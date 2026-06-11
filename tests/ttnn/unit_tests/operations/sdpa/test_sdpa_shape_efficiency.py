# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
SDPA prefill shape-efficiency demo.

Companion to test_matmul_shape_efficiency.py, applied to causal scaled dot-product
attention during prefill.  The Tensix attention engine processes Q, K, V in chunks
of size q_chunk × k_chunk, so the SEQUENCE LENGTH S decides:

  A. Chunk utilization — short sequences fill only a few chunks; the engine spends
     more time on setup than on compute, so short-S TFLOPS is low.

  B. Compute/memory balance — SDPA FLOPs scale as O(S²·D) while the memory traffic
     scales as O(S·D), so longer sequences become more compute-bound and TFLOPS rises.

  C. Head-dimension effect — wider heads (larger D) pack more FLOPs per bandwidth byte,
     so D=128 should reach higher TFLOPS than D=64 at the same S.

FLOPs counted as 4·B·H·S²·D (two S×D @ D×S matmuls for QK^T and PV, each 2·S·D·S ops).

Run:
    python tests/ttnn/unit_tests/operations/sdpa/test_sdpa_shape_efficiency.py
    pytest -s tests/ttnn/unit_tests/operations/sdpa/test_sdpa_shape_efficiency.py
"""

import time

import pytest
import torch
import torch.nn.functional as F

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

TILE = 32
TRACE_REGION_SIZE = 23887872

CORE_GRID = (8, 8)
DTYPE = ttnn.bfloat16

# Set A: fix H=8, D=128, sweep S (seq len) from small to large.
# Chunk size 64 divides all S values.
SET_A_H = 8
SET_A_D = 128
SET_A_SEQ_LENS = [64, 128, 256, 512, 1024, 2048, 4096]
SET_A_CHUNK = 64

# Set B: fix H=8, S=1024, sweep D (head dimension).
SET_B_H = 8
SET_B_S = 1024
SET_B_CHUNK = 128
SET_B_HEAD_DIMS = [64, 96, 128]


# --------------------------------------------------------------------------------------
# Timing helpers (same two-trace subtraction trick as test_matmul_shape_efficiency.py)
# --------------------------------------------------------------------------------------


def _time_with_trace(device, op_fn, warmup_iters, num_iters):
    op_fn()
    ttnn.synchronize_device(device)

    def capture(n):
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        for _ in range(n):
            op_fn()
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.synchronize_device(device)
        return tid

    warmup_trace = capture(warmup_iters)
    main_trace = capture(num_iters)

    t0 = time.perf_counter()
    ttnn.execute_trace(device, warmup_trace, blocking=False)
    ttnn.synchronize_device(device)
    warmup_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    ttnn.execute_trace(device, main_trace, blocking=False)
    ttnn.synchronize_device(device)
    main_s = time.perf_counter() - t0

    ttnn.release_trace(device, warmup_trace)
    ttnn.release_trace(device, main_trace)

    return (main_s - warmup_s) / (num_iters - warmup_iters)


def _time_without_trace(device, op_fn, num_iters):
    for _ in range(3):
        op_fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(num_iters):
        op_fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) / num_iters


def time_op(device, op_fn, warmup_iters=10, num_iters=30):
    try:
        return _time_with_trace(device, op_fn, warmup_iters, num_iters)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] trace timing unavailable ({e}); falling back to eager timing")
        return _time_without_trace(device, op_fn, num_iters)


# --------------------------------------------------------------------------------------
# SDPA benchmark helper
# --------------------------------------------------------------------------------------


def run_sdpa(device, b, h, s, d, chunk_size):
    """Time causal SDPA prefill for a single (b,h,s,d) shape.

    Returns (per_iter_s, tflops).
    FLOPs = 4 * b * h * s^2 * d  (two matmuls: QK^T and PV, each 2*s*d*s ops).
    """
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=CORE_GRID,
        q_chunk_size=chunk_size,
        k_chunk_size=chunk_size,
        exp_approx_mode=True,
    )

    tq = torch.randn(b, h, s, d, dtype=torch.bfloat16)
    tk = torch.randn(b, h, s, d, dtype=torch.bfloat16)
    tv = torch.randn(b, h, s, d, dtype=torch.bfloat16)

    q = ttnn.from_torch(tq, dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    k = ttnn.from_torch(tk, dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    v = ttnn.from_torch(tv, dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    per_iter_s = time_op(
        device,
        lambda: ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        ),
    )

    ttnn.deallocate(q)
    ttnn.deallocate(k)
    ttnn.deallocate(v)

    flops = 4 * b * h * s * s * d
    tflops = flops / per_iter_s / 1e12
    return per_iter_s, tflops


# --------------------------------------------------------------------------------------
# Result table
# --------------------------------------------------------------------------------------


class SdpaResultTable:
    """Prints a shape-efficiency table for SDPA, normalised to the best row."""

    def __init__(self, title, takeaway, sweep_col):
        self.title = title
        self.takeaway = takeaway
        self.sweep_col = sweep_col  # column label for the swept dimension
        self.rows = []  # (sweep_val, per_iter_s, tflops)

    def add(self, sweep_val, per_iter_s, tflops):
        self.rows.append((sweep_val, per_iter_s, tflops))

    def render(self):
        best = max((r[2] for r in self.rows), default=0.0)
        hdr = f"{self.sweep_col:>8} {'ms/iter':>10} {'TFLOPS':>10} {'% of best':>11}"
        sep = "-" * len(hdr)
        lines = [self.title, self.takeaway, sep, hdr, sep]
        for val, t, tflops in self.rows:
            pct = tflops / best if best > 0 else float("nan")
            lines.append(f"{val:>8} {t * 1e3:>10.3f} {tflops:>10.2f} {pct:>10.1%}")
        lines.append(sep)
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Correctness check
# --------------------------------------------------------------------------------------


def _correctness_check(device):
    b, h, s, d = 1, 4, 128, 64
    tq = torch.randn(b, h, s, d, dtype=torch.bfloat16)
    tk = torch.randn(b, h, s, d, dtype=torch.bfloat16)
    tv = torch.randn(b, h, s, d, dtype=torch.bfloat16)

    golden = F.scaled_dot_product_attention(tq.float(), tk.float(), tv.float(), is_causal=True).to(torch.bfloat16)

    prog_cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=CORE_GRID,
        q_chunk_size=64,
        k_chunk_size=64,
        exp_approx_mode=False,
    )
    compute_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    q = ttnn.from_torch(tq, dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    k = ttnn.from_torch(tk, dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    v = ttnn.from_torch(tv, dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    out = ttnn.transformer.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=True,
        program_config=prog_cfg,
        compute_kernel_config=compute_cfg,
    )
    assert_with_pcc(golden, ttnn.to_torch(out), 0.99)
    ttnn.deallocate(q)
    ttnn.deallocate(k)
    ttnn.deallocate(v)
    ttnn.deallocate(out)


# --------------------------------------------------------------------------------------
# pytest entry points
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
def test_correctness(device):
    """Verify causal SDPA prefill matches PyTorch SDPA within PCC 0.99."""
    _correctness_check(device)


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("seq_len", SET_A_SEQ_LENS)
def test_seq_len_scaling(device, seq_len):
    """Set A: sweep seq_len with H=8, D=128 fixed.

    Short sequences are overhead-bound (few chunks, low utilization); longer sequences
    become more compute-bound as S² FLOPs dominate over S·D memory traffic.
    """
    per_iter_s, tflops = run_sdpa(device, b=1, h=SET_A_H, s=seq_len, d=SET_A_D, chunk_size=SET_A_CHUNK)
    print(f"\nA  S={seq_len:<5} H={SET_A_H} D={SET_A_D}  " f"{per_iter_s * 1e3:.3f} ms/iter  {tflops:.2f} TFLOPS")


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("head_dim", SET_B_HEAD_DIMS)
def test_head_dim_scaling(device, head_dim):
    """Set B: sweep head dim D with H=8, S=1024 fixed.

    Larger D packs more FLOPs per byte of K/V bandwidth: FLOPs ∝ S²·D while
    bandwidth ∝ S·D.  Wider heads should approach higher TFLOPS when compute-bound.
    """
    per_iter_s, tflops = run_sdpa(device, b=1, h=SET_B_H, s=SET_B_S, d=head_dim, chunk_size=SET_B_CHUNK)
    print(f"\nB  D={head_dim:<4} H={SET_B_H} S={SET_B_S}  " f"{per_iter_s * 1e3:.3f} ms/iter  {tflops:.2f} TFLOPS")


# --------------------------------------------------------------------------------------
# Standalone entry point
# --------------------------------------------------------------------------------------


def main():
    device = ttnn.open_device(device_id=0, trace_region_size=TRACE_REGION_SIZE)
    try:
        _correctness_check(device)
        print("correctness (B=1,H=4,S=128,D=64 causal): PCC ok\n")

        table_a = SdpaResultTable(
            "Set A: seq-len scaling  H=%d D=%d  causal bf16" % (SET_A_H, SET_A_D),
            "Takeaway: short sequences are overhead-bound (few chunks); TFLOPS rises as S grows "
            "because S²·D FLOPs dominate S·D memory traffic.",
            sweep_col="S",
        )
        for s in SET_A_SEQ_LENS:
            per_iter_s, tflops = run_sdpa(device, b=1, h=SET_A_H, s=s, d=SET_A_D, chunk_size=SET_A_CHUNK)
            table_a.add(s, per_iter_s, tflops)

        table_b = SdpaResultTable(
            "Set B: head-dim scaling  H=%d S=%d  causal bf16" % (SET_B_H, SET_B_S),
            "Takeaway: wider heads pack more FLOPs per byte — D=128 should be more compute-bound "
            "than D=64 at the same S, reaching higher TFLOPS.",
            sweep_col="D",
        )
        for d in SET_B_HEAD_DIMS:
            per_iter_s, tflops = run_sdpa(device, b=1, h=SET_B_H, s=SET_B_S, d=d, chunk_size=SET_B_CHUNK)
            table_b.add(d, per_iter_s, tflops)

        print("\n" + table_a.render())
        print("\n" + table_b.render())
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
