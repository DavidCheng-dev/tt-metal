# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Matmul shape-efficiency demo for the Tensix matrix engine.

Companion to tech_reports/matrix_engine/matrix_engine.md. The matrix engine processes
fixed-size tiles (8x16 x 16x16 per cycle), so the SHAPE of a matmul decides how many of
those cycles do useful work. This demo measures, on real hardware, two consequences:

  A. Shape / tile utilization - a "skinny" matmul (small M, i.e. few activation rows) or a
     non-32-aligned shape (which the runtime pads up to a full tile) wastes engine cycles,
     so its achieved TFLOPS is far below a fat square matmul of the same fidelity.

  B. Core-grid / total size - a tiny matmul is dominated by host dispatch overhead and
     cannot fill the 8x8 core grid; only a large matmul saturates the grid and approaches
     the engine's peak throughput.

Fidelity is fixed at LoFi (the top matrix-engine rate for bf16) so shape is the only
variable.

Run:
    python tests/ttnn/unit_tests/operations/matmul/test_matmul_shape_efficiency.py
    pytest -s tests/ttnn/unit_tests/operations/matmul/test_matmul_shape_efficiency.py
"""

import time

import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

TILE = 32

# Trace region for the small/large demo workloads. Trace captures the op program once so
# timing excludes host-side dispatch overhead.
TRACE_REGION_SIZE = 23887872

# LoFi keeps the matrix engine at its top bf16 fidelity rate, so the only thing that moves
# the measured TFLOPS between cases is the matmul SHAPE.
COMPUTE_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=False,
)

# Case set A: fix K=N, sweep M from skinny (1 tile) to fat. The skinny cases starve the
# matrix engine of activation rows; the two *_misaligned cases are +1 / +8 over a tile
# boundary, so the runtime pads M up to the next multiple of 32 and the padding is wasted
# compute that does not count toward real FLOPs.
SET_A_KN = 512
SET_A_M = [32, 33, 40, 64, 65, 72, 128, 129, 136, 256, 257, 264, 512, 513, 520, 2048, 2049, 4096]

# Case set B: square M=K=N from tiny (overhead-bound) to large (grid-saturating).
SET_B_SQUARE = [128, 256, 512, 1024, 2048, 4096]

# Case set C: prefill seq-len sweep — fix K=N to a transformer-sized hidden dim and
# sweep M = seq_len from one tile to 4096 tokens.  This matches the prefill matmul
# pattern (activations: S×d_model; weights: d_model×d_out) and shows how TFLOPS rises
# as the sequence length grows long enough to saturate the matrix engine.
SET_C_KN = 2048  # d_model (hidden dim) — both K and N
SET_C_SEQ_LENS = [32, 64, 128, 256, 512, 1024, 2048, 4096]


def _time_with_trace(device, op_fn, warmup_iters, num_iters):
    """Trace-capture timing: capture the op warmup_iters and num_iters times into two
    traces, then subtract wall-times to cancel the fixed per-trace launch overhead,
    leaving (num_iters - warmup_iters) pure op executions."""
    op_fn()  # compile + cache kernels
    ttnn.synchronize_device(device)

    def capture(n_iters):
        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        for _ in range(n_iters):
            op_fn()
        ttnn.end_trace_capture(device, trace_id, cq_id=0)
        ttnn.synchronize_device(device)
        return trace_id

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
    """Mean per-iteration seconds for op_fn on a single device. Falls back to eager timing
    if trace capture is unavailable."""
    try:
        return _time_with_trace(device, op_fn, warmup_iters, num_iters)
    except Exception as e:  # noqa: BLE001 - demo: degrade gracefully
        print(f"  [warn] trace timing unavailable ({e}); falling back to eager timing")
        return _time_without_trace(device, op_fn, num_iters)


def run_matmul(device, m, k, n):
    """Time a single (m,k,n) bf16 LoFi matmul. Returns (per_iter_s, tflops) where TFLOPS is
    computed on the REAL (un-padded) m,k,n so tile-padding waste shows up as a drop."""
    a = ttnn.rand((m, k), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    b = ttnn.rand((k, n), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    per_iter_s = time_op(device, lambda: ttnn.matmul(a, b, compute_kernel_config=COMPUTE_KERNEL_CONFIG))

    ttnn.deallocate(a)
    ttnn.deallocate(b)

    tflops = (2 * m * k * n) / per_iter_s / 1e12
    return per_iter_s, tflops


def _tiles(m, k, n):
    """Padded tile counts (ceil to TILE) - this is the work the engine actually does."""
    return tuple(-(-x // TILE) for x in (m, k, n))


class ResultTable:
    """Collects (m,k,n, per_iter_s, tflops) rows and prints a shape-efficiency table.
    '% of best' is normalized to the highest TFLOPS in the table, so it is meaningful on
    any arch without hard-coding a peak constant."""

    def __init__(self, title, takeaway):
        self.title = title
        self.takeaway = takeaway
        self.rows = []  # (m, k, n, per_iter_s, tflops)

    def add(self, m, k, n, per_iter_s, tflops):
        self.rows.append((m, k, n, per_iter_s, tflops))

    def render(self):
        best = max((r[4] for r in self.rows), default=0.0)
        header = f"{'shape (MxKxN)':>20} {'tiles (MxKxN)':>16} {'ms/iter':>10} {'TFLOPS':>10} {'% of best':>11}"
        lines = [self.title, self.takeaway, "-" * len(header), header, "-" * len(header)]
        for m, k, n, t, tflops in self.rows:
            tm, tk, tn = _tiles(m, k, n)
            shape = f"{m}x{k}x{n}"
            tiles = f"{tm}x{tk}x{tn}"
            pct = (tflops / best) if best > 0 else float("nan")
            lines.append(f"{shape:>20} {tiles:>16} {t * 1e3:>10.3f} {tflops:>10.2f} {pct:>10.1%}")
        lines.append("-" * len(header))
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# pytest entry points
# --------------------------------------------------------------------------------------


def _correctness_check(device):
    """Prove we are running a real matmul before reporting perf (small shape only)."""
    m = k = n = 256
    ta = torch.rand(m, k, dtype=torch.bfloat16)
    tb = torch.rand(k, n, dtype=torch.bfloat16)
    golden = ta.float() @ tb.float()
    a = ttnn.from_torch(ta, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(tb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.matmul(a, b, compute_kernel_config=COMPUTE_KERNEL_CONFIG)
    assert_with_pcc(golden, ttnn.to_torch(out), 0.99)
    ttnn.deallocate(a)
    ttnn.deallocate(b)
    ttnn.deallocate(out)


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
def test_correctness(device):
    _correctness_check(device)


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("m", SET_A_M)
def test_shape_tile_utilization(device, m):
    k = n = SET_A_KN
    per_iter_s, tflops = run_matmul(device, m, k, n)
    tm, tk, tn = _tiles(m, k, n)
    print(f"\nA  {m}x{k}x{n}  tiles {tm}x{tk}x{tn}  {per_iter_s * 1e3:.3f} ms/iter  {tflops:.2f} TFLOPS")


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("s", SET_B_SQUARE)
def test_core_grid_total_size(device, s):
    per_iter_s, tflops = run_matmul(device, s, s, s)
    ts = s // TILE
    print(f"\nB  {s}x{s}x{s}  tiles {ts}x{ts}x{ts}  {per_iter_s * 1e3:.3f} ms/iter  {tflops:.2f} TFLOPS")


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("seq_len", SET_C_SEQ_LENS)
def test_prefill_seq_len(device, seq_len):
    """Set C: prefill seq-len sweep (M=seq_len, K=N=d_model).

    Models the token-parallel matmul during prefill: (S, d_model) @ (d_model, d_out).
    Small sequence lengths starve the matrix engine; longer contexts approach peak TFLOPS.
    """
    k = n = SET_C_KN
    per_iter_s, tflops = run_matmul(device, seq_len, k, n)
    tm, tk, tn = _tiles(seq_len, k, n)
    print(f"\nC  {seq_len}x{k}x{n}  tiles {tm}x{tk}x{tn}  {per_iter_s * 1e3:.3f} ms/iter  {tflops:.2f} TFLOPS")


# --------------------------------------------------------------------------------------
# standalone entry point - builds the two formatted tables
# --------------------------------------------------------------------------------------


def main():
    device = ttnn.open_device(device_id=0, trace_region_size=TRACE_REGION_SIZE)
    try:
        _correctness_check(device)
        print("correctness (256x256x256): PCC ok\n")

        table_a = ResultTable(
            "Set A: shape / tile utilization (K=N=%d, LoFi bf16)" % SET_A_KN,
            "Takeaway: skinny (small-M) and non-32-aligned shapes waste matrix-engine cycles "
            "-> lower TFLOPS than a fat square matmul.",
        )
        for m in SET_A_M:
            per_iter_s, tflops = run_matmul(device, m, SET_A_KN, SET_A_KN)
            table_a.add(m, SET_A_KN, SET_A_KN, per_iter_s, tflops)

        table_b = ResultTable(
            "Set B: core-grid / total size (square M=K=N, LoFi bf16)",
            "Takeaway: tiny matmuls are dispatch/overhead-bound; only large ones saturate the "
            "core grid and approach peak TFLOPS.",
        )
        for s in SET_B_SQUARE:
            per_iter_s, tflops = run_matmul(device, s, s, s)
            table_b.add(s, s, s, per_iter_s, tflops)

        table_c = ResultTable(
            "Set C: prefill seq-len sweep (K=N=%d, LoFi bf16)" % SET_C_KN,
            "Takeaway: short sequences (small M) starve the matrix engine; as seq_len grows "
            "the engine saturates and TFLOPS rises toward peak.",
        )
        for seq_len in SET_C_SEQ_LENS:
            per_iter_s, tflops = run_matmul(device, seq_len, SET_C_KN, SET_C_KN)
            table_c.add(seq_len, SET_C_KN, SET_C_KN, per_iter_s, tflops)

        print("\n" + table_a.render())
        print("\n" + table_b.render())
        print("\n" + table_c.render())
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
