# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Scaled-dot-product attention precision/data-format efficiency benchmark.

SDPA combines both the matrix engine (Q·Kᵀ and softmax·V) and the SFPU (softmax),
so it is sensitive to all four precision axes. This demo fixes a typical decode-stage
shape — (1, 8, 2048, 128): 1 batch, 8 heads, 2048 seq, 128 head-dim — and sweeps:

  A. dtype          — bfloat16 / bfloat8_b
                       (bfloat4_b is typically not supported for SDPA on current hardware)
  B. math_fidelity  — LoFi / HiFi2 / HiFi3 / HiFi4
                       (matrix-engine fidelity for QK and AV products)
  C. fp32_dest_acc_en — False / True
                       (fp32 accumulation in QK/AV dest registers)
  D. packer_l1_acc    — False / True

Throughput is reported as equivalent TFLOPS using the standard attention FLOP count:
  FLOPs = 4 * B * H * S * S * D   (two matrix products: Q·Kᵀ and softmax(QKᵀ)·V)

Correctness is checked against torch.nn.functional.scaled_dot_product_attention with
PCC ≥ 0.99. Note that ttnn SDPA uses chunked flash-attention internally, so the
tolerance is looser than plain matmul — PCC is the right metric here.

Run:
    python tests/ttnn/unit_tests/operations/precision_efficiency/test_attention_precision.py
    pytest -s tests/ttnn/unit_tests/operations/precision_efficiency/test_attention_precision.py
"""

import pytest
import torch
import torch.nn.functional as F

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.unit_tests.operations.precision_efficiency.precision_utils import (
    TRACE_REGION_SIZE,
    MATH_FIDELITIES,
    FP32_DEST_ACC_VALS,
    PACKER_L1_ACC_VALS,
    BASELINE,
    make_compute_config,
    run_axis_sweep,
    time_op,
)

# Q/K/V shape: (batch, num_heads, seq_len, head_dim)
BATCH, HEADS, SEQ, HEAD_DIM = 1, 8, 2048, 128

# SDPA dtype sweep: bfloat4_b is unsupported for flash-attention; float32 is excluded
# because SDPA input in float32 requires special handling outside this scope.
SDPA_DATA_FORMATS = [
    (ttnn.bfloat16, "bfloat16"),
    (ttnn.bfloat8_b, "bfloat8_b"),
]

# SDPA program config — chunk sizes must be multiples of 32 and ≤ seq_len.
# 256 is the standard default used in production SDPA tests.
_SDPA_PROGRAM_CONFIG = None  # lazily built per device (needs grid size)


def _sdpa_program_config(device):
    grid = device.compute_with_storage_grid_size()
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
        q_chunk_size=256,
        k_chunk_size=256,
        exp_approx_mode=False,
    )


# ---------------------------------------------------------------------------
# Equivalent FLOP count for attention
# ---------------------------------------------------------------------------


def _attn_tflops(per_iter_s):
    """TFLOPS using the standard 4·B·H·S·S·D FLOP count for two matrix products."""
    flops = 4 * BATCH * HEADS * SEQ * SEQ * HEAD_DIM
    return flops / per_iter_s / 1e12


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


def run_attention(device, dtype, math_fidelity, fp32_dest_acc_en, packer_l1_acc):
    """Time SDPA at the given configuration.

    Returns (per_iter_s, tflops).
    """
    cfg = make_compute_config(
        device, math_fidelity=math_fidelity, fp32_dest_acc_en=fp32_dest_acc_en, packer_l1_acc=packer_l1_acc
    )
    spc = _sdpa_program_config(device)
    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    q = ttnn.rand((BATCH, HEADS, SEQ, HEAD_DIM), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem)
    k = ttnn.rand((BATCH, HEADS, SEQ, HEAD_DIM), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem)
    v = ttnn.rand((BATCH, HEADS, SEQ, HEAD_DIM), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem)

    per_iter_s = time_op(
        device,
        lambda: ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            program_config=spc,
            compute_kernel_config=cfg,
        ),
    )

    ttnn.deallocate(q)
    ttnn.deallocate(k)
    ttnn.deallocate(v)

    return per_iter_s, _attn_tflops(per_iter_s)


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------


def _correctness_check(device):
    b, h, s, d = 1, 4, 512, 64
    tq = torch.rand(b, h, s, d, dtype=torch.bfloat16)
    tk = torch.rand(b, h, s, d, dtype=torch.bfloat16)
    tv = torch.rand(b, h, s, d, dtype=torch.bfloat16)
    golden = F.scaled_dot_product_attention(tq.float(), tk.float(), tv.float(), is_causal=True).to(torch.bfloat16)

    spc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(
            device.compute_with_storage_grid_size().x,
            device.compute_with_storage_grid_size().y,
        ),
        q_chunk_size=128,
        k_chunk_size=128,
        exp_approx_mode=False,
    )
    cfg = make_compute_config(device)
    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    q = ttnn.from_torch(tq, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    k = ttnn.from_torch(tk, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    v = ttnn.from_torch(tv, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)

    out = ttnn.transformer.scaled_dot_product_attention(
        q, k, v, is_causal=True, program_config=spc, compute_kernel_config=cfg
    )
    assert_with_pcc(golden, ttnn.to_torch(out), 0.99)

    for t in [q, k, v, out]:
        ttnn.deallocate(t)


# ---------------------------------------------------------------------------
# pytest entry points
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
def test_correctness(device):
    """Verify ttnn SDPA output matches torch SDPA within PCC 0.99."""
    _correctness_check(device)


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("dtype,dtype_name", SDPA_DATA_FORMATS)
def test_axis_dtype(device, dtype, dtype_name):
    """Set A: sweep dtype — bfloat8_b halves KV cache size and reduces DRAM traffic."""
    per_iter_s, tflops = run_attention(
        device, dtype, **{k: BASELINE[k] for k in ("math_fidelity", "fp32_dest_acc_en", "packer_l1_acc")}
    )
    print(
        f"\nA  dtype={dtype_name}  {BATCH}x{HEADS}x{SEQ}x{HEAD_DIM}  {per_iter_s * 1e3:.3f} ms/iter  {tflops:.3f} TFLOPS"
    )


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("fidelity,fidelity_name", MATH_FIDELITIES)
def test_axis_math_fidelity(device, fidelity, fidelity_name):
    """Set B: sweep math_fidelity for the QK and AV matrix products inside SDPA."""
    per_iter_s, tflops = run_attention(
        device, BASELINE["dtype"], fidelity, BASELINE["fp32_dest_acc_en"], BASELINE["packer_l1_acc"]
    )
    print(
        f"\nB  fidelity={fidelity_name}  {BATCH}x{HEADS}x{SEQ}x{HEAD_DIM}  {per_iter_s * 1e3:.3f} ms/iter  {tflops:.3f} TFLOPS"
    )


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("fp32_dest_acc_en,label", FP32_DEST_ACC_VALS)
def test_axis_fp32_dest_acc_en(device, fp32_dest_acc_en, label):
    """Set C: sweep fp32_dest_acc_en — trades throughput for numerical precision in QK/AV accumulators."""
    per_iter_s, tflops = run_attention(
        device, BASELINE["dtype"], BASELINE["math_fidelity"], fp32_dest_acc_en, BASELINE["packer_l1_acc"]
    )
    print(
        f"\nC  fp32_dest_acc_en={label}  {BATCH}x{HEADS}x{SEQ}x{HEAD_DIM}  {per_iter_s * 1e3:.3f} ms/iter  {tflops:.3f} TFLOPS"
    )


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("packer_l1_acc,label", PACKER_L1_ACC_VALS)
def test_axis_packer_l1_acc(device, packer_l1_acc, label):
    """Set D: sweep packer_l1_acc for the attention output write-back path."""
    per_iter_s, tflops = run_attention(
        device, BASELINE["dtype"], BASELINE["math_fidelity"], BASELINE["fp32_dest_acc_en"], packer_l1_acc
    )
    print(
        f"\nD  packer_l1_acc={label}  {BATCH}x{HEADS}x{SEQ}x{HEAD_DIM}  {per_iter_s * 1e3:.3f} ms/iter  {tflops:.3f} TFLOPS"
    )


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------


def main():
    device = ttnn.open_device(device_id=0, trace_region_size=TRACE_REGION_SIZE)
    try:
        _correctness_check(device)
        print(f"correctness (1x4x512x64, bfloat16, causal): PCC ok\n")

        table_a = run_axis_sweep(
            device,
            run_attention,
            "dtype",
            SDPA_DATA_FORMATS,
            BASELINE,
            title=f"Set A: dtype sweep — SDPA {BATCH}x{HEADS}x{SEQ}x{HEAD_DIM}",
            takeaway=(
                "Takeaway: bfloat8_b halves KV cache DRAM traffic; expect higher TFLOPS "
                "for memory-bound sequence lengths."
            ),
            throughput_unit="TFLOPS",
        )

        table_b = run_axis_sweep(
            device,
            run_attention,
            "math_fidelity",
            MATH_FIDELITIES,
            BASELINE,
            title=f"Set B: math_fidelity sweep — SDPA {BATCH}x{HEADS}x{SEQ}x{HEAD_DIM}",
            takeaway=(
                "Takeaway: LoFi is fastest; HiFi4 adds matrix-engine cycles for each QK and AV product → lower TFLOPS."
            ),
            throughput_unit="TFLOPS",
        )

        table_c = run_axis_sweep(
            device,
            run_attention,
            "fp32_dest_acc_en",
            FP32_DEST_ACC_VALS,
            BASELINE,
            title=f"Set C: fp32_dest_acc_en sweep — SDPA {BATCH}x{HEADS}x{SEQ}x{HEAD_DIM}",
            takeaway=(
                "Takeaway: fp32 dest accumulation trades TFLOPS for numerical stability in QK and AV dot products."
            ),
            throughput_unit="TFLOPS",
        )

        table_d = run_axis_sweep(
            device,
            run_attention,
            "packer_l1_acc",
            PACKER_L1_ACC_VALS,
            BASELINE,
            title=f"Set D: packer_l1_acc sweep — SDPA {BATCH}x{HEADS}x{SEQ}x{HEAD_DIM}",
            takeaway=(
                "Takeaway: L1 packer accumulation reduces output DRAM writes across chunks — may improve TFLOPS for long sequences."
            ),
            throughput_unit="TFLOPS",
        )

        for t in [table_a, table_b, table_c, table_d]:
            print("\n" + t.render())
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
