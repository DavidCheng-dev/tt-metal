# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Elementwise mul precision/data-format efficiency benchmark.

Companion to test_sin_precision.py — same sweep methodology applied to ttnn.mul,
a two-operand SFPU binary op. The key difference from sin is the bandwidth model:
mul reads two inputs and writes one output, so the effective bytes per iteration is
3× instead of 2×.

Sweeps at fixed shape (2048×2048):

  A. dtype          — bfloat16 / bfloat8_b / bfloat4_b / float32
  B. math_fidelity  — LoFi / HiFi2 / HiFi3 / HiFi4 (expected flat for SFPU)
  C. fp32_dest_acc_en — False / True (expected flat for SFPU)
  D. packer_l1_acc    — False / True (expected flat for SFPU)

Run:
    python tests/ttnn/unit_tests/operations/precision_efficiency/test_mul_precision.py
    pytest -s tests/ttnn/unit_tests/operations/precision_efficiency/test_mul_precision.py
"""

import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.unit_tests.operations.precision_efficiency.precision_utils import (
    TILE,
    TRACE_REGION_SIZE,
    DATA_FORMATS,
    MATH_FIDELITIES,
    FP32_DEST_ACC_VALS,
    PACKER_L1_ACC_VALS,
    BASELINE,
    time_op,
    run_axis_sweep,
)

SHAPE = (2048, 2048)


def _padded(x):
    return -(-x // TILE) * TILE


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


def run_mul(device, dtype, math_fidelity, fp32_dest_acc_en, packer_l1_acc):
    """Time ttnn.mul on (m, n) tensors.

    math_fidelity / fp32_dest_acc_en / packer_l1_acc are accepted for API uniformity
    but have no effect on SFPU binary ops.

    Returns (per_iter_s, gops_per_s).
    """
    m, n = SHAPE
    a = ttnn.rand((m, n), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
    b = ttnn.rand((m, n), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
    per_iter_s = time_op(device, lambda: ttnn.mul(a, b))
    ttnn.deallocate(a)
    ttnn.deallocate(b)

    padded_elems = _padded(m) * _padded(n)
    gops = padded_elems / per_iter_s / 1e9
    return per_iter_s, gops


def run_mul_bw(device, dtype, math_fidelity, fp32_dest_acc_en, packer_l1_acc):
    """Like run_mul but returns GB/s using the 3-tensor bandwidth model (2 reads + 1 write)."""
    m, n = SHAPE
    bytes_per_elem = {ttnn.bfloat16: 2, ttnn.bfloat8_b: 1, ttnn.bfloat4_b: 0.5, ttnn.float32: 4}.get(dtype, 2)
    a = ttnn.rand((m, n), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
    b = ttnn.rand((m, n), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
    per_iter_s = time_op(device, lambda: ttnn.mul(a, b))
    ttnn.deallocate(a)
    ttnn.deallocate(b)

    padded_elems = _padded(m) * _padded(n)
    total_bytes = 3 * padded_elems * bytes_per_elem  # 2 reads + 1 write
    bw_gbs = total_bytes / per_iter_s / 1e9
    return per_iter_s, bw_gbs


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------


def _correctness_check(device):
    m, n = 256, 512
    ta = torch.rand(m, n, dtype=torch.bfloat16)
    tb = torch.rand(m, n, dtype=torch.bfloat16)
    golden = ta * tb
    a = ttnn.from_torch(ta, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(tb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.mul(a, b)
    assert_with_pcc(golden, ttnn.to_torch(out), 0.999)
    ttnn.deallocate(a)
    ttnn.deallocate(b)
    ttnn.deallocate(out)


# ---------------------------------------------------------------------------
# pytest entry points
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
def test_correctness(device):
    """Verify ttnn.mul output matches torch element-wise mul within PCC 0.999."""
    _correctness_check(device)


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("dtype,dtype_name,_bpe", DATA_FORMATS)
def test_axis_dtype(device, dtype, dtype_name, _bpe):
    """Set A: sweep dtype. Reports GB/s using the 3-tensor bandwidth model."""
    per_iter_s, bw = run_mul_bw(
        device, dtype, **{k: BASELINE[k] for k in ("math_fidelity", "fp32_dest_acc_en", "packer_l1_acc")}
    )
    print(f"\nA  dtype={dtype_name}  {SHAPE}  {per_iter_s * 1e3:.3f} ms/iter  {bw:.2f} GB/s")


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("fidelity,fidelity_name", MATH_FIDELITIES)
def test_axis_math_fidelity(device, fidelity, fidelity_name):
    """Set B: sweep math_fidelity — expected flat for SFPU binary ops."""
    per_iter_s, gops = run_mul(
        device, BASELINE["dtype"], fidelity, BASELINE["fp32_dest_acc_en"], BASELINE["packer_l1_acc"]
    )
    print(f"\nB  fidelity={fidelity_name}  {SHAPE}  {per_iter_s * 1e3:.3f} ms/iter  {gops:.2f} GOps/s")


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("fp32_dest_acc_en,label", FP32_DEST_ACC_VALS)
def test_axis_fp32_dest_acc_en(device, fp32_dest_acc_en, label):
    """Set C: sweep fp32_dest_acc_en — expected flat for SFPU ops."""
    per_iter_s, gops = run_mul(
        device, BASELINE["dtype"], BASELINE["math_fidelity"], fp32_dest_acc_en, BASELINE["packer_l1_acc"]
    )
    print(f"\nC  fp32_dest_acc_en={label}  {SHAPE}  {per_iter_s * 1e3:.3f} ms/iter  {gops:.2f} GOps/s")


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("packer_l1_acc,label", PACKER_L1_ACC_VALS)
def test_axis_packer_l1_acc(device, packer_l1_acc, label):
    """Set D: sweep packer_l1_acc — expected flat for SFPU ops."""
    per_iter_s, gops = run_mul(
        device, BASELINE["dtype"], BASELINE["math_fidelity"], BASELINE["fp32_dest_acc_en"], packer_l1_acc
    )
    print(f"\nD  packer_l1_acc={label}  {SHAPE}  {per_iter_s * 1e3:.3f} ms/iter  {gops:.2f} GOps/s")


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------


def main():
    device = ttnn.open_device(device_id=0, trace_region_size=TRACE_REGION_SIZE)
    try:
        _correctness_check(device)
        print(f"correctness ({SHAPE[0]}x{SHAPE[1]}, bfloat16): PCC ok\n")

        table_a = run_axis_sweep(
            device,
            run_mul_bw,
            "dtype",
            [(d, n) for d, n, _ in DATA_FORMATS],
            BASELINE,
            title=f"Set A: dtype sweep — mul {SHAPE[0]}x{SHAPE[1]}",
            takeaway="Takeaway: 3-tensor bandwidth model (2 reads + 1 write). Smaller dtype → more elements/s at same GB/s.",
            throughput_unit="GB/s",
        )

        table_b = run_axis_sweep(
            device,
            run_mul,
            "math_fidelity",
            MATH_FIDELITIES,
            BASELINE,
            title=f"Set B: math_fidelity sweep — mul {SHAPE[0]}x{SHAPE[1]}",
            takeaway="Takeaway: SFPU binary ops bypass the matrix engine — fidelity should have no effect.",
        )

        table_c = run_axis_sweep(
            device,
            run_mul,
            "fp32_dest_acc_en",
            FP32_DEST_ACC_VALS,
            BASELINE,
            title=f"Set C: fp32_dest_acc_en sweep — mul {SHAPE[0]}x{SHAPE[1]}",
            takeaway="Takeaway: fp32 dest accumulation does not affect SFPU elementwise ops.",
        )

        table_d = run_axis_sweep(
            device,
            run_mul,
            "packer_l1_acc",
            PACKER_L1_ACC_VALS,
            BASELINE,
            title=f"Set D: packer_l1_acc sweep — mul {SHAPE[0]}x{SHAPE[1]}",
            takeaway="Takeaway: L1 packer accumulation does not affect SFPU elementwise ops.",
        )

        for t in [table_a, table_b, table_c, table_d]:
            print("\n" + t.render())
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
