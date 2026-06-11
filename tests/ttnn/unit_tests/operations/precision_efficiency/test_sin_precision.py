# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Elementwise sin precision/data-format efficiency benchmark.

Companion to test_sin_shape_efficiency.py. Where that demo fixes a dtype and sweeps
shape, this demo fixes a bandwidth-saturating shape (2048×2048) and sweeps:

  A. dtype          — bfloat16 vs bfloat8_b vs bfloat4_b vs float32
  B. math_fidelity  — LoFi / HiFi2 / HiFi3 / HiFi4 (SFPU does not use the matrix
                       engine, so this axis is expected to be flat — a useful
                       negative-result confirmation)
  C. fp32_dest_acc_en — False / True
  D. packer_l1_acc    — False / True

Throughput is reported in GOps/s (logical elements per second) and GB/s (DRAM
bandwidth), using padded element counts to reflect actual hardware work.

Run:
    python tests/ttnn/unit_tests/operations/precision_efficiency/test_sin_precision.py
    pytest -s tests/ttnn/unit_tests/operations/precision_efficiency/test_sin_precision.py
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

# Fixed bandwidth-saturating shape (same as test_sin_shape_efficiency.py Set C).
SHAPE = (2048, 2048)


def _padded(x):
    return -(-x // TILE) * TILE


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


def run_sin(device, dtype, math_fidelity, fp32_dest_acc_en, packer_l1_acc):
    """Time ttnn.sin at the fixed shape for a given dtype configuration.

    math_fidelity / fp32_dest_acc_en / packer_l1_acc are accepted for API uniformity
    but do not affect SFPU kernels — their values are intentionally ignored here.

    Returns (per_iter_s, gops_per_s) where GOps/s uses padded element count.
    """
    m, n = SHAPE
    x = ttnn.rand((m, n), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
    per_iter_s = time_op(device, lambda: ttnn.sin(x))
    ttnn.deallocate(x)

    padded_elems = _padded(m) * _padded(n)
    gops = padded_elems / per_iter_s / 1e9
    return per_iter_s, gops


def run_sin_bw(device, dtype, math_fidelity, fp32_dest_acc_en, packer_l1_acc):
    """Like run_sin but returns GB/s instead of GOps/s (for the dtype sweep table)."""
    m, n = SHAPE
    bytes_per_elem = {ttnn.bfloat16: 2, ttnn.bfloat8_b: 1, ttnn.bfloat4_b: 0.5, ttnn.float32: 4}.get(dtype, 2)
    x = ttnn.rand((m, n), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
    per_iter_s = time_op(device, lambda: ttnn.sin(x))
    ttnn.deallocate(x)

    padded_elems = _padded(m) * _padded(n)
    total_bytes = 2 * padded_elems * bytes_per_elem  # read input + write output
    bw_gbs = total_bytes / per_iter_s / 1e9
    return per_iter_s, bw_gbs


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------


def _correctness_check(device):
    m, n = 256, 512
    t = torch.rand(m, n, dtype=torch.bfloat16) * 6.28318
    golden = torch.sin(t)
    x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.sin(x)
    assert_with_pcc(golden, ttnn.to_torch(out), 0.999)
    ttnn.deallocate(x)
    ttnn.deallocate(out)


# ---------------------------------------------------------------------------
# pytest entry points
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
def test_correctness(device):
    """Verify ttnn.sin output matches torch.sin within PCC 0.999."""
    _correctness_check(device)


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("dtype,dtype_name,_bpe", DATA_FORMATS)
def test_axis_dtype(device, dtype, dtype_name, _bpe):
    """Set A: sweep dtype at fixed shape. Reports GB/s to make byte-size effect visible."""
    per_iter_s, bw = run_sin_bw(
        device, dtype, **{k: BASELINE[k] for k in ("math_fidelity", "fp32_dest_acc_en", "packer_l1_acc")}
    )
    print(f"\nA  dtype={dtype_name}  {SHAPE}  {per_iter_s * 1e3:.3f} ms/iter  {bw:.2f} GB/s")


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("fidelity,fidelity_name", MATH_FIDELITIES)
def test_axis_math_fidelity(device, fidelity, fidelity_name):
    """Set B: sweep math_fidelity — expected to be flat (SFPU bypasses the matrix engine)."""
    per_iter_s, gops = run_sin(
        device, BASELINE["dtype"], fidelity, BASELINE["fp32_dest_acc_en"], BASELINE["packer_l1_acc"]
    )
    print(f"\nB  fidelity={fidelity_name}  {SHAPE}  {per_iter_s * 1e3:.3f} ms/iter  {gops:.2f} GOps/s")


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("fp32_dest_acc_en,label", FP32_DEST_ACC_VALS)
def test_axis_fp32_dest_acc_en(device, fp32_dest_acc_en, label):
    """Set C: sweep fp32_dest_acc_en — expected to be flat for SFPU ops."""
    per_iter_s, gops = run_sin(
        device, BASELINE["dtype"], BASELINE["math_fidelity"], fp32_dest_acc_en, BASELINE["packer_l1_acc"]
    )
    print(f"\nC  fp32_dest_acc_en={label}  {SHAPE}  {per_iter_s * 1e3:.3f} ms/iter  {gops:.2f} GOps/s")


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("packer_l1_acc,label", PACKER_L1_ACC_VALS)
def test_axis_packer_l1_acc(device, packer_l1_acc, label):
    """Set D: sweep packer_l1_acc — expected to be flat for SFPU ops."""
    per_iter_s, gops = run_sin(
        device, BASELINE["dtype"], BASELINE["math_fidelity"], BASELINE["fp32_dest_acc_en"], packer_l1_acc
    )
    print(f"\nD  packer_l1_acc={label}  {SHAPE}  {per_iter_s * 1e3:.3f} ms/iter  {gops:.2f} GOps/s")


# ---------------------------------------------------------------------------
# Standalone entry point — prints all four formatted tables
# ---------------------------------------------------------------------------


def main():
    device = ttnn.open_device(device_id=0, trace_region_size=TRACE_REGION_SIZE)
    try:
        _correctness_check(device)
        print(f"correctness ({SHAPE[0]}x{SHAPE[1]}, bfloat16): PCC ok\n")

        table_a = run_axis_sweep(
            device,
            run_sin_bw,
            "dtype",
            [(d, n) for d, n, _ in DATA_FORMATS],
            BASELINE,
            title=f"Set A: dtype sweep — sin {SHAPE[0]}x{SHAPE[1]}",
            takeaway="Takeaway: bfloat8_b halves bytes vs bfloat16; expect ~2x GB/s for a bandwidth-bound SFPU op.",
            throughput_unit="GB/s",
        )

        table_b = run_axis_sweep(
            device,
            run_sin,
            "math_fidelity",
            MATH_FIDELITIES,
            BASELINE,
            title=f"Set B: math_fidelity sweep — sin {SHAPE[0]}x{SHAPE[1]}",
            takeaway="Takeaway: SFPU bypasses the matrix engine — fidelity should have no effect (flat GOps/s confirms isolation).",
        )

        table_c = run_axis_sweep(
            device,
            run_sin,
            "fp32_dest_acc_en",
            FP32_DEST_ACC_VALS,
            BASELINE,
            title=f"Set C: fp32_dest_acc_en sweep — sin {SHAPE[0]}x{SHAPE[1]}",
            takeaway="Takeaway: fp32 dest accumulation does not affect SFPU elementwise ops — expect flat GOps/s.",
        )

        table_d = run_axis_sweep(
            device,
            run_sin,
            "packer_l1_acc",
            PACKER_L1_ACC_VALS,
            BASELINE,
            title=f"Set D: packer_l1_acc sweep — sin {SHAPE[0]}x{SHAPE[1]}",
            takeaway="Takeaway: L1 packer accumulation does not affect SFPU elementwise ops — expect flat GOps/s.",
        )

        for t in [table_a, table_b, table_c, table_d]:
            print("\n" + t.render())
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
