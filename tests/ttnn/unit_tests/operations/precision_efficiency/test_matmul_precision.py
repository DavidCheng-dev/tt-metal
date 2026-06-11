# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Matmul precision/data-format efficiency benchmark.

Companion to test_matmul_shape_efficiency.py. Where that demo fixes dtype/fidelity
and sweeps shape, this demo fixes a grid-saturating square shape (2048×2048×2048) and
sweeps all four precision axes:

  A. dtype          — bfloat16 / bfloat8_b / bfloat4_b / float32
                       (smaller dtype → less DRAM traffic → higher TFLOPS)
  B. math_fidelity  — LoFi / HiFi2 / HiFi3 / HiFi4
                       (more fidelity → more matrix-engine cycles → lower TFLOPS)
  C. fp32_dest_acc_en — False / True
                       (True uses fp32 in the destination register → fewer dest tiles
                        fit → potentially lower TFLOPS; improved numerical stability)
  D. packer_l1_acc    — False / True
                       (True accumulates in L1 between output packer passes → affects
                        write-back throughput for large K)

Run:
    python tests/ttnn/unit_tests/operations/precision_efficiency/test_matmul_precision.py
    pytest -s tests/ttnn/unit_tests/operations/precision_efficiency/test_matmul_precision.py
"""

import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.unit_tests.operations.precision_efficiency.precision_utils import (
    TRACE_REGION_SIZE,
    DATA_FORMATS,
    MATH_FIDELITIES,
    FP32_DEST_ACC_VALS,
    PACKER_L1_ACC_VALS,
    BASELINE,
    make_compute_config,
    run_axis_sweep,
)

M = K = N = 2048


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


def run_matmul(device, dtype, math_fidelity, fp32_dest_acc_en, packer_l1_acc):
    """Time a (M, K) x (K, N) matmul at the given precision configuration.

    Returns (per_iter_s, tflops) where TFLOPS uses the real (unpadded) MKN counts
    so tile-padding waste shows up as a throughput drop.
    """
    cfg = make_compute_config(
        device, math_fidelity=math_fidelity, fp32_dest_acc_en=fp32_dest_acc_en, packer_l1_acc=packer_l1_acc
    )
    a = ttnn.rand((M, K), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
    b = ttnn.rand((K, N), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)

    from tests.ttnn.unit_tests.operations.precision_efficiency.precision_utils import time_op

    per_iter_s = time_op(device, lambda: ttnn.matmul(a, b, compute_kernel_config=cfg))

    ttnn.deallocate(a)
    ttnn.deallocate(b)

    tflops = (2 * M * K * N) / per_iter_s / 1e12
    return per_iter_s, tflops


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------


def _correctness_check(device):
    m = k = n = 256
    ta = torch.rand(m, k, dtype=torch.bfloat16)
    tb = torch.rand(k, n, dtype=torch.bfloat16)
    golden = ta.float() @ tb.float()
    cfg = make_compute_config(device)
    a = ttnn.from_torch(ta, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(tb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.matmul(a, b, compute_kernel_config=cfg)
    assert_with_pcc(golden, ttnn.to_torch(out), 0.99)
    ttnn.deallocate(a)
    ttnn.deallocate(b)
    ttnn.deallocate(out)


# ---------------------------------------------------------------------------
# pytest entry points
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
def test_correctness(device):
    """Verify ttnn.matmul output matches torch.matmul within PCC 0.99."""
    _correctness_check(device)


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("dtype,dtype_name,_bpe", DATA_FORMATS)
def test_axis_dtype(device, dtype, dtype_name, _bpe):
    """Set A: sweep dtype — smaller dtype reduces DRAM traffic → higher TFLOPS."""
    per_iter_s, tflops = run_matmul(
        device, dtype, **{k: BASELINE[k] for k in ("math_fidelity", "fp32_dest_acc_en", "packer_l1_acc")}
    )
    print(f"\nA  dtype={dtype_name}  {M}x{K}x{N}  {per_iter_s * 1e3:.3f} ms/iter  {tflops:.2f} TFLOPS")


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("fidelity,fidelity_name", MATH_FIDELITIES)
def test_axis_math_fidelity(device, fidelity, fidelity_name):
    """Set B: sweep math_fidelity — higher fidelity uses more matrix-engine cycles → lower TFLOPS."""
    per_iter_s, tflops = run_matmul(
        device, BASELINE["dtype"], fidelity, BASELINE["fp32_dest_acc_en"], BASELINE["packer_l1_acc"]
    )
    print(f"\nB  fidelity={fidelity_name}  {M}x{K}x{N}  {per_iter_s * 1e3:.3f} ms/iter  {tflops:.2f} TFLOPS")


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("fp32_dest_acc_en,label", FP32_DEST_ACC_VALS)
def test_axis_fp32_dest_acc_en(device, fp32_dest_acc_en, label):
    """Set C: sweep fp32_dest_acc_en — fp32 dest halves the dest tile capacity → potential TFLOPS drop."""
    per_iter_s, tflops = run_matmul(
        device, BASELINE["dtype"], BASELINE["math_fidelity"], fp32_dest_acc_en, BASELINE["packer_l1_acc"]
    )
    print(f"\nC  fp32_dest_acc_en={label}  {M}x{K}x{N}  {per_iter_s * 1e3:.3f} ms/iter  {tflops:.2f} TFLOPS")


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("packer_l1_acc,label", PACKER_L1_ACC_VALS)
def test_axis_packer_l1_acc(device, packer_l1_acc, label):
    """Set D: sweep packer_l1_acc — L1 accumulation reduces writes back to DRAM for large K."""
    per_iter_s, tflops = run_matmul(
        device, BASELINE["dtype"], BASELINE["math_fidelity"], BASELINE["fp32_dest_acc_en"], packer_l1_acc
    )
    print(f"\nD  packer_l1_acc={label}  {M}x{K}x{N}  {per_iter_s * 1e3:.3f} ms/iter  {tflops:.2f} TFLOPS")


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------


def main():
    device = ttnn.open_device(device_id=0, trace_region_size=TRACE_REGION_SIZE)
    try:
        _correctness_check(device)
        print(f"correctness (256x256x256, bfloat16): PCC ok\n")

        table_a = run_axis_sweep(
            device,
            run_matmul,
            "dtype",
            [(d, n) for d, n, _ in DATA_FORMATS],
            BASELINE,
            title=f"Set A: dtype sweep — matmul {M}x{K}x{N}",
            takeaway="Takeaway: smaller dtype → less DRAM traffic per tile → higher achieved TFLOPS.",
            throughput_unit="TFLOPS",
        )

        table_b = run_axis_sweep(
            device,
            run_matmul,
            "math_fidelity",
            MATH_FIDELITIES,
            BASELINE,
            title=f"Set B: math_fidelity sweep — matmul {M}x{K}x{N}",
            takeaway="Takeaway: LoFi is fastest; each step toward HiFi4 adds matrix-engine cycles → lower TFLOPS.",
            throughput_unit="TFLOPS",
        )

        table_c = run_axis_sweep(
            device,
            run_matmul,
            "fp32_dest_acc_en",
            FP32_DEST_ACC_VALS,
            BASELINE,
            title=f"Set C: fp32_dest_acc_en sweep — matmul {M}x{K}x{N}",
            takeaway="Takeaway: fp32 dest halves the dest tile capacity, often causing a measurable TFLOPS drop.",
            throughput_unit="TFLOPS",
        )

        table_d = run_axis_sweep(
            device,
            run_matmul,
            "packer_l1_acc",
            PACKER_L1_ACC_VALS,
            BASELINE,
            title=f"Set D: packer_l1_acc sweep — matmul {M}x{K}x{N}",
            takeaway="Takeaway: L1 accumulation in the packer can improve throughput for large-K matmuls by reducing DRAM writes.",
            throughput_unit="TFLOPS",
        )

        for t in [table_a, table_b, table_c, table_d]:
            print("\n" + t.render())
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
