# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Matmul memory-placement benchmark.

Holds shape fixed and sweeps five memory configurations to show how input
memory placement affects matrix-engine throughput.

Two shapes are benchmarked:
  - Prefill (2048×2048×2048): compute-bound, large batch.
  - Decode  (32×4096×4096):   memory-bandwidth-bound, single-tile batch.
    M=32 represents one padded decode token (or a batch of 32).  The weight
    matrix B is 32 MB — bandwidth dominates, TFLOPS utilisation is < 1%.

Both A and B share the same memory configuration so the comparison is
all-DRAM vs all-L1.  L1_MEMORY_CONFIG is interleaved, so the matmul op
accepts it for B.  Sharded configs are rejected by the op for B and appear
as (skipped).  Putting B in L1 is the dominant effect for decode throughput:
B (32 MB) dwarfs A (256 KB), so moving only A to L1 changes little.

Run:
    python tests/ttnn/unit_tests/operations/memory_perf/test_matmul_memory_perf.py
    pytest -s tests/ttnn/unit_tests/operations/memory_perf/test_matmul_memory_perf.py
"""

import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.unit_tests.operations.memory_perf._memcfg_utils import (
    TRACE_REGION_SIZE,
    MEMCFG_NAMES,
    build_memcfgs,
    time_op,
    MemcfgResultTable,
)

# Prefill: 2048×2048 — HEIGHT/WIDTH → 32 rows/cols per core (valid), BLOCK → 256×256 per core.
M = K = N = 2048

# Decode: M=32 (one tile row, padded single token or batch-32); large K/N typical of LLM FFN.
# Sharding of A (32×4096): HEIGHT → 32/64 = 0.5 per core (invalid);
# BLOCK → 32/8 = 4 rows per core (invalid); WIDTH → 4096/64 = 64 per core (valid geometry
# but rejected by the matmul op for A).  Only interleaved configs run.
M_DEC, K_DEC, N_DEC = 32, 4096, 4096

CORE_GRID = (8, 8)
DTYPE = ttnn.bfloat16

# Same LoFi config as test_matmul_shape_efficiency.py so only memory varies.
COMPUTE_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=False,
)


# --------------------------------------------------------------------------------------
# Benchmark helper
# --------------------------------------------------------------------------------------


def run_matmul(device, memcfg, m=M, k=K, n=N):
    """Time ttnn.matmul (m×k @ k×n) with memcfg applied to both A and B.

    Both inputs share the same memory configuration so the comparison is
    all-DRAM vs all-L1.  For sharded configs, the matmul op rejects sharded B
    (it requires B to be INTERLEAVED); those cases surface as (skipped).
    Returns (per_iter_s, tflops).
    """
    ta = torch.rand(m, k, dtype=torch.bfloat16)
    tb = torch.rand(k, n, dtype=torch.bfloat16)
    a = ttnn.from_torch(ta, dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memcfg)
    b = ttnn.from_torch(tb, dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memcfg)

    per_iter_s = time_op(device, lambda: ttnn.matmul(a, b, compute_kernel_config=COMPUTE_KERNEL_CONFIG))

    ttnn.deallocate(a)
    ttnn.deallocate(b)

    tflops = (2 * m * k * n) / per_iter_s / 1e12
    return per_iter_s, tflops


# --------------------------------------------------------------------------------------
# Correctness check
# --------------------------------------------------------------------------------------


def _correctness_check(device):
    m = k = n = 256
    ta = torch.rand(m, k, dtype=torch.bfloat16)
    tb = torch.rand(k, n, dtype=torch.bfloat16)
    golden = ta.float() @ tb.float()
    a = ttnn.from_torch(ta, dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b = ttnn.from_torch(tb, dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.matmul(a, b, compute_kernel_config=COMPUTE_KERNEL_CONFIG)
    assert_with_pcc(golden, ttnn.to_torch(out), 0.99)
    ttnn.deallocate(a)
    ttnn.deallocate(b)
    ttnn.deallocate(out)


# --------------------------------------------------------------------------------------
# pytest entry points
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
def test_correctness(device):
    """Verify ttnn.matmul matches torch matmul within PCC 0.99."""
    _correctness_check(device)


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("memcfg_name", MEMCFG_NAMES)
def test_matmul_memory_config(device, memcfg_name):
    """Sweep all five memory configs for A in ttnn.matmul; B stays DRAM-interleaved."""
    memcfgs = build_memcfgs((M, K), CORE_GRID)
    memcfg = memcfgs[memcfg_name]
    if memcfg is None:
        pytest.skip(f"{memcfg_name} not valid for shape ({M},{K})")

    try:
        per_iter_s, tflops = run_matmul(device, memcfg)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"{memcfg_name} rejected by op: {e}")
    print(f"\nmatmul  {memcfg_name:<24}  {per_iter_s * 1e3:.3f} ms/iter  {tflops:.4f} TFLOPS")


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("memcfg_name", MEMCFG_NAMES)
def test_matmul_decode_memory_config(device, memcfg_name):
    """Decode-shape sweep: M=32 (single-tile batch), K=4096, N=4096; B stays DRAM-interleaved.

    With M=32 the activation tensor A is (32, 4096).  HEIGHT and BLOCK sharding
    require ≥ 32 rows per core (32/64 < 32 and 32/8 < 32 respectively) and are
    skipped at geometry.  WIDTH sharding produces valid geometry but is typically
    rejected by the matmul op.  Only interleaved configs are expected to run.
    """
    memcfgs = build_memcfgs((M_DEC, K_DEC), CORE_GRID)
    memcfg = memcfgs[memcfg_name]
    if memcfg is None:
        pytest.skip(f"{memcfg_name} not valid for decode shape ({M_DEC},{K_DEC})")

    try:
        per_iter_s, tflops = run_matmul(device, memcfg, m=M_DEC, k=K_DEC, n=N_DEC)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"{memcfg_name} rejected by op: {e}")
    print(f"\nmatmul_decode  {memcfg_name:<24}  {per_iter_s * 1e3:.3f} ms/iter  {tflops:.4f} TFLOPS")


# --------------------------------------------------------------------------------------
# Standalone entry point
# --------------------------------------------------------------------------------------


def main():
    device = ttnn.open_device(device_id=0, trace_region_size=TRACE_REGION_SIZE)
    try:
        _correctness_check(device)
        print(f"correctness ({M}x{K}x{N}, bfloat16 LoFi): PCC ok\n")

        # ── Prefill benchmark ──────────────────────────────────────────────────
        table = MemcfgResultTable(
            f"matmul  shape={M}x{K}x{N}  dtype=bfloat16  fidelity=LoFi  (A and B same memcfg)",
            "Takeaway: both A and B share the same memcfg. "
            "Prefill is compute-bound so DRAM vs L1 matters less; sharded B is rejected by the op.",
            metric_label="TFLOPS",
        )

        memcfgs = build_memcfgs((M, K), CORE_GRID)
        for name, memcfg in memcfgs.items():
            if memcfg is None:
                table.add(name, 0.0, None)
                continue
            try:
                per_iter_s, tflops = run_matmul(device, memcfg)
                table.add(name, per_iter_s, tflops)
            except Exception as e:  # noqa: BLE001
                print(f"  [info] {name} failed (op constraint): {e}")
                table.add(name, 0.0, None)

        print(table.render())

        # ── Decode benchmark ───────────────────────────────────────────────────
        print()
        table_dec = MemcfgResultTable(
            f"matmul decode  shape={M_DEC}x{K_DEC}x{N_DEC}  dtype=bfloat16  fidelity=LoFi  (A and B same memcfg)",
            "Takeaway: B=32 MB dominates bandwidth. Moving both A and B from DRAM to L1 shows the "
            "true memory benefit — only interleaved configs run for this tiny M.",
            metric_label="TFLOPS",
        )

        memcfgs_dec = build_memcfgs((M_DEC, K_DEC), CORE_GRID)
        for name, memcfg in memcfgs_dec.items():
            if memcfg is None:
                table_dec.add(name, 0.0, None)
                continue
            try:
                per_iter_s, tflops = run_matmul(device, memcfg, m=M_DEC, k=K_DEC, n=N_DEC)
                table_dec.add(name, per_iter_s, tflops)
            except Exception as e:  # noqa: BLE001
                print(f"  [info] {name} failed (op constraint): {e}")
                table_dec.add(name, 0.0, None)

        print(table_dec.render())
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
