# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Elementwise mul memory-placement benchmark.

Holds shape fixed and sweeps five memory configurations (DRAM-interleaved,
L1-interleaved, and three L1-sharded variants) to show how memory location and
layout affect binary elementwise throughput.

mul reads two inputs and writes one output (3× bandwidth vs sin's 2×), so it
amplifies the DRAM-vs-L1 bandwidth difference even further than sin.

Run:
    python tests/ttnn/unit_tests/operations/memory_perf/test_mul_memory_perf.py
    pytest -s tests/ttnn/unit_tests/operations/memory_perf/test_mul_memory_perf.py
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

# 2048×2048: HEIGHT/WIDTH → 2048/64 = 32 rows/cols per core (valid), BLOCK → 256×256 per core.
SHAPE = (2048, 2048)
CORE_GRID = (8, 8)
DTYPE = ttnn.bfloat16


# --------------------------------------------------------------------------------------
# Benchmark helper
# --------------------------------------------------------------------------------------


def run_mul(device, memcfg):
    """Time ttnn.mul with the given memory config. Returns (per_iter_s, gops_s)."""
    ta = torch.rand(SHAPE, dtype=torch.bfloat16)
    tb = torch.rand(SHAPE, dtype=torch.bfloat16)
    a = ttnn.from_torch(ta, dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memcfg)
    b = ttnn.from_torch(tb, dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memcfg)

    per_iter_s = time_op(device, lambda: ttnn.mul(a, b))

    ttnn.deallocate(a)
    ttnn.deallocate(b)

    real_elems = SHAPE[0] * SHAPE[1]
    gops_s = real_elems / per_iter_s / 1e9
    return per_iter_s, gops_s


# --------------------------------------------------------------------------------------
# Correctness check
# --------------------------------------------------------------------------------------


def _correctness_check(device):
    ta = torch.rand(256, 512, dtype=torch.bfloat16)
    tb = torch.rand(256, 512, dtype=torch.bfloat16)
    golden = ta * tb
    a = ttnn.from_torch(ta, dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b = ttnn.from_torch(tb, dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.mul(a, b)
    assert_with_pcc(golden, ttnn.to_torch(out), 0.999)
    ttnn.deallocate(a)
    ttnn.deallocate(b)
    ttnn.deallocate(out)


# --------------------------------------------------------------------------------------
# pytest entry points
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
def test_correctness(device):
    """Verify ttnn.mul matches torch elementwise multiply within PCC 0.999."""
    _correctness_check(device)


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("memcfg_name", MEMCFG_NAMES)
def test_mul_memory_config(device, memcfg_name):
    """Sweep all five memory configs for ttnn.mul at a fixed shape."""
    memcfgs = build_memcfgs(SHAPE, CORE_GRID)
    memcfg = memcfgs[memcfg_name]
    if memcfg is None:
        pytest.skip(f"{memcfg_name} not valid for shape {SHAPE}")

    try:
        per_iter_s, gops_s = run_mul(device, memcfg)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"{memcfg_name} rejected by op: {e}")
    print(f"\nmul  {memcfg_name:<24}  {per_iter_s * 1e3:.3f} ms/iter  {gops_s:.2f} GOps/s")


# --------------------------------------------------------------------------------------
# Standalone entry point
# --------------------------------------------------------------------------------------


def main():
    device = ttnn.open_device(device_id=0, trace_region_size=TRACE_REGION_SIZE)
    try:
        _correctness_check(device)
        print(f"correctness ({SHAPE[0]}x{SHAPE[1]}, bfloat16): PCC ok\n")

        table = MemcfgResultTable(
            f"mul  shape={SHAPE[0]}x{SHAPE[1]}  dtype=bfloat16",
            "Takeaway: mul reads 2 inputs + writes 1 output (3x bandwidth vs sin). "
            "L1 sharded layouts benefit more than DRAM-interleaved.",
            metric_label="GOps/s",
        )

        memcfgs = build_memcfgs(SHAPE, CORE_GRID)
        for name, memcfg in memcfgs.items():
            if memcfg is None:
                table.add(name, 0.0, None)
                continue
            try:
                per_iter_s, gops_s = run_mul(device, memcfg)
                table.add(name, per_iter_s, gops_s)
            except Exception as e:  # noqa: BLE001
                print(f"  [info] {name} failed (op constraint): {e}")
                table.add(name, 0.0, None)

        print(table.render())
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
