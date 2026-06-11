# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Sin memory-placement benchmark.

Holds shape fixed and sweeps five memory configurations (DRAM-interleaved,
L1-interleaved, and three L1-sharded variants) to show how memory location and
layout affect SFPU elementwise throughput.

sin is bandwidth-bound at the shapes used here, so it is the clearest showcase
of the L1 bandwidth advantage over DRAM.

Run:
    python tests/ttnn/unit_tests/operations/memory_perf/test_sin_memory_perf.py
    pytest -s tests/ttnn/unit_tests/operations/memory_perf/test_sin_memory_perf.py
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

# Fixed shape: 2048×2048 ensures each sharding strategy's per-core shard is ≥ 32 (one tile).
# HEIGHT/WIDTH: 2048 / 64 cores = 32 rows/cols per core (exactly one tile, valid).
# BLOCK: 2048 / 8 = 256 rows and cols per core (eight tiles, valid).
# 2048×2048 bf16 = 8 MB → 128 KB per core at 8×8 = 64 cores, within the ~1.5 MB L1 budget.
SHAPE = (2048, 2048)
CORE_GRID = (8, 8)
DTYPE = ttnn.bfloat16


# --------------------------------------------------------------------------------------
# Benchmark helper
# --------------------------------------------------------------------------------------


def run_sin(device, memcfg):
    """Time ttnn.sin with the given memory config. Returns (per_iter_s, gops_s)."""
    t = torch.rand(SHAPE, dtype=torch.bfloat16)
    x = ttnn.from_torch(t, dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memcfg)

    per_iter_s = time_op(device, lambda: ttnn.sin(x))

    ttnn.deallocate(x)

    real_elems = SHAPE[0] * SHAPE[1]
    gops_s = real_elems / per_iter_s / 1e9
    return per_iter_s, gops_s


# --------------------------------------------------------------------------------------
# Correctness check
# --------------------------------------------------------------------------------------


def _correctness_check(device):
    t = torch.rand(256, 512, dtype=torch.bfloat16) * 6.28
    golden = torch.sin(t)
    x = ttnn.from_torch(t, dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.sin(x)
    assert_with_pcc(golden, ttnn.to_torch(out), 0.999)
    ttnn.deallocate(x)
    ttnn.deallocate(out)


# --------------------------------------------------------------------------------------
# pytest entry points
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
def test_correctness(device):
    """Verify ttnn.sin matches torch.sin within PCC 0.999."""
    _correctness_check(device)


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("memcfg_name", MEMCFG_NAMES)
def test_sin_memory_config(device, memcfg_name):
    """Sweep all five memory configs for ttnn.sin at a fixed shape."""
    memcfgs = build_memcfgs(SHAPE, CORE_GRID)
    memcfg = memcfgs[memcfg_name]
    if memcfg is None:
        pytest.skip(f"{memcfg_name} not valid for shape {SHAPE}")

    try:
        per_iter_s, gops_s = run_sin(device, memcfg)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"{memcfg_name} rejected by op: {e}")
    print(f"\nsin  {memcfg_name:<24}  {per_iter_s * 1e3:.3f} ms/iter  {gops_s:.2f} GOps/s")


# --------------------------------------------------------------------------------------
# Standalone entry point
# --------------------------------------------------------------------------------------


def main():
    device = ttnn.open_device(device_id=0, trace_region_size=TRACE_REGION_SIZE)
    try:
        _correctness_check(device)
        print(f"correctness ({SHAPE[0]}x{SHAPE[1]}, bfloat16): PCC ok\n")

        table = MemcfgResultTable(
            f"sin  shape={SHAPE[0]}x{SHAPE[1]}  dtype=bfloat16",
            "Takeaway: sin is bandwidth-bound — L1 sharded variants should outperform "
            "DRAM-interleaved because L1 offers ~10x higher bandwidth per core.",
            metric_label="GOps/s",
        )

        memcfgs = build_memcfgs(SHAPE, CORE_GRID)
        for name, memcfg in memcfgs.items():
            if memcfg is None:
                table.add(name, 0.0, None)
                continue
            try:
                per_iter_s, gops_s = run_sin(device, memcfg)
                table.add(name, per_iter_s, gops_s)
            except Exception as e:  # noqa: BLE001
                print(f"  [info] {name} failed (op constraint): {e}")
                table.add(name, 0.0, None)

        print(table.render())
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
