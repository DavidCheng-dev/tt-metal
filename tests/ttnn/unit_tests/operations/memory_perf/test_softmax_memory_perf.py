# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Softmax memory-placement benchmark.

Holds shape fixed and sweeps five memory configurations (DRAM-interleaved,
L1-interleaved, and three L1-sharded variants) to show how memory location and
layout affect softmax throughput.

softmax performs a row-wise reduction (exp → row-sum → divide), so height-sharding
is the natural fit: each core owns a private set of rows and the reduction stays
local.  Width- and block-sharded layouts spread a single row across cores, which
forces cross-core communication for the reduction; those configs may be rejected by
the op.  This makes softmax a richer showcase than sin (pure bandwidth) or mul
(binary bandwidth) because layout constraints, not just bandwidth, determine which
configs survive.

Run:
    python tests/ttnn/unit_tests/operations/memory_perf/test_softmax_memory_perf.py
    pytest -s tests/ttnn/unit_tests/operations/memory_perf/test_softmax_memory_perf.py
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
# Softmax is row-wise (dim=-1): height-sharding keeps each row on one core (reduction local).
# Width- and block-sharding split a row across cores; the op may reject those configs.
SHAPE = (2048, 2048)
CORE_GRID = (8, 8)
DTYPE = ttnn.bfloat16
DIM = -1  # row-wise softmax


# --------------------------------------------------------------------------------------
# Benchmark helper
# --------------------------------------------------------------------------------------


def run_softmax(device, memcfg):
    """Time ttnn.softmax with the given memory config. Returns (per_iter_s, gops_s)."""
    t = torch.rand(SHAPE, dtype=torch.bfloat16)
    x = ttnn.from_torch(t, dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memcfg)

    per_iter_s = time_op(device, lambda: ttnn.softmax(x, dim=DIM))

    ttnn.deallocate(x)

    real_elems = SHAPE[0] * SHAPE[1]
    gops_s = real_elems / per_iter_s / 1e9
    return per_iter_s, gops_s


# --------------------------------------------------------------------------------------
# Correctness check
# --------------------------------------------------------------------------------------


def _correctness_check(device):
    t = torch.rand(256, 512, dtype=torch.bfloat16)
    golden = torch.softmax(t, dim=-1)
    x = ttnn.from_torch(t, dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.softmax(x, dim=DIM)
    assert_with_pcc(golden, ttnn.to_torch(out), 0.997)
    ttnn.deallocate(x)
    ttnn.deallocate(out)


# --------------------------------------------------------------------------------------
# pytest entry points
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
def test_correctness(device):
    """Verify ttnn.softmax matches torch.softmax within PCC 0.999."""
    _correctness_check(device)


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("memcfg_name", MEMCFG_NAMES)
def test_softmax_memory_config(device, memcfg_name):
    """Sweep all five memory configs for ttnn.softmax at a fixed shape."""
    memcfgs = build_memcfgs(SHAPE, CORE_GRID)
    memcfg = memcfgs[memcfg_name]
    if memcfg is None:
        pytest.skip(f"{memcfg_name} not valid for shape {SHAPE}")

    try:
        per_iter_s, gops_s = run_softmax(device, memcfg)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"{memcfg_name} rejected by op: {e}")
    print(f"\nsoftmax  {memcfg_name:<24}  {per_iter_s * 1e3:.3f} ms/iter  {gops_s:.2f} GOps/s")


# --------------------------------------------------------------------------------------
# Standalone entry point
# --------------------------------------------------------------------------------------


def main():
    device = ttnn.open_device(device_id=0, trace_region_size=TRACE_REGION_SIZE)
    try:
        _correctness_check(device)
        print(f"correctness ({SHAPE[0]}x{SHAPE[1]}, bfloat16): PCC ok\n")

        table = MemcfgResultTable(
            f"softmax  shape={SHAPE[0]}x{SHAPE[1]}  dim={DIM}  dtype=bfloat16",
            "Takeaway: softmax is a row-wise reduction — height-sharding keeps each "
            "row on one core (local reduction). Width/block sharding may be rejected "
            "because the reduction crosses shard boundaries.",
            metric_label="GOps/s",
        )

        memcfgs = build_memcfgs(SHAPE, CORE_GRID)
        for name, memcfg in memcfgs.items():
            if memcfg is None:
                table.add(name, 0.0, None)
                continue
            try:
                per_iter_s, gops_s = run_softmax(device, memcfg)
                table.add(name, per_iter_s, gops_s)
            except Exception as e:  # noqa: BLE001
                print(f"  [info] {name} failed (op constraint): {e}")
                table.add(name, 0.0, None)

        print(table.render())
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
