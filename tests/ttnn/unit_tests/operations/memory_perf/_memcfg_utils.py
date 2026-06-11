# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared utilities for the memory-perf benchmark suite.

All four test files (sin, mul, matmul, attention) import from here so the
timing, config-building, and table-rendering logic stays in one place.
"""

import time
from collections import OrderedDict

import ttnn

TRACE_REGION_SIZE = 23887872
TILE_DIM = 32  # bf16 tile side length in elements

# Five memory configurations swept by every benchmark.
MEMCFG_NAMES = [
    "dram_interleaved",
    "l1_interleaved",
    "l1_height_sharded",
    "l1_width_sharded",
    "l1_block_sharded",
]


# --------------------------------------------------------------------------------------
# Timing helpers (same two-trace subtraction trick as the shape-efficiency demos)
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
    """Mean per-iteration seconds. Falls back to eager timing when trace is unavailable."""
    try:
        return _time_with_trace(device, op_fn, warmup_iters, num_iters)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] trace timing unavailable ({e}); falling back to eager timing")
        return _time_without_trace(device, op_fn, num_iters)


# --------------------------------------------------------------------------------------
# Memory-config factory
# --------------------------------------------------------------------------------------


def _shard_valid(shape, core_grid, strategy):
    """Return True iff the per-core shard dimensions are ≥ TILE_DIM and tile-aligned."""
    rows, cols = shape[-2], shape[-1]
    grid_y, grid_x = core_grid
    if strategy == ttnn.ShardStrategy.HEIGHT:
        num_cores = grid_y * grid_x
        shard_h = (rows + num_cores - 1) // num_cores
        return shard_h >= TILE_DIM and shard_h % TILE_DIM == 0
    if strategy == ttnn.ShardStrategy.WIDTH:
        num_cores = grid_y * grid_x
        shard_w = (cols + num_cores - 1) // num_cores
        return shard_w >= TILE_DIM and shard_w % TILE_DIM == 0
    if strategy == ttnn.ShardStrategy.BLOCK:
        shard_h = (rows + grid_y - 1) // grid_y
        shard_w = (cols + grid_x - 1) // grid_x
        return shard_h >= TILE_DIM and shard_h % TILE_DIM == 0 and shard_w >= TILE_DIM and shard_w % TILE_DIM == 0
    return True


def build_memcfgs(shape, core_grid=(8, 8)):
    """Return OrderedDict[name -> ttnn.MemoryConfig | None] for the five configs.

    shape is the 2-D logical tensor shape (rows, cols) as seen by the sharding API.
    Returns None for any config that cannot be validly constructed for this shape
    (e.g. shard dimension not tile-aligned after dividing across the core grid).
    Tests should call pytest.skip() when they receive None.
    """
    configs = OrderedDict()
    configs["dram_interleaved"] = ttnn.DRAM_MEMORY_CONFIG
    configs["l1_interleaved"] = ttnn.L1_MEMORY_CONFIG

    grid_y, grid_x = core_grid
    ttnn_core_grid = ttnn.CoreGrid(y=grid_y, x=grid_x)

    for name, strategy in (
        ("l1_height_sharded", ttnn.ShardStrategy.HEIGHT),
        ("l1_width_sharded", ttnn.ShardStrategy.WIDTH),
        ("l1_block_sharded", ttnn.ShardStrategy.BLOCK),
    ):
        if not _shard_valid(shape, core_grid, strategy):
            print(f"  [info] {name} skipped: shard not tile-aligned for shape {shape} on {core_grid} grid")
            configs[name] = None
            continue
        try:
            cfg = ttnn.create_sharded_memory_config(
                shape=shape,
                core_grid=ttnn_core_grid,
                strategy=strategy,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=False,
            )
            configs[name] = cfg
        except Exception as e:  # noqa: BLE001
            print(f"  [info] {name} not valid for shape {shape}: {e}")
            configs[name] = None

    return configs


# --------------------------------------------------------------------------------------
# Result table
# --------------------------------------------------------------------------------------


class MemcfgResultTable:
    """Collects (config_name, per_iter_s, metric) rows and prints a comparison table.

    metric_label: e.g. "GOps/s" or "TFLOPS"
    % of best is normalised to the fastest row so it's meaningful on any arch.
    """

    def __init__(self, title, takeaway, metric_label="GOps/s"):
        self.title = title
        self.takeaway = takeaway
        self.metric_label = metric_label
        self.rows = []  # (name, per_iter_s, metric)

    def add(self, name, per_iter_s, metric):
        self.rows.append((name, per_iter_s, metric))

    def render(self):
        best = max((r[2] for r in self.rows if r[2] is not None), default=0.0)
        ml = self.metric_label
        hdr = f"{'memory config':>24} {'ms/iter':>10} {ml:>12} {'% of best':>11}"
        sep = "-" * len(hdr)
        lines = [self.title, self.takeaway, sep, hdr, sep]
        for name, t, metric in self.rows:
            if metric is None:
                lines.append(f"{name:>24} {'(skipped)':>10} {'':>12} {'':>11}")
            else:
                pct = metric / best if best > 0 else float("nan")
                lines.append(f"{name:>24} {t * 1e3:>10.3f} {metric:>12.2f} {pct:>10.1%}")
        lines.append(sep)
        return "\n".join(lines)
