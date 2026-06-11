# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared helpers for the MeshDevice scale-up demos.

These demos show how a single TT-NN program scales from 1 chip up to 8 chips on a
Wormhole Galaxy. The helpers here are intentionally op-agnostic: opening a (1, N) mesh,
trace-based timing of a single op, a PCC golden check, and a small table that turns a set
of per-iteration timings into a speedup / scaling-efficiency report.
"""

import contextlib
import os
import sys
import time

# Make the repo root importable so this works both under pytest and as a standalone
# `python tests/ttnn/distributed/scale_up_demos/demo_xx.py` script (where only the script's
# own directory is on sys.path).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

# Mesh sizes the demos sweep over: 1 -> 2 -> 4 -> 8 chips, i.e. MeshShape(1, N).
MESH_SIZES = [1, 2, 4, 8]

# Trace region large enough for the small demo workloads. Trace captures the op program
# once so timing excludes host-side dispatch overhead.
DEFAULT_TRACE_REGION_SIZE = 23887872


def available_mesh_sizes():
    """MESH_SIZES filtered to what the current system actually has."""
    num_devices = ttnn.get_num_devices()
    return [n for n in MESH_SIZES if n <= num_devices]


@contextlib.contextmanager
def demo_mesh(n, trace_region_size=DEFAULT_TRACE_REGION_SIZE, fabric_config=None):
    """Open a mesh for the standalone (``python demo_xx.py``) run path.

    ``n`` may be an int (opens ``MeshShape(1, n)`` — the default line layout used by
    demos 1-4) or a ``(rows, cols)`` tuple for demos that need an explicit shape


    The pytest path uses the repo's ``mesh_device`` fixture instead.

    Pass ``fabric_config`` (e.g. ``ttnn.FabricConfig.FABRIC_1D``) for demos that issue
    on-device collectives like ``ttnn.all_gather`` — fabric must be enabled BEFORE the
    mesh is opened, and reset to DISABLED after it closes.
    """
    shape = ttnn.MeshShape(*n) if isinstance(n, tuple) else ttnn.MeshShape(1, n)
    if fabric_config is not None:
        ttnn.set_fabric_config(fabric_config)
    mesh_device = ttnn.open_mesh_device(shape, trace_region_size=trace_region_size)
    try:
        yield mesh_device
    finally:
        ttnn.close_mesh_device(mesh_device)
        if fabric_config is not None:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _time_with_trace(mesh_device, op_fn, warmup_iters, num_iters):
    """Trace-capture timing. Mirrors tests/ttnn/.../test_ag_rs_llama_prefill_TG.py.

    The op is captured ``warmup_iters`` times into one trace and ``num_iters`` times into
    another. Subtracting the warmup wall-time from the main wall-time cancels the fixed
    per-trace launch overhead, leaving (num_iters - warmup_iters) op executions.
    """
    # Compile run (kernels compiled + cached) so capture/exec are pure execution.
    op_fn()
    ttnn.synchronize_device(mesh_device)

    def capture(n_iters):
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for _ in range(n_iters):
            # Output is discarded each iter; refcount frees it so the trace region only
            # needs room for a single output buffer, not n_iters of them.
            op_fn()
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        return trace_id

    warmup_trace = capture(warmup_iters)
    main_trace = capture(num_iters)

    t0 = time.perf_counter()
    ttnn.execute_trace(mesh_device, warmup_trace, blocking=False)
    ttnn.synchronize_device(mesh_device)
    warmup_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    ttnn.execute_trace(mesh_device, main_trace, blocking=False)
    ttnn.synchronize_device(mesh_device)
    main_s = time.perf_counter() - t0

    ttnn.release_trace(mesh_device, warmup_trace)
    ttnn.release_trace(mesh_device, main_trace)

    return (main_s - warmup_s) / (num_iters - warmup_iters)


def _time_without_trace(mesh_device, op_fn, num_iters):
    """Fallback timing without trace (includes host dispatch overhead)."""
    for _ in range(3):
        op_fn()
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(num_iters):
        op_fn()
    ttnn.synchronize_device(mesh_device)
    return (time.perf_counter() - t0) / num_iters


def time_op(mesh_device, op_fn, warmup_iters=10, num_iters=30):
    """Mean per-iteration seconds for ``op_fn`` on ``mesh_device``.

    ``op_fn`` runs the op once using device tensors captured in its closure and returns
    (or discards) the output. Falls back to non-trace timing if trace capture is
    unavailable (e.g. no trace_region_size).
    """
    try:
        return _time_with_trace(mesh_device, op_fn, warmup_iters, num_iters)
    except Exception as e:  # noqa: BLE001 - demo: degrade gracefully to non-trace timing
        print(f"  [warn] trace timing unavailable ({e}); falling back to eager timing")
        return _time_without_trace(mesh_device, op_fn, num_iters)


def pcc_check(golden, tt_torch, pcc=0.99):
    """Assert the TT result matches the torch/CPU golden within a PCC threshold."""
    return assert_with_pcc(golden, tt_torch, pcc)


class SpeedupTable:
    """Collects per-mesh-size timings and prints a speedup / efficiency table.

    Speedup is measured against the 1-chip baseline (strong scaling): the *total* problem
    size is fixed, so as we shard it across more chips each chip does less work and the
    per-iteration wall time should fall ~N x. Efficiency = speedup / N (1.0 is ideal linear
    scaling; it drops as communication / dispatch overhead grows relative to compute).
    """

    def __init__(self, title, work_per_iter=None, work_unit="elem", col0_header="chips"):
        self.title = title
        self.work_per_iter = work_per_iter  # constant total work per iter, for throughput
        self.work_unit = work_unit
        self.col0_header = col0_header  # first-column header ("chips" for 1D, "mesh" for 2D)
        self.entries = []  # list of (n, per_iter_s, label)

    def add(self, n, per_iter_s, label=None):
        """``n`` is the chip count (used for the speedup baseline ordering and efficiency
        denominator). ``label`` optionally overrides the first column's text, e.g. "2x4"."""
        self.entries.append((n, per_iter_s, label))

    def render(self):
        if not self.entries:
            return f"{self.title}: (no data)"
        entries = sorted(self.entries, key=lambda e: (e[0], e[1]))
        baseline = entries[0][1]
        has_tput = self.work_per_iter is not None

        header = f"{self.col0_header:>6} {'time/iter (ms)':>16} {'speedup':>9} {'efficiency':>11}"
        if has_tput:
            header += f" {'throughput (G' + self.work_unit + '/s)':>22}"
        lines = [self.title, "-" * len(header), header, "-" * len(header)]
        for n, t, label in entries:
            speedup = baseline / t if t > 0 else float("nan")
            eff = speedup / n
            col0 = label if label is not None else str(n)
            row = f"{col0:>6} {t * 1e3:>16.3f} {speedup:>8.2f}x {eff:>10.2%}"
            if has_tput:
                gtput = (self.work_per_iter / t) / 1e9
                row += f" {gtput:>22.2f}"
            lines.append(row)
        lines.append("-" * len(header))
        return "\n".join(lines)
