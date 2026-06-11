# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared utilities for the precision/data-format efficiency benchmark suite.

Each test file in this package measures throughput of one op across four independent
axes: dtype, math_fidelity, fp32_dest_acc_en, packer_l1_acc. This module provides
common timing helpers, sweep constants, and table formatting so each test file stays
concise and the methodology stays consistent.
"""

import time

import ttnn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TILE = 32
TRACE_REGION_SIZE = 23887872

# (ttnn_dtype, label, bytes_per_element)
DATA_FORMATS = [
    (ttnn.bfloat16, "bfloat16", 2),
    (ttnn.bfloat8_b, "bfloat8_b", 1),
    (ttnn.bfloat4_b, "bfloat4_b", 0.5),
    (ttnn.float32, "float32", 4),
]

# (ttnn_math_fidelity, label)
MATH_FIDELITIES = [
    (ttnn.MathFidelity.LoFi, "LoFi"),
    (ttnn.MathFidelity.HiFi2, "HiFi2"),
    (ttnn.MathFidelity.HiFi3, "HiFi3"),
    (ttnn.MathFidelity.HiFi4, "HiFi4"),
]

FP32_DEST_ACC_VALS = [(False, "False"), (True, "True")]
PACKER_L1_ACC_VALS = [(False, "False"), (True, "True")]

BASELINE = dict(
    dtype=ttnn.bfloat16,
    math_fidelity=ttnn.MathFidelity.HiFi2,
    fp32_dest_acc_en=False,
    packer_l1_acc=False,
)

# ---------------------------------------------------------------------------
# Timing helpers (same methodology as test_matmul_shape_efficiency.py)
# ---------------------------------------------------------------------------


def _time_with_trace(device, op_fn, warmup_iters, num_iters):
    op_fn()
    ttnn.synchronize_device(device)

    def capture(n):
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        outputs = [op_fn() for _ in range(n)]
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.synchronize_device(device)
        for t in outputs:
            ttnn.deallocate(t)
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
    try:
        return _time_with_trace(device, op_fn, warmup_iters, num_iters)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] trace timing unavailable ({e}); falling back to eager timing")
        return _time_without_trace(device, op_fn, num_iters)


# ---------------------------------------------------------------------------
# Compute kernel config builder
# ---------------------------------------------------------------------------


def make_compute_config(device, math_fidelity=None, fp32_dest_acc_en=False, packer_l1_acc=False):
    if math_fidelity is None:
        math_fidelity = BASELINE["math_fidelity"]
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=packer_l1_acc,
    )


# ---------------------------------------------------------------------------
# Result table
# ---------------------------------------------------------------------------


class PrecisionResultTable:
    """Formats one precision-sweep table.

    Each row is (axis_label, per_iter_s, throughput, skip_reason).
    throughput is GOps/s for SFPU ops, TFLOPS for matrix ops.
    """

    def __init__(self, title, takeaway, throughput_unit="GOps/s"):
        self.title = title
        self.takeaway = takeaway
        self.throughput_unit = throughput_unit
        self.rows = []  # (label, per_iter_s, throughput, skip_reason)

    def add(self, label, per_iter_s, throughput):
        self.rows.append((label, per_iter_s, throughput, None))

    def add_skip(self, label, reason):
        self.rows.append((label, None, None, reason))

    def render(self):
        unit = self.throughput_unit
        valid = [r for r in self.rows if r[2] is not None]
        best = max((r[2] for r in valid), default=0.0)

        hdr = f"{'value':>14} {'ms/iter':>10} {unit:>12} {'% of best':>11}"
        sep = "-" * len(hdr)
        lines = [self.title, self.takeaway, sep, hdr, sep]
        for label, t, throughput, skip in self.rows:
            if skip is not None:
                lines.append(f"{label:>14}  [skip] {skip}")
            else:
                pct = throughput / best if best > 0 else float("nan")
                lines.append(f"{label:>14} {t * 1e3:>10.3f} {throughput:>12.2f} {pct:>10.1%}")
        lines.append(sep)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Generic axis-sweep driver
# ---------------------------------------------------------------------------


def run_axis_sweep(device, op_runner, axis_name, axis_values, baseline, title, takeaway, throughput_unit="GOps/s"):
    """Sweep one axis at a time, holding the other three at baseline.

    op_runner(device, dtype, math_fidelity, fp32_dest_acc_en, packer_l1_acc) -> (per_iter_s, throughput)

    Returns a populated PrecisionResultTable.
    """
    table = PrecisionResultTable(title, takeaway, throughput_unit)
    cfg = dict(baseline)

    for val, label in axis_values:
        cfg[axis_name] = val
        try:
            per_iter_s, throughput = op_runner(
                device,
                cfg["dtype"],
                cfg["math_fidelity"],
                cfg["fp32_dest_acc_en"],
                cfg["packer_l1_acc"],
            )
            table.add(label, per_iter_s, throughput)
        except Exception as e:  # noqa: BLE001
            short = str(e).splitlines()[0][:80]
            print(f"  [skip] {axis_name}={label}: {short}")
            table.add_skip(label, short)
        finally:
            cfg[axis_name] = baseline[axis_name]

    return table
