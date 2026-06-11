# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Elementwise sin shape-efficiency demo for the Tensix SFPU engine.

Companion to test_matmul_shape_efficiency.py — same measurement methodology applied to
a SFPU (Special Function Processing Unit) unary op instead of the matrix engine.

Key differences between SFPU and matrix engine:
  - SFPU is compute-bound on transcendental ops (sin uses a polynomial approximation)
  - Elementwise ops are memory-bandwidth-bound for large tensors but SFPU-bound when small
  - There is no "M x K x N" shape interaction — only the total element count matters for bandwidth
  - Tile alignment still matters: a non-32-aligned shape pads to the next tile boundary,
    wasting both SFPU cycles and DRAM bandwidth for those extra elements

This demo measures, on real hardware, three characteristics:

  A. Tile alignment - shapes that are NOT multiples of 32 pad up to the next tile,
     so the "real" element count (M*N) under-counts the actual work done.

  B. Total size / bandwidth saturation - small tensors don't saturate DRAM bandwidth
     and are dominated by host dispatch overhead; large tensors approach peak bandwidth.

  C. Data format - bfloat16 vs bfloat8_b changes element size (2 vs 1 byte) and
     therefore throughput in elements/s for the same bandwidth.

Run:
    python tests/ttnn/unit_tests/operations/eltwise/test_sin_shape_efficiency.py
    pytest -s tests/ttnn/unit_tests/operations/eltwise/test_sin_shape_efficiency.py
"""

import time

import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

TILE = 32
TRACE_REGION_SIZE = 23887872

# Set A: fix N=1024, sweep M.  Same M values as matmul Set A so the two demos are comparable.
SET_A_N = 512
SET_A_M = [32, 33, 64, 65, 128, 129, 256, 257, 512, 513, 1024, 1025, 2048, 2049, 4096]

# Set B: square (M=N), sweep from tiny to large to show bandwidth saturation.
SET_B_SQUARE = [64, 128, 256, 512, 1024, 2048, 4096, 4097]

# Set C: data format sweep (fixed large shape to be bandwidth-bound, not dispatch-bound).
SET_C_SHAPE = (2048, 2048)
SET_C_DTYPES = [
    (ttnn.bfloat16, "bfloat16"),
    (ttnn.bfloat8_b, "bfloat8_b"),
]


# --------------------------------------------------------------------------------------
# Timing helpers (identical to test_matmul_shape_efficiency.py)
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
    try:
        return _time_with_trace(device, op_fn, warmup_iters, num_iters)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] trace timing unavailable ({e}); falling back to eager timing")
        return _time_without_trace(device, op_fn, num_iters)


# --------------------------------------------------------------------------------------
# sin benchmark helpers
# --------------------------------------------------------------------------------------


def _padded(x):
    """Ceil-divide to TILE boundary — the actual element count processed by hardware."""
    return -(-x // TILE) * TILE


def run_sin(device, m, n, dtype=ttnn.bfloat16):
    """Time ttnn.sin on an (m, n) tensor.

    Returns (per_iter_s, real_gops, padded_gops) where:
      real_gops    - GOps/s based on the logical (unpadded) element count
      padded_gops  - GOps/s based on the padded (tile-aligned) element count

    The gap between real and padded quantifies the waste from non-tile-aligned shapes.
    """
    x = ttnn.rand((m, n), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)

    per_iter_s = time_op(device, lambda: ttnn.sin(x))

    ttnn.deallocate(x)

    real_elems = m * n
    padded_elems = _padded(m) * _padded(n)

    real_gops = real_elems / per_iter_s / 1e9
    padded_gops = padded_elems / per_iter_s / 1e9
    return per_iter_s, real_gops, padded_gops


def run_sin_bandwidth(device, m, n, dtype=ttnn.bfloat16):
    """Return achieved memory bandwidth (GB/s) for a sin pass.

    sin reads input and writes output: total bytes = 2 * elements * bytes_per_element.
    This converts the SFPU throughput into a bandwidth figure for comparing data formats.
    """
    bytes_per_elem = 2 if dtype == ttnn.bfloat16 else 1
    x = ttnn.rand((m, n), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)

    per_iter_s = time_op(device, lambda: ttnn.sin(x))

    ttnn.deallocate(x)

    padded_elems = _padded(m) * _padded(n)
    total_bytes = 2 * padded_elems * bytes_per_elem  # read + write
    bandwidth_gbs = total_bytes / per_iter_s / 1e9
    return per_iter_s, bandwidth_gbs


# --------------------------------------------------------------------------------------
# Result table
# --------------------------------------------------------------------------------------


class SinResultTable:
    def __init__(self, title, takeaway):
        self.title = title
        self.takeaway = takeaway
        self.rows = []

    def add_shape(self, m, n, per_iter_s, real_gops, padded_gops):
        self.rows.append(("shape", m, n, per_iter_s, real_gops, padded_gops, None, None))

    def add_bw(self, m, n, dtype_name, per_iter_s, bw_gbs):
        self.rows.append(("bw", m, n, per_iter_s, None, None, dtype_name, bw_gbs))

    def render(self):
        mode = self.rows[0][0] if self.rows else "shape"
        lines = [self.title, self.takeaway]

        if mode == "shape":
            best_real = max(r[4] for r in self.rows)
            hdr = f"{'shape (MxN)':>14} {'padded (MxN)':>14} {'ms/iter':>10} {'GOps/s(real)':>14} {'GOps/s(pad)':>13} {'util%':>8}"
            sep = "-" * len(hdr)
            lines += [sep, hdr, sep]
            for _, m, n, t, real, pad, _, _ in self.rows:
                pm, pn = _padded(m), _padded(n)
                util = real / best_real if best_real > 0 else float("nan")
                shape_str = f"{m}x{n}"
                padded_str = f"{pm}x{pn}"
                lines.append(
                    f"{shape_str:>14} {padded_str:>14} {t * 1e3:>10.3f} {real:>14.2f} {pad:>13.2f} {util:>7.1%}"
                )
            lines.append(sep)
        else:
            hdr = f"{'dtype':>12} {'shape (MxN)':>14} {'ms/iter':>10} {'BW (GB/s)':>12}"
            sep = "-" * len(hdr)
            lines += [sep, hdr, sep]
            for _, m, n, t, _, _, dtype_name, bw in self.rows:
                shape_str = f"{m}x{n}"
                lines.append(f"{dtype_name:>12} {shape_str:>14} {t * 1e3:>10.3f} {bw:>12.2f}")
            lines.append(sep)

        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Correctness check
# --------------------------------------------------------------------------------------


def _correctness_check(device):
    m, n = 256, 512
    t = torch.rand(m, n, dtype=torch.bfloat16) * 6.28  # [0, 2π)
    golden = torch.sin(t)
    x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.sin(x)
    assert_with_pcc(golden, ttnn.to_torch(out), 0.999)
    ttnn.deallocate(x)
    ttnn.deallocate(out)


# --------------------------------------------------------------------------------------
# pytest entry points
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
def test_correctness(device):
    """Verify ttnn.sin output matches torch.sin within PCC 0.999."""
    _correctness_check(device)


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("m", SET_A_M)
def test_tile_alignment(device, m):
    """Set A: sweep M with fixed N, showing padding waste for non-32-aligned shapes.

    Shapes like M=33 pad to M=64 (two tile rows), so the padded GOps/s is higher than
    real GOps/s — the gap is cycles wasted on padding elements.
    """
    n = SET_A_N
    per_iter_s, real_gops, padded_gops = run_sin(device, m, n)
    pm, pn = _padded(m), _padded(n)
    print(
        f"\nA  {m}x{n}  padded {pm}x{pn}  "
        f"{per_iter_s * 1e3:.3f} ms/iter  "
        f"real {real_gops:.2f} GOps/s  padded {padded_gops:.2f} GOps/s"
    )


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("s", SET_B_SQUARE)
def test_size_bandwidth_saturation(device, s):
    """Set B: square tensors from tiny to large, showing dispatch-overhead vs bandwidth saturation.

    Small tensors are dominated by host dispatch; only tensors large enough to saturate DRAM
    bandwidth approach the peak GOps/s.
    """
    per_iter_s, real_gops, padded_gops = run_sin(device, s, s)
    print(f"\nB  {s}x{s}  {per_iter_s * 1e3:.3f} ms/iter  {real_gops:.2f} GOps/s")


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("dtype,dtype_name", SET_C_DTYPES)
def test_data_format(device, dtype, dtype_name):
    """Set C: bfloat16 vs bfloat8_b at a fixed large shape.

    bfloat8_b is half the byte size of bfloat16, so for a bandwidth-bound op you expect ~2x
    the element throughput.  The actual ratio measures how close the hardware gets to that ideal.
    """
    m, n = SET_C_SHAPE
    per_iter_s, bw_gbs = run_sin_bandwidth(device, m, n, dtype=dtype)
    print(f"\nC  {dtype_name}  {m}x{n}  {per_iter_s * 1e3:.3f} ms/iter  {bw_gbs:.2f} GB/s")


# --------------------------------------------------------------------------------------
# Standalone entry point — builds all three formatted tables
# --------------------------------------------------------------------------------------


def main():
    device = ttnn.open_device(device_id=0, trace_region_size=TRACE_REGION_SIZE)
    try:
        _correctness_check(device)
        print("correctness (256x512, bfloat16): PCC ok\n")

        table_a = SinResultTable(
            "Set A: tile alignment (N=%d, bfloat16)" % SET_A_N,
            "Takeaway: non-32-aligned M pads to the next tile row.  "
            "The gap between real and padded GOps/s shows wasted SFPU cycles.",
        )
        for m in SET_A_M:
            per_iter_s, real_gops, padded_gops = run_sin(device, m, SET_A_N)
            table_a.add_shape(m, SET_A_N, per_iter_s, real_gops, padded_gops)

        table_b = SinResultTable(
            "Set B: size / bandwidth saturation (square MxN, bfloat16)",
            "Takeaway: tiny tensors are dispatch-bound; only large tensors saturate DRAM "
            "bandwidth and approach peak elementwise throughput.",
        )
        for s in SET_B_SQUARE:
            per_iter_s, real_gops, padded_gops = run_sin(device, s, s)
            table_b.add_shape(s, s, per_iter_s, real_gops, padded_gops)

        table_c = SinResultTable(
            "Set C: data format (%dx%d)" % SET_C_SHAPE,
            "Takeaway: bfloat8_b is 1 byte/elem vs 2 bytes/elem for bfloat16.  "
            "For a bandwidth-bound workload you expect ~2x more elements/s.",
        )
        for dtype, dtype_name in SET_C_DTYPES:
            per_iter_s, bw_gbs = run_sin_bandwidth(device, *SET_C_SHAPE, dtype=dtype)
            table_c.add_bw(*SET_C_SHAPE, dtype_name, per_iter_s, bw_gbs)

        print("\n" + table_a.render())
        print("\n" + table_b.render())
        print("\n" + table_c.render())
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
