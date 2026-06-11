# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tutorial 02 — The performance impact of tracing.

This script runs the *same* op chain two ways and times them:

    EAGER : the host re-dispatches every op, every iteration.
    TRACED: the chain is captured once, then replayed with a single host command.

Because the chain is made of many small ops, eager execution is "dispatch-bound" — the
device spends time idle waiting for the host. Tracing removes that overhead, so the
traced path should be noticeably faster. The script prints average per-iteration latency,
throughput, and the speedup ratio.

Tip: increase CHAIN_REPEATS to add more small ops per iteration — the more (smaller) ops
there are, the larger the tracing speedup, because there is more dispatch overhead to
eliminate.

Run inside the ct_metal container:
    docker exec -it -w /root/tt-metal ct_metal bash -c \
        "python ttnn/tutorials/tracing/02_trace_performance.py"
"""

import time

import torch
import ttnn
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc

# Tunables ------------------------------------------------------------------------------
SHAPE = (1, 1, 512, 512)
CHAIN_REPEATS = 10  # how many times to repeat the small op chain per iteration
WARMUP_ITERS = 10  # untimed iterations to stabilize timing
TIMED_ITERS = 100  # measured iterations


def run_op_chain(x):
    """A dispatch-bound chain of small eltwise ops, applied CHAIN_REPEATS times."""
    for _ in range(CHAIN_REPEATS):
        x = ttnn.gelu(x)
        x = ttnn.relu(x)
        x = ttnn.neg(x)
        x = ttnn.add(x, x)
    return x


def time_loop(device, fn, iters):
    """Run fn() `iters` times and synchronize once at the end, returning total seconds.

    The trailing synchronize_device is essential: TT-NN op enqueue is asynchronous, so
    without it we would only measure how fast the host can *enqueue* work, not how long
    the device takes to *finish* it.
    """
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(device)
    return time.perf_counter() - start


def main():
    device = ttnn.open_device(device_id=0, trace_region_size=16 << 20)

    try:
        # One fixed input, reused for every iteration (we are measuring dispatch, not data
        # movement). Pre-allocate it once so the traced path can replay against it.
        torch_in = torch.rand(SHAPE, dtype=torch.bfloat16)
        input_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(SHAPE), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch_in, layout=ttnn.TILE_LAYOUT), input_dev)

        # ---------------------------------------------------------------------------
        # EAGER: host dispatches every op every iteration.
        # ---------------------------------------------------------------------------
        logger.info("Benchmarking EAGER execution...")

        def eager_step():
            run_op_chain(input_dev)

        for _ in range(WARMUP_ITERS):  # warm up (also compiles kernels)
            eager_step()
        ttnn.synchronize_device(device)

        eager_secs = time_loop(device, eager_step, TIMED_ITERS)

        # Keep one eager result as the golden reference for correctness.
        eager_golden = ttnn.to_torch(run_op_chain(input_dev))
        ttnn.synchronize_device(device)

        # ---------------------------------------------------------------------------
        # TRACED: capture once, then replay with a single host command per iteration.
        # ---------------------------------------------------------------------------
        logger.info("Capturing trace...")
        # (kernels already compiled by the eager warm-up above)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        traced_output = run_op_chain(input_dev)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.synchronize_device(device)

        logger.info("Benchmarking TRACED execution...")

        def traced_step():
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)

        for _ in range(WARMUP_ITERS):
            traced_step()
        ttnn.synchronize_device(device)

        traced_secs = time_loop(device, traced_step, TIMED_ITERS)

        # Correctness: traced result must match the eager golden.
        assert_with_pcc(ttnn.to_torch(traced_output), eager_golden, pcc=0.99)
        logger.info("Traced output matches eager golden ✓")

        ttnn.release_trace(device, tid)

        # ---------------------------------------------------------------------------
        # Report
        # ---------------------------------------------------------------------------
        eager_ms = 1000.0 * eager_secs / TIMED_ITERS
        traced_ms = 1000.0 * traced_secs / TIMED_ITERS
        speedup = eager_secs / traced_secs if traced_secs > 0 else float("inf")

        logger.info("")
        logger.info("================ Trace performance summary ================")
        logger.info(f"  op chain         : {4 * CHAIN_REPEATS} small ops / iteration")
        logger.info(f"  timed iterations : {TIMED_ITERS}")
        logger.info(f"  EAGER            : {eager_ms:8.3f} ms/iter   ({1000.0 / eager_ms:8.1f} iter/s)")
        logger.info(f"  TRACED           : {traced_ms:8.3f} ms/iter   ({1000.0 / traced_ms:8.1f} iter/s)")
        logger.info(f"  SPEEDUP          : {speedup:8.2f}x")
        logger.info("===========================================================")
        logger.info("The gap is host dispatch overhead that tracing removes; it grows")
        logger.info("as you add more (smaller) ops — try increasing CHAIN_REPEATS.")

    finally:
        ttnn.close_device(device)
        logger.info("Device closed. Done.")


if __name__ == "__main__":
    main()
