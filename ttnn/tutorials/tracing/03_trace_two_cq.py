# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tutorial 03 — ADVANCED: overlapping data movement and traced compute with two command queues.

Read tutorials 01 and 02 first.

Tracing removes host *dispatch* overhead, but a real inference loop still has to move data:
copy the next input to the device and copy each result back to the host. If you do that on
the same command queue as compute, the device stalls during every transfer. The standard
production pattern uses TWO command queues so the transfers overlap with compute:

    CQ0 : runs the captured compute trace
    CQ1 : copies the next input in, and copies the current output back out

The two queues are coordinated with events (ttnn.record_event / ttnn.wait_for_event) so
that, e.g., CQ0 does not overwrite the input buffer before CQ1 has finished loading it,
and CQ1 does not read the output buffer before CQ0 has finished producing it.

This script benchmarks the two-CQ pipelined approach against a single-CQ baseline where
data movement and compute are serialized on the same queue. It prints per-iteration
latency, throughput, and the pipelining speedup.

This is intentionally the most intricate script in the tutorial; each event is commented
with the hazard it guards against. This is a condensed version of the pattern used in
models/experimental/functional_unet/tests/test_unet_trace.py.

Run inside the ct_metal container:
    docker exec -it -w /root/tt-metal ct_metal bash -c \
        "python ttnn/tutorials/tracing/03_trace_two_cq.py"
"""

import time

import torch
import ttnn
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc

SHAPE = (1, 1, 512, 512)
WARMUP_ITERS = 8  # untimed warm-up iterations
TIMED_ITERS = 32  # measured iterations

COMPUTE_CQ = 0  # CQ0 runs the trace
DATA_CQ = 1  # CQ1 moves data


def run_op_chain(x):
    """Small dispatch-bound compute "model" to be traced."""
    for _ in range(8):
        x = ttnn.gelu(x)
        x = ttnn.relu(x)
        x = ttnn.neg(x)
        x = ttnn.add(x, x)
    return x


def bench_single_cq(device, host_in, input_dev, output_dev, iters):
    """Single-CQ baseline: data movement and compute are serialized on CQ0.

    Each iteration: copy input → execute trace → copy output back.
    No parallelism between transfers and compute.
    """
    outputs = [
        ttnn.allocate_tensor_on_host(output_dev.shape, output_dev.dtype, output_dev.layout, device)
        for _ in range(iters)
    ]

    # Compile + capture a single-CQ trace.
    ttnn.copy_host_to_device_tensor(host_in, input_dev, cq_id=0)
    run_op_chain(input_dev)
    ttnn.synchronize_device(device)

    tid = ttnn.begin_trace_capture(device, cq_id=0)
    output_dev = run_op_chain(input_dev)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)

    # Warm up.
    for _ in range(WARMUP_ITERS):
        ttnn.copy_host_to_device_tensor(host_in, input_dev, cq_id=0)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.copy_device_to_host_tensor(output_dev, outputs[0], blocking=False, cq_id=0)
    ttnn.synchronize_device(device)

    # Timed loop.
    t0 = time.perf_counter()
    for i in range(iters):
        ttnn.copy_host_to_device_tensor(host_in, input_dev, cq_id=0)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.copy_device_to_host_tensor(output_dev, outputs[i], blocking=False, cq_id=0)
    ttnn.synchronize_device(device)
    elapsed = time.perf_counter() - t0

    ttnn.release_trace(device, tid)
    return elapsed, outputs


def bench_two_cq(device, host_in, input_dev, iters):
    """Two-CQ pipelined execution: CQ1 data movement overlaps CQ0 compute.

    The event coordination ensures:
    - CQ0 does not read input_dev before CQ1 has finished writing it.
    - CQ1 does not read output_dev before CQ0 has finished producing it.
    - CQ0 does not overwrite input_dev before CQ1 has finished reading it for the trace.
    """
    # Warm-up compile run + initial events.
    op_event = ttnn.record_event(device, COMPUTE_CQ)
    ttnn.copy_host_to_device_tensor(host_in, input_dev, cq_id=DATA_CQ)
    write_event = ttnn.record_event(device, DATA_CQ)
    ttnn.wait_for_event(COMPUTE_CQ, write_event)
    op_event = ttnn.record_event(device, COMPUTE_CQ)
    output_dev = run_op_chain(input_dev)
    ttnn.synchronize_device(device)

    golden = ttnn.to_torch(output_dev)

    # Capture compute trace on CQ0.
    logger.info("Capturing compute trace on CQ0")
    ttnn.wait_for_event(DATA_CQ, op_event)
    ttnn.copy_host_to_device_tensor(host_in, input_dev, cq_id=DATA_CQ)
    write_event = ttnn.record_event(device, DATA_CQ)
    ttnn.wait_for_event(COMPUTE_CQ, write_event)
    op_event = ttnn.record_event(device, COMPUTE_CQ)
    output_dev.deallocate(force=True)

    tid = ttnn.begin_trace_capture(device, cq_id=COMPUTE_CQ)
    output_dev = run_op_chain(input_dev)
    ttnn.end_trace_capture(device, tid, cq_id=COMPUTE_CQ)
    ttnn.synchronize_device(device)

    outputs = [
        ttnn.allocate_tensor_on_host(output_dev.shape, output_dev.dtype, output_dev.layout, device)
        for _ in range(iters)
    ]

    def _run_pipeline(n):
        nonlocal write_event, op_event
        read_event = ttnn.record_event(device, DATA_CQ)
        for i in range(n):
            ttnn.wait_for_event(COMPUTE_CQ, write_event)
            ttnn.wait_for_event(COMPUTE_CQ, read_event)
            op_event = ttnn.record_event(device, COMPUTE_CQ)
            ttnn.execute_trace(device, tid, cq_id=COMPUTE_CQ, blocking=False)
            model_event = ttnn.record_event(device, COMPUTE_CQ)

            ttnn.wait_for_event(DATA_CQ, op_event)
            ttnn.copy_host_to_device_tensor(host_in, input_dev, cq_id=DATA_CQ)
            write_event = ttnn.record_event(device, DATA_CQ)

            ttnn.wait_for_event(DATA_CQ, model_event)
            ttnn.copy_device_to_host_tensor(output_dev, outputs[i % len(outputs)], blocking=False, cq_id=DATA_CQ)
            read_event = ttnn.record_event(device, DATA_CQ)
        ttnn.synchronize_device(device)

    # Warm up.
    _run_pipeline(WARMUP_ITERS)

    # Timed loop.
    t0 = time.perf_counter()
    _run_pipeline(iters)
    elapsed = time.perf_counter() - t0

    ttnn.release_trace(device, tid)
    return elapsed, outputs, golden


def main():
    # Two command queues are requested at device open time.
    device = ttnn.open_device(device_id=0, trace_region_size=16 << 20, num_command_queues=2)

    try:
        torch_in = torch.rand(SHAPE, dtype=torch.bfloat16)
        host_in = ttnn.from_torch(torch_in, layout=ttnn.TILE_LAYOUT)

        input_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(SHAPE), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)

        # Pre-allocate a placeholder output_dev for single-CQ bench (it reallocates inside).
        output_placeholder = ttnn.allocate_tensor_on_device(ttnn.Shape(SHAPE), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)

        # ---- Single-CQ baseline --------------------------------------------------------
        logger.info("Benchmarking single-CQ (serialized) execution...")
        single_secs, single_outputs = bench_single_cq(device, host_in, input_dev, output_placeholder, TIMED_ITERS)

        # ---- Two-CQ pipelined ----------------------------------------------------------
        logger.info("Benchmarking two-CQ pipelined execution...")
        two_secs, two_outputs, golden = bench_two_cq(device, host_in, input_dev, TIMED_ITERS)

        # ---- Correctness ---------------------------------------------------------------
        assert_with_pcc(ttnn.to_torch(two_outputs[-1]), golden, pcc=0.99)
        logger.info("Pipelined traced output matches golden ✓")

        # ---- Report --------------------------------------------------------------------
        single_ms = 1000.0 * single_secs / TIMED_ITERS
        two_ms = 1000.0 * two_secs / TIMED_ITERS
        speedup = single_secs / two_secs if two_secs > 0 else float("inf")

        logger.info("")
        logger.info("================ Two-CQ pipeline performance summary ================")
        logger.info(f"  op chain         : {4 * 8} small ops / iteration (32 eltwise)")
        logger.info(f"  timed iterations : {TIMED_ITERS}")
        logger.info(f"  Single CQ (serial)  : {single_ms:8.3f} ms/iter   ({1000.0 / single_ms:8.1f} iter/s)")
        logger.info(f"  Two CQ (pipelined)  : {two_ms:8.3f} ms/iter   ({1000.0 / two_ms:8.1f} iter/s)")
        logger.info(f"  SPEEDUP          : {speedup:8.2f}x")
        logger.info("=====================================================================")
        logger.info("The two-CQ speedup comes from overlapping host<->device data movement")
        logger.info("(CQ1) with traced compute (CQ0). It grows with larger inputs, where")
        logger.info("data transfer is a bigger fraction of total iteration time.")

    finally:
        ttnn.close_device(device)
        logger.info("Device closed. Done.")


if __name__ == "__main__":
    main()
