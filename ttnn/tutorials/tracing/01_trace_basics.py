# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tutorial 01 — Trace capture & replay basics.

This script walks through the complete lifecycle of a TT-NN execution trace on a single
device. A "trace" is a recording of the device-side commands for a sequence of ops; once
captured it can be replayed with ttnn.execute_trace, issuing one host command instead of
re-dispatching every op. See the README in this folder for the concepts.

The 5 lifecycle steps are labelled in the code below:
    1. open device with a trace_region_size
    2. warm-up / compile run
    3. capture
    4. execute (replay) with fresh inputs fed into preallocated buffers
    5. release + close

Run inside the ct_metal container:
    docker exec -it -w /root/tt-metal ct_metal bash -c \
        "python ttnn/tutorials/tracing/01_trace_basics.py"
"""

import torch
import ttnn
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc


def main():
    # ---- Step 1: open the device WITH a trace region -----------------------------------
    # trace_region_size reserves on-device DRAM to hold the recorded commands. It must be
    # non-zero, otherwise capture has nowhere to store the trace.
    device = ttnn.open_device(device_id=0, trace_region_size=16 << 20)  # 16 MB

    try:
        shape = (1, 1, 512, 512)

        # Pre-allocate the INPUT buffers once. The trace will be tied to these exact
        # buffers, so for every replay we update their *contents* rather than allocating
        # new tensors. This is the single most important rule of tracing.
        input_0_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
        input_1_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)

        # The op chain we want to trace. Several small eltwise ops — exactly the kind of
        # dispatch-bound sequence that benefits from tracing.
        def run_op_chain(input_0, input_1):
            return ttnn.neg(ttnn.add(ttnn.mul(input_1, ttnn.neg(ttnn.gelu(input_0))), ttnn.relu(input_1)))

        # ---- Step 2: warm-up / compile run --------------------------------------------
        # Kernels are compiled on first use. Capture records *dispatch*, not compilation,
        # so we must run the chain once before capturing.
        logger.info("Step 2: warm-up / compile run")
        run_op_chain(input_0_dev, input_1_dev)

        # ---- Step 3: capture the trace ------------------------------------------------
        # Everything between begin/end is RECORDED, not executed. The returned output
        # tensor is the buffer the trace writes its result into on every replay.
        logger.info("Step 3: capturing trace")
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        output_dev = run_op_chain(input_0_dev, input_1_dev)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        logger.info(f"Captured trace id = {tid}")

        # ---- Step 4: execute (replay) several times -----------------------------------
        # Each iteration we load NEW input data into the preallocated buffers, replay the
        # trace, then read back the result and check it against a torch golden.
        for i in range(3):
            logger.info(f"Step 4: replay #{i}")

            torch_in0 = torch.rand(shape, dtype=torch.bfloat16)
            torch_in1 = torch.rand(shape, dtype=torch.bfloat16)

            # Golden reference computed on the host with torch.
            torch_golden = torch.neg(
                torch.add(
                    torch.mul(torch_in1, torch.neg(torch.nn.functional.gelu(torch_in0))),
                    torch.relu(torch_in1),
                )
            )

            # Load the new inputs into the SAME device buffers the trace was captured on.
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch_in0, layout=ttnn.TILE_LAYOUT), input_0_dev)
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch_in1, layout=ttnn.TILE_LAYOUT), input_1_dev)

            # One command replays the whole op chain on the device.
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)

            # Read back from the trace's output buffer and verify correctness.
            tt_out = ttnn.to_torch(output_dev)
            assert_with_pcc(tt_out, torch_golden, pcc=0.99)
            logger.info(f"  replay #{i} output matches torch golden ✓")

        # ---- Step 5: release the trace ------------------------------------------------
        # Frees the on-device trace buffer. Always release traces you no longer need.
        logger.info("Step 5: releasing trace")
        ttnn.release_trace(device, tid)

    finally:
        ttnn.close_device(device)
        logger.info("Device closed. Done.")


if __name__ == "__main__":
    main()
