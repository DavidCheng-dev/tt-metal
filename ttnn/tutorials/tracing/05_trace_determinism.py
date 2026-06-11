# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tutorial 05 — Bit-exact determinism of trace replay.

Read tutorial 01 (basics) first. This script answers a simple question: if you
replay the same trace twice with *identical* input data, are the outputs bit-for-bit
identical?

The answer is yes — TT hardware executes ops deterministically. There is no
floating-point non-determinism of the kind that can arise on GPUs with certain
reduction ops. This script verifies that guarantee with torch.equal(), which
requires every bit to match, rather than the looser PCC check used in earlier
tutorials.

Run inside the ct_metal container:
    docker exec -it -w /root/tt-metal ct_metal bash -c \
        "python ttnn/tutorials/tracing/05_trace_determinism.py"
"""

import torch
import ttnn
from loguru import logger


def main():
    device = ttnn.open_device(device_id=0, trace_region_size=16 << 20)

    try:
        shapes = [(1, 1, 512, 512), (1, 1, 32, 32), (1, 3, 128, 128)]

        for shape in shapes:
            input_0_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
            input_1_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)

            def run_op_chain(x0, x1):
                return ttnn.neg(ttnn.add(ttnn.mul(x1, ttnn.neg(ttnn.gelu(x0))), ttnn.relu(x1)))

            run_op_chain(input_0_dev, input_1_dev)  # compile

            tid = ttnn.begin_trace_capture(device, cq_id=0)
            output_dev = run_op_chain(input_0_dev, input_1_dev)
            ttnn.end_trace_capture(device, tid, cq_id=0)

            torch_in0 = torch.rand(shape, dtype=torch.bfloat16)
            torch_in1 = torch.rand(shape, dtype=torch.bfloat16)
            host_0 = ttnn.from_torch(torch_in0, layout=ttnn.TILE_LAYOUT)
            host_1 = ttnn.from_torch(torch_in1, layout=ttnn.TILE_LAYOUT)

            def replay():
                ttnn.copy_host_to_device_tensor(host_0, input_0_dev)
                ttnn.copy_host_to_device_tensor(host_1, input_1_dev)
                ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
                return ttnn.to_torch(output_dev)

            out1 = replay()
            out2 = replay()
            ttnn.release_trace(device, tid)

            assert torch.equal(out1, out2), f"FAILED: shape={shape} produced different outputs on identical inputs"
            logger.info(f"  shape={shape}: bit-exact ✓")

        logger.info("All bit-exact determinism checks passed ✓")
    finally:
        ttnn.close_device(device)
        logger.info("Done.")


if __name__ == "__main__":
    main()
