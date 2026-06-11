# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tutorial 04 — Tracing on a MeshDevice (multi-device).

Read tutorial 01 (basics) first. This script shows that the trace capture/replay API is
identical for multi-device meshes — you simply pass a MeshDevice instead of a Device. The
only new concepts are:

  1. ttnn.open_mesh_device(ttnn.MeshShape(rows, cols), trace_region_size=...)
        Opens a logical 2-D mesh of physical devices. MeshShape(1, N) gives N devices
        in a single row, which is the usual layout for T3K / N300 systems.

  2. mesh_mapper — controls how a host tensor is *distributed* across the mesh:
        ShardTensorToMesh(mesh_device, dim=0)   — splits along the batch dim (one slice per device)
        ReplicateTensorToMesh(mesh_device)       — copies the same tensor onto every device

  3. mesh_composer — controls how results are *collected* back from the mesh:
        ConcatMeshToTensor(mesh_device, dim=0)  — concatenates per-device slices along dim 0

The trace capture/execute/release calls are *byte-for-byte identical* to the single-device
API — `ttnn.begin_trace_capture(mesh_device, ...)`, etc. The MeshDevice broadcasts every
command to all member devices under the hood.

This script auto-detects the number of available devices and uses a 1×N mesh (up to 4).
It skips gracefully if only one device is present.

Run inside the ct_metal container:
    docker exec -it -w /root/tt-metal ct_metal bash -c \
        "python ttnn/tutorials/tracing/04_trace_mesh_device.py"
"""

import sys
import torch
import ttnn
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh, ConcatMeshToTensor
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc

# Use at most this many devices so the tutorial runs quickly even on large systems.
MAX_DEVICES = 4


def main():
    num_available = ttnn.GetNumAvailableDevices()
    logger.info(f"System has {num_available} Tenstorrent device(s).")

    num_devices = min(num_available, MAX_DEVICES)
    if num_devices < 2:
        logger.warning("Only one device available — this tutorial is most instructive with ≥2 devices.")
        logger.warning("Continuing with a 1×1 mesh (effectively the same as the single-device tutorial).")

    # ---- Step 1: open the mesh --------------------------------------------------------
    # MeshShape(rows, cols): here we use a 1-row mesh with `num_devices` columns, which is
    # the standard layout for Wormhole systems (N150/N300/T3K).
    mesh_shape = ttnn.MeshShape(1, num_devices)
    mesh_device = ttnn.open_mesh_device(
        mesh_shape,
        trace_region_size=16 << 20,  # 16 MB per device
    )
    logger.info(f"Opened MeshDevice with shape {mesh_shape} ({mesh_device.get_num_devices()} devices).")

    try:
        n = mesh_device.get_num_devices()
        shape = (1, 1, 512, 512)

        # ---- Pre-allocate ALL device buffers before any trace is captured -------------
        # Tensors must be allocated before begin_trace_capture: allocating inside a live
        # trace region is unsafe because the trace will write to those buffers on replay
        # even after they may have been reallocated.
        input_0_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device)
        input_1_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device)
        weight_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device)

        # The compute op chain — identical to tutorial 01.
        def run_op_chain(input_0, input_1):
            return ttnn.neg(ttnn.add(ttnn.mul(input_1, ttnn.neg(ttnn.gelu(input_0))), ttnn.relu(input_1)))

        # ---- Step 2: warm-up / compile ------------------------------------------------
        logger.info("Step 2: warm-up / compile run (broadcasts to all devices)")
        run_op_chain(input_0_dev, input_1_dev)

        # ---- Step 3: capture trace on the mesh ----------------------------------------
        # ttnn.begin_trace_capture accepts a MeshDevice just like a Device. Under the hood
        # it broadcasts the capture command to every member device simultaneously.
        logger.info("Step 3: capturing trace on MeshDevice")
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        output_dev = run_op_chain(input_0_dev, input_1_dev)
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        logger.info(f"Captured trace id = {tid}")

        # ---- Step 4: replay with sharded inputs and verify ----------------------------
        # We shard the batch dimension across devices: each device gets a different slice.
        # ShardTensorToMesh splits a (N, C, H, W) host tensor along dim=0 so device i
        # receives slice i. This is how data-parallel inference works in practice.
        logger.info(f"Step 4: running {n} replay(s) with ShardTensorToMesh inputs")
        for i in range(3):
            # Create a host tensor shaped (N, C, H, W) — one sample per device.
            torch_in0 = torch.rand((n,) + shape[1:], dtype=torch.bfloat16)
            torch_in1 = torch.rand((n,) + shape[1:], dtype=torch.bfloat16)

            # Golden: each row is the expected output for the corresponding device.
            torch_golden = torch.neg(
                torch.add(
                    torch.mul(torch_in1, torch.neg(torch.nn.functional.gelu(torch_in0))),
                    torch.relu(torch_in1),
                )
            )

            # Map host tensors to devices (ShardTensorToMesh splits along dim=0).
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(torch_in0, layout=ttnn.TILE_LAYOUT, mesh_mapper=ShardTensorToMesh(mesh_device, dim=0)),
                input_0_dev,
            )
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(torch_in1, layout=ttnn.TILE_LAYOUT, mesh_mapper=ShardTensorToMesh(mesh_device, dim=0)),
                input_1_dev,
            )

            # Replay: single host command broadcasts to ALL devices in the mesh.
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)

            # Collect results: ConcatMeshToTensor reassembles shards along dim=0.
            tt_out = ttnn.to_torch(
                output_dev,
                mesh_composer=ConcatMeshToTensor(mesh_device, dim=0),
                device=mesh_device,
            )
            assert_with_pcc(tt_out, torch_golden, pcc=0.99)
            logger.info(f"  replay #{i}: output across {n} devices matches torch golden ✓")

        # ---- Bonus: ReplicateTensorToMesh ---------------------------------------------
        # For weights / shared inputs that are the same on every device, use replication
        # instead of sharding. Here we replicate a weight and verify a simple scaled op.
        logger.info("Bonus: capturing a second trace that uses a replicated weight tensor")

        def run_weighted_chain(x, w):
            return ttnn.relu(ttnn.mul(x, w))

        # ---- Step 5: release first trace before capturing second ----------------------
        # Always release a trace before allocating new buffers or capturing another trace
        # on the same device — live traces hold exclusive access to the trace DRAM region.
        logger.info("Step 5a: releasing first trace before capturing second")
        ttnn.release_trace(mesh_device, tid)

        # Warm-up the new op chain so its kernels are compiled.
        run_weighted_chain(input_0_dev, weight_dev)

        tid2 = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        out2_dev = run_weighted_chain(input_0_dev, weight_dev)
        ttnn.end_trace_capture(mesh_device, tid2, cq_id=0)

        torch_in = torch.rand((n,) + shape[1:], dtype=torch.bfloat16)
        torch_w = torch.rand(shape, dtype=torch.bfloat16)  # same weight for all devices
        torch_golden2 = torch.relu(torch.mul(torch_in, torch_w.expand((n,) + shape[1:])))

        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch_in, layout=ttnn.TILE_LAYOUT, mesh_mapper=ShardTensorToMesh(mesh_device, dim=0)),
            input_0_dev,
        )
        # ReplicateTensorToMesh broadcasts the same weight to every device.
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch_w, layout=ttnn.TILE_LAYOUT, mesh_mapper=ReplicateTensorToMesh(mesh_device)),
            weight_dev,
        )
        ttnn.execute_trace(mesh_device, tid2, cq_id=0, blocking=True)
        tt_out2 = ttnn.to_torch(
            out2_dev,
            mesh_composer=ConcatMeshToTensor(mesh_device, dim=0),
            device=mesh_device,
        )
        assert_with_pcc(tt_out2, torch_golden2, pcc=0.99)
        logger.info("Replicated-weight trace output matches torch golden ✓")

        # ---- Step 5b: release second trace --------------------------------------------
        logger.info("Step 5b: releasing second trace")
        ttnn.release_trace(mesh_device, tid2)

    finally:
        ttnn.close_mesh_device(mesh_device)
        logger.info("MeshDevice closed. Done.")


if __name__ == "__main__":
    main()
