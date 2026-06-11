# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Demo 1 - ttnn.sin on a mesh (DATA PARALLEL).

The simplest scale-up case: a unary elementwise op has no weights and no cross-device
reduction, so it is "embarrassingly parallel". We keep the TOTAL tensor size fixed and
SHARD it across the mesh on the batch dimension, so each chip processes 1/N of the rows.
As we go 1 -> 2 -> 4 -> 8 chips the per-iteration time should drop ~N x (strong scaling).

Run:
    python tests/ttnn/distributed/scale_up_demos/demo_01_sin.py
    pytest tests/ttnn/distributed/scale_up_demos/demo_01_sin.py -s
"""

import pytest
import torch

import ttnn

from scale_up_common import SpeedupTable, available_mesh_sizes, demo_mesh, pcc_check, time_op

# Fixed total problem size (shared across all mesh sizes). Batch is divisible by 8 so it
# shards evenly onto 1/2/4/8 chips.
TOTAL_BATCH = 8
HEIGHT = 1024
WIDTH = 1024


def run_sin(mesh_device, n):
    """Run ttnn.sin sharded over the mesh, check PCC vs torch, return per-iter seconds."""
    torch_in = torch.rand(TOTAL_BATCH, 1, HEIGHT, WIDTH, dtype=torch.bfloat16)

    # CPU golden.
    golden = torch.sin(torch_in)

    # DATA PARALLEL: shard the batch dim across the N chips; each chip gets TOTAL_BATCH/N.
    tt_in = ttnn.from_torch(
        torch_in,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    # Correctness: gather the per-chip results back and compare to the golden.
    tt_out = ttnn.sin(tt_in)
    result = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    pcc_check(golden, result, pcc=0.999)

    # Speed: time just the op (inputs already on device).
    return time_op(mesh_device, lambda: ttnn.sin(tt_in))


def run_sin_2d(mesh_device):
    """2D data-parallel sin: shard batch on mesh axis 0 and height on mesh axis 1."""
    rows, cols = tuple(mesh_device.shape)
    torch_in = torch.rand(TOTAL_BATCH, 1, HEIGHT, WIDTH, dtype=torch.bfloat16)

    golden = torch.sin(torch_in)

    # Split the data across BOTH mesh axes: batch over the rows, height over the columns.
    tt_in = ttnn.from_torch(
        torch_in,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(0, 2)),
    )

    tt_out = ttnn.sin(tt_in)
    result = ttnn.to_torch(
        tt_out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(rows, cols), dims=(0, 2))
    )
    pcc_check(golden, result, pcc=0.999)

    return time_op(mesh_device, lambda: ttnn.sin(tt_in))


@pytest.mark.parametrize("device_params", [{"trace_region_size": 23887872}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1, 2, 4, 8], indirect=True)
def test_sin(mesh_device):
    n = mesh_device.get_num_devices()
    per_iter_s = run_sin(mesh_device, n)
    print(f"\nsin  {n} chip(s): {per_iter_s * 1e3:.3f} ms/iter")


@pytest.mark.parametrize("device_params", [{"trace_region_size": 23887872}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4), (4, 2)], indirect=True)
def test_sin_2d(mesh_device):
    rows, cols = tuple(mesh_device.shape)
    per_iter_s = run_sin_2d(mesh_device)
    print(f"\nsin  {rows}x{cols} mesh: {per_iter_s * 1e3:.3f} ms/iter")


def main():
    table = SpeedupTable("Demo 1: ttnn.sin (data parallel)", work_per_iter=TOTAL_BATCH * HEIGHT * WIDTH)
    for n in available_mesh_sizes():
        with demo_mesh(n) as mesh_device:
            table.add(n, run_sin(mesh_device, n))
    print("\n" + table.render())

    if ttnn.get_num_devices() >= 8:
        table2d = SpeedupTable(
            "Demo 1b: ttnn.sin (2D mesh, data parallel on both axes: batch x height)",
            work_per_iter=TOTAL_BATCH * HEIGHT * WIDTH,
            col0_header="mesh",
        )
        for shape in [(1, 1), (2, 4), (4, 2)]:
            with demo_mesh(shape) as mesh_device:
                table2d.add(shape[0] * shape[1], run_sin_2d(mesh_device), label=f"{shape[0]}x{shape[1]}")
        print("\n" + table2d.render())


if __name__ == "__main__":
    main()
