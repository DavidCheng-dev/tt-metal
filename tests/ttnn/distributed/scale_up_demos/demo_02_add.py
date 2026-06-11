# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Demo 2 - ttnn.add on a mesh (DATA PARALLEL).

Same idea as demo 1, but a binary elementwise op: BOTH inputs are sharded the same way on
the batch dimension, so chip i adds its slice of A to its slice of B. No cross-device
communication is needed. Total size is fixed, so per-iteration time falls ~N x as we add
chips (strong scaling).

Run:
    python tests/ttnn/distributed/scale_up_demos/demo_02_add.py
    pytest tests/ttnn/distributed/scale_up_demos/demo_02_add.py -s
"""

import pytest
import torch

import ttnn

from scale_up_common import SpeedupTable, available_mesh_sizes, demo_mesh, pcc_check, time_op

TOTAL_BATCH = 8
HEIGHT = 1024
WIDTH = 1024


def run_add(mesh_device, n):
    """Run ttnn.add sharded over the mesh, check PCC vs torch, return per-iter seconds."""
    torch_a = torch.rand(TOTAL_BATCH, 1, HEIGHT, WIDTH, dtype=torch.bfloat16)
    torch_b = torch.rand(TOTAL_BATCH, 1, HEIGHT, WIDTH, dtype=torch.bfloat16)

    # CPU golden.
    golden = torch_a + torch_b

    # DATA PARALLEL: shard both operands on the batch dim so each chip owns matching slices.
    shard = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    tt_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=shard)
    tt_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=shard)

    # Correctness: gather per-chip results and compare to golden.
    tt_out = ttnn.add(tt_a, tt_b)
    result = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    pcc_check(golden, result, pcc=0.999)

    # Speed: time just the op.
    return time_op(mesh_device, lambda: ttnn.add(tt_a, tt_b))


def run_add_2d(mesh_device):
    """2D data-parallel add: shard both operands on batch (mesh axis 0) and height (axis 1)."""
    rows, cols = tuple(mesh_device.shape)
    torch_a = torch.rand(TOTAL_BATCH, 1, HEIGHT, WIDTH, dtype=torch.bfloat16)
    torch_b = torch.rand(TOTAL_BATCH, 1, HEIGHT, WIDTH, dtype=torch.bfloat16)

    golden = torch_a + torch_b

    # Split the data across BOTH mesh axes: batch over the rows, height over the columns.
    shard = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(0, 2))
    tt_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=shard)
    tt_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=shard)

    tt_out = ttnn.add(tt_a, tt_b)
    result = ttnn.to_torch(
        tt_out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(rows, cols), dims=(0, 2))
    )
    pcc_check(golden, result, pcc=0.999)

    return time_op(mesh_device, lambda: ttnn.add(tt_a, tt_b))


@pytest.mark.parametrize("device_params", [{"trace_region_size": 23887872}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1, 2, 4, 8], indirect=True)
def test_add(mesh_device):
    n = mesh_device.get_num_devices()
    per_iter_s = run_add(mesh_device, n)
    print(f"\nadd  {n} chip(s): {per_iter_s * 1e3:.3f} ms/iter")


@pytest.mark.parametrize("device_params", [{"trace_region_size": 23887872}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4), (4, 2)], indirect=True)
def test_add_2d(mesh_device):
    rows, cols = tuple(mesh_device.shape)
    per_iter_s = run_add_2d(mesh_device)
    print(f"\nadd  {rows}x{cols} mesh: {per_iter_s * 1e3:.3f} ms/iter")


def main():
    table = SpeedupTable("Demo 2: ttnn.add (data parallel)", work_per_iter=TOTAL_BATCH * HEIGHT * WIDTH)
    for n in available_mesh_sizes():
        with demo_mesh(n) as mesh_device:
            table.add(n, run_add(mesh_device, n))
    print("\n" + table.render())

    if ttnn.get_num_devices() >= 8:
        table2d = SpeedupTable(
            "Demo 2b: ttnn.add (2D mesh, data parallel on both axes: batch x height)",
            work_per_iter=TOTAL_BATCH * HEIGHT * WIDTH,
            col0_header="mesh",
        )
        for shape in [(1, 1), (2, 4), (4, 2)]:
            with demo_mesh(shape) as mesh_device:
                table2d.add(shape[0] * shape[1], run_add_2d(mesh_device), label=f"{shape[0]}x{shape[1]}")
        print("\n" + table2d.render())


if __name__ == "__main__":
    main()
