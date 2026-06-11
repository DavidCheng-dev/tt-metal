# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Demo 3 - ttnn.matmul on a mesh (DATA PARALLEL and TENSOR PARALLEL).

matmul has weights, so there are two natural ways to scale it:

  * DATA PARALLEL  - replicate the weight W on every chip, shard the activation A on the
    batch dim. Chip i computes A_i @ W. No communication. Used when the model fits on one
    chip and you just want more throughput.

  * TENSOR PARALLEL - replicate A, shard the weight W on its output (width) dim. Chip i
    computes A @ W_i, producing a column-slice of the output. Used when the weight is too
    big for one chip. In production the slices are stitched back ON-DEVICE with an
    all-gather CCL op (ttnn.experimental.all_gather_async; see
    tests/ttnn/unit_tests/operations/ccl/test_ag_rs_llama_prefill_TG.py). Here we gather on
    the host instead, which keeps the correctness check simple and robust.

Both keep the TOTAL matmul fixed and split it across chips, so per-iteration time falls
~N x as we add chips (strong scaling).

Run:
    python tests/ttnn/distributed/scale_up_demos/demo_03_matmul.py
    pytest tests/ttnn/distributed/scale_up_demos/demo_03_matmul.py -s
"""

import pytest
import torch

import ttnn

from scale_up_common import SpeedupTable, available_mesh_sizes, demo_mesh, pcc_check, time_op

# Data-parallel problem: batch is divisible by 8 so it shards evenly onto 1/2/4/8 chips.
DP_BATCH = 8
DP_M, DP_K, DP_N = 512, 1024, 1024

# Tensor-parallel problem: output width N is divisible by 8 so the weight shards evenly.
# Sized large enough that per-chip compute dominates dispatch overhead (stable scaling).
TP_M, TP_K, TP_N = 1024, 2048, 8192

# 2D problem: data-parallel on batch (rows) x tensor-parallel on width N (cols).
# Sized so that even at the widest split (batch/4 down the rows, N/4 across the cols) each
# chip still has a fat matmul, so the core grid stays saturated. The small DP problem above
# goes skinny under 4-way TP (N/4 = 8 tiles) and underutilizes the grid; these dims keep
# N/4 = 64 tiles, M = 32 tiles, K = 64 tiles so every chip stays compute-bound.
TD_BATCH = 8  # divisible by max rows (4)
TD_M, TD_K, TD_N = 1024, 2048, 8192  # N divisible by max cols (4) and tile size (32)

# Accumulate the matmul in fp32 so a deep (K=1024/2048) bf16 reduction stays accurate
# enough to compare against the CPU golden. This is the main precision knob for matmul.
COMPUTE_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)


def run_matmul_dp(mesh_device, n):
    """Data-parallel matmul: shard A on batch, replicate W. Returns per-iter seconds."""
    torch_a = torch.rand(DP_BATCH, 1, DP_M, DP_K, dtype=torch.bfloat16)
    torch_w = torch.rand(1, 1, DP_K, DP_N, dtype=torch.bfloat16)
    golden = torch_a.float() @ torch_w.float()

    tt_a = ttnn.from_torch(
        torch_a,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    tt_w = ttnn.from_torch(
        torch_w,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_out = ttnn.matmul(tt_a, tt_w, compute_kernel_config=COMPUTE_KERNEL_CONFIG)
    result = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    pcc_check(golden, result, pcc=0.99)

    return time_op(mesh_device, lambda: ttnn.matmul(tt_a, tt_w, compute_kernel_config=COMPUTE_KERNEL_CONFIG))


def run_matmul_tp(mesh_device, n):
    """Tensor-parallel matmul: replicate A, shard W on width. Returns per-iter seconds."""
    torch_a = torch.rand(1, 1, TP_M, TP_K, dtype=torch.bfloat16)
    torch_w = torch.rand(1, 1, TP_K, TP_N, dtype=torch.bfloat16)
    golden = torch_a.float() @ torch_w.float()

    tt_a = ttnn.from_torch(
        torch_a,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_w = ttnn.from_torch(
        torch_w,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
    )

    # Each chip produces a column-slice [.., M, N/n]; gather slices on the host (dim=-1).
    tt_out = ttnn.matmul(tt_a, tt_w, compute_kernel_config=COMPUTE_KERNEL_CONFIG)
    result = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    pcc_check(golden, result, pcc=0.99)

    return time_op(mesh_device, lambda: ttnn.matmul(tt_a, tt_w, compute_kernel_config=COMPUTE_KERNEL_CONFIG))


def run_matmul_2d(mesh_device):
    """2D matmul: DATA PARALLEL on mesh axis 0 (rows) x TENSOR PARALLEL on mesh axis 1 (cols).

    The activation A is sharded on the batch dim across the rows and replicated down the
    columns; the weight W is replicated across the rows and sharded on its output (width)
    dim across the columns. Chip (r, c) computes A_r @ W_c -> a [batch-slice, M, N-slice]
    block. So a (2, 4) mesh is 2-way DP x 4-way TP, and (4, 2) is 4-way DP x 2-way TP.
    """
    rows, cols = tuple(mesh_device.shape)
    torch_a = torch.rand(TD_BATCH, 1, TD_M, TD_K, dtype=torch.bfloat16)
    torch_w = torch.rand(1, 1, TD_K, TD_N, dtype=torch.bfloat16)
    golden = torch_a.float() @ torch_w.float()

    tt_a = ttnn.from_torch(
        torch_a,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(0, None)),
    )
    tt_w = ttnn.from_torch(
        torch_w,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(None, 3)),
    )

    # Gather the batch slices from the rows (dim 0) and the width slices from the cols (dim 3).
    tt_out = ttnn.matmul(tt_a, tt_w, compute_kernel_config=COMPUTE_KERNEL_CONFIG)
    result = ttnn.to_torch(
        tt_out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(rows, cols), dims=(0, 3))
    )
    pcc_check(golden, result, pcc=0.99)

    return time_op(mesh_device, lambda: ttnn.matmul(tt_a, tt_w, compute_kernel_config=COMPUTE_KERNEL_CONFIG))


@pytest.mark.parametrize("device_params", [{"trace_region_size": 23887872}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1, 2, 4, 8], indirect=True)
def test_matmul_dp(mesh_device):
    n = mesh_device.get_num_devices()
    print(f"\nmatmul DP {n} chip(s): {run_matmul_dp(mesh_device, n) * 1e3:.3f} ms/iter")


@pytest.mark.parametrize("device_params", [{"trace_region_size": 23887872}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1, 2, 4, 8], indirect=True)
def test_matmul_tp(mesh_device):
    n = mesh_device.get_num_devices()
    print(f"\nmatmul TP {n} chip(s): {run_matmul_tp(mesh_device, n) * 1e3:.3f} ms/iter")


@pytest.mark.parametrize("device_params", [{"trace_region_size": 23887872}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4), (4, 2)], indirect=True)
def test_matmul_2d(mesh_device):
    rows, cols = tuple(mesh_device.shape)
    print(f"\nmatmul DPxTP {rows}x{cols} mesh: {run_matmul_2d(mesh_device) * 1e3:.3f} ms/iter")


def main():
    dp = SpeedupTable(
        "Demo 3a: ttnn.matmul (data parallel)", work_per_iter=2 * DP_BATCH * DP_M * DP_K * DP_N, work_unit="FLOP"
    )
    tp = SpeedupTable("Demo 3b: ttnn.matmul (tensor parallel)", work_per_iter=2 * TP_M * TP_K * TP_N, work_unit="FLOP")
    for n in available_mesh_sizes():
        with demo_mesh(n) as mesh_device:
            dp.add(n, run_matmul_dp(mesh_device, n))
            tp.add(n, run_matmul_tp(mesh_device, n))
    print("\n" + dp.render())
    print("\n" + tp.render())

    if ttnn.get_num_devices() >= 8:
        table2d = SpeedupTable(
            "Demo 3c: ttnn.matmul (2D mesh, data parallel x tensor parallel)",
            work_per_iter=2 * TD_BATCH * TD_M * TD_K * TD_N,
            work_unit="FLOP",
            col0_header="mesh",
        )
        for shape in [(1, 1), (2, 4), (4, 2)]:
            with demo_mesh(shape) as mesh_device:
                table2d.add(shape[0] * shape[1], run_matmul_2d(mesh_device), label=f"{shape[0]}x{shape[1]}")
        print("\n" + table2d.render())


if __name__ == "__main__":
    main()
