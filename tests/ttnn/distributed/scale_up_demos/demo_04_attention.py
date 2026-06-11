# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Demo 4 - scaled dot product attention on a mesh (DATA PARALLEL and TENSOR PARALLEL).

Attention is the most complex op in this series, but it scales cleanly two ways:

  * DATA PARALLEL  - shard Q/K/V on the batch dim. Chip i runs full attention for its
    slice of the batch. No communication.

  * TENSOR PARALLEL - shard Q/K/V on the HEAD dim. Attention heads are independent, so
    chip i computes a subset of heads with NO cross-chip communication, and we just
    concatenate the heads back. This is the natural multi-head split used by real LLMs.

Both keep the TOTAL attention work fixed and split it across chips, so per-iteration time
falls ~N x as we add chips (strong scaling).

Run:
    python tests/ttnn/distributed/scale_up_demos/demo_04_attention.py
    pytest tests/ttnn/distributed/scale_up_demos/demo_04_attention.py -s
"""

import pytest
import torch

import ttnn

from scale_up_common import SpeedupTable, available_mesh_sizes, demo_mesh, pcc_check, time_op

# Batch divisible by 8 (for data-parallel) and heads divisible by 8 (for tensor-parallel).
BATCH = 8
NUM_HEADS = 8
SEQ_LEN = 2048  # attention compute is O(S^2), so a longer sequence makes scaling visible
HEAD_DIM = 128

# Causal self-attention FLOPs ~ 4 * B * H * S^2 * D (QK^T + softmax-weighted V).
ATTN_FLOPS = 4 * BATCH * NUM_HEADS * SEQ_LEN * SEQ_LEN * HEAD_DIM


def _sdpa_configs(mesh_device):
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
        q_chunk_size=128,
        k_chunk_size=128,
        exp_approx_mode=False,  # exact exp keeps the softmax accurate over a long sequence
    )
    # fp32 accumulation in the attention matmuls keeps PCC high at long sequence lengths.
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    return program_config, compute_kernel_config


def _to_mesh(t, mesh_device, dim):
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=dim),
        pad_value=0.0,
    )


def _run_attention(mesh_device, shard_dim, concat_dim):
    """Shared body: build Q/K/V, shard on ``shard_dim``, run SDPA, check PCC, time it."""
    q = torch.rand(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16)
    k = torch.rand(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16)
    v = torch.rand(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16)

    # CPU golden (float32 accumulation for a clean reference).
    golden = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True)

    program_config, compute_kernel_config = _sdpa_configs(mesh_device)
    tt_q = _to_mesh(q, mesh_device, shard_dim)
    tt_k = _to_mesh(k, mesh_device, shard_dim)
    tt_v = _to_mesh(v, mesh_device, shard_dim)

    def sdpa():
        return ttnn.transformer.scaled_dot_product_attention(
            tt_q, tt_k, tt_v, is_causal=True, program_config=program_config, compute_kernel_config=compute_kernel_config
        )

    result = ttnn.to_torch(sdpa(), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=concat_dim))
    result = result[:, :, :SEQ_LEN, :]  # drop any tile padding
    pcc_check(golden, result, pcc=0.98)  # bf16 SDPA with approximate exp

    return time_op(mesh_device, sdpa)


def run_attention_dp(mesh_device, n):
    """Data-parallel: shard the batch dim (each chip runs all heads for fewer sequences)."""
    return _run_attention(mesh_device, shard_dim=0, concat_dim=0)


def run_attention_tp(mesh_device, n):
    """Tensor-parallel: shard the head dim, keep the full batch on each chip (no CCL needed)."""
    return _run_attention(mesh_device, shard_dim=1, concat_dim=1)


def run_attention_2d(mesh_device):
    """2D attention: DATA PARALLEL on mesh axis 0 (batch) x TENSOR PARALLEL on axis 1 (heads).

    Q/K/V are sharded on the batch dim across the rows and on the head dim across the
    columns; heads are independent so no cross-chip communication is needed. So a (2, 4)
    mesh is 2-way batch-DP x 4-way head-TP, and (4, 2) is 4-way batch-DP x 2-way head-TP.
    """
    rows, cols = tuple(mesh_device.shape)
    q = torch.rand(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16)
    k = torch.rand(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16)
    v = torch.rand(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16)

    golden = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True)

    program_config, compute_kernel_config = _sdpa_configs(mesh_device)

    def to_mesh_2d(t):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(0, 1)),
            pad_value=0.0,
        )

    tt_q, tt_k, tt_v = to_mesh_2d(q), to_mesh_2d(k), to_mesh_2d(v)

    def sdpa():
        return ttnn.transformer.scaled_dot_product_attention(
            tt_q, tt_k, tt_v, is_causal=True, program_config=program_config, compute_kernel_config=compute_kernel_config
        )

    result = ttnn.to_torch(
        sdpa(), mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(rows, cols), dims=(0, 1))
    )
    result = result[:, :, :SEQ_LEN, :]  # drop any tile padding
    pcc_check(golden, result, pcc=0.98)

    return time_op(mesh_device, sdpa)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 23887872}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1, 2, 4, 8], indirect=True)
def test_attention_dp(mesh_device):
    n = mesh_device.get_num_devices()
    print(f"\nattention DP {n} chip(s): {run_attention_dp(mesh_device, n) * 1e3:.3f} ms/iter")


@pytest.mark.parametrize("device_params", [{"trace_region_size": 23887872}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1, 2, 4, 8], indirect=True)
def test_attention_tp(mesh_device):
    n = mesh_device.get_num_devices()
    print(f"\nattention TP {n} chip(s): {run_attention_tp(mesh_device, n) * 1e3:.3f} ms/iter")


@pytest.mark.parametrize("device_params", [{"trace_region_size": 23887872}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4), (4, 2)], indirect=True)
def test_attention_2d(mesh_device):
    rows, cols = tuple(mesh_device.shape)
    print(f"\nattention DPxTP {rows}x{cols} mesh: {run_attention_2d(mesh_device) * 1e3:.3f} ms/iter")


def main():
    dp = SpeedupTable("Demo 4a: attention (data parallel)", work_per_iter=ATTN_FLOPS, work_unit="FLOP")
    tp = SpeedupTable(
        "Demo 4b: attention (tensor parallel / head-parallel)", work_per_iter=ATTN_FLOPS, work_unit="FLOP"
    )
    for n in available_mesh_sizes():
        with demo_mesh(n) as mesh_device:
            dp.add(n, run_attention_dp(mesh_device, n))
            tp.add(n, run_attention_tp(mesh_device, n))
    print("\n" + dp.render())
    print("\n" + tp.render())

    if ttnn.get_num_devices() >= 8:
        table2d = SpeedupTable(
            "Demo 4c: attention (2D mesh, data parallel x tensor parallel / head-parallel)",
            work_per_iter=ATTN_FLOPS,
            work_unit="FLOP",
            col0_header="mesh",
        )
        for shape in [(1, 1), (2, 4), (4, 2)]:
            with demo_mesh(shape) as mesh_device:
                table2d.add(shape[0] * shape[1], run_attention_2d(mesh_device), label=f"{shape[0]}x{shape[1]}")
        print("\n" + table2d.render())


if __name__ == "__main__":
    main()
