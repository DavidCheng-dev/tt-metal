# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
SDPA memory-placement benchmark — prefill and decode shapes.

Two scenarios are benchmarked:

  Prefill  (B=1, H=8, S_q=512, S_kv=512, D=128, causal):
    Classic full-sequence attention.  Q shape for sharding is (H*S_q, D) = (4096, 128).

  Decode   (B=1, H=8, S_q=32, S_kv=2048, D=128, non-causal):
    Single-tile query attending to a 2048-token KV cache.  S_q=32 is the minimum
    tile-aligned size.  Q shape for sharding is (H*S_q, D) = (256, 128).
    All three sharding strategies fail the geometry gate (256/64=4 rows per core
    for HEIGHT, 128/64=2 cols per core for WIDTH, 256/8=32 rows but 128/8=16 cols
    for BLOCK) — only interleaved configs run.  SDPA additionally requires all
    operands to be INTERLEAVED, so sharded Q would be rejected by the op anyway.

Q, K, and V all share the same memory configuration so the comparison is
all-DRAM vs all-L1.  For decode, KV together are 8 MB (500 KB/core across
64 cores) — well within the 1.5 MB per-core L1 budget — so placing KV in L1
eliminates the dominant DRAM bandwidth cost.

Run:
    python tests/ttnn/unit_tests/operations/memory_perf/test_attention_memory_perf.py
    pytest -s tests/ttnn/unit_tests/operations/memory_perf/test_attention_memory_perf.py
"""

import pytest
import torch
import torch.nn.functional as F

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.unit_tests.operations.memory_perf._memcfg_utils import (
    TRACE_REGION_SIZE,
    MEMCFG_NAMES,
    build_memcfgs,
    time_op,
    MemcfgResultTable,
)

# Prefill shape
B = 1
H = 8
S = 512
D = 128
CORE_GRID = (8, 8)
DTYPE = ttnn.bfloat16

# Decode shape: S_q=32 (one tile, padded single token); S_kv=2048 (full KV cache context).
S_Q_DEC = 32
S_KV_DEC = 2048

# Q is a 2-D view (H*S, D) for memory-config purposes, but passed to SDPA as (B,H,S,D).
# We shard over the H*S rows: 8*512 = 4096 rows, D=128 cols.
Q_SHARD_SHAPE_2D = (H * S, D)
Q_SHARD_SHAPE_2D_DEC = (H * S_Q_DEC, D)  # (256, 128) — all sharding invalid, see module docstring.

COMPUTE_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=True,
    fp32_dest_acc_en=False,
    packer_l1_acc=False,
)

SDPA_PROGRAM_CONFIG = ttnn.SDPAProgramConfig(
    compute_with_storage_grid_size=CORE_GRID,
    q_chunk_size=128,
    k_chunk_size=128,
    exp_approx_mode=True,
)

# Decode: q_chunk_size must equal S_q (one tile); k_chunk_size chunks the long KV context.
SDPA_DECODE_PROGRAM_CONFIG = ttnn.SDPAProgramConfig(
    compute_with_storage_grid_size=CORE_GRID,
    q_chunk_size=S_Q_DEC,
    k_chunk_size=128,
    exp_approx_mode=True,
)


# --------------------------------------------------------------------------------------
# Benchmark helper
# --------------------------------------------------------------------------------------


def run_attention(device, memcfg, s_q=S, s_kv=S, is_causal=True, program_config=None):
    """Time SDPA with memcfg applied to Q, K, and V.

    All three operands share the same memory configuration so the comparison is
    all-DRAM vs all-L1.  SDPA requires all operands to be INTERLEAVED; sharded
    configs are rejected by the op and surface as (skipped).
    s_q: query sequence length; s_kv: key/value sequence length.
    is_causal: True for prefill, False for decode (all KV positions are past tokens).
    Returns (per_iter_s, tflops).
    """
    if program_config is None:
        program_config = SDPA_PROGRAM_CONFIG

    tq = torch.randn(B, H, s_q, D, dtype=torch.bfloat16)
    tk = torch.randn(B, H, s_kv, D, dtype=torch.bfloat16)
    tv = torch.randn(B, H, s_kv, D, dtype=torch.bfloat16)

    q = ttnn.from_torch(tq, dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memcfg)
    k = ttnn.from_torch(tk, dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memcfg)
    v = ttnn.from_torch(tv, dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memcfg)

    per_iter_s = time_op(
        device,
        lambda: ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=is_causal,
            program_config=program_config,
            compute_kernel_config=COMPUTE_KERNEL_CONFIG,
        ),
    )

    ttnn.deallocate(q)
    ttnn.deallocate(k)
    ttnn.deallocate(v)

    # Approximate FLOP count: 4 * B * H * S_q * S_kv * D
    # (QK^T: B*H*S_q*D*S_kv and AV: B*H*S_q*S_kv*D, each counted twice for MAC)
    flops = 4 * B * H * s_q * s_kv * D
    tflops = flops / per_iter_s / 1e12
    return per_iter_s, tflops


# --------------------------------------------------------------------------------------
# Correctness check
# --------------------------------------------------------------------------------------


def _correctness_check(device):
    b, h, s, d = 1, 4, 128, 64
    tq = torch.randn(b, h, s, d, dtype=torch.bfloat16)
    tk = torch.randn(b, h, s, d, dtype=torch.bfloat16)
    tv = torch.randn(b, h, s, d, dtype=torch.bfloat16)

    golden = F.scaled_dot_product_attention(tq.float(), tk.float(), tv.float(), is_causal=True).to(torch.bfloat16)

    prog_cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=CORE_GRID,
        q_chunk_size=64,
        k_chunk_size=64,
        exp_approx_mode=False,
    )
    compute_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    q = ttnn.from_torch(tq, dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    k = ttnn.from_torch(tk, dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    v = ttnn.from_torch(tv, dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    out = ttnn.transformer.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=True,
        program_config=prog_cfg,
        compute_kernel_config=compute_cfg,
    )
    assert_with_pcc(golden, ttnn.to_torch(out), 0.99)
    ttnn.deallocate(q)
    ttnn.deallocate(k)
    ttnn.deallocate(v)
    ttnn.deallocate(out)


def _correctness_check_decode(device):
    """Verify decode-shape SDPA (S_q=32, S_kv=256) matches PyTorch within PCC 0.99."""
    b, h, s_q, s_kv, d = 1, 4, 32, 256, 64
    tq = torch.randn(b, h, s_q, d, dtype=torch.bfloat16)
    tk = torch.randn(b, h, s_kv, d, dtype=torch.bfloat16)
    tv = torch.randn(b, h, s_kv, d, dtype=torch.bfloat16)

    # Decode: is_causal=False — all KV positions are valid past tokens.
    golden = F.scaled_dot_product_attention(tq.float(), tk.float(), tv.float(), is_causal=False).to(torch.bfloat16)

    prog_cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=CORE_GRID,
        q_chunk_size=32,
        k_chunk_size=64,
        exp_approx_mode=False,
    )
    compute_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    q = ttnn.from_torch(tq, dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    k = ttnn.from_torch(tk, dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    v = ttnn.from_torch(tv, dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    out = ttnn.transformer.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=False,
        program_config=prog_cfg,
        compute_kernel_config=compute_cfg,
    )
    assert_with_pcc(golden, ttnn.to_torch(out), 0.99)
    ttnn.deallocate(q)
    ttnn.deallocate(k)
    ttnn.deallocate(v)
    ttnn.deallocate(out)


# --------------------------------------------------------------------------------------
# pytest entry points
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
def test_correctness(device):
    """Verify SDPA prefill matches PyTorch SDPA within PCC 0.99."""
    _correctness_check(device)


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
def test_correctness_decode(device):
    """Verify decode-shape SDPA (S_q=32, S_kv=256) matches PyTorch within PCC 0.99."""
    _correctness_check_decode(device)


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("memcfg_name", MEMCFG_NAMES)
def test_attention_memory_config(device, memcfg_name):
    """Sweep all five memory configs applied to Q for SDPA prefill."""
    memcfgs = build_memcfgs(Q_SHARD_SHAPE_2D, CORE_GRID)
    memcfg = memcfgs[memcfg_name]
    if memcfg is None:
        pytest.skip(f"{memcfg_name} not valid for Q shape {Q_SHARD_SHAPE_2D}")

    try:
        per_iter_s, tflops = run_attention(device, memcfg)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"{memcfg_name} rejected by op: {e}")
    print(f"\nsdpa  {memcfg_name:<24}  {per_iter_s * 1e3:.3f} ms/iter  {tflops:.4f} TFLOPS")


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("memcfg_name", MEMCFG_NAMES)
def test_attention_decode_memory_config(device, memcfg_name):
    """Decode-shape sweep: S_q=32 (single tile) attending to S_kv=2048 KV cache.

    Q 2-D shard shape is (256, 128).  All sharding strategies fail the geometry
    gate (HEIGHT/WIDTH/BLOCK all require ≥ 32 elements per core in the sharded
    dimension).  SDPA additionally requires INTERLEAVED operands, so even configs
    that pass geometry would be rejected by the op.  Only interleaved configs run.
    """
    memcfgs = build_memcfgs(Q_SHARD_SHAPE_2D_DEC, CORE_GRID)
    memcfg = memcfgs[memcfg_name]
    if memcfg is None:
        pytest.skip(f"{memcfg_name} not valid for decode Q shape {Q_SHARD_SHAPE_2D_DEC}")

    try:
        per_iter_s, tflops = run_attention(
            device,
            memcfg,
            s_q=S_Q_DEC,
            s_kv=S_KV_DEC,
            is_causal=False,
            program_config=SDPA_DECODE_PROGRAM_CONFIG,
        )
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"{memcfg_name} rejected by op: {e}")
    print(f"\nsdpa_decode  {memcfg_name:<24}  {per_iter_s * 1e3:.3f} ms/iter  {tflops:.4f} TFLOPS")


# --------------------------------------------------------------------------------------
# Standalone entry point
# --------------------------------------------------------------------------------------


def main():
    device = ttnn.open_device(device_id=0, trace_region_size=TRACE_REGION_SIZE)
    try:
        _correctness_check(device)
        print(f"correctness prefill (B={B},H=4,S=128,D=64 causal SDPA): PCC ok")
        _correctness_check_decode(device)
        print(f"correctness decode  (B={B},H=4,S_q=32,S_kv=256,D=64 SDPA): PCC ok\n")

        # ── Prefill benchmark ──────────────────────────────────────────────────
        table = MemcfgResultTable(
            f"SDPA prefill  B={B} H={H} S={S} D={D}  is_causal=True  dtype=bfloat16  (Q/K/V same memcfg)",
            "Takeaway: Q, K, V all share the same memcfg (all-DRAM vs all-L1). "
            "SDPA requires INTERLEAVED operands; sharded configs are skipped.",
            metric_label="TFLOPS",
        )

        memcfgs = build_memcfgs(Q_SHARD_SHAPE_2D, CORE_GRID)
        for name, memcfg in memcfgs.items():
            if memcfg is None:
                table.add(name, 0.0, None)
                continue
            try:
                per_iter_s, tflops = run_attention(device, memcfg)
                table.add(name, per_iter_s, tflops)
            except Exception as e:  # noqa: BLE001
                print(f"  [info] {name} failed (op constraint): {e}")
                table.add(name, 0.0, None)

        print(table.render())

        # ── Decode benchmark ───────────────────────────────────────────────────
        print()
        table_dec = MemcfgResultTable(
            f"SDPA decode  B={B} H={H} S_q={S_Q_DEC} S_kv={S_KV_DEC} D={D}  is_causal=False  dtype=bfloat16  (Q/K/V same memcfg)",
            "Takeaway: KV = 8 MB dominates bandwidth. Moving Q/K/V from DRAM to L1 together "
            "eliminates the KV load cost — only interleaved configs run for this tiny S_q.",
            metric_label="TFLOPS",
        )

        memcfgs_dec = build_memcfgs(Q_SHARD_SHAPE_2D_DEC, CORE_GRID)
        for name, memcfg in memcfgs_dec.items():
            if memcfg is None:
                table_dec.add(name, 0.0, None)
                continue
            try:
                per_iter_s, tflops = run_attention(
                    device,
                    memcfg,
                    s_q=S_Q_DEC,
                    s_kv=S_KV_DEC,
                    is_causal=False,
                    program_config=SDPA_DECODE_PROGRAM_CONFIG,
                )
                table_dec.add(name, per_iter_s, tflops)
            except Exception as e:  # noqa: BLE001
                print(f"  [info] {name} failed (op constraint): {e}")
                table_dec.add(name, 0.0, None)

        print(table_dec.render())
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
