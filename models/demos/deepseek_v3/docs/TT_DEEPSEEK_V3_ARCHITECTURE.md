# How Tenstorrent Supports DeepSeek V3

A technical architecture-and-code tour of the DeepSeek‑V3 implementation in tt-metal.

This document explains **how the DeepSeek‑V3 architecture maps onto Tenstorrent hardware**
and **where each piece lives in the source tree**. It is written for an engineer who is new
to this code and needs to navigate it quickly: every section ends with a **Key files** list
of clickable, workspace-relative paths.

> **How to read this doc:** If you read nothing else, read
> [§2 The Core Design Pattern](#2-the-core-design-pattern-3-stage-module-model) first.
> The entire codebase is structured around a single "3‑stage module" convention, and the
> rest of the document assumes you understand it.

---

## Architecture → Implementation map

| DeepSeek‑V3 feature | TT module (Python) | Custom ttnn op(s) (C++) |
|---|---|---|
| Multi-head Latent Attention (MLA) | [tt/mla/mla1d.py](../tt/mla/mla1d.py), [tt/mla/mla2d.py](../tt/mla/mla2d.py) | [experimental/deepseek/mla/matmul_wo](../../../../ttnn/cpp/ttnn/operations/experimental/deepseek/mla/matmul_wo) |
| Mixture of Experts (routed + shared) | [tt/moe.py](../tt/moe.py), [tt/moe_optimized.py](../tt/moe_optimized.py), [tt/moe_gate.py](../tt/moe_gate.py), [tt/experts.py](../tt/experts.py), [tt/mlp/shared_expert.py](../tt/mlp/shared_expert.py) | [moe/deepseek_moe_gate](../../../../ttnn/cpp/ttnn/operations/experimental/deepseek/moe/deepseek_moe_gate), [deepseek_moe_post_combine_tilize](../../../../ttnn/cpp/ttnn/operations/experimental/deepseek_moe_post_combine_tilize), [deepseek_prefill/*](../../../../ttnn/cpp/ttnn/operations/experimental/deepseek_prefill) |
| RoPE + YaRN scaling | [tt/rope.py](../tt/rope.py) | — |
| Multi-Token Prediction (MTP) | [tt/mtp.py](../tt/mtp.py) | — |
| Decoder stack / model assembly | [tt/model/row_batched_model.py](../tt/model/row_batched_model.py), [tt/decoder_block/](../tt/decoder_block) | — |
| Distributed collectives | [tt/ccl.py](../tt/ccl.py) | [ccl/deepseek_moe_reduce_scatter](../../../../ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_moe_reduce_scatter), [reduction/deepseek_moe_fast_reduce_nc](../../../../ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc) |
| Normalization | [tt/rms_norm/](../tt/rms_norm) | — |
| Weight conversion / loading | [scripts/dequantize_hf_checkpoint.py](../scripts/dequantize_hf_checkpoint.py), [utils/hf_model_utils.py](../utils/hf_model_utils.py) | — |
| Generation / serving | [tt/generator.py](../tt/generator.py), [tt/generator_vllm.py](../tt/generator_vllm.py), [demo/demo.py](../demo/demo.py) | — |

---

## 1. Introduction & Scope

**DeepSeek‑V3** is a ~671B-parameter Mixture-of-Experts language model. Its three
architecturally distinctive features are:

- **MLA (Multi-head Latent Attention):** attention with low-rank compression of the K/V
  representation, dramatically shrinking the KV cache.
- **MoE (Mixture of Experts):** each MoE layer has **256 routed experts** plus **1 shared
  expert**, with **8 experts activated per token**.
- **MTP (Multi-Token Prediction):** an auxiliary head that predicts the next-next token,
  enabling speculative decoding.

The reference configuration ([reference/config.json](../reference/config.json)) pins the
exact shapes this implementation targets:

| Param | Value | Meaning |
|---|---|---|
| `num_hidden_layers` | 61 | total decoder layers |
| `first_k_dense_replace` | 3 | first 3 layers are dense MLP, the rest are MoE |
| `n_routed_experts` | 256 | routed experts per MoE layer |
| `n_shared_experts` | 1 | always-on shared expert |
| `num_experts_per_tok` | 8 | top-k routing width |
| `q_lora_rank` | 1536 | Q low-rank dimension (MLA) |
| `kv_lora_rank` | 512 | KV latent dimension (MLA) |
| `qk_nope_head_dim` / `qk_rope_head_dim` | 128 / 64 | non-RoPE and RoPE parts of the Q/K head |
| `v_head_dim` | 128 | value head dimension |
| `num_nextn_predict_layers` | 1 | MTP heads |

**Scope of this document.** This covers the **main production implementation** at
[models/demos/deepseek_v3/](../), the TT‑NN pipeline targeting `deepseek-ai/DeepSeek-R1-0528`
and compatible V3 checkpoints. Two sibling implementations and the training stack are
described briefly in [Appendix B](#appendix-b-sibling-implementations-orientation-only).

**Target hardware.** Galaxy (Wormhole), in **2x or 4x multi-host** configurations. A single
Galaxy (`MESH_DEVICE=TG`) can run the demo in a reduced-layer (5-layer) data-parallel mode
for development, and most submodule unit tests run on a single Galaxy.

**Key files**
- [models/demos/deepseek_v3/README.md](../README.md) — setup, multi-host launch, demo CLI, vLLM serving
- [reference/config.json](../reference/config.json) — the 671B model config
- [reference/configuration_deepseek.py](../reference/configuration_deepseek.py) — `DeepseekV3Config` schema

---

## 2. The Core Design Pattern (3-stage module model)

The single most important thing to understand: **modules are not instantiated objects.**
Each module (MLA, MoE, MLP, Embedding, …) is a class used purely as a **namespace of
classmethods**. This deliberately separates the *stateless* description of the model from
the *stateful* weights, so the same weights can be reused under different operator configs,
and prefill vs decode can be tuned independently. The convention is documented in
[tt/README.md](../tt/README.md) and enforced via [utils/abstract_module.py](../utils/abstract_module.py).

Every module implements the same lifecycle, executed in **three stages**:

1. **`convert_weights(hf_config, state_dict, output_path, mesh_device) -> WeightConfig`**
   Converts PyTorch weights into TTNN tensors (choosing dtype, layout, sharding across the
   mesh) and returns a `WeightConfig` — a dict mapping each op's keyword argument (e.g.
   `["w1"]["input_tensor_b"]`) to the resulting tensor. For the DeepSeek‑V3 runtime these
   are usually materialized in memory rather than written to an on-disk cache.

2. **`prefill_model_config(...)` / `decode_model_config(...)` -> ModelConfig**
   Build the *static* operator configuration: a nested dict of dataclass instances
   (`LinearConfig`, `MulConfig`, `AllReduceConfig`, …) drawn from
   [utils/config_dataclass.py](../utils/config_dataclass.py). Prefill and decode get
   **separate** configs because their performance characteristics differ sharply (DRAM,
   long sequences, low-fidelity math for prefill; L1-sharded, single-token, high-fidelity
   for decode).

3. **`create_state(...)` + `run_config(...)` -> RunConfig, then `forward_prefill` / `forward_decode`**
   `create_state` holds non-weight runtime tensors (e.g. the KV cache). `run_config`
   ([utils/run_config.py](../utils/run_config.py)) hierarchically merges the model config,
   the weight config, and the state into a single `RunConfig`. Forward passes then unpack
   configs directly into ttnn calls: `ttnn.linear(x, **cfg["w1"])`.

Two helper sentinels make stage 3 work: **`FromWeightConfig`** marks a config slot that
should be filled by a tensor loaded from the `WeightConfig`, and **`MeshDeviceStub`** marks
a slot that should be replaced with the actual `ttnn.MeshDevice` at run time. All op config
dataclasses inherit from **`OpConfigBase`**, which gives them dict-like access and—crucially—
**filters out `None` fields** during `**cfg` expansion, so `None` means "use the ttnn
default" rather than "pass None".

**Key files**
- [tt/README.md](../tt/README.md) — the canonical description of this pattern
- [utils/config_dataclass.py](../utils/config_dataclass.py) — `OpConfigBase`, `FromWeightConfig`, `MeshDeviceStub`, and all op configs
- [utils/run_config.py](../utils/run_config.py) — the hierarchical merge into a `RunConfig`
- [utils/abstract_module.py](../utils/abstract_module.py) — base contract for modules
- [utils/config_helpers.py](../utils/config_helpers.py) — grid/sharding/sequence helpers (`find_prefill_grid`, `dram_shard_core_grid_for_k`, …)

---

## 3. Model Assembly & Decoder Stack

The full model is assembled in [tt/model/row_batched_model.py](../tt/model/row_batched_model.py)
(`RowBatchedModel`). The structure mirrors the reference architecture:

1. **Token embedding** ([tt/embedding/](../tt/embedding)) — `Embedding1D` / `Embedding2D`.
2. **First `first_k_dense_replace` (=3) layers**: dense decoder blocks using a plain MLP.
3. **Remaining layers (4…61)**: MoE decoder blocks (shared + routed experts).
4. **Final RMSNorm** ([tt/rms_norm/](../tt/rms_norm)).
5. **LM head** ([tt/lm_head1d.py](../tt/lm_head1d.py), `LMHead1D`).
6. **Optional MTP head** ([tt/mtp.py](../tt/mtp.py)) for speculative decoding.

Each decoder block is `Attention (MLA) → residual → MLP/MoE → residual`, with RMSNorm
before each sub-layer. The block hierarchy has a base (`DecoderBlockBase`), a 2D-mesh base
(`DecoderBlock2DBase`), the dense variant (`DecoderBlock2D`), and the MoE variant
(`MoEDecoderBlock2D`). The 1D vs 2D split reflects whether the block is laid out across one
mesh axis or two (tensor-parallel × expert/data-parallel) — see
[§8 Distributed Execution](#8-distributed-execution--collectives).

**Key files**
- [tt/model/row_batched_model.py](../tt/model/row_batched_model.py) — `RowBatchedModel`, full assembly
- [tt/decoder_block/decoder_block_base.py](../tt/decoder_block/decoder_block_base.py) — `DecoderBlockBase`
- [tt/decoder_block/decoder_block_2d_base.py](../tt/decoder_block/decoder_block_2d_base.py) — `DecoderBlock2DBase`
- [tt/decoder_block/decoder_block_2d.py](../tt/decoder_block/decoder_block_2d.py) — `DecoderBlock2D` (dense)
- [tt/decoder_block/moe_decoder_block_2d.py](../tt/decoder_block/moe_decoder_block_2d.py) — `MoEDecoderBlock2D`
- [tt/embedding/embedding1d.py](../tt/embedding/embedding1d.py), [tt/embedding/embedding2d.py](../tt/embedding/embedding2d.py)
- [tt/lm_head1d.py](../tt/lm_head1d.py) — `LMHead1D`
- [tt/rms_norm/rms_norm.py](../tt/rms_norm/rms_norm.py), [tt/rms_norm/distributed_rms_norm.py](../tt/rms_norm/distributed_rms_norm.py), [tt/rms_norm/rms_norm_base.py](../tt/rms_norm/rms_norm_base.py)

---

## 4. Multi-head Latent Attention (MLA)

MLA replaces standard multi-head attention with a **low-rank latent** formulation. Instead
of caching full K and V, it caches a compact `kv_lora_rank` (512) latent plus a small RoPE
component, then expands to per-head K/V on the fly. This is what makes DeepSeek's KV cache
small enough to serve long contexts on a fixed device budget.

The projection chain implemented in `MLA1D` / `MLA2D`:

- **`wq_a` → `wq_b`**: Q is projected down to `q_lora_rank` (1536), then up to
  `num_heads × (qk_nope_head_dim + qk_rope_head_dim)`.
- **`kv_a_proj_with_mqa`**: produces the `kv_lora_rank` (512) latent plus the
  `qk_rope_head_dim` (64) RoPE part.
- **`wkv_b1` / `wkv_b2`**: expand the latent into per-head K (`qk_nope_head_dim`) and V
  (`v_head_dim`) — implemented as batched matmuls.
- **`o_proj` (wo)**: output projection, fused with a ring all-reduce via the custom op
  [experimental/deepseek/mla/matmul_wo](../../../../ttnn/cpp/ttnn/operations/experimental/deepseek/mla/matmul_wo).

The attention score/softmax/▢V step uses a **FlashMLA**-style kernel configured with
`head_dim_v = kv_lora_rank` (the latent acts as the value dimension), with causal masking
and a **paged KV cache**. RoPE is applied to only the rope portion of Q/K (see [§6](#6-rope-with-yarn-scaling)).

**Prefill vs decode** differ in layout: prefill processes long sequences in DRAM, chunked
across mesh rows (see `PrefillChunkSizes` in [utils/config_dataclass.py](../utils/config_dataclass.py)),
with low-fidelity (LoFi) math; decode runs single-token, L1-width-sharded, with HiFi math.
`MLA2D` extends `MLA1D` to lay the heads/projections across a 2D mesh.

**Key files**
- [tt/mla/mla1d.py](../tt/mla/mla1d.py) — `MLA1D` (base implementation, 1D mesh)
- [tt/mla/mla2d.py](../tt/mla/mla2d.py) — `MLA2D` (2D mesh, extends `MLA1D`)
- [ttnn/cpp/ttnn/operations/experimental/deepseek/mla/matmul_wo](../../../../ttnn/cpp/ttnn/operations/experimental/deepseek/mla/matmul_wo) — fused ring all-reduce output matmul
- [tests/test_mla.py](../tests/test_mla.py) — MLA correctness tests

---

## 5. Mixture of Experts (MoE)

Each MoE layer routes every token to **8 of 256** routed experts (top-k) **and** runs a
single **shared expert** on all tokens; the two contributions are summed. On hardware this
becomes a four-phase pipeline: **gate → dispatch → expert FFN → combine/reduce**.

- **Gate / routing** ([tt/moe_gate.py](../tt/moe_gate.py), `MoEGate`): computes the routing
  scores with sigmoid gating + `score_correction_bias`, applies grouped top-k, and emits the
  per-token expert assignment and weights. It is backed by the custom op
  [moe/deepseek_moe_gate](../../../../ttnn/cpp/ttnn/operations/experimental/deepseek/moe/deepseek_moe_gate)
  (with a companion matmul [moe/moe_gate_mm](../../../../ttnn/cpp/ttnn/operations/experimental/deepseek/moe/moe_gate_mm)),
  surfaced to Python via [tt/deepseek_moe_gate/op.py](../tt/deepseek_moe_gate/op.py).
- **Experts** ([tt/experts.py](../tt/experts.py), `Experts`): the routed expert FFNs
  (`gate_proj`, `up_proj`, `down_proj`) sharded across mesh rows (**expert parallelism**).
  Supports both legacy per-expert and stacked checkpoints; weights are dequantized to bf16
  and stored compactly (e.g. bf8_b / bf4_b).
- **Shared expert** ([tt/mlp/shared_expert.py](../tt/mlp/shared_expert.py), `SharedExpert`):
  an always-on FFN sized by `moe_intermediate_size`, computed in parallel and added back.
- **Dispatch / combine / reduce**: tokens are routed to the devices holding their experts
  and the outputs are gathered back. The standard path is in [tt/moe.py](../tt/moe.py) (`MoE`);
  an optimized ring-fabric variant for the quad mesh is in
  [tt/moe_optimized.py](../tt/moe_optimized.py) (`MoEOptimized`), which uses
  [deepseek_moe_post_combine_tilize](../../../../ttnn/cpp/ttnn/operations/experimental/deepseek_moe_post_combine_tilize)
  and fused reduce-scatter.

The dense (non-MoE) layers and the per-block dense MLP use
[tt/mlp/mlp.py](../tt/mlp/mlp.py) (`MLP`), [tt/mlp/non_expert.py](../tt/mlp/non_expert.py)
(`NonExpert`), and [tt/mlp/mlp_dequant.py](../tt/mlp/mlp_dequant.py) (`MLPDequant`).

**Custom ttnn ops backing MoE**

- Decode/inference reductions:
  [reduction/deepseek_grouped_gate](../../../../ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_grouped_gate),
  [reduction/deepseek_moe_fast_reduce_nc](../../../../ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc),
  [reduction/deepseek_moe_fast_reduce_nc_fused](../../../../ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc_fused)
- Cross-device combine:
  [ccl/deepseek_moe_reduce_scatter](../../../../ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_moe_reduce_scatter)
- Prefill MoE family (token bookkeeping at scale):
  [deepseek_prefill/](../../../../ttnn/cpp/ttnn/operations/experimental/deepseek_prefill) —
  `moe_grouped_topk`, `dispatch`, `extract`, `insert`, `combine`, `masked_bincount`,
  `offset_cumsum`, `post_combine_reduce`, `routed_expert_ffn`.

**Key files**
- [tt/moe.py](../tt/moe.py), [tt/moe_optimized.py](../tt/moe_optimized.py), [tt/moe_gate.py](../tt/moe_gate.py), [tt/deepseek_moe_gate/op.py](../tt/deepseek_moe_gate/op.py)
- [tt/experts.py](../tt/experts.py), [tt/mlp/shared_expert.py](../tt/mlp/shared_expert.py), [tt/mlp/non_expert.py](../tt/mlp/non_expert.py), [tt/mlp/mlp.py](../tt/mlp/mlp.py), [tt/mlp/mlp_dequant.py](../tt/mlp/mlp_dequant.py)
- [tests/test_moe.py](../tests/test_moe.py), [tests/test_moe_gate.py](../tests/test_moe_gate.py), [tests/test_moe_experts.py](../tests/test_moe_experts.py)

---

## 6. RoPE with YaRN Scaling

Rotary position embeddings are set up by `RotarySetup` in [tt/rope.py](../tt/rope.py). Two
points matter for correctness on TT hardware:

- **Layout conversion.** HuggingFace stores cos/sin interleaved as
  `[t1..t_{d/2}, t1..t_{d/2}]`; the TT kernels expect the "meta" layout
  `[t1, t1, t2, t2, …]`. `RotarySetup` performs this remap when building the cos/sin tables.
- **YaRN long-context scaling.** The reference uses `DeepseekV3YarnRotaryEmbedding`. The
  YaRN parameters — `factor`, `mscale` (applied as `0.1 * mscale * log(factor) + 1.0`),
  `beta_fast`, `beta_slow`, `original_max_position_embeddings`, `rope_theta` — are carried
  through `RopeConfig` / `YarnConfig` in [utils/config_dataclass.py](../utils/config_dataclass.py).

RoPE is applied only to the `qk_rope_head_dim` (64) portion of Q/K. Decode shards the cos/sin
tables height-wise across rows for batch parallelism; prefill keeps the full table in DRAM.

**Key files**
- [tt/rope.py](../tt/rope.py) — `RotarySetup`
- [utils/config_dataclass.py](../utils/config_dataclass.py) — RoPE/YaRN config fields

---

## 7. Multi-Token Prediction (MTP)

MTP adds `num_nextn_predict_layers` (=1) auxiliary head(s) that predict the token after next,
enabling speculative decoding. Implemented as `MTP2D` in [tt/mtp.py](../tt/mtp.py):

- **Shares** the main model's token embedding and LM head.
- Has its **own decoder block** (a full MoE layer) plus dedicated norms
  (`hidden_norm`, `token_norm`, `head_norm`).
- Projects the concatenated `[hidden, next-token-embedding]` through `eh_proj`
  (`hidden_size → 2·hidden_size`) before its decoder block, then reuses the shared head.

The decode driver logic lives alongside it (`_MtpDecodeBootstrap`, `_MtpDecodeLoopResult`,
`_MtpPromptLayout` in the same file), and is exercised through the generator.

**Key files**
- [tt/mtp.py](../tt/mtp.py) — `MTP2D` and decode helpers
- [tests/test_mtp.py](../tests/test_mtp.py) — MTP tests

---

## 8. Distributed Execution & Collectives

DeepSeek‑V3 only runs across a **2D device mesh** (multi-host Galaxy). The two axes carry
different forms of parallelism:

- **Columns (axis 1) — Tensor Parallel:** hidden dimensions are split across columns
  (`hidden_size / TP`).
- **Rows (axis 0) — Expert / Data Parallel:** the 256 experts are distributed across rows;
  prefill additionally chunks the sequence dimension across rows.

The collective primitives are wrapped in [tt/ccl.py](../tt/ccl.py) (`CCL`), which manages the
semaphores and links for:

- **All-Gather** — assemble full hidden states before Q/KV/output projections.
- **All-to-All (dispatch/combine)** — route tokens to expert-holding devices and gather
  results back (`AllToAllDispatchConfig`, `AllToAllCombineConfig` in
  [utils/config_dataclass.py](../utils/config_dataclass.py)).
- **Reduce-Scatter** — sum expert outputs back to token locations, including the fused fast
  path ([reduction/deepseek_moe_fast_reduce_nc_fused](../../../../ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc_fused))
  and the dedicated [ccl/deepseek_moe_reduce_scatter](../../../../ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_moe_reduce_scatter).

Memory placement follows the prefill/decode split: prefill stages tensors in **DRAM**
(long sequences), decode uses **L1** width/height sharding for latency. Grid sizes and shard
specs are computed by helpers in [utils/config_helpers.py](../utils/config_helpers.py).

**Key files**
- [tt/ccl.py](../tt/ccl.py) — `CCL`, collective orchestration
- [utils/config_helpers.py](../utils/config_helpers.py) — mesh/grid/shard helpers
- [utils/config_dataclass.py](../utils/config_dataclass.py) — `AllGatherConfig`, `AllReduceConfig`, `ReduceScatterConfig`, `AllToAll*Config`, `PrefillChunkSizes`

---

## 9. Weights: Conversion & Loading

DeepSeek‑V3 ships as **FP8** safetensors. The recommended runtime path converts them once
to a **stacked dequantized** checkpoint, then loads tensors directly into TTNN memory (no
on-disk TT weight cache):

1. **Dequantize + stack:** [scripts/dequantize_hf_checkpoint.py](../scripts/dequantize_hf_checkpoint.py)
   produces a `*-dequantized-stacked` directory. Point `DEEPSEEK_V3_HF_MODEL` (or
   `--model-path`) at it.
2. **Load:** [utils/hf_model_utils.py](../utils/hf_model_utils.py) discovers/validates the
   checkpoint and tokenizer; [utils/lazy_state_dict.py](../utils/lazy_state_dict.py) loads
   weights lazily to bound host memory.
3. **Convert per module:** each module's `convert_weights` turns PyTorch tensors into TTNN
   tensors with appropriate sharding and dtype (bf16 for norms/gates/output projections,
   bf8_b/bf4_b for the heavy expert/MLP projections). Metadata/versioning is tracked by
   `SavedWeight` in [utils/config_dataclass.py](../utils/config_dataclass.py) and
   [utils/weight_config.py](../utils/weight_config.py).

A legacy/BSPM TT weight cache can still be consumed with `--use-weight-cache --cache-dir`,
but caches predating the current `SavedWeight` versioning are rejected and must be
regenerated.

**Key files**
- [scripts/dequantize_hf_checkpoint.py](../scripts/dequantize_hf_checkpoint.py) — main conversion path
- [scripts/convert_bspm_weights.py](../scripts/convert_bspm_weights.py), [scripts/validate_weight_cache.py](../scripts/validate_weight_cache.py), [scripts/prepare_quad_ring_hf_checkpoint.py](../scripts/prepare_quad_ring_hf_checkpoint.py)
- [utils/weight_config.py](../utils/weight_config.py), [utils/hf_model_utils.py](../utils/hf_model_utils.py), [utils/lazy_state_dict.py](../utils/lazy_state_dict.py)

---

## 10. Running the Model: Demo, Generators & Serving

**Demo entry point** ([demo/demo.py](../demo/demo.py)) supports: full-model generation from
real weights, `--random-weights` smoke tests (single dense layer), `--token-accuracy`
teacher-forced verification, and batch generation from a `--prompts-file` JSON. It logs
prefill/decode timing and tokens/sec, and can stream the first prompt with
`--early_print_first_user`.

**Generation engine.** [tt/generator.py](../tt/generator.py) implements the batch-parallel
(`bp`) `DeepseekGenerator`: prefill + decode loop, paged KV cache, optional MTP, teacher
forcing, and tracing (`--enable-trace`). [tt/generator_vllm.py](../tt/generator_vllm.py)
adapts it to the vLLM TT backend (`DeepseekV3ForCausalLM`).

**Multi-host launch.** [scripts/launch_multihost_galaxy.py](../scripts/launch_multihost_galaxy.py)
selects the cluster config, sets `MESH_DEVICE` (`DUAL` for 2x, `QUAD` for 4x), exports the
model/cache env vars, and wraps the command in `tt-run` (MPI). Example:

```bash
./models/demos/deepseek_v3/scripts/launch_multihost_galaxy.py 2x -- \
  python models/demos/deepseek_v3/demo/demo.py \
    --model-path $DEEPSEEK_V3_HF_MODEL "Write a haiku about the sea"
```

**Key files**
- [demo/demo.py](../demo/demo.py), [demo/README.md](../demo/README.md)
- [tt/generator.py](../tt/generator.py) — `DeepseekGenerator`
- [tt/generator_vllm.py](../tt/generator_vllm.py) — `DeepseekV3ForCausalLM` (vLLM)
- [scripts/launch_multihost_galaxy.py](../scripts/launch_multihost_galaxy.py)

---

## 11. Testing & CI

Testing spans three layers:

- **Submodule pytests** — one per module, runnable on a single Galaxy:
  [tests/test_mla.py](../tests/test_mla.py), [tests/test_moe.py](../tests/test_moe.py),
  [tests/test_moe_gate.py](../tests/test_moe_gate.py), [tests/test_moe_experts.py](../tests/test_moe_experts.py),
  [tests/test_mlp.py](../tests/test_mlp.py), [tests/test_rms_norm.py](../tests/test_rms_norm.py),
  [tests/test_embedding.py](../tests/test_embedding.py), [tests/test_lm_head1d.py](../tests/test_lm_head1d.py),
  [tests/test_decoder_block.py](../tests/test_decoder_block.py), [tests/test_mtp.py](../tests/test_mtp.py),
  [tests/test_sampling.py](../tests/test_sampling.py).
- **Integration** — [tests/test_model.py](../tests/test_model.py),
  [tests/test_row_batched_model.py](../tests/test_row_batched_model.py),
  [tests/test_generator_teacher_forcing.py](../tests/test_generator_teacher_forcing.py),
  [tests/test_generator_vllm.py](../tests/test_generator_vllm.py).
- **Infra/weights** — [tests/test_dequantize.py](../tests/test_dequantize.py),
  [tests/test_get_weight_config.py](../tests/test_get_weight_config.py),
  [tests/test_hf_model_utils.py](../tests/test_hf_model_utils.py),
  [tests/test_lazy_state_dict.py](../tests/test_lazy_state_dict.py),
  [tests/test_config_helpers.py](../tests/test_config_helpers.py).

Mesh-device fixtures and device auto-detection live in [conftest.py](../conftest.py).

**CI workflows** (Galaxy hardware): [galaxy-deepseek-tests.yaml](../../../../.github/workflows/galaxy-deepseek-tests.yaml)
(+ `-impl`), the prefill suite [galaxy-deepseek-prefill-tests.yaml](../../../../.github/workflows/galaxy-deepseek-prefill-tests.yaml)
(+ `-impl`), and the multi-host pipeline
[multi-host-deepseekv3.yaml](../../../../.github/workflows/multi-host-deepseekv3.yaml).

---

## 12. Reference Model

A cleaned-up HuggingFace reference is kept in-tree for numerical parity (PCC) checks against
the TT implementation — every submodule test compares its TTNN output to the reference.

**Key files**
- [reference/modeling_deepseek.py](../reference/modeling_deepseek.py) — PyTorch reference model
- [reference/configuration_deepseek.py](../reference/configuration_deepseek.py) — `DeepseekV3Config`
- [reference/reference_utils.py](../reference/reference_utils.py) — reference helpers
- [reference/config.json](../reference/config.json) — 671B config

---

## Appendix A. Directory Map

```
models/demos/deepseek_v3/
├── README.md                 # setup, multi-host launch, demo CLI, vLLM
├── conftest.py               # pytest mesh-device fixtures / auto-detect
├── demo/                     # entry points
│   ├── demo.py               # main demo (full / random / teacher-forced / batch)
│   └── README.md
├── reference/                # HF reference model (parity checks)
│   ├── modeling_deepseek.py
│   ├── configuration_deepseek.py
│   └── config.json
├── tt/                       # TTNN model implementation
│   ├── model/row_batched_model.py   # full model assembly
│   ├── decoder_block/        # dense + MoE decoder blocks (1D/2D)
│   ├── mla/                  # MLA1D / MLA2D (latent attention)
│   ├── moe.py, moe_optimized.py, moe_gate.py, experts.py   # MoE
│   ├── deepseek_moe_gate/op.py      # gate custom-op wrapper
│   ├── mlp/                  # dense MLP + shared/non expert
│   ├── embedding/            # Embedding1D / Embedding2D
│   ├── rms_norm/             # RMSNorm + distributed
│   ├── rope.py               # RoPE + YaRN
│   ├── mtp.py                # multi-token prediction
│   ├── lm_head1d.py          # LM head
│   ├── ccl.py                # collective communication wrapper
│   ├── generator.py          # batch-parallel generator
│   └── generator_vllm.py     # vLLM backend
├── utils/                    # the 3-stage framework + helpers
│   ├── config_dataclass.py   # op config dataclasses, OpConfigBase, sentinels
│   ├── run_config.py         # config/weight/state merge
│   ├── config_helpers.py     # grid/shard/sequence helpers
│   ├── abstract_module.py    # module base contract
│   ├── weight_config.py, hf_model_utils.py, lazy_state_dict.py
└── scripts/                  # weight conversion + multi-host launch
    ├── dequantize_hf_checkpoint.py
    └── launch_multihost_galaxy.py
```

Custom C++ ops added for DeepSeek live outside this directory under
[ttnn/cpp/ttnn/operations/experimental/](../../../../ttnn/cpp/ttnn/operations/experimental)
(`deepseek/`, `deepseek_prefill/`, `deepseek_moe_post_combine_tilize/`,
`ccl/deepseek_moe_reduce_scatter/`, `reduction/deepseek_*`).

---

## Appendix B. Sibling Implementations (orientation only)

These are **separate** implementations, out of scope for this document but worth knowing:

- **`models/demos/deepseek_v3_b1/`** — "Blitz": a lower-level, performance-focused
  implementation built on a *unified kernel descriptor* abstraction and hand-tuned micro-ops,
  with scaleout configs for single/dual/super pods.
- **`models/demos/deepseek_v3_d_p/`** — disaggregated **prefill** stage, organized around an
  explicit MoE **dispatch/combine** architecture (`tt/moe/`), with its own detailed README.
- **`tt-train/sources/ttml/ttml/models/deepseek/`** — DeepSeek **MoE training** (forward +
  autograd), with model configs under `tt-train/configs/model_configs/moe/`.
