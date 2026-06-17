# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Configuration for DeepSeek-V4-Pro (https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro).
# Mirrors models/demos/deepseek_v3/reference/configuration_deepseek.py (DeepseekV3Config),
# adding the fields needed for Heavily Compressed Attention (HCA) / Compressed Sparse
# Attention (CSA): `head_dim`, `o_lora_rank`, `o_groups`, `sliding_window`,
# `compress_ratios`, `compress_rope_theta`, and the (currently unused outside HCA)
# `hc_*` / `index_*` indexer fields.

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


def _default_compress_ratios(num_hidden_layers: int) -> list:
    """Default per-layer compress ratio: layers 0/1 = HCA (128), middle layers alternate
    CSA (4) / HCA (128), final layer = 0 (sliding-window-only MTP layer)."""
    if num_hidden_layers <= 1:
        return [0] * num_hidden_layers
    ratios = [128, 128]
    toggle = [4, 128]
    i = 0
    while len(ratios) < num_hidden_layers - 1:
        ratios.append(toggle[i % 2])
        i += 1
    ratios = ratios[: num_hidden_layers - 1]
    ratios.append(0)
    return ratios


class DeepseekV4Config(PretrainedConfig):
    r"""
    Configuration class for the DeepSeek-V4-Pro hybrid-attention model.

    In addition to the DeepseekV3Config fields, this adds:
        head_dim (`int`, *optional*, defaults to 512):
            Per-head dimension for HCA/CSA attention (Q, K-latent, and V all share this
            width). `qk_nope_head_dim = head_dim - qk_rope_head_dim` and
            `kv_lora_rank = v_head_dim = head_dim`.
        o_lora_rank (`int`, *optional*, defaults to 1024):
            Output dimension of the grouped low-rank attention output projection.
        o_groups (`int`, *optional*, defaults to 16):
            Number of independent groups used by the grouped low-rank output projection.
            `num_heads * v_head_dim` and `o_lora_rank` must both be divisible by this.
        sliding_window (`int`, *optional*, defaults to 128):
            Width of the local (uncompressed) attention window.
        compress_ratios (`List[int]`, *optional*):
            Per-layer compression ratio. `128` selects Heavily Compressed Attention (HCA),
            `4` selects Compressed Sparse Attention (CSA, not implemented here), and `0`
            selects a sliding-window-only (MTP) layer. Defaults to an alternating
            HCA/CSA pattern with a trailing `0`.
        compress_rope_theta (`float`, *optional*, defaults to 160000.0):
            RoPE base for the compressed KV stream (separate from `rope_theta`, which is
            used for the uncompressed Q/K-latent stream).
        hc_mult (`int`, *optional*, defaults to 4):
        hc_sinkhorn_iters (`int`, *optional*, defaults to 20):
        hc_eps (`float`, *optional*, defaults to 1e-6):
            Reserved Heavily-Compressed hyperparameters for a future Sinkhorn-normalized
            gating refinement of the Compressor. Stored but unused by the v1
            softmax-gated-pooling Compressor implemented here.
        index_head_dim, index_n_heads, index_topk:
            Lightning-indexer dimensions used by CSA's top-k sparse selection. Stored but
            unused by HCA.
    """

    model_type = "deepseek_v4"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=129280,
        hidden_size=7168,
        intermediate_size=18432,
        moe_intermediate_size=3072,
        num_hidden_layers=61,
        num_nextn_predict_layers=1,
        num_attention_heads=128,
        num_key_value_heads=1,
        n_shared_experts=1,
        n_routed_experts=384,
        ep_size=1,
        routed_scaling_factor=2.5,
        head_dim=512,
        q_lora_rank=1536,
        o_lora_rank=1024,
        o_groups=16,
        qk_rope_head_dim=64,
        sliding_window=128,
        compress_ratios=None,
        compress_rope_theta=160000.0,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        index_head_dim=128,
        index_n_heads=64,
        index_topk=1024,
        topk_method="noaux_tc",
        n_group=1,
        topk_group=1,
        num_experts_per_tok=6,
        moe_layer_freq=1,
        first_k_dense_replace=1,
        norm_topk_prob=True,
        scoring_func="sqrtsoftplus",
        aux_loss_alpha=0.001,
        seq_aux=True,
        hidden_act="silu",
        max_position_embeddings=1048576,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=0,
        eos_token_id=1,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        swiglu_limit=10.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.ep_size = ep_size
        self.routed_scaling_factor = routed_scaling_factor

        # HCA/CSA attention dimensions (head_dim subsumes V3's separate
        # kv_lora_rank / v_head_dim / qk_nope_head_dim constants).
        self.head_dim = head_dim
        self.q_lora_rank = q_lora_rank
        self.o_lora_rank = o_lora_rank
        self.o_groups = o_groups
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = head_dim - qk_rope_head_dim
        self.kv_lora_rank = head_dim
        self.v_head_dim = head_dim
        self.q_head_dim = head_dim

        self.sliding_window = sliding_window
        self.compress_ratios = (
            compress_ratios if compress_ratios is not None else _default_compress_ratios(num_hidden_layers)
        )
        self.compress_rope_theta = compress_rope_theta
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_topk = index_topk
        self.swiglu_limit = swiglu_limit

        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux

        if num_key_value_heads is None:
            num_key_value_heads = 1
        self.num_key_value_heads = num_key_value_heads

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
