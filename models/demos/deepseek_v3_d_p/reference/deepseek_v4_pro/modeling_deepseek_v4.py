# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
DeepSeek-V4-Pro Heavily Compressed Attention (HCA) reference implementation.

HCA replaces V3's MLA with a hybrid scheme that attends densely to a
`compress_ratio`-times-compressed KV stream (the whole sequence summarized into
`seq_len // compress_ratio` entries) plus a recent uncompressed sliding window, with a
learned per-head attention sink. Everything else (low-rank Q, MLA-style KV latent +
RoPE, grouped low-rank output projection) follows the V3 MLA pattern.

This module is a from-scratch, self-consistent implementation written against the
DeepSeek-V4-Pro `config.json` field set (`head_dim`, `qk_rope_head_dim`, `q_lora_rank`,
`o_lora_rank`, `o_groups`, `sliding_window`, `compress_ratios`, `compress_rope_theta`,
`hc_*`) and the publicly described algorithm (gated-pool Compressor with learned
absolute positional bias `ape`, dense-over-compressed + sliding-window + sink
attention). Several details are not published, so the following design choices were
made explicitly (and are mirrored exactly by the tt-metal composite implementation, so
golden PCC comparisons are self-consistent):

  * Compressor output width ("coff * head_dim" in the public sketch) is fixed to
    `kvpe_dim = kv_lora_rank + qk_rope_head_dim` (576 for V4-Pro), i.e. the same width
    as the per-token KV latent (`kv_a_proj_with_mqa` output). The last `qk_rope_head_dim`
    channels carry RoPE (with `compress_rope_theta`), the first `kv_lora_rank` channels
    are the latent K/V (nope) and get their own RMSNorm (`compress_kv_norm`).
  * The compressed stream's RoPE uses plain (non-YaRN) `DeepseekV3RotaryEmbedding` with
    `base=compress_rope_theta`, positions = compressed-block index (0, 1, 2, ...) — the
    compressed stream is short enough that YaRN long-context extrapolation isn't needed.
  * Causality: compressed block `b` (covering tokens `[b*r, (b+1)*r)`) is visible to a
    query at position `p` iff `b < (p + 1) // r`. The sliding window covers
    `[p - sliding_window + 1, p]`. Together these give full causal coverage with a
    deliberate small overlap at block boundaries (the just-completed block's tokens are
    visible via both its own compressed summary and the window).
  * Attention sink: one learned scalar logit per head (`attn_sink`), included as an
    extra always-visible "key" with no value contribution — it only inflates the
    softmax denominator (standard streaming-attention sink behavior).
  * Output projection: `o_groups`-way grouped low-rank. `num_heads * v_head_dim` is
    split into `o_groups` equal chunks, each independently projected down to
    `o_lora_rank // o_groups`, concatenated to `o_lora_rank`, RMSNorm'd, then `wo`.

The HCA layer is selected by `config.compress_ratios[layer_idx] > 0` (128 = HCA,
4 = CSA — not implemented here, 0 = sliding-window-only MTP layer).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.demos.deepseek_v3.reference.modeling_deepseek import (
    DeepseekV3RMSNorm,
    DeepseekV3RotaryEmbedding,
    DeepseekV3YarnRotaryEmbedding,
    apply_rotary_pos_emb,
    yarn_get_mscale,
)
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_pro.configuration_deepseek_v4 import DeepseekV4Config


class Compressor(nn.Module):
    """Turns every `compress_ratio` consecutive tokens into one compressed KV entry via
    learned softmax-gated pooling:

        x_blk = reshape(x, [B, S/r, r, hidden])
        kv    = wkv(x_blk)
        score = wgate(x_blk) + ape          # ape: learned [r, kvpe_dim] positional bias
        w     = softmax(score, dim=block)   # over the r tokens, per output channel
        comp  = sum(kv * w, dim=block)      # [B, S/r, kvpe_dim]
    """

    def __init__(self, config: DeepseekV4Config, compress_ratio: int):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.kvpe_dim = config.kv_lora_rank + config.qk_rope_head_dim
        self.wkv = nn.Linear(config.hidden_size, self.kvpe_dim, bias=False)
        self.wgate = nn.Linear(config.hidden_size, self.kvpe_dim, bias=False)
        self.ape = nn.Parameter(torch.empty(compress_ratio, self.kvpe_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, num_blocks, compress_ratio, hidden] -> [B, num_blocks, kvpe_dim]
        kv = self.wkv(x)
        score = self.wgate(x) + self.ape
        weights = F.softmax(score.float(), dim=2).to(kv.dtype)
        return (kv * weights).sum(dim=2)


class HCACache:
    """Minimal decode-time cache for `DeepseekV4HCAAttention`.

    Holds the sliding-window ring buffer of per-token KV latents (`window_kvpe`), the
    compressed-KV stream built so far (`compressed_kv`), and the not-yet-compressed tail
    of raw `hidden_states` (`pending_hidden`) needed to compute the next compressed
    entry once a full `compress_ratio`-token block has been seen.
    """

    def __init__(
        self,
        batch_size: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        hidden_size: int,
        sliding_window: int,
        compress_ratio: int,
        device=None,
        dtype=torch.bfloat16,
    ):
        self.kvpe_dim = kv_lora_rank + qk_rope_head_dim
        self.hidden_size = hidden_size
        self.sliding_window = sliding_window
        self.compress_ratio = compress_ratio
        self.window_kvpe = torch.empty(batch_size, 0, self.kvpe_dim, device=device, dtype=dtype)
        self.compressed_kv = torch.empty(batch_size, 0, self.kvpe_dim, device=device, dtype=dtype)
        self.pending_hidden = torch.empty(batch_size, 0, hidden_size, device=device, dtype=dtype)
        self.seen_tokens = 0

    def get_seq_length(self) -> int:
        return self.seen_tokens

    def update_window(self, kvpe: torch.Tensor) -> None:
        self.window_kvpe = torch.cat([self.window_kvpe, kvpe], dim=1)
        if self.window_kvpe.shape[1] > self.sliding_window:
            self.window_kvpe = self.window_kvpe[:, -self.sliding_window :]

    def append_pending_hidden(self, hidden_states: torch.Tensor) -> None:
        self.pending_hidden = torch.cat([self.pending_hidden, hidden_states], dim=1)

    def take_pending_blocks(self) -> Optional[torch.Tensor]:
        """Pop complete `compress_ratio`-token blocks off `pending_hidden`.

        Returns `[B, num_new_blocks, compress_ratio, hidden]`, or `None` if no full
        block is available yet.
        """
        r = self.compress_ratio
        num_blocks = self.pending_hidden.shape[1] // r
        if num_blocks == 0:
            return None
        take = num_blocks * r
        bsz = self.pending_hidden.shape[0]
        blocks = self.pending_hidden[:, :take].reshape(bsz, num_blocks, r, self.hidden_size)
        self.pending_hidden = self.pending_hidden[:, take:]
        return blocks

    def append_compressed(self, comp: torch.Tensor) -> None:
        self.compressed_kv = torch.cat([self.compressed_kv, comp], dim=1)


class DeepseekV4HCAAttention(nn.Module):
    """Heavily Compressed Attention (HCA) for DeepSeek-V4-Pro."""

    def __init__(self, config: DeepseekV4Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx if layer_idx is not None else 0

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.q_head_dim = config.q_head_dim
        self.o_lora_rank = config.o_lora_rank
        self.o_groups = config.o_groups
        self.sliding_window = config.sliding_window
        self.attention_dropout = config.attention_dropout
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.compress_rope_theta = config.compress_rope_theta

        if self.layer_idx < len(config.compress_ratios):
            self.compress_ratio = config.compress_ratios[self.layer_idx]
        else:
            self.compress_ratio = 0
        if self.compress_ratio <= 0:
            raise ValueError(
                f"DeepseekV4HCAAttention requires compress_ratios[{self.layer_idx}] > 0 "
                f"(got {self.compress_ratio}); 128 selects HCA, 4 selects CSA (not "
                "implemented here), 0 is a sliding-window-only MTP layer."
            )

        kvpe_dim = self.kv_lora_rank + self.qk_rope_head_dim  # 576 for V4-Pro

        # --- Q: low-rank, MLA-style ---
        self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=config.attention_bias)
        self.q_a_layernorm = DeepseekV3RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)

        # --- KV latent (per-token) ---
        self.kv_a_proj_with_mqa = nn.Linear(self.hidden_size, kvpe_dim, bias=config.attention_bias)
        self.kv_a_layernorm = DeepseekV3RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False
        )

        # --- Compressed KV stream ---
        self.compressor = Compressor(config, self.compress_ratio)
        self.compress_kv_norm = DeepseekV3RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)

        # --- Attention sink: one learned scalar logit per head ---
        self.attn_sink = nn.Parameter(torch.zeros(self.num_heads))

        # --- Output: o_groups-way grouped low-rank projection ---
        if (self.num_heads * self.v_head_dim) % self.o_groups != 0 or self.o_lora_rank % self.o_groups != 0:
            raise ValueError("num_heads * v_head_dim and o_lora_rank must both be divisible by o_groups")
        group_in = (self.num_heads * self.v_head_dim) // self.o_groups
        group_out = self.o_lora_rank // self.o_groups
        self.o_down_proj = nn.Parameter(torch.empty(self.o_groups, group_in, group_out))
        self.o_norm = DeepseekV3RMSNorm(self.o_lora_rank, eps=config.rms_norm_eps)
        self.wo = nn.Linear(self.o_lora_rank, self.hidden_size, bias=config.attention_bias)

        self.is_causal = True
        self._init_rope()

        self.softmax_scale = self.q_head_dim ** (-0.5)
        if config.rope_scaling is not None:
            mscale_all_dim = config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = DeepseekV3RotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type != "yarn":
                raise ValueError(f"Unsupported rope_scaling type for HCA: {scaling_type}")
            kwargs = {
                key: self.config.rope_scaling[key]
                for key in ["original_max_position_embeddings", "beta_fast", "beta_slow", "mscale", "mscale_all_dim"]
                if key in self.config.rope_scaling
            }
            self.rotary_emb = DeepseekV3YarnRotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=scaling_factor,
                base=self.rope_theta,
                **kwargs,
            )

        # Separate, non-YaRN RoPE for the compressed stream: positions are compressed-
        # block indices (0, 1, 2, ...), so no long-context extrapolation is needed.
        max_blocks = max(self.max_position_embeddings // self.compress_ratio, 1) + 1
        self.compress_rotary_emb = DeepseekV3RotaryEmbedding(
            self.qk_rope_head_dim,
            max_position_embeddings=max_blocks,
            base=self.compress_rope_theta,
        )

    def _apply_compress_rope(self, comp: torch.Tensor, start_block: int) -> torch.Tensor:
        # comp: [B, num_new_blocks, kvpe_dim] -> RMSNorm(nope) || RoPE(rope)
        bsz, num_new, _ = comp.shape
        nope, rope = torch.split(comp, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        nope = self.compress_kv_norm(nope)

        rope4 = rope.view(bsz, 1, num_new, self.qk_rope_head_dim)
        block_pos = torch.arange(start_block, start_block + num_new, device=comp.device).unsqueeze(0)
        cos, sin = self.compress_rotary_emb(rope4, seq_len=start_block + num_new, meta_style=True)
        rope4, _ = apply_rotary_pos_emb(rope4, rope4, cos, sin, block_pos, meta_style=True)
        rope = rope4.view(bsz, num_new, self.qk_rope_head_dim)
        return torch.cat([nope, rope], dim=-1)

    def _attention(
        self,
        query_states: torch.Tensor,
        window_kvpe: torch.Tensor,
        compressed_kv: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, num_heads, q_len, _ = query_states.shape
        device = query_states.device
        t_w = window_kvpe.shape[2]
        n_b = compressed_kv.shape[2]
        r = self.compress_ratio

        q_pos = position_ids[0]  # [S], shared across batch
        cur_pos = int(q_pos[-1].item())
        window_key_pos = (cur_pos + 1 - t_w) + torch.arange(t_w, device=device)  # [Tw]
        block_idx = torch.arange(n_b, device=device)  # [Nb]

        attn_w = torch.matmul(query_states, window_kvpe.transpose(2, 3)) * self.softmax_scale  # [B,H,S,Tw]
        attn_c = torch.matmul(query_states, compressed_kv.transpose(2, 3)) * self.softmax_scale  # [B,H,S,Nb]

        causal = window_key_pos[None, :] <= q_pos[:, None]
        windowed = (q_pos[:, None] - window_key_pos[None, :]) < self.sliding_window
        mask_w = (causal & windowed)[None, None]  # [1,1,S,Tw]
        attn_w = attn_w.masked_fill(~mask_w, float("-inf"))

        q_block = (q_pos + 1) // r  # [S]
        mask_c = (block_idx[None, :] < q_block[:, None])[None, None]  # [1,1,S,Nb]
        attn_c = attn_c.masked_fill(~mask_c, float("-inf"))

        sink = self.attn_sink.view(1, num_heads, 1, 1).expand(bsz, num_heads, q_len, 1).to(attn_w.dtype)

        all_scores = torch.cat([attn_w, attn_c, sink], dim=-1)
        probs = F.softmax(all_scores.float(), dim=-1).to(query_states.dtype)
        probs = F.dropout(probs, p=self.attention_dropout, training=self.training)
        probs_w, probs_c, _ = torch.split(probs, [t_w, n_b, 1], dim=-1)

        value_w = window_kvpe[..., : self.kv_lora_rank]
        value_c = compressed_kv[..., : self.kv_lora_rank]
        attn_output = torch.matmul(probs_w, value_w) + torch.matmul(probs_c, value_c)
        return attn_output, probs

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[HCACache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[HCACache]]:
        bsz, q_len, _ = hidden_states.shape
        device = hidden_states.device
        r = self.compress_ratio

        past_len = past_key_value.get_seq_length() if past_key_value is not None else 0
        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + q_len, device=device).unsqueeze(0)

        # ---- Q: low-rank projection, split nope/rope, absorb nope into KV-latent space ----
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)  # [B,H,S,head_dim]
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        kv_b1_proj = self.kv_b_proj.weight.view(self.num_heads, -1, self.kv_lora_rank)[:, : self.qk_nope_head_dim]
        q_nope = torch.matmul(q_nope, kv_b1_proj)  # [B,H,S,kv_lora_rank]

        # ---- KV latent (per-token) + RoPE ----
        kv_latent = self.kv_a_proj_with_mqa(hidden_states)
        k_nope_raw, k_pe = torch.split(kv_latent, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(bsz, 1, q_len, self.qk_rope_head_dim)
        k_nope = self.kv_a_layernorm(k_nope_raw).view(bsz, 1, q_len, self.kv_lora_rank)

        kv_seq_len = past_len + q_len
        cos, sin = self.rotary_emb(k_nope, seq_len=kv_seq_len, meta_style=True)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids, meta_style=True)

        query_states = q_pe.new_empty(bsz, self.num_heads, q_len, self.kv_lora_rank + self.qk_rope_head_dim)
        query_states[..., : self.kv_lora_rank] = q_nope
        query_states[..., self.kv_lora_rank :] = q_pe

        new_window_kvpe = k_pe.new_empty(bsz, 1, q_len, self.kv_lora_rank + self.qk_rope_head_dim)
        new_window_kvpe[..., : self.kv_lora_rank] = k_nope
        new_window_kvpe[..., self.kv_lora_rank :] = k_pe

        # ---- Compressed KV stream + sliding-window cache ----
        if past_key_value is not None:
            past_key_value.update_window(new_window_kvpe[:, 0])
            past_key_value.append_pending_hidden(hidden_states)

            blocks = past_key_value.take_pending_blocks()
            if blocks is not None:
                start_block = past_key_value.compressed_kv.shape[1]
                comp = self.compressor(blocks)  # [B, num_new_blocks, kvpe_dim]
                comp = self._apply_compress_rope(comp, start_block)
                past_key_value.append_compressed(comp)

            window_kvpe = past_key_value.window_kvpe.to(new_window_kvpe.dtype).unsqueeze(1)
            compressed_kv = past_key_value.compressed_kv.to(new_window_kvpe.dtype).unsqueeze(1)
            past_key_value.seen_tokens += q_len
        else:
            if q_len % r != 0:
                raise ValueError(f"HCA prefill requires seq_len % compress_ratio == 0 (seq_len={q_len}, r={r})")
            window_kvpe = new_window_kvpe
            num_blocks = q_len // r
            blocks = hidden_states.view(bsz, num_blocks, r, self.hidden_size)
            comp = self.compressor(blocks)
            comp = self._apply_compress_rope(comp, 0)
            compressed_kv = comp.unsqueeze(1)

        attn_output, probs = self._attention(query_states, window_kvpe, compressed_kv, position_ids)

        # ---- KV b2 projection: expand latent V back to per-head v_head_dim ----
        kv_b2_proj = self.kv_b_proj.weight.view(self.num_heads, -1, self.kv_lora_rank)[:, -self.v_head_dim :].transpose(
            1, 2
        )
        attn_output = torch.matmul(attn_output, kv_b2_proj)  # [B,H,S,v_head_dim]

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.num_heads * self.v_head_dim)

        # ---- Output: o_groups-way grouped low-rank projection ----
        group_in = (self.num_heads * self.v_head_dim) // self.o_groups
        grouped = attn_output.view(bsz, q_len, self.o_groups, group_in)
        grouped = torch.einsum("bsgi,gio->bsgo", grouped, self.o_down_proj.to(grouped.dtype))
        grouped = grouped.reshape(bsz, q_len, self.o_lora_rank)
        grouped = self.o_norm(grouped)
        attn_output = self.wo(grouped)

        if not output_attentions:
            probs = None
        return attn_output, probs, past_key_value
