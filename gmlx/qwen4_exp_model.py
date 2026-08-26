# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
# Portions copyright (c) 2024 Apple Inc. (mlx-lm qwen3_next/qwen3_5 skeleton, MIT)
"""Vendored mlx-lm-style model for Qwen3.8-Flash-Next (GGUF arch ``qwen4exp``).

Neither pinned mlx-lm nor pinned mlx-vlm ships a qwen4_exp class; this module
is the runtime for llama.cpp's ``LLM_ARCH_QWEN4EXP`` conversions. It is the
qwen3.5 hybrid (gated DeltaNet on three of every four layers, gated full
attention on the fourth, a 512-expert top-10 MoE with a sigmoid-gated shared
expert on every layer) plus three mechanisms of its own:

  1. Hyper-connections. The residual is four parallel streams ``[T, 4, D]``
     initialised from the token embedding. Each sub-layer reads one mixed
     ``[T, D]`` row (grouped RMSNorm per stream, low-rank silu down / sigmoid
     up gate, mean over the streams) and writes back ``out * inject``, where
     ``inject = 2 * sigmoid(W xn / 4)`` is one scalar per stream. A final
     mixer without inject replaces output_norm.
  2. QSA sparse attention. A 4-head indexer scores mean-pooled blocks of
     ``compress_ratio`` keys (k_norm + rope at the block start position,
     ``sum_h relu(q_h . k_b)``), keeps the ``budget / ratio`` best complete
     blocks plus the incomplete tail, and attention is masked to that set.
     Dense while at most ``budget / ratio`` complete blocks exist.
  3. PLE n-gram hash embeddings on one layer: 2-gram and 3-gram hashes of the
     token history (EOS resets the context) index a huge quantized table; the
     rows gate into all four streams plus a dilated depthwise conv branch.

Wire facts the forward relies on: every norm weight is already ``(1 + w)``
baked (plain RMSNorm); the GDN output gate is sigmoid; GDN V heads are tiled
(V head ``hv`` reads K head ``hv % Hk``), so q/k are tiled explicitly before
the scan; attention ``q_proj`` carries ``[q | gate]`` per head; rotary is 64
of 256 dims with interleaved mrope sections, which for text-only positions is
plain NEOX rope.
"""

import importlib
import math
import sys
from dataclasses import dataclass, field
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.gated_delta import gated_delta_update
from mlx_lm.models.switch_layers import SwitchGLU


def ensure_registered() -> None:
    """Make ``import mlx_lm.models.qwen4_exp`` resolve, preferring upstream,
    and expose ``QSAKVCache`` on the cache modules (prompt-cache save/load
    resolves cache classes by name there; mlx-vlm >= 0.6.4 vendors its own
    models/cache.py, so register on both when it is loaded)."""
    import mlx_lm.models.cache as _mlx_cache

    vlm_cache = sys.modules.get("mlx_vlm.models.cache")
    for mod in (_mlx_cache, vlm_cache):
        if mod is not None and not hasattr(mod, "QSAKVCache"):
            mod.QSAKVCache = QSAKVCache
    if "mlx_lm.models.qwen4_exp" in sys.modules:
        return
    try:
        importlib.import_module("mlx_lm.models.qwen4_exp")  # upstream wins
    except ImportError:
        sys.modules["mlx_lm.models.qwen4_exp"] = sys.modules[__name__]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "qwen4_exp"
    hidden_size: int = 2560
    num_hidden_layers: int = 48
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    full_attention_interval: int = 4
    layer_types: Optional[List[str]] = None
    linear_num_value_heads: int = 48
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    rope_theta: float = 10000000.0
    partial_rotary_factor: float = 0.25
    mrope_section: List[int] = field(default_factory=lambda: [11, 11, 10])
    max_position_embeddings: int = 262144
    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    norm_topk_prob: bool = True
    hc_count: int = 4
    hc_lowrank: int = 320
    indexer_n_heads: int = 4
    indexer_head_dim: int = 128
    indexer_budget: int = 2048
    compress_ratios: Optional[List[int]] = None
    ple_layer_ids: List[int] = field(default_factory=list)
    ple_ngram_size: int = 3
    ple_heads_per_ngram: int = 8
    ple_conv_kernel: int = 4
    ple_eos_token_id: int = 0
    ple_image_token_id: Optional[int] = None
    ple_embed_dim: int = 160
    ple_table_rows: int = 0
    ple_layer_multipliers: List[int] = field(default_factory=list)
    ple_head_offsets: List[int] = field(default_factory=list)
    ple_head_vocab_sizes: List[int] = field(default_factory=list)
    tie_word_embeddings: bool = False
    kv_head_layout: str = "tiled"

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = [
                "full_attention"
                if (i + 1) % self.full_attention_interval == 0
                else "linear_attention"
                for i in range(self.num_hidden_layers)
            ]
        if self.compress_ratios is None:
            self.compress_ratios = [0] * self.num_hidden_layers


# Norms


def _grouped_rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """RMSNorm over the last axis of ``[..., hc, D]`` with a ``[hc * D]``
    gamma: one statistic per stream, one gain per element."""
    hc, d = x.shape[-2], x.shape[-1]
    return mx.fast.rms_norm(x, None, eps) * weight.reshape(hc, d)


class GroupedRMSNorm(nn.Module):
    """RMSNorm applied per residual stream, gamma over all streams."""

    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return _grouped_rms_norm(x, self.weight, self.eps)


class RMSNormGatedSigmoid(nn.Module):
    """``rms_norm(x) * sigmoid(gate)``: the GDN output norm (qwen3.5 uses
    silu here; this is the one numerical difference in the GDN)."""

    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps) * mx.sigmoid(gate)


# Hyper-connections


class HyperConnection(nn.Module):
    """Low-rank sigmoid-gated stream mixer (+ optional per-stream inject)."""

    def __init__(self, hidden: int, hc: int, lowrank: int, eps: float,
                 inject: bool = True):
        super().__init__()
        self.hc = hc
        self.hidden = hidden
        self.norm = GroupedRMSNorm(hc * hidden, eps)
        self.down = nn.Linear(hc * hidden, lowrank, bias=False)
        self.up = nn.Linear(lowrank, hc * hidden, bias=False)
        if inject:
            self.inject = nn.Linear(hc * hidden, hc, bias=False)

    def __call__(self, h: mx.array):
        B, T, hc, D = h.shape
        xn = self.norm(h)
        xf = xn.reshape(B, T, hc * D)
        lo = nn.silu(self.down(xf) * (1.0 / hc))
        gate = mx.sigmoid(self.up(lo)).reshape(B, T, hc, D)
        mixed = (gate * xn).mean(axis=2)
        if "inject" not in self:
            return mixed
        inj = 2.0 * mx.sigmoid(self.inject(xf) * (1.0 / hc))
        return mixed, inj


def _hc_combine(h: mx.array, out: mx.array, inject: mx.array) -> mx.array:
    return h + out[:, :, None, :] * inject[..., None]


# Gated DeltaNet


class GatedDeltaNet(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_v_heads = args.linear_num_value_heads
        self.num_k_heads = args.linear_num_key_heads
        self.head_k_dim = args.linear_key_head_dim
        self.head_v_dim = args.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = args.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.eps = args.rms_norm_eps

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
            bias=False,
        )
        self.dt_bias = mx.ones((self.num_v_heads,))
        self.A_log = mx.zeros((self.num_v_heads,))
        self.norm = RMSNormGatedSigmoid(self.head_v_dim, eps=self.eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def __call__(self, inputs: mx.array, mask: Optional[mx.array] = None,
                 cache=None) -> mx.array:
        B, S, _ = inputs.shape
        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        n_keep = self.conv_kernel_size - 1
        conv_state = None
        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
            if conv_state.shape[0] != B:
                conv_state = None
        if conv_state is None:
            conv_state = mx.zeros((B, n_keep, self.conv_dim), dtype=inputs.dtype)

        if mask is not None:
            if mask.shape[0] != B:
                mask = None
            else:
                qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            lengths = getattr(cache, "lengths", None)
            if lengths is not None:
                ends = mx.clip(lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], axis=-1),
                (self.num_k_heads, self.num_k_heads, self.num_v_heads),
                (self.head_k_dim, self.head_k_dim, self.head_v_dim),
            )
        ]
        inv_scale = self.head_k_dim ** -0.5
        q = (inv_scale ** 2) * mx.fast.rms_norm(q, None, self.eps)
        k = inv_scale * mx.fast.rms_norm(k, None, self.eps)
        if self.num_v_heads != self.num_k_heads:
            # GGUF tiled V layout: V head hv pairs with K head hv % Hk.
            r = self.num_v_heads // self.num_k_heads
            q = mx.tile(q, [1, 1, r, 1])
            k = mx.tile(k, [1, 1, r, 1])

        state = cache[1] if cache is not None else None
        if state is not None and state.shape[0] != B:
            state = None
        out, state = gated_delta_update(
            q, k, v, a, b, self.A_log, self.dt_bias, state, mask,
            use_kernel=not self.training,
        )
        if cache is not None:
            cache[1] = state
            if hasattr(cache, "advance"):
                cache.advance(S)
        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1))


# QSA: cache, indexer, attention


class QSAKVCache(KVCache):
    """KVCache extended with the QSA indexer key stream.

    ``ik`` holds one raw (pre-norm, pre-rope) indexer key per cached token,
    appended in lockstep with K/V so ``offset`` covers all three streams.
    ``blocks`` caches the finished (mean-pooled, normed, roped) block keys
    derived from ``ik``; it is a pure function of the raw stream, so trim and
    state restore just invalidate it. Quantizing would drop the stream, so
    ``to_quantized`` is refused.
    """

    kv_quant_unsupported = True

    def __init__(self, ratio: int = 4):
        super().__init__()
        self.ratio = ratio
        self.ik = None       # [B, cap, index_dim]
        self.blocks = None   # [B, n_blocks, index_dim]
        self.n_blocks = 0

    def update_and_fetch_qsa(self, keys, values, ik):
        prev = self.offset
        k, v = super().update_and_fetch(keys, values)
        n = ik.shape[1]
        if self.ik is None or (prev + n) > self.ik.shape[1]:
            B, _, idx_dim = ik.shape
            n_steps = (self.step + n - 1) // self.step
            new = mx.zeros((B, n_steps * self.step, idx_dim), ik.dtype)
            if self.ik is not None:
                if prev % self.step != 0:
                    self.ik = self.ik[:, :prev, :]
                self.ik = mx.concatenate([self.ik, new], axis=1)
            else:
                self.ik = new
        self.ik[:, prev:prev + n, :] = ik
        return k, v, self.ik[:, :self.offset, :]

    def finished_blocks(self, n_blocks: int, finish):
        """Block keys ``[B, n_blocks, D]``; ``finish(raw, start_block)`` turns
        ``[B, m * ratio, D]`` raw keys into ``[B, m, D]`` finished ones."""
        r = self.ratio
        if self.blocks is not None and self.blocks.shape[0] != self.ik.shape[0]:
            self.blocks, self.n_blocks = None, 0
        if n_blocks > self.n_blocks:
            raw = self.ik[:, self.n_blocks * r:n_blocks * r, :]
            new = finish(raw, self.n_blocks)
            self.blocks = new if self.blocks is None else mx.concatenate(
                [self.blocks[:, :self.n_blocks], new], axis=1)
            self.n_blocks = n_blocks
        return self.blocks[:, :n_blocks]

    @property
    def state(self):
        if self.offset == self.keys.shape[2]:
            return self.keys, self.values, self.ik[:, :self.offset]
        return (
            self.keys[..., :self.offset, :],
            self.values[..., :self.offset, :],
            self.ik[:, :self.offset],
        )

    @state.setter
    def state(self, v):
        self.keys, self.values, self.ik = v
        self.offset = self.keys.shape[2]
        self.blocks, self.n_blocks = None, 0

    @property
    def meta_state(self):
        return str(self.ratio)

    @meta_state.setter
    def meta_state(self, v):
        if v:
            self.ratio = int(v)

    def trim(self, n):
        n = super().trim(n)
        self.n_blocks = min(self.n_blocks, self.offset // self.ratio)
        return n

    def to_quantized(self, group_size: int = 64, bits: int = 4):
        raise NotImplementedError(
            "QSAKVCache cannot quantize: the QSA indexer key stream has no "
            "quantized form (drop --kv-bits for qwen4exp)")

    @property
    def nbytes(self):
        n = super().nbytes
        if self.ik is not None:
            n += self.ik.nbytes
        if self.blocks is not None:
            n += self.blocks.nbytes
        return n


class QSAIndexer(nn.Module):
    """Block selector for one attention layer."""

    def __init__(self, args: ModelArgs, ratio: int, rotary_dims: int):
        super().__init__()
        self.n_heads = args.indexer_n_heads
        self.head_dim = args.indexer_head_dim
        self.ratio = ratio
        self.block_topk = args.indexer_budget // ratio
        self.rotary_dims = rotary_dims
        self.rope_theta = args.rope_theta
        self.q_proj = nn.Linear(args.hidden_size, self.n_heads * self.head_dim,
                                bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.head_dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

    def _rope(self, x, offset, scale=1.0):
        return mx.fast.rope(x, self.rotary_dims, traditional=False,
                            base=self.rope_theta, scale=scale, offset=offset)

    def finish_blocks(self, raw: mx.array, start_block: int) -> mx.array:
        """Mean-pool ``[B, m * r, D]`` raw keys into ``[B, m, D]`` block keys,
        k_norm, then rope at the block start positions (``scale=r`` makes
        position ``i`` of the call land on token ``(start_block + i) * r``)."""
        B, n, D = raw.shape
        m = n // self.ratio
        pooled = raw[:, :m * self.ratio].reshape(B, m, self.ratio, D)
        pooled = pooled.astype(mx.float32).mean(axis=2).astype(raw.dtype)
        pooled = self.k_norm(pooled)
        return self._rope(pooled[:, None], start_block,
                          scale=float(self.ratio))[:, 0]

    def scores(self, x: mx.array, blocks: mx.array, offset: int) -> mx.array:
        """``[B, L, n_blocks]`` f32 block scores for the queries in ``x``."""
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        q = self._rope(q, offset)
        s = q.astype(mx.float32) @ blocks.astype(mx.float32)[:, None].transpose(0, 1, 3, 2)
        s = mx.maximum(s, 0).sum(axis=1)
        return s * (1.0 / math.sqrt(self.head_dim))

    def select(self, x: mx.array, ik_all: mx.array, cache, offset: int):
        """Top-k complete blocks per query, or None when every query is
        still dense. Returns ``(blocks [B, L, topk], complete_counts [L])``."""
        B, L, _ = x.shape
        key_len = ik_all.shape[1]
        r = self.ratio
        n_blocks = key_len // r
        if n_blocks <= self.block_topk:
            return None
        if cache is not None:
            blocks = cache.finished_blocks(n_blocks, self.finish_blocks)
        else:
            blocks = self.finish_blocks(ik_all[:, :n_blocks * r], 0)
        s = self.scores(x, blocks, offset)
        query_ends = offset + mx.arange(L) + 1
        complete = query_ends // r
        valid = mx.arange(n_blocks)[None, None, :] < complete[None, :, None]
        s = mx.where(valid, s, -mx.inf)
        k = self.block_topk
        sel = mx.argpartition(s, kth=-k, axis=-1)[..., -k:]
        return sel, complete


def _qsa_token_mask(sel, complete, offset, L, key_len, ratio, topk):
    """Boolean ``[B, 1, L, key_len]`` attention mask from the block selection:
    selected block members plus the incomplete tail, or plain causal for
    queries with at most ``topk`` complete blocks."""
    B = sel.shape[0]
    n_blocks = key_len // ratio
    blk = mx.zeros((B, L, n_blocks + 1), dtype=mx.bool_)
    blk = mx.put_along_axis(blk, sel, mx.array(True), axis=-1)[..., :n_blocks]
    tok_sel = mx.repeat(blk, ratio, axis=-1)
    pad = key_len - n_blocks * ratio
    if pad:
        tok_sel = mx.concatenate(
            [tok_sel, mx.zeros((B, L, pad), dtype=mx.bool_)], axis=-1)
    tok = mx.arange(key_len)[None, :]
    query_ends = (offset + mx.arange(L) + 1)[:, None]
    tail_start = (complete * ratio)[:, None]
    causal = tok < query_ends
    tail = (tok >= tail_start) & causal
    use_sparse = (complete > topk)[:, None]
    m = mx.where(use_sparse[None], tok_sel | tail[None], causal[None])
    return m[:, None]


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim ** -0.5
        self.rotary_dims = int(self.head_dim * args.partial_rotary_factor)
        self.rope_theta = args.rope_theta
        H, Hkv, D = self.num_attention_heads, self.num_key_value_heads, self.head_dim
        self.q_proj = nn.Linear(args.hidden_size, H * D * 2, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, Hkv * D, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, Hkv * D, bias=False)
        self.o_proj = nn.Linear(H * D, args.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(D, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(D, eps=args.rms_norm_eps)
        self.ratio = int(args.compress_ratios[layer_idx])
        if self.ratio > 0:
            self.indexer = QSAIndexer(args, self.ratio, self.rotary_dims)

    def _rope(self, x, offset):
        return mx.fast.rope(x, self.rotary_dims, traditional=False,
                            base=self.rope_theta, scale=1.0, offset=offset)

    def _gathered_attention(self, q, k, v, sel, complete, offset, L):
        """Decode / verify: gather the selected keys per query and run one
        ``B * L``-batched qL=1 sdpa over ``topk * r + r`` keys."""
        B, H, _, D = q.shape
        Hkv = k.shape[1]
        r, topk = self.ratio, self.indexer.block_topk
        members = (sel[..., None] * r + mx.arange(r)).reshape(B, L, topk * r)
        tail_start = (complete * r)[None, :, None]
        tail = tail_start + mx.arange(r)[None, None, :]
        query_ends = (offset + mx.arange(L) + 1)[None, :, None]
        tail_ok = tail < query_ends
        idx = mx.concatenate([members, mx.minimum(tail, query_ends - 1)], axis=-1)
        W = idx.shape[-1]
        bias = mx.concatenate(
            [mx.ones((B, L, topk * r), dtype=mx.bool_), tail_ok], axis=-1)
        flat = mx.broadcast_to(idx.reshape(B, 1, L * W, 1), (B, Hkv, L * W, 1))
        k_sel = mx.take_along_axis(k, flat, axis=2).reshape(B, Hkv, L, W, D)
        v_sel = mx.take_along_axis(v, flat, axis=2).reshape(B, Hkv, L, W, D)
        k_sel = k_sel.transpose(0, 2, 1, 3, 4).reshape(B * L, Hkv, W, D)
        v_sel = v_sel.transpose(0, 2, 1, 3, 4).reshape(B * L, Hkv, W, D)
        q_l = q.transpose(0, 2, 1, 3).reshape(B * L, H, 1, D)
        out = mx.fast.scaled_dot_product_attention(
            q_l, k_sel, v_sel, scale=self.scale,
            mask=bias.reshape(B * L, 1, 1, W))
        return out.reshape(B, L, H, D).transpose(0, 2, 1, 3)

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        B, L, _ = x.shape
        H, Hkv, D = self.num_attention_heads, self.num_key_value_heads, self.head_dim
        qg = self.q_proj(x).reshape(B, L, H, 2 * D)
        q, gate = mx.split(qg, 2, axis=-1)
        gate = gate.reshape(B, L, -1)
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(B, L, Hkv, D)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, Hkv, D).transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        q = self._rope(q, offset)
        k = self._rope(k, offset)

        selection = None
        if "indexer" in self:
            ik = self.indexer.k_proj(x)
            if cache is not None and hasattr(cache, "update_and_fetch_qsa"):
                k, v, ik_all = cache.update_and_fetch_qsa(k, v, ik)
                selection = self.indexer.select(x, ik_all, cache, offset)
            elif cache is not None:
                k, v = cache.update_and_fetch(k, v)
            else:
                selection = self.indexer.select(x, ik, None, 0)
        elif cache is not None:
            k, v = cache.update_and_fetch(k, v)

        if selection is None:
            out = scaled_dot_product_attention(
                q, k, v, cache=cache, scale=self.scale, mask=mask)
        else:
            sel, complete = selection
            key_len = k.shape[2]
            all_sparse = (offset + 1) // self.ratio > self.indexer.block_topk
            if L <= 8 and all_sparse and not isinstance(mask, mx.array):
                out = self._gathered_attention(q, k, v, sel, complete, offset, L)
            else:
                qsa = _qsa_token_mask(sel, complete, offset, L, key_len,
                                      self.ratio, self.indexer.block_topk)
                if isinstance(mask, mx.array):
                    if mask.dtype == mx.bool_:
                        qsa = qsa & mask
                    else:
                        qsa = mask + mx.where(qsa, 0.0, -mx.inf).astype(mask.dtype)
                out = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=self.scale, mask=qsa)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out * mx.sigmoid(gate))


# MoE (qwen3-next shape: router + SwitchGLU + sigmoid-gated shared expert)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.norm_topk_prob = args.norm_topk_prob
        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.gate = nn.Linear(dim, self.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(dim, args.moe_intermediate_size, self.num_experts)
        self.shared_expert = MLP(dim, args.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(dim, 1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)
        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        shared_y = self.shared_expert(x)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
        return y + shared_y


# PLE n-gram hash embeddings


class PLEEmbedding(nn.Module):
    """Hash the token history into per-head row ids and gather the rows.

    Head ``h`` of n-gram order ``n`` (2 <= n <= ngram_size) hashes
    ``ctx[0] * m[0] ^ ... ^ ctx[n-1] * m[n-1]`` (uint64 wraparound) into
    ``mixed % vocab[h] + offset[h]``. ``ctx[s]`` is the token ``s`` positions
    back, or EOS when an EOS sits anywhere in between (the token's own EOS
    does not cut its context). The multipliers are 45-bit and token ids
    18-bit, so the products never reach the sign bit and signed (HF) and
    unsigned (llama.cpp) arithmetic agree.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.ngram_size = args.ple_ngram_size
        self.heads_per_ngram = args.ple_heads_per_ngram
        self.context_len = self.ngram_size - 1
        self.n_heads = self.context_len * self.heads_per_ngram
        self.eos_token_id = args.ple_eos_token_id
        self.embed_dim = args.ple_embed_dim
        self._mults = [int(m) for m in args.ple_layer_multipliers]
        self._sizes = mx.array([int(s) for s in args.ple_head_vocab_sizes],
                               dtype=mx.uint64)
        self._offsets = mx.array([int(o) for o in args.ple_head_offsets],
                                 dtype=mx.uint64)

    def _shift_right_ignore_eos(self, tokens: mx.array, shift: int) -> mx.array:
        if shift == 0:
            return tokens
        B, T = tokens.shape
        positions = mx.arange(T, dtype=mx.int64)
        eos_pos = mx.where(tokens == self.eos_token_id, positions[None], -1)
        prev_eos_incl = mx.cummax(eos_pos, axis=1)
        prev_eos = mx.concatenate(
            [mx.full((B, 1), -1, dtype=mx.int64), prev_eos_incl[:, :-1]], axis=1)
        src = positions - shift
        gathered = mx.take_along_axis(
            tokens, mx.broadcast_to(mx.maximum(src, 0)[None], (B, T)), axis=1)
        valid = (src[None] > prev_eos) & (src[None] >= 0)
        return mx.where(valid, gathered, self.eos_token_id)

    def row_ids(self, input_ids: mx.array, cache) -> mx.array:
        """``[B, T, n_heads]`` table rows for the tokens in ``input_ids``."""
        ids = input_ids.astype(mx.int64)
        B, T = ids.shape
        prev = None
        if cache is not None and cache[3] is not None:
            prev = cache[3]
            if prev.shape[0] != B:
                prev = None
        if prev is None:
            prev = mx.full((B, self.context_len), self.eos_token_id, dtype=mx.int64)
        hist = mx.concatenate([prev, ids], axis=1)
        if cache is not None:
            cache[3] = mx.contiguous(hist[:, -self.context_len:])
        shifted = [
            self._shift_right_ignore_eos(hist, s).astype(mx.uint64)
            for s in range(self.ngram_size)
        ]
        mults = [mx.array(m, dtype=mx.uint64) for m in self._mults]
        blocks = []
        for n in range(2, self.ngram_size + 1):
            start = (n - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * mults[0]
            for p in range(1, n):
                mixed = mx.bitwise_xor(mixed, shifted[p] * mults[p])
            rows = mixed[..., None] % self._sizes[start:end] + self._offsets[start:end]
            blocks.append(rows)
        rows = mx.concatenate(blocks, axis=-1)[:, -T:]
        return rows.astype(mx.int32)

    def __call__(self, input_ids: mx.array, cache, table) -> mx.array:
        """``table`` is the model-level row table (``model.ple_embed``); it is
        passed in rather than owned so the 320M-row weight has one parameter
        path independent of which layer hosts the PLE block."""
        rows = self.row_ids(input_ids, cache)
        emb = table(rows)
        return emb.reshape(*emb.shape[:-2], -1)


class PLELayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.hc = args.hc_count
        hc_dim = self.hidden_size * self.hc
        eps = args.rms_norm_eps
        self.embedding = PLEEmbedding(args)
        emb_dim = self.embedding.n_heads * args.ple_embed_dim
        self.key_proj = nn.Linear(emb_dim, hc_dim, bias=False)
        self.value_proj = nn.Linear(emb_dim, self.hidden_size, bias=False)
        self.norm_key = GroupedRMSNorm(hc_dim, eps)
        self.norm_query = GroupedRMSNorm(hc_dim, eps)
        self.norm_conv = GroupedRMSNorm(hc_dim, eps)
        self.conv_dilation = args.ple_ngram_size
        self.conv_kernel = args.ple_conv_kernel
        self.conv_state_len = (self.conv_kernel - 1) * self.conv_dilation
        self.conv1d = nn.Conv1d(
            hc_dim, hc_dim, kernel_size=self.conv_kernel,
            dilation=self.conv_dilation, groups=hc_dim, bias=False)

    def _conv(self, x: mx.array, cache) -> mx.array:
        B = x.shape[0]
        state = None
        if cache is not None and cache[2] is not None:
            state = cache[2]
            if state.shape[0] != B:
                state = None
        if state is None:
            state = mx.zeros((B, self.conv_state_len, x.shape[-1]), dtype=x.dtype)
        conv_input = mx.concatenate([state, x], axis=1)
        if cache is not None:
            cache[2] = mx.contiguous(conv_input[:, -self.conv_state_len:])
        return nn.silu(self.conv1d(conv_input))

    def __call__(self, h: mx.array, input_ids: mx.array, cache, table,
                 mask=None) -> mx.array:
        B, T, hc, D = h.shape
        emb = self.embedding(input_ids, cache, table)
        keys = self.norm_key(self.key_proj(emb).reshape(B, T, hc, D))
        values = self.value_proj(emb)
        queries = self.norm_query(h)
        gate = (keys.astype(mx.float32) * queries.astype(mx.float32)).sum(
            axis=-1, keepdims=True) * (1.0 / math.sqrt(D))
        gate = mx.sign(gate) * mx.sqrt(mx.maximum(mx.abs(gate), 1e-6))
        gate = mx.sigmoid(gate).astype(h.dtype)
        gated = gate * values[:, :, None, :]
        normed = self.norm_conv(gated).reshape(B, T, hc * D)
        gated = gated.reshape(B, T, hc * D)
        if isinstance(mask, mx.array) and mask.ndim == 2 and mask.shape[0] == B:
            gated = mx.where(mask[..., None], gated, 0)
            normed = mx.where(mask[..., None], normed, 0)
        out = gated + self._conv(normed, cache)
        return h + out.reshape(B, T, hc, D)


# Layers and model


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_linear = args.layer_types[layer_idx] == "linear_attention"
        eps = args.rms_norm_eps
        self.hc_attn = HyperConnection(args.hidden_size, args.hc_count,
                                       args.hc_lowrank, eps)
        self.hc_ffn = HyperConnection(args.hidden_size, args.hc_count,
                                      args.hc_lowrank, eps)
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = Attention(args, layer_idx)
        self.mlp = SparseMoeBlock(args)
        if layer_idx in args.ple_layer_ids:
            self.ple = PLELayer(args)

    def __call__(self, h: mx.array, input_ids: mx.array, mask=None, cache=None,
                 ple_table=None):
        if "ple" in self:
            h = self.ple(h, input_ids, cache, ple_table, mask)
        mixed, inject = self.hc_attn(h)
        if self.is_linear:
            out = self.linear_attn(mixed, mask=mask, cache=cache)
        else:
            out = self.self_attn(mixed, mask=mask, cache=cache)
        h = _hc_combine(h, out, inject)
        mixed, inject = self.hc_ffn(h)
        out = self.mlp(mixed)
        return _hc_combine(h, out, inject)


class Qwen4ExpModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.hc = args.hc_count
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.hc_head = HyperConnection(args.hidden_size, args.hc_count,
                                       args.hc_lowrank, args.rms_norm_eps,
                                       inject=False)
        if args.ple_layer_ids:
            # The PLE row table (wire bytes at load; KQuantEmbedding gathers
            # and dequantizes only the touched rows).
            self.ple_embed = nn.Embedding(max(int(args.ple_table_rows), 1),
                                          args.ple_embed_dim)
        self.ssm_idx = next(
            (i for i, t in enumerate(args.layer_types) if t == "linear_attention"), 0)
        self.fa_idx = next(
            (i for i, t in enumerate(args.layer_types) if t == "full_attention"), 0)

    def __call__(self, inputs: mx.array, cache=None, input_embeddings=None,
                 return_streams: bool = False):
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        B, T, D = h.shape
        h = mx.broadcast_to(h[:, :, None, :], (B, T, self.hc, D))
        if cache is None:
            cache = [None] * len(self.layers)
        probe = h[:, :, 0, :]
        fa_mask = create_attention_mask(probe, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(probe, cache[self.ssm_idx])
        table = self.ple_embed if "ple_embed" in self else None
        for layer, c in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            h = layer(h, inputs, mask=mask, cache=c, ple_table=table)
        out = self.hc_head(h)
        if return_streams:
            return out, h
        return out


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen4ExpModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None, input_embeddings=None):
        out = self.model(inputs, cache, input_embeddings=input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        return weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches: List[Any] = []
        for layer in self.layers:
            if layer.is_linear:
                caches.append(ArraysCache(size=4 if "ple" in layer else 2))
            elif layer.self_attn.ratio > 0:
                caches.append(QSAKVCache(layer.self_attn.ratio))
            else:
                caches.append(KVCache())
        return caches
