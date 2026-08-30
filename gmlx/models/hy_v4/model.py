# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
# Portions copyright (c) 2026 Apple Inc. (mlx-lm deepseek_v32 skeleton, MIT)
"""Vendored mlx-lm-style model for HY4-preview (GGUF arch ``hyv4``).

mlx-lm has no hy_v4 class. The runtime follows the llama.cpp reference
(``src/models/hyv4.cpp``) and recombines mechanisms gmlx already runs:

  1. Absorbed MLA in the DeepSeek-V3.2 shape: q_a/q_b, a 512-wide latent
     plus a 64-wide rope half, and per-head ``embed_q``/``unembed_out``
     up-projections. The score scale is over the MLA head size
     (``256**-0.5``), not the latent width.
  2. Per-head learnable attention sinks. The sink logit joins the softmax
     normalizer, so a token can hold back attention mass from every key.
  3. A sigmoid output gate on the decompressed attention result, applied
     before ``o_proj``.
  4. A DSA lightning indexer on the layers marked "full", selecting the
     top ``index_topk`` keys. Rope covers the LAST ``qk_rope_head_dim``
     dims of the 128-wide indexer head, the reverse of mlx-lm's
     deepseek_v32. Layers marked "shared" reuse the most recent preceding
     full layer's selection and carry no indexer weights. Below
     ``index_topk`` cached keys the selection is the identity and
     attention is dense.
  5. iHC (independent hyper-connections): 4 parallel residual streams. One
     ``hc_fn`` per sublayer produces 4 ``pre`` gates that collapse the
     streams and 4 ``post`` gates that redistribute the sublayer output.
     There is no sinkhorn/comb term, which is the delta from DeepSeek-V4's
     hyper-connections; the final collapse is deepseek_v4's learned
     ``HyperHead``.
  6. Sigmoid MoE (256 experts top-8, selection-only correction bias,
     renormalized then scaled) with clamped SwiGLU on the ROUTED experts
     only. The leading dense layer and the shared expert are unclamped.

The model has no MTP block: the converter drops the nextn layers, so this
family cannot drive speculative decoding.
"""

import importlib
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import BaseModelArgs, create_attention_mask
from mlx_lm.models.cache import KVCache
from mlx_lm.models.mla import MultiLinear
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.models.switch_layers import SwitchGLU

from gmlx.models.deepseek_v4.hyper_connection import HyperHead
from gmlx.models.deepseek_v4.model import (
    LimitedSwiGLU,
    _expert_select,
)
from gmlx.models.deepseek_v4.model import (
    ensure_registered as _ds4_ensure_registered,
)


def ensure_registered() -> None:
    """Make ``import mlx_lm.models.hy_v4`` resolve, preferring upstream, and
    register the deepseek_v4 companions this module borrows from."""
    _ds4_ensure_registered()
    if "mlx_lm.models.hy_v4" not in sys.modules:
        try:
            importlib.import_module("mlx_lm.models.hy_v4")  # upstream wins
        except ImportError:
            sys.modules["mlx_lm.models.hy_v4"] = sys.modules[__name__]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    n_routed_experts: int
    num_experts_per_tok: int
    n_shared_experts: int
    first_k_dense_replace: int
    rms_norm_eps: float
    hc_mult: int
    hc_eps: float
    hc_magnitude: float
    index_n_heads: int = 0
    index_head_dim: int = 0
    index_topk: int = 0
    index_is_full: List[int] = field(default_factory=list)
    routed_scaling_factor: float = 1.0
    norm_topk_prob: bool = True
    scoring_func: str = "sigmoid"
    swiglu_limit: float = 0.0
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    # Interleaved (consecutive-pair) rope, matching the deployed vLLM
    # (is_neox_style=False) and llama.cpp's LLAMA_ROPE_TYPE_NORM. Kept a
    # config key because transformers 5.15 switched hy_v4 to rotate_half;
    # the GGUF supports either, so the convention is a graph-side choice.
    rope_traditional: bool = True
    max_position_embeddings: int = 1048576
    attention_bias: bool = False
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.scoring_func != "sigmoid":
            raise ValueError(
                f"hy_v4 routes sigmoid-only; got {self.scoring_func!r}")
        if self.index_topk and self.index_is_full and not self.index_is_full[0]:
            raise ValueError(
                "hy_v4 layer 0 must own an indexer; nothing precedes it "
                "to share a selection from")


class HyV4KVCache(KVCache):
    """Latent + rope-half KV cache that refuses ``--kv-bits``.

    Quantized SDPA has no sinks argument, so mlx-lm's helper raises on any
    cache carrying ``bits`` while a sink term is present, and HY4 has
    per-head sinks on every layer. ``kv_quant_unsupported`` is the seam
    ``gmlx.gen.generation.kv_quantization_unsupported`` reads to drop the
    flag up front with a named reason instead of failing at the first
    quantized step (MSAKVCache precedent).
    """

    kv_quant_unsupported = True


def split_cache(cache):
    """``(latent_cache, indexer_cache)`` for one layer's cache entry.

    A full layer's entry is a ``CacheList`` of two; a shared layer owns no
    indexer and its entry is the latent cache alone. Carrying an unwritten
    second slot on the 57 shared layers is not free: ``CacheList.state``
    reads ``keys.shape`` on every entry, so an empty slot crashes the
    generation loop's ``mx.eval`` of the cache state.
    """
    if cache is None:
        return None, None
    inner = getattr(cache, "caches", None)
    if inner is None:
        return cache, None
    return inner[0], (inner[1] if len(inner) > 1 else None)


class IndependentHyperConnection(nn.Module):
    """One iHC sublayer gate pair.

    ``pre`` collapses the ``hc_mult`` residual streams into the sublayer
    input; ``post`` scales the sublayer output back into each stream. Both
    come from one [2*hc, hc*hidden] mixing matrix over the weightless
    RMSNorm of the flattened streams. ``fn``/``base``/``scale`` are raw
    fp32 arrays, not Linear modules, so the loader's fp32 pins apply by
    name and nn.quantize never visits them.

    The reference runs the whole cycle in fp32: 78 layers x 2 sublayers of
    bf16 rounding on the residual streams compounds.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        hc = args.hc_mult
        self.hc_mult = hc
        self.norm_eps = args.rms_norm_eps
        self.hc_eps = args.hc_eps
        self.magnitude = args.hc_magnitude
        self.fn = mx.zeros((2 * hc, hc * args.hidden_size), dtype=mx.float32)
        self.base = mx.zeros((2 * hc,), dtype=mx.float32)
        self.scale = mx.ones((2,), dtype=mx.float32)

    def pre(self, x: mx.array):
        """``x`` [B, L, hc, D] -> (collapsed [B, L, D], post [B, L, hc])."""
        hc = self.hc_mult
        y = x.astype(mx.float32)
        z = mx.fast.rms_norm(y.flatten(-2), None, self.norm_eps)
        mixes = z @ self.fn.T
        pre = mx.sigmoid(
            mixes[..., :hc] * self.scale[0] + self.base[:hc]) + self.hc_eps
        post = self.magnitude * mx.sigmoid(
            mixes[..., hc:] * self.scale[1] + self.base[hc:]) + self.hc_eps
        collapsed = (pre[..., None] * y).sum(axis=-2)
        return collapsed.astype(x.dtype), post

    def expand(self, y: mx.array, residual: mx.array, post: mx.array):
        """``y`` [B, L, D] scaled by ``post`` into each residual stream."""
        out = (residual.astype(mx.float32)
               + post[..., None] * y.astype(mx.float32)[..., None, :])
        return out.astype(residual.dtype)


class HyV4Indexer(nn.Module):
    """DSA lightning indexer: per-token top-k key selection.

    Scores are ``relu(q . k)`` per head, weighted by a per-token head
    projection and summed; both reference scale factors fold into the head
    weights once, before the [B, H, L, S] score tensor exists. Rope covers
    the trailing ``qk_rope_head_dim`` of each 128-wide head.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.index_topk = args.index_topk
        self.wq_b = nn.Linear(
            args.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.hidden_size, self.head_dim, bias=False)
        # LayerNorm with weight and bias, at the model's rms eps: the
        # reference plumbs rms_norm_eps into this norm.
        self.k_norm = nn.LayerNorm(self.head_dim, eps=args.rms_norm_eps)
        self.weights_proj = nn.Linear(
            args.hidden_size, self.n_heads, bias=False)
        self.weight_scale = (self.head_dim * self.n_heads) ** -0.5
        self.rope = initialize_rope(
            dims=args.qk_rope_head_dim,
            base=args.rope_theta,
            traditional=args.rope_traditional,
            max_position_embeddings=args.max_position_embeddings,
            scaling_config=args.rope_scaling,
        )

    def _rope_tail(self, x: mx.array, offset) -> mx.array:
        nope, pe = mx.split(x, [self.nope_head_dim], axis=-1)
        return mx.concatenate([nope, self.rope(pe, offset)], axis=-1)

    def __call__(
        self,
        x: mx.array,
        qr: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any] = None,
    ) -> Optional[mx.array]:
        B, L, _ = x.shape
        q = self.wq_b(qr).reshape(
            B, L, self.n_heads, self.head_dim).swapaxes(1, 2)
        k = self.k_norm(self.wk(x)).reshape(B, 1, L, self.head_dim)

        offset = cache.offset if cache is not None else 0
        q = self._rope_tail(q, offset)
        k = self._rope_tail(k, offset)

        if cache is not None:
            k, _ = cache.update_and_fetch(k, mx.zeros([B, 1, L, 0], k.dtype))
        if k.shape[2] <= self.index_topk:
            return None

        scores = q @ k.swapaxes(-1, -2)
        scores = mx.maximum(scores, 0)
        weights = self.weights_proj(x) * self.weight_scale
        scores = scores * weights.swapaxes(-1, -2)[..., None]
        scores = scores.sum(axis=1, keepdims=True)
        if mask is not None:
            scores = mx.where(mask, scores, -float("inf"))
        return mx.argpartition(scores, kth=-self.index_topk, axis=-1)[
            ..., -self.index_topk:]


# The fused attention kernel caps at head dim 128, so an L>1 MLA forward
# runs composite and materializes a [B, 64, L, S] score tensor, plus a
# second one for the float additive mask the rope half rides in on: ~17 GB
# at an 8192-token chunk against 8192 keys. Past the thresholds below,
# prefill switches to an exact online softmax over [_STREAM_Q x
# _STREAM_BLOCK] tiles. Seeding the running maximum with the sink term
# folds it in without an extra pass, and keeps a query row finite when the
# selection leaves it no visible key.
_STREAM_MIN_KEYS = 4096
_STREAM_BLOCK = 2048
_STREAM_Q = 512
_NEG = mx.array(-1e30, dtype=mx.float32)


def _sparse_disabled() -> bool:
    """``GMLX_HY4_SPARSE_DISABLE=1`` forces dense attention on every layer.

    The reference for the sparse path. A llama.cpp oracle past the top_k
    boundary is expensive here (the STQ1_0 patch has no Metal kernel, so
    the oracle is CPU-only and prefills at ~2.3 tok/s), so the key-selection
    chain is also validated against this model's own dense forward: the two
    must agree wherever the selection is the identity, and must both hold a
    mid-context fact where it is not. The indexer still runs and still
    writes its cache under this flag - only the selection is dropped.
    Read per call so a test can flip it without a module reload.
    """
    return os.environ.get("GMLX_HY4_SPARSE_DISABLE", "") not in ("", "0")


def _tiled_absorbed_attention(q_n, q_pe, latent, k_pe, mask, scale, sinks):
    """Absorbed MQA over the latent plus the rope half, streamed in tiles.

    ``q_n`` [B, H, L, R] (post embed_q), ``q_pe`` [B, H, L, P], ``latent``
    [B, 1, S, R], ``k_pe`` [B, 1, S, P], ``mask`` bool with trailing axes
    [.., L, S] (True = attend) or None, ``sinks`` [H]. Exact online softmax
    in fp32.

    ``mx.depends`` chains every tile on the previous tile's accumulators.
    Without that edge the scheduler sees the score tiles as independent and
    materializes all of them at once, which is worse than the composite
    path this replaces (the glm5_next finding).
    """
    B, H, L, _ = q_n.shape
    S = latent.shape[2]
    m0 = sinks.astype(mx.float32).reshape(1, H, 1, 1)
    outs = []
    prev = None
    for q0 in range(0, L, _STREAM_Q):
        q1 = min(q0 + _STREAM_Q, L)
        qn32 = (q_n[:, :, q0:q1] * scale).astype(mx.float32)
        qp32 = (q_pe[:, :, q0:q1] * scale).astype(mx.float32)
        # Seed the running maximum with the sink logit and its weight with
        # exp(sink - m) = 1: the sink is one more column of the softmax
        # that contributes no value, so it belongs in the normalizer from
        # the start rather than in a correction afterwards.
        m = m0
        lse = mx.ones_like(m0)
        acc = mx.zeros((1, 1, 1, 1), dtype=mx.float32)
        for s0 in range(0, S, _STREAM_BLOCK):
            s1 = min(s0 + _STREAM_BLOCK, S)
            kb = latent[:, :, s0:s1].astype(mx.float32)
            pb = k_pe[:, :, s0:s1].astype(mx.float32)
            if prev is not None:
                kb, pb = mx.depends([kb, pb], prev)
            s = (qn32 @ kb.swapaxes(-1, -2)) + (qp32 @ pb.swapaxes(-1, -2))
            if mask is not None:
                s = mx.where(mask[..., q0:q1, s0:s1], s, _NEG)
            m_new = mx.maximum(m, s.max(axis=-1, keepdims=True))
            p = mx.exp(s - m_new)
            corr = mx.exp(m - m_new)
            acc = acc * corr + p @ kb
            lse = lse * corr + p.sum(axis=-1, keepdims=True)
            m = m_new
            prev = [acc, lse]
        outs.append((acc / lse).astype(q_n.dtype))
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)


class HyV4Attention(nn.Module):
    """Gated absorbed MLA with per-head sinks and DSA key selection.

    Module names follow mlx-lm deepseek_v32 so the DEEPSEEK2-style remap
    rows and KQuantMultiLinear (``embed_q``/``unembed_out``) engage
    unchanged.

    The rope half never enters the SDPA call: its scores are computed
    separately and passed as the float ``mask`` argument, the same trick
    mlx-lm's deepseek_v32 uses. The top-k selection is naturally a bool
    mask, and folding it into those float scores is a REQUIREMENT here,
    not an accident of the shape: mx.fast SDPA deviates measurably when a
    bool mask and a sinks term meet in the same call.
    """

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.q_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        # Over the MLA head size (256), never the 512-wide latent.
        self.scale = self.q_head_dim**-0.5

        hidden = args.hidden_size
        self.q_a_proj = nn.Linear(
            hidden, args.q_lora_rank, bias=args.attention_bias)
        self.q_a_layernorm = nn.RMSNorm(args.q_lora_rank, eps=args.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            args.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden, args.kv_lora_rank + args.qk_rope_head_dim,
            bias=args.attention_bias)
        self.kv_a_layernorm = nn.RMSNorm(
            args.kv_lora_rank, eps=args.rms_norm_eps)
        self.embed_q = MultiLinear(
            self.qk_nope_head_dim, args.kv_lora_rank, self.num_heads)
        self.unembed_out = MultiLinear(
            args.kv_lora_rank, self.v_head_dim, self.num_heads)
        self.attn_gate = nn.Linear(
            hidden, self.num_heads * self.v_head_dim, bias=False)
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim, hidden, bias=args.attention_bias)

        # Per-head learnable sinks, fp32 (a raw array, not a module).
        self.sinks = mx.zeros((self.num_heads,), dtype=mx.float32)

        self.is_full = bool(
            args.index_topk
            and (not args.index_is_full or args.index_is_full[layer_idx]))
        self.indexer = HyV4Indexer(args) if self.is_full else None

        self.rope = initialize_rope(
            dims=args.qk_rope_head_dim,
            base=args.rope_theta,
            traditional=args.rope_traditional,
            max_position_embeddings=args.max_position_embeddings,
            scaling_config=args.rope_scaling,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        top_k: Optional[mx.array] = None,
    ):
        B, L, _ = x.shape

        qr = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(qr).reshape(
            B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed = self.kv_a_proj_with_mqa(x)
        compressed, k_pe = mx.split(compressed, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(
            B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        latent = mx.expand_dims(self.kv_a_layernorm(compressed), axis=1)

        # A full layer's cache is a CacheList (latent, indexer keys); a
        # shared layer owns no indexer, so its cache is the latent alone.
        kv_cache, idx_cache = split_cache(cache)

        offset = kv_cache.offset if kv_cache is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)

        if kv_cache is not None:
            latent, k_pe = kv_cache.update_and_fetch(latent, k_pe)

        if self.indexer is not None:
            top_k = self.indexer(x, qr, mask, cache=idx_cache)
        if _sparse_disabled():
            # Drop the selection, but only AFTER the indexer has written its
            # cache: skipping the call would leave the slot unwritten, and
            # CacheList.state reads keys.shape on every entry it holds. The
            # two paths then differ in the selection alone, which is the
            # whole point of the comparison.
            top_k = None
        if top_k is not None:
            if L == 1:
                idx = top_k[:, :, 0, :, None]
                latent = mx.take_along_axis(
                    latent,
                    mx.broadcast_to(idx, idx.shape[:-1] + (latent.shape[-1],)),
                    axis=2)
                k_pe = mx.take_along_axis(
                    k_pe,
                    mx.broadcast_to(idx, idx.shape[:-1] + (k_pe.shape[-1],)),
                    axis=2)
                if mask is not None:
                    mask = mx.take_along_axis(mask, top_k, axis=-1)
            else:
                shape = list(top_k.shape)
                shape[-1] = latent.shape[2]
                sparse = mx.zeros(shape, dtype=mx.bool_)
                sparse = mx.put_along_axis(
                    sparse, top_k, mx.array(True), axis=-1)
                mask = sparse if mask is None else (sparse & mask)

        # Keep the indexer cache write on the graph even when the selection
        # goes unused, so the graph does not grow across steps.
        if kv_cache is not None and idx_cache is not None \
                and idx_cache.keys is not None:
            kv_cache.keys = mx.depends(
                kv_cache.keys, (idx_cache.keys, idx_cache.values))

        S = latent.shape[2]
        if (L > 1 and not isinstance(mask, str)
                and (L > _STREAM_Q or S > _STREAM_MIN_KEYS)):
            # Deep prefill chunk: exact online softmax over score tiles.
            # The bool mask goes straight in - the tiled path runs its own
            # fp32 softmax, so there is no mx.fast call to keep a float
            # mask for. It needs a materialized mask, so a string ("causal")
            # one keeps the path below; HyV4Model always builds an array.
            bmask = mask
            if bmask is not None and bmask.ndim == 3:
                bmask = bmask[:, None]
            out = _tiled_absorbed_attention(
                self.embed_q(q_nope), q_pe, latent, k_pe, bmask,
                self.scale, self.sinks)
            out = self.unembed_out(out)
            out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
            out = out * mx.sigmoid(self.attn_gate(x)).astype(out.dtype)
            return self.o_proj(out), top_k

        # The rope half rides in as the float additive mask, so the bool
        # selection mask folds into it here. That is a requirement, not an
        # accident of the shape: mx.fast SDPA deviates measurably when a
        # bool mask and a sinks term meet in the same call.
        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask, pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype))

        sinks = self.sinks.astype(q_nope.dtype)
        if L == 1:
            out = mx.fast.scaled_dot_product_attention(
                self.embed_q(q_nope), latent, latent,
                scale=self.scale, mask=pe_scores, sinks=sinks)
            out = self.unembed_out(out)
        else:
            k = self.embed_q(latent, transpose=False)
            v = self.unembed_out(latent)
            out = mx.fast.scaled_dot_product_attention(
                q_nope, k, v, scale=self.scale, mask=pe_scores, sinks=sinks)

        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        out = out * mx.sigmoid(self.attn_gate(x)).astype(out.dtype)
        return self.o_proj(out), top_k


class HyV4MLP(nn.Module):
    """Plain SwiGLU MLP: the leading dense layer and the shared expert.

    Only the routed experts are clamped; the converter leaves
    ``swiglu_clamp_shexp`` at its 0 default on purpose.
    """

    def __init__(self, args: ModelArgs, intermediate_size: Optional[int] = None):
        super().__init__()
        dim = args.hidden_size
        hidden = intermediate_size or args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class HyV4MoEGate(nn.Module):
    """Sigmoid router: the correction bias steers selection only, weights
    are the unbiased sigmoid scores, renormalized then scaled.

    A gate submodule returning ``(inds, weights)`` with an int ``top_k`` -
    the shape stream/moe_experts and stream/lookahead duck-type on.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.n_routed_experts = args.n_routed_experts
        self.routed_scaling_factor = args.routed_scaling_factor
        self.norm_topk_prob = args.norm_topk_prob
        # fp32 like the F32 wire tensors: sigmoid top-8-of-256 with a
        # correction bias is near-tie-heavy, and llama.cpp routes fp32.
        self.weight = mx.zeros(
            (args.n_routed_experts, args.hidden_size), dtype=mx.float32)
        self.e_score_correction_bias = mx.zeros(
            (args.n_routed_experts,), dtype=mx.float32)

    def __call__(self, x: mx.array):
        logits = x.astype(mx.float32) @ self.weight.T
        return _expert_select(
            logits,
            self.e_score_correction_bias,
            self.top_k,
            self.routed_scaling_factor,
            self.norm_topk_prob,
            "sigmoid",
        )


class HyV4MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate = HyV4MoEGate(args)
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.n_routed_experts,
            activation=LimitedSwiGLU(args.swiglu_limit),
        )
        # Shared expert: unclamped, and unscaled (routed_scaling_factor is
        # already folded into the routed weights by the gate).
        self.shared_experts = HyV4MLP(
            args, args.moe_intermediate_size * args.n_shared_experts)

    def __call__(self, x: mx.array) -> mx.array:
        inds, scores = self.gate(x)
        if getattr(self.switch_mlp, "_kq_mix_scores", False):
            y = self.switch_mlp(x, inds, scores)
        else:
            y = self.switch_mlp(x, inds)
            if y.ndim == scores.ndim + 1:
                y = (y * scores[..., None].astype(y.dtype)).sum(-2)
        return y + self.shared_experts(x)


class HyV4DecoderLayer(nn.Module):
    """Two iHC cycles per block: collapse, sublayer, redistribute."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = HyV4Attention(args, layer_idx)
        self.mlp = (
            HyV4MLP(args) if layer_idx < args.first_k_dense_replace
            else HyV4MoE(args))
        self.input_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps)
        self.attn_hc = IndependentHyperConnection(args)
        self.ffn_hc = IndependentHyperConnection(args)

    def __call__(
        self,
        h: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        top_k: Optional[mx.array] = None,
    ):
        x, post = self.attn_hc.pre(h)
        x, top_k = self.self_attn(
            self.input_layernorm(x), mask=mask, cache=cache, top_k=top_k)
        h = self.attn_hc.expand(x, h, post)

        x, post = self.ffn_hc.pre(h)
        x = self.mlp(self.post_attention_layernorm(x))
        return self.ffn_hc.expand(x, h, post), top_k


class HyV4Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [HyV4DecoderLayer(args, i)
                       for i in range(args.num_hidden_layers)]
        self.hc_head = HyperHead(args)
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)

        latent_cache, _ = split_cache(cache[0])
        mask = create_attention_mask(h, latent_cache, return_array=True)

        # hc_mult exact copies of the embedding: no scaling, no one-hot.
        h = mx.contiguous(mx.broadcast_to(
            h[:, :, None, :],
            (h.shape[0], h.shape[1], self.args.hc_mult, h.shape[2])))

        top_k = None
        for layer, layer_cache in zip(self.layers, cache):
            h, top_k = layer(h, mask, layer_cache, top_k=top_k)

        return self.norm(self.hc_head(h))


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = HyV4Model(args)
        if args.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(
                args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        out = self.model(inputs, cache)
        if self.lm_head is None:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        # Full layers carry two streams: the latent + rope half, and the
        # indexer keys. Shared layers own no indexer, so they carry the
        # latent alone - an unwritten second slot is not inert, because
        # CacheList.state reads keys.shape on every entry it holds.
        # Per-layer cache shapes are the norm here (qwen4_exp mixes three).
        from gmlx.cache.compat import construction_cache_module

        cmod = construction_cache_module()
        kv_cls = getattr(cmod, "KVCache", KVCache)
        list_cls = getattr(cmod, "CacheList", None)
        if list_cls is None:
            from mlx_lm.models.cache import CacheList as list_cls

        return [list_cls(HyV4KVCache(), kv_cls())
                if layer.self_attn.indexer is not None else HyV4KVCache()
                for layer in self.model.layers]

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        n = self.args.num_hidden_layers

        def _past_trunk(k: str) -> bool:
            if not k.startswith("model.layers."):
                return False
            idx = k.split(".", 3)[2]
            return idx.isdigit() and int(idx) >= n

        weights = {k: v for k, v in weights.items() if not _past_trunk(k)}
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    @property
    def cast_predicate(self):
        def predicate(path: str):
            # iHC mixers and the final collapse head: the reference keeps
            # them fp32 and the streams accumulate over 78 layers.
            if ".attn_hc." in path or ".ffn_hc." in path:
                return False
            if path.startswith("model.hc_head."):
                return False
            # Per-head sinks and the routing pair are semantically fp32.
            if path.endswith("self_attn.sinks"):
                return False
            if "e_score_correction_bias" in path:
                return False
            if path.endswith("mlp.gate.weight"):
                return False
            # Indexer head weights: a sign-free fp32 GEMM decides near-tied
            # key rankings.
            if path.endswith("indexer.weights_proj.weight"):
                return False
            return True

        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            # nn.quantize visits Linear/Embedding modules; the raw-array
            # leaves (sinks, hc fn/base/scale, gate.weight) are never seen.
            if path.endswith("indexer.weights_proj"):
                return False
            if path.endswith(("indexer.wq_b", "indexer.wk")):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate
