# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
# Portions copyright (c) 2026 Apple Inc. (mlx-lm kimi_linear skeleton, MIT)
"""Vendored mlx-lm-style model for GLM-5.3-Flash (GGUF arch ``glm5next``).

mlx-lm has no glm5_next class; this module is the runtime for llama.cpp PR
27754 conversions. The architecture is a hybrid of mechanisms gmlx already
runs, recombined:

  1. KDA linear attention (34 of 45 layers): kimi_k3's per-key-channel-decay
     delta rule (``exp(lb * sigmoid(exp(A_log) * (f(x) + dt_bias)))``,
     lb = -5.0, ``ssm_a`` arrives folded as ``-exp(A_log)``), with a
     low-rank ``g_a/g_b`` sigmoid output gate instead of K3's full-rank
     ``g_proj``.
  2. Nope-only MLA (every 4th layer): absorbed embed_q/unembed_out over a
     512-wide latent, head dim 256, scale ``256**-0.5``, no rope anywhere.
  3. A pooled DSA lightning indexer on the MLA layers: per-token keys
     (LayerNorm at eps 1e-6, NOT the 1e-5 rms eps) and gates are pooled
     4:1 by a per-channel softmax-over-slots compressor with an additive
     positional table; queries score POOLS, the top index_topk/kpool = 512
     pools expand x4 to token ids, and the query's own incomplete trailing
     pool is always attended. Sparse attention engages only above
     ``index_topk + kpool - 1`` = 2051 total keys; below that selection is
     the identity and the dense path is the same function.

     Convention (pinned against the PR's kpool masks): pool ``p`` is
     scoreable by the query at absolute position ``q`` iff it is complete
     and its last member is causally visible, i.e. ``p < (q + 1) // 4`` -
     the query's own just-completed pool included at ``q % 4 == 3``, where
     the tail is empty. ``PoolingCache.make_mask`` computes exactly this.
  4. Sigmoid MoE (288 experts top-8, selection-only correction bias, x2.5
     renorm, one shared expert, 3 leading dense layers) with clamped
     SwiGLU (limit 10) on every FFN, dense and shared included.
  5. deepseek_v4's 4-stream sinkhorn hyper-connections, reused verbatim;
     the final collapse is an unweighted mean (no learned head).

``GMLX_GLM5_SPARSE_DISABLE=1`` forces dense attention at any depth (A/B
debugging); ``GMLX_GLM5_ABSORBED_PREFILL=1`` runs the absorbed MLA form at
all lengths (the llama.cpp evaluation order, for parity bisects).
"""

import importlib
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.gated_delta import gated_delta_kernel, gated_delta_ops
from mlx_lm.models.mla import MultiLinear
from mlx_lm.models.switch_layers import SwitchGLU

from gmlx.models.deepseek_v4.cache import PoolingCache
from gmlx.models.deepseek_v4.hyper_connection import (
    HyperConnection,
    hc_expand,
    hc_expand_collapse,
    hc_expand_m1,
)
from gmlx.models.deepseek_v4.model import (
    LimitedSwiGLU,
    _expert_select,
    _limited_swiglu,
)
from gmlx.models.deepseek_v4.model import (
    ensure_registered as _ds4_ensure_registered,
)
from gmlx.models.kimi_k3 import ShortConv1d, _kda_decay_lb

_MOE_MIX_SCORES = os.environ.get("GMLX_GLM5_MOE_MIX", "1") != "0"
_SPARSE_DISABLE = os.environ.get("GMLX_GLM5_SPARSE_DISABLE", "0") == "1"
_ABSORBED_PREFILL = os.environ.get("GMLX_GLM5_ABSORBED_PREFILL", "0") == "1"
_SPARSE_GATHER = os.environ.get("GMLX_GLM5_SPARSE_GATHER", "1") != "0"


def ensure_registered() -> None:
    """Make ``import mlx_lm.models.glm5_next`` resolve, preferring upstream,
    and register the deepseek_v4 companions (PoolingCache on the cache
    namespaces) this model's caches depend on."""
    _ds4_ensure_registered()
    if "mlx_lm.models.glm5_next" not in sys.modules:
        try:
            importlib.import_module("mlx_lm.models.glm5_next")  # upstream wins
        except ImportError:
            sys.modules["mlx_lm.models.glm5_next"] = sys.modules[__name__]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    rms_norm_eps: float
    # per-layer schedule: "linear_attention" (KDA) | "full_attention" (MLA)
    layer_types: List[str] = field(default_factory=list)
    # KDA
    kda_head_dim: int = 128
    ssm_conv_kernel: int = 4
    kda_gate_lower_bound: float = -5.0
    # MLA (nope-only)
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 256
    qk_rope_head_dim: int = 0
    v_head_dim: int = 256
    # Pooled lightning indexer
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    index_kpool: int = 4
    index_knorm_eps: float = 1e-6
    # MoE
    n_routed_experts: int = 0
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 0
    n_shared_experts: int = 1
    first_k_dense_replace: int = 0
    routed_scaling_factor: float = 1.0
    norm_topk_prob: bool = True
    scoring_func: str = "sigmoid"
    swiglu_limit: float = 10.0
    # Hyper-connections
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    # Native MTP head depth (blk.45 in the GGUF; the trunk ignores it)
    mtp_num_hidden_layers: int = 0
    max_position_embeddings: int = 1048576
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.qk_rope_head_dim != 0:
            raise ValueError(
                "glm5_next is nope-only; a conversion with "
                f"rope.dimension_count={self.qk_rope_head_dim} is not this "
                "architecture")
        if self.scoring_func != "sigmoid":
            raise ValueError(
                f"glm5_next routes sigmoid-only; got {self.scoring_func!r}")


class Glm5NextMLP(nn.Module):
    """Dense clamped-SwiGLU MLP (leading dense layers and the shared
    expert). The clamp is not MoE-only: the reference routes every FFN
    through the same limited activation."""

    def __init__(self, args: ModelArgs, intermediate_size: Optional[int] = None):
        super().__init__()
        dim = args.hidden_size
        hidden = intermediate_size or args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        self._limit = args.swiglu_limit

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(
            _limited_swiglu(self.gate_proj(x), self.up_proj(x), self._limit))


class Glm5NextMoEGate(nn.Module):
    """Sigmoid noaux-tc router: the correction bias steers selection only,
    weights are the unbiased sigmoid scores, renormalized then x2.5.

    A gate submodule returning ``(inds, weights)`` with an int ``top_k`` -
    the shape stream/moe_experts and stream/lookahead duck-type on."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.n_routed_experts = args.n_routed_experts
        self.routed_scaling_factor = args.routed_scaling_factor
        self.norm_topk_prob = args.norm_topk_prob
        # fp32 like the F32 wire tensor: sigmoid top-8-of-288 with a
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


class Glm5NextMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate = Glm5NextMoEGate(args)
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.n_routed_experts,
            activation=LimitedSwiGLU(args.swiglu_limit),
        )
        # Shared expert: full width, unscaled (routed_scaling_factor is
        # already folded into the routed weights by the gate).
        self.shared_experts = Glm5NextMLP(
            args, args.moe_intermediate_size * args.n_shared_experts)

    def __call__(self, x: mx.array) -> mx.array:
        inds, scores = self.gate(x)
        if _MOE_MIX_SCORES and getattr(self.switch_mlp, "_kq_mix_scores", False):
            # Fused arm folds the score-weighted sum into the down gather.
            y = self.switch_mlp(x, inds, scores)
        else:
            y = self.switch_mlp(x, inds)
            if y.ndim == scores.ndim + 1:
                y = (y * scores[..., None].astype(y.dtype)).sum(-2)
        return y + self.shared_experts(x)


@mx.compile
def _pool_windows(keys: mx.array, gates: mx.array, ape: mx.array) -> mx.array:
    # Per-channel softmax over the slot axis: ape[slot, channel] adds to the
    # gate logits pre-softmax (slot = position % kpool by construction - the
    # windows are position-aligned), then the pooled key is the per-channel
    # weighted average of the member keys. fp32 like the reference.
    g = gates.astype(mx.float32) + ape
    probs = mx.softmax(g, axis=-2, precise=True)
    return (keys.astype(mx.float32) * probs).sum(axis=-2)


class Glm5NextCompressor(nn.Module):
    """kpool-4 disjoint-window pooling of the indexer's keys and gates.

    Unlike deepseek_v4's ratio-4 compressor there is no separate wkv (the
    scorer's own keys are pooled), no norm, no rope, and no cross-window
    overlap - hence PoolingCache(kpool, lookback=False)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.ratio = args.index_kpool
        self.head_dim = args.index_head_dim
        self.wgate = nn.Linear(args.hidden_size, args.index_head_dim, bias=False)
        self.ape = mx.zeros(
            (args.index_kpool, args.index_head_dim), dtype=mx.float32)

    def __call__(
        self,
        keys: mx.array,
        gates: mx.array,
        pool_cache: Optional[PoolingCache],
        offset: Any,
    ) -> mx.array:
        B, L, D = keys.shape
        if pool_cache is None:
            usable = (L // self.ratio) * self.ratio
            ready_k, ready_g = keys[:, :usable], gates[:, :usable]
        else:
            ready_k, ready_g, _ = pool_cache.accumulate_windows(
                keys, gates, offset)

        if ready_k.size == 0:
            new_pooled = mx.zeros((B, 0, D), dtype=keys.dtype)
        else:
            kw = mx.unflatten(ready_k, 1, (-1, self.ratio))
            gw = mx.unflatten(ready_g, 1, (-1, self.ratio))
            new_pooled = _pool_windows(kw, gw, self.ape).astype(keys.dtype)

        if pool_cache is not None:
            return pool_cache.update_and_fetch(new_pooled)
        return new_pooled


class Glm5NextIndexer(nn.Module):
    """Pooled lightning indexer: scores pools, selects whole pools.

    The key+gate store is NOT gated on the sparse path - below the
    selection width the scoring is skipped but every token's key and gate
    must still land in the pool cache, or the first forward to cross the
    width would pool windows that were never written."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.index_topk = args.index_topk
        self.kpool = args.index_kpool
        # 512 whole pools; the reference's select_k = index_topk // kpool.
        self.select_k = args.index_topk // args.index_kpool
        # Selection width: select_k pools plus the (kpool-1)-wide tail.
        self.n_select = args.index_topk + args.index_kpool - 1
        self.wq_b = nn.Linear(
            args.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.hidden_size, self.head_dim, bias=False)
        # A genuine LayerNorm (weight AND bias) at eps 1e-6 - hardcoded in
        # the reference, distinct from the 1e-5 rms eps.
        self.k_norm = nn.LayerNorm(args.index_head_dim, eps=args.index_knorm_eps)
        self.weights_proj = nn.Linear(
            args.hidden_size, args.index_n_heads, bias=False)
        self.compressor = Glm5NextCompressor(args)
        # Folded score scale: head_dim**-0.5 * n_heads**-0.5. relu is
        # positively homogeneous, so folding both constants into the head
        # weights is the same function as scaling the scores.
        self._w_scale = (args.index_head_dim * args.index_n_heads) ** -0.5

    def __call__(
        self,
        x: mx.array,
        q_residual: mx.array,
        pool_cache: Optional[PoolingCache],
        offset: Any,
        total_keys: int,
    ) -> Optional[mx.array]:
        """Store this chunk's keys+gates; return the selected pool ordinals
        ``[B, L, select_k]`` when sparse selection binds, else None (dense:
        every visible key would be selected anyway)."""
        B, L, _ = x.shape
        keys = self.k_norm(self.wk(x))
        gates = self.compressor.wgate(x)
        pooled = self.compressor(keys, gates, pool_cache, offset)

        if _SPARSE_DISABLE or total_keys <= self.n_select:
            return None
        if pooled.shape[1] < self.select_k:
            return None

        q = self.wq_b(q_residual).reshape(B, L, self.n_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)

        # fp32 head weights: the weights are sign-free (no softmax, no relu
        # on them) and a bf16 head gate moves logits enough to flip
        # near-tied pools; the weight tensor itself is kept fp32.
        w = (x.astype(mx.float32) @ self.weights_proj.weight.T) * self._w_scale

        if pool_cache is not None:
            pmask = pool_cache.make_mask(L, offset)
        elif L > 1:
            # Cacheless forward: apply the same visibility predicate
            # (p < (q + 1) // kpool) so early queries cannot burn selection
            # slots on their own future's pools.
            pool_idx = mx.arange(pooled.shape[1])
            query_idx = mx.arange(offset + 1, offset + L + 1)
            pmask = pool_idx < query_idx[:, None] // self.kpool
        else:
            pmask = None
        if pmask is not None and pmask.ndim == 2:
            pmask = pmask[None]

        # Scoring is chunked over queries (the reference chunks by 512):
        # the fp32 [B, heads, chunk, pools] score tile plus its relu copy
        # stay bounded no matter the prefill chunk width. (The pool axis
        # still scales with depth; the kq indexer kernels take over there.)
        pooled_t = pooled[:, None].swapaxes(-1, -2).astype(mx.float32)
        sel_parts = []
        for j0 in range(0, L, 512):
            j1 = min(j0 + 512, L)
            # relu sits BETWEEN the per-head dot and the head weighting:
            # the weights are sign-free, so moving it is a different
            # function.
            s = q[:, :, j0:j1].astype(mx.float32) @ pooled_t
            s = mx.maximum(s, 0)
            s = (s * w[:, j0:j1].swapaxes(-1, -2)[..., None]).sum(axis=1)
            if pmask is not None:
                s = mx.where(
                    pmask[:, j0:j1], s, mx.finfo(s.dtype).min)
            # Top-k over POOLS, never over member cells: relu drives most
            # pool scores to exactly 0.0, tie groups span pools, and an
            # unordered cell-level cut leaves partial pools (measured 5-7%
            # of query rows in the reference tree). Whole-pool selection
            # leaves none.
            sel_parts.append(
                mx.argpartition(-s, kth=self.select_k - 1, axis=-1)[
                    ..., : self.select_k])
        if len(sel_parts) == 1:
            return sel_parts[0]
        return mx.concatenate(sel_parts, axis=1)


def _dequantized(kv, cache):
    """Materialize a possibly-quantized KV fetch as a dense array."""
    if isinstance(kv, tuple):
        return mx.dequantize(
            *kv, group_size=cache.group_size, bits=cache.bits)
    return kv


# MLX's fused attention kernel caps at head dim 128, so every L>1 MLA
# forward here runs composite and materializes [B, H, L, S] scores - at
# 16k keys that is a multi-GB transient on top of a ~101 GB resident
# model. Beyond _STREAM_MIN_KEYS the L>1 paths switch to an absorbed
# online-softmax accumulation over [_STREAM_Q x _STREAM_BLOCK] tiles: no
# per-head K/V expansion, peak bounded by a few score tiles at any depth.
_STREAM_MIN_KEYS = 4096
_STREAM_BLOCK = 2048
_STREAM_Q = 512
_NEG = mx.array(-1e30, dtype=mx.float32)


def _streamed_absorbed_attention(q_n, latent, mask, scale):
    """Absorbed MQA over the latent, streamed over key blocks.

    ``q_n`` [B, H, L, D] (post embed_q), ``latent`` [B, 1, S, D], ``mask``
    bool with trailing axes [.., L, S] (True = attend) or None. Exact
    online softmax in fp32, tiled [_STREAM_Q x _STREAM_BLOCK]. The mask
    folds into the score GEMM as a finite -1e30 additive bias (mx.addmm):
    masked lanes underflow to exactly zero through exp once any real key
    raises the row max, and an all-masked tile stays finite (exp(0))
    until a later block's corr factor annihilates it; a row with no
    visible key anywhere cannot occur under causal masking. mx.depends
    chains every tile on the previous tile's accumulators - without the
    edge the scheduler sees the score tiles as independent and
    materializes all of them at once, which is worse than the composite
    path this replaces."""
    B, H, L, D = q_n.shape
    S = latent.shape[2]
    outs = []
    prev = None
    for q0 in range(0, L, _STREAM_Q):
        q1 = min(q0 + _STREAM_Q, L)
        q32 = (q_n[:, :, q0:q1] * scale).astype(mx.float32)
        m = lse = acc = None
        for s0 in range(0, S, _STREAM_BLOCK):
            s1 = min(s0 + _STREAM_BLOCK, S)
            kb = latent[:, :, s0:s1].astype(mx.float32)
            if prev is not None:
                (kb,) = mx.depends([kb], prev)
            kt = kb.swapaxes(-1, -2)
            if mask is not None:
                am = mx.where(mask[..., q0:q1, s0:s1], 0.0, _NEG)
                s = mx.addmm(am, q32, kt)  # [B, H, q1-q0, s1-s0]
            else:
                s = q32 @ kt
            if acc is None:
                m = s.max(axis=-1, keepdims=True)
                p = mx.exp(s - m)
                acc = p @ kb
                lse = p.sum(axis=-1, keepdims=True)
            else:
                m_new = mx.maximum(m, s.max(axis=-1, keepdims=True))
                p = mx.exp(s - m_new)
                corr = mx.exp(m - m_new)
                acc = acc * corr + p @ kb
                lse = lse * corr + p.sum(axis=-1, keepdims=True)
                m = m_new
            prev = [acc, lse]
        outs.append((acc / mx.maximum(lse, 1e-30)).astype(q_n.dtype))
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)


class Glm5NextMLAAttention(nn.Module):
    """Nope-only MLA with the pooled DSA indexer.

    Module names follow mlx-lm deepseek_v3/kimi_k3 so the DEEPSEEK2-style
    remap rows and KQuantMultiLinear (embed_q/unembed_out) engage
    unchanged. The cache is CacheList(KVCache latent-only, PoolingCache).
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.q_head_dim = args.qk_nope_head_dim  # nope-only
        self.v_head_dim = args.v_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        # Over the MLA head size, not the post-absorption latent width:
        # 256**-0.5 = 0.0625, never 512**-0.5.
        self.scale = self.q_head_dim**-0.5

        hidden = args.hidden_size
        self.q_a_proj = nn.Linear(hidden, args.q_lora_rank, bias=False)
        self.q_a_layernorm = nn.RMSNorm(args.q_lora_rank, eps=args.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            args.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)
        # kv_a is the bare 512-wide latent: no rope half, no split.
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden, args.kv_lora_rank, bias=False)
        self.kv_a_layernorm = nn.RMSNorm(args.kv_lora_rank,
                                         eps=args.rms_norm_eps)
        self.embed_q = MultiLinear(
            self.qk_nope_head_dim, args.kv_lora_rank, self.num_heads)
        self.unembed_out = MultiLinear(
            args.kv_lora_rank, self.v_head_dim, self.num_heads)
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim, hidden, bias=False)
        self.indexer = Glm5NextIndexer(args)
        self._kpool = args.index_kpool
        # Debug/test seam: False routes sparse decode through the masked
        # path instead of the latent gather (same function).
        self._decode_gather = True

    def _gathered_sparse_prefill(
        self, q_n, latent_d, sel_pools, offset: int, L: int, S: int
    ) -> mx.array:
        """Sparse prefill over gathered keys instead of a full-width mask.

        Selections inside a 512-query block overlap heavily (~0.5-0.8 of
        the visible pools measured on real prompts), so per block the keys
        reduce to: the union of pools every block query could see whole
        (pool id < the block's earliest tail start), gathered once, plus
        the block's dense local span, where per-query tails and pools that
        complete mid-block live. Visibility per row is rebuilt over the
        gathered columns - selected-pool membership for the union part,
        ``causal & (tail | selected-member)`` for the local part - so the
        attended set per query is exactly the masked path's.

        Spilled selections (queries with fewer visible pools than
        select_k) can only name invisible pools, which are all >= the
        query's own pool horizon and therefore >= the union cutoff: the
        keep filters drop every spill.

        One bounded host sync per block reads the union size; each block's
        output is eval'd so the [H, block, G] score transient is freed
        before the next block (the _chunked_prefill memory idiom)."""
        r = self._kpool
        n_pools = S // r
        outs = []
        for j0 in range(0, L, _STREAM_Q):
            j1 = min(j0 + _STREAM_Q, L)
            lb = j1 - j0
            ls = (offset + j0 + 1) // r * r
            lp = ls // r
            sel_j = sel_pools[0, j0:j1]

            parts = []
            mask_parts = []
            q_abs = offset + mx.arange(j0, j1)[:, None]

            if lp > 0:
                keep = sel_j < lp
                safe = mx.where(keep, sel_j, n_pools)
                present = mx.zeros((n_pools + 1,), dtype=mx.bool_)
                present = mx.put_along_axis(
                    present, safe.reshape(-1), mx.array(True), axis=0
                )[:lp]
                pres_i = present.astype(mx.int32)
                ranks = mx.cumsum(pres_i) - pres_i
                U = int(present.sum())
                if U > 0:
                    ids = mx.arange(lp)
                    order = mx.argsort(mx.where(present, ids, lp + ids))
                    ptok = (order[:U, None] * r
                            + mx.arange(r)).reshape(-1)
                    parts.append(ptok)
                    sel_rank = mx.where(
                        keep,
                        mx.take(ranks, mx.minimum(sel_j, lp - 1)),
                        U)
                    vis_p = mx.zeros((lb, U + 1), dtype=mx.bool_)
                    vis_p = mx.put_along_axis(
                        vis_p, sel_rank, mx.array(True), axis=-1)[:, :U]
                    mask_parts.append(mx.broadcast_to(
                        vis_p[..., None], (lb, U, r)).reshape(lb, -1))

            local = mx.arange(ls, offset + j1)
            parts.append(local)
            causal = local[None] <= q_abs
            tail = local[None] >= (q_abs + 1) // r * r
            npl = (offset + j1 - 1) // r - lp + 1
            keep_l = (sel_j >= lp) & (sel_j < lp + npl)
            bm = mx.zeros((lb, npl + 1), dtype=mx.bool_)
            bm = mx.put_along_axis(
                bm, mx.where(keep_l, sel_j - lp, npl), mx.array(True),
                axis=-1)[:, :npl]
            member = mx.take_along_axis(
                bm, mx.broadcast_to((local // r - lp)[None], causal.shape),
                axis=-1)
            mask_parts.append(causal & (tail | member))

            tok = parts[0] if len(parts) == 1 else mx.concatenate(parts)
            kv_g = mx.take(latent_d[0, 0], tok, axis=0)[None, None]
            blk_mask = (mask_parts[0] if len(mask_parts) == 1
                        else mx.concatenate(mask_parts, axis=-1))
            out_b = scaled_dot_product_attention(
                q_n[:, :, j0:j1], kv_g, kv_g, cache=None, scale=self.scale,
                mask=blk_mask[None, None])
            mx.eval(out_b)
            outs.append(out_b)
        return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)

    def _sparse_mask(self, sel_pools, offset: int, L: int, S: int):
        """Bool visibility mask [B, L, S] for the masked-SDPA sparse path:
        causal AND (own trailing tail OR a selected pool's member).

        Spilled selections (a query with fewer visible pools than select_k
        picks arbitrary losers among the -inf ties) expand to members of
        incomplete or future pools; with contiguous per-sequence positions
        those members are exactly the tail-or-future positions, so the
        causal AND tail terms neutralize every spill - the llama.cpp
        cand_mask argument, collapsed for hole-free caches."""
        B = sel_pools.shape[0]
        r = self._kpool
        q_pos = offset + mx.arange(L)[:, None]  # [L, 1]
        key_pos = mx.arange(S)  # [S]
        causal = key_pos <= q_pos
        tail = key_pos >= (q_pos + 1) // r * r
        # Expand selected pools to member token ids and scatter.
        tok = (sel_pools[..., None] * r + mx.arange(r)).reshape(B, L, -1)
        sel = mx.zeros((B, L, S), dtype=mx.bool_)
        sel = mx.put_along_axis(
            sel, mx.minimum(tok, S - 1), mx.array(True), axis=-1)
        return (causal & tail)[None] | (causal[None] & sel)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        kv_cache = pool_cache = None
        if cache is not None:
            kv_cache, pool_cache = cache[0], cache[1]

        offset = kv_cache.offset if kv_cache is not None else 0

        q_residual = self.q_a_layernorm(self.q_a_proj(x))
        latent = self.kv_a_layernorm(self.kv_a_proj_with_mqa(x))
        latent = mx.expand_dims(latent, axis=1)  # [B, 1, L, 512]

        if kv_cache is not None:
            # Latent-only: the zero-width value keeps KVCache's (keys,
            # values) contract without storing anything for V.
            values = mx.zeros(latent.shape[:-1] + (0,), dtype=latent.dtype)
            latent_all, _ = kv_cache.update_and_fetch(latent, values)
        else:
            latent_all = latent

        if isinstance(offset, int):
            total_keys = offset + L
        else:
            # Batched per-row offsets: sparse selection needs per-row tail
            # arithmetic (the llama.cpp cell machinery); run dense and say
            # so once. Exact below 2051 keys per row, approximate beyond.
            total_keys = 0
            if not getattr(Glm5NextMLAAttention, "_warned_batch", False):
                Glm5NextMLAAttention._warned_batch = True
                print(
                    "warning: glm5_next batched decode runs dense attention "
                    "(sparse top-2048 selection is single-sequence for now); "
                    "exact to 2051 tokens per row",
                    file=sys.stderr,
                )

        sel_pools = self.indexer(
            x, q_residual, pool_cache, offset, total_keys)

        q = self.q_b_proj(q_residual)
        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(
            0, 2, 1, 3)

        if (sel_pools is not None and L == 1 and isinstance(offset, int)
                and self._decode_gather):
            # Sparse decode: gather the selected latents + the tail rows and
            # run absorbed MQA over them. Every gathered row is a complete
            # visible pool member or the tail, so no mask is needed.
            latent_d = _dequantized(latent_all, kv_cache)
            r = self._kpool
            tok = (sel_pools[..., None] * r + mx.arange(r)).reshape(B, -1)
            q_abs = offset  # this token's absolute position
            tail_start = (q_abs + 1) // r * r
            if q_abs >= tail_start:
                tail = mx.broadcast_to(
                    mx.arange(tail_start, q_abs + 1)[None],
                    (B, q_abs + 1 - tail_start))
                tok = mx.concatenate([tok, tail], axis=1)
            kv_g = mx.take_along_axis(
                latent_d[:, 0], tok[..., None], axis=1)[:, None]
            q_n = self.embed_q(q)
            out = scaled_dot_product_attention(
                q_n, kv_g, kv_g, cache=None, scale=self.scale, mask=None)
            out = self.unembed_out(out)
        elif sel_pools is not None:
            # Sparse prefill/chunk: gathered keys past the streaming
            # threshold, a selection mask over the full latent below it.
            latent_d = _dequantized(latent_all, kv_cache)
            S = latent_d.shape[2]
            if (S > _STREAM_MIN_KEYS and B == 1 and _SPARSE_GATHER
                    and isinstance(offset, int)):
                q_n = self.embed_q(q)
                out = self._gathered_sparse_prefill(
                    q_n, latent_d, sel_pools, offset, L, S)
                out = self.unembed_out(out)
            elif S > _STREAM_MIN_KEYS:
                smask = self._sparse_mask(sel_pools, offset, L, S)[:, None]
                q_n = self.embed_q(q)
                out = _streamed_absorbed_attention(
                    q_n, latent_d, smask, self.scale)
                out = self.unembed_out(out)
            elif _ABSORBED_PREFILL:
                smask = self._sparse_mask(sel_pools, offset, L, S)[:, None]
                q_n = self.embed_q(q)
                out = scaled_dot_product_attention(
                    q_n, latent_d, latent_d, cache=None, scale=self.scale,
                    mask=smask)
                out = self.unembed_out(out)
            else:
                smask = self._sparse_mask(sel_pools, offset, L, S)[:, None]
                k = self.embed_q(latent_d, transpose=False)
                v = self.unembed_out(latent_d)
                out = scaled_dot_product_attention(
                    q, k, v, cache=None, scale=self.scale, mask=smask)
        elif L == 1:
            # Dense absorbed (MQA over the latent; V is the same rows).
            q_n = self.embed_q(q)
            out = scaled_dot_product_attention(
                q_n, latent_all, latent_all, cache=kv_cache, scale=self.scale,
                mask=mask)
            out = self.unembed_out(out)
        elif (offset + L > _STREAM_MIN_KEYS
              and isinstance(offset, int) and not isinstance(mask, str)):
            # Deep dense prefill chunk: streamed absorbed attention.
            latent_d = _dequantized(latent_all, kv_cache)
            q_n = self.embed_q(q)
            out = _streamed_absorbed_attention(
                q_n, latent_d, mask[:, None] if (
                    mask is not None and mask.ndim == 3) else mask,
                self.scale)
            out = self.unembed_out(out)
        elif _ABSORBED_PREFILL:
            q_n = self.embed_q(q)
            out = scaled_dot_product_attention(
                q_n, latent_all, latent_all, cache=kv_cache, scale=self.scale,
                mask=mask)
            out = self.unembed_out(out)
        else:
            # Dense naive prefill: expand the latent to per-head K/V. Pass
            # the causal mask straight through - with no rope half there is
            # no pe_scores term to fold it into.
            latent_d = _dequantized(latent_all, kv_cache)
            k = self.embed_q(latent_d, transpose=False)
            v = self.unembed_out(latent_d)
            out = scaled_dot_product_attention(
                q, k, v, cache=None, scale=self.scale, mask=mask)

        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class Glm5NextDeltaAttention(nn.Module):
    """KDA with the low-rank g_a/g_b output gate; recurrence via mlx-lm
    gated_delta (vectorized per-key-channel decay, g: [B, T, H, Dk])."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.head_dim = args.kda_head_dim
        self.conv_kernel = args.ssm_conv_kernel
        self.gate_lower_bound = args.kda_gate_lower_bound
        self.projection_dim = self.num_heads * self.head_dim
        self.scale = float(self.head_dim) ** -0.5
        hidden = args.hidden_size

        self.q_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.q_conv = ShortConv1d(self.projection_dim, self.conv_kernel)
        self.k_conv = ShortConv1d(self.projection_dim, self.conv_kernel)
        self.v_conv = ShortConv1d(self.projection_dim, self.conv_kernel)

        self.f_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, self.projection_dim, bias=False)
        self.b_proj = nn.Linear(hidden, self.num_heads, bias=False)
        # Wire ssm_a: the folded -exp(A_log), [num_heads], fp32. Kept folded.
        self.a_folded = -mx.ones((self.num_heads,))
        self.dt_bias = mx.zeros((self.projection_dim,))
        # Low-rank sigmoid output gate (kimi_k3 has a full-rank g_proj).
        self.g_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, self.projection_dim, bias=False)
        self.o_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.o_proj = nn.Linear(self.projection_dim, hidden, bias=False)

        # The metal kernel needs Dk % 32 == 0; fall back to ops otherwise.
        self._can_kernel = (self.head_dim % 32 == 0) and mx.metal.is_available()

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        gdn_sink: Optional[list] = None,
    ) -> mx.array:
        B, T, _ = x.shape
        dtype = x.dtype

        if cache is not None:
            q_state, k_state, v_state, ssm_state = cache
            lengths = cache.lengths
        else:
            q_state = k_state = v_state = ssm_state = None
            lengths = None

        if gdn_sink is not None and cache is not None:
            # MTP verify forward: record the pre-update recurrent state (the
            # slot arrays are immutable; the writes below replace them) plus
            # this call's inputs so rollback_verify_sink can replay the
            # accepted prefix from the pre-state.
            gdn_sink.append({
                "layer": self, "cache": cache,
                "pre": (q_state, k_state, v_state, ssm_state),
                "inputs": x, "mask": mask,
            })

        q_conv, q_state = self.q_conv(self.q_proj(x), q_state, mask, lengths)
        k_conv, k_state = self.k_conv(self.k_proj(x), k_state, mask, lengths)
        v_conv, v_state = self.v_conv(self.v_proj(x), v_state, mask, lengths)
        if cache is not None:
            cache[0] = q_state
            cache[1] = k_state
            cache[2] = v_state

        q = q_conv.reshape(B, T, self.num_heads, self.head_dim)
        k = k_conv.reshape(B, T, self.num_heads, self.head_dim)
        v = v_conv.reshape(B, T, self.num_heads, self.head_dim)

        # l2-norm with the attention scale folded in (kimi_linear convention:
        # l2norm(x) = rms_norm(x)/sqrt(d), q additionally carries 1/sqrt(d)).
        q = (self.scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = self.scale * mx.fast.rms_norm(k, None, 1e-6)

        # Decay: exp(lb * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))),
        # per key channel. f/g/beta read the layer input, not the conv out.
        a_raw = self.f_b_proj(self.f_a_proj(x)).reshape(
            B, T, self.num_heads, self.head_dim)
        dt = self.dt_bias.reshape(self.num_heads, self.head_dim)
        g = _kda_decay_lb(self.a_folded, a_raw, dt, self.gate_lower_bound)
        beta = mx.sigmoid(self.b_proj(x).reshape(B, T, self.num_heads))

        if ssm_state is None:
            ssm_state = mx.zeros(
                (B, self.num_heads, self.head_dim, self.head_dim),
                dtype=mx.float32)

        if self._can_kernel and mx.default_device() == mx.gpu and not self.training:
            out, ssm_state = gated_delta_kernel(q, k, v, g, beta, ssm_state, mask)
        else:
            out, ssm_state = gated_delta_ops(q, k, v, g, beta, ssm_state, mask)

        if cache is not None:
            cache[3] = ssm_state
            cache.advance(T)

        gate = self.g_b_proj(self.g_a_proj(x)).reshape(
            B, T, self.num_heads, self.head_dim)
        # RMS over head_dim with one shared weight, then a plain sigmoid
        # gate (not a silu-gated norm).
        out = (self.o_norm(out.reshape(B, T, self.num_heads, self.head_dim))
               * mx.sigmoid(gate)).reshape(B, T, -1)
        return self.o_proj(out.astype(dtype))


def rollback_verify_sink(sink: list, n: int) -> None:
    """Rewind the KDA caches after an MTP verify forward over S positions
    to the state after its first ``n`` (the accepted prefix): restore the
    recorded pre-verify conv tails + recurrent state, then replay the layer
    over the accepted prefix. O(n <= block) per layer; the KV/pool leaves
    are trimmed by the caller."""
    for e in sink:
        cache = e["cache"]
        for i, v in enumerate(e["pre"]):
            cache[i] = v
        mask = e["mask"]
        if isinstance(mask, mx.array):
            mask = mask[:, :n]
        e["layer"](e["inputs"][:, :n], mask=mask, cache=cache)


class Glm5NextDecoderLayer(nn.Module):
    """Hyper-connected block: hc_pre -> norm -> sublayer -> hc_expand, twice.
    The eager and fused-M=1 routes mirror deepseek_v4's block and must stay
    in lockstep (the fused front folds the sublayer RMSNorm)."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.is_linear = args.layer_types[layer_idx] == "linear_attention"
        if self.is_linear:
            self.self_attn = Glm5NextDeltaAttention(args)
        else:
            self.self_attn = Glm5NextMLAAttention(args)

        if layer_idx < args.first_k_dense_replace or args.n_routed_experts == 0:
            self.mlp = Glm5NextMLP(args)
        else:
            self.mlp = Glm5NextMoE(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps)
        self.attn_hc = HyperConnection(args)
        self.ffn_hc = HyperConnection(args)

    def __call__(
        self,
        h: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any],
        carry: Optional[tuple] = None,
        carry_mode: bool = False,
        gdn_sink: Optional[list] = None,
    ):
        akw = {"gdn_sink": gdn_sink} if (
            gdn_sink is not None and self.is_linear) else {}
        if self.attn_hc.m1_fused_ok(h):
            # attn front; a pending expand from the caller folds into it
            if carry is not None:
                h, front = self.attn_hc.fused_m1_expand(
                    carry, self.input_layernorm.weight)
            else:
                front = self.attn_hc.fused_m1(h, self.input_layernorm.weight)
            x, post, comb = front
            x = self.self_attn(x, mask=mask, cache=cache, **akw)

            # ffn front always consumes the attn expand inline
            h, front = self.ffn_hc.fused_m1_expand(
                (x, h, post, comb), self.post_attention_layernorm.weight)
            x, post, comb = front
            x = self.mlp(x)
            if carry_mode:
                return h, (x, h, post, comb)
            return hc_expand_m1(x, h, post, comb)

        if carry is not None:
            h = hc_expand_m1(*carry)
        residual = h
        x, post, comb = self.attn_hc(h)
        x = self.self_attn(self.input_layernorm(x), mask=mask, cache=cache,
                           **akw)
        residual, x, post, comb = hc_expand_collapse(
            self.ffn_hc, x, residual, post, comb)
        x = self.mlp(self.post_attention_layernorm(x))
        out = hc_expand(x, residual, post, comb)
        if carry_mode:
            return out, None
        return out


class Glm5NextModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [Glm5NextDecoderLayer(args, i)
                       for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        types = args.layer_types
        self.ssm_idx = types.index("linear_attention")
        self.attn_idx = types.index("full_attention")

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
        return_raw_hidden: bool = False,
        gdn_sink: Optional[list] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)

        attn_cache = cache[self.attn_idx]
        if attn_cache is not None:
            attn_cache = attn_cache[0]  # CacheList -> latent KVCache
        ssm_mask = create_ssm_mask(h, cache[self.ssm_idx])
        attn_mask = create_attention_mask(h, attn_cache, return_array=True)

        # hc_mult exact copies of the embedding: no scaling, no one-hot.
        h = mx.broadcast_to(
            h[:, :, None, :],
            (h.shape[0], h.shape[1], self.args.hc_mult, h.shape[2]))
        h = mx.contiguous(h)

        carry = None
        for layer, layer_cache in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else attn_mask
            h, carry = layer(h, mask, layer_cache, carry=carry,
                             carry_mode=True, gdn_sink=gdn_sink)
        if carry is not None:
            h = hc_expand_m1(*carry)

        # Final collapse: unweighted mean over the streams (no learned head).
        collapsed = h.astype(mx.float32).mean(axis=2).astype(h.dtype)
        if return_raw_hidden:
            return self.norm(collapsed), collapsed
        return self.norm(collapsed)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Glm5NextModel(args)
        if args.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size,
                                     bias=False)

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
        # Plain caches take the consuming stack's class identities (mlx-vlm
        # vs mlx-lm); PoolingCache stays the deepseek_v4 class every APC
        # seam isinstance-keys on. Heterogeneous list => the hybrid cache is
        # non-trimmable and chat re-prefills on partial prefix reuse.
        from gmlx.cache.compat import construction_cache_module

        cmod = construction_cache_module()
        arrays_cls = getattr(cmod, "ArraysCache", ArraysCache)
        kv_cls = getattr(cmod, "KVCache", KVCache)
        list_cls = getattr(cmod, "CacheList", None)
        if list_cls is None:
            from mlx_lm.models.cache import CacheList as list_cls

        caches = []
        for layer in self.layers:
            if layer.is_linear:
                # slots 0..2 = q/k/v conv tails, 3 = fp32 recurrent state.
                caches.append(arrays_cls(size=4))
            else:
                # Indexer pool: read in full at every scoring step, so it
                # opts out of --kv-bits packing (deepseek_v4 precedent).
                pool = PoolingCache(self.args.index_kpool, lookback=False)
                pool.quantizable = False
                caches.append(list_cls(kv_cls(), pool))
        return caches

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        # The GGUF's MTP block (blk.45) remaps onto model.layers.45.*, one
        # past the trunk; the MTP drafter loads those tensors separately.
        # Also covers HF-style leftovers on the text-only path.
        n = self.args.num_hidden_layers

        def _mtp_key(k: str) -> bool:
            if not k.startswith("model.layers."):
                return False
            idx = k.split(".", 3)[2]
            return idx.isdigit() and int(idx) >= n

        weights = {k: v for k, v in weights.items()
                   if not _mtp_key(k)
                   and not k.startswith(("vision_tower.", "mm_projector."))}
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    @property
    def cast_predicate(self):
        def predicate(path: str):
            if "e_score_correction_bias" in path:
                return False
            if path.endswith(("a_folded", "dt_bias")):
                return False
            # Hyper-connection params and the router weight route fp32.
            if ".attn_hc." in path or ".ffn_hc." in path:
                return False
            if path.endswith("mlp.gate.weight"):
                return False
            # Indexer fp32 pins: sign-free head weights (near-tie flips)
            # and the additive positional table.
            if path.endswith("indexer.weights_proj.weight"):
                return False
            if path.endswith("compressor.ape"):
                return False
            return True

        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            # Module paths (nn.quantize visits Linear/Embedding modules;
            # raw-array leaves like ape/gate.weight are never visited).
            # weights_proj stays float: its fp32 GEMM decides near-tied
            # pool rankings. The indexer projections cap at 8-bit g64.
            if path.endswith("indexer.weights_proj"):
                return False
            if path.endswith(("indexer.wq_b", "indexer.wk",
                              "compressor.wgate")):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate
