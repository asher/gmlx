# Copyright (c) 2026 Apple Inc.
#
"""MiniMax-M3 text backbone, vendored from mlx-lm PR #1401 (unmerged).

mlx-lm (0.31.3) ships no `models/minimax_m3.py`, so gmlx carries the
class and grafts it into the `mlx_lm.models` namespace at load time via
``ensure_registered()`` (importlib consults ``sys.modules`` first, so
``mlx_lm.utils._get_classes`` and every downstream path resolve it). The
registration is upstream-first: once an installed mlx-lm provides
``mlx_lm.models.minimax_m3``, that module wins and this copy is dead code -
delete the file and the loader hook. NOTE: the MSA implementation below is
a gmlx extension beyond the PR body; upstream adoption must carry it too.

Source: https://github.com/ml-explore/mlx-lm/pull/1401 for the dense
backbone; the MiniMax Sparse Attention (MSA) path follows llama.cpp
PR #24908 and the transformers reference (modular_minimax_m3_vl).

M3 extends MiniMax-M2 with: Gemma-style RMSNorm (scale by 1+w, fp32),
per-head QK-norm, partial RoPE (rotary_dim < head_dim), SwiGLU-OAI activation
(clamped gate/up with an (up+1) term), a shared expert + routed_scaling_factor
in the MoE, and the first few layers being dense MLPs instead of MoE.

MiniMax Sparse Attention (MSA): on every non-dense-lead layer a lightweight
indexer (one head per GQA group) scores the visible causal context, the
scores are max-pooled into ``sparse_block_size`` (128) token key blocks, and
the top ``sparse_topk_blocks`` (16) blocks - the query's local block always
force-included - are selected per (query, GQA group). Main attention then
runs only over the selected blocks (16*128 = 2048 KV per query). MSA is not
an optional speed feature: the model is trained with it, and dense attention
is an out-of-distribution approximation that degrades output beyond
~topk*block tokens (observed as endless reasoning loops). For sequences with
at most ``sparse_topk_blocks`` key blocks every block is selected, so MSA
reduces to exact dense attention there (and this module short-circuits to
the dense path in that regime).

The indexer runs only when the GGUF (or an indexer sidecar - see
loader.load_model) carries the indexer tensors and the cache is an
``MSAKVCache`` (which additionally stores the indexer key stream). Anything
else - indexless GGUFs, batch-merged or quantized caches - falls back to
dense attention with a one-time quality warning, mirroring llama.cpp's
fallback. ``GMLX_MSA_DISABLE=1`` forces dense for A/B comparison.
"""

import importlib
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models import cache as _lm_cache
from mlx_lm.models.cache import KVCache
from mlx_lm.models.switch_layers import SwitchGLU


def ensure_registered() -> None:
    """Make ``import mlx_lm.models.minimax_m3`` resolve, preferring upstream."""
    if "mlx_lm.models.minimax_m3" not in sys.modules:
        try:
            importlib.import_module("mlx_lm.models.minimax_m3")  # upstream wins
        except ImportError:
            sys.modules["mlx_lm.models.minimax_m3"] = sys.modules[__name__]
    # Snapshot restore resolves cache classes by name inside mlx_lm.models.cache;
    # graft the MSA cache alongside so saved M3 prompt caches round-trip.
    if not hasattr(_lm_cache, "MSAKVCache"):
        _lm_cache.MSAKVCache = MSAKVCache


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    intermediate_size: int
    dense_intermediate_size: int
    shared_intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    num_local_experts: int
    num_experts_per_tok: int
    rms_norm_eps: float
    rope_theta: float
    rotary_dim: int
    vocab_size: int
    head_dim: int = 128
    max_position_embeddings: int = 1048576
    routed_scaling_factor: float = 2.0
    swiglu_alpha: float = 1.702
    swiglu_limit: float = 7.0
    scoring_func: str = "sigmoid"
    use_qk_norm: bool = True
    tie_word_embeddings: bool = False
    # Per-layer MLP dispatch: "sparse" -> MoE block, "dense" -> dense MLP.
    mlp_layer_types: Optional[List[str]] = None
    # MiniMax Sparse Attention. Enabled by config synth when the GGUF (or a
    # sidecar) carries the indexer tensors; the MoE layers then build the
    # indexer branch (dense-lead layers never carry one).
    use_sparse_attention: bool = False
    sparse_index_dim: int = 128
    sparse_num_index_heads: int = 4
    sparse_topk_blocks: int = 16
    sparse_block_size: int = 128
    sparse_local_block: int = 1


class GemmaRMSNorm(nn.Module):
    """Gemma-style RMSNorm: normalize in fp32 and scale by ``weight + 1``."""

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.zeros((dims,))
        self.eps = eps

    def _extra_repr(self):
        return f"{self.weight.shape[0]}, eps={self.eps}"

    def __call__(self, x):
        ot = x.dtype
        x = x.astype(mx.float32)
        x = x * mx.rsqrt(x.square().mean(-1, keepdims=True) + self.eps)
        return (x * (1.0 + self.weight.astype(mx.float32))).astype(ot)


def swiglu_oai(x_gate, x_up, alpha: float, limit: float):
    """GPT-OSS / MiniMax-M3 clamped SwiGLU: (clamp(up)+1) * gate*sigmoid(alpha*gate)."""
    gate = mx.minimum(x_gate, limit)
    up = mx.clip(x_up, -limit, limit)
    return (up + 1.0) * (gate * mx.sigmoid(gate * alpha))


class SwiGLUOAI(nn.Module):
    """Activation callable for SwitchGLU: receives (x_up, x_gate)."""

    def __init__(self, alpha: float, limit: float):
        super().__init__()
        self.alpha = alpha
        self.limit = limit

    def __call__(self, x_up, x_gate):
        return swiglu_oai(x_gate, x_up, self.alpha, self.limit)


class MiniMaxM3MLP(nn.Module):
    """Dense SwiGLU-OAI MLP (used by the first dense layers and the shared expert)."""

    def __init__(self, args: ModelArgs, intermediate_size: int):
        super().__init__()
        self.alpha = args.swiglu_alpha
        self.limit = args.swiglu_limit
        self.gate_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(
            swiglu_oai(self.gate_proj(x), self.up_proj(x), self.alpha, self.limit)
        )


class MSAKVCache(KVCache):
    """KVCache extended with the MSA indexer key stream.

    ``ik`` holds one f32 vector of ``sparse_index_dim`` per cached token
    (single indexer key head, shared by all groups), appended in lockstep with
    K/V through ``update_and_fetch_msa`` so ``offset`` covers all three
    streams; ``trim`` therefore trims them together. Quantizing would drop the
    stream mid-generation, so ``to_quantized`` is refused and the loader is
    expected to drop ``--kv-bits`` up front (``kv_quant_unsupported``).
    """

    kv_quant_unsupported = True

    def __init__(self):
        super().__init__()
        self.ik = None  # [B, 1, cap, index_dim]

    def update_and_fetch_msa(self, keys, values, ik):
        prev = self.offset
        k, v = super().update_and_fetch(keys, values)  # advances offset
        n = ik.shape[2]
        if self.ik is None or (prev + n) > self.ik.shape[2]:
            B, _, _, idx_dim = ik.shape
            n_steps = (self.step + n - 1) // self.step
            new = mx.zeros((B, 1, n_steps * self.step, idx_dim), ik.dtype)
            if self.ik is not None:
                if prev % self.step != 0:
                    self.ik = self.ik[..., :prev, :]
                self.ik = mx.concatenate([self.ik, new], axis=2)
            else:
                self.ik = new
        self.ik[..., prev : prev + n, :] = ik
        return k, v, self.ik[..., : self.offset, :]

    @property
    def state(self):
        if self.offset == self.keys.shape[2]:
            return self.keys, self.values, self.ik
        return (
            self.keys[..., : self.offset, :],
            self.values[..., : self.offset, :],
            self.ik[..., : self.offset, :],
        )

    @state.setter
    def state(self, v):
        self.keys, self.values, self.ik = v
        self.offset = self.keys.shape[2]

    def to_quantized(self, group_size: int = 64, bits: int = 4):
        raise NotImplementedError(
            "MSAKVCache cannot quantize: the MSA indexer key stream has no "
            "quantized form (drop --kv-bits for MiniMax-M3 with MSA)"
        )

    @property
    def nbytes(self):
        n = super().nbytes
        return n if self.ik is None else n + self.ik.nbytes


def _warn_once(key: str, msg: str, _seen=set()):
    if key not in _seen:
        _seen.add(key)
        print(f"[minimax-m3] {msg}", file=sys.stderr)


def _msa_disabled() -> bool:
    return os.environ.get("GMLX_MSA_DISABLE", "") == "1"


class MiniMaxM3Attention(nn.Module):
    def __init__(self, args: ModelArgs, is_sparse: bool = False):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size, self.num_attention_heads * head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * head_dim, args.hidden_size, bias=False
        )

        # M3 uses per-head Gemma QK-norm over the head dimension.
        self.q_norm = GemmaRMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(head_dim, eps=args.rms_norm_eps)

        self.rope = nn.RoPE(args.rotary_dim, traditional=False, base=args.rope_theta)

        # MSA indexer branch (MoE layers only; dense-lead layers carry none).
        self.msa = args.use_sparse_attention and is_sparse and not _msa_disabled()
        if self.msa:
            self.idx_dim = args.sparse_index_dim
            self.n_idx_heads = args.sparse_num_index_heads
            self.topk_blocks = args.sparse_topk_blocks
            self.block_size = args.sparse_block_size
            self.local_blocks = args.sparse_local_block
            if self.n_idx_heads != self.num_key_value_heads:
                raise ValueError(
                    "MSA expects one indexer head per GQA group "
                    f"(got {self.n_idx_heads} indexer heads, "
                    f"{self.num_key_value_heads} KV heads)"
                )
            self.index_q_proj = nn.Linear(
                args.hidden_size, self.n_idx_heads * self.idx_dim, bias=False
            )
            self.index_k_proj = nn.Linear(args.hidden_size, self.idx_dim, bias=False)
            self.index_q_norm = GemmaRMSNorm(self.idx_dim, eps=args.rms_norm_eps)
            self.index_k_norm = GemmaRMSNorm(self.idx_dim, eps=args.rms_norm_eps)

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.num_attention_heads, self.head_dim)
        keys = self.k_proj(x).reshape(B, L, self.num_key_value_heads, self.head_dim)
        values = self.v_proj(x).reshape(B, L, self.num_key_value_heads, self.head_dim)

        # Per-head QK-norm over the head dim, before transpose / RoPE.
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if self.msa and isinstance(cache, MSAKVCache):
            return self._msa_attention(x, queries, keys, values, cache)
        if self.msa and cache is not None:
            _warn_once(
                "cache",
                f"cache {type(cache).__name__} cannot carry the MSA indexer "
                "stream; running DENSE attention (output may degrade)",
            )

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)

    # -- MSA ------------------------------------------------------------------

    def _msa_attention(self, x, queries, keys, values, cache):
        """Sparse attention over the top-k selected key blocks.

        Selection follows llama.cpp PR #24908 / the MSA paper: per GQA group,
        block score = maxpool_block(idx_q @ idx_k^T + causal mask), a large
        positive bias force-includes each query's ``local_blocks`` trailing
        blocks, and the top ``topk_blocks`` blocks are selected per (query,
        group). Scores are unscaled - only the ranking matters. Decode gathers
        exactly topk*block KV rows per group; prefill runs masked dense
        attention per group (correctness-first, like the reference).
        """
        B, _, L, D = queries.shape
        offset = cache.offset  # before this chunk lands

        # Indexer branch: project, per-head norm, partial RoPE, cache. The
        # whole branch runs f32 (matching the reference's F32 indexer ops).
        iq = self.index_q_proj(x).reshape(B, L, self.n_idx_heads, self.idx_dim)
        ik = self.index_k_proj(x).reshape(B, L, 1, self.idx_dim)
        iq = self.index_q_norm(iq).transpose(0, 2, 1, 3).astype(mx.float32)
        ik = self.index_k_norm(ik).transpose(0, 2, 1, 3).astype(mx.float32)
        iq = self.rope(iq, offset=offset)
        ik = self.rope(ik, offset=offset)

        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)
        K, V, IK = cache.update_and_fetch_msa(keys, values, ik)
        S = K.shape[2]

        blk = self.block_size
        nblk = (S + blk - 1) // blk
        if nblk <= self.topk_blocks:
            # Every block is selectable -> selection keeps everything and MSA
            # is exactly dense causal attention. Run the cheap form.
            out = self._dense_sdpa(queries, K, V, offset, L, S)
        elif L == 1:
            out = self._msa_decode(queries, K, V, IK, iq, S, nblk)
        else:
            out = self._msa_prefill(queries, K, V, IK, iq, offset, L, S, nblk)

        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)

    def _dense_sdpa(self, queries, K, V, offset, L, S):
        mask = None
        if L > 1:
            mask = self._causal(offset, L, S).astype(queries.dtype)[None, None]
        return mx.fast.scaled_dot_product_attention(
            queries, K, V, scale=self.scale, mask=mask
        )

    @staticmethod
    def _causal(offset, L, S):
        """Additive causal mask [L, S] in f32 for queries at offset..offset+L-1."""
        q_pos = mx.arange(offset, offset + L)
        k_pos = mx.arange(S)
        return mx.where(k_pos[None, :] <= q_pos[:, None], 0.0, -float("inf")).astype(
            mx.float32
        )

    def _block_select(self, scores, q_pos, nblk):
        """Top-k block indices from causal-masked scores.

        ``scores``: [..., Lq, S] f32, -inf on invisible keys. Returns int
        indices [..., Lq, K]. The trailing ``local_blocks`` blocks of each
        query are force-included via a +1e30 bias (selection only - the
        token-level causal mask still applies inside them).
        """
        blk = self.block_size
        S = scores.shape[-1]
        pad = nblk * blk - S
        if pad:
            scores = mx.pad(
                scores,
                [(0, 0)] * (scores.ndim - 1) + [(0, pad)],
                constant_values=-float("inf"),
            )
        bs = scores.reshape(*scores.shape[:-1], nblk, blk).max(axis=-1)
        own = q_pos // blk  # [Lq]
        blk_idx = mx.arange(nblk)
        local = (blk_idx[None, :] <= own[:, None]) & (
            blk_idx[None, :] > (own[:, None] - self.local_blocks)
        )
        bs = bs + local.astype(mx.float32) * 1e30
        k = self.topk_blocks
        return mx.argpartition(-bs, kth=k - 1, axis=-1)[..., :k]

    def _msa_decode(self, queries, K, V, IK, iq, S, nblk):
        """Single-token step: batched top-k + gather, one grouped-GQA SDPA."""
        B = queries.shape[0]
        blk, k = self.block_size, self.topk_blocks
        Hkv = self.num_key_value_heads
        gq = self.num_attention_heads // Hkv

        # Scores [B, Hkv, 1, S]: every cached key is visible at decode.
        sc = mx.matmul(iq, IK.swapaxes(-1, -2))
        idx = self._block_select(sc, mx.array([S - 1]), nblk)  # [B, Hkv, 1, k]

        # Expand blocks to token rows; rows past S are gathered clipped and
        # masked out (the partial tail block - the nblk<=k shortcut already
        # excludes the fewer-blocks-than-topk regime).
        tok = (idx * blk)[..., None] + mx.arange(blk)  # [B, Hkv, 1, k, blk]
        tok = tok.reshape(B, Hkv, k * blk)
        tok_mask = mx.where(tok < S, 0.0, -float("inf"))  # [B, Hkv, k*blk]
        tok = mx.minimum(tok, S - 1)

        Kg = mx.take_along_axis(K, tok[..., None], axis=2)  # [B, Hkv, k*blk, D]
        Vg = mx.take_along_axis(V, tok[..., None], axis=2)

        # One native-GQA call: q heads are group-major, so repeating the
        # per-group mask across each group's heads lines up with head h
        # reading KV head h // gq.
        mask = mx.repeat(tok_mask[:, :, None, None, :], gq, axis=1)
        mask = mask.reshape(B, Hkv * gq, 1, k * blk).astype(queries.dtype)
        return mx.fast.scaled_dot_product_attention(
            queries, Kg, Vg, scale=self.scale, mask=mask
        )

    def _msa_prefill(self, queries, K, V, IK, iq, offset, L, S, nblk):
        """Chunk of L queries: per-group masked dense attention over the full
        cache with non-selected blocks -inf'd out (correctness-first, mirrors
        llama.cpp's batch regime). Queries are tiled to bound the transient
        f32 score/mask buffers at depth."""
        B = queries.shape[0]
        blk = self.block_size
        Hkv = self.num_key_value_heads
        gq = self.num_attention_heads // Hkv
        tile = int(os.environ.get("GMLX_MSA_PREFILL_TILE", "2048"))

        outs = []
        for t0 in range(0, L, tile):
            t1 = min(t0 + tile, L)
            Lt = t1 - t0
            q_pos = mx.arange(offset + t0, offset + t1)
            causal = self._causal(offset + t0, Lt, S)  # [Lt, S] f32

            group_outs = []
            for g in range(Hkv):
                # Scores for this group [B, 1, Lt, S] f32 (+ causal).
                sc = mx.matmul(iq[:, g : g + 1, t0:t1], IK.swapaxes(-1, -2))
                sc = sc + causal[None, None]
                idx = self._block_select(sc, q_pos, nblk)  # [B, 1, Lt, k]

                # Block keep-mask -> token level, then combine with causal.
                keep = mx.put_along_axis(
                    mx.full((B, 1, Lt, nblk), -float("inf"), dtype=mx.float32),
                    idx,
                    mx.zeros_like(idx).astype(mx.float32),
                    axis=-1,
                )
                keep = mx.repeat(keep[..., None], blk, axis=-1)
                keep = keep.reshape(B, 1, Lt, nblk * blk)[..., :S]
                gmask = (keep + causal[None, None]).astype(queries.dtype)

                o = mx.fast.scaled_dot_product_attention(
                    queries[:, g * gq : (g + 1) * gq, t0:t1],
                    K[:, g : g + 1],
                    V[:, g : g + 1],
                    scale=self.scale,
                    mask=gmask,
                )
                group_outs.append(o)
            outs.append(mx.concatenate(group_outs, axis=1))
        return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)


_MOE_MIX_SCORES = os.environ.get("GMLX_M3_MOE_MIX", "1") != "0"


class MiniMaxM3SparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok
        self.routed_scaling_factor = args.routed_scaling_factor

        self.gate = nn.Linear(args.hidden_size, args.num_local_experts, bias=False)
        self.e_score_correction_bias = mx.zeros((args.num_local_experts,))
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.intermediate_size,
            args.num_local_experts,
            activation=SwiGLUOAI(args.swiglu_alpha, args.swiglu_limit),
        )
        self.shared_experts = MiniMaxM3MLP(args, args.shared_intermediate_size)

    def __call__(self, x):
        gates = self.gate(x.astype(mx.float32))
        scores = mx.sigmoid(gates)
        orig_scores = scores
        scores = scores + self.e_score_correction_bias

        k = self.num_experts_per_tok
        inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
        weights = mx.take_along_axis(orig_scores, inds, axis=-1)
        weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
        weights = (weights * self.routed_scaling_factor).astype(x.dtype)

        # Hand the normalized routing weights to the swapped module when
        # it accepts them (the mix seam is also what feeds miss-shed its
        # scores); an unmixed return keeps the stock python-side sum.
        if _MOE_MIX_SCORES and getattr(
                self.switch_mlp, "_kq_mix_scores", False):
            y = self.switch_mlp(x, inds, weights)
        else:
            y = self.switch_mlp(x, inds)
        if y.ndim == weights.ndim + 1:
            y = (y * weights[..., None]).sum(axis=-2)
        return y + self.shared_experts(x)


class MiniMaxM3DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.is_sparse = (args.mlp_layer_types or ["sparse"] * args.num_hidden_layers)[
            layer_idx
        ] == "sparse"
        self.self_attn = MiniMaxM3Attention(args, is_sparse=self.is_sparse)
        if self.is_sparse:
            self.block_sparse_moe = MiniMaxM3SparseMoeBlock(args)
        else:
            self.mlp = MiniMaxM3MLP(args, args.dense_intermediate_size)
        self.input_layernorm = GemmaRMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(self, x, mask=None, cache=None):
        r = x + self.self_attn(self.input_layernorm(x), mask, cache)
        mlp = self.block_sparse_moe if self.is_sparse else self.mlp
        return r + mlp(self.post_attention_layernorm(r))


class MiniMaxM3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            MiniMaxM3DecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.norm = GemmaRMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs, mask=None, cache=None):
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        if mask is None:
            mask = create_attention_mask(h, cache[0])
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MiniMaxM3Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, mask=None, cache=None):
        out = self.model(inputs, mask, cache)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def make_cache(self):
        return [
            MSAKVCache() if getattr(layer.self_attn, "msa", False) else KVCache()
            for layer in self.layers
        ]

    def sanitize(self, weights):
        skip_prefixes = (
            "vision_tower",
            "multi_modal_projector",
            "patch_merge_mlp",
            "model.vision_tower",
            "model.multi_modal_projector",
        )
        # Must track the attention modules' gate: under GMLX_MSA_DISABLE the
        # indexer submodules are never built, so their weights must drop too.
        keep_indexer = self.args.use_sparse_attention and not _msa_disabled()

        def keep(k):
            if k.startswith(skip_prefixes):
                return False
            if ".self_attn.index_" in k:  # MSA indexer - kept only when armed
                return keep_indexer
            if ".mtp." in k or k.startswith("mtp.") or "model.mtp" in k:
                return False
            return True

        def rename(k):
            if k.startswith("language_model.model."):
                return "model." + k[len("language_model.model.") :]
            if k.startswith("language_model.lm_head."):
                return "lm_head." + k[len("language_model.lm_head.") :]
            if k.startswith("language_model."):
                return k[len("language_model.") :]
            return k

        renamed = {}
        for k, v in weights.items():
            if not keep(k):
                continue
            if (
                ".self_attn.index_" in k
                and k.endswith("proj.weight")
                and v.dtype == mx.float32
            ):
                # Some MSA GGUFs store the indexer projections as F32; the
                # source checkpoint is BF16, so narrowing recovers the
                # original bits at half the memory (norms stay F32 - tiny,
                # and the norm math runs in F32 regardless).
                v = v.astype(mx.bfloat16)
            renamed[rename(k)] = v
        weights = renamed

        # Stack per-expert w1/w2/w3 into SwitchGLU's batched experts.
        if (
            "model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight"
            not in weights
        ):
            mapping = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
            for i in range(self.args.num_hidden_layers):
                prefix = f"model.layers.{i}.block_sparse_moe"
                if f"{prefix}.experts.0.w1.weight" not in weights:
                    continue
                for orig, new in mapping.items():
                    stacked = mx.stack(
                        [
                            weights.pop(f"{prefix}.experts.{e}.{orig}.weight")
                            for e in range(self.args.num_local_experts)
                        ]
                    )
                    weights[f"{prefix}.switch_mlp.{new}.weight"] = stacked

        return weights

    @property
    def layers(self):
        return self.model.layers

    @property
    def cast_predicate(self):
        # Keep the router correction bias in fp32.
        return lambda k: "e_score_correction_bias" not in k

    @property
    def quant_predicate(self):
        def predicate(path, _):
            # Routers stay high-precision (small, sensitive to quantization).
            if path.endswith("block_sparse_moe.gate"):
                return {"group_size": 64, "bits": 8}
            # Indexer projections drive block selection (a discrete retrieval
            # decision); quant error there changes which KV is read at all.
            if ".self_attn.index_" in path:
                return False
            return True

        return predicate
