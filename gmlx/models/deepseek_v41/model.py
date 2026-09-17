"""DeepSeek-V4.1-Flash.

The V4 skeleton with three changes, all derived from the reference
``inference/model.py`` and ``inference/engram.py``:

- Engram conditional memory on a few layers. A hash of the last
  ``max_ngram_size`` token ids picks 24 rows from a per-layer table; the
  rows become one key per residual stream plus a shared value, and a
  normalized stream-key dot product gates the value into the streams.
- The hyper-connection collapse lags one sublayer. Attention collapses
  with the previous FFN's pre-mix, the FFN with this attention's, and the
  final collapse uses the last layer's. There is no collapse head.
- CSA2 attention. Every layer keeps a sliding window; layers with a
  compress ratio also read a pooled-KV stream and a top-k selection that
  a few source layers produce and the layers after them reuse.

Everything else - the low-rank q/o projections, the sqrt-softplus MoE,
the Sinkhorn mixing, the pooled cache - is imported from
``gmlx.models.deepseek_v4``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    create_causal_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.mla import MultiLinear
from mlx_lm.models.pipeline import PipelineMixin

from gmlx.models.deepseek_v4 import model as _v4
from gmlx.models.deepseek_v4.cache import PoolingCache
from gmlx.models.deepseek_v4.hyper_connection import (
    HyperConnection,
    _hc_split_sinkhorn_ops,
    hc_expand,
    hc_expand_m1,
)
from gmlx.models.deepseek_v41.engram_codec import (
    decode_rows,
    parse_row_encoding,
    row_width,
)
from gmlx.models.deepseek_v41.engram_codec import tables as engram_tables

DeepseekV4RoPE = _v4.DeepseekV4RoPE
class DeepseekV41MoE(_v4.DeepseekV4MoE):
    """The V4 MoE, plus the level-2 decode profile marks."""

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self._li = layer_idx

    def __call__(self, x, input_ids, image_mask=None):
        if not _subprof_on(x.shape[1]) or self.sharding_group is not None:
            return super().__call__(x, input_ids, image_mask)
        import time

        t0 = time.perf_counter()
        inds, scores = self.gate(x, input_ids, image_mask)
        t0 = _prof_mark("f.route", self._li, (inds, scores), t0)
        y = self._routed(x, inds, scores)
        t0 = _prof_mark("f.exp", self._li, y, t0)
        y = self._with_shared(x, y, scores)
        _prof_mark("f.shexp", self._li, y, t0)
        return y


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "deepseek_v41"
    vocab_size: int = 129280
    hidden_size: int = 5120
    moe_intermediate_size: int = 2304
    num_hidden_layers: int = 40
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    n_shared_experts: int = 1
    n_routed_experts: int = 384
    routed_scaling_factor: float = 1.5
    q_lora_rank: int = 1280
    qk_rope_head_dim: int = 64
    num_experts_per_tok: int = 6
    norm_topk_prob: bool = True
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-20
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    attention_bias: bool = False
    head_dim: int = 512
    scoring_func: str = "sqrtsoftplus"
    compress_ratios: List[int] = field(default_factory=list)
    compress_rope_theta: float = 160000.0
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    num_hash_layers: int = 0
    swiglu_limit: float = 10.0
    sliding_window: int = 128
    o_groups: int = 8
    o_lora_rank: int = 1024
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512
    tie_word_embeddings: bool = False
    vision_router_bias: bool = False
    media_token_ids: List[int] = field(default_factory=list)
    # Shared-stream roles, derived at conversion from tensor presence.
    kv_source_layers: List[int] = field(default_factory=list)
    index_key_layers: List[int] = field(default_factory=list)
    index_source_layers: List[int] = field(default_factory=list)
    # Two-level candidate selection; only engages past
    # candidate_topk_blocks * candidate_block_size compressed positions.
    candidate_source_layer: int = -1
    candidate_topk_blocks: int = 0
    candidate_block_size: int = 0
    # Engram tables.
    engram_layer_ids: List[int] = field(default_factory=list)
    engram_table_rows: List[int] = field(default_factory=list)
    engram_max_ngram_size: int = 4
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    engram_pad_id: int = 2
    engram_multipliers: List[int] = field(default_factory=list)
    engram_primes: List[int] = field(default_factory=list)
    engram_offsets: List[int] = field(default_factory=list)
    engram_token_map: Optional[List[int]] = None
    # Row byte layout, when the conversion stores the tables as raw bytes
    # rather than a GGUF quant (see engram_codec).
    engram_row_encoding: Optional[str] = None
    # intermediate_size is unused: every V4.1 layer is MoE.
    intermediate_size: int = 0

    def __post_init__(self):
        n = self.num_hidden_layers
        if not self.compress_ratios:
            self.compress_ratios = [0] * n
        self.compress_ratios = list(self.compress_ratios[:n])
        if len(self.compress_ratios) != n:
            raise ValueError(
                "`compress_ratios` must have one entry per hidden layer, got "
                f"{len(self.compress_ratios)} for {n} layers."
            )
        bad = [r for r in self.compress_ratios if r not in (0, 1, 2)]
        if bad:
            raise ValueError(f"Unsupported DeepSeek-V4.1 compress ratios: {bad}")
        if not set(self.kv_source_layers) <= set(self.index_source_layers):
            raise ValueError(
                "every kv-source layer must also be an index source: it owns "
                "the index keys the later layers read"
            )
        if self.engram_layer_ids:
            n_t = len(self.engram_layer_ids)
            cols = (self.engram_max_ngram_size - 1) * self.engram_n_heads
            if len(self.engram_primes) != n_t * cols:
                raise ValueError(
                    f"{len(self.engram_primes)} engram primes for {n_t} tables "
                    f"x {cols} buckets"
                )
            if len(self.engram_table_rows) != n_t:
                raise ValueError(
                    "engram_table_rows needs one row count per engram layer"
                )


def _rms(x: mx.array, eps: float) -> mx.array:
    return mx.fast.rms_norm(x, None, eps)


# --- engram ----------------------------------------------------------------


class EngramHash(nn.Module):
    """Table row ids for the n-grams ending at each position.

    A position is hashed as ``max_ngram_size - 1`` n-grams, each split
    over ``n_heads`` heads; every (n-gram size, head) pair owns a disjoint
    prime-sized bucket range. Ids run through a compressed token map
    first, and a look-back that crosses the start of the sequence reads
    the pad id instead.
    """

    DEAD = -1

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.n_tables = len(config.engram_layer_ids)
        self.max_ngram = config.engram_max_ngram_size
        self.context_len = self.max_ngram - 1
        n_h = config.engram_n_heads
        n_ng = self.context_len
        if config.engram_token_map is None:
            raise ValueError(
                "deepseek_v41: engram_token_map is missing from the config; "
                "the hash cannot be computed without it"
            )
        self._token_map = mx.array(config.engram_token_map, dtype=mx.int64)
        self.pad_id = int(config.engram_token_map[config.engram_pad_id])
        self._multipliers = mx.array(
            config.engram_multipliers, dtype=mx.int64
        ).reshape(self.n_tables, self.max_ngram)
        self._primes = mx.array(config.engram_primes, dtype=mx.int64).reshape(
            self.n_tables, n_ng, n_h
        )
        self._offsets = mx.array(config.engram_offsets, dtype=mx.int64).reshape(
            self.n_tables, n_ng * n_h
        )

    def __call__(self, input_ids, history, token_mask=None):
        """Returns ``[B, L, n_tables, cols]`` row ids and the new history.

        ``token_mask`` is False on tokens that take no part in an n-gram
        (image spans); those are marked dead, and a look-back that reaches
        one reads the pad id from there on."""
        B, L = input_ids.shape
        compressed = self._token_map[input_ids.astype(mx.int64)]
        if token_mask is not None:
            compressed = mx.where(token_mask, compressed, self.DEAD)
        if history is None or history.shape[0] != B:
            history = mx.full((B, self.context_len), self.pad_id, dtype=mx.int64)
        full = mx.concatenate([history, compressed], axis=1)
        new_history = (
            mx.contiguous(full[:, -self.context_len:])
            if self.context_len
            else history
        )

        # Shift s reads the token s positions back. The pad-filled history
        # covers a look-back that crosses the start of the sequence; a dead
        # token blocks that shift and every longer one.
        blocked = mx.zeros((B, L), dtype=mx.bool_)
        tokens = []
        for s in range(self.max_ngram):
            src = full[:, self.context_len - s: self.context_len - s + L]
            blocked = mx.logical_or(blocked, src == self.DEAD)
            tokens.append(mx.where(blocked, self.pad_id, src))
        tokens = mx.stack(tokens, axis=-1)

        # XOR the multiplied ids one look-back at a time, so the running
        # value after step i hashes the (i+1)-gram; each lands in its own
        # prime-sized bucket range.
        products = tokens[:, :, None, :] * self._multipliers[None, None]
        rolling = products[..., 0]
        cols = []
        for i in range(1, self.max_ngram):
            rolling = mx.bitwise_xor(rolling, products[..., i])
            cols.append(rolling[..., None] % self._primes[:, i - 1])
        ids = mx.concatenate(cols, axis=-1) + self._offsets
        return ids, new_history


class EngramTable(nn.Embedding):
    """Row gather over one n-gram table.

    A conversion that stores rows as a GGUF quant leaves this a plain
    gather, which the loader swaps for the kquant one. A conversion that
    stores raw bytes names its layout instead, so the gather returns
    ``row_bytes`` of uint8 and the codec turns them into ``head_dim``
    values (reference ``ds4_engram_read``, which rounds to bfloat16).
    """

    def __init__(self, rows: int, head_dim: int, encoding: Optional[str] = None):
        codec = parse_row_encoding(encoding)
        super().__init__(max(int(rows), 1), row_width(head_dim, encoding))
        self.head_dim = int(head_dim)
        self._codec = codec
        if codec is not None:
            values, scales = engram_tables()
            self._value_lut = mx.array(values)
            self._scale_lut = mx.array(scales)

    def decode_gathered(self, raw: mx.array) -> mx.array:
        if self._codec is None:
            return raw
        return decode_rows(raw, self._codec, self.head_dim,
                           self._value_lut, self._scale_lut).astype(mx.bfloat16)

    def __call__(self, x: mx.array) -> mx.array:
        return self.decode_gathered(super().__call__(x))


class Engram(nn.Module):
    """One n-gram lookup written into the residual streams, gated by how
    well it matches them."""

    def __init__(self, config: ModelArgs, table_rows: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.eps = config.rms_norm_eps
        self.clamp_value = 1e-6
        cols = (config.engram_max_ngram_size - 1) * config.engram_n_heads
        self.embed = EngramTable(table_rows, config.engram_head_dim,
                                 config.engram_row_encoding)
        self.wkv = nn.Linear(
            cols * config.engram_head_dim,
            config.hidden_size * (config.hc_mult + 1),
            bias=False,
        )
        self.q_weight = mx.ones((config.hc_mult, config.hidden_size), dtype=mx.float32)
        self.k_weight = mx.ones((config.hc_mult, config.hidden_size), dtype=mx.float32)

    def __call__(self, h: mx.array, row_ids: mx.array,
                 token_mask: Optional[mx.array] = None) -> mx.array:
        B, L = row_ids.shape[0], row_ids.shape[1]
        rows = self.embed(row_ids)
        kv = self.wkv(rows.reshape(B, L, -1))
        split = self.hc_mult * self.hidden_size
        key = kv[..., :split].astype(mx.float32).reshape(
            B, L, self.hc_mult, self.hidden_size
        )
        value = kv[..., split:].astype(mx.float32)
        weight = self.q_weight.astype(mx.float32) * self.k_weight.astype(mx.float32)

        hf = h.astype(mx.float32)
        rstd = mx.rsqrt(mx.mean(mx.square(hf), axis=-1) + self.eps) * mx.rsqrt(
            mx.mean(mx.square(key), axis=-1) + self.eps
        )
        dot = (hf * weight * key).sum(axis=-1) * rstd * (self.hidden_size ** -0.5)
        # signed sqrt before the sigmoid, matching the training kernel;
        # copysign, so a +0 dot keeps the positive root
        mag = mx.sqrt(mx.maximum(mx.abs(dot), self.clamp_value))
        gate = mx.sigmoid(mx.where(dot < 0, -mag, mag))
        if token_mask is not None:
            gate = mx.where(token_mask[..., None], gate, mx.zeros_like(gate))
        return (hf + gate[..., None] * value[:, :, None, :]).astype(h.dtype)



# --- shared compressed and index streams ------------------------------------


class SharedStreams:
    """What a source layer hands to the layers after it.

    Built fresh for every forward. Layers run in order and every source
    writes before its readers read, so one slot each is enough.
    (reference SharedAttentionRuntime)
    """

    __slots__ = ("pooled", "pool_cache", "index_k", "topk", "candidates", "ratio")

    def __init__(self):
        self.pooled = None
        self.pool_cache = None
        self.index_k = None
        self.topk = None
        self.candidates = None
        self.ratio = 0


@partial(mx.compile, shapeless=True)
def _pool_group(kv: mx.array, gate: mx.array) -> mx.array:
    weights = mx.softmax(gate.astype(mx.float32), axis=-2).astype(kv.dtype)
    return (kv * weights).sum(axis=-2)


class Compressor(nn.Module):
    """Pools ``compress_ratio`` tokens into one KV latent, before RoPE.

    Ratio 1 is a plain projection with no gate. Above that a learned
    softmax weights the tokens in each group. The latent comes back
    unrotated because the index keys derive from that form.
    (reference Compressor.forward)
    """

    def __init__(self, config: ModelArgs, compress_ratio: int):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.head_dim = config.head_dim
        self.wkv = nn.Linear(config.hidden_size, config.head_dim, bias=False)
        if compress_ratio > 1:
            self.wgate = nn.Linear(config.hidden_size, config.head_dim, bias=False)
        self.norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, pool_cache: Optional[PoolingCache], offset):
        B = x.shape[0]
        kv = self.wkv(x)
        # Ratio 1 has no gate; kv stands in for it so the cache keeps the
        # window bookkeeping its trim path needs.
        gate = self.wgate(x) if self.compress_ratio > 1 else kv
        if pool_cache is None:
            usable = (kv.shape[1] // self.compress_ratio) * self.compress_ratio
            ready_kv, ready_gate, pool_base = kv[:, :usable], gate[:, :usable], offset
        else:
            ready_kv, ready_gate, pool_base = pool_cache.accumulate_windows(
                kv, gate, offset
            )
        if ready_kv.shape[1] == 0:
            return mx.zeros((B, 0, self.head_dim), dtype=x.dtype), pool_base
        if self.compress_ratio == 1:
            return self.norm(ready_kv), pool_base
        pooled = _pool_group(
            mx.unflatten(ready_kv, 1, (-1, self.compress_ratio)),
            mx.unflatten(ready_gate, 1, (-1, self.compress_ratio)),
        )
        return self.norm(pooled), pool_base


class Indexer(nn.Module):
    """Keeps the ``index_topk`` best compressed positions per query.

    An index-key owner also turns its layer's unrotated latent into the
    one shared key per compressed position that every later indexer
    scores against. (reference Indexer)
    """

    def __init__(self, config: ModelArgs, layer_idx: int, owns_k: bool):
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.scale = self.head_dim ** -0.5
        self.owns_k = owns_k
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        if owns_k:
            self.wk = nn.Linear(config.head_dim, self.head_dim, bias=False)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.is_candidate_source = layer_idx == config.candidate_source_layer
        self.uses_candidates = 0 <= config.candidate_source_layer < layer_idx
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size

    def make_keys(self, latent: mx.array, pool_rope, pool_base) -> mx.array:
        """Index keys for freshly pooled latents, rotated on the tail.
        Stored fp16: the kq scorers read fp16 operands (exact for the
        FP4-grid rows), so the cache needs no cast per step."""
        k = self.k_norm(self.wk(latent))
        k = pool_rope(k[:, None], offset=pool_base).squeeze(1)
        return _indexer_qat(k).astype(mx.float16)

    def __call__(self, x, q_residual, rope, index_k, pmask, offset, streams):
        B, L, _ = x.shape
        P = index_k.shape[1]
        k = min(self.index_topk, P)
        q = self.wq_b(q_residual).reshape(B, L, self.n_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)
        q = rope(q, offset)
        q = _indexer_qat(q)

        weights = _v4._skinny_linear(self.weights_proj, x) * (
            self.n_heads ** -0.5
        )
        scorer = _indexer_kernel_scorer(
            q, index_k, weights, self.scale, k,
            offset if pmask is not None else None, streams.ratio,
        )
        if scorer is None:
            q = q.astype(mx.float32)
            keys = index_k[:, None].swapaxes(-1, -2).astype(mx.float32)
            weights = weights.astype(mx.float32).swapaxes(-1, -2)[..., None]
            floor = mx.finfo(mx.float32).min
        else:
            floor = mx.finfo(mx.float16).min
        if pmask is not None and pmask.ndim == 2:
            pmask = pmask[None]
        source = self.is_candidate_source and self.candidate_block_size > 0
        given = (
            streams.candidates
            if self.uses_candidates and not source
            else None
        )
        if given is not None and given.shape[1] != L:
            # Prefill tail: this layer runs fewer rows than the source.
            given = given[:, given.shape[1] - L:]
        # Query blocks: a full-length pass holds an [heads, L, P] score
        # per layer, square in the prompt for the ratio-1 layers.
        block = _v4._prefill_block()
        block = L if not 0 < block < L else block
        tops, cands = [], []
        for qs in range(0, L, block):
            qe = min(L, qs + block)
            if scorer is not None:
                scores = scorer(qs, qe)
            else:
                scores = q[:, :, qs:qe] @ keys
                scores = mx.maximum(scores, 0) * self.scale
                scores = (scores * weights[:, :, qs:qe]).sum(axis=1)
            if pmask is not None:
                scores = mx.where(pmask[:, qs:qe], scores, floor)
            if source:
                cands.append(
                    _select_candidate_blocks(
                        scores, self.candidate_topk_blocks,
                        self.candidate_block_size, floor,
                    )
                )
            elif given is not None:
                scores = mx.where(given[:, qs:qe], scores, floor)
            if scorer is not None:
                tops.append(_indexer_kernel_topk(scores, k))
            else:
                tops.append(
                    mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
                )
        if source:
            streams.candidates = (
                None if any(c is None for c in cands)
                else cands[0] if len(cands) == 1
                else mx.concatenate(cands, axis=1)
            )
        return tops[0] if len(tops) == 1 else mx.concatenate(tops, axis=1)


def _indexer_kernel_scorer(q, index_k, weights, scale, k, offset=None, ratio=0):
    """Prefill scores through the kq indexer GEMM (dsa_indexer_scores): the
    relu, scale and head sum run in the kernel, so no [heads, L, P] score
    is materialized. fp16 operands hold the FP4-grid q and k exactly; the
    scale folds into the per-head weights. Keys pad once to the 64-row
    tile, a query block pads per call.

    ``offset`` (absolute position of query row 0) with the pool ``ratio``
    arms the kernel's causal tile skip. The caller's mask hides pooled row
    n from query row m once n >= (offset + m + 1) // ratio, and
    (offset + qs + 1) // ratio - 1 + m never falls under that bound, so
    the tiles past it hold only hidden scores and skip; the mask still
    decides the exact set. Pass ``offset`` only under that mask.

    On tensor-op hardware the int8 arm (dsa_indexer_scores_q) runs on the
    packed FP4 codes, bit-identical to the fp16 GEMM on the same rows.
    Returns ``score(qs, qe) -> [B, qe - qs, P]`` float16, or None to keep
    the inline fp32 path (decode widths, other geometries, no Metal,
    GMLX_DSA_INDEXER=0)."""
    B, H, L, D = q.shape
    P = index_k.shape[1]
    if L <= 4:
        return _indexer_decode_scorer(q, index_k, weights, scale, offset, ratio)
    if (
        H not in (32, 64)
        or D != 128
        or k not in (512, 2048)
        or not mx.metal.is_available()
        or mx.default_device() != mx.Device(mx.gpu)
        or not _v4._dsa_probe("indexer")
    ):
        return None
    import mlx_kquant as kq

    causal = isinstance(offset, int) and isinstance(ratio, int) and ratio >= 1
    q16 = q.astype(mx.float16)
    w16 = (weights * scale).astype(mx.float16)
    keys = index_k.astype(mx.float16)[:, None]
    pad_n = (-P) % 64
    if pad_n:
        keys = mx.concatenate(
            [keys, mx.zeros((B, 1, pad_n, D), dtype=mx.float16)], axis=2
        )
    packed = None
    if _qat_enabled() and _v4._dsa_probe("indexer_q"):
        try:
            packed = (kq.dsa_indexer_qat_pack(q16), kq.dsa_indexer_qat_pack(keys))
        except Exception as exc:  # noqa: BLE001 - permanent fallback
            _v4._dsa_disable("indexer_q", exc)

    def score(qs, qe):
        m = qe - qs
        pad_l = (-m) % 64
        wb = w16[:, qs:qe]
        if pad_l:
            wb = mx.concatenate(
                [wb, mx.zeros((B, pad_l, H), dtype=mx.float16)], axis=1
            )
        flags = (
            dict(
                causal=True,
                skip_causal_future_store=True,
                causal_q_offset=(offset + qs + 1) // ratio - 1,
            )
            if causal
            else dict(causal=False)
        )
        if packed is not None:
            (qc, qsc), (kc, ksc) = packed
            qc, qsc = qc[:, :, qs:qe], qsc[:, :, qs:qe]
            if pad_l:
                qc = mx.concatenate(
                    [qc, mx.zeros((B, H, pad_l, D), dtype=qc.dtype)], axis=2
                )
                qsc = mx.concatenate(
                    [qsc, mx.zeros((B, H, pad_l, qsc.shape[-1]), dtype=qsc.dtype)],
                    axis=2,
                )
            s = kq.dsa_indexer_scores_q(qc, qsc, kc, ksc, wb, **flags)
        else:
            qb = q16[:, :, qs:qe]
            if pad_l:
                qb = mx.concatenate(
                    [qb, mx.zeros((B, H, pad_l, D), dtype=mx.float16)], axis=2
                )
            s = kq.dsa_indexer_scores(qb, keys, wb, **flags)
        return s[:, 0, :m, :P]

    return score


def _indexer_decode_scorer(q, index_k, weights, scale, offset, ratio):
    """Decode-width scores through the fused kq kernel
    (dsa_indexer_score_decode): the relu, scale and head sum run in one
    dispatch and no [heads, L, P] score is materialized. The kernel hides
    pooled row n from query row m once n >= (offset + m + 1) // ratio,
    which is the caller's pool mask, and shows every row to a lone query;
    so a step of several rows needs ``offset``. Returns
    ``score(qs, qe) -> [B, qe - qs, P]`` float16, or None to keep the
    inline fp32 path (other geometries, no Metal,
    GMLX_DS41_INDEXER_DECODE=0)."""
    B, H, L, D = q.shape
    if (
        H not in (4, 32, 64)
        or D != 128
        or (L > 1 and not isinstance(offset, int))
        or os.environ.get("GMLX_DS41_INDEXER_DECODE", "1") == "0"
        or not mx.metal.is_available()
        or mx.default_device() != mx.Device(mx.gpu)
        or not _v4._dsa_probe("indexer")
    ):
        return None
    import mlx_kquant as kq

    if not hasattr(kq, "dsa_indexer_score_decode"):
        return None
    q16 = q.astype(mx.float16)
    keys = index_k if index_k.dtype == mx.float16 else index_k.astype(mx.float16)
    w16 = (weights * scale).astype(mx.float16)
    base = offset if isinstance(offset, int) else 0
    r = ratio if isinstance(ratio, int) and ratio >= 1 else 1

    def score(qs, qe):
        s = kq.dsa_indexer_score_decode(
            q16[:, :, qs:qe], keys, w16[:, qs:qe], base + qs, r
        )
        return s[:, 0]

    return score


def _indexer_kernel_topk(scores, k):
    """Top-k over kernel scores [B, L, P] float16 through the kq radix
    arg-select: the same index set as argpartition, order unspecified."""
    import mlx_kquant as kq

    return kq.dsa_topk_indices(scores[:, None], k, bucketed=True)[:, 0]


def _select_candidate_blocks(scores, topk_blocks, block_size, floor):
    """Level one of the two-level top-k: keep the highest-scoring blocks
    per query, plus the block holding that query's newest position.

    ``scores`` already carries ``floor`` on every position the query
    cannot reach, so a block that scores ``floor`` is out of reach and is
    dropped even when the top-k has room for it. Returns None when every
    block fits, which leaves the caller's scores untouched.
    (reference select_candidate_blocks)
    """
    width = scores.shape[-1]
    pad = (-width) % block_size
    if pad:
        scores = mx.concatenate(
            [scores, mx.full(scores.shape[:-1] + (pad,), floor, dtype=scores.dtype)],
            axis=-1,
        )
    blocks = mx.unflatten(scores, -1, (-1, block_size)).max(axis=-1)
    n_blocks = blocks.shape[-1]
    if topk_blocks >= n_blocks:
        return None
    # The newest block is only part filled, so an older full block can
    # outscore it; pin it in.
    reach = (blocks > floor).astype(mx.int32).sum(axis=-1, keepdims=True)
    pin = mx.arange(n_blocks) == (reach - 1)
    blocks = mx.where(pin, mx.finfo(blocks.dtype).max, blocks)
    idx = mx.argpartition(-blocks, kth=topk_blocks - 1, axis=-1)[..., :topk_blocks]
    keep = mx.put_along_axis(
        mx.zeros(blocks.shape, dtype=mx.bool_),
        idx,
        mx.take_along_axis(blocks, idx, axis=-1) > floor,
        axis=-1,
    )
    return mx.repeat(keep, block_size, axis=-1)[..., :width]


# --- QAT round-trips --------------------------------------------------------
# V4.1 quantizes differently from V4: the window KV takes FP8 block-32 over
# the whole post-RoPE row with no F16 round, the compressed latent takes FP4
# in groups of 16 with an E4M3 scale, and the indexer takes FP4 block-32 with
# an E8M0 scale and no Hadamard.
# (reference kernel.py act_quant / fp4_act_quant and their call sites)


def _qat_enabled() -> bool:
    return os.environ.get("GMLX_DS41_QAT", "1") != "0"


# Fused QAT kernels (mlx-kquant dsa_kv_qat block 32 / dsa_indexer_qat
# without the Hadamard), bit-identical to the compiled chains below and one
# dispatch each in place of ~6. Probed once with a real eval so a kq without
# the arguments, or a CPU default device, leaves the chains in charge;
# GMLX_DS41_QAT_FUSED=0 keeps them off for A/Bs.
_QAT_FUSED: Dict[str, Optional[bool]] = {"kv": None, "indexer": None}


def _qat_fused(which: str) -> bool:
    on = _QAT_FUSED[which]
    if on is None:
        on = False
        if (
            os.environ.get("GMLX_DS41_QAT_FUSED", "1") != "0"
            and mx.metal.is_available()
            and mx.default_device() == mx.Device(mx.gpu)
        ):
            try:
                import mlx_kquant as kq

                x = mx.arange(128, dtype=mx.float32).reshape(1, 128)
                if which == "kv":
                    got = kq.dsa_kv_qat(x, 0, f16_round=False, block=32)
                    ref = _v4._fp8_e4m3_roundtrip(x, block=32)
                else:
                    got = kq.dsa_indexer_qat(x, hadamard=False)
                    ref = _v4._fp4_e2m1_roundtrip(x, block=32)
                on = bool(mx.array_equal(got, ref).item())
            except Exception:  # noqa: BLE001 - any failure means the chain
                on = False
        _QAT_FUSED[which] = on
    return on


def _kv_qat(kv: mx.array) -> mx.array:
    if not _qat_enabled():
        return kv
    if kv.shape[-1] % 32 == 0 and _qat_fused("kv"):
        import mlx_kquant as kq

        return kq.dsa_kv_qat(kv, 0, f16_round=False, block=32)
    return _v4._fp8_e4m3_roundtrip(kv, block=32)


def _indexer_qat(x: mx.array) -> mx.array:
    if not _qat_enabled():
        return x
    if x.shape[-1] == 128 and _qat_fused("indexer"):
        import mlx_kquant as kq

        return kq.dsa_indexer_qat(x, hadamard=False)
    return _v4._fp4_e2m1_roundtrip(x, block=32)


@partial(mx.compile, shapeless=True)
def _latent_qat_core(v: mx.array) -> mx.array:
    # E4M3 scale, so no power-of-two rounding; the amax floor keeps an
    # all-zero group's scale nonzero.
    amax = mx.maximum(mx.max(mx.abs(v), axis=-1, keepdims=True), 6.0 * 2.0**-9)
    scale = _v4._e4m3_round(amax / 6.0)
    return _v4._e2m1_round(mx.clip(v / scale, -6.0, 6.0)) * scale


def _latent_qat(x: mx.array) -> mx.array:
    if not _qat_enabled() or x.shape[-1] % 16:
        return x
    orig = x.dtype
    v = mx.unflatten(x.astype(mx.float32), -1, (-1, 16))
    return mx.flatten(_latent_qat_core(v), -2).astype(orig)


# --- attention --------------------------------------------------------------


class DeepseekV41Attention(nn.Module):
    """Sliding-window latent attention, plus the shared compressed stream
    on layers whose compress ratio is non-zero.

    A non-zero ratio does not mean the layer compresses its own KV: only
    a kv-source layer does, and the layers after it read that pool and
    the top-k their index source published. (reference Attention)
    """

    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.is_kv_source = layer_idx in config.kv_source_layers
        self.is_index_source = layer_idx in config.index_source_layers
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.index_head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.scale = self.head_dim ** -0.5

        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.wo_a = MultiLinear(
            self.n_heads * self.head_dim // config.o_groups,
            config.o_lora_rank,
            config.o_groups,
        )
        self.wo_b = nn.Linear(
            config.o_groups * config.o_lora_rank,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)

        # One frequency table per layer, shared by q, the window KV, the
        # latent and the indexer. A ratio-0 layer keeps the plain theta and
        # drops YaRN.
        theta = config.compress_rope_theta if self.compress_ratio else config.rope_theta
        scaling = config.rope_scaling if self.compress_ratio else None
        self.rope = DeepseekV4RoPE(
            config.qk_rope_head_dim, theta, scaling, config.max_position_embeddings
        )
        if self.is_kv_source:
            # A latent stands for the first token of its group, so group j
            # takes position j * ratio.
            self.pool_rope = DeepseekV4RoPE(
                config.qk_rope_head_dim, theta, scaling,
                config.max_position_embeddings, freq_scale=self.compress_ratio,
            )
            self.compressor = Compressor(config, self.compress_ratio)
        if self.is_index_source:
            self.indexer = Indexer(
                config, layer_idx, owns_k=layer_idx in config.index_key_layers
            )
        self.sharding_group = None

    def _publish(self, x, pool_cache, idx_cache, offset, streams, L):
        """Compress this layer's KV and publish the latent pool and the
        index keys the layers after it read. (reference _compress_kv)"""
        B = x.shape[0]
        latent, pool_base = self.compressor(x, pool_cache, offset)
        streams.pool_cache = pool_cache
        streams.ratio = self.compress_ratio
        streams.topk = None

        # The indexer needs the latent before RoPE, so the keys come first.
        if self.is_index_source and self.indexer.owns_k:
            if idx_cache is not None:
                # Zero-width windows: the key rows arrive already pooled, so
                # this call is only the bookkeeping trim() replays from.
                z = mx.zeros((B, L, 0), dtype=x.dtype)
                idx_cache.accumulate_windows(z, z, offset)
            keys = (
                self.indexer.make_keys(latent, self.pool_rope, pool_base)
                if latent.shape[1] > 0
                else mx.zeros((B, 0, self.index_head_dim), dtype=mx.float16)
            )
            streams.index_k = (
                idx_cache.update_and_fetch(keys) if idx_cache is not None else keys
            )

        if latent.shape[1] > 0:
            roped = self.pool_rope(latent[:, None], offset=pool_base).squeeze(1)
            latent = _latent_qat(roped)
        streams.pooled = (
            pool_cache.update_and_fetch(latent) if pool_cache is not None else latent
        )

    def publish(self, x, caches, offset, streams):
        """The shared streams from ``x`` alone (a prefill-tail layer with
        no query rows)."""
        _, pool_cache, idx_cache = caches
        self._publish(x, pool_cache, idx_cache, offset, streams, x.shape[1])

    def __call__(self, x, mask, caches, offset, streams, n_prep=0, pub=None):
        """Rows [0, n_prep) of ``x`` feed the window KV only; the rest are
        queries. ``pub`` = (rows, offset) publishes the shared streams from
        a wider span than ``x`` (prefill tail). None when there are no
        query rows."""
        B, L, _ = x.shape
        local_cache, pool_cache, idx_cache = caches
        prof = _subprof_on(L)
        if prof:
            import time

            t0 = time.perf_counter()

        n_full = L - n_prep
        q0 = offset + n_prep
        xq = x[:, n_prep:] if n_prep else x
        if n_full:
            q_residual = self.q_norm(self.wq_a(xq))
            q = self.wq_b(q_residual).reshape(B, n_full, self.n_heads, self.head_dim)
            q = q.transpose(0, 2, 1, 3)
            q = self.rope(q, q0)
            if prof:
                t0 = _prof_mark("a.q", self.layer_idx, q, t0)

        kv = self.kv_norm(self.wkv(x)).reshape(B, 1, L, self.head_dim)
        kv = self.rope(kv, offset)
        kv = _kv_qat(kv)
        if local_cache is not None:
            kv, _ = local_cache.update_and_fetch(kv, mx.zeros((B, 1, L, 0)))
        if prof:
            t0 = _prof_mark("a.kv", self.layer_idx, kv, t0)

        if self.is_kv_source:
            rows, base = pub if pub is not None else (x, offset)
            self._publish(rows, pool_cache, idx_cache, base, streams, rows.shape[1])
            if prof:
                t0 = _prof_mark(
                    "a.pub", self.layer_idx, (streams.pooled, streams.index_k), t0
                )
        if n_full == 0:
            return None
        if n_prep or (mask is None and n_full > 1):
            mask = create_causal_mask(
                n_full, kv.shape[2] - n_full,
                window_size=self.config.sliding_window,
            )

        sinks = self.attn_sink.astype(q.dtype)
        block = _v4._prefill_block()
        arrays = isinstance(kv, mx.array) and isinstance(mask, mx.array)
        banded = 0 < block < n_full and arrays
        if not self.compress_ratio:
            out = self._window_attention(
                q, kv, mask, local_cache, sinks, banded, block
            )
            if prof:
                t0 = _prof_mark("a.core", self.layer_idx, out, t0)
            out = self._project(out, q0, B, n_full)
            if prof:
                _prof_mark("a.out", self.layer_idx, out, t0)
            return out

        pooled = streams.pooled
        plen = 0 if pooled is None else pooled.shape[1]

        if plen == 0:
            out = self._window_attention(
                q, kv, mask, local_cache, sinks, banded, block
            )
        else:
            src = streams.pool_cache
            pmask = (
                src.make_mask(n_full, q0)
                if src is not None
                else _v4._cacheless_pool_mask(plen, n_full, q0, streams.ratio)
            )
            if plen <= self.index_topk:
                # Every reachable position fits, so the gather would select
                # all of them; a masked dense pass is the same result.
                full_kv = mx.concatenate([kv, pooled[:, None]], axis=2)
                out = scaled_dot_product_attention(
                    q, full_kv, full_kv, cache=local_cache, scale=self.scale,
                    mask=_v4._extend_mask(mask, pmask, full_kv.shape[2]),
                    sinks=sinks,
                )
            else:
                if self.is_index_source:
                    streams.topk = self.indexer(
                        xq, q_residual, self.rope, streams.index_k, pmask,
                        q0, streams,
                    )
                    if prof:
                        t0 = _prof_mark("a.idx", self.layer_idx, streams.topk, t0)
                topk = streams.topk
                if topk.shape[1] != n_full:
                    # Prefill tail: fewer rows than the index source ran.
                    topk = topk[:, topk.shape[1] - n_full:]
                sparse_mask = None
                if pmask is not None:
                    sparse_mask = mx.take_along_axis(
                        pmask[None] if pmask.ndim == 2 else pmask, topk, axis=2
                    )[:, None]
                out = None
                if _v4._sparse_kernel_ok():
                    kblock = (
                        _v4._sparse_kernel_block() if arrays and n_full > 16 else 0
                    )
                    if 0 < kblock < n_full:
                        out = _v4._sparse_kernel_attention_banded(
                            q, kv, pooled, topk, mask, sparse_mask, self.scale,
                            sinks, self.config.sliding_window, kblock,
                        )
                    else:
                        out = _v4._sparse_kernel_attention(
                            q, kv, pooled, topk, mask, sparse_mask, self.scale,
                            sinks,
                        )
                if out is not None:
                    pass
                elif (
                    n_full <= 4
                    and _v4._COMPILE_SPARSE
                    and kv.shape[2] >= self.config.sliding_window
                ):
                    # Decode on a full window: steady shapes, so the
                    # compiled core traces once. The gather stays eager
                    # (it alone sees the growing pool).
                    gathered = _v4._sparse_topk_gather(
                        pooled, topk, n_full, self.head_dim
                    )
                    out = _v4._sparse_gathered_attention_c(
                        q, kv, gathered, mask, sparse_mask, self.scale, sinks
                    )
                elif banded:
                    out = _v4._sparse_pooled_attention_banded(
                        q, kv, pooled, topk, mask, sparse_mask, self.scale,
                        sinks, self.config.sliding_window, block,
                    )
                else:
                    out = _v4._sparse_pooled_attention(
                        q, kv, pooled, topk, mask, sparse_mask, self.scale,
                        sinks,
                    )
        if prof:
            t0 = _prof_mark("a.core", self.layer_idx, out, t0)
        out = self._project(out, q0, B, n_full)
        if prof:
            _prof_mark("a.out", self.layer_idx, out, t0)
        return out

    def _window_attention(self, q, kv, mask, local_cache, sinks, banded, block):
        if banded:
            return _v4._banded_window_attention(
                q, kv, mask, self.scale, sinks, self.config.sliding_window, block
            )
        return scaled_dot_product_attention(
            q, kv, kv, cache=local_cache, scale=self.scale, mask=mask,
            sinks=sinks,
        )

    def _project(self, out, offset, B, L):
        """De-rope, then the block-diagonal output projection: group g
        reads only its own heads."""
        out = self.rope(out, offset, inverse=True)
        out = out.reshape(B, self.o_groups, -1, L, self.head_dim)
        out = out.transpose(0, 1, 3, 2, 4).flatten(-2)
        out = self.wo_a(out)
        out = out.transpose(0, 2, 1, 3).flatten(-2)
        return self.wo_b(out)


# --- block and model --------------------------------------------------------


def _hc_mixes(hc: HyperConnection, x: mx.array):
    """The pre / post / comb coefficients this sublayer hands to the next
    one. The same projection HyperConnection.__call__ runs, without the
    collapse: V4.1 collapses with the previous sublayer's pre.
    (reference Block.hc_mixes)"""
    fn_t = getattr(hc, "_fn_t", None)
    if fn_t is None:
        fn_t = mx.contiguous(hc.fn.T)
        mx.eval(fn_t)
        hc._fn_t = fn_t
    y = x.astype(mx.float32)
    mixes = mx.fast.rms_norm(y.flatten(-2), None, hc.norm_eps) @ fn_t
    return _hc_split_sinkhorn_ops(
        mixes, hc.scale, hc.base, hc.hc_mult, hc.sinkhorn_iters, hc.hc_eps
    )


def _hc_collapse(x: mx.array, pre: mx.array) -> mx.array:
    return (pre[..., None] * x.astype(mx.float32)).sum(axis=2).astype(x.dtype)


def _hc_fused_route() -> bool:
    """GMLX_DS41_HC_FUSED=0 keeps the ops route at decode width, for A/Bs."""
    return os.environ.get("GMLX_DS41_HC_FUSED", "1") != "0"


# GMLX_DECODE_LAYER_PROFILE=1: eval after each decode-layer component and
# print the per-component wall per token at exit. Attribution only: the
# extra syncs slow the run and split each layer into one command buffer
# per component (which is what makes a GPU trace of it readable).
_LAYER_PROFILE_LEVEL = int(os.environ.get("GMLX_DECODE_LAYER_PROFILE", "0") or 0)
_LAYER_PROFILE = _LAYER_PROFILE_LEVEL >= 1
# Level 2 also evals inside attention (q, kv, publish, indexer, core, out)
# and the MoE (route, experts, shared), one command buffer each in a trace.
_SUB_PROFILE = _LAYER_PROFILE_LEVEL >= 2
# GMLX_LAYER_PROFILE_PREFILL=1 also marks steps wider than one token (the
# prefill chunks, on the plain block route).
_PROFILE_WIDE = os.environ.get("GMLX_LAYER_PROFILE_PREFILL", "0") == "1"
_PROF: Dict[tuple, float] = {}
_PROF_CALLS = [0]


def _prof_on(width: int) -> bool:
    return _LAYER_PROFILE and (width == 1 or _PROFILE_WIDE)


def _subprof_on(width: int) -> bool:
    return _SUB_PROFILE and (width == 1 or _PROFILE_WIDE)


_PROF_LOG: list = []


def _prof_mark(key: str, li: int, arr, t0: float) -> float:
    import time

    mx.eval(arr)
    t1 = time.perf_counter()
    _PROF[(key, li)] = _PROF.get((key, li), 0.0) + (t1 - t0)
    if _SUB_PROFILE:
        # Wall-clock window of the mark, for aligning a GPU trace's command
        # buffers to the marks (GMLX_DECODE_LAYER_PROFILE_LOG=<path>).
        w1 = time.time()
        _PROF_LOG.append((key, li, w1 - (t1 - t0), w1))
    return t1


def _prof_dump() -> None:
    n = _PROF_CALLS[0]
    if not n:
        return
    keys = ("engram", "hc", "attn", "ffn")
    tot = {k: sum(v for (kk, _), v in _PROF.items() if kk == k) for k in keys}
    print(
        "[layerprof] per token ms: "
        + " | ".join(f"{k} {1e3 * tot[k] / n:.1f}" for k in keys)
        + f" | total {1e3 * sum(tot.values()) / n:.1f} over {n} tokens",
        flush=True,
    )
    for k in ("attn", "ffn", "engram"):
        rows = sorted((li, v) for (kk, li), v in _PROF.items() if kk == k)
        if rows:
            print(
                f"[layerprof] {k} by layer ms/token: "
                + " ".join(f"{li}:{1e3 * v / n:.2f}" for li, v in rows),
                flush=True,
            )
    path = os.environ.get("GMLX_DECODE_LAYER_PROFILE_LOG")
    if path and _PROF_LOG:
        with open(path, "w") as f:
            for key, li, w0, w1 in _PROF_LOG:
                f.write(f"{key} {li} {w0:.6f} {w1:.6f}\n")
    subs = sorted({kk for (kk, _) in _PROF if "." in kk})
    if subs:
        print(
            "[layerprof] sub per token ms: "
            + " | ".join(
                f"{k} {1e3 * sum(v for (kk, _), v in _PROF.items() if kk == k) / n:.1f}"
                for k in subs
            ),
            flush=True,
        )


if _LAYER_PROFILE:
    import atexit

    atexit.register(_prof_dump)


def _prefill_tail_on() -> bool:
    """GMLX_DS41_PREFILL_TAIL=0 runs every layer on every prompt row."""
    return os.environ.get("GMLX_DS41_PREFILL_TAIL", "1") != "0"


class DeepseekV41Block(nn.Module):
    """Attention and FFN between a collapse and an expand, with the
    collapse weights coming from the previous sublayer.
    (reference Block.forward)"""

    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = DeepseekV41Attention(config, layer_idx)
        self.ffn = DeepseekV41MoE(config, layer_idx)
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = HyperConnection(config)
        self.ffn_hc = HyperConnection(config)
        self.engram = None
        self.engram_slot = -1

    def __call__(self, h, pre_mix, mask, caches, offset, streams, input_ids,
                 image_mask=None):
        if (
            _hc_fused_route()
            and h.dtype in (mx.float16, mx.bfloat16)
            and self.attn_hc.wide_fused_ok(h)
        ):
            return self._wide_step(
                h, pre_mix, mask, caches, offset, streams, input_ids, image_mask
            )
        prof = _prof_on(h.shape[1])
        if prof:
            import time

            if self.layer_idx == 0:
                _PROF_CALLS[0] += 1
            t0 = time.perf_counter()
        residual = h
        attn_pre, attn_post, attn_comb = _hc_mixes(self.attn_hc, h)
        x = self.attn_norm(_hc_collapse(h, pre_mix))
        if prof:
            t0 = _prof_mark("hc", self.layer_idx, x, t0)
        x = self.attn(x, mask, caches, offset, streams)
        if prof:
            t0 = _prof_mark("attn", self.layer_idx, x, t0)
        h = hc_expand(x, residual, attn_post, attn_comb)

        residual = h
        ffn_pre, ffn_post, ffn_comb = _hc_mixes(self.ffn_hc, h)
        x = self.ffn_norm(_hc_collapse(h, attn_pre))
        if prof:
            t0 = _prof_mark("hc", self.layer_idx, x, t0)
        x = self.ffn(x, input_ids, image_mask)
        if prof:
            t0 = _prof_mark("ffn", self.layer_idx, x, t0)
        h = hc_expand(x, residual, ffn_post, ffn_comb)
        if prof:
            _prof_mark("hc", self.layer_idx, h, t0)
        return h, ffn_pre

    def _wide_step(self, h, pre_mix, mask, caches, offset, streams, input_ids,
                   image_mask):
        """``__call__`` past the M=1 width: the GEMM front, then the lag
        collapse and the expand as one kernel each."""
        prof = _prof_on(h.shape[1])
        if prof:
            import time

            if self.layer_idx == 0:
                _PROF_CALLS[0] += 1
            t0 = time.perf_counter()
        hc = self.attn_hc
        mixes_raw, ssq = hc.front_wide(h)
        x, attn_pre, attn_post, attn_comb = hc.lag_collapse_m1(
            h, mixes_raw, ssq, self.attn_norm.weight, pre_mix
        )
        if prof:
            t0 = _prof_mark("hc", self.layer_idx, x, t0)
        x = self.attn(x, mask, caches, offset, streams)
        if prof:
            t0 = _prof_mark("attn", self.layer_idx, x, t0)
        h = hc_expand_m1(x, h, attn_post, attn_comb)

        hc = self.ffn_hc
        mixes_raw, ssq = hc.front_wide(h)
        x, ffn_pre, ffn_post, ffn_comb = hc.lag_collapse_m1(
            h, mixes_raw, ssq, self.ffn_norm.weight, attn_pre
        )
        if prof:
            t0 = _prof_mark("hc", self.layer_idx, x, t0)
        x = self.ffn(x, input_ids, image_mask)
        if prof:
            t0 = _prof_mark("ffn", self.layer_idx, x, t0)
        h = hc_expand_m1(x, h, ffn_post, ffn_comb)
        if prof:
            _prof_mark("hc", self.layer_idx, h, t0)
        return h, ffn_pre

    def _front(self, hc, norm, h, pre_in):
        """One sublayer's hyper-connection front on the route the row
        count picks: (x, pre, post, comb, fused). ``fused`` names the
        expand that pairs with these mixes."""
        if (
            _hc_fused_route()
            and h.dtype in (mx.float16, mx.bfloat16)
            and hc.wide_fused_ok(h)
        ):
            mixes_raw, ssq = hc.front_wide(h)
            x, pre, post, comb = hc.lag_collapse_m1(
                h, mixes_raw, ssq, norm.weight, pre_in
            )
            return x, pre, post, comb, True
        pre, post, comb = _hc_mixes(hc, h)
        return norm(_hc_collapse(h, pre_in)), pre, post, comb, False

    @staticmethod
    def _expand(x, residual, post, comb, fused):
        if fused:
            return hc_expand_m1(x, residual, post, comb)
        return hc_expand(x, residual, post, comb)

    def tail_step(self, h, pre_mix, caches, offset, streams, input_ids,
                  image_mask, n_pub, n_prep, publish):
        """``__call__`` on a prefill tail. Rows [0, n_pub) of ``h`` only
        feed the shared streams (``publish``), the next ``n_prep`` rows
        only the window KV; the rest run in full. ``input_ids`` and
        ``image_mask`` cover the full rows. Returns 0 rows when there
        are none."""
        x, attn_pre, attn_post, attn_comb, fused = self._front(
            self.attn_hc, self.attn_norm, h, pre_mix
        )
        if x.shape[1] == n_pub:
            if publish:
                self.attn.publish(x, caches, offset, streams)
            return h[:, :0], attn_pre[:, :0]
        pub = (x, offset) if publish else None
        if n_pub:
            x = x[:, n_pub:]
        x = self.attn(x, None, caches, offset + n_pub, streams, n_prep, pub)
        if x is None:
            return h[:, :0], attn_pre[:, :0]
        a = n_pub + n_prep
        if a:
            h, attn_pre, attn_post, attn_comb = (
                h[:, a:], attn_pre[:, a:], attn_post[:, a:], attn_comb[:, a:]
            )
        h = self._expand(x, h, attn_post, attn_comb, fused)

        x, ffn_pre, ffn_post, ffn_comb, fused = self._front(
            self.ffn_hc, self.ffn_norm, h, attn_pre
        )
        x = self.ffn(x, input_ids, image_mask)
        h = self._expand(x, h, ffn_post, ffn_comb, fused)
        return h, ffn_pre

    def fused_step(self, h, pre_mix, carry, mask, caches, offset, streams,
                   input_ids, image_mask=None):
        """``__call__`` on the fused M=1 route: two dispatches per
        hyper-connection cycle instead of the Sinkhorn's ~100 ops. The
        FFN expand is not applied here; it returns as ``carry`` for the
        next layer's front (or the model's final expand) to fold in.
        Returns (h, ffn_pre, carry) with h the post-attention stream."""
        prof = _LAYER_PROFILE and h.shape[1] == 1
        if prof:
            import time

            if self.layer_idx == 0:
                _PROF_CALLS[0] += 1
            t0 = time.perf_counter()
        hc = self.attn_hc
        if carry is None:
            mixes_raw, ssq = hc.front_m1(h)
        else:
            h, mixes_raw, ssq = hc.front_expand_m1(carry)
        x, attn_pre, attn_post, attn_comb = hc.lag_collapse_m1(
            h, mixes_raw, ssq, self.attn_norm.weight, pre_mix)
        if prof:
            t0 = _prof_mark("hc", self.layer_idx, x, t0)
        x = self.attn(x, mask, caches, offset, streams)
        if prof:
            t0 = _prof_mark("attn", self.layer_idx, x, t0)

        hc = self.ffn_hc
        h, mixes_raw, ssq = hc.front_expand_m1((x, h, attn_post, attn_comb))
        x, ffn_pre, ffn_post, ffn_comb = hc.lag_collapse_m1(
            h, mixes_raw, ssq, self.ffn_norm.weight, attn_pre)
        if prof:
            t0 = _prof_mark("hc", self.layer_idx, x, t0)
        x = self.ffn(x, input_ids, image_mask)
        if prof:
            _prof_mark("ffn", self.layer_idx, x, t0)
        return h, ffn_pre, (x, h, ffn_post, ffn_comb)


class DeepseekV41Model(PipelineMixin, nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            DeepseekV41Block(config, idx) for idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.engram_hash = None
        if config.engram_layer_ids:
            if config.engram_token_map is not None:
                self.engram_hash = EngramHash(config)
            for slot, layer_id in enumerate(config.engram_layer_ids):
                self.layers[layer_id].engram = Engram(
                    config, config.engram_table_rows[slot]
                )
                self.layers[layer_id].engram_slot = slot

    def _split_cache(self, layer_idx: int, entry, cache_list_types):
        """(window, latent pool, index-key pool) for one layer. Only a
        kv-source layer owns pools; every other layer has just its
        window, and the pool members never move."""
        if entry is None:
            return None, None, None
        if not isinstance(entry, cache_list_types):
            return entry, None, None
        members = list(entry)
        if layer_idx in self.args.kv_source_layers:
            return members[0], members[1], members[2]
        return members[0], None, None

    def _tail_plan(self, entry, offset, L, B, fused):
        """(prompt end, first tail layer, window history) when this chunk
        runs under an armed prompt end (``_gmlx_prefill_end`` on the first
        cache entry), else None. A chunk that reaches the end clears the
        arm. Layers past the last kv-source layer then run only the rows
        the layers after them can still reach: the last layer needs the
        last row, and each layer before it one window more.
        (ds4 decoder suffix)"""
        end = getattr(entry, "_gmlx_prefill_end", None)
        if end is None:
            return None
        end = int(end)
        if offset + L >= end:
            entry._gmlx_prefill_end = None
        hist = (self.args.sliding_window or 0) - 1
        sources = self.args.kv_source_layers
        if (
            fused or B != 1 or hist <= 0 or not sources
            or offset + L > end or end - 1 - hist <= offset
            or not _prefill_tail_on()
        ):
            return None
        return end, max(sources), hist

    def _limit_feeder_pass(self, tail, offset, L):
        """Tell the prefill feeder the last layer whose experts this pass
        uses, so it stages nothing for the layers the tail skips."""
        feeder = None
        for layer in self.layers:
            feeder = getattr(
                getattr(layer.ffn, "switch_mlp", None), "_kq_feeder", None
            )
            if feeder is not None:
                break
        if feeder is None or not hasattr(feeder, "limit_pass"):
            return
        last = None
        if tail is not None:
            end, first, hist = tail
            n = len(self.layers)
            last = first - 1
            for idx in range(first, n):
                if end - 1 - hist * (n - 1 - idx) < offset + L:
                    last = idx
        feeder.limit_pass(last)

    def _touch_window(self, local, B, dtype):
        """A never-written tail window cache gets an empty state (the
        generation loop evaluates every cache after a chunk). Its offset
        counts the rows it holds, as the rotating cache expects."""
        if local is not None and getattr(local, "keys", None) is None:
            local.update_and_fetch(
                mx.zeros((B, 1, 0, self.args.head_dim), dtype=dtype),
                mx.zeros((B, 1, 0, 0)),
            )

    def _history_slot(self, cache, cache_list_types):
        """The engram token history, held on the first engram layer."""
        ids = self.args.engram_layer_ids
        if not ids or cache is None or cache[ids[0]] is None:
            return None
        entry = cache[ids[0]]
        if not isinstance(entry, cache_list_types):
            return None
        member = list(entry)[-1]
        return member if hasattr(member, "cache") else None

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
        image_mask: Optional[mx.array] = None,
    ) -> mx.array:
        from gmlx.cache.compat import cache_types

        cache_list_types = cache_types("CacheList")
        if cache is None:
            cache = [None] * len(self.layers)
        mask_cache, _, _ = self._split_cache(0, cache[0], cache_list_types)
        offset = mask_cache.offset if mask_cache is not None else 0

        # Image tokens take no part in an n-gram and get no engram write.
        engram_mask = None if image_mask is None else ~image_mask
        if image_mask is not None:
            # Image ids sit past the vocab: clamp before ANY gather
            # (embedding, n-gram token map). The container supplies the
            # embeddings for those rows.
            inputs = mx.where(image_mask, mx.zeros_like(inputs), inputs)
        row_ids = None
        if self.engram_hash is not None:
            slot = self._history_slot(cache, cache_list_types)
            history = slot[0] if slot is not None else None
            row_ids, new_history = self.engram_hash(inputs, history, engram_mask)
            if slot is not None:
                slot[0] = new_history

        h = self.embed_tokens(inputs) if input_embeddings is None else input_embeddings
        h = mx.contiguous(
            mx.broadcast_to(
                h[:, :, None, :],
                (h.shape[0], h.shape[1], self.args.hc_mult, h.shape[2]),
            )
        )
        mask = create_attention_mask(
            h[:, :, 0, :], mask_cache,
            window_size=self.args.sliding_window, return_array=True,
        )

        if row_ids is not None:
            # File-backed tables read their rows off-thread from here on,
            # under the layers that run before each engram.
            for layer in self.layers:
                pf = getattr(getattr(layer.engram, "embed", None), "prefetch", None)
                if pf is not None:
                    pf(row_ids[:, :, layer.engram_slot, :])
        streams = SharedStreams()
        pre_mix = mx.zeros(
            (h.shape[0], h.shape[1], self.args.hc_mult), dtype=mx.float32
        )
        pre_mix[:, :, 0] = 1.0
        fused = (
            _hc_fused_route()
            and h.dtype in (mx.float16, mx.bfloat16)
            and self.layers[0].attn_hc.m1_fused_ok(h)
        )
        carry = None
        prof = _LAYER_PROFILE and (
            (fused and h.shape[1] == 1) or (_PROFILE_WIDE and h.shape[1] > 1)
        )
        L = h.shape[1]
        tail = self._tail_plan(cache[0], offset, L, h.shape[0], fused)
        n_layers = len(self.layers)
        self._limit_feeder_pass(tail, offset, L)
        for idx, layer in enumerate(self.layers):
            # Rows of h are the chunk's last h.shape[1] rows (prefill tail).
            r0 = L - h.shape[1]
            if layer.engram is not None and row_ids is not None and h.shape[1]:
                if prof:
                    import time

                    t0 = time.perf_counter()
                if carry is not None:
                    h, carry = hc_expand_m1(*carry), None
                h = layer.engram(
                    h, row_ids[:, r0:, layer.engram_slot, :],
                    engram_mask if engram_mask is None else engram_mask[:, r0:],
                )
                if prof:
                    _prof_mark("engram", idx, h, t0)
            caches = self._split_cache(idx, cache[idx], cache_list_types)
            if tail is not None and idx >= tail[1]:
                end, _, hist = tail
                if h.shape[1]:
                    f = end - 1 - hist * (n_layers - 1 - idx)
                    lo = min(L, max(0, f - hist - offset))
                    a = min(L, max(0, f - offset))
                    h, pre_mix = layer.tail_step(
                        h, pre_mix, caches, offset + r0, streams,
                        inputs[:, a:],
                        image_mask if image_mask is None else image_mask[:, a:],
                        lo - r0, a - lo, idx in self.args.kv_source_layers,
                    )
                self._touch_window(caches[0], h.shape[0], h.dtype)
                continue
            if fused:
                h, pre_mix, carry = layer.fused_step(
                    h, pre_mix, carry, mask, caches, offset, streams,
                    inputs, image_mask,
                )
            else:
                h, pre_mix = layer(
                    h, pre_mix, mask, caches, offset, streams, inputs,
                    image_mask,
                )
        if carry is not None:
            h = hc_expand_m1(*carry)
        _v4._materialize_cache_arrays(cache)
        return self.norm(_hc_collapse(h, pre_mix))


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = DeepseekV41Model(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None, input_embeddings=None,
                 image_mask=None, **kwargs) -> mx.array:
        out = self.model(inputs, cache, input_embeddings=input_embeddings,
                         image_mask=image_mask)
        if out.shape[1] == 0:
            # A prefill-tail chunk with no row the last layer needs.
            return mx.zeros(
                (out.shape[0], 0, self.args.vocab_size), dtype=out.dtype
            )
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        from gmlx.cache.compat import construction_cache_module

        cmod = construction_cache_module()
        args = self.args
        first_engram = args.engram_layer_ids[:1]
        caches = []
        for idx in range(args.num_hidden_layers):
            slots = [cmod.RotatingKVCache(max_size=args.sliding_window)]
            if idx in args.kv_source_layers:
                ratio = args.compress_ratios[idx]
                # The index pool opts out of --kv-bits packing: the score
                # path reads it in full every step.
                idx_pool = PoolingCache(ratio)
                idx_pool.quantizable = False
                slots += [PoolingCache(ratio), idx_pool]
            if idx in first_engram:
                slots.append(cmod.ArraysCache(1))
            caches.append(slots[0] if len(slots) == 1 else cmod.CacheList(*slots))
        return caches

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        n_layers = self.args.num_hidden_layers
        out = {}
        for k, v in weights.items():
            if k.startswith("mtp.") or ".rotary_emb." in k:
                continue
            parts = k.split(".")
            if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
                if parts[2].isdigit() and int(parts[2]) >= n_layers:
                    continue
            out[k] = v

        # wo_a ships 2D; MultiLinear wants (groups, rank, in).
        for idx in range(n_layers):
            prefix = f"model.layers.{idx}.attn.wo_a"
            for key in (f"{prefix}.weight", f"{prefix}.scales", f"{prefix}.biases"):
                arr = out.get(key)
                if arr is not None and arr.ndim == 2:
                    out[key] = arr.reshape(self.args.o_groups, self.args.o_lora_rank, -1)
        return out

    @property
    def cast_predicate(self):
        # Kept in step with loader._FP32_KEEP_BY_MODEL_TYPE["deepseek_v41"].
        keep = (
            "_hc.", ".attn_sink", ".e_score_correction_bias",
            ".gate.weight", ".engram.q_weight", ".engram.k_weight",
        )

        def predicate(path):
            return not any(k in path for k in keep)

        return predicate


def ensure_registered() -> None:
    """Expose this package as ``mlx_lm.models.deepseek_v41`` so every
    importer resolves it, and register deepseek_v4's companions too."""
    import sys

    _v4.ensure_registered()
    import mlx_lm.models as _models

    mod = sys.modules[__name__]
    sys.modules.setdefault("mlx_lm.models.deepseek_v41", mod)
    if not hasattr(_models, "deepseek_v41"):
        _models.deepseek_v41 = mod
