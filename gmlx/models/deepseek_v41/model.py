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
)
from gmlx.models.deepseek_v41.engram_codec import (
    decode_rows,
    parse_row_encoding,
    row_width,
)
from gmlx.models.deepseek_v41.engram_codec import tables as engram_tables

DeepseekV4RoPE = _v4.DeepseekV4RoPE
DeepseekV41MoE = _v4.DeepseekV4MoE


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
        """Index keys for freshly pooled latents, rotated on the tail."""
        k = self.k_norm(self.wk(latent))
        k = pool_rope(k[:, None], offset=pool_base).squeeze(1)
        return _indexer_qat(k)

    def __call__(self, x, q_residual, rope, index_k, pmask, offset, streams):
        B, L, _ = x.shape
        P = index_k.shape[1]
        k = min(self.index_topk, P)
        q = self.wq_b(q_residual).reshape(B, L, self.n_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)
        q = rope(q, offset)
        q = _indexer_qat(q)

        scores = q.astype(mx.float32) @ index_k[:, None].swapaxes(-1, -2).astype(
            mx.float32
        )
        scores = mx.maximum(scores, 0) * self.scale
        weights = _v4._skinny_linear(self.weights_proj, x).astype(mx.float32) * (
            self.n_heads ** -0.5
        )
        scores = (scores * weights.swapaxes(-1, -2)[..., None]).sum(axis=1)
        floor = mx.finfo(scores.dtype).min
        if pmask is not None:
            scores = mx.where(
                pmask if pmask.ndim == 3 else pmask[None], scores, floor
            )
        if self.is_candidate_source and self.candidate_block_size > 0:
            streams.candidates = _select_candidate_blocks(
                scores, self.candidate_topk_blocks, self.candidate_block_size, floor
            )
        elif self.uses_candidates and streams.candidates is not None:
            scores = mx.where(streams.candidates, scores, floor)
        return mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]


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


def _kv_qat(kv: mx.array) -> mx.array:
    if not _qat_enabled():
        return kv
    return _v4._fp8_e4m3_roundtrip(kv, block=32)


def _indexer_qat(x: mx.array) -> mx.array:
    if not _qat_enabled():
        return x
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
                else mx.zeros((B, 0, self.index_head_dim), dtype=x.dtype)
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

    def __call__(self, x, mask, caches, offset, streams):
        B, L, _ = x.shape
        local_cache, pool_cache, idx_cache = caches

        q_residual = self.q_norm(self.wq_a(x))
        q = self.wq_b(q_residual).reshape(B, L, self.n_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)
        q = self.rope(q, offset)

        kv = self.kv_norm(self.wkv(x)).reshape(B, 1, L, self.head_dim)
        kv = self.rope(kv, offset)
        kv = _kv_qat(kv)
        if local_cache is not None:
            kv, _ = local_cache.update_and_fetch(kv, mx.zeros((B, 1, L, 0)))

        sinks = self.attn_sink.astype(q.dtype)
        if not self.compress_ratio:
            out = scaled_dot_product_attention(
                q, kv, kv, cache=local_cache, scale=self.scale, mask=mask,
                sinks=sinks,
            )
            return self._project(out, offset, B, L)

        if self.is_kv_source:
            self._publish(x, pool_cache, idx_cache, offset, streams, L)
        pooled = streams.pooled
        plen = 0 if pooled is None else pooled.shape[1]

        if plen == 0:
            out = scaled_dot_product_attention(
                q, kv, kv, cache=local_cache, scale=self.scale, mask=mask,
                sinks=sinks,
            )
        else:
            src = streams.pool_cache
            pmask = (
                src.make_mask(L, offset)
                if src is not None
                else _v4._cacheless_pool_mask(plen, L, offset, streams.ratio)
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
                        x, q_residual, self.rope, streams.index_k, pmask,
                        offset, streams,
                    )
                topk = streams.topk
                sparse_mask = None
                if pmask is not None:
                    sparse_mask = mx.take_along_axis(
                        pmask[None] if pmask.ndim == 2 else pmask, topk, axis=2
                    )[:, None]
                out = _v4._sparse_pooled_attention(
                    q, kv, pooled, topk, mask, sparse_mask, self.scale, sinks
                )
        return self._project(out, offset, B, L)

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
        residual = h
        attn_pre, attn_post, attn_comb = _hc_mixes(self.attn_hc, h)
        x = self.attn_norm(_hc_collapse(h, pre_mix))
        x = self.attn(x, mask, caches, offset, streams)
        h = hc_expand(x, residual, attn_post, attn_comb)

        residual = h
        ffn_pre, ffn_post, ffn_comb = _hc_mixes(self.ffn_hc, h)
        x = self.ffn_norm(_hc_collapse(h, attn_pre))
        x = self.ffn(x, input_ids, image_mask)
        h = hc_expand(x, residual, ffn_post, ffn_comb)
        return h, ffn_pre


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

        streams = SharedStreams()
        pre_mix = mx.zeros(
            (h.shape[0], h.shape[1], self.args.hc_mult), dtype=mx.float32
        )
        pre_mix[:, :, 0] = 1.0
        for idx, layer in enumerate(self.layers):
            if layer.engram is not None and row_ids is not None:
                h = layer.engram(
                    h, row_ids[:, :, layer.engram_slot, :], engram_mask
                )
            h, pre_mix = layer(
                h, pre_mix, mask,
                self._split_cache(idx, cache[idx], cache_list_types),
                offset, streams, inputs, image_mask,
            )
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
