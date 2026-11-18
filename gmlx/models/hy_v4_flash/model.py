# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Model for Hy4-flash (GGUF arch ``hy_v4_flash``).

The runtime follows the llama.cpp reference (``src/models/hy-v4-flash.cpp``).
It keeps HY4-preview's gated absorbed MLA, per-head sinks and independent
hyper-connections (``gmlx.models.hy_v4``) and changes the rest:

  1. A hybrid stack over one MLA latent geometry. Sparse layers attend the
     whole context through a pooled lightning indexer; sliding-window
     layers attend the last ``sliding_window`` tokens. The two kinds have
     their own head count and rope base.
  2. A pooled indexer on the sparse layers. Keys are pooled ``index_kpool``
     to one, a query scores only the pools that end before its own group,
     keeps the top ``index_topk / index_kpool - 1`` of them and always
     attends its own group. Up to ``index_topk`` cached keys the selection
     is every key and attention is dense.
  3. Sandwich norms: an RMSNorm after the attention output projection and
     after the FFN.
  4. Routed experts that run at ``moe_latent_size`` between a down and an
     up projection. The router and the shared expert read the full width.
  5. Engram: an n-gram hash table lookup written into the residual streams
     of a few layers, gated by how well it matches them (deepseek_v41's
     mechanism with normed rows, a projection bias, a normed value and a
     per-stream offset).

The MTP block past the trunk is dropped here; the drafter loads it.
"""

import functools
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from gmlx._vendor.mlx_lm.models.base import BaseModelArgs
from gmlx._vendor.mlx_lm.models.mla import MultiLinear
from gmlx._vendor.mlx_lm.models.rope_utils import initialize_rope
from gmlx._vendor.mlx_lm.models.switch_layers import SwitchGLU

import gmlx.gen.prefill_decay as _prefill_decay
from gmlx.cache.kv_append import depends_keys
from gmlx.cache.kv_cache import ArraysCache, CacheList, RotatingKVCache
from gmlx.cache.window_sdpa import _HD_OK, _MIN_ROWS, _kq_window_kernel
from gmlx.envflags import env_bool
from gmlx.models.deepseek_v4.cache import (
    DS4BatchPoolingCache,
    DS4PoolingCache,
)
from gmlx.models.deepseek_v4.hyper_connection import HyperHead
from gmlx.models.deepseek_v4.model import (
    LimitedSwiGLU,
    _expert_select,
)
from gmlx.models.deepseek_v4.model import (
    ensure_registered as _ds4_ensure_registered,
)
from gmlx.models.deepseek_v41.model import EngramTable
from gmlx.models.hy_v4_flash import row_kernels
from gmlx.models.hy_v4.model import (
    _STREAM_MIN_KEYS,
    _STREAM_Q,
    HyV4KVCache,
    HyV4MLP,
    IndependentHyperConnection,
    _tiled_absorbed_attention,
)

# The reference builds every engram norm with eps unset, which resolves to
# the fp32 machine epsilon, not the model's rms_norm_eps.
ENGRAM_EPS = 1.1920929e-07

# Queries per sparse decode call that take the per-query key gather: a
# decode step, or an MTP verify block.
_GATHER_MAX_L = 8
# Queries per call kq's decode-width pool score takes. Longer calls take
# its chunk score.
_IDX_KQ_MAX_L = 4
# The pool keys are stored in fp16 whatever the model's 16-bit type: the
# kernel route emits its scores in the key type, and fp16 keeps 11 bits of
# a score where bf16 keeps 8.
_POOL_DTYPE = mx.float16
# Queries per gather in a sparse prefill: 256 queries hold 0.6 GB of
# gathered keys at a time.
_GATHER_TILE = 256


def ensure_registered() -> None:
    """Register the deepseek_v4 companions this module borrows from."""
    _ds4_ensure_registered()


@functools.cache
def _kq_tail_kernel(name: str):
    """kq's sparse attention kernel ``name`` when the mlx_kquant build
    takes a score-only tail on its rows (the rope half beside the latent),
    else None."""
    try:
        import mlx_kquant as kq
    except ImportError:
        return None
    fn = getattr(kq, name, None)
    if fn is None or "score_tail" not in (fn.__doc__ or ""):
        return None
    return fn


@functools.cache
def _kq_router_kernel():
    """kq.moe_router_topk when the mlx_kquant build has its sigmoid arm,
    else None."""
    try:
        import mlx_kquant as kq
    except ImportError:
        return None
    fn = getattr(kq, "moe_router_topk", None)
    if fn is None or "sigmoid" not in (fn.__doc__ or ""):
        return None
    return fn


@functools.cache
def _kq_indexer_kernels():
    """kq's decode-width pool score and its radix select when the
    mlx_kquant build takes 16 indexer heads and a 511-wide select, else
    None."""
    try:
        import mlx_kquant as kq
    except ImportError:
        return None
    score = getattr(kq, "dsa_indexer_score_decode", None)
    select = getattr(kq, "dsa_topk_indices", None)
    if (score is None or select is None
            or "16, 32, 64" not in (score.__doc__ or "")
            or "511" not in (select.__doc__ or "")):
        return None
    return score, select


@functools.cache
def _kq_indexer_chunk_kernel():
    """kq's pool score for a prefill chunk when the mlx_kquant build
    takes pooled keys (a ratio in its causal rule), else None."""
    try:
        import mlx_kquant as kq
    except ImportError:
        return None
    score = getattr(kq, "dsa_indexer_scores", None)
    if score is None or "causal_ratio" not in (score.__doc__ or ""):
        return None
    return score


@functools.cache
def _row_offsets(r: int) -> mx.array:
    """``arange(r)`` as int32, evaluated once: a constant of every decode
    step's row list."""
    out = mx.arange(r, dtype=mx.int32)
    mx.eval(out)
    return out


@functools.cache
def _zero_values_const(B: int, L: int, dtype) -> mx.array:
    out = mx.zeros((B, 1, L, 1), dtype=dtype)
    mx.eval(out)
    return out


def _zero_values(B: int, L: int, dtype) -> mx.array:
    """What the unused value slot of a cache takes. A decode step gets one
    evaluated constant behind a fresh handle, since a cache may keep the
    array it is handed and write into it; a chunk gets its own zeros."""
    if L > _GATHER_MAX_L:
        return mx.zeros((B, 1, L, 1), dtype=dtype)
    return _zero_values_const(B, L, dtype).__copy__()


@functools.cache
def _no_rows(B: int, L: int) -> mx.array:
    """An index list that names no row, for a decode step with no picked
    pools. Evaluated once."""
    out = mx.full((B, L, 1), -1, dtype=mx.int32)
    mx.eval(out)
    return out


def _head_keys(embed_q, latent: mx.array) -> mx.array:
    """Per-head keys ``latent @ W[h]`` [B, H, S, qk_nope_head_dim] of a
    head-shared ``latent`` [B, 1, S, kv_lora_rank]. A K-quant stack is
    dequantized for the call and multiplied by MLX's matmul, which runs
    on the tensor-op units where kq's untransposed gather does not.
    ``GMLX_HY4_KEYS_DENSE=0`` restores the gather."""
    if (getattr(embed_q, "mode", None) == "kquant"
            and latent.dtype == mx.bfloat16
            and env_bool("GMLX_HY4_KEYS_DENSE", True)):
        import mlx_kquant as kq
        w = kq.dequantize(embed_q["weight"], embed_q["scales"],
                          embed_q.kquant_type, latent.dtype)
        return latent @ w[None]
    return embed_q(latent, transpose=False)


@functools.partial(mx.compile, shapeless=True)
def _gated(out: mx.array, gate: mx.array) -> mx.array:
    # One dispatch for the sigmoid and the product.
    return out * mx.sigmoid(gate).astype(out.dtype)


def _sparse_disabled() -> bool:
    """``GMLX_HY4_SPARSE_DISABLE=1`` makes every sparse layer attend its
    whole context. The indexer still runs and writes its cache, so the two
    paths differ in the selection alone. Read per call."""
    return os.environ.get("GMLX_HY4_SPARSE_DISABLE", "") not in ("", "0")


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    # One head count per trunk layer: sparse and sliding-window layers
    # differ.
    num_attention_heads: List[int]
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
    sliding_window: int
    # True on the sliding-window layers.
    sliding_window_pattern: List[bool]
    rope_theta: float = 10000000.0
    rope_theta_swa: float = 10000.0
    moe_latent_size: int = 0
    index_n_heads: int = 0
    index_head_dim: int = 0
    index_topk: int = 0
    index_kpool: int = 0
    index_select_tail: bool = True
    # True on the layers that own an indexer.
    index_types: List[bool] = field(default_factory=list)
    routed_scaling_factor: float = 1.0
    norm_topk_prob: bool = True
    scoring_func: str = "sigmoid"
    swiglu_limit: float = 0.0
    rope_scaling: Optional[Dict] = None
    rope_traditional: bool = True
    max_position_embeddings: int = 1048576
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    num_nextn_predict_layers: int = 0
    # Engram. Multipliers are [module][ngram]; primes and offsets are
    # [module][column], with the offsets rebased into the one shared table.
    engram_layer_ids: List[int] = field(default_factory=list)
    engram_max_ngram_size: int = 0
    engram_n_heads: int = 0
    engram_head_dim: int = 0
    engram_multipliers: List[int] = field(default_factory=list)
    engram_primes: List[int] = field(default_factory=list)
    engram_offsets: List[int] = field(default_factory=list)
    engram_pad_id: int = 0
    engram_token_map: Optional[List[int]] = None
    engram_table_rows: int = 0

    def __post_init__(self):
        if self.scoring_func != "sigmoid":
            raise ValueError(
                f"hy_v4_flash routes sigmoid-only; got {self.scoring_func!r}")
        n = self.num_hidden_layers
        if isinstance(self.num_attention_heads, int):
            self.num_attention_heads = [self.num_attention_heads] * n
        for name in ("num_attention_heads", "sliding_window_pattern"):
            if len(getattr(self, name)) < n:
                raise ValueError(
                    f"hy_v4_flash: {name} has {len(getattr(self, name))} "
                    f"entries for {n} layers")
        if self.index_topk:
            if len(self.index_types) < n:
                raise ValueError(
                    f"hy_v4_flash: index_types has {len(self.index_types)} "
                    f"entries for {n} layers")
            if not self.index_select_tail:
                raise ValueError(
                    "hy_v4_flash: an indexer that does not force the "
                    "query's own group is not supported")
            if self.index_kpool < 2 or self.index_topk % self.index_kpool:
                raise ValueError(
                    f"hy_v4_flash: index_topk {self.index_topk} must be a "
                    f"multiple of index_kpool {self.index_kpool}")
            for i in range(n):
                if self.index_types[i] and self.sliding_window_pattern[i]:
                    raise ValueError(
                        f"hy_v4_flash: layer {i} is a sliding-window layer "
                        "and cannot own an indexer")

    def is_swa(self, layer_idx: int) -> bool:
        return bool(self.sliding_window_pattern[layer_idx])

    def has_indexer(self, layer_idx: int) -> bool:
        return bool(self.index_topk and self.index_types[layer_idx])


# Rope at explicit positions


def rope_at(x: mx.array, positions: mx.array, base: float) -> mx.array:
    """Interleaved-pair rope of ``x`` [..., N, D] at ``positions`` [N].

    The same rotation ``nn.RoPE(traditional=True)`` applies at consecutive
    positions, for rows whose positions are not consecutive (a pooled key
    sits at its group's first position)."""
    d = x.shape[-1]
    freqs = mx.exp(
        -mx.arange(0, d // 2, dtype=mx.float32) * (np.log(base) / (d // 2)))
    ang = positions.astype(mx.float32)[:, None] * freqs
    cos, sin = mx.cos(ang), mx.sin(ang)
    xf = x.astype(mx.float32)
    x1, x2 = xf[..., 0::2], xf[..., 1::2]
    out = mx.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    return out.flatten(-2).astype(x.dtype)


# Engram


class EngramHash(nn.Module):
    """Token ids to n-gram table rows, one set of columns per engram layer.

    The hash runs in uint64, as the reference does: two multipliers times a
    mapped token id pass 2**63, and a signed modulo of the wrapped product
    lands in another bucket. A look-back slot before the start of the
    sequence, or on a token with no map entry, reads the pad id; only that
    slot is replaced. A token with no map entry of its own (an image
    position) still hashes, from the pad id, and is reported dead so the
    engram write is masked.
    """

    DEAD = -1

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.n_modules = len(config.engram_layer_ids)
        self.ngram = int(config.engram_max_ngram_size)
        self.context_len = self.ngram - 1
        n_h = int(config.engram_n_heads)
        self.n_cols = self.context_len * n_h
        if config.engram_token_map is None:
            raise ValueError(
                "hy_v4_flash: engram_token_map is missing from the config; "
                "the hash cannot be computed without it")
        self._token_map = mx.array(config.engram_token_map, dtype=mx.int64)
        self.pad_id = int(config.engram_token_map[config.engram_pad_id])
        self._multipliers = mx.array(
            np.asarray(config.engram_multipliers, dtype=np.uint64)
        ).reshape(self.n_modules, self.ngram)
        self._primes = mx.array(
            np.asarray(config.engram_primes, dtype=np.uint64)
        ).reshape(self.n_modules, self.context_len, n_h)
        self._offsets = mx.array(
            np.asarray(config.engram_offsets, dtype=np.int64)
        ).reshape(self.n_modules, self.n_cols)

    def __call__(self, input_ids, history, token_mask=None):
        """``(row ids [B, L, modules, cols], live [B, L], new history)``.

        ``token_mask`` is False on tokens that take no part in an n-gram.
        ``history`` holds the mapped ids of the last ``ngram - 1`` tokens,
        DEAD where a token had none."""
        B, L = input_ids.shape
        mapped = self._token_map[input_ids.astype(mx.int64)]
        if token_mask is not None:
            mapped = mx.where(token_mask, mapped, self.DEAD)
        if history is None or history.shape[0] != B:
            history = mx.full((B, self.context_len), self.pad_id,
                              dtype=mx.int64)
        full = mx.concatenate([history, mapped], axis=1)
        new_history = mx.contiguous(full[:, -self.context_len:])
        live = mapped != self.DEAD

        tokens = []
        for s in range(self.ngram):
            src = full[:, self.context_len - s: self.context_len - s + L]
            tokens.append(mx.where(src == self.DEAD, self.pad_id, src))
        tokens = mx.stack(tokens, axis=-1).astype(mx.uint64)

        products = tokens[:, :, None, :] * self._multipliers[None, None]
        rolling = products[..., 0]
        cols = []
        for i in range(1, self.ngram):
            rolling = mx.bitwise_xor(rolling, products[..., i])
            cols.append(rolling[..., None] % self._primes[:, i - 1])
        ids = mx.concatenate(cols, axis=-1).astype(mx.int64) + self._offsets
        return ids, live, new_history


class Engram(nn.Module):
    """One n-gram lookup written into the residual streams."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        d = config.hidden_size
        hc = config.hc_mult
        self.hidden_size = d
        self.hc_mult = hc
        self.n_cols = (config.engram_max_ngram_size - 1) * config.engram_n_heads
        self.head_dim = config.engram_head_dim
        # The stream keys and the one value come out of a single
        # projection; unlike deepseek_v41's it has a bias.
        self.wkv = nn.Linear(self.n_cols * self.head_dim, d * (hc + 1),
                             bias=False)
        self.kv_bias = mx.zeros((hc + 1, d), dtype=mx.float32)
        self.gate_weight = mx.ones((hc, d), dtype=mx.float32)
        self.lookup_norm = mx.ones((self.n_cols, self.head_dim),
                                   dtype=mx.float32)
        self.pre_gamma = mx.ones((d,), dtype=mx.float32)
        self.hc_scale = mx.zeros((hc, d), dtype=mx.float32)

    def __call__(self, h: mx.array, rows: mx.array,
                 live: mx.array) -> mx.array:
        """``h`` [B, L, hc, D], ``rows`` [B, L, cols, head_dim], ``live``
        [B, L] bool: False where the token takes no engram write."""
        B, L = rows.shape[0], rows.shape[1]
        hc, d = self.hc_mult, self.hidden_size
        r = mx.fast.rms_norm(rows.astype(mx.float32), None, ENGRAM_EPS)
        r = r * self.lookup_norm
        kv = self.wkv(r.reshape(B, L, -1).astype(rows.dtype))
        kv = kv.astype(mx.float32).reshape(B, L, hc + 1, d) + self.kv_bias
        key = kv[:, :, :hc]
        value = kv[:, :, hc]

        hf = h.astype(mx.float32)
        dot = (mx.fast.rms_norm(hf, None, ENGRAM_EPS)
               * mx.fast.rms_norm(key, None, ENGRAM_EPS)
               * self.gate_weight).sum(axis=-1) * (d ** -0.5)
        # Signed sqrt before the sigmoid; a zero dot keeps the positive
        # root, as copysign does.
        mag = mx.sqrt(mx.maximum(mx.abs(dot), 1e-6))
        gate = mx.sigmoid(mx.where(dot < 0, -mag, mag))

        signal = mx.fast.rms_norm(value, None, ENGRAM_EPS) * self.pre_gamma
        delta = signal[:, :, None, :] * (gate[..., None] + self.hc_scale)
        # The projection bias and the stream offset reach the value path
        # even when every row is masked, so the whole write is dropped.
        delta = mx.where(live[..., None, None], delta, 0.0)
        return (hf + delta).astype(h.dtype)


def _slot_rows(table, row_ids: mx.array) -> list:
    """The rows of each engram layer, [B, L, cols, head_dim], by slot. A
    file-backed table starts one read per slot here: the first engram
    layer waits for its own rows, and the rest come off the drive under
    the layers ahead of their readers."""
    by_slot = getattr(table, "rows_by_slot", None)
    rows = by_slot(row_ids) if by_slot is not None else None
    if rows is None:
        # A table in memory: one gather serves every engram layer.
        every = table(row_ids)
        rows = [every[:, :, slot] for slot in range(row_ids.shape[2])]
    return rows


# Pooled indexer


@mx.compile
def _pool_windows(values: mx.array, gates: mx.array, ape: mx.array) -> mx.array:
    # Per-channel softmax over the slot axis, then the weighted average of
    # the member values. fp32 like the reference.
    g = gates.astype(mx.float32) + ape
    probs = mx.softmax(g, axis=-2, precise=True)
    return (values.astype(mx.float32) * probs).sum(axis=-2)


class HyV4FlashIndexer(nn.Module):
    """Pooled lightning indexer: scores pools, selects whole pools.

    Each token stores a value and a gate, both projections of its normed
    key. A completed group of ``kpool`` tokens pools to one key: the
    gate-weighted average of the values, normed, with rope on the trailing
    ``qk_rope_head_dim`` at the group's first position. A query at
    position t scores the pools that end before its own group
    (``g < t // kpool``) and the caller adds the group itself.

    The store is not gated on the sparse path: below the selection width
    the scoring is skipped, but every token's value and gate must still
    land in the pool cache.
    """

    def __init__(self, args: ModelArgs, rope_theta: float):
        super().__init__()
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.index_topk = args.index_topk
        self.kpool = args.index_kpool
        # One slot of the budget is the query's own group.
        self.select_k = args.index_topk // args.index_kpool - 1
        self.rope_theta = rope_theta
        self.wq_b = nn.Linear(
            args.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.pool_value = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.pool_gate = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.ape = mx.zeros((self.kpool, self.head_dim), dtype=mx.float32)
        self.pool_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.weights_proj = nn.Linear(
            args.hidden_size, self.n_heads, bias=False)
        self._w_scale = (self.head_dim * self.n_heads) ** -0.5
        self._w_scaled = None
        self.rope = initialize_rope(
            dims=args.qk_rope_head_dim,
            base=rope_theta,
            traditional=args.rope_traditional,
            max_position_embeddings=args.max_position_embeddings,
            scaling_config=args.rope_scaling,
        )

    def _pool(self, values, gates, pool_cache, offset):
        B, L, D = values.shape
        if pool_cache is None:
            usable = (L // self.kpool) * self.kpool
            ready_v, ready_g, base = values[:, :usable], gates[:, :usable], offset
        else:
            ready_v, ready_g, base = pool_cache.accumulate_windows(
                values, gates, offset)
        dtype = (_POOL_DTYPE if values.dtype == mx.bfloat16
                 else values.dtype)
        if ready_v.size == 0:
            new_pooled = mx.zeros((B, 0, D), dtype=dtype)
        else:
            vw = mx.unflatten(ready_v, 1, (-1, self.kpool))
            gw = mx.unflatten(ready_g, 1, (-1, self.kpool))
            pooled = self.pool_norm(_pool_windows(vw, gw, self.ape))
            nope, pe = mx.split(pooled, [self.nope_head_dim], axis=-1)
            pos = base + self.kpool * mx.arange(pooled.shape[1])
            new_pooled = mx.concatenate(
                [nope, rope_at(pe, pos, self.rope_theta)], axis=-1
            ).astype(dtype)
        if pool_cache is not None:
            return pool_cache.update_and_fetch(new_pooled)
        return new_pooled

    def _select_kq(self, q, x, pooled, offset: int):
        """The selection on kq's kernels: the pool rows scored where they
        lie and a radix select, in place of a fp32 score of every head
        against the whole pool and a sort per query. The head sum runs in
        fp32 with fp32 head weights and the scores are stored in the
        pool's 16-bit type, so pools tied in 16 bits at the threshold are
        taken lowest index first where the ops path orders them in fp32.
        None where the kernels do not apply."""
        kernels = _kq_indexer_kernels()
        B, H, L, D = q.shape
        chunk = None
        if L > _IDX_KQ_MAX_L:
            chunk = _kq_indexer_chunk_kernel()
        if (kernels is None or (L > _IDX_KQ_MAX_L and chunk is None)
                or pooled.dtype not in (mx.float16, mx.bfloat16)
                or q.dtype not in (mx.float16, mx.bfloat16)
                or mx.default_device().type != mx.DeviceType.gpu
                or not env_bool("GMLX_HY4_IDX_KQ", True)):
            return None
        score, select = kernels
        proj = self.weights_proj.weight
        held = self._w_scaled
        if held is None or held[0] is not proj:
            # The scale folded into the transposed weight once.
            held = (proj, (proj.astype(mx.float32) * self._w_scale).T)
            # Not a parameter: kept out of the module's own dict.
            object.__setattr__(self, "_w_scaled", held)
        w = x.astype(mx.float32) @ held[1]
        if chunk is not None:
            return self._select_chunk(chunk, select, q, w, pooled, offset)
        if L == 1:
            # The kernel scores every row it is given to one query: hand
            # it the pools that end before the query's group.
            keys, q_offset = pooled[:, :offset // self.kpool], 0
        else:
            # Row p is visible to query j when p < (offset + j) // kpool.
            keys, q_offset = pooled, offset - 1
        scores = score(q.astype(keys.dtype), keys, w, q_offset, self.kpool)
        sel = select(scores, self.select_k)[:, 0]
        return mx.stop_gradient(sel.astype(mx.int32))

    def _select_chunk(self, score, select, q, w, pooled, offset: int):
        """A prefill chunk's selection. The score kernel works in tiles of
        64 rows by 64 keys, so both pad: a padded key lies past the pools
        of every row and a padded row is dropped. A row that sees fewer
        pools than it picks is handed pools it cannot see, as on the ops
        path, held inside the pool rows."""
        L, P = q.shape[2], pooled.shape[1]
        keys = pooled[:, None]
        q = q.astype(keys.dtype)
        pad_n, pad_l = -P % 64, -L % 64
        if pad_n:
            keys = mx.pad(keys, [(0, 0), (0, 0), (0, pad_n), (0, 0)])
        if pad_l:
            q = mx.pad(q, [(0, 0), (0, 0), (0, pad_l), (0, 0)])
            w = mx.pad(w, [(0, 0), (0, pad_l), (0, 0)])
        # Row p is visible to query j when p < (offset + j) // kpool.
        scores = score(q, keys, w, causal=True, causal_q_offset=offset - 1,
                       causal_ratio=self.kpool)
        sel = select(scores, self.select_k)[:, 0, :L].astype(mx.int32)
        if pad_n:
            sel = mx.minimum(sel, P - 1)
        return mx.stop_gradient(sel)

    def __call__(
        self,
        x: mx.array,
        qr: mx.array,
        pool_cache: Optional[DS4PoolingCache],
        offset: int,
    ) -> Optional[mx.array]:
        """Store this chunk's values and gates; return the selected pool
        ordinals ``[B, L, select_k]`` when the selection binds, else None."""
        B, L, _ = x.shape
        ik = self.k_norm(self.wk(x))
        pooled = self._pool(
            self.pool_value(ik), self.pool_gate(ik), pool_cache, offset)

        if offset + L <= self.index_topk:
            return None
        if pooled.shape[1] < self.select_k:
            return None

        q = self.wq_b(qr).reshape(B, L, self.n_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)
        nope, pe = mx.split(q, [self.nope_head_dim], axis=-1)
        q = mx.concatenate([nope, self.rope(pe, offset)], axis=-1)

        if (self.n_heads == 16 and self.head_dim == 128 and self.kpool > 1
                and self.select_k == 511 and "weight" in self.weights_proj):
            sel = self._select_kq(q, x, pooled, offset)
            if sel is not None:
                return sel

        # fp32 head weights: they are sign-free, and a bf16 head gate moves
        # scores enough to flip near-tied pools.
        w = (x.astype(mx.float32) @ self.weights_proj.weight.T
             if "weight" in self.weights_proj
             else self.weights_proj(x.astype(mx.float32)))
        w = w * self._w_scale

        n_pool = pooled.shape[1]
        pool_idx = mx.arange(n_pool)
        pooled_t = pooled[:, None].swapaxes(-1, -2).astype(mx.float32)
        sel_parts = []
        for j0 in range(0, L, 512):
            j1 = min(j0 + 512, L)
            s = q[:, :, j0:j1].astype(mx.float32) @ pooled_t
            s = mx.maximum(s, 0)
            s = (s * w[:, j0:j1].swapaxes(-1, -2)[..., None]).sum(axis=1)
            q_pos = offset + mx.arange(j0, j1)
            visible = pool_idx < (q_pos // self.kpool)[:, None]
            s = mx.where(visible[None], s, mx.finfo(s.dtype).min)
            sel_parts.append(
                mx.argpartition(-s, kth=self.select_k - 1, axis=-1)[
                    ..., :self.select_k])
        sel = sel_parts[0] if len(sel_parts) == 1 else mx.concatenate(
            sel_parts, axis=1)
        return mx.stop_gradient(sel)


# Attention


class HyV4FlashAttention(nn.Module):
    """Gated absorbed MLA with per-head sinks, in three forms: a sparse
    layer with a pooled indexer, a sliding-window layer, and a full layer
    with neither.

    The rope half never enters the SDPA call: its scores are passed as the
    float ``mask`` argument, and the bool visibility folds into them,
    because mx.fast SDPA deviates when a bool mask and a sinks term meet.
    """

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.num_heads = args.num_attention_heads[layer_idx]
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.q_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.scale = self.q_head_dim**-0.5
        self.window = args.sliding_window if args.is_swa(layer_idx) else 0
        self.kpool = args.index_kpool
        theta = args.rope_theta_swa if self.window else args.rope_theta

        hidden = args.hidden_size
        self.q_a_proj = nn.Linear(hidden, args.q_lora_rank, bias=False)
        self.q_a_layernorm = nn.RMSNorm(args.q_lora_rank, eps=args.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            args.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden, args.kv_lora_rank + args.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = nn.RMSNorm(
            args.kv_lora_rank, eps=args.rms_norm_eps)
        self.embed_q = MultiLinear(
            self.qk_nope_head_dim, args.kv_lora_rank, self.num_heads)
        self.unembed_out = MultiLinear(
            args.kv_lora_rank, self.v_head_dim, self.num_heads)
        self.attn_gate = nn.Linear(
            hidden, self.num_heads * self.v_head_dim, bias=False)
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim, hidden, bias=False)
        self.post_norm = nn.RMSNorm(hidden, eps=args.rms_norm_eps)
        self.sinks = mx.zeros((self.num_heads,), dtype=mx.float32)

        self.indexer = (HyV4FlashIndexer(args, theta)
                        if args.has_indexer(layer_idx) else None)
        self.rope = initialize_rope(
            dims=args.qk_rope_head_dim,
            base=theta,
            traditional=args.rope_traditional,
            max_position_embeddings=args.max_position_embeddings,
            scaling_config=args.rope_scaling,
        )

    def _attend(self, q_nope, q_pe, latent, k_pe, mask):
        """Absorbed attention of ``q`` [B, H, L, .] over ``latent`` /
        ``k_pe`` [B, 1, S, .]. ``mask`` is bool with trailing axes [L, S]
        (True = attend) or None. Returns [B, H, L, v_head_dim]."""
        L, S = q_nope.shape[2], latent.shape[2]
        if L > 1 and (L > _STREAM_Q or S > _STREAM_MIN_KEYS):
            bmask = mask
            if bmask is not None and bmask.ndim == 3:
                bmask = bmask[:, None]
            out = _tiled_absorbed_attention(
                self.embed_q(q_nope), q_pe, latent, k_pe, bmask,
                self.scale, self.sinks)
            return self.unembed_out(out)

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            if mask.ndim == 3:
                mask = mask[:, None]
            pe_scores = mx.where(
                mask, pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype))
        sinks = self.sinks.astype(q_nope.dtype)
        if L == 1:
            out = mx.fast.scaled_dot_product_attention(
                self.embed_q(q_nope), latent, latent,
                scale=self.scale, mask=pe_scores, sinks=sinks)
            return self.unembed_out(out)
        k = self.embed_q(latent, transpose=False)
        v = self.unembed_out(latent)
        return mx.fast.scaled_dot_product_attention(
            q_nope, k, v, scale=self.scale, mask=pe_scores, sinks=sinks)

    def _window_prefill(self, q_nope, q_pe, latent, k_pe):
        """A sliding-window chunk on kq's window kernel, which reads only
        the key blocks inside each query block's window. The kernel takes
        per-head keys and values, so the latent is expanded for the window
        rows. None where the kernel does not apply."""
        B, H, L, _ = q_nope.shape
        kernel = _kq_window_kernel()
        if (kernel is None or L < _MIN_ROWS
                or self.q_head_dim not in _HD_OK
                or self.v_head_dim != self.q_head_dim
                or q_nope.dtype not in (mx.float16, mx.bfloat16)
                or mx.default_device().type != mx.DeviceType.gpu
                or not env_bool("GMLX_SWA_KQ", True)):
            return None
        S = latent.shape[2]
        q = mx.concatenate([q_nope, q_pe], axis=-1)
        k = mx.concatenate(
            [_head_keys(self.embed_q, latent),
             mx.broadcast_to(k_pe, (B, H, S, k_pe.shape[-1]))], axis=-1)
        v = self.unembed_out(latent)
        return kernel(q, k, v, self.scale, self.window, sinks=self.sinks)

    def _kq_ok(self, kernel, q_nope) -> bool:
        return (kernel is not None and self.kv_lora_rank == 512
                and self.qk_rope_head_dim == 64
                and q_nope.dtype in (mx.float16, mx.bfloat16)
                and mx.default_device().type == mx.DeviceType.gpu
                and env_bool("GMLX_HY4_SPARSE_KQ", True))

    def _picked_rows(self, sel, q_pos, B: int, L: int):
        """Row list [B, L, select_k * kpool] of the picked pools that end
        before each query's group; -1 for a pool the query cannot see."""
        r = self.kpool
        own = (q_pos // r)[None, :, None]
        pos = (sel[..., None] * r + mx.arange(r)).reshape(B, L, -1)
        seen = mx.broadcast_to(
            (sel < own)[..., None], sel.shape + (r,)).reshape(B, L, -1)
        return mx.where(seen, pos, -1)

    def _step_kq(self, q_nope, q_pe, kv, sel, offset: int, base: int = 0,
                 q=None):
        """A decode step (a few queries) as one kq kernel call over the
        cache rows: the window rows read whole, the picked rows through an
        index list, the rope half as the score-only tail. Row 0 of ``kv``
        is position ``base``; ``q`` is the query rows when the caller has
        built them. None where the kernel does not apply."""
        kernel = _kq_tail_kernel("sdpa_sparse_decode")
        if not self._kq_ok(kernel, q_nope):
            return None
        B, H, L, _ = q_nope.shape
        rows = kv.shape[2]
        r = self.kpool
        if q is None:
            q = mx.concatenate([self.embed_q(q_nope), q_pe], axis=-1)
        q_pos = offset + mx.arange(L)
        # Window rows every query may read, from the first query's bound.
        low = None
        if self.window:
            start = max(0, offset + 1 - self.window)
            low = q_pos - self.window + 1
        elif sel is not None:
            start = offset // r * r
            low = q_pos // r * r
        else:
            start = 0
        # A cache that rolls may hold less than the window after a restore.
        start = max(start, base)
        lo = start - base
        win = kv[:, :, lo:] if lo else kv
        win_mask = None
        if L > 1:
            key_pos = start + mx.arange(rows - lo)
            win_mask = key_pos <= q_pos[:, None]
            if low is not None:
                win_mask = win_mask & (key_pos >= low[:, None])
        if sel is not None:
            pool = kv[:, 0]
            if L == 1:
                # One query past the key budget sees more pools than it
                # picks, so every picked pool ends before its group.
                idx = (sel[..., None] * r + _row_offsets(r)).reshape(B, 1, -1)
            else:
                idx = self._picked_rows(sel, q_pos, B, L)
            idx = idx.astype(mx.int32)
        else:
            pool = win[:, 0, :1]
            idx = _no_rows(B, L)
        out = kernel(q, win, pool, idx, self.scale, sinks=self.sinks,
                     win_mask=win_mask, score_tail=64)
        return self.unembed_out(out)

    def _sparse_prefill_kq(self, q_nope, q_pe, kv, sel, offset: int):
        """The chunk on kq's sparse prefill kernel, which reads each
        query's listed rows where they lie. None where it does not apply."""
        kernel = _kq_tail_kernel("sdpa_sparse_prefill")
        if not self._kq_ok(kernel, q_nope):
            return None
        B, H, L, _ = q_nope.shape
        r = self.kpool
        q = mx.concatenate([self.embed_q(q_nope), q_pe], axis=-1)
        q_pos = offset + mx.arange(L)
        own = (q_pos // r)[None, :, None]
        # Listed rows: the picked pools that end before the query's group,
        # then the group's rows before the query. The query's own row is
        # the kernel's band of one.
        tail = (own * r + mx.arange(r - 1))
        tail = mx.broadcast_to(tail, (B, L, r - 1))
        idx = mx.concatenate(
            [self._picked_rows(sel, q_pos, B, L),
             mx.where(tail < q_pos[None, :, None], tail, -1)],
            axis=-1).astype(mx.int32)
        out = kernel(q, kv, kv[:, 0], idx, self.scale, 1,
                     sinks=self.sinks, score_tail=64)
        return self.unembed_out(out)

    def _sparse_prefill(self, q_nope, q_pe, latent, k_pe, sel, offset: int):
        """Sparse attention for a chunk: each query attends its selected
        pools and its own group up to itself. The keys are gathered a pool
        at a time, a tile of queries per step, so the work per query is
        the key budget whatever the depth."""
        B, H, L, _ = q_nope.shape
        S, R = latent.shape[2], latent.shape[3]
        r = self.kpool
        kv = mx.concatenate([latent[:, 0], k_pe[:, 0]], axis=-1)
        if S % r:
            kv = mx.pad(kv, [(0, 0), (0, -S % r), (0, 0)])
        W = kv.shape[-1]
        kv = kv.reshape(B, -1, r * W)
        q_abs = self.embed_q(q_nope)
        sinks = self.sinks.astype(q_nope.dtype)
        neg = mx.array(mx.finfo(q_pe.dtype).min, q_pe.dtype)
        rows = mx.arange(B)[:, None, None]
        within = mx.arange(r)
        outs = []
        for j0 in range(0, L, _GATHER_TILE):
            j1 = min(j0 + _GATHER_TILE, L)
            Q = j1 - j0
            q_pos = offset + mx.arange(j0, j1)
            own = (q_pos // r)[None, :, None]
            picked = sel[:, j0:j1]
            pools = mx.concatenate(
                [picked, mx.broadcast_to(own, (B, Q, 1))], axis=-1)
            if outs:
                # One tile's gathered keys at a time.
                pools = mx.depends([pools], [outs[-1]])[0]
            g = kv[rows, pools].reshape(B * Q, 1, -1, W)
            lat_g, pe_g = mx.split(g, [R], axis=-1)
            # A picked pool counts when it ends before the query's group;
            # the group itself counts up to the query.
            seen = mx.concatenate(
                [picked < own, mx.ones((B, Q, 1), dtype=mx.bool_)], axis=-1)
            pos = pools[..., None] * r + within
            keep = seen[..., None] & (pos <= q_pos[None, :, None, None])
            qn = q_abs[:, :, j0:j1].transpose(0, 2, 1, 3).reshape(
                B * Q, H, 1, R)
            qp = q_pe[:, :, j0:j1].transpose(0, 2, 1, 3).reshape(
                B * Q, H, 1, -1)
            pe_scores = (qp * self.scale) @ pe_g.swapaxes(-1, -2)
            pe_scores = mx.where(
                keep.reshape(B * Q, 1, 1, -1), pe_scores, neg)
            out = mx.fast.scaled_dot_product_attention(
                qn, lat_g, lat_g, scale=self.scale, mask=pe_scores,
                sinks=sinks)
            outs.append(out.reshape(B, Q, H, R).transpose(0, 2, 1, 3))
        out = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)
        return self.unembed_out(out)

    def _sparse_keys(self, sel, q_abs: int, B: int):
        """Key positions one query attends: its selected pools expanded
        ``kpool`` wide, then its own group up to itself. A pool the query
        cannot see (handed out when fewer than select_k are visible) is
        masked; its members lie in the group or after it."""
        r = self.kpool
        tail_start = q_abs // r * r
        tok = (sel[..., None] * r + mx.arange(r)).reshape(B, -1)
        keep = mx.broadcast_to(
            (sel < tail_start // r)[..., None], sel.shape + (r,)
        ).reshape(B, -1)
        tok = mx.where(keep, tok, 0)
        tail = mx.broadcast_to(
            mx.arange(tail_start, q_abs + 1)[None], (B, q_abs + 1 - tail_start))
        tok = mx.concatenate([tok, tail], axis=1)
        keep = mx.concatenate(
            [keep, mx.ones(tail.shape, dtype=mx.bool_)], axis=1)
        return tok, keep

    def _sparse_mask(self, sel_pools, offset: int, L: int, S: int):
        """Bool visibility [B, L, S]: causal and (own group or a member of
        a selected pool). A selected pool the query cannot see has only
        members in its group or after it, which the other terms cover."""
        B = sel_pools.shape[0]
        r = self.kpool
        q_pos = offset + mx.arange(L)[:, None]
        key_pos = mx.arange(S)
        causal = key_pos <= q_pos
        tail = key_pos >= q_pos // r * r
        tok = (sel_pools[..., None] * r + mx.arange(r)).reshape(B, L, -1)
        sel = mx.zeros((B, L, S), dtype=mx.bool_)
        sel = mx.put_along_axis(
            sel, mx.minimum(tok, S - 1), mx.array(True), axis=-1)
        return (causal & tail)[None] | (causal[None] & sel)

    def __call__(self, x: mx.array, cache: Optional[Any] = None) -> mx.array:
        B, L, _ = x.shape

        qr = self.q_a_layernorm(self.q_a_proj(x))
        q_flat = self.q_b_proj(qr)
        q = q_flat.reshape(
            B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed = self.kv_a_proj_with_mqa(x)
        kv_cache, pool_cache = split_cache(cache)
        offset = kv_cache.offset if kv_cache is not None else 0
        # One row per key, the latent then its rope half, so a kernel reads
        # both in place. The cache's value slot holds nothing.
        q_rows = None
        if (L <= _GATHER_MAX_L
                and self._kq_ok(_kq_tail_kernel("sdpa_sparse_decode"), q_nope)
                and q_flat.dtype == compressed.dtype
                and row_kernels.eligible(
                    compressed, self.kv_a_layernorm, self.rope,
                    self.kv_lora_rank, self.qk_rope_head_dim)):
            # A decode step: the key row and the query rows, a kernel each.
            kv = row_kernels.kv_row(
                compressed, self.kv_a_layernorm, self.rope, offset)
            q_rows = row_kernels.q_row(
                self.embed_q(q_nope).astype(q_flat.dtype), q_flat,
                self.rope, offset)
        else:
            compressed, k_pe = mx.split(
                compressed, [self.kv_lora_rank], axis=-1)
            k_pe = k_pe.reshape(
                B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
            latent = mx.expand_dims(self.kv_a_layernorm(compressed), axis=1)
            q_pe = self.rope(q_pe, offset)
            k_pe = self.rope(k_pe, offset)
            kv = mx.concatenate([latent, k_pe], axis=-1)
        # Row 0 of the fetched rows is position ``base``: 0 for a cache
        # that keeps every key, the window's first key for one that rolls.
        base = 0
        if kv_cache is not None:
            kv, _ = kv_cache.update_and_fetch(
                kv, _zero_values(B, L, kv.dtype))
            base = kv_cache.offset - kv.shape[2]
        S = kv.shape[2]

        sel = None
        if self.indexer is not None:
            sel = self.indexer(x, qr, pool_cache, offset)
            if pool_cache is not None and kv_cache is not None:
                # Keep the pool write on the graph when the selection goes
                # unused, so the graph does not grow across steps.
                buf = pool_cache.buf_kv
                if buf is not None:
                    depends_keys(kv_cache, (buf, pool_cache.buf_gate))
            if _sparse_disabled():
                sel = None

        out = None
        if L <= _GATHER_MAX_L:
            out = self._step_kq(
                q_nope, q_pe, kv, sel, offset, base, q=q_rows)
        elif sel is not None:
            out = self._sparse_prefill_kq(q_nope, q_pe, kv, sel, offset)
        latent, k_pe = mx.split(kv, [self.kv_lora_rank], axis=-1)
        if out is not None:
            pass
        elif self.window:
            # Sliding window: a query sees itself and the window - 1 keys
            # before it. One query over a rolled cache reads every row it
            # is handed, in whatever order the ring holds them.
            start = max(0, offset + 1 - self.window, base)
            lo = start - base
            if lo:
                latent, k_pe = latent[:, :, lo:], k_pe[:, :, lo:]
            out = self._window_prefill(q_nope, q_pe, latent, k_pe)
            if out is None:
                mask = None
                if L > 1:
                    q_pos = offset + mx.arange(L)[:, None]
                    key_pos = start + mx.arange(S - lo)
                    mask = ((key_pos <= q_pos)
                            & (key_pos > q_pos - self.window))
                    mask = mask[None, None]
                out = self._attend(q_nope, q_pe, latent, k_pe, mask)
        elif sel is not None and L <= _GATHER_MAX_L:
            # Sparse decode: gather each query's keys and attend those.
            outs = []
            for j in range(L):
                tok, keep = self._sparse_keys(sel[:, j], offset + j, B)
                lat_g = mx.take_along_axis(
                    latent[:, 0], tok[..., None], axis=1)[:, None]
                pe_g = mx.take_along_axis(
                    k_pe[:, 0], tok[..., None], axis=1)[:, None]
                outs.append(self._attend(
                    q_nope[:, :, j:j + 1], q_pe[:, :, j:j + 1], lat_g, pe_g,
                    keep[:, None, None, :]))
            out = outs[0] if L == 1 else mx.concatenate(outs, axis=2)
        elif sel is not None:
            out = self._sparse_prefill(q_nope, q_pe, latent, k_pe, sel, offset)
        else:
            mask = None
            if L > 1:
                q_pos = offset + mx.arange(L)[:, None]
                mask = (mx.arange(S) <= q_pos)[None, None]
            out = self._attend(q_nope, q_pe, latent, k_pe, mask)

        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        out = _gated(out, self.attn_gate(x))
        return self.post_norm(self.o_proj(out))


class HyV4FlashKVCache(HyV4KVCache):
    """The key rows of one layer, appended in place. A slice write copies
    the whole buffer whenever the step before still holds it, which is
    every step of a pipelined decode: a copy of the cache per layer per
    token, growing with depth. Rows at or past the offset are never
    handed out, which is what the in-place append needs
    (gmlx/cache/kv_append.py)."""

    _kv_routed = True
    _kv_grid = True


def split_cache(cache):
    """``(latent_cache, pool_cache)`` for one layer's cache entry. A layer
    with an indexer holds a ``CacheList``; the first engram layer's list
    also ends with the token history, which is not returned here."""
    if cache is None:
        return None, None
    inner = getattr(cache, "caches", None)
    if inner is None:
        return cache, None
    pool = next((c for c in inner[1:] if isinstance(c, DS4PoolingCache)), None)
    return inner[0], pool


# MoE


class HyV4FlashMoEGate(nn.Module):
    """Sigmoid router over the full-width input: the correction bias steers
    selection only, weights are the unbiased scores, renormalized then
    scaled."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.n_routed_experts = args.n_routed_experts
        self.routed_scaling_factor = args.routed_scaling_factor
        self.norm_topk_prob = args.norm_topk_prob
        self.weight = mx.zeros(
            (args.n_routed_experts, args.hidden_size), dtype=mx.float32)
        self.e_score_correction_bias = mx.zeros(
            (args.n_routed_experts,), dtype=mx.float32)

    def __call__(self, x: mx.array):
        if not self.training and row_kernels.router_eligible(x, self.weight):
            logits = row_kernels.router_logits(x, self.weight)
        else:
            logits = x.astype(mx.float32) @ self.weight.T
        t = logits.size // self.n_routed_experts
        kernel = _kq_router_kernel()
        # The kernel has no backward and no CPU form.
        if (kernel is not None and t * self.top_k < 64 and not self.training
                and mx.default_device().type == mx.DeviceType.gpu
                and env_bool("GMLX_HY4_KQ_ROUTER", True)):
            # A decode step: score, pick, renormalize and scale in one
            # dispatch. Ties at the boundary go to the lower index here
            # and by argpartition order below; fp32 scores make them rare.
            inds, weights = kernel(
                logits.reshape(t, self.n_routed_experts),
                self.top_k,
                self.norm_topk_prob,
                shared_gate=False,
                bias=self.e_score_correction_bias,
                scoring="sigmoid",
                scale=self.routed_scaling_factor,
            )
            shp = logits.shape[:-1]
            return (inds.reshape(*shp, self.top_k),
                    weights.reshape(*shp, self.top_k))
        return _expert_select(
            logits,
            self.e_score_correction_bias,
            self.top_k,
            self.routed_scaling_factor,
            self.norm_topk_prob,
            "sigmoid",
        )


class HyV4FlashMoE(nn.Module):
    """Routed experts at the latent width, plus a full-width shared
    expert. Only the routed experts are clamped."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        latent = args.moe_latent_size or args.hidden_size
        self.gate = HyV4FlashMoEGate(args)
        if args.moe_latent_size:
            self.routed_down = nn.Linear(args.hidden_size, latent, bias=False)
            self.routed_up = nn.Linear(latent, args.hidden_size, bias=False)
        self.switch_mlp = SwitchGLU(
            latent,
            args.moe_intermediate_size,
            args.n_routed_experts,
            activation=LimitedSwiGLU(args.swiglu_limit),
        )
        self.shared_experts = HyV4MLP(
            args, args.moe_intermediate_size * args.n_shared_experts)

    def __call__(self, x: mx.array) -> mx.array:
        inds, scores = self.gate(x)
        xr = self.routed_down(x) if "routed_down" in self else x
        if getattr(self.switch_mlp, "_kq_mix_scores", False):
            y = self.switch_mlp(xr, inds, scores)
        else:
            y = self.switch_mlp(xr, inds)
            if y.ndim == scores.ndim + 1:
                y = (y * scores[..., None].astype(y.dtype)).sum(-2)
        if "routed_up" in self:
            y = self.routed_up(y)
        return y + self.shared_experts(x)


# Trunk


class HyV4FlashDecoderLayer(nn.Module):
    """Two iHC cycles per block, each sublayer followed by its own norm."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = HyV4FlashAttention(args, layer_idx)
        self.mlp = (
            HyV4MLP(args) if layer_idx < args.first_k_dense_replace
            else HyV4FlashMoE(args))
        self.input_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps)
        self.ffn_post_norm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps)
        self.attn_hc = IndependentHyperConnection(args)
        self.ffn_hc = IndependentHyperConnection(args)
        self.engram = None
        self.engram_slot = None

    def __call__(self, h: mx.array, cache: Optional[Any] = None) -> mx.array:
        return self.close(self.body(h, None, cache))

    def body(self, h: mx.array, pending, cache: Optional[Any] = None):
        """The block up to its last expand, which it hands back unapplied
        as ``(y, streams, post)``. ``pending`` is the layer before's, and
        ``h`` is unused then: each expand runs in the dispatch of the
        collapse after it."""
        if pending is None:
            x, post = self.attn_hc.pre_norm(h, self.input_layernorm)
        else:
            h, x, post = self.attn_hc.expand_pre_norm(
                *pending, self.input_layernorm)
        x = self.self_attn(x, cache=cache)
        h, x, post = self.ffn_hc.expand_pre_norm(
            x, h, post, self.post_attention_layernorm)
        return self.ffn_post_norm(self.mlp(x)), h, post

    def close(self, pending) -> mx.array:
        """The streams after a ``body``'s last expand."""
        return self.ffn_hc.expand(*pending)


class HyV4FlashModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [HyV4FlashDecoderLayer(args, i)
                       for i in range(args.num_hidden_layers)]
        self.hc_head = HyperHead(args)
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.engram_hash = None
        self.engram_embed = None
        if args.engram_layer_ids:
            self.engram_hash = EngramHash(args)
            # One table holds every engram layer's buckets.
            self.engram_embed = EngramTable(
                args.engram_table_rows, args.engram_head_dim)
            for slot, layer_id in enumerate(args.engram_layer_ids):
                self.layers[layer_id].engram = Engram(args)
                self.layers[layer_id].engram_slot = slot

    def _history_slot(self, cache):
        """The engram token history, held on the first engram layer."""
        ids = self.args.engram_layer_ids
        if not ids or cache is None or cache[ids[0]] is None:
            return None
        inner = getattr(cache[ids[0]], "caches", None)
        if inner is None:
            return None
        member = inner[-1]
        return member if isinstance(member, ArraysCache) else None

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
        input_embeddings: Optional[mx.array] = None,
        token_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """``token_mask`` is False on positions that take no part in an
        n-gram and get no engram write (image positions)."""
        if cache is None:
            cache = [None] * len(self.layers)

        rows = live = None
        if self.engram_hash is not None:
            slot = self._history_slot(cache)
            history = slot[0] if slot is not None else None
            row_ids, live, new_history = self.engram_hash(
                inputs, history, token_mask)
            if slot is not None:
                slot[0] = new_history
            # A file-backed table reads its rows off-thread from here on,
            # under the layers that run before their engram.
            pf = getattr(self.engram_embed, "prefetch", None)
            if pf is not None:
                pf(row_ids)
            if live is not None:
                rows = _slot_rows(self.engram_embed, row_ids)

        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)
        # hc_mult exact copies of the embedding.
        h = mx.contiguous(mx.broadcast_to(
            h[:, :, None, :],
            (h.shape[0], h.shape[1], self.args.hc_mult, h.shape[2])))

        pending = prev = None
        for layer, layer_cache in zip(self.layers, cache):
            if layer.engram is not None and live is not None:
                if pending is not None:
                    h, pending = prev.close(pending), None
                h = layer.engram(h, rows[layer.engram_slot], live)
            pending, prev = layer.body(h, pending, layer_cache), layer
        if pending is not None:
            h = prev.close(pending)

        return self.norm(self.hc_head(h))


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = HyV4FlashModel(args)
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
        args = self.args
        first_engram = args.engram_layer_ids[:1]
        caches = []
        roll = env_bool("GMLX_HY4_SWA_ROLL", True)
        for idx in range(args.num_hidden_layers):
            # A sliding-window layer keeps its window alone, so a long
            # context costs it nothing past that.
            slots: List[Any] = [
                RotatingKVCache(max_size=args.sliding_window)
                if roll and args.is_swa(idx) else HyV4FlashKVCache()]
            if args.has_indexer(idx):
                # The score path reads the pool in full every step, so it
                # opts out of --kv-bits packing.
                pool = DS4PoolingCache(args.index_kpool, lookback=False)
                pool.quantizable = False
                slots.append(pool)
            if idx in first_engram:
                slots.append(ArraysCache(1))
            caches.append(slots[0] if len(slots) == 1 else CacheList(*slots))
        return caches

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
            if ".attn_hc." in path or ".ffn_hc." in path:
                return False
            if path.startswith("model.hc_head."):
                return False
            if path.endswith("self_attn.sinks"):
                return False
            if "e_score_correction_bias" in path:
                return False
            if path.endswith("mlp.gate.weight"):
                return False
            if path.endswith(("indexer.weights_proj.weight", "indexer.ape")):
                return False
            if ".engram." in path and not path.endswith("wkv.weight"):
                return False
            return True

        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("indexer.weights_proj"):
                return False
            if path.endswith(("indexer.wq_b", "indexer.wk",
                              "indexer.pool_value", "indexer.pool_gate")):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate


def _prefill_score_profile(model, prompt_cache):
    """What a prefill chunk's scores hold, for the chunk sizing in
    ``prefill_decay``. No layer scores a query against the whole depth: a
    window layer sees its window, a sparse layer the keys its indexer
    picked, and the indexer one row per pool of ``index_kpool`` tokens. On
    kq's chunk kernel the head sum runs inside, so a chunk holds one
    16-bit score per query and pool, priced as fp32 for the select beside
    it. The ops path scores 512 queries at a time against every head in
    fp32. A batched pool keeps the dense model."""
    for c in _prefill_decay._walk_caches(prompt_cache):
        if isinstance(c, DS4BatchPoolingCache):
            return None
    cfg = getattr(model, "config", model)
    kpool = int(_prefill_decay._config_value(cfg, "index_kpool") or 0)
    heads = int(_prefill_decay._config_value(cfg, "index_n_heads") or 0)
    if kpool < 2 or heads < 1:
        return None
    if (_kq_indexer_kernels() is not None
            and _kq_indexer_chunk_kernel() is not None
            and env_bool("GMLX_HY4_IDX_KQ", True)
            and mx.default_device().type == mx.DeviceType.gpu):
        return _prefill_decay.ScoreTransientProfile(
            heads=1, bytes_per_elem=4, depth_divisor=kpool)
    return _prefill_decay.ScoreTransientProfile(
        heads=heads + 1, bytes_per_elem=4, depth_divisor=kpool, rows=512)


_prefill_decay.register_score_profile("hy_v4_flash", _prefill_score_profile)
