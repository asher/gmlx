# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Model for Thinking Machines Inkling 2 (GGUF arch ``inkling2`` and
``inkling2_loop``).

Ported from llama.cpp ``src/models/inkling2.cpp`` and
``src/models/inkling2-loop.cpp`` (MIT). Each physical layer runs:

  1. Block attention residuals: the inputs of attention and of the FFN are
     a softmax mix over depth of the rms-normed earlier block outputs and
     the current stream, with learned queries. Full-attention layers mix
     separate inputs for Q, K and V.
  2. Attention with no RoPE. A relative-position bias ``R . B[d]`` with a
     per-query sink, an L2 clip on Q, K and R, a log-scaled position factor
     on the full layers' Q and R, rms-normed V, and a sigmoid output gate.
  3. Causal short convolutions on K, V and both residual branches. Full
     layers add input-dependent taps per head on K and V.
  4. A sigmoid-routed MoE with situ experts.

A looped model runs its core layers several passes with shared weights.
Each pass restarts from the prelude output and reads the previous pass
through four cross-pass messages, gated into the Q, K, V and FFN inputs.
Every layer visit keeps its own KV and convolution state.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from gmlx._vendor.mlx_lm.models.base import BaseModelArgs
from gmlx._vendor.mlx_lm.models.switch_layers import SwitchGLU

from gmlx.cache.attention import create_ssm_mask
from gmlx.cache.kv_cache import (
    ArraysCache,
    BatchKVCache,
    BatchRotatingKVCache,
    BufferedRotatingKVCache,
    KVCache,
    RotatingKVCache,
)
from gmlx.load.row_fusion import fuse_float_rows, fuse_kquant_rows, split_rows
from gmlx.models.inkling2 import row_kernels
from gmlx.models.kimi_k3 import SituActivation
from gmlx.upstream.attn_hd512 import note_route

SLIDING = "sliding_attention"
FULL = "full_attention"

# The four cross-pass message controllers, in msg_query row order.
CONTROLLERS = ("q", "k", "v", "ffn")

# A prefill chunk at least this wide evaluates each visit as it finishes,
# so one chunk does not hold every visit's intermediates at once.
PREFILL_EVAL_ROWS = 64

# The bias tensor of one attention call stays under this many elements;
# wider calls split the queries.
_BIAS_ELEMENTS = 1 << 25
# Mask elements per banded call: its mask is built in the query dtype with
# no gather index, so it holds more elements in fewer bytes.
_BANDED_ELEMENTS = 1 << 26

# The epsilon of the unweighted norms in the depth mix and the controllers.
_MIX_EPS = 1e-6


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    unpadded_vocab_size: int
    layer_types: List[str]
    layer_visits: List[int] = field(default_factory=list)
    loop: Optional[Dict[str, int]] = None
    sliding_window: int = 1024
    rel_length: int = 16
    rel_extent: int = 1024
    rel_clip: float = 8.0
    qk_clip: float = 22.627416
    log_scale_floor: int = 8192
    log_scale_alpha: float = 0.1
    attn_res_block_size: int = 6
    conv_L_cache: int = 4
    num_experts: int = 128
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 1152
    expert_weights_scale: float = 1.0
    situ_beta: float = 4.0
    situ_linear_beta: float = 25.0
    use_gate_bias: bool = True
    final_logit_softcapping: float = 0.0
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 1048576
    tie_word_embeddings: bool = False
    intermediate_size: int = 0

    def __post_init__(self):
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"inkling2: layer_types has {len(self.layer_types)} entries "
                f"for {self.num_hidden_layers} layers")
        if not self.layer_visits:
            self.layer_visits = list(range(self.num_hidden_layers))
        if self.loop and self.loop.get("varying"):
            raise NotImplementedError(
                "inkling2: variable loop counts are not supported")


def _rms(x: mx.array, eps: float = _MIX_EPS) -> mx.array:
    """Unweighted RMS norm in float32."""
    if row_kernels.norm_eligible(x, None):
        return row_kernels.rms_norm(x, None, eps, mx.float32)
    return mx.fast.rms_norm(x.astype(mx.float32), None, eps)


# Decode runs these small chains about a hundred times per token each, so
# they are compiled: one kernel launch for a chain instead of one per op.


@lru_cache(maxsize=None)
def _clipper(scaled: bool, tau: bool, dtype):
    def run(x, scale, threshold, factor):
        x = x.astype(mx.float32)
        if scaled:
            x = x * scale.astype(mx.float32)
        norm = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True)) + 1e-6
        x = x * mx.clip(threshold / norm, 0.0, 1.0)
        if tau:
            x = x * factor
        return x.astype(dtype)

    return mx.compile(run)


def clip(x: mx.array, scale: Optional[mx.array], threshold: float,
         tau: Optional[mx.array] = None, dtype=mx.float32) -> mx.array:
    """Scale each row (last axis) down to an L2 norm of at most
    ``threshold``, after the optional per-channel ``scale``. ``tau`` then
    multiplies the float32 result, which is cast to ``dtype``."""
    if row_kernels.clip_eligible(x, scale, tau):
        return row_kernels.clip(x, scale, threshold, tau, dtype,
                                _clip_ops(float(threshold), dtype))
    t = mx.array(threshold, dtype=mx.float32)
    run = _clipper(scale is not None, tau is not None, dtype)
    return run(x, scale, t, tau)


@lru_cache(maxsize=None)
def _clip_ops(threshold: float, dtype):
    def ops(x, scale, tau):
        t = mx.array(threshold, dtype=mx.float32)
        return _clipper(scale is not None, tau is not None, dtype)(
            x, scale, t, tau)

    return ops


def depth_mix(blocks: List[mx.array], cur: mx.array,
              query: mx.array) -> mx.array:
    """Attention over depth: a softmax over ``blocks`` (already rms-normed,
    see ``push_block``) and the rms-normed ``cur``, with ``query`` rows
    [nq, D]. Returns [B, L, nq, D] float32."""
    if row_kernels.mix_eligible(blocks, cur, query):
        return row_kernels.depth_mix(blocks, cur, query, _MIX_EPS, _depth_mix)
    return _depth_mix(tuple(blocks), cur, query)


@mx.compile
def _depth_mix(blocks: tuple, cur: mx.array, query: mx.array) -> mx.array:
    states = mx.stack([*blocks, _rms(cur)], axis=-2)  # [B, L, n, D]
    q = query.astype(mx.float32).reshape(-1, states.shape[-1])
    # Depth on the last axis, so the softmax is one kernel.
    scores = (q @ mx.swapaxes(states, -1, -2)) * (
        states.shape[-1] ** -0.5)                      # [B, L, nq, n]
    return mx.softmax(scores, axis=-1) @ states        # [B, L, nq, D]


def push_block(blocks: List[mx.array], state: mx.array) -> None:
    """Add a block checkpoint. It is stored rms-normed, as every depth mix
    reads it, so the norm runs once per block, not once per mix."""
    blocks.append(_rms(state))


def log_scale(positions: mx.array, floor: int, alpha: float) -> mx.array:
    """1 + alpha * log(max(1, (pos + 1) / floor)), float32."""
    t = (positions.astype(mx.float32) + 1.0) / float(floor)
    return 1.0 + alpha * mx.log(mx.maximum(t, 1.0))


class _Norm(nn.RMSNorm):
    """RMSNorm with a float32 weight; the output keeps the input dtype."""

    def __call__(self, x):
        if row_kernels.norm_eligible(x, self.weight):
            return row_kernels.rms_norm(x, self.weight, self.eps, x.dtype)
        return mx.fast.rms_norm(
            x.astype(mx.float32), self.weight.astype(mx.float32),
            self.eps).astype(x.dtype)


class Controller(nn.Module):
    """One cross-pass message controller:
    ``(1 - tanh(F h)) * cur + tanh(G h) * msg``,
    ``h = D [rms(cur); rms(msg)]``."""

    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.down = nn.Linear(2 * dim, rank, bias=False)
        self.gain = nn.Linear(rank, dim, bias=False)
        self.forget = nn.Linear(rank, dim, bias=False)

    def __call__(self, cur: mx.array, msg: mx.array) -> mx.array:
        mods = (self.down, self.gain, self.forget)
        if all(type(m) is nn.Linear and "bias" not in m for m in mods):
            w = (self.down.weight, self.gain.weight, self.forget.weight)
            if row_kernels.control_eligible(cur, msg, *w):
                return row_kernels.control(cur, msg, *w, _MIX_EPS, _control)
            return _control(cur, msg, *w)
        h = self.down(mx.concatenate([_rms(cur), _rms(msg)], axis=-1))
        return _blend(cur, msg, self.gain(h), self.forget(h))


@mx.compile
def _blend(cur, msg, gain, forget):
    gain = mx.tanh(gain.astype(mx.float32))
    forget = mx.tanh(forget.astype(mx.float32))
    c = cur.astype(mx.float32)
    return (c - forget * c + gain * msg.astype(mx.float32)).astype(cur.dtype)


@mx.compile
def _control(cur, msg, down, gain, forget):
    h = mx.concatenate([_rms(cur), _rms(msg)], axis=-1) @ down.T
    return _blend(cur, msg, h @ gain.T, h @ forget.T)


class LoopControl(nn.Module):
    """The message queries and controllers of one core layer."""

    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.msg_query = mx.zeros((len(CONTROLLERS), dim), dtype=mx.float32)
        self.q = Controller(dim, rank)
        self.k = Controller(dim, rank)
        self.v = Controller(dim, rank)
        self.ffn = Controller(dim, rank)


def _conv_taps(window: mx.array, w: mx.array, L: int) -> mx.array:
    """Static taps of a short convolution: ``w[:, k]`` reads window row
    t + k for output row t; the last tap is the current input."""
    w = w.astype(mx.float32)
    out = window[:, 0:L] * w[:, 0]
    for k in range(1, w.shape[1]):
        out = out + window[:, k:k + L] * w[:, k]
    return out


@mx.compile
def _conv_static(state: mx.array, x: mx.array, w: mx.array):
    """(window, output): the kept inputs ``state`` followed by ``x``, and
    the convolution plus its residual, scaled, in the dtype of ``x``
    (static taps only)."""
    xf = x.astype(mx.float32)
    window = mx.concatenate([state.astype(mx.float32), xf], axis=1)
    out = _conv_taps(window, w, xf.shape[1])
    return window, ((xf + out) * (w.shape[1] + 1) ** -0.5).astype(x.dtype)


@mx.compile
def _conv_dynamic(state: mx.array, x: mx.array, w: mx.array,
                  dynamic: mx.array):
    """As ``_conv_static``, plus per-head input-dependent taps; lag 0 is
    the current input."""
    xf = x.astype(mx.float32)
    window = mx.concatenate([state.astype(mx.float32), xf], axis=1)
    B, L, C = xf.shape
    width = w.shape[1]
    n_keep = width - 1
    out = _conv_taps(window, w, L)
    n_heads = dynamic.shape[-1] // width
    taps = dynamic.astype(mx.float32).reshape(B, L, n_heads, width)
    for lag in range(width):
        past = window[:, n_keep - lag:n_keep - lag + L]
        past = past.reshape(B, L, n_heads, C // n_heads)
        out = out + (past * taps[..., lag:lag + 1]).reshape(B, L, C)
    return window, ((xf + out) * (2 * width + 1) ** -0.5).astype(x.dtype)


@mx.compile
def _residual(a: mx.array, b: mx.array) -> mx.array:
    """``a + b`` in float32, in the dtype of ``a``."""
    return (a.astype(mx.float32) + b.astype(mx.float32)).astype(a.dtype)


@mx.compile
def _gate(out: mx.array, g: mx.array) -> mx.array:
    return out * mx.sigmoid(g.astype(mx.float32)).astype(out.dtype)


def _iota(n: int) -> mx.array:
    """A cached int32 arange of at least ``n``; slice it to length. A
    slice is a view, so a decode step builds no index arrays."""
    size = 1 << max(10, (n - 1).bit_length())
    return _iota_of(size)


@lru_cache(maxsize=None)
def _iota_of(size: int) -> mx.array:
    # Not evaluated here: training builds this under a transform, where an
    # eval raises. The first step that reads it evaluates it.
    return mx.arange(size, dtype=mx.int32)


def _key_distances(cache, n_queries: int, n_keys: int, rows=None) -> mx.array:
    """[L, K] distances pos_q - pos_k from the queries of this update to
    every key slot the cache returned; ``rows`` (start, stop) selects some
    of the queries. Left padding shifts queries and keys alike, so buffer
    coordinates give the distance."""
    r0, r1 = rows if rows is not None else (0, n_queries)
    i = _iota(r1)[r0:r1, None]
    s = _iota(n_keys)[None, :n_keys]
    # Query i sits n_queries - 1 - i positions before the newest key.
    back = n_queries - 1 - i
    kind = type(cache)
    # A buffered sliding cache returns its keys in time order, newest last.
    if cache is None or kind in (KVCache, BatchKVCache,
                                 BufferedRotatingKVCache):
        if kind is BatchKVCache and getattr(cache, "_row_ends", None) is not None:
            raise NotImplementedError(
                "inkling2: rows with different ends are not supported")
        return (n_keys - 1 - s) - back
    if kind in (RotatingKVCache, BatchRotatingKVCache):
        # The newest key sits at slot _idx - 1; the slots before it run
        # back in time and wrap.
        idx = int(cache._idx)
        return mx.remainder(idx - 1 - s, n_keys) - back
    raise NotImplementedError(
        f"inkling2: attention does not support cache type {kind.__name__}")


@lru_cache(maxsize=None)
def _zero_sinks(n_heads: int, dtype) -> mx.array:
    """The zero sink logit per head, built once from host data, so a step
    launches no fill for it and nothing evaluates inside a transform."""
    return mx.array([0.0] * n_heads, dtype=dtype)


@lru_cache(maxsize=None)
def _masker(extent: int, dtype):
    """Compiled mask build for ``Attention._mask``. Shapeless, so a cache
    that grows by one key per step does not retrace."""
    def run(r, sink, table, dist, valid):
        scores = r @ table.T - sink[..., None]           # [B, L, H, E+1]
        index = mx.where((dist >= 0) & (dist < extent), dist, extent)
        bias = mx.take_along_axis(scores, index[None, :, None, :], axis=-1)
        return mx.where(valid, bias.transpose(0, 2, 1, 3),
                        -mx.inf).astype(dtype)

    return mx.compile(run, shapeless=True)


# Caches whose keys come back in time order, newest last.
_TIME_ORDERED = (KVCache, BatchKVCache, BufferedRotatingKVCache)


def _refuse_quantized(cache) -> None:
    """The relative bias and the sink need full-precision keys."""
    if cache is None:
        return
    if getattr(cache, "bits", None) is not None or "Quant" in type(
            cache).__name__ or "KVarN" in type(cache).__name__:
        raise NotImplementedError(
            "inkling2: KV cache quantization is not supported for this "
            "model (relative-position attention needs full keys)")


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.is_sliding = args.layer_types[layer_idx] == SLIDING
        self.window = args.sliding_window if self.is_sliding else None
        self.extent = args.rel_extent
        self.qk_clip = args.qk_clip
        self.rel_clip_threshold = args.rel_clip
        self.rel_length = args.rel_length
        self.width = args.conv_L_cache
        self.eps = args.rms_norm_eps
        self.scale = self.head_dim ** -0.5
        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim

        # Row-fused projection groups (install_row_fusion): slot ->
        # (member names, member rows). Kept out of the parameter tree.
        object.__setattr__(self, "_row_groups", {})
        object.__setattr__(self, "_table_cache", None)
        kind = "sliding" if self.is_sliding else "full"
        object.__setattr__(self, "_route_tags", (
            f"inkling2_banded_{kind}", f"inkling2_dense_{kind}"))
        self.hidden_size = dim
        self.q_proj = nn.Linear(dim, q_dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, dim, bias=False)
        self.gate_proj = nn.Linear(dim, q_dim, bias=False)
        self.rel_proj = nn.Linear(dim, self.n_heads * self.rel_length,
                                  bias=False)
        self.sink_proj = nn.Linear(dim, self.n_heads, bias=False)
        if not self.is_sliding:
            self.k_conv_proj = nn.Linear(dim, self.width * self.n_kv_heads,
                                         bias=False)
            self.v_conv_proj = nn.Linear(dim, self.width * self.n_kv_heads,
                                         bias=False)
        self.q_clip = mx.ones((self.n_heads, self.head_dim), dtype=mx.float32)
        self.rel_clip = mx.ones((self.n_heads, self.rel_length),
                                dtype=mx.float32)
        self.rel_bias = mx.zeros((self.extent, self.rel_length),
                                 dtype=mx.float32)
        self.k_conv = mx.zeros((kv_dim, self.width), dtype=mx.float32)
        self.v_conv = mx.zeros((kv_dim, self.width), dtype=mx.float32)
        self.post_conv = mx.zeros((dim, self.width), dtype=mx.float32)

    def project(self, x: mx.array, names: tuple) -> list:
        """The projections ``names`` of ``x``, in order. A fused group
        whose members are all asked for runs as one matmul; the rest run
        alone."""
        out = {}
        for slot, (members, sizes) in self._row_groups.items():
            fused = self.get(slot)
            if fused is None or not all(n in names for n in members):
                continue
            out.update(zip(members, split_rows(fused, sizes, x)))
        return [out[n] if n in out else self[n](x) for n in names]

    def _mask(self, r: mx.array, sink: mx.array, dist: mx.array,
              valid: mx.array, dtype) -> mx.array:
        """[B, H, L, K] additive mask in ``dtype``: the bias
        ``R . B[d] - sink`` where ``valid``, with the zero pad row for
        distances outside [0, extent), and -inf elsewhere."""
        return _masker(self.extent, dtype)(r, sink, self._padded_table(),
                                           dist, valid)

    def _padded_table(self) -> mx.array:
        """``rel_bias`` in float32 with the zero pad row, built once per
        weight array. The entry holds the array it came from, so a new
        weight (training, a load) misses; it is not evaluated here, since
        an eval under a transform raises."""
        held = self._table_cache
        if held is None or held[0] is not self.rel_bias:
            table = mx.pad(self.rel_bias.astype(mx.float32), [(0, 1), (0, 0)])
            held = (self.rel_bias, table)
            object.__setattr__(self, "_table_cache", held)
        return held[1]

    def attend(self, q, k, v, r, sink, cache, pad_mask) -> mx.array:
        """Attention with the folded bias and a zero sink logit. ``q`` is
        [B, H, L, D]; returns [B, H, L, D]."""
        if cache is None or type(cache) in _TIME_ORDERED:
            return self._attend_banded(q, k, v, r, sink, cache, pad_mask)
        return self._attend_dense(q, k, v, r, sink, cache, pad_mask)

    def _attend_banded(self, q, k, v, r, sink, cache, pad_mask) -> mx.array:
        """``attend`` for keys in time order, newest last. Query row i of
        this update sits at key slot K - L + i. A key at least ``extent``
        slots behind every query of a sub-chunk takes the bias ``-sink``,
        so that region is filled with the constant, and only the near band
        gathers from the bias table. Keys outside the window are dropped.
        The mask values equal those of ``_attend_dense``."""
        B, H, L, _ = q.shape
        K = k.shape[2]
        _key_distances(cache, L, K, (0, 1))  # refuses unsupported rows
        sinks = _zero_sinks(H, q.dtype)
        outs = []
        i0 = 0
        while i0 < L:
            # First key any query of the sub-chunk can see.
            start = 0
            if self.window is not None:
                start = max(0, K - L + i0 - self.window + 1)
            step = max(1, _BANDED_ELEMENTS // max(1, B * H * (K - start)))
            i1 = min(L, i0 + step)
            # Keys before ``near`` are far for every query of the sub-chunk.
            near = min(K, max(start, K - L + i0 - self.extent + 1))
            dist = _key_distances(cache, L, K, (i0, i1))[:, near:]
            valid = dist >= 0
            if self.window is not None:
                valid = valid & (dist < self.window)
            pm = None if pad_mask is None else pad_mask[..., i0:i1, :]
            if pm is not None:
                valid = valid & pm[..., near:]
            parts = [self._mask(r[:, i0:i1], sink[:, i0:i1], dist, valid,
                                q.dtype)]
            if near > start:
                far = (-sink[:, i0:i1]).astype(q.dtype).transpose(0, 2, 1)
                far = mx.broadcast_to(far[..., None],
                                      (B, H, i1 - i0, near - start))
                if pm is not None:
                    far = mx.where(pm[..., start:near], far,
                                   mx.array(-mx.inf, dtype=q.dtype))
                parts.insert(0, far)
            mask = parts[0] if len(parts) == 1 else mx.concatenate(parts, -1)
            kk, vv = (k, v) if start == 0 else (k[:, :, start:], v[:, :, start:])
            qq = q[:, :, i0:i1]
            note_route(self._route_tags[0], qq, kk, mask)
            outs.append(mx.fast.scaled_dot_product_attention(
                qq, kk, vv, scale=self.scale, mask=mask,
                sinks=sinks))
            i0 = i1
        return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)

    def _attend_dense(self, q, k, v, r, sink, cache, pad_mask) -> mx.array:
        """``attend`` for any supported cache: the bias of every key slot
        from its distance."""
        B, H, L, _ = q.shape
        K = k.shape[2]
        sinks = _zero_sinks(H, q.dtype)
        step = max(1, _BIAS_ELEMENTS // max(1, B * H * K))
        outs = []
        for i0 in range(0, L, step):
            i1 = min(L, i0 + step)
            dist = _key_distances(cache, L, K, (i0, i1))
            valid = dist >= 0
            if self.window is not None:
                valid = valid & (dist < self.window)
            if pad_mask is not None:
                valid = valid & pad_mask[..., i0:i1, :]
            mask = self._mask(r[:, i0:i1], sink[:, i0:i1], dist, valid,
                              q.dtype)
            qq = q[:, :, i0:i1]
            note_route(self._route_tags[1], qq, k, mask)
            outs.append(mx.fast.scaled_dot_product_attention(
                qq, k, v, scale=self.scale, mask=mask,
                sinks=sinks))
        return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)


class Layer(nn.Module):
    """One physical layer: attention, the FFN, and their depth mixes."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        dim = args.hidden_size
        self.layer_idx = layer_idx
        self.is_sliding = args.layer_types[layer_idx] == SLIDING
        self.block_size = args.attn_res_block_size
        self.width = args.conv_L_cache
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.n_heads = args.num_attention_heads
        self.self_attn = Attention(args, layer_idx)
        if self.is_sliding:
            self.input_layernorm = _Norm(dim, eps=args.rms_norm_eps)
        else:
            self.q_inp_norm = _Norm(dim, eps=args.rms_norm_eps)
            self.k_inp_norm = _Norm(dim, eps=args.rms_norm_eps)
            self.v_inp_norm = _Norm(dim, eps=args.rms_norm_eps)
        if layer_idx > 0:
            self.attn_res_query = mx.zeros(
                (1 if self.is_sliding else 3, dim), dtype=mx.float32)
        self.ffn_res_query = mx.zeros((1, dim), dtype=mx.float32)
        self.post_attention_layernorm = _Norm(dim, eps=args.rms_norm_eps)
        self.mlp = MoE(args)
        self.ffn_post_conv = mx.zeros((dim, self.width), dtype=mx.float32)
        loop = args.loop
        if loop and loop["start"] <= layer_idx < loop["end"]:
            self.loop = LoopControl(dim, loop["gate_rank"])

    def _convolve(self, x, weight, dynamic, cache, slot, pad_mask,
                  sink=None):
        """Causal short convolution with a residual. ``dynamic`` adds
        per-head input-dependent taps. ``cache`` slot ``slot`` keeps the
        last ``width - 1`` inputs. ``sink`` receives (cache, slot, window)
        for a speculative rollback."""
        B, L, C = x.shape
        n_keep = self.width - 1
        if pad_mask is not None:
            x = mx.where(pad_mask[..., None], x, 0)
        state = None if cache is None else cache[slot]
        if state is None:
            state = mx.zeros((B, n_keep, C), dtype=mx.float32)
        if dynamic is None:
            window, out = _conv_static(state, x, weight)
        else:
            window, out = _conv_dynamic(state, x, weight, dynamic)
        if cache is not None:
            cache.update_window(slot, window, n_keep, lengths=cache.lengths)
            if sink is not None:
                sink.append((cache, slot, window))
        return out

    def __call__(self, h, ip, blocks, cache, pad_mask, positions,
                 messages=None, conv_sink=None):
        """Run this layer as physical layer ``ip`` of the block schedule.
        ``cache`` is (conv ArraysCache, attention cache) for this visit."""
        conv_cache, kv_cache = cache if cache is not None else (None, None)
        attn = self.self_attn
        B, L, D = h.shape
        inp_sa = h

        if hasattr(self, "attn_res_query"):
            mixed = depth_mix(blocks, h, self.attn_res_query)
            if self.is_sliding:
                inputs = [mixed[:, :, 0]] * 3
            else:
                inputs = [mixed[:, :, j] for j in range(3)]
            inputs = [x.astype(h.dtype) for x in inputs]
        else:
            inputs = [h, h, h]

        if messages is not None:
            ctrl = (self.loop.q, self.loop.k, self.loop.v)
            inputs = [c(x, messages[:, :, j]) for j, (c, x) in
                      enumerate(zip(ctrl, inputs))]

        if self.is_sliding:
            if messages is not None:
                inputs = [self.input_layernorm(x) for x in inputs]
            else:
                inputs = [self.input_layernorm(inputs[0])] * 3
        else:
            inputs = [self.q_inp_norm(inputs[0]), self.k_inp_norm(inputs[1]),
                      self.v_inp_norm(inputs[2])]

        q, r, sink, gate = attn.project(
            inputs[0], ("q_proj", "rel_proj", "sink_proj", "gate_proj"))
        k_dyn = v_dyn = None
        if not self.is_sliding:
            k, k_dyn = attn.project(inputs[1], ("k_proj", "k_conv_proj"))
            v, v_dyn = attn.project(inputs[2], ("v_proj", "v_conv_proj"))
        elif inputs[1] is inputs[2]:
            k, v = attn.project(inputs[1], ("k_proj", "v_proj"))
        else:
            k, v = attn.k_proj(inputs[1]), attn.v_proj(inputs[2])
        k = self._convolve(k, attn.k_conv, k_dyn, conv_cache, 0, pad_mask,
                           conv_sink)
        v = self._convolve(v, attn.v_conv, v_dyn, conv_cache, 1, pad_mask,
                           conv_sink)

        dtype = h.dtype
        tau = None if self.is_sliding else positions[..., None, None]
        q = clip(q.reshape(B, L, self.n_heads, self.head_dim), attn.q_clip,
                 attn.qk_clip, tau, dtype)
        k = clip(k.reshape(B, L, self.n_kv_heads, self.head_dim), None,
                 attn.qk_clip, None, dtype)
        v = _rms(v.reshape(B, L, self.n_kv_heads, self.head_dim), attn.eps)
        r = clip(r.reshape(B, L, self.n_heads, attn.rel_length),
                 attn.rel_clip, attn.rel_clip_threshold, tau)
        sink = sink.astype(mx.float32)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.astype(dtype).transpose(0, 2, 1, 3)
        _refuse_quantized(kv_cache)
        key_mask = None
        if kv_cache is not None:
            # A batch cache's mask covers the keys the update returns, so
            # it is built before the update.
            key_mask = _pad_key_mask(kv_cache, L)
            k, v = kv_cache.update_and_fetch(k, v)
        out = attn.attend(q, k, v, r, sink, kv_cache, key_mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        out = _gate(out, gate)
        out = self._convolve(attn.o_proj(out), attn.post_conv, None,
                             conv_cache, 2, pad_mask, conv_sink)

        if ip % self.block_size == 0:
            push_block(blocks, inp_sa)
            h = out
        else:
            h = _residual(inp_sa, out)
        ffn_inp = h

        x = depth_mix(blocks, h, self.ffn_res_query)[:, :, 0].astype(dtype)
        if messages is not None:
            x = self.loop.ffn(x, messages[:, :, 3])
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = self._convolve(x, self.ffn_post_conv, None, conv_cache, 3,
                           pad_mask, conv_sink)
        if conv_cache is not None:
            conv_cache.advance(L)
        return _residual(ffn_inp, x)


def _pad_key_mask(cache, n_queries: int):
    """[B, 1, L, K] key validity of a batch cache's left padding, or None.
    Call it before the cache update."""
    if type(cache) not in (BatchKVCache, BatchRotatingKVCache):
        return None
    mask = cache.make_mask(n_queries, return_array=True)
    if mask is None or isinstance(mask, str):
        return None
    return mask


@lru_cache(maxsize=None)
def _router(k: int, scale: float, biased: bool, dtype):
    """Compiled routing: gate logits (and the selection bias) to the top-k
    expert ids and their weights, normalized and scaled, in ``dtype``."""
    def run(logits, bias):
        scores = mx.sigmoid(logits.astype(mx.float32))
        select = scores
        if biased:
            select = scores + bias.astype(mx.float32)
        inds = mx.stop_gradient(
            mx.argpartition(-select, kth=k - 1, axis=-1)[..., :k])
        w = mx.take_along_axis(scores, inds, axis=-1)
        w = w / (mx.sum(w, axis=-1, keepdims=True) + 1e-20)
        return inds, (w * scale).astype(dtype)

    return mx.compile(run)


class MoE(nn.Module):
    """Sigmoid-routed experts with a selection bias, normalized weights
    times ``expert_weights_scale``, and situ."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        if args.use_gate_bias:
            self.e_score_correction_bias = mx.zeros((args.num_experts,),
                                                    dtype=mx.float32)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts,
            activation=SituActivation(args.situ_beta, args.situ_linear_beta))

    def __call__(self, x: mx.array) -> mx.array:
        from gmlx.stream.moe_experts import expert_controls_active

        args = self.args
        logits = self.gate(x.astype(mx.float32))
        bias = self.e_score_correction_bias if args.use_gate_bias else None
        route = _router(args.num_experts_per_tok, args.expert_weights_scale,
                        bias is not None, x.dtype)
        if row_kernels.router_eligible(logits, args.num_experts_per_tok):
            inds, weights = row_kernels.router(
                logits, bias, args.num_experts_per_tok,
                args.expert_weights_scale, x.dtype)
        else:
            inds, weights = route(logits, bias)

        if expert_controls_active(self):
            from gmlx.stream.moe_experts import _apply_expert_controls

            scores = mx.sigmoid(logits.astype(mx.float32))

            def weights_at(inds):
                w = mx.take_along_axis(scores, inds, axis=-1)
                w = w / (mx.sum(w, axis=-1, keepdims=True) + 1e-20)
                return (w * args.expert_weights_scale).astype(x.dtype)

            inds, weights = _apply_expert_controls(
                self, inds, weights, weights_at)

        if (getattr(self.switch_mlp, "_kq_mix_scores", False)
                or getattr(self.switch_mlp, "_kq_scores_sink", False)):
            y = self.switch_mlp(x, inds, weights)
        else:
            y = self.switch_mlp(x, inds)
        if y.ndim == weights.ndim + 1:
            y = (y * weights[..., None]).sum(axis=-2)
        return y


class Inkling2Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        dim = args.hidden_size
        self.embed_tokens = nn.Embedding(args.vocab_size, dim)
        self.embed_norm = _Norm(dim, eps=args.rms_norm_eps)
        self.layers = [Layer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = _Norm(dim, eps=args.rms_norm_eps)
        self.output_res_query = mx.zeros((1, dim), dtype=mx.float32)
        self.visits = list(args.layer_visits)
        loop = args.loop
        self.loop = loop
        if loop:
            self.core = (loop["start"], loop["end"])
            self.passes = loop["passes"]

    def _positions(self, cache, L: int) -> mx.array:
        """The log-scale factor per row and query, [B or 1, L] float32."""
        offset = 0
        if cache is not None:
            for c in cache:
                if c is not None and not isinstance(c, ArraysCache):
                    offset = c.offset
                    break
        if isinstance(offset, mx.array):
            pos = offset.reshape(-1, 1) + _iota(L)[None, :L]
        else:
            pos = (offset + _iota(L)[:L])[None, :]
        return log_scale(pos, self.args.log_scale_floor,
                         self.args.log_scale_alpha)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
        capture_layer_ids=None,
        hidden_sink: Optional[list] = None,
        embed_mask: Optional[mx.array] = None,
        trunk_sink: Optional[list] = None,
        conv_sink: Optional[list] = None,
    ):
        if input_embeddings is None:
            h = self.embed_norm(self.embed_tokens(inputs))
        elif embed_mask is None:
            # Token embeddings, as embed_tokens gives them.
            h = self.embed_norm(input_embeddings)
        else:
            # embed_mask marks the rows that take the norm. Image rows go in
            # as the vision tower made them.
            h = input_embeddings
            h = mx.where(embed_mask[..., None], self.embed_norm(h), h)
        B, L, _ = h.shape
        n_visits = len(self.visits)
        per_visit = cache is not None and L >= PREFILL_EVAL_ROWS
        conv_caches = cache[:n_visits] if cache is not None else None
        kv_caches = cache[n_visits:] if cache is not None else None
        # Positions for the log scale come from the attention caches.
        positions = self._positions(kv_caches, L)
        pad_mask = None
        if conv_caches is not None:
            m = create_ssm_mask(h, conv_caches[0])
            pad_mask = m if isinstance(m, mx.array) else None

        capture = set(capture_layer_ids) if capture_layer_ids else ()
        blocks: List[mx.array] = []
        prelude = prelude_blocks = None
        bank = bank_out = None
        start, end = self.core if self.loop else (0, 0)
        core = end - start

        for visit, ip in enumerate(self.visits):
            pass_idx = 0
            if self.loop and start <= ip < end:
                pass_idx = (visit - start) // core
                if ip == start:
                    if pass_idx == 0:
                        prelude, prelude_blocks = h, list(blocks)
                    else:
                        # The previous pass's blocks after its first core
                        # block, and its output, feed the messages.
                        bank = blocks[len(prelude_blocks) + 1:]
                        bank_out = h
                        h, blocks = prelude, list(prelude_blocks)
            if hidden_sink is not None and visit in capture:
                hidden_sink.append(h)
            layer = self.layers[ip]
            messages = None
            if pass_idx > 0:
                messages = depth_mix(bank, bank_out, layer.loop.msg_query)
            c = None if cache is None else (conv_caches[visit],
                                            kv_caches[visit])
            h = layer(h, ip, blocks, c, pad_mask, positions, messages,
                      conv_sink)
            if per_visit:
                mx.eval(h)

        h = depth_mix(blocks, h, self.output_res_query)[:, :, 0].astype(
            h.dtype)
        if trunk_sink is not None:
            trunk_sink.append(h)
        h = self.norm(h)
        if hidden_sink is not None and n_visits in capture:
            hidden_sink.append(h)
        return h


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Inkling2Model(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size,
                                 bias=False)

    def head(self, h: mx.array) -> mx.array:
        """Logits over the unpadded vocabulary, softcapped in float32."""
        logits = self.lm_head(h).astype(mx.float32)
        cap = self.args.final_logit_softcapping
        if cap:
            logits = mx.tanh(logits / cap) * cap
        return logits[..., : self.args.unpadded_vocab_size]

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
        capture_layer_ids=None,
        hidden_sink: Optional[list] = None,
        embed_mask: Optional[mx.array] = None,
    ):
        return self.head(self.model(
            inputs, cache, input_embeddings,
            capture_layer_ids=capture_layer_ids, hidden_sink=hidden_sink,
            embed_mask=embed_mask))

    def sanitize(self, weights):
        """Split the looped layout's packed message queries per core layer:
        rows (q[2i], kv[2i], kv[2i+1], q[2i+1]) are the Q, K, V and FFN
        queries of core layer start + i."""
        q = weights.pop("model.loop_res_query", None)
        kv = weights.pop("model.loop_kv_res_query", None)
        loop = self.args.loop
        if (q is None) != (kv is None):
            raise ValueError("inkling2: the GGUF has one of the two packed "
                             "loop message query tensors")
        if q is not None:
            if not loop:
                raise ValueError("inkling2: loop message queries in a model "
                                 "with no loop")
            n_core = loop["end"] - loop["start"]
            if q.shape[0] != 2 * n_core or kv.shape[0] != 2 * n_core:
                raise ValueError(
                    f"inkling2: packed loop message queries have "
                    f"{q.shape[0]} and {kv.shape[0]} rows for {n_core} core "
                    f"layers")
            for i in range(n_core):
                rows = mx.stack([q[2 * i], kv[2 * i], kv[2 * i + 1],
                                 q[2 * i + 1]])
                weights[f"model.layers.{loop['start'] + i}.loop.msg_query"] = \
                    rows
        for name in list(weights):
            if name.endswith(("attn_res_query", "ffn_res_query",
                              "output_res_query")):
                arr = weights[name]
                if arr.ndim == 1:
                    weights[name] = arr.reshape(1, -1)
        return weights

    @property
    def layers(self):
        return self.model.layers

    @property
    def n_visits(self) -> int:
        return len(self.model.visits)

    def make_cache(self):
        """One convolution state per visit, then one attention cache per
        visit, in execution order."""
        convs = [ArraysCache(size=4) for _ in self.model.visits]
        kvs = []
        for ip in self.model.visits:
            if self.layers[ip].is_sliding:
                c = RotatingKVCache(max_size=self.args.sliding_window)
            else:
                c = KVCache()
            # The relative bias needs float keys: the KV policy drops
            # --kv-bits with a named reason.
            c.kv_quant_unsupported = True
            kvs.append(c)
        return convs + kvs

    @property
    def cast_predicate(self):
        # Kept in step with the loader's fp32 pins (loader.fp32_keep_for).
        keep = FP32_KEEP

        def predicate(path: str):
            return not any(k in path for k in keep)

        return predicate


# The parameters that stay float32: routing, the depth-mix and message
# queries, the clip scales, the bias tables, the convolution taps and the
# loop controllers (the reference keeps the controllers float32).
FP32_KEEP = (
    ".mlp.gate.weight", ".e_score_correction_bias", "_res_query",
    ".msg_query", ".q_clip", ".rel_clip", ".rel_bias", "_conv",
    ".loop.", "norm.weight",
)


# Projection groups that read one input, tried in order; the first group
# that fuses claims its members. Float groups past the row cap stay split:
# MLX's float mat-vec loses to two launches there (deepseek_v4 note).
_ROW_GROUPS = (
    ("_q_fused", ("q_proj", "gate_proj", "rel_proj", "sink_proj")),
    ("_q_fused", ("q_proj", "gate_proj", "rel_proj")),
    ("_q_fused", ("rel_proj", "sink_proj")),
    ("_kv_fused", ("k_proj", "v_proj")),
    ("_kc_fused", ("k_proj", "k_conv_proj")),
    ("_vc_fused", ("v_proj", "v_conv_proj")),
)
_MAX_FLOAT_ROWS = 1024
# k and v read one input on sliding layers only; the conv projections exist
# on full layers only.
_SLIDING_ONLY = ("_kv_fused",)
_FULL_ONLY = ("_kc_fused", "_vc_fused")


def install_row_fusion(model) -> int:
    """Fuse each attention's same-input projections into one matmul per
    group (gmlx.load.row_fusion). k and v share an input only on sliding
    layers, and only outside loop passes, where ``Layer`` checks it.
    Idempotent: an attention that already has groups is left alone.
    Returns the number of groups fused. GMLX_GEMV_FUSE=0 disables."""
    import os

    if os.environ.get("GMLX_GEMV_FUSE", "1") == "0":
        return 0
    inner = getattr(model, "model", model)
    root = "model." if inner is not model else ""
    members: set = set()
    n = 0
    for li, layer in enumerate(getattr(inner, "layers", ())):
        attn = layer.self_attn
        if attn._row_groups:
            continue
        taken: set = set()
        for slot, names in _ROW_GROUPS:
            if slot in attn._row_groups or taken & set(names):
                continue
            if slot in _SLIDING_ONLY and not attn.is_sliding or (
                    slot in _FULL_ONLY and attn.is_sliding):
                continue
            mods = [attn.get(name) for name in names]
            if any(m is None for m in mods):
                continue
            fused = fuse_kquant_rows(mods, attn.hidden_size)
            if fused is None and sum(
                    m.weight.shape[0] for m in mods) <= _MAX_FLOAT_ROWS:
                fused = fuse_float_rows(mods)
            if fused is None:
                continue
            mx.eval([m.weight for m in mods])
            setattr(attn, slot, fused)
            attn._row_groups[slot] = (
                names, tuple(m.weight.shape[0] for m in mods))
            taken |= set(names)
            path = f"{root}layers.{li}.self_attn"
            members |= {f"{path}.{name}.weight" for name in names}
            n += 1
    if members:
        from gmlx.stream.pin_weights import FUSED_MEMBERS_ATTR

        prev = getattr(model, FUSED_MEMBERS_ATTR, None) or frozenset()
        object.__setattr__(model, FUSED_MEMBERS_ATTR, prev | members)
    return n
