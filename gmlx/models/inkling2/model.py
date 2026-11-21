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
from gmlx.models.kimi_k3 import SituActivation

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
    return mx.fast.rms_norm(x.astype(mx.float32), None, eps)


def clip(x: mx.array, scale: Optional[mx.array], threshold: float) -> mx.array:
    """Scale each row (last axis) down to an L2 norm of at most
    ``threshold``, after the optional per-channel ``scale``."""
    x = x.astype(mx.float32)
    if scale is not None:
        x = x * scale.astype(mx.float32)
    norm = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True)) + 1e-6
    return x * mx.clip(threshold / norm, 0.0, 1.0)


def depth_mix(blocks: List[mx.array], cur: mx.array,
              query: mx.array) -> mx.array:
    """Attention over depth: a softmax over the rms-normed ``blocks`` and
    ``cur`` with ``query`` rows [nq, D]. Returns [B, L, nq, D] float32."""
    states = mx.stack([*blocks, cur], axis=-2)        # [B, L, n, D]
    states = _rms(states)
    q = query.astype(mx.float32).reshape(-1, states.shape[-1])
    scores = (states @ q.T) * (states.shape[-1] ** -0.5)   # [B, L, n, nq]
    p = mx.softmax(scores, axis=-2)
    return mx.swapaxes(p, -1, -2) @ states             # [B, L, nq, D]


def log_scale(positions: mx.array, floor: int, alpha: float) -> mx.array:
    """1 + alpha * log(max(1, (pos + 1) / floor)), float32."""
    t = (positions.astype(mx.float32) + 1.0) / float(floor)
    return 1.0 + alpha * mx.log(mx.maximum(t, 1.0))


class _Norm(nn.RMSNorm):
    """RMSNorm with a float32 weight; the output keeps the input dtype."""

    def __call__(self, x):
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
        h = self.down(mx.concatenate([_rms(cur), _rms(msg)], axis=-1))
        gain = mx.tanh(self.gain(h).astype(mx.float32))
        forget = mx.tanh(self.forget(h).astype(mx.float32))
        c = cur.astype(mx.float32)
        return (c - forget * c + gain * msg.astype(mx.float32)).astype(
            cur.dtype)


class LoopControl(nn.Module):
    """The message queries and controllers of one core layer."""

    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.msg_query = mx.zeros((len(CONTROLLERS), dim), dtype=mx.float32)
        self.q = Controller(dim, rank)
        self.k = Controller(dim, rank)
        self.v = Controller(dim, rank)
        self.ffn = Controller(dim, rank)


def _key_distances(cache, n_queries: int, n_keys: int, rows=None) -> mx.array:
    """[L, K] distances pos_q - pos_k from the queries of this update to
    every key slot the cache returned; ``rows`` (start, stop) selects some
    of the queries. Left padding shifts queries and keys alike, so buffer
    coordinates give the distance."""
    r0, r1 = rows if rows is not None else (0, n_queries)
    i = mx.arange(r0, r1, dtype=mx.int32)[:, None]
    s = mx.arange(n_keys, dtype=mx.int32)[None, :]
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

    def _bias(self, r: mx.array, sink: mx.array, dist: mx.array) -> mx.array:
        """[B, H, L, K] float32 bias ``R . B[d] - sink`` with the zero pad
        row for distances outside [0, extent)."""
        B, L, H, _ = r.shape
        table = mx.pad(self.rel_bias.astype(mx.float32), [(0, 1), (0, 0)])
        scores = r @ table.T                              # [B, L, H, E+1]
        scores = scores - sink[..., None]
        index = mx.where((dist >= 0) & (dist < self.extent), dist,
                         self.extent)
        K = dist.shape[1]
        index = mx.broadcast_to(index[None, :, None, :], (B, L, H, K))
        bias = mx.take_along_axis(scores, index, axis=-1)  # [B, L, H, K]
        return bias.transpose(0, 2, 1, 3)

    def attend(self, q, k, v, r, sink, cache, pad_mask) -> mx.array:
        """Attention with the folded bias and a zero sink logit. ``q`` is
        [B, H, L, D]; returns [B, H, L, D]."""
        B, H, L, _ = q.shape
        K = k.shape[2]
        sinks = mx.zeros((H,), dtype=q.dtype)
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
            bias = self._bias(r[:, i0:i1], sink[:, i0:i1], dist)
            mask = mx.where(valid, bias, -mx.inf).astype(q.dtype)
            outs.append(mx.fast.scaled_dot_product_attention(
                q[:, :, i0:i1], k, v, scale=self.scale, mask=mask,
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
        xf = x.astype(mx.float32)
        if pad_mask is not None:
            xf = mx.where(pad_mask[..., None], xf, 0)
        if cache is not None:
            state = cache[slot]
            if state is None:
                state = mx.zeros((B, n_keep, C), dtype=mx.float32)
            window = mx.concatenate([state.astype(mx.float32), xf], axis=1)
            cache.update_window(slot, window, n_keep, lengths=cache.lengths)
            if sink is not None:
                sink.append((cache, slot, window))
        else:
            window = mx.pad(xf, [(0, 0), (n_keep, 0), (0, 0)])
        w = weight.astype(mx.float32)
        out = window[:, 0:L] * w[:, 0]
        for k in range(1, self.width):
            out = out + window[:, k:k + L] * w[:, k]
        n = 1
        if dynamic is not None:
            n = 2
            n_heads = dynamic.shape[-1] // self.width
            taps = dynamic.astype(mx.float32).reshape(B, L, n_heads,
                                                      self.width)
            for lag in range(self.width):
                past = window[:, n_keep - lag:n_keep - lag + L]
                past = past.reshape(B, L, n_heads, C // n_heads)
                out = out + (past * taps[..., lag:lag + 1]).reshape(B, L, C)
        return ((xf + out) * (n * self.width + 1) ** -0.5).astype(x.dtype)

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

        q = attn.q_proj(inputs[0])
        k = attn.k_proj(inputs[1])
        v = attn.v_proj(inputs[2])
        k = self._convolve(
            k, attn.k_conv,
            None if self.is_sliding else attn.k_conv_proj(inputs[1]),
            conv_cache, 0, pad_mask, conv_sink)
        v = self._convolve(
            v, attn.v_conv,
            None if self.is_sliding else attn.v_conv_proj(inputs[2]),
            conv_cache, 1, pad_mask, conv_sink)
        r = attn.rel_proj(inputs[0])

        q = clip(q.reshape(B, L, self.n_heads, self.head_dim), attn.q_clip,
                 attn.qk_clip)
        k = clip(k.reshape(B, L, self.n_kv_heads, self.head_dim), None,
                 attn.qk_clip)
        v = _rms(v.reshape(B, L, self.n_kv_heads, self.head_dim), attn.eps)
        r = clip(r.reshape(B, L, self.n_heads, attn.rel_length),
                 attn.rel_clip, attn.rel_clip_threshold)
        if not self.is_sliding:
            tau = positions[..., None, None]
            q = q * tau
            r = r * tau
        sink = attn.sink_proj(inputs[0]).astype(mx.float32)

        dtype = h.dtype
        q = q.astype(dtype).transpose(0, 2, 1, 3)
        k = k.astype(dtype).transpose(0, 2, 1, 3)
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
        out = out * mx.sigmoid(attn.gate_proj(inputs[0]).astype(mx.float32)
                               ).astype(dtype)
        out = self._convolve(attn.o_proj(out), attn.post_conv, None,
                             conv_cache, 2, pad_mask, conv_sink)

        if ip % self.block_size == 0:
            blocks.append(inp_sa)
            h = out
        else:
            h = (inp_sa.astype(mx.float32) + out.astype(mx.float32)).astype(
                dtype)
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
        return (ffn_inp.astype(mx.float32) + x.astype(mx.float32)).astype(
            dtype)


def _pad_key_mask(cache, n_queries: int):
    """[B, 1, L, K] key validity of a batch cache's left padding, or None.
    Call it before the cache update."""
    if type(cache) not in (BatchKVCache, BatchRotatingKVCache):
        return None
    mask = cache.make_mask(n_queries, return_array=True)
    if mask is None or isinstance(mask, str):
        return None
    return mask


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
        scores = mx.sigmoid(self.gate(x.astype(mx.float32)).astype(
            mx.float32))
        select = scores
        if self.args.use_gate_bias:
            select = scores + self.e_score_correction_bias.astype(mx.float32)
        k = self.args.num_experts_per_tok

        def weights_at(inds):
            w = mx.take_along_axis(scores, inds, axis=-1)
            w = w / (mx.sum(w, axis=-1, keepdims=True) + 1e-20)
            return (w * self.args.expert_weights_scale).astype(x.dtype)

        inds = mx.stop_gradient(
            mx.argpartition(-select, kth=k - 1, axis=-1)[..., :k])
        weights = weights_at(inds)

        from gmlx.stream.moe_experts import expert_controls_active

        if expert_controls_active(self):
            from gmlx.stream.moe_experts import _apply_expert_controls

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
            pos = offset.reshape(-1, 1) + mx.arange(L)[None, :]
        else:
            pos = (offset + mx.arange(L))[None, :]
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
