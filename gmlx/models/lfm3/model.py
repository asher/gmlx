# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Model for Liquid AI LFM3 (GGUF arch ``lfm3``).

This is the runtime for llama.cpp's ``LLM_ARCH_LFM3`` conversions, ported
from ``src/models/lfm3.cpp`` and ``src/models/lfm-base.cpp`` (MIT, written
by Liquid AI) and checked against the Apache-2.0 ``modeling_lfm3.py`` of
the LFM3 checkpoints. Each layer runs one of three operators:

  1. A gated short convolution: ``in_proj`` gives B, C and x, a depthwise
     causal convolution of width ``conv_L_cache`` runs over ``B * x``, and
     ``out_proj`` reads ``C * conv``.
  2. Sliding-window attention over the last ``sliding_window`` positions.
  3. Full attention.

Attention is GQA with a per-head Q/K RMSNorm and a sigmoid output gate of
one value per head, projected from the normed input. Sliding layers rotate
every dimension at ``rope_theta_swa``. Full layers rotate the leading
dimensions at ``rope_theta`` and leave the rest unrotated; the GGUF stores
that split as ``rope_freqs`` factors, which the loader hands to
``set_full_rope_factors``. Sandwich norms follow the operator and the FFN.
The head is tied to the embedding.
"""

from dataclasses import dataclass
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from gmlx._vendor.mlx_lm.models.base import BaseModelArgs
from gmlx._vendor.mlx_lm.models.rope_utils import ProportionalRoPE

from gmlx.cache import window_sdpa
from gmlx.cache.attention import (
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from gmlx.cache.kv_cache import ArraysCache, KVCache, RotatingKVCache
from gmlx.load.dtypes import (
    NORM_WEIGHT_KEY,
    activation_dtype,
    fp32_norms,
    wider_float,
)

CONV = "conv"
SLIDING = "sliding_attention"
FULL = "full_attention"
LAYER_TYPES = (CONV, SLIDING, FULL)

# A prefill chunk at least this wide evaluates each layer as it finishes,
# so one chunk does not hold every layer's intermediates at once. Decode
# and speculative verify rounds stay one lazy graph.
PREFILL_EVAL_ROWS = 64

# The tokens that switch a reply to speech, and the media placeholders. A
# text reply must not sample them, on any path, so the head masks them.
# The speech path asks the head for the unmasked logits.
MODE_MARKERS = ("<|audio_start|>", "<|mixed_start|>", "<|audio_out|>")
MEDIA_PLACEHOLDERS = ("<|image|>", "<|video|>", "<|audio|>")
# Finite in every 16-bit float type, so a masked logit never makes a NaN.
_SUPPRESS_BIAS = -3.0e4

# A rope_freqs factor at or above this leaves its dimension unrotated.
# llama.cpp's converter writes 1e30 there.
_UNROTATED_FACTOR = 1e29


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    layer_types: List[str]
    sliding_window: int = 256
    conv_L_cache: int = 7
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1000000.0
    rope_theta_swa: float = 10000.0
    max_position_embeddings: int = 128000
    use_sandwich_norm: bool = True
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"lfm3: layer_types has {len(self.layer_types)} entries for "
                f"{self.num_hidden_layers} layers")
        bad = sorted(set(self.layer_types) - set(LAYER_TYPES))
        if bad:
            raise ValueError(f"lfm3: unknown layer types {bad}")


def suppressed_token_ids(vocab: dict) -> list[int]:
    """The ids of the mode markers and media placeholders that ``vocab``
    (token text to id) holds."""
    return sorted(int(vocab[t]) for t in MODE_MARKERS + MEDIA_PLACEHOLDERS
                  if t in vocab)


def rotated_dims_from_factors(factors, head_dim: int) -> int:
    """The rotated width a ``rope_freqs`` tensor describes: a run of 1.0
    factors, then factors that leave the rest unrotated. Any other pattern
    raises, because the full layers have no rope for it."""
    values = [float(v) for v in factors.tolist()]
    if len(values) != head_dim // 2:
        raise ValueError(
            f"lfm3: rope_freqs has {len(values)} factors for head_dim "
            f"{head_dim}")
    n = 0
    while n < len(values) and values[n] == 1.0:
        n += 1
    if n == 0 or any(v < _UNROTATED_FACTOR for v in values[n:]):
        raise ValueError(
            "lfm3: rope_freqs is not a run of 1.0 factors followed by "
            "unrotated dimensions; the full-attention layers cannot use it")
    return 2 * n


class _Norm(nn.RMSNorm):
    """RMSNorm whose weight the loader may keep float32. The output takes
    the wider of the input and activation dtypes, as a weight in the
    activation dtype gives, so a float32 weight never widens the stream."""

    OUTPUT_IN_INPUT_DTYPE = True

    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__(dims, eps=eps)
        self._act = activation_dtype("lfm3")

    def __call__(self, x):
        out = wider_float(x.dtype, self._act)
        return mx.fast.rms_norm(x, self.weight, self.eps).astype(out)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, use_sliding: bool):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.use_sliding = use_sliding
        self.window = args.sliding_window if use_sliding else None
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        # One gate value per head.
        self.gate_proj = nn.Linear(dim, self.n_heads, bias=False)

        self.q_norm = _Norm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = _Norm(self.head_dim, eps=args.rms_norm_eps)

        # The loader replaces a full layer's rope from the GGUF rope_freqs
        # factors (Model.set_full_rope_factors).
        self.rope = nn.RoPE(
            self.head_dim,
            traditional=False,
            base=args.rope_theta_swa if use_sliding else args.rope_theta,
        )

    def _rotate(self, x: mx.array, offset) -> mx.array:
        # RoPE runs in float32, as the reference does.
        return self.rope(x.astype(mx.float32), offset=offset).astype(x.dtype)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        offset = cache.offset if cache is not None else 0
        queries = self._rotate(queries, offset)
        keys = self._rotate(keys, offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = window_sdpa.claim(
            queries, keys, values, cache, self.scale, self.window)
        if output is None:
            output = scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask
            )
        gate = mx.sigmoid(self.gate_proj(x))
        output = output.transpose(0, 2, 1, 3) * gate[..., None]
        return self.o_proj(output.reshape(B, L, -1))


class ShortConv(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.width = args.conv_L_cache
        self.in_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        # Depthwise taps, [channels, width], float32 like the F32 wire
        # tensor. Tap width - 1 reads the current position.
        self.kernel = mx.zeros((dim, self.width), dtype=mx.float32)

    def _convolve(self, window: mx.array, length: int) -> mx.array:
        """Depthwise causal convolution over ``window``, which holds
        ``width - 1`` earlier positions before its ``length`` new ones.
        The sum runs in float32, as the reference does."""
        w = self.kernel.astype(mx.float32)
        win = window.astype(mx.float32)
        if length == 1:
            out = (win * w.T).sum(axis=1, keepdims=True)
        else:
            out = win[:, :length, :] * w[:, 0]
            for k in range(1, self.width):
                out = out + win[:, k:k + length, :] * w[:, k]
        return out.astype(window.dtype)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        b, c, v = mx.split(self.in_proj(x), 3, axis=-1)
        bx = b * v
        if mask is not None:
            bx = mx.where(mask[..., None], bx, 0)

        length = bx.shape[1]
        n_keep = self.width - 1
        if cache is not None:
            state = cache[0]
            if state is None:
                state = mx.zeros((bx.shape[0], n_keep, bx.shape[2]),
                                 dtype=bx.dtype)
            window = mx.concatenate([state, bx], axis=1)
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, length)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(window, positions, axis=1)
            elif length > n_keep:
                # A copy, so the state does not hold the chunk's window.
                cache[0] = window[:, -n_keep:, :] * 1
            else:
                cache[0] = window[:, -n_keep:, :]
            cache.advance(length)
        else:
            window = mx.pad(bx, [(0, 0), (n_keep, 0), (0, 0)])

        return self.out_proj(c * self._convolve(window, length))


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        dim = args.hidden_size
        self.layer_type = args.layer_types[layer_idx]
        self.is_attention_layer = self.layer_type != CONV
        self.use_sliding = self.layer_type == SLIDING
        if self.is_attention_layer:
            self.self_attn = Attention(args, self.use_sliding)
        else:
            self.conv = ShortConv(args)
        self.mlp = MLP(dim, args.intermediate_size)
        self.input_layernorm = _Norm(dim, eps=args.rms_norm_eps)
        # The FFN pre-norm, under the name stream/lookahead reads.
        self.post_attention_layernorm = _Norm(dim, eps=args.rms_norm_eps)
        self.use_sandwich_norm = args.use_sandwich_norm
        if self.use_sandwich_norm:
            self.post_attn_norm = _Norm(dim, eps=args.rms_norm_eps)
            self.post_ffw_norm = _Norm(dim, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        op = self.self_attn if self.is_attention_layer else self.conv
        r = op(self.input_layernorm(x), mask, cache)
        if self.use_sandwich_norm:
            r = self.post_attn_norm(r)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        if self.use_sandwich_norm:
            r = self.post_ffw_norm(r)
        return h + r


def _first(layer_types: List[str], kind: str) -> Optional[int]:
    return next((i for i, t in enumerate(layer_types) if t == kind), None)


class Lfm3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.sliding_window = args.sliding_window
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = _Norm(args.hidden_size, eps=args.rms_norm_eps)

        self.fa_idx = _first(args.layer_types, FULL)
        self.swa_idx = _first(args.layer_types, SLIDING)
        self.conv_idx = _first(args.layer_types, CONV)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
        capture_layer_ids=None,
        hidden_sink: Optional[list] = None,
    ):
        h = self.embed_tokens(inputs) if input_embeddings is None \
            else input_embeddings
        per_layer = cache is not None and h.shape[1] >= PREFILL_EVAL_ROWS
        if cache is None:
            cache = [None] * len(self.layers)

        masks: dict[str, Any] = {CONV: None, SLIDING: None, FULL: None}
        if self.fa_idx is not None:
            masks[FULL] = create_attention_mask(h, cache[self.fa_idx])
        if self.swa_idx is not None:
            masks[SLIDING] = create_attention_mask(
                h, cache[self.swa_idx], window_size=self.sliding_window)
        if self.conv_idx is not None:
            masks[CONV] = create_ssm_mask(h, cache[self.conv_idx])

        capture = set(capture_layer_ids) if capture_layer_ids else ()
        for index, (layer, c) in enumerate(zip(self.layers, cache)):
            # A drafter reads each captured layer's input, as llama.cpp
            # records it (t_layer_inp).
            if hidden_sink is not None and index in capture:
                hidden_sink.append(h)
            h = layer(h, masks[layer.layer_type], c)
            if per_layer:
                # The conv tail is a leaf of its own. Attention evaluates
                # its keys and values with its output.
                tail = c[0] if layer.layer_type == CONV and c is not None \
                    else None
                mx.eval(h, *([tail] if tail is not None else []))

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Lfm3Model(args)
        self._suppressed: tuple = ()
        self._suppress_bias: dict = {}
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def set_suppressed_tokens(self, ids) -> None:
        """Mask ``ids`` in every text reply. The loader passes the mode
        markers and media placeholders of the file's vocabulary."""
        ids = sorted({int(i) for i in ids})
        if any(i < 0 or i >= self.args.vocab_size for i in ids):
            raise ValueError(
                f"lfm3: a suppressed token id is outside the vocabulary of "
                f"{self.args.vocab_size}: {ids}")
        self._suppressed = tuple(ids)
        # One bias row per float type, built here so a traced forward
        # evaluates nothing.
        self._suppress_bias = {}
        if ids:
            row = mx.zeros((self.args.vocab_size,), dtype=mx.float32)
            row[mx.array(ids)] = _SUPPRESS_BIAS
            for dtype in (mx.float32, mx.bfloat16, mx.float16):
                self._suppress_bias[dtype] = row.astype(dtype)
            mx.eval(list(self._suppress_bias.values()))

    @property
    def suppressed_tokens(self) -> tuple:
        return self._suppressed

    def suppress_bias(self, dtype) -> Optional[mx.array]:
        """The bias row :meth:`head` adds to logits of ``dtype``. None
        when no token is suppressed."""
        if not self._suppressed:
            return None
        bias = self._suppress_bias.get(dtype)
        if bias is None:
            bias = self._suppress_bias[mx.float32].astype(dtype)
        return bias

    def head(self, h: mx.array, suppress: bool = True) -> mx.array:
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(h)
        else:
            logits = self.lm_head(h)
        if suppress and self._suppressed:
            logits = logits + self.suppress_bias(logits.dtype)
        return logits

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
        capture_layer_ids=None,
        hidden_sink: Optional[list] = None,
    ):
        return self.head(self.model(
            inputs, cache, input_embeddings,
            capture_layer_ids=capture_layer_ids, hidden_sink=hidden_sink))

    def set_full_rope_factors(self, factors: mx.array) -> int:
        """Build each full-attention layer's rope from the GGUF
        ``rope_freqs`` factors. Returns the rotated width."""
        rotated = rotated_dims_from_factors(factors, self.args.head_dim)
        for layer in self.model.layers:
            if layer.layer_type == FULL:
                layer.self_attn.rope = ProportionalRoPE(
                    self.args.head_dim, rotated, traditional=False,
                    base=self.args.rope_theta)
        return rotated

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.layer_type == CONV:
                caches.append(ArraysCache(size=1))
            elif layer.use_sliding:
                caches.append(
                    RotatingKVCache(max_size=self.model.sliding_window))
            else:
                caches.append(KVCache())
        return caches

    @property
    def cast_predicate(self):
        # Kept in step with the loader's fp32 pins (loader.fp32_keep_for).
        norms = fp32_norms("lfm3")

        def predicate(path: str):
            if path.endswith(".conv.kernel"):
                return False
            return not (norms and NORM_WEIGHT_KEY in path)

        return predicate
