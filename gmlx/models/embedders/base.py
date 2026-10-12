"""Outputs, masks, activations and sentence pooling shared by the encoders.

The pooling follows the sentence-transformers modules a checkpoint lists in
``modules.json``: ``Pooling`` (mean over real tokens, the first token, or
the last real token), then any ``Dense`` layers, then ``Normalize`` (L2).
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

POOLING_MODES = ("mean", "cls", "last")


@dataclass
class EncoderOutput:
    """``last_hidden_state`` is ``[B, L, H]``; ``text_embeds`` is ``[B, D]``,
    pooled, projected and L2-normalized."""

    last_hidden_state: mx.array
    text_embeds: mx.array


def activation(name: str):
    """The function for a Hugging Face ``hidden_act`` name."""
    if name == "gelu":
        return nn.gelu
    if name in ("gelu_new", "gelu_pytorch_tanh", "gelu_fast"):
        return nn.gelu_approx
    if name in ("silu", "swish"):
        return nn.silu
    if name == "relu":
        return nn.relu
    raise ValueError(f"unsupported activation {name!r}")


def padding_mask(attention_mask: mx.array | None) -> mx.array | None:
    """``[B, L]`` 0/1 mask to a ``[B, 1, 1, L]`` boolean key mask, or None
    when every position is real (no mask needed)."""
    if attention_mask is None:
        return None
    return (attention_mask != 0)[:, None, None, :]


def window_mask(length: int, radius: int) -> mx.array | None:
    """``[L, L]`` boolean mask allowing ``|q - k| <= radius``, or None when
    the window covers the whole sequence."""
    if length <= radius + 1:
        return None
    pos = mx.arange(length)
    return mx.abs(pos[:, None] - pos[None, :]) <= radius


def additive_mask(mask: mx.array | None, dtype) -> mx.array | None:
    """A boolean mask as an additive one in ``dtype``: 0 where allowed and
    the dtype's finite minimum where not, as transformers builds it.

    A query row with no allowed key (a padding query outside every real
    token's window) attends to every key instead, as transformers'
    ``_unmask_unattended`` does. Its output is never pooled, and a row of
    minimums would overflow to -inf in float16 arithmetic and give NaN."""
    if mask is None:
        return None
    unattended = mx.logical_not(mx.any(mask, axis=-1, keepdims=True))
    return mx.where(mx.logical_or(mask, unattended), mx.array(0, dtype),
                    mx.array(mx.finfo(dtype).min, dtype))


def combine_masks(*masks: mx.array | None) -> mx.array | None:
    """Logical AND of the given boolean masks, skipping None."""
    out = None
    for m in masks:
        if m is None:
            continue
        out = m if out is None else mx.logical_and(out, m)
    return out


def pool(hidden: mx.array, attention_mask: mx.array | None, mode: str) -> mx.array:
    """Pool ``[B, L, H]`` hidden states to ``[B, H]`` in float32.

    ``mean`` averages the real tokens (the sentence-transformers mean with its
    1e-9 floor on the token count); ``cls`` takes the first token; ``last``
    takes each row's last real token, wherever the padding sits."""
    if mode == "last":
        if attention_mask is None:
            return hidden[:, -1].astype(mx.float32)
        pos = mx.arange(hidden.shape[1])
        last = mx.max(mx.where(attention_mask != 0, pos, -1), axis=1)
        rows = mx.take_along_axis(hidden, last[:, None, None], axis=1)
        return rows[:, 0].astype(mx.float32)
    h = hidden.astype(mx.float32)
    if mode == "cls":
        return h[:, 0]
    if mode != "mean":
        raise ValueError(f"unsupported pooling mode {mode!r}")
    if attention_mask is None:
        return h.mean(axis=1)
    m = attention_mask.astype(mx.float32)[..., None]
    return (h * m).sum(axis=1) / mx.maximum(m.sum(axis=1), 1e-9)


def l2_normalize(x: mx.array, eps: float = 1e-12) -> mx.array:
    """Rows of ``x`` scaled to unit L2 norm (norm floored at ``eps``)."""
    norm = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True))
    return x / mx.maximum(norm, eps)
