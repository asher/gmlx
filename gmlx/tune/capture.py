"""Residual-stream capture: the output of every decoder block.

A block's output is the residual stream after the block, the tensor that
llama.cpp's ``tools/cvector-generator`` reads as ``l_out``. ``layer_capture``
rewrites the ``__call__`` of every decoder-layer class of a model, in the
pattern of ``checkpoint_layers``, and marks each layer with its index. While
a recorder is active, a marked layer hands its output to it. The rewrite is
undone when the block ends.

The wrapper:

- reads element 0 of a tuple return (a gemma-4 layer returns the hidden
  state with its shared key and value arrays and the rope offset);
- records once per layer index, because a fused layer calls the stock
  ``__call__`` of its base class through ``super()``;
- returns the very object the layer returned, because the next fused
  gemma layer tests ``ho[0] is x`` to reuse the normed input it was
  handed;
- averages the streams of a hyper-connection model, whose wide residual
  is ``[B, T, streams, n_embd]``, as llama.cpp's ``hc_stream_mean`` does.

``block_outputs`` runs one prompt with a fresh cache and stops after the
trunk, so the vocabulary head never runs. A trunk that takes
``keep_last`` gets None: every layer runs on every column.
"""
from __future__ import annotations

import contextlib
import contextvars

import mlx.core as mx
import mlx.nn as nn

from gmlx.tune.checkpoint import layer_list
from gmlx.upstream.patching import link, wrapped_with

_FLAG = "_gmlx_capture"
_IDX = "_gmlx_capture_idx"
_RECORDER: contextvars.ContextVar = contextvars.ContextVar(
    "gmlx_capture_recorder", default=None)


def summarize(h: mx.array, positions: int) -> mx.array:
    """The f32 mean of the last ``positions`` columns of row 0 of a block
    output, ``[n_embd]``. A 4-D output holds the streams of a
    hyper-connection model on axis 2, and they are averaged first."""
    h = h.astype(mx.float32)
    if h.ndim == 4:
        h = h.mean(axis=2)
    t = h.shape[1]
    return h[0, max(t - positions, 0):, :].mean(axis=0)


class Recorder:
    """The block outputs of one forward, by layer index."""

    def __init__(self, n_layer: int, positions: int):
        self.n_layer = n_layer
        self.positions = positions
        self.active: set[int] = set()
        self.begin()

    def begin(self) -> None:
        self.rows: list = [None] * self.n_layer
        self.calls = [0] * self.n_layer

    def record(self, idx: int, out) -> None:
        h = out[0] if isinstance(out, tuple) else out
        self.calls[idx] += 1
        self.rows[idx] = summarize(h, self.positions)

    def finish(self) -> mx.array:
        """``[n_layer, n_embd]`` f32. Raises RuntimeError unless every
        layer ran exactly once."""
        bad = [i for i, n in enumerate(self.calls) if n != 1]
        if bad:
            raise RuntimeError(
                f"capture expected one output from each of {self.n_layer} "
                f"layers; layers {bad[:8]} ran {[self.calls[i] for i in bad[:8]]} times")
        return mx.stack(self.rows)


def _wrap(fn):
    def captured(self, *args, **kwargs):
        rec = _RECORDER.get()
        idx = self.__dict__.get(_IDX)
        if rec is None or idx is None or idx in rec.active:
            return fn(self, *args, **kwargs)
        rec.active.add(idx)
        try:
            out = fn(self, *args, **kwargs)
        finally:
            rec.active.discard(idx)
        rec.record(idx, out)
        return out
    return link(captured, fn, _FLAG)


@contextlib.contextmanager
def layer_capture(layers, positions: int):
    """Mark ``layers`` and wrap their classes for the block. Yields the
    active ``Recorder``."""
    wrapped = []
    for i, layer in enumerate(layers):
        layer.__dict__[_IDX] = i
        cls = type(layer)
        if not wrapped_with(cls.__call__, _FLAG):
            own = "__call__" in cls.__dict__
            orig = cls.__call__
            cls.__call__ = _wrap(orig)
            wrapped.append((cls, orig, own))
    rec = Recorder(len(layers), positions)
    tok = _RECORDER.set(rec)
    try:
        yield rec
    finally:
        _RECORDER.reset(tok)
        for layer in layers:
            layer.__dict__.pop(_IDX, None)
        for cls, orig, own in reversed(wrapped):
            if cls.__dict__.get("__call__", None) is None:
                continue
            if cls.__dict__["__call__"].__dict__.get("_gmlx_orig") is not orig:
                continue                 # another wrap went on top; leave both
            if own:
                cls.__call__ = orig
            else:
                del cls.__call__


def capture_layers(model, n_layer: int | None = None) -> list:
    """The decoder layers to capture: ``layer_list`` cut to ``n_layer``, the
    trunk's count without MTP or nextn blocks, when a model keeps such
    blocks after its trunk layers."""
    layers = list(layer_list(model))
    if n_layer is not None and 0 < n_layer < len(layers):
        layers = layers[:n_layer]
    return layers


class _TrunkDone(Exception):
    pass


class _TrunkOnly:
    """Stands in for a wrapped model's trunk for one forward: runs the trunk
    with ``keep_last=None`` and stops the forward before the head.
    Attribute reads reach the trunk."""

    def __init__(self, trunk):
        self.__dict__["_trunk"] = trunk

    def __getattr__(self, name):
        return getattr(self.__dict__["_trunk"], name)

    def __call__(self, *args, **kwargs):
        from gmlx.models.vlm_text_only import _takes_keep_last

        trunk = self.__dict__["_trunk"]
        if _takes_keep_last(trunk):
            kwargs["keep_last"] = None
        trunk(*args, **kwargs)
        raise _TrunkDone


def _trunk_owner(model):
    """The wrapped model whose ``model`` attribute is its trunk, or None."""
    from gmlx.models.vlm_text_only import LanguageModel

    lm = getattr(model, "language_model", None)
    if not isinstance(lm, LanguageModel):
        return None
    inner = lm._model
    if "model" in vars(inner) or not isinstance(getattr(inner, "model", None), nn.Module):
        return None
    return inner


def block_outputs(model, ids, rec: Recorder) -> mx.array:
    """Run ``ids`` (one prompt) through ``model`` under ``rec`` and return
    the evaluated ``[n_layer, n_embd]`` f32 block outputs."""
    x = mx.array([list(ids)], dtype=mx.int32)
    cache = model.make_cache() if hasattr(model, "make_cache") else None
    rec.begin()
    owner = _trunk_owner(model)
    if owner is None:
        model(x, cache=cache)
    else:
        attrs = vars(owner)
        attrs["model"] = _TrunkOnly(owner.model)
        try:
            model(x, cache=cache)
        except _TrunkDone:
            pass
        finally:
            attrs.pop("model", None)
    out = rec.finish()
    mx.eval(out)
    return out
