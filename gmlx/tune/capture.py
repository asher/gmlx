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
  is ``[B, T, streams, n_embd]``, as llama.cpp's ``hc_stream_mean`` does;
- applies the pending expand of a hyper-connection layer in carry mode,
  which returns the residual before its last expand and the expand's
  inputs;
- takes the active stream, stream 0, of a gemma-3n AltUp output
  ``[streams, B, T, n_embd]``, the slice that cvector-generator reads
  from llama.cpp's ``[n_embd, n_tokens, n_altup]`` ``l_out``.

Any other layout raises RuntimeError.

``block_outputs`` runs one prompt with a fresh cache and stops after the
trunk, so the vocabulary head never runs. A long prompt runs in chunks
through the cache, as a prefill does, and only the last chunk is
recorded; it holds every averaged position. A trunk that takes
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


def summarize(h: mx.array, positions: int, tokens: int) -> mx.array:
    """The f32 mean of the last ``positions`` columns of a block output for
    one ``tokens``-column forward, ``[n_embd]``. Raises RuntimeError on a
    layout other than ``[1, T, D]``, ``[1, T, streams, D]`` (the streams
    are averaged) and ``[streams, 1, T, D]`` (stream 0 is taken)."""
    s = tuple(h.shape)
    t0 = max(tokens - positions, 0)
    if h.ndim in (3, 4) and s[:2] == (1, tokens):
        h = h[0, t0:].astype(mx.float32)
        if h.ndim == 3:
            h = h.mean(axis=1)
    elif h.ndim == 4 and s[1:3] == (1, tokens):
        h = h[0, 0, t0:].astype(mx.float32)
    else:
        raise RuntimeError(f"capture cannot read a block output of shape {s} "
                           f"for a {tokens}-token prompt")
    return h.mean(axis=0)


def _block_output(out) -> mx.array:
    """The hidden state a layer returned. A hyper-connection layer in
    carry mode returns ``(h, (x, residual, post, comb))``, where ``h`` is
    the residual before its last expand; the block output is that expand."""
    if not isinstance(out, tuple):
        return out
    h = out[0]
    carry = out[1] if len(out) == 2 else None
    if (h.ndim == 4 and isinstance(carry, tuple) and len(carry) == 4
            and all(isinstance(a, mx.array) for a in carry)):
        from gmlx.models.deepseek_v4 import hyper_connection

        return hyper_connection.hc_expand_m1(*carry)
    return h


class Recorder:
    """The block outputs of one forward, by layer index."""

    def __init__(self, n_layer: int, positions: int):
        self.n_layer = n_layer
        self.positions = positions
        self.active: set[int] = set()
        self.begin()

    def begin(self, tokens: int = 0) -> None:
        """Start a forward of ``tokens`` columns."""
        self.tokens = tokens
        self.rows: list = [None] * self.n_layer
        self.calls = [0] * self.n_layer

    def record(self, idx: int, out) -> None:
        self.calls[idx] += 1
        self.rows[idx] = summarize(_block_output(out), self.positions, self.tokens)

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
    """The module whose ``model`` attribute is the trunk that holds the
    layers, found under the text wrapper and any ``language_model`` links
    below it (qwen3.5 nests its text model so), or None."""
    from gmlx.models.vlm_text_only import LanguageModel

    lm = getattr(model, "language_model", None)
    if not isinstance(lm, LanguageModel):
        return None
    owner = lm._model
    for _ in range(4):
        if "model" in vars(owner):
            return None
        trunk = getattr(owner, "model", None)
        if isinstance(trunk, nn.Module) and getattr(trunk, "layers", None) is not None:
            return owner
        owner = getattr(owner, "language_model", None)
        if not isinstance(owner, nn.Module):
            return None
    return None


def _chunks(n: int, positions: int, step: int | None) -> list[int]:
    """Chunk widths for an ``n``-token prompt: pieces of ``step``, with a
    last piece of ``max(positions, step)`` tokens or the whole prompt."""
    if step is None or n <= step:
        return [n]
    last = min(n, max(positions, step))
    head, rest = divmod(n - last, step)
    return [step] * head + ([rest] if rest else []) + [last]


def _forward(model, x, cache) -> None:
    owner = _trunk_owner(model)
    if owner is None:
        model(x, cache=cache)
        return
    attrs = vars(owner)
    attrs["model"] = _TrunkOnly(owner.model)
    try:
        model(x, cache=cache)
    except _TrunkDone:
        pass
    finally:
        attrs.pop("model", None)


def block_outputs(model, ids, rec: Recorder, step: int | None = None) -> mx.array:
    """Run ``ids`` (one prompt) through ``model`` under ``rec`` and return
    the evaluated ``[n_layer, n_embd]`` f32 block outputs. With ``step``, a
    longer prompt runs in chunks through the cache (see ``_chunks``)."""
    ids = list(ids)
    cache = model.make_cache() if hasattr(model, "make_cache") else None
    widths = [len(ids)] if cache is None else _chunks(len(ids), rec.positions, step)
    start = 0
    for n in widths[:-1]:
        tok = _RECORDER.set(None)
        try:
            _forward(model, mx.array([ids[start:start + n]], dtype=mx.int32), cache)
        finally:
            _RECORDER.reset(tok)
        mx.eval([c.state for c in cache])
        mx.clear_cache()
        start += n
    rec.begin(widths[-1])
    _forward(model, mx.array([ids[start:]], dtype=mx.int32), cache)
    out = rec.finish()
    mx.eval(out)
    return out
