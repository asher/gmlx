# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Whole-block prefill chunking for prompts carrying media token blocks.

Some VLM prompts carry image token blocks whose attention is not causal
inside the block, so the text tower can only honor it when every block
lands whole inside one prefill chunk. Two sources name the blocks:

- ``image_spans``, a Python list of absolute ``[start, end)`` prompt
  offsets riding the prompt kwargs (DeepSeek-V4-Flash-Vision-Exp, where a
  query inside a block sees the whole block and the tower raises on a
  block a chunk cuts);
- the token types of a gemma-4 prompt under bidirectional image attention
  (``use_bidirectional_attention == "vision"``): each run of image or
  video tokens is a block, as in transformers. Audio stays causal. In a
  multi-row batch the rows' runs are taken in column order and runs that
  overlap across rows merge into one span.

Rule, for a chunk starting at offset ``off`` with a candidate width
``n``: a boundary ``off + n`` strictly inside a block moves back to the
block's first token when that still makes progress, else (the chunk
begins at the block and the step is shorter than the block) forward to
the token after the block. The extension is bounded by one block and is
pinned past the depth-decay resolver so decay cannot re-cut it. An exact
APC checkpoint column inside the extended chunk is dropped, and a
right-padded row whose prompt ends inside it keeps the logits of its last
real column.

Three prefill loops consult this: the stock serve ``prompt_step``
(wrapped by ``install_span_aware_prompt_step``), the owned MTP prefill
(calls ``span_aware_prompt_n``), and the fixed-step loop of mlx-vlm's
``generate_step`` behind run and chat, which
``install_block_aware_generate_step`` steers by passing each block start
as a checkpoint length. That loop runs the last prompt token alone, so
for a block that ends on it the wrapper holds the block's chunk and runs
it with the last token in the loop's final forward.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)

_FLAG = "_gmlx_span_aware_prompt_step"
_GEN_FLAG = "_gmlx_block_aware_generate_step"
PINNED_STEP = "_gmlx_pinned_step"
_ABSENT = object()


def normalize_spans(image_spans) -> list[tuple[int, int]]:
    """``image_spans`` as sorted ``(start, end)`` int pairs; [] when absent."""
    if not image_spans:
        return []
    out = []
    for span in image_spans:
        s, e = int(span[0]), int(span[1])
        if e > s:
            out.append((s, e))
    out.sort()
    return out


def vision_spans(token_types) -> list[tuple[int, int]]:
    """Column spans ``[start, end)`` of the image and video blocks (token
    types 1 and 2) in ``token_types`` of shape (B, L), with runs that
    overlap across rows merged."""
    if token_types is None:
        return []
    t = np.asarray(token_types)
    if t.ndim == 1:
        t = t[None]
    visual = ((t == 1) | (t == 2)).astype(np.int8)
    runs = []
    for row in visual:
        edges = np.flatnonzero(np.diff(np.concatenate(([0], row, [0]))))
        runs.extend(zip(edges[0::2].tolist(), edges[1::2].tolist()))
    runs.sort()
    merged: list[tuple[int, int]] = []
    for a, b in runs:
        if merged and a < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def vision_blocks(model) -> bool:
    """True when ``model`` (a language model or its container) attends
    bidirectionally inside image and video blocks."""
    for m in (model, getattr(model, "language_model", None)):
        config = getattr(m, "config", None)
        for c in (config, getattr(config, "text_config", None)):
            if getattr(c, "use_bidirectional_attention", None) == "vision":
                return True
    return False


def token_types(kwargs: dict):
    """A prompt's token types from its kwargs, or None."""
    types = kwargs.get("mm_token_type_ids", None)
    return kwargs.get("token_type_ids", None) if types is None else types


def span_aware_chunk(image_spans, offset: int, n: int, max_n: int) -> int:
    """Chunk width for a chunk starting at ``offset`` with the candidate
    width ``n`` (``max_n`` = the widest chunk allowed)."""
    n = int(n)
    if n <= 0:
        return n
    off = int(offset)
    for a, b in normalize_spans(image_spans):
        p = off + n
        if a < p < b:
            if a > off:
                n = a - off
            elif a == off:
                n = min(b - off, int(max_n))
            # a < off: the chunk already starts inside the block; an earlier
            # boundary cut it and the tower reports that.
    return n


def _absolute_offset(batch) -> int:
    """Absolute prompt position of the batch's next chunk: the row's real
    tokens processed (restored prefix plus suffix columns, padding
    excluded) when APC metadata is present, else the processed columns."""
    meta = getattr(batch, "_apc_meta", None) or []
    if meta and meta[0] is not None:
        real: Callable[[int], int] | None = getattr(
            batch, "_row_real_tokens_processed", None)
        if callable(real):
            try:
                return int(real(0))
            except Exception:  # noqa: S110 - odd batch without the row helper; the column count is the offset
                pass
    return int(getattr(batch, "_processed_prompt_columns", 0))


def _batch_spans(batch) -> tuple[list[tuple[int, int]], int]:
    """The blocks of a prompt batch and the offset of its next chunk in
    their coordinates. ``image_spans`` are absolute and serve single-row
    batches only; token-type spans are columns of the remaining prompt,
    since the loops slice the types with the embeddings."""
    kwargs = getattr(batch, "_prompt_kwargs", None) or {}
    spans = kwargs.get("image_spans")
    if spans:
        if len(getattr(batch, "uids", ())) != 1:
            return [], 0
        return normalize_spans(spans), _absolute_offset(batch)
    types = token_types(kwargs)
    if types is None or not vision_blocks(getattr(batch, "model", None)):
        return [], 0
    return vision_spans(types), 0


def _drop_checkpoint_inside(batch, n: int) -> None:
    col_fn: Callable[[], int | None] | None = getattr(
        batch, "_next_apc_checkpoint_column", None)
    col = col_fn() if callable(col_fn) else None
    if col is None:
        return
    start = int(getattr(batch, "_processed_prompt_columns", 0))
    if not start < col < start + n:
        return
    for meta in getattr(batch, "_apc_meta", None) or ():
        if meta is not None and not meta.get("checkpoint_done"):
            meta["checkpoint_done"] = True
    _log.info("APC exact checkpoint at column %d dropped: it falls "
              "inside an image block", col)


def span_aware_prompt_n(batch, n: int) -> int:
    """Block-aware width for a prompt batch's next chunk (``n`` = the width
    the loop would use)."""
    spans, offset = _batch_spans(batch)
    embeds = getattr(batch, "_inputs_embeds", None)
    if not spans or embeds is None:
        return n
    n2 = span_aware_chunk(spans, offset, n, int(embeds.shape[1]) - 1)
    if n2 > n:
        _drop_checkpoint_inside(batch, n2)
    return n2


def rows_ending_inside(batch, n: int) -> list[int]:
    """Rows of a right-padded batch whose prompt ends strictly inside the
    next ``n`` columns. Their last real logits sit inside the chunk."""
    if getattr(batch, "_right_pad_per_row", None) is None:
        return []
    start = int(batch._processed_prompt_columns)
    return [i for i, e in enumerate(batch._suffix_lens)
            if start < e < start + n]


def _pending_end(batch, n: int) -> int:
    """``n`` clipped to the next right-padded row end, as stock clips it."""
    if getattr(batch, "_right_pad_per_row", None) is None:
        return n
    start = int(batch._processed_prompt_columns)
    ends = [e for e in batch._suffix_lens if e > start]
    return min(n, min(ends) - start) if ends else n


def install_span_aware_prompt_step(cls=None) -> bool:
    """Wrap ``PromptProcessingBatch.prompt_step`` (or ``cls``'s) with the
    block rule. Idempotent. Install before the prefill decay wrap so this
    runs inside it and reads the decayed step; either order stays correct
    for the extension, which is pinned past the decay resolver."""
    if cls is None:
        from mlx_vlm.generate.ar import PromptProcessingBatch as cls

    if getattr(cls, _FLAG, False):
        return True
    from gmlx.gen import prefill_decay
    from gmlx.spec.engine import _FULL_PREFILL_FLAG

    orig = cls.prompt_step

    def _span_prompt_step(self):
        # The owned MTP body keeps blocks whole itself, whichever of the
        # two wrappers installed first.
        if (getattr(self, "draft_kind", None) == "mtp"
                and getattr(type(self), _FULL_PREFILL_FLAG, False)):
            return orig(self)
        spans, _offset = _batch_spans(self)
        embeds = self._inputs_embeds
        if not spans or embeds is None:
            return orig(self)
        base = self.prefill_step_size
        step = prefill_decay.decayed_for_batch(self) or embeds.shape[1]
        n = _pending_end(self, min(step, embeds.shape[1] - 1))
        col = self._next_apc_checkpoint_column()
        if col is not None:
            n = min(n, col - self._processed_prompt_columns)
        n2 = span_aware_prompt_n(self, n)
        if n2 <= 0 or n2 == n:
            return orig(self)
        self.prefill_step_size = n2
        setattr(self, PINNED_STEP, n2)
        try:
            inside = rows_ending_inside(self, n2)
            if inside:
                return _step_past_row_ends(self, orig, inside, n2)
            return orig(self)
        finally:
            self.prefill_step_size = base
            setattr(self, PINNED_STEP, None)

    cls.prompt_step = _span_prompt_step
    setattr(cls, _FLAG, True)
    return True


def _step_past_row_ends(batch, step, rows: list[int], n: int):
    """Run ``step`` for a chunk of ``n`` columns that passes the ends of the
    right-padded ``rows``. Stock ends a chunk at the next row end, so the
    rows are presented as ending with the chunk, and each one then keeps
    the logits of its own last real column."""
    import mlx.core as mx

    start = int(batch._processed_prompt_columns)
    lens = batch._suffix_lens
    prefill = batch._speculative_prefill
    own = prefill.__dict__.get("append")
    append = prefill.append
    outputs: list[Any] = []

    def capture(out):
        outputs.append(out)
        append(out)

    batch._suffix_lens = [start + n if i in rows else e
                          for i, e in enumerate(lens)]
    prefill.append = capture
    try:
        result = step(batch)
    finally:
        batch._suffix_lens = lens
        if own is None:
            del prefill.append
        else:
            prefill.append = own
    if outputs:
        out = outputs[0]
        logits = out.logits if hasattr(out, "logits") else out
        kept = [mx.contiguous(logits[i, lens[i] - start - 1]) for i in rows]
        for i, row_logits in zip(rows, kept):
            batch._finished_prompt_logits[i] = row_logits
        mx.eval(kept)
    return result


def run_checkpoints(spans, length: int, step, checkpoints
                    ) -> tuple[Any, list[int], set[int], int | None]:
    """The step and checkpoint lengths that keep mlx-vlm's fixed-step loop
    off every block of a ``length``-token prompt: each block start becomes
    a checkpoint, the step grows to the longest block, and a caller
    checkpoint inside a block is dropped.

    The loop always runs the prompt's last token alone in its final
    forward, so a block of two or more tokens that ends on that token
    would be cut before it. Its start is returned as the tail, for
    ``_TailForward`` to run the block and the last token in that final
    forward. A block that is the whole prompt makes the step None: one
    pass, over the block alone.

    Returns (step, lengths, the caller checkpoints kept, the tail start or
    None)."""
    kept = set()
    for c in checkpoints:
        c = int(c)
        if any(a < c < b for a, b in spans):
            _log.info("APC exact checkpoint at %d dropped: it falls inside "
                      "an image block", c)
        else:
            kept.add(c)
    starts = {a for a, _b in spans if 0 < a < length}
    tail = None
    if step is not None and spans:
        step = max(int(step), max(b - a for a, b in spans))
        ending = [a for a, b in spans if b >= length > a + 1]
        if ending:
            tail = min(ending)
            if tail <= 0:
                step, tail = None, None
    return step, sorted(starts | kept), kept, tail


class _TailForward:
    """Stands in for ``model.language_model`` from the tail block's start
    to the end of mlx-vlm's prefill, on the thread that armed it.

    The loop's chunk from the tail start covers the block without its last
    token. That chunk is held, and it returns a stub whose hidden states
    are an empty list. The loop's final forward then runs the held columns
    and the last token in one call to the real language model, with the
    final forward's arguments, so the first logits come from it. The
    hidden states split at the last token: the held columns fill the
    stub's list, where the drafter's prefill capture reads them, and the
    final output keeps the last column, as after a one-token forward.
    Other threads, and any call that does not match, reach the real
    language model, and a mismatched chunk disarms the stand-in.
    """

    def __init__(self, model, width: int):
        self._model = model
        self._width = int(width)
        self._lm: Any = None
        self._thread: int | None = None
        self._held: tuple[Any, Any] | None = None
        self._hidden: list[Any] = []
        self._shadowed: Any = _ABSENT

    def arm(self) -> None:
        import threading

        self._lm = self._model.language_model
        self._thread = threading.get_ident()
        self._held = None
        # The instance dict, not setattr: an mlx Module's setattr of a
        # non-array value drops the child module from its tree.
        attrs = vars(self._model)
        self._shadowed = attrs.get("language_model", _ABSENT)
        attrs["language_model"] = self

    def disarm(self) -> None:
        attrs = vars(self._model)
        if attrs.get("language_model") is self:
            if self._shadowed is _ABSENT:
                del attrs["language_model"]
            else:
                attrs["language_model"] = self._shadowed
        self._thread = None

    def __getattr__(self, name: str):
        lm = self.__dict__.get("_lm")
        if lm is None:
            raise AttributeError(name)
        return getattr(lm, name)

    def __call__(self, *args, **kwargs):
        import threading

        lm = self._lm
        if threading.get_ident() != self._thread:
            return lm(*args, **kwargs)
        if self._held is None:
            ids, embeds = kwargs.get("inputs"), kwargs.get("inputs_embeds")
            if (kwargs.get("n_to_process") == self._width and not args
                    and ids is not None and embeds is not None):
                self._held = (ids, embeds)
                return _HeldChunk(self._hidden)
            self.disarm()
            return lm(*args, **kwargs)
        self.disarm()
        embeds = kwargs.get("inputs_embeds")
        if len(args) != 1 or embeds is None:
            return lm(*args, **kwargs)
        import mlx.core as mx

        held_ids, held_embeds = self._held
        self._held = None
        kwargs["inputs_embeds"] = mx.concatenate([held_embeds, embeds], axis=1)
        out = lm(mx.concatenate([held_ids, args[0]], axis=1), **kwargs)
        self._split_hidden(out, self._width + int(args[0].shape[1]))
        return out

    def _split_hidden(self, out, width: int) -> None:
        hidden = getattr(out, "hidden_states", None)
        if not isinstance(hidden, (list, tuple)) or not hidden:
            return
        if any(getattr(h, "ndim", 0) < 2 or h.shape[1] != width
               for h in hidden):
            return
        self._hidden.extend(h[:, :-1] for h in hidden)
        try:
            out.hidden_states = [h[:, -1:] for h in hidden]
        except AttributeError:
            del self._hidden[:]


class _HeldChunk:
    """The output of a held chunk: no logits (the loop drops them) and the
    hidden-state list the final forward fills."""

    logits = None

    def __init__(self, hidden: list[Any]):
        self.hidden_states = hidden


def _disarm_after(gen, forward: _TailForward):
    try:
        yield from gen
    finally:
        forward.disarm()


def _cache_offset(prompt_cache) -> int | None:
    """Tokens already in ``prompt_cache``: 0 for none, None when its first
    layer reports no single offset."""
    if not prompt_cache:
        return 0
    c = prompt_cache[0]
    if hasattr(c, "caches"):
        c = c.caches[0]
    off = getattr(c, "offset", None)
    if hasattr(off, "size"):
        off = int(off.item()) if off.size == 1 else None
    return off if isinstance(off, int) else None


def _run_spans(input_ids, model, kwargs: dict) -> list[tuple[int, int]]:
    """Block spans of the prompt ``generate_step`` prefills, in its own
    columns. ``image_spans`` are absolute, so they shift by the cached
    prefix. The token types cover the whole prompt, the cached prefix
    included, so the columns start where the prefix ends."""
    if input_ids is None:
        return []
    image_spans = kwargs.get("image_spans")
    if image_spans:
        prefix = _cache_offset(kwargs.get("prompt_cache"))
        blocks = normalize_spans(image_spans)
    else:
        types = token_types(kwargs)
        if types is None or not vision_blocks(model):
            return []
        prefix = int(types.shape[-1]) - int(input_ids.shape[-1])
        blocks = vision_spans(types)
    if prefix is None or prefix < 0:
        return []
    spans = []
    for a, b in blocks:
        if b > prefix:
            spans.append((max(a - prefix, 0), b - prefix))
    return spans


def install_block_aware_generate_step() -> bool:
    """Wrap the ``generate_step`` that mlx-vlm's ``stream_generate`` (run
    and chat) drives so its fixed-step prefill never cuts a block.
    Idempotent."""
    try:
        from mlx_vlm.generate import ar, dispatch
    except ImportError:
        return False
    orig = dispatch.generate_step
    if getattr(orig, _GEN_FLAG, False):
        return True

    def generate_step(input_ids, model, pixel_values, mask, **kwargs):
        spans = _run_spans(input_ids, model, kwargs)
        forward = None
        if spans:
            length = int(input_ids.shape[-1])
            caller = kwargs.get("prompt_cache_checkpoint")
            lengths = list(kwargs.get("prompt_cache_checkpoint_lengths") or [])
            if kwargs.get("prompt_cache_checkpoint_len") is not None:
                lengths.append(kwargs["prompt_cache_checkpoint_len"])
            if caller is None:
                lengths = []
            step, lengths, kept, tail = run_checkpoints(
                spans, length,
                kwargs.get("prefill_step_size", ar.DEFAULT_PREFILL_STEP_SIZE),
                lengths)
            if tail is not None:
                forward = _TailForward(model, length - 1 - tail)

            def checkpoint(n: int, prompt_cache) -> None:
                if caller is not None and n in kept:
                    caller(n, prompt_cache)
                if forward is not None and n == tail:
                    forward.arm()

            kwargs.update(prefill_step_size=step,
                          prompt_cache_checkpoint=checkpoint,
                          prompt_cache_checkpoint_len=None,
                          prompt_cache_checkpoint_lengths=lengths)
        gen = orig(input_ids, model, pixel_values, mask, **kwargs)
        return gen if forward is None else _disarm_after(gen, forward)

    setattr(generate_step, _GEN_FLAG, True)
    generate_step.__wrapped__ = orig  # type: ignore[attr-defined]
    dispatch.generate_step = generate_step
    return True
