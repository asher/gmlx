# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Whole-block prefill chunking for prompts carrying media token blocks.

Some VLM prompts carry image token blocks whose attention is not causal
inside the block (DeepSeek-V4-Flash-Vision-Exp: a query inside a block
sees the whole block, END included). The text tower can only honor that
when every block lands whole inside one prefill chunk, so no chunk
boundary may fall inside a block. The processor that expands the
placeholders reports the blocks as ``image_spans``, a Python list of
absolute ``[start, end)`` prompt offsets (lead pads through END) riding
the prompt kwargs; the tower raises on a block a chunk cuts.

Rule, for a chunk starting at absolute offset ``off`` with a candidate
width ``n``: a boundary ``off + n`` strictly inside a block moves back to
the block's first token (cut in front of the block) when that still makes
progress, else (the chunk begins at the block and the step is shorter than
the block) forward to the token after END. The cut never widens a chunk;
the extension is bounded by one block (at most 384 tokens on DeepSeek-V4)
and is pinned past the depth-decay resolver so decay cannot re-cut it.

The stock exact-APC checkpoint column is a second boundary. The media
guard (``gmlx.cache.apc_media``) keeps it past the last block; should a
column still land inside one, the extension drops that checkpoint rather
than cut the block.

Three prefill loops consult this: the stock serve ``prompt_step`` (wrapped
by ``install_span_aware_prompt_step``), the owned MTP prefill (calls
``span_aware_prompt_n``), and run/chat's fixed-step loop, which cannot be
steered per chunk; there the language model's ``chunked_prefill_policy``
reads the step ``stamp_prefill_step`` recorded and declines chunking when
a boundary would cut a block.
"""

from __future__ import annotations

import logging

_log = logging.getLogger(__name__)

_FLAG = "_gmlx_span_aware_prompt_step"
STEP_STAMP = "_gmlx_prefill_step"
PINNED_STEP = "_gmlx_pinned_step"


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


def span_aware_chunk(image_spans, offset: int, n: int, max_n: int) -> int:
    """Chunk width for a chunk starting at absolute ``offset`` with the
    candidate width ``n`` (``max_n`` = the widest chunk allowed)."""
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


def chunk_boundaries_cut(image_spans, step: int, length: int) -> bool:
    """True when mlx-vlm's fixed-step loop over ``length`` prompt tokens
    (chunks of ``step``, the final token reserved for the first decode
    forward) would put a boundary strictly inside a block."""
    spans = normalize_spans(image_spans)
    step, length = int(step), int(length)
    if not spans or step <= 0 or length <= step:
        return False
    bounds = list(range(step, length - 1, step)) + [length - 1]
    return any(a < p < b for p in bounds for a, b in spans)


def prefill_step_stamp(language_model):
    return getattr(language_model, STEP_STAMP, None)


def stamp_prefill_step(model, step) -> None:
    """Record the run/chat prefill chunk width on the language model
    (``None`` = mlx-vlm's default) for ``chunked_prefill_policy``."""
    if step is None:
        try:
            from mlx_vlm.generate.ar import DEFAULT_PREFILL_STEP_SIZE as step
        except ImportError:
            step = 2048
    lm = getattr(model, "language_model", model)
    try:
        setattr(lm, STEP_STAMP, int(step))
    except (AttributeError, TypeError):
        pass


def _absolute_offset(batch) -> int:
    """Absolute prompt position of the batch's next chunk: the row's real
    tokens processed (restored prefix plus suffix columns, padding
    excluded) when APC metadata is present, else the processed columns."""
    meta = getattr(batch, "_apc_meta", None) or []
    if meta and meta[0] is not None:
        real = getattr(batch, "_row_real_tokens_processed", None)
        if callable(real):
            try:
                return int(real(0))
            except Exception:
                pass
    return int(getattr(batch, "_processed_prompt_columns", 0))


def _drop_checkpoint_inside(batch, n: int) -> None:
    col_fn = getattr(batch, "_next_apc_checkpoint_column", None)
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
    the loop would use). Single-row batches only."""
    spans = (getattr(batch, "_prompt_kwargs", None) or {}).get("image_spans")
    if not spans or len(getattr(batch, "uids", ())) != 1:
        return n
    embeds = getattr(batch, "_inputs_embeds", None)
    if embeds is None:
        return n
    n2 = span_aware_chunk(
        spans, _absolute_offset(batch), n, int(embeds.shape[1]) - 1)
    if n2 > n:
        _drop_checkpoint_inside(batch, n2)
    return n2


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

    orig = cls.prompt_step

    def _span_prompt_step(self):
        spans = (getattr(self, "_prompt_kwargs", None) or {}).get("image_spans")
        embeds = self._inputs_embeds
        if not spans or embeds is None:
            return orig(self)
        base = self.prefill_step_size
        step = prefill_decay.decayed_for_batch(self) or embeds.shape[1]
        n = min(step, embeds.shape[1] - 1)
        col = self._next_apc_checkpoint_column()
        if col is not None:
            n = min(n, col - self._processed_prompt_columns)
        n2 = span_aware_prompt_n(self, n)
        if n2 <= 0 or n2 == n:
            return orig(self)
        self.prefill_step_size = n2
        setattr(self, PINNED_STEP, n2)
        try:
            return orig(self)
        finally:
            self.prefill_step_size = base
            setattr(self, PINNED_STEP, None)

    cls.prompt_step = _span_prompt_step
    setattr(cls, _FLAG, True)
    return True
