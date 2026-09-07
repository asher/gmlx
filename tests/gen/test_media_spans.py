"""Whole-block prefill chunking for media token blocks (gmlx.gen.media_spans)."""

import logging

import mlx.core as mx

from gmlx.gen import media_spans as ms
from gmlx.gen import prefill_decay

# One 24-token block at [40, 64) and one at [100, 124).
SPANS = [[40, 64], [100, 124]]


def test_span_aware_chunk_cuts_at_block_start():
    # step lands inside the first block: cut in front of it.
    assert ms.span_aware_chunk(SPANS, 0, 50, 400) == 40
    # boundary exactly at the block start or end is not a cut
    assert ms.span_aware_chunk(SPANS, 0, 40, 400) == 40
    assert ms.span_aware_chunk(SPANS, 0, 64, 400) == 64
    # a chunk that ends before the block is untouched
    assert ms.span_aware_chunk(SPANS, 0, 30, 400) == 30


def test_span_aware_chunk_extends_only_at_zero_progress():
    # chunk begins at the block, step shorter than the block: extend to END
    assert ms.span_aware_chunk(SPANS, 40, 8, 400) == 24
    # the extension is capped by max_n (the reserve for the final token)
    assert ms.span_aware_chunk(SPANS, 40, 8, 20) == 20
    # lead pads are inside the block: a cut at 41 (past the first pad) is
    # never produced, the cut lands on the block's first token
    assert ms.span_aware_chunk(SPANS, 38, 5, 400) == 2


def test_span_aware_chunk_two_blocks_and_no_spans():
    assert ms.span_aware_chunk(SPANS, 0, 110, 400) == 100
    assert ms.span_aware_chunk(SPANS, 64, 40, 400) == 36
    assert ms.span_aware_chunk(None, 0, 50, 400) == 50
    assert ms.span_aware_chunk([], 7, 50, 400) == 50
    assert ms.span_aware_chunk(SPANS, 0, 0, 400) == 0


def test_normalize_spans_sorts_and_drops_empty():
    assert ms.normalize_spans([[100, 124], (40, 64), [70, 70]]) == [
        (40, 64), (100, 124)]
    assert ms.normalize_spans(None) == []


def test_chunk_boundaries_cut_matches_fixed_step_loop():
    # 140-token prompt, step 50: boundaries 50, 100, 139 -> 50 cuts block 1
    assert ms.chunk_boundaries_cut(SPANS, 50, 140)
    # step 40: boundaries 40, 80, 120, 139 -> 120 cuts block 2
    assert ms.chunk_boundaries_cut(SPANS, 40, 140)
    # step 64: boundaries 64, 128, 139 -> clean
    assert not ms.chunk_boundaries_cut(SPANS, 64, 140)
    # the final-token reserve: a block ending on the last token is cut
    assert ms.chunk_boundaries_cut([[100, 140]], 64, 140)
    # no chunking at all when the prompt fits one step
    assert not ms.chunk_boundaries_cut(SPANS, 200, 140)
    assert not ms.chunk_boundaries_cut(None, 50, 140)


class _LM:
    pass


class _Container:
    def __init__(self):
        self.language_model = _LM()


def test_stamp_prefill_step_lands_on_language_model():
    m = _Container()
    ms.stamp_prefill_step(m, 512)
    assert ms.prefill_step_stamp(m.language_model) == 512
    ms.stamp_prefill_step(m, None)
    from mlx_vlm.generate.ar import DEFAULT_PREFILL_STEP_SIZE
    assert ms.prefill_step_stamp(m.language_model) == DEFAULT_PREFILL_STEP_SIZE
    bare = _LM()
    ms.stamp_prefill_step(bare, 7)
    assert ms.prefill_step_stamp(bare) == 7


class _Batch:
    """The PromptProcessingBatch surface span_aware_prompt_n reads."""

    def __init__(self, spans, processed=0, length=200, uids=(1,),
                 checkpoint=None, meta=None):
        self._prompt_kwargs = {"image_spans": spans}
        self._processed_prompt_columns = processed
        self._inputs_embeds = mx.zeros((1, length, 4))
        self.uids = list(uids)
        self._checkpoint = checkpoint
        self._apc_meta = meta

    def _next_apc_checkpoint_column(self):
        return self._checkpoint


def test_span_aware_prompt_n_uses_processed_columns():
    assert ms.span_aware_prompt_n(_Batch(SPANS, processed=0), 50) == 40
    assert ms.span_aware_prompt_n(_Batch(SPANS, processed=40), 8) == 24
    assert ms.span_aware_prompt_n(_Batch(SPANS, processed=64), 50) == 36
    assert ms.span_aware_prompt_n(_Batch(SPANS, processed=64), 100) == 100
    # multi-row batches and text prompts pass through
    assert ms.span_aware_prompt_n(_Batch(SPANS, uids=(1, 2)), 50) == 50
    assert ms.span_aware_prompt_n(_Batch(None), 50) == 50


def test_span_aware_prompt_n_extension_capped_by_final_token_reserve():
    # the batch trims processed columns off _inputs_embeds: 24 remain
    b = _Batch([[40, 64]], processed=40, length=24)
    assert ms.span_aware_prompt_n(b, 8) == 23


def test_extension_drops_checkpoint_inside_block(caplog):
    meta = [{"checkpoint_done": False}]
    b = _Batch(SPANS, processed=40, checkpoint=50, meta=meta)
    with caplog.at_level(logging.INFO, logger="gmlx.gen.media_spans"):
        assert ms.span_aware_prompt_n(b, 8) == 24
    assert meta[0]["checkpoint_done"] is True
    assert "checkpoint" in caplog.text
    # a checkpoint past the block is left alone
    meta = [{"checkpoint_done": False}]
    b = _Batch(SPANS, processed=40, checkpoint=90, meta=meta)
    assert ms.span_aware_prompt_n(b, 8) == 24
    assert meta[0]["checkpoint_done"] is False


def test_absolute_offset_prefers_real_tokens_with_apc_meta():
    b = _Batch(SPANS, processed=10, meta=[{"checkpoint_done": True}])
    b._row_real_tokens_processed = lambda i: 1000 + i
    assert ms._absolute_offset(b) == 1000
    b2 = _Batch(SPANS, processed=10)
    b2._row_real_tokens_processed = lambda i: 1000
    assert ms._absolute_offset(b2) == 10


def test_pinned_step_wins_in_decay_resolver():
    b = _Batch(SPANS)
    b.prefill_step_size = 2048
    b._gmlx_pinned_step = 24
    assert prefill_decay.decayed_for_batch(b) == 24
    b._gmlx_pinned_step = None
    b.prefill_step_size = 0
    assert prefill_decay.decayed_for_batch(b) == 0


def test_install_wrapper_pins_the_extension(monkeypatch):
    seen = []

    class Stub:
        _prompt_kwargs = {"image_spans": [[40, 64]]}
        _inputs_embeds = mx.zeros((1, 200, 4))
        _processed_prompt_columns = 40
        uids = [1]
        _apc_meta = None
        prefill_step_size = 8

        def _next_apc_checkpoint_column(self):
            return None

        def prompt_step(self):
            seen.append((self.prefill_step_size,
                         getattr(self, ms.PINNED_STEP, None)))
            return "ok"

    monkeypatch.setattr(prefill_decay, "decayed_for_batch",
                        lambda batch: batch.prefill_step_size)
    assert ms.install_span_aware_prompt_step(Stub)
    assert ms.install_span_aware_prompt_step(Stub)  # idempotent
    inst = Stub()
    assert inst.prompt_step() == "ok"
    assert seen == [(24, 24)]
    assert inst.prefill_step_size == 8
    assert getattr(inst, ms.PINNED_STEP) is None
    # a cut (no extension) also runs through the pin
    seen.clear()
    inst._processed_prompt_columns = 0
    inst.prefill_step_size = 50
    assert inst.prompt_step() == "ok"
    assert seen == [(40, 50)] or seen == [(40, 40)]
    # text prompts chain straight through
    seen.clear()
    inst._prompt_kwargs = {}
    assert inst.prompt_step() == "ok"
    assert seen == [(50, None)]
    # the real class takes the wrap too (idempotent)
    from mlx_vlm.generate.ar import PromptProcessingBatch
    assert ms.install_span_aware_prompt_step()
    assert getattr(PromptProcessingBatch, "_gmlx_span_aware_prompt_step")
