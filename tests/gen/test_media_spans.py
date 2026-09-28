"""Whole-block prefill chunking for media token blocks (gmlx.gen.media_spans)."""

import logging

import mlx.core as mx
import numpy as np
import pytest

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


def test_vision_spans_merge_rows_and_keep_audio_causal():
    types = mx.array([[0, 1, 1, 0, 2, 2, 0, 3, 3, 0],
                      [0, 0, 1, 1, 1, 0, 0, 0, 1, 0]])
    # image and video runs are blocks; audio (3) is not; runs overlapping
    # across rows merge, touching ones do not
    assert ms.vision_spans(types) == [(1, 6), (8, 9)]
    assert ms.vision_spans(mx.array([[1, 2, 2, 1]])) == [(0, 4)]
    assert ms.vision_spans(mx.array([[0, 3, 0]])) == []
    assert ms.vision_spans(None) == []


def test_vision_blocks_reads_the_text_config():
    from types import SimpleNamespace as NS

    vision = NS(use_bidirectional_attention="vision")
    assert ms.vision_blocks(NS(config=vision))
    assert ms.vision_blocks(NS(config=NS(text_config=vision)))
    assert ms.vision_blocks(NS(config=NS(), language_model=NS(config=vision)))
    assert not ms.vision_blocks(NS(config=NS(use_bidirectional_attention=None)))
    assert not ms.vision_blocks(None)


def test_span_aware_prompt_n_reads_token_types_on_bidirectional_models():
    from types import SimpleNamespace as NS

    b = _Batch(None, length=40, uids=(1, 2))
    b._prompt_kwargs = {"mm_token_type_ids": mx.array(
        [[0] * 6 + [1] * 14 + [0] * 20, [0] * 14 + [1] * 12 + [0] * 14])}
    b.model = NS(config=NS(use_bidirectional_attention="vision"))
    # the rows' blocks merge into [6, 26); columns are the remaining prompt
    assert ms.span_aware_prompt_n(b, 8) == 6
    assert ms.span_aware_prompt_n(b, 30) == 30
    b._prompt_kwargs["mm_token_type_ids"] = b._prompt_kwargs[
        "mm_token_type_ids"][:, 6:]
    assert ms.span_aware_prompt_n(b, 8) == 20
    b.model = NS(config=NS(use_bidirectional_attention=None))
    assert ms.span_aware_prompt_n(b, 8) == 8


def test_run_checkpoints_start_blocks_and_drop_caller_checkpoints_inside():
    spans = [(6, 20), (24, 34)]
    step, lengths, kept, tail = ms.run_checkpoints(spans, 40, 8, [10, 22, 36])
    assert step == 14  # grows to the longest block
    assert lengths == [6, 22, 24, 36]
    assert kept == {22, 36}
    assert tail is None
    assert ms.run_checkpoints(spans, 40, 2048, [])[0] == 2048
    # a None step (the caller declined chunking) stays None
    assert ms.run_checkpoints(spans, 40, None, [])[0] is None
    # a block at column 0 needs no checkpoint
    assert ms.run_checkpoints([(0, 5)], 40, 8, [])[1] == []


def test_run_checkpoints_a_block_ending_on_the_last_token():
    # step 50 and 64 around two blocks that end before the last token
    assert ms.run_checkpoints(SPANS, 140, 50, []) == (50, [40, 100], set(),
                                                      None)
    assert ms.run_checkpoints(SPANS, 140, 64, []) == (64, [40, 100], set(),
                                                      None)
    # the final-token reserve: the loop's final forward takes the block
    assert ms.run_checkpoints([(100, 140)], 140, 64, []) == (
        64, [100], set(), 100)
    # a prompt that fits one step still splits before its last token
    assert ms.run_checkpoints(SPANS, 140, 200, [])[3] is None
    assert ms.run_checkpoints([(100, 140)], 140, 200, [])[3] == 100
    # a caller checkpoint on the last token falls inside the block
    assert ms.run_checkpoints([(100, 140)], 140, 64, [139, 80]) == (
        64, [80, 100], {80}, 100)
    # a one-token block on the last token runs alone, as the loop runs it
    assert ms.run_checkpoints([(139, 140)], 140, 50, [])[3] is None
    # a block that is the whole prompt: one pass over the block
    assert ms.run_checkpoints([(0, 140)], 140, 50, []) == (None, [], set(),
                                                           None)
    # a one-token prompt is never chunked
    assert ms.run_checkpoints([(0, 1)], 1, 200, []) == (200, [], set(), None)
    assert ms.run_checkpoints([], 140, 50, []) == (50, [], set(), None)
    # a None step never chunks, so there is no tail to take
    assert ms.run_checkpoints([(100, 140)], 140, None, [])[3] is None


class _TailLM:
    """Records each call; outputs carry per-column hidden states."""

    def __init__(self):
        self.calls = []
        self.config = "cfg"

    def __call__(self, *args, **kwargs):
        from types import SimpleNamespace as NS

        ids = args[0] if args else kwargs["inputs"]
        self.calls.append((np.array(ids).tolist(), kwargs))
        cols = mx.array(np.array(ids), dtype=mx.float32)[..., None]
        return NS(logits=cols, hidden_states=[cols, cols * 2])


def test_tail_forward_runs_the_held_chunk_with_the_last_token():
    import threading
    from types import SimpleNamespace as NS

    lm = _TailLM()
    model = NS(language_model=lm)
    fwd = ms._TailForward(model, 3)
    fwd.arm()
    assert model.language_model is fwd and fwd.config == "cfg"
    ids = mx.array([[5, 6, 7, 8]])
    emb = mx.zeros((1, 4, 2))
    # another thread reaches the real model
    other = []
    t = threading.Thread(target=lambda: other.append(
        model.language_model(ids[:, :1], inputs_embeds=emb[:, :1])))
    t.start()
    t.join()
    assert len(lm.calls) == 1 and other[0].logits.shape == (1, 1, 1)
    held = model.language_model(inputs=ids[:, :3], inputs_embeds=emb[:, :3],
                                cache="c", n_to_process=3, logits_to_keep=1)
    assert held.logits is None and held.hidden_states == []
    assert len(lm.calls) == 1
    out = model.language_model(ids[:, 3:], inputs_embeds=emb[:, 3:],
                               cache="c", logits_to_keep=1, capture=True)
    assert model.language_model is lm
    call_ids, kw = lm.calls[1]
    assert call_ids == [[5, 6, 7, 8]]
    assert kw["inputs_embeds"].shape == (1, 4, 2)
    assert kw["cache"] == "c" and kw["capture"] and "n_to_process" not in kw
    assert np.array(out.logits).ravel().tolist() == [5, 6, 7, 8]
    # the held columns fill the chunk's list, the output keeps the last
    assert [np.array(h).ravel().tolist() for h in held.hidden_states] == [
        [5, 6, 7], [10, 12, 14]]
    assert [np.array(h).ravel().tolist() for h in out.hidden_states] == [
        [8], [16]]


def test_tail_forward_disarms_on_a_chunk_it_does_not_expect():
    from types import SimpleNamespace as NS

    lm = _TailLM()
    model = NS(language_model=lm)
    fwd = ms._TailForward(model, 3)
    fwd.arm()
    ids = mx.array([[5, 6]])
    out = model.language_model(inputs=ids, inputs_embeds=mx.zeros((1, 2, 2)),
                               n_to_process=2)
    assert model.language_model is lm and out.logits.shape == (1, 2, 1)
    fwd.arm()
    list(ms._disarm_after(iter([1, 2]), fwd))
    assert model.language_model is lm



def test_tail_forward_raises_on_a_final_forward_it_cannot_join():
    from types import SimpleNamespace as NS

    ids, emb = mx.array([[5, 6, 7, 8]]), mx.zeros((1, 4, 2))
    for final in ({"inputs": ids[:, 3:], "inputs_embeds": emb[:, 3:]},
                  {"args": (ids[:, 3:],)}):
        lm = _TailLM()
        model = NS(language_model=lm)
        fwd = ms._TailForward(model, 3)
        fwd.arm()
        model.language_model(inputs=ids[:, :3], inputs_embeds=emb[:, :3],
                             n_to_process=3)
        args = final.pop("args", ())
        with pytest.raises(RuntimeError, match="held chunk"):
            model.language_model(*args, **final)
        # Neither the held chunk nor the final forward reached the model.
        assert lm.calls == [] and model.language_model is lm
        assert fwd._held is None

def test_run_spans_start_after_the_cached_prefix():
    from types import SimpleNamespace as NS

    model = NS(config=NS(text_config=NS(use_bidirectional_attention="vision")))
    types = mx.array([[0] * 6 + [1] * 14 + [0] * 4 + [1] * 10 + [0] * 6])
    ids = mx.zeros((1, 40), dtype=mx.int32)
    assert ms._run_spans(ids, model, {"mm_token_type_ids": types}) == [
        (6, 20), (24, 34)]
    # a reused 20-token prefix: the suffix holds the second block only
    assert ms._run_spans(ids[:, 20:], model, {"mm_token_type_ids": types}) == [
        (4, 14)]
    assert ms._run_spans(ids, NS(config=NS()), {"mm_token_type_ids": types}) == []
    assert ms._run_spans(ids, model, {}) == []


def test_run_spans_shift_image_spans_by_the_cache_offset():
    from types import SimpleNamespace as NS

    ids = mx.zeros((1, 100), dtype=mx.int32)
    kw = {"image_spans": [[100, 124], [40, 64]]}
    assert ms._run_spans(ids, NS(), kw) == [(40, 64), (100, 124)]
    # a 64-token cached prefix, the first layer a cache list
    cache = [NS(caches=[NS(offset=64)])]
    assert ms._run_spans(ids, NS(), {**kw, "prompt_cache": cache}) == [
        (36, 60)]
    # an offset the wrapper cannot read leaves the loop unsteered
    assert ms._run_spans(ids, NS(), {**kw, "prompt_cache": [NS()]}) == []
