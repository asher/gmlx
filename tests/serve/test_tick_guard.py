"""Engine-tick OOM containment: ledger, retry, retire-requeue, red exit."""

import types

import pytest

from gmlx._vendor.mlx_vlm.generate import ar

import gmlx.eval_guard as eg
import gmlx.serve.tick_guard as tg


class FakeGB:
    def __init__(self, uids, num_tokens):
        self.uids = list(uids)
        self._all_uids = list(uids)
        self._num_tokens = list(num_tokens)

    def __len__(self):
        return len(self.uids)


class FakeGen:
    def __init__(self, pending, gb):
        self._unprocessed_sequences = list(pending)
        self._generation_batch = gb
        self.removed = []

    def remove(self, uid):
        self.removed.append(uid)
        if uid in self._generation_batch.uids:
            self._generation_batch.uids.remove(uid)
            return True
        return False


def resp(uid, token, finish=None):
    return types.SimpleNamespace(uid=uid, token=token,
                                 finish_reason=finish)


MEM_ERR = RuntimeError(
    "[metal::malloc] Attempting to allocate 103918075904 bytes")

OOM_ERR = RuntimeError(
    "[METAL] Command buffer execution failed: Insufficient Memory "
    "(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).")

TIMEOUT_ERR = RuntimeError(
    "[METAL] Command buffer execution failed: Caused GPU Timeout Error "
    "(00000002:kIOGPUCommandBufferCallbackErrorTimeout).")

PAGE_FAULT_ERR = RuntimeError(
    "[METAL] Command buffer execution failed: Caused GPU Page Fault Error "
    "(00000003:kIOGPUCommandBufferCallbackErrorPageFault).")


@pytest.fixture
def wrapped(monkeypatch):
    """Install the guard over a scripted _next; yields (gen, script)."""
    monkeypatch.delenv("GMLX_TICK_GUARD", raising=False)
    script = []

    def scripted_next(self, **kw):
        step = script.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step

    monkeypatch.setattr(ar.BatchGenerator, "_next", scripted_next)
    drains = []
    monkeypatch.setattr(eg, "drain_for", lambda site: drains.append(site))
    assert tg.install_tick_guard() is True
    pending = [(7, [1, 2, 3], 16, {}, None, None),
               (8, [4, 5, 6, 9, 9, 9], 16, {}, None, None)]
    gen = FakeGen(pending, FakeGB([], []))
    return gen, script, drains


def tick(gen, **kw):
    return ar.BatchGenerator._next(gen, **kw)


def test_ledger_harvests_prompts_and_committed_tokens(wrapped):
    gen, script, _ = wrapped
    script.append(([], [resp(7, 11), resp(8, 21)]))
    script.append(([], [resp(7, 12), resp(8, 22, finish="stop")]))
    tick(gen)
    tick(gen)
    st = gen._kq_tick_guard
    assert st.ledger[7].prompt_ids == [1, 2, 3]
    assert st.ledger[7].committed == [11, 12]
    assert 8 not in st.ledger  # finished rows leave the ledger
    assert st.fail_streak == 0


def test_first_failure_drains_reclaims_and_retries(wrapped, monkeypatch):
    gen, script, drains = wrapped
    cleared = []
    monkeypatch.setattr(tg.mx, "clear_cache", lambda: cleared.append(1))
    script.append(MEM_ERR)
    script.append(([], [resp(7, 11)]))
    assert tick(gen) == ([], [])  # contained: empty tick
    assert drains == ["engine-tick"] and cleared == [1]
    assert gen.removed == []      # no shed on first failure
    tick(gen)
    assert gen._kq_tick_guard.fail_streak == 0  # success resets


def test_repeat_failure_retires_largest_and_requeues_same_uid(
        wrapped, monkeypatch):
    gen, script, _ = wrapped
    monkeypatch.setattr(tg.mx, "clear_cache", lambda: None)
    # both rows decoding; row 8 is larger (prompt 6 + 5 emitted)
    gen._generation_batch = FakeGB([7, 8], [3, 5])
    script.append(([], [resp(7, 11), resp(8, 21)]))
    tick(gen)
    script.append(MEM_ERR)
    script.append(MEM_ERR)
    tick(gen)
    assert gen.removed == []
    tick(gen)
    assert gen.removed == [8]
    uid, replay, remaining, kw, lp, tc = gen._unprocessed_sequences[0]
    assert uid == 8
    assert replay == [4, 5, 6, 9, 9, 9, 21]  # prompt + delivered
    assert remaining == 15                    # 16 - 1 delivered
    assert 8 in gen._kq_tick_guard.rebuilt


def test_third_strike_fails_row_permanently(wrapped, monkeypatch):
    gen, script, _ = wrapped
    monkeypatch.setattr(tg.mx, "clear_cache", lambda: None)
    failed = []
    monkeypatch.setattr(tg, "_row_failed_callbacks",
                        [lambda uid, info: failed.append((uid, info))])
    gen._generation_batch = FakeGB([8], [2])
    gen._kq_tick_guard = st = tg._TickState()
    st.rebuilt.add(8)
    script.append(([], [resp(8, 21)]))
    tick(gen)
    script.extend([MEM_ERR, MEM_ERR])
    tick(gen)
    tick(gen)
    assert gen.removed == [8]
    assert failed and failed[0][0] == 8
    assert failed[0][1]["delivered"] == 1
    assert 8 not in st.ledger


def test_non_memory_errors_propagate(wrapped):
    gen, script, drains = wrapped
    script.append(RuntimeError("unrelated explosion"))
    with pytest.raises(RuntimeError, match="unrelated"):
        tick(gen)
    assert drains == []


def test_inject_throw_is_contained(wrapped, monkeypatch):
    gen, script, drains = wrapped
    monkeypatch.setenv("GMLX_OOM_INJECT", "throw@2")
    tg._INJECT_FIRED.clear()
    monkeypatch.setattr(tg.mx, "clear_cache", lambda: None)
    script.append(([], [resp(7, 11)]))
    tick(gen)                              # tick 1: normal
    assert tick(gen) == ([], [])           # tick 2: synthetic, contained
    assert drains == ["engine-tick"]
    script.append(([], [resp(7, 12)]))
    tick(gen)                              # fires once; tick 3 normal
    assert gen._kq_tick_guard.fail_streak == 0


def test_kill_switch(monkeypatch):
    monkeypatch.setenv("GMLX_TICK_GUARD", "0")
    orig = ar.BatchGenerator._next
    assert tg.install_tick_guard() is False
    assert ar.BatchGenerator._next is orig


def test_classifier_separates_memory_from_gpu_faults():
    """Only the allocator's own words are a memory error. A timeout or a
    page fault says the submitted work was wrong, not that the box is
    full, so the ladder that sheds a row must not see them."""
    for e in (MEM_ERR, OOM_ERR):
        assert tg.is_memory_error(e) and not tg.is_gpu_fault(e), e
    for e in (TIMEOUT_ERR, PAGE_FAULT_ERR):
        assert tg.is_gpu_fault(e) and not tg.is_memory_error(e), e
    other = RuntimeError("shapes do not match")
    assert not tg.is_memory_error(other) and not tg.is_gpu_fault(other)


def test_gpu_fault_is_contained_once_then_propagates(wrapped, monkeypatch):
    """One retry covers a driver hiccup. A second fault propagates with
    the Metal error rather than shedding a row, because reclaiming
    buffers cannot fix a fault in the work itself."""
    gen, script, drains = wrapped
    cleared = []
    monkeypatch.setattr(tg.mx, "clear_cache", lambda: cleared.append(1))
    script.append(TIMEOUT_ERR)
    assert tick(gen) == ([], [])
    assert drains == ["engine-tick"] and cleared == [1]
    assert gen.removed == []

    script.append(TIMEOUT_ERR)
    with pytest.raises(RuntimeError, match="Timeout"):
        tick(gen)
    assert gen.removed == []          # never sheds a row on a fault
    assert drains == ["engine-tick"]  # no second reclaim


def test_gpu_fault_streak_resets_on_a_good_tick(wrapped, monkeypatch):
    gen, script, _ = wrapped
    monkeypatch.setattr(tg.mx, "clear_cache", lambda: None)
    script.append(PAGE_FAULT_ERR)
    tick(gen)
    script.append(([], [resp(7, 11)]))
    tick(gen)
    assert gen._kq_tick_guard.fault_streak == 0
    script.append(PAGE_FAULT_ERR)
    assert tick(gen) == ([], [])      # contained again after recovery


def test_metal_oom_still_takes_the_memory_ladder(wrapped, monkeypatch):
    """The OOM that arrives as a command buffer failure keeps the old
    behaviour: contained, then a row retired and requeued."""
    gen, script, drains = wrapped
    monkeypatch.setattr(tg.mx, "clear_cache", lambda: None)
    gen._generation_batch = FakeGB([7, 8], [4, 40])
    script.append(OOM_ERR)
    tick(gen)
    assert gen.removed == []
    script.append(OOM_ERR)
    tick(gen)
    assert gen.removed == [8]         # largest row retired, as before


def test_a_rebuilt_row_penalizes_its_delivered_tokens(wrapped, monkeypatch):
    from gmlx._vendor.mlx_lm.sample_utils import make_logits_processors

    gen, script, _ = wrapped
    monkeypatch.setattr(tg.mx, "clear_cache", lambda: None)
    procs = make_logits_processors(presence_penalty=1.5)
    gen._unprocessed_sequences[1] = (8, [4, 5, 6], 16, {}, procs, None)
    gen._generation_batch = FakeGB([7, 8], [1, 3])
    # The decode loop drives the row's processors as it delivers 21, 22.
    for history in ([4, 5, 6], [4, 5, 6, 21], [4, 5, 6, 21, 22]):
        procs[0](tg.mx.array(history), tg.mx.zeros((1, 32)))
    script.append(([], [resp(7, 11), resp(8, 21)]))
    script.append(([], [resp(8, 22)]))
    script.extend([MEM_ERR, MEM_ERR])
    for _ in range(4):
        tick(gen)
    uid, replay, _rem, _kw, lp, _tc = gen._unprocessed_sequences[0]
    assert uid == 8 and replay == [4, 5, 6, 21, 22]
    assert lp[0] is not procs[0]
    # An APC hit prefills only the uncached tail, so the replay's first
    # call sees just that.
    out = lp[0](tg.mx.array([22]), tg.mx.zeros((1, 32))).reshape(-1)
    assert {t: -v for t, v in enumerate(out.tolist()) if v} == {
        21: 1.5, 22: 1.5}


def test_a_rebuilt_row_keeps_its_row_settings(monkeypatch):
    """A row the guard retires and requeues keeps its uid, so it keeps its
    sampler and logprob settings: a greedy logprob row comes back greedy,
    with logprobs, in a generator whose default row samples."""
    import mlx.core as mx

    from gmlx._vendor.mlx_lm.models import llama
    from gmlx._vendor.mlx_vlm.utils import StoppingCriteria
    from gmlx.models.vlm_text_only import TextOnlyModel
    from gmlx.serve.patches.sampling import _FastPositionedSampler
    from gmlx.serve.row_sampling import RowSettings

    monkeypatch.setenv("GMLX_PREFILL_DECAY", "0")
    monkeypatch.setenv("GMLX_CB_PHASE", "0")
    mx.random.seed(0)
    model = TextOnlyModel(llama.Model(llama.ModelArgs(
        model_type="llama", hidden_size=32, num_hidden_layers=2,
        intermediate_size=64, num_attention_heads=4, num_key_value_heads=2,
        rms_norm_eps=1e-5, vocab_size=64)))
    lm = model.language_model
    proc = types.SimpleNamespace(tokenizer=types.SimpleNamespace(
        stopping_criteria=StoppingCriteria([], None)))
    gen = ar.BatchGenerator(lm, proc, sampler=_FastPositionedSampler(temperature=8.0),
                            greedy_sampling=False, prefill_step_size=None,
                            logprobs_cap=3)
    prompt = [3, 9, 4, 7]
    rs = RowSettings(logprobs=True, top_k=3)
    (uid,) = gen.insert([prompt], 8, prompt_kwargs=[
        {"inputs_embeds": lm.input_embeds(mx.array([prompt]))}], row_settings=[rs])
    st = tg._state(gen)
    tg._harvest_pending(gen, st)
    delivered = []
    for _ in range(3):
        delivered += [r.token for r in gen.next()[1]]
    st.ledger[uid].committed = list(delivered)
    replay = prompt + delivered
    st.ledger[uid].kwargs = {"inputs_embeds": lm.input_embeds(mx.array([replay]))}
    tg._rebuild_row(gen, st, uid)
    assert gen._row_settings[uid] is rs
    out = []
    while gen.has_work:
        out += gen.next()[1]
    gen.close()
    assert out and all(r.uid == uid for r in out)
    for r in out:
        assert r.token_logprob is not None and len(r.top_logprobs) == 3
        assert r.token == r.top_logprobs[0][0]          # greedy


def test_a_rebuilt_adapted_row_keeps_its_scales(monkeypatch, engine_seams):
    """In rows mode a row the guard retires and requeues keeps its uid, so
    its RowLora stays in the generator's store and each forward after the
    rebuild publishes the row's own strengths."""
    import mlx.core as mx
    import numpy as np

    from gmlx import lora_rows
    from gmlx._vendor.mlx_lm.models import llama
    from gmlx._vendor.mlx_vlm.utils import StoppingCriteria
    from gmlx.models.vlm_text_only import TextOnlyModel
    from gmlx.serve.lora_request import RowLora
    from gmlx.serve.row_sampling import RowSettings

    monkeypatch.setenv("GMLX_PREFILL_DECAY", "0")
    monkeypatch.setenv("GMLX_CB_PHASE", "0")
    mx.random.seed(0)
    model = TextOnlyModel(llama.Model(llama.ModelArgs(
        model_type="llama", hidden_size=32, num_hidden_layers=2,
        intermediate_size=64, num_attention_heads=4, num_key_value_heads=2,
        rms_norm_eps=1e-5, vocab_size=64)))
    lm = model.language_model
    seen = []
    forward = type(lm).__call__

    def recording(self, *a, **kw):
        seen.append(lora_rows.row_scales(0).tolist())
        return forward(self, *a, **kw)

    monkeypatch.setattr(type(lm), "__call__", recording)
    seams = engine_seams.snapshot()
    lora_rows.install_row_channel()
    lora_rows.configure("rows", 1)
    try:
        proc = types.SimpleNamespace(tokenizer=types.SimpleNamespace(
            stopping_criteria=StoppingCriteria([], None)))
        gen = ar.BatchGenerator(lm, proc, prefill_step_size=None)
        prompt = [3, 9, 4, 7]
        lora = RowLora((0.75,), np.asarray((0.75,), np.float32), 5, "m")
        (uid,) = gen.insert([prompt], 8, prompt_kwargs=[
            {"inputs_embeds": lm.input_embeds(mx.array([prompt]))}],
            row_settings=[RowSettings(lora=lora)])
        st = tg._state(gen)
        tg._harvest_pending(gen, st)
        delivered = []
        for _ in range(3):
            delivered += [r.token for r in gen.next()[1]]
        st.ledger[uid].committed = list(delivered)
        replay = prompt + delivered
        st.ledger[uid].kwargs = {"inputs_embeds": lm.input_embeds(mx.array([replay]))}
        tg._rebuild_row(gen, st, uid)
        assert gen._row_settings[uid].lora is lora
        seen.clear()
        while gen.has_work:
            gen.next()
        gen.close()
        assert seen and all(row == [0.75] for row in seen)
    finally:
        lora_rows.configure("static", 1)
        engine_seams.restore(seams)


def test_resumed_processors_resume_every_processor_that_can():
    """A grammar processor and its thinking wrapper resume from the
    delivered tokens, as the penalties do; any other processor is kept."""
    from gmlx._vendor.mlx_vlm.structured import (
        LLGuidanceLogitsProcessor,
        ThinkingAwareLogitsProcessor,
    )

    seen = []

    class _Resumable:
        def __init__(self, name):
            self.name = name

        def resumed(self, committed):
            seen.append((self.name, list(committed)))
            return self.name + "'"

    def plain(tokens, logits):
        return logits

    assert hasattr(LLGuidanceLogitsProcessor, "resumed")
    assert hasattr(ThinkingAwareLogitsProcessor, "resumed")
    lp = [_Resumable("grammar"), plain, _Resumable("wrapper")]
    out = tg._resumed_processors(lp, [5, 6])
    assert out == ["grammar'", plain, "wrapper'"]
    assert seen == [("grammar", [5, 6]), ("wrapper", [5, 6])]


def _salvaged(err, *responses):
    e = RuntimeError(str(err))
    e._kq_salvage = list(responses)
    return e


def test_a_contained_refusal_delivers_its_salvage(wrapped, monkeypatch):
    """What the failed tick computed goes out: the guard returns it and
    the ledger holds it, and a salvaged finish closes the row."""
    gen, script, drains = wrapped
    monkeypatch.setattr(tg.mx, "clear_cache", lambda: None)
    gen._generation_batch = FakeGB([7, 8], [1, 1])
    script.append(([], [resp(7, 11), resp(8, 21)]))
    tick(gen)
    salvage = [resp(7, 12), resp(8, 22, finish="stop")]
    script.append(_salvaged(MEM_ERR, *salvage))
    assert tick(gen) == ([], salvage)
    st = gen._kq_tick_guard
    assert st.ledger[7].committed == [11, 12]
    assert 8 not in st.ledger
    assert drains == ["engine-tick"] and st.fail_streak == 1


def test_a_command_buffer_oom_sends_no_salvage(wrapped, monkeypatch):
    """A command buffer that ran out of memory spoiled its outputs, so
    nothing it computed goes out."""
    gen, script, _ = wrapped
    monkeypatch.setattr(tg.mx, "clear_cache", lambda: None)
    script.append(_salvaged(OOM_ERR, resp(7, 11)))
    assert tick(gen) == ([], [])
    assert gen._kq_tick_guard.ledger[7].committed == []


def test_salvage_is_harvested_before_a_rebuild(wrapped, monkeypatch):
    """A row rebuilt on the strike that salvaged its token replays it."""
    gen, script, _ = wrapped
    monkeypatch.setattr(tg.mx, "clear_cache", lambda: None)
    gen._generation_batch = FakeGB([7, 8], [3, 5])
    script.append(([], [resp(7, 11), resp(8, 21)]))
    tick(gen)
    script.append(MEM_ERR)
    tick(gen)
    script.append(_salvaged(MEM_ERR, resp(8, 22)))
    _, out = tick(gen)
    assert [r.token for r in out] == [22] and gen.removed == [8]
    uid, replay, remaining, *_ = gen._unprocessed_sequences[0]
    assert uid == 8 and replay == [4, 5, 6, 9, 9, 9, 21, 22]
    assert remaining == 14


def test_salvage_rides_the_exception_ahead_of_inner_frames():
    e = RuntimeError(MEM_ERR)
    tg.salvage(e, [1, 2])
    tg.salvage(e, [0], ahead=True)
    tg.salvage(e, [])
    assert e._kq_salvage == [0, 1, 2]
