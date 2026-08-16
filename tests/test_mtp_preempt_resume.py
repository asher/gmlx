"""MTP preempt + resume: a live B=1 speculative generation preempts at a
verify-round boundary when prefilled waiters queue (the scalar generator
closes, the batch rebuilds armless on the batch loop, waiters join through
the injection drain), and a width-gated batch that drains back under the cap
resumes speculation through a capture round.

Loop-level tests drive _owned_decode_rounds_batch with an armable fake
drafter whose draft_block returns an EMPTY draft, so armed rounds run the
full speculative machinery (S=1 verify + zero-draft walk) with deterministic
echo streams. Engine-level tests run the real SpeculativeGenerationBatch
against a recording fake rounds generator, the same pattern as
test_spec_engine_release.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from gmlx.speculative import _owned_decode_rounds_batch, _width_cap_logged
import gmlx.speculative as spec

from test_mtp_width_cap import _FakeCache, _StrictDrafter

VOCAB = 32


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("GMLX_MTP_WIDTH_CAP", "MLX_VLM_GGUF_SPEC_WIDTH_CAP",
                "GMLX_MTP_PREEMPT", "GMLX_MTP_RESUME"):
        monkeypatch.delenv(var, raising=False)
    spec._width_cap_memo = ("", None)
    _width_cap_logged.clear()
    yield
    spec._width_cap_memo = ("", None)
    _width_cap_logged.clear()


class _VerifyEchoLM:
    """Echo target (next token = input + 1) that also serves the
    plain-forward verify branch (hidden_states / shared_kv / logits), so
    capture rounds and zero-draft speculative rounds run end to end.
    Plain and verify forwards are recorded separately: the consume-then-
    capture protocol is asserted from these counters."""

    def __init__(self):
        self.plain_widths = []
        self.verify_widths = []
        self._rope_deltas = None

    def __call__(self, x, cache=None, return_hidden=False,
                 return_shared_kv=False, **kw):
        B, S = x.shape
        nxt = (x + 1) % VOCAB
        onehot = (mx.arange(VOCAB)[None, None, :] == nxt[:, :, None])
        logits = onehot.astype(mx.float32) * 10.0
        if return_hidden:
            self.verify_widths.append(B)
            return SimpleNamespace(
                logits=logits,
                hidden_states=[mx.zeros((B, S, 8))],
                shared_kv_states={"full": (mx.zeros((B, 2, S, 4)),
                                           mx.zeros((B, 2, S, 4)))},
                gdn_states=None)
        self.plain_widths.append(B)
        return SimpleNamespace(logits=logits)


class _ArmableDrafter:
    """Cold-startable fake: the seed calls (reset / prefill_from_target_hidden
    / set_shared_kv) are recorded, draft_block returns an empty draft so armed
    rounds exercise verify + walk without draft-quality modeling."""

    uses_shared_kv = True

    def __init__(self, cap=2, block_size=4):
        self.config = SimpleNamespace(block_size=block_size)
        self.accept_lens = []
        self.draft_lens = []
        self.mtp_width_cap = cap
        self.mtp_width_limit = 0
        self.reset_calls = []
        self.prefill_calls = []
        self.shared_kv_calls = []
        self.draft_calls = []

    def reset(self, model, left_padding=None):
        self.reset_calls.append(left_padding)
        return []

    def set_shared_kv(self, shared_kv, kv_offset=0, position=None,
                      kv_valid_len=None, left_padding=None):
        self.shared_kv_calls.append(kv_offset)

    def prefill_from_target_hidden(self, tokens, hidden, next_tokens,
                                   sampler, dtype, **kw):
        self.prefill_calls.append(tuple(hidden.shape))

    def draft_block(self, b, hidden, kv, n, sampler, dtype, **kw):
        self.draft_calls.append(int(b.shape[0]))
        return mx.zeros((int(b.shape[0]), 0), dtype=dtype)


def _drive_armless(drafter, *, B, max_tokens, lm=None, model=None,
                   rounds=None, stop_check=None):
    """Run the batch loop from an armless start (hidden=None, shared_kv=None),
    the state a preempted scalar generation rebuilds into."""
    model = model if model is not None else SimpleNamespace()
    lm = lm if lm is not None else _VerifyEchoLM()
    prompt_cache = [_FakeCache(width=B)]
    gen = _owned_decode_rounds_batch(
        model, drafter, lm, prompt_cache,
        hidden=None,
        b=list(range(1, B + 1)),
        shared_kv=None,
        seed_tokens=None,
        emitted=[1] * B,
        max_tokens=max_tokens,
        sampler=None,
        draft_block_size=None,
        stop_check=stop_check,
    )
    out = []
    for toks, meta in gen:
        out.append((toks, meta))
        if rounds is not None and len(out) >= rounds:
            gen.close()
            break
    return out, lm, prompt_cache


# -- armless entry (rebuilt after preempt) --------------------------------


def test_armless_entry_over_cap_gates_plain():
    """A rebuilt batch born over the cap decodes plain with hidden=None
    everywhere: nothing on the gated path may dereference it."""
    d = _StrictDrafter(cap=2)
    lm = _VerifyEchoLM()
    out, _, _ = _drive_armless(d, B=3, max_tokens=4, lm=lm)
    assert d.forward_calls == []
    assert d.shared_kv_calls == 0
    assert lm.verify_widths == []
    # echo chains from seeds [1, 2, 3]
    assert [toks for toks, _ in out] == [[2, 3, 4], [3, 4, 5], [4, 5, 6]]


def test_armless_entry_capture_arms_then_speculates():
    """Under the cap, the first round is a capture round (S=1 verify, drafter
    cold start) and later rounds run the speculative machinery. The token
    stream is the same plain echo chain: arming must not skip or duplicate."""
    d = _ArmableDrafter(cap=0)
    out, lm, _ = _drive_armless(d, B=2, max_tokens=4)
    assert [toks for toks, _ in out] == [[2, 3], [3, 4], [4, 5]]
    # every round is a verify (capture round + zero-draft rounds), no plain
    assert lm.verify_widths == [2, 2, 2]
    assert lm.plain_widths == []
    # cold start ran exactly once, from the 1-token capture
    assert d.prefill_calls == [(2, 1, 8)]
    # rounds after the capture actually drafted (empty blocks)
    assert d.draft_calls == [2, 2]
    # tail re-set the drafter's shared-KV view each armed round except the
    # last (the all-finished break exits before the tail)
    assert len(d.shared_kv_calls) == 2


def test_armless_capture_skips_entry_seed():
    """The entry seed block (prefill from seed_tokens + set_shared_kv at
    L_prefill) reads state an armless start does not have; the capture round
    is the only seed path."""
    d = _ArmableDrafter(cap=0)
    _drive_armless(d, B=2, max_tokens=2)
    assert len(d.prefill_calls) == 1


# -- resume (gated batch drains under the cap) ----------------------------


def _finish_row0(orig, tok):
    return orig == 0


def test_row_finish_filters_without_shared_kv():
    """A drafter with uses_shared_kv=False leaves next_shared_kv None; the
    first mid-batch row finish takes the keep-slots filter, which must not
    iterate it (the qwen35 owned-MTP concurrent-serve crash)."""
    d = _ArmableDrafter(cap=4)
    d.uses_shared_kv = False
    out, lm, cache = _drive_armless(
        d, B=2, max_tokens=40, rounds=6, stop_check=_finish_row0)
    assert cache[0].width == 1  # row 0 filtered out, loop kept running
    assert all(toks[0] is None for toks, _ in out[1:])
    assert all(toks[1] is not None for toks, _ in out)


def test_resume_after_drain_rearms_and_streams():
    """B=3 over cap=2 gates at formation; row 0 finishing drains the batch to
    the cap. The next round consumes the dispatched plain lookahead (its KV
    is already in the cache), the round after arms via capture, and the
    stream stays the exact echo chain across both transitions."""
    d = _ArmableDrafter(cap=2)
    out, lm, _ = _drive_armless(
        d, B=3, max_tokens=40, rounds=5, stop_check=_finish_row0)
    assert [toks for toks, _ in out] == [
        [2, 3, 4],        # gated round; row 0 finishes
        [None, 4, 5],     # consume round: dispatched lookahead, no re-prime
        [None, 5, 6],     # capture round (arm)
        [None, 6, 7],     # speculative round
        [None, 7, 8],
    ]
    # gated rounds: prime + one dispatch at width 3, then nothing plain --
    # a discarded (re-primed) buffer would show a third plain forward
    assert lm.plain_widths == [3, 3]
    # capture + speculative rounds verify at the drained width
    assert lm.verify_widths == [2, 2, 2]
    assert d.prefill_calls == [(2, 1, 8)]
    assert d.draft_calls == [2, 2]


def test_resume_env_kill_switch(monkeypatch):
    """GMLX_MTP_RESUME=0 keeps the drained batch on plain decode (the old
    latch behavior); the strict drafter would raise on any arm attempt."""
    monkeypatch.setenv("GMLX_MTP_RESUME", "0")
    d = _StrictDrafter(cap=2)
    out, lm, _ = _drive_armless(
        d, B=3, max_tokens=40, rounds=5, stop_check=_finish_row0,
        lm=_VerifyEchoLM())
    assert d.forward_calls == []
    assert len(out) == 5
    assert out[-1][0] == [None, 7, 8]


def test_resume_skipped_near_budget_end():
    """Re-arming costs a capture forward + a drafter seed; rows within
    _RESUME_MIN_REMAINING of their budget finish plain instead."""
    d = _StrictDrafter(cap=2)
    out, _, _ = _drive_armless(
        d, B=3, max_tokens=8, lm=_VerifyEchoLM(), stop_check=_finish_row0)
    assert d.forward_calls == []
    # rows 1..2 run to their budget (emitted 1 -> 8 = 7 rounds)
    assert len(out) == 7


def test_resume_rechecks_cap_on_new_admission():
    """An injection landing in the same round a resume would fire must win:
    the drain runs first and re-trips the gate, so the batch never arms over
    the cap."""
    d = _StrictDrafter(cap=2)
    model = SimpleNamespace()
    lm = _VerifyEchoLM()
    prompt_cache = [_FakeCache(width=3)]
    gen = _owned_decode_rounds_batch(
        model, d, lm, prompt_cache,
        hidden=None, b=[1, 2, 3], shared_kv=None, seed_tokens=None,
        emitted=[1, 1, 1], max_tokens=40, sampler=None,
        draft_block_size=None, stop_check=_finish_row0)
    next(gen)          # gated round; row 0 finishes -> batch at the cap
    next(gen)          # consume round; a resume is now due next round
    model._generator_injections = [{
        "uids": ["w", "x"],
        "prompt_cache": [_FakeCache(width=2, offset=9)],
        "hidden": mx.zeros((2, 1, 8)),
        "prompt_tokens": mx.zeros((2, 4), dtype=mx.int32),
        "first_tokens": mx.array([7, 8], dtype=mx.int32),
        "first_tokens_list": [7, 8],
        "shared_kv_states": None,
    }]
    toks, _ = next(gen)
    gen.close()
    assert d.forward_calls == []          # never armed
    assert len(toks) == 5                 # both admissions joined the round


# -- rebuilt-generator emitted override -----------------------------------


def test_rebuild_emitted_override_consumed():
    """owned_server_rounds_batch picks up model._kq_rebuild_emitted (the
    preempted host's real emitted count), applies it to the budget, and
    deletes the attribute."""
    from gmlx.speculative import owned_server_rounds_batch

    d = _ArmableDrafter(cap=0)
    lm = _VerifyEchoLM()
    model = SimpleNamespace()
    model._kq_rebuild_emitted = [5]
    gen = owned_server_rounds_batch(
        model, d, [_FakeCache(width=1)],
        None,
        first_bonus=mx.array([3], dtype=mx.int32),
        max_tokens=8,
        sampler=None,
        shared_kv_states=None,
        prompt_tokens=None,
        greedy_sampling=True,
    )
    # patch lm resolution: owned_server_rounds_batch derives lm from model
    model.language_model = lm
    out = [toks for toks, _ in gen]
    assert not hasattr(model, "_kq_rebuild_emitted")
    # emitted resumes at 5, so budget 8 leaves exactly 3 more tokens
    assert out == [[4], [5], [6]]


def test_rebuild_emitted_length_mismatch_ignored():
    """A stale or mis-sized override must not survive into an unrelated
    batch: wrong length falls back to the fresh-generator default."""
    from gmlx.speculative import owned_server_rounds_batch

    d = _ArmableDrafter(cap=0)
    lm = _VerifyEchoLM()
    model = SimpleNamespace()
    model._kq_rebuild_emitted = [5]
    gen = owned_server_rounds_batch(
        model, d, [_FakeCache(width=2)],
        None,
        first_bonus=mx.array([3, 4], dtype=mx.int32),
        max_tokens=3,
        sampler=None,
        shared_kv_states=None,
        prompt_tokens=None,
        greedy_sampling=True,
    )
    model.language_model = lm
    out = [toks for toks, _ in gen]
    assert not hasattr(model, "_kq_rebuild_emitted")
    assert out == [[4, 5], [5, 6]]


# -- engine-level preempt (real SpeculativeGenerationBatch) ---------------


class _HostCache:
    """Single-sequence cache: no filter/extend, so the preempt must lift it
    through type(c).merge([c])."""

    def __init__(self):
        self._gmlx_cascade = "stamp"

    @classmethod
    def merge(cls, caches):
        lifted = _BatchCache()
        lifted.merged_from = list(caches)
        return lifted


class _BatchCache:
    merged_from = None

    def filter(self, keep):
        pass

    def extend(self, other):
        pass


class _EngineDrafter:
    def __init__(self):
        self.reset_calls = 0

    def reset(self, model, left_padding=None):
        self.reset_calls += 1
        return []


def _make_batch(ar, *, uids=(0,), model=None, max_tokens=6, cache=None):
    return ar.SpeculativeGenerationBatch(
        model=model if model is not None else SimpleNamespace(),
        draft_model=_EngineDrafter(),
        draft_kind="mtp",
        uids=list(uids),
        first_tokens=mx.array([5 + u for u in uids]),
        prompt_cache=[cache if cache is not None else _HostCache()],
        sampler=None,
        stop_criteria=lambda tok: False,
        max_tokens=[max_tokens] * len(uids),
        hidden=mx.zeros((len(uids), 4, 8)),
        shared_kv_states={"full": None},
        prompt_tokens=mx.array([[1, 2, 3]] * len(uids)),
        greedy_sampling=True,
    )


def _recording_rounds(calls):
    """Fake rounds generator: records its (hidden, first_bonus, cache) per
    call, drains model._generator_injections the way the real batch loop
    does (widening its yields), and flags close."""

    def fake_rounds(model, draft_model, prompt_cache, hidden, **kw):
        entry = {"hidden": hidden, "first_bonus": kw.get("first_bonus"),
                 "cache": list(prompt_cache), "closed": False}
        calls.append(entry)
        width = int(entry["first_bonus"].shape[0])
        base = 100 * len(calls)
        n = 0
        try:
            while True:
                inj = getattr(model, "_generator_injections", None)
                if inj:
                    width += sum(len(e["uids"]) for e in inj)
                    inj.clear()
                n += 1
                yield [base + n] * width, None
        finally:
            entry["closed"] = True

    return fake_rounds


def test_preempt_rebuilds_scalar_for_waiters(monkeypatch):
    from mlx_vlm.generate import ar
    from gmlx.spec_engine import install_continuous_batch_admission

    install_continuous_batch_admission()
    calls = []
    monkeypatch.setattr(ar, "run_speculative_server_rounds",
                        _recording_rounds(calls))

    model = SimpleNamespace()
    host = _make_batch(ar, uids=(0,), model=model)
    assert [r.token for r in host.next()] == [5]        # first token
    assert [r.token for r in host.next()] == [101]      # scalar round

    waiter = _make_batch(ar, uids=(7,), model=model)
    host.extend(waiter)                                  # buffered

    responses = host.next()
    # scalar generator closed at its round boundary
    assert calls[0]["closed"] is True
    # rebuilt armless from the last delivered token
    assert len(calls) == 2
    assert calls[1]["hidden"] is None
    assert calls[1]["first_bonus"].tolist() == [101]
    # host cache lifted to the batch class, cascade stamp preserved
    lifted = calls[1]["cache"][0]
    assert isinstance(lifted, _BatchCache)
    assert isinstance(lifted.merged_from[0], _HostCache)
    assert lifted._gmlx_cascade == "stamp"
    # real emitted count handed to the rebuilt generator
    assert model._kq_rebuild_emitted == [2]
    # waiter delivered its first token and joined the same round
    assert [(r.uid, r.token) for r in responses] == [
        (7, 12), (0, 201), (7, 201)]


def test_preempt_env_kill_switch(monkeypatch):
    from mlx_vlm.generate import ar
    from gmlx.spec_engine import install_continuous_batch_admission

    install_continuous_batch_admission()
    monkeypatch.setenv("GMLX_MTP_PREEMPT", "0")
    calls = []
    monkeypatch.setattr(ar, "run_speculative_server_rounds",
                        _recording_rounds(calls))

    model = SimpleNamespace()
    host = _make_batch(ar, uids=(0,), model=model)
    host.next()
    host.next()
    host.extend(_make_batch(ar, uids=(7,), model=model))

    responses = host.next()
    # old behavior: scalar keeps running, waiter stays buffered until drain
    assert len(calls) == 1
    assert calls[0]["closed"] is False
    assert [(r.uid, r.token) for r in responses] == [(0, 102)]
    assert len(host._pending_injections) == 1


def test_preempt_waits_for_first_delivery(monkeypatch):
    """A batch that has not delivered its first tokens has no bonus to
    rebuild from; the preempt fires on the following next() instead."""
    from mlx_vlm.generate import ar
    from gmlx.spec_engine import install_continuous_batch_admission

    install_continuous_batch_admission()
    calls = []
    monkeypatch.setattr(ar, "run_speculative_server_rounds",
                        _recording_rounds(calls))

    model = SimpleNamespace()
    host = _make_batch(ar, uids=(0,), model=model)
    host.extend(_make_batch(ar, uids=(7,), model=model))

    assert [r.token for r in host.next()] == [5]     # no preempt yet
    assert calls == []

    responses = host.next()
    assert len(calls) == 1
    assert calls[0]["hidden"] is None
    assert calls[0]["first_bonus"].tolist() == [5]
    assert model._kq_rebuild_emitted == [1]
    assert [(r.uid, r.token) for r in responses] == [
        (7, 12), (0, 101), (7, 101)]
