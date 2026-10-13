"""Mixed batches at the width cap: a batch wider than the drafter's cap keeps
speculating on the oldest ``cap`` rows and verifies every other row in the
same forward behind padding drafts, instead of converting to plain decode.

Loop-level tests drive _owned_decode_rounds_batch with an echo target (next
token = input + 1) and a drafter that drafts the echo chain, so a speculating
row emits a full block per round and a plain row one token, and every row's
stream is checkable against its chain.
"""

from types import SimpleNamespace

import numpy as np
import mlx.core as mx
import pytest

import gmlx.spec.speculative as spec
from gmlx.spec.speculative import (
    _mixed_policy,
    _mixed_walk,
    _owned_decode_rounds_batch,
    _width_cap_logged,
    annotate_sampling_params,
    owned_server_rounds_batch,
)

from test_mtp_width_cap import _FakeCache, _StrictDrafter

VOCAB = 64


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("GMLX_MTP_WIDTH_CAP", "MLX_VLM_GGUF_SPEC_WIDTH_CAP",
                "GMLX_MTP_MIXED", "GMLX_MTP_RESUME"):
        monkeypatch.delenv(var, raising=False)
    spec._width_cap_memo = ("", None)
    _width_cap_logged.clear()
    yield
    spec._width_cap_memo = ("", None)
    _width_cap_logged.clear()


class _EchoLM:
    """Echo target serving the plain-forward verify branch. Records each
    verify's input rows and each rollback's per-row accepts."""

    def __init__(self, step=1):
        self.step = step
        self.inputs = []
        self.rollbacks = []
        self._rope_deltas = None

    def __call__(self, x, cache=None, return_hidden=False,
                 return_shared_kv=False, **kw):
        B, S = x.shape
        self.inputs.append(x.tolist())
        nxt = (x + self.step) % VOCAB
        logits = (mx.arange(VOCAB)[None, None, :] == nxt[:, :, None])
        logits = logits.astype(mx.float32) * 10.0
        # hidden carries the input token so contexts can be checked
        hidden = mx.broadcast_to(x[:, :, None].astype(mx.float32), (B, S, 8))
        return SimpleNamespace(logits=logits, hidden_states=[hidden],
                               shared_kv_states={}, gdn_states=None)

    def rollback_speculative_cache(self, caches, gdn_states, accepted,
                                   block_size):
        self.rollbacks.append(
            [int(accepted)] if isinstance(accepted, int) else list(accepted))


class _RowDrafter:
    """B=1-only drafter that owns its state and drafts the echo chain.
    ``contexts`` turns the parked row-context methods on."""

    uses_shared_kv = False
    supports_greedy_draft_argmax = True

    def __init__(self, cap=1, limit=1, block_size=4, contexts=True,
                 step=1, batch_accept=False):
        self.config = SimpleNamespace(block_size=block_size)
        self.accept_lens = []
        self.draft_lens = []
        self.mtp_width_cap = cap
        self.mtp_width_limit = limit
        self.step = step
        self.log = []
        self.live = []
        if not contexts:
            for name in ("row_context_open", "row_context_ingest",
                         "row_context_adopt", "row_context_state"):
                setattr(self, name, None)
        if batch_accept:
            self.accept_verified_tokens_batch = self._accept_batch

    def reset(self, model, left_padding=None):
        if left_padding is not None:
            raise NotImplementedError("B=1 only")
        self.log.append(("reset",))
        self.live = []
        return []

    def set_shared_kv(self, *a, **kw):
        raise AssertionError("own-state drafter must not see shared KV")

    def prefill_from_target_hidden(self, toks, hidden, bonus, sampler, dtype,
                                   **kw):
        self.log.append(("prefill", int(hidden.shape[1])))
        self.live = [float(v) for v in hidden[0, :, 0].tolist()]

    def draft_block(self, b, hidden, cache, n, sampler, dtype, **kw):
        if not isinstance(b, mx.array):
            b = mx.array([b])
        assert int(b.shape[0]) == 1, "B=1-only drafter drafted a batch"
        self.log.append(("draft", int(b.reshape(-1)[0].item())))
        chain = (b[:, None] + self.step * mx.arange(1, n)[None, :]) % VOCAB
        return chain.astype(dtype)

    def accept_verified_tokens(self, vh, drafts, accepted, new, sampler,
                               dtype, **kw):
        self.log.append(("accept", int(accepted)))
        self.live += [float(v) for v in vh[0, :accepted + 1, 0].tolist()]

    def _accept_batch(self, vh, drafts, accepted, new, sampler, dtype, **kw):
        self.log.append(("accept_batch", list(accepted)))

    def row_context_open(self, toks, hidden, bonus, kv=None):
        self.log.append(("open", int(hidden.shape[1])))
        return {"rows": [float(v) for v in hidden[0, :, 0].tolist()]}

    def row_context_ingest(self, ctx, vh, new_tokens, accepted):
        ctx["rows"] += [float(v) for v in vh[0, :accepted + 1, 0].tolist()]

    def row_context_state(self, ctx):
        return []

    def row_context_adopt(self, ctx, sampler, token_dtype=mx.int32,
                          greedy=False, stash=None):
        self.log.append(("adopt", len(ctx["rows"])))
        self.live = list(ctx["rows"])

    def names(self):
        return [e[0] for e in self.log]


def _chain(start, n, step=1):
    return [(start + step * i) % VOCAB for i in range(1, n + 1)]


def _streams(out, rows):
    return {r: [t[r] for t, _ in out if r < len(t) and t[r] is not None]
            for r in range(rows)}


def _drive(drafter, *, B, max_tokens, lm=None, model=None, b=None,
           hidden=True, stop_check=None, sampler=None, rounds=None,
           handoff=None):
    model = model if model is not None else SimpleNamespace()
    lm = lm if lm is not None else _EchoLM()
    b = b if b is not None else [10 * (i + 1) for i in range(B)]
    gen = _owned_decode_rounds_batch(
        model, drafter, lm, [_FakeCache(width=B)],
        hidden=(mx.broadcast_to(mx.arange(5, dtype=mx.float32)[None, :, None],
                                (B, 5, 8)) if hidden else None),
        b=list(b), shared_kv=None,
        seed_tokens=mx.zeros((B, 5), dtype=mx.int32) if hidden else None,
        emitted=[1] * B, max_tokens=max_tokens, sampler=sampler,
        draft_block_size=None, stop_check=stop_check, handoff=handoff)
    out = []
    for toks, meta in gen:
        out.append((toks, meta))
        if rounds is not None and len(out) >= rounds:
            gen.close()
            break
    return out, lm, model


# -- policy ---------------------------------------------------------------


def test_policy_needs_cap_owned_state_and_ragged_rollback(monkeypatch):
    lm = _EchoLM()
    d = _RowDrafter()
    assert _mixed_policy(d, lm, 1)
    assert not _mixed_policy(d, lm, 0)               # uncapped never trips
    shared = _RowDrafter()
    shared.uses_shared_kv = True
    assert not _mixed_policy(shared, lm, 1)
    lm_uniform = _EchoLM()
    lm_uniform.requires_uniform_batch_acceptance = True
    assert not _mixed_policy(d, lm_uniform, 1)
    assert not _mixed_policy(d, SimpleNamespace(), 1)  # no rollback hook
    monkeypatch.setenv("GMLX_MTP_MIXED", "0")
    assert not _mixed_policy(d, lm, 1)


def test_policy_auto_leaves_moe_targets_on_the_trip(monkeypatch):
    lm = _EchoLM()
    d = _RowDrafter()
    d.mtp_target_moe = True
    assert not _mixed_policy(d, lm, 1)
    monkeypatch.setenv("GMLX_MTP_MIXED", "1")
    assert _mixed_policy(d, lm, 1)


def test_stamp_records_moe_target():
    from gmlx.spec.mtp_load import _stamp_mtp_width_cap
    from test_mtp_width_cap import DENSE_PROJ, MOE_PROJ_3D, _fake_model

    quiet = dict(log=lambda *a, **k: None)
    d = _stamp_mtp_width_cap(SimpleNamespace(), "dflash2",
                             target=_fake_model(DENSE_PROJ), hard_limit=1,
                             **quiet)
    assert d.mtp_target_moe is False
    d = _stamp_mtp_width_cap(SimpleNamespace(), "dflash2",
                             target=_fake_model(MOE_PROJ_3D), hard_limit=1,
                             **quiet)
    assert d.mtp_target_moe is True


# -- formation past the cap -----------------------------------------------


def test_formation_over_cap_speculates_on_the_oldest_row():
    d = _RowDrafter(block_size=4)
    out, lm, _ = _drive(d, B=3, max_tokens=13)
    streams = _streams(out, 3)
    # every row streams its own echo chain to its budget (12 after the
    # first token), the speculating row a full block of 4 a round
    for r, start in enumerate([10, 20, 30]):
        assert streams[r] == _chain(start, 12)
    # row 0 drafts until it finishes, then row 1, then row 2 is promoted
    names = d.names()
    first = names.index("adopt")
    second = names.index("adopt", first + 1)
    assert [e[1] for e in d.log[:first] if e[0] == "draft"] == [10, 14, 18]
    assert all(20 < e[1] < 33
               for e in d.log[first:second] if e[0] == "draft")
    assert all(e[1] > 33 for e in d.log[second:] if e[0] == "draft")
    # one verify per round over every row, each a full block
    assert all(len(rows) == 3 and len(rows[0]) == 4
               for rows in lm.inputs[:3])
    # plain rows pad their block with their bonus; the pads never match
    # an echo target, so their rollbacks are per row
    assert lm.inputs[0][1] == [20, 20, 20, 20]
    assert lm.rollbacks[0] == [3, 0, 0]
    # drafter seeded the speculating row, parked contexts for the others
    assert d.names()[:4] == ["reset", "prefill", "open", "open"]


def test_old_trip_is_reachable(monkeypatch):
    """GMLX_MTP_MIXED=0 restores the plain-decode trip: the strict drafter
    raises on any drafting work."""
    monkeypatch.setenv("GMLX_MTP_MIXED", "0")
    d = _StrictDrafter(cap=1)
    d.uses_shared_kv = False
    lm = _EchoLM()
    out, _, _ = _drive(d, B=2, max_tokens=4, lm=lm)
    assert d.forward_calls == []
    assert all(meta["round_len"] == 1 for _, meta in out)


def test_b1_only_drafter_accepts_through_its_scalar_hook():
    """A DFlash-style drafter has no batched accept: a mixed batch drives
    its scalar accept, so its context grows by each round's commits."""
    d = _RowDrafter(block_size=4)
    _drive(d, B=2, max_tokens=7)
    accepts = [e for e in d.log if e[0] == "accept"]
    assert accepts[0] == ("accept", 3) and len(accepts) >= 2


def test_padding_drafts_are_accepted_only_where_the_target_agrees():
    """A target that repeats its input accepts every pad, so the plain row
    advances a full block too, still on its own exact stream."""
    lm = _EchoLM(step=0)
    d = _RowDrafter(block_size=4, step=0)
    out, _, _ = _drive(d, B=2, max_tokens=9, lm=lm)
    streams = _streams(out, 2)
    assert streams[0] == [10] * 8
    assert streams[1] == [20] * 8
    assert lm.rollbacks == []


# -- admission ------------------------------------------------------------


def _queue(model, rows=1, first=50, max_tokens=None, hidden_len=3):
    entry = {
        "uids": [f"u{i}" for i in range(rows)],
        "prompt_cache": [_FakeCache(width=rows, offset=9)],
        "hidden": mx.full((rows, hidden_len, 8), 7.0),
        "prompt_tokens": mx.zeros((rows, hidden_len), dtype=mx.int32),
        "first_tokens": mx.array([first + i for i in range(rows)],
                                 dtype=mx.int32),
        "first_tokens_list": [first + i for i in range(rows)],
        "shared_kv_states": None,
    }
    if max_tokens is not None:
        entry["max_tokens"] = [max_tokens] * rows
    model._generator_injections = [entry]


def test_admission_past_the_cap_keeps_the_incumbent_speculating():
    d = _RowDrafter(block_size=4)
    model = SimpleNamespace()
    _queue(model, max_tokens=10)
    out, lm, _ = _drive(d, B=1, max_tokens=13, model=model)
    streams = _streams(out, 2)
    assert streams[0] == _chain(10, 12)
    assert streams[1] == _chain(50, 9)
    # the injected row parked a context from its 3-row prompt capture
    assert ("open", 3) in d.log
    # the incumbent drafted every round; nothing re-seeded it
    assert d.names().count("prefill") == 1


def test_promotion_adopts_the_parked_context():
    """When the speculating row finishes, the oldest plain row takes over
    the drafter with its parked context: prompt capture plus every
    committed position it decoded plain."""
    d = _RowDrafter(block_size=4)
    out, lm, _ = _drive(d, B=2, max_tokens=20,
                        stop_check=lambda r, t: r == 0 and t >= 16)
    streams = _streams(out, 2)
    assert streams[0] == _chain(10, 6)
    assert streams[1] == _chain(20, 19)
    adopt = [e for e in d.log if e[0] == "adopt"]
    assert len(adopt) == 1
    # 5 prompt rows + one committed position per plain round (2 rounds)
    assert adopt[0][1] == 5 + 2
    # after the promotion the row drafts from its own bonus
    later = [e[1] for e in d.log[d.log.index(adopt[0]):] if e[0] == "draft"]
    assert later and all(20 < t < 40 for t in later)
    assert d.names().count("prefill") == 1


def test_promotion_without_contexts_cold_starts_from_the_capture():
    d = _RowDrafter(block_size=4, contexts=False)
    out, _, _ = _drive(d, B=2, max_tokens=20,
                       stop_check=lambda r, t: r == 0 and t >= 16)
    streams = _streams(out, 2)
    assert streams[1] == _chain(20, 19)
    prefills = [e for e in d.log if e[0] == "prefill"]
    # formation seed, then the cold start from the plain row's one
    # committed position of the round
    assert prefills == [("prefill", 5), ("prefill", 1)]


def test_all_finished_adoption_seeds_the_newcomer():
    d = _RowDrafter(block_size=4)
    model = SimpleNamespace()
    gen = _owned_decode_rounds_batch(
        model, d, _EchoLM(), [_FakeCache(width=2)],
        hidden=mx.zeros((2, 5, 8)), b=[10, 20], shared_kv=None,
        seed_tokens=mx.zeros((2, 5), dtype=mx.int32), emitted=[1, 1],
        max_tokens=2, sampler=None, draft_block_size=None)
    next(gen)
    _queue(model, first=40, max_tokens=8)
    rest = [t for t, _ in gen]
    stream = [t[2] for t in rest if len(t) > 2 and t[2] is not None]
    assert stream == _chain(40, 7)
    assert d.names().count("reset") == 2


# -- preempt handoff ------------------------------------------------------


def test_handoff_skips_the_reseed():
    """A rebuilt row whose scalar round committed into the drafter starts
    drafting at once: no reset, no capture round, no prefill."""
    d = _RowDrafter(block_size=4)
    out, lm, _ = _drive(d, B=1, max_tokens=7, hidden=False,
                        handoff={"drafter": d, "hidden": mx.zeros((1, 1, 8))})
    assert "reset" not in d.names() and "prefill" not in d.names()
    assert d.log[0] == ("draft", 10)
    assert _streams(out, 1)[0] == _chain(10, 6)


def test_handoff_for_another_drafter_is_ignored():
    d = _RowDrafter(block_size=4)
    other = _RowDrafter(block_size=4)
    _drive(d, B=1, max_tokens=4, hidden=False,
           handoff={"drafter": other, "hidden": mx.zeros((1, 1, 8))})
    # armless entry: capture round, then a cold seed
    assert d.names()[:2] == ["reset", "reset"]
    assert ("prefill", 1) in d.log


def test_owned_server_rounds_batch_pops_the_handoff():
    d = _RowDrafter(block_size=4)
    lm = _EchoLM()
    model = SimpleNamespace(language_model=lm)
    model._kq_preempt_handoff = {"drafter": d, "hidden": mx.zeros((1, 1, 8))}
    gen = owned_server_rounds_batch(
        model, d, [_FakeCache(width=1)], None,
        first_bonus=mx.array([10], dtype=mx.int32), max_tokens=7,
        sampler=None, shared_kv_states=None, prompt_tokens=None,
        greedy_sampling=True)
    out = [t for t, _ in gen]
    assert not hasattr(model, "_kq_preempt_handoff")
    assert [t[0] for t in out] == _chain(10, 6)
    assert "reset" not in d.names()


# -- the walk -------------------------------------------------------------


def _sampler():
    def s(logprobs):
        return mx.random.categorical(logprobs, axis=-1)
    annotate_sampling_params(s, temp=1.0, top_p=1.0, top_k=0, min_p=0.0)
    return s


def _verify_with(logits):
    lm = SimpleNamespace(
        speculative_logits_from_hidden=lambda hidden: mx.array(logits))
    verify = SimpleNamespace(target_tokens=None, hidden=mx.zeros((1, 1, 1)),
                             logits=None)
    return lm, verify


def test_mixed_walk_stochastic_row_takes_the_pq_rule():
    """The speculating row accepts by min(1, p/q): q equal to p accepts
    every draft whatever was drawn, while the plain row beside it is held
    to exact match and rejects a pad the target rules out."""
    rng = np.random.default_rng(1)
    logits = rng.normal(0, 2, size=(2, 3, 16)).astype(np.float32)
    logits[1, :, 9] = 40.0              # plain row's target is token 9
    p = np.exp(logits[0] - logits[0].max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    lm, verify = _verify_with(logits)
    drafts = mx.array([[3, 4], [2, 2]])
    q_rows = [mx.array(p[0]), mx.array(p[1])]
    for _ in range(8):
        acc, new = _mixed_walk(lm, verify, drafts, _sampler(), [9, 9],
                               stoch_row=0, q_rows=q_rows)
        assert acc[0] == 2 and new[0][:2] == [3, 4] and len(new[0]) == 3
        assert acc[1] == 0 and new[1] == [9]


def test_mixed_walk_residual_excludes_a_rejected_point_mass():
    """A copied or drafted token proposed with q=1 that the target gives
    p=0 is rejected, and the residual (p - q)+ never re-emits it."""
    logits = np.zeros((1, 2, 16), dtype=np.float32)
    logits[0, 0, 5] = 30.0
    lm, verify = _verify_with(logits)
    q = np.zeros(16, dtype=np.float32)
    q[7] = 1.0
    acc, new = _mixed_walk(lm, verify, mx.array([[7]]), _sampler(), [4],
                           stoch_row=0, q_rows=[mx.array(q)])
    assert acc == [0] and new == [[5]]


def test_mixed_walk_misaligned_stash_falls_back_to_exact_match():
    logits = np.zeros((1, 3, 16), dtype=np.float32)
    logits[0, :, 6] = 30.0
    lm, verify = _verify_with(logits)
    acc, new = _mixed_walk(lm, verify, mx.array([[6, 6]]), _sampler(), [9],
                           stoch_row=0, q_rows=[mx.zeros(16)])
    assert acc == [2] and new == [[6, 6, 6]]


# -- parked contexts on the real drafters ---------------------------------


def _close(a, b, tol=5e-3):
    a, b = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    assert a.shape == b.shape and np.allclose(a, b, atol=tol), \
        np.abs(a - b).max()


def test_dflash_parked_ring_matches_a_live_row():
    """A parked DFlash ring fed the prompt capture and two plain rounds
    drafts what the live ring fed the same positions drafts."""
    from test_dflash2_drafter import CAPTURE, _drafter, _target, _tcfg

    lm = _target()
    drafter = _drafter(lm)
    width = _tcfg().hidden_size * (1 + len(CAPTURE))
    mx.random.seed(3)
    prompt = mx.random.normal((1, 6, width))
    rounds = [mx.random.normal((1, 4, width)) for _ in range(2)]
    block = mx.array([[3, 7, 7, 7]])

    drafter.prefill_from_target_hidden(None, prompt, 5, None)
    for vh in rounds:
        drafter.accept_verified_tokens(vh, None, 0, [9], None)
    want = drafter._draft_hidden(block)
    mx.eval(want)

    drafter.reset(lm)
    ctx = drafter.row_context_open(None, prompt, 5)
    for vh in rounds:
        drafter.row_context_ingest(ctx, vh, [9], 0)
    mx.eval(drafter.row_context_state(ctx))
    drafter.reset(lm)
    drafter.row_context_adopt(ctx, None)
    _close(want, drafter._draft_hidden(block))


def _mtp_head():
    from test_mtp_round_kv import _head
    return _head()


def test_qwen_mtp_parked_head_matches_a_live_row():
    """A parked native MTP head, teacher-forced from the prompt capture and
    two plain rounds, adopts with the KV and seed the live head reaches
    over the same committed positions."""
    head, text = _mtp_head()
    mx.random.seed(4)
    toks = mx.array([[11, 12, 13, 14, 15]])
    prompt = mx.random.normal((1, 5, text.hidden_size))
    rounds = [(mx.random.normal((1, 3, text.hidden_size)), [21, 22, 23]),
              (mx.random.normal((1, 3, text.hidden_size)), [31])]

    # the live row of a mixed batch runs on the batch cache class
    head._cache = head.make_cache(left_padding=[0])
    head.prefill_from_target_hidden(toks, prompt, 16, None, greedy=True)
    for vh, new in rounds:
        a = len(new) - 1
        head.accept_verified_tokens(vh, mx.array([new[:a]]), a, new, None,
                                    greedy=True)
    want_kv = [s for c in head._cache for s in c.state]
    want_seed = (head._seed_token, head._seed_hidden)
    mx.eval(want_kv, want_seed)

    ctx = head.row_context_open(toks, prompt, 16)
    for vh, new in rounds:
        head.row_context_ingest(ctx, vh, new, len(new) - 1)
    head._cache = head.make_cache(left_padding=[0])
    head.row_context_adopt(ctx, None, greedy=True)
    got_kv = [s for c in head._cache for s in c.state]
    assert len(got_kv) == len(want_kv)
    for w, g in zip(want_kv, got_kv):
        _close(w, g)
    assert head._seed_token.tolist() == want_seed[0].tolist()
    _close(want_seed[1], head._seed_hidden)


def _scalar_preempt(d, model=None):
    d.set_shared_kv = lambda *a, **k: None    # the scalar entry binds it
    model = model if model is not None else SimpleNamespace()
    gen = spec._owned_decode_rounds(
        model, d, _EchoLM(), [_FakeCache(width=1)],
        hidden=mx.zeros((1, 4, 8)), b=5, shared_kv=None,
        seed_tokens=mx.zeros((1, 4), dtype=mx.int32), emitted=1,
        max_tokens=20, sampler=None, draft_block_size=None)
    assert next(gen) == 6              # round is [6, 7, 8, 9]; one delivered
    captured, slot = [], {}
    model._kq_preempt_capture = captured
    model._kq_preempt_handoff = slot
    gen.close()
    assert captured == [7, 8, 9]
    return slot


def test_scalar_preempt_hands_its_drafter_to_a_mixed_batch():
    """A preempt close commits the closed round into the drafter, as the
    loop would after delivery, and hands it over with the next hidden."""
    d = _RowDrafter(block_size=4)
    slot = _scalar_preempt(d)
    assert slot["drafter"] is d
    assert tuple(slot["hidden"].shape) == (1, 1, 8)
    assert d.log[-1] == ("accept", 3)


def test_scalar_preempt_keeps_the_old_contract_without_a_mixed_batch(
        monkeypatch):
    shared = _RowDrafter(block_size=4)
    shared.uses_shared_kv = True
    assert _scalar_preempt(shared) == {}
    monkeypatch.setenv("GMLX_MTP_MIXED", "0")
    d = _RowDrafter(block_size=4)
    assert _scalar_preempt(d) == {}
    assert ("accept", 3) not in d.log
