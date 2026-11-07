"""Logprobs on the owned speculative rounds, from the target's verify
logits, and each row's own sampler in every round kind.

A toy target scores each next token from its input token alone, with
non-integer bf16 logits, so a teacher-forced reference is the f32
log_softmax of the row for the token before each emitted one. It serves
the hidden-state verify hooks, the fused argmax hook and the plain
forward, and records which ran.
"""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import gmlx.spec.speculative as spec
from gmlx.gen.logprobs import RoundLogprobs
from gmlx.serve.row_sampling import GREEDY_KEY, RowSettings
from gmlx.spec.speculative import _width_cap_logged

from test_mtp_preempt_resume import _ArmableDrafter
from test_mtp_width_cap import _FakeCache

V = 64
K = 5
_rng = np.random.default_rng(7)
TABLE = mx.array(_rng.standard_normal((V, V)).astype(np.float32) * 2.5).astype(
    mx.bfloat16)
_REF = np.array(TABLE.astype(mx.float32))
_REF = _REF - np.log(np.exp(_REF).sum(axis=-1, keepdims=True))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("GMLX_MTP_WIDTH_CAP", "MLX_VLM_GGUF_SPEC_WIDTH_CAP",
                "GMLX_MTP_MIXED", "GMLX_MTP_RESUME", "GMLX_MTP_STOCH"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(spec, "_ROUND_GATE", 0.0)
    monkeypatch.setattr(spec, "_ROUND_GATE_AUTO", False)
    spec._width_cap_memo = ("", None)
    _width_cap_logged.clear()
    yield
    spec._width_cap_memo = ("", None)
    _width_cap_logged.clear()


def _greedy_next(tok):
    return int(np.argmax(_REF[int(tok)]))


def _shift(lp):
    """A deterministic 'sampler' that never takes the argmax."""
    return ((mx.argmax(lp, axis=-1) + 1) % V).astype(mx.int32)


class _TableLM:
    """Next-token logits are TABLE[input token]."""

    def __init__(self):
        self._rope_deltas = None
        self.hooks = []
        self.plain = 0

    @staticmethod
    def _hidden(x):
        B, S = x.shape
        return mx.broadcast_to(x[:, :, None].astype(mx.float32), (B, S, 8))

    @staticmethod
    def _kv(B, S):
        return {"full": (mx.zeros((B, 2, S, 4)), mx.zeros((B, 2, S, 4)))}

    def __call__(self, x, cache=None, return_hidden=False,
                 return_shared_kv=False, **kw):
        logits = TABLE[x]
        if return_hidden:
            B, S = x.shape
            return SimpleNamespace(logits=logits, hidden_states=[self._hidden(x)],
                                   shared_kv_states=self._kv(B, S), gdn_states=None)
        self.plain += 1
        return SimpleNamespace(logits=logits)

    def speculative_verify_hidden(self, x, cache, **kw):
        self.hooks.append("verify_hidden")
        B, S = x.shape
        return self._hidden(x), self._kv(B, S)

    def speculative_logits_from_hidden(self, h):
        self.hooks.append("logits")
        return TABLE[h[..., 0].astype(mx.int32)]

    def speculative_argmax_from_hidden(self, h):
        self.hooks.append("argmax")
        return mx.argmax(TABLE[h[..., 0].astype(mx.int32)], axis=-1)


class _RaggedLM(_TableLM):
    ragged_spec_rollback = True

    def rollback_speculative_cache(self, caches, gdn_states, accepted, bs):
        pass


class _ChainDrafter(_ArmableDrafter):
    """Drafts the greedy next token, then a token the target refuses."""

    uses_shared_kv = False

    def draft_block(self, b, hidden, kv, n, sampler, dtype, **kw):
        rows = mx.array(b).reshape(-1).tolist()
        out = []
        for t in rows:
            d1 = _greedy_next(t)
            out.append([d1, (_greedy_next(d1) + 7) % V][: n - 1])
        self.draft_calls.append(len(rows))
        return mx.array(out, dtype=dtype)


class _RowChainDrafter(_ChainDrafter):
    """A B=1-only drafter with parked row contexts, for mixed batches."""

    def row_context_open(self, toks, hidden, bonus, kv=None):
        return {}

    def row_context_ingest(self, ctx, vh, new_tokens, accepted):
        pass

    def row_context_state(self, ctx):
        return []

    def row_context_adopt(self, ctx, sampler, token_dtype=mx.int32,
                          greedy=False, stash=None):
        pass


def _check(first, toks, entries, k=K):
    """Each entry against the teacher-forced f32 reference."""
    assert len(entries) == len(toks)
    prev = [first] + list(toks[:-1])
    for p, t, (lp, ids, vals) in zip(prev, toks, entries):
        row = _REF[int(p)]
        assert lp == pytest.approx(float(row[int(t)]), abs=1e-5)
        if k:
            assert len(ids) == k
            np.testing.assert_allclose(vals, np.sort(row)[::-1][:k], atol=1e-5)
            np.testing.assert_allclose(vals, row[ids], atol=1e-5)


# -- the B=1 scalar loop ----------------------------------------------------


def _scalar(*, sampler=None, max_tokens=8, b=3, row_procs=None, hook=None,
            lm=None, drafter=None, model=None):
    lm = lm or _TableLM()
    round_lp: list = []
    gen = spec._owned_decode_rounds(
        model or SimpleNamespace(), drafter or _ChainDrafter(cap=0, block_size=3),
        lm, [_FakeCache(width=1)], hidden=mx.zeros((1, 4, 8)), b=b,
        shared_kv=None, seed_tokens=None, emitted=1, max_tokens=max_tokens,
        sampler=sampler, draft_block_size=None, round_pos=(pos := [0, 1]),
        row_procs=row_procs, thinking_hook=hook, round_lp=round_lp, lp_k=K)
    toks, entries = [], []
    for tok in gen:
        toks.append(int(tok))
        entries.append(round_lp[pos[0]])
    return toks, entries, lm


def test_scalar_rounds_take_entries_from_the_verify_logits():
    toks, entries, lm = _scalar()
    _check(3, toks, entries)
    # Greedy: the target's own chain.
    assert toks == [_greedy_next(t) for t in [3] + toks[:-1]]
    # The fused argmax gave way to one head pass shared with the walk.
    assert "argmax" not in lm.hooks and "logits" in lm.hooks


def test_scalar_rounds_without_logprobs_keep_the_fused_argmax():
    lm = _TableLM()
    gen = spec._owned_decode_rounds(
        SimpleNamespace(), _ChainDrafter(cap=0, block_size=3), lm,
        [_FakeCache(width=1)], hidden=mx.zeros((1, 4, 8)), b=3, shared_kv=None,
        seed_tokens=None, emitted=1, max_tokens=6, sampler=None,
        draft_block_size=None)
    list(gen)
    assert "argmax" in lm.hooks and "logits" not in lm.hooks


def test_scalar_entries_ignore_the_row_processors():
    from gmlx._vendor.mlx_vlm.sample_utils import make_logits_processors

    from gmlx.spec.row_procs import SpecRowProcessors

    rows = SpecRowProcessors.from_processors(
        make_logits_processors({11: 30.0}), [3])
    rows.start(3)
    toks, entries, _ = _scalar(row_procs=rows)
    assert set(toks) == {11}             # the bias steers every token
    _check(3, toks, entries)             # the logprobs stay raw


def test_scalar_sampled_entries_are_the_targets():
    toks, entries, _ = _scalar(sampler=_shift)
    assert toks[0] != _greedy_next(3)
    _check(3, toks, entries)


def test_a_gated_round_never_evaluates_the_dropped_verify(monkeypatch):
    built = []
    orig = RoundLogprobs.__init__

    def spy(self, logits, k, drafts=None):
        built.append(tuple(logits.shape))
        orig(self, logits, k, drafts=drafts)

    monkeypatch.setattr(RoundLogprobs, "__init__", spy)
    monkeypatch.setattr(spec, "_ROUND_GATE", 1.0)
    d = _ChainDrafter(cap=0, block_size=3)
    d.round_estimate = mx.array(0.0)
    toks, entries, lm = _scalar(drafter=d)
    _check(3, toks, entries)
    assert built and all(shape == (1, 1, V) for shape in built)
    assert "logits" not in lm.hooks      # no head pass on a dropped verify


class _Hook:
    """A thinking hook that closes once, after the first round."""

    def __init__(self, ids):
        self.ids = list(ids)
        self.seen = 0

    def observe(self, tok):
        self.seen += 1

    def take_forced(self):
        ids, self.ids = (self.ids, []) if self.seen else ([], self.ids)
        return ids or None


def test_a_forced_close_reports_the_forced_ids_own_logprobs():
    toks, entries, _ = _scalar(hook=_Hook([20, 21, 22]), max_tokens=12)
    i = toks.index(20)
    assert toks[i:i + 3] == [20, 21, 22]
    _check(3, toks, entries)


def test_a_preempt_tail_carries_its_entries():
    model = SimpleNamespace(_kq_preempt_capture=[], _kq_preempt_capture_lp=[])
    round_lp: list = []
    gen = spec._owned_decode_rounds(
        model, _ChainDrafter(cap=0, block_size=4), _TableLM(),
        [_FakeCache(width=1)], hidden=mx.zeros((1, 4, 8)), b=3, shared_kv=None,
        seed_tokens=None, emitted=1, max_tokens=12, sampler=None,
        draft_block_size=None, round_lp=round_lp, lp_k=K)
    first = int(next(gen))
    gen.close()
    tail = model._kq_preempt_capture
    assert tail and len(model._kq_preempt_capture_lp) == len(tail)
    _check(first, tail, model._kq_preempt_capture_lp)


# -- owned_server_rounds (B=1 serve) ------------------------------------------


def _server(row, *, sampler=None, greedy_sampling=True, uid=4, max_tokens=7):
    model = SimpleNamespace(language_model=_TableLM(), _kq_row_uids=[uid])
    gen = spec.owned_server_rounds(
        model, _ChainDrafter(cap=0, block_size=3), [_FakeCache(width=1)],
        mx.zeros((1, 4, 8)), first_bonus=3, max_tokens=max_tokens,
        sampler=sampler, shared_kv_states=None, prompt_tokens=None,
        greedy_sampling=greedy_sampling, row_settings={uid: row},
        default_row=RowSettings(), logprobs_cap=K)
    toks, entries = [], []
    for tok_list, meta in gen:
        toks.append(tok_list[0])
        entries.append(meta.get("logprobs", [None])[0])
    return toks, entries


def test_a_lone_row_decodes_with_its_own_settings():
    # The generator's first request sampled; this row is greedy.
    toks, entries = _server(RowSettings(logprobs=True, top_k=K),
                            sampler=_shift, greedy_sampling=False)
    assert toks == [_greedy_next(t) for t in [3] + toks[:-1]]
    _check(3, toks, entries)
    # And the other way around.
    toks, _ = _server(RowSettings(sampler=_shift, key=("shift",)))
    assert toks[0] == (_greedy_next(3) + 1) % V


def test_a_seeded_lone_row_replays():
    from gmlx.serve.patches.sampling import _FastPositionedSampler
    from gmlx.serve.row_sampling import register_seed

    def run(seed):
        s = _FastPositionedSampler(temperature=1.5)
        register_seed(s, 4, seed)
        return _server(RowSettings(sampler=s, seed=seed, key=("t",)),
                       max_tokens=10)[0]

    assert run(1234) == run(1234)
    assert run(1234) != run(99)


# -- the batch loop -----------------------------------------------------------


def _settings(rows):
    return {uid: rs for uid, rs in rows.items()}


def _batch(rows, *, B=None, b=None, lm=None, drafter=None, hidden=True,
           max_tokens=8, model=None, hook=None, inject=None, uids=None,
           collect=None):
    uids = list(uids or rows)
    B = B or len(uids)
    b = b or [3 + 5 * i for i in range(B)]
    model = model or SimpleNamespace()
    gen = spec._owned_decode_rounds_batch(
        model, drafter or _ChainDrafter(cap=0, block_size=3), lm or _TableLM(),
        [_FakeCache(width=B)],
        hidden=mx.zeros((B, 4, 8)) if hidden else None, b=list(b),
        shared_kv=None, seed_tokens=None, emitted=[1] * B,
        max_tokens=max_tokens, sampler=None, draft_block_size=None,
        thinking_hook=hook, row_settings=_settings(rows),
        default_row=RowSettings(), lp_k=K, uids=uids)
    out = {r: ([], []) for r in range(B + 4)}
    steps = 0
    for toks, meta in gen:
        steps += 1
        if inject is not None and steps == 1:
            inject(model)
        for r, t in enumerate(toks):
            if t is not None:
                out[r][0].append(int(t))
                lps = meta.get("logprobs")
                out[r][1].append(lps[r] if lps is not None else None)
        if collect is not None:
            collect.append(meta)
    return out, b


_LP = RowSettings(logprobs=True, top_k=K)


def test_batch_rounds_give_entries_to_the_rows_that_ask():
    out, b = _batch({1: _LP, 2: RowSettings()})
    _check(b[0], *out[0])
    assert out[1][1] and all(e is None for e in out[1][1])


def test_a_batch_without_logprob_or_sampled_rows_runs_todays_verify(monkeypatch):
    calls, built = [], []
    orig = spec._mtp_verify_target

    def verify(*a, **kw):
        calls.append(kw.get("sample_target_tokens"))
        return orig(*a, **kw)

    monkeypatch.setattr(spec, "_mtp_verify_target", verify)
    monkeypatch.setattr(RoundLogprobs, "__init__",
                        lambda *a, **k: built.append(1))
    lm = _TableLM()
    out, _ = _batch({1: RowSettings(), 2: RowSettings()}, lm=lm)
    assert calls and all(calls)
    assert not built and "argmax" in lm.hooks
    assert all(e is None for e in out[0][1])


def test_batch_logprob_rows_skip_the_fused_argmax(monkeypatch):
    calls = []
    orig = spec._mtp_verify_target

    def verify(*a, **kw):
        calls.append(kw.get("sample_target_tokens"))
        return orig(*a, **kw)

    monkeypatch.setattr(spec, "_mtp_verify_target", verify)
    lm = _TableLM()
    out, b = _batch({1: _LP, 2: _LP}, lm=lm)
    assert calls and not any(calls)
    assert "argmax" not in lm.hooks
    _check(b[0], *out[0])
    _check(b[1], *out[1])


def test_a_mixed_greedy_and_sampled_batch_samples_each_row_as_itself():
    out, b = _batch({1: _LP, 2: RowSettings(sampler=_shift, key=("s",),
                                            logprobs=True, top_k=K)})
    greedy, sampled = out[0][0], out[1][0]
    assert greedy == [_greedy_next(t) for t in [b[0]] + greedy[:-1]]
    assert sampled == [(_greedy_next(t) + 1) % V for t in [b[1]] + sampled[:-1]]
    _check(b[0], *out[0])
    _check(b[1], *out[1])


def test_gated_batch_rounds_carry_entries(monkeypatch):
    lm = _TableLM()
    out, b = _batch({1: _LP, 2: RowSettings(sampler=_shift, key=("s",))},
                    lm=lm, drafter=_ChainDrafter(cap=1, block_size=3))
    assert lm.plain > 0 and not lm.hooks
    _check(b[0], *out[0])
    assert out[1][0][0] == (_greedy_next(b[1]) + 1) % V


def test_an_arm_capture_round_carries_entries():
    out, b = _batch({1: _LP, 2: _LP}, hidden=False)
    _check(b[0], *out[0])
    _check(b[1], *out[1])


def test_a_batch_forced_close_carries_entries():
    out, b = _batch({1: _LP}, hook=_Hook([20, 21, 22]), max_tokens=12)
    toks = out[0][0]
    assert 20 in toks
    _check(b[0], *out[0])


def test_mixed_rounds_carry_entries(monkeypatch):
    monkeypatch.setenv("GMLX_MTP_MIXED", "1")
    walks = []
    orig = spec._mixed_walk
    monkeypatch.setattr(spec, "_mixed_walk",
                        lambda *a, **k: walks.append(1) or orig(*a, **k))
    d = _RowChainDrafter(cap=1, block_size=3)
    d.mtp_width_limit = 1
    out, b = _batch({1: _LP, 2: _LP}, lm=_RaggedLM(), drafter=d)
    assert walks
    _check(b[0], *out[0])
    _check(b[1], *out[1])


def _injection(uid, first, hidden=True):
    def inject(model):
        model._generator_injections = [{
            "uids": [uid],
            "prompt_cache": [_FakeCache(width=1)],
            "hidden": mx.zeros((1, 1, 8)) if hidden else None,
            "shared_kv_states": None,
            "prompt_tokens": mx.array([[1]]),
            "first_tokens": mx.array([first]),
            "first_tokens_list": [first],
            "max_tokens": [8],
            "row_procs": None,
        }]
    return inject


def test_injected_rows_keep_their_own_samplers():
    # A sampled row joins a greedy batch.
    out, b = _batch({1: _LP, 9: RowSettings(sampler=_shift, key=("s",))},
                    B=1, uids=[1], inject=_injection(9, 40))
    joined = out[1][0]
    assert joined and joined[0] == (_greedy_next(40) + 1) % V
    _check(b[0], *out[0])
    # A greedy row joins a sampled batch.
    out, b = _batch({1: RowSettings(sampler=_shift, key=("s",)), 9: _LP},
                    B=1, uids=[1], inject=_injection(9, 40))
    joined = out[1][0]
    assert joined == [_greedy_next(t) for t in [40] + joined[:-1]]
    _check(40, *out[1])


def test_a_sampled_row_joining_a_gated_greedy_batch_is_sampled():
    # Cap 1: the injection trips the width gate, and the gated step must
    # sample the new row with its own sampler.
    out, _b = _batch({1: _LP, 9: RowSettings(sampler=_shift, key=("s",))},
                     B=1, uids=[1], inject=_injection(9, 40),
                     drafter=_ChainDrafter(cap=1, block_size=3))
    joined = out[1][0]
    assert joined and joined == [(_greedy_next(t) + 1) % V
                                 for t in [40] + joined[:-1]]


def test_the_row_sampler_default_is_greedy():
    assert RowSettings().key == GREEDY_KEY and RowSettings().sampler is None


# -- SpeculativeGenerationBatch -------------------------------------------------


def _entry(x):
    return (x, list(range(K)), [x - i for i in range(K)])


def _tail_rounds(calls, tail):
    """Fake rounds: yields one token per step with its entry in the meta;
    on close, fills the preempt capture with ``tail`` the way the scalar
    loop does."""

    def fake_rounds(model, draft_model, prompt_cache, hidden, **kw):
        calls.append(kw)
        width = int(kw["first_bonus"].shape[0])
        n = 0
        try:
            while True:
                inj = getattr(model, "_generator_injections", None)
                if inj:
                    width += sum(len(e["uids"]) for e in inj)
                    inj.clear()
                n += 1
                tok = 100 * len(calls) + n
                yield [tok] * width, {"logprobs": [_entry(-float(n))] * width}
        finally:
            cap = getattr(model, "_kq_preempt_capture", None)
            if cap is not None and len(calls) == 1:
                cap.extend(tail)
                model._kq_preempt_capture_lp.extend(_entry(-7.0) for _ in tail)

    return fake_rounds


def _spec_batch(ar, uid, model, row, first_lp):
    from test_mtp_preempt_resume import _make_batch

    sb = _make_batch(ar, uids=(uid,), model=model, max_tokens=20)
    ar._bind_rows(sb, ({uid: row}, RowSettings(), K))
    sb._first_lps = mx.array([first_lp])
    sb._first_top_idx = mx.array([list(range(K))], dtype=mx.int32)
    sb._first_top_lp = mx.array([[first_lp - i for i in range(K)]])
    return sb


def test_spec_batch_entries_for_first_tokens_rounds_tails_and_injections(
        monkeypatch):
    from gmlx._vendor.mlx_vlm.generate import ar
    from gmlx.spec.admission import install_continuous_batch_admission

    install_continuous_batch_admission()
    calls = []
    monkeypatch.setattr(ar, "run_speculative_server_rounds",
                        _tail_rounds(calls, [61, 62]))
    model = SimpleNamespace()
    host = _spec_batch(ar, 0, model, _LP, -0.5)
    (first,) = host.next()
    assert first.token_logprob == pytest.approx(-0.5)
    assert first.top_logprobs == [(i, pytest.approx(-0.5 - i)) for i in range(K)]
    (r,) = host.next()
    assert r.token_logprob == -1.0 and len(r.top_logprobs) == K
    assert calls[0]["row_settings"] == {0: _LP}
    assert calls[0]["logprobs_cap"] == K

    # A waiter joins: the scalar closes, its tail keeps its entries, and
    # the waiter's first token reports its own.
    host.extend(_spec_batch(ar, 7, model, _LP, -0.25))
    got = [(x.uid, x.token, x.token_logprob) for x in host.next()]
    assert got[:2] == [(0, 61, -7.0), (0, 62, -7.0)]
    assert (7, 12, -0.25) in got
    assert not hasattr(model, "_kq_preempt_capture_lp")
    assert not hasattr(model, "_kq_preempt_capture")


@pytest.mark.parametrize("kind,want", [
    ("mtp", None), (None, None), ("eagle", "eagle"), ("dflash", "dflash"),
])
def test_spec_kind_unsupported(kind, want):
    from gmlx.spec.engine import spec_kind_unsupported

    assert spec_kind_unsupported(SimpleNamespace(draft_kind=kind)) == want
    assert spec_kind_unsupported(SimpleNamespace()) is None
