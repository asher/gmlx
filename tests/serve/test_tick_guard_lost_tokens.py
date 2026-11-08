"""A contained memory error delivers what its tick computed.

A real BatchGenerator runs a tiny llama under the tick guard. The forward
runs, so each row's cache moves as in serving, but the logits pick "last
input + 1": greedy decode counts up, and a lost token shows as a skip.
Each case injects one error at a named call and compares every stream
with the uninjected run. The gap of a row is its cache offset less its
prompt and its delivered tokens; it is 0 in an uninjected run.
"""

from __future__ import annotations

import sys
import types

import pytest

import mlx.core as mx

import gmlx.eval_guard as eg
import gmlx.serve.tick_guard as tg
from gmlx._vendor.mlx_vlm.generate import ar
from gmlx.upstream.patching import root

MEM = "[metal::malloc] Attempting to allocate 1 bytes (test)"
OOM = ("[METAL] Command buffer execution failed: Insufficient Memory "
       "(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).")

A_PROMPT, B_PROMPT, C_PROMPT = [3, 9, 4, 7], [5, 6, 2], [40, 41]
FORCED = 300


class Shot:
    """One throw, at the first matching call once armed."""

    def __init__(self, text=MEM):
        self.text = text
        self.armed = False
        self.fired = 0

    def fire(self):
        if self.armed and not self.fired:
            self.fired += 1
            raise RuntimeError(self.text)


class _Mx:
    """ar's mx, with a throw at an eval called from the named function."""

    def __init__(self, real, shot, *where, name="async_eval"):
        self._real, self._shot, self._where, self._name = real, shot, where, name

    def __getattr__(self, n):
        fn = getattr(self._real, n)
        if n != self._name:
            return fn

        def call(*a, **k):
            if sys._getframe(1).f_code.co_name in self._where:
                self._shot.fire()
            return fn(*a, **k)
        return call


class _Forcing:
    """A criteria that forces one id after the row's third token."""

    def __init__(self):
        self.n = 0
        self.forced_token_id = None

    def __call__(self, tok):
        self.n += 1
        if self.n == 3:
            self.forced_token_id = FORCED
        return self.forced_token_id

    def pop_forced_token_id(self):
        out, self.forced_token_id = self.forced_token_id, None
        return out


@pytest.fixture
def engine(monkeypatch):
    """A guarded generator factory over the counting llama."""
    from gmlx._vendor.mlx_lm.models import llama
    from gmlx._vendor.mlx_vlm.utils import StoppingCriteria
    from gmlx.models.vlm_text_only import TextOnlyModel

    monkeypatch.setenv("GMLX_PREFILL_DECAY", "0")
    monkeypatch.setenv("GMLX_CB_PHASE", "0")
    monkeypatch.delenv("GMLX_TICK_GUARD", raising=False)
    monkeypatch.delenv("GMLX_OOM_INJECT", raising=False)
    monkeypatch.setattr(eg, "drain_for", lambda site: None)
    monkeypatch.setattr(ar.BatchGenerator, "_next",
                        root(ar.BatchGenerator._next))
    assert tg.install_tick_guard()
    mx.random.seed(0)
    model = TextOnlyModel(llama.Model(llama.ModelArgs(
        model_type="llama", hidden_size=32, num_hidden_layers=2,
        intermediate_size=64, num_attention_heads=4, num_key_value_heads=2,
        rms_norm_eps=1e-5, vocab_size=512)))
    lm = model.language_model
    forward = type(lm).__call__

    def counting(self, inputs, *a, inputs_embeds=None, **k):
        # The KV comes from the ids, so a replay holds what its ids say.
        out = forward(self, inputs, *a, **k)
        logits = out.logits if hasattr(out, "logits") else out
        V, L = logits.shape[-1], logits.shape[1]
        nxt = (inputs[:, -L:].astype(mx.int32) + 1) % V
        hot = (mx.arange(V)[None, None, :] == nxt[..., None])
        hot = hot.astype(logits.dtype) * 20.0
        if hasattr(out, "logits"):
            out.logits = hot
            return out
        return hot

    monkeypatch.setattr(type(lm), "__call__", counting)
    proc = types.SimpleNamespace(tokenizer=types.SimpleNamespace(
        stopping_criteria=StoppingCriteria([], None)))

    def make(**kw):
        kw.setdefault("prefill_step_size", None)
        return ar.BatchGenerator(lm, proc, logprobs_cap=3, **kw), lm

    return make


def _kw(lm, ids):
    return {"inputs_embeds": lm.input_embeds(mx.array([ids]))}


def _offsets(gen):
    gb = gen._generation_batch
    caches = getattr(gb, "prompt_cache", None)
    if not caches:
        return {}
    off = getattr(caches[0], "offset", None)
    if isinstance(off, mx.array):
        off = off.tolist()
    elif isinstance(off, int):
        off = [off] * len(gb.uids)
    else:
        return {}
    return dict(zip(gb.uids, off))


def run(make, *, plan=None, b_max=6, c_at=None, forcing=False,
        chunked=False, eval_every=0):
    """Drive two or three rows to the end. ``plan(tick, gen, out)`` arms
    an injection. Returns the streams, the finishes and the gaps."""
    from gmlx.serve.row_sampling import RowSettings

    gen, lm = make(prefill_step_size=2 if chunked else None)
    if eval_every:
        gen._cache_eval_interval = eval_every
    prompts = {}
    crit = [_Forcing()] if forcing else None
    (ua,) = gen.insert([A_PROMPT], 12, prompt_kwargs=[_kw(lm, A_PROMPT)],
                       thinking_budget_criteria=crit,
                       row_settings=[RowSettings(logprobs=True, top_k=3)])
    prompts[ua] = A_PROMPT
    out, finishes, gaps = {}, {}, {}
    ticks = 0
    late = []
    try:
        while gen.has_work and ticks < 80:
            ticks += 1
            if ticks == 4:
                (ub,) = gen.insert([B_PROMPT], b_max,
                                   prompt_kwargs=[_kw(lm, B_PROMPT)])
                prompts[ub] = B_PROMPT
            if (c_at is not None and not late
                    and len(out.get(1, [])) == c_at):
                (uc,) = gen.insert([C_PROMPT], 2,
                                   prompt_kwargs=[_kw(lm, C_PROMPT)])
                prompts[uc] = C_PROMPT
                late.append(uc)
            if plan is not None:
                plan(ticks, gen, out)
            _, resps = gen.next()
            for r in resps:
                assert r.uid not in finishes, f"uid {r.uid} emits after finish"
                if r.token is not None:
                    out.setdefault(r.uid, []).append(r)
                if r.finish_reason:
                    finishes[r.uid] = r.finish_reason
            st = tg._state(gen)
            if st.fail_streak or st.fault_streak:
                continue    # a failed tick can owe a token it computed
            for uid, off in _offsets(gen).items():
                n = len(out.get(uid, []))
                gaps.setdefault(uid, set()).add(off - len(prompts[uid]) - n)
    finally:
        gen.close()
    return out, finishes, gaps


def _tokens(out):
    return {uid: [r.token for r in rs] for uid, rs in out.items()}


def check(make, case, **kw):
    ref, ref_fin, _ = run(make, **{k: v for k, v in kw.items()
                                   if k != "plan"})
    got, fin, gaps = run(make, **kw)
    assert case.fired == 1
    assert _tokens(got) == _tokens(ref)
    assert fin == ref_fin and set(fin) == set(ref)
    for r in got[0]:
        assert r.token_logprob is not None and len(r.top_logprobs) == 3
    for uid, g in gaps.items():
        assert g == {0}, (uid, g)
    return got


def _arm_at(shot, tick):
    def plan(t, gen, out):
        if t == tick:
            shot.armed = True
    return plan


def test_e_generate_entry(engine, monkeypatch):
    """E: the prefill arm throws after the decode step of its tick."""
    shot = Shot()
    orig = ar.PromptProcessingBatch.generate

    def generate(self, *a, **k):
        shot.fire()
        return orig(self, *a, **k)

    monkeypatch.setattr(ar.PromptProcessingBatch, "generate", generate)
    check(engine, shot, plan=_arm_at(shot, 4))


def test_e_cache_eval(engine, monkeypatch):
    """E: the cache eval after the decode step throws."""
    shot = Shot()
    monkeypatch.setattr(ar, "mx", _Mx(ar.mx, shot, "_next", "_next_after_decode"))
    check(engine, shot, plan=_arm_at(shot, 6), eval_every=1)


def test_f_finish_on_a_failed_tick(engine, monkeypatch):
    """F: a row finishes on the tick whose prefill arm throws. Its finish
    was its only close."""
    shot = Shot()
    orig = ar.PromptProcessingBatch.generate

    def generate(self, *a, **k):
        shot.fire()
        return orig(self, *a, **k)

    def plan(t, gen, out):
        if len(out.get(1, [])) == 2:
            shot.armed = True

    monkeypatch.setattr(ar.PromptProcessingBatch, "generate", generate)
    got = check(engine, shot, plan=plan, b_max=3, c_at=2)
    assert got[1][-1].finish_reason == "length"


def test_d_filter(engine, monkeypatch):
    """D: the filter of a finishing row throws. The finished row leaves
    on the next tick instead of decoding past its stop."""
    shot = Shot()
    orig = ar.GenerationBatch.filter

    def filt(self, keep):
        shot.fire()
        return orig(self, keep)

    def plan(t, gen, out):
        if len(out.get(1, [])) == 2:
            shot.armed = True

    monkeypatch.setattr(ar.GenerationBatch, "filter", filt)
    check(engine, shot, plan=plan, b_max=3)


def test_d_forced_eval(engine, monkeypatch):
    """D: the eval of a forced id throws."""
    shot = Shot()
    monkeypatch.setattr(ar, "mx", _Mx(ar.mx, shot, "_force_next_tokens"))
    got = check(engine, shot, plan=_arm_at(shot, 1), forcing=True)
    assert FORCED in _tokens(got)[0]


def test_c_after_the_sample(engine, monkeypatch):
    """C: the decode step throws after its sample. The next step returns
    the token the failed one owed, without a forward."""
    shot = Shot()
    monkeypatch.setattr(ar, "mx", _Mx(ar.mx, shot, "_step"))
    check(engine, shot, plan=_arm_at(shot, 6))


def test_c_in_the_emit_eval(engine, monkeypatch):
    shot = Shot()
    monkeypatch.setattr(ar, "mx", _Mx(ar.mx, shot, "_step", "_emit", name="eval"))
    check(engine, shot, plan=_arm_at(shot, 6))


def test_a_forced_id_on_the_resume_tick_keeps_its_logprob(engine, monkeypatch):
    """The step that resumes feeds the owed token to the criteria, which
    forces an id. The id reports its own raw logprob, which needs the
    failed step's raw logits."""
    shot = Shot()
    monkeypatch.setattr(ar, "mx", _Mx(ar.mx, shot, "_step"))

    def plan(t, gen, out):
        if len(out.get(0, [])) == 2:
            shot.armed = True

    ref, _, _ = run(engine, forcing=True)
    got = check(engine, shot, plan=plan, forcing=True)
    want = [r for r in ref[0] if r.token == FORCED]
    have = [r for r in got[0] if r.token == FORCED]
    assert len(have) == len(want) == 1
    assert have[0].token_logprob == pytest.approx(want[0].token_logprob)
    assert have[0].token_logprob < -10
