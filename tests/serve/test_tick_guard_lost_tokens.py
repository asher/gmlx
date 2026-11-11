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

    def rewind(self, committed):
        """The state after ``committed``, as a requeue needs."""
        self.n = len(committed)
        self.forced_token_id = FORCED if self.n == 3 else None


@pytest.fixture
def engine(monkeypatch):
    """A guarded generator factory over the counting llama."""
    from gmlx._vendor.mlx_lm.models import llama
    from gmlx._vendor.mlx_vlm.utils import StoppingCriteria
    from gmlx.models.vlm_text_only import TextOnlyModel
    from gmlx.serve.replay_prompt import embed_text

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
        gen = ar.BatchGenerator(lm, proc, logprobs_cap=3, **kw)
        gen._kq_embed_text = lambda ids, rs: embed_text(model, ids, rs)
        return gen, lm

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


def run(make, *, plan=None, b_max=6, c_at=None, c_max=2, forcing=False,
        chunked=False, eval_every=0, processors=None):
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
                       logits_processors=[processors] if processors else None,
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
                (uc,) = gen.insert([C_PROMPT], c_max,
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


@pytest.fixture(params=[MEM, OOM], ids=["mem", "cb-oom"])
def shot(request):
    """The injection: an allocator refusal, and again a command buffer OOM,
    after which nothing the engine holds is valid (G)."""
    return Shot(request.param)


def _arm_at(shot, tick):
    def plan(t, gen, out):
        if t == tick:
            shot.armed = True
    return plan


def test_e_generate_entry(engine, monkeypatch, shot):
    """E: the prefill arm throws after the decode step of its tick."""
    orig = ar.PromptProcessingBatch.generate

    def generate(self, *a, **k):
        shot.fire()
        return orig(self, *a, **k)

    monkeypatch.setattr(ar.PromptProcessingBatch, "generate", generate)
    check(engine, shot, plan=_arm_at(shot, 4))


def test_e_cache_eval(engine, monkeypatch, shot):
    """E: the cache eval after the decode step throws."""
    monkeypatch.setattr(ar, "mx", _Mx(ar.mx, shot, "_next", "_next_after_decode"))
    check(engine, shot, plan=_arm_at(shot, 6), eval_every=1)


def test_f_finish_on_a_failed_tick(engine, monkeypatch, shot):
    """F: a row finishes on the tick whose prefill arm throws. Its finish
    was its only close."""
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


def test_d_filter(engine, monkeypatch, shot):
    """D: the filter of a finishing row throws. The finished row leaves
    on the next tick instead of decoding past its stop."""
    orig = ar.GenerationBatch.filter

    def filt(self, keep):
        shot.fire()
        return orig(self, keep)

    def plan(t, gen, out):
        if len(out.get(1, [])) == 2:
            shot.armed = True

    monkeypatch.setattr(ar.GenerationBatch, "filter", filt)
    check(engine, shot, plan=plan, b_max=3)


class _Refusals:
    """Throws at the cache evals of a filter and of what follows it.
    ``shot`` goes off at the eval of layer ``layer``'s kept rows inside a
    filter. Each site of ``later`` then gets one allocator refusal, in
    order: "next" is the eval a step makes first for the layers a failed
    filter left, "before" the evals a filter makes before it changes the
    batch."""

    def __init__(self, monkeypatch, shot, *later, layer=0):
        self.shot, self.later, self.layer = shot, list(later), layer
        self.seen = 0
        orig = ar._eval_cache

        def eval_cache(c):
            caller = sys._getframe(1).f_code.co_name
            if caller == "filter":
                self.at("filter")
            elif caller == "_kq_eval_filtered":
                outer = sys._getframe(2).f_code.co_name
                self.at("next" if outer == "next" else "before")
            return orig(c)

        monkeypatch.setattr(ar, "_eval_cache", eval_cache)
        monkeypatch.setattr(ar, "mx", _Mx(ar.mx, self, "_eval_pending_state",
                                          name="eval"))

    def fire(self):
        self.at("before")

    def at(self, site):
        if not self.shot.fired:
            if site == "filter" and self.shot.armed:
                self.seen += 1
                if self.seen == self.layer + 1:
                    self.shot.fire()
        elif self.later and self.later[0] == site:
            self.later.pop(0)
            raise RuntimeError(MEM)


def _arm_at_b_finish(shot):
    def plan(t, gen, out):
        if len(out.get(1, [])) == 2:
            shot.armed = True
    return plan


@pytest.mark.parametrize("layer", [0, 1])
def test_d_filter_cache_eval(engine, monkeypatch, shot, layer):
    """D: the eval of one layer's kept rows throws inside the filter of a
    finishing row. The filter still takes the row out of every layer and
    of the pending tokens, so the retried tick decodes the row that is
    left."""
    _Refusals(monkeypatch, shot, layer=layer)
    check(engine, shot, plan=_arm_at_b_finish(shot), b_max=3)


@pytest.mark.parametrize("later", [("next",), ("next", "before")],
                         ids=["retry", "retry-and-victim"])
def test_d_filter_cache_eval_refused_again(engine, monkeypatch, later):
    """D: the allocator refuses the same eval on the retried tick, before
    its step, and then again in the filter that takes the guard's victim
    out. The victim leaves without an eval, and is rebuilt."""
    shot = Shot(MEM)
    refusals = _Refusals(monkeypatch, shot, *later)
    check(engine, shot, plan=_arm_at_b_finish(shot), b_max=3)
    assert not refusals.later


def test_d_filter_host_read(engine, monkeypatch, shot):
    """D: a layer's own filter throws, after the layer before it dropped
    the finished row. No filter can finish that batch, so every row left
    in it is rebuilt."""
    def plan(t, gen, out):
        if len(out.get(1, [])) == 2 and not shot.armed:
            shot.armed = True
            second = gen._generation_batch.prompt_cache[1]
            orig = type(second).filter

            def filt(self, idx):
                if self is second:
                    shot.fire()
                return orig(self, idx)

            monkeypatch.setattr(type(second), "filter", filt)

    check(engine, shot, plan=plan, b_max=3)


def _cancel(uid, when, shot=None):
    done = []

    def plan(t, gen, out):
        if shot is not None and len(out.get(1, [])) == 2:
            shot.armed = True
        if not done and uid in gen._generation_batch.uids and when(out):
            done.append(getattr(gen._generation_batch, "_kq_filter_lazy",
                                False))
            assert gen.remove(uid)
            tg.forget(gen, uid)
            assert uid not in gen._generation_batch.uids
    return plan, done


def test_remove_cache_eval_refused(engine, monkeypatch):
    """A cancelled row is taken out beside one that decodes, and the
    allocator refuses the eval of the rows left. The row is out, and the
    other stream is whole."""
    def when(out):
        return len(out.get(1, [])) == 2

    shot = Shot(MEM)
    _Refusals(monkeypatch, shot)
    ref, ref_fin, _ = run(engine, plan=_cancel(1, when)[0], b_max=6)
    plan, done = _cancel(1, when, shot)
    got, fin, gaps = run(engine, plan=plan, b_max=6)
    assert done == [False] and shot.fired == 1
    assert _tokens(got) == _tokens(ref) and len(got[1]) == 2
    assert fin == ref_fin and set(fin) == {0}
    assert gaps[0] == {0}


def test_remove_after_a_refused_filter(engine, monkeypatch):
    """A row is cancelled after a finish whose filter was refused, before
    the next step evaluated the layers it left, and the allocator refuses
    again. The row is out, and the stream beside it is whole."""
    def when(out):
        return len(out.get(1, [])) == 3

    kw = dict(b_max=3, c_at=1, c_max=6)
    shot = Shot(MEM)
    refusals = _Refusals(monkeypatch, shot, "before")
    ref, ref_fin, _ = run(engine, plan=_cancel(2, when)[0], **kw)
    plan, done = _cancel(2, when, shot)
    got, fin, gaps = run(engine, plan=plan, **kw)
    assert done == [True] and shot.fired == 1 and not refusals.later
    assert _tokens(got) == _tokens(ref) and len(got[2]) < 6
    assert fin == ref_fin and set(fin) == {0, 1}
    assert gaps[0] == {0}


def test_d_forced_eval(engine, monkeypatch, shot):
    """D: the eval of a forced id throws."""
    monkeypatch.setattr(ar, "mx", _Mx(ar.mx, shot, "_force_next_tokens"))
    got = check(engine, shot, plan=_arm_at(shot, 1), forcing=True)
    assert FORCED in _tokens(got)[0]


def test_c_after_the_sample(engine, monkeypatch, shot):
    """C: the decode step throws after its sample. The next step returns
    the token the failed one owed, without a forward."""
    monkeypatch.setattr(ar, "mx", _Mx(ar.mx, shot, "_step"))
    check(engine, shot, plan=_arm_at(shot, 6))


def test_c_in_the_emit_eval(engine, monkeypatch, shot):
    monkeypatch.setattr(ar, "mx", _Mx(ar.mx, shot, "_step", "_emit", name="eval"))
    check(engine, shot, plan=_arm_at(shot, 6))


def test_a_forced_id_on_the_resume_tick_keeps_its_logprob(engine, monkeypatch, shot):
    """The step that resumes feeds the owed token to the criteria, which
    forces an id. The id reports its own raw logprob, which needs the
    failed step's raw logits."""
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


class _Proc:
    """A logits processor that leaves the logits as they are."""

    def __init__(self, shot):
        self.shot = shot

    def __call__(self, tokens, logits):
        self.shot.fire()
        return logits


def test_b_in_a_logits_processor(engine, shot):
    """B: the step throws between its forward and its sample. Every row of
    the batch is rebuilt, so no KV holds a token twice."""
    got = check(engine, shot, plan=_arm_at(shot, 6), processors=[_Proc(shot)])
    assert len(got) == 2


def test_p_in_generate(engine, monkeypatch, shot):
    """P: the first eval of the prompt batch's generate throws after its
    forward moved the new row's cache."""
    monkeypatch.setattr(ar, "mx", _Mx(ar.mx, shot, "generate"))
    check(engine, shot, plan=_arm_at(shot, 4))


def test_p_in_prompt_step(engine, monkeypatch, shot):
    """P: a chunk of a chunked prefill throws after its forward."""
    monkeypatch.setattr(ar, "mx", _Mx(ar.mx, shot, "prompt_step"))
    check(engine, shot, plan=_arm_at(shot, 4), chunked=True)


def test_p0_at_the_prompt_batch_constructor(engine, monkeypatch, shot):
    """P0: the rows have left the queue and no prompt batch holds them."""
    orig = ar.PromptProcessingBatch.__init__

    def init(self, *a, **k):
        shot.fire()
        orig(self, *a, **k)

    monkeypatch.setattr(ar.PromptProcessingBatch, "__init__", init)
    check(engine, shot, plan=_arm_at(shot, 4))


def test_x_before_extend_changes_the_batch(engine, monkeypatch, shot):
    """X: the new batch is lost before it joins; the decode batch is
    whole."""
    orig = ar.GenerationBatch.extend

    def extend(self, other):
        shot.fire()
        return orig(self, other)

    monkeypatch.setattr(ar.GenerationBatch, "extend", extend)
    check(engine, shot, plan=_arm_at(shot, 4))


def test_x_after_extend_changes_the_batch(engine, monkeypatch, shot):
    """X: extend throws half way, so the decode batch is dropped whole and
    every row is rebuilt."""
    orig = ar._extend_cache

    def extend_cache(*a, **k):
        shot.fire()
        return orig(*a, **k)

    monkeypatch.setattr(ar, "_extend_cache", extend_cache)
    check(engine, shot, plan=_arm_at(shot, 4))
