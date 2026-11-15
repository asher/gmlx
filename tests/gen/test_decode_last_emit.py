"""A decode step whose emit ends every row runs no forward.

A real BatchGenerator runs a minted model, and a wrapper on the language
model call counts the forwards. A reference run turns the skip off, so each
case compares the tokens, logprobs and finishes with the stock step. The
tick guard is off in this module (the stock ``_next``), as with
``GMLX_TICK_GUARD=0``: a failed tick raises, and the batch survives it.
"""

from __future__ import annotations

import contextlib
import sys
import types

import pytest

import mlx.core as mx

from gmlx._vendor.mlx_vlm.generate import ar
from gmlx.upstream.patching import root

MEM = "[metal::malloc] Attempting to allocate 1 bytes (test)"
NL, START, END = 10, 99, 100
FORCED = 300


def _llama():
    from gmlx._vendor.mlx_lm.models import llama

    return llama.Model(llama.ModelArgs(
        model_type="llama", hidden_size=32, num_hidden_layers=2,
        intermediate_size=64, num_attention_heads=4, num_key_value_heads=2,
        rms_norm_eps=1e-5, vocab_size=512))


def _qwen3_next():
    from gmlx._vendor.mlx_lm.models import qwen3_next as m

    return m.Model(m.ModelArgs(
        model_type="qwen3_next", hidden_size=64, num_hidden_layers=4,
        intermediate_size=128, num_attention_heads=4,
        linear_num_value_heads=4, linear_num_key_heads=2,
        linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4, num_experts=0, num_experts_per_tok=1,
        decoder_sparse_step=1, shared_expert_intermediate_size=64,
        mlp_only_layers=[], moe_intermediate_size=64, rms_norm_eps=1e-5,
        vocab_size=97, num_key_value_heads=2, rope_theta=10000.0,
        partial_rotary_factor=0.5, max_position_embeddings=512,
        head_dim=16, full_attention_interval=2))


def _deepseek_v4():
    from gmlx.models.deepseek_v4 import model as v4

    return v4.Model(v4.ModelArgs(
        vocab_size=64, hidden_size=64, intermediate_size=64,
        moe_intermediate_size=32, num_hidden_layers=3, num_attention_heads=2,
        n_routed_experts=4, num_experts_per_tok=2, q_lora_rank=32,
        qk_rope_head_dim=16, head_dim=32, compress_ratios=[0, 4, 0],
        hc_mult=2, num_hash_layers=0, sliding_window=8, o_groups=2,
        o_lora_rank=32, index_n_heads=2, index_head_dim=16, index_topk=4,
        num_nextn_predict_layers=0))


_MODELS = {"llama": _llama, "qwen3_next": _qwen3_next,
           "deepseek_v4": _deepseek_v4}


class Rig:
    """A minted model behind a BatchGenerator, with its forwards counted."""

    def __init__(self, monkeypatch, arch="llama"):
        from gmlx._vendor.mlx_vlm.utils import StoppingCriteria
        from gmlx.models.vlm_text_only import TextOnlyModel
        from gmlx.serve.replay_prompt import embed_text

        self.mp = monkeypatch
        mx.random.seed(0)
        self.model = TextOnlyModel(_MODELS[arch]())
        mx.eval(self.model.parameters())
        self.lm = self.model.language_model
        self.calls: list[tuple] = []
        forward = type(self.lm).__call__
        calls = self.calls

        def counting(lm, inputs, *a, **k):
            calls.append(tuple(inputs.shape))
            return forward(lm, inputs, *a, **k)

        monkeypatch.setattr(type(self.lm), "__call__", counting)
        self._embed_text = embed_text
        self.proc = types.SimpleNamespace(tokenizer=types.SimpleNamespace(
            stopping_criteria=StoppingCriteria([], None)))

    def make(self):
        gen = ar.BatchGenerator(self.lm, self.proc, logprobs_cap=3,
                                prefill_step_size=None)
        gen._kq_embed_text = lambda ids, rs: self._embed_text(
            self.model, ids, rs)
        return gen

    def insert(self, gen, prompts, max_tokens, *, criteria=None):
        from gmlx.serve.row_sampling import RowSettings

        return gen.insert(
            prompts, max_tokens,
            prompt_kwargs=[{"inputs_embeds": self.lm.input_embeds(
                mx.array([p]))} for p in prompts],
            thinking_budget_criteria=criteria,
            row_settings=[RowSettings(logprobs=True, top_k=3)
                          for _ in prompts])

    def run(self, rows, *, skip=True, plan=None, ticks=40):
        """Drive ``rows`` to the end. Each row is (tick, prompt, max_tokens,
        criteria or None); rows of one tick go in one insert. Returns the
        responses per uid and the decode forwards of each tick."""
        out: dict[int, list] = {}
        per_tick: list[list] = []
        gen = self.make()
        with (contextlib.nullcontext() if skip else _no_skip(self.mp)):
            try:
                t = 0
                while t < ticks and (t == 0 or gen.has_work
                                     or any(r[0] > t for r in rows)):
                    due = [r for r in rows if r[0] == t]
                    if due:
                        crit = [r[3] for r in due]
                        self.insert(gen, [r[1] for r in due],
                                    [r[2] for r in due],
                                    criteria=crit if any(crit) else None)
                    if plan is not None:
                        plan(t, gen)
                    seen = len(self.calls)
                    _, resps = gen.next()
                    per_tick.append([s for s in self.calls[seen:]
                                     if s[1] == 1])
                    for r in resps:
                        out.setdefault(r.uid, []).append(r)
                    t += 1
            finally:
                gen.close()
        return out, per_tick


@contextlib.contextmanager
def _no_skip(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(ar.GenerationBatch, "_ends_on_this_emit",
                  lambda self: False)
        yield


def _view(out):
    return {uid: [(r.token, r.token_logprob, r.top_logprobs, r.finish_reason)
                  for r in rs] for uid, rs in out.items()}


def _decodes(per_tick):
    return sum(len(c) for c in per_tick)


@pytest.fixture(autouse=True)
def _plain_engine(monkeypatch):
    """The stock tick with no guard, and an extend that never takes a
    batch whose next tokens a skip dropped."""
    monkeypatch.setenv("GMLX_PREFILL_DECAY", "0")
    monkeypatch.setenv("GMLX_CB_PHASE", "0")
    monkeypatch.setenv("GMLX_TICK_GUARD", "0")
    monkeypatch.setattr(ar.BatchGenerator, "_next",
                        root(ar.BatchGenerator._next))
    orig = ar.GenerationBatch.extend

    def extend(self, other):
        assert not self.uids or self._next_tokens is not None
        return orig(self, other)

    monkeypatch.setattr(ar.GenerationBatch, "extend", extend)


@pytest.fixture
def rig(monkeypatch):
    return Rig(monkeypatch)


A, B, C = [3, 9, 4, 7], [5, 6, 2], [40, 41]


def test_a_one_token_row_makes_one_forward(rig):
    ref, ref_ticks = rig.run([(0, A, 1, None)], skip=False)
    n_ref = len(rig.calls)
    rig.calls.clear()
    got, ticks = rig.run([(0, A, 1, None)])
    assert n_ref == 2 and len(rig.calls) == 1
    assert _decodes(ticks) == 0 and _decodes(ref_ticks) == 1
    assert _view(got) == _view(ref)
    (rs,) = got.values()
    assert [r.finish_reason for r in rs] == ["length"]
    assert rs[0].token_logprob is not None and len(rs[0].top_logprobs) == 3


@pytest.mark.parametrize("rows", [
    [(0, A, 2, None), (0, B, 2, None)],
    [(0, A, 3, None), (0, B, 1, None)],
    [(0, A, 2, None), (2, B, 1, None), (3, C, 3, None)],
    [(0, A, 4, None), (2, B, 2, None)],
], ids=["together", "different-max", "staggered", "join"])
def test_rows_give_the_stock_tokens(rig, rows):
    """Rows of different lengths and arrivals give the stock tokens and
    logprobs. Each skip saves one decode forward."""
    ref, ref_ticks = rig.run(rows, skip=False)
    got, ticks = rig.run(rows)
    assert _view(got) == _view(ref)
    for rs in got.values():
        assert rs[-1].finish_reason == "length"
    skips = sum(1 for a, b in zip(ticks, ref_ticks) if b and not a)
    assert skips >= 1
    assert _decodes(ticks) == _decodes(ref_ticks) - skips


def test_a_row_that_continues_keeps_the_forward(rig):
    """B ends on the first emit while A goes on, so that step forwards both
    rows. A's last emit skips."""
    _, ticks = rig.run([(0, A, 3, None), (0, B, 1, None)])
    assert ticks[0] == []                    # the prefill tick
    assert ticks[1] == [(2, 1)]
    assert ticks[2] == [(1, 1)]
    assert ticks[3] == []


def test_a_speech_row_keeps_the_forward(rig, monkeypatch):
    from gmlx.serve import speech_rows

    def forward(batch, inputs, fwd_kwargs, speaking):
        out = batch._language_model(inputs[:, None], cache=batch.prompt_cache,
                                    **fwd_kwargs)
        return out.logits[:, -1, :], None, ()

    monkeypatch.setattr(speech_rows, "rows_of",
                        lambda batch: {0: object()} if batch.uids else None)
    monkeypatch.setattr(speech_rows, "forward", forward)
    monkeypatch.setattr(speech_rows, "sample_frames",
                        lambda batch, sampled, hidden, speaking: sampled)
    got, ticks = rig.run([(0, A, 1, None)])
    assert _decodes(ticks) == 1
    (rs,) = got.values()
    assert [r.finish_reason for r in rs] == ["length"]


@pytest.mark.parametrize("closed", [False, True])
def test_only_an_open_speech_row_blocks_the_skip(closed):
    gb = ar.GenerationBatch.empty(None, None, None)
    gb.uids, gb.max_tokens, gb._num_tokens = [7], [1], [0]
    gb._row_settings = {7: types.SimpleNamespace(
        speech=types.SimpleNamespace(closed=closed))}
    assert gb._ends_on_this_emit() is closed


class _Forcing:
    """Forces FORCED after the row's ``at``th token, then nothing."""

    def __init__(self, at=1):
        self.n, self.at = 0, at
        self.forced_token_id = None
        self.popped = []

    def __call__(self, tok):
        self.n += 1
        self.forced_token_id = FORCED if self.n == self.at else None
        return self.forced_token_id

    def pop_forced_token_id(self):
        out, self.forced_token_id = self.forced_token_id, None
        if out is not None:
            self.popped.append(out)
        return out


def test_a_budget_row_that_ends_skips(rig):
    """The budget forces a token on the row's last emit. The row has no next
    token to force, and the step forwards nothing."""
    crit = _Forcing(at=1)
    got, ticks = rig.run([(0, A, 1, crit)])
    assert crit.popped == [FORCED]
    assert _decodes(ticks) == 0
    (rs,) = got.values()
    assert [r.finish_reason for r in rs] == ["length"]


def test_a_budget_row_that_continues_gets_its_forced_token(rig):
    ref, ref_ticks = rig.run([(0, A, 3, _Forcing(at=1))], skip=False)
    got, ticks = rig.run([(0, A, 3, _Forcing(at=1))])
    assert _view(got) == _view(ref)
    (rs,) = got.values()
    assert rs[1].token == FORCED
    assert _decodes(ticks) == _decodes(ref_ticks) - 1 == 2


class _Tok:
    _MAP = {"\n": [NL], "</think>": [END], "<think>": [START]}

    def encode(self, text, add_special_tokens=True):
        return self._MAP.get(text, [])


def _off_budget():
    from gmlx._vendor.mlx_vlm.utils import ThinkingBudgetCriteria

    crit = ThinkingBudgetCriteria(_Tok(), thinking_budget=0,
                                  thinking_start_token="<think>",
                                  enable_thinking=True,
                                  prompt_preopens_thinking=True)
    popped = []
    pop = crit.pop_forced_token_id

    def spy():
        out = pop()
        if out is not None:
            popped.append(out)
        return out

    crit.pop_forced_token_id = spy
    return crit, popped


@pytest.mark.parametrize("max_tokens", [1, 3])
def test_a_budget_zero_thinking_off_row_skips(rig, max_tokens):
    """Thinking off as a budget of 0 forces the close from the first token.
    The row's last emit skips, and the close it forces before then lands."""
    ref_crit, _ = _off_budget()
    ref, ref_ticks = rig.run([(0, A, max_tokens, ref_crit)], skip=False)
    crit, popped = _off_budget()
    got, ticks = rig.run([(0, A, max_tokens, crit)])
    assert popped and popped[0] == NL
    assert _view(got) == _view(ref)
    assert _decodes(ticks) == _decodes(ref_ticks) - 1
    if max_tokens == 3:
        assert [r.token for r in got[0]][1:] == [NL, END]


def test_force_next_tokens_without_next_tokens_returns():
    gb = ar.GenerationBatch.empty(None, None, None)
    gb._next_raw = mx.zeros((1, 4))
    gb._force_next_tokens([FORCED])
    assert gb._next_tokens is None and gb._next_raw is None


class _Mx:
    """ar's mx. ``eval`` from ``_emit`` throws once when armed; ``async_eval``
    calls from ``_step`` are counted."""

    def __init__(self, real):
        self._real = real
        self.armed = False
        self.fired = 0
        self.step_async = 0

    def __getattr__(self, n):
        fn = getattr(self._real, n)
        if n not in ("eval", "async_eval"):
            return fn

        def call(*a, **k):
            caller = sys._getframe(1).f_code.co_name
            if n == "eval" and caller == "_emit" and self.armed \
                    and not self.fired:
                self.fired += 1
                raise RuntimeError(MEM)
            if n == "async_eval" and caller == "_step":
                self.step_async += 1
            return fn(*a, **k)
        return call


def test_an_emit_failure_after_a_skip_owes_the_token(rig, monkeypatch):
    ref, _ = rig.run([(0, A, 1, None)], skip=False)
    proxy = _Mx(ar.mx)
    monkeypatch.setattr(ar, "mx", proxy)
    rig.calls.clear()
    gen = rig.make()
    try:
        rig.insert(gen, [A], 1)
        gen.next()                            # the prefill tick
        gb = gen._generation_batch
        proxy.armed = True
        with pytest.raises(RuntimeError, match="malloc"):
            gen.next()
        assert gb._kq_owed is not None and gb._next_tokens is None
        assert gb._num_tokens == [0]
        proxy.step_async = 0
        _, resps = gen.next()
        assert proxy.step_async == 0
        assert gb._kq_owed is None
        assert _view({0: resps}) == _view(ref)
        assert not gen.has_work
    finally:
        gen.close()
    assert len(rig.calls) == 1


def test_a_failed_filter_after_a_skip_drops_the_rows_next_call(
        rig, monkeypatch):
    """The filter of a step that skipped throws. Its responses ride the
    error, the next call drops the rows, and a row inserted meanwhile
    decodes as alone."""
    ref, _ = rig.run([(0, C, 3, None)], skip=False)
    shot = {"armed": False, "fired": 0}
    orig = ar.GenerationBatch.filter

    def filt(self, keep):
        if shot["armed"] and not shot["fired"]:
            shot["fired"] += 1
            raise RuntimeError(MEM)
        return orig(self, keep)

    monkeypatch.setattr(ar.GenerationBatch, "filter", filt)
    gen = rig.make()
    try:
        ua, ub = rig.insert(gen, [A, B], 1)
        gen.next()
        gb = gen._generation_batch
        shot["armed"] = True
        with pytest.raises(RuntimeError, match="malloc") as err:
            gen.next()
        salvaged = err.value._kq_salvage
        assert sorted((r.uid, r.finish_reason) for r in salvaged) == [
            (ua, "length"), (ub, "length")]
        assert sorted(gb.uids) == [ua, ub] and gb._next_tokens is None
        (uc,) = rig.insert(gen, [C], 3)
        out = []
        while gen.has_work:
            _, resps = gen.next()
            out.extend(resps)
        assert {r.uid for r in out} == {uc}
        assert _view({0: out}) == _view(ref)
    finally:
        gen.close()


# Retirement after a skip. A raw-completion row's stash rides the plain
# ckpt hooks; render_ctx None keys the whole sequence.


@pytest.fixture
def retire_hooks(monkeypatch):
    import gmlx.cache.snapshot as snapshot
    import gmlx.spec.ckpt as ckpt

    for name in ("_step", "filter", "extend"):
        monkeypatch.setattr(ar.GenerationBatch, name,
                            getattr(ar.GenerationBatch, name))
    ckpt._install_plain_ckpt_decode()
    seen = {"extract": 0, "retire": []}
    extract = snapshot.row_snapshot
    retire = ckpt._plain_retire

    def row_snapshot(*a, **k):
        seen["extract"] += 1
        return extract(*a, **k)

    def plain_retire(stash, caches):
        seen["retire"].append(len(stash["full_ids"]) + len(stash["gen"]))
        return retire(stash, caches)

    monkeypatch.setattr(snapshot, "row_snapshot", row_snapshot)
    monkeypatch.setattr(ckpt, "_plain_retire", plain_retire)
    return seen


def _stash_plan(prompts, mode, man):
    def plan(t, gen):
        if t == 1:
            gb = gen._generation_batch
            # uids count from 0 in insert order; the batch may reorder.
            gb._kq_apc_retire_rows = {
                uid: {"full_ids": list(prompts[uid]), "extra_hash": 0,
                      "mode": mode, "manager": man, "render_ctx": None,
                      "snap_ok": False, "gen": []}
                for uid in gb.uids}
    return plan


@pytest.mark.parametrize("prompts", [
    [list(range(3, 43))],
    [list(range(3, 43)), list(range(50, 83))],
], ids=["b1-live", "b2-extract"])
def test_a_ckpt_row_retires_without_its_last_token(
        monkeypatch, retire_hooks, prompts):
    from gmlx._vendor.mlx_vlm.apc import APCManager
    from gmlx.cache.snapshot import ckpt_lookup

    rig = Rig(monkeypatch, "qwen3_next")
    man = APCManager(num_blocks=64, block_size=16)
    rows = [(0, p, 3, None) for p in prompts]
    out, ticks = rig.run(rows, plan=_stash_plan(prompts, "ckpt", man))
    assert ticks[-1] == []                       # the last emit skipped
    assert (retire_hooks["extract"] > 0) == (len(prompts) > 1)
    for uid, p in enumerate(prompts):
        gen = [r.token for r in out[uid]]
        assert len(gen) == 3
        _, got = ckpt_lookup(man, p + gen + [1], extra_hash=0)
        assert got == len(p) + len(gen) - 1


def test_a_pooling_row_retires_exact_without_its_last_token(
        monkeypatch, retire_hooks):
    from gmlx._vendor.mlx_vlm.apc import APCManager
    from gmlx.cache.apc_pooling import install_pooling_apc_support

    install_pooling_apc_support()
    rig = Rig(monkeypatch, "deepseek_v4")
    man = APCManager(num_blocks=64, block_size=16)
    prompt = list(range(1, 20))
    out, ticks = rig.run([(0, prompt, 3, None)],
                         plan=_stash_plan([prompt], "exact", man))
    assert ticks[-1] == []
    assert retire_hooks["extract"] == 0
    gen = [r.token for r in out[0]]
    warm, got = man.lookup_exact_cache(prompt + gen + [1], extra_hash=0)
    assert warm is not None and got == len(prompt) + len(gen) - 1
