"""The batch engine's logprobs are the model's own distribution: f32
log_softmax of the raw logits, before logit bias and penalties, for the
first token, every decode token and every forced token. Each row keeps its
own sampler and logprob settings in any batch, and the settings store
follows the shared uid lifetime rule."""

import importlib
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from gmlx._vendor.mlx_lm.models import llama
from gmlx._vendor.mlx_lm.models.llama import ModelArgs
from gmlx._vendor.mlx_vlm.sample_utils import make_logits_processors
from gmlx._vendor.mlx_vlm.server.generation import _position_seed
from gmlx._vendor.mlx_vlm.utils import StoppingCriteria
from gmlx.models.vlm_text_only import TextOnlyModel
from gmlx.serve import row_sampling as rsm
from gmlx.serve.patches.sampling import _FastPositionedSampler
from gmlx.serve.row_sampling import RowSettings

ar = importlib.import_module("gmlx._vendor.mlx_vlm.generate.ar")

_V = 64
_K = 8
_BIAS = 7


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("GMLX_PREFILL_DECAY", "0")
    monkeypatch.setenv("GMLX_CB_PHASE", "0")


@pytest.fixture(scope="module")
def model():
    mx.random.seed(0)
    inner = llama.Model(ModelArgs(
        model_type="llama", hidden_size=32, num_hidden_layers=2,
        intermediate_size=64, num_attention_heads=4,
        num_key_value_heads=2, rms_norm_eps=1e-5, vocab_size=_V))
    m = TextOnlyModel(inner)
    m.set_dtype(mx.bfloat16)
    mx.eval(m.parameters())
    return m


def _gen(model, **kw):
    proc = SimpleNamespace(tokenizer=SimpleNamespace(
        stopping_criteria=StoppingCriteria([], None)))
    kw.setdefault("logprobs_cap", _K)
    return ar.BatchGenerator(model.language_model, proc, prefill_step_size=None, **kw)


def _ins(gen, model, prompts, n, **kw):
    lm = model.language_model
    kw["prompt_kwargs"] = [{"inputs_embeds": lm.input_embeds(mx.array([p]))}
                           for p in prompts]
    return gen.insert(prompts, n, **kw)


def _procs():
    return make_logits_processors({_BIAS: 6.0}, repetition_penalty=1.5)


def _lp_row(**kw):
    kw.setdefault("logprobs", True)
    kw.setdefault("top_k", _K)
    return RowSettings(**kw)


def _run(gen, max_steps=64):
    out = {}
    for _ in range(max_steps):
        if not gen.has_work:
            break
        _p, resp = gen.next()
        for r in resp:
            out.setdefault(r.uid, []).append(r)
    return out


def _teacher(model, tokens):
    """f32 log_softmax of a whole-sequence forward, one row per position."""
    logits = model.language_model(mx.array([tokens]))
    logits = getattr(logits, "logits", logits)[0].astype(mx.float32)
    return np.array(logits - mx.logsumexp(logits, axis=-1, keepdims=True))


def _check_against_teacher(model, prompt, resps):
    toks = [r.token for r in resps]
    ref = _teacher(model, prompt + toks[:-1])
    for i, r in enumerate(resps):
        row = ref[len(prompt) - 1 + i]
        # Incremental decode against a whole forward, in bf16 activations.
        assert r.token_logprob == pytest.approx(row[r.token], abs=3e-2)
        ids = [t for t, _ in r.top_logprobs]
        vals = [v for _, v in r.top_logprobs]
        assert len(ids) == _K
        np.testing.assert_allclose(vals, np.sort(row)[::-1][:_K], atol=3e-2)
        np.testing.assert_allclose(vals, row[ids], atol=3e-2)
    return ref


def test_first_and_decode_logprobs_are_raw_f32(model):
    prompt = [3, 9, 4, 7, 1]
    gen = _gen(model, max_tokens=6)
    (uid,) = _ins(gen, model, [prompt], 6, logits_processors=[_procs()],
                        row_settings=[_lp_row()])
    resps = _run(gen)[uid]
    gen.close()
    assert len(resps) == 6
    ref = _check_against_teacher(model, prompt, resps)
    # The bias makes token 7 the sampled argmax; its raw logprob does not
    # carry the bias.
    assert all(r.token == _BIAS for r in resps[:1])
    assert resps[0].token_logprob == pytest.approx(ref[len(prompt) - 1][_BIAS], abs=3e-2)
    # f32, not rounded to bf16.
    vals = np.array([r.token_logprob for r in resps], dtype=np.float32)
    rounded = np.array(mx.array(vals).astype(mx.bfloat16).astype(mx.float32))
    assert (vals != rounded).any()


class _Force:
    """A thinking-budget stand-in that forces ``ids`` after the first token."""

    def __init__(self, ids):
        self.queue = list(ids)
        self.seen = 0

    def __call__(self, tok):
        self.seen += 1

    def pop_forced_token_id(self):
        return self.queue.pop(0) if self.seen >= 1 and self.queue else None


@pytest.mark.parametrize("forced", [[11], [11, 12, 13]])
def test_a_forced_token_reports_its_own_raw_logprob(model, forced):
    prompt = [3, 9, 4, 7, 1]
    n = 2 + len(forced)
    gen = _gen(model, max_tokens=n)
    (uid,) = _ins(gen, model, [prompt], n, thinking_budget_criteria=[_Force(forced)],
                        row_settings=[_lp_row()])
    resps = _run(gen)[uid]
    gen.close()
    assert [r.token for r in resps[1:1 + len(forced)]] == forced
    _check_against_teacher(model, prompt, resps)


def test_retirement_follows_the_shared_rule(model, monkeypatch):
    gen = _gen(model, max_tokens=2)
    (a,) = _ins(gen, model, [[1, 2, 3]], 2, row_settings=[_lp_row()])
    assert gen.remove(a)
    assert a in gen._row_settings               # remove keeps the settings
    (b,) = _ins(gen, model, [[4, 5]], 2, row_settings=[_lp_row()])
    _run(gen)
    assert b not in gen._row_settings           # its finish popped it
    # Past the bound, insert drops only uids that are not live.
    monkeypatch.setattr(rsm, "MAX_ROWS", 2)
    gen._row_settings.update({100: _lp_row(), 101: _lp_row(), 102: _lp_row()})
    gen._unprocessed_sequences.append((100, [1], 1, {}, None, None))
    gen._generation_batch = SimpleNamespace(uids=[], _all_uids=[101])
    (c,) = _ins(gen, model, [[6]], 1, row_settings=[_lp_row()])
    assert sorted(gen._row_settings) == sorted([100, 101, c])
    gen.close()


class _Spy:
    """Records every row-sampler draw: bound uids, logprobs, positions, out."""

    def __init__(self, monkeypatch):
        self.draws = []
        orig = rsm.RowSampler.sample_target
        spy = self

        def sample_target(this, lp, *, row_ids, positions):
            out = orig(this, lp, row_ids=row_ids, positions=positions)
            spy.draws.append((list(this._uids), np.array(lp.astype(mx.float32)),
                              list(positions), np.array(out).tolist()))
            return out

        monkeypatch.setattr(rsm.RowSampler, "sample_target", sample_target)


def test_a_greedy_row_in_a_sampled_batch_stays_greedy(model, monkeypatch):
    spy = _Spy(monkeypatch)
    gen = _gen(model, max_tokens=6, sampler=_FastPositionedSampler(temperature=3.0),
               greedy_sampling=False)
    hot = RowSettings(sampler=_FastPositionedSampler(temperature=3.0), key=("t3",))
    greedy, sampled = _ins(gen, model,
        [[3, 9, 4], [5, 2, 8, 1]], 6, row_settings=[RowSettings(), hot])
    _run(gen)
    gen.close()
    joined = [d for d in spy.draws if {greedy, sampled} <= set(d[0])]
    assert joined
    for uids, lp, _pos, out in spy.draws:
        if greedy in uids:
            j = uids.index(greedy)
            assert out[j] == int(np.argmax(lp[j]))


def test_a_seeded_row_in_a_greedy_batch_draws_its_seed_keys(model, monkeypatch):
    spy = _Spy(monkeypatch)
    gen = _gen(model, max_tokens=5)
    seeded = RowSettings(sampler=_FastPositionedSampler(temperature=2.0), seed=42,
                         key=("t2",))
    greedy, row = _ins(gen, model, [[3, 9, 4], [5, 2, 8, 1]], 5,
                       row_settings=[RowSettings(), seeded])
    _run(gen)
    gen.close()
    assert [d for d in spy.draws if {greedy, row} <= set(d[0])]
    for uids, lp, pos, out in spy.draws:
        if row not in uids:
            continue
        j = uids.index(row)
        key = mx.random.key(_position_seed(42, 0, pos[j]))
        want = mx.random.categorical(mx.array(lp[j]) * 0.5, key=key)
        assert out[j] == int(want.item())


def test_a_logprob_row_joining_a_plain_batch_gets_its_top_k(model):
    gen = _gen(model, max_tokens=8)
    (plain,) = _ins(gen, model, [[3, 9, 4, 7]], 8, row_settings=[RowSettings()])
    for _ in range(3):
        gen.next()
    (lp_uid,) = _ins(gen, model, [[5, 2]], 4, row_settings=[_lp_row(top_k=3)])
    out = _run(gen)
    gen.close()
    assert all(r.token_logprob is None and r.top_logprobs is None
               for r in out[plain])
    first = out[lp_uid][0]
    assert first.token_logprob is not None and len(first.top_logprobs) == 3


def test_a_late_logprob_row_gets_top_k_at_the_server_cap(model):
    # The loop builds the generator from its first request, which asked
    # for no logprobs, and passes the server cap apart.
    gen = _gen(model, max_tokens=3, compute_logprobs=False, top_logprobs_k=0,
               logprobs_cap=5)
    _ins(gen, model, [[1, 2]], 3, row_settings=[RowSettings()])
    (uid,) = _ins(gen, model, [[4, 5, 6]], 3, row_settings=[_lp_row(top_k=5)])
    out = _run(gen)
    gen.close()
    assert all(len(r.top_logprobs) == 5 for r in out[uid])


@pytest.mark.parametrize("row,fused", [
    (RowSettings(), True),
    (RowSettings(logprobs=True), False),
    (RowSettings(sampler=lambda x: mx.argmax(x, axis=-1), key=("s",)), False),
])
def test_the_fused_greedy_step_needs_greedy_rows_without_logprobs(model, monkeypatch,
                                                                  row, fused):
    lm = model.language_model
    target = getattr(lm, "language_model", lm)
    calls = []

    def fused_greedy_decode(self, *a, **k):
        calls.append(1)
        return None

    monkeypatch.setattr(type(target), "fused_greedy_decode", fused_greedy_decode,
                        raising=False)
    gen = _gen(model, max_tokens=3)
    _ins(gen, model, [[1, 2, 3]], 3, row_settings=[row])
    _run(gen)
    gen.close()
    assert bool(calls) is fused

