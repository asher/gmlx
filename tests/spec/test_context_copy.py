"""Context copy: drafts copied from the prompt, verified like any draft.

Covers the prompt index and copy-length policy, the acceptance law for a
copied token (a point-mass proposal under the stochastic walk), the scalar
round that verifies a copy in place of the drafter's block on a tiny real
target and DFlash 2 drafter, the switches, and the serve transport of the
prompt ids.
"""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import gmlx.spec.context_copy as cc
import gmlx.spec.speculative as spec
from gmlx.spec.context_copy import KEY, PromptCopy


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("GMLX_CONTEXT_COPY", "GMLX_CONTEXT_COPY_MAX",
                "MLX_VLM_GGUF_CONTEXT_COPY"):
        monkeypatch.delenv(var, raising=False)


# -- index and policy -----------------------------------------------------


def _gen(copier, tokens):
    copier.extend(tokens)
    return copier


def test_lookup_proposes_the_prompt_continuation():
    prompt = list(range(100, 140))
    c = PromptCopy(prompt, longest=24)
    _gen(c, [7, 7, 7] + prompt[10:10 + KEY])
    assert c.propose(1, 100) == prompt[10 + KEY:10 + KEY + 8]   # START


def test_lookup_prefers_the_longest_agreement_then_the_latest():
    key = [1, 2, 3, 4, 5, 6]
    # the same key twice: once after 9, once after 8; then once more after 8
    prompt = [9] + key + [50, 51] + [8] + key + [60, 61] + [8] + key + [70, 71]
    c = PromptCopy(prompt, longest=24)
    _gen(c, [9] + key)
    assert c.propose(1, 2) == [50, 51]
    c = PromptCopy(prompt, longest=24)
    _gen(c, [8] + key)
    assert c.propose(1, 2) == [70, 71]          # latest of the equal pair


def test_no_proposal_without_a_full_key_match():
    prompt = list(range(100, 140))
    c = _gen(PromptCopy(prompt), [5] + prompt[10:10 + KEY - 1])
    assert c.propose(1, 100) == []


def test_proposal_respects_the_shortest_the_budget_and_the_prompt_end():
    prompt = list(range(100, 140))
    c = _gen(PromptCopy(prompt, longest=24), prompt[28:28 + KEY])
    assert c.propose(1, 100) == prompt[34:40]   # 6 left before the end
    assert c.propose(7, 100) == []               # shorter than the drafter
    assert c.propose(1, 3) == prompt[34:37]      # budget


def test_whole_accepts_grow_the_copy_and_a_misfire_pauses():
    prompt = list(range(100, 200))
    c = _gen(PromptCopy(prompt, longest=24), prompt[:KEY])
    assert len(c.propose(1, 100)) == 8
    c.record(8, 8)
    assert len(c.propose(1, 100)) == 16
    c.record(16, 16)
    assert len(c.propose(1, 100)) == 24
    c.record(24, 3)                              # under a quarter: misfire
    assert c.propose(1, 100) == []
    c.extend([0] * (cc.PAUSE - 1))
    assert c.propose(1, 100) == []
    c.extend([0])
    c.extend(prompt[40:40 + KEY])
    assert len(c.propose(1, 100)) == 8           # length started over
    c.record(8, 1)                               # second misfire in a row
    c.extend(prompt[60:60 + KEY])
    c.extend([0] * (2 * cc.PAUSE - KEY - 1))
    assert c.propose(1, 100) == []


def test_partial_accept_keeps_the_length():
    prompt = list(range(100, 200))
    c = _gen(PromptCopy(prompt, longest=24), prompt[:KEY])
    c.record(8, 8)
    c.record(16, 6)
    assert len(c.propose(1, 100)) == 16


def test_switches(monkeypatch):
    on = SimpleNamespace(context_copy=True)
    off = SimpleNamespace()
    assert cc.context_copy_enabled(on) and not cc.context_copy_enabled(off)
    monkeypatch.setenv("GMLX_CONTEXT_COPY", "0")
    assert not cc.context_copy_enabled(on)
    monkeypatch.setenv("GMLX_CONTEXT_COPY", "1")
    assert cc.context_copy_enabled(off)
    assert cc.longest_copy() == cc.LONGEST
    monkeypatch.setenv("GMLX_CONTEXT_COPY_MAX", "200")
    assert cc.longest_copy() == cc.CEILING
    monkeypatch.setenv("GMLX_CONTEXT_COPY_MAX", "x")
    assert cc.longest_copy() == cc.LONGEST


def test_load_stamps_the_config_key(monkeypatch):
    from gmlx.spec.mtp_load import _stamp_mtp_width_cap

    quiet = dict(log=lambda *a, **k: None)
    d = _stamp_mtp_width_cap(SimpleNamespace(), "dflash2", hard_limit=1,
                             **quiet)
    assert d.context_copy is False
    monkeypatch.setenv("MLX_VLM_GGUF_CONTEXT_COPY", "1")
    d = _stamp_mtp_width_cap(SimpleNamespace(), "dflash2", hard_limit=1,
                             **quiet)
    assert d.context_copy is True


# -- acceptance law -------------------------------------------------------


def _sampler(temp=1.0):
    def s(logprobs):
        return mx.random.categorical(logprobs / temp, axis=-1)
    spec.annotate_sampling_params(s, temp=temp, top_p=1.0, top_k=0,
                                  min_p=0.0)
    return s


def test_point_mass_walk_keeps_the_target_law():
    """A copied token d proposed with q = 1 is accepted with p(d), and a
    rejection draws from (p - q)+, which is p without d. The emitted token's
    law is then p itself. Every position below is an independent trial of
    the same copy."""
    mx.random.seed(0)
    V, n = 8, 20000
    p = np.array([0.30, 0.25, 0.15, 0.10, 0.08, 0.06, 0.04, 0.02])
    logits = mx.array(np.log(np.tile(p, (n + 1, 1))).astype(np.float32))
    d = 1
    acc, res, _ = spec._stoch_walk_graph(
        _sampler(), logits, mx.full((n,), d, dtype=mx.int32), None)
    acc, res = np.array(acc), np.array(res)
    assert abs(acc.mean() - p[d]) < 0.015
    assert not (res == d).any()
    emitted = np.where(acc, d, res)
    freq = np.bincount(emitted, minlength=V) / n
    assert np.abs(freq - p).max() < 0.015


def test_point_mass_walk_rejects_what_the_target_rules_out():
    logits = np.full((3, 16), -1e9, dtype=np.float32)
    logits[:, 5] = 0.0
    graph = spec._stoch_walk_graph(
        _sampler(), mx.array(logits), mx.array([7, 7], dtype=mx.int32), None)
    mx.eval(*graph)
    acc_i, new = spec._stoch_walk_finish(graph, [7, 7], 10)
    assert acc_i == 0 and new == [5]


def test_point_mass_matches_an_explicit_one_hot_stash():
    """The None form is the one-hot q rows the general walk takes."""
    rng = np.random.default_rng(3)
    logits = mx.array(rng.normal(0, 2, (4, 12)).astype(np.float32))
    draft = mx.array([3, 9, 0], dtype=mx.int32)
    one_hot = [mx.array(np.eye(12, dtype=np.float32)[t]) for t in (3, 9, 0)]
    mx.random.seed(5)
    a = spec._stoch_walk_graph(_sampler(), logits, draft, None)
    mx.eval(*a)
    mx.random.seed(5)
    b = spec._stoch_walk_graph(_sampler(), logits, draft, one_hot)
    mx.eval(*b)
    for x, y in zip(a, b):
        assert np.array_equal(np.array(x), np.array(y))


# -- the scalar round -----------------------------------------------------


class _Scripted:
    """Stands in for PromptCopy: proposes the next tokens of a script, with
    an optional wrong token, and records the outcome of each round."""

    def __init__(self, script, wrong_at=None, every=1):
        self.script = list(script)
        self.wrong_at = wrong_at
        self.every = every
        self.seen = []
        self.records = []
        self.calls = 0

    def extend(self, tokens):
        self.seen.extend(int(t) for t in tokens)

    def propose(self, shortest, budget):
        self.calls += 1
        if self.calls % self.every:
            return []
        i = len(self.seen)
        out = self.script[i:i + min(12, budget)]
        if self.wrong_at is not None and len(out) > self.wrong_at:
            out = list(out)
            out[self.wrong_at] = (out[self.wrong_at] + 1) % 100
        return out if len(out) >= max(shortest, 1) else []

    def record(self, copied, accepted):
        self.records.append((copied, accepted))


def _dflash_walk(monkeypatch, scripted, n):
    from test_dflash2_drafter import (BLOCK, CAPTURE, _drafter,
                                      _engine_reference, _target)

    lm = _target()
    prompt = mx.array([[1, 2, 3, 4, 5]])
    ref = _engine_reference(lm, prompt, n)
    script = scripted(ref)
    monkeypatch.setenv("GMLX_CONTEXT_COPY", "1")
    monkeypatch.setattr(spec, "PromptCopy", lambda p: script)
    drafter = _drafter(lm)
    widths = []
    orig = spec._mtp_verify_target

    def verify(lm_, ids, *a, **k):
        widths.append(int(ids.shape[1]))
        return orig(lm_, ids, *a, **k)

    monkeypatch.setattr(spec, "_mtp_verify_target", verify)
    lm.set_dflash_capture(CAPTURE)
    cache = lm.make_cache()
    out = lm(prompt, cache=cache, return_hidden=True, return_shared_kv=True)
    b = int(mx.argmax(out.logits[:, -1, :], axis=-1).item())
    spec._buffer_mtp_target_cache(cache, drafter, BLOCK)
    toks = [b]
    toks.extend(spec._owned_decode_rounds(
        lm, drafter, lm, cache, hidden=out.hidden_states[-1], b=b,
        shared_kv=out.shared_kv_states, seed_tokens=prompt, emitted=1,
        max_tokens=n, sampler=None, draft_block_size=BLOCK,
        copy_prompt=[1, 2, 3, 4, 5]))
    return toks, ref, script, widths, drafter, BLOCK


@pytest.mark.parametrize("wrong_at", [None, 4])
def test_copy_rounds_keep_the_greedy_chain(monkeypatch, wrong_at):
    """Copies longer than the drafter's block verify in one forward, a
    wrong copied token cuts the round there, and the output is the target's
    greedy chain either way."""
    n = 40
    toks, ref, script, widths, drafter, block = _dflash_walk(
        monkeypatch, lambda r: _Scripted(r, wrong_at=wrong_at), n)
    assert toks[:n] == ref
    assert max(widths) > block
    assert script.records
    if wrong_at is None:
        assert all(a == c for c, a in script.records)
    else:
        assert all(a == min(wrong_at, c) for c, a in script.records)
    # the drafter's stats hold only its own rounds
    assert len(drafter.accept_lens) == len(widths) - len(script.records)


def test_copy_and_drafter_rounds_alternate(monkeypatch):
    """Every other round copies; the drafter rounds between them draft from
    a context that ingested the copied positions, and the chain holds."""
    n = 40
    toks, ref, script, widths, drafter, block = _dflash_walk(
        monkeypatch, lambda r: _Scripted(r, every=2), n)
    assert toks[:n] == ref
    assert drafter.accept_lens and script.records
    ring = drafter._cache[0]
    # prompt capture plus every committed position, copied or drafted
    assert ring.offset >= 5 + sum(a + 1 for _, a in script.records)


def test_copy_off_by_default(monkeypatch):
    from test_dflash2_drafter import _drafter, _target

    made = []
    monkeypatch.setattr(spec, "PromptCopy", lambda p: made.append(p))
    lm = _target()
    assert not cc.context_copy_enabled(_drafter(lm))
    assert made == []


def test_stochastic_copy_round_uses_the_point_mass_walk(monkeypatch):
    from test_mtp_mixed_batch import _EchoLM, _FakeCache, _RowDrafter

    seen = []
    orig = spec._stochastic_walk

    def walk(lm, verify, drafts, sampler, budget, q_rows, process=None):
        seen.append(q_rows)
        return orig(lm, verify, drafts, sampler, budget, q_rows,
                    process=process)

    monkeypatch.setattr(spec, "_STOCH_ACCEPT", True)
    monkeypatch.setattr(spec, "_stochastic_walk", walk)
    monkeypatch.setenv("GMLX_CONTEXT_COPY", "1")
    d = _RowDrafter(block_size=4)
    d.set_shared_kv = lambda *a, **k: None
    # no q stash from this drafter: its own rounds fall back to exact match
    prompt = list(range(40)) + [0, 1, 2, 3, 4, 5]
    gen = spec._owned_decode_rounds(
        SimpleNamespace(), d, _EchoLM(), [_FakeCache(width=1)],
        hidden=mx.zeros((1, 4, 8)), b=6, shared_kv=None,
        seed_tokens=mx.zeros((1, 4), dtype=mx.int32), emitted=1,
        max_tokens=24, sampler=_sampler(0.01), draft_block_size=None,
        copy_prompt=prompt)
    out = list(gen)
    # the echo target continues 6 -> 7 -> 8 ..., which the prompt holds
    assert out == list(range(7, 30))
    assert seen and all(q is None for q in seen)


# -- serve transport ------------------------------------------------------


def test_transport_carries_a_lone_rows_full_prompt(monkeypatch):
    from gmlx.serve.patches import context_copy as tp

    class Spec:
        def __init__(self, n):
            self._all_uids = list(range(n))
            self._rounds_iter = None
            self.prompt_cache = [SimpleNamespace()]
            self.hidden = mx.zeros((n, 1, 2))

        def _start_rounds(self):
            return "started"

    class PPB:
        def __init__(self, input_ids, apc_meta=None, draft_model=None,
                     draft_kind=None):
            self.draft_model = draft_model
            self.draft_kind = draft_kind
            self.n = len(input_ids)

        def generate(self):
            return Spec(self.n)

    tp._install_capture(PPB)
    tp._install_build(PPB, Spec)
    tp._install_start(Spec)

    drafter = SimpleNamespace(context_copy=True)
    warm = PPB([[7, 8]], apc_meta=[{"full_input_ids": [1, 2, 7, 8]}],
               draft_model=drafter, draft_kind="mtp")
    sb = warm.generate()
    assert sb._start_rounds() == "started"
    assert sb.prompt_cache[0]._kq_copy_prompt == [1, 2, 7, 8]
    assert spec._pop_copy_prompt(sb.prompt_cache) == [1, 2, 7, 8]
    assert not hasattr(sb.prompt_cache[0], "_kq_copy_prompt")

    pair = PPB([[1, 2], [3, 4]], draft_model=drafter, draft_kind="mtp")
    sb = pair.generate()
    sb._start_rounds()
    assert not hasattr(sb.prompt_cache[0], "_kq_copy_prompt")

    off = PPB([[1, 2]], draft_model=SimpleNamespace(), draft_kind="mtp")
    assert not hasattr(off, "_kq_copy_prompts")
