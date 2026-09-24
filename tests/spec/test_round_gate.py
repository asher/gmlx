"""Round gate on a tiny random qwen3_5 target with a DFlash 2 drafter.

A round whose lattice estimate falls under the threshold runs as one plain
target step. The verify graph is built before the estimate is read, over a
snapshot of the cache state, so a gated round drops the graph unevaluated
and restores the cache. Whatever the gate decides, the engine must emit the
target's own greedy chain.
"""

import numpy as np
import mlx.core as mx
import pytest

from mlx_vlm.models.cache import KVCache

import gmlx.spec.dflash_drafter as dd
import gmlx.spec.speculative as sp
from gmlx.spec.dflash_drafter import greedy_path, path_estimate
from gmlx.upstream.gdn_patches import _patch_gated_delta_tiled_v

from test_dflash2_drafter import (
    BLOCK,
    CAPTURE,
    N_GEN,
    TOP_K,
    _drafter,
    _engine_reference,
    _engine_walk,
    _target,
    _verify,
)

PROMPT = [[1, 2, 3, 4, 5]]


@pytest.fixture(scope="module", autouse=True)
def _tiled_v():
    # A gated round runs the plain forward, whose scan takes mlx-lm's
    # gated_delta, and the verify rounds take the tiled ops. Every qwen3.5
    # GGUF load tiles mlx-lm's K->V map too, and the two routes agree only
    # under it.
    _patch_gated_delta_tiled_v()


def _log_softmax(v):
    v = np.asarray(v, dtype=np.float64)
    return v - v.max() - np.log(np.exp(v - v.max()).sum())


def test_path_estimate_matches_a_numpy_transcription():
    mx.random.seed(3)
    L = BLOCK - 1
    first = 3 * mx.random.normal((TOP_K,))
    edges = 3 * mx.random.normal((L - 1, TOP_K, TOP_K))
    sel = greedy_path(first, edges)
    got = path_estimate(first, edges, sel).item()
    f, e, s = np.array(first), np.array(edges), np.array(sel)
    lp = [_log_softmax(f)[s[0]]]
    lp += [_log_softmax(e[p][s[p]])[s[p + 1]] for p in range(L - 1)]
    ref = 1.0 + np.exp(np.cumsum(lp)).sum()
    assert abs(got - ref) < 1e-4
    assert 1.0 < got < L + 1


def test_a_one_position_block_estimates_from_its_first_row():
    first = mx.array([2.0, 0.0, -1.0, 0.5])
    edges = mx.zeros((0, 4, 4))
    sel = greedy_path(first, edges)
    ref = 1.0 + np.exp(_log_softmax(np.array(first))[0])
    assert sel.tolist() == [0]
    assert abs(path_estimate(first, edges, sel).item() - ref) < 1e-5


def _estimates(monkeypatch, drafter):
    """Record every round estimate the engine reads."""
    seen = []
    orig = type(drafter).draft_block

    def draft_block(self, *a, **kw):
        out = orig(self, *a, **kw)
        if self.round_estimate is not None:
            seen.append(self.round_estimate.item())
        return out

    monkeypatch.setattr(type(drafter), "draft_block", draft_block)
    return seen


def test_every_round_gated_emits_the_greedy_chain(monkeypatch):
    monkeypatch.setattr(sp, "_ROUND_GATE", 1e9)
    lm = _target()
    prompt = mx.array(PROMPT)
    ref = _engine_reference(lm, prompt, N_GEN)
    drafter = _drafter(lm)
    got = _engine_walk(lm, drafter, prompt, N_GEN)
    assert got[:N_GEN] == ref
    assert drafter.gated_rounds == N_GEN - 1
    assert drafter.accept_lens == []


def test_a_mixed_gate_emits_the_greedy_chain(monkeypatch):
    lm = _target()
    prompt = mx.array(PROMPT)
    ref = _engine_reference(lm, prompt, N_GEN)
    monkeypatch.setattr(sp, "_ROUND_GATE", 1e9)
    probe = _drafter(lm)
    seen = _estimates(monkeypatch, probe)
    _engine_walk(lm, probe, prompt, N_GEN)
    # A threshold between the lowest and highest estimate of the all-gated
    # run gates some rounds and verifies others.
    gate = float(np.median(seen))
    assert min(seen) < gate < max(seen)
    monkeypatch.setattr(sp, "_ROUND_GATE", gate)
    drafter = _drafter(lm)
    got = _engine_walk(lm, drafter, prompt, N_GEN)
    assert got[:N_GEN] == ref
    assert drafter.gated_rounds > 0 and drafter.accept_lens


def test_the_gate_off_builds_no_estimate(monkeypatch):
    monkeypatch.setattr(sp, "_ROUND_GATE", 0.0)
    monkeypatch.setattr(sp, "_ROUND_GATE_AUTO", False)
    built = []

    def estimate(*a):
        built.append(1)
        return path_estimate(*a)

    monkeypatch.setattr(dd, "path_estimate", estimate)
    lm = _target()
    drafter = _drafter(lm)
    _engine_walk(lm, drafter, mx.array(PROMPT), N_GEN)
    assert not built
    assert drafter.gated_rounds == 0 and drafter.accept_lens


def test_the_auto_threshold_is_the_cost_ratio_of_timed_rounds():
    cost = sp._GateCost()
    assert cost.threshold() == 0.0
    cost.add(False, 0.08)
    assert cost.threshold() == sp._GATE_PRIOR
    cost.add(True, 0.04)
    assert cost.threshold() == pytest.approx(2.0)
    cost.add(True, 0.09)
    assert cost.gated == pytest.approx(0.04 + sp._GATE_EMA * 0.05)
    assert cost.threshold() == pytest.approx(0.08 / cost.gated)


def test_the_auto_gate_emits_the_greedy_chain(monkeypatch):
    # The first round only times a verify; a prior this high gates the next
    # one, and the measured ratio decides the rest.
    monkeypatch.setattr(sp, "_ROUND_GATE_AUTO", True)
    monkeypatch.setattr(sp, "_GATE_PRIOR", 1e9)
    lm = _target()
    prompt = mx.array(PROMPT)
    ref = _engine_reference(lm, prompt, N_GEN)
    drafter = _drafter(lm)
    got = _engine_walk(lm, drafter, prompt, N_GEN)
    assert got[:N_GEN] == ref
    assert drafter.gated_rounds >= 1 and drafter.accept_lens


def test_a_stack_the_snapshot_cannot_copy_decides_before_the_build(monkeypatch):
    monkeypatch.setattr(sp, "_ROUND_GATE", 1e9)
    monkeypatch.setattr(sp, "_cache_snapshot", lambda cache: None)
    built = []
    orig = sp._mtp_verify_target

    def verify_target(*a, **kw):
        built.append(1)
        return orig(*a, **kw)

    monkeypatch.setattr(sp, "_mtp_verify_target", verify_target)
    lm = _target()
    prompt = mx.array(PROMPT)
    ref = _engine_reference(lm, prompt, N_GEN)
    got = _engine_walk(lm, _drafter(lm), prompt, N_GEN)
    assert got[:N_GEN] == ref
    assert not built


@pytest.mark.parametrize("head", [False, True])
def test_restore_undoes_a_verify_that_was_built_but_not_evaluated(head):
    """With ``head`` the verify's leading GDN layers run before the restore,
    as they do while a gated round waits for its estimate."""
    lm = _target()
    lm.set_dflash_capture(CAPTURE)
    prompt = mx.array(PROMPT)
    caches = []
    for _ in range(2):
        cache = lm.make_cache()
        out = lm(prompt, cache=cache, return_hidden=True)
        mx.eval(out.hidden_states[-1], [c.state for c in cache])
        caches.append(cache)
    live, ref = caches
    snap = sp._cache_snapshot(live)
    assert snap is not None
    _, gdn = _verify(lm, mx.array([[3, 1, 4, 1]]), live)
    if head:
        index = sp._leading_gdn_layers(lm) - 1
        assert index >= 1
        mx.eval(gdn[index][9])
    sp._cache_restore(snap)
    del snap
    step = mx.array([[3]])
    a = lm(step, cache=live, return_hidden=True)
    b = lm(step, cache=ref, return_hidden=True)
    assert mx.array_equal(a.logits, b.logits)
    assert mx.array_equal(a.hidden_states[-1], b.hidden_states[-1])


def test_snapshot_refuses_state_it_cannot_copy():
    class Storage:
        pass

    plain = KVCache()
    assert sp._cache_snapshot([plain]) is not None
    odd = KVCache()
    odd.storage = Storage()
    assert sp._cache_snapshot([plain, odd]) is None


def test_snapshot_handles_do_not_follow_a_slice_write():
    c = KVCache()
    c.keys = mx.zeros((1, 1, 4, 2))
    mx.eval(c.keys)
    snap = sp._cache_snapshot([c])
    c.keys[..., 1:2, :] = 1.0
    sp._cache_restore(snap)
    assert c.keys.sum().item() == 0.0

