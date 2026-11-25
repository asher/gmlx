"""GPU row cut: the engine verifies the whole block under the target's
verify_row_keep and emits only what the kept rows decide. A target whose
rows past the keep are garbage still yields the greedy chain."""

import contextlib
import itertools

import mlx.core as mx
import pytest

import gmlx.spec.speculative as sp
from gmlx.spec.dflash_drafter import DFlash2Drafter

from test_dflash2_drafter import (
    BLOCK,
    N_GEN,
    _config,
    _engine_reference,
    _engine_walk,
    _target,
    _tcfg,
)

PROMPT = [[1, 2, 3, 4, 5]]


class _KeepDrafter(DFlash2Drafter):
    """Drafts the target's own greedy chain (every valid row accepts) and
    leaves a scripted keep count with each block, on the plain round path
    (no lazy bonus, no deferred cut)."""

    supports_lazy_bonus = False
    supports_deferred_row_cut = False

    def draft_block(self, *a, **kw):
        n = int(super().draft_block(*a, **kw).shape[1])
        pos = sum(int(x) + 1 for x in self.accept_lens)
        drafts = (self.ref[pos + 1:pos + 1 + n] + [0] * n)[:n]
        object.__setattr__(self, "_keep", mx.array(next(self.script)))
        return mx.array([drafts], dtype=mx.int32)

    def take_row_keep(self):
        keep, self._keep = self._keep, None
        return keep


def _drafter(lm, script, ref):
    d = _KeepDrafter(_config(_tcfg()))
    mx.eval(d.parameters())
    d.reset(lm)
    object.__setattr__(d, "ref", ref)
    object.__setattr__(d, "script", itertools.cycle(script))
    object.__setattr__(d, "_keep", None)
    return d


def _poisoned(lm, builds):
    """Give ``lm`` a verify_row_keep whose verify returns noise for the
    rows past the keep."""
    held = [None]
    real = lm.speculative_verify_hidden

    @contextlib.contextmanager
    def verify_row_keep(keep):
        held[0] = keep
        builds.append(1)
        try:
            yield
        finally:
            held[0] = None

    def speculative_verify_hidden(ids, cache, *a, **kw):
        hidden, *rest = real(ids, cache, *a, **kw)
        if held[0] is not None:
            rows = mx.arange(hidden.shape[1])[None, :, None]
            noise = mx.random.normal(hidden.shape, key=mx.random.key(0)) * 50
            hidden = mx.where(rows > held[0], noise.astype(hidden.dtype), hidden)
        return (hidden, *rest)

    object.__setattr__(lm, "verify_row_keep", verify_row_keep)
    object.__setattr__(lm, "speculative_verify_hidden", speculative_verify_hidden)


@pytest.fixture(autouse=True)
def _plain_rounds(monkeypatch):
    monkeypatch.setattr(sp, "_ROUND_GATE", 0.0)
    monkeypatch.setattr(sp, "_ROUND_GATE_AUTO", False)
    monkeypatch.setattr(sp, "_ROUND_PROFILE", False)
    monkeypatch.setattr(sp, "_DRAFT_EARLY", False)


@pytest.mark.parametrize("script", [(0, 1, 2, BLOCK - 1), (0,), (1, 2)])
def test_a_cut_round_emits_the_greedy_chain(script):
    lm = _target()
    prompt = mx.array(PROMPT)
    ref = _engine_reference(lm, prompt, N_GEN)
    builds: list = []
    _poisoned(lm, builds)
    drafter = _drafter(lm, script, ref)
    got = _engine_walk(lm, drafter, prompt, N_GEN)
    assert got[:N_GEN] == ref
    assert builds
    # Every kept draft is the target's own, so a round accepts its keep.
    assert list(drafter.accept_lens[:len(script)]) == list(script)


def test_without_the_target_hook_the_round_is_whole():
    lm = _target()
    prompt = mx.array(PROMPT)
    ref = _engine_reference(lm, prompt, N_GEN)
    drafter = _drafter(lm, (0,), ref)
    assert _engine_walk(lm, drafter, prompt, N_GEN)[:N_GEN] == ref
    assert drafter.accept_lens[0] == BLOCK - 1
