#!/usr/bin/env python3
"""The arena token split rule. A multi-token call that the prefill ring can
also serve takes the split only when its routed union is a small share of
the layer's experts. Otherwise it takes the ring, which reads the whole
layer once. A stock SwitchGLU behind stub feeders. The streamed paths run
on the GPU stream, so the reference runs there too, through the body the
path runs. A reference on the default device, which may be the CPU, or
through the sorted GEMM, which may run fp32 at TF32, differs by far more
than the tolerance."""
from __future__ import annotations

import contextlib

import mlx.core as mx
import numpy as np
import pytest

from gmlx._vendor.mlx_lm.models.switch_layers import SwitchGLU
from gmlx.load.loader import _split_beats_ring
from gmlx.load.modules import glu_unsorted_call
from gmlx.stream.expert_streaming import install_expert_streaming

E, TOPK = 16, 2


class _Arena:
    """Decode feeder stub with identity slots (an arena of all experts,
    which refuses calls over ``slots`` distinct experts)."""

    def __init__(self, slots):
        self.slots = slots
        self.leaves = []
        self.rule = []

    def covers(self, li):
        return True

    def can_stage_smaller(self, li):
        return True

    def slot_count(self, li):
        return self.slots

    def wedged_at(self, li):
        return False

    def repay_at(self, li, ids, ring_ok=False):
        return False

    def ensure_wired(self):
        pass

    def note_split_rule(self, split, ahead=False):
        self.rule.append((split, ahead))

    def stage(self, li, ids):
        if np.unique(ids.reshape(-1)).size > self.slots:
            return None
        self.leaves.append(ids.size // ids.shape[-1])
        return ids.astype(np.uint32)

    @contextlib.contextmanager
    def swapped(self, li):
        yield


class _Ring:
    """Prefill feeder stub: the whole-layer and the partial ring calls run
    against the resident weights."""

    def __init__(self, live=True, ahead=()):
        self.live = live
        self.ahead = set(ahead)
        self.whole = 0
        self.partial = 0

    def covers(self, li):
        return True

    def ring_live(self):
        return self.live

    def staged_ahead(self, li):
        return li in self.ahead

    @contextlib.contextmanager
    def prefill_call(self, module, li, ids=None):
        self.whole += 1
        yield

    @contextlib.contextmanager
    def prefill_partial_call(self, module, li, ids, routing=None):
        self.partial += 1
        yield


class _Owed(_Arena):
    """An arena that owes a ring lend and repays it at this layer."""

    def __init__(self, slots):
        super().__init__(slots)
        self.events = []

    def repay_at(self, li, ids, ring_ok=False):
        self.events.append(("repay_at", ids.size, ring_ok))
        return True

    def ensure_wired(self):
        self.events.append("wired")

    def stage(self, li, ids):
        self.events.append("stage")
        return super().stage(li, ids)


def _glu(arena, ring):
    mx.random.seed(9)
    glu = SwitchGLU(16, 32, E)
    mx.eval(glu.parameters())

    class _L:
        pass

    layer = _L()
    layer.modules = lambda: [glu]
    model = _L()
    model.layers = [layer]
    assert install_expert_streaming(model)[0] == 1
    object.__setattr__(glu, "_kq_cpu_only", True)
    object.__setattr__(glu, "_kq_li", 3)
    object.__setattr__(glu, "_kq_decode_feeder", arena)
    object.__setattr__(glu, "_kq_feeder", ring)
    return glu


def _inds(t, n_experts):
    """``t`` tokens over the first ``n_experts`` experts, all of them used."""
    ids = (np.arange(t * TOPK) % n_experts).reshape(1, t, TOPK)
    return mx.array(ids.astype(np.uint32))


def _check(glu, t, n_experts, rows):
    """``rows``: the arena serves the call with the unsorted per-row body.
    Otherwise the ring runs the stock body."""
    x = mx.random.normal((1, t, 16))
    inds = _inds(t, n_experts)
    y = glu(x, inds)
    with mx.stream(mx.gpu):
        if rows:
            ref = glu_unsorted_call(glu, x, inds)
        else:
            ref = SwitchGLU.__call__(glu, x, inds)
    mx.eval(ref, y)
    assert mx.allclose(ref, y, atol=1e-5, rtol=1e-5)


def test_rule_thresholds(monkeypatch):
    monkeypatch.delenv("GMLX_ARENA_SPLIT_MAX_SHARE", raising=False)
    assert _split_beats_ring(2, 256)
    assert _split_beats_ring(128, 256)
    assert not _split_beats_ring(129, 256)
    assert not _split_beats_ring(250, 256)
    monkeypatch.setenv("GMLX_ARENA_SPLIT_MAX_SHARE", "0.25")
    assert not _split_beats_ring(128, 256)
    assert _split_beats_ring(64, 256)
    monkeypatch.setenv("GMLX_ARENA_SPLIT_MAX_SHARE", "1")
    assert _split_beats_ring(256, 256)
    monkeypatch.setenv("GMLX_ARENA_SPLIT_MAX_SHARE", "0")
    assert not _split_beats_ring(1, 256)


def test_small_union_splits():
    arena, ring = _Arena(slots=4), _Ring()
    _check(_glu(arena, ring), 128, 4, rows=True)
    assert arena.rule == [(True, False)]
    assert arena.leaves == [64, 64]
    assert ring.whole == 0


def test_near_full_union_takes_the_ring():
    arena, ring = _Arena(slots=4), _Ring()
    _check(_glu(arena, ring), 128, E, rows=False)
    assert arena.rule == [(False, False)]
    assert arena.leaves == []
    assert ring.whole == 1


def test_layer_staged_ahead_takes_the_ring():
    arena, ring = _Arena(slots=4), _Ring(ahead={3})
    _check(_glu(arena, ring), 128, 4, rows=False)
    assert arena.rule == [(False, True)]
    assert ring.whole == 1


def test_small_call_takes_the_partial_ring_while_live():
    arena, ring = _Arena(slots=4), _Ring(live=True)
    _check(_glu(arena, ring), 40, E, rows=False)
    assert arena.rule == [(False, False)]
    assert ring.partial == 1


def test_small_call_splits_once_the_ring_is_released():
    arena, ring = _Arena(slots=4), _Ring(live=False)
    _check(_glu(arena, ring), 40, E, rows=True)
    assert arena.rule == []
    assert arena.leaves and ring.partial == 0


@pytest.mark.parametrize("t", [2, 16, 31])
def test_verify_widths_keep_the_split(t):
    arena, ring = _Arena(slots=4), _Ring(live=True)
    _check(_glu(arena, ring), t, E, rows=True)
    assert arena.rule == []
    assert ring.whole == ring.partial == 0


def test_share_one_restores_the_split(monkeypatch):
    monkeypatch.setenv("GMLX_ARENA_SPLIT_MAX_SHARE", "1")
    arena, ring = _Arena(slots=4), _Ring()
    _check(_glu(arena, ring), 128, E, rows=True)
    assert arena.rule == [(True, False)]
    assert ring.whole == 0


@pytest.mark.parametrize("t", [8, 40])
def test_wide_call_repays_a_lend_before_it_stages(t):
    """A batched or verify decode step is never one token wide. The arena
    decides from the call's routing, and whether the call is wide enough
    to take the ring, if it repays a ring lend. The repay runs before the
    layer stages."""
    arena, ring = _Owed(slots=4), _Ring(live=False)
    _check(_glu(arena, ring), t, 4, rows=True)
    assert arena.events == [("repay_at", t * TOPK, t >= 32), "wired", "stage"]
