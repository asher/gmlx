#!/usr/bin/env python3
"""Prefill feeder (``gmlx.feeder``): router-aware partial staging -
sparse slot fill at original expert indices, drain-on-new-pass interplay
with whole-layer staging, weight-swap restore - and the offload wrapper's
branch ordering between arena, partial and whole-layer paths. Pure CPU:
``arena_alloc`` is faked with numpy buffers and the "GGUF" is a temp file
of known bytes (fixture shared with test_decode_feeder)."""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import mlx.core as mx
from mlx_lm.models.switch_layers import SwitchGLU

from gmlx.loader import install_expert_streaming

from test_decode_feeder import (
    _KINDS,
    _STRIDE,
    _expert_bytes,
    _fake_arena_alloc,
    _holder_model,
    _make_fixture,
)


def _make_prefill_feeder(monkeypatch, tmp_path, n_layers=2):
    import mlx_kquant as kq
    from gmlx.feeder import PrefillFeeder

    monkeypatch.setattr(kq, "arena_alloc", _fake_arena_alloc, raising=False)
    offsets, modules = _make_fixture(tmp_path, n_layers)
    # PrefillFeeder's offsets tuples are (path, off, nbytes, n_exp, kind)
    return PrefillFeeder(offsets, modules), modules


def _slot_expert(feeder, li, kind, e):
    view = feeder._views[(li, feeder._slot_of[li])][kind]
    return bytes(view[e].reshape(-1))


def test_gapped_coverage_alternates_slots(monkeypatch, tmp_path):
    """Covered layers with an interval (e.g. every other MoE layer) must
    alternate ring slots by covered position, not absolute layer parity:
    same-parity neighbors would otherwise share a slot and the pipelined
    staging of the next layer would overwrite the one in use."""
    import mlx_kquant as kq
    from gmlx.feeder import PrefillFeeder

    monkeypatch.setattr(kq, "arena_alloc", _fake_arena_alloc, raising=False)
    offsets, modules = _make_fixture(tmp_path, 5)
    for li in (1, 3):
        del offsets[li]
        del modules[li]
    feeder = PrefillFeeder(offsets, modules)
    assert feeder._slot_of[0] != feeder._slot_of[2]
    assert feeder._slot_of[2] != feeder._slot_of[4]
    with feeder.prefill_call(modules[2][0], 2):
        # prefill_call pipelines staging of the next covered layer (4);
        # layer 2's staged bytes must survive it.
        assert feeder._ready[4].wait(5)
        for kind in _KINDS:
            for e in (0, 3):
                assert _slot_expert(feeder, 2, kind, e) == _expert_bytes(2, kind, e)
    assert feeder._error is None


def test_partial_call_stages_only_routed_slices(monkeypatch, tmp_path):
    feeder, modules = _make_prefill_feeder(monkeypatch, tmp_path)
    mod = modules[0][0]
    orig = {k: getattr(mod, f"{k}_proj").weight for k in _KINDS}
    with feeder.prefill_partial_call(mod, 0, [1, 3]):
        for kind in _KINDS:
            assert getattr(mod, f"{kind}_proj").weight is not orig[kind]
            for e in (1, 3):
                assert _slot_expert(feeder, 0, kind, e) == _expert_bytes(0, kind, e)
            for e in (0, 2):  # unrouted slices stay unstaged
                assert _slot_expert(feeder, 0, kind, e) == b"\x00" * _STRIDE[kind]
    for kind in _KINDS:
        assert getattr(mod, f"{kind}_proj").weight is orig[kind]


def test_partial_call_after_whole_pass_drains_and_restages(monkeypatch, tmp_path):
    feeder, modules = _make_prefill_feeder(monkeypatch, tmp_path)
    with feeder.prefill_call(modules[0][0], 0):
        for e in range(4):  # whole-layer staging filled every slice
            assert _slot_expert(feeder, 0, "gate", e) == _expert_bytes(0, "gate", e)
    # Same layer again = new pass (short final chunk after a big one):
    # partial staging reuses the slot; its routed slices are freshly read.
    with feeder.prefill_partial_call(modules[0][0], 0, [2]):
        assert _slot_expert(feeder, 0, "gate", 2) == _expert_bytes(0, "gate", 2)
    assert feeder._error is None


def test_wrapper_partial_branch_and_ordering(monkeypatch):
    """32..64-token calls with a prefill feeder take the partial path with
    the routed unique ids; >64-token calls take the whole-layer path; when a
    decode feeder is present and fits, it wins over partial staging."""
    mx.random.seed(13)
    glu = SwitchGLU(16, 32, 4)
    mx.eval(glu.parameters())
    model = _holder_model(glu)
    model.parameters = lambda: {"glu": glu.parameters()}
    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": 1024}
    )
    real_set_wired = mx.set_wired_limit
    try:
        install_expert_streaming(model)

        class _FakeFdr:
            partial_calls = []
            whole_calls = []

            def covers(self, li):
                return True

            @contextmanager
            def prefill_partial_call(self, module, li, ids):
                self.partial_calls.append((li, list(ids)))
                yield

            @contextmanager
            def prefill_call(self, module, li):
                self.whole_calls.append(li)
                yield

        fdr = _FakeFdr()
        object.__setattr__(glu, "_kq_feeder", fdr)
        object.__setattr__(glu, "_kq_li", 3)

        x40 = mx.random.normal((1, 40, 16))
        i40 = mx.concatenate(
            [mx.zeros((1, 40, 1), dtype=mx.uint32),
             mx.full((1, 40, 1), 2, dtype=mx.uint32)], axis=-1)
        ref40 = mx.array(glu(x40, i40))
        mx.eval(ref40)
        assert fdr.partial_calls == [(3, [0, 2])]
        assert fdr.whole_calls == []

        x80 = mx.random.normal((1, 80, 16))
        i80 = mx.zeros((1, 80, 2), dtype=mx.uint32)
        mx.eval(glu(x80, i80))
        assert fdr.partial_calls == [(3, [0, 2])]
        assert fdr.whole_calls == [3]

        class _FakeDF:
            stage_calls = []
            fits = True

            def covers(self, li):
                return True

            def stage(self, li, ids):
                self.stage_calls.append(li)
                return ids.astype(np.uint32) if self.fits else None

            def wedged_at(self, li):
                return False

            def has_dead(self, li):
                return False

            def redirect_dead(self, li, ids):
                return ids

            @contextmanager
            def swapped(self, li):
                yield

        df = _FakeDF()
        object.__setattr__(glu, "_kq_decode_feeder", df)
        out = mx.array(glu(x40, i40))
        mx.eval(out)
        assert mx.allclose(ref40, out, atol=1e-6, rtol=1e-6)
        assert df.stage_calls == [3]  # arena won
        assert fdr.partial_calls == [(3, [0, 2])]  # partial not re-entered

        df.fits = False  # arena overflow falls through to partial staging
        mx.eval(glu(x40, i40))
        assert df.stage_calls == [3, 3]
        assert fdr.partial_calls == [(3, [0, 2]), (3, [0, 2])]
    finally:
        mx.set_wired_limit = real_set_wired
