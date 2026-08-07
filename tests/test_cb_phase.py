"""cb_phase flip dedupe and engine install marking."""

from types import SimpleNamespace

import pytest


@pytest.fixture
def fake_kq(monkeypatch):
    calls = []
    fake = SimpleNamespace(
        set_cb_caps=lambda ops, mb: calls.append((ops, mb)) or (50, 50))
    from gmlx import cb_phase
    monkeypatch.setitem(cb_phase._state, "kq", fake)
    monkeypatch.setitem(cb_phase._state, "phase", None)
    return calls


def test_flip_dedupes(fake_kq):
    from gmlx import cb_phase

    cb_phase.flip("decode")
    cb_phase.flip("decode")
    cb_phase.flip("prefill")
    cb_phase.flip("decode")
    assert fake_kq == [cb_phase.COARSE, cb_phase.FINE, cb_phase.COARSE]


def test_flip_noop_without_kq(monkeypatch):
    from gmlx import cb_phase

    monkeypatch.setitem(cb_phase._state, "kq", False)
    monkeypatch.setitem(cb_phase._state, "phase", None)
    cb_phase.flip("decode")  # must not raise


def test_install_disabled_by_env(monkeypatch, fake_kq):
    from gmlx import cb_phase

    monkeypatch.setenv("GMLX_CB_PHASE", "0")
    assert cb_phase.install_cb_phase_flips() is False


def test_install_wraps_engine_once(monkeypatch, fake_kq):
    pytest.importorskip("mlx_vlm")
    from mlx_vlm.generate import ar

    from gmlx import cb_phase

    saved = (ar.GenerationBatch._step, ar.SpeculativeGenerationBatch.next,
             ar.PromptProcessingBatch.prompt_step)
    try:
        assert cb_phase.install_cb_phase_flips() is True
        assert ar.GenerationBatch._step._gmlx_cb_phase
        assert ar.SpeculativeGenerationBatch.next._gmlx_cb_phase
        assert ar.PromptProcessingBatch.prompt_step._gmlx_cb_phase
        first = ar.GenerationBatch._step
        assert cb_phase.install_cb_phase_flips() is True
        assert ar.GenerationBatch._step is first
    finally:
        (ar.GenerationBatch._step, ar.SpeculativeGenerationBatch.next,
         ar.PromptProcessingBatch.prompt_step) = saved
