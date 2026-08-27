"""cb_phase flip dedupe and engine install marking."""

from types import SimpleNamespace

import pytest


@pytest.fixture
def fake_kq(monkeypatch):
    calls = []
    fake = SimpleNamespace(
        set_cb_caps=lambda ops, mb: calls.append((ops, mb)) or (50, 50))
    import gmlx.serve.cb_phase as cb_phase
    monkeypatch.setitem(cb_phase._state, "kq", fake)
    monkeypatch.setitem(cb_phase._state, "phase", None)
    return calls


def test_flip_dedupes(fake_kq):
    import gmlx.serve.cb_phase as cb_phase

    cb_phase.flip("decode")
    cb_phase.flip("decode")
    cb_phase.flip("prefill")
    cb_phase.flip("decode")
    assert fake_kq == [cb_phase.COARSE, cb_phase.FINE, cb_phase.COARSE]


def test_flip_noop_without_kq(monkeypatch):
    import gmlx.serve.cb_phase as cb_phase

    monkeypatch.setitem(cb_phase._state, "kq", False)
    monkeypatch.setitem(cb_phase._state, "phase", None)
    cb_phase.flip("decode")  # must not raise


def test_install_disabled_by_env(monkeypatch, fake_kq):
    import gmlx.serve.cb_phase as cb_phase

    monkeypatch.setenv("GMLX_CB_PHASE", "0")
    assert cb_phase.install_cb_phase_flips() is False


def test_install_wraps_engine_once(monkeypatch, fake_kq):
    pytest.importorskip("mlx_vlm")
    from mlx_vlm.generate import ar

    import gmlx.serve.cb_phase as cb_phase

    saved = (ar.GenerationBatch._step, ar.SpeculativeGenerationBatch.next,
             ar.PromptProcessingBatch.prompt_step,
             ar.PromptProcessingBatch.generate)
    try:
        assert cb_phase.install_cb_phase_flips() is True
        assert ar.GenerationBatch._step._gmlx_cb_phase
        assert ar.SpeculativeGenerationBatch.next._gmlx_cb_phase
        assert ar.PromptProcessingBatch.prompt_step._gmlx_cb_phase
        # short prompts skip prompt_step; generate() is their only prefill
        assert ar.PromptProcessingBatch.generate._gmlx_cb_phase
        first = ar.GenerationBatch._step
        assert cb_phase.install_cb_phase_flips() is True
        assert ar.GenerationBatch._step is first
    finally:
        (ar.GenerationBatch._step, ar.SpeculativeGenerationBatch.next,
         ar.PromptProcessingBatch.prompt_step,
         ar.PromptProcessingBatch.generate) = saved


def test_phase_invariant_across_request_sequences(monkeypatch, fake_kq):
    """Every engine entry point sees its own phase at entry, in the call
    orders the scheduler actually produces. The short-prompt case is the
    regression: a prompt at or under the chunk size skips prompt_step, so
    generate() is its entire prefill and must flip fine itself."""
    pytest.importorskip("mlx_vlm")
    from mlx_vlm.generate import ar

    import gmlx.serve.cb_phase as cb_phase

    seen = []

    def rec(name):
        def stub(self, *args, **kwargs):
            seen.append((name, cb_phase._state["phase"]))
        return stub

    monkeypatch.setattr(ar.GenerationBatch, "_step", rec("step"))
    monkeypatch.setattr(ar.SpeculativeGenerationBatch, "next", rec("next"))
    monkeypatch.setattr(ar.PromptProcessingBatch, "prompt_step",
                        rec("prompt_step"))
    monkeypatch.setattr(ar.PromptProcessingBatch, "generate", rec("generate"))
    assert cb_phase.install_cb_phase_flips() is True

    obj = object.__new__(ar.PromptProcessingBatch)
    gen = object.__new__(ar.GenerationBatch)

    # chunked prompt: prompt_step chunks, generate finishes, decode follows
    ar.PromptProcessingBatch.prompt_step(obj)
    ar.PromptProcessingBatch.generate(obj)
    ar.GenerationBatch._step(gen)
    # short prompt after a decode: generate() is the only prefill call
    ar.PromptProcessingBatch.generate(obj)
    ar.GenerationBatch._step(gen)

    assert seen == [
        ("prompt_step", "prefill"),
        ("generate", "prefill"),
        ("step", "decode"),
        ("generate", "prefill"),
        ("step", "decode"),
    ]
