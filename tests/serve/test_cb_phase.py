"""cb_phase flip dedupe, per-model decode caps and engine install marking."""

import importlib
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
    monkeypatch.setitem(cb_phase._state, "caps", None)
    monkeypatch.setitem(cb_phase._state, "lifted", False)
    return calls


def _model(streamed):
    """A model the streaming install marked, in RAM or streamed."""
    return SimpleNamespace(_kq_streaming=streamed)


def _keep_step_targets(monkeypatch):
    """Restore at teardown the generate_step functions an install wraps."""
    for modname, name in (("gmlx._vendor.mlx_lm.generate", "generate_step"),
                          ("gmlx._vendor.mlx_lm.generate",
                           "speculative_generate_step"),
                          ("gmlx._vendor.mlx_vlm.generate.ar", "generate_step"),
                          ("gmlx.spec.speculative", "stream_speculative")):
        try:
            mod = importlib.import_module(modname)
        except ImportError:
            continue
        monkeypatch.setattr(mod, name, getattr(mod, name))


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
    pytest.importorskip("gmlx._vendor.mlx_vlm")
    from gmlx._vendor.mlx_vlm.generate import ar

    import gmlx.serve.cb_phase as cb_phase

    from gmlx.upstream.patching import root

    saved = (ar.GenerationBatch._step, ar.SpeculativeGenerationBatch.next,
             ar.PromptProcessingBatch.prompt_step,
             ar.PromptProcessingBatch.generate)
    try:
        # The copy's own methods, without what an earlier install left.
        (ar.GenerationBatch._step, ar.SpeculativeGenerationBatch.next,
         ar.PromptProcessingBatch.prompt_step,
         ar.PromptProcessingBatch.generate) = map(root, saved)
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
    pytest.importorskip("gmlx._vendor.mlx_vlm")
    from gmlx._vendor.mlx_vlm.generate import ar

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


def test_phased_steps_flips_at_first_token(fake_kq):
    import gmlx.serve.cb_phase as cb_phase

    def steps():
        yield "t0"
        yield "t1"
        yield "t2"

    gen = cb_phase.phased_steps(steps())
    assert fake_kq == []  # nothing until iteration starts
    assert next(gen) == "t0"
    assert fake_kq == [cb_phase.FINE, cb_phase.COARSE]
    assert list(gen) == ["t1", "t2"]
    assert fake_kq == [cb_phase.FINE, cb_phase.COARSE]
    # the next prompt's prefill flips fine again
    list(cb_phase.phased_steps(steps()))
    assert fake_kq == [cb_phase.FINE, cb_phase.COARSE] * 2


def test_phased_steps_close_closes_wrapped_generator(fake_kq):
    # A consumer that stops mid-stream closes the wrapper, and the wrapped
    # generator's cleanup (a speculative loop's cache rollback) runs then.
    import gmlx.serve.cb_phase as cb_phase

    events = []

    def steps():
        try:
            yield "t0"
            yield "t1"
        finally:
            events.append("inner closed")

    gen = cb_phase.phased_steps(steps())
    assert next(gen) == "t0"
    gen.close()
    assert events == ["inner closed"]


def test_install_steps_wraps_generate_step_once(monkeypatch, fake_kq):
    lmgen = importlib.import_module("gmlx._vendor.mlx_lm.generate")
    spec = importlib.import_module("gmlx.spec.speculative")

    import gmlx.serve.cb_phase as cb_phase

    monkeypatch.delenv("MLX_MAX_OPS_PER_BUFFER", raising=False)
    saved = (lmgen.generate_step, lmgen.speculative_generate_step)
    monkeypatch.setattr(spec, "stream_speculative", spec.stream_speculative)
    try:
        assert cb_phase.install_cb_phase_steps() is True
        assert lmgen.generate_step._gmlx_cb_phase
        assert lmgen.speculative_generate_step._gmlx_cb_phase
        # gmlx run's own speculative loop decodes on coarse caps too
        assert spec.stream_speculative._gmlx_cb_phase
        first = lmgen.generate_step
        assert cb_phase.install_cb_phase_steps() is True
        assert lmgen.generate_step is first
        # the wrapper drives the original through phased_steps
        monkeypatch.setattr(lmgen.generate_step, "__wrapped__",
                            lambda *a, **k: iter(["a", "b"]))
    finally:
        lmgen.generate_step, lmgen.speculative_generate_step = saved


def test_install_steps_skips_a_user_pin_and_the_env_switch(monkeypatch,
                                                          fake_kq):
    """A user's MLX_MAX_OPS_PER_BUFFER keeps its caps for the whole run, as
    does GMLX_CB_PHASE=0. The env the streaming lift wrote is no pin."""
    from gmlx.commands.cli import _lift_stream_cb_caps

    lmgen = importlib.import_module("gmlx._vendor.mlx_lm.generate")

    import gmlx.serve.cb_phase as cb_phase

    _keep_step_targets(monkeypatch)
    saved = lmgen.generate_step
    monkeypatch.delenv("MLX_MAX_MB_PER_BUFFER", raising=False)
    monkeypatch.setenv("MLX_MAX_OPS_PER_BUFFER", "100")
    _lift_stream_cb_caps(["run", "m.gguf", "--stream-experts"])
    assert cb_phase.install_cb_phase_steps(streaming=True) is False
    monkeypatch.delenv("MLX_MAX_OPS_PER_BUFFER")
    monkeypatch.setenv("GMLX_CB_PHASE", "0")
    assert cb_phase.install_cb_phase_steps() is False
    assert lmgen.generate_step is saved
    monkeypatch.delenv("GMLX_CB_PHASE")
    monkeypatch.delenv("MLX_MAX_MB_PER_BUFFER")
    _lift_stream_cb_caps(["run", "m.gguf", "--stream-experts"])
    assert cb_phase.install_cb_phase_steps(streaming=True) is True
    assert lmgen.generate_step._gmlx_cb_phase


@pytest.mark.parametrize("flags, streaming",
                         [([], False), (["--stream-experts"], True)])
def test_chat_installs_the_phased_steps_as_run_does(monkeypatch, tmp_path,
                                                    flags, streaming):
    """gmlx chat sets the caps per phase as run does, so a model decodes at
    the same speed in both, and a streamed chat prefills on the lift."""
    import gmlx.commands.cli as cli
    import gmlx.serve.cb_phase as cb_phase
    from gmlx.tui import chat

    seen = []
    monkeypatch.setattr(cli, "maybe_load_from_config", lambda *a: None)
    monkeypatch.setattr(cli, "preload_wire_check", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_ensure_stream_cb_caps", lambda args: None)
    monkeypatch.setattr(cb_phase, "install_cb_phase_steps",
                        lambda streaming=False: seen.append(streaming) or True)
    missing = str(tmp_path / "missing.gguf")
    assert chat.cmd_chat([missing, *flags]) == 2
    assert seen == [streaming]


def test_decode_caps_follow_the_streaming_verdict():
    """A streamed decode syncs on the host at every layer, so it decodes on
    the streamed caps; a model in RAM keeps the coarse decode caps."""
    import gmlx.serve.cb_phase as cb_phase

    assert cb_phase.STREAMED_DECODE != cb_phase.COARSE
    assert cb_phase.decode_caps(_model(True)) == cb_phase.STREAMED_DECODE
    assert cb_phase.decode_caps(_model(False)) == cb_phase.COARSE
    assert cb_phase.decode_caps(None) == cb_phase.COARSE


def test_engine_steps_set_each_models_caps(monkeypatch, fake_kq):
    """One server, a model in RAM and a streamed one: each decode step sets
    its own model's caps, and a step that keeps the caps sets nothing."""
    pytest.importorskip("gmlx._vendor.mlx_vlm")
    from gmlx._vendor.mlx_vlm.generate import ar

    import gmlx.serve.cb_phase as cb_phase

    for cls, name in ((ar.GenerationBatch, "_step"),
                      (ar.SpeculativeGenerationBatch, "next"),
                      (ar.PromptProcessingBatch, "prompt_step"),
                      (ar.PromptProcessingBatch, "generate")):
        monkeypatch.setattr(cls, name, lambda self, *a, **k: None)
    assert cb_phase.install_cb_phase_flips() is True
    resident = object.__new__(ar.GenerationBatch)
    resident.model = _model(False)
    streamed = object.__new__(ar.GenerationBatch)
    streamed.model = _model(True)
    prompt = object.__new__(ar.PromptProcessingBatch)
    prompt.model = streamed.model
    for batch in (resident, resident, streamed, resident):
        ar.GenerationBatch._step(batch)
    ar.PromptProcessingBatch.generate(prompt)
    ar.GenerationBatch._step(streamed)
    wanted = [cb_phase.COARSE, cb_phase.COARSE, cb_phase.STREAMED_DECODE,
              cb_phase.COARSE, cb_phase.FINE, cb_phase.STREAMED_DECODE]
    assert fake_kq == [c for i, c in enumerate(wanted)
                       if i == 0 or c != wanted[i - 1]]


def test_a_streamed_run_prefills_on_the_lift_and_decodes_on_the_streamed_caps(
        monkeypatch, fake_kq):
    """gmlx run --stream-experts keeps the lift's coarse caps through the
    prefill and decodes on the streamed caps from the first token."""
    import mlx.nn as nn

    import gmlx.serve.cb_phase as cb_phase

    lmgen = importlib.import_module("gmlx._vendor.mlx_lm.generate")
    _keep_step_targets(monkeypatch)
    monkeypatch.delenv("MLX_MAX_OPS_PER_BUFFER", raising=False)
    monkeypatch.setattr(lmgen, "generate_step",
                        lambda prompt, model, **kwargs: iter(["t0", "t1"]))
    assert cb_phase.install_cb_phase_steps(streaming=True) is True
    model = nn.Module()
    model._kq_streaming = True
    assert list(lmgen.generate_step([1, 2], model)) == ["t0", "t1"]
    assert fake_kq == [cb_phase.COARSE, cb_phase.STREAMED_DECODE]


def test_a_placement_lift_goes_through_the_caps_record(monkeypatch, fake_kq):
    """A model that streams for want of memory lifts the live caps at its
    placement (a chat model switch, say). The lift goes through cb_phase,
    so that model's next decode step still sets its decode caps."""
    from gmlx.commands.cli import _lift_live_cb_caps

    import gmlx.serve.cb_phase as cb_phase

    monkeypatch.delenv("MLX_MAX_OPS_PER_BUFFER", raising=False)
    streamed = _model(True)
    cb_phase.flip("decode", streamed)
    _lift_live_cb_caps()
    cb_phase.flip("decode", streamed)
    assert fake_kq == [cb_phase.STREAMED_DECODE, cb_phase.COARSE,
                       cb_phase.STREAMED_DECODE]
