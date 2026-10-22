"""cb_phase flip dedupe, per-model caps, the user's pin and engine install
marking."""

import importlib
import os
import threading
from types import SimpleNamespace

import pytest

_HOLDS = ("MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER", "GMLX_CB_PHASE")


@pytest.fixture
def fake_kq(monkeypatch):
    calls = []
    fake = SimpleNamespace(
        set_cb_caps=lambda ops, mb: calls.append((ops, mb)) or (50, 50))
    import gmlx.serve.cb_phase as cb_phase
    for k in _HOLDS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setitem(cb_phase._state, "kq", fake)
    monkeypatch.setitem(cb_phase._state, "phase", None)
    monkeypatch.setitem(cb_phase._state, "caps", None)
    monkeypatch.setitem(cb_phase._state, "lifted", False)
    return calls


@pytest.fixture(params=_HOLDS)
def held(request, monkeypatch, fake_kq):
    """The user's pin of either cap, or GMLX_CB_PHASE=0. Returns the key
    and the caps writes."""
    key = request.param
    monkeypatch.setenv(key, "0" if key == "GMLX_CB_PHASE" else "200")
    return key, fake_kq


def _model(streamed):
    """A model the streaming install marked, in RAM or streamed."""
    return SimpleNamespace(_kq_streaming=streamed)


@pytest.fixture
def distinct_prefill(monkeypatch):
    """STREAMED_PREFILL as caps no other phase sets. The shipped value is
    FINE, so a routing test sets its own to show which verdict a step
    read."""
    import gmlx.serve.cb_phase as cb_phase

    monkeypatch.setattr(cb_phase, "STREAMED_PREFILL", (100, 100))


def _stub_engine_steps(monkeypatch):
    """The engine's step entrypoints as no-ops, restored at teardown."""
    from gmlx._vendor.mlx_vlm.generate import ar

    for cls, name in ((ar.GenerationBatch, "_step"),
                      (ar.SpeculativeGenerationBatch, "next"),
                      (ar.PromptProcessingBatch, "prompt_step"),
                      (ar.PromptProcessingBatch, "generate")):
        monkeypatch.setattr(cls, name, lambda self, *a, **k: None)
    return ar


def _lm(ids, cache=None, n_to_process=0):
    """A language model forward for _chunked_prefill_cache."""


def _keep_step_targets(monkeypatch):
    """Restore at teardown the generate_step functions an install wraps."""
    for modname, name in (("gmlx._vendor.mlx_lm.generate", "generate_step"),
                          ("gmlx._vendor.mlx_lm.generate",
                           "speculative_generate_step"),
                          ("gmlx._vendor.mlx_vlm.generate.ar", "generate_step"),
                          ("gmlx._vendor.mlx_vlm.generate.dispatch",
                           "generate_step"),
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
    assert cb_phase.install_cb_phase_steps() is False
    monkeypatch.delenv("MLX_MAX_OPS_PER_BUFFER")
    monkeypatch.setenv("GMLX_CB_PHASE", "0")
    assert cb_phase.install_cb_phase_steps() is False
    assert lmgen.generate_step is saved
    monkeypatch.delenv("GMLX_CB_PHASE")
    monkeypatch.delenv("MLX_MAX_MB_PER_BUFFER")
    _lift_stream_cb_caps(["run", "m.gguf", "--stream-experts"])
    assert cb_phase.install_cb_phase_steps() is True
    assert lmgen.generate_step._gmlx_cb_phase


@pytest.mark.parametrize("flags", [[], ["--stream-experts"]])
def test_chat_installs_the_phased_steps_as_run_does(monkeypatch, tmp_path,
                                                    flags):
    """gmlx chat sets the caps per phase as run does, so a model decodes at
    the same speed in both, in RAM and streamed."""
    import gmlx.commands.cli as cli
    import gmlx.serve.cb_phase as cb_phase
    from gmlx.tui import chat

    seen = []
    monkeypatch.setattr(cli, "maybe_load_from_config", lambda *a: None)
    monkeypatch.setattr(cli, "preload_wire_check", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_ensure_stream_cb_caps", lambda args: None)
    monkeypatch.setattr(cb_phase, "install_cb_phase_steps",
                        lambda: seen.append("installed") or True)
    missing = str(tmp_path / "missing.gguf")
    assert chat.cmd_chat([missing, *flags]) == 2
    assert seen == ["installed"]


def test_decode_caps_follow_the_streaming_verdict():
    """A streamed decode syncs on the host at every layer, so it decodes on
    the streamed caps; a model in RAM keeps the coarse decode caps."""
    import gmlx.serve.cb_phase as cb_phase

    assert cb_phase.STREAMED_DECODE != cb_phase.COARSE
    assert cb_phase.decode_caps(_model(True)) == cb_phase.STREAMED_DECODE
    assert cb_phase.decode_caps(_model(False)) == cb_phase.COARSE
    assert cb_phase.decode_caps(None) == cb_phase.COARSE


def test_prefill_caps_follow_the_streaming_verdict(fake_kq, distinct_prefill):
    """A streamed model prefills on the streamed prefill caps and a model
    in RAM on the fine caps, in a chunked prefill as in a step."""
    import mlx.core as mx

    import gmlx.serve.cb_phase as cb_phase
    from gmlx.gen.generation import _chunked_prefill_cache

    assert cb_phase.prefill_caps(_model(True)) == cb_phase.STREAMED_PREFILL
    assert cb_phase.prefill_caps(_model(False)) == cb_phase.FINE
    assert cb_phase.prefill_caps(None) == cb_phase.FINE
    ids = mx.array([[1, 2, 3, 4]])
    _chunked_prefill_cache(_lm, ids, 2, cache=[], owner=_model(True))
    _chunked_prefill_cache(_lm, ids, 2, cache=[], owner=_model(False))
    assert fake_kq == [cb_phase.STREAMED_PREFILL, cb_phase.FINE]


def test_engine_steps_set_each_models_caps(monkeypatch, fake_kq,
                                           distinct_prefill):
    """One server, a model in RAM and a streamed one: each step sets its
    own model's caps for its phase, and a step that keeps the caps sets
    nothing."""
    pytest.importorskip("gmlx._vendor.mlx_vlm")
    import gmlx.serve.cb_phase as cb_phase

    ar = _stub_engine_steps(monkeypatch)
    assert cb_phase.install_cb_phase_flips() is True
    resident = object.__new__(ar.GenerationBatch)
    resident.model = _model(False)
    streamed = object.__new__(ar.GenerationBatch)
    streamed.model = _model(True)
    prompt = object.__new__(ar.PromptProcessingBatch)
    prompt.model = streamed.model
    prompt_ram = object.__new__(ar.PromptProcessingBatch)
    prompt_ram.model = resident.model
    for batch in (resident, resident, streamed, resident):
        ar.GenerationBatch._step(batch)
    ar.PromptProcessingBatch.generate(prompt_ram)
    ar.PromptProcessingBatch.prompt_step(prompt)
    ar.GenerationBatch._step(streamed)
    wanted = [cb_phase.COARSE, cb_phase.COARSE, cb_phase.STREAMED_DECODE,
              cb_phase.COARSE, cb_phase.FINE, cb_phase.STREAMED_PREFILL,
              cb_phase.STREAMED_DECODE]
    assert fake_kq == [c for i, c in enumerate(wanted)
                       if i == 0 or c != wanted[i - 1]]


def test_a_streamed_run_prefills_and_decodes_on_the_streamed_caps(
        monkeypatch, fake_kq, distinct_prefill):
    """gmlx run on a streamed model prefills on the streamed prefill caps
    and decodes on the streamed decode caps from the first token, with
    --stream-experts or with experts that stream for want of memory."""
    import mlx.nn as nn

    import gmlx.serve.cb_phase as cb_phase

    lmgen = importlib.import_module("gmlx._vendor.mlx_lm.generate")
    _keep_step_targets(monkeypatch)
    monkeypatch.setattr(lmgen, "generate_step",
                        lambda prompt, model, **kwargs: iter(["t0", "t1"]))
    assert cb_phase.install_cb_phase_steps() is True
    model = nn.Module()
    model._kq_streaming = True
    assert list(lmgen.generate_step([1, 2], model)) == ["t0", "t1"]
    assert fake_kq == [cb_phase.STREAMED_PREFILL, cb_phase.STREAMED_DECODE]


def test_a_streamed_prefill_runs_on_the_fine_caps(monkeypatch, fake_kq):
    """A streamed model prefills on (50, 50) in a serve prompt step, the
    last prompt columns, a run and a chunked prefill. A coarse buffer keeps
    a finished layer's buffers in flight while the next layer allocates:
    on DeepSeek V4.1 Flash served streamed, an 8k prompt on (400, 100000)
    left 5.4 GB of pages free."""
    pytest.importorskip("gmlx._vendor.mlx_vlm")
    import mlx.core as mx
    import mlx.nn as nn

    import gmlx.serve.cb_phase as cb_phase
    from gmlx.gen.generation import _chunked_prefill_cache

    ar = _stub_engine_steps(monkeypatch)
    lmgen = importlib.import_module("gmlx._vendor.mlx_lm.generate")
    _keep_step_targets(monkeypatch)
    monkeypatch.setattr(lmgen, "generate_step",
                        lambda prompt, model, **kwargs: iter(["t0"]))
    assert cb_phase.install_cb_phase_flips() is True
    assert cb_phase.install_cb_phase_steps() is True
    model = nn.Module()
    model._kq_streaming = True
    prompt = object.__new__(ar.PromptProcessingBatch)
    prompt.model = model
    ids = mx.array([[1, 2, 3, 4]])
    prefills = (
        lambda: ar.PromptProcessingBatch.prompt_step(prompt),
        lambda: ar.PromptProcessingBatch.generate(prompt),
        lambda: list(lmgen.generate_step([1, 2], model)),
        lambda: _chunked_prefill_cache(_lm, ids, 2, cache=[], owner=model))
    for prefill in prefills:
        # A model in RAM decodes in between, so each prefill writes its caps.
        cb_phase.flip("decode", _model(False))
        prefill()
    assert fake_kq == [(400, 100000), (50, 50)] * len(prefills)



def _caps_forward(seen):
    """A language model forward that records the caps each call runs on."""
    import gmlx.serve.cb_phase as cb_phase

    def forward(ids, cache=None, n_to_process=0, **kwargs):
        seen.append(cb_phase._state["caps"])

    forward.make_cache = lambda: []
    return forward


def _tokenizer():
    detok = SimpleNamespace(reset=lambda: None, finalize=lambda: None,
                            add_token=lambda tok: None, text="",
                            last_segment="")
    return SimpleNamespace(eos_token_ids=[0], eos_token_id=0,
                           detokenizer=detok)


def _stock_round(monkeypatch):
    """The stock MTP round on 2-token chunks, its generate_step a no-op.
    Returns the drafter."""
    import gmlx.gen.generation as generation

    ar = importlib.import_module("gmlx._vendor.mlx_vlm.generate.ar")
    monkeypatch.setenv("GMLX_OWNED_ROUND", "0")
    monkeypatch.setattr(generation, "_PREFILL_CHUNK", 2)
    monkeypatch.setattr(ar, "generate_step", lambda *a, **k: iter(()))
    return SimpleNamespace(config=SimpleNamespace(block_size=3))


def _job_step(model, monkeypatch):
    from gmlx.serve import engine_jobs

    with engine_jobs._step_scope(model):
        model.language_model(None)


def _mtp_chunk(model, monkeypatch):
    pytest.importorskip("gmlx._vendor.mlx_vlm")
    from gmlx._vendor.mlx_vlm.generate.ar import PromptProcessingBatch
    from gmlx.spec.mtp_prefill import install_full_prompt_mtp_prefill
    from gmlx.upstream.patching import _walk

    install_full_prompt_mtp_prefill()
    # The MTP body itself, under whatever wraps prompt_step in this process.
    step = next(f for f in _walk(PromptProcessingBatch.prompt_step)
                if getattr(f, "__name__", "") == "_mtp_prompt_step")
    batch = object.__new__(PromptProcessingBatch)
    batch.draft_kind = "mtp"
    batch.model = model
    batch.prefill_step_size = batch._gmlx_pinned_step = 2
    batch._mtp_full_input_ids = None
    # Called after the flip; None ends the chunk before any forward.
    batch.needs_processing = lambda: model.language_model(None)
    assert step(batch) == 0


def _spec_round(model, monkeypatch):
    import gmlx.gen.generation as generation

    drafter = _stock_round(monkeypatch)
    generation._generate_speculative(
        model, drafter, _tokenizer(), [1, 2, 3, 4, 5], draft_block_size=3,
        apply_chat_template=False)


def _cross_turn(model, monkeypatch):
    import gmlx.gen.generation as generation

    drafter = _stock_round(monkeypatch)
    list(generation._stream_generate_speculative(
        model, drafter, _tokenizer(), [1, 2, 3, 4, 5], prompt_cache=[],
        draft_block_size=3))


def _ar_bench(model, monkeypatch):
    from gmlx.gen import benchmarks

    _stock_round(monkeypatch)
    benchmarks._bench_ar_tps(model, _tokenizer(), [1, 2, 3, 4, 5],
                             decode_tokens=1, prefill_chunk=2)


@pytest.mark.parametrize("site", [_job_step, _mtp_chunk, _spec_round,
                                  _cross_turn, _ar_bench],
                         ids=["job step", "mtp chunk", "stock round",
                              "stock round next turn", "depth bench plain arm"])
def test_each_prefill_site_takes_the_caps_of_the_model_it_drives(
        monkeypatch, fake_kq, distinct_prefill, site):
    """A serve job step, an MTP prompt chunk, the chunked prefills of the
    stock MTP round (a first turn and a chat's next turn) and the plain arm
    of the depth bench each prefill on the caps of the model they drive.
    The verdict sits on that model only, so a site that reads its language
    model, or no model, runs on FINE."""
    import gmlx.serve.cb_phase as cb_phase

    seen = []
    model = SimpleNamespace(_kq_streaming=True,
                            language_model=_caps_forward(seen))
    cb_phase.flip("decode", _model(False))
    site(model, monkeypatch)
    assert seen and set(seen) == {cb_phase.STREAMED_PREFILL}


@pytest.mark.parametrize("media_first", [True, False],
                         ids=["media_spans first", "cb_phase first"])
def test_the_binding_stream_generate_calls_is_phased(monkeypatch, fake_kq,
                                                     media_first):
    """run --mmproj and chat's media turns drive mlx-vlm's stream_generate,
    which calls the generate_step its module bound at import. That binding
    runs in phased_steps, under or over media_spans' block-aware step."""
    pytest.importorskip("gmlx._vendor.mlx_vlm")
    import gmlx.gen.media_spans as media_spans
    import gmlx.serve.cb_phase as cb_phase
    from gmlx.upstream.patching import root, wrapped_with

    dispatch = importlib.import_module("gmlx._vendor.mlx_vlm.generate.dispatch")
    _keep_step_targets(monkeypatch)
    monkeypatch.setattr(dispatch, "generate_step", root(dispatch.generate_step))
    if media_first:
        media_spans.install_block_aware_generate_step()
    assert cb_phase.install_cb_phase_steps() is True
    media_spans.install_block_aware_generate_step()
    called = dispatch.stream_generate.__globals__["generate_step"]
    assert called is dispatch.generate_step
    assert wrapped_with(called, cb_phase._FLAG)
    assert wrapped_with(called, media_spans._GEN_FLAG)

def test_a_placement_lift_goes_through_the_caps_record(monkeypatch, fake_kq):
    """A config placement lifts the live caps for its load (a chat model
    switch, say). The lift goes through cb_phase, so a decode step after
    it still sets the model's decode caps."""
    from gmlx.commands.cli import _lift_live_cb_caps

    import gmlx.serve.cb_phase as cb_phase

    monkeypatch.delenv("MLX_MAX_OPS_PER_BUFFER", raising=False)
    streamed = _model(True)
    cb_phase.flip("decode", streamed)
    _lift_live_cb_caps()
    cb_phase.flip("decode", streamed)
    assert fake_kq == [cb_phase.STREAMED_DECODE, cb_phase.COARSE,
                       cb_phase.STREAMED_DECODE]


@pytest.mark.parametrize("on_language_model", [True, False])
@pytest.mark.parametrize("streamed", [True, False])
def test_the_depth_bench_plain_arm_decodes_on_the_caps_of_the_mtp_arm(
        on_language_model, streamed):
    """bench_tg_depth runs the plain arm of an owned drafter on a view of
    the target's language model and the MTP arm on the target: both decode
    on the target's caps, or the speedup it reports is off by the gap."""
    import mlx.nn as nn

    import gmlx.serve.cb_phase as cb_phase
    from gmlx.gen.benchmarks import _RawLogitsLM

    target = nn.Module()
    target.language_model = nn.Module()
    marked = target.language_model if on_language_model else target
    marked._kq_streaming = streamed
    plain = _RawLogitsLM(target)
    assert plain.lm is target.language_model
    assert cb_phase.decode_caps(plain) == cb_phase.decode_caps(target)


def test_serve_steps_keep_a_pin_or_the_switch(monkeypatch, held):
    """The user's pin of either cap, or GMLX_CB_PHASE=0, holds through
    every serve step: the flips do not install, and flips installed before
    write nothing."""
    pytest.importorskip("gmlx._vendor.mlx_vlm")
    import gmlx.serve.cb_phase as cb_phase

    key, writes = held
    ar = _stub_engine_steps(monkeypatch)
    assert cb_phase.install_cb_phase_flips() is False
    value = os.environ.pop(key)
    assert cb_phase.install_cb_phase_flips() is True
    monkeypatch.setenv(key, value)
    batch = object.__new__(ar.GenerationBatch)
    batch.model = _model(False)
    prompt = object.__new__(ar.PromptProcessingBatch)
    prompt.model = _model(True)
    ar.PromptProcessingBatch.prompt_step(prompt)
    ar.PromptProcessingBatch.generate(prompt)
    ar.GenerationBatch._step(batch)
    assert writes == []


def test_run_steps_keep_a_pin_or_the_switch(held):
    """The pin or the switch holds through each phase of run and chat."""
    import gmlx.serve.cb_phase as cb_phase

    _, writes = held
    assert cb_phase.install_cb_phase_steps() is False
    for model in (_model(True), _model(False)):
        assert list(cb_phase.phased_steps(iter(["t0", "t1"]), model)) == [
            "t0", "t1"]
    assert writes == []


def test_a_job_step_keeps_a_pin_or_the_switch(held):
    """A serve job step between decode steps writes no prefill caps under
    the pin or the switch."""
    from gmlx.serve import engine_jobs

    _, writes = held
    with engine_jobs._step_scope(None):
        pass
    assert writes == []


def test_a_chunked_prefill_keeps_a_pin_or_the_switch(held):
    """The chunked prefill of the speculative round and the depth bench's
    baseline writes no caps under the pin or the switch."""
    import mlx.core as mx

    from gmlx.gen.generation import _chunked_prefill_cache

    _, writes = held
    _chunked_prefill_cache(_lm, mx.array([[1, 2, 3, 4]]), 2, cache=[],
                           owner=_model(True))
    assert writes == []


def test_a_buffer_size_pin_holds_with_stream_experts(monkeypatch, fake_kq):
    """MLX_MAX_MB_PER_BUFFER alone is the user's pin too: the streaming
    lift fills in the ops cap beside it, and no phase writes over it."""
    from gmlx.commands.cli import _lift_live_cb_caps, _lift_stream_cb_caps

    import gmlx.serve.cb_phase as cb_phase

    monkeypatch.setenv("MLX_MAX_MB_PER_BUFFER", "200")
    _lift_stream_cb_caps(["run", "m.gguf", "--stream-experts"])
    assert os.environ["MLX_MAX_OPS_PER_BUFFER"] == "400"
    assert os.environ["MLX_MAX_MB_PER_BUFFER"] == "200"
    assert cb_phase.install_cb_phase_steps() is False
    _lift_live_cb_caps()
    cb_phase.flip("prefill", _model(True))
    cb_phase.flip("decode", _model(True))
    assert fake_kq == []


def test_engine_threads_leave_the_record_on_the_device_caps(monkeypatch,
                                                            fake_kq):
    """Engines of a residency pool step on their own threads. A step that
    sets the caps while another is between its write and its record must
    leave the record on the caps the device holds, or the next decode step
    of the other engine dedupes to no write and runs on the wrong caps."""
    import gmlx.serve.cb_phase as cb_phase

    device = []
    entered, go = threading.Event(), threading.Event()

    def set_cb_caps(ops, mb):
        device.append((ops, mb))
        if (ops, mb) == cb_phase.COARSE and not go.is_set():
            entered.set()
            go.wait(5)
        return ops, mb

    monkeypatch.setitem(cb_phase._state, "kq",
                        SimpleNamespace(set_cb_caps=set_cb_caps))
    resident = threading.Thread(target=cb_phase.flip,
                                args=("decode", _model(False)))
    streamed = threading.Thread(target=cb_phase.flip,
                                args=("decode", _model(True)))
    resident.start()
    assert entered.wait(5)
    streamed.start()
    streamed.join(0.2)   # it would write and record here without the lock
    go.set()
    resident.join(5)
    streamed.join(5)
    assert device == [cb_phase.COARSE, cb_phase.STREAMED_DECODE]
    assert cb_phase._state["caps"] == device[-1]
