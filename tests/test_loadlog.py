#!/usr/bin/env python3
"""`gmlx.loadlog`: load-time output routing - verbose gating, stage/spinner
plumbing, warn channel, facts, and the `load_ui` CLI wrapper. Also covers
`generation._prefill_progress_ui` (the run prefill spinner). CPU-only."""
from __future__ import annotations

import io
import sys
import threading
import time
from collections import Counter

import pytest

from gmlx import loadlog


class _FakeSpinner:
    def __init__(self):
        self.updates: list[str] = []
        self.lines: list[str] = []

    def update(self, text):
        self.updates.append(text)

    def println(self, msg):
        self.lines.append(msg)


def _session(verbose=False, spinner=None, base="loading x"):
    """Install a loadlog state; returns the reset token."""
    state = loadlog._State(verbose=verbose, spinner=spinner, base=base)
    return loadlog._STATE.set(state), state


# -- no-state back-compat -----------------------------------------------------


def test_no_state_is_verbose_and_prints(capsys):
    assert loadlog.is_verbose()
    loadlog.verbose_print("[x] hello")
    assert capsys.readouterr().out == "[x] hello\n"


def test_no_state_warn_goes_to_stderr(capsys):
    loadlog.warn("WARNING: boom")
    err = capsys.readouterr().err
    assert "WARNING: boom" in err


def test_no_state_stage_and_fact_are_noops():
    loadlog.stage("reading tensors")     # must not raise
    loadlog.fact("arch", "llama")
    assert loadlog.facts() == {}


# -- seeded sessions ----------------------------------------------------------


def test_seeds_quiet_silences_and_resets(capsys):
    @loadlog.seeds
    def fake_load(*, verbose=True):
        loadlog.verbose_print("[gguf] diag")
        return "ok"

    assert fake_load(verbose=False) == "ok"
    assert capsys.readouterr().out == ""
    assert loadlog.is_verbose()          # state reset after the call


def test_seeds_verbose_prints(capsys):
    @loadlog.seeds
    def fake_load(*, verbose=True):
        loadlog.verbose_print("[gguf] diag")

    fake_load(verbose=True)
    assert "[gguf] diag" in capsys.readouterr().out


def test_outer_session_wins_over_inner_verbose(capsys):
    @loadlog.seeds
    def inner(*, verbose=True):
        loadlog.verbose_print("[inner] diag")

    tok, _ = _session(verbose=False)
    try:
        inner(verbose=True)              # nested load defers to the quiet outer
    finally:
        loadlog._STATE.reset(tok)
    assert capsys.readouterr().out == ""


def test_stage_updates_spinner_and_records_label():
    sp = _FakeSpinner()
    tok, state = _session(spinner=sp, base="loading m.gguf")
    try:
        loadlog.stage("remapping tensors")
        loadlog.stage("building model")
    finally:
        loadlog._STATE.reset(tok)
    assert sp.updates == [
        "loading m.gguf: remapping tensors",
        "loading m.gguf: building model",
    ]
    assert state.stage_label == "building model"


def test_warn_routes_through_active_spinner(capsys):
    sp = _FakeSpinner()
    tok, _ = _session(spinner=sp)
    try:
        loadlog.warn("WARNING: odd tensor")
    finally:
        loadlog._STATE.reset(tok)
    assert sp.lines == ["WARNING: odd tensor"]
    assert capsys.readouterr().err == ""


def test_facts_round_trip():
    tok, _ = _session()
    try:
        loadlog.fact("arch", "qwen35moe")
        loadlog.fact("codecs", Counter({"q6_k": 80}))
        assert loadlog.facts()["arch"] == "qwen35moe"
    finally:
        loadlog._STATE.reset(tok)


def test_thread_isolation():
    tok, _ = _session(verbose=False)
    seen = {}

    def body():
        seen["verbose"] = loadlog.is_verbose()

    try:
        t = threading.Thread(target=body)
        t.start()
        t.join()
    finally:
        loadlog._STATE.reset(tok)
    assert seen["verbose"] is True       # fresh thread sees no session


# -- summary line -------------------------------------------------------------


def test_summary_top3_codecs_and_truncation():
    line = loadlog._summary_line(
        "m.gguf",
        6.84,
        {
            "arch": "qwen35moe",
            "codecs": Counter({"q8_0": 302, "q6_k": 80, "q4_k": 9, "q5_k": 2, "f32": 1}),
            "drafter": "native-head",
        },
    )
    assert line.startswith("[load] m.gguf | arch qwen35moe | ")
    assert "q8_0 x302 q6_k x80 q4_k x9 +2 more" in line
    assert "| 6.8s" in line
    assert line.endswith("| drafter native-head")


def test_summary_no_kquant_and_mmproj():
    line = loadlog._summary_line("m.gguf", 1.0, {"arch": "llama", "mmproj": True})
    assert "no kquant tensors" in line
    assert "+mmproj" in line


def test_summary_attn_and_indexer_sidecar():
    line = loadlog._summary_line(
        "m.gguf",
        0.8,
        {
            "arch": "minimax-m3",
            "attn": "msa",
            "indexer-sidecar": "minimax-m3-indexer-bf16.gguf",
        },
    )
    assert "| attn msa |" in line
    assert line.endswith("| indexer sidecar minimax-m3-indexer-bf16.gguf")


# -- load_ui ------------------------------------------------------------------


def test_load_ui_verbose_is_null_context(capsys):
    with loadlog.load_ui(True, "/x/m.gguf"):
        loadlog.verbose_print("[gguf] diag")
    out = capsys.readouterr().out
    assert "[gguf] diag" in out
    assert "[load]" not in out           # no summary in verbose mode


def test_load_ui_quiet_prints_summary_only(capsys):
    with loadlog.load_ui(False, "/x/m.gguf"):
        loadlog.verbose_print("[gguf] diag")
        loadlog.fact("arch", "llama")
    out = capsys.readouterr().out
    assert "[gguf] diag" not in out
    assert out.startswith("[load] m.gguf | arch llama | ")


def test_load_ui_failure_names_stage_and_reraises(capsys):
    with pytest.raises(RuntimeError, match="boom") as ei:
        with loadlog.load_ui(False, "/x/m.gguf"):
            loadlog.stage("building model")
            raise RuntimeError("boom")
    captured = capsys.readouterr()
    # One merged line: stage + reason; flagged so the CLI won't print again.
    assert "load failed building model: boom" in captured.err
    assert "--verbose" in captured.err
    assert getattr(ei.value, "_gmlx_reported", False)
    assert "[load]" not in captured.out  # no summary on failure


def test_load_ui_gguf_metadata_failure_hints_truncation(capsys):
    with pytest.raises(ValueError):
        with loadlog.load_ui(False, "/x/m.gguf"):
            loadlog.stage("reading gguf metadata")
            raise ValueError("bad magic")
    err = capsys.readouterr().err
    assert "load failed reading gguf metadata: bad magic" in err
    assert "truncated download or not a GGUF file?" in err


def test_load_ui_keyboard_interrupt_is_silent(capsys):
    with pytest.raises(KeyboardInterrupt):
        with loadlog.load_ui(False, "/x/m.gguf"):
            raise KeyboardInterrupt
    captured = capsys.readouterr()
    assert "load failed" not in captured.err
    assert "[load]" not in captured.out


def test_load_ui_resets_state_on_success_and_failure():
    with loadlog.load_ui(False, "/x/m.gguf"):
        assert not loadlog.is_verbose()
    assert loadlog.is_verbose()
    with pytest.raises(ValueError):
        with loadlog.load_ui(False, "/x/m.gguf"):
            raise ValueError
    assert loadlog.is_verbose()


# -- capture (chat background load) --------------------------------------------


def test_capture_defers_stray_writes(capsys):
    with loadlog.capture("/x/m.gguf") as cap:
        loadlog.fact("arch", "llama")
        print("W: added a special token", file=sys.stderr)
        print("progress dots")
    captured = capsys.readouterr()
    assert captured.err == "" and captured.out == ""    # nothing smeared live
    assert "W: added a special token\n" in cap.stray    # handed to the joiner
    assert "progress dots\n" in cap.stray
    assert cap.summary.startswith("[load] m.gguf | arch llama")


def test_capture_stray_kept_on_failure(capsys):
    with pytest.raises(RuntimeError, match="boom"):
        with loadlog.capture("/x/m.gguf") as cap:
            loadlog.stage("building tokenizer")
            print("clue about the failure", file=sys.stderr)
            raise RuntimeError("boom")
    assert cap.stage == "building tokenizer"
    assert "clue about the failure" in cap.stray
    assert capsys.readouterr().err == ""


def test_router_passes_uncaptured_contexts_through(capsys):
    with loadlog.capture("/x/m.gguf"):
        seen = {}

        def body():
            # A thread with no capture session (e.g. the main REPL thread
            # while a background load runs) writes straight through.
            print("live line", file=sys.stderr)
            seen["done"] = True

        t = threading.Thread(target=body)
        t.start()
        t.join()
    assert seen["done"]
    assert "live line" in capsys.readouterr().err


# -- prefill progress spinner (generation._prefill_progress_ui) ----------------


class _FakeTty(io.StringIO):
    def isatty(self):
        return True


def _wait_for(pred, timeout=2.0):
    """Poll until pred() or timeout; the spinner writes from its own thread."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


def _prefill_ui():
    pytest.importorskip("mlx_lm")
    from gmlx import generation

    stream = _FakeTty()
    cb, close = generation._prefill_progress_ui(stream=stream)
    return cb, close, stream


def test_prefill_short_prompt_never_spins():
    cb, close, stream = _prefill_ui()
    cb(0, 511)                           # below the 512-token threshold
    cb(511, 511)
    close()
    assert stream.getvalue() == ""


def test_prefill_spinner_lifecycle():
    cb, close, stream = _prefill_ui()
    cb(0, 4096)                          # first chunk: spinner starts
    assert _wait_for(lambda: "prefill 0/4096 tok" in stream.getvalue())
    cb(2048, 4096)                       # label swap, visible on the next frame
    assert _wait_for(lambda: "prefill 2048/4096 tok" in stream.getvalue())
    cb(4096, 4096)                       # final call, before the first token
    assert stream.getvalue().endswith("\r\x1b[K")   # line cleared
    frozen = stream.getvalue()
    close()                              # the finally-path close is a no-op
    close()
    assert stream.getvalue() == frozen


def test_generate_wires_progress_callback_only_when_opted_in(monkeypatch):
    pytest.importorskip("mlx_lm")
    import mlx_lm

    from gmlx import generation

    captured: list[dict] = []

    def fake_generate(model, tokenizer, prompt, **kw):
        captured.append(kw)
        return "ok"

    monkeypatch.setattr(mlx_lm, "generate", fake_generate)

    class _Tok:
        chat_template = None
        eos_token_ids = [0]

    def run(*, tty, **kw):
        monkeypatch.setattr(sys, "stderr", _FakeTty() if tty else io.StringIO())
        assert generation.generate(object(), _Tok(), "hi", **kw) == "ok"
        return captured.pop()

    assert "prompt_progress_callback" in run(tty=True, prefill_progress=True)
    assert "prompt_progress_callback" not in run(tty=False, prefill_progress=True)
    assert "prompt_progress_callback" not in run(tty=True)


def test_seeds_defaults_to_quiet(capsys):
    # Library callers get a quiet load unless they opt in with verbose=True.
    @loadlog.seeds
    def fake_load(*, verbose=False):
        loadlog.verbose_print("[gguf] diag")
        return "ok"

    assert fake_load() == "ok"
    assert capsys.readouterr().out == ""
