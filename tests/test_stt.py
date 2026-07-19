#!/usr/bin/env python3
"""Speech-to-text core (``gmlx.stt``): alias resolution, the request-model
policy, OpenAI response shaping, and the transcription driver with a stubbed
mlx-whisper. Pure CPU - mlx-whisper itself is never imported (the import is
forced-missing where the gate is under test), no audio is decoded."""
from __future__ import annotations

import contextlib
import sys
import threading
import types

import pytest


from gmlx import stt  # noqa: E402
from gmlx.config import build_config  # noqa: E402

TURBO = "mlx-community/whisper-large-v3-turbo"


@pytest.fixture(autouse=True)
def _noop_offline_resolve(monkeypatch):
    """Keep STT unit tests hermetic: neutralize the HF-cache offline window so a
    repo-id ref doesn't trigger a real ``snapshot_download``. The real
    ``offline_resolve`` (cache-only resolve + download-once) is covered in
    ``test_hf_cache.py``."""
    @contextlib.contextmanager
    def _noop(_ref):
        yield
    monkeypatch.setattr(stt, "offline_resolve", _noop)


# resolve_stt_model
def test_resolve_aliases_and_default():
    assert stt.resolve_stt_model("whisper-turbo") == TURBO
    assert stt.resolve_stt_model("WHISPER-TURBO-Q4") == TURBO + "-q4"
    for v in (True, "default", "true", "whisper-1"):
        assert stt.resolve_stt_model(v) == TURBO
    for alias, repo in stt.STT_ALIASES.items():
        assert stt.resolve_stt_model(alias) == repo


def test_resolve_passthrough_repo_and_local_dir(tmp_path):
    # An explicit repo id (e.g. the fp16 repo when q4 isn't wanted) passes through.
    assert stt.resolve_stt_model("mlx-community/whisper-large-v3-mlx") == \
        "mlx-community/whisper-large-v3-mlx"
    d = tmp_path / "my-whisper"
    d.mkdir()
    assert stt.resolve_stt_model(str(d)) == str(d)


# effective_model (request `model` policy)
def test_effective_model_accepts_conventional_names():
    for req in ("", "whisper-1", "default", "WHISPER-1", TURBO, "whisper-turbo"):
        assert stt.effective_model(req, TURBO) == TURBO


def test_effective_model_rejects_other_models():
    # Clients must not be able to make the server pull arbitrary repos.
    with pytest.raises(stt.STTRequestError) as ei:
        stt.effective_model("mlx-community/whisper-tiny", TURBO)
    assert ei.value.status_code == 400
    assert "whisper-1" in str(ei.value)


# format_result (OpenAI response shapes)
_RESULT = {
    "text": " Hello world. ",
    "language": "en",
    "segments": [
        {"id": 0, "seek": 0, "start": 0.0, "end": 1.25, "text": " Hello",
         "tokens": [1], "temperature": 0.0, "avg_logprob": -0.1,
         "compression_ratio": 1.0, "no_speech_prob": 0.01},
        {"id": 1, "seek": 0, "start": 1.25, "end": 3661.5, "text": " world.",
         "tokens": [2], "temperature": 0.0, "avg_logprob": -0.2,
         "compression_ratio": 1.0, "no_speech_prob": 0.02},
    ],
}


def test_format_json_default():
    content, media = stt.format_result(_RESULT, "json")
    assert content == {"text": "Hello world."} and media == "application/json"


def test_format_text():
    content, media = stt.format_result(_RESULT, "text")
    assert content == "Hello world.\n" and media == "text/plain"


def test_format_verbose_json():
    content, media = stt.format_result(_RESULT, "verbose_json")
    assert media == "application/json"
    assert content["task"] == "transcribe"
    assert content["language"] == "en"
    assert content["duration"] == 3661.5
    assert [s["text"] for s in content["segments"]] == [" Hello", " world."]
    assert "tokens" in content["segments"][0]


def test_format_srt_and_vtt_timestamps():
    srt, _ = stt.format_result(_RESULT, "srt")
    # SRT: comma millis, 1-based counters; 3661.5s == 01:01:01,500.
    assert "1\n00:00:00,000 --> 00:00:01,250\nHello" in srt
    assert "01:01:01,500" in srt
    vtt, _ = stt.format_result(_RESULT, "vtt")
    assert vtt.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.250\nHello" in vtt


# run_transcription (driver; mlx-whisper stubbed)
def _stub_whisper(monkeypatch, calls):
    mod = types.ModuleType("mlx_whisper")

    def transcribe(audio_path, *, path_or_hf_repo, temperature, **decode_options):
        calls.append({"path": audio_path, "repo": path_or_hf_repo,
                      "temperature": temperature, **decode_options})
        return _RESULT

    mod.transcribe = transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", mod)


def test_run_transcription_happy_path(monkeypatch, tmp_path):
    calls = []
    _stub_whisper(monkeypatch, calls)
    content, media = stt.run_transcription(
        b"RIFFfake", filename="clip.ogg", configured_model=TURBO,
        model="whisper-1", language=" en ", prompt="Names: Asher",
        temperature="0.2", response_format="json")
    assert content == {"text": "Hello world."} and media == "application/json"
    (call,) = calls
    assert call["repo"] == TURBO
    assert call["path"].endswith(".ogg")     # upload suffix preserved for ffmpeg
    assert call["temperature"] == 0.2
    assert call["language"] == "en"
    assert call["initial_prompt"] == "Names: Asher"
    import os
    assert not os.path.exists(call["path"])  # temp upload cleaned up


def test_run_transcription_translate_task(monkeypatch):
    calls = []
    _stub_whisper(monkeypatch, calls)
    content, media = stt.run_transcription(
        b"RIFFfake", filename="clip.wav", configured_model=TURBO,
        task="translate", response_format="verbose_json")
    (call,) = calls
    assert call["task"] == "translate"            # Whisper translate task forwarded
    assert content["task"] == "translate"         # echoed in the verbose body


def test_run_transcription_default_task_omits_task_option(monkeypatch):
    calls = []
    _stub_whisper(monkeypatch, calls)
    stt.run_transcription(b"x", filename="a.wav", configured_model=TURBO)
    (call,) = calls
    assert "task" not in call                      # transcribe is Whisper's default


def test_format_verbose_json_translate_reports_task():
    content, _ = stt.format_result(_RESULT, "verbose_json", task="translate")
    assert content["task"] == "translate"


def test_run_transcription_field_validation(monkeypatch):
    calls = []
    _stub_whisper(monkeypatch, calls)
    with pytest.raises(stt.STTRequestError, match="'file' is required"):
        stt.run_transcription(b"", filename="a.wav", configured_model=TURBO)
    with pytest.raises(stt.STTRequestError, match="response_format"):
        stt.run_transcription(b"x", filename="a.wav", configured_model=TURBO,
                              response_format="yaml")
    with pytest.raises(stt.STTRequestError, match="temperature"):
        stt.run_transcription(b"x", filename="a.wav", configured_model=TURBO,
                              temperature="warm")
    assert not calls                          # nothing reached the model


def test_run_transcription_wraps_backend_errors(monkeypatch):
    mod = types.ModuleType("mlx_whisper")

    def transcribe(*a, **k):
        raise RuntimeError("[Errno 2] No such file or directory: 'ffmpeg'")

    mod.transcribe = transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", mod)
    with pytest.raises(RuntimeError, match="brew install ffmpeg"):
        stt.run_transcription(b"x", filename="a.wav", configured_model=TURBO)


# import gate + config key
def test_import_gate_message(monkeypatch):
    # sys.modules[name] = None forces ImportError even if mlx-whisper IS installed.
    monkeypatch.setitem(sys.modules, "mlx_whisper", None)
    with pytest.raises(ImportError, match=r"gmlx\[stt\]"):
        stt.import_mlx_whisper()


def test_config_stt_key_round_trips():
    cfg = build_config({"server": {"stt": "whisper-turbo-q4"}})
    assert cfg.stt == "whisper-turbo-q4"      # raw; resolved at serve time
    assert build_config({"server": {}}).stt is None
    assert build_config({"server": {"stt": False}}).stt is None
    # YAML `stt: true` enables the default alias downstream.
    assert build_config({"server": {"stt": True}}).stt is True
    assert stt.resolve_stt_model(True) == TURBO


# prewarm (background model load; never touches a real GPU/HF here)
def test_prewarm_loads_in_background(monkeypatch):
    monkeypatch.setattr(stt, "import_mlx_whisper", lambda: None)
    loaded = []
    monkeypatch.setattr(stt, "_load_stt_model", lambda p: loaded.append(p))
    fut = stt.prewarm(TURBO)
    fut.result(timeout=5)
    assert fut.done()
    assert loaded == [TURBO]


def test_prewarm_is_best_effort_on_load_failure(monkeypatch, capsys):
    monkeypatch.setattr(stt, "import_mlx_whisper", lambda: None)

    def boom(_p):
        raise RuntimeError("HF offline")

    monkeypatch.setattr(stt, "_load_stt_model", boom)
    stt.prewarm(TURBO).result(timeout=5)      # must not raise
    assert "stt prewarm failed" in capsys.readouterr().err


def test_prewarm_is_best_effort_when_extra_missing(monkeypatch, capsys):
    # Forced ImportError from the real install gate is swallowed too.
    monkeypatch.setitem(sys.modules, "mlx_whisper", None)
    stt.prewarm(TURBO).result(timeout=5)
    assert "stt prewarm failed" in capsys.readouterr().err


def test_load_stt_model_warms_holder_with_float16(monkeypatch):
    import mlx.core as mx

    seen = {}

    class FakeHolder:
        @staticmethod
        def get_model(path, dtype):
            seen["path"] = path
            seen["dtype"] = dtype

    fake_tr = types.ModuleType("mlx_whisper.transcribe")
    fake_tr.ModelHolder = FakeHolder
    fake_mw = types.ModuleType("mlx_whisper")
    fake_mw.transcribe = fake_tr
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mw)
    monkeypatch.setitem(sys.modules, "mlx_whisper.transcribe", fake_tr)

    stt._load_stt_model(TURBO)
    assert seen["path"] == TURBO
    assert seen["dtype"] == mx.float16        # the dtype run_transcription serves


def test_transcription_runs_on_dedicated_worker(monkeypatch):
    """Transcription must run off the caller, on the persistent STT worker:
    mlx-whisper binds a model to its loading thread, and the resulting
    'no Stream(gpu, N)' surfaces async and aborts the whole process."""
    where = {}
    mod = types.ModuleType("mlx_whisper")

    def transcribe(audio_path, *, path_or_hf_repo, temperature, **opts):
        where["transcribe"] = threading.current_thread().name
        return _RESULT

    mod.transcribe = transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", mod)
    stt.run_transcription(b"RIFFfake", filename="a.wav", configured_model=TURBO)
    assert where["transcribe"].startswith("stt-worker")       # off the caller
    assert where["transcribe"] != threading.current_thread().name
