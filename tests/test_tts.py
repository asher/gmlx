#!/usr/bin/env python3
"""Text-to-speech core (``gmlx.tts``): alias resolution, the request-model
policy, audio encoding, the synthesis driver with a stubbed mlx-audio, and the
background pre-warm. Pure CPU - mlx-audio itself is never imported (the import
is forced-missing where the gate is under test), no model is loaded."""
from __future__ import annotations

import sys
import threading
import types

import numpy as np
import pytest

from gmlx import tts  # noqa: E402
from gmlx.config import build_config  # noqa: E402

KOKORO = "mlx-community/Kokoro-82M-bf16"


# resolve_tts_model
def test_resolve_aliases_and_default():
    assert tts.resolve_tts_model("kokoro") == KOKORO
    assert tts.resolve_tts_model("KOKORO-8BIT") == "mlx-community/Kokoro-82M-8bit"
    for v in (True, "default", "true", "tts-1", "tts-1-hd"):
        assert tts.resolve_tts_model(v) == KOKORO
    for alias, repo in tts.TTS_ALIASES.items():
        assert tts.resolve_tts_model(alias) == repo


def test_resolve_passthrough_repo_and_local_dir(tmp_path):
    repo = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
    assert tts.resolve_tts_model(repo) == repo
    d = tmp_path / "my-tts"
    d.mkdir()
    assert tts.resolve_tts_model(str(d)) == str(d)


# effective_model (request `model` policy)
def test_effective_model_accepts_conventional_names():
    for n in ("", "tts-1", "tts-1-hd", "gpt-4o-mini-tts", "default", KOKORO):
        assert tts.effective_model(n, KOKORO) == KOKORO
    assert tts.effective_model("kokoro", KOKORO) == KOKORO   # alias -> configured


def test_effective_model_rejects_other_repos():
    with pytest.raises(tts.TTSRequestError) as exc:
        tts.effective_model("some/other-repo", KOKORO)
    assert exc.value.status_code == 400


# encode_audio
def test_encode_pcm_is_int16_le():
    b = tts.encode_audio(np.array([0.0, 1.0, -1.0], dtype=np.float32), 24000, "pcm")
    assert b == b"\x00\x00\xff\x7f\x01\x80"   # 0, +32767, -32767


def test_encode_pcm_clips_out_of_range():
    b = tts.encode_audio(np.array([2.0, -2.0], dtype=np.float32), 24000, "pcm")
    assert b == b"\xff\x7f\x01\x80"           # clipped to +/-32767


def test_encode_non_pcm_goes_through_writer(monkeypatch):
    calls = {}

    def fake_write(buf, data, samplerate, format=None):
        calls["sr"] = samplerate
        calls["fmt"] = format
        buf.write(b"AUDIO")

    audio_io = types.ModuleType("mlx_audio.audio_io")
    audio_io.write = fake_write
    mlx_audio = types.ModuleType("mlx_audio")
    mlx_audio.audio_io = audio_io
    monkeypatch.setitem(sys.modules, "mlx_audio", mlx_audio)
    monkeypatch.setitem(sys.modules, "mlx_audio.audio_io", audio_io)
    out = tts.encode_audio(np.array([0.1, 0.2], dtype=np.float32), 24000, "wav")
    assert out == b"AUDIO" and calls == {"sr": 24000, "fmt": "wav"}


# run_synthesis (driver; mlx-audio stubbed)
def _stub_synth(monkeypatch, sr=24000, audio=(0.0, 0.5)):
    monkeypatch.setattr(tts, "import_mlx_audio", lambda: None)
    monkeypatch.setattr(tts._TTSModelHolder, "get",
                        classmethod(lambda cls, p: ("MODEL", p)))
    cap = {}

    def fake_synth(model_obj, text, voice, speed):
        cap.update(model=model_obj, text=text, voice=voice, speed=speed)
        return np.array(audio, dtype=np.float32), sr

    monkeypatch.setattr(tts, "_synthesize", fake_synth)
    monkeypatch.setattr(
        tts, "available_voices",
        lambda p: ["af_bella", tts.DEFAULT_VOICE] if "Kokoro" in p else
        list(tts.QWEN3_TTS_VOICES) if "qwen3-tts" in p.lower() else [])
    return cap


def test_run_synthesis_happy_path_pcm(monkeypatch):
    cap = _stub_synth(monkeypatch)
    content, media = tts.run_synthesis(
        "Hello world", configured_model=KOKORO, model="tts-1",
        voice="af_bella", speed="1.5", response_format="pcm")
    assert media == "audio/pcm"
    assert content == tts.encode_audio(
        np.array([0.0, 0.5], dtype=np.float32), 24000, "pcm")
    assert cap["text"] == "Hello world"
    assert cap["voice"] == "af_bella"
    assert cap["speed"] == 1.5
    assert cap["model"] == ("MODEL", KOKORO)   # resolved target reached the holder


def test_run_synthesis_defaults_voice_speed_and_mp3(monkeypatch):
    cap = _stub_synth(monkeypatch)
    # stub the writer so the default mp3 path needs no real mlx-audio/ffmpeg
    audio_io = types.ModuleType("mlx_audio.audio_io")
    audio_io.write = lambda buf, data, sr, format=None: buf.write(b"MP3")
    mlx_audio = types.ModuleType("mlx_audio")
    mlx_audio.audio_io = audio_io
    monkeypatch.setitem(sys.modules, "mlx_audio", mlx_audio)
    monkeypatch.setitem(sys.modules, "mlx_audio.audio_io", audio_io)
    content, media = tts.run_synthesis("Hi", configured_model=KOKORO)
    assert media == "audio/mpeg" and content == b"MP3"
    assert cap["voice"] == tts.DEFAULT_VOICE and cap["speed"] == 1.0


def test_run_synthesis_no_default_voice_for_other_families(monkeypatch):
    # The Kokoro af_heart preset must not be forced on model families that
    # don't ship it; no voice lets the model's own default apply.
    cap = _stub_synth(monkeypatch)
    tts.run_synthesis("Hi", configured_model="mlx-community/qwen3-tts-4bit",
                      response_format="pcm")
    assert cap["voice"] is None
    cap2 = _stub_synth(monkeypatch)
    tts.run_synthesis("Hi", configured_model="mlx-community/qwen3-tts-4bit",
                      voice="Cherry", response_format="pcm")
    assert cap2["voice"] == "Cherry"


def test_run_synthesis_field_validation(monkeypatch):
    _stub_synth(monkeypatch)
    with pytest.raises(tts.TTSRequestError, match="'input' is required"):
        tts.run_synthesis("   ", configured_model=KOKORO)
    with pytest.raises(tts.TTSRequestError, match="response_format"):
        tts.run_synthesis("hi", configured_model=KOKORO, response_format="ogg")
    with pytest.raises(tts.TTSRequestError, match="is not a number"):
        tts.run_synthesis("hi", configured_model=KOKORO,
                          response_format="pcm", speed="fast")
    with pytest.raises(tts.TTSRequestError, match="between 0.25 and 4.0"):
        tts.run_synthesis("hi", configured_model=KOKORO,
                          response_format="pcm", speed="9")


def test_run_synthesis_wraps_backend_errors(monkeypatch):
    monkeypatch.setattr(tts, "import_mlx_audio", lambda: None)
    monkeypatch.setattr(tts._TTSModelHolder, "get",
                        classmethod(lambda cls, p: object()))

    def boom(*a, **k):
        raise RuntimeError("kokoro exploded")

    monkeypatch.setattr(tts, "_synthesize", boom)
    with pytest.raises(RuntimeError, match="synthesis failed: kokoro exploded"):
        tts.run_synthesis("hi", configured_model=KOKORO, response_format="pcm")


# prewarm (background model load; never touches a real GPU/HF here)
def test_prewarm_loads_in_background(monkeypatch):
    monkeypatch.setattr(tts, "import_mlx_audio", lambda: None)
    loaded = []
    monkeypatch.setattr(tts, "_load_tts_model", lambda p: loaded.append(p))
    fut = tts.prewarm(KOKORO)
    fut.result(timeout=5)
    assert fut.done()
    assert loaded == [KOKORO]


def test_prewarm_is_best_effort_on_load_failure(monkeypatch, capsys):
    monkeypatch.setattr(tts, "import_mlx_audio", lambda: None)

    def boom(_p):
        raise RuntimeError("HF offline")

    monkeypatch.setattr(tts, "_load_tts_model", boom)
    tts.prewarm(KOKORO).result(timeout=5)      # must not raise
    assert "tts prewarm failed" in capsys.readouterr().err


def test_prewarm_is_best_effort_when_extra_missing(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "mlx_audio", None)
    tts.prewarm(KOKORO).result(timeout=5)
    assert "tts prewarm failed" in capsys.readouterr().err


def test_load_tts_model_warms_holder(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        tts._TTSModelHolder, "get",
        classmethod(lambda cls, p: seen.update(path=p)))
    tts._load_tts_model(KOKORO)
    assert seen["path"] == KOKORO


def test_synthesis_runs_on_dedicated_worker(monkeypatch):
    """Load + synthesis must share one persistent thread, distinct from the
    caller: mlx-audio binds a model's Metal work to its loading thread, so
    generating on Starlette's rotating threadpool raises 'no Stream(gpu, N)'."""
    monkeypatch.setattr(tts, "import_mlx_audio", lambda: None)
    where = {}
    monkeypatch.setattr(
        tts._TTSModelHolder, "get",
        classmethod(lambda cls, p: where.setdefault(
            "load", threading.current_thread().name)))

    def fake_synth(model_obj, text, voice, speed):
        where["synth"] = threading.current_thread().name
        return np.array([0.0], dtype=np.float32), 24000

    monkeypatch.setattr(tts, "_synthesize", fake_synth)
    tts.run_synthesis("hi", configured_model=KOKORO, response_format="pcm")
    assert where["synth"].startswith("tts-worker")          # off the caller
    assert where["synth"] != threading.current_thread().name
    assert where["load"] == where["synth"]                  # loader == generator


# import gate + config key
def test_import_gate_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx_audio", None)
    with pytest.raises(ImportError, match=r"gmlx\[tts\]"):
        tts.import_mlx_audio()


def test_config_tts_key_round_trips():
    cfg = build_config({"server": {"tts": "kokoro-8bit"}})
    assert cfg.tts == "kokoro-8bit"            # raw; resolved at serve time
    assert build_config({"server": {}}).tts is None
    assert build_config({"server": {"tts": False}}).tts is None
    assert build_config({"server": {"tts": True}}).tts is True
    assert tts.resolve_tts_model(True) == KOKORO


def test_available_voices_from_local_voices_dir(tmp_path):
    vd = tmp_path / "voices"
    vd.mkdir()
    for name in ("af_heart.safetensors", "am_adam.pt", "notes.txt"):
        (vd / name).write_bytes(b"")
    assert tts.available_voices(str(tmp_path)) == ["af_heart", "am_adam"]


def test_available_voices_qwen3_static_set():
    voices = tts.available_voices("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit")
    assert "Chelsie" in voices and len(voices) >= 4


def test_available_voices_unknown_is_empty(tmp_path):
    assert tts.available_voices(str(tmp_path)) == []
    assert tts.available_voices("nobody/UnknownTTS") == []


def test_kokoro_sine_length_guard():
    # mlx-audio <=0.4.4: SineGen._f02sine's interpolate roundtrip
    # (scale_factor=1/upsample_scale, then upsample_scale) emits one extra
    # frame for lengths where N/scale rounds up in fp, and the result
    # broadcast-fails against the uv mask (value-dependent 500s: "hello",
    # "The API returns JSON quickly."). The guard trims/pads to the input
    # length.
    istftnet = pytest.importorskip("mlx_audio.tts.models.kokoro.istftnet")
    import mlx.core as mx

    tts._patch_kokoro_sine_length()
    gen = istftnet.SineGen(samp_rate=24000, upsample_scale=300, harmonic_num=8)
    for n in (300, 33600, 33900, 72000):  # 33600/72000 reproduce the bug
        f0 = mx.full((1, n, 1), 120.0)
        sine, uv, noise = gen(f0)
        mx.eval(sine, uv, noise)
        assert sine.shape == (1, n, 9)
        assert uv.shape == (1, n, 1)


def test_sanitize_strips_markdown_keeps_scripts():
    s = tts.sanitize_speech_text
    assert s("**Pasta**\n\n* 8 oz spaghetti") == "Pasta\n8 oz spaghetti"
    assert s("See [the docs](https://x.y/z) for more.") == "See the docs for more."
    assert s("# Heading\n> quoted") == "Heading\nquoted"
    assert s("snake_case and `code`") == "snake case and code"
    # dashes become pauses; NFKC folds fractions/fullwidth forms
    assert s("Cook — about 10 minutes — until done.") == \
        "Cook, about 10 minutes, until done."
    assert s("Add ½ cup.") == "Add 1/2 cup."
    # non-Latin scripts pass through untouched (i18n: letters never dropped)
    assert s("日本語のテキスト。") == "日本語のテキスト。"
    assert s("tiếng Việt; हिन्दी; русский") == "tiếng Việt; हिन्दी; русский"
    # emoji and control chars are unspeakable
    assert s("Buon appetito! 🍝") == "Buon appetito!"
    assert s("🍝🎉✨") == ""


def test_run_synthesis_rejects_unspeakable_input():
    with pytest.raises(tts.TTSRequestError) as e:
        tts.run_synthesis("🍝🎉", configured_model="kokoro")
    assert e.value.status_code == 400
    assert "speakable" in str(e.value)


def test_configure_espeak_sets_bundled_library(monkeypatch):
    lib, data = "/fake/libespeak-ng.dylib", "/fake/espeak-ng-data"
    calls = {}

    class _Wrapper:
        _ESPEAK_LIBRARY = None
        @classmethod
        def set_library(cls, p):
            calls["lib"] = p
        @classmethod
        def set_data_path(cls, p):
            calls["data"] = p

    loader = types.SimpleNamespace(get_library_path=lambda: lib,
                                   get_data_path=lambda: data)
    wrapper_mod = types.ModuleType("phonemizer.backend.espeak.wrapper")
    wrapper_mod.EspeakWrapper = _Wrapper
    for name, mod in {
        "espeakng_loader": loader,
        "phonemizer": types.ModuleType("phonemizer"),
        "phonemizer.backend": types.ModuleType("phonemizer.backend"),
        "phonemizer.backend.espeak": types.ModuleType("phonemizer.backend.espeak"),
        "phonemizer.backend.espeak.wrapper": wrapper_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    tts._configure_espeak()
    assert calls == {"lib": lib, "data": data}

    _Wrapper._ESPEAK_LIBRARY = "/already/set.dylib"  # respected, not clobbered
    calls.clear()
    tts._configure_espeak()
    assert calls == {}
