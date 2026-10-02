"""gmlx/serve/media_programs.py: the server runs ffmpeg and ffprobe only from
the Homebrew and system folders. A folder on the server's PATH can lie in a
share that a container client changes, such as a project's .venv/bin."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import re
import sys
import types

import numpy as np
import pytest

from gmlx.container import settings
from gmlx.serve import media_programs, media_sinks, server, stt, tts

MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 16
# What the fake ffmpeg prints: two 16-bit samples.
PCM = b"abcd"
SAMPLES = np.frombuffer(PCM, dtype=np.int16)


def _program(folder, name: str, log, body: str) -> None:
    path = folder / name
    path.write_text(f'#!/bin/sh\necho "$0" >> "{log}"\ncat > /dev/null\n{body}\n')
    path.chmod(0o755)


@pytest.fixture
def ran(tmp_path, monkeypatch):
    """Plants ffmpeg and ffprobe first on PATH, and puts working fakes in
    the folder where the server looks. Gives the programs that ran."""
    log = tmp_path / "ran.txt"
    planted = tmp_path / "share" / ".venv" / "bin"
    system = tmp_path / "system" / "bin"
    for folder in (planted, system):
        folder.mkdir(parents=True)
        for name in ("ffmpeg", "ffprobe"):
            _program(folder, name, log, "exit 1")
    _program(system, "ffprobe", log,
             """echo '{"streams": [{"sample_rate": "16000", "channels": 1}]}'""")
    _program(system, "ffmpeg", log, f"printf '{PCM.decode()}'")
    monkeypatch.setenv("PATH", f"{planted}:/usr/bin:/bin")
    monkeypatch.setattr(settings, "SYSTEM_PATH", str(system))
    media_sinks.install()

    def programs() -> list[str]:
        lines = log.read_text().splitlines() if log.exists() else []
        return [line.replace(str(tmp_path), "") for line in lines]
    yield programs
    media_sinks.uninstall()


def _whisper(monkeypatch) -> list:
    heard = []
    mod = types.ModuleType("mlx_whisper")

    def transcribe(audio, *, path_or_hf_repo, temperature, **decode_options):
        heard.append(audio)
        return {"text": "hi", "segments": []}

    mod.transcribe = transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", mod)

    @contextlib.contextmanager
    def online(_ref):
        yield
    monkeypatch.setattr(stt, "offline_resolve", online)
    return heard


def test_request_audio_is_decoded_with_the_programs_where_the_server_looks(ran):
    audio_io = importlib.import_module("mlx_audio.audio_io")
    samples, rate = audio_io.read(MP4, dtype="int16")
    assert rate == 16000 and samples.tolist() == SAMPLES.tolist()
    assert ran() == ["/system/bin/ffprobe", "/system/bin/ffmpeg"]


def test_speech_is_encoded_with_the_ffmpeg_where_the_server_looks(ran):
    audio = np.zeros(2400, dtype=np.float32)
    assert tts.encode_audio(audio, 24000, "mp3") == PCM
    assert tts.encode_audio(audio, 24000, "opus") == PCM
    assert ran() == ["/system/bin/ffmpeg", "/system/bin/ffmpeg"]


def test_an_upload_is_decoded_with_the_ffmpeg_where_the_server_looks(ran, monkeypatch):
    heard = _whisper(monkeypatch)
    content, _ = stt.run_transcription(b"x", filename="a.m4a", configured_model="whisper-1")
    assert content == {"text": "hi"}
    (audio,) = heard
    assert audio.dtype == np.float32 and audio.tolist() == (SAMPLES / 32768.0).tolist()
    assert ran() == ["/system/bin/ffmpeg"]


@pytest.mark.filterwarnings("ignore:ffmpeg is required:RuntimeWarning")
def test_a_missing_program_names_where_the_server_looks(ran, monkeypatch, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(settings, "SYSTEM_PATH", f"{empty}:/nowhere")
    audio_io = importlib.import_module("mlx_audio.audio_io")
    want = rf"ffmpeg is not in {empty} or /nowhere, .*`brew install ffmpeg`"
    with pytest.raises(media_programs.ProgramMissing, match=want):
        audio_io.read(MP4)
    with pytest.raises(media_programs.ProgramMissing, match=want):
        tts.encode_audio(np.zeros(4, dtype=np.float32), 24000, "mp3")
    _whisper(monkeypatch)
    with pytest.raises(RuntimeError, match="transcription failed: " + want):
        stt.run_transcription(b"x", filename="a.m4a", configured_model="whisper-1")
    assert ran() == []


def test_the_install_hints_name_where_the_server_looks_for_ffmpeg(monkeypatch):
    monkeypatch.setattr(settings, "SYSTEM_PATH", "/nowhere/a/bin:/nowhere/b/bin")
    for name, load in (("mlx_whisper", stt.import_mlx_whisper),
                       ("mlx_audio", tts.import_mlx_audio)):
        monkeypatch.setitem(sys.modules, name, None)
        with pytest.raises(ImportError) as err:
            load()
        assert "on PATH" not in str(err.value)
        assert ("ffmpeg in /nowhere/a/bin or /nowhere/b/bin - `brew install ffmpeg`"
                in str(err.value))


def test_the_serve_help_names_where_the_server_looks_for_ffmpeg():
    ap = argparse.ArgumentParser()
    server._add_serve_args(ap)
    helps = {action.dest: action.help or "" for action in ap._actions}
    for dest in ("stt", "tts"):
        assert "on PATH" not in helps[dest]
        named = re.findall(r"/[\w/]+/bin\b", helps[dest])
        assert named and set(named) <= set(media_programs.folders()), helps[dest]
