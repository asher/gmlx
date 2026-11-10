#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""``gmlx run --speech-out``: the flag refusals before the load, the
dispatch, and the run's output with the model and generator stubbed.
CPU-only."""

from __future__ import annotations

import wave
from types import SimpleNamespace

import numpy as np
import pytest

import gmlx.commands.cli as cli
from gmlx.commands import speech_run
from tests.lfm3_header import write_projector_header


@pytest.fixture
def gguf(tmp_path):
    p = tmp_path / "m.gguf"
    p.write_bytes(b"GGUF")
    return str(p)


@pytest.fixture
def speaks(tmp_path):
    return write_projector_header(tmp_path / "mmproj-speaks.gguf",
                                  audio=True, speech=True)


@pytest.fixture
def hears(tmp_path):
    return write_projector_header(tmp_path / "mmproj-hears.gguf", audio=True)


def _args(**kw):
    base = dict(speech_out=None, tts=False, voice=None, speech_language=None,
                mmproj=None, image=None, audio=None, system_prompt=None,
                no_chat_template=False)
    base.update(kw)
    return SimpleNamespace(**base)


def test_a_run_without_speech_flags_is_not_refused():
    assert speech_run.refusal(_args()) is None


@pytest.mark.parametrize("flags, message", [
    (dict(tts=True), "--tts needs --speech-out FILE.wav"),
    (dict(voice="male"), "--voice needs --speech-out FILE.wav"),
    (dict(speech_language="fr"),
     "--speech-language needs --speech-out FILE.wav"),
])
def test_the_speech_flags_need_the_output_file(flags, message):
    assert speech_run.refusal(_args(**flags)) == message


def test_the_refusals_of_a_speech_run(speaks, hears):
    ok = dict(speech_out="reply.wav", mmproj=speaks)
    assert speech_run.refusal(_args(**ok)) is None
    assert speech_run.refusal(_args(**ok, tts=True, voice="male",
                                    speech_language="fr")) is None
    assert "Name it FILE.wav" in speech_run.refusal(
        _args(**{**ok, "speech_out": "reply.mp3"}))
    assert speech_run.refusal(_args(**ok, voice="male")).startswith(
        "--voice applies to --tts")
    assert speech_run.refusal(_args(**ok, speech_language="fr")).startswith(
        "--speech-language applies to --tts")
    for mmproj in (None, hears):
        assert speech_run.refusal(_args(
            speech_out="reply.wav", mmproj=mmproj)).startswith(
                "--speech-out needs a model with speech output")
    assert "takes no --image or --audio" in speech_run.refusal(
        _args(**ok, tts=True, audio="a.wav"))
    assert "Drop --system-prompt" in speech_run.refusal(
        _args(**ok, tts=True, system_prompt="Be brief."))
    assert "Drop --no-chat-template" in speech_run.refusal(
        _args(**ok, tts=True, no_chat_template=True))
    # A spoken reply takes media and a system prompt.
    assert speech_run.refusal(_args(**ok, audio="a.wav",
                                    system_prompt="Be brief.")) is None


def test_run_refuses_before_the_load(gguf, hears, capsys):
    assert cli.main([gguf, "--prompt", "hi", "--tts"]) == 2
    assert "--tts needs --speech-out FILE.wav" in capsys.readouterr().err
    assert cli.main([gguf, "--mmproj", hears, "--prompt", "hi",
                     "--speech-out", "reply.wav"]) == 2
    assert "needs a model with speech output" in capsys.readouterr().err


def test_run_dispatches_a_speech_run(gguf, speaks, monkeypatch):
    seen = []
    monkeypatch.setattr(speech_run, "run",
                        lambda args: seen.append(args) or 0)
    monkeypatch.setattr(cli, "_run_vlm", lambda args: 1)
    assert cli.main([gguf, "--mmproj", speaks, "--prompt", "Good morning.",
                     "--tts", "--voice", "male", "--speech-language", "fr",
                     "--speech-out", "reply.wav"]) == 0
    args = seen[0]
    assert (args.tts, args.voice, args.speech_language, args.speech_out) == (
        True, "male", "fr", "reply.wav")
    assert cli._run_mode(args) == "speech"


def test_the_wav_is_16_bit_mono_and_clipped(tmp_path):
    path = str(tmp_path / "a.wav")
    speech_run.write_wav(path, np.array([0.0, 0.5, 2.0, -2.0], np.float32),
                         24000)
    with wave.open(path, "rb") as f:
        assert (f.getnchannels(), f.getsampwidth(), f.getframerate()) == (
            1, 2, 24000)
        pcm = np.frombuffer(f.readframes(4), dtype="<i2")
    assert pcm.tolist() == [0, 16383, 32767, -32767]


class _Model:
    def __init__(self, speaks=True):
        self.language_model = object()
        self._speaks = speaks

    def speech_capabilities(self):
        return {"sample_rate": 24000} if self._speaks else None


@pytest.fixture
def stubbed(monkeypatch, gguf, speaks, tmp_path):
    """A run with the load, the fit check and the generator stubbed.
    Returns ``(run, calls)``: ``run(*flags)`` gives the exit code."""
    from gmlx.load import vlm
    from gmlx.models.lfm3 import speak
    from gmlx.stream import expert_streaming

    calls: dict = {"model": _Model(), "events": []}
    monkeypatch.setattr(vlm, "load_vlm_model", lambda *a, **k: (
        calls["model"], {}, SimpleNamespace(tokenizer=object())))
    monkeypatch.setattr(cli, "_apply_placement", lambda *a: None)
    monkeypatch.setattr(cli, "print_family_note", lambda *a: None)
    monkeypatch.setattr(cli, "unfit_request", lambda *a: None)
    monkeypatch.setattr(expert_streaming, "_resolve_prefill_step",
                        lambda lm, step: (step, False))

    def tts_prompt(tokenizer, text, voice=None, language=None):
        calls["tts"] = (text, voice, language)
        return "TTS " + text

    def stream_speech(model, processor, prompt, **kw):
        calls["prompt"], calls["kw"] = prompt, kw
        yield from calls["events"]

    monkeypatch.setattr(speak, "tts_prompt", tts_prompt)
    monkeypatch.setattr(speak, "stream_speech", stream_speech)

    def run(*flags):
        return cli.main([gguf, "--mmproj", speaks, *flags])

    return run, calls


def _events(frames=2):
    from gmlx.models.lfm3.speak import SpeechEvent

    chunk = np.full(1920, 0.25, dtype=np.float32)
    return ([SpeechEvent(token=5, text="Good ")]
            + [SpeechEvent(codes=np.zeros(12), audio=chunk)] * frames
            + [SpeechEvent(text="day.", finish_reason="stop", prompt_tokens=9,
                           text_tokens=1, frames=frames, prompt_seconds=0.5,
                           first_audio_seconds=0.02, generation_seconds=0.1,
                           peak_memory_gb=1.5)])


def test_tts_speaks_the_prompt_and_saves_the_file(stubbed, tmp_path, capsys):
    run, calls = stubbed
    calls["events"] = _events()
    out = tmp_path / "hello.wav"
    assert run("--tts", "--voice", "male", "--speech-language", "fr",
               "--prompt", "Bonjour.", "--speech-out", str(out),
               "--seed", "4", "--max-tokens", "50") == 0
    assert calls["tts"] == ("Bonjour.", "male", "fr")
    assert calls["prompt"] == "TTS Bonjour."
    assert calls["kw"]["mode"] == "audio" and calls["kw"]["max_tokens"] == 50
    assert calls["kw"]["key"] is not None
    with wave.open(str(out), "rb") as f:
        assert f.getnframes() == 3840 and f.getframerate() == 24000
    printed = capsys.readouterr().out
    assert "Good day." in printed
    assert f"Speech: 0.16 s saved to {out}" in printed
    assert "Prompt: 9 tokens, 18.000 tokens-per-sec" in printed
    assert "Generation: 1 text tokens and 2 speech frames, 30.000 per sec" \
        in printed
    assert "Speech speed: 1.6 times real time, first audio 20 ms" in printed
    assert "Peak memory: 1.500 GB" in printed


def test_a_spoken_reply_uses_the_cards_system_prompt(stubbed, tmp_path,
                                                     monkeypatch):
    from gmlx._vendor.mlx_vlm import prompt_utils
    from gmlx.models.lfm3 import speak

    run, calls = stubbed
    calls["events"] = _events()
    seen = {}

    def template(processor, config, messages, **kw):
        seen["messages"], seen["kw"] = messages, kw
        return "RENDERED"

    monkeypatch.setattr(prompt_utils, "apply_chat_template", template)
    assert run("--prompt", "Hello?", "--audio", "a.wav,b.wav",
               "--speech-out", str(tmp_path / "r.wav")) == 0
    assert seen["messages"] == [
        {"role": "system", "content": speak.INTERLEAVED_SYSTEM},
        {"role": "user", "content": "Hello?"}]
    assert seen["kw"]["num_audios"] == 2 and seen["kw"]["num_images"] == 0
    assert calls["prompt"] == "RENDERED"
    assert calls["kw"]["mode"] == "interleaved"
    assert calls["kw"]["audio"] == ["a.wav", "b.wav"]
    # The user's system prompt replaces the card's.
    assert run("--prompt", "Hello?", "--system-prompt", "Speak slowly.",
               "--speech-out", str(tmp_path / "r.wav")) == 0
    assert seen["messages"][0]["content"] == "Speak slowly."


def test_a_reply_with_no_speech_writes_no_file(stubbed, tmp_path, capsys):
    run, calls = stubbed
    calls["events"] = _events(frames=0)
    out = tmp_path / "none.wav"
    assert run("--tts", "--prompt", "x", "--speech-out", str(out)) == 0
    assert not out.exists()
    assert "The model gave no speech, so no file was written." in \
        capsys.readouterr().out


def test_a_model_gmlx_cannot_make_speak_is_refused_after_the_load(
        stubbed, tmp_path, capsys):
    run, calls = stubbed
    calls["model"] = _Model(speaks=False)
    assert run("--tts", "--prompt", "x",
               "--speech-out", str(tmp_path / "x.wav")) == 1
    assert "gmlx has no speech output for this model" in \
        capsys.readouterr().err
