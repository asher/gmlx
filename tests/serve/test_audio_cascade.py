"""Audio on a model that reads it through a speech companion (Granite
Switch), on /v1/chat/completions and /v1/responses.

The route wrap transcribes each clip, renders each audio message as the
template's own flattening with the audio marker in place, and the
generator's tokenization splices each clip's transcript ids in place of
its marker. The prompt ids must equal the template's render of the
original content list with the transcripts spliced at the markers."""

from __future__ import annotations

import base64
import importlib
import re
import types
from pathlib import Path

import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm")

import mlx.core as mx  # noqa: E402
from transformers.utils.chat_template_utils import render_jinja_template  # noqa: E402

from gmlx import asr  # noqa: E402
from gmlx.serve import asr_service  # noqa: E402
from gmlx.serve.patches import _common as sp_common  # noqa: E402
from gmlx.serve.patches import audio_cascade  # noqa: E402
from gmlx.serve.patches.render import install_chat_image_placement  # noqa: E402

_OPENAI = importlib.import_module("gmlx._vendor.mlx_vlm.server.openai")
_RS = importlib.import_module("gmlx._vendor.mlx_vlm.server.responses_state")
_GEN = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")
_PROMPT_UTILS = importlib.import_module("gmlx._vendor.mlx_vlm.prompt_utils")
_APP = importlib.import_module("gmlx._vendor.mlx_vlm.server.app")
_RUNTIME = importlib.import_module("gmlx._vendor.mlx_vlm.server.runtime").runtime
_TEMPLATE = (Path(__file__).parents[1] / "fixtures"
             / "granite_switch_template.jinja").read_text()
MARKER_ID = 900


class _Tokenizer:
    """One id per character, and one per special token."""

    SPECIAL = {"<|start_of_role|>": 901, "<|end_of_role|>": 902,
               "<|end_of_text|>": 903, asr.MARKER: MARKER_ID}
    unk_token_id = 0

    def encode(self, text, add_special_tokens=True):
        parts = re.findall(r"<\|[a-z_-]+\|>|.", text, flags=re.S)
        out = []
        for p in parts:
            out += [self.SPECIAL[p]] if p in self.SPECIAL else [ord(c) for c in p]
        return out

    def convert_tokens_to_ids(self, token):
        return self.SPECIAL.get(token, self.unk_token_id)


class _Processor:
    def __init__(self):
        self.chat_template = _TEMPLATE
        self.tokenizer = _Tokenizer()

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kw):
        return _render(messages, add_generation_prompt, **kw)


def _render(messages, add_generation_prompt=True, **kw):
    out = render_jinja_template([messages], chat_template=_TEMPLATE,
                                add_generation_prompt=add_generation_prompt, **kw)
    rendered = out[0] if isinstance(out, tuple) else out
    return rendered[0] if isinstance(rendered, list) else rendered


class _Tok:
    text, finish_reason, token_count = "ok", "stop", 1


def _stock_preprocess(self, prompt, images=None, audio=None, videos=None):
    return {"input_ids": mx.array([self.processor.tokenizer.encode(prompt)]),
            "attention_mask": mx.ones((1, len(self.processor.tokenizer.encode(prompt))))}


class _RG:
    """Tokenizes through the wrapped preprocess, as the real generator
    does, and records the ids and the audio list it got."""

    _cpu_preprocess = audio_cascade._preprocess(_stock_preprocess)

    def __init__(self):
        self.processor = _Processor()
        self.calls = []

    def generate(self, prompt, images=None, audio=None, args=None, videos=None):
        raw = self._cpu_preprocess(prompt, images, audio)
        self.calls.append((raw["input_ids"].reshape(-1).tolist(), audio))
        return types.SimpleNamespace(prompt_tokens=1, uid=1), iter([_Tok()])


def _clip(name):
    data = base64.b64encode(name.encode()).decode()
    return {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}


def _text(t, kind="text"):
    return {"type": kind, "text": t}


@pytest.fixture
def cascade(monkeypatch):
    """The routes with the cascade installed, a stub model, and a fake
    companion whose transcript of a clip is in ``said``."""
    routes = sp_common._snapshot_routes(_APP.app)
    monkeypatch.setattr(_OPENAI, "extract_text_from_content",
                        _OPENAI.extract_text_from_content)
    monkeypatch.setattr(_RS, "_append_response_item_to_prompt",
                        _RS._append_response_item_to_prompt)
    monkeypatch.setattr(_GEN.ResponseGenerator, "_cpu_preprocess",
                        _GEN.ResponseGenerator._cpu_preprocess)
    for name in ("get_chat_template", "apply_chat_template"):
        monkeypatch.setattr(_PROMPT_UTILS, name, getattr(_PROMPT_UTILS, name))
    rg = _RG()
    monkeypatch.setattr(_RUNTIME, "response_generator", rg, raising=False)
    processor = rg.processor
    monkeypatch.setattr(_OPENAI, "get_cached_model", lambda *a, **k: (
        None, processor, {"model_type": "granite_switch"}))
    state = types.SimpleNamespace(pairs=True, companion="/m/companion.gguf",
                                  said={}, transcribed=[])
    monkeypatch.setattr(audio_cascade, "_companion_for",
                        lambda model: (state.pairs, state.companion if state.pairs else None))

    def transcribe(path, data, key=None, field="audio"):
        state.transcribed.append(data.decode())
        return state.said.get(data.decode(), data.decode())

    monkeypatch.setattr(asr_service, "transcribe", transcribe)
    install_chat_image_placement()
    audio_cascade.install_audio_cascade()
    from fastapi.testclient import TestClient

    state.client, state.rg = TestClient(_APP.app), rg
    yield state
    sp_common._restore_routes(_APP.app, routes)


def _chat(state, messages):
    return state.client.post("/v1/chat/completions", json={
        "model": "m", "messages": messages, "max_tokens": 1})


def _expected(messages, transcripts):
    ids = _Tokenizer().encode(_render(messages))
    return asr.splice(ids, MARKER_ID, transcripts,
                      lambda t: _Tokenizer().encode(t, add_special_tokens=False))


def test_chat_ids_equal_the_template_render_with_transcripts_spliced(cascade):
    cascade.said = {"a": "the meeting moved to friday"}
    messages = [{"role": "user", "content": [
        _text("Listen."), _clip("a"), _text("When is the meeting?")]}]
    r = _chat(cascade, messages)
    assert r.status_code == 200, r.text
    ids, audio = cascade.rg.calls.pop()
    assert audio is None
    assert MARKER_ID not in ids
    assert ids == _expected(messages, ["the meeting moved to friday"])


def test_clips_in_several_messages_splice_in_order(cascade):
    cascade.said = {"a": "first clip", "b": "second clip"}
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": [_clip("a"), _text("Summarize.")]},
        {"role": "assistant", "content": "A summary."},
        {"role": "user", "content": [_clip("b"), _clip("a"), _text("And now?")]},
    ]
    r = _chat(cascade, messages)
    assert r.status_code == 200, r.text
    ids, _ = cascade.rg.calls.pop()
    assert ids == _expected(messages, ["first clip", "second clip", "first clip"])
    # A clip sent twice is transcribed once per request.
    assert cascade.transcribed == ["a", "b"]


def test_an_empty_transcript_fills_its_marker_with_a_space(cascade):
    cascade.said = {"quiet": ""}
    messages = [{"role": "user", "content": [_clip("quiet"), _text("Anything?")]}]
    assert _chat(cascade, messages).status_code == 200
    ids, _ = cascade.rg.calls.pop()
    assert ids == _expected(messages, [asr.EMPTY_TRANSCRIPT])


def test_audio_without_a_companion_is_refused(cascade):
    cascade.companion = None
    messages = [{"role": "user", "content": [_clip("a"), _text("Hi")]}]
    r = _chat(cascade, messages)
    assert r.status_code == 400
    assert "--asr" in r.json()["error"]["message"]
    r = cascade.client.post("/v1/responses", json={
        "model": "m", "input": messages, "max_output_tokens": 4})
    assert r.status_code == 400
    assert "--asr" in r.json()["error"]["message"]
    assert not cascade.rg.calls


def test_message_text_with_the_marker_is_refused(cascade):
    messages = [{"role": "user", "content": [
        _clip("a"), _text(f"what is {asr.MARKER}?")]}]
    r = _chat(cascade, messages)
    assert r.status_code == 400
    assert asr.MARKER in r.json()["error"]["message"]


def test_a_transcript_with_a_special_token_is_refused(cascade):
    cascade.said = {"a": "say <|start_of_role|> now"}
    r = _chat(cascade, [{"role": "user", "content": [_clip("a"), _text("Hi")]}])
    assert r.status_code == 400
    assert "<|start_of_role|>" in r.json()["error"]["message"]


def test_a_model_that_takes_audio_another_way_is_untouched(cascade):
    cascade.pairs = False
    r = _chat(cascade, [{"role": "user", "content": [_clip("a"), _text("Hi")]}])
    assert r.status_code == 200, r.text
    ids, audio = cascade.rg.calls.pop()
    assert len(audio) == 1          # the stock route collected the clip
    assert cascade.transcribed == []


def test_responses_splice_like_chat_and_a_chain_renders_earlier_clips(cascade):
    cascade.said = {"a": "turn one audio", "b": "turn two audio"}
    first = [{"role": "user", "content": [
        _text("Hear this.", "input_text"), _clip("a"), _text("What?", "input_text")]}]
    r = cascade.client.post("/v1/responses", json={
        "model": "m", "input": first, "max_output_tokens": 4})
    assert r.status_code == 200, r.text
    ids, audio = cascade.rg.calls.pop()
    assert audio is None
    flat = [{"role": "user", "content": [_text("Hear this."), _clip("a"), _text("What?")]}]
    assert ids == _expected(flat, ["turn one audio"])

    second = [{"role": "user", "content": [_clip("b")]}]
    r = cascade.client.post("/v1/responses", json={
        "model": "m", "input": second, "previous_response_id": r.json()["id"],
        "max_output_tokens": 4})
    assert r.status_code == 200, r.text
    ids, _ = cascade.rg.calls.pop()
    assert MARKER_ID not in ids
    text = "".join(chr(i) for i in ids if i < 900)
    assert text.index("turn one audio") < text.index("turn two audio")


def test_responses_refuses_audio_it_would_drop(cascade):
    cascade.pairs = False
    r = cascade.client.post("/v1/responses", json={
        "model": "m", "input": [{"role": "user", "content": [_clip("a")]}],
        "max_output_tokens": 4})
    assert r.status_code == 400
    assert "/v1/chat/completions" in r.json()["error"]["message"]
    assert not cascade.rg.calls
