"""Audio on a model that hears it through its own speech encoder (Nemotron
3.5 Transcribe), on /v1/chat/completions and /v1/responses.

The route wraps decode each clip, runs the context and memory checks, then
encodes the clips and hands them to the generator's tokenization in place
of the request's audio. The encoder itself is stubbed here; the encode job
is tested beside the model (tests/models/test_nemotron_transcribe_input.py).
"""

from __future__ import annotations

import asyncio
import base64
import importlib
import json
import math
import types
from pathlib import Path

import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm")

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from transformers.utils.chat_template_utils import render_jinja_template  # noqa: E402

from gmlx.models.nemotron_h.transcribe_model import EncodedClip  # noqa: E402
from gmlx.serve import mem_preflight  # noqa: E402
from gmlx.serve.engine_jobs import JobCancelled  # noqa: E402
from gmlx.serve.patches import _common as sp_common  # noqa: E402
from gmlx.serve.patches import native_audio  # noqa: E402
from gmlx.serve.patches.audio_cascade import install_audio_cascade  # noqa: E402
from gmlx.serve.patches.render import install_chat_image_placement  # noqa: E402

_OPENAI = importlib.import_module("gmlx._vendor.mlx_vlm.server.openai")
_GEN = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")
_PROMPT_UTILS = importlib.import_module("gmlx._vendor.mlx_vlm.prompt_utils")
_RS = importlib.import_module("gmlx._vendor.mlx_vlm.server.responses_state")
_APP = importlib.import_module("gmlx._vendor.mlx_vlm.server.app")
_RUNTIME = importlib.import_module("gmlx._vendor.mlx_vlm.server.runtime").runtime
_TEMPLATE = (Path(__file__).parents[1] / "fixtures" / "nemotron_transcribe"
             / "chat_template.jinja").read_text()
RATE = 16000


class _Tokenizer:
    """One id per character."""

    def encode(self, text, add_special_tokens=True):
        return [ord(c) for c in text]


class _Processor:
    def __init__(self):
        self.chat_template = _TEMPLATE
        self.tokenizer = _Tokenizer()

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kw):
        out = render_jinja_template([messages], chat_template=_TEMPLATE,
                                    add_generation_prompt=add_generation_prompt,
                                    **kw)
        rendered = out[0] if isinstance(out, tuple) else out
        return rendered[0] if isinstance(rendered, list) else rendered


class _Tok:
    text, finish_reason, token_count = "ok", "stop", 1


def _stock_preprocess(self, prompt, images=None, audio=None, videos=None):
    self.seen.append((prompt, audio))
    ids = self.processor.tokenizer.encode(prompt)
    return {"input_ids": mx.array([ids]), "attention_mask": mx.ones((1, len(ids)))}


class _RG:
    _cpu_preprocess = native_audio._preprocess(_stock_preprocess)

    def __init__(self):
        self.processor = _Processor()
        self.seen = []

    def generate(self, prompt, images=None, audio=None, args=None, videos=None):
        self._cpu_preprocess(prompt, images, audio)
        return types.SimpleNamespace(prompt_tokens=1, uid=1), iter([_Tok()])


def _tokens(samples: int) -> int:
    return math.ceil(samples / 1280)


def _clip(seconds: float):
    """An input_audio part whose decoded length is ``seconds``."""
    data = base64.b64encode(str(int(seconds * RATE)).encode()).decode()
    return {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}


@pytest.fixture
def native(monkeypatch):
    routes = sp_common._snapshot_routes(_APP.app)
    monkeypatch.setattr(_OPENAI, "extract_text_from_content",
                        _OPENAI.extract_text_from_content)
    monkeypatch.setattr(_GEN.ResponseGenerator, "_cpu_preprocess",
                        _GEN.ResponseGenerator._cpu_preprocess)
    for name in ("get_chat_template", "apply_chat_template"):
        monkeypatch.setattr(_PROMPT_UTILS, name, getattr(_PROMPT_UTILS, name))
    monkeypatch.setattr(_RS, "_append_response_item_to_prompt",
                        _RS._append_response_item_to_prompt)
    monkeypatch.setattr(_OPENAI, "_response_items_audio",
                        _OPENAI._response_items_audio)
    rg = _RG()
    monkeypatch.setattr(_RUNTIME, "response_generator", rg, raising=False)
    model = types.SimpleNamespace(audio_preflight=lambda waves: (
        sum(_tokens(len(w)) for w in waves), 0))
    loaded = (model, rg.processor, {"model_type": "nemotron_h_transcribe"})
    monkeypatch.setattr(_OPENAI, "get_cached_model", lambda *a, **k: loaded)
    monkeypatch.setattr(_APP, "get_cached_model", lambda *a, **k: loaded)
    state = types.SimpleNamespace(native=True, events=[], encoded=[], fail=None)
    monkeypatch.setattr(native_audio, "_is_native", lambda m: state.native)
    monkeypatch.setattr(native_audio, "_decode", lambda data: np.zeros(
        int(data.decode()), np.float32))

    def preflight(rg_, prompt, audio=None, held_bytes=0, **kw):
        state.events.append("preflight")
        if isinstance(state.fail, _GEN.PromptTooLongError):
            raise state.fail

    monkeypatch.setattr(mem_preflight, "preflight_prompt_memory", preflight)

    def encode(rg_, model_, waves, stop, timeout_s, **kw):
        state.events.append("encode")
        if isinstance(state.fail, JobCancelled):
            raise state.fail
        clips = [EncodedClip.of(len(w), f"d{len(w)}", mx.zeros((_tokens(len(w)), 4)),
                                np.zeros((_tokens(len(w)), 8))) for w in waves]
        state.encoded.append(clips)
        captures = native_audio.CAPTURE.get()
        if captures is not None:
            captures.extend(f"capture{len(w)}" for w in waves)
        return clips

    monkeypatch.setattr(native_audio, "encode", encode)
    install_chat_image_placement()
    install_audio_cascade()      # installed first in serve, as here
    native_audio.install_native_audio()
    from fastapi.testclient import TestClient

    state.client, state.rg = TestClient(_APP.app), rg
    yield state
    sp_common._restore_routes(_APP.app, routes)


def _chat(state, messages, **body):
    return state.client.post("/v1/chat/completions", json={
        "model": "m", "messages": messages, "max_tokens": 1, **body})


def test_the_generator_gets_the_encoded_clips(native):
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "What is said?"}, _clip(1.0)]}]
    r = _chat(native, messages)
    assert r.status_code == 200, r.text
    assert native.events == ["preflight", "encode"]
    prompt, audio = native.rg.seen[-1]
    assert audio == native.encoded[-1] and isinstance(audio[0], EncodedClip)
    assert prompt.count("<|audio|>") == 1 and "What is said? <|audio|>" in prompt


def test_clips_of_every_user_turn_encode_in_order(native):
    messages = [
        {"role": "user", "content": [_clip(1.0)]},
        {"role": "assistant", "content": "One."},
        {"role": "user", "content": [{"type": "text", "text": "And?"}, _clip(2.0)]},
    ]
    assert _chat(native, messages).status_code == 200
    prompt, audio = native.rg.seen[-1]
    assert [len(c) for c in audio] == [RATE, 2 * RATE]
    # Each clip renders in the turn that sent it, so the next turn's prompt
    # starts with this one's.
    assert "user\n<|audio|><|im_end|>" in prompt
    assert "user\nAnd? <|audio|><|im_end|>" in prompt


def test_a_seventh_clip_is_refused_before_any_encode(native):
    messages = [{"role": "user", "content": [_clip(0.5)] * 7}]
    r = _chat(native, messages)
    assert r.status_code == 400
    assert "limit is 6" in r.json()["error"]["message"]
    assert native.events == []


def test_past_the_context_is_refused_before_any_encode(native):
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "x" * 21_000}, _clip(3600.0)]}]
    r = _chat(native, messages)
    assert r.status_code == 400
    assert "65,536" in r.json()["error"]["message"]
    assert native.events == []


def test_a_memory_refusal_spends_no_encoder_time(native):
    native.fail = _GEN.PromptTooLongError("prompt is too long for the memory")
    r = _chat(native, [{"role": "user", "content": [_clip(1.0)]}])
    assert r.status_code == 400 and "memory" in r.json()["error"]["message"]
    assert native.events == ["preflight"]


def test_a_cancelled_encode_answers_499(native):
    native.fail = JobCancelled("the audio encoder was cancelled")
    r = _chat(native, [{"role": "user", "content": [_clip(1.0)]}])
    assert r.status_code == 499
    assert native.events == ["preflight", "encode"] and not native.rg.seen


def test_other_models_keep_the_stock_route(native):
    native.native = False
    r = _chat(native, [{"role": "user", "content": [_clip(1.0)]}])
    assert r.status_code == 200, r.text
    assert native.events == []
    _, audio = native.rg.seen[-1]
    assert not isinstance(audio[0], EncodedClip)


def test_text_only_requests_skip_the_wrap(native):
    r = _chat(native, [{"role": "user", "content": "hello"}])
    assert r.status_code == 200
    assert native.events == [] and native.rg.seen[-1][1] is None


def _salt(raw):
    """The APC salt the stock generate call gives a request whose
    tokenization returns ``raw``."""
    rg = object.__new__(_GEN.ResponseGenerator)
    queued = []

    class _Queue:
        def put(self, request):
            queued.append(request)
            request.rqueue.put(_GEN.GenerationContext(uid=1, prompt_tokens=1))

    rg.__dict__.update(
        wait_until_ready=lambda: None, draft_model=None, apc_mode="on",
        _preprocess_request=lambda *a: raw, requests=_Queue(),
        _make_thinking_budget_criteria=lambda *a: None, _cancel=None,
        model=types.SimpleNamespace(), processor=None)
    rg.generate("p")
    return queued[0].apc_semantic_hash


def test_the_prefix_cache_keys_encoded_audio_by_its_digests():
    ids = {"input_ids": mx.array([[1, 2, 3]])}
    a = _salt({**ids, "audio_digests": ("a",)})
    assert a == _salt({**ids, "audio_digests": ("a",)})
    assert a != _salt({**ids, "audio_digests": ("b",)})
    assert a != _salt({**ids, "audio_digests": ("a", "a")})
    assert a != _salt(ids)


def _respond(state, content, **body):
    return state.client.post("/v1/responses", json={
        "model": "m", "max_output_tokens": 1,
        "input": [{"type": "message", "role": "user", "content": content}],
        **body})


def _input_clip(seconds: float):
    return {**_clip(seconds), "type": "input_audio"}


def test_responses_hear_the_clips_in_place(native):
    r = _respond(native, [{"type": "input_text", "text": "What is said?"},
                          _input_clip(1.0)])
    assert r.status_code == 200, r.text
    assert native.events == ["preflight", "encode"]
    prompt, audio = native.rg.seen[-1]
    assert audio == native.encoded[-1] and isinstance(audio[0], EncodedClip)
    assert prompt.count("<|audio|>") == 1 and "What is said? <|audio|>" in prompt


def test_a_responses_follow_up_keeps_the_earlier_clip_in_its_turn(native):
    first = _respond(native, [_input_clip(1.0)])
    assert first.status_code == 200, first.text
    r = _respond(native, [{"type": "input_text", "text": "And?"}],
                 previous_response_id=first.json()["id"])
    assert r.status_code == 200, r.text
    prompt, audio = native.rg.seen[-1]
    assert [len(c) for c in audio] == [RATE]
    assert "user\n<|audio|><|im_end|>" in prompt
    assert prompt.count("<|audio|>") == 1 and "user\nAnd?<|im_end|>" in prompt


def test_counting_responses_tokens_decodes_without_an_encode(native):
    r = native.client.post("/v1/responses/input_tokens", json={
        "model": "m",
        "input": [{"type": "message", "role": "user",
                   "content": [_input_clip(2.0)]}]})
    assert r.status_code == 200, r.text
    assert native.events == []
    prompt, audio = native.rg.seen[-1]
    assert [len(c) for c in audio] == [2 * RATE]
    assert not isinstance(audio[0], EncodedClip)
    assert prompt.count("<|audio|>") == 1


def test_responses_audio_on_other_models_is_still_refused(native):
    native.native = False
    r = _respond(native, [_input_clip(1.0)])
    assert r.status_code == 400
    assert "takes no audio" in r.json()["error"]["message"]
    assert native.events == []


# -- word timestamps on chat -------------------------------------------------


@pytest.fixture
def stamps(native, monkeypatch):
    from gmlx.serve import word_timestamps

    seen = []

    async def chat_result(http_request, model_id, capture, transcript):
        seen.append((capture, transcript))
        return {"words": [], "diarization": [], "error": None,
                "speaker_tag_to_diarization_speaker": {}}

    monkeypatch.setattr(word_timestamps, "chat_result", chat_result)
    model = _APP.get_cached_model("m")[0]
    model.audio_tower = types.SimpleNamespace(cfg=types.SimpleNamespace(
        asr=types.SimpleNamespace(d_model=4), num_speakers=8))
    native.stamped = seen
    return native


def test_chat_word_timestamps_ride_the_completion(stamps):
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hi"}, _clip(1.0)]}]
    # The template thinks by default, so the reply would land in reasoning
    # once serve's template-kwargs patch is installed; transcription runs
    # with thinking off.
    for opt in ({"mm_processor_kwargs": {"capture_ctc_timestamps": True}},
                {"timestamp_granularities": ["word"]}):
        r = _chat(stamps, messages, enable_thinking=False, **opt)
        assert r.status_code == 200, r.text
        assert r.json()["ctc_timestamps"]["error"] is None
    assert stamps.stamped == [("capture16000", "ok")] * 2
    r = _chat(stamps, messages)
    assert "ctc_timestamps" not in r.json() and len(stamps.stamped) == 2
    assert native_audio.CAPTURE.get() is None


def test_chat_word_timestamps_need_one_clip_and_one_choice(stamps):
    opt = {"timestamp_granularities": ["word"]}
    two = [{"role": "user", "content": [_clip(1.0), _clip(2.0)]}]
    r = _chat(stamps, two, **opt)
    assert r.status_code == 400 and "exactly one audio clip" in r.text
    one = [{"role": "user", "content": [_clip(1.0)]}]
    r = _chat(stamps, one, n=2, **opt)
    assert r.status_code == 400 and "n = 1" in r.text
    assert stamps.encoded == []


def _sse(*chunks):
    async def body():
        for c in chunks:
            yield c if isinstance(c, str) else f"data: {json.dumps(c)}\n\n"
        yield "data: [DONE]\n\n"
    return body()


def test_a_stream_holds_only_its_final_chunk_for_the_timestamps(monkeypatch):
    from gmlx.serve import word_timestamps

    order = []

    async def chat_result(http_request, model_id, capture, transcript):
        order.append(("align", transcript))
        return {"words": [], "error": None}

    monkeypatch.setattr(word_timestamps, "chat_result", chat_result)

    def delta(text=None, finish=None):
        d = {"role": "assistant"} if text is None else {"content": text}
        return {"id": "c", "choices": [{"index": 0, "delta": d, "finish_reason": finish}]}

    async def collect(body):
        out = []
        async for event in word_timestamps._stream(body, None, "m", "cap"):
            order.append("event")
            out.append(event)
        return out

    usage = {"id": "c", "choices": [], "usage": {"total_tokens": 3}}
    out = asyncio.run(collect(_sse(delta("Hel"), delta("lo"), delta(finish="stop"), usage)))
    events = [e[len("data: "):].strip() for e in out]
    assert json.loads(events[2])["ctc_timestamps"] == {"words": [], "error": None}
    assert json.loads(events[2])["choices"][0]["finish_reason"] == "stop"
    assert "usage" in json.loads(events[3]) and events[4] == "[DONE]"
    assert order == ["event", "event", ("align", "Hello"), "event", "event", "event"]
    # Text sharing the finish chunk goes out before the wait.
    order.clear()
    out = asyncio.run(collect(_sse(delta("Hi"), delta(" there", "stop"))))
    first = [json.loads(e[len("data: "):]) for e in out[:3]]
    assert first[1]["choices"][0] == {"index": 0, "delta": {"content": " there"},
                                      "finish_reason": None}
    assert first[2]["choices"][0]["delta"] == {} and "ctc_timestamps" in first[2]
