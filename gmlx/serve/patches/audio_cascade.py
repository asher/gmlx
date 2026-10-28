"""Audio on a model that reads it through a speech companion (Granite
Switch), on the chat completions and Responses routes.

A route wrap transcribes each clip of the request on the companion's worker
before the stock endpoint runs. Each message with audio then renders as the
text the model's template builds from it, with the audio marker in place of
each clip (:func:`gmlx.asr.marked_text`). The transcripts ride a context
variable to the response generator's tokenization, where
:func:`gmlx.asr.splice` replaces each marker id with that clip's transcript
ids, as IBM's cascade does.

The Responses route rebuilds the prompt from its stored chain, so an earlier
turn's clip renders again on every later turn. Its message conversion takes
the transcripts from the request state, and the service's cache keeps the
companion from transcribing the same clip twice
(:mod:`gmlx.serve.asr_service`).
"""

from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import hashlib
import importlib
import io

from gmlx import asr

from ._common import _CHAT_PATHS, _error_content, _wrap_post_routes

_FLAG = "_gmlx_audio_cascade"
_RESPONSES_PATHS = ("/responses", "/v1/responses")


@dataclasses.dataclass
class _Cascade:
    companion: str
    texts: dict          # clip digest -> transcript
    order: list          # transcripts in prompt marker order


_STATE: contextvars.ContextVar[_Cascade | None] = contextvars.ContextVar(
    "gmlx_audio_cascade", default=None)


class _Refused(Exception):
    """A client error, answered with a 400."""


def _companion_for(model_field) -> tuple[bool, str | None]:
    """Whether the request's model reads audio through a companion, and
    the companion it pairs with."""
    from .. import bridge_vlm as serving

    try:
        path, spec = serving.resolve_request_model(
            model_field, profile_field=serving.get_request_profile())
    except Exception:  # noqa: BLE001  the stock route answers the error
        return False, None
    if not asr.pairs_with_companion(path):
        return False, None
    return True, asr.companion_path(path, getattr(spec, "asr", None),
                                    getattr(spec, "no_asr", False))


def _plain(part):
    if hasattr(part, "model_dump"):
        return part.model_dump(exclude_none=True)
    return part


def _clip_bytes(part) -> bytes:
    from ..media_sinks import _AUDIO_WANT, reference_bytes
    from .media_gate import MediaRefused

    data = part.get("input_audio") if isinstance(part, dict) else None
    if not isinstance(data, dict) or not data.get("data"):
        raise _Refused("an audio part needs input_audio.data")
    openai = importlib.import_module("gmlx._vendor.mlx_vlm.server.openai")
    try:
        got = openai._decode_input_audio_data(data)
        if isinstance(got, io.BytesIO):
            return got.getvalue()
        return reference_bytes(str(got), "audio", "input_audio.data", _AUDIO_WANT)
    except MediaRefused as e:
        raise _Refused(str(e)) from None
    except Exception as e:  # noqa: BLE001  HTTPException from the decode
        raise _Refused(str(getattr(e, "detail", e))) from None


def _digest(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


async def _transcribe_all(path: str, clips: list[bytes]) -> dict:
    from .. import asr_service

    texts: dict = {}
    for data in clips:
        key = _digest(data)
        if key in texts:
            continue
        try:
            text = await asyncio.to_thread(asr_service.transcribe, path, data, key,
                                           "input_audio")
            asr.check_transcript(text)
        except (asr_service.ASRRequestError, ValueError) as e:
            raise _Refused(str(e)) from None
        texts[key] = text
    return texts


def _audio_parts(value) -> list:
    """Every audio part inside ``value``, a request body or part of one."""
    value = _plain(value)
    if isinstance(value, dict):
        if asr.is_audio_part(value):
            return [value]
        return [p for v in value.values() for p in _audio_parts(v)]
    if isinstance(value, list):
        return [p for v in value for p in _audio_parts(v)]
    return []


def _texts_hold_marker(value) -> bool:
    value = _plain(value)
    if isinstance(value, str):
        return asr.MARKER in value
    if isinstance(value, dict):
        if asr.is_audio_part(value):
            return False
        return any(_texts_hold_marker(v) for v in value.values())
    if isinstance(value, list):
        return any(_texts_hold_marker(v) for v in value)
    return False


async def _prepare(model_field, parts: list, texts_with_marker: bool):
    """The request's cascade state, None for a model that takes audio
    another way. Raises _Refused for a client error."""
    pairs, path = _companion_for(model_field)
    if not pairs:
        return None
    if path is None:
        raise _Refused(asr.NO_COMPANION)
    if texts_with_marker:
        raise _Refused(f"message text holds {asr.MARKER}, which this model "
                       "reads as the place of an audio clip")
    clips = [_clip_bytes(p) for p in parts]
    return _Cascade(path, await _transcribe_all(path, clips), [])


def _refusal(http_request, e: _Refused):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=400, content=_error_content(
        http_request.url.path, 400, "invalid_request_error", str(e)))


def _flatten_message(message, state: _Cascade) -> None:
    content = [_plain(p) for p in message.content]
    for part in content:
        if asr.is_audio_part(part):
            state.order.append(state.texts[_digest(_clip_bytes(part))])
    message.content = asr.marked_text(content)


def _chat_wrap(original):
    async def endpoint(request, http_request):
        messages = list(getattr(request, "messages", None) or ())
        with_audio = [m for m in messages if asr.has_audio(
            [_plain(p) for p in m.content] if isinstance(m.content, list) else None)]
        if not with_audio:
            return await original(request, http_request)
        try:
            state = await _prepare(
                request.model, _audio_parts([m.content for m in with_audio]),
                any(_texts_hold_marker(m.content) for m in messages))
            if state is None:
                return await original(request, http_request)
            for message in with_audio:
                _flatten_message(message, state)
        except _Refused as e:
            return _refusal(http_request, e)
        _STATE.set(state)        # read in the generate call, maybe in the stream
        return await original(request, http_request)
    return endpoint


def _responses_wrap(original):
    async def endpoint(request):
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001  the stock route answers the error
            return await original(request)
        if not isinstance(body, dict):
            return await original(request)
        openai = importlib.import_module("gmlx._vendor.mlx_vlm.server.openai")
        try:
            chain = openai._response_chain_items(body.get("previous_response_id"))
        except Exception:  # noqa: BLE001
            chain = []
        current = _audio_parts(body.get("input"))
        parts = _audio_parts(chain) + current
        if not parts:
            return await original(request)
        try:
            state = await _prepare(body.get("model"), parts,
                                   _texts_hold_marker(body.get("input")))
            if state is None:
                # The stock route drops every clip.
                raise _Refused("/v1/responses takes no audio for this model. "
                               "Send audio to /v1/chat/completions.")
        except _Refused as e:
            return _refusal(request, e)
        _STATE.set(state)
        return await original(request)
    return endpoint


def _item_converter(stock):
    """The Responses message conversion, with each message that holds audio
    rendered as marked text while a cascade request is on."""
    def _append_response_item_to_prompt(item, chat_messages, images):
        state = _STATE.get()
        content = item.get("content") if isinstance(item, dict) else None
        if (state is not None and isinstance(content, list)
                and item.get("type") == "message"):
            plain = [_plain(p) for p in content]
            if asr.has_audio(plain):
                for part in plain:
                    if asr.is_audio_part(part):
                        state.order.append(state.texts[_digest(_clip_bytes(part))])
                item = {**item, "content": asr.marked_text(plain)}
        return stock(item, chat_messages, images)
    _append_response_item_to_prompt.__wrapped__ = stock  # type: ignore[attr-defined]
    setattr(_append_response_item_to_prompt, _FLAG, True)
    return _append_response_item_to_prompt


def _preprocess(stock):
    """The response generator's tokenization, with the request's markers
    replaced by its transcripts."""
    def _cpu_preprocess(self, *args, **kwargs):
        raw = stock(self, *args, **kwargs)
        state = _STATE.get()
        if state is None or not state.order or not isinstance(raw, dict):
            return raw
        import mlx.core as mx

        tok = getattr(self.processor, "tokenizer", self.processor)
        marker = asr.marker_id(tok)
        if marker is None:
            raise ValueError("the model's tokenizer has no audio marker token")
        ids = asr.splice(raw["input_ids"].reshape(-1).tolist(), marker,
                         state.order,
                         lambda t: tok.encode(t, add_special_tokens=False))
        raw = dict(raw)
        raw["input_ids"] = mx.array([ids], dtype=raw["input_ids"].dtype)
        if raw.get("attention_mask") is not None:
            raw["attention_mask"] = mx.ones((1, len(ids)),
                                            dtype=raw["attention_mask"].dtype)
        return raw
    _cpu_preprocess.__wrapped__ = stock  # type: ignore[attr-defined]
    setattr(_cpu_preprocess, _FLAG, True)
    return _cpu_preprocess


def install_audio_cascade() -> None:
    """Install the route wraps, the Responses conversion and the splice.
    Idempotent."""
    from gmlx.upstream.patching import wrapped_with

    rs = importlib.import_module("gmlx._vendor.mlx_vlm.server.responses_state")
    if not wrapped_with(rs._append_response_item_to_prompt, _FLAG):
        rs._append_response_item_to_prompt = _item_converter(
            rs._append_response_item_to_prompt)
    gen = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")
    cls = gen.ResponseGenerator
    if not wrapped_with(cls._cpu_preprocess, _FLAG):
        cls._cpu_preprocess = _preprocess(cls._cpu_preprocess)
    app = importlib.import_module("gmlx._vendor.mlx_vlm.server.app").app
    _wrap_post_routes(app, _CHAT_PATHS, _FLAG, _chat_wrap)
    _wrap_post_routes(app, _RESPONSES_PATHS, _FLAG, _responses_wrap)
