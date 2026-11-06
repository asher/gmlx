"""Word timestamps and speakers for a served model that hears audio through
its own speech encoder (Nemotron 3.5 Transcribe).

A request that asks for them sets ``native_audio.CAPTURE``; the encoder then
keeps each clip's ASR states and speaker activity. Once the decoder is done,
:func:`align` runs the word timestamp head over the capture as a step job on
the engine and NeMo's aligner (:mod:`gmlx.models.nemotron_h.align`) off it,
holding the model resident until the words are timed.

Chat completions return the result as NeMo's ``ctc_timestamps`` object
(``ctc_serving.py``): :func:`public`. A chat request opts in with
``mm_processor_kwargs: {"capture_ctc_timestamps": true}`` or
``timestamp_granularities: ["word"]``.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import threading

_log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
OPT_IN = "capture_ctc_timestamps"
FAILED = "alignment_failed"


def wanted(request) -> bool:
    """Whether a chat request asks for word timestamps."""
    extra = getattr(request, "model_extra", None) or {}
    kw = extra.get("mm_processor_kwargs")
    if isinstance(kw, dict) and kw.get(OPT_IN):
        return True
    grains = extra.get("timestamp_granularities")
    return isinstance(grains, (list, tuple)) and "word" in grains


def _special_tokens(tokenizer) -> list[str]:
    tokens = set(getattr(tokenizer, "all_special_tokens", None) or ())
    added = getattr(tokenizer, "get_added_vocab", None)
    if callable(added):
        tokens.update(added())
    return sorted(tokens, key=len, reverse=True)


def diarization(capture) -> list[dict]:
    """The capture's 10 ms diarization segments: they need no transcript."""
    from gmlx.models.nemotron_h import align as al

    if capture is None or capture.labels is None:
        return []
    return al.diarization_segments(capture.labels, al.DIARIZATION_FRAME_SECONDS,
                                   0.0, capture.samples / SAMPLE_RATE)


async def align(http_request, model_id: str, capture, transcript: str):
    """``(Alignment or None, error or None)`` for ``transcript`` over
    ``capture``. An empty transcript has no words and no error; a
    transcript the aligner refuses has no Alignment and names why."""
    from gmlx.models.nemotron_h import align as al

    from .patches import native_audio
    from .patches.images import _watch_disconnect

    app_mod = importlib.import_module("gmlx._vendor.mlx_vlm.server.app")
    runtime = importlib.import_module("gmlx._vendor.mlx_vlm.server.runtime").runtime
    gen = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")
    # The model stays resident until the words are timed.
    model, processor, _ = app_mod.get_cached_model(model_id)
    hold = _current_hold()
    try:
        ctc = model.audio_tower.cfg.ctc
        tok = model.ctc_tokenizer()
        if ctc is None or tok is None:
            return None, "the speech encoder has no word timestamp head"
        text = al.clean_transcript(
            transcript, _special_tokens(getattr(processor, "tokenizer", processor)))
        if not text:
            return al.Alignment(words=[], diarization=diarization(capture),
                                mapping={}, failed=[]), None
        frames = int(capture.states.shape[0]) * ctc.upsampling
        try:
            prepared = al.prepare(text, tok, ctc.blank_id, frames)
        except al.AlignmentError as e:
            return None, str(e)
        stop = threading.Event()
        watcher = asyncio.create_task(_watch_disconnect(http_request, stop))
        try:
            logp = await asyncio.to_thread(
                native_audio.ctc_log_probs, runtime.response_generator, model,
                capture, prepared.columns, stop, gen.get_token_queue_timeout())
        finally:
            stop.set()
            watcher.cancel()
        result = await asyncio.to_thread(
            al.align, prepared, logp, capture.speaker_probs, capture.labels,
            capture.samples / SAMPLE_RATE, ctc.upsampling)
        return result, None
    finally:
        if hold is not None:
            hold.release()


def _current_hold():
    try:
        from . import residency
    except ImportError:
        return None
    return residency._active_hold.get()


def public(result, error, capture) -> dict:
    """NeMo's ``ctc_timestamps`` object: ``words``, ``diarization``,
    ``speaker_tag_to_diarization_speaker`` and ``error``, which names why
    ``words`` is empty."""
    from gmlx.models.nemotron_h import align as al

    if result is None:
        return al.public_result(None, FAILED if error else None, diarization(capture))
    return al.public_result(result)


async def chat_result(http_request, model_id: str, capture, transcript: str) -> dict:
    """:func:`public` for a chat completion; a failure is an ``error``."""
    from gmlx.serve.engine_jobs import JobCancelled

    try:
        result, error = await align(http_request, model_id, capture, transcript)
    except asyncio.CancelledError:
        raise
    except JobCancelled:
        _log.debug("word timestamps: the client left")
        return public(None, FAILED, capture)
    except Exception as e:  # noqa: BLE001  headers may be out: answer in the field
        _log.error("word timestamps failed: %s", e, exc_info=e)
        return public(None, FAILED, capture)
    if error:
        _log.warning("word timestamps: %s", error)
    return public(result, error, capture)


def refusal(request, clips: int) -> str | None:
    """Why a chat request cannot have word timestamps, or None: they need
    one completion and one recording."""
    n = (getattr(request, "model_extra", None) or {}).get("n")
    if isinstance(n, int) and n > 1:
        return "word timestamps need n = 1"
    if clips != 1:
        return f"word timestamps need exactly one audio clip, not {clips}"
    return None


async def attach(response, http_request, model_id: str, capture):
    """The chat response with ``ctc_timestamps``: on the body, or on the
    stream's final choice chunk, ahead of the usage chunk and ``[DONE]``.
    Only that chunk waits for the alignment."""
    import json

    from fastapi.responses import JSONResponse, Response
    from starlette.responses import StreamingResponse

    if isinstance(response, StreamingResponse):
        response.body_iterator = _stream(response.body_iterator, http_request,
                                         model_id, capture)
        return response
    if isinstance(response, Response):
        if response.status_code >= 400:
            return response
        body = json.loads(bytes(response.body))
    else:
        dump = getattr(response, "model_dump", None)
        body = dump() if callable(dump) else dict(response or {})
    choices = body.get("choices") or []
    text = (choices[0].get("message") or {}).get("content") if choices else ""
    body["ctc_timestamps"] = await chat_result(http_request, model_id, capture,
                                               str(text or ""))
    return JSONResponse(body)


async def _stream(body, http_request, model_id: str, capture):
    """NeMo ``ctc_serving``'s stream order: text deltas pass at once; the
    final choice chunk is held, with what follows it, until ``[DONE]``,
    then sent with ``ctc_timestamps``."""
    import json

    text: list[str] = []
    terminal = None
    trailing: list[str] = []
    failed = False
    pending = ""
    try:
        async for raw in body:
            pending += raw.decode() if isinstance(raw, (bytes, bytearray)) else raw
            while "\n\n" in pending:
                event, pending = pending.split("\n\n", 1)
                event += "\n\n"
                data = event[len("data: "):].strip() if event.startswith("data: ") else None
                if data == "[DONE]":
                    if terminal is not None and not failed:
                        terminal["ctc_timestamps"] = await chat_result(
                            http_request, model_id, capture, "".join(text))
                        yield f"data: {json.dumps(terminal, ensure_ascii=False)}\n\n"
                    for tail in trailing:
                        yield tail
                    yield event
                    continue
                if data is None:
                    yield event
                    continue
                try:
                    chunk = json.loads(data)
                except ValueError:
                    yield event
                    continue
                if "error" in chunk:
                    failed, terminal = True, None
                    yield event
                    continue
                if terminal is not None:
                    trailing.append(event)
                    continue
                choices = chunk.get("choices") or []
                if failed or not choices:
                    yield event
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                text.append(delta.get("content") or "")
                if choice.get("finish_reason") is None:
                    yield event
                    continue
                terminal = chunk
                if delta.get("content") or choice.get("logprobs") is not None:
                    partial = {**choice, "finish_reason": None}
                    yield f"data: {json.dumps({**chunk, 'choices': [partial]}, ensure_ascii=False)}\n\n"
                    terminal["choices"] = [{**choice, "delta": {}, "logprobs": None}]
        if pending:
            yield pending
    finally:
        aclose = getattr(body, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:  # noqa: S110 - upstream body may already be closed
                pass
