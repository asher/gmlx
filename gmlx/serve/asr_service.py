"""The speech companion on the server: one worker thread that owns every
loaded companion, shared by the chat routes' audio cascade and the
speech-to-text route."""

from __future__ import annotations

import concurrent.futures
import io
import threading
from collections import OrderedDict

from gmlx import asr

from .subservice import SingleWorker, SubserviceRequestError

ALIAS = "granite-speech"
_WORKER = SingleWorker("asr-worker")
# Transcripts by (companion, clip digest). A Responses chain renders an
# earlier turn's clip again on every later turn.
_CACHE_MAX = 64
_cache: OrderedDict = OrderedDict()
_cache_lock = threading.Lock()


class ASRRequestError(SubserviceRequestError):
    """A client-side problem with a clip (HTTP 4xx)."""


def prewarm(path: str) -> concurrent.futures.Future:
    from . import subservice

    return subservice.prewarm(_WORKER, lambda: asr.companion(path).speech(),
                              "speech companion")


def release() -> concurrent.futures.Future:
    """Drop every loaded companion, after the work queued before it."""
    def _run():
        for c in list(asr._companions.values()):
            c.release()
        with _cache_lock:
            _cache.clear()
        import mlx.core as mx
        mx.clear_cache()
    return _WORKER.submit(_run)


def _transcribe(path: str, data: bytes, field: str) -> str:
    from .media_decode import load_audio
    from .patches.media_gate import MediaRefused

    speech = asr.companion(path)
    try:
        wave = load_audio(io.BytesIO(data), speech.sample_rate, field=field)
    except MediaRefused as e:
        raise ASRRequestError(400, str(e)) from None
    return speech.transcribe(wave)


def transcribe(path: str, data: bytes, key: bytes | None = None,
               field: str = "audio") -> str:
    """The transcript of the encoded clip ``data``, from the cache when
    ``key`` (the clip's digest) is there. Blocks; call it off the event
    loop."""
    if key is not None:
        with _cache_lock:
            text = _cache.get((path, key))
            if text is not None:
                _cache.move_to_end((path, key))
                return text
    text = _WORKER.submit(_transcribe, path, data, field).result()
    if key is not None:
        with _cache_lock:
            _cache[(path, key)] = text
            while len(_cache) > _CACHE_MAX:
                _cache.popitem(last=False)
    return text
