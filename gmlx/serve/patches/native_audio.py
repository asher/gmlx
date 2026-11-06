"""Audio on a model that hears it through its own speech encoder (Nemotron
3.5 Transcribe), on the chat completions and Responses routes.

A route wrap reads each clip of the request once, as 16 kHz samples, and
encodes it on the model's engine thread as a step job of one encoder window
per step. The engine decodes other requests between the steps, and a
disconnect stops the job. The memory preflight runs first, so a request
that cannot fit spends no encoder time.

The encoded clips ride a context variable to the response generator's
tokenization, which hands them to the processor in place of the request's
audio. The prefix cache keys a prompt by each clip's digest. A byte-capped
cache of encoder rows, kept per loaded model, spares a follow-up turn the
encode of the clips it repeats.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import importlib
import logging
import re
import threading
import types
import weakref
from collections import OrderedDict

import numpy as np

from ._common import _CHAT_PATHS, _error_content, _wrap_post_routes

_log = logging.getLogger(__name__)

_FLAG = "_gmlx_native_audio"
_RESPONSES_PATHS = ("/responses", "/v1/responses")
SAMPLE_RATE = 16000
# Encoder rows kept per loaded model: about two hours of audio, and at
# most an eighth of the prefill room (gmlx's ceiling less the weights).
CACHE_BYTES = 512 << 20
CACHE_ROOM_SHARE = 8

_CLIPS: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "gmlx_native_audio", default=None)
# A caller that decoded the request's audio itself (the transcription
# route) sets its samples here, and a hook that sees the encoded clips
# before the decoder runs.
DECODED: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "gmlx_native_audio_decoded", default=None)
ON_ENCODED: contextvars.ContextVar = contextvars.ContextVar(
    "gmlx_native_audio_on_encoded", default=None)
# A caller that aligns words to the transcript sets a list here; the
# encoder adds one Capture per clip, and re-encodes cached clips for it.
CAPTURE: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "gmlx_native_audio_capture", default=None)


class Capture:
    """What aligning a clip's transcript needs once the decoder is done:
    its ASR states in bfloat16 (NeMo keeps them so), its 80 ms speaker
    probabilities, its 10 ms diarization labels and its length."""

    def __init__(self, digest: str, samples: int, states, speaker_probs,
                 labels):
        self.digest, self.samples = digest, int(samples)
        self.states, self.speaker_probs, self.labels = states, speaker_probs, labels

    @property
    def nbytes(self) -> int:
        return int(self.states.nbytes) + int(self.labels.nbytes)


def capture_bytes(model, waves) -> int:
    """What the captures of ``waves`` will hold: ASR states in bfloat16
    and the 10 ms labels."""
    from gmlx.models.nemotron_h import speech

    cfg = model.audio_tower.cfg
    return sum(speech.num_audio_tokens(w.size) * cfg.asr.d_model * 2
               + speech.num_frames(w.size) * cfg.num_speakers for w in waves)


class _Refused(Exception):
    """A client error, answered with a 400."""


# -- encoder rows ------------------------------------------------------------


def _cache_cap() -> int:
    from gmlx.gen.prefill_decay import prefill_room_bytes

    room = prefill_room_bytes()
    if room is None:
        return CACHE_BYTES
    return max(0, min(CACHE_BYTES, int(room) // CACHE_ROOM_SHARE))


class _RowCache:
    """Encoder rows by clip digest, least recently used out first past
    ``cap`` bytes (by default a share of the room, read at each put)."""

    def __init__(self, cap: int | None = None):
        self._cap = cap
        self.entries: OrderedDict = OrderedDict()
        self.bytes = 0
        self.lock = threading.Lock()

    def get(self, digest: str):
        with self.lock:
            hit = self.entries.get(digest)
            if hit is not None:
                self.entries.move_to_end(digest)
            return hit

    @property
    def cap(self) -> int:
        return _cache_cap() if self._cap is None else self._cap

    def put(self, digest: str, samples: int, embeds, probs) -> None:
        size = int(embeds.nbytes)
        cap = self.cap
        if size > cap:
            return
        with self.lock:
            if digest in self.entries:
                return
            self.entries[digest] = (samples, embeds, probs)
            self.bytes += size
            while self.bytes > cap:
                _, (_, old, _) = self.entries.popitem(last=False)
                self.bytes -= int(old.nbytes)

    def evict_one(self, keep=()) -> bool:
        """Drop the least recently used entry not in ``keep``. False when
        there is none."""
        with self.lock:
            for digest in self.entries:
                if digest not in keep:
                    _, old, _ = self.entries.pop(digest)
                    self.bytes -= int(old.nbytes)
                    return True
        return False


# id(model) -> its row cache; an entry leaves with its model.
_CACHES: dict[int, _RowCache] = {}
# Clips of requests in flight, by id: their rows count as held memory
# until the request lets them go.
_LIVE: dict[int, weakref.ref] = {}
# Captures of requests in flight, by id: held until aligned.
_CAPTURES: dict[int, weakref.ref] = {}
# id(response generator) -> requests encoding on it now.
_ENCODING: dict[int, int] = {}
_lock = threading.Lock()


def row_cache(model) -> _RowCache:
    key = id(model)
    with _lock:
        cache = _CACHES.get(key)
        if cache is None:
            cache = _CACHES[key] = _RowCache()
            try:
                weakref.finalize(model, _CACHES.pop, key, None)
            except TypeError:
                pass             # a stand-in model: the cache stays
        return cache


def held_bytes(skip=()) -> int:
    """Encoder rows the server holds: every row cache, the clips of
    requests in flight that no cache holds, and the captures waiting for
    alignment. Clips whose digest is in ``skip`` (a request's own, which
    it prices itself) do not count."""
    seen: set[int] = set()
    total = 0
    with _lock:
        caches = list(_CACHES.values())
    for cache in caches:
        with cache.lock:
            for digest, (_, embeds, _) in cache.entries.items():
                if digest not in skip and id(embeds) not in seen:
                    seen.add(id(embeds))
                    total += int(embeds.nbytes)
    for ref in list(_LIVE.values()):
        clip = ref()
        embeds = clip.embeds if clip is not None else None
        if (embeds is not None and clip.digest not in skip
                and id(embeds) not in seen):
            seen.add(id(embeds))
            total += int(embeds.nbytes)
    for ref in list(_CAPTURES.values()):
        cap = ref()
        if cap is not None and cap.digest not in skip:
            total += cap.nbytes
    return total


def encoding_requests(rg) -> int:
    """Requests encoding audio on ``rg``'s engine now, for the queue cap."""
    return _ENCODING.get(id(getattr(rg, "_rg", rg)), 0)


# -- the request -------------------------------------------------------------


def _digest(samples: np.ndarray) -> str:
    return hashlib.blake2b(np.ascontiguousarray(samples).tobytes(),
                           digest_size=16).hexdigest()


def _decode(data: bytes) -> np.ndarray:
    from ..media_programs import decode

    pcm, _, _ = decode(data, sample_rate=SAMPLE_RATE, nchannels=1,
                       field="input_audio")
    return pcm.astype(np.float32) / 32768.0


def _encode_job(model, todo: list, out: list, **kw):
    """A step job encoding each ``(features, valid frames, tokens)`` of
    ``todo`` into ``out``, one encoder window per step."""
    def steps(rg, should_stop):
        for feats, valid, tokens in todo:
            out.append((yield from model.encode_steps(feats, valid, tokens, **kw)))
        return out
    return steps


def _ctc_job(model, states, columns: list[int]):
    def steps(rg, should_stop):
        tower = model.audio_tower
        return (yield from tower.ctc_log_prob_steps(states.astype(tower.dtype),
                                                    columns))
    return steps


def ctc_log_probs(rg, model, capture: Capture, columns: list[int],
                  stop: threading.Event, timeout_s):
    """The CTC head's log probabilities of ``columns`` over ``capture``,
    ``(40 ms frames, len(columns))``, from a step job on ``rg``'s engine of
    one slice per step. Runs off the event loop."""
    from gmlx.serve.engine_jobs import run_on_engine

    return run_on_engine(
        getattr(rg, "_rg", rg), _ctc_job(model, capture.states, columns),
        request_id=f"ctc-{id(capture):x}", prompt_tokens=int(capture.states.shape[0]),
        stop=stop, timeout_s=timeout_s, stepwise=True,
        what="the word timestamp head")


def _mel(model, processor) -> np.ndarray:
    """The encoder's mel filterbank as NumPy: the processor's copy, taken
    at load. MLX on a request thread hits the thread-local stream trap."""
    fe = getattr(processor, "feature_extractor", None)
    mel = getattr(fe, "mel_filters", None)
    if isinstance(mel, np.ndarray):
        return mel
    import mlx.core as mx

    return np.array(model.audio_tower.mel_filters.astype(mx.float32))


def _priced(model, waves, digests) -> list:
    """``waves`` for the memory check, with an :class:`EncodedClip` in
    place of each clip the row cache holds: those cost no encoder work."""
    from gmlx.models.nemotron_h.transcribe_model import EncodedClip

    cache = row_cache(model)
    out = []
    for w, d in zip(waves, digests):
        hit = cache.get(d)
        if hit is None:
            out.append(w)
        else:
            n, embeds, probs = hit
            out.append(EncodedClip.of(n, d, embeds, probs))
    return out


def encode(rg, model, waves: list[np.ndarray], stop: threading.Event,
           timeout_s, *, processor=None, digests: list[str] | None = None
           ) -> list:
    """The :class:`EncodedClip` of each of ``waves``, from the model's row
    cache or a step job on ``rg``'s engine. Runs off the event loop."""
    from gmlx.models.nemotron_h import speech
    from gmlx.models.nemotron_h.transcribe_model import EncodedClip
    from gmlx.serve.engine_jobs import run_on_engine

    cache = row_cache(model)
    digests = digests or [_digest(w) for w in waves]
    captures = CAPTURE.get()
    # A capture needs the encoder's ASR states, which no cache keeps.
    hits = {d: None if captures is not None else cache.get(d) for d in digests}
    todo, fresh = [], {}
    mel = None
    for w, d in zip(waves, digests):
        if hits[d] is None and d not in fresh:
            fresh[d] = w.size
            if mel is None:
                mel = _mel(model, processor)
            todo.append((speech.features(w, mel), speech.num_frames(w.size),
                         speech.num_audio_tokens(w.size)))
    kw = {}
    if captures is not None:
        from gmlx.models.nemotron_h.align import CHECKPOINT

        kw = {"keep_states": True, "label_threshold": CHECKPOINT.vad_threshold}
    by_digest = {}
    if todo:
        key = id(getattr(rg, "_rg", rg))
        with _lock:
            _ENCODING[key] = _ENCODING.get(key, 0) + 1
        try:
            outs = run_on_engine(
                getattr(rg, "_rg", rg), _encode_job(model, todo, [], **kw),
                request_id=f"aud-{id(todo):x}",
                prompt_tokens=sum(t for _, _, t in todo), stop=stop,
                timeout_s=timeout_s, stepwise=True, what="the audio encoder")
        finally:
            with _lock:
                _ENCODING[key] -= 1
                if not _ENCODING[key]:
                    del _ENCODING[key]
        for (d, n), out in zip(fresh.items(), outs):
            hits[d] = (n, out.embeds, out.speaker_probs)
            cache.put(d, n, out.embeds, out.speaker_probs)
            by_digest[d] = out
    if captures is not None:
        import mlx.core as mx

        for d in digests:
            n, _, probs = hits[d]
            out = by_digest[d]
            states = out.asr_states.astype(mx.bfloat16)
            mx.eval(states)
            cap = Capture(d, n, states, probs, out.diarization_labels)
            key = id(cap)
            _CAPTURES[key] = weakref.ref(cap, lambda _, k=key: _CAPTURES.pop(k, None))
            captures.append(cap)
    clips = []
    for d in digests:
        n, embeds, probs = hits[d]
        clip = EncodedClip.of(n, d, embeds, probs)
        key = id(clip)
        _LIVE[key] = weakref.ref(clip, lambda _, k=key: _LIVE.pop(k, None))
        clips.append(clip)
    return clips


# -- the route wraps ---------------------------------------------------------


def _is_native(model_field) -> bool:
    from gmlx.load.discovery import is_speech_encoder

    from .. import bridge_vlm as serving

    try:
        _, spec = serving.resolve_request_model(
            model_field, profile_field=serving.get_request_profile())
    except Exception:  # noqa: BLE001  the stock route answers the error
        return False
    return is_speech_encoder(getattr(spec, "mmproj", None))


def _texts(content) -> list[str]:
    """The text of a message content: a string, or its text parts."""
    from .audio_cascade import _plain

    if isinstance(content, str):
        return [content]
    out = []
    for part in content if isinstance(content, list) else ():
        part = _plain(part)
        if (isinstance(part, dict)
                and part.get("type") in ("text", "input_text", "output_text")):
            out.append(str(part.get("text") or ""))
    return out


def _user_audio(messages) -> list:
    """The input_audio parts of the user messages: the ones the chat route
    reads, in its order."""
    from .audio_cascade import _plain

    return [p for m in messages
            if getattr(m, "role", None) == "user" and isinstance(m.content, list)
            for p in map(_plain, m.content)
            if isinstance(p, dict) and p.get("type") == "input_audio"]


# The model's own output tags, which a client may quote back.
_OUTPUT_TAG = re.compile(r"<(?:spk:\d+|ev:[^<>]*|sty:[^<>]*)>")
# The template's audio marker: the processor reads it as a clip.
_AUDIO_MARKER = "<|audio|>"
# id(tokenizer) -> (tokenizer, pattern of its control tokens).
_CONTROL: dict[int, tuple] = {}


def _control_pattern(tokenizer):
    hit = _CONTROL.get(id(tokenizer))
    if hit is not None and hit[0] is tokenizer:
        return hit[1]
    get = getattr(tokenizer, "get_added_vocab", None)
    names = {n for n in (get() if callable(get) else {})
             if len(n) > 1 and not _OUTPUT_TAG.fullmatch(n)} | {_AUDIO_MARKER}
    pattern = re.compile("|".join(
        re.escape(n) for n in sorted(names, key=len, reverse=True)))
    _CONTROL[id(tokenizer)] = (tokenizer, pattern)
    return pattern


def _check_control_tokens(tokenizer, texts: list[str]) -> None:
    """Refuse client text that holds one of the model's control tokens
    (the audio slot, the chat markers): it would be read as the token."""
    pattern = _control_pattern(tokenizer)
    for text in texts:
        found = pattern.search(text or "")
        if found:
            raise _Refused(
                f"the message text holds {found.group(0)}, a control token "
                "of this model. Remove it from the text")


# Tokens the chat template adds around the texts.
TEMPLATE_TOKENS = 64


def output_cap(used: int, max_tokens) -> int:
    """``max_tokens`` cut to the context a prompt of ``used`` tokens
    leaves; that room when the client set no cap."""
    from gmlx.models.nemotron_h.transcription import CONTEXT_TOKENS

    room = max(1, CONTEXT_TOKENS - int(used))
    mt = max_tokens if isinstance(max_tokens, int) and max_tokens > 0 else None
    return room if mt is None else min(mt, room)


def _context_check(model, tokenizer, texts: list[str], waves) -> int:
    """The prompt's tokens: the texts', the clips' and the template's (a
    bound). Raises _Refused past the model's context."""
    from gmlx.models.nemotron_h.transcription import CONTEXT_TOKENS

    audio_tokens, _ = model.audio_preflight(waves)
    text_tokens = len(tokenizer.encode("\n".join(texts))) if texts else 0
    used = audio_tokens + text_tokens + TEMPLATE_TOKENS
    if used >= CONTEXT_TOKENS:
        raise _Refused(
            f"the conversation is {used:,} tokens with its audio, and the "
            f"model's context is {CONTEXT_TOKENS:,}. Send less audio or text")
    return used


async def _decoded(parts: list) -> list:
    """The clips decoded to 16 kHz mono. Raises _Refused for a client
    error."""
    from gmlx.models.nemotron_h.transcription import check_clips

    from .audio_cascade import _clip_bytes

    try:
        waves = DECODED.get()
        if waves is None:
            data = [_clip_bytes(p) for p in parts]
            waves = await asyncio.to_thread(lambda: [_decode(d) for d in data])
        elif len(waves) != len(parts):
            raise ValueError(f"{len(parts)} audio parts but {len(waves)} "
                             "decoded clips")
        check_clips([w.size for w in waves], SAMPLE_RATE)
    except (RuntimeError, ValueError) as e:
        raise _Refused(str(e)) from None
    return waves


def _check_memory(rg, model, texts: list[str], waves, digests: list[str],
                  max_tokens) -> None:
    """The memory check for the request. The row cache yields to it, least
    recently used entry first, and the request is refused only once no
    other entry is left."""
    from .. import mem_preflight

    gen = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")
    own = set(digests)
    # A request that aligns holds its captures until then.
    mine = capture_bytes(model, waves) if CAPTURE.get() is not None else 0
    while True:
        try:
            mem_preflight.preflight_prompt_memory(
                getattr(rg, "_rg", rg), "\n".join(texts),
                # A capture re-encodes even the clips the cache holds.
                audio=waves if mine else _priced(model, waves, digests),
                args=types.SimpleNamespace(max_tokens=max_tokens),
                held_bytes=held_bytes(skip=own) + mine)
            return
        except gen.PromptTooLongError:
            if not row_cache(model).evict_one(keep=own):
                raise


async def _prepare(model_field, parts: list, texts: list[str], http_request,
                   max_tokens=None, client_texts: list[str] | None = None):
    """The request's encoded clips, and its output cap within the model's
    context. ``client_texts``, the user and system text, may hold no
    control token. Raises _Refused for a client error."""
    from .images import _watch_disconnect

    waves = await _decoded(parts)
    app_mod = importlib.import_module("gmlx._vendor.mlx_vlm.server.app")
    runtime = importlib.import_module("gmlx._vendor.mlx_vlm.server.runtime").runtime
    model, processor, _ = app_mod.get_cached_model(model_field)
    rg = runtime.response_generator
    tok = getattr(processor, "tokenizer", processor)
    _check_control_tokens(tok, texts if client_texts is None else client_texts)
    cap = output_cap(_context_check(model, tok, texts, waves), max_tokens)
    digests = await asyncio.to_thread(lambda: [_digest(w) for w in waves])
    # Before the encoder: a request that cannot fit spends no encoder time.
    _check_memory(rg, model, texts, waves, digests,
                  cap if max_tokens is not None else None)
    gen = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")
    stop = threading.Event()
    watcher = asyncio.create_task(_watch_disconnect(http_request, stop))
    try:
        clips = await asyncio.to_thread(
            lambda: encode(rg, model, waves, stop, gen.get_token_queue_timeout(),
                           processor=processor, digests=digests))
        return clips, cap
    finally:
        stop.set()
        watcher.cancel()


def _answer(path: str, e: Exception):
    from fastapi.responses import JSONResponse

    from gmlx.serve.engine_jobs import JobCancelled, JobTimeout

    gen = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")
    if isinstance(e, _Refused | gen.PromptTooLongError):
        return JSONResponse(status_code=400, content=_error_content(
            path, 400, "invalid_request_error", str(e)))
    if isinstance(e, JobCancelled):
        return JSONResponse(status_code=499, content=_error_content(
            path, 499, "client_closed_request", str(e)))
    if isinstance(e, JobTimeout):
        return JSONResponse(status_code=504, content=_error_content(
            path, 504, "timeout", str(e)))
    return None


_CLIENT_ROLES = ("user", "system", "developer")


def _chat_wrap(original):
    from .audio_cascade import _audio_parts

    async def endpoint(request, http_request):
        messages = list(getattr(request, "messages", None) or ())
        if not any(_audio_parts(m.content) for m in messages
                   if isinstance(m.content, list)):
            return await original(request, http_request)
        if not _is_native(request.model):
            return await original(request, http_request)
        from .. import word_timestamps

        stamps = word_timestamps.wanted(request)
        captures: list | None = None
        token = None
        try:
            parts = _user_audio(messages)
            if stamps:
                why = word_timestamps.refusal(request, len(parts))
                if why:
                    raise _Refused(why)
                captures = []
                token = CAPTURE.set(captures)
            texts = [t for m in messages for t in _texts(m.content)]
            asked = (getattr(request, "max_completion_tokens", None)
                     if "max_completion_tokens" in request.model_fields_set
                     else request.max_tokens
                     if "max_tokens" in request.model_fields_set else None)
            client = [t for m in messages
                      if getattr(m, "role", None) in _CLIENT_ROLES
                      for t in _texts(m.content)]
            clips, cap = await _prepare(request.model, parts, texts,
                                        http_request, asked, client)
        except Exception as e:  # noqa: BLE001  answered or raised below
            answer = _answer(http_request.url.path, e)
            if answer is None:
                raise
            return answer
        finally:
            if token is not None:
                CAPTURE.reset(token)
        hook = ON_ENCODED.get()
        if hook is not None:
            hook(clips)
        # The output stays inside the model's context.
        request.max_tokens = cap
        if "max_completion_tokens" in request.model_fields_set:
            request.max_completion_tokens = cap
        from .render import _PlacedMedia

        # Each clip renders in the message that sent it.
        for m in messages:
            if getattr(m, "role", None) == "user" and isinstance(m.content, list):
                m.content = _PlacedMedia(m.content)
        _CLIPS.set(clips)        # read in the generate call, maybe in the stream
        response = await original(request, http_request)
        if captures:
            return await word_timestamps.attach(response, http_request,
                                                request.model, captures[0])
        return response
    return endpoint


_RESPONSES_PATHS = ("/responses", "/v1/responses")
_COUNT_PATHS = ("/responses/input_tokens", "/v1/responses/input_tokens")


def _response_items(body) -> list:
    """The stored chain and the current input of a Responses body, as the
    route converts them."""
    openai = importlib.import_module("gmlx._vendor.mlx_vlm.server.openai")
    try:
        return (openai._response_chain_items(body.get("previous_response_id"))
                + openai._normalize_response_input(body.get("input")))
    except Exception:  # noqa: BLE001  the stock route answers the error
        return []


def _is_user_message(item) -> bool:
    return (isinstance(item, dict) and item.get("type") == "message"
            and (item.get("role") or "user") == "user")


def _item_audio(items) -> list:
    """The input_audio parts of the user messages, in prompt order."""
    from .audio_cascade import _plain

    return [p for item in items if _is_user_message(item)
            and isinstance(item.get("content"), list)
            for p in map(_plain, item["content"])
            if isinstance(p, dict) and p.get("type") == "input_audio"]


def _responses_wrap(original, count_only: bool = False):
    async def endpoint(request):
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001  the stock route answers the error
            return await original(request)
        if not isinstance(body, dict):
            return await original(request)
        items = _response_items(body)
        parts = _item_audio(items)
        if not parts or not _is_native(body.get("model")):
            return await original(request)
        try:
            if count_only:
                clips = await _decoded(parts)
            else:
                texts = [t for item in items if isinstance(item, dict)
                         for t in _texts(item.get("content"))]
                client = [t for item in items if isinstance(item, dict)
                          and item.get("role", "user") in _CLIENT_ROLES
                          for t in _texts(item.get("content"))]
                client += _texts(body.get("instructions") or "")
                clips, cap = await _prepare(body.get("model"), parts, texts,
                                            request, body.get("max_output_tokens"),
                                            client)
                # The stock route reads this same parsed body.
                body["max_output_tokens"] = cap
        except Exception as e:  # noqa: BLE001  answered or raised below
            answer = _answer(request.url.path, e)
            if answer is None:
                raise
            return answer
        _CLIPS.set(clips)
        return await original(request)
    return endpoint


def _item_converter(stock):
    """The Responses message conversion, with a user message's clips placed
    in that message while a native request is on. The stock conversion
    drops them."""
    def _append_response_item_to_prompt(item, chat_messages, images):
        from .audio_cascade import _plain
        from .render import _PlacedMedia, _placed_content

        content = item.get("content") if isinstance(item, dict) else None
        if (_CLIPS.get() is None or not _is_user_message(item)
                or not isinstance(content, list)):
            return stock(item, chat_messages, images)
        plain = _PlacedMedia(_plain(p) for p in content)
        if not any(isinstance(p, dict) and p.get("type") == "input_audio"
                   for p in plain):
            return stock(item, chat_messages, images)
        before = len(chat_messages)
        stock(item, chat_messages, images)
        if len(chat_messages) == before + 1:
            chat_messages[-1]["content"] = _placed_content(plain)
    _append_response_item_to_prompt.__wrapped__ = stock  # type: ignore[attr-defined]
    setattr(_append_response_item_to_prompt, _FLAG, True)
    return _append_response_item_to_prompt


def _items_audio(stock):
    def _response_items_audio(items):
        clips = _CLIPS.get()
        return list(clips) if clips is not None else stock(items)
    _response_items_audio.__wrapped__ = stock  # type: ignore[attr-defined]
    setattr(_response_items_audio, _FLAG, True)
    return _response_items_audio


def _preprocess(stock):
    """The response generator's tokenization, with the request's encoded
    clips in place of its audio."""
    def _cpu_preprocess(self, prompt, images=None, audio=None, videos=None):
        clips = _CLIPS.get()
        if clips is not None and audio:
            if len(audio) != len(clips):
                raise ValueError(f"{len(audio)} clips in the prompt but "
                                 f"{len(clips)} encoded")
            audio = list(clips)
        if videos is None:
            return stock(self, prompt, images, audio)
        return stock(self, prompt, images, audio, videos)
    _cpu_preprocess.__wrapped__ = stock  # type: ignore[attr-defined]
    setattr(_cpu_preprocess, _FLAG, True)
    return _cpu_preprocess


def install_native_audio() -> None:
    """Install the route wraps, the Responses conversion and the
    tokenization seam. Idempotent."""
    from gmlx.upstream.patching import wrapped_with

    gen = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")
    cls = gen.ResponseGenerator
    if not wrapped_with(cls._cpu_preprocess, _FLAG):
        cls._cpu_preprocess = _preprocess(cls._cpu_preprocess)
    rs = importlib.import_module("gmlx._vendor.mlx_vlm.server.responses_state")
    if not wrapped_with(rs._append_response_item_to_prompt, _FLAG):
        rs._append_response_item_to_prompt = _item_converter(
            rs._append_response_item_to_prompt)
    openai = importlib.import_module("gmlx._vendor.mlx_vlm.server.openai")
    if not wrapped_with(openai._response_items_audio, _FLAG):
        openai._response_items_audio = _items_audio(openai._response_items_audio)
    app = importlib.import_module("gmlx._vendor.mlx_vlm.server.app").app
    _wrap_post_routes(app, _CHAT_PATHS, _FLAG, _chat_wrap)
    _wrap_post_routes(app, _RESPONSES_PATHS, _FLAG, _responses_wrap)
    _wrap_post_routes(app, _COUNT_PATHS, _FLAG,
                      lambda original: _responses_wrap(original, count_only=True))
