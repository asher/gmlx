"""OpenAI ``/v1/completions``: raw prompts with no chat template.

A prompt is a string, a list of strings, a list of token ids or a list of
such lists, as OpenAI and vLLM take it. The route turns every prompt into
ids before anything queues: a string through the engine's own tokenizer
binding, with ``add_special_tokens`` (default true), and a list of ids as
sent. Each prompt becomes its own engine row, and ``_build_gen_args``
applies the same sampling defaults, profiles, XTC and adapters as on
``/v1/chat/completions``. ``logprobs: N`` returns the OpenAI completions
shape. ``echo`` puts the prompt before the completion. Out of scope (400):
``n > 1``, ``suffix``, ``best_of > 1``, a streamed list of prompts."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import importlib
import inspect
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, List, Optional

from fastapi import HTTPException, Request  # module-level: stringized annotations
from pydantic import BaseModel, ConfigDict

from gmlx.gen.generation import StopScanner
from gmlx.gen.logprobs import json_logprob
from gmlx.upstream.single_bos_fix import raw_prompt_special_tokens
from ._common import _remove_routes
from .api_contract import COMPLETIONS_CONSUMED, warn_ignored_fields
from .chat_behavior import _request_stop_sequences

_COMPLETIONS_PATHS = ("/completions", "/v1/completions")
# Prompts one request may carry when the queue depth cap is off.
MAX_PROMPTS = 64
_SHAPE = ("a prompt is a string, a list of strings, a list of token ids or a "
          "list of such lists")

_log = logging.getLogger(__name__)

# Bound once at install time (the route cannot run before install); the
# sibling patches bind their upstream modules the same way.
_app_mod = None
_gen_mod = None
_runtime_mod = None


def _modules():
    global _app_mod, _gen_mod, _runtime_mod
    if _app_mod is None:
        _app_mod = importlib.import_module("gmlx._vendor.mlx_vlm.server.app")
        _gen_mod = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")
        _runtime_mod = importlib.import_module("gmlx._vendor.mlx_vlm.server.runtime")
    return _app_mod, _gen_mod, _runtime_mod


class CompletionRequest(BaseModel):
    """Local request shape - upstream ``schemas.py`` stays untouched. Extras
    ride along (``extra="allow"``) so ``_build_gen_args`` and the gmlx
    gen-args transforms read top_k / min_p / profile / xtc_* off it like any
    other request; ``prompt``, ``logprobs`` and ``add_special_tokens`` stay
    ``Any`` so the handler can 400 a bad shape with a clear message instead
    of a pydantic 422."""
    model_config = ConfigDict(extra="allow")

    model: str = ""
    prompt: Any = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    seed: Optional[int] = None
    stop: Optional[Any] = None
    stream: bool = False
    stream_options: Optional[Any] = None
    echo: Optional[bool] = None
    suffix: Optional[str] = None
    n: Optional[int] = None
    best_of: Optional[int] = None
    logprobs: Optional[Any] = None
    add_special_tokens: Optional[Any] = None
    return_tokens_as_token_ids: Optional[bool] = None


class CompletionChoice(BaseModel):
    text: str = ""
    index: int = 0
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None


class CompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CompletionResponse(BaseModel):
    id: str = ""
    object: str = "text_completion"
    created: int = 0
    model: str = ""
    choices: List[CompletionChoice] = []
    usage: Optional[CompletionUsage] = None


def _bad(detail: str) -> HTTPException:
    return HTTPException(status_code=400, detail=detail)


def _is_id(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _id_error(items) -> HTTPException:
    for t in items:
        if isinstance(t, bool):
            return _bad("token ids must be integers, not booleans")
        if isinstance(t, float):
            return _bad("token ids must be integers, not floats")
        if isinstance(t, list):
            return _bad("token-id prompts nest at most two levels: a list "
                        "of token ids or a list of such lists")
    return _bad(f"'prompt' must be {_SHAPE}")


def parse_prompts(prompt) -> list:
    """The prompts of a request body, each a string or a list of ids."""
    if isinstance(prompt, str):
        if not prompt:
            raise _bad("prompt must not be empty")
        return [prompt]
    if not isinstance(prompt, list):
        raise _bad(f"'prompt' must be {_SHAPE}")
    if not prompt:
        raise _bad("prompt must not be empty")
    if all(isinstance(p, str) for p in prompt):
        if not all(prompt):
            raise _bad("prompt must not be empty")
        return list(prompt)
    if all(_is_id(t) for t in prompt):
        return [list(prompt)]
    if all(isinstance(p, list) for p in prompt):
        out = []
        for p in prompt:
            if not p:
                raise _bad("prompt must not be empty")
            if not all(_is_id(t) for t in p):
                raise _id_error(p)
            out.append(list(p))
        return out
    if any(isinstance(p, str) for p in prompt):
        raise _bad(f"'prompt' must be {_SHAPE}; a list that mixes text and "
                   "token ids is not supported")
    raise _id_error(prompt)


def max_prompts() -> int:
    """Prompts one request may carry: the queue depth cap when it is on."""
    from gmlx.serve.queue_cap import _cap

    cap = _cap()
    return cap if cap > 0 else MAX_PROMPTS


def logprobs_cap() -> int:
    _, gen_mod, _ = _modules()
    return int(gen_mod.get_top_logprobs_k())


def _validate(request: CompletionRequest) -> list:
    """400s for the shapes the route does not serve, with a next step.
    Returns the prompts."""
    if request.n is not None and request.n != 1:
        raise _bad("'n' greater than 1 is not supported; send parallel "
                   "requests instead")
    if request.suffix:
        raise _bad("'suffix' (fill-in-the-middle) is not supported on this "
                   "server")
    if request.best_of is not None and request.best_of != 1:
        raise _bad("'best_of' greater than 1 is not supported")
    lp = request.logprobs
    if lp is not None:
        if isinstance(lp, bool):
            raise _bad("'logprobs' on /v1/completions is the number of "
                       "alternatives per token, an integer such as 0, not a "
                       "boolean")
        if not isinstance(lp, int) or lp < 0:
            raise _bad("'logprobs' must be an integer of 0 or more")
        cap = logprobs_cap()
        if lp > cap:
            raise _bad(f"logprobs is {lp}, above the server cap {cap} "
                       "(TOP_LOGPROBS_K)")
    flag = request.add_special_tokens
    if flag is not None and not isinstance(flag, bool):
        raise _bad("'add_special_tokens' must be true or false")
    if request.max_tokens == 0 and not request.echo:
        raise _bad("'max_tokens' 0 needs 'echo': it returns the prompt "
                   "without generating")
    if request.stream and request.echo and lp is not None:
        raise _bad("'echo' with 'logprobs' scores the prompt and does not "
                   "stream; send it without 'stream'")
    prompts = parse_prompts(request.prompt)
    if request.stream and len(prompts) > 1:
        raise _bad("a list of prompts does not stream; send it without "
                   "'stream', or one request per prompt")
    limit = max_prompts()
    if len(prompts) > limit:
        raise _bad(f"{len(prompts)} prompts in one request, above the limit "
                   f"{limit}; split the request")
    return prompts


def _include_usage(request: CompletionRequest) -> bool:
    so = request.stream_options
    if isinstance(so, dict):
        return bool(so.get("include_usage"))
    return bool(getattr(so, "include_usage", False))


_RECORD_FAILURE_WARNED = False


def _record_failure(runtime, model: str, stream: bool, error: str) -> None:
    global _RECORD_FAILURE_WARNED
    try:
        runtime.metrics.record_failure(endpoint="/v1/completions",
                                       model=model, stream=stream,
                                       error=error)
    except Exception:
        if not _RECORD_FAILURE_WARNED:
            _RECORD_FAILURE_WARNED = True
            _log.warning("failure-metrics record raised; failed requests "
                         "are missing from /v1/metrics", exc_info=True)


def _completion_envelope(gen_mod, *, model, stream, prompt_tokens,
                         completion_tokens, request_start, metrics,
                         finish_reason):
    return gen_mod._build_metrics_envelope(
        endpoint="/v1/completions",
        model=model,
        stream=stream,
        backend="continuous_batching",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        generated_tokens=completion_tokens,
        request_elapsed_s=time.perf_counter() - request_start,
        request_started_s=request_start,
        token_times=metrics.token_times,
        prompt_tps=metrics.prompt_tps,
        generation_tps=metrics.generation_tps,
        peak_memory_gb=metrics.peak_memory or None,
        finish_reason=finish_reason,
    )


@dataclass
class _Row:
    """One prompt: its ids, and the text sent for a string prompt."""
    ids: list
    text: Optional[str] = None


@dataclass
class _Prepared:
    rg: Any
    hold: Any
    gen_args: Any
    rows: list
    tokenizer: Any
    lock: Any = field(default_factory=contextlib.nullcontext)

    def release(self) -> None:
        hold, self.hold = self.hold, None
        if hold is not None:
            hold.release()


def _tokenizer_of(processor):
    return getattr(processor, "tokenizer", processor)


def _vocab_size(tokenizer) -> Optional[int]:
    for tok in (tokenizer, getattr(tokenizer, "_tokenizer", None)):
        if tok is None:
            continue
        try:
            return int(len(tok))
        except TypeError:
            continue
    size = getattr(tokenizer, "vocab_size", None)
    return int(size) if isinstance(size, int) else None


def _row_ids(raw) -> list:
    ids = raw["input_ids"]
    row = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    if row and isinstance(row[0], list):
        row = row[0]
    return [int(t) for t in row]


def _tokenize(rg, tokenizer, prompts, add_special_tokens: bool) -> list:
    """Each prompt as ids: a string through the engine's tokenizer binding,
    as chat tokenizes, a list of ids checked against the vocabulary and
    the model's media placeholders."""
    from gmlx._vendor.mlx_vlm.apc import multimodal_token_ids_from_config

    vocab = _vocab_size(tokenizer)
    config = getattr(getattr(rg, "model", None), "config", None)
    media = multimodal_token_ids_from_config(config) if config is not None else set()
    lock = getattr(rg, "_tokenizer_lock", None) or contextlib.nullcontext()
    rows = []
    for p in prompts:
        if isinstance(p, str):
            with lock, raw_prompt_special_tokens(add_special_tokens):
                rows.append(_Row(_row_ids(rg._preprocess_request(p)), p))
            continue
        for t in p:
            if t < 0 or (vocab is not None and t >= vocab):
                top = f"0 to {vocab - 1}" if vocab is not None else "0 up"
                raise _bad(f"token id {t} is out of range; this model's ids "
                           f"run from {top}")
            if t in media:
                raise _bad(f"token id {t} is a media placeholder of this "
                           "model; send images and video through chat")
        rows.append(_Row(list(p)))
    return rows


def _prepare(request, http_request, prompts) -> _Prepared:
    """Resolve and load the model, build the generation arguments and
    tokenize every prompt, all in one worker thread so the request context
    (profile var, active spec, residency entry) stays coherent. Raising here
    (resolver 404/400, a bad id, PromptTooLongError) surfaces before any
    streaming starts. The caller releases the model's busy hold."""
    from gmlx.gen.diffusion import is_diffusion_model

    app_mod, gen_mod, runtime_mod = _modules()
    _model, processor, _config = app_mod.get_cached_model(request.model)
    # The (patched) _build_gen_args applies profile injection / XTC /
    # template-kwargs transforms exactly as on the chat routes.
    gen_args = app_mod._build_gen_args(
        request, processor, tenant_id=app_mod._read_tenant_id(http_request))
    gen_args.logprobs = request.logprobs is not None
    gen_args.top_logprobs = int(request.logprobs or 0)
    # runtime.response_generator resolves against the residency entry
    # get_cached_model bound to this context. The request holds the entry
    # until its last row ends.
    guard = runtime_mod.runtime.response_generator
    if guard is None:
        raise HTTPException(
            status_code=500,
            detail="continuous-batching engine unavailable; restart the "
                   "server")
    rg = getattr(guard, "_rg", guard)
    out = _Prepared(rg, getattr(guard, "_hold", None), gen_args, [],
                    _tokenizer_of(processor if processor is not None
                                  else getattr(rg, "processor", None)),
                    getattr(rg, "_tokenizer_lock", None) or contextlib.nullcontext())
    try:
        if request.logprobs is not None and is_diffusion_model(getattr(rg, "model", None)):
            raise _bad("'logprobs' is not supported on a diffusion model")
        flag = True if request.add_special_tokens is None else request.add_special_tokens
        out.rows = _tokenize(rg, out.tokenizer, prompts, flag)
        for row in out.rows:
            gen_mod._check_configured_context_budget(
                len(row.ids), gen_args.max_tokens if request.max_tokens != 0 else 0)
    except BaseException:
        out.release()
        raise
    return out


class _Logprobs:
    """The OpenAI completions ``logprobs`` object of one choice, built a
    token at a time. Names are the token text, or ``token_id:<id>``."""

    def __init__(self, prep: _Prepared, k: int, as_ids: bool, offset: int = 0):
        self.prep = prep
        self.k = int(k)
        self.as_ids = bool(as_ids)
        self.offset = int(offset)
        self._names: dict = {}
        self.clear()

    def clear(self) -> None:
        self.tokens: list = []
        self.token_logprobs: list = []
        self.top_logprobs: list = []
        self.text_offset: list = []

    def name(self, tid: int) -> str:
        tid = int(tid)
        if self.as_ids:
            return f"token_id:{tid}"
        got = self._names.get(tid)
        if got is None:
            with self.prep.lock:
                try:
                    got = self.prep.tokenizer.decode([tid])
                except Exception:  # noqa: BLE001 - an undecodable id names nothing
                    got = ""
            self._names[tid] = got
        return got

    def add(self, tid: int, lp, top: Optional[list] = None) -> None:
        """One position: ``lp`` None for the first prompt token, else its
        logprob and the top alternatives as ``(id, logprob)`` pairs. The
        alternatives are the top ``k`` plus the token itself."""
        name = self.name(tid)
        self.tokens.append(name)
        if lp is None:
            self.token_logprobs.append(None)
            self.top_logprobs.append(None)
        else:
            self.token_logprobs.append(json_logprob(lp))
            alts = {name: json_logprob(lp)}
            alternatives: list = list(top) if top else []
            for alt, alp in alternatives[:self.k]:
                alts.setdefault(self.name(alt), json_logprob(alp))
            self.top_logprobs.append(alts)
        self.text_offset.append(self.offset)
        self.offset += len(name)

    def add_token(self, tok) -> None:
        lp = getattr(tok, "logprobs", None)
        if lp is not None:
            self.add(getattr(tok, "token", 0), lp, getattr(tok, "top_logprobs", None))

    def body(self) -> dict:
        return {"tokens": list(self.tokens),
                "token_logprobs": list(self.token_logprobs),
                "top_logprobs": list(self.top_logprobs),
                "text_offset": list(self.text_offset)}

    def take(self) -> dict:
        out = self.body()
        self.clear()
        return out


def _echo_text(prep: _Prepared, row: _Row) -> str:
    if row.text is not None:
        return row.text
    with prep.lock:
        return prep.tokenizer.decode(row.ids)


def _run_row(ctx, token_iter, stops, gen_mod, entries: Optional[_Logprobs]):
    """Drain one row: ``(text, output_tokens, finish, metrics)``."""
    scanner = StopScanner(stops)
    metrics = gen_mod.GenerationMetrics()
    text = ""
    output_tokens = 0
    finish = None
    hit_stop = False
    try:
        for tok in token_iter:
            output_tokens += getattr(tok, "token_count", 1)
            metrics.record_chunk(tok)
            if entries is not None:
                entries.add_token(tok)
            if stops:
                piece, hit_stop = scanner.feed(tok.text)
                text += piece
                if hit_stop:
                    break
            else:
                text += tok.text
            if tok.finish_reason:
                finish = tok.finish_reason
                break
    finally:
        try:
            token_iter.close()
        except Exception:  # noqa: S110 - iterator already finished
            pass
    if stops and not hit_stop:
        text += scanner.flush()
    finish = "stop" if hit_stop else (finish or "stop")
    return text, output_tokens, finish, metrics


def _close_all(started) -> None:
    for _ctx, it in started:
        try:
            it.close()
        except Exception:  # noqa: S110 - the row already ended
            pass


async def completions_endpoint(request: CompletionRequest,
                               http_request: Request):
    # Default-model fill and the request-profile contextvar are applied by
    # request_flow's profile-capture wrapper, which wraps this route
    # (_PROFILE_CAPTURE_PATHS lists it; install order in server_patches
    # __init__ registers the route first). Generation starts inside the
    # wrapper's window - the prompts queue before the StreamingResponse is
    # returned - so nothing here re-reads the profile after reset.
    request_start = time.perf_counter()
    app_mod, gen_mod, runtime_mod = _modules()
    runtime = runtime_mod.runtime

    prompts = _validate(request)
    warn_ignored_fields("/v1/completions",
                        set(request.model_fields_set) - COMPLETIONS_CONSUMED)
    stops = _request_stop_sequences(request)

    def _prep():
        return _prepare(request, http_request, prompts)

    if request.stream:
        return await _stream_completion(
            request, runtime, gen_mod, _prep, stops, request_start)
    return await _blocking_completion(
        request, runtime, gen_mod, _prep, stops, request_start)


def _failure(runtime, request, stream: bool, e: BaseException, gen_mod):
    """The HTTP error for a failed request, after recording it."""
    if isinstance(e, gen_mod.PromptTooLongError):
        _record_failure(runtime, request.model, stream, str(e))
        return HTTPException(status_code=400, detail=str(e))
    if isinstance(e, HTTPException):
        _record_failure(runtime, request.model, stream, str(e.detail))
        return e
    _record_failure(runtime, request.model, stream, str(e))
    from .row_failed import RowShedError

    if isinstance(e, RowShedError):
        # The shed contract: the 500 body carries the numbers and the
        # retry-after-backoff hint (message built for clients; no paths or
        # internals).
        return HTTPException(status_code=500, detail=str(e))
    # Generic body: raw exception text can carry paths and internals; the
    # recorded failure reaches the operator via the [req] log line.
    return HTTPException(status_code=500, detail=(
        f"generation failed ({type(e).__name__}); see the server log"))


def _generate_rows(request, prep: _Prepared, stops, gen_mod, scored=None):
    """Queue every row, then drain them in order. A row that fails cancels
    the rows already queued, and the request fails with its error.
    ``scored`` holds a scoring request's prompt entries per row."""
    choices = []
    started: list = []
    try:
        if request.max_tokens != 0:
            for row in prep.rows:
                started.append(prep.rg.generate(row.ids, args=copy.copy(prep.gen_args)))
        for i, row in enumerate(prep.rows):
            entries = None
            if request.logprobs is not None:
                entries = (scored[i] if scored is not None
                           else _Logprobs(prep, request.logprobs,
                                          request.return_tokens_as_token_ids))
            echo = _echo_text(prep, row) if request.echo else ""
            if request.max_tokens == 0:
                choices.append((row, echo, len(row.ids), 0, "length", None, entries))
                continue
            ctx, it = started[i]
            text, n_out, finish, metrics = _run_row(ctx, it, stops, gen_mod, entries)
            choices.append((row, echo + text, ctx.prompt_tokens, n_out, finish,
                            metrics, entries))
    except BaseException:
        _close_all(started)
        raise
    return choices


async def _blocking_completion(request, runtime, gen_mod, prep_fn, stops,
                               request_start, scored=None):
    runtime.metrics.begin_request(endpoint="/v1/completions",
                                  model=request.model, stream=False)

    def _run():
        prep = prep_fn()
        try:
            return _generate_rows(request, prep, stops, gen_mod, scored)
        finally:
            prep.release()

    try:
        rows = await asyncio.to_thread(_run)
    except Exception as e:
        raise _failure(runtime, request, False, e, gen_mod) from None

    choices, prompt_tokens, output_tokens = [], 0, 0
    first_metrics = None
    for i, (_row, text, p_tok, n_out, finish, metrics, entries) in enumerate(rows):
        prompt_tokens += p_tok
        output_tokens += n_out
        first_metrics = first_metrics or metrics
        choices.append(CompletionChoice(
            text=text, index=i, finish_reason=finish,
            logprobs=entries.body() if entries is not None else None))
    runtime.metrics.record_success(_completion_envelope(
        gen_mod, model=request.model, stream=False,
        prompt_tokens=prompt_tokens, completion_tokens=output_tokens,
        request_start=request_start,
        metrics=first_metrics or gen_mod.GenerationMetrics(),
        finish_reason=choices[0].finish_reason if choices else None))
    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4()}",
        created=int(time.time()),
        model=request.model,
        choices=choices,
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=output_tokens,
            total_tokens=prompt_tokens + output_tokens))


async def _stream_completion(request, runtime, gen_mod, prep_fn, stops,
                             request_start):
    from starlette.responses import StreamingResponse

    runtime.metrics.begin_request(endpoint="/v1/completions",
                                  model=request.model, stream=True)

    def _start():
        prep = prep_fn()
        try:
            if request.max_tokens == 0:
                return prep, None, None
            ctx, it = prep.rg.generate(prep.rows[0].ids, args=prep.gen_args)
        except BaseException:
            prep.release()
            raise
        return prep, ctx, it

    try:
        prep, ctx, token_iter = await asyncio.to_thread(_start)
    except Exception as e:
        raise _failure(runtime, request, True, e, gen_mod) from None

    row = prep.rows[0]
    prompt_tokens = ctx.prompt_tokens if ctx is not None else len(row.ids)
    echo = _echo_text(prep, row) if request.echo else ""
    entries = (_Logprobs(prep, request.logprobs, request.return_tokens_as_token_ids)
               if request.logprobs is not None else None)
    rid = f"cmpl-{uuid.uuid4()}"
    created = int(time.time())
    include_usage = _include_usage(request)

    def _chunk(text: str, finish: str | None) -> str:
        lp = entries.take() if entries is not None else None
        return "data: " + json.dumps({
            "id": rid, "object": "text_completion", "created": created,
            "model": request.model,
            "choices": [{"index": 0, "text": text, "logprobs": lp,
                         "finish_reason": finish}],
        }) + "\n\n"

    async def stream_generator():
        import concurrent.futures
        import threading

        scanner = StopScanner(stops)
        metrics = gen_mod.GenerationMetrics()
        output_tokens = 0
        finish = None
        hit_stop = False
        # One long-lived pump thread per stream, not one executor hop (plus
        # contextvars copy) per token: the pump drains the blocking iterator
        # into a bounded asyncio.Queue for backpressure. The pump owns
        # ``token_iter.close()`` - generators are not thread-safe, so only
        # the iterating thread may close it; the consumer's ``finally`` just
        # sets ``closed`` and the pump notices within its put timeout.
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        closed = threading.Event()
        pumping = False
        DONE = object()

        def _put(item) -> bool:
            fut = asyncio.run_coroutine_threadsafe(q.put(item), loop)
            while True:
                try:
                    fut.result(0.25)
                    return True
                except concurrent.futures.TimeoutError:
                    if closed.is_set():
                        fut.cancel()
                        return False
                except Exception:
                    return False        # event loop gone
        def _pump():
            try:
                for tok in token_iter:
                    if not _put(tok):
                        return
                _put(DONE)
            except BaseException as e:  # surface the error on the consumer
                _put(e)
            finally:
                try:
                    # normal end or disconnect: cancels the in-flight
                    # batch generation
                    token_iter.close()
                except Exception:  # noqa: S110 - iterator already finished
                    pass
                prep.release()

        try:
            if echo:
                yield _chunk(echo, None)
            if token_iter is None:
                finish = "length"
            else:
                pumping = True
                threading.Thread(target=_pump, name="gmlx-completions-pump",
                                 daemon=True).start()
                while True:
                    tok = await q.get()
                    if tok is DONE:
                        break
                    if isinstance(tok, BaseException):
                        raise tok
                    output_tokens += getattr(tok, "token_count", 1)
                    metrics.record_chunk(tok)
                    if entries is not None:
                        entries.add_token(tok)
                    piece = tok.text
                    if stops:
                        piece, hit_stop = scanner.feed(tok.text)
                    if piece:
                        yield _chunk(piece, None)
                    if hit_stop:
                        break
                    if tok.finish_reason:
                        finish = tok.finish_reason
                        break
                if stops and not hit_stop:
                    tail = scanner.flush()
                    if tail:
                        yield _chunk(tail, None)
                finish = "stop" if hit_stop else (finish or "stop")
            yield _chunk("", finish)
            if include_usage:
                yield "data: " + json.dumps({
                    "id": rid, "object": "text_completion",
                    "created": created, "model": request.model,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": output_tokens,
                        "total_tokens": prompt_tokens + output_tokens,
                    }}) + "\n\n"
            yield "data: [DONE]\n\n"
            runtime.metrics.record_success(_completion_envelope(
                gen_mod, model=request.model, stream=True,
                prompt_tokens=prompt_tokens,
                completion_tokens=output_tokens,
                request_start=request_start, metrics=metrics,
                finish_reason=finish))
        except Exception as e:
            _record_failure(runtime, request.model, True, str(e))
            yield "data: " + json.dumps(
                {"error": {"message": str(e), "type": "server_error"}}) \
                + "\n\n"
        finally:
            closed.set()    # unblock the pump; it closes token_iter
            if not pumping:
                if token_iter is not None:
                    _close_all([(ctx, token_iter)])
                prep.release()

    return StreamingResponse(stream_generator(),
                             media_type="text/event-stream")


# Real-class signature, not this module's stringized annotations: the route
# wrappers (_wrap_post_routes) copy ``__signature__`` into endpoints living in
# other modules, where FastAPI could not resolve the strings "CompletionRequest"
# / "Request" against their globals and would 422 every request.
completions_endpoint.__signature__ = inspect.Signature([
    inspect.Parameter("request", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                      annotation=CompletionRequest),
    inspect.Parameter("http_request", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                      annotation=Request),
])


def install_completions_route() -> None:
    """Register ``POST /v1/completions`` (+ ``/completions``). Install before
    the load-offload / keepalive route wrappers so those wrap this route
    too."""
    app = importlib.import_module("gmlx._vendor.mlx_vlm.server.app").app
    _remove_routes(app, *_COMPLETIONS_PATHS)
    app.add_api_route("/v1/completions", completions_endpoint,
                      methods=["POST"], include_in_schema=False)
    app.add_api_route("/completions", completions_endpoint,
                      methods=["POST"], include_in_schema=False)
