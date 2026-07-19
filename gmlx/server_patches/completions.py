"""Minimal OpenAI ``/v1/completions`` route (single string prompt, n=1).

mlx-vlm serves only the chat-shaped routes; legacy clients and benchmark
harnesses still probe the classic text-completions endpoint. This route
drives the exact engine path the chat handler uses minus the chat template:
``ResponseGenerator.generate`` tokenizes a raw prompt string directly, and
``_build_gen_args`` is request-shape agnostic, so sampling defaults, profile
injection, XTC, and the residency pool behave exactly as on
``/v1/chat/completions``. Out of scope (400): list / token-array prompts,
``n > 1``, ``echo``, ``suffix``, ``best_of > 1``."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import time
import uuid
from typing import Any, List, Optional

from fastapi import HTTPException, Request  # module-level: stringized annotations
from pydantic import BaseModel, ConfigDict

from ..generation import StopScanner
from ._common import _remove_routes
from .api_contract import COMPLETIONS_CONSUMED, warn_ignored_fields
from .chat_behavior import _request_stop_sequences

_COMPLETIONS_PATHS = ("/completions", "/v1/completions")

# Bound once at install time (the route cannot run before install); the
# sibling patches bind their upstream modules the same way.
_app_mod = None
_gen_mod = None
_runtime_mod = None


def _modules():
    global _app_mod, _gen_mod, _runtime_mod
    if _app_mod is None:
        _app_mod = importlib.import_module("mlx_vlm.server.app")
        _gen_mod = importlib.import_module("mlx_vlm.server.generation")
        _runtime_mod = importlib.import_module("mlx_vlm.server.runtime")
    return _app_mod, _gen_mod, _runtime_mod


class CompletionRequest(BaseModel):
    """Local request shape - upstream ``schemas.py`` stays untouched. Extras
    ride along (``extra="allow"``) so ``_build_gen_args`` and the gmlx
    gen-args transforms read top_k / min_p / profile / xtc_* off it like any
    other request; ``prompt`` stays ``Any`` so the handler can 400 the
    unsupported shapes with a clear message instead of a pydantic 422."""
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


def _validate(request: CompletionRequest) -> None:
    """400s for the documented out-of-scope shapes, with a next step."""
    if not isinstance(request.prompt, str) or not request.prompt:
        raise HTTPException(
            status_code=400,
            detail="'prompt' must be a single non-empty string; lists and "
                   "token arrays are not supported (send one request per "
                   "prompt)")
    if request.n is not None and request.n != 1:
        raise HTTPException(
            status_code=400,
            detail="'n' greater than 1 is not supported; send parallel "
                   "requests instead")
    if request.echo:
        raise HTTPException(
            status_code=400,
            detail="'echo' is not supported; prepend the prompt client-side")
    if request.suffix:
        raise HTTPException(
            status_code=400,
            detail="'suffix' (fill-in-the-middle) is not supported on this "
                   "server")
    if request.best_of is not None and request.best_of != 1:
        raise HTTPException(
            status_code=400,
            detail="'best_of' greater than 1 is not supported")


def _include_usage(request: CompletionRequest) -> bool:
    so = request.stream_options
    if isinstance(so, dict):
        return bool(so.get("include_usage"))
    return bool(getattr(so, "include_usage", False))


def _record_failure(runtime, model: str, stream: bool, error: str) -> None:
    try:
        runtime.metrics.record_failure(endpoint="/v1/completions",
                                       model=model, stream=stream,
                                       error=error)
    except Exception:
        pass


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


async def completions_endpoint(request: CompletionRequest,
                               http_request: Request):
    # Default-model fill and the request-profile contextvar are applied by
    # request_flow's profile-capture wrapper, which wraps this route
    # (_PROFILE_CAPTURE_PATHS lists it; install order in server_patches
    # __init__ registers the route first). Generation starts inside the
    # wrapper's window - ``_start`` runs before the StreamingResponse is
    # returned - so nothing here re-reads the profile after reset.
    request_start = time.perf_counter()
    app_mod, gen_mod, runtime_mod = _modules()
    runtime = runtime_mod.runtime

    _validate(request)
    warn_ignored_fields("/v1/completions",
                        set(request.model_fields_set) - COMPLETIONS_CONSUMED)
    stops = _request_stop_sequences(request)
    prompt = request.prompt

    def _start():
        """Resolve + load + build args + start generation, all in one worker
        thread so the request context (profile var, active spec, residency
        entry) stays coherent from resolution through ``generate``. Raising
        here (resolver 404/400, PromptTooLongError) surfaces before any
        streaming starts."""
        _model, processor, _config = app_mod.get_cached_model(request.model)
        # The (patched) _build_gen_args applies profile injection / XTC /
        # template-kwargs transforms exactly as on the chat routes.
        gen_args = app_mod._build_gen_args(
            request, processor, tenant_id=app_mod._read_tenant_id(http_request))
        # Text-completions never returns token logprobs; don't collect them.
        gen_args.logprobs = False
        # runtime.response_generator resolves against the residency entry
        # get_cached_model bound to this context.
        rg = runtime.response_generator
        if rg is None:
            raise HTTPException(
                status_code=500,
                detail="continuous-batching engine unavailable; restart the "
                       "server")
        return rg.generate(prompt, args=gen_args)

    if request.stream:
        return await _stream_completion(
            request, runtime, gen_mod, _start, stops, request_start)
    return await _blocking_completion(
        request, runtime, gen_mod, _start, stops, request_start)


async def _blocking_completion(request, runtime, gen_mod, start, stops,
                               request_start):
    runtime.metrics.begin_request(endpoint="/v1/completions",
                                  model=request.model, stream=False)

    def _run():
        ctx, token_iter = start()
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
            except Exception:
                pass
        if stops and not hit_stop:
            text += scanner.flush()
        finish = "stop" if hit_stop else (finish or "stop")
        return ctx.prompt_tokens, text, output_tokens, finish, metrics

    try:
        prompt_tokens, text, output_tokens, finish, metrics = \
            await asyncio.to_thread(_run)
    except gen_mod.PromptTooLongError as e:
        _record_failure(runtime, request.model, False, str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        _record_failure(runtime, request.model, False, str(e.detail))
        raise
    except Exception as e:
        _record_failure(runtime, request.model, False, str(e))
        # Generic body: raw exception text can carry paths and internals;
        # the recorded failure reaches the operator via the [req] log line.
        raise HTTPException(status_code=500,
                            detail=f"generation failed "
                                   f"({type(e).__name__}); see the server log")

    runtime.metrics.record_success(_completion_envelope(
        gen_mod, model=request.model, stream=False,
        prompt_tokens=prompt_tokens, completion_tokens=output_tokens,
        request_start=request_start, metrics=metrics, finish_reason=finish))
    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4()}",
        created=int(time.time()),
        model=request.model,
        choices=[CompletionChoice(text=text, finish_reason=finish)],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=output_tokens,
            total_tokens=prompt_tokens + output_tokens))


async def _stream_completion(request, runtime, gen_mod, start, stops,
                             request_start):
    from starlette.responses import StreamingResponse

    runtime.metrics.begin_request(endpoint="/v1/completions",
                                  model=request.model, stream=True)
    try:
        ctx, token_iter = await asyncio.to_thread(start)
    except gen_mod.PromptTooLongError as e:
        _record_failure(runtime, request.model, True, str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        _record_failure(runtime, request.model, True, str(e.detail))
        raise
    except Exception as e:
        _record_failure(runtime, request.model, True, str(e))
        raise HTTPException(status_code=500,
                            detail=f"generation failed "
                                   f"({type(e).__name__}); see the server log")

    rid = f"cmpl-{uuid.uuid4()}"
    created = int(time.time())
    include_usage = _include_usage(request)

    def _chunk(text: str, finish: str | None) -> str:
        return "data: " + json.dumps({
            "id": rid, "object": "text_completion", "created": created,
            "model": request.model,
            "choices": [{"index": 0, "text": text, "logprobs": None,
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
                    token_iter.close()  # normal end or disconnect: cancels
                except Exception:       # the in-flight batch generation
                    pass

        threading.Thread(target=_pump, name="gmlx-completions-pump",
                         daemon=True).start()
        try:
            while True:
                tok = await q.get()
                if tok is DONE:
                    break
                if isinstance(tok, BaseException):
                    raise tok
                output_tokens += getattr(tok, "token_count", 1)
                metrics.record_chunk(tok)
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
                        "prompt_tokens": ctx.prompt_tokens,
                        "completion_tokens": output_tokens,
                        "total_tokens": ctx.prompt_tokens + output_tokens,
                    }}) + "\n\n"
            yield "data: [DONE]\n\n"
            runtime.metrics.record_success(_completion_envelope(
                gen_mod, model=request.model, stream=True,
                prompt_tokens=ctx.prompt_tokens,
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
    app = importlib.import_module("mlx_vlm.server.app").app
    _remove_routes(app, *_COMPLETIONS_PATHS)
    app.add_api_route("/v1/completions", completions_endpoint,
                      methods=["POST"], include_in_schema=False)
    app.add_api_route("/completions", completions_endpoint,
                      methods=["POST"], include_in_schema=False)
