"""``POST /v1/systemone``: structured decisions on a served DiffusionGemma
model, in the Jev decision API's request and answer shapes.

The route parses the body, picks the model, checks the request against
the context and memory budgets, and runs the decision as one job on the
model's engine thread (``engine_jobs``). The decision logic and the reads
live in ``gmlx.systemone``."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import logging
import threading
import time
import uuid

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response

import gmlx.serve.bridge_vlm as serving
from gmlx.systemone import (
    Limits,
    SchemaError,
    TemplateResolver,
    decide,
    jev_answers,
    jev_schema,
    jev_state,
    log_labels,
    parse_seed,
    system_text,
    usage,
)
from gmlx.systemone.decide import chunk_groups
from gmlx.systemone.schema import schedule

from ._common import _error_content, _remove_routes
from .api_contract import SYSTEMONE_CONSUMED, warn_ignored_fields

SYSTEMONE_PATHS = ("/systemone", "/v1/systemone")
_ENDPOINT = "/v1/systemone"
_RUNTIME_ATTR = "_kq_systemone_runtime"
_DISCONNECT_POLL_S = 0.5
_NO_IMAGES = "images are not supported: the served model is text-only"

_log = logging.getLogger(__name__)
_runtime_lock = threading.Lock()


def _error(status: int, err_type: str, message: str, **extra) -> JSONResponse:
    return JSONResponse(status_code=status, content=_error_content(
        _ENDPOINT, status, err_type, message, **extra))


def _tokens(rg, state_text: str = ""):
    from gmlx.systemone.engine import ChatTokens

    return ChatTokens(rg.processor, state_text,
                      lock=rg._tokenizer_lock)


class _ModelRuntime:
    """Per-model state kept on the model's ResponseGenerator: the template
    resolver, the reader (built on the engine thread by the first job) and
    the served canvas."""

    def __init__(self, rg, canvas: int):
        self.canvas_len = min(int(canvas), int(rg.model.config.canvas_length))
        self.prefill_step_size = int(rg._effective_prefill_step_size())
        self.resolver = TemplateResolver(_tokens(rg).enc, self.canvas_len)
        self.reader = None


def _runtime_for(rg, canvas: int) -> _ModelRuntime:
    """The model's runtime for a canvas setting, built once per pair."""
    with _runtime_lock:
        by_canvas = getattr(rg, _RUNTIME_ATTR, None)
        if by_canvas is None:
            by_canvas = {}
            setattr(rg, _RUNTIME_ATTR, by_canvas)
        rt = by_canvas.get(int(canvas))
        if rt is None:
            rt = by_canvas[int(canvas)] = _ModelRuntime(rg, canvas)
    return rt


def _settings(installed):
    """``server.systemone`` from the live config, so a reload applies it,
    else the settings the route was installed with."""
    live = serving.server_config()
    return getattr(live, "systemone", None) or installed


def _pick_model(field, profile, cfg) -> str:
    """The model string to load. A name the resolver knows wins. An absent
    or unknown name falls back to ``server.systemone.model``, then to the
    server's default model. A name the client sent that nothing resolves
    keeps its own 404."""
    if serving.server_config() is None:
        return str(field or "")
    raw = str(field or "").strip()
    not_found = None
    if raw:
        try:
            serving.resolve_request_model(raw, profile_field=profile)
            return raw
        except serving.ModelNotFound as e:
            not_found = e
    if cfg.model:
        return cfg.model
    try:
        serving.resolve_request_model(None, profile_field=profile)
    except serving.NoModelSpecified:
        if not_found is not None:
            raise not_found from None
        raise
    return ""


def _sequential_reads(schema, resolver) -> int:
    """The most reads that can run one after another on one growing prompt:
    one per stage, or one per chunk when chunks run in sequence."""
    qs = [q for q in schema["questions"]
          if not schema.get("ask") or q["id"] in schema["ask"]]
    levels = schedule(qs)
    if not schema["sequential"]:
        return len(levels)
    return sum(len(chunk_groups(schema, level, resolver)) for level in levels)


def _admit(rg, rt, tokens, schema) -> int:
    """Check the largest prompt the decision can build against the memory
    and context budgets. Returns its token count."""
    gen = importlib.import_module("mlx_vlm.server.generation")
    from gmlx.serve.mem_preflight import preflight_prompt_memory

    resolver = rt.resolver
    # With think "auto", the second run's thought is the largest prompt.
    think = int(schema["think"] or (schema.get("think_auto") or {}).get("budget", 0))
    bound_text = tokens.render(system_text(schema, chunked=True), think > 0)
    bound_ids = tokens.encode_prompt(bound_text)
    growth = (think + len(resolver.thought_open) + len(resolver.thought_close)
              + len(resolver.scaffold)
              + (_sequential_reads(schema, resolver) - 1) * rt.canvas_len)
    preflight_prompt_memory(
        rg, bound_text,
        args=gen.GenerationArguments(max_tokens=growth + rt.canvas_len))
    gen._check_configured_context_budget(len(bound_ids) + growth, rt.canvas_len)
    return len(bound_ids)


def _record_failure(runtime, model, error: str) -> None:
    try:
        runtime.metrics.record_failure(endpoint=_ENDPOINT, model=model,
                                       stream=False, error=error)
    except Exception:  # noqa: BLE001 - metrics must not fail the request
        _log.debug("failure-metrics record raised", exc_info=True)


async def _watch_disconnect(http_request: Request, stop: threading.Event):
    try:
        while not stop.is_set():
            if await http_request.is_disconnected():
                stop.set()
                return
            await asyncio.sleep(_DISCONNECT_POLL_S)
    except asyncio.CancelledError:
        pass


def make_systemone_endpoint(installed):
    async def systemone_endpoint(http_request: Request):
        request_start = time.perf_counter()
        cfg = _settings(installed)
        limits = Limits(max_questions=cfg.max_questions,
                        max_samples=cfg.max_samples)
        runtime = importlib.import_module("mlx_vlm.server.runtime").runtime
        shown = {"model": ""}
        runtime.metrics.begin_request(endpoint=_ENDPOINT, model="", stream=False)

        def fail(status: int, err_type: str, message: str) -> JSONResponse:
            _record_failure(runtime, shown["model"], message)
            return _error(status, err_type, message)

        ctype = http_request.headers.get("content-type", "").lower()
        if ctype.startswith("multipart/form-data"):
            return fail(400, "invalid_request_error", _NO_IMAGES)
        try:
            body = json.loads(await http_request.body())
        except Exception as e:  # noqa: BLE001 - any parse failure is a 400
            return fail(400, "invalid_request_error", f"invalid body: {e}")
        if not isinstance(body, dict):
            return fail(400, "invalid_request_error",
                        "invalid body: expected a JSON object")
        requested = body.get("model")
        shown["model"] = requested if isinstance(requested, str) else ""
        if body.get("images"):
            return fail(400, "invalid_request_error", _NO_IMAGES)
        warn_ignored_fields(_ENDPOINT, set(body) - SYSTEMONE_CONSUMED)
        try:
            schema = jev_schema(body, limits, cfg.request_defaults())
            state = jev_state(body)
            seed = parse_seed(body)
        except SchemaError as e:
            return fail(422, "validation_error", str(e))
        profile = body.get("profile") if isinstance(body.get("profile"), str) else None

        app_mod = importlib.import_module("mlx_vlm.server.app")
        stop = threading.Event()
        request_id = f"so-{uuid.uuid4().hex[:12]}"

        def _run():
            from gmlx.gen.diffusion import is_diffusion_model
            from gmlx.serve.engine_jobs import run_on_engine
            from gmlx.serve.residency import _http_from_resolver_error
            from gmlx.systemone.engine import BoundReader, StructuredReader, engine_scope

            gen = importlib.import_module("mlx_vlm.server.generation")
            # The residency seam resolves the model string with the body
            # profile from this context variable.
            token = serving.set_request_profile(profile)
            try:
                try:
                    model_str = _pick_model(body.get("model"), profile, cfg)
                except (serving.ModelNotFound, serving.ModelFileMissing,
                        serving.UnknownProfile, serving.NoModelSpecified) as e:
                    raise _http_from_resolver_error(e) from e
                app_mod.get_cached_model(model_str)
                spec = serving.get_active_spec()
            finally:
                serving.reset_request_profile(token)
            guard = runtime.response_generator
            rg = getattr(guard, "_rg", guard)
            hold = getattr(guard, "_hold", None)
            try:
                if rg is None:
                    raise HTTPException(status_code=500, detail=(
                        "the model engine is unavailable; restart the server"))
                resolved = spec.id if spec is not None else (model_str or shown["model"])
                shown["model"] = resolved
                if not is_diffusion_model(getattr(rg, "model", None)):
                    raise HTTPException(status_code=400, detail=(
                        f"model {resolved!r} is not a diffusion model; "
                        "/v1/systemone needs a DiffusionGemma model"))
                rt = _runtime_for(rg, cfg.canvas)
                tokens = _tokens(rg, state)
                admitted = _admit(rg, rt, tokens, schema)

                def job(engine_rg, should_stop):
                    with engine_scope(engine_rg.model, seed):
                        tok = engine_rg.tokenizer
                        criteria = getattr(tok, "stopping_criteria", None)
                        if criteria is not None:
                            criteria.reset(engine_rg.config.eos_token_id)
                        if rt.reader is None:
                            rt.reader = StructuredReader(
                                engine_rg.model,
                                prefill_step_size=rt.prefill_step_size)
                        engine = BoundReader(
                            rt.reader, processor=engine_rg.processor,
                            backend=tok, should_stop=should_stop)
                        return decide(
                            schema, state, engine=engine, resolver=rt.resolver,
                            chat_ids=tokens.chat_ids, seed=seed,
                            constrained=cfg.constrained,
                            canvas_len=rt.canvas_len, decode=tokens.decode,
                            should_stop=should_stop)

                result, completion_tokens = run_on_engine(
                    rg, job, request_id=request_id, prompt_tokens=admitted,
                    stop=stop, timeout_s=gen.get_token_queue_timeout())
                return resolved, result, completion_tokens
            finally:
                if hold is not None:
                    hold.release()

        watcher = asyncio.create_task(_watch_disconnect(http_request, stop))
        try:
            resolved, result, completion_tokens = await asyncio.to_thread(_run)
        except SchemaError as e:
            return fail(422, "validation_error", str(e))
        except HTTPException as e:
            _record_failure(runtime, shown["model"], str(e.detail))
            raise
        except Exception as e:  # noqa: BLE001 - mapped to a status below
            from gmlx.serve.engine_jobs import JobCancelled, JobTimeout

            gen = importlib.import_module("mlx_vlm.server.generation")
            if isinstance(e, JobCancelled):
                _record_failure(runtime, shown["model"], "client disconnected")
                return Response(status_code=499)
            if isinstance(e, JobTimeout):
                return fail(504, "timeout", str(e))
            if isinstance(e, gen.PromptTooLongError):
                return fail(400, "invalid_request_error", str(e))
            _log.exception("systemone: decision failed")
            _record_failure(runtime, shown["model"], str(e))
            return _error(500, "server_error",
                          f"decision failed ({type(e).__name__}); see the server log")
        finally:
            stop.set()
            watcher.cancel()

        diagnostics = result["diagnostics"]
        input_tokens = int(diagnostics.get("prompt_tokens") or 0)
        auto = diagnostics.get("think_auto") or {}
        _log.info("systemone: %s reads=%d %.0fms%s", log_labels(result["answers"]),
                  diagnostics["timing"]["reads"], diagnostics["timing"]["total_ms"],
                  " thought=auto" if auto.get("thought") else "")
        gen = importlib.import_module("mlx_vlm.server.generation")
        try:
            runtime.metrics.record_success(gen._build_metrics_envelope(
                endpoint=_ENDPOINT, model=resolved, stream=False,
                backend="diffusion", prompt_tokens=input_tokens,
                completion_tokens=int(completion_tokens),
                generated_tokens=int(completion_tokens),
                request_elapsed_s=time.perf_counter() - request_start,
                request_started_s=request_start))
        except Exception:  # noqa: BLE001 - metrics must not fail the request
            _log.debug("success-metrics record raised", exc_info=True)
        return JSONResponse(content={
            "model": resolved,
            "answers": jev_answers(schema, result),
            "usage": usage(input_tokens, completion_tokens),
            "diagnostics": diagnostics,
        })

    # The request wrappers copy this signature onto endpoints defined in
    # other modules, where the deferred annotation would not resolve.
    systemone_endpoint.__signature__ = inspect.Signature([inspect.Parameter(
        "http_request", inspect.Parameter.POSITIONAL_OR_KEYWORD,
        annotation=Request)])
    return systemone_endpoint


def install_systemone_route(cfg) -> None:
    """Register ``POST /v1/systemone`` (+ ``/systemone``). Install before the
    load-offload, profile-capture and queue-cap wrappers so they wrap it."""
    app = importlib.import_module("mlx_vlm.server.app").app
    endpoint = make_systemone_endpoint(cfg)
    _remove_routes(app, *SYSTEMONE_PATHS)
    for path in SYSTEMONE_PATHS:
        app.add_api_route(path, endpoint, methods=["POST"],
                          include_in_schema=False)
