"""``POST /v1/systemone``: structured decisions in the Jev decision API's
request and answer shapes.

The route parses the body, picks the model, checks the request against
the context and memory budgets, and runs the decision as a job on the
model's engine thread (``engine_jobs``). A DiffusionGemma model reads its
answer slots in one job. Any other text model answers with the letter
readout, as a step job between batch steps. The decision logic and the
reads live in ``gmlx.systemone``."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import inspect
import json
import logging
import threading
import time
import uuid
from typing import TYPE_CHECKING, Optional

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response

import gmlx.serve.bridge_vlm as serving
from gmlx import lora_rows
from gmlx.systemone import ar_reader, letters
from gmlx.systemone import (
    Limits,
    SchemaError,
    TemplateResolver,
    decide,
    jev_response,
    log_labels,
    system_text,
)
from gmlx.systemone.backends import ParsedRequest
from gmlx.systemone.decide import chunk_groups
from gmlx.systemone.schema import schedule

from ._common import _error_content, _remove_routes
from .api_contract import SYSTEMONE_CONSUMED, warn_ignored_fields

if TYPE_CHECKING:
    from gmlx.systemone.engine import StructuredReader

SYSTEMONE_PATHS = ("/systemone", "/v1/systemone")
_ENDPOINT = "/v1/systemone"
_RUNTIME_ATTR = "_kq_systemone_runtime"
_LETTERS_ATTR = "_kq_systemone_letters"
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
        self.reader: Optional[StructuredReader] = None


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


class _LetterRuntime:
    """Per-model letter state kept on the model's ResponseGenerator: the
    letter ids and the reader, built on the engine thread by the first job."""

    def __init__(self, rg):
        self.letter_ids = ar_reader.LetterTokens(
            rg.processor, lock=rg._tokenizer_lock).letter_ids()
        self.reader: Optional[ar_reader.LetterReader] = None


def _letters_for(rg) -> _LetterRuntime:
    with _runtime_lock:
        rt = getattr(rg, _LETTERS_ATTR, None)
        if rt is None:
            rt = _LetterRuntime(rg)
            setattr(rg, _LETTERS_ATTR, rt)
    return rt


def _rows_scope(scales):
    """Publishes the request's adapter scales around each reader forward
    while the server runs adapters per row."""
    @contextlib.contextmanager
    def scope(rows: int):
        if lora_rows.mode() != "rows":
            yield
            return
        pad = (0.0,) * max(lora_rows.n_slots() - len(scales), 0)
        lora_rows.set_rows([tuple(scales) + pad] * rows)
        try:
            yield
        finally:
            lora_rows.clear_rows()
    return scope


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


def _carried_canvases(schema, resolver) -> int:
    """How many reads' answer lines the last prompt of a decision can carry,
    each at most one canvas. A later stage carries every chunk of every
    earlier stage, and a chunk read in sequence also carries the chunks
    before it in its own stage."""
    qs = [q for q in schema["questions"]
          if not schema.get("ask") or q["id"] in schema["ask"]]
    chunks = [len(chunk_groups(schema, level, resolver)) for level in schedule(qs)]
    if schema["sequential"]:
        return sum(chunks) - 1
    return sum(chunks[:-1])


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
              + _carried_canvases(schema, resolver) * rt.canvas_len)
    preflight_prompt_memory(
        rg, bound_text,
        args=gen.GenerationArguments(max_tokens=growth + rt.canvas_len))
    gen._check_configured_context_budget(len(bound_ids) + growth, rt.canvas_len)
    return len(bound_ids)


def _admit_letters(rg, tokens, schema, state: str) -> int:
    """Tokenize every pass the decision can send and check the longest
    against the memory and context budgets. Returns its token count. The
    reader holds at most one forward of tail rows beyond that prompt."""
    gen = importlib.import_module("mlx_vlm.server.generation")
    from gmlx.serve.mem_preflight import preflight_prompt_memory

    tokens.ids(letters.prefix_text(state))
    texts = letters.pass_texts(schema, state) + letters.winner_bounds(schema, state)
    longest = max(texts, key=lambda t: len(tokens.ids(t)))
    n = len(tokens.ids(longest))
    preflight_prompt_memory(
        rg, tokens.render(longest),
        args=gen.GenerationArguments(max_tokens=ar_reader.FORWARD_TOKENS))
    gen._check_configured_context_budget(n, 1)
    return n


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
        unread = set(body) - SYSTEMONE_CONSUMED
        parsed = ParsedRequest(body, limits, cfg.request_defaults())
        if parsed.error is not None:
            warn_ignored_fields(_ENDPOINT, unread)
            return fail(422, "validation_error", str(parsed.error))
        profile = body.get("profile") if isinstance(body.get("profile"), str) else None

        app_mod = importlib.import_module("mlx_vlm.server.app")
        stop = threading.Event()
        request_id = f"so-{uuid.uuid4().hex[:12]}"

        def _backend(kind: str):
            try:
                got = parsed.get(kind)
            except SchemaError:
                warn_ignored_fields(_ENDPOINT, unread)
                raise
            warn_ignored_fields(_ENDPOINT, unread | parsed.ignored(kind))
            return got

        def _diffusion_job(rg):
            from gmlx.systemone.engine import BoundReader, StructuredReader, engine_scope

            schema, state, seed = _backend("diffusion")
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

            return schema, job, admitted

        def _letters_job(rg, spec, resolved):
            schema, state = _backend("letters")
            if letters.has_image(body.get("state")):
                raise HTTPException(status_code=400, detail=_NO_IMAGES)
            try:
                lrt = _letters_for(rg)
                tokens = ar_reader.LetterTokens(rg.processor, lock=rg._tokenizer_lock)
                tokens.render(letters.prefix_text(state))
            except ValueError as e:
                raise HTTPException(status_code=400, detail=(
                    f"model {resolved!r} cannot answer letter reads: {e}")) from e
            admitted = _admit_letters(rg, tokens, schema, state)
            scope = _rows_scope(lora_rows.request_scales(spec))

            def job(engine_rg, should_stop):
                if lrt.reader is None:
                    lrt.reader = ar_reader.LetterReader(engine_rg.model, lrt.letter_ids)
                return ar_reader.decide_letters(
                    lrt.reader, tokens, schema, state,
                    should_stop=should_stop, rows_scope=scope)

            return schema, job, admitted

        def _run():
            from gmlx.gen.diffusion import is_diffusion_model
            from gmlx.serve.engine_jobs import run_on_engine
            from gmlx.serve.residency import _http_from_resolver_error

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
                if is_diffusion_model(getattr(rg, "model", None)):
                    schema, job, admitted = _diffusion_job(rg)
                    result, completion_tokens = run_on_engine(
                        rg, job, request_id=request_id, prompt_tokens=admitted,
                        stop=stop, timeout_s=gen.get_token_queue_timeout())
                    return "diffusion", schema, resolved, result, completion_tokens
                schema, job, admitted = _letters_job(rg, spec, resolved)
                result = run_on_engine(
                    rg, job, request_id=request_id, prompt_tokens=admitted,
                    stop=stop, timeout_s=gen.get_token_queue_timeout(),
                    stepwise=True)
                return "letters", schema, resolved, result, 0
            finally:
                if hold is not None:
                    hold.release()

        watcher = asyncio.create_task(_watch_disconnect(http_request, stop))
        try:
            backend, schema, resolved, result, completion_tokens = (
                await asyncio.to_thread(_run))
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

        try:
            content = jev_response(schema, result, completion_tokens, resolved)
        except Exception as e:  # noqa: BLE001 - a 500 like any engine failure
            _log.exception("systemone: response assembly failed")
            _record_failure(runtime, resolved, str(e))
            return _error(500, "server_error",
                          f"decision failed ({type(e).__name__}); see the server log")
        diagnostics = result["diagnostics"]
        input_tokens = content["usage"]["input_tokens"]
        auto = diagnostics.get("think_auto") or {}
        _log.info("systemone: %s reads=%d %.0fms%s", log_labels(result["answers"]),
                  diagnostics["timing"]["reads"], diagnostics["timing"]["total_ms"],
                  " thought=auto" if auto.get("thought") else "")
        gen = importlib.import_module("mlx_vlm.server.generation")
        try:
            runtime.metrics.record_success(gen._build_metrics_envelope(
                endpoint=_ENDPOINT, model=resolved, stream=False,
                backend=backend, prompt_tokens=input_tokens,
                completion_tokens=int(completion_tokens),
                generated_tokens=int(completion_tokens),
                request_elapsed_s=time.perf_counter() - request_start,
                request_started_s=request_start))
        except Exception:  # noqa: BLE001 - metrics must not fail the request
            _log.debug("success-metrics record raised", exc_info=True)
        return JSONResponse(content=content)

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
