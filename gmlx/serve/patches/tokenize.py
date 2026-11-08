"""``POST /tokenize`` and ``POST /detokenize``, with vLLM's bodies.

``/tokenize`` takes a ``prompt`` string, tokenized as ``/v1/completions``
tokenizes it (``add_special_tokens`` default true), or chat ``messages``,
rendered and tokenized as the chat route renders and tokenizes them. So a
client gets the exact ids a chat request reads, and can send them back as a
completions id prompt. The answer is ``{"tokens", "count",
"max_model_len"}``, plus ``token_strs`` when ``return_token_strs`` is set.
``/detokenize`` turns ``tokens`` back into ``{"prompt": text}``."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import inspect
import logging
from typing import Any, Optional, cast

from fastapi import HTTPException, Request  # module-level: stringized annotations
from pydantic import BaseModel, ConfigDict

from ._common import _remove_routes

TOKENIZE_PATHS = ("/tokenize", "/v1/tokenize")
DETOKENIZE_PATHS = ("/detokenize", "/v1/detokenize")
_NO_MEDIA = ("/tokenize takes text messages only; images, audio and video "
             "expand to ids the model's processor makes, which a client "
             "cannot send back")

_log = logging.getLogger(__name__)


class TokenizeRequest(BaseModel):
    """A prompt or chat messages. Extras ride along for the chat render
    (``chat_template_kwargs``, ``add_generation_prompt`` and the like)."""
    model_config = ConfigDict(extra="allow")

    model: str = ""
    prompt: Any = None
    messages: Any = None
    add_special_tokens: Any = None
    return_token_strs: Any = None


class DetokenizeRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = ""
    tokens: Any = None


def _bad(detail: str) -> HTTPException:
    return HTTPException(status_code=400, detail=detail)


def _flag(value, name: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise _bad(f"'{name}' must be true or false")
    return value


@contextlib.contextmanager
def _engine(model: str):
    """The engine of ``model``, loaded and held for the block."""
    app_mod = importlib.import_module("gmlx._vendor.mlx_vlm.server.app")
    runtime = importlib.import_module("gmlx._vendor.mlx_vlm.server.runtime").runtime
    _model, processor, config = app_mod.get_cached_model(model)
    guard = runtime.response_generator
    if guard is None:
        raise HTTPException(status_code=500, detail=(
            "the model engine is unavailable; restart the server"))
    rg = getattr(guard, "_rg", guard)
    hold = getattr(guard, "_hold", None)
    try:
        yield rg, processor if processor is not None else rg.processor, config
    finally:
        if hold is not None:
            hold.release()


def _tokenizer(processor):
    return getattr(processor, "tokenizer", processor)


def _token_strs(tokenizer, ids: list, lock) -> list:
    with lock:
        for tok in (tokenizer, getattr(tokenizer, "_tokenizer", None)):
            convert = getattr(tok, "convert_ids_to_tokens", None)
            if callable(convert):
                return [str(s) for s in cast(list, convert(ids))]
        return [tokenizer.decode([t]) for t in ids]


def max_model_len(model: str) -> Optional[int]:
    """The context a request may fill: the configured limit, else the
    model's trained context. None when neither can be read."""
    gen_mod = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")
    try:
        limit = gen_mod.get_configured_context_limit()
        if limit is not None:
            return int(limit)
        import gmlx.serve.bridge_vlm as serving
        from gmlx.serve import capacity

        if serving.server_config() is None:
            return None
        path, spec = serving.resolve_request_model(
            model, profile_field=serving.get_request_profile())
        trained = capacity.effective_context_length(
            path, getattr(spec, "rope_scaling", None))
        return int(trained) if trained else None
    except Exception:  # noqa: BLE001 - an unknown limit is reported as null
        _log.debug("tokenize: context limit unreadable", exc_info=True)
        return None


def _message_ids(request: TokenizeRequest, rg, processor, config, http_request) -> list:
    from gmlx._vendor.mlx_vlm.server.app import _read_tenant_id
    from gmlx.serve.estimate import _normalize_messages, render_chat
    from gmlx.serve.patches.completions import _row_ids
    from gmlx.serve.patches.prefill import body_error

    if not isinstance(request.messages, list) or not request.messages:
        raise _bad("'messages' must be a non-empty list")
    body = request.model_dump()
    body.pop("prompt", None)
    rule = body_error(body)
    if rule:
        raise _bad(rule)
    msgs, media = _normalize_messages(request.messages)
    if media:
        raise _bad(_NO_MEDIA)
    try:
        prompt, _ = render_chat(body, processor, config, msgs,
                                tenant_id=_read_tenant_id(http_request))
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 - a template error is the client's 400
        raise _bad(f"cannot render the messages: {e}") from None
    lock = getattr(rg, "_tokenizer_lock", None) or contextlib.nullcontext()
    with lock:
        return _row_ids(rg._preprocess_request(prompt))


async def tokenize_endpoint(request: TokenizeRequest, http_request: Request):
    from gmlx.serve.patches.completions import _tokenize

    has_prompt, has_messages = request.prompt is not None, request.messages is not None
    if has_prompt == has_messages:
        raise _bad("send one of 'prompt' (a string) or 'messages'")
    if has_prompt and (not isinstance(request.prompt, str) or not request.prompt):
        raise _bad("'prompt' must be a non-empty string")
    special = _flag(request.add_special_tokens, "add_special_tokens", True)
    want_strs = _flag(request.return_token_strs, "return_token_strs", False)

    def _run():
        with _engine(request.model) as (rg, processor, config):
            tokenizer = _tokenizer(processor)
            if has_prompt:
                ids = _tokenize(rg, tokenizer, [request.prompt], special)[0].ids
            else:
                ids = _message_ids(request, rg, processor, config, http_request)
            lock = getattr(rg, "_tokenizer_lock", None) or contextlib.nullcontext()
            strs = _token_strs(tokenizer, ids, lock) if want_strs else None
        return ids, strs

    ids, strs = await asyncio.to_thread(_run)
    out: dict = {"tokens": ids, "count": len(ids),
                 "max_model_len": max_model_len(request.model)}
    if strs is not None:
        out["token_strs"] = strs
    return out


async def detokenize_endpoint(request: DetokenizeRequest, http_request: Request):
    from gmlx.serve.patches.completions import _is_id, _vocab_size

    ids = request.tokens
    if not isinstance(ids, list) or not all(_is_id(t) for t in ids):
        raise _bad("'tokens' must be a list of token ids (integers)")

    def _run():
        with _engine(request.model) as (rg, processor, _config):
            tokenizer = _tokenizer(processor)
            vocab = _vocab_size(tokenizer)
            for t in ids:
                if t < 0 or (vocab is not None and t >= vocab):
                    raise _bad(f"token id {t} is out of range")
            lock = getattr(rg, "_tokenizer_lock", None) or contextlib.nullcontext()
            with lock:
                return tokenizer.decode(ids)

    return {"prompt": await asyncio.to_thread(_run)}


# Real-class signatures, as the completions route sets them: the route
# wrappers copy ``__signature__`` into endpoints in other modules.
for _endpoint, _cls in ((tokenize_endpoint, TokenizeRequest),
                        (detokenize_endpoint, DetokenizeRequest)):
    _endpoint.__signature__ = inspect.Signature([  # type: ignore[attr-defined]
        inspect.Parameter("request", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                          annotation=_cls),
        inspect.Parameter("http_request", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                          annotation=Request),
    ])


def install_tokenize_routes() -> None:
    """Register ``POST /tokenize`` and ``POST /detokenize`` (and their
    ``/v1`` spellings). Install before the profile-capture wrapper so it
    fills the default model and binds the profile."""
    app = importlib.import_module("gmlx._vendor.mlx_vlm.server.app").app
    _remove_routes(app, *TOKENIZE_PATHS, *DETOKENIZE_PATHS)
    for path in TOKENIZE_PATHS:
        app.add_api_route(path, tokenize_endpoint, methods=["POST"],
                          include_in_schema=False)
    for path in DETOKENIZE_PATHS:
        app.add_api_route(path, detokenize_endpoint, methods=["POST"],
                          include_in_schema=False)
