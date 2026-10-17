"""Prompt upsampling on the image routes.

``revise`` gives the caption the image entry's upsampler writes for a
typed prompt, or None when the prompt runs as typed. The caption comes
from the server's own chat generation on the upsampler, a ``models:``
entry acquired from the residency pool as a chat request acquires it.
A bounded table maps (upsampler, the family's turn, the system prompt
by its sha256, typed prompt) to the caption: decoding is greedy, so a rerun with another seed writes the
same caption, skips the rewrite, and finds its conditioning cached. The
entry's ``upsample_prompt`` file is read on each request, so an edit to
it writes new captions.
"""

from __future__ import annotations

import contextvars
import importlib
import logging
import threading
import time
from collections import OrderedDict

import gmlx.serve.bridge_vlm as serving

_log = logging.getLogger(__name__)

MAX_CAPTIONS = 128


class _Captions:
    """The captions of the latest prompts, at most ``size`` of them."""

    def __init__(self, size: int):
        self.size = size
        self._items: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            caption = self._items.get(key)
            if caption is not None:
                self._items.move_to_end(key)
            return caption

    def put(self, key, caption: str) -> None:
        with self._lock:
            self._items[key] = caption
            self._items.move_to_end(key)
            while len(self._items) > self.size:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


CAPTIONS = _Captions(MAX_CAPTIONS)
_WARNED: set = set()


def upsampler_for(spec, family, requested: bool | None,
                  has_refs: bool) -> str | None:
    """The ``models:`` id that rewrites a prompt of the image entry
    ``spec``, or None to run it as typed. ``requested`` is the request's
    ``prompt_upsampling``, None when it gives none. Raises
    ``ImageRequestError`` for a request that asks for a rewrite this
    entry cannot run."""
    from .image_service import _bad

    if has_refs:
        return None
    if family.upsample is None:
        if requested:
            raise _bad(f"{family.title} takes no prompt upsampling; send "
                       "the request without prompt_upsampling",
                       "prompt_upsampling")
        return None
    if requested is False:
        return None
    upsampler = getattr(spec, "upsampler", None)
    if upsampler is None:
        if requested:
            raise _bad(f"prompt_upsampling needs an upsampler: set "
                       f"upsampler to a models: id on the image_models entry "
                       f"{spec.id!r}", "prompt_upsampling")
        if family.upsample.expected and spec.id not in _WARNED and \
                getattr(spec, "upsample", True):
            _WARNED.add(spec.id)
            _log.warning("image model %r: %s expects its prompts upsampled; "
                         "set upsampler on its image_models entry to the "
                         "models: id of a %s GGUF", spec.id, family.title,
                         family.upsample.suggested)
        return None
    if requested is None and not getattr(spec, "upsample", True):
        return None
    return upsampler


def revise(model_field, prompt: str, requested: bool | None = None,
           has_refs: bool = False, *, stop=None,
           resolve=None) -> str | None:
    """The caption of ``prompt`` for the image model ``model_field``, or
    None when it runs as typed. ``resolve(model_field)`` gives the
    entry's ``(path, spec)``, the server's image models by default. A
    model field that does not resolve, or a file that is no image model,
    returns None, so the image model's own acquire refuses it. ``stop``
    is a threading.Event that ends the rewrite between tokens."""
    from gmlx.image.upsample import (UpsampleError, cache_key, system_prompt,
                                     upsample)

    from .image_service import ImageRequestError
    from .image_steps import family_of

    if not prompt or not prompt.strip():
        return None
    try:
        _path, spec = (resolve or serving.resolve_image_request)(model_field)
        family, _variant = family_of(spec)
    except Exception:  # noqa: BLE001 - the acquire refuses it with its status
        return None
    up = upsampler_for(spec, family, requested, has_refs)
    if up is None:
        return None
    t0 = time.monotonic()
    try:
        system = system_prompt(family, getattr(spec, "upsample_prompt", None))
        key = cache_key(_upsampler_identity(up), family, prompt, system)
        caption = CAPTIONS.get(key)
        if caption is not None:
            return caption
        caption = upsample(family, prompt,
                           lambda req: _chat(up, req, stop),
                           name=f"the upsampler {up!r}", system=system)
    except UpsampleError as e:
        raise ImageRequestError(500, str(e), err_type="server_error") from e
    CAPTIONS.put(key, caption)
    _log.info("[image] prompt upsampled by %s in %.1f s, %d words", up,
              time.monotonic() - t0, len(caption.split()))
    return caption


def _upsampler_identity(up: str) -> tuple:
    """What tells one chat model of the pool from another: the load
    signature of its resolved entry, its adapter and its template
    controls. A config reload that changes any of them writes new
    captions."""
    from .residency import _http_from_resolver_error

    try:
        spec = serving.resolve_request_model(up)[1]
    except (serving.ModelNotFound, serving.ModelFileMissing,
            serving.UnknownProfile, serving.NoModelSpecified) as e:
        raise (_http_from_resolver_error(e) or e) from e
    kwargs = getattr(spec, "chat_template_kwargs", None) or {}
    return (spec.load_signature(), getattr(spec, "adapter", None),
            tuple(sorted((k, repr(v)) for k, v in kwargs.items())),
            repr(getattr(spec, "thinking", None)))


def _chat(up: str, req, stop) -> str:
    """The reply of the chat model ``up`` to the upsampler turn ``req``,
    in a context of its own, so the chat model's request state never
    reaches the image request."""
    return contextvars.copy_context().run(_chat_in_context, up, req, stop)


def _chat_in_context(up: str, req, stop) -> str:
    from fastapi import HTTPException

    from . import residency
    from .image_service import ImageRequestError, JobStopped

    app_mod = importlib.import_module("gmlx._vendor.mlx_vlm.server.app")
    openai_mod = importlib.import_module("gmlx._vendor.mlx_vlm.server.openai")
    gen_mod = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")
    runtime_mod = importlib.import_module("gmlx._vendor.mlx_vlm.server.runtime")
    schemas = importlib.import_module("gmlx._vendor.mlx_vlm.server.schemas")

    messages = [dict(m) for m in req.messages]
    _model, processor, config = app_mod.get_cached_model(up)
    hold = residency._active_hold.get()
    token_iter = None
    try:
        # Every sampling field is set, so no profile of the chat model
        # changes the decode.
        request = schemas.ChatRequest(
            model=up, messages=messages, max_tokens=req.max_tokens,
            temperature=req.temperature, top_p=1.0, top_k=0, min_p=0.0,
            presence_penalty=req.presence_penalty, repetition_penalty=None,
            frequency_penalty=None, xtc_probability=0.0,
            thinking=req.thinking, stream=False)
        gen_args = app_mod._build_gen_args(request, processor)
        gen_args.logprobs = False
        prompt = openai_mod.apply_chat_template(
            processor, config, messages, num_images=0,
            **gen_args.to_template_kwargs())
        rg = runtime_mod.runtime.response_generator
        if rg is None:
            raise HTTPException(status_code=503, detail="the chat engine of "
                                "the upsampler is not running")
        try:
            _ctx, token_iter = rg.generate(prompt, args=gen_args)
        except gen_mod.PromptTooLongError as e:
            limit = gen_mod.get_configured_context_limit()
            room = f" of {limit} tokens" if limit else ""
            raise ImageRequestError(
                500, f"the upsampler {up!r} has no room in its context{room} "
                f"for its system prompt and {req.max_tokens} caption tokens: "
                f"{e}", err_type="server_error") from e
        from gmlx.gen.thinking_budget import prompt_open_think_tag

        # A template that opens the thinking block in the prompt leaves its
        # tag out of the reply, which the parser reads it by.
        text = [prompt_open_think_tag(prompt) or ""]
        for tok in token_iter:
            if stop is not None and stop.is_set():
                raise JobStopped(timed_out=False)
            text.append(tok.text)
            if tok.finish_reason:
                break
        return "".join(text)
    finally:
        if token_iter is not None:
            try:
                token_iter.close()
            except Exception:  # noqa: S110 - the iterator already ended
                pass
        if hold is not None:
            hold.release()
