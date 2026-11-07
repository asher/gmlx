"""Assistant prefill on the chat route: ``continue_final_message`` and
``add_generation_prompt``, with llama-server's rules.

A request that ends with a partial assistant message and sets
``continue_final_message: true, add_generation_prompt: false`` gets a reply
that continues that message. The prompt is the one llama-server renders for
the same messages (:mod:`gmlx.gen.continuation`). ``add_generation_prompt:
false`` alone renders the messages with no generation prompt. A request that
sends neither field renders as before.

Five parts, each idempotent:

* a chat-route wrap that refuses an invalid request before the handler
  loads a model;
* a gen-args transform that checks the template and stashes the
  continuation on the request's ``GenerationArguments``;
* a ``to_template_kwargs`` wrap that passes it to the render;
* a render wrap on every ``apply_chat_template`` binding, between the retire
  capture and the faithful-history wrap;
* the retire capture (``apc.py``) records a reduced context for these
  renders.
"""

from __future__ import annotations

import importlib

from gmlx.gen.continuation import (
    INVALID_VALUE,
    RECIPES,
    continued_prompt,
    handler_for,
    parse_continue,
    recipe_supported,
    resolve,
)
from gmlx.upstream.patching import link, wrapped_with

from .render import _LAST_MESSAGE_ONLY
from ._common import (
    _CHAT_PATHS,
    _error_content,
    _install_gen_args_transform,
    _render_target_modules,
    _wrap_post_routes,
    refusal_type,
)

_FLAG = "_kq_gguf_assistant_prefill"
# The render kwarg that carries the continuation from to_template_kwargs to
# the render wrap. render_continued pops it, so it never reaches Jinja.
CONTINUE_KEY = "_gmlx_continue"

AGP_NOT_BOOL = "add_generation_prompt must be a boolean"
BOTH_TRUE = ("Cannot set both add_generation_prompt and continue_final_message "
             'to true. Send "add_generation_prompt": false with '
             "continue_final_message.")
NOT_ASSISTANT = ("continue_final_message needs the last message to be an "
                 "assistant message.")
TOOL_CALLS = "Cannot continue an assistant message that contains tool calls."
EMPTY = ("continue_final_message needs content or reasoning_content in the "
         "last message.")
RESPONSE_FORMAT = "continue_final_message does not combine with response_format."
UNSUPPORTED = ("continue_final_message is not supported for this model's chat "
               "template.")

_MISSING = object()

# Processor apply_chat_template overrides known to render the template text
# that handler_for reads, as llama-server renders it: (module, class, whether
# it renders with its tokenizer's template). Gemma4Processor renders with
# its tokenizer's, so that must be the template handler_for reads.
_JINJA_OVERRIDES = (
    ("gmlx._vendor.mlx_lm.tokenizer_utils", "TokenizerWrapper", False),
    ("gmlx.serve.bridge_vlm", "_GgufServerProcessor", False),
    ("gmlx._vendor.mlx_vlm.models.gemma4.processing_gemma4", "Gemma4Processor",
     True),
)


class _NotText(Exception):
    """The render returned something other than a prompt string."""


def _fields(request) -> tuple:
    """(continue_final_message, add_generation_prompt) as sent, each
    ``_MISSING`` when absent. Both arrive as ChatRequest extras."""
    extra = getattr(request, "model_extra", None) or {}
    return (extra.get("continue_final_message", _MISSING),
            extra.get("add_generation_prompt", _MISSING))


def _get(obj, key):
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _response_format_set(request) -> bool:
    from gmlx._vendor.mlx_vlm.server.request_normalization import (
        _extract_response_format_schema,
    )

    try:
        return _extract_response_format_schema(request) is not None
    except ValueError:
        # The route answers a malformed format itself.
        return False


def parse_request(request):
    """``(requested, add_generation_prompt, error)`` for a chat request.
    ``requested`` is None without a continuation, ``add_generation_prompt``
    is None when not sent, and ``error`` is the refusal text or None."""
    raw_cont, raw_agp = _fields(request)
    if raw_agp is _MISSING or raw_agp is None:
        agp = None
    elif isinstance(raw_agp, bool):
        agp = raw_agp
    else:
        return None, None, AGP_NOT_BOOL
    try:
        requested = parse_continue(None if raw_cont is _MISSING else raw_cont)
    except ValueError:
        return None, agp, INVALID_VALUE
    if requested is None:
        return None, agp, None
    if agp is not False:
        return requested, agp, BOTH_TRUE
    messages = getattr(request, "messages", None) or []
    last = messages[-1] if messages else None
    if last is None or _get(last, "role") != "assistant":
        return requested, agp, NOT_ASSISTANT
    if _get(last, "tool_calls"):
        return requested, agp, TOOL_CALLS
    if resolve(requested, last) is None:
        return requested, agp, EMPTY
    if _response_format_set(request):
        return requested, agp, RESPONSE_FORMAT
    return requested, agp, None


def body_error(body: dict) -> str | None:
    """The request rule a chat ``body`` breaks, or None. For a caller that
    answers before any model loads, as the route wrap does."""
    try:
        request = _chat_request_type()(**body)
    except Exception:
        # The caller's own validation answers a malformed body.
        return None
    return parse_request(request)[2]


def _chat_request_type():
    return importlib.import_module("gmlx._vendor.mlx_vlm.server.schemas").ChatRequest


def template_for(processor, kw=None) -> str:
    """The template text a render of ``processor`` uses: a ``chat_template``
    kwarg overrides the model's."""
    from .chat_behavior import _model_template_text

    override = (kw or {}).get("chat_template")
    if isinstance(override, str) and override:
        return override
    return _model_template_text(processor)


def _renders_template_text(processor) -> bool:
    """Whether a render of ``processor`` runs the Jinja text ``template_for``
    returns: no Python renderer and no override that renders other text."""
    template = template_for(processor)
    if "{%" not in template and "{{" not in template:
        return False
    for obj in (processor, getattr(processor, "tokenizer", None)):
        if obj is not None and callable(getattr(obj, "_chat_template", None)):
            return False
    owner = next((c for c in type(processor).__mro__
                  if "apply_chat_template" in vars(c)), None)
    if owner is None:
        # A delegating wrapper: the tokenizer it forwards to was checked
        # above.
        return True
    fn = vars(owner)["apply_chat_template"]
    module = getattr(fn, "__module__", "") or ""
    qualname = getattr(fn, "__qualname__", "") or ""
    if module.startswith("transformers."):
        return True
    for m, cls, via_tokenizer in _JINJA_OVERRIDES:
        if module == m and qualname == f"{cls}.apply_chat_template":
            if not via_tokenizer:
                return True
            tokenizer = getattr(processor, "tokenizer", None)
            return (getattr(processor, "chat_template", None)
                    == getattr(tokenizer, "chat_template", None))
    return False


def _stash_continuation(args, request, processor):
    if not isinstance(request, _chat_request_type()):
        return args
    requested, agp, error = parse_request(request)
    Refusal = refusal_type()
    if error:
        raise Refusal(400, error)
    if requested is None:
        if agp is not None:
            args._kq_add_gen = agp
        return args
    template = template_for(processor)
    if not _renders_template_text(processor):
        raise Refusal(400, UNSUPPORTED)
    handler = handler_for(template)
    if not recipe_supported(handler, template, request.messages[:-1]):
        raise Refusal(400, UNSUPPORTED)
    args._kq_continue = resolve(requested, request.messages[-1], handler)
    args._kq_add_gen = False
    return args


def render_continued(fn, processor, config, prompt, *a, **kw):
    """Render through ``fn``, continuing the final assistant message when
    ``kw`` carries a continuation. Every caller that renders the output of
    ``to_template_kwargs`` for a chat request goes through here."""
    cont = kw.pop(CONTINUE_KEY, None)
    if cont is None:
        return fn(processor, config, prompt, *a, **kw)
    if (not isinstance(prompt, list) or not prompt
            or _get(prompt[-1], "role") != "assistant"):
        raise RuntimeError("continue_final_message reached the render without "
                           "a final assistant message")
    cfg = config if isinstance(config, dict) else getattr(config, "__dict__", {})
    if cfg.get("model_type") in _LAST_MESSAGE_ONLY:
        # The render returns the last message alone, not a prompt.
        raise refusal_type()(400, UNSUPPORTED)
    if cont.handler not in RECIPES:
        raise refusal_type()(400, UNSUPPORTED)
    head = list(prompt[:-1])

    def render(gen):
        out = fn(processor, config, head, *a,
                 **{**kw, "add_generation_prompt": gen})
        if not isinstance(out, str):
            raise _NotText
        return out

    try:
        return continued_prompt(cont, template_for(processor, kw), render)
    except _NotText:
        raise refusal_type()(400, UNSUPPORTED) from None


def install_assistant_prefill() -> None:
    """Install the route check, the gen-args transform, the
    ``to_template_kwargs`` wrap and the render wraps. Install right before
    the retire capture, so the render wrap lands outside faithful history
    and inside the capture. Idempotent."""
    from fastapi import Request
    from fastapi.responses import JSONResponse

    app = importlib.import_module("gmlx._vendor.mlx_vlm.server.app").app

    def make_endpoint(original):
        async def chat_endpoint(*args, **kwargs):
            values = list(args) + list(kwargs.values())
            http = next((v for v in values if isinstance(v, Request)), None)
            req = kwargs.get("request")
            if req is None:
                req = next((v for v in args if not isinstance(v, Request)), None)
            _requested, _agp, error = parse_request(req)
            if error:
                path = http.url.path if http is not None else _CHAT_PATHS[-1]
                return JSONResponse(status_code=400, content=_error_content(
                    path, 400, "invalid_request_error", error))
            return await original(*args, **kwargs)
        return chat_endpoint

    _wrap_post_routes(app, _CHAT_PATHS, _FLAG, make_endpoint)
    _install_gen_args_transform(_FLAG, _stash_continuation)

    gen = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")
    cls = gen.GenerationArguments
    original = cls.to_template_kwargs
    if not wrapped_with(original, _FLAG):
        def to_template_kwargs(self):
            kw = original(self)
            cont = getattr(self, "_kq_continue", None)
            if cont is not None:
                return {**kw, "add_generation_prompt": False,
                        CONTINUE_KEY: cont}
            if getattr(self, "_kq_add_gen", None) is False:
                return {**kw, "add_generation_prompt": False}
            return kw

        cls.to_template_kwargs = link(to_template_kwargs, original, _FLAG)

    def _wrap(fn):
        def apply_chat_template(processor, config, prompt, *a, **kw):
            return render_continued(fn, processor, config, prompt, *a, **kw)
        return link(apply_chat_template, fn, _FLAG)

    for target in _render_target_modules():
        fn = getattr(target, "apply_chat_template", None)
        if fn is not None and not wrapped_with(fn, _FLAG):
            setattr(target, "apply_chat_template", _wrap(fn))
