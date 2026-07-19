"""API-contract patches on the generation routes: a per-request warning
for parameters the server accepts but never reads (the request schemas are
``extra="allow"``, so unknown fields ride along silently), and server-side
enforcement of ``tool_choice: "none"`` (drop the tools before the chat
template ever sees them). ``required``/named tool_choice can't be enforced
without per-template grammars, so a request that asked for a forced call and
got none back logs one warning instead."""

from __future__ import annotations

import importlib
import logging
import os

from ._common import _CHAT_PATHS, _wrap_post_routes

_log = logging.getLogger(__name__)

_API_CONTRACT_FLAG = "_kq_gguf_api_contract"

_RESPONSES_PATHS = ("/responses", "/v1/responses")
_MESSAGES_PATHS = ("/messages", "/v1/messages")


# Consumed-parameter sets, one per dialect. "Consumed" means some code path
# actually reads the field off the request: the stock handler, the
# shape-agnostic ``_build_gen_args`` (getattr-based, so it reads these off any
# request object), or a gmlx server patch. A set-but-unlisted field draws the
# one-line warning below. The docs/server-config.md "Parameter support" table
# is derived from these sets, and tests/test_api_contract.py cross-checks
# both (and fails when an upstream schema grows a field not classified here),
# so update table, test, and set together.

# mlx_vlm.server.app._build_gen_args + _extract_response_format_schema:
# every field read there via getattr/_request_field_or_default.
_GEN_ARGS_CONSUMED = frozenset({
    "max_tokens", "max_output_tokens", "temperature", "top_p", "top_k",
    "min_p", "seed", "logprobs", "repetition_penalty",
    "repetition_context_size", "presence_penalty", "presence_context_size",
    "frequency_penalty", "frequency_context_size", "logit_bias",
    "enable_thinking", "thinking_budget", "thinking_start_token",
    "thinking_end_token", "response_format", "text",
    # masked-diffusion knobs (read for every request shape)
    "max_denoising_steps", "block_length", "num_to_transfer",
    "max_transfer_per_step", "editing_threshold", "max_post_steps",
    "stability_steps", "diffusion_full_canvas",
    "diffusion_min_canvas_length", "diffusion_max_canvas_length",
    "diffusion_sampler", "threshold", "min_threshold",
})

# gmlx server patches (stop resolver, profile capture, template kwargs,
# XTC) read these off every generation request.
_GMLX_CONSUMED = frozenset({
    "profile", "chat_template_kwargs", "xtc_probability", "xtc_threshold",
})

# /v1/chat/completions (openai.py chat_completions_endpoint + the gmlx stop
# filter). ``tool_choice`` is consumed here: "none" is enforced below, other
# values are documented as template-dependent.
CHAT_CONSUMED = _GEN_ARGS_CONSUMED | _GMLX_CONSUMED | frozenset({
    "model", "messages", "stream", "stream_options", "adapter_path",
    "resize_shape", "tools", "tool_choice", "top_logprobs", "stop",
})

# /v1/responses (openai.py responses_endpoint). No ``stop`` here: the gmlx
# stop filter wraps only the chat routes, so a Responses ``stop`` is unread.
# ``logprobs`` is excluded: _build_gen_args reads it, but the Responses shape
# never returns logprobs, so setting it changes nothing observable.
RESPONSES_CONSUMED = (_GEN_ARGS_CONSUMED - {"logprobs"}) | _GMLX_CONSUMED \
    | frozenset({
        "model", "input", "instructions", "previous_response_id", "tools",
        "tool_choice", "store", "stream", "adapter_path",
    })

# /v1/messages (anthropic.py). ``output_config`` is consumed (its
# ``json_schema`` format maps onto ``response_format``); ``metadata`` is
# declared on the schema but never read, so it warns. ``logprobs`` excluded
# for the same reason as on /v1/responses.
ANTHROPIC_CONSUMED = (_GEN_ARGS_CONSUMED - {"logprobs"}) | _GMLX_CONSUMED \
    | frozenset({
        "model", "messages", "system", "stream", "stop_sequences", "tools",
        "tool_choice", "thinking", "output_config", "adapter_path",
    })

# /v1/completions (gmlx's own route, server_patches/completions.py).
# ``logprobs`` is excluded: the field is neutralized there (token logprobs
# are never returned on the text-completions shape), so it warns.
COMPLETIONS_CONSUMED = (
    (_GEN_ARGS_CONSUMED - {"logprobs"}) | _GMLX_CONSUMED | frozenset({
        "model", "prompt", "stream", "stream_options", "stop",
        "echo", "suffix", "n", "best_of",
    })
)


def warn_ignored_fields(path: str, ignored) -> None:
    """One warning line naming the set-but-unread request fields."""
    ignored = sorted(str(f) for f in ignored)
    if not ignored:
        return
    _log.warning(
        "%s: ignoring unsupported parameter(s): %s "
        "(see the parameter support table in docs/server-config.md)",
        path, ", ".join(ignored))


def _is_tool_choice_none(tc) -> bool:
    """True for OpenAI ``"none"`` and Anthropic ``{"type": "none"}`` (dict or
    pydantic-shaped)."""
    if tc == "none":
        return True
    if isinstance(tc, dict):
        return tc.get("type") == "none"
    return getattr(tc, "type", None) == "none"


def _tool_choice_wants_calls(tc) -> bool:
    """True when the request demanded a tool call: OpenAI ``"required"`` /
    ``{"type": "function", ...}``, Anthropic ``{"type": "any"}`` /
    ``{"type": "tool", ...}``."""
    if tc == "required":
        return True
    t = tc.get("type") if isinstance(tc, dict) else getattr(tc, "type", None)
    return t in ("function", "tool", "any", "required")


def _result_tool_calls(result):
    """Whether a non-streaming response carries tool calls: True / False, or
    None when undeterminable (streams get their own watcher below; error
    bodies and unknown shapes stay None)."""
    choices = getattr(result, "choices", None)
    if choices:                                       # ChatResponse
        for choice in choices:
            if getattr(choice, "finish_reason", None) == "tool_calls":
                return True
            if getattr(getattr(choice, "message", None), "tool_calls", None):
                return True
        return False
    if getattr(result, "stop_reason", None) is not None:   # Anthropic message
        if result.stop_reason == "tool_use":
            return True
        blocks = getattr(result, "content", None) or []
        return any(isinstance(b, dict) and b.get("type") == "tool_use"
                   for b in blocks)
    output = getattr(result, "output", None)          # Responses API
    if isinstance(output, list):
        return any(getattr(item, "type", None) == "function_call"
                   or (isinstance(item, dict)
                       and item.get("type") == "function_call")
                   for item in output)
    return None


def _warn_unfulfilled(path: str) -> None:
    _log.warning(
        "%s: tool_choice requested a forced tool call but none was "
        "parsed from the output; required/named tool_choice is only "
        "honored when the model's chat template implements it "
        "(see docs/server-config.md)", path)


# Tool-call presence markers in the three dialects' stream chunks: chat SSE
# deltas carry "tool_calls", Anthropic content_block_start events "tool_use",
# Responses API events "function_call". A substring scan (not JSON parsing)
# keeps the watcher off the latency path; the failure mode is fail-quiet - a
# marker appearing in generated text suppresses a warning, never adds one.
_STREAM_TOOL_MARKERS = ('"tool_calls"', '"tool_use"', '"function_call"')


def _watch_stream_for_tool_calls(path: str, result) -> None:
    """Forced tool_choice on a streaming response: pass the SSE chunks
    through unchanged and warn at clean end-of-stream when no tool-call
    marker ever appeared. An early close (client disconnect) skips the
    warning - the stream never finished, so absence proves nothing."""
    body = result.body_iterator

    async def watched():
        seen = False
        async for chunk in body:
            if not seen:
                text = (chunk.decode("utf-8", "ignore")
                        if isinstance(chunk, (bytes, bytearray)) else
                        str(chunk))
                seen = any(m in text for m in _STREAM_TOOL_MARKERS)
            yield chunk
        if not seen:
            _warn_unfulfilled(path)

    result.body_iterator = watched()


def _maybe_warn_unfulfilled_tool_choice(path: str, tc, result) -> None:
    if tc is None or not _tool_choice_wants_calls(tc):
        return
    got = _result_tool_calls(result)
    if got is False:
        _warn_unfulfilled(path)
    elif got is None and hasattr(result, "body_iterator"):
        _watch_stream_for_tool_calls(path, result)


def _maybe_warn_top_logprobs(path: str, request) -> None:
    """Warn when a request asks for more logprob alternatives than the
    server-side ``TOP_LOGPROBS_K`` cap allows (default 0): the response
    silently carries empty/truncated alternatives otherwise."""
    want = getattr(request, "top_logprobs", None)
    if not want:
        return
    try:
        cap = int(os.environ.get("TOP_LOGPROBS_K", 0) or 0)
    except ValueError:
        cap = 0
    # Mirror the engine's effective cap (get_top_logprobs_k clamps to 20):
    # comparing against the raw env value would stay silent for a request in
    # (20, TOP_LOGPROBS_K] that the server still truncates.
    cap = max(0, min(cap, 20))
    if want > cap:
        _log.warning(
            "%s: top_logprobs=%d exceeds the server cap %d (TOP_LOGPROBS_K, "
            "hard-capped at 20 by the engine), so alternatives beyond the "
            "cap are omitted (see docs/server-config.md)",
            path, want, cap)


def _http_request_of(values):
    return next((o for o in values
                 if hasattr(o, "receive") and hasattr(o, "json")), None)


def _make_chat_endpoint(original):
    """Wrapper for the parsed-body chat routes: warn on unread fields off
    ``model_fields_set`` (extras included) and drop ``tools`` when
    ``tool_choice`` is "none" - the handler then never flips
    ``skip_special_tokens`` and the template renders without a tool block."""
    async def endpoint(*args, **kwargs):
        values = list(args) + list(kwargs.values())
        request = next(
            (o for o in values if hasattr(o, "model_fields_set")), None)
        http = _http_request_of(values)
        path = str(http.url.path) if http is not None else _CHAT_PATHS[-1]
        tc = getattr(request, "tool_choice", None) if request is not None \
            else None
        if request is not None:
            warn_ignored_fields(
                path, set(request.model_fields_set) - CHAT_CONSUMED)
            _maybe_warn_top_logprobs(path, request)
            if _is_tool_choice_none(tc) and getattr(request, "tools", None):
                request.tools = None
        result = await original(*args, **kwargs)
        _maybe_warn_unfulfilled_tool_choice(path, tc, result)
        return result
    return endpoint


def _make_raw_endpoint(consumed, fallback_path):
    """Wrapper factory for the raw-``Request`` routes (responses/anthropic):
    the handler parses the JSON body itself, so warn/enforce on the cached
    body dict (starlette caches ``_json``; the handler's own ``json()`` call
    returns the same mutated object)."""
    def make(original):
        async def endpoint(*args, **kwargs):
            values = list(args) + list(kwargs.values())
            http = _http_request_of(values)
            path = str(http.url.path) if http is not None else fallback_path
            tc = None
            if http is not None:
                try:
                    body = await http.json()
                except Exception:
                    body = None
                if isinstance(body, dict):
                    tc = body.get("tool_choice")
                    warn_ignored_fields(path, set(body) - consumed)
                    if _is_tool_choice_none(tc):
                        body.pop("tools", None)
                        body.pop("tool_choice", None)
            result = await original(*args, **kwargs)
            _maybe_warn_unfulfilled_tool_choice(path, tc, result)
            return result
        return endpoint
    return make


def install_api_contract() -> None:
    """Wrap the three generation dialects with the unread-parameter warning
    and ``tool_choice: "none"`` enforcement. Idempotent per route."""
    app = importlib.import_module("mlx_vlm.server.app").app
    _wrap_post_routes(app, _CHAT_PATHS, _API_CONTRACT_FLAG,
                      _make_chat_endpoint)
    _wrap_post_routes(app, _RESPONSES_PATHS, _API_CONTRACT_FLAG,
                      _make_raw_endpoint(RESPONSES_CONSUMED,
                                         _RESPONSES_PATHS[-1]))
    _wrap_post_routes(app, _MESSAGES_PATHS, _API_CONTRACT_FLAG,
                      _make_raw_endpoint(ANTHROPIC_CONSUMED,
                                         _MESSAGES_PATHS[-1]))
