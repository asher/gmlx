"""Chat-completions behavior patches: chat_template_kwargs
passthrough, the thinking-budget fix, stream thinking seed, ignore-eos,
OpenAI stop sequences, and exclude_none streaming chunks."""

from __future__ import annotations

import contextvars
import importlib
import os
import sys


from .. import server_bridge_vlm as serving
from ._common import (
    _CHAT_PATHS,
    _PATCH_FLAG,
    _install_gen_args_transform,
    _wrap_post_routes,
)


# Generic chat_template_kwargs passthrough
# mlx-vlm's GenerationArguments.to_template_kwargs() only forwards its own thinking
# knobs (enable_thinking + thinking_budget/start/end) to apply_chat_template. A
# template may expose other variables (e.g. Qwen3.6 / Gemma-4 `preserve_thinking`,
# which keeps prior-turn <think> blocks for agent / tool-use turns). This seam
# merges arbitrary kwargs from the active profile (config `chat_template_kwargs`)
# and the request (an `extra="allow"` `chat_template_kwargs` field, the vLLM
# convention) into that dict - request keys win. Per-request only; nothing is
# auto-enabled, the user controls every template variable.
_CTKW_FLAG = "_kq_gguf_chat_template_kwargs_patch"


def _merged_template_kwargs(request, spec) -> dict:
    """Profile ``chat_template_kwargs`` as the base, request ``chat_template_kwargs``
    merged on top (request wins). Either side absent => the other; both absent => {}."""
    merged: dict = {}
    spec_kw = getattr(spec, "chat_template_kwargs", None) if spec is not None else None
    if isinstance(spec_kw, dict):
        merged.update(spec_kw)
    req_kw = getattr(request, "chat_template_kwargs", None)
    if isinstance(req_kw, dict):
        merged.update(req_kw)
    return merged


def _stash_template_kwargs(args, request, _processor):
    """Stash the merged template kwargs on the GenerationArguments instance so the
    patched ``to_template_kwargs`` can fold them in. Returns ``args`` (mutated)."""
    spec = serving.get_active_spec()
    merged = _merged_template_kwargs(request, spec)
    if merged:
        args._kq_template_kwargs = merged
    fields_set = getattr(request, "model_fields_set", None) or set()
    spec_sampling = getattr(spec, "sampling", None) or {} if spec else {}
    thinking_explicit = (
        "enable_thinking" in fields_set
        or "enable_thinking" in spec_sampling
        or os.environ.get("MLX_VLM_ENABLE_THINKING") is not None
    )
    args._kq_thinking_explicit = thinking_explicit
    if not thinking_explicit:
        args.enable_thinking = True
    return args


def install_chat_template_kwargs() -> None:
    """Forward arbitrary ``chat_template_kwargs`` (active profile + request, request
    wins) into ``apply_chat_template``. Two idempotent halves: a gen-args transform
    stashes the merged dict on the args object; a ``GenerationArguments.to_template_kwargs``
    wrapper folds that stash into the kwargs mlx-vlm passes to the template."""
    _install_gen_args_transform(_CTKW_FLAG, _stash_template_kwargs)

    gen = importlib.import_module("mlx_vlm.server.generation")
    cls = gen.GenerationArguments
    original = cls.to_template_kwargs
    if getattr(original, _CTKW_FLAG, False):
        return

    def to_template_kwargs(self):
        kw = original(self)
        if not getattr(self, "_kq_thinking_explicit", True):
            kw.pop("enable_thinking", None)
        extra = getattr(self, "_kq_template_kwargs", None)
        if extra:
            kw = {**kw, **extra}
        return kw

    to_template_kwargs.__dict__[_CTKW_FLAG] = True
    cls.to_template_kwargs = to_template_kwargs


# thinking_budget for models that generate <think> (not just pre-fill it)
# mlx-vlm's ThinkingBudgetCriteria gates both token counting and the forced close
# on ``enable_thinking``, which ResponseGenerator only sets True when the *prompt*
# already opened a <think> block (``_prompt_has_open_thinking``). GLM-5.2 pre-fills
# <think>, so it works; Qwen3 and most reasoning models *generate* <think> as the
# first token, so the criteria stays disarmed and the budget is a silent no-op.
# Fix (no fork): keep the gates armed whenever a budget is set (regardless of
# enable_thinking - a group/profile config may disable it, or a model may ignore
# the flag and emit <think> anyway), and instead seed the initial in-thinking
# state from whether the prompt pre-filled - so a generated <think> still flips
# counting on, while a non-thinking answer (no <think> ever emitted) is never
# force-closed. The batched decode loop that drives the criteria
# (mlx_vlm/generate/ar.py) is unchanged; only construction is.
_TBUDGET_FLAG = "_kq_gguf_thinking_budget_fix"


def _armed_thinking_budget_criteria_cls():
    """Subclass of mlx-vlm's criteria that decouples 'armed' (counting + forced
    close, gated on enable_thinking) from 'starts inside a block' (in_thinking).
    Built lazily so importing this module never imports mlx-vlm."""
    from mlx_vlm.utils import ThinkingBudgetCriteria

    class _ArmedThinkingBudgetCriteria(ThinkingBudgetCriteria):
        def __init__(self, *args, prompt_open_thinking=False, **kw):
            super().__init__(*args, **kw)  # sets in_thinking = enable_thinking
            self._prompt_open = bool(prompt_open_thinking)
            self.in_thinking = self._prompt_open
        def reset_thinking_state(self):
            super().reset_thinking_state()
            self.in_thinking = self._prompt_open

    return _ArmedThinkingBudgetCriteria


def _prompt_opens_thinking_tokens(rg, args, input_ids) -> bool:
    """Whether the prompt ``input_ids`` ends inside an open <think> block, read
    straight from the tokens. This is the enable_thinking-independent core of
    mlx-vlm's ``_prompt_has_open_thinking`` (which returns False whenever
    ``enable_thinking`` is off - and the server default is off - so it misses
    pre-fill models whose template opened <think> regardless of the flag, exactly
    the case an explicit budget must cap)."""
    if input_ids is None:
        return False
    start_id, end_id = rg._thinking_token_ids(args)
    toks = input_ids.flatten().tolist() if hasattr(input_ids, "flatten") \
        else list(input_ids)
    try:
        last_start = len(toks) - 1 - toks[::-1].index(start_id)
    except ValueError:
        return False
    try:
        last_end = len(toks) - 1 - toks[::-1].index(end_id)
    except ValueError:
        last_end = -1
    return last_start > last_end


def install_thinking_budget_fix() -> None:
    """Make ``thinking_budget`` cap generate-<think> models too, by constructing
    the criteria armed (enable_thinking=True) with the in-thinking seed taken from
    the prompt. Idempotent. No-op if mlx-vlm's shape changed."""
    gen = importlib.import_module("mlx_vlm.server.generation")
    cls = getattr(gen, "ResponseGenerator", None)
    if cls is None or not hasattr(cls, "_make_thinking_budget_criteria"):
        return
    if getattr(cls._make_thinking_budget_criteria, _TBUDGET_FLAG, False):
        return
    armed_cls = _armed_thinking_budget_criteria_cls()

    def _make_thinking_budget_criteria(self, args, input_ids):
        if getattr(args, "thinking_budget", None) is None:
            return None
        # Honor an explicit budget regardless of enable_thinking: a group/profile
        # config may disable it, or the model may ignore the flag and emit <think>
        # anyway. The gates arm but in_thinking is seeded from whether the prompt
        # actually opens <think> (computed independently of enable_thinking, so a
        # pre-fill model still caps), and a never-thinking answer is never closed.
        start = args.thinking_start_token or gen.DEFAULT_THINKING_START_TOKEN
        end = args.thinking_end_token or gen.DEFAULT_THINKING_END_TOKEN
        prompt_open = _prompt_opens_thinking_tokens(self, args, input_ids)
        return armed_cls(
            tokenizer=self.tokenizer,
            thinking_budget=args.thinking_budget,
            thinking_end_token=end,
            thinking_start_token=start,
            enable_thinking=True,             # arm the gates regardless of origin
            prompt_open_thinking=prompt_open,  # but only start inside if pre-filled
        )

    _make_thinking_budget_criteria.__dict__[_TBUDGET_FLAG] = True
    cls._make_thinking_budget_criteria = _make_thinking_budget_criteria


# Streaming think-splitter seeded from the rendered prompt, not the flag
# mlx-vlm's ThinkingStreamState starts in reasoning mode when
# gen_args.enable_thinking is True, and only exits on a close marker in the
# output. Our 7b transform forces enable_thinking=True whenever thinking isn't
# explicit (so the template's own default drives it) - correct for templates
# that then PRE-FILL the open marker (Qwen3.6 appends '<think>\n' to the
# generation prompt, so the model never emits it), but wrong for templates
# whose own default is thinking-OFF (gemma-4 renders no thinking scaffold):
# the output never contains a close marker, the splitter never leaves
# reasoning mode, and the whole answer streams as `reasoning` deltas with
# content empty - while the non-streamed path (marker-driven _split_thinking)
# returns the same answer as content. Ground truth is the rendered prompt:
# start inside a think block iff the prompt actually ends inside one. The
# rendered prompt is stashed per request (contextvar - the render and the
# splitter construction run on the same asyncio task chain) by wrapping the
# apply_chat_template binding each protocol module uses; the splitter's
# __init__ then reseeds in_thinking from the prompt tail using its own marker
# pairs (so custom start/end tokens are honored). No stashed prompt (raw
# /v1/completions) keeps the stock seed.
_STREAM_SEED_FLAG = "_kq_gguf_stream_thinking_seed"

_LAST_RENDERED_PROMPT: contextvars.ContextVar = contextvars.ContextVar(
    "kq_last_rendered_prompt", default=None)


def _prompt_tail_opens_thinking(text, open_close_markers) -> bool:
    """Whether ``text`` ends inside an open think block for any marker pair
    (last open marker after the pair's last close marker)."""
    if not isinstance(text, str) or not text:
        return False
    for open_m, close_m in open_close_markers:
        o = text.rfind(open_m)
        if o >= 0 and o > text.rfind(close_m):
            return True
    return False


def install_stream_thinking_seed() -> None:
    """Reseed ThinkingStreamState.in_thinking from the rendered prompt (see
    the 7e comment). Idempotent; no-op if mlx-vlm's shape changed."""
    rs = importlib.import_module("mlx_vlm.server.responses_state")
    cls = getattr(rs, "ThinkingStreamState", None)
    if cls is None:
        return
    if not getattr(cls.__init__, _STREAM_SEED_FLAG, False):
        original_init = cls.__init__

        def __init__(self, *args, **kw):
            original_init(self, *args, **kw)
            prompt = _LAST_RENDERED_PROMPT.get()
            if prompt is not None:
                self.in_thinking = _prompt_tail_opens_thinking(
                    prompt, self.open_close_markers)

        __init__.__dict__[_STREAM_SEED_FLAG] = True
        cls.__init__ = __init__

    def _wrap(fn):
        def apply_chat_template(*a, **kw):
            out = fn(*a, **kw)
            _LAST_RENDERED_PROMPT.set(out if isinstance(out, str) else None)
            return out
        apply_chat_template.__dict__[_STREAM_SEED_FLAG] = True
        return apply_chat_template

    targets = [sys.modules.get(m) or importlib.import_module(m)
               for m in ("mlx_vlm.server.openai", "mlx_vlm.server.anthropic")]
    app = importlib.import_module("mlx_vlm.server.app")
    deps = getattr(app, "_protocol_deps", None)
    if deps is not None:
        targets.append(deps)
    for target in targets:
        fn = getattr(target, "apply_chat_template", None)
        if fn is not None and not getattr(fn, _STREAM_SEED_FLAG, False):
            target.apply_chat_template = _wrap(fn)


# ignore-eos: forced-length decode (server-level)
# A server switch (serve --ignore-eos / GMLX_IGNORE_EOS) that makes decode run
# to max_tokens instead of stopping on an EOS / stop token, mirroring llama-server's
# --ignore-eos launch flag. StoppingCriteria.__call__ is the single predicate both
# the normal and speculative decode loops consult (mlx_vlm.generate.ar's
# GenerationBatch and SpeculativeGenerationBatch), so neutralizing it there covers
# every batch in one place. Request-level OpenAI stop sequences are unaffected:
# those are filtered at the SSE layer (install_openai_stop_sequences), not here.
_IGNORE_EOS_FLAG = "_kq_gguf_ignore_eos_patch"


def install_ignore_eos() -> None:
    """Suppress EOS / stop-token termination so decode runs to max_tokens. Patches
    mlx_vlm.utils.StoppingCriteria.__call__ to always return False. Server-global
    and idempotent; for benchmarking forced-length throughput. No-op if mlx-vlm's
    shape changed."""
    utils = importlib.import_module("mlx_vlm.utils")
    cls = getattr(utils, "StoppingCriteria", None)
    if cls is None:
        return
    original = cls.__call__
    if getattr(original, _IGNORE_EOS_FLAG, False):
        return

    def __call__(self, input_ids):
        return False

    __call__.__dict__[_IGNORE_EOS_FLAG] = True
    cls.__call__ = __call__


# OpenAI-API stop sequences (chat completions)
# mlx-vlm's Anthropic endpoint honours ``stop_sequences`` but its OpenAI chat
# schema has no ``stop`` field - an OpenAI-API compliance hole that coding
# harnesses hit (they send stop sequences on /v1/chat/completions). The request
# schemas are ``extra="allow"`` so a client-sent ``stop`` already arrives on the
# request object; this patch wraps the chat-completions routes to honour it:
# non-stream responses are trimmed at the earliest stop, streams are filtered
# through a StopScanner (hold-back so a stop split across chunks still matches)
# and closed early - ending the upstream generator cancels the generation.
_STOP_FLAG = "_kq_gguf_openai_stop_patch"


_MAX_REQUEST_STOPS = 16   # OpenAI caps at 4; the scanner is O(stops) per token


def _request_stop_sequences(request) -> list:
    """The effective stop sequences: the request ``stop`` (str or list) wins;
    else the resolved profile's ``sampling.stop``. Empty list = none."""
    raw = getattr(request, "stop", None)
    from_request = raw is not None
    if raw is None and getattr(serving, "server_config", lambda: None)() is not None:
        try:
            _path, spec = serving.resolve_request_model(
                getattr(request, "model", None),
                profile_field=getattr(request, "profile", None))
        except Exception:
            spec = None
        if spec is not None:
            raw = (spec.sampling or {}).get("stop")
    if raw is None:
        return []
    if isinstance(raw, str):
        seqs = [raw]
    elif isinstance(raw, (list, tuple)) and all(isinstance(s, str) for s in raw):
        seqs = list(raw)
    else:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=400,
            detail="'stop' must be a string or an array of strings")
    if from_request and len(seqs) > _MAX_REQUEST_STOPS:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=400,
            detail=f"'stop' allows at most {_MAX_REQUEST_STOPS} sequences "
                   f"(got {len(seqs)})")
    return [s for s in seqs if s]


def _trim_chat_response(resp, stops: list):
    """Trim each choice's message content at the earliest stop sequence and mark
    ``finish_reason="stop"``. Non-chat shapes (error JSONResponse) pass through."""
    for choice in getattr(resp, "choices", None) or []:
        msg = getattr(choice, "message", None)
        text = getattr(msg, "content", None)
        if isinstance(text, str):
            cuts = [i for i in (text.find(s) for s in stops) if i != -1]
            if cuts:
                msg.content = text[:min(cuts)]
                choice.finish_reason = "stop"
    return resp


def _filter_sse_event(event: str, st: dict) -> tuple:
    """Filter one SSE event through the scanner. Returns ``(events, done)`` -
    the (possibly rewritten / synthesized) events to emit, and whether the
    stream is finished (stop hit or upstream [DONE])."""
    import json

    scanner = st["scanner"]
    if not event.startswith("data: "):
        return [event + "\n\n"], False
    payload = event[len("data: "):]
    if payload.strip() == "[DONE]":
        return [event + "\n\n"], True
    try:
        obj = json.loads(payload)
    except ValueError:
        return [event + "\n\n"], False
    choices = obj.get("choices") or []
    if not choices:
        return [event + "\n\n"], False          # usage-only chunk
    st["meta"] = {k: obj[k] for k in ("id", "created", "model", "object")
                  if k in obj}
    choice = choices[0]
    delta = choice.get("delta") or {}
    content = delta.get("content")

    if choice.get("finish_reason"):
        # Natural end: release the held-back tail into the final chunk (or trim
        # it if the tail itself completes a stop sequence).
        text, hit = scanner.feed(content) if isinstance(content, str) and content \
            else ("", False)
        if hit:
            choice["finish_reason"] = "stop"
        else:
            text += scanner.flush()
        if content is not None or text:
            delta["content"] = text
            choice["delta"] = delta
        return [f"data: {json.dumps(obj)}\n\n"], False

    if not isinstance(content, str) or not content:
        return [event + "\n\n"], False          # role / empty delta
    text, hit = scanner.feed(content)
    events = []
    if text:
        delta["content"] = text
        choice["delta"] = delta
        events.append(f"data: {json.dumps(obj)}\n\n")
    if hit:
        fin = {**st["meta"],
               "choices": [{"index": choice.get("index", 0),
                            "delta": {}, "finish_reason": "stop"}]}
        events.append(f"data: {json.dumps(fin)}\n\n")
        events.append("data: [DONE]\n\n")
        return events, True
    return events, False


async def _stop_filter_sse(body, stops: list):
    """Wrap a chat-completions SSE body iterator with stop-sequence filtering.
    On a hit: emit the pre-stop text, a ``finish_reason="stop"`` chunk and
    ``[DONE]``, then close the upstream generator (cancelling generation)."""
    from ..generation import StopScanner

    st = {"scanner": StopScanner(stops), "meta": {}}
    pending = ""    # partial SSE event split across upstream yields
    try:
        async for raw in body:
            pending += raw.decode() if isinstance(raw, (bytes, bytearray)) else raw
            while "\n\n" in pending:
                event, pending = pending.split("\n\n", 1)
                events, done = _filter_sse_event(event, st)
                for out in events:
                    yield out
                if done:
                    return
        if pending:
            yield pending
        # Upstream ended without a finish_reason chunk (engine error/cancel):
        # release the scanner's held-back tail so those characters aren't lost.
        # No [DONE] is synthesized - an abnormal end should look abnormal.
        tail = st["scanner"].flush()
        if tail and st["meta"]:
            import json
            obj = {**st["meta"],
                   "choices": [{"index": 0, "delta": {"content": tail},
                                "finish_reason": None}]}
            yield f"data: {json.dumps(obj)}\n\n"
    finally:
        aclose = getattr(body, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:
                pass


def install_openai_stop_sequences() -> None:
    """Wrap the chat-completions routes so an OpenAI ``stop`` parameter (or a
    profile ``sampling.stop``) is honoured for both stream and non-stream
    responses. ``__signature__`` is copied from the original endpoint so FastAPI
    still parses the body into ChatRequest. Idempotent per route."""
    from starlette.responses import StreamingResponse

    app = importlib.import_module("mlx_vlm.server.app").app

    def _make(original):
        async def endpoint(request, http_request):
            stops = _request_stop_sequences(request)
            result = await original(request, http_request)
            if not stops:
                return result
            if isinstance(result, StreamingResponse):
                result.body_iterator = _stop_filter_sse(
                    result.body_iterator, stops)
                return result
            return _trim_chat_response(result, stops)
        return endpoint

    _wrap_post_routes(app, _CHAT_PATHS, _STOP_FLAG, _make)


def install_vanilla_stream_chunks() -> None:
    """Serialize streaming chat-completion chunks as byte-vanilla OpenAI.

    mlx-vlm emits each ``ChatStreamChunk`` with a plain ``model_dump_json()``, so
    every chunk carries non-standard *null* fields a real OpenAI server omits:
    ``timings: null`` at the top level (plus ``usage``/``logprobs``/``finish_reason``
    and, in the delta, ``reasoning``/``tool_calls``/``tool_call_id``/``name``). The
    ``timings: null`` is actively breaking: Open WebUI's stream relay does
    ``raw_usage.update(chunk.get("timings", {}))`` - a present-but-null key returns
    ``None`` (not the ``{}`` default), so ``{}.update(None)`` raises ``TypeError``,
    the relay's per-line handler swallows it with ``continue``, and *every* content
    chunk is dropped: the assistant message renders blank while tokens are served.

    Fix: make ``ChatStreamChunk.model_dump_json`` default ``exclude_none=True`` so
    streaming chunks drop their null fields. Real values (the final chunk's
    ``usage``/``timings`` dicts, a set ``finish_reason``) are kept; ``choices: []``
    and ``content: ""`` are not ``None`` so they survive. The non-streaming
    ``ChatResponse`` is a different class and is untouched. Idempotent."""
    schemas = importlib.import_module("mlx_vlm.server.schemas")
    chunk_cls = schemas.ChatStreamChunk
    original = chunk_cls.model_dump_json
    if getattr(original, _PATCH_FLAG, False):
        return

    def model_dump_json(self, **kwargs):
        kwargs.setdefault("exclude_none", True)
        return original(self, **kwargs)

    model_dump_json.__dict__[_PATCH_FLAG] = True
    chunk_cls.model_dump_json = model_dump_json
