"""Sampling-surface patches: profile injection into unset request
fields, the chat ``max_completion_tokens`` cap, the thinking-budget fields
and their llama-server names, the until-EOS default cap, XTC as a
per-request logits processor, and the fast positioned sampler."""

from __future__ import annotations

import contextvars
import importlib
import logging

import gmlx.serve.bridge_vlm as serving
from gmlx.spec.topk import HIER_MIN_V, topk_ids
from gmlx.upstream.patching import link, wrapped_with

from ._common import (
    _CHAT_PATHS,
    _PATCH_FLAG,
    _install_gen_args_transform,
    _wrap_post_routes,
    refuse_both_spellings,
)

_log = logging.getLogger(__name__)


# Sampling-profile injection
# GenerationArguments attr <- profile sampling key (1:1 names). Each maps to the
# request field(s) whose presence in model_fields_set means "client set it".
_GEN_ARG_REQUEST_FIELDS = {
    "max_tokens": ("max_tokens", "max_output_tokens"),
    "temperature": ("temperature",),
    "top_p": ("top_p",),
    "top_k": ("top_k",),
    "min_p": ("min_p",),
    "seed": ("seed",),
    "repetition_penalty": ("repetition_penalty",),
    "repetition_context_size": ("repetition_context_size",),
    "presence_penalty": ("presence_penalty",),
    "frequency_penalty": ("frequency_penalty",),
    "enable_thinking": ("enable_thinking",),
    "thinking_budget": ("thinking_budget", "reasoning_budget_tokens"),
    "thinking_budget_message": ("thinking_budget_message",
                                "reasoning_budget_message"),
}


def _inject_profile_sampling(args, request, spec):
    """Override an unset sampling field on ``args`` (a GenerationArguments) with the
    resolved profile's value. A field the client explicitly set (in
    ``request.model_fields_set``) always wins. Returns ``args`` (mutated)."""
    if spec is None or not getattr(spec, "sampling", None):
        return args
    fields_set = getattr(request, "model_fields_set", None) or set()
    for key, value in spec.sampling.items():
        if not hasattr(args, key):
            continue
        request_fields = _GEN_ARG_REQUEST_FIELDS.get(key, (key,))
        if any(f in fields_set for f in request_fields):
            continue                     # client set it -> request wins
        setattr(args, key, value)
    return args


def install_gen_args_profile_injection() -> None:
    """Wrap ``_build_gen_args`` so the active profile seeds unset sampling fields.
    Idempotent (see :func:`_install_gen_args_transform`)."""
    _install_gen_args_transform(
        _PATCH_FLAG,
        lambda args, request, _processor:
            _inject_profile_sampling(args, request, serving.get_active_spec()))


# max_completion_tokens on /v1/chat/completions
# OpenAI's current chat spelling of the output cap. mlx-vlm's ChatRequest
# declares only max_tokens, so this field arrives as an extra that nothing read.
_MAX_COMPLETION_FLAG = "_kq_gguf_max_completion_tokens"


def client_max_tokens(value):
    """``value`` as a client output cap: a positive int (not a bool), else None."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _apply_max_completion_tokens(args, request, _processor):
    """Set ``args.max_tokens`` from a chat request's ``max_completion_tokens``.
    It wins over ``max_tokens`` and over a profile value, as in vLLM. Other
    request shapes are untouched. Returns ``args`` (mutated)."""
    fields = getattr(request, "model_fields_set", None) or set()
    if "max_completion_tokens" not in fields:
        return args
    schemas = importlib.import_module("gmlx._vendor.mlx_vlm.server.schemas")
    if not isinstance(request, schemas.ChatRequest):
        return args
    value = getattr(request, "max_completion_tokens", None)
    cap = client_max_tokens(value)
    if cap is None:
        _log.warning("ignoring max_completion_tokens=%r: not a positive "
                     "integer", value)
        return args
    args.max_tokens = cap
    return args


def install_max_completion_tokens() -> None:
    """Honor ``max_completion_tokens`` on chat requests at the gen-args seam.
    Install after :func:`install_gen_args_profile_injection`: each install
    wraps the previous one, so this transform runs later and a client value
    beats a profile ``max_tokens``."""
    _install_gen_args_transform(_MAX_COMPLETION_FLAG,
                                _apply_max_completion_tokens)


# Thinking-budget fields
# thinking_budget_message is forced ahead of the end marker when the budget
# runs out. reasoning_budget_tokens and reasoning_budget_message are
# llama-server's names for the two fields. A -1 budget asks for the server
# default, as on llama-server, and any negative budget left after that is
# no budget.
_BUDGET_FIELDS_FLAG = "_kq_gguf_thinking_budget_fields"
_BUDGET_ROUTE_FLAG = "_kq_gguf_thinking_budget_route"
# The routes that generate. The count, estimate and upsample callers of
# _build_gen_args are not on them, so they never refuse a budget.
_GENERATION_PATHS = _CHAT_PATHS + (
    "/responses", "/v1/responses", "/messages", "/v1/messages",
    "/completions", "/v1/completions")
_generation_route: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "gmlx_generation_route", default=False)


def _bad_field(detail: str):
    from fastapi import HTTPException

    return HTTPException(status_code=400, detail=detail)


def _budget_field(name: str, value):
    """A client budget: an int of -1 or more (not a bool), or None."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < -1:
        raise _bad_field(f"{name} must be an integer of -1 or more.")
    return value


def _message_field(name: str, value):
    if value is not None and not isinstance(value, str):
        raise _bad_field(f"{name} must be a string.")
    return value


def _server_thinking_budget():
    """The budget a client -1 asks for: the active spec's (profile, model
    override, --thinking-budget, family default), else the environment's."""
    spec = serving.get_active_spec()
    sampling = (getattr(spec, "sampling", None) or {}) if spec is not None else {}
    value = sampling.get("thinking_budget")
    if value is None:
        gen = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")
        value = gen.get_server_thinking_budget()
    return value


def _budget_display_name(request) -> str:
    """``thinking.budget_tokens`` when a Messages request's budget came from
    there, else ``thinking_budget``. The Messages route copies
    ``budget_tokens`` only into a null ``thinking_budget``, as an int."""
    schemas = importlib.import_module("gmlx._vendor.mlx_vlm.server.schemas")
    thinking = getattr(request, "thinking", None)
    if not (isinstance(request, schemas.AnthropicRequest)
            and isinstance(thinking, dict)
            and thinking.get("budget_tokens") is not None):
        return "thinking_budget"
    budget = getattr(request, "thinking_budget", None)
    try:
        derived = type(budget) is int and budget == int(thinking["budget_tokens"])
    except (TypeError, ValueError):
        derived = False
    return "thinking.budget_tokens" if derived else "thinking_budget"


def separate_drafter_kind(rg) -> str | None:
    """The draft kind of a loaded drafter that cannot run a thinking budget,
    else None."""
    from gmlx.spec.engine import spec_kind_unsupported

    if rg is None or getattr(rg, "draft_model", None) is None:
        return None
    return spec_kind_unsupported(rg)


def _apply_thinking_budget_fields(args, request, _processor):
    """Resolve the budget and the message from either spelling. Runs after
    profile injection, which skips a field the client set under either
    name. Returns ``args`` (mutated)."""
    fields = getattr(request, "model_fields_set", None) or set()
    present = {f for f in fields if getattr(request, f, None) is not None}
    name = _budget_display_name(request)
    fs = present | ({name} if "thinking_budget" in present else set())
    refuse_both_spellings(fs, name, "reasoning_budget_tokens")
    refuse_both_spellings(fs, "thinking_budget_message",
                          "reasoning_budget_message")

    args._kq_budget_field = None
    for field, shown in (("thinking_budget", name),
                         ("reasoning_budget_tokens", "reasoning_budget_tokens")):
        if field not in present:
            continue
        value = _budget_field(shown, getattr(request, field))
        if value == -1:
            args.thinking_budget = _server_thinking_budget()
        else:
            args.thinking_budget = value
            args._kq_budget_field = shown
    budget = getattr(args, "thinking_budget", None)
    if isinstance(budget, int) and budget < 0:
        args.thinking_budget = None

    messages = ("thinking_budget_message", "reasoning_budget_message")
    for field in messages:
        if field in present:
            args.thinking_budget_message = _message_field(
                field, getattr(request, field))
            break
    else:
        if any(f in fields for f in messages):
            args.thinking_budget_message = None  # an explicit null

    kind = (separate_drafter_kind(_runtime().response_generator)
            if _generation_route.get() and args._kq_budget_field
            and args.thinking_budget is not None else None)
    if kind:
        raise _bad_field(
            f"This model cannot run a thinking budget while it speculates "
            f"with draft kind '{kind}'. Remove {args._kq_budget_field}, or "
            f"serve the model without the drafter.")
    return args


def _runtime():
    return importlib.import_module("gmlx._vendor.mlx_vlm.server.runtime").runtime


def install_thinking_budget_fields() -> None:
    """Honor ``thinking_budget_message`` and llama-server's
    ``reasoning_budget_tokens`` and ``reasoning_budget_message``. Install
    after :func:`install_max_completion_tokens` and before the template
    kwargs transform, so profile injection runs first and the off-by-budget
    rule sees the result."""
    _install_gen_args_transform(_BUDGET_FIELDS_FLAG,
                                _apply_thinking_budget_fields)


def install_thinking_budget_route() -> None:
    """Mark the generation routes, so a client budget that the drafter
    cannot run is a 400 before the response starts. Install after the
    completions route is registered. Idempotent per route."""
    app = importlib.import_module("gmlx._vendor.mlx_vlm.server.app").app

    def _make(original):
        async def endpoint(*args, **kwargs):
            token = _generation_route.set(True)
            try:
                return await original(*args, **kwargs)
            finally:
                _generation_route.reset(token)
        return endpoint

    _wrap_post_routes(app, _GENERATION_PATHS, _BUDGET_ROUTE_FLAG, _make)


# Until-EOS default output cap
# A request with no output cap from the client, a profile or --max-tokens
# runs until the model stops or its context fills, as `gmlx run` does.
# mlx-vlm reads its default cap from MLX_VLM_MAX_TOKENS everywhere, so the
# server sets it to UNTIL_EOS and each request swaps that value for the room
# its context has left, once the prompt length is known.
UNTIL_EOS = 1 << 30
_UNTIL_EOS_FLAG = "_kq_gguf_until_eos"


def native_context(model) -> int | None:
    """The context length the model was trained for, or None."""
    for obj in (model, getattr(model, "language_model", None)):
        for cfg in (getattr(obj, "config", None), getattr(obj, "args", None)):
            for c in (cfg, getattr(cfg, "text_config", None)):
                if isinstance(c, dict):
                    v = c.get("max_position_embeddings")
                else:
                    v = getattr(c, "max_position_embeddings", None)
                if isinstance(v, int) and v > 0:
                    return v
    return None


def until_eos_cap(prompt_tokens: int, limit: int | None, fallback: int) -> int:
    """The output cap that fills the context: ``limit`` less the prompt, at
    least 1. ``fallback`` when the context length is unknown."""
    if not limit:
        return fallback
    return max(1, int(limit) - int(prompt_tokens))


def install_until_eos_default() -> None:
    """Swap the UNTIL_EOS marker for the room left in the context, at the
    first ResponseGenerator hook that sees both the arguments and the prompt.
    Install after every other ``_make_thinking_budget_criteria`` patch.
    Idempotent."""
    from .api_contract import model_context_limit

    gen = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")
    cls = getattr(gen, "ResponseGenerator", None)
    if cls is None or not hasattr(cls, "_make_thinking_budget_criteria"):
        return
    orig = cls._make_thinking_budget_criteria
    if wrapped_with(orig, _UNTIL_EOS_FLAG):
        return

    def _make_thinking_budget_criteria(self, args, input_ids):
        if getattr(args, "max_tokens", None) == UNTIL_EOS:
            n = getattr(input_ids, "size", None)
            n = int(n) if n is not None else len(input_ids)
            limit = (model_context_limit(serving.get_active_spec())
                     or native_context(self.model))
            args.max_tokens = until_eos_cap(n, limit, gen.DEFAULT_MAX_TOKENS)
        return orig(self, args, input_ids)

    _make_thinking_budget_criteria.__dict__.update(getattr(orig, "__dict__", {}))
    _make_thinking_budget_criteria.__dict__[_UNTIL_EOS_FLAG] = True
    cls._make_thinking_budget_criteria = link(_make_thinking_budget_criteria, orig)


# XTC sampling (request extras / profile -> per-request logits processor)
# mlx-vlm's batch engine samples with its own positioned sampler (temperature +
# top_p only) - XTC can't ride the sampler. But XTC is a pure logits transform,
# and GenerationArguments.logits_processors is applied per row in the batched
# decode loop, so we inject mlx-lm's apply_xtc there at the gen-args seam.
# MTP rounds run it at every verify position (gmlx.spec.row_procs).
_XTC_FLAG = "_kq_gguf_xtc_patch"


def _effective_request_param(request, spec, key, default=None):
    """A request field (schemas are ``extra="allow"``, so unknown fields ride
    along) wins; else the resolved profile's sampling value; else ``default``."""
    val = getattr(request, key, None)
    if val is not None:
        return val
    sampling = getattr(spec, "sampling", None) if spec is not None else None
    if sampling and sampling.get(key) is not None:
        return sampling[key]
    return default


_NEWLINE_PROBE_WARNED = False


def _warn_newline_probe_once(exc: BaseException) -> None:
    global _NEWLINE_PROBE_WARNED
    if not _NEWLINE_PROBE_WARNED:
        _NEWLINE_PROBE_WARNED = True
        _log.warning(
            "XTC newline probe failed (%s: %s); newline stays maskable "
            "by XTC for this process", type(exc).__name__, exc)


def _xtc_special_tokens(processor) -> list:
    """Newline + EOS token ids, excluded from XTC masking (the same convention
    as the run/chat CLI). Defensive: any tokenizer shape miss degrades to the
    ids that did resolve, with a warn-once for the newline probe."""
    tok = getattr(processor, "tokenizer", processor)
    if tok is None:
        return []
    ids: list = []
    try:
        ids.extend(tok.encode("\n", add_special_tokens=False))
    except TypeError:
        try:
            ids.extend(tok.encode("\n"))
        except Exception as exc:
            _warn_newline_probe_once(exc)
    except Exception as exc:
        _warn_newline_probe_once(exc)
    eos = getattr(tok, "eos_token_ids", None)
    if eos is None:
        eos = getattr(tok, "eos_token_id", None)
    if isinstance(eos, int):    # some backends expose eos_token_ids as a bare id
        eos = [eos]
    ids.extend(int(t) for t in (eos or []) if t is not None)
    return list(dict.fromkeys(ids))


def _sampling_float(name: str, value) -> float:
    """A request-supplied sampling knob as a float, or HTTP 400. These ride the
    ChatRequest's ``extra="allow"`` passthrough, so pydantic never type-checks
    them and a bad value would otherwise 500 out of the handler."""
    try:
        return float(value)
    except (TypeError, ValueError):
        from fastapi import HTTPException

        raise HTTPException(status_code=400,
                            detail=f"{name!r} must be a number (got {value!r})")


def _attach_xtc(args, request, processor):
    """Append an XTC logits processor to ``args`` when the request (or profile)
    asks for it. Returns ``args`` (mutated)."""
    spec = serving.get_active_spec()
    prob = _effective_request_param(request, spec, "xtc_probability")
    if prob is None:
        return args
    # extras arrive untyped (extra="allow"): coerce before the disable check so
    # a client's "0" (string) still disables instead of truthily attaching.
    p = _sampling_float("xtc_probability", prob)
    if not p:
        return args
    threshold = _sampling_float(
        "xtc_threshold",
        _effective_request_param(request, spec, "xtc_threshold", 0.0))
    from gmlx._vendor.mlx_lm.sample_utils import apply_xtc   # lazy: pulls in mlx
    special = _xtc_special_tokens(processor)

    def xtc_processor(_tokens, logits):
        return apply_xtc(logits, p, threshold, special)

    procs = list(args.logits_processors or [])
    procs.append(xtc_processor)
    args.logits_processors = procs
    return args


def install_xtc_sampling() -> None:
    """Honour ``xtc_probability`` / ``xtc_threshold`` from request extras or the
    active profile by injecting a logits processor at the gen-args seam."""
    _install_gen_args_transform(_XTC_FLAG, _attach_xtc)


# 7a2. top_k / min_p aware batch sampler
# mlx-vlm's batch engine samples through ResponseGenerator._make_sampler, which
# returns a _PositionedTargetSampler that knows only temperature + top_p and runs
# a full-vocab argsort every step. So for any temperature>0 request it (a) silently
# drops a client-set top_k / min_p and (b) uses the wrong filter order: llama.cpp
# truncates to top_k first, so top_p / min_p run over k candidates, not the whole
# vocab. This patch swaps in a sampler that applies the llama.cpp order -- top_k ->
# top_p -> min_p -> temperature (last) -> sample -- and bounds every sort to k. It
# preserves the per-(row, position) key contract so MTP ragged verification stays
# deterministic: only the final categorical-over-k is keyed (the proven mlx-vlm
# vmap pattern), the heavy filtering is fully batched. The single-stream generate
# path (mlx_vlm.generate.ar) already routes to make_sampler when top_k / min_p are
# set, so only this batch seam needs the fix.
_FAST_SAMPLER_FLAG = "_kq_gguf_fast_sampler_patch"

_TOPP_ONLY_CAP = 1024   # top_p-only nucleus is bounded to this many candidates
_MIN_KEEP = 1           # always keep the argmax so a row can never go all -inf
_HIER_MIN_V = HIER_MIN_V
_topk_ids = topk_ids


class _FastPositionedSampler:
    """top_k / min_p aware drop-in for mlx-vlm's _PositionedTargetSampler.

    Consumed purely by duck typing (``sample_target`` for ragged verification,
    ``__call__`` otherwise); exposes the same attributes plus top_k / min_p.

    Conventions match unpatched mlx_lm / mlx-vlm: ``top_p`` and ``min_p`` of 0
    mean *disabled* (no filter), NOT "keep only the argmax" - so ``top_p: 0`` is
    a no-op, exactly as on the stock server (and as an mlx_lm-style model config
    carrying ``top_p: 0`` expects). When only ``top_p`` is set (``top_k == 0``),
    the nucleus is bounded to ``_TOPP_ONLY_CAP`` (1024) candidates so the sort
    stays batched; on a very flat distribution the tail past rank 1024 is dropped.
    """

    def __init__(self, *, temperature, top_p=1.0, top_k=0, min_p=0.0, seed=None):
        from gmlx._vendor.mlx_vlm.server.generation import DEFAULT_SEED
        self.temperature = float(temperature)
        self.top_p = float(top_p or 1.0)
        self.top_k = int(top_k or 0)
        self.min_p = float(min_p or 0.0)
        self.seed = DEFAULT_SEED if seed is None else int(seed)
        # Per-request seeds (uid -> seed) and the row->uid context for the
        # draw in flight, published by gmlx.serve.row_sampling. A row with a
        # registered seed draws from its own key stream; every other row
        # keeps the stock derivation byte for byte.
        self._kq_row_seeds: dict = {}
        self._kq_rows = None

    def _row_keys(self, row_ids, positions):
        import mlx.core as mx
        from gmlx._vendor.mlx_vlm.server.generation import _position_keys, _position_seed
        rows = self._kq_rows
        seeds = self._kq_row_seeds
        if not seeds or rows is None or len(rows) != len(row_ids):
            return _position_keys(self.seed, row_ids, positions)
        return mx.stack([
            mx.random.key(_position_seed(
                seeds.get(u, self.seed), row, pos))
            for u, row, pos in zip(rows, row_ids, positions)])

    @property
    def _has_filter(self):
        return self.top_k > 0 or self.top_p < 1.0 or self.min_p > 0.0

    def _filtered(self, logprobs):
        # logprobs [B, V] -> (masked [B, k] temp-scaled logits, part, order) where
        # part [B, k] holds the k survivors' vocab ids and order sorts them by prob
        # desc, so a sampled rank maps back to a vocab id via _resolve. Bounds the
        # sort to k instead of the full vocab.
        import mlx.core as mx
        lp = logprobs.astype(mx.float32)
        v = lp.shape[-1]
        k = self.top_k if 0 < self.top_k < v else min(_TOPP_ONLY_CAP, v)
        if k <= 64 and v >= _HIER_MIN_V:
            part = _topk_ids(lp, k)                              # [B, k] ids
        else:
            part = mx.argpartition(-lp, kth=k - 1, axis=-1)[:, :k]  # [B, k] ids
        cand = mx.take_along_axis(lp, part, axis=-1)             # [B, k] raw logits
        probs = mx.softmax(cand, axis=-1)                       # over the survivors
        order = mx.argsort(-probs, axis=-1)                    # [B, k] desc by prob
        sp = mx.take_along_axis(probs, order, axis=-1)
        ranks = mx.arange(k)
        keep = ranks < _MIN_KEEP
        if self.top_p < 1.0:
            keep = keep | ((mx.cumsum(sp, axis=-1) - sp) < self.top_p)  # mass above
        else:
            keep = keep | mx.ones_like(ranks).astype(mx.bool_)
        if self.min_p > 0.0:
            keep = keep & ((sp >= sp[:, :1] * self.min_p) | (ranks < _MIN_KEEP))
        kept = mx.take_along_axis(cand, order, axis=-1)
        masked = mx.where(keep, kept * (1.0 / self.temperature),
                          mx.array(-float("inf"), dtype=mx.float32))
        return masked, part, order

    @staticmethod
    def _resolve(part, order, pos):
        import mlx.core as mx
        chosen = mx.take_along_axis(order, pos[:, None], axis=-1)   # rank -> sorted
        return mx.take_along_axis(part, chosen, axis=-1)[:, 0]      # -> vocab id

    def __call__(self, logprobs):
        # logprobs is [B, V] on the main path but [B, 1, V] from a drafter's
        # draft_block; flatten the leading dims for the 2-D filter, then restore
        # them so the returned ids keep the caller's shape (matches top_p_sampling).
        import mlx.core as mx
        if not self._has_filter:
            # f32: float16 logprobs would otherwise reach categorical unwidened
            lp = logprobs.astype(mx.float32)
            return mx.random.categorical(lp * (1.0 / self.temperature), axis=-1)
        lead = logprobs.shape[:-1]
        lp2 = logprobs.reshape(-1, logprobs.shape[-1])
        masked, part, order = self._filtered(lp2)
        tok = self._resolve(part, order, mx.random.categorical(masked, axis=-1))
        return tok.reshape(lead)

    def sample_target(self, logprobs, *, row_ids, positions):
        import mlx.core as mx
        if logprobs.shape[0] != len(row_ids) or len(row_ids) != len(positions):
            raise ValueError("row_ids and positions must match logprobs batch size.")
        keys = self._row_keys(row_ids, positions)                   # [B, 2]

        def _cat(row, key):
            return mx.random.categorical(row, key=key)

        if not self._has_filter:
            scaled = logprobs.astype(mx.float32) * (1.0 / self.temperature)
            return mx.vmap(_cat, in_axes=(0, 0))(scaled, keys)
        # only this tiny categorical-over-k is per-row keyed; the filter is batched.
        masked, part, order = self._filtered(logprobs)
        pos = mx.vmap(_cat, in_axes=(0, 0))(masked, keys)
        return self._resolve(part, order, pos)


def install_fast_sampler() -> None:
    """Swap ``ResponseGenerator._make_sampler`` for one that honours top_k / min_p
    and applies the llama.cpp filter order. Idempotent; greedy (temperature 0)
    still returns ``None`` so the batch engine keeps its argmax fast path."""
    from gmlx._vendor.mlx_vlm.server import generation as gen
    if wrapped_with(gen.ResponseGenerator._make_sampler, _FAST_SAMPLER_FLAG):
        return
    stock_make_sampler = gen.ResponseGenerator._make_sampler

    def _make_sampler(self, args):
        if args.temperature == 0:
            return None
        # 0.6.15 samplers the fast path does not implement; use stock.
        if (getattr(args, "top_n_sigma", 0.0) > 0
                or getattr(args, "p_less", False)
                or getattr(args, "typical_p", 1.0) < 1.0):
            return stock_make_sampler(self, args)
        return _FastPositionedSampler(
            temperature=args.temperature,
            top_p=getattr(args, "top_p", 1.0),
            top_k=getattr(args, "top_k", 0),
            min_p=getattr(args, "min_p", 0.0),
            seed=getattr(args, "seed", None))

    _make_sampler.__dict__[_FAST_SAMPLER_FLAG] = True
    gen.ResponseGenerator._make_sampler = link(_make_sampler, stock_make_sampler)
