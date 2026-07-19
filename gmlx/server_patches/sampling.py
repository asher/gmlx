"""Sampling-surface patches: profile injection into unset request
fields, XTC as a per-request logits processor, and the fast positioned
sampler."""

from __future__ import annotations

from .. import server_bridge_vlm as serving
from ._common import (
    _PATCH_FLAG,
    _install_gen_args_transform,
)


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
    "thinking_budget": ("thinking_budget",),
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


# XTC sampling (request extras / profile -> per-request logits processor)
# mlx-vlm's batch engine samples with its own positioned sampler (temperature +
# top_p only) - XTC can't ride the sampler. But XTC is a pure logits transform,
# and GenerationArguments.logits_processors is applied per row in the batched
# decode loop, so we inject mlx-lm's apply_xtc there at the gen-args seam.
# Caveat: the engines reject logits_processors under speculative decoding, so a
# request with XTC against a speculative model errors (documented).
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


def _xtc_special_tokens(processor) -> list:
    """Newline + EOS token ids, excluded from XTC masking (the same convention
    as the run/chat CLI). Defensive: any tokenizer shape miss degrades to []."""
    tok = getattr(processor, "tokenizer", processor)
    if tok is None:
        return []
    ids: list = []
    try:
        ids.extend(tok.encode("\n", add_special_tokens=False))
    except TypeError:
        try:
            ids.extend(tok.encode("\n"))
        except Exception:
            pass
    except Exception:
        pass
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
    from mlx_lm.sample_utils import apply_xtc   # lazy: pulls in mlx
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
_HIER_ROW = 128         # hierarchical top-k row width (see _topk_ids)
_HIER_MIN_V = 8192      # below this the flat argpartition is already cheap


def _topk_ids(lp, k):
    """Top-k vocab ids of ``lp`` [1, V] f32, hierarchically.

    ``mx.argpartition`` over a 200k vocab is a full-width sort on the GPU
    sitting on the decode critical path. Reshape to rows of ``_HIER_ROW``,
    reduce each row to its max (one bandwidth-bound pass), argpartition the
    ~V/128 row-maxes, then argpartition the k*128 surviving candidates.
    Exact: a value in the global top-k bounds its row's max from below, so
    fewer than k rows can outrank that row -- every top-k value's row
    survives the row cut. Only tie ORDER at the k-th value can differ from
    the flat argpartition.
    """
    import mlx.core as mx
    v = lp.shape[-1]
    n_rows = (v + _HIER_ROW - 1) // _HIER_ROW
    if n_rows * _HIER_ROW != v:
        pad = mx.full(
            (1, n_rows * _HIER_ROW - v), -float("inf"), dtype=lp.dtype)
        lp = mx.concatenate([lp, pad], axis=-1)
    rows = lp.reshape(n_rows, _HIER_ROW)
    top_rows = mx.argpartition(-rows.max(axis=1), kth=k - 1)[:k]   # [k]
    cand = rows[top_rows].reshape(-1)                              # [k * 128]
    loc = mx.argpartition(-cand, kth=k - 1)[:k]                    # [k]
    ids = top_rows[loc // _HIER_ROW] * _HIER_ROW + (loc % _HIER_ROW)
    return ids[None]


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
        from mlx_vlm.server.generation import DEFAULT_SEED
        self.temperature = float(temperature)
        self.top_p = float(top_p or 1.0)
        self.top_k = int(top_k or 0)
        self.min_p = float(min_p or 0.0)
        self.seed = DEFAULT_SEED if seed is None else int(seed)

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
        if lp.shape[0] == 1 and k <= 64 and v >= _HIER_MIN_V:
            part = _topk_ids(lp, k)                              # [1, k] ids
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
            return mx.random.categorical(logprobs * (1.0 / self.temperature), axis=-1)
        lead = logprobs.shape[:-1]
        lp2 = logprobs.reshape(-1, logprobs.shape[-1])
        masked, part, order = self._filtered(lp2)
        tok = self._resolve(part, order, mx.random.categorical(masked, axis=-1))
        return tok.reshape(lead)

    def sample_target(self, logprobs, *, row_ids, positions):
        import mlx.core as mx
        from mlx_vlm.server.generation import _position_keys
        if logprobs.shape[0] != len(row_ids) or len(row_ids) != len(positions):
            raise ValueError("row_ids and positions must match logprobs batch size.")
        keys = _position_keys(self.seed, row_ids, positions)        # [B, 2]

        def _cat(row, key):
            return mx.random.categorical(row, key=key)

        if not self._has_filter:
            scaled = logprobs * (1.0 / self.temperature)
            return mx.vmap(_cat, in_axes=(0, 0))(scaled, keys)
        # only this tiny categorical-over-k is per-row keyed; the filter is batched.
        masked, part, order = self._filtered(logprobs)
        pos = mx.vmap(_cat, in_axes=(0, 0))(masked, keys)
        return self._resolve(part, order, pos)


def install_fast_sampler() -> None:
    """Swap ``ResponseGenerator._make_sampler`` for one that honours top_k / min_p
    and applies the llama.cpp filter order. Idempotent; greedy (temperature 0)
    still returns ``None`` so the batch engine keeps its argmax fast path."""
    from mlx_vlm.server import generation as gen
    if getattr(gen.ResponseGenerator._make_sampler, _FAST_SAMPLER_FLAG, False):
        return

    def _make_sampler(self, args):
        if args.temperature == 0:
            return None
        return _FastPositionedSampler(
            temperature=args.temperature,
            top_p=getattr(args, "top_p", 1.0),
            top_k=getattr(args, "top_k", 0),
            min_p=getattr(args, "min_p", 0.0),
            seed=getattr(args, "seed", None))

    _make_sampler.__dict__[_FAST_SAMPLER_FLAG] = True
    gen.ResponseGenerator._make_sampler = _make_sampler
