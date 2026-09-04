"""Prompt-side memory preflight (GMLX_PREFLIGHT_MEM).

A request whose KV cannot fit dies mid-stream today, or trips the
pressure machinery late. An error before the SSE stream opens is
retryable and reportable by every client; a mid-stream abort is a
half-answer. This preflight estimates the prompt-side peak (prompt KV
at the model's per-token cost plus the prefill score transient) and
rejects with HTTP 400 and the numbers in the body only on prompt-side
impossibility: the prompt alone cannot fit even with the batch drained.

Generation-side risk is deliberately not rejected. ``max_tokens`` is
counted only when the client pinned it explicitly, and even then only
prompt plus pinned-max against the drained budget. Default-max requests
are never rejected on generation length; the pressure machinery owns
the tail. Preflight declines the impossible, not the unlikely.

Every estimate errs on the admit side: quantized KV prices at its bits
without scale overhead, sliding windows cap the token count, MLA prices
the compressed latent, and unprobeable geometry skips the check. Media
requests skip too (image KV and encoder transients are not estimated
in v1). Errors raise a PromptTooLongError subclass, so every existing
handler mapping to 400 applies unchanged.

Knobs:
    GMLX_PREFLIGHT_MEM=0   kill switch, checked per request
"""

from __future__ import annotations

import logging
import os

_log = logging.getLogger(__name__)

_INSTALLED_FLAG = "_kq_gguf_mem_preflight"
GB = 1e9


_ERR_CLS = None


def _preflight_error_cls():
    global _ERR_CLS
    if _ERR_CLS is None:
        from mlx_vlm.server.generation import PromptTooLongError

        class MemoryPreflightError(PromptTooLongError):
            """Prompt-side KV cannot fit the drained working set."""

        _ERR_CLS = MemoryPreflightError
    return _ERR_CLS


def _get(cfg, name, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _lm_config(model):
    cfg = getattr(model, "config", None)
    text = _get(cfg, "text_config")
    return text if text is not None else cfg


def _layer_windows(c, layers):
    """Per-layer sliding window from the config, None where a layer keeps
    full attention."""
    window = _get(c, "sliding_window")
    window = window if isinstance(window, int) and window > 0 else None
    if window is None:
        return [None] * layers
    types = _get(c, "layer_types")
    if isinstance(types, (list, tuple)) and len(types) == layers:
        return [window if "sliding" in str(t) else None for t in types]
    pattern = _get(c, "sliding_window_pattern")
    if isinstance(pattern, int) and pattern > 0:
        return [None if (i + 1) % pattern == 0 else window
                for i in range(layers)]
    return [window] * layers


class FixedRows(int):
    """Cost-entry window for a buffer allocated in full at the first
    token (kvarn's fp16 sink, horizon and tail): every row is charged
    however short the context."""


class StepTokens(int):
    """Cost-entry window for storage allocated in slabs of this many
    tokens (kvarn record codes): charged at the slab ceiling."""


def span_tokens(window, tokens: int) -> int:
    """Rows a cost entry charges at ``tokens`` of context."""
    if tokens <= 0:
        return 0
    if window is None:
        return tokens
    if isinstance(window, FixedRows):
        return int(window)
    if isinstance(window, StepTokens):
        step = int(window)
        return -(-tokens // step) * step
    return min(tokens, int(window))


def per_token_bytes(costs) -> float:
    """Bytes of KV per token of context; fixed buffers excluded."""
    return sum(bpt for w, bpt in costs if not isinstance(w, FixedRows))


def kv_layer_costs(model, bytes_per_elem: float = 2.0, per_layer_bpe=None,
                   per_layer_regions=None, per_layer_steps=None):
    """Per-layer ``(window_or_None, bytes_per_token)`` for the KV cache,
    or None when the geometry cannot be read. Admit-side throughout: MLA
    prices the compressed latent, a configured sliding window caps every
    layer it could apply to. ``per_layer_bpe`` (from the KV policy)
    prices each layer's storage exactly; ``per_layer_steps`` marks layers
    whose storage grows in slabs (StepTokens windows); and
    ``per_layer_regions`` appends the fixed-size buffers a scheme holds
    beside the per-token record -- kvarn's fp16 sink, horizon and tail
    -- each as a FixedRows entry, so the list runs longer than the layer
    count. All three must match the layer count or they are ignored."""
    c = _lm_config(model)
    layers = _get(c, "num_hidden_layers")
    if not isinstance(layers, int) or layers <= 0:
        return None
    if per_layer_bpe is not None and len(per_layer_bpe) != layers:
        per_layer_bpe = None
    if per_layer_regions is not None and len(per_layer_regions) != layers:
        per_layer_regions = None
    if per_layer_steps is not None and len(per_layer_steps) != layers:
        per_layer_steps = None

    def bpe(i):
        return per_layer_bpe[i] if per_layer_bpe is not None else bytes_per_elem

    lora = _get(c, "kv_lora_rank")
    if isinstance(lora, int) and lora > 0:
        rope = _get(c, "qk_rope_head_dim", 0) or 0
        elems = (lora + rope)
        windows = [None] * layers
    else:
        heads = _get(c, "num_attention_heads")
        n_kv = _get(c, "num_key_value_heads") or heads
        head_dim = _get(c, "head_dim")
        hidden = _get(c, "hidden_size")
        if not head_dim and heads and hidden:
            head_dim = hidden // heads
        if not (isinstance(n_kv, int) and n_kv > 0
                and isinstance(head_dim, int) and head_dim > 0):
            return None
        elems = 2 * n_kv * head_dim
        windows = _layer_windows(c, layers)

    costs = []
    for i in range(layers):
        w = windows[i]
        step = per_layer_steps[i] if per_layer_steps is not None else 0
        if w is None and step:
            w = StepTokens(step)
        costs.append((w, elems * bpe(i)))
    for regions in per_layer_regions or ():
        for rows, region_bpe in regions:
            costs.append((FixedRows(rows), elems * region_bpe))
    return costs


def _policy_costs(rg, model):
    """kv_layer_costs priced from rg's resolved KV policy (batched
    mode). Without a policy, pricing stays uniform fp16."""
    c = _lm_config(model)
    layers = _get(c, "num_hidden_layers")
    vec = regions = steps = None
    if isinstance(layers, int) and layers > 0:
        from .kv_policy import pricing_vector, region_vector, step_vector

        vec = pricing_vector(rg, layers)
        regions = region_vector(rg, layers)
        steps = step_vector(rg, layers)
    return kv_layer_costs(model, 2.0, per_layer_bpe=vec,
                          per_layer_regions=regions, per_layer_steps=steps)


def prompt_kv_bytes(costs, tokens: int) -> float:
    return sum(bpt * span_tokens(w, tokens) for w, bpt in costs)


def available_drained_bytes():
    """Working set minus zero-copy weights minus the admission reserve:
    what a lone request could hold with the batch drained. MLX-tracked
    weight allocations are not subtracted, which only admits more."""
    import mlx.core as mx

    from gmlx.gen.prefill_decay import untracked_weight_bytes
    from .memory import admit_reserve_bytes

    try:
        ws = float(mx.device_info()["max_recommended_working_set_size"])
    except Exception:
        return None
    if ws <= 0:
        return None
    return ws - untracked_weight_bytes() - admit_reserve_bytes(ws)


def _need_bytes(model, costs, prompt_tokens: int, gen_tokens: int = 0):
    from gmlx.gen.prefill_decay import score_transient_bytes

    kv = prompt_kv_bytes(costs, prompt_tokens + gen_tokens)
    return kv + score_transient_bytes(model, None, prompt_tokens)


def preflight_prompt_memory(rg, prompt, images=None, audio=None,
                            videos=None, args=None) -> None:
    """Raise MemoryPreflightError when the prompt cannot fit. Best
    effort: any probe failure admits."""
    try:
        if os.environ.get("GMLX_PREFLIGHT_MEM", "1") == "0":
            return
        if images or audio or videos:
            return
        model = getattr(rg, "model", None)
        if model is None or not isinstance(prompt, str):
            return
        costs = _policy_costs(rg, model)
        if not costs:
            return
        avail = available_drained_bytes()
        if avail is None:
            return
        # A text token is at least one character, so the character count
        # bounds the token count; most requests never pay a tokenize.
        if _need_bytes(model, costs, len(prompt)) <= avail:
            pinned = _pinned_max_tokens(args)
            if not pinned or _need_bytes(
                    model, costs, len(prompt), pinned) <= avail:
                return
        from mlx_vlm.server.generation import _count_prompt_tokens
        tokens = _count_prompt_tokens(rg._preprocess_request(prompt))
        if tokens <= 0:
            return
        need = _need_bytes(model, costs, tokens)
        pinned = _pinned_max_tokens(args)
        need_pinned = (_need_bytes(model, costs, tokens, pinned)
                       if pinned else 0.0)
        if need <= avail and need_pinned <= avail:
            return
        what = ("prompt" if need > avail
                else f"prompt + max_tokens {pinned}")
        worst = max(need, need_pinned)
        raise _preflight_error_cls()(
            f"request cannot fit: {what} needs an estimated "
            f"{worst / GB:.1f} GB of KV and prefill transient, but only "
            f"{avail / GB:.1f} GB remains with the batch drained "
            f"(prompt_tokens={tokens}). Reduce the prompt"
            + (" or max_tokens" if need_pinned > avail else "") + ".")
    except Exception as e:
        from mlx_vlm.server.generation import PromptTooLongError
        if isinstance(e, PromptTooLongError):
            raise
        _log.warning("memory preflight failed; admitting", exc_info=True)


def _pinned_max_tokens(args):
    """The request's max_tokens, only when the client pinned it away
    from the server default."""
    mt = getattr(args, "max_tokens", None)
    if not isinstance(mt, int) or mt <= 0:
        return 0
    try:
        from mlx_vlm.server.generation import get_server_max_tokens
        if mt == get_server_max_tokens():
            return 0
    except Exception:
        return 0
    return mt


def install_memory_preflight() -> None:
    """Run the memory preflight on both request entry points.

    ``validate_context_budget`` covers streaming routes before the SSE
    stream opens; ``generate`` covers non-streaming handlers and the
    gmlx completions route, which call it before any response starts.
    Idempotent."""
    from mlx_vlm.server.generation import ResponseGenerator

    if getattr(ResponseGenerator.generate, _INSTALLED_FLAG, False):
        return

    _orig_generate = ResponseGenerator.generate
    _orig_validate = ResponseGenerator.validate_context_budget

    def _generate(self, prompt, images=None, audio=None, args=None,
                  videos=None):
        preflight_prompt_memory(self, prompt, images, audio, videos, args)
        return _orig_generate(self, prompt, images=images, audio=audio,
                              args=args, videos=videos)

    def _validate(self, prompt, images=None, audio=None, args=None,
                  videos=None):
        _orig_validate(self, prompt, images=images, audio=audio,
                       args=args, videos=videos)
        preflight_prompt_memory(self, prompt, images, audio, videos, args)

    _generate.__dict__[_INSTALLED_FLAG] = True
    _validate.__dict__[_INSTALLED_FLAG] = True
    ResponseGenerator.generate = _generate
    ResponseGenerator.validate_context_budget = _validate
    _log.info("memory preflight installed")
