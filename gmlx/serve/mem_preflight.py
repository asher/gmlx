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
from dataclasses import dataclass

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
    token (a recurrent layer's state): every row is charged however
    short the context."""


def span_tokens(window, tokens: int) -> int:
    """Rows a cost entry charges at ``tokens`` of context."""
    if tokens <= 0:
        return 0
    if window is None:
        return tokens
    if isinstance(window, FixedRows):
        return int(window)
    return min(tokens, int(window))


def per_token_bytes(costs) -> float:
    """Bytes of KV per token of context; fixed buffers excluded."""
    return sum(bpt for w, bpt in costs if not isinstance(w, FixedRows))


_STATE_BYTES = 4    # recurrent state is held fp32
_CONV_BYTES = 2     # conv tails are held in the activation dtype
_STATE_WORDS = ("linear", "mamba", "recurrent", "ssm", "conv")


def recurrent_state_bytes(c) -> float | None:
    """Fixed bytes one recurrent layer holds per sequence row (the fp32
    recurrent state plus the conv tails), or None when the config carries
    no recurrent geometry this can read. Families: gated DeltaNet
    (qwen3_5, qwen3_next, qwen4_exp), Mamba2 (nemotron_h, falcon_h1,
    granitemoehybrid), KDA (kimi_k3, glm5_next)."""
    nv = _get(c, "linear_num_value_heads")
    if nv:
        nk = _get(c, "linear_num_key_heads") or nv
        dk = _get(c, "linear_key_head_dim")
        dv = _get(c, "linear_value_head_dim")
        k = _get(c, "linear_conv_kernel_dim") or 4
        if not (dk and dv):
            return None
        conv_dim = 2 * nk * dk + nv * dv
        return float((k - 1) * conv_dim * _CONV_BYTES
                     + nv * dk * dv * _STATE_BYTES)
    kda = _get(c, "kda_head_dim")
    if kda:
        heads = _get(c, "num_attention_heads")
        k = _get(c, "ssm_conv_kernel") or 4
        if not heads:
            return None
        return float(3 * (k - 1) * heads * kda * _CONV_BYTES
                     + heads * kda * kda * _STATE_BYTES)
    heads = _get(c, "mamba_num_heads") or _get(c, "mamba_n_heads")
    head_dim = _get(c, "mamba_head_dim") or _get(c, "mamba_d_head")
    state = _get(c, "ssm_state_size") or _get(c, "mamba_d_state")
    if heads and head_dim and state:
        k = _get(c, "conv_kernel") or _get(c, "mamba_d_conv") or 4
        groups = _get(c, "n_groups") or _get(c, "mamba_n_groups") or 1
        inner = heads * head_dim
        conv_dim = inner + 2 * groups * state
        return float((k - 1) * conv_dim * _CONV_BYTES
                     + inner * state * _STATE_BYTES)
    return None


@dataclass(frozen=True)
class LayerGeometry:
    """One cache-stack entry: a growing KV region (capped at ``window``
    rows when it rotates) and/or a fixed recurrent state."""
    attn: bool = True
    window: int | None = None
    state: float = 0.0


def _config_kinds(c, layers):
    """Per config layer: ``kv``, ``state``, ``both``, or None for a block
    that owns no cache. Read from the family's layout key; a layer type
    this cannot name counts as growing KV."""
    types = _get(c, "layer_types")
    if isinstance(types, (list, tuple)) and len(types) == layers:
        return ["state" if any(w in str(t) for w in _STATE_WORDS) else "kv"
                for t in types]
    pattern = _get(c, "hybrid_override_pattern")
    if isinstance(pattern, (list, tuple, str)) and len(pattern) == layers:
        return [{"M": "state", "*": "kv"}.get(str(p)) for p in pattern]
    interval = _get(c, "full_attention_interval")
    if (isinstance(interval, int) and interval > 0
            and _get(c, "linear_num_value_heads")):
        return ["kv" if (i + 1) % interval == 0 else "state"
                for i in range(layers)]
    if _get(c, "model_type") == "falcon_h1":
        return ["both"] * layers
    return ["kv"] * layers


def config_geometry(c):
    """The cache-stack geometry a model of this config builds, one entry
    per cache the stack owns, or None when the layer count is unreadable.
    A recurrent family whose state this cannot size is priced as growing
    KV (admit-side)."""
    layers = _get(c, "num_hidden_layers")
    if not isinstance(layers, int) or layers <= 0:
        return None
    kinds = _config_kinds(c, layers)
    windows = _layer_windows(c, layers)
    state = recurrent_state_bytes(c)
    geo = []
    for i, kind in enumerate(kinds):
        if kind is None:
            continue
        if kind == "kv" or state is None:
            geo.append(LayerGeometry(True, windows[i]))
        elif kind == "state":
            geo.append(LayerGeometry(False, None, state))
        else:
            geo.append(LayerGeometry(True, windows[i], state))
    return geo


def stack_geometry(model, stack):
    """Geometry read off a constructed cache stack: kinds from the cache
    classes, rotating windows from the caches, state bytes from the
    config. A recurrent cache whose state the config cannot size is
    priced as growing KV."""
    from gmlx.cache.kv_policy import _cache_kind_types, _classify

    types = _cache_kind_types()
    state = recurrent_state_bytes(_lm_config(model))
    geo = []
    for c in stack:
        kind = _classify(c, types)
        members = getattr(c, "caches", None)
        members = list(members) if members is not None else [c]
        has_state = (state is not None
                     and any(isinstance(m, types["state"]) for m in members))
        window = None
        if kind == "window":
            for m in members:
                if isinstance(m, types["window"]):
                    window = getattr(m, "max_size", None)
                    break
            window = window if isinstance(window, int) and window > 0 else None
        if kind == "state" and has_state:
            geo.append(LayerGeometry(False, None, state))
        else:
            geo.append(LayerGeometry(True, window, state if has_state else 0.0))
    return geo


def kv_layer_costs(model, bytes_per_elem: float = 2.0, per_layer_bpe=None,
                   geometry=None):
    """Cost entries ``(window_or_None, bytes_per_token)`` for the model's
    cache stack, or None when the geometry cannot be read: one entry per
    growing KV region (a rotating window caps it) and one FixedRows(1)
    entry per recurrent state. Admit-side throughout: MLA prices the
    compressed latent, a configured sliding window caps every layer it
    could apply to. ``geometry`` (from stack_geometry) replaces the
    config's layer pattern; ``per_layer_bpe`` (from the KV policy) prices
    each entry's storage exactly and must match the geometry length or it
    is ignored."""
    c = _lm_config(model)
    if geometry is None:
        geometry = config_geometry(c)
    if not geometry:
        return None
    if per_layer_bpe is not None and len(per_layer_bpe) != len(geometry):
        per_layer_bpe = None

    def bpe(i):
        return per_layer_bpe[i] if per_layer_bpe is not None else bytes_per_elem

    lora = _get(c, "kv_lora_rank")
    if isinstance(lora, int) and lora > 0:
        elems = lora + (_get(c, "qk_rope_head_dim", 0) or 0)
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

    costs = []
    for i, g in enumerate(geometry):
        if g.attn:
            costs.append((g.window, elems * bpe(i)))
        if g.state:
            costs.append((FixedRows(1), float(g.state)))
    return costs


def _policy_costs(rg, model):
    """kv_layer_costs priced from rg's resolved KV policy (batched mode)
    over the stack the policy was resolved on. Without a policy, or when
    the stack cannot be probed, pricing stays uniform fp16 over the
    config's geometry."""
    from .kv_policy import _probe_stack, pricing_vector

    geometry = None
    try:
        geometry = stack_geometry(model, _probe_stack(model))
    except Exception:
        _log.debug("cache stack probe failed; pricing from the config",
                   exc_info=True)
    if not geometry:
        geometry = config_geometry(_lm_config(model))
    if not geometry:
        return None
    return kv_layer_costs(model, 2.0, per_layer_bpe=pricing_vector(
        rg, len(geometry)), geometry=geometry)


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
