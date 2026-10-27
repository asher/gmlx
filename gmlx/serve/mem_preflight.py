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
in v1). The media limits of :mod:`gmlx.serve.media_decode` bound the
memory that decoding a request's media takes. Errors raise a
PromptTooLongError subclass, so every existing handler mapping to 400
applies unchanged.

Knobs:
    GMLX_PREFLIGHT_MEM=0   kill switch, checked per request
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from gmlx.upstream.patching import link, wrapped_with

_log = logging.getLogger(__name__)

_INSTALLED_FLAG = "_kq_gguf_mem_preflight"
GB = 1e9


_ERR_CLS = None


def _preflight_error_cls():
    global _ERR_CLS
    if _ERR_CLS is None:
        from gmlx._vendor.mlx_vlm.server.generation import PromptTooLongError

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
    token (a recurrent layer's state; kvarn's fp16 sink, horizon and
    tail): every row is charged however short the context."""


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
    rows when it rotates), a fixed recurrent state, and the bytes per
    token of the pools that grow to the full depth beside it."""
    attn: bool = True
    window: int | None = None
    state: float = 0.0
    pool: float = 0.0


_POOL_BYTES = 2     # pooled rows are priced fp16, the unpacked bound


def _int_set(c, name):
    v = _get(c, name)
    return {int(i) for i in v} if isinstance(v, (list, tuple)) else None


def _indexer_layers(c, layers):
    """The layers of a DSA model that own a lightning indexer, read the
    way the model builds them, or None for another family."""
    mt = _get(c, "model_type")
    if mt == "deepseek_v32":
        return set(range(layers))
    if mt == "hy_v4":
        if not _get(c, "index_topk"):
            return set()
        full = _get(c, "index_is_full")
        if not isinstance(full, (list, tuple)) or not full:
            return set(range(layers))
        return {i for i in range(min(layers, len(full))) if full[i]}
    if mt != "glm_moe_dsa":
        return None
    types = _get(c, "indexer_types")
    pattern = _get(c, "index_topk_pattern")
    if not isinstance(types, (list, tuple)) and pattern is not None:
        types = ([{"F": "full", "S": "shared"}.get(ch) for ch in pattern]
                 if isinstance(pattern, str) else pattern)
    if isinstance(types, (list, tuple)):
        return {i for i in range(min(layers, len(types)))
                if types[i] != "shared"}
    freq = max(_get(c, "index_topk_freq", 1) or 1, 1)
    offset = _get(c, "index_skip_topk_offset", 2)
    offset = 2 if offset is None else offset
    return {i for i in range(layers) if max(i - offset + 1, 0) % freq == 0}


def pool_bytes_per_token(c, layers):
    """Per config layer, the bytes per token of context its pools hold to
    the full depth: one row per ``compress_ratio`` tokens. DeepSeek V4
    keeps a latent pool ``head_dim`` wide on every compressed layer and an
    indexer key pool ``index_head_dim`` wide on the sparse ones (ratio 4).
    V4.1 keeps both pools only on its ``kv_source_layers``, the indexer
    pool filled only on its ``index_key_layers``; the other layers read
    them. glm5_next keeps an indexer pool on each attention layer, one row
    per ``index_kpool`` tokens. The DSA models (deepseek_v32, glm_moe_dsa,
    hy_v4) keep their indexer keys one row per token on each layer that
    owns an indexer. Zero for any other layout."""
    out = [0.0] * layers
    mt = _get(c, "model_type")
    width = _get(c, "head_dim") or 0
    index = _get(c, "index_head_dim") or 0
    indexed = _indexer_layers(c, layers)
    if indexed is not None:
        return [index * _POOL_BYTES if i in indexed else 0.0
                for i in range(layers)]
    if mt == "glm5_next":
        kpool = _get(c, "index_kpool")
        if isinstance(kpool, int) and kpool > 0 and index:
            return [index * _POOL_BYTES / kpool] * layers
        return out
    if mt not in ("deepseek_v4", "deepseek_v41"):
        return out
    ratios = _get(c, "compress_ratios")
    if not isinstance(ratios, (list, tuple)) or len(ratios) < layers:
        return out
    sources = _int_set(c, "kv_source_layers")
    keys = _int_set(c, "index_key_layers")
    for i in range(layers):
        r = int(ratios[i])
        if r <= 0 or (sources is not None and i not in sources):
            continue
        rows = width
        if (i in keys) if keys is not None else r != 128:
            rows += index
        out[i] = rows * _POOL_BYTES / r
    return out


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
    pools = pool_bytes_per_token(c, layers)
    geo = []
    for i, kind in enumerate(kinds):
        if kind is None:
            continue
        if kind == "kv" or state is None:
            pool = pools[i] if kind != "state" else 0.0
            geo.append(LayerGeometry(True, windows[i], pool=pool))
        elif kind == "state":
            geo.append(LayerGeometry(False, None, state))
        else:
            geo.append(LayerGeometry(True, windows[i], state, pools[i]))
    return geo


def stack_geometry(model, stack):
    """Geometry read off a constructed cache stack: kinds from the cache
    classes, rotating windows from the caches, state bytes and pool
    prices from the config. A recurrent cache whose state the config
    cannot size is priced as growing KV."""
    from gmlx.cache.kv_policy import _cache_kind_types, _classify

    types = _cache_kind_types()
    cfg = _lm_config(model)
    state = recurrent_state_bytes(cfg)
    layers = _get(cfg, "num_hidden_layers")
    # A kv_policy_exempt side cache (Granite Switch's router lanes) is not
    # a config layer. It grows at the few bytes per token it declares.
    exempt = [bool(getattr(c, "kv_policy_exempt", False)) for c in stack]
    # Pool prices are per config layer, so they apply only to a stack
    # with one entry per layer.
    real = len(stack) - sum(exempt)
    pools = (pool_bytes_per_token(cfg, layers) if layers == real
             else [0.0] * real)
    geo = []
    i = -1
    for c, side in zip(stack, exempt):
        if side:
            side_bpt = float(getattr(c, "side_bytes_per_token", 0.0))
            geo.append(LayerGeometry(False, None, 0.0, side_bpt))
            continue
        i += 1
        kind = _classify(c, types)
        members = getattr(c, "caches", None)
        members = list(members) if members is not None else [c]
        kinds = [_classify(m, types) for m in members]
        has_state = (state is not None
                     and any(isinstance(m, types["state"]) for m in members))
        # The config prices a layer's pools when the stack holds them: a
        # pooling cache, or a second growing cache (a DSA indexer's keys).
        pooled = any(hasattr(m, "quantize_storage") for m in members)
        growing = sum(k in ("kv", "optout") for k in kinds)
        pool = pools[i] if pools[i] and (pooled or growing > 1) else 0.0
        # The rotating window caps the layer whatever rides beside it, as
        # long as nothing else there grows unpriced: a side cache holds no
        # KV, and a recurrent state and a priced pool are priced on their
        # own. A plain KV member or an unpriced pool does grow, so those
        # layers keep the growing price.
        window = None
        if "kv" not in kinds and ("pool" not in kinds or pool):
            for m in members:
                if isinstance(m, types["window"]):
                    window = getattr(m, "max_size", None)
                    break
            window = window if isinstance(window, int) and window > 0 else None
        if kind == "state" and has_state:
            geo.append(LayerGeometry(False, None, state))
        else:
            geo.append(LayerGeometry(True, window, state if has_state else 0.0,
                                     pool))
    return geo


def kv_layer_costs(model, bytes_per_elem: float = 2.0, per_layer_bpe=None,
                   per_layer_regions=None, per_layer_steps=None,
                   geometry=None):
    """Cost entries ``(window_or_None, bytes_per_token)`` for the model's
    cache stack, or None when the geometry cannot be read: one entry per
    growing KV region (a rotating window caps it), one FixedRows(1)
    entry per recurrent state and one growing entry per layer's pools
    (compressed pools, DSA indexer keys), priced fp16. Admit-side
    throughout: MLA prices the compressed latent, a configured sliding
    window caps every layer it could apply to. ``geometry`` (from
    stack_geometry) replaces the config's layer pattern. Per geometry
    entry: ``per_layer_bpe`` (from the KV policy) prices the storage
    exactly; ``per_layer_steps`` marks entries whose storage grows in
    slabs (StepTokens windows); and ``per_layer_regions`` appends the
    fixed-size buffers a scheme holds beside the per-token record
    (kvarn's fp16 sink, horizon and tail), each as a FixedRows entry
    after the stack entries. All three must match the geometry length
    or they are ignored."""
    c = _lm_config(model)
    if geometry is None:
        geometry = config_geometry(c)
    if not geometry:
        return None
    n = len(geometry)
    if per_layer_bpe is not None and len(per_layer_bpe) != n:
        per_layer_bpe = None
    if per_layer_regions is not None and len(per_layer_regions) != n:
        per_layer_regions = None
    if per_layer_steps is not None and len(per_layer_steps) != n:
        per_layer_steps = None

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
            w = g.window
            step = per_layer_steps[i] if per_layer_steps is not None else 0
            if w is None and step:
                w = StepTokens(step)
            costs.append((w, elems * bpe(i)))
        if g.state:
            costs.append((FixedRows(1), float(g.state)))
        if g.pool:
            costs.append((None, float(g.pool)))
    for regions in per_layer_regions or ():
        for rows, region_bpe in regions:
            costs.append((FixedRows(rows), elems * region_bpe))
    return costs


def _policy_costs(rg, model):
    """kv_layer_costs priced from rg's resolved KV policy (batched mode)
    over the stack the policy was resolved on. Without a policy, or when
    the stack cannot be probed, pricing stays uniform fp16 over the
    config's geometry."""
    from .kv_policy import (_probe_stack, pricing_vector, region_vector,
                            step_vector)

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
    n = len(geometry)
    return kv_layer_costs(model, 2.0, per_layer_bpe=pricing_vector(rg, n),
                          per_layer_regions=region_vector(rg, n),
                          per_layer_steps=step_vector(rg, n),
                          geometry=geometry)


def prompt_kv_bytes(costs, tokens: int) -> float:
    return sum(bpt * span_tokens(w, tokens) for w, bpt in costs)


def available_drained_bytes():
    """Working set minus zero-copy weights, minus a streaming model's
    decode arena, minus the admission reserve: what a lone request could
    hold with the batch drained. The arena is MLX-tracked, sized at
    install and never drained. Other MLX-tracked weight allocations are
    not subtracted, which only admits more."""
    from gmlx.device import metal_value
    from gmlx.gen.prefill_decay import untracked_weight_bytes
    from gmlx.stream.installs import live_arena_bytes
    from .memory import admit_reserve_bytes

    try:
        ws = float(metal_value("max_recommended_working_set_size"))
    except Exception:
        return None
    if ws <= 0:
        return None
    return (ws - untracked_weight_bytes() - live_arena_bytes()
            - admit_reserve_bytes(ws))


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
        from gmlx._vendor.mlx_vlm.server.generation import _count_prompt_tokens
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
            f"prompt is too long for the memory available: {what} needs an "
            f"estimated {worst / GB:.1f} GB of KV and prefill transient, but "
            f"only {avail / GB:.1f} GB remains with the batch drained "
            f"(prompt_tokens={tokens}). Reduce the prompt"
            + (" or max_tokens" if need_pinned > avail else "") + ".")
    except Exception as e:
        from gmlx._vendor.mlx_vlm.server.generation import PromptTooLongError
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
        from gmlx._vendor.mlx_vlm.server.generation import get_server_max_tokens
        if mt == get_server_max_tokens():
            return 0
    except Exception:
        return 0
    return mt


def _check_media(images, audio, videos) -> None:
    """Refuse a media path or URL that reached the generation path, and
    media past the limits of one request, with the rules of
    :mod:`gmlx.serve.patches.media_gate`, before anything decodes. The
    refusal is a PromptTooLongError, which the generation routes answer
    with 400."""
    from .patches.media_gate import MediaRefused, check_media_lists, route_refusal

    try:
        check_media_lists(images, audio, videos)
    except MediaRefused as e:
        raise route_refusal(e) from None


def install_memory_preflight() -> None:
    """Run the memory preflight on both request entry points.

    ``validate_context_budget`` covers streaming routes before the SSE
    stream opens; ``generate`` covers non-streaming handlers and the
    gmlx completions route, which call it before any response starts.
    Idempotent."""
    from gmlx._vendor.mlx_vlm.server.generation import ResponseGenerator

    if wrapped_with(ResponseGenerator.generate, _INSTALLED_FLAG):
        return

    _orig_generate = ResponseGenerator.generate
    _orig_validate = ResponseGenerator.validate_context_budget

    def _generate(self, prompt, images=None, audio=None, args=None,
                  videos=None):
        _check_media(images, audio, videos)
        preflight_prompt_memory(self, prompt, images, audio, videos, args)
        return _orig_generate(self, prompt, images=images, audio=audio,
                              args=args, videos=videos)

    def _validate(self, prompt, images=None, audio=None, args=None,
                  videos=None):
        _check_media(images, audio, videos)
        _orig_validate(self, prompt, images=images, audio=audio,
                       args=args, videos=videos)
        preflight_prompt_memory(self, prompt, images, audio, videos, args)

    _generate.__dict__[_INSTALLED_FLAG] = True
    _validate.__dict__[_INSTALLED_FLAG] = True
    ResponseGenerator.generate = link(_generate, _orig_generate)
    ResponseGenerator.validate_context_budget = link(_validate, _orig_validate)
    _log.info("memory preflight installed")
