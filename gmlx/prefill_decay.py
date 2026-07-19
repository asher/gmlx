"""Depth-decay prefill chunk scaling.

Chunked prefill on head dims stock MLX has no fused prefill kernel for
(256, 512) materializes a [heads, step, S] score tensor per full-attention
layer, so the transient grows linearly in step x depth and crests on the
last chunk. The blunt lever (PREFILL_STEP_SIZE) shrinks the chunk globally,
which taxes MoE prefill at every depth: lost weight amortization prefills a
122B A10B at roughly 100/91/71/50% of full throughput at step
2048/1024/512/256. Decay instead halves the step only once the current
depth would push the transient past a cap:

    step(S) = largest tier in {base, base/2, ..., min_step} with
              heads * step * (S + step) * 2 bytes <= cap

Shallow prefill keeps the full step; only the deep tail (where attention
dominates the chunk anyway) pays the small-step tax. Applied to the serve
prefill loop (prompt_step and needs_processing -- the loop hands its final
<=step tail to generate(), which is exactly the cresting chunk), the MTP
prefill override, and the MTP teacher-force head seed.

The dense [heads, step, S] model is the conservative default. Archs whose
prefill attention runs fused/streaming kernels never materialize it; their
real transient is far smaller (e.g. deepseek_v4's native DSA path peaks at
the indexer's [1, step, S/ratio] fp32 block). Such models register a
ScoreTransientProfile provider keyed by config.model_type; when the provider
arms (kernels present, single sequence), the fit test sizes the true
transient and the chunk stays full until that genuinely nears the cap. A
profile may also carry an arch-default base step, honored only when no
explicit PREFILL_STEP_SIZE is in force.

Knobs:
    GMLX_PREFILL_DECAY=0        kill switch (default on)
    GMLX_PREFILL_SCORE_CAP_GB   transient cap (default 5% of the GPU
                                    working set, floor 2 GB)
    GMLX_PREFILL_MIN_STEP       decay floor (default 256)
    GMLX_PREFILL_SCORE_PROFILE=0  ignore arch score profiles (dense model
                                    everywhere, pre-profile behavior)
    GMLX_PREFILL_HEADROOM_CAP=1 raise the body cap to min(half the live
                                    free working set, the configured
                                    GMLX_CACHE_LIMIT_GB); requires a cache
                                    limit (the churn-safe clamp) and is off
                                    by default pending receipts
    GMLX_MTP_SEED_SCORE_CAP_GB  explicit cap for the MTP teacher-force
                                    seed chunks (default: half the estimated
                                    free working set at seed time, floored
                                    at the prefill cap)
"""

from __future__ import annotations

import os
from typing import Callable, NamedTuple, Optional

import mlx.core as mx

from .envflags import env_bool, env_int

_WS_CAP_BYTES: float | None = None
_FLAG = "_gmlx_prefill_decay"


class ScoreTransientProfile(NamedTuple):
    """Per-arch description of the real prefill score transient."""

    heads: int = 1  # effective head count in the transient
    bytes_per_elem: int = 4  # effective bytes per score element
    depth_divisor: int = 1  # transient key length = (depth + step) / divisor
    base_step: int | None = None  # arch-default chunk (None = stock base)


# model_type -> provider(model, prompt_cache) -> profile | None. A provider
# returns None whenever the arch would take a materializing path (kernels
# unavailable, batched sequences, quantized KV) so the dense default stays
# authoritative there.
_SCORE_PROFILES: dict[str, Callable] = {}
_PROFILE_LOGGED: set[str] = set()


def register_score_profile(model_type: str, provider: Callable) -> None:
    _SCORE_PROFILES[model_type] = provider


def resolve_score_profile(model, prompt_cache) -> Optional[ScoreTransientProfile]:
    """Look up and invoke the arch's profile provider; None keeps the dense
    model. Re-resolved per chunk so runtime kernel disables take effect on
    the next chunk."""
    if not env_bool("GMLX_PREFILL_SCORE_PROFILE", True):
        return None
    cfg = getattr(model, "config", model)
    mt = getattr(cfg, "model_type", None)
    if mt is None and isinstance(cfg, dict):
        mt = cfg.get("model_type")
    provider = _SCORE_PROFILES.get(mt) if mt else None
    if provider is None:
        return None
    try:
        return provider(model, prompt_cache)
    except Exception:
        return None


def _walk_caches(prompt_cache):
    """Yield leaf caches, flattening CacheList-shaped entries."""
    for entry in prompt_cache or ():
        for c in getattr(entry, "caches", None) or (entry,):
            yield c


def _cache_is_quantized(c) -> bool:
    # gmlx pooling caches expose is_quantized; mlx-lm/vlm quantized caches
    # carry a bits attribute.
    return (bool(getattr(c, "is_quantized", False))
            or getattr(c, "bits", None) is not None)


def build_score_profile(*, profile,
                        kernels_armed: Callable[[], bool] | None = None,
                        require_cache: type | tuple | None = None,
                        disarm_cache: type | tuple | None = None,
                        allow_quantized_pools: bool = False) -> Callable:
    """Provider factory composing the common opt-out predicates, so a
    per-arch provider is a declarative call instead of a bespoke cache walk.
    Returns closure(model, prompt_cache) -> profile | None:

    - kernels_armed(): live gate, checked first (re-resolved per chunk, so a
      runtime kernel disable reverts to dense decay on the next chunk).
    - disarm_cache: any instance (e.g. a batched cache class) -> None.
    - require_cache: at least one instance must be present; each must have an
      int offset (non-int = batched) and, unless allow_quantized_pools, be
      unquantized. Any OTHER cache with a bits attr (quantized KV) -> None.
    - profile: the ScoreTransientProfile, or a callable returning one
      (callable form re-resolves live state such as arch base-step envs).

    Pass dual-origin mlx-lm/vlm classes as cache_compat.cache_types(...)
    tuples; gmlx-native classes are single-origin and safe directly."""
    def _provider(model, prompt_cache):
        matched = 0
        if kernels_armed is not None and not kernels_armed():
            return None
        for c in _walk_caches(prompt_cache):
            if disarm_cache is not None and isinstance(c, disarm_cache):
                return None
            if require_cache is not None and isinstance(c, require_cache):
                if not isinstance(getattr(c, "offset", 0), int):
                    return None
                if not allow_quantized_pools and _cache_is_quantized(c):
                    return None
                matched += 1
            elif getattr(c, "bits", None) is not None:
                return None
        if require_cache is not None and not matched:
            return None
        return profile() if callable(profile) else profile

    return _provider


def _enabled() -> bool:
    return env_bool("GMLX_PREFILL_DECAY", True)


def _env_cap_bytes() -> float | None:
    # Re-read per call (bench harnesses flip it); absolute authority.
    env = os.environ.get("GMLX_PREFILL_SCORE_CAP_GB")
    if env:
        try:
            return max(0.1, float(env)) * 1e9
        except ValueError:
            pass
    return None


def _cap_bytes() -> float:
    # Explicit env wins; the device-derived default is probed once.
    env = _env_cap_bytes()
    if env is not None:
        return env
    global _WS_CAP_BYTES
    if _WS_CAP_BYTES is None:
        try:
            ws = mx.device_info()["max_recommended_working_set_size"]
            _WS_CAP_BYTES = max(2e9, 0.05 * float(ws))
        except Exception:
            _WS_CAP_BYTES = 4e9
    return _WS_CAP_BYTES


# MLX buffer-cache limit as configured by the server (bytes), when one is in
# force. Doubles as the churn-safe certificate for the headroom-raised body
# cap: a held transient above the recyclable cache limit churns the OS
# allocator every chunk, which is exactly why small decayed chunks beat full
# ones on materializing archs -- so the raise clamps to it and never
# activates without it.
_NOTED_CACHE_LIMIT: float | None = None


def note_cache_limit(nbytes: float | None) -> None:
    """Server hook: record the byte value passed to mx.set_cache_limit
    (None clears)."""
    global _NOTED_CACHE_LIMIT
    _NOTED_CACHE_LIMIT = None if nbytes is None else float(nbytes)


def _body_cap_bytes() -> float:
    """Body transient cap. Env keeps absolute authority; with
    GMLX_PREFILL_HEADROOM_CAP=1 AND a noted cache limit, the legacy 5% cap is
    raised to min(half the live free working set, the cache limit) -- big
    boxes stop over-decaying, held transients stay recyclable. Without both
    conditions (or on probe failure) the legacy cap is returned unchanged."""
    env = _env_cap_bytes()
    if env is not None:
        return env
    legacy = _cap_bytes()
    if (not env_bool("GMLX_PREFILL_HEADROOM_CAP", False)
            or _NOTED_CACHE_LIMIT is None):
        return legacy
    room = _headroom_bytes()
    if room is None:
        return legacy
    cap = max(legacy, min(_HEADROOM_FRACTION * room, _NOTED_CACHE_LIMIT))
    if cap > legacy and env_bool("GMLX_PREFILL_DECAY_LOG", False) \
            and "headroom-cap" not in _PROFILE_LOGGED:
        _PROFILE_LOGGED.add("headroom-cap")
        print(f"[prefill-decay] headroom cap active: {cap / 1e9:.2f} GB "
              f"(legacy {legacy / 1e9:.2f})", flush=True)
    return cap


def score_heads(obj) -> int:
    """Attention head count that sizes the [heads, step, S] score transient.
    Accepts a model or a config; falls back conservatively."""
    cfg = getattr(obj, "config", obj)
    for c in (cfg, getattr(cfg, "text_config", None)):
        n = getattr(c, "num_attention_heads", None)
        if n:
            return int(n)
    return 32


def kv_depth(prompt_cache) -> int:
    """Current KV depth = max cache offset (0 when nothing is cached)."""
    d = 0
    for c in prompt_cache or ():
        off = getattr(c, "offset", None)
        if off is None:
            continue
        try:
            d = max(d, int(off))
        except (TypeError, ValueError):
            continue
    return d


def decayed_step(base: int, depth: int, heads: int,
                 cap: float | None = None,
                 profile: ScoreTransientProfile | None = None) -> int:
    """Largest halving tier of base whose score transient at this depth fits
    the cap. A base at or below the floor passes through untouched (an
    explicit small PREFILL_STEP_SIZE stays authoritative). Idempotent:
    re-decaying a decayed step returns it unchanged. A profile swaps the
    dense (heads, 2 bytes, full depth) transient model for the arch's real
    one; without a profile the fit test is unchanged."""
    if not base or base <= 0:
        return base
    min_step = max(1, env_int("GMLX_PREFILL_MIN_STEP", 256))
    if cap is None:
        cap = _body_cap_bytes()
    if profile is not None:
        h, bpe, div = profile.heads, profile.bytes_per_elem, profile.depth_divisor
    else:
        h, bpe, div = heads, 2, 1
    step = int(base)
    while step > min_step and h * step * (depth + step) * bpe > cap * div:
        step //= 2
    return step


_UNTRACKED_WEIGHTS = 0.0
_HEADROOM_FRACTION = 0.5


def note_untracked_weights(nbytes: float) -> None:
    """Loader hook: bytes wired at inference but invisible to
    mx.get_active_memory (zero-copy mmap weights). Accumulates across loads
    (target model + drafter)."""
    global _UNTRACKED_WEIGHTS
    _UNTRACKED_WEIGHTS += float(nbytes)


def _headroom_bytes() -> float | None:
    """Estimated live free working set: recommended working set minus
    zero-copy weights minus MLX-tracked allocations. The buffer cache counts
    as free (the allocator evicts it under pressure). Sampled fresh per call,
    never memoized."""
    try:
        ws = float(mx.device_info()["max_recommended_working_set_size"])
        active = float(mx.get_active_memory())
    except Exception:
        return None
    return ws - _UNTRACKED_WEIGHTS - active


def _seed_cap_bytes() -> float:
    # Explicit env wins; otherwise size the seed cap from live headroom.
    # The seed runs once per request at worst-case residency (post-prefill),
    # so no fixed fraction of the box can both pin a small model's seed at
    # full chunks and stay inside a near-capacity model's remaining
    # gigabytes. Floor at the body cap: the seed never decays harder than
    # the prefill body.
    env = os.environ.get("GMLX_MTP_SEED_SCORE_CAP_GB")
    if env:
        try:
            return max(0.1, float(env)) * 1e9
        except ValueError:
            pass
    room = _headroom_bytes()
    if room is None:
        return _body_cap_bytes()
    return max(_body_cap_bytes(), _HEADROOM_FRACTION * room)


def decayed_seed_step(base: int, depth: int, heads: int,
                      profile: ScoreTransientProfile | None = None) -> int:
    """Depth-decayed chunk for the MTP teacher-force head seed. The seed is a
    single block (churn cost ~1/n_layers of the prefill body's) and larger
    seed chunks measurably raise draft acceptance and decode throughput, so
    its cap defaults to half the live free working set rather than the body
    cap: full chunks whenever memory allows, decay only near capacity.
    Honors the kill switch, which the body-side wrappers gate at install."""
    if not _enabled():
        return base
    step = decayed_step(base, depth, heads, cap=_seed_cap_bytes(),
                        profile=profile)
    if step != base and env_bool("GMLX_PREFILL_DECAY_LOG", False):
        print(f"[prefill-decay] seed depth {depth}: step {base} -> {step}",
              flush=True)
    return step


# Stock serve base (mlx_vlm DEFAULT_PREFILL_STEP_SIZE); a batch base equal to
# this with no PREFILL_STEP_SIZE env means "defaulted", the only state an
# arch-default base_step may override.
_STOCK_BASE = 2048


def decayed_for_batch(batch) -> int | None:
    """Depth-decayed prefill step for a PromptProcessingBatch-shaped object
    (None passes through: caller falls back to one-shot)."""
    base = batch.prefill_step_size
    if not base or not _enabled():
        return base
    profile = resolve_score_profile(batch.model, batch.prompt_cache)
    if (profile is not None and profile.base_step
            and base == _STOCK_BASE
            and "PREFILL_STEP_SIZE" not in os.environ):
        base = int(profile.base_step)
    depth = kv_depth(batch.prompt_cache)
    step = decayed_step(base, depth, score_heads(batch.model),
                        profile=profile)
    if env_bool("GMLX_PREFILL_DECAY_LOG", False):
        if profile is not None and "profile" not in _PROFILE_LOGGED:
            _PROFILE_LOGGED.add("profile")
            print(f"[prefill-decay] score profile active: {profile}",
                  flush=True)
        if step != base:
            print(f"[prefill-decay] depth {depth}: step {base} -> {step}",
                  flush=True)
    return step


def install_prefill_decay() -> bool:
    """Wrap PromptProcessingBatch.prompt_step and needs_processing so both
    consult the depth-decayed step. Idempotent; composes with the MTP
    prompt_step override in either install order (decay is idempotent, and
    the MTP body resolves its own step via decayed_for_batch)."""
    if not _enabled():
        return False
    from mlx_vlm.generate.ar import PromptProcessingBatch

    if getattr(PromptProcessingBatch, _FLAG, False):
        return True

    orig_step = PromptProcessingBatch.prompt_step
    orig_needs = PromptProcessingBatch.needs_processing

    def _with_decayed_step(self, orig):
        base = self.prefill_step_size
        step = decayed_for_batch(self)
        if step == base:
            return orig(self)
        self.prefill_step_size = step
        try:
            return orig(self)
        finally:
            self.prefill_step_size = base

    def _decay_prompt_step(self):
        return _with_decayed_step(self, orig_step)

    def _decay_needs_processing(self):
        return _with_decayed_step(self, orig_needs)

    PromptProcessingBatch.prompt_step = _decay_prompt_step
    PromptProcessingBatch.needs_processing = _decay_needs_processing
    setattr(PromptProcessingBatch, _FLAG, True)
    return True
