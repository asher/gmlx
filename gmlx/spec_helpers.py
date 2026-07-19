"""Owned speculative-decoding round helpers.

The round orchestration the owned engine needs, vendored from mlx-vlm's
speculative.{common,mtp} so the engine does not depend on mlx-vlm's private
speculative API (underscore-prefixed, unstable across versions). The drafter
model classes and the model/cache runtime (mlx_vlm.models.cache) are still
consumed from mlx-vlm by design -- only the round logic is owned here.

Logic is a faithful copy (acceptance must stay token-identical to the validated
path); keep it in sync when mlx-vlm's round changes in a way we want to track.
"""
from dataclasses import dataclass
from typing import Any
from collections.abc import Callable

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.models import cache

from .cache_compat import cache_types

# Shared generation stream (the drafter model pins no stream of its own, so the
# round and verify forward run here; cross-stream deps are handled by MLX events).
generation_stream = mx.new_thread_local_stream(mx.default_device())


# --- draft/target sampler RNG coupling -------------------------------------

def _copy_rng_state() -> list[mx.array]:
    return [mx.array(state) for state in mx.random.state]


def _restore_rng_state(state: list[mx.array]) -> None:
    for i, value in enumerate(state):
        mx.random.state[i] = value


def _append_arrays(value: Any, arrays: list[mx.array]) -> None:
    if isinstance(value, mx.array):
        arrays.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            _append_arrays(item, arrays)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _append_arrays(item, arrays)


def _draft_sampler_state_arrays(draft_model: nn.Module) -> list[mx.array]:
    state_fn = getattr(draft_model, "draft_eval_state", None)
    if callable(state_fn):
        arrays: list[mx.array] = []
        _append_arrays(state_fn(), arrays)
        return arrays

    attrs = getattr(draft_model, "sampler_state_attrs", ("_seed_token",))
    if isinstance(attrs, str):
        attrs = (attrs,)

    arrays = []
    for attr in attrs:
        _append_arrays(getattr(draft_model, attr, None), arrays)
    return arrays


class _SpeculativeSamplerRNG:
    """Keep target and drafter sampler RNG streams independent."""

    def __init__(self, draft_model: nn.Module, *, enabled: bool):
        self.draft_model = draft_model
        self.enabled = bool(enabled)
        self._target_rng_state = _copy_rng_state() if self.enabled else None
        self._draft_rng_state = _copy_rng_state() if self.enabled else None

    def draft_call(self, fn: Callable, *args, **kwargs):
        if not self.enabled:
            result = fn(*args, **kwargs)
            arrays = []
            _append_arrays(result, arrays)
            arrays.extend(_draft_sampler_state_arrays(self.draft_model))
            if arrays:
                mx.async_eval(*arrays)
            return result

        self._target_rng_state = _copy_rng_state()
        _restore_rng_state(self._draft_rng_state)
        result = fn(*args, **kwargs)

        arrays = _draft_sampler_state_arrays(self.draft_model)
        arrays.extend(mx.random.state)
        if arrays:
            mx.async_eval(*arrays)

        self._draft_rng_state = _copy_rng_state()
        _restore_rng_state(self._target_rng_state)
        return result

    def draft_tokens(self, fn: Callable, *args, **kwargs):
        if not self.enabled:
            result = fn(*args, **kwargs)
            arrays: list[mx.array] = []
            _append_arrays(result, arrays)
            if arrays:
                mx.async_eval(*arrays)
            return result

        self._target_rng_state = _copy_rng_state()
        _restore_rng_state(self._draft_rng_state)
        result = fn(*args, **kwargs)

        arrays = []
        _append_arrays(result, arrays)
        arrays.extend(_draft_sampler_state_arrays(self.draft_model))
        arrays.extend(mx.random.state)
        if arrays:
            mx.async_eval(*arrays)

        self._draft_rng_state = _copy_rng_state()
        _restore_rng_state(self._target_rng_state)
        return result

    def target_sampled(self, *, sync_draft: bool = False) -> None:
        if self.enabled:
            self._target_rng_state = _copy_rng_state()
            if sync_draft:
                self._draft_rng_state = _copy_rng_state()


# --- round bookkeeping + block sizing --------------------------------------

def _record_speculative_round(
    draft_model: nn.Module, accepted: float, draft_count: int
) -> None:
    draft_model.accept_lens.append(accepted)
    if hasattr(draft_model, "draft_lens"):
        draft_model.draft_lens.append(int(draft_count))


def _dflash_block_total(draft_model: nn.Module, draft_block_size: int | None) -> int:
    if draft_block_size is not None:
        return int(draft_block_size)

    configured = int(draft_model.config.block_size)
    runtime = getattr(draft_model.config, "runtime_block_size", None)
    if runtime is None:
        return configured
    return min(configured, max(1, int(runtime)))


def _effective_mtp_block_size(
    requested_block_total: int,
    configured_block_total: int,
    accept_lens: list[int],
    remaining_budget: int,
) -> int:
    """Choose the MTP block size for the next round.

    Treat user-provided block sizes above the assistant's configured depth as a
    ceiling. Larger tails are useful only if the prefix reaches the configured
    depth often enough; otherwise each round pays extra autoregressive drafter
    forwards for tokens that cannot be accepted.
    """
    block_total = min(requested_block_total, remaining_budget)
    configured_block_total = min(configured_block_total, block_total)
    if block_total <= configured_block_total or configured_block_total <= 1:
        return block_total

    if len(accept_lens) < 8:
        return configured_block_total

    recent = accept_lens[-32:]
    configured_draft_count = configured_block_total - 1
    configured_prefix_hits = sum(
        1 for accepted in recent if accepted >= configured_draft_count
    )
    configured_prefix_hit_rate = configured_prefix_hits / len(recent)
    if configured_prefix_hit_rate < 0.65:
        return configured_block_total

    return block_total


def _mtp_next_block_size(
    draft_model: nn.Module,
    requested_block_total: int,
    configured_block_total: int,
    remaining_budget: int,
) -> int:
    budget = min(requested_block_total, remaining_budget)
    if getattr(draft_model, "cap_at_configured_depth", False):
        native = getattr(draft_model, "_native_block_size", configured_block_total)
        return min(budget, native)
    if getattr(draft_model, "prefer_requested_block_size", False):
        return budget
    return _effective_mtp_block_size(
        requested_block_total,
        configured_block_total,
        draft_model.accept_lens,
        remaining_budget,
    )


# --- target cache + shared-KV plumbing -------------------------------------

def _buffer_mtp_target_cache(
    prompt_cache: list[Any],
    draft_model: nn.Module,
    draft_block_size: int | None,
) -> None:
    configured = int(getattr(draft_model.config, "block_size", draft_block_size or 1))
    requested = int(draft_block_size or configured)
    buffer_size = max(32, min(128, max(configured, requested) * 8))

    def buffer_entry(entry):
        if isinstance(entry, cache_types("CacheList")):
            entry.caches = tuple(buffer_entry(child) for child in entry.caches)
            return entry
        if isinstance(entry, cache.BufferedRotatingKVCache):
            entry.buffer_size = max(entry.buffer_size, buffer_size)
        elif (
            isinstance(entry, cache_types("RotatingKVCache"))
            and getattr(entry, "keep", 0) == 0
        ):
            return cache.BufferedRotatingKVCache.from_cache(
                entry, buffer_size=buffer_size
            )
        return entry

    for idx, entry in enumerate(prompt_cache):
        prompt_cache[idx] = buffer_entry(entry)


def _mtp_cache_offset(prompt_cache: list[Any]) -> Any:
    for cache_entry in prompt_cache:
        offset = getattr(cache_entry, "offset", None)
        if offset is not None:
            return offset
    for cache_entry in prompt_cache:
        idx = getattr(cache_entry, "_idx", None)
        if idx is not None:
            return idx
    return 0


def _mtp_cache_offset_max(prompt_cache: list[Any]) -> int:
    offset = _mtp_cache_offset(prompt_cache)
    return int(offset.max().item()) if isinstance(offset, mx.array) else int(offset)


def _mtp_draft_position(kv_valid_len: Any) -> Any:
    if isinstance(kv_valid_len, int):
        return max(kv_valid_len - 1, 0)
    if isinstance(kv_valid_len, mx.array):
        return mx.maximum(kv_valid_len.astype(mx.int32) - 1, 0)
    return mx.maximum(mx.array(kv_valid_len, dtype=mx.int32) - 1, 0)


def _slice_shared_kv_after_reject(shared_kv_states: dict, rejected: int) -> dict:
    if rejected <= 0:
        return shared_kv_states

    next_shared_kv = {}
    for k, kv in shared_kv_states.items():
        K, V = kv
        # rejected >= 1 here (early return above), so valid < K.shape[-2] always.
        valid = K.shape[-2] - rejected
        if valid <= 0:
            next_shared_kv[k] = (K[..., :1, :], V[..., :1, :])
        else:
            next_shared_kv[k] = (K[..., :valid, :], V[..., :valid, :])
    return next_shared_kv


# --- verify forward over the draft block -----------------------------------

# One-shot note naming the verify branch the target resolves to (hook vs plain
# forward), opt in with GMLX_MTP_DEBUG=1. The branch decides the verify
# Attention shape (hook path: target_verify per-token/folded; plain forward:
# one standard causal call) and whether gdn_states reach rollback -- exactly
# the facts a perf/parity investigation needs pinned first.
_VERIFY_BRANCH_NOTED: set = set()


def _note_verify_branch(branch: str, lm) -> None:
    import os
    import sys
    if os.environ.get("GMLX_MTP_DEBUG", "") not in ("1", "true", "TRUE"):
        return
    key = (branch, type(lm).__name__)
    if key in _VERIFY_BRANCH_NOTED:
        return
    _VERIFY_BRANCH_NOTED.add(key)
    print(f"[mtp] verify branch: {branch} (lm={type(lm).__name__})",
          file=sys.stderr, flush=True)


@dataclass
class _MTPVerifyResult:
    hidden: mx.array
    shared_kv_states: dict
    target_tokens: mx.array | None = None
    gdn_states: list | None = None


def _mtp_draft_hidden(lm: nn.Module, hidden: mx.array) -> mx.array:
    prepare = getattr(lm, "speculative_draft_hidden", None)
    return prepare(hidden) if callable(prepare) else hidden


def _mtp_shared_kv_from_prompt_cache(lm: nn.Module, prompt_cache: list[Any]) -> dict:
    layers = getattr(getattr(lm, "model", None), "layers", [])
    if len(prompt_cache) != len(layers):
        return {}

    shared_kv_states = {}
    for layer, layer_cache in zip(layers, prompt_cache):
        if layer_cache is None or not hasattr(layer_cache, "state"):
            continue
        state = layer_cache.state
        if state is None or len(state) < 2:
            continue
        keys, values = state[:2]
        if keys is None or values is None:
            continue
        if (
            isinstance(layer_cache, cache_types("RotatingKVCache"))
            and not isinstance(layer_cache, cache.BufferedRotatingKVCache)
            and hasattr(layer_cache, "_temporal_order")
        ):
            keys = layer_cache._temporal_order(keys)
            values = layer_cache._temporal_order(values)
        shared_kv_states[layer.layer_type] = (keys, values)
    return shared_kv_states


def _mtp_verify_without_logits(
    lm: nn.Module,
    verify_input: mx.array,
    prompt_cache: list[Any],
) -> _MTPVerifyResult | None:
    verify_hidden = getattr(lm, "speculative_verify_hidden", None)
    if callable(verify_hidden):
        _note_verify_branch("hook:speculative_verify_hidden", lm)
        result = verify_hidden(verify_input, prompt_cache)
        if isinstance(result, tuple):
            if len(result) == 3:
                hidden, shared_kv_states, gdn_states = result
            elif len(result) == 2:
                hidden, shared_kv_states = result
                gdn_states = None
            else:
                raise ValueError(
                    "speculative_verify_hidden() must return "
                    "(hidden, shared_kv_states) or "
                    "(hidden, shared_kv_states, gdn_states)."
                )
        else:
            hidden = result
            shared_kv_states = {}
            gdn_states = None
        return _MTPVerifyResult(
            hidden=hidden,
            shared_kv_states=shared_kv_states or {},
            gdn_states=gdn_states,
        )

    layers = getattr(getattr(lm, "model", None), "layers", [])
    if len(prompt_cache) == len(layers):
        _note_verify_branch("inner-model-forward", lm)
        hidden = lm.model(verify_input, cache=prompt_cache, skip_final_norm=True)
        shared_kv_states = _mtp_shared_kv_from_prompt_cache(lm, prompt_cache)
        if shared_kv_states:
            return _MTPVerifyResult(hidden=hidden, shared_kv_states=shared_kv_states)

    shared_kv_sink: dict = {}
    hidden = lm.model(
        verify_input,
        cache=prompt_cache,
        shared_kv_sink=shared_kv_sink,
        skip_final_norm=True,
    )
    if not shared_kv_sink:
        return None
    return _MTPVerifyResult(hidden=hidden, shared_kv_states=shared_kv_sink)


def _mtp_verify_with_model_method(
    lm: nn.Module,
    verify_input: mx.array,
    prompt_cache: list[Any],
    sampler: Callable[[mx.array], mx.array],
) -> _MTPVerifyResult | None:
    verify_logits = getattr(lm, "speculative_verify_logits", None)
    if not callable(verify_logits):
        return None

    _note_verify_branch("hook:speculative_verify_logits", lm)
    result = verify_logits(verify_input, prompt_cache, sampler)
    if not isinstance(result, tuple) or len(result) != 4:
        raise ValueError(
            "speculative_verify_logits() must return "
            "(hidden, shared_kv_states, gdn_states, target_tokens)."
        )

    hidden, shared_kv_states, gdn_states, target_tokens = result
    return _MTPVerifyResult(
        hidden=hidden,
        shared_kv_states=shared_kv_states or {},
        target_tokens=target_tokens,
        gdn_states=gdn_states,
    )


def _mtp_verify_target(
    lm: nn.Module,
    verify_input: mx.array,
    prompt_cache: list[Any],
    sampler: Callable[[mx.array], mx.array],
    *,
    sample_target_tokens: bool = True,
) -> _MTPVerifyResult:
    if sample_target_tokens:
        argmax_from_hidden = getattr(lm, "speculative_argmax_from_hidden", None)
        if callable(argmax_from_hidden):
            result = _mtp_verify_without_logits(lm, verify_input, prompt_cache)
            if result is not None:
                target_tokens = argmax_from_hidden(result.hidden)
                if target_tokens is not None:
                    result.target_tokens = target_tokens
                    return result

        result = _mtp_verify_with_model_method(lm, verify_input, prompt_cache, sampler)
        if result is not None:
            return result

    if hasattr(lm, "speculative_logits_from_hidden"):
        result = _mtp_verify_without_logits(lm, verify_input, prompt_cache)
        if result is not None:
            return result

    _note_verify_branch("plain-forward", lm)
    verify_out = lm(
        verify_input,
        cache=prompt_cache,
        return_hidden=True,
        return_shared_kv=True,
    )
    return _MTPVerifyResult(
        hidden=verify_out.hidden_states[-1],
        shared_kv_states=verify_out.shared_kv_states,
        # Greedy callers pass sampler=None; argmax instead of calling None
        # (mirrors the hooked path in speculative.py).
        target_tokens=(mx.argmax(verify_out.logits, axis=-1)
                       if sampler is None else sampler(verify_out.logits)),
        gdn_states=verify_out.gdn_states,
    )
