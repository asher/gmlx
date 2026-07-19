"""Batch drafter contract, adapter, and load-time validation.

Formalizes the ~22 touchpoints the speculative engine (speculative.py,
spec_helpers.py) consumes on a drafter object. Two implementations exist:

- QwenMTPDrafter (native-head, owns KV, full batch support) -- satisfies
  the full contract natively.
- Gemma4AssistantDraftModel (mlx-vlm foreign, shared-KV, no batch methods)
  -- wrapped in DrafterAdapter at load time for the reset shim.

This module covers the drafter axis only (~1/3 of new-model fluidity).
It does not cover target-side coupling (S=0 guard, prefix_cache type set,
gated-delta rollback).

How to adapt a new mlx-vlm drafter
-----------------------------------
1. Subclass from mlx-vlm's drafter or implement from scratch.
2. Ensure required members exist:
   - config.block_size (int)
   - accept_lens (list -- engine appends directly)
   - reset(target_model, left_padding=None) -> list
   - set_shared_kv(shared_kv, kv_offset, position=, kv_valid_len=,
     left_padding=) -> None
   - draft_block(last_bonus, hidden, cache, block_size, sampler,
     token_dtype, greedy=) -> mx.array
   - bind(target_model) -> self
3. Wrap with DrafterAdapter(drafter) if the drafter's reset() does not
   accept left_padding. The adapter is a pure delegator -- inner methods
   always take precedence; missing optional members raise AttributeError,
   which the engine's getattr/hasattr guards handle correctly.
4. Call validate_drafter(drafter) at load time to catch missing required
   members before they crash mid-batch.
"""
from __future__ import annotations

import inspect
from typing import Any, Protocol, runtime_checkable

import mlx.core as mx


# 1. BatchDrafterProtocol -- the contract as typed documentation

@runtime_checkable
class BatchDrafterProtocol(Protocol):
    """Structural contract for speculative drafters.

    isinstance checks verify method names exist, not signatures.
    validate_drafter() does deeper checks to compensate.

    Required members (crash if missing -- directly accessed by the engine):
        config          -- must have .block_size (int)
        accept_lens     -- list, engine appends directly
        reset           -- (target_model, left_padding=None) -> list
        set_shared_kv   -- (shared_kv, kv_offset, ...) -> None
        draft_block     -- (last_bonus, hidden, cache, ...) -> mx.array
        bind            -- (target_model) -> self

    Guarded-optional (engine uses getattr with safe defaults):
        supports_greedy_draft_argmax    -- default False
        prefill_from_target_hidden      -- default None (skip)
        accept_verified_tokens          -- default None (B=1 skip)
        accept_verified_tokens_batch    -- default None (B>1 skip)
        uses_shared_kv                  -- default True
        prefer_requested_block_size     -- default False
        inject_rows                     -- default None (skip)
        filter_batch                    -- default None (skip)
        draft_eval_state                -- default None -> sampler_state_attrs
        sampler_state_attrs             -- default ("_seed_token",)
        draft_lens                      -- hasattr guard
        config.runtime_block_size       -- default None
        cap_at_configured_depth         -- default False
        _native_block_size              -- default configured_block_total

    Load-path only (not runtime):
        sanitize        -- weight remapping at load
        make_cache      -- internal to reset()
        model.embed_tokens -- gemma4 ordered-embeddings dequant only
    """

    config: Any
    accept_lens: list[float]

    def reset(self, target_model: Any, left_padding: list[int] | None = None) -> list:
        ...

    def set_shared_kv(
        self,
        shared_kv_states: dict,
        kv_offset: Any,
        position: Any = None,
        kv_valid_len: Any = None,
        left_padding: Any = None,
    ) -> None:
        ...

    def draft_block(
        self,
        last_bonus: mx.array,
        hidden: mx.array,
        cache: list,
        block_size: int,
        sampler: Any,
        token_dtype: Any = mx.int32,
        greedy: bool = False,
    ) -> mx.array:
        ...

    def bind(self, target_model: Any) -> "BatchDrafterProtocol":
        ...


# 2. DrafterAdapter -- pure delegator, per-instance

def _check_accepts_left_padding(inner: Any) -> bool:
    """True if inner.reset accepts a left_padding kwarg.

    Determined via inspect.signature at construction time -- not
    try/except TypeError, which would mask body errors inside reset
    and double-execute side effects.
    """
    try:
        sig = inspect.signature(inner.reset)
        params = sig.parameters
        if "left_padding" in params:
            return True
        return any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
    except (ValueError, TypeError):
        # co_varnames fallback for C extensions or uninspectable methods.
        # Can false-positive if left_padding is a local, not a param.
        code = getattr(inner.reset, "__code__", None)
        if code and "left_padding" in code.co_varnames:
            return True
        return False


class DrafterAdapter:
    """Per-instance adapter for foreign (mlx-vlm) drafters.

    Pure delegator: reset() absorbs left_padding, bind() re-wraps,
    everything else delegates to the inner drafter. If the inner
    lacks an attribute, AttributeError propagates -- the engine's
    own getattr/hasattr guards handle absence correctly.

    Phase 2 adds no-op defaults paired with engine guard removal.

    Transparency assumptions: the wrapper assumes the drafter is used
    only via named method calls and attribute reads, and never has
    attributes set on it directly. A future drafter.foo = x would set
    foo on the adapter (not the inner), and drafter(...) would fail
    (__call__ is not delegated by __getattr__). Both are non-issues
    for the current engine.
    """

    def __init__(self, inner: Any):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(
            self, "_accepts_left_padding", _check_accepts_left_padding(inner)
        )

    def reset(self, target_model: Any, left_padding: list[int] | None = None) -> list:
        inner = object.__getattribute__(self, "_inner")
        if object.__getattribute__(self, "_accepts_left_padding"):
            return inner.reset(target_model, left_padding=left_padding)
        return inner.reset(target_model)

    def bind(self, target_model: Any) -> "DrafterAdapter":
        inner = object.__getattribute__(self, "_inner")
        inner.bind(target_model)
        return self

    def __getattr__(self, name: str) -> Any:
        inner = object.__getattribute__(self, "_inner")
        return getattr(inner, name)


# 3. validate_drafter -- load-time crash prevention

def validate_drafter(drafter: Any) -> None:
    """Check required drafter members exist at load time.

    Raises RuntimeError with a clear message on failure -- crash at
    load, not mid-batch. Prevents the bug class from def3d87 (gemma
    reset missing left_padding).

    This validates crash-safety, not semantic correctness. A passing
    validation does not prove batch no-ops are correct -- that
    requires the B>1 parity test.
    """
    errors: list[str] = []
    name = type(drafter).__name__
    inner = drafter
    if isinstance(drafter, DrafterAdapter):
        inner = object.__getattribute__(drafter, "_inner")
        name = f"DrafterAdapter({type(inner).__name__})"

    if not hasattr(drafter, "config"):
        errors.append("missing config")
    else:
        try:
            int(drafter.config.block_size)
        except (AttributeError, TypeError, ValueError):
            errors.append("config.block_size must be int-castable")

    if not hasattr(drafter, "accept_lens"):
        errors.append("missing accept_lens (must be a list)")
    elif not isinstance(drafter.accept_lens, list):
        errors.append(
            f"accept_lens must be a list, got {type(drafter.accept_lens).__name__}"
        )

    for method in ("draft_block", "set_shared_kv", "bind"):
        if not callable(getattr(drafter, method, None)):
            errors.append(f"missing or non-callable: {method}")

    if not callable(getattr(drafter, "reset", None)):
        errors.append("missing or non-callable: reset")
    elif not isinstance(drafter, DrafterAdapter):
        if not _check_accepts_left_padding(drafter):
            errors.append(
                "reset() must accept a left_padding kwarg "
                "(or wrap with DrafterAdapter)"
            )

    if errors:
        raise RuntimeError(
            f"Drafter validation failed for {name}:\n  "
            + "\n  ".join(errors)
        )
