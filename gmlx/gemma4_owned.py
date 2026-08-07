"""Owned mask building and attention for gemma4 text MTP targets.

Construction-time owned classes for ``mlx_vlm.models.gemma4.language``,
selected in ``loader._mtp_target_classes("gemma4_text")`` behind
``GMLX_GEMMA_OWNED`` (default on). The text-MTP selector builds the vlm
LanguageModel even for text-only GGUFs, so this is the class every
drafter-bearing gemma text checkpoint runs.

Scope is deliberately narrow: gemma4's model-level ``__call__`` carries
none of the qwen-style per-step taxes and is not patched today, so it is
inherited stock. The owned surface is exactly the patched one:

- ``_make_masks`` carries the gemma4_sync nosync semantics natively (int
  offsets compare host-side, array offsets skip the ``.item()`` probe).
  One mask object per layer type is preserved; the batched-sdpa pad
  relay keys on that identity.
- ``Attention.__call__`` carries the offset handling natively: int rope
  offsets pass to rope unwrapped, array offsets are snapshot-copied
  (``update_and_fetch`` advances ``cache.offset`` with an in-place ``+=``
  between the key rope and the query rope, so an aliased array offset
  rotates queries one position ahead of keys). The SDPA tail goes
  through ``_sdpa_dispatch``, which composes the gemma4_batched_sdpa
  row-route claim directly and falls back to the base implementation,
  so the module-global patch is not needed on owned calls.
- Attention ownership is an instance ``__class__`` rebind during owned
  model construction; DecoderLayer/Router/Experts/MLP stay stock
  classes so the fused-MoE swap eligibility keeps firing.

The gemma4_sync and gemma4_batched_sdpa installers stay in place for
stock-built trees (multimodal construction, ``GMLX_GEMMA_OWNED=0``):
subclass overrides shadow the base-class patches structurally, so both
regimes coexist in one process. Mirror drift is alarmed by
substitution-normalized source equality against the pinned upstream
bodies in ``tests/test_gemma4_owned.py``.
"""

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.models.base import (
    scaled_dot_product_attention as _base_sdpa,
)
from mlx_vlm.models.gemma4 import language as _G

from . import attn_hd512, gemma4_batched_sdpa
from .envflags import env_bool

_OWNED_MASK_CALLS = 0
_OWNED_SDPA_CALLS = 0


def owned_mask_call_count() -> int:
    """Engagement counter: owned _make_masks invocations this process."""
    return _OWNED_MASK_CALLS


def owned_sdpa_call_count() -> int:
    """Engagement counter: owned attention SDPA dispatches this process
    (one per owned attention forward, all routes)."""
    return _OWNED_SDPA_CALLS


def _cache_has_prefix(c) -> bool:
    off = getattr(c, "offset", 0)
    if isinstance(off, mx.array):
        # Batched cache: probing max(offset) is the host sync the owned
        # path removes. Assume a prefix; the array mask is correct at
        # offset 0.
        return True
    try:
        return int(off) > 0
    except (TypeError, ValueError):
        return True


def _sdpa_dispatch(queries, keys, values, cache=None, scale=1.0, mask=None):
    """Owned SDPA tail: the batched hd512 row-route claim, else base sdpa.

    Calls the row route's claim directly (shared claims() counter), so
    owned attention never needs the module-global patch; the env kill
    switch stays live per call, matching the patch's gate.
    """
    global _OWNED_SDPA_CALLS
    _OWNED_SDPA_CALLS += 1
    if type(keys).__name__ == "KVarNView":
        # Backstop: kvarn_unsupported declines the owned tree at setup
        # (this call site bypasses the module-attribute sweep). If an arm
        # is ever wanted here, route kvarn_attention(queries, keys.cache,
        # scale, mask) as kvarn_sdpa's wrapper does.
        raise RuntimeError(
            "[kvarn] gemma-4 owned attention reached a kvarn cache; "
            "the scheme should have been declined at setup"
        )
    if attn_hd512._installed and env_bool("GMLX_G4_BATCHED_SDPA", True):
        out = gemma4_batched_sdpa._claim(
            queries, keys, values, cache, scale, mask, None
        )
        if out is not None:
            return out
    return _base_sdpa(queries, keys, values, cache=cache, scale=scale, mask=mask)


def _owned_make_masks(self, h, cache, mm_token_type_ids: Optional[mx.array] = None):
    # Upstream body with the sliding-branch offset probe replaced by
    # _cache_has_prefix (no .item()); one mask object per layer type is
    # preserved (the batched-sdpa pad relay keys on that identity).
    mask = {}
    masks = []
    has_audio_tokens = (
        mm_token_type_ids is not None and int(mx.sum(mm_token_type_ids == 3).item()) > 0
    )
    has_visual_tokens = (
        mm_token_type_ids is not None
        and int(mx.sum((mm_token_type_ids == 1) | (mm_token_type_ids == 2)).item()) > 0
    )
    use_bidirectional_vision = (
        getattr(self.config, "use_bidirectional_attention", None) == "vision"
        and mm_token_type_ids is not None
        and has_visual_tokens
        and not has_audio_tokens
        and h.shape[1] > 1
    )
    for l, c in zip(self.layers, cache):
        if l.layer_type not in mask:
            if l.layer_type == "full_attention":
                return_array = (
                    use_bidirectional_vision
                    or getattr(c, "left_padding", None) is not None
                )
                mask["full_attention"] = _G.create_attention_mask(
                    h, c, return_array=return_array
                )
            elif l.layer_type == "sliding_attention":
                return_array = (
                    h.shape[1] > 1 and c is not None and _cache_has_prefix(c)
                ) or use_bidirectional_vision
                mask["sliding_attention"] = _G.create_attention_mask(
                    h, c, window_size=self.window_size, return_array=return_array
                )
            if (
                use_bidirectional_vision
                and isinstance(mask[l.layer_type], str)
                and mask[l.layer_type] == "causal"
            ):
                window = (
                    self.window_size if l.layer_type == "sliding_attention" else None
                )
                mask[l.layer_type] = _G.create_causal_mask(
                    h.shape[1], window_size=window
                )
            if use_bidirectional_vision and isinstance(mask[l.layer_type], mx.array):
                mask[l.layer_type] = self._apply_blockwise_bidirectional_overlay(
                    mask[l.layer_type],
                    mm_token_type_ids,
                )
        masks.append(mask[l.layer_type])
    return masks


def _owned_attention_call(
    self,
    x: mx.array,
    mask: Optional[mx.array] = None,
    cache: Optional[Any] = None,
    shared_kv: Optional[tuple] = None,
    offset: Optional[Any] = None,
) -> mx.array:
    # Upstream body with the offset wrap replaced (ints pass to rope
    # unwrapped; arrays are snapshot-copied against the in-place +=
    # advance inside update_and_fetch) and the SDPA tail routed through
    # the owned dispatch.
    B, L, _ = x.shape

    queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
    queries = self.q_norm(queries)

    if shared_kv is not None:
        keys, values = shared_kv
    else:
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

        if self.use_k_eq_v:
            values = keys
        else:
            values = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

        offset = cache.offset if cache is not None else 0
        if isinstance(offset, mx.array):
            offset = mx.array(offset)

        keys = self.k_norm(keys)
        keys = keys.transpose(0, 2, 1, 3)
        keys = self.rope(keys, offset=offset)

        values = self.v_norm(values)
        values = values.transpose(0, 2, 1, 3)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

    queries = queries.transpose(0, 2, 1, 3)
    queries = self.rope(queries, offset=offset)

    output = _sdpa_dispatch(
        queries, keys, values, cache=cache, scale=self.scale, mask=mask
    )
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

    return self.o_proj(output), (keys, values), offset


class OwnedGemma4Attention(_G.Attention):
    """Stock gemma4 attention with owned offset handling and SDPA tail.

    Installed by instance ``__class__`` rebind from
    ``OwnedGemma4TextModel.__init__``; adds no attributes.
    """

    __call__ = _owned_attention_call


class OwnedGemma4TextModel(_G.Gemma4TextModel):
    """Stock gemma4 text model with the owned mask builder and attention.

    The stock ``__init__`` builds the full stock module tree; only the
    attention instances are rebound to the owned class afterwards.
    ``__call__`` is inherited stock (unpatched upstream, no per-step
    taxes; nothing to retire by mirroring it).
    """

    def __init__(self, config, kv_shared_only: bool = False):
        super().__init__(config, kv_shared_only=kv_shared_only)
        for layer in self.layers:
            layer.self_attn.__class__ = OwnedGemma4Attention

    def _make_masks(self, h, cache, mm_token_type_ids=None):
        global _OWNED_MASK_CALLS
        _OWNED_MASK_CALLS += 1
        return _owned_make_masks(self, h, cache, mm_token_type_ids)


class OwnedGemma4LanguageModel(_G.LanguageModel):
    """Stock gemma4 LanguageModel over the owned text model.

    Mirrors the stock constructor body (construction-pair tested); every
    ``speculative_*`` hook, ``chunked_prefill_policy``,
    ``rollback_speculative_cache``, ``sanitize``, and ``make_cache``
    inherit stock.
    """

    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.model_type = config.model_type
        self.model = OwnedGemma4TextModel(config)
        self.final_logit_softcapping = getattr(config, "final_logit_softcapping", None)


def is_owned_language_model(model) -> bool:
    """True when the built tree is the owned gemma4 LanguageModel.

    Gates on what construction produced, never on config model_type: the
    multimodal build paths never consult the selector and stay stock.
    """
    return isinstance(model, OwnedGemma4LanguageModel)
