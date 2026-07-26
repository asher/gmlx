"""Upstream-bug fix: quantized-KV SDPA mask broadcast at B>1 with GQA.

Both mlx_lm.models.base and mlx_vlm.models.base carry the same
quantized_scaled_dot_product_attention: when n_repeats > 1 it reshapes
queries to the grouped 5D layout (B, n_kv_heads, n_repeats, L, D), so the
score tensor is 5D -- but an array mask is applied as-is. A batched
left-pad mask (B, 1, 1, kv) left-pads to (1, B, 1, 1, kv) under broadcast,
its batch dim lands in the n_kv_heads slot, and the mx.where raises a
broadcast error. Every batched (B>1) masked GQA call with a quantized KV
cache crashes; B=1 works because a leading 1 broadcasts anywhere, which is
why single-stream kv4/kv8 measurements never saw it.

The fix inserts one axis: a 4D array mask becomes (B, 1, 1, L, kv) before
the wrapped call whenever the call is grouped. Masks without a real batch
dim ((1, 1, L, kv), 2D, "causal", None) are unchanged in effect -- the
expand is a no-op under broadcasting -- so the wrapper applies it to every
4D array mask rather than sniffing shapes.

Install is idempotent; GMLX_QSDPA_MASK_FIX=0 disables. Patched at each
base module's symbol (their scaled_dot_product_attention dispatchers read
the module global at call time).
"""
from __future__ import annotations

import mlx.core as mx

from .envflags import env_bool

_installed = False

_MODULES = ("mlx_lm.models.base", "mlx_vlm.models.base")


def _make_fixed(orig):
    def _masked_grouped_qsdpa(queries, q_keys, q_values, scale, mask,
                              group_size: int = 64, bits: int = 8):
        if (
            isinstance(mask, mx.array)
            and mask.ndim == 4
            and queries.ndim == 4
            and queries.shape[1] != q_keys[0].shape[-3]
        ):
            mask = mask[:, None]
        return orig(queries, q_keys, q_values, scale, mask,
                    group_size=group_size, bits=bits)

    _masked_grouped_qsdpa._gmlx_orig = orig
    _masked_grouped_qsdpa._gmlx_qsdpa_fix = True
    return _masked_grouped_qsdpa


def install_quantized_sdpa_mask_fix() -> bool:
    """Fix the 5D-scores vs 4D-batch-mask broadcast crash in upstream
    quantized SDPA (both base modules). Idempotent; no-op when
    GMLX_QSDPA_MASK_FIX=0. Returns True if the patch is active."""
    global _installed
    if not env_bool("GMLX_QSDPA_MASK_FIX", True):
        return False
    if _installed:
        return True
    import importlib

    patched = 0
    for name in _MODULES:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        cur = getattr(mod, "quantized_scaled_dot_product_attention", None)
        if cur is None:
            continue
        if getattr(cur, "_gmlx_qsdpa_fix", False):
            patched += 1
            continue
        mod.quantized_scaled_dot_product_attention = _make_fixed(cur)
        patched += 1
    if not patched:
        return False
    _installed = True
    return True
