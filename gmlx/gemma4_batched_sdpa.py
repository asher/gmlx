"""Batched-decode row route for gemma-4 hd512 global layers (WS1b ungating).

Every kq attention route in attn_hd512 gates on q.shape[0] == 1, and stock
MLX has no fused kernel for head_dim 512 at any width -- so a batched decode
step pays a materialized matmul -> softmax -> matmul on every global layer.
Measured on gemma-4-31b Q6_K at d14k (in-process, ABBA): the fallback costs
3.4 ms/step at B=2 and 7.8 ms/step at B=4 (~3.4% / ~6.5% of the whole step),
all of it recoverable by running each row through the existing B=1 kernel
routes -- the extra launches hide in in-encoder concurrency.

This claims hd512 qL==1 B>1 calls at the gemma4 module-level SDPA seam (the
only seam where the cache is visible) and loops rows through
mx.fast.scaled_dot_product_attention, i.e. through attn_hd512's wrapper, so
each row picks sdpa_vector or sdpa_decode_gqa by its own depth. Left
padding is honored by slicing each padded row's K/V to its visible tail
([pad, L) -- BatchKVCache.make_mask encodes exactly this at decode width;
right padding is transient inside update_and_fetch and never live here).
The pad list is host-read once per left_padding rebinding (identity-keyed
memo, the qwen35_verify_fold._pads pattern), never per step.

Tail slices keep the head-dim stride of the parent buffer; the kq kernels
already consume step-padded cache views in production, and a kernel that
rejects the shape raises inside attn_hd512's try/except and lands that row
on stock -- visible in route_counts(), which certs must check.

Install is idempotent; GMLX_G4_BATCHED_SDPA=0 disables; requires the
attn_hd512 wrapper to be installed (otherwise per-row calls would hit the
stock materialized path row by row, paying launches for nothing).
"""
from __future__ import annotations

import mlx.core as mx

from . import attn_hd512
from .envflags import env_bool

_installed = False
_orig = None
_CLAIMS = [0]


def claims() -> int:
    """Calls claimed by the row route since import (test/cert hook)."""
    return _CLAIMS[0]


def _pads(cache):
    """Host-side left padding, memoized on the array object (one sync per
    rebinding: injection/retirement/filter replace the array wholesale)."""
    lp = getattr(cache, "left_padding", None)
    if lp is None:
        return None
    cached = getattr(cache, "_gmlx_g4_pads", None)
    if cached is not None and cached[0] is lp:
        return cached[1]
    pads = [int(x) for x in lp.tolist()] if isinstance(lp, mx.array) else [
        int(x) for x in lp
    ]
    cache._gmlx_g4_pads = (lp, pads)
    return pads


def _mask_claimable(mask):
    if mask is None or (isinstance(mask, str) and mask == "causal"):
        return True
    # decode-width array mask: pure left-pad visibility (the qL>1 vision
    # overlay never reaches decode width), which the tail slice reproduces
    return isinstance(mask, mx.array) and mask.ndim >= 2 and mask.shape[-2] == 1


def _batched_rows_sdpa(queries, keys, values, cache=None, scale=1.0,
                       mask=None, sinks=None):
    if (
        sinks is None
        and isinstance(keys, mx.array)
        and queries.ndim == 4
        and queries.shape[0] > 1
        and queries.shape[2] == 1
        and queries.shape[-1] == 512
        and values.shape[-1] == 512
        and keys.shape[-1] == 512
        and keys.shape[1] >= 1
        and queries.shape[1] % keys.shape[1] == 0
        and not hasattr(cache, "bits")
        and getattr(cache, "_right_padding", None) is None
        and keys.shape[2] >= attn_hd512._MIN_KV
        and _mask_claimable(mask)
    ):
        pads = _pads(cache)
        B = queries.shape[0]
        if pads is None:
            pads = [0] * B
        if len(pads) >= B and max(pads[:B]) < keys.shape[2]:
            _CLAIMS[0] += 1
            outs = []
            for i in range(B):
                p = pads[i]
                if p <= 0:
                    ki, vi = keys[i:i + 1], values[i:i + 1]
                else:
                    ki = keys[i:i + 1, :, p:, :]
                    vi = values[i:i + 1, :, p:, :]
                outs.append(mx.fast.scaled_dot_product_attention(
                    queries[i:i + 1], ki, vi, scale=scale, mask=None))
            return mx.concatenate(outs, axis=0)
    return _orig(queries, keys, values, cache=cache, scale=scale, mask=mask,
                 sinks=sinks)


def install_gemma4_batched_sdpa() -> bool:
    """Route gemma4 hd512 batched decode through per-row B=1 kernel calls.
    Idempotent; no-op when GMLX_G4_BATCHED_SDPA=0, when the gemma4 module is
    unavailable, or when the hd512 wrapper is not installed. Returns True if
    the patch is active."""
    global _installed, _orig
    if not env_bool("GMLX_G4_BATCHED_SDPA", True):
        return False
    if _installed:
        return True
    if not attn_hd512._installed:
        return False
    try:
        from mlx_vlm.models.gemma4 import language as g4
    except ImportError as e:
        print(f"[g4-batched-sdpa] disabled: gemma4 module unavailable ({e})",
              flush=True)
        return False

    cur = g4.scaled_dot_product_attention
    _orig = getattr(cur, "_gmlx_orig", cur)
    _batched_rows_sdpa._gmlx_orig = _orig
    g4.scaled_dot_product_attention = _batched_rows_sdpa
    _installed = True
    return True
