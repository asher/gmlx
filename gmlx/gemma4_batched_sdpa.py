"""Batched-decode row route for gemma-4 hd512 global layers (WS1b ungating).

Every kq attention route in attn_hd512 gates on q.shape[0] == 1, and stock
MLX has no fused kernel for head_dim 512 at any width -- so a batched decode
step pays a materialized matmul -> softmax -> matmul on every global layer.
Measured on gemma-4-31b Q6_K at d14k (in-process, ABBA): the fallback costs
3.4 ms/step at B=2 and 7.8 ms/step at B=4 (~3.4% / ~6.5% of the whole step),
all of it recoverable by running each row through the existing B=1 kernel
routes -- the extra launches hide in in-encoder concurrency.

Gemma-4 text attention lives in two upstream modules, one per load path:
text-only GGUFs build from mlx_lm.models.gemma4_text (wrapped in
gmlx.vlm_text_only), multimodal loads from mlx_vlm.models.gemma4's
language module. Both import a module-level scaled_dot_product_attention and
call it with the cache in hand -- the only seam where left padding is
visible -- so the route installs on both, each chained to that module's own
original. This claims hd512 B>1 calls at decode width (qL==1) and verify
width (qL 2..8, the MTP block range; the block occupies the last qL key
positions, so a per-row end-aligned "causal" mask after the tail slice is
exact). Left padding is honored by restricting each row to its visible key
tail [pad, L) -- BatchKVCache.make_mask encodes exactly this at decode
width; right padding is transient inside update_and_fetch and never live
here. The pad list is host-read once per left_padding rebinding
(identity-keyed memo, the qwen35_verify_fold._pads pattern), never per step.

Decode width goes out as ONE batched kq.sdpa_decode_gqa call with per-row
key start offsets (the kernel's `starts` buffer; pad bytes are skipped, not
staged) when the resident mlx-kquant ships it -- measured 1.14-1.38x over
the per-row loop at the 31b global shape from d4k to d49k, B 2-16: the
per-row grid (Hkv, 1, splits) starves the GPU, the batched grid fills it.
GMLX_G4_SDPA_STARTS=0 falls back to the loop; an older mlx-kquant falls
back silently. Verify width (qL 2..8) stays on the per-row loop through
mx.fast.scaled_dot_product_attention, i.e. through attn_hd512's wrapper, so
each row picks the B=1 kernel route (fa_verify, verify_gemm, sdpa_vector)
by its own shape -- the batched kernel measured 0.5-0.6x against fa_verify
at block width, so one call is not the right shape there.

KV-shared consumer layers arrive with cache=None but share the producers'
per-layer-type mask object (both upstream modules build one mask per layer
type); a small identity-keyed relay carries the producer's pads to the
consumer. A cold miss falls through to stock rather than guessing pads.

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

try:
    import mlx_kquant as _kq
except Exception:  # pragma: no cover - mlx_kquant always present in practice
    _kq = None

_installed = False
_CLAIMS = [0]
_ONECALL = [0]

_MODULES = (
    ("mlx_lm.models", "gemma4_text"),
    ("mlx_vlm.models.gemma4", "language"),
)


def _probe_starts() -> bool:
    # Kernel capability: sdpa_decode_gqa grew the per-row `starts` buffer
    # after 0.3.6. The op validator runs at build time, so probing costs no
    # GPU work (nothing is evaled); an older wheel raises TypeError.
    if _kq is None or not hasattr(_kq, "sdpa_decode_gqa"):
        return False
    try:
        q = mx.zeros((2, 1, 1, 512), dtype=mx.float16)
        kv = mx.zeros((2, 1, 8, 512), dtype=mx.float16)
        _kq.sdpa_decode_gqa(q, kv, kv, 1.0,
                            starts=mx.zeros((2,), dtype=mx.int32))
        return True
    except Exception:
        return False


_HAS_STARTS = _probe_starts()


def claims() -> int:
    """Calls claimed by the route since import (test/cert hook)."""
    return _CLAIMS[0]


def onecall_claims() -> int:
    """Decode-width claims served by the single batched starts-kernel call
    (subset of claims(); test/cert engagement hook)."""
    return _ONECALL[0]


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


_STARTS_RING: list = []  # (tuple(pads), int32 device array), bounded


def _starts_for(pads, B):
    """Device starts buffer for the batched kernel call, memoized by value
    (pads lists are tiny; the ring avoids a per-layer-per-step upload)."""
    key = tuple(pads[:B])
    for k2, arr in _STARTS_RING:
        if k2 == key:
            return arr
    arr = mx.array(list(key), dtype=mx.int32)
    _STARTS_RING.append((key, arr))
    if len(_STARTS_RING) > 8:
        _STARTS_RING.pop(0)
    return arr


_MASK_PADS: list = []  # ring of (mask_obj, pads): producer -> consumer relay


def _remember_mask_pads(mask, pads):
    _MASK_PADS.append((mask, pads))
    if len(_MASK_PADS) > 4:
        _MASK_PADS.pop(0)


def _pads_for_mask(mask):
    for m, p in reversed(_MASK_PADS):
        if m is mask:
            return p
    return None


def _mask_claimable(mask, qL):
    if isinstance(mask, str) and mask == "causal":
        return True
    if mask is None:
        # None at qL>1 would mean unmasked block attention; never claim it
        return qL == 1
    # decode-width (shape[-2]==1) or verify-width (shape[-2]==qL) array
    # mask: pure left-pad visibility plus in-block causal, which the tail
    # slice plus per-row causal reproduces. The gemma vision overlay only
    # appears on image prefills (>=256 tokens/image), never at qL<=8 -- and
    # the text-only load path has no overlay at all.
    return (isinstance(mask, mx.array) and mask.ndim >= 2
            and mask.shape[-2] == qL)


def _claim(queries, keys, values, cache, scale, mask, sinks):
    """Batched-row output for a claimable call, else None (fall through)."""
    if not (
        sinks is None
        and isinstance(keys, mx.array)
        and queries.ndim == 4
        and queries.shape[0] > 1
        and 1 <= queries.shape[2] <= 8
        and queries.shape[-1] == 512
        and values.shape[-1] == 512
        and keys.shape[-1] == 512
        and keys.shape[1] >= 1
        and queries.shape[1] % keys.shape[1] == 0
        and not hasattr(cache, "bits")
        and getattr(cache, "_right_padding", None) is None
        and keys.shape[2] >= attn_hd512._MIN_KV
        and _mask_claimable(mask, queries.shape[2])
    ):
        return None
    pads = _pads(cache)
    B = queries.shape[0]
    if pads is None:
        if isinstance(mask, mx.array):
            # kv-shared consumer layers arrive with cache=None but share
            # the producers' mask object: relay the pads by identity. A
            # cold miss (no producer claimed this mask) falls through --
            # the mask may encode padding this route cannot reconstruct.
            pads = _pads_for_mask(mask)
            if pads is None:
                return None
        else:
            pads = [0] * B
    elif isinstance(mask, mx.array):
        _remember_mask_pads(mask, pads)
    qL = queries.shape[2]
    if len(pads) < B or max(pads[:B]) > keys.shape[2] - qL:
        return None
    _CLAIMS[0] += 1
    # Decode width: one batched kernel call, per-row starts. The explicit
    # dtype/gqa gates mirror the op validator so an ineligible call walks
    # the row loop without paying an exception; anything else the op
    # rejects at build time lands there through the except.
    if (
        qL == 1
        and _HAS_STARTS
        and queries.dtype in (mx.float16, mx.bfloat16)
        and keys.dtype == queries.dtype
        and values.dtype == queries.dtype
        and queries.shape[1] // keys.shape[1] <= 16
        and env_bool("GMLX_G4_SDPA_STARTS", True)
    ):
        try:
            starts = (_starts_for(pads, B) if max(pads[:B]) > 0 else None)
            out = _kq.sdpa_decode_gqa(queries, keys, values, float(scale),
                                      starts=starts)
            _ONECALL[0] += 1
            return out
        except Exception:
            pass  # op-build rejection -> per-row loop
    # qL==1 needs no mask after the tail slice; verify blocks (qL 2..8)
    # occupy the LAST qL key positions, which is exactly mx.fast's
    # end-aligned "causal" semantics on the sliced row.
    row_mask = None if qL == 1 else "causal"
    outs = []
    for i in range(B):
        p = pads[i]
        if p <= 0:
            ki, vi = keys[i:i + 1], values[i:i + 1]
        else:
            ki = keys[i:i + 1, :, p:, :]
            vi = values[i:i + 1, :, p:, :]
        outs.append(mx.fast.scaled_dot_product_attention(
            queries[i:i + 1], ki, vi, scale=scale, mask=row_mask))
    return mx.concatenate(outs, axis=0)


def _make_route(orig):
    def _batched_rows_sdpa(queries, keys, values, cache=None, scale=1.0,
                           mask=None, sinks=None):
        out = _claim(queries, keys, values, cache, scale, mask, sinks)
        if out is not None:
            return out
        return orig(queries, keys, values, cache=cache, scale=scale,
                    mask=mask, sinks=sinks)

    _batched_rows_sdpa._gmlx_orig = orig
    _batched_rows_sdpa._gmlx_g4_route = True
    return _batched_rows_sdpa


def install_gemma4_batched_sdpa() -> bool:
    """Route gemma4 hd512 batched decode through per-row B=1 kernel calls,
    at both upstream seams (mlx_lm gemma4_text for text-only loads, mlx_vlm
    gemma4 language for multimodal). Idempotent; no-op when
    GMLX_G4_BATCHED_SDPA=0, when no gemma4 module is importable, or when the
    hd512 wrapper is not installed. Returns True if the patch is active."""
    global _installed
    if not env_bool("GMLX_G4_BATCHED_SDPA", True):
        return False
    if _installed:
        return True
    if not attn_hd512._installed:
        return False
    import importlib

    patched = 0
    for pkg, name in _MODULES:
        try:
            mod = importlib.import_module(f"{pkg}.{name}")
        except ImportError:
            continue
        cur = getattr(mod, "scaled_dot_product_attention", None)
        if cur is None:
            continue
        if getattr(cur, "_gmlx_g4_route", False):
            patched += 1
            continue
        mod.scaled_dot_product_attention = _make_route(cur)
        patched += 1
    if not patched:
        print("[g4-batched-sdpa] disabled: no gemma4 module available",
              flush=True)
        return False
    _installed = True
    return True
