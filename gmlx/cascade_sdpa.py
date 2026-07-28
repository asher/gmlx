"""Shared-prefix (cascade) decode attention for concurrent serving.

When B concurrent requests share a rendered prefix (same system prompt or
context restored from the APC), the batch cache holds B byte-identical
copies of that prefix's K/V and a plain batched decode step reads all B of
them every layer. The fused mlx_kquant.sdpa_decode_gqa_cascade kernel reads
the prefix ONCE for the whole batch (one matrix-unit row-tile walk serves
every row) and each row's private suffix per row, dropping attention traffic
per step from B*(P+S) to P + B*S. Measured per call vs the per-row path:
1.6-4.2x at P 14k-32k, B 4-16 (lab results/cascade/design-probe.md).

Two pieces, both in this module:

* A stamp at the duplication point. make_warm_batch_kv_cache_multi is where
  the APC materializes per-row prefix copies into one BatchKVCache; when
  every row's pick resolves to the same block chain (object identity on
  APCBlock entries -- concurrent hits on one cached prefix return the same
  blocks) or the same exact-cache snapshot, the shared token length P is
  stamped on each layer cache as ``_gmlx_cascade``. Rows may extend past a
  common chain: the stamp records the COMMON prefix only. The stamp's
  invariant is "keys[b, :, L_b : L_b + P] is byte-identical across rows",
  with L_b read live from cache.left_padding -- filter()'s uniform
  left-shift preserves it, so retirement keeps the stamp; extend() admits a
  row with an arbitrary prefix, so an instance-level extend wrapper drops
  the stamp on injection.

* A decode route. Module-global scaled_dot_product_attention seams (the
  gemma4_batched_sdpa pattern) claim stamped B>1 qL==1 calls and issue the
  fused op on zero-copy views of the live cache buffer: k_shared is row 0's
  prefix copy, the private slab starts at C0 = min_b(L_b + P), and
  starts[b] = L_b + P - C0 masks each row's own prefix bytes inside the
  slab. Anything else falls through to the wrapped original, and any kernel
  rejection lands on stock via try/except -- the route can only decline,
  never break a step.

Install is idempotent; GMLX_CASCADE_SDPA=0 disables both pieces;
GMLX_CASCADE_MIN_P (default 1024) sets the smallest shared prefix worth
routing. Quantized-KV caches (kv_bits) are never claimed -- the fused op is
fp16/bf16 only today.
"""
from __future__ import annotations

import mlx.core as mx

from .envflags import env_bool, env_int

_installed_route = False
_installed_stamp = False
_CLAIMS = [0]
_STAMPS = [0]

_MODULES = (
    ("mlx_lm.models", "llama"),
    ("mlx_lm.models", "gemma4_text"),
    ("mlx_vlm.models.gemma4", "language"),
)

_HD_OK = (64, 128, 256, 512)


def claims() -> int:
    """Decode calls claimed by the cascade route since import."""
    return _CLAIMS[0]


def stamps() -> int:
    """Warm batches stamped with a shared prefix since import."""
    return _STAMPS[0]


def _pads(cache):
    """Host-side left padding, memoized on the array object (one sync per
    rebinding: injection/retirement/filter replace the array wholesale)."""
    lp = getattr(cache, "left_padding", None)
    if lp is None:
        return None
    cached = getattr(cache, "_gmlx_casc_pads", None)
    if cached is not None and cached[0] is lp:
        return cached[1]
    pads = [int(x) for x in lp.tolist()] if isinstance(lp, mx.array) else [
        int(x) for x in lp
    ]
    cache._gmlx_casc_pads = (lp, pads)
    return pads


def _starts_for(cache, pads, P):
    """int32 starts row vector for the private slab, memoized alongside the
    pads (both are functions of the left_padding array identity)."""
    cached = getattr(cache, "_gmlx_casc_starts", None)
    lp = cache.left_padding
    if cached is not None and cached[0] is lp and cached[1] == P:
        return cached[2], cached[3]
    c0 = min(pads) + P
    starts = mx.array([p + P - c0 for p in pads], dtype=mx.int32)
    cache._gmlx_casc_starts = (lp, P, starts, c0)
    return starts, c0


def _mask_claimable(mask):
    # Decode-width claims only: None, mx.fast's "causal" string, or a
    # boolean visibility mask of decode width. The route reproduces
    # left-pad visibility exactly (full shared region + per-row starts on
    # the private slab), so the array mask's content is redundant.
    if mask is None or (isinstance(mask, str) and mask == "causal"):
        return True
    return isinstance(mask, mx.array) and mask.ndim >= 2 and mask.shape[-2] == 1


def _claim(queries, keys, values, cache, scale, mask, sinks):
    """Cascade output for a claimable stamped decode call, else None."""
    info = getattr(cache, "_gmlx_cascade", None)
    if info is None:
        return None
    B, Hq = queries.shape[0], queries.shape[1]
    hd = queries.shape[-1]
    if not (
        sinks is None
        and isinstance(keys, mx.array)
        and queries.ndim == 4
        and B > 1
        and queries.shape[2] == 1
        and hd in _HD_OK
        and keys.shape[-1] == hd
        and values.shape[-1] == hd
        and keys.shape[0] == B
        and keys.shape[1] >= 1
        and Hq % keys.shape[1] == 0
        and queries.dtype in (mx.float16, mx.bfloat16)
        and keys.dtype == queries.dtype
        and not hasattr(cache, "bits")
        and getattr(cache, "_right_padding", None) is None
        and _mask_claimable(mask)
    ):
        return None
    gqa = Hq // keys.shape[1]
    if gqa > 16 or B * gqa > (32 if hd == 512 else 64):
        return None
    P = info["P"]
    if P < env_int("GMLX_CASCADE_MIN_P", 1024):
        return None
    pads = _pads(cache)
    if pads is None or len(pads) < B:
        return None
    kL = keys.shape[2]
    starts, c0 = _starts_for(cache, pads[:B], P)
    if kL - c0 < 1 or max(pads[:B]) + P > kL:
        return None
    l0 = pads[0]
    try:
        from mlx_kquant import sdpa_decode_gqa_cascade

        out = sdpa_decode_gqa_cascade(
            queries,
            keys[0:1, :, l0:l0 + P],
            values[0:1, :, l0:l0 + P],
            keys[:, :, c0:],
            values[:, :, c0:],
            scale,
            starts=starts,
        )
    except Exception:
        return None
    if _CLAIMS[0] == 0:
        import sys

        print(f"[cascade] claimed: B={B} P={P} hd={hd} gqa={gqa}",
              file=sys.stderr, flush=True)
    _CLAIMS[0] += 1
    return out


def _make_route(orig):
    def _cascade_sdpa(queries, keys, values, cache=None, scale=1.0,
                      mask=None, sinks=None):
        out = _claim(queries, keys, values, cache, scale, mask, sinks)
        if out is not None:
            return out
        return orig(queries, keys, values, cache=cache, scale=scale,
                    mask=mask, sinks=sinks)

    _cascade_sdpa._gmlx_orig = orig
    _cascade_sdpa._gmlx_cascade_route = True
    return _cascade_sdpa


def install_cascade_sdpa() -> bool:
    """Claim stamped shared-prefix batched decode at the module-global SDPA
    seams. Idempotent; no-op when GMLX_CASCADE_SDPA=0 or no seam module is
    importable. Returns True if the route is active."""
    global _installed_route
    if not env_bool("GMLX_CASCADE_SDPA", True):
        return False
    if _installed_route:
        return True
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
        if getattr(cur, "_gmlx_cascade_route", False):
            patched += 1
            continue
        mod.scaled_dot_product_attention = _make_route(cur)
        patched += 1
    if not patched:
        return False
    _installed_route = True
    return True


# ---------------------------------------------------------------------------
# Stamp: shared-prefix detection at the APC warm-batch duplication point
# ---------------------------------------------------------------------------


def _common_prefix_tokens(picks) -> int:
    """Shared token length across all picks, 0 when nothing is shared.

    Block mode: the longest common APCBlock chain by object identity (two
    rows that hit the same cached prefix get the same block objects from the
    manager). Exact mode: the full prefix when every row carries the same
    warm_cache snapshot object. Mixed or cold rows share nothing.
    """
    if len(picks) < 2 or any(p is None for p in picks):
        return 0
    if all(p.get("warm_cache") is not None for p in picks):
        first = picks[0]["warm_cache"]
        if all(p["warm_cache"] is first for p in picks[1:]) and len(
            {p["prefix_len"] for p in picks}
        ) == 1:
            return picks[0]["prefix_len"]
        return 0
    if any(p.get("matched_blocks") is None for p in picks):
        return 0
    chains = [p["matched_blocks"] for p in picks]
    shared = 0
    for blocks in zip(*chains):
        first = blocks[0]
        if any(b is not first for b in blocks[1:]):
            break
        shared += first.keys[0].shape[-2]
    return shared


def _drop_stamp_on_extend(cache):
    """Instance-level extend wrapper: an injected row need not share the
    prefix, so admission invalidates the stamp. filter() is left alone --
    its uniform left-shift preserves the stamp's invariant."""
    orig = cache.extend

    def _extend(other, _c=cache, _orig=orig):
        _c.__dict__.pop("_gmlx_cascade", None)
        return _orig(other)

    cache.extend = _extend


def _stamp_caches(caches, shared_tokens):
    for c in caches:
        c._gmlx_cascade = {"P": shared_tokens}
        _drop_stamp_on_extend(c)
    if _STAMPS[0] == 0:
        import sys

        print(f"[cascade] stamped warm batch: shared P={shared_tokens}",
              file=sys.stderr, flush=True)
    _STAMPS[0] += 1


def _make_stamped_warm_multi(orig):
    def _warm_multi(picks, num_layers):
        out = orig(picks, num_layers)
        try:
            caches, max_prefix = out
            if caches and max_prefix:
                shared = _common_prefix_tokens(picks)
                if shared > 0:
                    _stamp_caches(caches, shared)
        except Exception:
            pass
        return out

    _warm_multi._gmlx_orig = orig
    _warm_multi._gmlx_cascade_stamp = True
    return _warm_multi


def install_cascade_stamp() -> bool:
    """Stamp shared-prefix warm batches as they are built by
    mlx_vlm.apc.make_warm_batch_kv_cache_multi. Idempotent; no-op when
    GMLX_CASCADE_SDPA=0 or the apc module is unavailable."""
    global _installed_stamp
    if not env_bool("GMLX_CASCADE_SDPA", True):
        return False
    if _installed_stamp:
        return True
    try:
        from mlx_vlm import apc
    except ImportError:
        return False
    cur = getattr(apc, "make_warm_batch_kv_cache_multi", None)
    if cur is None:
        return False
    if not getattr(cur, "_gmlx_cascade_stamp", False):
        apc.make_warm_batch_kv_cache_multi = _make_stamped_warm_multi(cur)
    _installed_stamp = True
    return True
