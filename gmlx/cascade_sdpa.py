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
  stamped on each layer cache as ``_gmlx_cascade`` together with the block
  chain itself (strong refs). Rows may extend past a common chain: the
  stamp records the COMMON prefix only. The stamp's invariant is
  "keys[b, :, L_b : L_b + P] is byte-identical across rows", with L_b read
  live from cache.left_padding -- filter()'s uniform left-shift preserves
  it, so retirement keeps the stamp. extend() (admission of new rows into
  a live batch) re-derives the stamp as the longest common object-identity
  chain of both sides: same-prefix admission keeps cascading, a diverging
  or cold row clears it. B=1 warm batches are stamped too -- they never
  route (the claim gate needs B>1) but they carry the chain into the
  merges, which is exactly the steady serve flow (one live request, more
  arriving on the same prefix).

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


_DEBUGGED = [0]


def _debug_decline(reason, **kw):
    import os
    import sys

    if os.environ.get("GMLX_CASCADE_DEBUG") == "1" and _DEBUGGED[0] < 8:
        _DEBUGGED[0] += 1
        print(f"[cascade] decline: {reason} {kw}", file=sys.stderr, flush=True)


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
        _debug_decline(
            "shape/dtype/mask", B=B, qL=queries.shape[2], hd=hd,
            kshape=tuple(keys.shape), qdt=str(queries.dtype),
            kdt=str(keys.dtype), bits=hasattr(cache, "bits"),
            rp=getattr(cache, "_right_padding", None) is not None,
            mask=type(mask).__name__ if not isinstance(mask, str) else mask,
            mshape=tuple(mask.shape) if isinstance(mask, mx.array) else None,
            sinks=sinks is not None)
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
        _debug_decline("geometry", kL=kL, c0=c0, P=P, pads=pads[:B])
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
    except Exception as e:
        _debug_decline("kernel", err=str(e)[:120])
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


def _common_chain(picks):
    """(chain, widths) shared across all picks, ((), ()) when nothing is.

    Block mode: the longest common APCBlock chain by object identity (rows
    that hit the same cached prefix get the same block objects from the
    manager); a single warm pick shares its whole chain -- a B=1 stamp
    never routes, but it carries the chain into extend() merges when new
    same-prefix rows are admitted. Exact mode: the snapshot object stands
    in as a one-element chain. Cold rows share nothing.
    """
    if not picks or any(p is None for p in picks):
        return (), ()
    if all(p.get("warm_cache") is not None for p in picks):
        first = picks[0]["warm_cache"]
        if all(p["warm_cache"] is first for p in picks[1:]) and len(
            {p["prefix_len"] for p in picks}
        ) == 1:
            return (first,), (picks[0]["prefix_len"],)
        return (), ()
    if any(p.get("matched_blocks") is None for p in picks):
        return (), ()
    chain, widths = [], []
    for blocks in zip(*[p["matched_blocks"] for p in picks]):
        first = blocks[0]
        if any(b is not first for b in blocks[1:]):
            break
        chain.append(first)
        widths.append(first.keys[0].shape[-2])
    return tuple(chain), tuple(widths)


def _merge_stamps(a, b):
    """Stamp for a batch formed by extend(): the longest common
    object-identity prefix of the two chains, None when nothing survives."""
    if a is None or b is None:
        return None
    chain, widths = [], []
    for x, w, y in zip(a["chain"], a["widths"], b["chain"]):
        if x is not y:
            break
        chain.append(x)
        widths.append(w)
    P = sum(widths)
    if P <= 0:
        return None
    return {"P": P, "chain": tuple(chain), "widths": tuple(widths)}


def _merge_stamp_on_extend(cache):
    """Instance-level extend wrapper: admission re-derives the stamp as
    the common chain of both sides (an unstamped or cold side clears it).
    filter() is left alone -- its uniform left-shift preserves the stamp's
    invariant for surviving rows."""
    if getattr(cache.extend, "_gmlx_cascade_merge", False):
        return
    orig = cache.extend

    def _extend(other, _c=cache, _orig=orig):
        merged = _merge_stamps(
            _c.__dict__.get("_gmlx_cascade"),
            getattr(other, "_gmlx_cascade", None))
        out = _orig(other)
        if merged is None:
            _c.__dict__.pop("_gmlx_cascade", None)
        else:
            _c._gmlx_cascade = merged
        return out

    _extend._gmlx_cascade_merge = True
    cache.extend = _extend


def _stamp_caches(caches, chain, widths):
    P = sum(widths)
    for c in caches:
        c._gmlx_cascade = {"P": P, "chain": chain, "widths": widths}
        _merge_stamp_on_extend(c)
    if _STAMPS[0] == 0:
        import sys

        print(f"[cascade] stamped warm batch: shared P={P}",
              file=sys.stderr, flush=True)
    _STAMPS[0] += 1


def _make_stamped_warm_multi(orig):
    def _warm_multi(picks, num_layers):
        out = orig(picks, num_layers)
        try:
            caches, max_prefix = out
            if caches and max_prefix:
                chain, widths = _common_chain(picks)
                if sum(widths) > 0:
                    _stamp_caches(caches, chain, widths)
        except Exception:
            pass
        return out

    _warm_multi._gmlx_orig = orig
    _warm_multi._gmlx_cascade_stamp = True
    return _warm_multi


def _make_stamped_warm_single(orig):
    def _warm_single(matched_blocks):
        out = orig(matched_blocks)
        try:
            if out and matched_blocks:
                chain = tuple(matched_blocks)
                widths = tuple(b.keys[0].shape[-2] for b in chain)
                if sum(widths) > 0:
                    _stamp_caches(out, chain, widths)
        except Exception:
            pass
        return out

    _warm_single._gmlx_orig = orig
    _warm_single._gmlx_cascade_stamp = True
    return _warm_single


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
    single = getattr(apc, "make_warm_batch_kv_cache", None)
    if single is not None and not getattr(
        single, "_gmlx_cascade_stamp", False
    ):
        apc.make_warm_batch_kv_cache = _make_stamped_warm_single(single)
    _installed_stamp = True
    return True
