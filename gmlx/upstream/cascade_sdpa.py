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

* A stamp at batch formation. PromptProcessingBatch is where every row's
  prompt tokens and the freshly built per-layer caches coexist in matching
  order -- for cold, warm, and mixed batches alike -- so the stamp derives
  the shared prefix from TOKEN IDS (packed bytes, memcmp compare): rows
  whose token prefixes match hold interchangeable KV for that span. The
  common token length P is stamped on each layer cache as
  ``_gmlx_cascade`` with the per-row token bytes. The invariant is
  "keys[b, :, L_b : L_b + P] is interchangeable across rows", with L_b
  read live from cache.left_padding -- filter()'s uniform left-shift
  preserves it, so retirement keeps the stamp. extend() (admission into a
  live batch) re-derives the stamp from both sides' tokens: same-prefix
  admission keeps cascading, a diverging row clears it; the _extend_cache
  merge lift carries stamps across B=1-to-batch conversion. No APC mode
  is required, so hybrid models (exact-snapshot APC) and APC-off serving
  cascade too. B=1 batches are stamped -- inert until rows join.

* A decode route. Module-global scaled_dot_product_attention seams (the
  gemma4_batched_sdpa pattern) claim stamped B>1 calls at decode width
  (qL 1) and speculative-verify width (qL 2..8, end-aligned causal on the
  private slab) and issue the fused op on zero-copy views of the live
  cache buffer: k_shared is row 0's prefix copy, the private slab starts
  at C0 = min_b(L_b + P), and starts[b] = L_b + P - C0 masks each row's
  own prefix bytes inside the slab. Anything else falls through to the
  wrapped original, and any kernel rejection lands on stock via
  try/except -- the route can only decline, never break a step.

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
_SEEN_B = set()

_MODULES = (
    ("mlx_lm.models", "llama"),
    ("mlx_lm.models", "gemma4_text"),
    ("mlx_vlm.models.gemma4", "language"),
    # text qwen3.5/3.6 (dense + MoE): the attention call site lives in
    # qwen3_next; qwen3_5/qwen3_5_moe import its Attention class
    ("mlx_lm.models", "qwen3_next"),
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


def _mask_claimable(mask, qL):
    # Decode width (qL 1): None, mx.fast's "causal" string, or a boolean
    # visibility mask of decode width. Verify width (qL 2..8): "causal"
    # or an array mask of matching width -- its content is left-pad
    # visibility plus the end-aligned in-block causal, which the fused
    # op reproduces (full shared region + per-row starts + causal clamp
    # on the private slab). None at qL > 1 would mean an unmasked block:
    # never claim it.
    if isinstance(mask, str) and mask == "causal":
        return True
    if mask is None:
        return qL == 1
    return isinstance(mask, mx.array) and mask.ndim >= 2 and mask.shape[-2] == qL


_DEBUGGED = [0]


def _debug_decline(reason, **kw):
    import os
    import sys

    if os.environ.get("GMLX_CASCADE_DEBUG") == "1" and _DEBUGGED[0] < env_int(
            "GMLX_CASCADE_DEBUG_N", 8):
        _DEBUGGED[0] += 1
        print(f"[cascade] decline: {reason} {kw}", file=sys.stderr, flush=True)


def _claim(queries, keys, values, cache, scale, mask, sinks):
    """Cascade output for a claimable stamped decode call, else None."""
    info = getattr(cache, "_gmlx_cascade", None)
    if info is None:
        _debug_decline("no-stamp", B=queries.shape[0],
                       cache=type(cache).__name__)
        return None
    B, Hq = queries.shape[0], queries.shape[1]
    hd = queries.shape[-1]
    qL = queries.shape[2] if queries.ndim == 4 else 0
    # Quantized batch caches (BatchQuantizedKVCache, mlx affine bits 8 /
    # group 64) present k/v as (wire, scales, biases) tuples; the fused op
    # takes the wire directly and dequants at tile stage. Other quant
    # configs decline.
    quant = (
        isinstance(keys, (tuple, list))
        and len(keys) == 3
        and isinstance(values, (tuple, list))
        and len(values) == 3
        and getattr(cache, "bits", None) == 8
        and getattr(cache, "group_size", None) == 64
        and hd != 512
    )
    if not quant and isinstance(keys, (tuple, list)):
        _debug_decline("quant-config", bits=getattr(cache, "bits", None),
                       gs=getattr(cache, "group_size", None), hd=hd)
        return None
    ka = keys[0] if quant else keys
    va = values[0] if quant else values
    kv_last = hd // 4 if quant else hd
    dt_ok = (
        (ka.dtype == mx.uint32 and keys[1].dtype == queries.dtype)
        if quant
        else (ka.dtype == queries.dtype and not hasattr(cache, "bits"))
    )
    if not (
        sinks is None
        and isinstance(ka, mx.array)
        and queries.ndim == 4
        and B > 1
        and 1 <= qL <= 8
        and hd in _HD_OK
        and ka.shape[-1] == kv_last
        and va.shape[-1] == kv_last
        and ka.shape[0] == B
        and ka.shape[1] >= 1
        and Hq % ka.shape[1] == 0
        and queries.dtype in (mx.float16, mx.bfloat16)
        and dt_ok
        and getattr(cache, "_right_padding", None) is None
        and _mask_claimable(mask, qL)
    ):
        _debug_decline(
            "shape/dtype/mask", B=B, qL=queries.shape[2], hd=hd,
            kshape=tuple(ka.shape) if isinstance(ka, mx.array) else None,
            qdt=str(queries.dtype),
            kdt=str(ka.dtype) if isinstance(ka, mx.array) else None,
            bits=getattr(cache, "bits", None), quant=quant,
            rp=getattr(cache, "_right_padding", None) is not None,
            mask=type(mask).__name__ if not isinstance(mask, str) else mask,
            mshape=tuple(mask.shape) if isinstance(mask, mx.array) else None,
            sinks=sinks is not None)
        return None
    gqa = Hq // ka.shape[1]
    if gqa > 16 or B * gqa * qL > (32 if hd == 512 else 64):
        return None
    if qL > 1 and gqa * ((qL + 1) // 2) > 32:
        return None
    P = info["P"]
    if P < env_int("GMLX_CASCADE_MIN_P", 1024):
        return None
    pads = _pads(cache)
    if pads is None or len(pads) < B:
        return None
    kL = ka.shape[2]
    starts, c0 = _starts_for(cache, pads[:B], P)
    if kL - c0 < qL or max(pads[:B]) + P > kL - qL + 1:
        _debug_decline("geometry", kL=kL, c0=c0, P=P, pads=pads[:B])
        return None
    l0 = pads[0]
    try:
        from mlx_kquant import sdpa_decode_gqa_cascade

        if quant:
            kw, ks, kb = keys
            vw, vs, vb = values
            out = sdpa_decode_gqa_cascade(
                queries,
                kw[0:1, :, l0:l0 + P],
                vw[0:1, :, l0:l0 + P],
                kw[:, :, c0:],
                vw[:, :, c0:],
                scale,
                starts=starts,
                k_shared_scales=ks[0:1, :, l0:l0 + P],
                k_shared_biases=kb[0:1, :, l0:l0 + P],
                v_shared_scales=vs[0:1, :, l0:l0 + P],
                v_shared_biases=vb[0:1, :, l0:l0 + P],
                k_priv_scales=ks[:, :, c0:],
                k_priv_biases=kb[:, :, c0:],
                v_priv_scales=vs[:, :, c0:],
                v_priv_biases=vb[:, :, c0:],
            )
        else:
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
    if (B, qL, quant) not in _SEEN_B:  # one line per (batch, query) width
        _SEEN_B.add((B, qL, quant))
        import sys

        print(f"[cascade] claimed: B={B} qL={qL} P={P} hd={hd} gqa={gqa}"
              + (" kv=q8" if quant else ""),
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
# Stamp v3: token-prefix currency (cold + warm rows alike)
# ---------------------------------------------------------------------------


# The stamp's currency is the row's PROMPT TOKEN IDS, packed to bytes.
# Rows whose token prefixes match hold interchangeable KV for that span
# (same tokens, same positions under left padding => same prefill math;
# byte-identical when chunking matches, and within GEMM reduction-order
# noise when the prefill tick resized chunks -- far below kv8 noise).
# One stamp point covers every formation path: PromptProcessingBatch
# builds caches and rows in matching order for cold, warm, and mixed
# batches. Merges compare token bytes, so cold rows cascade with warm
# ones and no APC mode is required at all.


def _row_bytes(ids):
    import array

    return array.array("i", [int(t) for t in ids]).tobytes()


def _lcp_rows(rows):
    """Common-prefix token count across packed rows (memcmp-speed)."""
    import os

    if not rows or any(not r for r in rows):
        return 0
    return len(os.path.commonprefix(list(rows))) // 4


def _merge_stamps(a, b):
    """Stamp for a batch formed by extend(): self rows then other rows
    (BatchKVCache.extend row order). None when nothing survives."""
    if a is None or b is None:
        return None
    P = _lcp_rows([a["tok"][0][: 4 * a["P"]], b["tok"][0][: 4 * b["P"]]])
    if P <= 0:
        return None
    return {"P": P, "tok": a["tok"] + b["tok"]}


def _merge_stamp_on_extend(cache):
    """Instance-level extend wrapper: admission re-derives the stamp from
    both sides' token bytes (an unstamped side clears it)."""
    fn = getattr(cache, "extend", None)
    if fn is None or getattr(fn, "_gmlx_cascade_merge", False):
        return
    orig = fn

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


def _mask_stamp_on_filter(cache):
    """Instance-level filter wrapper: retired rows leave the stamp; the
    shared prefix stays valid for every surviving row."""
    fn = getattr(cache, "filter", None)
    if fn is None or getattr(fn, "_gmlx_cascade_filter", False):
        return
    orig = fn

    def _filter(keep, *a, _c=cache, _orig=orig, **kw):
        info = _c.__dict__.get("_gmlx_cascade")
        out = _orig(keep, *a, **kw)
        if info is not None:
            ks = keep.tolist() if hasattr(keep, "tolist") else list(keep)
            tok = tuple(t for t, k in zip(info["tok"], ks) if k)
            if tok:
                _c._gmlx_cascade = {"P": info["P"], "tok": tok}
            else:
                _c.__dict__.pop("_gmlx_cascade", None)
        return out

    _filter._gmlx_cascade_filter = True
    cache.filter = _filter


def carry_stamp(src, dst):
    """Copy a live cascade stamp from one cache object to another.

    Owned prefill paths that rebuild per-layer caches through class
    merge (fresh objects) call this so the formation stamp survives;
    row order is unchanged there, so the stamp stays valid as-is."""
    st = src.__dict__.get("_gmlx_cascade") if src is not None else None
    if st is None or dst is None:
        return
    dst._gmlx_cascade = st
    _merge_stamp_on_extend(dst)
    _mask_stamp_on_filter(dst)


def _stamp_caches(caches, rows):
    """Stamp every layer cache with the batch's shared token prefix.
    rows: per-row packed token bytes, cache row order. No-op when the
    common prefix is empty."""
    P = _lcp_rows(rows)
    if P <= 0:
        return False
    tok = tuple(rows)
    for c in caches:
        c._gmlx_cascade = {"P": P, "tok": tok}
        _merge_stamp_on_extend(c)
        _mask_stamp_on_filter(c)
    if _STAMPS[0] == 0:
        import sys

        print(f"[cascade] stamped batch: shared P={P} rows={len(rows)}",
              file=sys.stderr, flush=True)
    _STAMPS[0] += 1
    return True


def _rows_from_prompt_batch(pb):
    """Per-row full prompt tokens (packed) for a just-built prompt batch.

    APC on: _apc_meta carries full_input_ids for every row (warm rows'
    _input_ids hold only the suffix, so meta is the ONLY safe source --
    bail if any row lacks it rather than mix currencies). APC off: every
    row is cold, so the untouched _input_ids ARE the full prompts; strip
    each row's pads. Must run before prompt_step consumes _input_ids.
    """
    # PPB normalizes absent meta to [] (kv-quantized serving disables APC
    # entirely) -- an empty list means every row is cold, NOT "meta without
    # ids"; fall through to the _input_ids currency.
    meta = getattr(pb, "_apc_meta", None)
    if meta:
        rows = []
        for m in meta:
            ids = (m or {}).get("full_input_ids")
            if not ids:
                return None
            rows.append(_row_bytes(ids))
        return rows
    ii = getattr(pb, "_input_ids", None)
    lps = getattr(pb, "_left_padding_per_row", None)
    if ii is None or lps is None or getattr(ii, "ndim", 0) != 2:
        return None
    rps = getattr(pb, "_right_pad_per_row", None) or [0] * len(lps)
    L = ii.shape[1]
    rows = []
    for i, lp in enumerate(lps):
        rp = rps[i] if i < len(rps) else 0
        rows.append(_row_bytes(ii[i, lp:L - rp].tolist()))
    return rows


def _make_stamped_ppb_init(orig):
    def _init(self, *a, **kw):
        orig(self, *a, **kw)
        try:
            caches = getattr(self, "prompt_cache", None)
            if caches:
                rows = _rows_from_prompt_batch(self)
                if rows:
                    _stamp_caches(caches, rows)
                else:
                    _debug_decline(
                        "stamp-no-rows",
                        meta=getattr(self, "_apc_meta", None) is not None,
                        cache=type(caches[0]).__name__)
            else:
                _debug_decline("stamp-no-caches")
        except Exception:
            if env_bool("GMLX_CASCADE_DEBUG", False):
                import traceback

                traceback.print_exc()

    _init._gmlx_orig = orig
    _init._gmlx_cascade_stamp = True
    return _init


def _make_stamped_extend_cache(orig):
    """_extend_cache lifts B=1 caches via merge([ca]), which drops
    instance attrs; carry the stamps across the whole merge."""

    def _extend_cache(a, b):
        stamps = [
            (getattr(ca, "_gmlx_cascade", None),
             getattr(cb, "_gmlx_cascade", None))
            for ca, cb in zip(a, b)
        ]
        out = orig(a, b)
        for oc, (sa, sb) in zip(out, stamps):
            merged = _merge_stamps(sa, sb)
            if merged is None:
                oc.__dict__.pop("_gmlx_cascade", None)
            else:
                oc._gmlx_cascade = merged
                _merge_stamp_on_extend(oc)
                _mask_stamp_on_filter(oc)
        return out

    _extend_cache._gmlx_orig = orig
    _extend_cache._gmlx_cascade_stamp = True
    return _extend_cache


def install_cascade_stamp() -> bool:
    """Stamp shared-prefix batches at formation (PromptProcessingBatch)
    and keep stamps alive across admission merges (_extend_cache lift).
    Token-currency: works for cold, warm, and mixed batches on every
    cache mode, including hybrid models whose APC is exact-snapshot
    only. Idempotent; no-op when GMLX_CASCADE_SDPA=0."""
    global _installed_stamp
    if not env_bool("GMLX_CASCADE_SDPA", True):
        return False
    if _installed_stamp:
        return True
    try:
        from mlx_vlm.generate import ar
    except ImportError:
        return False
    ppb = getattr(ar, "PromptProcessingBatch", None)
    ext = getattr(ar, "_extend_cache", None)
    if ppb is None or ext is None:
        return False
    if not getattr(ppb.__init__, "_gmlx_cascade_stamp", False):
        ppb.__init__ = _make_stamped_ppb_init(ppb.__init__)
    if not getattr(ext, "_gmlx_cascade_stamp", False):
        ar._extend_cache = _make_stamped_extend_cache(ext)
    _installed_stamp = True
    return True
