"""Lossy sparse (top-k page) decode attention for deep contexts.

At depth, decode attention reads the whole KV cache every step, but the
softmax mass concentrates on a small set of positions. This route keeps
an approximate index over the cache -- one mean key per page of
``tile_c`` tokens, maintained incrementally as blocks fill -- and each
step attends only the top-k scoring pages (plus the sink page and the
two most recent pages, always resident) through the fused
mlx_kquant.sdpa_decode_gqa_paged kernel. Attention cost becomes flat in
context depth: the walk sees k selected tokens, not S.

Measured on Dolphin3-Llama3.1-8B Q6_K at d32k (lab
results/sparse-attn/stage4-e2e.md): budget 2048 gives mean KLD 0.0079
vs full attention -- inside the quantization-noise acceptance band --
at 1.40x end-to-end decode throughput, selection overhead included.

This is LOSSY and therefore strictly opt-in: GMLX_SPARSE_ATTN=1 enables
the route, GMLX_SPARSE_K sets the kept-token budget (default 2048), and
GMLX_SPARSE_MIN_S (default 8192) keeps it off shallow contexts where
full attention is already cheap. Forced sink + recency residency is
load-bearing: without it approximate indexers fail catastrophically
(stage-2 finding), so it is not configurable.

Claims are limited to quality-gated architectures (llama-family full
attention today): the property being traded on -- top-k concentration
of decode attention -- is architectural, and it measurably does NOT
hold on SWA hybrids (gemma-4's global layers fail even the exact
oracle at k=2048). Within a validated arch, claims cover decode width
(qL==1) on fp16/bf16 caches, B=1 or batched.
Left-padded rows are handled end to end: pages fully inside a row's
pad region are biased out of selection, the row's sink page is the
first page holding its real tokens, and the kernel masks pad positions
inside selected boundary pages via per-row starts. Quantized-KV
caches, sinks, and speculative verify widths decline. Any kernel
rejection lands on stock via try/except: the route can only decline,
never break a step.
"""
from __future__ import annotations

import mlx.core as mx

from .envflags import env_bool, env_int

_installed = False
_CLAIMS = [0]
_SEEN_B = set()

_MODULES = (
    ("mlx_lm.models", "llama"),
    ("mlx_lm.models", "gemma4_text"),
    ("mlx_vlm.models.gemma4", "language"),
)

# Only architectures with a PASSED KLD gate are patched. gemma-4 is
# measured NOT top-k sparse: even the exact oracle at k=2048 sits ~15x
# above the acceptance band on its global layers (SWA hybrids aggregate
# diffusely there; lab results/sparse-attn/gemma-gate.md). Extend via
# GMLX_SPARSE_ARCHS=name[,name] only to run a new arch's quality gate.
_VALIDATED = ("llama",)

# page size must match the paged kernel's staging tile per head dim
_TILE_C = {64: 32, 128: 32, 256: 16, 512: 8}


def claims() -> int:
    """Decode calls claimed by the sparse route since import."""
    return _CLAIMS[0]


def _block_means(cache, keys, bl):
    """fp16 mean key per full page, maintained incrementally on the cache.

    Memo: [means [B,Hkv,nb,D], covered_blocks, last_S, B]. Appends cover
    newly completed pages only; any shrink or width change (trim, filter,
    admission merge) drops the memo entirely -- filter's left-shift moves
    absolute positions, so sliced reuse would be silently wrong.
    """
    B, Hkv, S, D = keys.shape
    nb = S // bl
    memo = getattr(cache, "_gmlx_sp_means", None)
    if memo is not None and (memo[3] != B or S < memo[2]):
        memo = None
    if memo is None:
        m = keys[:, :, : nb * bl].reshape(B, Hkv, nb, bl, D).mean(axis=-2)
        memo = [m, nb, S, B]
    elif memo[1] < nb:
        n0 = memo[1]
        m = keys[:, :, n0 * bl : nb * bl].reshape(
            B, Hkv, nb - n0, bl, D
        ).mean(axis=-2)
        memo = [mx.concatenate([memo[0], m], axis=2), nb, S, B]
    else:
        memo = [memo[0], memo[1], S, B]
    cache._gmlx_sp_means = memo
    return memo[0]


def _select_pages(queries, keys, cache, bl, nkeep, pads=None):
    """Top-nkeep page indices per kv head, int32 [B,Hkv,nkeep], sorted.

    Group-shared scoring (mean query over the GQA group against the page
    means, f32). Per row, the sink page (the first page holding the
    row's real tokens) and the last two pages -- including an unscored
    partial tail -- are forced resident via a score bias, and pages
    fully inside the row's left pad are biased out; the kept-page count
    stays fixed and argpartition-friendly.
    """
    B, Hq, _, D = queries.shape
    Hkv, S = keys.shape[1], keys.shape[2]
    rep = Hq // Hkv
    nb_total = (S + bl - 1) // bl
    means = _block_means(cache, keys, bl)
    qg = queries.reshape(B, Hkv, rep, 1, D).mean(axis=2).astype(mx.float32)
    bs = (qg @ means.astype(mx.float32).transpose(0, 1, 3, 2))[:, :, 0]
    nb_full = means.shape[2]
    if nb_total > nb_full:
        bs = mx.concatenate(
            [bs, mx.zeros((B, Hkv, nb_total - nb_full))], axis=-1
        )
    pos = mx.arange(nb_total)[None, None, :]
    pad = mx.array(pads if pads is not None else [0] * B).reshape(B, 1, 1)
    local = (pos >= nb_total - 2)
    sink = (pos == pad // bl)
    dead = ((pos + 1) * bl <= pad)  # page holds only this row's pad bytes
    bias = mx.where(sink | local, mx.array(1e9), mx.array(0.0)) + mx.where(
        dead, mx.array(-1e9), mx.array(0.0)
    )
    idx = mx.argpartition(-(bs + bias), nkeep - 1, axis=-1)[..., :nkeep]
    return mx.sort(idx, axis=-1).astype(mx.int32)


def _mask_claimable(mask):
    if mask is None or (isinstance(mask, str) and mask == "causal"):
        return True
    return isinstance(mask, mx.array) and mask.ndim >= 2 and mask.shape[-2] == 1


def _pads(cache, B):
    """Host-side left padding as a python list (zeros when absent), plus
    the int32 starts array for the kernel (None when unpadded). Memoized
    on the left_padding array identity."""
    lp = getattr(cache, "left_padding", None)
    if lp is None:
        return [0] * B, None
    cached = getattr(cache, "_gmlx_sp_pads", None)
    if cached is not None and cached[0] is lp:
        return cached[1], cached[2]
    vals = [int(x) for x in (lp.tolist() if isinstance(lp, mx.array) else lp)]
    starts = None
    if any(v != 0 for v in vals):
        starts = mx.array(vals, dtype=mx.int32)
    cache._gmlx_sp_pads = (lp, vals, starts)
    return vals, starts


def _claim(queries, keys, values, cache, scale, mask, sinks):
    """Sparse output for a claimable deep decode call, else None."""
    if cache is None or not isinstance(keys, mx.array):
        return None
    B, Hq = queries.shape[0], queries.shape[1]
    hd = queries.shape[-1]
    bl = _TILE_C.get(hd)
    if not (
        sinks is None
        and bl is not None
        and queries.ndim == 4
        and queries.shape[2] == 1
        and keys.shape[-1] == hd
        and values.shape[-1] == hd
        and keys.shape[0] == B
        and keys.shape[1] >= 1
        and Hq % keys.shape[1] == 0
        and Hq // keys.shape[1] <= 16
        and queries.dtype in (mx.float16, mx.bfloat16)
        and keys.dtype == queries.dtype
        and not hasattr(cache, "bits")
        and getattr(cache, "_right_padding", None) is None
        and _mask_claimable(mask)
    ):
        return None
    S = keys.shape[2]
    pads, starts = _pads(cache, B)
    if len(pads) < B:
        return None
    if S - max(pads[:B]) < env_int("GMLX_SPARSE_MIN_S", 8192):
        return None
    k = env_int("GMLX_SPARSE_K", 2048)
    nkeep = max(1, k // bl) + 3
    nb_total = (S + bl - 1) // bl
    if nb_total <= nkeep:
        return None
    try:
        from mlx_kquant import sdpa_decode_gqa_paged

        pages = _select_pages(queries, keys, cache, bl, nkeep, pads[:B])
        out = sdpa_decode_gqa_paged(queries, keys, values, scale, pages,
                                    starts=starts)
    except Exception:
        return None
    if B not in _SEEN_B:  # one line per batch width, not per call
        _SEEN_B.add(B)
        import sys

        print(
            f"[sparse] claimed: B={B} S={S} k={k} pages={nkeep}/{nb_total}",
            file=sys.stderr,
            flush=True,
        )
    _CLAIMS[0] += 1
    return out


def _make_route(orig):
    def _sparse_sdpa(queries, keys, values, cache=None, scale=1.0,
                     mask=None, sinks=None):
        out = _claim(queries, keys, values, cache, scale, mask, sinks)
        if out is not None:
            return out
        return orig(queries, keys, values, cache=cache, scale=scale,
                    mask=mask, sinks=sinks)

    _sparse_sdpa._gmlx_orig = orig
    _sparse_sdpa._gmlx_sparse_route = True
    return _sparse_sdpa


def install_sparse_sdpa() -> bool:
    """Claim deep unpadded decode at the module-global SDPA seams.
    Strictly opt-in (GMLX_SPARSE_ATTN=1): the route trades bounded
    quality (KLD ~ quantization noise at the default budget) for
    depth-flat attention cost. Idempotent. Returns True when active."""
    global _installed
    if not env_bool("GMLX_SPARSE_ATTN", False):
        return False
    if _installed:
        return True
    import importlib
    import os

    extra = tuple(
        x.strip() for x in os.environ.get("GMLX_SPARSE_ARCHS", "").split(",")
        if x.strip()
    )
    allowed = _VALIDATED + extra
    patched = 0
    for pkg, name in _MODULES:
        if name not in allowed:
            continue
        try:
            mod = importlib.import_module(f"{pkg}.{name}")
        except ImportError:
            continue
        cur = getattr(mod, "scaled_dot_product_attention", None)
        if cur is None:
            continue
        if getattr(cur, "_gmlx_sparse_route", False):
            patched += 1
            continue
        mod.scaled_dot_product_attention = _make_route(cur)
        patched += 1
    if not patched:
        return False
    _installed = True
    return True
