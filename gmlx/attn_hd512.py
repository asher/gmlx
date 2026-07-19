"""Fused vector attention for head dims stock MLX leaves unfused, via mlx-kquant.

Stock MLX's fused scaled_dot_product_attention has two head_dim allowlists: a
vector path {64,96,128,256} (qL==1 decode only) and a tiled path {64,80,128}
(qL>1). Two cases fall through to a materialized matmul -> softmax -> matmul:

  - head_dim 512 (gemma-4 global/full-attention layers): excluded from both, so
    even decode materializes and leaves the GPU bandwidth-idle.
  - head_dim 256 at verify width (Qwen3.x full-attention layers, speculative
    verify with qL>1): in the vector allowlist so qL==1 decode is fused, but the
    tiled path excludes 256, so a qL>1 verify forward materializes a [Hq, qL, kL]
    score per full-attn layer at depth.

This routes head_dim 512 through mlx_kquant.sdpa_vector, a compiled two-pass
online-softmax kernel (256/512 instantiations of MLX's own sdpa_vector_2pass), from
qL==1. Measured vs the stock materialized fallback at 16k-32k context: ~1.0-1.15x at
decode (qL=1) and ~1.3-1.9x at verify (qL 2..5), tapering to a wash by qL~7, so the
route is gated to qL <= GMLX_HD512_MAXQL (default 6). Output matches stock SDPA
to bf16 rounding.

The head_dim-256 verify route (qL>=2; its qL==1 decode is already fused by the
vector path) is opt-in, off by default. Enable with GMLX_HD256_VERIFY=1.

head_dim-512 verify widths (qL 3..5, kv >= 2) preferentially take the
kq.sdpa_fa_verify d-split kernel over the same GQA fold (one flash-attention
KV sweep on the matrix units, no materialized score), then fall back to a
gqa-folded GEMM route, then sdpa_vector: the GQA group folds into the query
rows so one matrix-unit KV sweep serves the whole verify block, where the
vector kernel walks the KV once per query row. GEMM route measured at the
gemma-4-31b global-layer shape: qL=4 1.6-2.2x per call from 2k to 131k
(qL=2 loses, stays on vector); this is the MTP verify depth-slope fix.
A/B fa-verify vs GEMM with GMLX_VERIFY_FA=0; disable the GEMM fold
with GMLX_VERIFY_GEMM=0.

Separately, GQA decode (qL==1, gqa 2..8) routes to mlx_kquant.sdpa_decode_gqa
once the KV is deep enough. head_dim 64 (e.g. gpt-oss full-attn layers) engages
at GMLX_GQA_SDPA_MINKV (default 4096): stock MLX's fused vector path
plateaus near 37% of read-once bandwidth at long KV; the kq kernel's coarse
contiguous splits + GQA-shared K/V tile staging reach ~55-60% (1.3-1.6x per
call from 16k up). head_dim 512 (gemma-4 global layers, >= 2 kv heads) engages
at GMLX_GQA_SDPA_MINKV512 (default 32768), where it overtakes the
kq.sdpa_vector route below (1.12x @32k -> 1.28x @131k per call; token-exact
+1.7% whole-step at 49k on gemma-4-31b). Attention sinks ride through the
kernel's merge pass. Disable with GMLX_GQA_SDPA=0.

Also tiles a pathological single-shot full-width prefill (qL > GMLX_HD512_TILE)
so its score materialization stays bounded; the runtime's normal chunked prefill
(<= 2048 per step) passes straight through. Disable the head_dim-512 kernel route
entirely with GMLX_HD512=0.
"""
from __future__ import annotations


import mlx.core as mx

from .envflags import env_bool, env_int

try:
    import mlx_kquant
    _HAS_COMPILED = hasattr(mlx_kquant, "sdpa_vector")
    _HAS_GQA_DECODE = hasattr(mlx_kquant, "sdpa_decode_gqa")
except Exception:  # pragma: no cover - mlx_kquant always present in practice
    mlx_kquant = None
    _HAS_COMPILED = False
    _HAS_GQA_DECODE = False

# Decode + verify: route to the compiled kernel up to this query width; above it
# the stock materialized fallback is as fast or faster (the two-pass partials
# buffer scales with qL). 6 is the measured crossover at 32k context.
_MAX_QL = env_int("GMLX_HD512_MAXQL", 6)
# Engage only once context is deep enough to matter; short context is cheap on the
# stock path.
_MIN_KV = env_int("GMLX_HD512_MINKV", 4096)
# Tile width for the chunked-prefill guard. Defaults to the runtime's prefill step
# (2048), so a normally-chunked prefill passes straight through (qL <= tile) and
# only a pathological single-shot full-width forward (qL > 2048) gets tiled to keep
# its score materialization bounded.
_PREFILL_TILE = env_int("GMLX_HD512_TILE", 2048)
# head dims stock MLX has a fused full/prefill SDPA kernel for; any other head_dim
# materializes the full score tensor at prefill (see MLX sdpa cpp dispatch).
_FULL_HD = (64, 80, 128)
# head dims mlx-kquant ships a kq.sdpa_vector instantiation for (bf16/f16 x {256,
# 512}); see metal/kq_sdpa.metal. 512 routes from qL==1; the 256 verify route
# (qL>=2; its qL==1 decode is already fused by stock) is on by default -- without
# it, stock MLX materializes the full [H, qL, kL] score tensor at verify width
# since 256 is not in its fused full-kernel allowlist {64, 80, 128}.
_KQ_HD = (256, 512)
_HD256_VERIFY = env_bool("GMLX_HD256_VERIFY", True)
# hd64 GQA decode route (kq.sdpa_decode_gqa): fixed coarse key splits +
# threadgroup-staged K/V tiles shared across the GQA group, sinks folded into
# the merge. hd64 beats the stock vector path 1.27-1.6x from 16k up at gpt-oss
# shapes (hd64 GQA-8). hd256 (qwen3.5/3.6 full-attn, GQA 4-6) is instantiated
# but off by default: stock's hd256 vector kernel already runs ~78-86% of
# peak, and while an isolated per-call probe read 1.08-1.19x at 49k+, the
# token-exact E2E on qwen3.6-27b lost (-2.9% at 65k, -4.2% at 110k) -- the
# probe pipelined independent calls (throughput) where the decode step
# serializes the two-pass dispatch (latency). Opt back in with
# GMLX_GQA_SDPA_HD256=1 (e.g. to re-measure under batch/verify). hd128
# is instantiated in mlx-kquant but unrouted: no resident model decodes
# hd128 GQA; measure before routing (llama3-class).
_GQA_DECODE = env_bool("GMLX_GQA_SDPA", True)
_GQA_MIN_KV = env_int("GMLX_GQA_SDPA_MINKV", 4096)
_GQA_HD256 = env_bool("GMLX_GQA_SDPA_HD256", False)
_GQA_MIN_KV_256 = env_int("GMLX_GQA_SDPA_MINKV256", 49152)
# hd512 GQA decode (gemma-4 global layers): chained per-call probe at the 31b
# shape (Hq/Hkv 32/4) reads 1.12x @32k -> 1.28x @131k over kq.sdpa_vector,
# crossing over between 16k and 32k. Requires >= 2 kv heads: at Hkv=1 (12b)
# the split grid is too starved and the kernel loses at every depth.
_GQA_MIN_KV_512 = env_int("GMLX_GQA_SDPA_MINKV512", 32768)
# hd512 speculative-verify width (qL 3..6): fold the GQA group into the query
# rows ([B,Hq,qL,D] -> [B,Hkv,G*qL,D]; exact -- query heads are grouped
# kv-major) and run plain batched-GEMM attention with a bottom-right causal
# mask and precise softmax. One matrix-unit KV sweep serves all G*qL rows
# where the kq.sdpa_vector route pays a strided sweep per query row; measured
# at the gemma-4-31b global shape (32/4, hd512, bf16): qL=4 1.6-2.2x per call
# from 2k to 131k, qL=3 1.1-1.5x, qL=2 LOSES mid-depth (stays on vector).
# Disable with GMLX_VERIFY_GEMM=0 (falls back to kq.sdpa_vector above).
_VERIFY_GEMM = env_bool("GMLX_VERIFY_GEMM", True)
# Speculative-verify on the matrix units: kq.sdpa_fa_verify, a
# simdgroup-matrix flash-attention pass over the same GQA fold. One KV sweep
# on matrix units vs the vector route's per-row strided walk; measured at
# 24/4 hd256 bf16 (qwen3.x full-attn): qL=4 1.14x @4k -> 1.53x @131k, qL=3
# wins from ~16k, qL=2 loses (stays on vector). The kernel takes one row
# tile (hd256: probed 32 or 64; hd512 d-split: 32); folds up to 4x that
# (e.g. 32/2 gqa16 at qL 4) run as per-chunk calls over a kv-major split of
# the GQA group -- each chunk re-sweeps the KV, still far ahead of the
# per-row vector walk at those shapes. hd512 (gemma-4 26b-moe/31b global
# layers, kv>=2) claims ahead of the verify_gemm fold below; A/B the two
# with GMLX_VERIFY_FA=0. Disable with GMLX_VERIFY_FA=0.
_VERIFY_FA = env_bool("GMLX_VERIFY_FA", True)
_FA_MIN_KV = env_int("GMLX_VERIFY_FA_MINKV", 4096)
_FA_MIN_KV_QL3 = 16384
_HAS_FA_VERIFY = mlx_kquant is not None and hasattr(mlx_kquant, "sdpa_fa_verify")
# hd256 wide-group decode (gqa > 8, i.e. qwen3.5-122b 32/2): the per-key dot
# fan-out (16 dots/staged element) is compute-bound on the FMA-path kernels
# -- stock and kq.sdpa_decode_gqa both plateau ~40% of read-once bandwidth --
# so decode routes to the matrix-unit fa kernel as a 1-query fold (G rows,
# q_len 1). Measured at 32/2 hd256: par to +10% per call from 32k up, and
# ~2.5x when stock hits its slow mode at 16-32k. Requires the q_len>=1
# mlx-kquant op gate. Disable with GMLX_FA_DECODE=0.
_FA_DECODE = env_bool("GMLX_FA_DECODE", True)
_FA_DECODE_MIN_KV = env_int("GMLX_FA_DECODE_MINKV", 32768)


def _probe_fa_max_rows():
    # Kernel tile capability: newer mlx-kquant accepts a 64-row fold (one
    # KV sweep for gqa16 x qL4); older builds cap at 32. The op validator
    # runs at build time, so probing costs no GPU work (nothing is evaled).
    if not _HAS_FA_VERIFY:
        return 0
    try:
        z = mx.zeros((1, 1, 64, 256), dtype=mx.float16)
        mlx_kquant.sdpa_fa_verify(z, z, z, 1.0, 4)
        return 64
    except Exception:
        return 32


_FA_MAX_ROWS = _probe_fa_max_rows()
_orig_sdpa = None
_installed = False


def _causal_str(mask):
    return mask is None or (isinstance(mask, str) and mask == "causal")


def _eligible(q, k, v, mask):
    # B==1, a kq.sdpa_vector head_dim (q and v), verify/decode width, deep enough,
    # causal/full mask.
    if q.shape[0] != 1 or q.ndim != 4:
        return False
    hd = q.shape[-1]
    if hd not in _KQ_HD or v.shape[-1] != hd:
        return False
    if hd == 512:
        min_ql = 1  # stock materializes hd512 even at decode
    elif hd == 256 and _HD256_VERIFY:
        min_ql = 2  # hd256 qL==1 decode is already fused on stock; verify is not
    else:
        return False
    qL = q.shape[2]
    kL = k.shape[2]
    if qL < min_ql or qL > _MAX_QL or qL > kL or kL < _MIN_KV:
        return False
    if q.shape[1] % k.shape[1] != 0:
        return False
    # Only a causal/full mask is handled (global/full-attn layers pass "causal"/None).
    return _causal_str(mask)


def _prefill_eligible(q, k, v, mask):
    # B==1, query wider than the verify path (a prefill), and a head_dim that stock
    # has no fused full-attention kernel for. Stock's full kernel only handles
    # head_dim in _FULL_HD; anything else materializes the whole [Hq, qL, kL] score
    # tensor for every such layer (~128 GB for a 32-head model at 32k) and thrashes
    # to swap. gemma-4 has two unsupported prefill head dims: 512 (global/causal
    # layers) and 256 (sliding/array-mask layers, the majority of the layer count),
    # so both must be chunked.
    if q.shape[0] != 1 or q.ndim != 4:
        return False
    hd = q.shape[-1]
    if hd == v.shape[-1] and hd in _FULL_HD:
        return False  # stock has a fused prefill kernel for this head_dim
    qL = q.shape[2]
    if qL <= 8 or qL > k.shape[2]:
        return False
    if _causal_str(mask):
        return True
    # an array mask we can slice along the query axis (sliding-window layers)
    return isinstance(mask, mx.array) and mask.ndim >= 2 and mask.shape[-2] == qL


def _chunked_prefill(q, k, v, scale, mask, tile, sinks=None):
    """Tile the query so the materialized scores stay [Hq, tile, kL]. Each query
    row's attention is independent, so slicing query rows is exact -- including
    under attention sinks, a per-head additive logit in every row's softmax
    denominator regardless of key slicing. Causal: also slice keys to the tile
    horizon (mask='causal' keeps it causal). Array mask (sliding): slice the
    mask's query rows, keep all keys.

    Each tile is eval'd before the next so its [Hq, tile, kL] score is freed
    instead of accumulating across tiles (and layers) in one lazy graph -- without
    this, peak memory stays as high as the full materialization and still swaps."""
    skw = {} if sinks is None else {"sinks": sinks}
    qL = q.shape[2]
    arr = isinstance(mask, mx.array)
    if qL <= tile:
        return _orig_sdpa(q, k, v, scale=scale,
                          mask=(mask if arr else "causal"), **skw)
    # With a cached prefix (kL > qL, chunk 2+ of a chunked prefill) the causal
    # horizon of query row t is offset + t, not t: slicing keys to t1 would
    # select only the head of the cached prefix and drop the chunk's own keys.
    offset = k.shape[2] - qL
    outs = []
    for t0 in range(0, qL, tile):
        t1 = min(t0 + tile, qL)
        qt = q[:, :, t0:t1, :]
        if arr:
            ot = _orig_sdpa(qt, k, v, scale=scale, mask=mask[..., t0:t1, :],
                            **skw)
        else:
            ot = _orig_sdpa(qt, k[:, :, :offset + t1, :],
                            v[:, :, :offset + t1, :],
                            scale=scale, mask="causal", **skw)
        mx.eval(ot)
        outs.append(ot)
    return mx.concatenate(outs, axis=2)


def _gqa_decode_eligible(q, k, v, mask):
    # hd64 (and opt-in hd256) GQA decode (qL == 1) at deep KV; sinks are
    # handled by the kernel. Batched serve decode with per-sequence KV padding
    # passes an array mask and falls through to stock.
    if q.ndim != 4 or q.shape[2] != 1:
        return False
    hd = q.shape[-1]
    if hd == 64:
        min_kv = _GQA_MIN_KV
    elif hd == 256 and _GQA_HD256:
        min_kv = _GQA_MIN_KV_256
    elif hd == 512 and k.shape[1] >= 2:
        min_kv = _GQA_MIN_KV_512
    else:
        return False
    if v.shape[-1] != hd or k.shape[-1] != hd:
        return False
    if k.shape[2] < min_kv:
        return False
    kv_heads = k.shape[1]
    if kv_heads == 0 or q.shape[1] % kv_heads != 0:
        return False
    if not 2 <= q.shape[1] // kv_heads <= 8:
        return False
    return _causal_str(mask)


def _verify_gemm_eligible(q, k, v, mask):
    # B==1 hd512 verify width at deep KV, causal/full mask. qL==2 stays on the
    # vector route (two sweeps amortize fine; GEMM measured 0.83-0.97x there);
    # from qL==3 the fold wins at every depth.
    if q.ndim != 4 or q.shape[0] != 1:
        return False
    hd = q.shape[-1]
    if hd != 512 or v.shape[-1] != hd or k.shape[-1] != hd:
        return False
    qL = q.shape[2]
    if not 3 <= qL <= _MAX_QL or qL > k.shape[2]:
        return False
    if k.shape[2] < _MIN_KV:
        return False
    # >= 2 kv heads: at Hkv=1 (gemma-4-12b globals) there is no GQA
    # amplification for the fold to remove -- stock's fallback is within
    # +-8% at every depth and slightly ahead past 32k.
    if k.shape[1] < 2 or q.shape[1] % k.shape[1] != 0:
        return False
    return _causal_str(mask)


def _verify_gemm(q, k, v, scale, causal):
    B, hq, qL, hd = q.shape
    kv = k.shape[1]
    g = hq // kv
    kL = k.shape[2]
    qg = (q * scale).reshape(B, kv, g * qL, hd)
    s = qg @ k.swapaxes(-1, -2)
    if causal and qL > 1:
        # bottom-right aligned: folded row r is query r % qL of head r // qL,
        # visible keys <= kL - qL + (r % qL)
        rows = mx.tile(mx.arange(kL - qL, kL), g).reshape(g * qL, 1)
        s = mx.where(mx.arange(kL).reshape(1, kL) <= rows, s,
                     mx.array(-float("inf"), dtype=s.dtype))
    o = mx.softmax(s, axis=-1, precise=True) @ v
    return o.reshape(B, hq, qL, hd)


def _fa_chunks(g, qL, cap=None):
    # Smallest kv-major split of the GQA group whose fold fits the kernel's
    # row tile (probed: 64 on newer mlx-kquant, else 32); each chunk pays
    # its own KV sweep, so cap the split at 4.
    cap = _FA_MAX_ROWS if cap is None else cap
    for n in (1, 2, 4):
        if g % n == 0 and (g // n) * qL <= cap:
            return n
    return None


def _fa_decode_eligible(q, k, v, mask):
    # B==1 hd256 decode with a wide GQA group (> 8, beyond the FMA kernels'
    # compute knee); the group must fit one fa tile (<= 64 rows).
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[2] != 1 or q.shape[-1] != 256:
        return False
    if v.shape[-1] != 256 or k.shape[-1] != 256:
        return False
    kv = k.shape[1]
    if kv == 0 or q.shape[1] % kv != 0:
        return False
    g = q.shape[1] // kv
    if not 8 < g <= 64:
        return False
    if k.shape[2] < _FA_DECODE_MIN_KV:
        return False
    return _causal_str(mask)


def _fa_row_cap(hd):
    # hd512 runs the d-split kernel variant: fixed 32-row tile (no 64-row
    # instantiation), regardless of the probed hd256 cap.
    return 32 if hd == 512 else _FA_MAX_ROWS


def _fa_verify_eligible(q, k, v, mask):
    # B==1 verify width, causal only, fold must split into <=4 row tiles
    # along the GQA group. hd256 = qwen3.x full-attn; hd512 = gemma-4
    # global layers (26b-moe/31b folds G8 x qL4 = one 32-row tile).
    if q.ndim != 4 or q.shape[0] != 1:
        return False
    hd = q.shape[-1]
    if hd not in (256, 512) or v.shape[-1] != hd or k.shape[-1] != hd:
        return False
    qL = q.shape[2]
    kv = k.shape[1]
    if kv == 0 or q.shape[1] % kv != 0:
        return False
    # Hkv==1 (gemma-4-12b globals): no GQA amplification for the fold to
    # remove; stock is par at depth (same finding as the verify_gemm gate).
    if hd == 512 and kv < 2:
        return False
    if not 3 <= qL <= 5 or _fa_chunks(q.shape[1] // kv, qL, _fa_row_cap(hd)) is None:
        return False
    if qL > k.shape[2] or k.shape[2] < (_FA_MIN_KV_QL3 if qL == 3 else _FA_MIN_KV):
        return False
    return isinstance(mask, str) and mask == "causal"


_SDPA_DEBUG = [24] if env_bool("GMLX_SDPA_DEBUG", False) else None

# Route visibility, always on: per-(route, head_dim, q_len, kv-bucket) call
# counts (dict bump, negligible next to the Python wrapper itself), an exit
# summary under GMLX_ROUTE_LOG=1, and a one-shot loud warning when a
# verify-shaped causal call at depth lands on the stock materialized path --
# the regression signature that previously needed ad-hoc taps to surface.
# Counts are taken at claim time; a kernel exception that falls through to a
# later branch double-counts, which is itself a signal worth seeing.
_ROUTE_COUNTS: dict = {}
_STOCK_WARNED: set = set()


def _kv_bucket(kv):
    return 0 if kv < 4096 else 1 << (kv.bit_length() - 1)


def route_counts() -> dict:
    """SDPA route -> call count, keyed (route, head_dim, q_len, kv_bucket)."""
    return dict(_ROUTE_COUNTS)


def _route(name, q, k, mask):
    key = (name, q.shape[3], q.shape[2], _kv_bucket(k.shape[2]))
    _ROUTE_COUNTS[key] = _ROUTE_COUNTS.get(key, 0) + 1
    _sdpa_debug(name, q, k, mask)


def _stock_depth_warning(q, k, mask, sinks):
    if sinks is not None or not (isinstance(mask, str) and mask == "causal"):
        return
    B, hq, qL, hd = q.shape
    if B != 1 or not 2 <= qL <= 8 or hd < 256 or k.shape[2] < 16384:
        return
    key = (hd, qL, hq, k.shape[1])
    if key in _STOCK_WARNED:
        return
    _STOCK_WARNED.add(key)
    import sys
    print(f"[gmlx] warning: verify-shaped causal SDPA q={tuple(q.shape)} "
          f"k={tuple(k.shape)} fell to the stock materialized path at depth; "
          f"a fused verify route (fa_verify/verify_gemm/sdpa_vector) should "
          f"claim this shape. GMLX_SDPA_DEBUG=1 traces routes.",
          file=sys.stderr, flush=True)


if env_bool("GMLX_ROUTE_LOG", False):
    import atexit

    def _dump_route_counts():
        import sys
        for (name, hd, qL, kvb), n in sorted(_ROUTE_COUNTS.items()):
            print(f"[sdpa-route] {name} hd={hd} qL={qL} kv_bucket={kvb}: {n}",
                  file=sys.stderr, flush=True)

    atexit.register(_dump_route_counts)


def _sdpa_debug(route, q, k, mask):
    if _SDPA_DEBUG is None or _SDPA_DEBUG[0] <= 0:
        return
    if k.shape[2] < 4096:  # shallow KV is stock by design; log deep routes only
        return
    _SDPA_DEBUG[0] -= 1
    import sys
    m = mask if isinstance(mask, (str, type(None))) else f"array{tuple(mask.shape)}"
    print(f"[sdpa-debug] route={route} q={tuple(q.shape)} k={tuple(k.shape)} "
          f"mask={m}", file=sys.stderr, flush=True)


def _wrapped_sdpa(q, k, v, *, scale=1.0, mask=None, **kw):
    if (_HAS_GQA_DECODE and _GQA_DECODE
            and _gqa_decode_eligible(q, k, v, mask)):
        try:
            _route("gqa_decode", q, k, mask)
            return mlx_kquant.sdpa_decode_gqa(
                q, k, v, float(scale), sinks=kw.get("sinks"))
        except Exception:
            pass  # any unsupported shape -> stock fallback
    if (_FA_DECODE and _HAS_FA_VERIFY and kw.get("sinks") is None
            and _fa_decode_eligible(q, k, v, mask)):
        try:
            B, hq, _, hd = q.shape
            kv = k.shape[1]
            _route("fa_decode", q, k, mask)
            out = mlx_kquant.sdpa_fa_verify(
                mx.contiguous(q.reshape(B, kv, hq // kv, hd)),
                k, v, float(scale), 1)
            return out.reshape(B, hq, 1, hd)
        except Exception:
            pass  # older kernel gate (q_len >= 2) -> stock fallback
    if (_VERIFY_FA and _HAS_FA_VERIFY and kw.get("sinks") is None
            and _fa_verify_eligible(q, k, v, mask)):
        try:
            B, hq, qL, hd = q.shape
            kv = k.shape[1]
            g = hq // kv
            n = _fa_chunks(g, qL, _fa_row_cap(hd))
            _route(f"fa_verify(chunks={n})", q, k, mask)
            if n == 1:
                out = mlx_kquant.sdpa_fa_verify(
                    q.reshape(B, kv, g * qL, hd), k, v, float(scale), qL)
            else:
                # Oversized fold (e.g. gqa16 x qL4 = 64 rows): kv-major chunks
                # of the GQA group, one kernel call (= one KV sweep) each.
                # Chunks must be materialized: the kernel mis-reads strided
                # q views (returns wrong rows for kv-head > 0).
                qc = q.reshape(B, kv, n, (g // n) * qL, hd)
                out = mx.concatenate(
                    [mlx_kquant.sdpa_fa_verify(mx.contiguous(qc[:, :, i]),
                                               k, v, float(scale), qL)
                     for i in range(n)],
                    axis=2)
            return out.reshape(B, hq, qL, hd)
        except Exception:
            pass  # any unsupported shape -> stock fallback
    if (_VERIFY_GEMM and kw.get("sinks") is None
            and _verify_gemm_eligible(q, k, v, mask)):
        try:
            _route("verify_gemm", q, k, mask)
            return _verify_gemm(q, k, v, float(scale), mask == "causal")
        except Exception:
            pass  # any unsupported shape -> stock fallback
    if (_HAS_COMPILED and _eligible(q, k, v, mask)
            and kw.get("sinks") is None):
        try:
            _route("sdpa_vector", q, k, mask)
            return mlx_kquant.sdpa_vector(
                q, k, v, float(scale), causal=(mask == "causal"))
        except Exception:
            pass  # any unsupported shape -> stock fallback
    elif _prefill_eligible(q, k, v, mask):
        try:
            _route("chunked_prefill", q, k, mask)
            return _chunked_prefill(q, k, v, scale, mask, _PREFILL_TILE,
                                    sinks=kw.get("sinks"))
        except Exception:
            pass  # any unsupported shape -> stock fallback
    _route("stock", q, k, mask)
    _stock_depth_warning(q, k, mask, kw.get("sinks"))
    return _orig_sdpa(q, k, v, scale=scale, mask=mask, **kw)


def install_hd512_sdpa() -> bool:
    """Route head_dim-512 SDPA through the compiled kernel + chunked prefill.
    Idempotent; no-op when GMLX_HD512=0. Returns True if the patch is active."""
    global _orig_sdpa, _installed
    if not env_bool("GMLX_HD512", True):
        return False
    if _installed:
        return True
    # A prior install (e.g. before a module reload) may have left its wrapper
    # in place; unwrap to the true original instead of wrapping the wrapper.
    cur = mx.fast.scaled_dot_product_attention
    _orig_sdpa = getattr(cur, "_gmlx_orig_sdpa", cur)
    _wrapped_sdpa._gmlx_orig_sdpa = _orig_sdpa
    mx.fast.scaled_dot_product_attention = _wrapped_sdpa
    _installed = True
    return True
