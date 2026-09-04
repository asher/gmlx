"""KVarN SDPA route: attention over KVarNKVCache regions.

install_kvarn_sdpa() rebinds ``scaled_dot_product_attention`` in every
loaded mlx_lm.models.* / mlx_vlm.models.* module (and both base modules,
so later imports inherit the wrapper). The wrapper only claims calls whose
keys are KVarNView handles; everything else passes through untouched.

Decode (qL 1, plain causal masking) runs fused on the vector kernel: the
query is WHT-rotated, kq.sdpa_decode_gqa_kvarn walks sink rows + sealed
records + live rows in one dispatch, and the output is un-rotated. Verify
width (qL 2 to 8) runs the same walk on the matrix-unit FA kernels
(kq.sdpa_fa_verify_kvarn over the kv-major GQA fold; a fold wider than the
tile splits the group into at most four chunks): the vector kernel's
per-query cost climbs steeply past two queries, the matrix tile prices
extra rows at nearly zero. When the cache carries a precision tail, the
body walk stops at the tail boundary (n_attend, causal clamp lifted) and a
second plain-fp16 attention over the original-domain tail rows merges in
through the log-sum-exp weights, so no token is counted twice and the
freshest context stays full fidelity.

Prefill (wider queries or array masks) materializes the rotated cache once
per call and runs stock mx.fast attention on it with the rotated query,
composing with the attn_hd512 wrapper's chunked-prefill routes.

Kill switches: GMLX_KVARN=0 drops the scheme when the policy resolves at
cache build (kvarn_cache.py); GMLX_KVARN_SDPA=0 forces the materialize
path for decode as well (correct, slower); GMLX_KVARN_FA=0 keeps verify
width on the vector kernel (an A/B). Both read at call time.
"""

from __future__ import annotations

import sys

import mlx.core as mx

from gmlx.envflags import env_bool
from .kvarn_cache import BatchKVarNKVCache, KVarNKVCache, KVarNView

_MODEL_PREFIXES = ("mlx_lm.models.", "mlx_vlm.models.", "gmlx.")
_BASE_MODULES = ("mlx_lm.models.base", "mlx_vlm.models.base")

_probe_result = None


def kvarn_ops_missing():
    """Reason the kvarn kernel surface is unusable, or None. Memoized."""
    global _probe_result
    if _probe_result is None:
        _probe_result = (_probe(),)
    return _probe_result[0]


def _probe():
    try:
        import mlx_kquant as kq
    except ImportError:
        return "mlx-kquant not importable"
    if mx.default_device().type != mx.DeviceType.gpu:
        return "kvarn kernels are Metal-only (cpu default device)"
    missing = [
        op
        for op in (
            "kvarn_quantize",
            "kvarn_dequant",
            "kvarn_rotate",
            "sdpa_decode_gqa_kvarn",
        )
        if not hasattr(kq, op)
    ]
    if missing:
        return "mlx-kquant build lacks " + ", ".join(missing)
    have = getattr(kq, "KVARN_RECORD_VERSION", None)
    want = KVarNKVCache.kvarn_layout_version
    if have != want:
        return (f"mlx-kquant kvarn record layout {have} does not match "
                f"gmlx layout {want}")
    return None


_sdpa_env = None


def _route_enabled() -> bool:
    """GMLX_KVARN_SDPA, read once: 0 forces the materialize path."""
    global _sdpa_env
    if _sdpa_env is None:
        _sdpa_env = env_bool("GMLX_KVARN_SDPA", True)
    return _sdpa_env


_tg_limits: dict[tuple[int, bool], int] = {}


def _tg_limit(d: int, multi: bool) -> int:
    """Largest fused-decode threadgroup this GPU runs for the kernel
    variant (head_dim, qL > 1), probed once per variant. The pipeline cap
    depends on the variant's register use as well as the GPU, and an
    oversized dispatch raises at eval time, past any call site that could
    catch it."""
    key = (d, multi)
    if key not in _tg_limits:
        _tg_limits[key] = _probe_tg_limit(d, 4 if multi else 1)
    return _tg_limits[key]


def _probe_tg_limit(d: int, ql: int) -> int:
    if kvarn_ops_missing():
        return 0
    from .kvarn_cache import GROUP

    c = KVarNKVCache(tail_tokens=0)
    kv = mx.zeros((1, 1, GROUP + 2 * ql, d), mx.float16)
    c.update_and_fetch(kv, kv)
    for gqa in (16, 8, 4, 2, 1):
        if gqa * ((ql + 1) // 2) > 32:
            continue
        try:
            mx.eval(_decode_vector(mx.zeros((1, gqa, ql, d), mx.float16), c, 1.0))
        except Exception:
            continue
        return _tg_threads(gqa, ql)
    return 0


def _tg_threads(gqa: int, qL: int) -> int:
    """Threadgroup width of one fused decode dispatch (kq_sdpa geometry)."""
    return 32 * gqa * ((qL + 1) // 2)


def _fa_available() -> bool:
    import mlx_kquant as kq

    return hasattr(kq, "sdpa_fa_verify_kvarn") and env_bool("GMLX_KVARN_FA", True)


def _fa_row_cap(d: int) -> int:
    """Query rows one FA verify tile holds (the hd512 d-split kernel is
    fixed at 32)."""
    return 32 if d == 512 else 64


def _fa_chunks(gqa: int, qL: int, cap: int):
    """Smallest kv-major split of the GQA group whose fold fits the tile,
    or None. Each chunk re-sweeps the keys, so the split stops at 4."""
    for n in (1, 2, 4):
        if gqa % n == 0 and (gqa // n) * qL <= cap:
            return n
    return None


def _fa_route(q, cache) -> bool:
    """Verify width (qL >= 2) runs on the matrix-unit FA kernels: B=1, the
    op present, and the GQA fold splitting into at most four tiles."""
    if q.shape[2] < 2 or q.shape[0] != 1 or not _fa_available():
        return False
    gqa = q.shape[1] // cache.stage_k.shape[1]
    return _fa_chunks(gqa, q.shape[2], _fa_row_cap(q.shape[-1])) is not None


def _fused_ok(q, cache) -> bool:
    kvh = cache.stage_k.shape[1]
    gqa = q.shape[1] // kvh
    qL = q.shape[2]
    if not (
        q.dtype in (mx.float16, mx.bfloat16)
        and 1 <= gqa <= 16
        and q.shape[1] % kvh == 0
        and _route_enabled()
    ):
        return False
    if _fa_route(q, cache):
        return True
    return qL <= 4 and _tg_threads(gqa, qL) <= _tg_limit(q.shape[-1], qL > 1)


def _legs(n: int, tail_len: int, qL: int) -> tuple[int, int]:
    """Body/tail key split for the fused decode. Each leg the merge runs
    needs qL keys (the body's full-visibility clamp and the tail kernel
    both require it). A body sliver widens into the tail when the tail
    can spare qL keys; otherwise the body takes the whole stream, whose
    stage rows are the same values the tail holds."""
    t = min(tail_len, n)
    n_body = n - t
    if 0 < n_body < qL and n >= 2 * qL:
        return qL, n - qL
    if 0 < t < qL or 0 < n_body < qL:
        return n, 0
    return n_body, t


def _lse_merge(out_a, lse_a, out_b, lse_b):
    """Numerically stable two-segment softmax merge in fp32."""
    a = out_a.astype(mx.float32)
    b = out_b.astype(mx.float32)
    m = mx.maximum(lse_a, lse_b)
    wa = mx.exp(lse_a - m)
    wb = mx.exp(lse_b - m)
    return (a * wa[..., None] + b * wb[..., None]) / (wa + wb)[..., None]


def _decode(q, cache, scale):
    if _fa_route(q, cache):
        return _decode_fa(q, cache, scale)
    return _decode_vector(q, cache, scale)


def _decode_fa(q, cache, scale):
    """Verify width on the FA kernels over the kv-major GQA fold: the body
    reads records through sdpa_fa_verify_kvarn (clamp lifted at a tail
    boundary), the tail runs sdpa_fa_verify over the fp16 rows with exact
    offset causality, and the two merge through their LSEs. A fold wider
    than the tile splits the GQA group; every chunk re-sweeps the keys."""
    import mlx_kquant as kq

    _, hq, qL, d = q.shape
    kvh = cache.stage_k.shape[1]
    gqa = hq // kvh
    n = cache.visible
    n_body, t = _legs(n, cache.tail_len, qL)
    g = gqa // _fa_chunks(gqa, qL, _fa_row_cap(d))
    rows = g * qL
    q5 = q.reshape(1, kvh, gqa, qL, d)
    q_rot5 = kq.kvarn_rotate(q).reshape(1, kvh, gqa, qL, d) if n_body else None
    if t:
        tk, tv = cache.tail_slices(t)
        tk, tv = tk.astype(q.dtype), tv.astype(q.dtype)
    outs = []
    for g0 in range(0, gqa, g):
        qf = mx.contiguous(q5[:, :, g0 : g0 + g]).reshape(1, kvh, rows, d)
        if n_body == 0:
            outs.append(kq.sdpa_fa_verify(qf, tk, tv, scale, qL))
            continue
        qf_rot = mx.contiguous(q_rot5[:, :, g0 : g0 + g]).reshape(1, kvh, rows, d)
        body_args = (
            qf_rot,
            cache.codes_k,
            cache.axes_k,
            cache.codes_v,
            cache.axes_v,
            cache.stage_k,
            cache.stage_v,
            n,
            scale,
            cache.k_bits,
            cache.v_bits,
            qL,
        )
        if t == 0:
            outs.append(kq.kvarn_rotate(kq.sdpa_fa_verify_kvarn(*body_args)))
            continue
        body, lse_b = kq.sdpa_fa_verify_kvarn(
            *body_args, n_attend=n_body, full_visibility=True, return_lse=True
        )
        tail, lse_t = kq.sdpa_fa_verify(qf, tk, tv, scale, qL, return_lse=True)
        merged = _lse_merge(kq.kvarn_rotate(body), lse_b, tail, lse_t)
        outs.append(merged.astype(q.dtype))
    if len(outs) == 1:
        return outs[0].reshape(1, hq, qL, d)
    out = mx.concatenate([o.reshape(1, kvh, g, qL, d) for o in outs], axis=2)
    return out.reshape(1, hq, qL, d)


def _decode_vector(q, cache, scale):
    import mlx_kquant as kq

    n = cache.visible
    n_body, t = _legs(n, cache.tail_len, q.shape[2])
    if n_body == 0:
        tk, tv = cache.tail_slices(t)
        return kq.sdpa_decode_gqa(q, tk.astype(q.dtype), tv.astype(q.dtype), scale)
    q_rot = kq.kvarn_rotate(q)
    body_args = (
        q_rot,
        cache.codes_k,
        cache.axes_k,
        cache.codes_v,
        cache.axes_v,
        cache.stage_k,
        cache.stage_v,
        n,
        scale,
        cache.k_bits,
        cache.v_bits,
    )
    if t == 0:
        return kq.kvarn_rotate(kq.sdpa_decode_gqa_kvarn(*body_args))
    body, lse_b = kq.sdpa_decode_gqa_kvarn(
        *body_args, n_attend=n_body, full_visibility=True, return_lse=True
    )
    tk, tv = cache.tail_slices(t)
    tail, lse_t = kq.sdpa_decode_gqa(
        q, tk.astype(q.dtype), tv.astype(q.dtype), scale, return_lse=True
    )
    merged = _lse_merge(kq.kvarn_rotate(body), lse_b, tail, lse_t)
    return merged.astype(q.dtype)


def _decode_batch(q, cache, scale, starts):
    """qL=1 batched decode: same body/tail split as _decode with per-row
    key starts (left padding). A row admitted deep into an older batch can
    start inside the tail window; its body leg then attends zero keys and
    contributes nothing through the LSE weights."""
    import mlx_kquant as kq

    n = cache._idx
    t = min(cache.tail_len, n)
    n_body = n - t
    if n_body == 0:
        tk, tv = cache.tail_slices(t)
        return kq.sdpa_decode_gqa(
            q, tk.astype(q.dtype), tv.astype(q.dtype), scale, starts=starts
        )
    q_rot = kq.kvarn_rotate(q)
    body_args = (
        q_rot,
        cache.codes_k,
        cache.axes_k,
        cache.codes_v,
        cache.axes_v,
        cache.stage_k,
        cache.stage_v,
        n,
        scale,
        cache.k_bits,
        cache.v_bits,
    )
    if t == 0:
        return kq.kvarn_rotate(kq.sdpa_decode_gqa_kvarn(*body_args, starts=starts))
    body, lse_b = kq.sdpa_decode_gqa_kvarn(
        *body_args,
        starts=mx.minimum(starts, n_body).astype(mx.int32),
        n_attend=n_body,
        full_visibility=True,
        return_lse=True,
    )
    tk, tv = cache.tail_slices(t)
    tail_starts = mx.maximum(starts - n_body, 0).astype(mx.int32)
    tail, lse_t = kq.sdpa_decode_gqa(
        q,
        tk.astype(q.dtype),
        tv.astype(q.dtype),
        scale,
        starts=tail_starts,
        return_lse=True,
    )
    merged = _lse_merge(kq.kvarn_rotate(body), lse_b, tail, lse_t)
    return merged.astype(q.dtype)


def _prefill(q, cache, scale, mask):
    import mlx_kquant as kq

    k, v = cache.materialize(dtype=q.dtype)
    out = mx.fast.scaled_dot_product_attention(
        kq.kvarn_rotate(q), k, v, scale=scale, mask=mask
    )
    return kq.kvarn_rotate(out)


def _batch_starts(cache, mask):
    """Per-row starts for a batched decode call, or None to decline. The
    mask must be one this cache's make_mask registered (provenance, not
    content -- inspecting mask contents is a GPU sync); windowed or foreign
    masks fall back to the materialize path."""
    if not isinstance(mask, mx.array):
        return None
    from gmlx.upstream.quantized_sdpa_fix import _registered_starts

    return _registered_starts(mask)


def _pad_mask(starts, n, qL):
    """Left-pad + causal bool mask [B,1,qL,n] for the materialize fallback
    when only per-row starts are known (no mask to reuse)."""
    t = mx.arange(n)[None, None, :]
    end = (n - qL) + mx.arange(qL)[None, :, None]
    return ((t >= starts[:, None, None]) & (t <= end))[:, None]


def kvarn_attention(q, cache, scale, mask, sinks=None, starts=None):
    """Attention over a kvarn cache. ``starts`` lets owned dispatches
    (qwen3.5) pass per-row left padding directly when their mask protocol
    carries none; unset, batched decode derives it from mask provenance."""
    if sinks is not None:
        raise RuntimeError(
            "[kvarn] attention sinks reached the kvarn route; this arch "
            "should have been declined at cache build time."
        )
    if isinstance(cache, BatchKVarNKVCache):
        if (
            q.shape[2] == 1
            and q.shape[0] == cache.stage_k.shape[0]
            and _fused_ok(q, cache)
        ):
            s = starts if starts is not None else _batch_starts(cache, mask)
            if s is not None:
                return _decode_batch(q, cache, float(scale), s)
        if starts is not None and not isinstance(mask, mx.array):
            # Declined decode with explicit pads: the materialize path
            # still needs the pad rows masked out.
            mask = _pad_mask(starts, cache._idx, q.shape[2])
        return _prefill(q, cache, float(scale), mask)
    plain_mask = mask is None or (isinstance(mask, str) and mask == "causal")
    if (
        1 <= q.shape[2] <= 8
        and plain_mask
        and q.shape[0] == 1
        and _fused_ok(q, cache)
    ):
        return _decode(q, cache, float(scale))
    return _prefill(q, cache, float(scale), mask)


def _make_wrapper(orig):
    def scaled_dot_product_attention(queries, keys, values, *args, **kwargs):
        if isinstance(keys, KVarNView):
            scale = kwargs.get("scale", args[1] if len(args) > 1 else 1.0)
            mask = kwargs.get("mask", args[2] if len(args) > 2 else None)
            sinks = kwargs.get("sinks", args[3] if len(args) > 3 else None)
            return kvarn_attention(queries, keys.cache, scale, mask, sinks)
        return orig(queries, keys, values, *args, **kwargs)

    scaled_dot_product_attention._gmlx_kvarn = True
    scaled_dot_product_attention._gmlx_orig = orig
    return scaled_dot_product_attention


def install_kvarn_sdpa() -> int:
    """Sweep-rebind the SDPA symbol over loaded model modules. Idempotent
    per module; returns the number of modules now carrying the wrapper."""
    import importlib

    for name in _BASE_MODULES:
        try:
            importlib.import_module(name)
        except ImportError:
            pass
    patched = 0
    for name, mod in list(sys.modules.items()):
        if mod is None or not (
            name in _BASE_MODULES or name.startswith(_MODEL_PREFIXES)
        ):
            continue
        cur = getattr(mod, "scaled_dot_product_attention", None)
        if cur is None or not callable(cur):
            continue
        if getattr(cur, "_gmlx_kvarn", False):
            patched += 1
            continue
        mod.scaled_dot_product_attention = _make_wrapper(cur)
        patched += 1
    return patched
