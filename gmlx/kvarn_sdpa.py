"""KVarN SDPA route: attention over KVarNKVCache regions.

install_kvarn_sdpa() rebinds ``scaled_dot_product_attention`` in every
loaded mlx_lm.models.* / mlx_vlm.models.* module (and both base modules,
so later imports inherit the wrapper). The wrapper only claims calls whose
keys are KVarNView handles; everything else passes through untouched.

Decode/verify (qL <= 4, plain causal masking) runs fused: the query is
WHT-rotated, kq.sdpa_decode_gqa_kvarn walks sink rows + sealed records +
live rows in one dispatch, and the output is un-rotated. When the cache
carries a precision tail, the body walk stops at the tail boundary
(n_attend, causal clamp lifted) and a second plain-fp16 attention over the
original-domain tail rows merges in through the log-sum-exp weights, so no
token is counted twice and the freshest context stays full fidelity.

Prefill (wider queries or array masks) materializes the rotated cache once
per call and runs stock mx.fast attention on it with the rotated query,
composing with the attn_hd512 wrapper's chunked-prefill routes.

Kill switches: GMLX_KVARN=0 disables the scheme at cache build time
(generation.py); GMLX_KVARN_SDPA=0 forces the materialize path for decode
as well (correct, slower). Both read at call time.
"""

from __future__ import annotations

import sys

import mlx.core as mx

from .envflags import env_bool
from .kvarn_cache import BatchKVarNKVCache, KVarNView

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
    doc = getattr(kq.sdpa_decode_gqa_kvarn, "__doc__", "") or ""
    if "n_attend" not in doc:
        return "mlx-kquant build predates kvarn precision-tail support"
    return None


def _lse_merge(out_a, lse_a, out_b, lse_b):
    """Numerically stable two-segment softmax merge in fp32."""
    a = out_a.astype(mx.float32)
    b = out_b.astype(mx.float32)
    m = mx.maximum(lse_a, lse_b)
    wa = mx.exp(lse_a - m)
    wb = mx.exp(lse_b - m)
    return (a * wa[..., None] + b * wb[..., None]) / (wa + wb)[..., None]


def _decode(q, cache, scale):
    import mlx_kquant as kq

    n = cache.offset
    qL = q.shape[2]
    t = min(cache.tail_len, n)
    n_body = n - t
    if 0 < n_body < qL:
        # A saturated tail can leave a sliver of body narrower than the
        # query block; widen the body to qL so the clamp lift stays valid.
        n_body = qL
        t = n - qL
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
    from .quantized_sdpa_fix import _registered_starts

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
            and q.dtype in (mx.float16, mx.bfloat16)
            and 1 <= q.shape[1] // cache.stage_k.shape[1] <= 16
            and q.shape[1] % cache.stage_k.shape[1] == 0
            and env_bool("GMLX_KVARN_SDPA", True)
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
        1 <= q.shape[2] <= 4
        and plain_mask
        and q.shape[0] == 1
        and q.dtype in (mx.float16, mx.bfloat16)
        and 1 <= q.shape[1] // cache.stage_k.shape[1] <= 16
        and q.shape[1] % cache.stage_k.shape[1] == 0
        and env_bool("GMLX_KVARN_SDPA", True)
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
