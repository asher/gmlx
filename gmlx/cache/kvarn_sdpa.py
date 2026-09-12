"""KVarN SDPA route: attention over KVarNKVCache regions.

install_kvarn_sdpa() rebinds ``scaled_dot_product_attention`` in every
loaded mlx_lm.models.* / mlx_vlm.models.* module (and both base modules,
so later imports inherit the wrapper). The wrapper only claims calls whose
keys are KVarNView handles; everything else passes through untouched.

Decode (qL 1, plain causal masking) runs fused on the vector kernel: the
query is WHT-rotated, kq.sdpa_decode_gqa_kvarn walks sink rows + sealed
records + live rows in one dispatch, and the output is un-rotated. A
batch cache (BatchKVarNKVCache) decodes and verifies at up to
KVARN_BATCH_QL queries on the same kernel with a geometry per row: the
body leg walks each row's records from its own start to tail_cap short of
its own end, the tail leg covers each row's own tail window, and the mask
the stack shares is never consulted (it describes the fp16 layers'
right-justified shadow). Verify
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

import logging
import sys

import mlx.core as mx

from gmlx.envflags import env_bool
from .kvarn_cache import BatchKVarNKVCache, KVarNKVCache, KVarNView

_MODEL_PREFIXES = ("mlx_lm.models.", "mlx_vlm.models.", "gmlx.")
_BASE_MODULES = ("mlx_lm.models.base", "mlx_vlm.models.base")

_log = logging.getLogger(__name__)

# Widest query block the batched kernel route serves (the decode kernels'
# qL limit); a wider batched round materializes.
KVARN_BATCH_QL = 4

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


_row_ends_result = None


def kvarn_row_ends_ok() -> bool:
    """Whether the installed mlx-kquant takes per-row ``ends`` on the decode
    ops (0.4.9 or later), memoized. The bound ops expose no signature, so
    the version string decides."""
    global _row_ends_result
    if _row_ends_result is None:
        _row_ends_result = (_probe_row_ends(),)
    return _row_ends_result[0]


def _probe_row_ends() -> bool:
    try:
        import mlx_kquant as kq
    except ImportError:
        return False
    return _version_tuple(getattr(kq, "__version__", "")) >= (0, 4, 9)


def _version_tuple(text) -> tuple:
    """The leading three integers of a version string, missing or
    non-numeric pieces as 0."""
    parts = []
    for piece in str(text).split(".")[:3]:
        digits = ""
        for ch in piece:
            if not ch.isdigit():
                break
            digits += ch
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


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

    import mlx_kquant as kq

    c = KVarNKVCache(tail_tokens=0)
    kv = mx.zeros((1, 1, GROUP + 2 * ql, d), mx.float16)
    c.update_and_fetch(kv, kv)
    for gqa in (16, 8, 4, 2, 1):
        if gqa * ((ql + 1) // 2) > 32:
            continue
        q = mx.zeros((1, gqa, ql, d), mx.float16)
        try:
            # Both kernels of the split: records, then the fp16 tail twin.
            mx.eval(_decode_vector_one(q, c, 1.0))
            mx.eval(kq.sdpa_decode_gqa(q, kv, kv, 1.0))
        except Exception:
            continue
        return _tg_threads(gqa, ql)
    return 0


def _tg_threads(gqa: int, qL: int) -> int:
    """Threadgroup width of one fused decode dispatch (kq_sdpa geometry)."""
    return 32 * gqa * ((qL + 1) // 2)


def _vector_chunks(gqa: int, qL: int, d: int):
    """Head chunks per kv head for the vector kernel: the smallest split of
    the GQA group whose threadgroup fits the probed cap, or None when a
    single head does not. Each chunk re-sweeps the keys; a GPU that caps
    the pipeline below the fold (M1-class at gqa 16) keeps the fused route
    at the cost of the extra sweeps."""
    limit = _tg_limit(d, qL > 1)
    for n in (1, 2, 4, 8, 16):
        if gqa % n == 0 and _tg_threads(gqa // n, qL) <= limit:
            return n
    return None


def _split_heads(q, kvh: int, n: int):
    """The q head axis split into n contiguous GQA sub-groups per kv head."""
    b, hq, qL, d = q.shape
    g = hq // kvh // n
    q5 = q.reshape(b, kvh, n, g, qL, d)
    return [mx.contiguous(q5[:, :, i]).reshape(b, kvh * g, qL, d) for i in range(n)]


def _join_heads(outs, kvh: int, hq: int):
    b, _, qL, d = outs[0].shape
    n = len(outs)
    out = mx.concatenate([o.reshape(b, kvh, 1, hq // kvh // n, qL, d) for o in outs], axis=2)
    return out.reshape(b, hq, qL, d)


def _fa_available() -> bool:
    import mlx_kquant as kq

    return hasattr(kq, "sdpa_fa_verify_kvarn") and env_bool("GMLX_KVARN_FA", True)


_fa_ok: dict[int, bool] = {}


def _fa_runs(d: int) -> bool:
    """Whether this GPU runs the verify kernels at head_dim d, probed once
    per head_dim on a two-query fold; a pipeline that caps below the tile
    raises at eval, and the verify width then takes the vector route."""
    if d not in _fa_ok:
        _fa_ok[d] = _probe_fa(d)
    return _fa_ok[d]


def _probe_fa(d: int) -> bool:
    if kvarn_ops_missing():
        return False
    from .kvarn_cache import GROUP

    c = KVarNKVCache(tail_tokens=0)
    kv = mx.zeros((1, 1, GROUP + 4, d), mx.float16)
    c.update_and_fetch(kv, kv)
    try:
        mx.eval(_decode_fa(mx.zeros((1, 1, 2, d), mx.float16), c, 1.0))
    except Exception:
        return False
    return True


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
    return (
        _fa_chunks(gqa, q.shape[2], _fa_row_cap(q.shape[-1])) is not None
        and _fa_runs(q.shape[-1])
    )


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
    return qL <= 4 and _vector_chunks(gqa, qL, q.shape[-1]) is not None


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
    kvh = cache.stage_k.shape[1]
    n = _vector_chunks(q.shape[1] // kvh, q.shape[2], q.shape[-1]) or 1
    if n == 1:
        return _decode_vector_one(q, cache, scale)
    outs = [_decode_vector_one(qc, cache, scale) for qc in _split_heads(q, kvh, n)]
    return _join_heads(outs, kvh, q.shape[1])


def _decode_vector_one(q, cache, scale):
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


def _decode_batch(q, cache, scale, starts, ends):
    kvh = cache.stage_k.shape[1]
    n = _vector_chunks(q.shape[1] // kvh, q.shape[2], q.shape[-1]) or 1
    if n == 1:
        return _decode_batch_one(q, cache, scale, starts, ends)
    outs = [
        _decode_batch_one(qc, cache, scale, starts, ends)
        for qc in _split_heads(q, kvh, n)
    ]
    return _join_heads(outs, kvh, q.shape[1])


def _decode_batch_one(q, cache, scale, starts, ends):
    """Batched decode and verify (qL 1 to KVARN_BATCH_QL) with a geometry
    per row. The body leg walks each row's records from its start to
    tail_cap short of its end, every query seeing every body key; the tail
    leg walks each row's last min(ends - starts, tail_cap) tail rows with
    the causal clamp at the row's end; the two merge through their LSEs.
    A row admitted deep into an older batch, or one still shorter than
    the tail, has an empty body: the kernel writes an empty partial that
    carries no weight in the merge. Without a tail the body leg runs
    alone under the per-row causal clamp."""
    import mlx_kquant as kq

    q_rot = kq.kvarn_rotate(q)
    body_args = (
        q_rot,
        cache.codes_k,
        cache.axes_k,
        cache.codes_v,
        cache.axes_v,
        cache.stage_k,
        cache.stage_v,
        cache._pos,
        scale,
        cache.k_bits,
        cache.v_bits,
    )
    if cache.tail_cap == 0:
        return kq.kvarn_rotate(
            kq.sdpa_decode_gqa_kvarn(*body_args, starts=starts, ends=ends)
        )
    own = starts is cache.starts_mx and ends is cache.ends_mx
    if own:
        tail_starts = cache.tail_leg_starts_mx
    else:
        tail_starts = cache.tail_ends_mx - mx.minimum(
            mx.maximum(ends - starts, 0), cache.tail_cap
        ).astype(mx.int32)
    tail_k = cache.tail_k.astype(q.dtype)
    tail_v = cache.tail_v.astype(q.dtype)
    if own and all(
        e - cache.tail_cap <= s for s, e in zip(cache.starts, cache.ends)
    ):
        # Every row sits inside its tail window: no records to read.
        return kq.sdpa_decode_gqa(
            q, tail_k, tail_v, scale, starts=tail_starts, ends=cache.tail_ends_mx
        )
    # With at least qL tail rows every body key precedes every query's
    # causal position; a shorter tail keeps the per-row clamp on the body.
    body, lse_b = kq.sdpa_decode_gqa_kvarn(
        *body_args,
        starts=starts,
        ends=ends,
        tail_rows=cache.tail_cap,
        full_visibility=cache.tail_cap >= q.shape[2],
        return_lse=True,
    )
    tail, lse_t = kq.sdpa_decode_gqa(
        q,
        tail_k,
        tail_v,
        scale,
        starts=tail_starts,
        ends=cache.tail_ends_mx,
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


def _ragged_mask(starts, ends, n, qL):
    """Per-row causal bool mask [B, 1, qL, n] over the physical geometry,
    for the materialize fallback: key p is visible to query i of row b
    when starts[b] <= p <= ends[b] - qL + i."""
    t = mx.arange(n, dtype=mx.int32)[None, None, :]
    last = (ends[:, None, None] - qL) + mx.arange(qL, dtype=mx.int32)[None, :, None]
    return ((t >= starts[:, None, None]) & (t <= last))[:, None]


_materialize_noted: set = set()


def _note_materialize(qL: int, cache) -> None:
    """One log line per reason a batched call left the kernel route."""
    if not _route_enabled():
        why = "GMLX_KVARN_SDPA=0"
    elif not kvarn_row_ends_ok():
        why = "mlx-kquant has no per-row ends (0.4.9 or later)"
    elif qL > KVARN_BATCH_QL:
        why = f"{qL} queries (the kernels take up to {KVARN_BATCH_QL})"
    else:
        why = "head geometry outside the fused route"
    if why in _materialize_noted:
        return
    _materialize_noted.add(why)
    _log.info(
        "[kvarn] batched attention over %d rows materializes: %s",
        cache.stage_k.shape[0],
        why,
    )


def kvarn_attention(q, cache, scale, mask, sinks=None, starts=None, ends=None):
    """Attention over a kvarn cache. A batch cache carries its own
    geometry (per-row physical starts and ends as int32 vectors);
    ``starts``/``ends`` may pass those same vectors explicitly, and the
    mask is never consulted for it: the stack's shared mask describes the
    fp16 layers' right-justified shadow, which the physical rows do not
    share. Decode and verify at up to KVARN_BATCH_QL queries run fused;
    anything wider, and the kill switch, materialize under a per-row
    mask."""
    if sinks is not None:
        raise RuntimeError(
            "[kvarn] attention sinks reached the kvarn route; this arch "
            "should have been declined at cache build time."
        )
    if isinstance(cache, BatchKVarNKVCache):
        qL = q.shape[2]
        if starts is None:
            starts = cache.starts_mx
        if ends is None:
            ends = cache.ends_mx
        if (
            1 <= qL <= KVARN_BATCH_QL
            and q.shape[0] == cache.stage_k.shape[0]
            and kvarn_row_ends_ok()
            and _fused_ok(q, cache)
        ):
            return _decode_batch(q, cache, float(scale), starts, ends)
        _note_materialize(qL, cache)
        return _prefill(
            q, cache, float(scale), _ragged_mask(starts, ends, cache._pos, qL)
        )
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
