"""Quantized-KV SDPA fixes at both base-module seams.

Both mlx_lm.models.base and mlx_vlm.models.base carry the same
quantized_scaled_dot_product_attention. Three independent problems live
there, patched by one wrapper:

Mask broadcast at B>1 with GQA (upstream bug): when n_repeats > 1 it
reshapes queries to the grouped 5D layout (B, n_kv_heads, n_repeats, L, D),
so the score tensor is 5D -- but an array mask is applied as-is. A batched
left-pad mask (B, 1, 1, kv) left-pads to (1, B, 1, 1, kv) under broadcast,
its batch dim lands in the n_kv_heads slot, and the mx.where raises a
broadcast error. Every batched (B>1) masked GQA call with a quantized KV
cache crashes; B=1 works because a leading 1 broadcasts anywhere, which is
why single-stream kv4/kv8 measurements never saw it. The fix inserts one
axis: a 4D array mask becomes (B, 1, 1, L, kv) before the wrapped call
whenever the call is grouped. Masks without a real batch dim
((1, 1, L, kv), 2D, "causal", None) are unchanged in effect -- the expand
is a no-op under broadcasting -- so the wrapper applies it to every 4D
array mask rather than sniffing shapes.

Prefill-width flash path (perf): the stock body runs two unfused
mx.quantized_matmul passes plus a separate masked softmax. At decode
width (qL == 1) that is at parity with the fused fp16 path, but at
prefill chunk widths it is 3.4-4.2x slower per call, which put
``kv_bits: 8`` serving 1.6-1.9x behind fp16 on prefill wall time. When
the query width reaches GMLX_KV8_PREFILL_FLASH_MIN_L (default 8) the
wrapper dequantizes the K/V wire once (a bandwidth-bound copy, amortized
over the chunk) and runs fused mx.fast.scaled_dot_product_attention
instead; grouping never happens on that path, so the mask needs no
reshaping. Kill with GMLX_KV8_PREFILL_FLASH=0.

Fused decode route (perf): at qL == 1 the stock body is correct and at
parity with fp16, but when the mlx_kquant build carries quantized-KV
operands on sdpa_decode_gqa (dequant fused into the staged K/V walk,
one online-softmax dispatch) the same call runs ~1.4x faster at depth.
Decode-shaped calls (qL == 1, group 64 / bits 8, hd 64-512, gqa <= 16)
route there. Serving batches carry a boolean left-pad mask; the kernel
wants per-row ``starts`` instead, and the conversion must not inspect
mask CONTENTS -- a data-dependent check is a GPU sync, and under paced
serving that sync lands behind whole in-flight prefill chunks on the
in-order queue, stalling decode enough that the pacing loop throttles
prefill admission into a spiral (measured: agg 54.5 -> 40.5). Instead
the route trusts mask PROVENANCE: BatchQuantizedKVCache.make_mask is
patched to register each mask it creates against the cache's
left_padding (python-held row pads -- exactly the kernel's starts), and
the wrapper routes only masks found in that registry. Unregistered
array masks fall back to the stock body untouched. Kill with
GMLX_QSDPA_KQ=0.

Install is idempotent; GMLX_QSDPA_MASK_FIX=0 disables the whole wrapper.
Patched at each base module's symbol (their scaled_dot_product_attention
dispatchers read the module global at call time).
"""
from __future__ import annotations

import collections

import mlx.core as mx

from gmlx.envflags import env_bool, env_int

_installed = False

_MODULES = ("mlx_lm.models.base", "mlx_vlm.models.base")

_HD_OK = (64, 128, 256, 512)

# id -> (mask object, starts array). Populated by the patched
# BatchQuantizedKVCache.make_mask; the mask object is pinned so its id
# cannot be recycled while the entry lives. All layers of a step share
# one mask object, so lookups after the first are dict hits.
_STARTS_MEMO: collections.OrderedDict = collections.OrderedDict()
_STARTS_MEMO_CAP = 8


def _kq_q8_route():
    """The fused-route callable, or None when unavailable."""
    try:
        import mlx_kquant as kq
    except ImportError:
        return None
    # Metal-only kernel: on a cpu default device (--cpu runs, no-Metal
    # hosts) the stock path is correct, so the route declines.
    if mx.default_device().type != mx.DeviceType.gpu:
        return None
    fn = getattr(kq, "sdpa_decode_gqa", None)
    doc = (getattr(fn, "__doc__", "") or "") if fn is not None else ""
    if "k_scales" not in doc or "starts" not in doc:
        return None
    return fn


def _register_starts(mask, left_padding):
    """Record a mask's per-row starts at its creation site. No syncs:
    left_padding is the creator's python-held pad state, taken on trust."""
    _STARTS_MEMO[id(mask)] = (mask, left_padding.astype(mx.int32))
    if len(_STARTS_MEMO) > _STARTS_MEMO_CAP:
        _STARTS_MEMO.popitem(last=False)


def _registered_starts(mask):
    """Starts registered for this exact mask object, else None."""
    ent = _STARTS_MEMO.get(id(mask))
    if ent is not None and ent[0] is mask:
        return ent[1]
    return None


def _patch_make_mask() -> bool:
    """Register left-pad decode masks as BatchQuantizedKVCache creates
    them. Plain causal+left-pad masks only: windowed or right-padded
    calls pass through unregistered (their masks are not pure left-pad,
    and the stock body handles them)."""
    try:
        from mlx_vlm.models.cache import BatchQuantizedKVCache
    except ImportError:
        return False
    orig = BatchQuantizedKVCache.make_mask
    if getattr(orig, "_gmlx_starts_reg", False):
        return True

    def make_mask(self, N, return_array=False, **kwargs):
        mask = orig(self, N, return_array=return_array, **kwargs)
        if (
            N == 1
            and isinstance(mask, mx.array)
            and mask.dtype == mx.bool_
            and not any(v is not None for v in kwargs.values())
        ):
            _register_starts(mask, self.left_padding)
        return mask

    make_mask._gmlx_starts_reg = True
    make_mask._gmlx_orig = orig
    BatchQuantizedKVCache.make_mask = make_mask
    return True


def _make_fixed(orig, kq_fn):
    def _masked_grouped_qsdpa(queries, q_keys, q_values, scale, mask,
                              group_size: int = 64, bits: int = 8):
        if (
            queries.shape[-2] >= env_int("GMLX_KV8_PREFILL_FLASH_MIN_L", 8)
            and env_bool("GMLX_KV8_PREFILL_FLASH", True)
        ):
            # Cache fetches are lazy step-buffer slices. The Metal
            # dequantize kernel misreads a strided biases operand (mlx
            # 0.32.1); make each operand contiguous first.
            keys = mx.dequantize(*(mx.contiguous(t) for t in q_keys),
                                 group_size=group_size, bits=bits)
            values = mx.dequantize(*(mx.contiguous(t) for t in q_values),
                                   group_size=group_size, bits=bits)
            return mx.fast.scaled_dot_product_attention(
                queries, keys, values, scale=scale, mask=mask)
        if (
            kq_fn is not None
            and env_bool("GMLX_QSDPA_KQ", True)
            and group_size == 64
            and bits == 8
            and queries.ndim == 4
            and queries.shape[2] == 1
            and queries.shape[3] in _HD_OK
            and queries.dtype in (mx.float16, mx.bfloat16)
            and queries.shape[1] % q_keys[0].shape[1] == 0
            and 1 <= queries.shape[1] // q_keys[0].shape[1] <= 16
        ):
            starts = None
            routable = not isinstance(mask, mx.array)
            if (
                not routable
                and mask.dtype == mx.bool_
                and mask.ndim == 4
                and mask.shape[0] == queries.shape[0]
                and mask.shape[1] == 1
                and mask.shape[2] == 1
                and mask.shape[3] == q_keys[0].shape[2]
            ):
                starts = _registered_starts(mask)
                routable = starts is not None
            if routable:
                try:
                    kw, ks, kb = q_keys
                    vw, vs, vb = q_values
                    return kq_fn(
                        queries, kw, vw, float(scale), starts=starts,
                        k_scales=ks, k_biases=kb,
                        v_scales=vs, v_biases=vb)
                except Exception:
                    pass  # any unsupported layout -> stock fallback
        if (
            isinstance(mask, mx.array)
            and mask.ndim == 4
            and queries.ndim == 4
            and queries.shape[1] != q_keys[0].shape[-3]
        ):
            mask = mask[:, None]
        return orig(queries, q_keys, q_values, scale, mask,
                    group_size=group_size, bits=bits)

    _masked_grouped_qsdpa._gmlx_orig = orig
    _masked_grouped_qsdpa._gmlx_qsdpa_fix = True
    return _masked_grouped_qsdpa


def install_quantized_sdpa_mask_fix() -> bool:
    """Fix the 5D-scores vs 4D-batch-mask broadcast crash in upstream
    quantized SDPA (both base modules). Idempotent; no-op when
    GMLX_QSDPA_MASK_FIX=0. Returns True if the patch is active."""
    global _installed
    if not env_bool("GMLX_QSDPA_MASK_FIX", True):
        return False
    if _installed:
        return True
    import importlib

    kq_fn = _kq_q8_route()
    if kq_fn is not None:
        _patch_make_mask()
    patched = 0
    for name in _MODULES:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        cur = getattr(mod, "quantized_scaled_dot_product_attention", None)
        if cur is None:
            continue
        if getattr(cur, "_gmlx_qsdpa_fix", False):
            patched += 1
            continue
        mod.quantized_scaled_dot_product_attention = _make_fixed(cur, kq_fn)
        patched += 1
    if not patched:
        return False
    _installed = True
    return True
