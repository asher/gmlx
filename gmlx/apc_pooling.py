"""APC participation + safe KV quantization for PoolingCache stacks.

Stock mlx-vlm's APC exact store supports the cache kinds it knows
(KVCache/RotatingKVCache/ChunkedKVCache/ArraysCache/CacheList/tuple) and
rejects anything else, so a deepseek4 row snapshot -- whose CacheLists carry
PoolingCaches -- always came back None and was never stored (no serve
warm-starts, memory or disk). The installers below add PoolingCache arms to
the in-memory clone and the disk shard writer/reader; packed (quantized)
pools are stored packed, and BatchPoolingCache.merge already reads through
the dequantizing ``pooled`` property, so restored entries splice into
batches unchanged.

``install_safe_kv_quantization`` replaces mlx-lm's per-step
``maybe_quantize_kv_cache``: the stock version calls ``to_quantized`` on any
cache exposing it, which raises NotImplementedError on (Batch)RotatingKVCache
mid-generation -- the serve-side KV_BITS crash on sliding-window arches.
The replacement packs quantizable PoolingCaches in place (conversion-at-start,
including pools that already hold rows), skips rotating caches, and keeps the
stock KVCache -> QuantizedKVCache conversion for standard models.
"""

from __future__ import annotations

import importlib
import json
import logging

_log = logging.getLogger(__name__)


def install_pooling_apc_support() -> None:
    """Add PoolingCache arms to the APC exact-store clone and disk paths.

    Idempotent. A missing upstream symbol raises: silently skipping would
    leave serve APC dead for every pooling-cache model (deepseek-v4), which
    is exactly the drift this install exists to prevent (see
    gmlx.upstream_seams).
    """
    apc = importlib.import_module("mlx_vlm.apc")
    if getattr(apc, "_kq_pooling_apc", False):
        return
    try:
        _copy = apc._copy_mlx_array
        stock_clone = apc._clone_cache_entry_for_apc
        stock_supports = apc._cache_entry_supports_exact_apc
        stock_merge = apc._merge_exact_cache_entries
        disk_cls = apc.DiskBlockStore
        stock_snap = disk_cls._snapshot_exact_cache_entry
        stock_load = disk_cls._load_exact_cache_entry
        _read = apc._read_safetensors_tensor
        stock_dtype_info = apc._safetensors_dtype_info
    except AttributeError as e:
        raise RuntimeError(
            "APC pooling support cannot install: mlx-vlm apc surface "
            f"changed ({e}) - re-audit against the pinned seams "
            "(gmlx.upstream_seams)") from e

    from .deepseek_v4_cache import BatchPoolingCache, PoolingCache

    def supports_exact(c):
        # Without this arm model_apc_mode resolves to None for pooling
        # stacks and APC never engages, making the clone/disk arms moot.
        return isinstance(c, PoolingCache) or stock_supports(c)

    def merge_exact_entries(entries, prefix_lens):
        # Warm-batch adoption: a restored exact hit is merged per entry into
        # batch caches; without this arm the merge returns None and the
        # engine silently falls back to a cold prefill after the hit.
        if entries and all(isinstance(c, PoolingCache) for c in entries):
            return BatchPoolingCache.merge(entries)
        return stock_merge(entries, prefix_lens)

    def dtype_info(dtype):
        # Packed pooled planes are uint32; the stock shard reader only
        # decodes float dtypes.
        if dtype == "U32":
            return apc.np.dtype("<u4"), apc.mx.uint32, None
        return stock_dtype_info(dtype)

    def clone_entry(c, *, min_capacity_tokens, eval_targets):
        if not isinstance(c, PoolingCache):
            return stock_clone(
                c,
                min_capacity_tokens=min_capacity_tokens,
                eval_targets=eval_targets,
            )
        out = PoolingCache(c.ratio)
        out.quantizable = c.quantizable
        out._qbits = c._qbits
        out._qgroup = c._qgroup
        if c.buf_kv is not None and c.remainder > 0:
            # Full staging buffers (capacity == ratio rows) so the clone
            # satisfies accumulate_windows' invariants as-is.
            out.buf_kv = _copy(c.buf_kv)
            out.buf_gate = _copy(c.buf_gate)
            out.remainder = c.remainder
            eval_targets.extend([out.buf_kv, out.buf_gate])
        if c._plen > 0:
            if isinstance(c._pbuf, tuple):
                bufs = tuple(_copy(b[:, : c._plen]) for b in c._pbuf)
                out._pbuf = bufs
                eval_targets.extend(bufs)
            else:
                out._pbuf = _copy(c._pbuf[:, : c._plen])
                eval_targets.append(out._pbuf)
            out._plen = c._plen
        if c._prev_kv is not None:
            out._prev_kv = _copy(c._prev_kv)
            out._prev_gate = _copy(c._prev_gate)
            eval_targets.extend([out._prev_kv, out._prev_gate])
        return out

    def snap_entry(self, c, prefix, arrays, metadata):
        if not isinstance(c, PoolingCache):
            before = set(arrays)
            ok = stock_snap(self, c, prefix, arrays, metadata)
            if ok:
                # mx.save_safetensors rejects zero-size arrays (deepseek4's
                # DSA local caches carry zero-width values); spill them to
                # metadata and synthesize zeros on load.
                for name in set(arrays) - before:
                    a = arrays[name]
                    if a.size == 0:
                        del arrays[name]
                        metadata[f"{name}__kq_empty"] = json.dumps({
                            "dtype": str(a.dtype).split(".")[-1],
                            "shape": list(a.shape),
                        })
            return ok
        metadata[f"{prefix}_kind"] = "kq_pooling"
        metadata[f"{prefix}_ratio"] = str(int(c.ratio))
        metadata[f"{prefix}_remainder"] = str(int(c.remainder))
        metadata[f"{prefix}_plen"] = str(int(c._plen))
        metadata[f"{prefix}_qbits"] = str(int(c._qbits or 0))
        metadata[f"{prefix}_qgroup"] = str(int(c._qgroup))
        metadata[f"{prefix}_quantizable"] = "1" if c.quantizable else "0"
        if c.buf_kv is not None and c.remainder > 0:
            arrays[f"{prefix}_rk"] = c.buf_kv[:, : c.remainder]
            arrays[f"{prefix}_rg"] = c.buf_gate[:, : c.remainder]
        if c._plen > 0:
            if isinstance(c._pbuf, tuple):
                for name, b in zip(("pq", "ps", "pb"), c._pbuf):
                    arrays[f"{prefix}_{name}"] = b[:, : c._plen]
            else:
                arrays[f"{prefix}_p"] = c._pbuf[:, : c._plen]
        if c._prev_kv is not None:
            arrays[f"{prefix}_lk"] = c._prev_kv
            arrays[f"{prefix}_lg"] = c._prev_gate
        return True

    def load_entry(
        self,
        path,
        tensor_entries,
        metadata,
        data_start,
        prefix,
        *,
        min_capacity_tokens,
        eval_targets,
    ):
        if metadata.get(f"{prefix}_kind") != "kq_pooling":
            for k, v in metadata.items():
                if k.endswith("__kq_empty"):
                    name = k[: -len("__kq_empty")]
                    if name not in tensor_entries:
                        try:
                            tensor_entries[name] = {
                                "__kq_empty__": True, **json.loads(v)
                            }
                        except (TypeError, ValueError):
                            pass
            return stock_load(
                self,
                path,
                tensor_entries,
                metadata,
                data_start,
                prefix,
                min_capacity_tokens=min_capacity_tokens,
                eval_targets=eval_targets,
            )
        try:
            ratio = int(metadata[f"{prefix}_ratio"])
            remainder = int(metadata.get(f"{prefix}_remainder", "0"))
            plen = int(metadata.get(f"{prefix}_plen", "0"))
            qbits = int(metadata.get(f"{prefix}_qbits", "0"))
            qgroup = int(metadata.get(f"{prefix}_qgroup", "64"))
        except (KeyError, TypeError, ValueError):
            return None
        out = PoolingCache(ratio)
        out.quantizable = metadata.get(f"{prefix}_quantizable", "1") != "0"
        if qbits:
            out._qbits = qbits
            out._qgroup = qgroup
        if remainder > 0:
            rk = tensor_entries.get(f"{prefix}_rk")
            rg = tensor_entries.get(f"{prefix}_rg")
            if rk is None or rg is None:
                return None
            rk = _read(path, data_start, rk)
            rg = _read(path, data_start, rg)
            if rk is None or rg is None:
                return None
            # Rebuild the staging buffers through the normal append path
            # (allocates ratio-row capacity); remainder < ratio, so no
            # window completes and no pooled row is emitted.
            out.accumulate_windows(rk, rg, 0)
            out._undo = None
            eval_targets.extend([out.buf_kv, out.buf_gate])
        if plen > 0:
            if qbits:
                bufs = []
                for name in ("pq", "ps", "pb"):
                    ent = tensor_entries.get(f"{prefix}_{name}")
                    if ent is None:
                        return None
                    arr = _read(path, data_start, ent)
                    if arr is None:
                        return None
                    bufs.append(arr)
                out._pbuf = tuple(bufs)
                eval_targets.extend(bufs)
            else:
                ent = tensor_entries.get(f"{prefix}_p")
                if ent is None:
                    return None
                arr = _read(path, data_start, ent)
                if arr is None:
                    return None
                out._pbuf = arr
                eval_targets.append(arr)
            out._plen = plen
        lk = tensor_entries.get(f"{prefix}_lk")
        lg = tensor_entries.get(f"{prefix}_lg")
        if lk is not None and lg is not None:
            lk = _read(path, data_start, lk)
            lg = _read(path, data_start, lg)
            if lk is None or lg is None:
                return None
            out._prev_kv = lk
            out._prev_gate = lg
            eval_targets.extend([lk, lg])
        return out

    def read_tensor(path, data_start, entry):
        if isinstance(entry, dict) and entry.get("__kq_empty__"):
            try:
                return apc.mx.zeros(
                    entry["shape"], getattr(apc.mx, entry["dtype"])
                )
            except (AttributeError, KeyError, TypeError):
                return None
        return _read(path, data_start, entry)

    apc._cache_entry_supports_exact_apc = supports_exact
    apc._merge_exact_cache_entries = merge_exact_entries
    apc._read_safetensors_tensor = read_tensor
    apc._clone_cache_entry_for_apc = clone_entry
    apc._safetensors_dtype_info = dtype_info
    disk_cls._snapshot_exact_cache_entry = snap_entry
    disk_cls._load_exact_cache_entry = load_entry
    apc._kq_pooling_apc = True


def install_safe_kv_quantization() -> None:
    """Rotating-safe, pool-aware replacement for maybe_quantize_kv_cache.

    Patches the mlx-lm module attribute and mlx-vlm's imported binding so
    both the lm and vlm generate loops resolve the replacement.
    """
    # Not `from mlx_lm import generate`: the package exports a `generate`
    # function under that name, and setting attributes on it is a silent
    # no-op against the module.
    lm_gen = importlib.import_module("mlx_lm.generate")

    if getattr(lm_gen, "_kq_safe_kv_quant", False):
        return
    from .cache_compat import cache_types
    from .deepseek_v4_cache import PoolingCache

    def _pack_pools(c, group_size, bits):
        subs = getattr(c, "caches", None)
        if subs is not None:
            for sub in subs:
                _pack_pools(sub, group_size, bits)
        elif isinstance(c, PoolingCache) and c.quantizable:
            c.quantize_storage(group_size=group_size, bits=bits)

    def safe_maybe_quantize(prompt_cache, quantized_kv_start, kv_group_size, kv_bits):
        if kv_bits is None:
            return
        # Per call, not at install: mlx-vlm (a second rotating-class origin
        # since 0.6.4) may load after this installer runs.
        rotating = (cache_types("RotatingKVCache")
                    + cache_types("BatchRotatingKVCache"))
        bits, group = int(kv_bits), int(kv_group_size)
        for e, c in enumerate(prompt_cache):
            _pack_pools(c, group, bits)
            if isinstance(c, rotating) or getattr(c, "caches", None) is not None:
                continue
            if hasattr(c, "to_quantized") and c.offset >= quantized_kv_start:
                prompt_cache[e] = c.to_quantized(group_size=group, bits=bits)

    lm_gen.maybe_quantize_kv_cache = safe_maybe_quantize
    lm_gen._kq_safe_kv_quant = True
    # No fallback: without the rebind, stock KV quantization corrupts
    # rotating/pooling caches on the VLM serve path.
    # importlib, not `import mlx_vlm.generate as ...`: the mlx_vlm package
    # exports a `generate` function that shadows the submodule attribute.
    vlm_generate = importlib.import_module("mlx_vlm.generate")
    vlm_common = importlib.import_module("mlx_vlm.generate.common")

    if hasattr(vlm_common, "mlx_maybe_quantize_kv_cache"):
        # <= 0.6.3: the stock wrapper delegates its non-turboquant tail to
        # this mlx-lm alias; rebinding it covers every caller.
        vlm_common.mlx_maybe_quantize_kv_cache = safe_maybe_quantize
        return
    # >= 0.6.4: the tail is inlined (and lost the rotating exclusion). The
    # AR engine resolves "maybe_quantize_kv_cache" from the mlx_vlm.generate
    # package at call time (_generate_module_override), so publish a wrapper
    # there that keeps the stock turboquant arm and routes the affine tail
    # through the safe path.
    stock = getattr(vlm_common, "maybe_quantize_kv_cache", None)
    turbo_on = getattr(vlm_common, "turboquant_enabled", None)
    if stock is None:
        raise RuntimeError(
            "mlx_vlm.generate.common.maybe_quantize_kv_cache is gone - "
            "re-audit against the pinned seams (gmlx.upstream_seams)")

    def safe_vlm_maybe_quantize(prompt_cache, quantized_kv_start,
                                kv_group_size, kv_bits, **kwargs):
        if kv_bits is None:
            return
        scheme = kwargs.get(
            "kv_quant_scheme",
            getattr(vlm_common, "DEFAULT_KV_QUANT_SCHEME", None))
        if turbo_on is not None and turbo_on(kv_bits, scheme):
            return stock(prompt_cache, quantized_kv_start, kv_group_size,
                         kv_bits, **kwargs)
        return safe_maybe_quantize(
            prompt_cache, quantized_kv_start, kv_group_size, kv_bits)

    vlm_generate.maybe_quantize_kv_cache = safe_vlm_maybe_quantize
    vlm_common.maybe_quantize_kv_cache = safe_vlm_maybe_quantize


def install_pooled_prompt_kv_quant() -> None:
    """Honor serve kv_bits on pooling-cache models at prompt-batch build.

    With kv_bits set, ``PromptProcessingBatch.__init__`` routes every cache
    through ``_make_cache``/``to_batch_cache``, which has no pooled arm: one
    request on a pooling-cache model (deepseek-v4) raises before prefill.
    Mirror the CLI ``generate()`` intercept instead: drop kv_bits from the
    stock init (a single cold row then builds the model's own
    single-sequence caches) and arm packed at-rest pooled storage on the
    result. Batched rows and MTP spec caches keep stock behavior (fp16
    pools) with a one-shot note; non-pooling models are untouched.
    Idempotent. Kill switch: GMLX_POOLED_KV_QUANT=0.
    """
    import os

    from mlx_vlm.generate import ar as _ar

    ppb = _ar.PromptProcessingBatch
    if getattr(ppb, "_kq_pooled_prompt_kv", False):
        return
    if os.environ.get("GMLX_POOLED_KV_QUANT", "1") == "0":
        return

    from .deepseek_v4_cache import PoolingCache

    def _has_pools(model) -> bool:
        cached = getattr(model, "_kq_has_pools", None)
        if cached is not None:
            return cached
        make = getattr(model, "make_cache", None)
        found = False
        if callable(make):
            try:
                stack = list(make() or [])
            except Exception:
                stack = []
            while stack:
                c = stack.pop()
                subs = getattr(c, "caches", None)
                if subs is not None:
                    stack.extend(subs)
                elif isinstance(c, PoolingCache):
                    found = True
                    break
        try:
            model._kq_has_pools = found
        except Exception:
            pass
        return found

    _orig_init = ppb.__init__
    _noted = [False]
    _skipped = [False]

    def _skip_note(msg, *fmt):
        if not _skipped[0]:
            _skipped[0] = True
            _log.warning(msg, *fmt)

    def _pooled_init(self, *args, **kwargs):
        bits = kwargs.get("kv_bits")
        model = kwargs.get("model") or (args[0] if args else None)
        if bits is None or model is None or not _has_pools(model):
            return _orig_init(self, *args, **kwargs)
        _orig_init(self, *args, **dict(kwargs, kv_bits=None))
        rows = kwargs.get("input_ids") or (args[2] if len(args) > 2 else ())
        if len(rows) != 1 or kwargs.get("draft_kind") is not None:
            _skip_note(
                "kv_bits on a pooling-cache model: packed pools are B=1 "
                "baseline only; this batch keeps fp16 pools")
            return
        try:
            fbits = float(bits)
        except (TypeError, ValueError):
            return
        if fbits != int(fbits):
            _skip_note(
                "kv_bits=%s: pooled storage needs an integer affine width; "
                "pools stay fp16", bits)
            return
        from .generation import quantize_pooled_caches

        group = int(kwargs.get("kv_group_size") or 64)
        n = quantize_pooled_caches(self.prompt_cache, int(fbits), group)
        if n and not _noted[0]:
            _noted[0] = True
            print(
                f"[kv] {int(fbits)}-bit pooled KV cache ({n} pools; "
                "sliding windows stay fp16)",
                flush=True,
            )

    ppb.__init__ = _pooled_init
    ppb._kq_pooled_prompt_kv = True
