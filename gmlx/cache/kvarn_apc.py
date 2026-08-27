"""kvarn arms for the APC exact tier (memory + disk).

install_kvarn_apc() chains kvarn arms onto the same seams
apc_pooling arms for pooling caches:

- supports/clone/merge: KVarNKVCache entries store and warm-adopt through
  the exact tier. A B=1 BatchKVarNKVCache (the lone-batch store shortcut
  hands the live batch caches over verbatim) normalizes to single-stream
  via extract(0). Warm-batch adoption is single-row only: multi-row exact
  merges return None and the engine falls back to a cold prefill, because
  mixed warm/cold prefill right-pads and kvarn cannot roll per-row right
  padding into sealed records.
- disk: kind "kq_kvarn" persists the full state tuple plus meta_state.
  The load arm fail-closes (None -> clean cold miss) on any layout or
  version mismatch instead of raising -- meta_state's own version check
  raises ValueError, which must never escape the lookup path.
- model_apc_mode: kvarn caches are built by the serve installers after
  model construction, so the stock probe of model.make_cache() sees plain
  fp16 caches and resolves "block" -- a tier kvarn cannot serve (128-token
  records do not split into 16-token blocks). Models stamped by the kvarn
  serve gate (stamp_model) resolve "exact" instead.

Entry isolation: the gmlx APC manager salts exact-tier extra hashes with
the kvarn wire config (apply_kvarn_salt, called at residency's post-load
manager pairing and gated on an actual-conversion probe), so entries
written under one scheme/width/tail config never warm-adopt under
another -- the disk namespace is the model path and carries no scheme.
"""

from __future__ import annotations

import hashlib
import importlib
import os

_FLAG = "_gmlx_kvarn_apc"
_MODE_STAMP = "_gmlx_kvarn_apc_exact"
_CONVERTS_ATTR = "_gmlx_kvarn_converts"
_STATE_ARITY = 10


def kvarn_apc_installed() -> bool:
    apc = importlib.import_module("mlx_vlm.apc")
    return bool(getattr(apc, _FLAG, False))


def stamp_model(model) -> None:
    """Mark a model as serving kvarn KV so model_apc_mode resolves exact.
    Stamps both the model and its language model: the live call sites
    disagree on which object they pass (the engine passes the top-level
    model, dispatch and spec_engine the language model)."""
    for obj in (model, getattr(model, "language_model", None)):
        if obj is not None:
            try:
                setattr(obj, _MODE_STAMP, True)
            except Exception:
                pass


def kvarn_model_converts(model) -> bool:
    """True when a kvarn boot converts at least one of this model's cache
    layers. Structural only -- the scheme window belongs to the caller
    (residency's env window, or the per-request kwarg in the serve gate).
    Computed once and stashed on the model so the salt gate and the serve
    gate read one answer; counts cache types only and retains nothing
    from make_cache (this runs on the residency load thread). Any failure
    means False: don't stamp, don't salt."""
    cached = getattr(model, _CONVERTS_ATTR, None)
    if cached is not None:
        return bool(cached)
    val = False
    try:
        from gmlx.gen.generation import kvarn_unsupported

        if kvarn_unsupported(model) is None:
            lm = getattr(model, "language_model", None) or model
            make = getattr(lm, "make_cache", None)
            if make is None:
                # No make_cache: upstream builds plain KV for every
                # layer, and serve converts all of them.
                val = True
            else:
                from .kvarn_cache import convertible_kv_types

                conv = convertible_kv_types()
                val = any(type(c) in conv for c in make())
    except Exception:
        val = False
    try:
        setattr(model, _CONVERTS_ATTR, val)
    except Exception:
        pass
    return val


def apply_kvarn_salt(manager, model) -> None:
    """Set the manager's exact-tier wire salt at residency's post-load
    pairing, gated on the model actually converting: a kvarn-window boot
    of a zero-conversion arch (deepseek4, recurrent_gemma) runs pure fp16
    caches, and salting its entries would cold-miss every cross-boot
    lookup. Failures leave the salt at its XOR-identity default."""
    try:
        if manager is None or model is None:
            return
        salt = kvarn_entry_salt()
        if salt and kvarn_model_converts(model):
            manager._exact_extra_salt = salt
    except Exception:
        pass


def kvarn_entry_salt() -> int:
    """Exact-tier extra-hash salt for the current env window's kvarn wire
    config, 0 outside a kvarn window (XOR identity)."""
    if os.environ.get("KV_QUANT_SCHEME", "") != "kvarn":
        return 0
    from .kvarn_cache import KVarNKVCache
    from .kvarn_serve import _serve_widths_and_tail

    k_bits, v_bits, tail = _serve_widths_and_tail()
    key = f"kvarn:{KVarNKVCache.kvarn_layout_version}:{k_bits}:{v_bits}:{tail}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")


def install_kvarn_apc() -> None:
    """Add KVarNKVCache arms to the APC exact-store clone, merge, mode and
    disk paths. Idempotent; a missing upstream symbol raises (the
    apc_pooling drift posture -- see gmlx.upstream.seams)."""
    apc = importlib.import_module("mlx_vlm.apc")
    if getattr(apc, _FLAG, False):
        return
    try:
        _copy = apc._copy_mlx_array
        stock_clone = apc._clone_cache_entry_for_apc
        stock_supports = apc._cache_entry_supports_exact_apc
        stock_merge = apc._merge_exact_cache_entries
        disk_cls = apc.DiskBlockStore
        stock_snap = disk_cls._snapshot_exact_cache_entry
        stock_load = disk_cls._load_exact_cache_entry
        stock_dtype_info = apc._safetensors_dtype_info
        stock_mode = apc.model_apc_mode
        _read = apc._read_safetensors_tensor
    except AttributeError as e:
        raise RuntimeError(
            "APC kvarn support cannot install: mlx-vlm apc surface "
            f"changed ({e}) - re-audit against the pinned seams "
            "(gmlx.upstream.seams)"
        ) from e

    from .kvarn_cache import (
        BatchKVarNKVCache,
        KVarNKVCache,
        ensure_registered,
    )

    ensure_registered()

    def supports_exact(c):
        return isinstance(c, KVarNKVCache) or stock_supports(c)

    def clone_entry(c, *, min_capacity_tokens, eval_targets):
        # min_capacity_tokens is a K/V-padding hint; kvarn grows its own
        # region buffers, so it is ignored here.
        if isinstance(c, BatchKVarNKVCache):
            if c.left_padding.shape[0] != 1:
                return None
            out = c.extract(0)
            eval_targets.extend(out.state)
            return out
        if not isinstance(c, KVarNKVCache):
            return stock_clone(
                c,
                min_capacity_tokens=min_capacity_tokens,
                eval_targets=eval_targets,
            )
        state = tuple(_copy(a) for a in c.state)
        out = type(c).from_state(state, c.meta_state)
        eval_targets.extend(state)
        return out

    def merge_exact_entries(entries, prefix_lens):
        if entries and any(isinstance(c, KVarNKVCache) for c in entries):
            if len(entries) != 1 or not isinstance(entries[0], KVarNKVCache):
                # Multi-row (or mixed warm/cold) adoption right-pads the
                # suffix prefill; None falls back to a cold prefill
                # instead of crashing at finalize.
                return None
            # A restart's first request can warm-adopt before any
            # _make_cache kvarn build armed the SDPA route.
            from .kvarn_sdpa import install_kvarn_sdpa

            install_kvarn_sdpa()
            return BatchKVarNKVCache.merge(list(entries))
        return stock_merge(entries, prefix_lens)

    def dtype_info(dtype):
        # kvarn codes are uint32; the stock shard reader only decodes
        # float dtypes. Chains after apc_pooling's identical arm.
        if dtype == "U32":
            return apc.np.dtype("<u4"), apc.mx.uint32, None
        return stock_dtype_info(dtype)

    def mode_wrap(language_model):
        # Check-down: stamp_model stamps both objects, but a caller may
        # hand over a wrapper whose language model carries the stamp
        # (there is no walking up -- children hold no parent reference).
        if getattr(language_model, _MODE_STAMP, False) or getattr(
            getattr(language_model, "language_model", None),
            _MODE_STAMP,
            False,
        ):
            return "exact"
        return stock_mode(language_model)

    def snap_entry(self, c, prefix, arrays, metadata):
        if not isinstance(c, KVarNKVCache):
            return stock_snap(self, c, prefix, arrays, metadata)
        metadata[f"{prefix}_kind"] = "kq_kvarn"
        metadata[f"{prefix}_meta"] = ",".join(c.meta_state)
        for j, a in enumerate(c.state):
            arrays[f"{prefix}_s{j}"] = a
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
        if metadata.get(f"{prefix}_kind") != "kq_kvarn":
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
        state = []
        for j in range(_STATE_ARITY):
            ent = tensor_entries.get(f"{prefix}_s{j}")
            if ent is None:
                return None
            arr = _read(path, data_start, ent)
            if arr is None:
                return None
            state.append(arr)
        try:
            out = KVarNKVCache.from_state(
                tuple(state), tuple(metadata[f"{prefix}_meta"].split(","))
            )
        except (KeyError, TypeError, ValueError):
            # Unknown layout or version: a clean cold miss, never a crash.
            return None
        eval_targets.extend(state)
        return out

    apc._cache_entry_supports_exact_apc = supports_exact
    apc._merge_exact_cache_entries = merge_exact_entries
    apc._clone_cache_entry_for_apc = clone_entry
    apc._safetensors_dtype_info = dtype_info
    apc.model_apc_mode = mode_wrap
    disk_cls._snapshot_exact_cache_entry = snap_entry
    disk_cls._load_exact_cache_entry = load_entry
    apc._gmlx_kvarn_apc = True
