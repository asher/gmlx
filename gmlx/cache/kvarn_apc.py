"""kvarn arms for the APC exact tier (memory + disk).

install_kvarn_apc() chains kvarn arms onto the same seams
apc_pooling arms for pooling caches:

- capability/clone/merge: KVarNKVCache entries store and warm-adopt through
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
- prefix-cache plan and model_apc_mode: kvarn caches are built by the
  serve installers after model construction, so the stock plan over
  model.make_cache() sees plain fp16 caches and resolves "block" -- a tier
  kvarn cannot serve (128-token records do not split into 16-token
  blocks). Models stamped by the kvarn serve gate (stamp_model) resolve
  the checkpoint strategy ("exact") instead. model_apc_mode answers a
  stamped model from the plan, since the stock function returns "block"
  for a model without make_cache before it plans.

Entry isolation: the gmlx APC manager salts exact-tier extra hashes with
the kvarn wire config (apply_kvarn_salt, called at residency's post-load
manager pairing and gated on an actual-conversion probe), so entries
written under one scheme/width/tail config never warm-adopt under
another -- the disk namespace is the model path and carries no scheme.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import secrets
from dataclasses import replace

from gmlx.upstream.patching import link

_log = logging.getLogger(__name__)

_FLAG = "_gmlx_kvarn_apc"
_MODE_STAMP = "_gmlx_kvarn_apc_exact"
_CONVERTS_ATTR = "_gmlx_kvarn_converts"
_STATE_ARITY = 10


def kvarn_apc_installed() -> bool:
    apc = importlib.import_module("mlx_vlm.apc")
    return bool(getattr(apc, _FLAG, False))


def stamp_model(model) -> None:
    """Mark a model as serving kvarn KV so its prefix-cache plan resolves
    the checkpoint strategy (model_apc_mode "exact").
    Stamps both the model and its language model: the live call sites
    disagree on which object they pass (the engine passes the top-level
    model, dispatch and spec_engine the language model)."""
    for obj in (model, getattr(model, "language_model", None)):
        if obj is not None:
            try:
                setattr(obj, _MODE_STAMP, True)
            except Exception:  # noqa: S110 - stamp only; an unsettable object reads as unstamped
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
        lm = getattr(model, "language_model", None) or model
        make = getattr(lm, "make_cache", None)
        if make is None:
            # No make_cache: upstream builds plain KV for every layer.
            from gmlx.cache.kvarn_cache import kvarn_unsupported

            val = kvarn_unsupported(model) is None
        else:
            # The policy answers "converts" exactly, carve-out included,
            # so the stamp and salt never disagree with the cache build.
            from .kvarn_serve import _serve_widths_and_tail, kvarn_batch_policy

            k_bits, v_bits, tail = _serve_widths_and_tail(model)
            policy = kvarn_batch_policy(model, make(), k_bits, v_bits, tail,
                                        mode="single")
            val = policy.n_quant > 0
    except Exception:
        val = False
    try:
        setattr(model, _CONVERTS_ATTR, val)
    except Exception:  # noqa: S110 - memo stamp; recomputed when the model forbids ad-hoc attrs
        pass
    return val


def apply_kvarn_salt(manager, model) -> None:
    """Set the manager's exact-tier wire salt at residency's post-load
    pairing, gated on the model actually converting: a kvarn-window boot
    of a zero-conversion arch (deepseek4, recurrent_gemma) runs pure fp16
    caches, and salting its entries would cold-miss every cross-boot
    lookup. The XOR-identity default is correct only for that
    non-converting case; a converting model whose salt computation fails
    gets a random per-boot salt instead, so its entries stay out of the
    fp16 hash space and cold-miss across boots rather than warm-adopt
    under the wrong wire config."""
    if manager is None or model is None:
        return
    if not kvarn_model_converts(model):
        return
    try:
        salt = kvarn_entry_salt(model)
    except Exception:
        salt = secrets.randbits(63) | 1
        _log.warning(
            "kvarn exact-tier salt computation failed; salting with a "
            "per-boot value (cross-boot APC reuse off for this model)",
            exc_info=True)
    if not salt:
        return
    try:
        manager._exact_extra_salt = salt
    except Exception:
        _log.warning(
            "kvarn exact-tier salt could not be applied; kvarn entries "
            "share the fp16 hash space until the next boot", exc_info=True)


def kvarn_entry_salt(model=None) -> int:
    """Exact-tier extra-hash salt for this model's kvarn wire config, 0
    (XOR identity) for an unstamped model or a non-kvarn stamp. Residency
    stamps the policy before it salts, so entries are salted with the
    config the caches were actually built at."""
    from .kvarn_cache import KVarNKVCache
    from .kvarn_serve import _serve_widths_and_tail, stamped_single_policy

    single = stamped_single_policy(model)
    if single is None or getattr(single, "scheme", None) != "kvarn":
        return 0

    k_bits, v_bits, tail = _serve_widths_and_tail(model)
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
        adapters = importlib.import_module("mlx_vlm.apc_adapters")
        coordinator = importlib.import_module("mlx_vlm.apc_coordinator")
        _copy = apc._copy_mlx_array
        stock_clone = adapters.clone_cache_entry
        stock_merge = adapters.merge_cache_entries
        stock_plan = adapters.build_prefix_cache_plan
        stock_mode = apc.model_apc_mode
        register_capability = adapters.register_capability
        checkpoint = adapters.Capability.CHECKPOINT
        disk_cls = apc.DiskBlockStore
        stock_snap = disk_cls._snapshot_exact_cache_entry
        stock_load = disk_cls._load_exact_cache_entry
        _read = apc._read_safetensors_tensor
    except (AttributeError, ModuleNotFoundError) as e:
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

    # Explicit rows: the plan would otherwise infer KVarNKVCache from its
    # KVCache base, and the batch class has no stock base at all.
    register_capability(KVarNKVCache, checkpoint)
    register_capability(BatchKVarNKVCache, checkpoint)

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
            return BatchKVarNKVCache.merge(list(entries))
        return stock_merge(entries, prefix_lens)

    def stamped(model):
        # Check-down: stamp_model stamps both objects, but a caller may
        # hand over a wrapper whose language model carries the stamp
        # (there is no walking up -- children hold no parent reference).
        return bool(getattr(model, _MODE_STAMP, False) or getattr(
            getattr(model, "language_model", None), _MODE_STAMP, False))

    def plan_wrap(model, overrides=None):
        # A stamped plan keeps its stock components but pages nothing, so
        # its strategy is "checkpoint" (legacy mode "exact").
        plan = stock_plan(model, overrides)
        if plan.restorable and stamped(model):
            plan.layer_specs = [
                replace(spec, block_eligible=False) for spec in plan.layer_specs
            ]
        return plan

    def mode_wrap(language_model):
        # The stock mode returns "block" for a model without make_cache
        # before it plans, while the batch generator's coordinator plans
        # that model from make_prompt_cache. A stamped model answers from
        # the plan, as the coordinator does.
        if stamped(language_model):
            return plan_wrap(language_model).legacy_mode
        return stock_mode(language_model)

    def snap_entry(self, c, prefix, arrays, metadata):
        # Exact class, not isinstance: load_entry rebuilds through
        # KVarNKVCache.from_state, whose meta arity the rotating subclass
        # does not match. Writing one would cost a disk record that can
        # only ever miss.
        if type(c) is not KVarNKVCache:
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
        prefix_len=None,
    ):
        # A trim request only accompanies an all-"kv" shard, never kq_kvarn.
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
                prefix_len=prefix_len,
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

    # Clone and merge recurse through the adapter module globals, and
    # the stock model_apc_mode imports the plan builder per call; the
    # coordinator module holds its own binding, rebound beside it. The
    # mode wrap covers the no-make_cache answer the plan never sees. The
    # shard reader decodes the uint32 kvarn codes natively.
    adapters.clone_cache_entry = link(clone_entry, stock_clone)
    adapters.merge_cache_entries = link(merge_exact_entries, stock_merge)
    adapters.build_prefix_cache_plan = link(plan_wrap, stock_plan)
    coordinator.build_prefix_cache_plan = plan_wrap
    apc.model_apc_mode = link(mode_wrap, stock_mode)
    disk_cls._snapshot_exact_cache_entry = link(snap_entry, stock_snap)
    disk_cls._load_exact_cache_entry = link(load_entry, stock_load)
    apc._gmlx_kvarn_apc = True
