"""Speculative KV quantization: lift helpers and the install block.

Split out of ``gmlx.spec.engine``.
"""

from __future__ import annotations

import logging
import os

import mlx.core as mx

from gmlx.spec.engine import _debug_note

_log = logging.getLogger(__name__)


def dequantize_lift_cache(c):
    """Dequantize a B=1 QuantizedKVCache into a one-row BatchKVCache.

    QuantizedKVCache has no merge, and the B>1 MTP arm runs fp16 KV
    anyway, so the lift performs the same conversion the batch-build
    swap does, at preemption or injection time, as a one-off O(depth)
    copy. Without this every kv-bits MTP preemption declined into a
    drain-wait and queued rows stalled behind the live generation."""
    from mlx_vlm.models.cache import BatchKVCache

    lifted = BatchKVCache([0])
    L = c.offset
    if L:
        keys = mx.dequantize(
            *(mx.contiguous(t[..., :L, :]) for t in c.keys),
            group_size=c.group_size, bits=c.bits)
        values = mx.dequantize(
            *(mx.contiguous(t[..., :L, :]) for t in c.values),
            group_size=c.group_size, bits=c.bits)
        lifted.update_and_fetch(keys, values)
    stamp = getattr(c, "_gmlx_cascade", None)
    if stamp is not None:
        lifted._gmlx_cascade = stamp
    return lifted


def kvarn_lift_cache(c):
    """Recover a B=1 KVarNKVCache into a one-row fp16 BatchKVCache, the
    lift for an mlx-kquant without per-row ends (the batched arm then
    runs fp16 KV).

    The kvarn twin of dequantize_lift_cache: materialize() returns
    rotated-domain K/V, which stock SDPA would attend with an un-rotated
    query -- no crash, just wrong logits on every preempted row.
    _raw_single is the original-domain accessor."""
    from mlx_vlm.models.cache import BatchKVCache

    lifted = BatchKVCache([0])
    if c.offset:
        keys, values = c._raw_single()
        lifted.update_and_fetch(keys, values)
    stamp = getattr(c, "_gmlx_cascade", None)
    if stamp is not None:
        lifted._gmlx_cascade = stamp
    return lifted


def mtp_kv_decline(lm, *, owned_round: bool = True) -> str | None:
    """Why this MTP verify walk cannot run on packed KV, or None.

    The owned rounds roll back with trim, and affine packing is
    per-token along head_dim, so a trim is an offset move: they take the
    same layers serve takes. The two stock walks slice keys as raw
    arrays and cannot read a packed tuple back. Shared by serve, run and
    chat so the three cannot drift.
    """
    if not owned_round:
        return "GMLX_OWNED_ROUND=0 stock rounds have no KV quantization hook"
    from gmlx.models.qwen35.gdn import stock_gdn_fallback

    mt = None
    for h in _spec_target_holders(lm):
        cfg = getattr(h, "config", None)
        mt = (getattr(h, "model_type", None)
              or (cfg.get("model_type") if isinstance(cfg, dict)
                  else getattr(cfg, "model_type", None)))
        if mt:
            break
    if stock_gdn_fallback(mt):
        return ("the GMLX_QWEN_OWNED=0 stock fallback cannot verify on a "
                "quantized KV cache")
    return None


def lift_single_cache(c):
    """Promote a single-sequence cache to its batch class. An affine B=1
    cache recovers to a one-row fp16 BatchKVCache (the batched MTP arm
    runs fp16 KV under uniform). A kvarn B=1 cache becomes a one-row
    BatchKVarNKVCache, buffers and horizon adopted bit-exactly, when the
    installed mlx-kquant takes per-row ends, and recovers to fp16 rows
    otherwise. Everything else lifts through its class's merge. The
    cascade stamp rides along. One lift for the preempted host row and
    the injected rows, so the two cannot drift."""
    from gmlx.cache.compat import cache_types

    if isinstance(c, cache_types("QuantizedKVCache")):
        return dequantize_lift_cache(c)
    if getattr(c, "kv_quant_scheme", None) == "kvarn" and batch_liftable(c):
        from gmlx.cache.kvarn_sdpa import kvarn_row_ends_ok

        if not kvarn_row_ends_ok():
            return kvarn_lift_cache(c)
        from gmlx.cache.kvarn_cache import BatchKVarNKVCache

        lifted = BatchKVarNKVCache.merge([c])
        stamp = getattr(c, "_gmlx_cascade", None)
        if stamp is not None:
            lifted._gmlx_cascade = stamp
        return lifted
    lifted = type(c).merge([c])
    stamp = getattr(c, "_gmlx_cascade", None)
    if stamp is not None:
        lifted._gmlx_cascade = stamp
    return lifted


def batch_liftable(c) -> bool:
    """Whether _lift_host_cache can promote this cache to a batch class.

    The preemption and pre-start-compaction gates must agree with it: a
    scheme it can lift but they refuse declines into the drain-wait, and
    one they admit but it cannot lift raises mid-rebuild.
    """
    if hasattr(c, "filter") and hasattr(c, "extend"):
        return True
    if getattr(c, "kv_quant_scheme", None) == "kvarn":
        from gmlx.cache.kvarn_cache import KVarNRotatingKVCache

        # The rotating subclass counts evicted tokens in offset that its
        # buffers no longer hold: kvarn_lift_cache would misplace rows.
        return not isinstance(c, KVarNRotatingKVCache)
    from gmlx.cache.compat import cache_types

    return (isinstance(c, cache_types("QuantizedKVCache"))
            or hasattr(type(c), "merge"))

_SPEC_KV_QUANT_FLAG = "_kq_gguf_spec_kv_quant"
_SPEC_KV_QUANT_WIDTHS = (2, 3, 4, 6, 8)  # mx.quantize affine widths


def _spec_kv_quant_params():
    """resolve_kv_quant_policy kwargs for the KV quantization serve's env
    asks the trimmable B=1 single-stream cache to honor, else None.
    Fractional widths and unknown schemes have no such cache; kvarn
    engages on the scheme alone (widths default like the CLI's)."""
    if os.environ.get("GMLX_SPEC_KV_QUANT", "1") == "0":
        return None
    scheme = os.environ.get("KV_QUANT_SCHEME", "uniform")
    raw = os.environ.get("KV_BITS", "")
    if scheme == "kvarn":
        from gmlx.cache.kvarn_cache import kvarn_widths, parse_tail_tokens

        try:
            bits = int(raw) if raw else None
            tail = parse_tail_tokens(os.environ.get("KV_TAIL_TOKENS"))
        except ValueError:
            _log.warning(
                "KV_BITS/KV_TAIL_TOKENS malformed under scheme kvarn; "
                "B=1 MTP target KV stays fp16"
            )
            return None
        k_bits, v_bits = kvarn_widths(bits)
        return dict(scheme="kvarn", kv_bits=k_bits, value_bits=v_bits,
                    tail_tokens=tail)
    if not raw:
        return None
    try:
        bits = float(raw)
    except ValueError:
        return None
    if bits <= 0:
        return None
    if (
        scheme != "uniform"
        or bits != int(bits)
        or int(bits) not in _SPEC_KV_QUANT_WIDTHS
    ):
        _log.warning(
            "KV_BITS=%s scheme=%s: no trimmable single-stream cache; "
            "B=1 MTP target KV stays fp16",
            raw,
            scheme,
        )
        return None
    return dict(scheme="uniform", kv_bits=int(bits),
                kv_group_size=int(os.environ.get("KV_GROUP_SIZE", "64")))


def _stamped_spec_params(lm):
    """resolve_kv_quant_policy kwargs from the KV policy residency stamped
    on this model at load, None when nothing is stamped. Per-model env
    windows are closed by request time, so the stamp rules the boot env;
    a stamp that quantizes nothing (off, dropped, error) yields {} (stay
    fp16)."""
    from gmlx.cache.kvarn_serve import stamped_single_policy

    single = stamped_single_policy(lm)
    if single is None:
        return None
    bits = getattr(single, "bits", None)
    if not bits or getattr(single, "verdict", None) not in ("full", "partial"):
        return {}
    if getattr(single, "scheme", None) == "kvarn":
        from gmlx.cache.kvarn_cache import KVARN_DEFAULT_TAIL

        tail = single.tail_tokens
        return dict(scheme="kvarn", kv_bits=int(bits),
                    value_bits=int(single.value_bits or bits),
                    tail_tokens=(KVARN_DEFAULT_TAIL if tail is None
                                 else int(tail)))
    if int(bits) != bits or int(bits) not in _SPEC_KV_QUANT_WIDTHS:
        return {}
    return dict(scheme="uniform", kv_bits=int(bits),
                kv_group_size=int(single.group_size))


def _spec_target_holders(lm) -> tuple:
    """The spec target and the language model it may wrap: serve hands
    the cache builder an MTPTextTarget, the CLI the bare model. Every
    probe on the target reads both."""
    inner = getattr(lm, "language_model", None)
    return (lm,) if inner is None or inner is lm else (lm, inner)


def _mtp_reads_kv_back(lm) -> bool:
    """True when the target's verify route re-reads K/V from the prompt
    cache (spec_helpers._mtp_shared_kv_from_prompt_cache): it computes
    logits from hidden but owns no verify hook, so the walk rebuilds the
    drafter's shared K/V from cache state -- raw arrays kvarn records
    cannot supply."""
    return any(
        callable(getattr(h, "speculative_logits_from_hidden", None))
        and not callable(getattr(h, "speculative_verify_hidden", None))
        and not callable(getattr(h, "speculative_verify_logits", None))
        for h in _spec_target_holders(lm)
    )


def _harden_spec_target(lm) -> None:
    """harden_mtp_rollback on every holder of the target's rollback."""
    from gmlx.gen.generation import harden_mtp_rollback

    for h in _spec_target_holders(lm):
        harden_mtp_rollback(h)


def _kvarn_spec_reason(lm):
    """The kvarn declines the shared policy cannot see: the target's own
    verify contract. Both MTP arms (B=1 and batched) check it."""
    from gmlx.cache.kvarn_cache import kvarn_unsupported

    reason = kvarn_unsupported(lm)
    if reason is None and _mtp_reads_kv_back(lm):
        reason = (
            "the target's verify path reads shared K/V back "
            "from the cache (kvarn records are not raw K/V)"
        )
    return reason


def _kvarn_batch_spec_cache(lm, caches, left_padding, params):
    """Convert a B>1 MTP target stack to kvarn batch rows when the batched
    arm is engaged, in place; None when it declines (the caller keeps the
    fp16 batch stack). The B=1 declines apply: an ineligible model, a
    target that reads K/V back, a sliding-window stack, and the policy's
    own drop when mlx-kquant lacks per-row ends. The verify block is not
    knowable here; batch formation clamps it to the kernels' width."""
    from gmlx.cache.kv_policy import kv_line, note_once
    from gmlx.cache.kvarn_cache import (KVARN_DEFAULT_TAIL,
                                        kvarn_mtp_window_decline)
    from gmlx.cache.kvarn_serve import (kvarn_batch_policy,
                                        kvarn_convert_batch_stack)

    def decline(reason):
        if note_once(lm, "spec-kv-kvarn-batched"):
            _log.warning(
                "KV_QUANT_SCHEME=kvarn dropped on the batched MTP path: %s; "
                "the batch runs fp16 KV", reason)
        return None

    reason = kvarn_mtp_window_decline(caches) or _kvarn_spec_reason(lm)
    if reason is not None:
        return decline(reason)
    k_bits = int(params["kv_bits"])
    v_bits = int(params.get("value_bits") or k_bits)
    tail = params.get("tail_tokens")
    tail = KVARN_DEFAULT_TAIL if tail is None else int(tail)
    policy = kvarn_batch_policy(lm, caches, k_bits, v_bits, tail,
                                mode="batched", mtp=True)
    if policy.verdict not in ("full", "partial"):
        return decline(policy.reason)
    n = kvarn_convert_batch_stack(caches, policy, left_padding, k_bits,
                                  v_bits, tail)
    if not n:
        return decline("no plain KV-cache layers in this arch's stack")
    _harden_spec_target(lm)
    if note_once(lm, "spec-kv-batched"):
        _log.info("%s", kv_line("MTP spec path (batched)", policy))
    return caches


def install_spec_kv_quant() -> None:
    """Honor KV_BITS on the B=1 MTP serve path.

    Stock ``make_speculative_prompt_cache`` returns plain fp16 caches for
    ``draft_kind == "mtp", batch_size == 1``, discarding the engine's
    kv_bits: ``BatchQuantizedKVCache`` cannot trim, and MTP rollback must
    trim the target. The single-stream ``QuantizedKVCache`` can trim --
    packing is per-token along head_dim, so trim is an offset move -- and
    the model rollback already goes through ``is_trimmable()``/``trim()``.
    The shared KV policy picks the layers: growing KV converts at
    construction (empty, so conversion is free), quantizable pools pack
    at rest, and windows, recurrent state, and opt-outs stay fp16 at any
    nesting depth. Scheme kvarn converts the same B=1 caches to
    ``KVarNKVCache`` instead (rollback rides the stage/horizon regions),
    declining targets whose verify path reads shared K/V back from cache
    state. B>1 MTP under kvarn converts the batch stack to
    ``BatchKVarNKVCache`` rows when the installed mlx-kquant takes
    per-row ends (each row rolls back by its own rejected count); under
    uniform, or on an older mlx-kquant, B>1 keeps fp16 batch KV with a
    one-shot warning (the packed batch cache cannot trim). Scheme and
    widths come from the policy stamped on the model at load; the boot
    env is the fallback for unstamped models. Kill switch:
    GMLX_SPEC_KV_QUANT=0."""
    from mlx_vlm.generate import ar as _ar
    from mlx_vlm.server import generation as _gen
    from mlx_vlm.speculative import utils as _su

    if getattr(_su.make_speculative_prompt_cache, _SPEC_KV_QUANT_FLAG, False):
        return
    if os.environ.get("GMLX_SPEC_KV_QUANT", "1") == "0":
        return
    boot_params = _spec_kv_quant_params()

    from gmlx.cache.compat import cache_types

    from gmlx.cache.kv_policy import note_once

    _orig = _su.make_speculative_prompt_cache

    def _decline_kvarn(lm, reason: str):
        if note_once(lm, "spec-kv-kvarn"):
            _log.warning(
                "KV_QUANT_SCHEME=kvarn dropped on the B=1 MTP path: %s", reason
            )

    def _quantizing_spec_cache(lm, *, draft_kind, batch_size, left_padding, make_cache):
        from gmlx.cache.kvarn_serve import spec_cache_build

        # The stock make_cache closure passes the boot scheme through;
        # suspend the serve wrap so spec targets never get a batch kvarn
        # cache (the verify walk needs trim, which it does not support).
        with spec_cache_build():
            caches = _orig(
                lm,
                draft_kind=draft_kind,
                batch_size=batch_size,
                left_padding=left_padding,
                make_cache=make_cache,
            )
        if draft_kind != "mtp":
            return caches
        params = _stamped_spec_params(lm)
        if params is None:
            params = boot_params
        if batch_size != 1:
            if params and params.get("scheme") == "kvarn":
                out = _kvarn_batch_spec_cache(lm, caches, left_padding, params)
                if out is not None:
                    return out
            # Force fp16 batch KV: the stock rollback misfiles
            # BatchQuantizedKVCache as an SSM cache and never trims
            # rejected drafts.
            # to_batch_cache also quantizes nested subcaches. Walk
            # into CacheList entries.
            from mlx_vlm.models.cache import BatchKVCache

            batch_quant = cache_types("BatchQuantizedKVCache")

            def _swap(c):
                if isinstance(c, batch_quant):
                    return BatchKVCache(left_padding), 1
                inner = getattr(c, "caches", None)
                if inner is None:
                    return c, 0
                subs = [_swap(s) for s in inner]
                n = sum(k for _, k in subs)
                if n:
                    c.caches = tuple(s for s, _ in subs)
                return c, n

            swapped = 0
            for e, c in enumerate(caches):
                caches[e], n_sw = _swap(c)
                swapped += n_sw
            if swapped and note_once(lm, "spec-kv-batch"):
                _log.warning(
                    "KV quantization with MTP at batch size %d: packed "
                    "batch rollback is unsupported; %d layers run fp16 KV",
                    batch_size, swapped)
            return caches
        if not params:
            return caches
        kind = params["scheme"]
        decline = mtp_kv_decline(lm)
        if decline is not None:
            if note_once(lm, "spec-kv-stock"):
                _log.warning(
                    "KV quantization dropped on the MTP path: %s", decline)
            return caches
        scheme_reason = None
        if kind == "kvarn":
            from gmlx.cache.kvarn_cache import kvarn_mtp_window_decline

            scheme_reason = (kvarn_mtp_window_decline(caches)
                             or _kvarn_spec_reason(lm))
            if scheme_reason is not None:
                _decline_kvarn(lm, scheme_reason)
                return caches
        # The shared policy owns layer selection: nested KV members,
        # pools, windows, and opt-outs at any depth.
        from gmlx.cache.kv_policy import (kv_line, quantize_stack,
                                          resolve_kv_quant_policy)

        policy = resolve_kv_quant_policy(
            caches, mode="single", scheme_reason=scheme_reason, **params)
        if policy.verdict not in ("full", "partial"):
            if kind == "kvarn":
                _decline_kvarn(lm, policy.reason)
            elif note_once(lm, "spec-kv-dropped"):
                _log.warning("%s", kv_line("MTP spec path", policy))
            return caches

        armed, n = quantize_stack(caches, policy)
        if kind == "kvarn":
            if not n:
                _decline_kvarn(lm, "no plain KV-cache layers in this arch's stack")
                return caches
            _harden_spec_target(lm)
        if (n or armed) and note_once(lm, "spec-kv"):
            _log.info("%s", kv_line("MTP spec path", policy))
        return caches

    _quantizing_spec_cache.__dict__[_SPEC_KV_QUANT_FLAG] = True
    _quantizing_spec_cache.__dict__["_gmlx_orig"] = _orig
    _su.make_speculative_prompt_cache = _quantizing_spec_cache
    _ar.make_speculative_prompt_cache = _quantizing_spec_cache
    _gen.make_speculative_prompt_cache = _quantizing_spec_cache
    _debug_note(f"[mtp] spec cache wrap armed (B=1); boot env {boot_params}")
