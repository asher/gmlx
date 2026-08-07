"""kvarn KV on the serve batch path.

install_kvarn_serve() arms three seams when the boot environment selects
KV_QUANT_SCHEME=kvarn:

- ``_make_cache`` (mlx_vlm.generate.ar + the server.generation from-import)
  builds BatchKVarNKVCache for every plain-KV layer of an eligible model.
  Deliberate divergence from upstream: the stock quantized builder skips
  the last layer ("sensitive to quantization"); kvarn6 measures at or
  above kv8 fidelity on the KLD harness, so all eligible layers convert
  and the memory story stays uniform. Ineligible models keep fp16 batch
  caches with a one-shot reason (never a silent affine fallback).
- ``BatchGenerator.__init__`` routes kvarn-eligible models onto the APC
  exact tier: stamps the model so model_apc_mode resolves "exact" (the
  block tier cannot split 128-token records) and drops the kv_bits kwarg
  so upstream's kv-quant opt-out keeps the manager. When the kvarn APC
  arms are not installed the manager is nulled instead -- a half-armed
  APC would fail on record-group geometry mid-request.
- ``PromptProcessingBatch.__init__`` re-routes the single-row fast path:
  stock init skips ``_make_cache`` entirely when B=1 and kv_bits is None
  (``cache.make_prompt_cache``), which under kvarn is every lone request.
  The wrap rebuilds the empty prompt cache through the wrapped
  ``_make_cache`` so B=1 batches quantize like the rest.

Speculative targets stay stock: ``spec_cache_build()`` marks the stock
speculative-cache construction so the ``_make_cache`` wrap declines inside
it (the spec engine owns rollback, and the batch kvarn cache cannot trim).

All three wraps install unconditionally at server boot and engage per
call: ``_make_cache`` on its ``kv_quant_scheme`` kwarg (the BatchGenerator
lambdas pass the per-model env value, applied through the residency pool's
transient load windows), the APC gate on the same env read at generator
construction, which happens inside that window.

Kill switch: GMLX_KVARN=0 (same as the CLI paths).
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager

_log = logging.getLogger("gmlx.kvarn_serve")

_MAKE_CACHE_FLAG = "_gmlx_kvarn_make_cache"
_APC_GATE_FLAG = "_gmlx_kvarn_apc_gate"
_PPB_FLAG = "_gmlx_kvarn_ppb_fastpath"

_spec_build = [False]


@contextmanager
def spec_cache_build():
    """Marks a stock speculative-cache build in progress; the _make_cache
    wrap stays stock inside so spec targets keep their own KV handling."""
    _spec_build[0] = True
    try:
        yield
    finally:
        _spec_build[0] = False


def _serve_widths_and_tail():
    from .generation import _kvarn_widths

    raw = os.environ.get("KV_BITS", "")
    try:
        bits = int(raw) if raw else None
    except ValueError:
        bits = None
    try:
        tail = int(os.environ.get("KV_TAIL_TOKENS", "") or 1024)
    except ValueError:
        tail = 1024
    k_bits, v_bits = _kvarn_widths(bits)
    return k_bits, v_bits, tail


def install_kvarn_serve() -> None:
    """Arm the serve batch path for kvarn (engagement is per call/model)."""
    from mlx_vlm.generate import ar as _ar
    from mlx_vlm.server import generation as _gen

    _install_make_cache(_ar, _gen)
    _install_ppb_fastpath(_ar)
    _install_apc_gate(_ar)


def _install_make_cache(_ar, _gen):
    if getattr(_ar._make_cache, _MAKE_CACHE_FLAG, False):
        return
    _orig = _ar._make_cache
    _noted = [False]
    _declined: set = set()

    def _kvarn_make_cache(
        model,
        left_padding,
        kv_bits=None,
        kv_group_size=64,
        kv_quant_scheme=None,
        **kwargs,
    ):
        if kv_quant_scheme != "kvarn" or _spec_build[0]:
            return _orig(
                model,
                left_padding,
                kv_bits=kv_bits,
                kv_group_size=kv_group_size,
                **(
                    {}
                    if kv_quant_scheme is None or _spec_build[0]
                    else {"kv_quant_scheme": kv_quant_scheme}
                ),
                **kwargs,
            )
        from .cache_compat import cache_types
        from .generation import kvarn_unsupported
        from .kvarn_cache import (
            KVARN_BITS,
            BatchKVarNKVCache,
            ensure_registered,
        )

        k_bits, v_bits, tail = _serve_widths_and_tail()
        reason = kvarn_unsupported(model)
        if reason is None and not (k_bits in KVARN_BITS and v_bits in KVARN_BITS):
            reason = f"kvarn bits must be one of {KVARN_BITS}"
        if reason is None and (tail < 0 or tail % 128):
            reason = "KV_TAIL_TOKENS must be a multiple of 128 (0 disables)"
        # The scheme owns the width request either way: a declined model
        # serves fp16, never the affine quantized batch cache.
        caches = _orig(model, left_padding, kv_bits=None, **kwargs)
        if reason is not None:
            key = type(model).__name__
            if key not in _declined:
                _declined.add(key)
                _log.warning(
                    "KV_QUANT_SCHEME=kvarn dropped for %s: %s; KV stays fp16",
                    key,
                    reason,
                )
            return caches
        ensure_registered()
        batch_kv = cache_types("BatchKVCache")
        n = 0
        for i, c in enumerate(caches):
            if type(c) in batch_kv:
                caches[i] = BatchKVarNKVCache(
                    left_padding,
                    k_bits=k_bits,
                    v_bits=v_bits,
                    tail_tokens=tail,
                )
                n += 1
        if n:
            from .kvarn_sdpa import install_kvarn_sdpa

            install_kvarn_sdpa()
            if not _noted[0]:
                _noted[0] = True
                width = (
                    f"kvarn{k_bits}"
                    if k_bits == v_bits
                    else f"kvarn k{k_bits} v{v_bits}"
                )
                regions = "sink+tail fp16" if tail else "sink fp16, tail off"
                print(
                    f"[kv] serve batch: {n}/{len(caches)}-layer {width} KV ({regions})",
                    flush=True,
                )
        return caches

    _kvarn_make_cache.__dict__[_MAKE_CACHE_FLAG] = True
    _ar._make_cache = _kvarn_make_cache
    _gen._make_cache = _kvarn_make_cache


def _install_ppb_fastpath(_ar):
    ppb = _ar.PromptProcessingBatch
    if getattr(ppb.__init__, _PPB_FLAG, False):
        return
    _orig_init = ppb.__init__

    def _kvarn_ppb_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        # Stock init's single-row fast path (B=1, kv_bits None, no right
        # padding) builds via cache.make_prompt_cache and never consults
        # the scheme. Rebuild through the wrapped _make_cache; the cache
        # is still empty here, so the swap is free.
        if (
            kwargs.get("kv_quant_scheme") != "kvarn"
            or kwargs.get("kv_bits") is not None
            or kwargs.get("warm_cache") is not None
            or kwargs.get("draft_model") is not None
            or kwargs.get("right_pad_per_row") is not None
            or len(self.uids) != 1
            or any(
                getattr(c, "kv_quant_scheme", None) == "kvarn"
                for c in self.prompt_cache
            )
        ):
            return
        from .generation import kvarn_unsupported

        # Ineligible models keep their model-built caches (hybrids carry
        # non-KV layers a generic rebuild would drop).
        if kvarn_unsupported(self.model) is not None:
            return
        self.prompt_cache = _ar._make_cache(
            self.model,
            list(self._left_padding_per_row),
            kv_bits=None,
            kv_group_size=kwargs.get("kv_group_size", 64),
            kv_quant_scheme="kvarn",
        )

    _kvarn_ppb_init.__dict__[_PPB_FLAG] = True
    ppb.__init__ = _kvarn_ppb_init


def _install_apc_gate(_ar):
    if getattr(_ar.BatchGenerator.__init__, _APC_GATE_FLAG, False):
        return
    _orig_init = _ar.BatchGenerator.__init__
    _noted = [False]

    def _gated_init(self, model, *args, **kwargs):
        # Scheme comes from the kwarg (the ResponseGenerator's env-window
        # capture), not the ambient env: per-model load windows are closed
        # by the time the first request constructs a BatchGenerator.
        if (
            kwargs.get("apc_manager") is not None
            and kwargs.get("kv_quant_scheme") == "kvarn"
        ):
            from .generation import kvarn_unsupported

            if kvarn_unsupported(model) is None:
                from .kvarn_apc import kvarn_apc_installed, stamp_model

                if kvarn_apc_installed():
                    # Exact tier only: stamp the model so model_apc_mode
                    # resolves "exact", and drop kv_bits (the scheme owns
                    # the width via env) so upstream's kv-quant opt-out
                    # does not null the manager.
                    stamp_model(model)
                    kwargs["kv_bits"] = None
                    if not _noted[0]:
                        _noted[0] = True
                        _log.info("[serve] APC exact tier active under kvarn KV")
                else:
                    kwargs["apc_manager"] = None
                    if not _noted[0]:
                        _noted[0] = True
                        _log.info(
                            "[serve] APC inactive under kvarn KV "
                            "(kvarn APC arms not installed)"
                        )
        return _orig_init(self, model, *args, **kwargs)

    _gated_init.__dict__[_APC_GATE_FLAG] = True
    _ar.BatchGenerator.__init__ = _gated_init
