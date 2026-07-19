"""Single-call speculative-verify attention for qwen3.5/3.6 full-attn layers.

Upstream mlx-vlm's ``Qwen3_5Attention`` has two verify shapes the folded
kernels in ``attn_hd512`` can never reach:

- B==1 hook-path verify (``speculative_verify_hidden``, target_verify=True)
  decomposes the block into L sequential qL==1 SDPA calls, sweeping the whole
  KV cache L times per MTP round. For a plain (non-quantized) cache the loop
  is numerically one bottom-right-"causal" SDPA over the block: row i attends
  keys [0, prefix + i], exactly MLX's "causal" alignment for qL < kL.
- Batched (B>=2) verify arrives at the module's ``scaled_dot_product_attention``
  as one call with a left-padded array mask (``BatchKVCache.make_mask``), which
  every fused route declines -- stock materializes [B, Hq, qL, kL] scores.
  Split per row (each row's keys sliced past its pad, mask "causal"), the rows
  are plain folded verifies: same bytes, fused kernels, pad columns skipped.

Patched seams:

- ``qwen3_5.language._target_verify_left_padded_attention``: claims the B==1
  case as one "causal" call, packed (quantized) KV included, routed through
  the quantized-SDPA dispatch; array masks defer upstream.
- ``qwen3_5.language.scaled_dot_product_attention``: claims verify-width
  (2 <= qL <= 8) batched calls on plain batch caches, using the cache's
  ``left_padding`` (host-synced once per membership change, identity-cached)
  to slice each row. Unknown array masks without pad info defer upstream.

Disable both with GMLX_QWEN35_VERIFY_FOLD=0.
"""

from __future__ import annotations

import sys

import mlx.core as mx

from .envflags import env_bool

_installed = False


def _mask_desc(mask) -> str:
    if isinstance(mask, (str, type(None))):
        return str(mask)
    return f"array{tuple(mask.shape)} {mask.dtype}"


def _fold_debug(dbg: list, line: str) -> None:
    """Consume one debug slot and print (callers gate on ``dbg[0] > 0``)."""
    dbg[0] -= 1
    print(f"[fold-debug] {line}", file=sys.stderr, flush=True)


def _pads_list(cache):
    """Per-row left padding of a batch cache as a Python list, or None.
    The tolist() host sync runs once per left_padding array (batch membership
    change), then rides an identity-keyed cache attr."""
    lp = getattr(cache, "left_padding", None)
    if lp is None:
        return None
    cached = getattr(cache, "_gmlx_pads_cache", None)
    if cached is not None and cached[0] is lp:
        return cached[1]
    pads = [int(x) for x in lp.tolist()] if isinstance(lp, mx.array) else [
        int(x) for x in lp
    ]
    cache._gmlx_pads_cache = (lp, pads)
    return pads


def install_qwen35_verify_fold() -> bool:
    """Route qwen3.5 verify attention through folded SDPA calls (B==1 single
    call; B>=2 one "causal" call per row). Idempotent; no-op when
    GMLX_QWEN35_VERIFY_FOLD=0 or the qwen3_5 module is unavailable.
    Returns True if the patch is active."""
    global _installed
    if not env_bool("GMLX_QWEN35_VERIFY_FOLD", True):
        return False
    if _installed:
        return True
    try:
        from mlx_vlm.models.qwen3_5 import language as q35
    except ImportError as e:
        print(f"[verify-fold] disabled: qwen3_5 module unavailable ({e})",
              flush=True)
        return False

    cur = q35._target_verify_left_padded_attention
    orig = getattr(cur, "_gmlx_orig", cur)
    dbg = [8] if env_bool("GMLX_VERIFY_FOLD_DEBUG", False) else None

    def _folded_verify(queries, keys, values, *, cache, scale, mask):
        # Packed (tuple) KV pairs with a quantized cache: the same single
        # "causal" call dispatches to quantized SDPA upstream, which builds
        # the identical bottom-right causal mask. Upstream's own fallback
        # cannot take this case (it slices keys as raw arrays).
        quant = isinstance(keys, (tuple, list))
        claimed = (
            queries.ndim == 4
            and (hasattr(cache, "bits") if quant
                 else keys.ndim == 4 and not hasattr(cache, "bits"))
            and queries.shape[0] == 1
            and queries.shape[2] > 1
            and (mask is None or (isinstance(mask, str) and mask == "causal"))
        )
        if dbg is not None and dbg[0] > 0 and queries.shape[2] > 1:
            kdesc = (f"quant{tuple(keys[0].shape)}" if quant
                     else f"{tuple(keys.shape)}")
            _fold_debug(dbg, f"claimed={claimed} q={tuple(queries.shape)} "
                             f"k={kdesc} mask={_mask_desc(mask)} "
                             f"cache={type(cache).__name__}")
        if claimed:
            return q35.scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=scale, mask="causal"
            )
        return orig(queries, keys, values, cache=cache, scale=scale, mask=mask)

    _folded_verify._gmlx_orig = orig
    q35._target_verify_left_padded_attention = _folded_verify

    sdpa_cur = q35.scaled_dot_product_attention
    sdpa_orig = getattr(sdpa_cur, "_gmlx_orig", sdpa_cur)

    def _folded_batch_sdpa(queries, keys, values, cache, scale, mask,
                           sinks=None):
        if isinstance(keys, (tuple, list)):
            # Packed KV: never touch .ndim/.shape; upstream dispatches
            # tuple pairs to quantized SDPA.
            return sdpa_orig(
                queries, keys, values, cache=cache, scale=scale, mask=mask,
                sinks=sinks
            )
        if (
            sinks is None
            and queries.ndim == 4
            and keys.ndim == 4
            and queries.shape[0] >= 2
            and 2 <= queries.shape[2] <= 8
            and keys.shape[2] >= 4096
            and not hasattr(cache, "bits")
        ):
            causal = mask is None or (
                isinstance(mask, str) and mask == "causal"
            )
            pads = _pads_list(cache)
            if pads is None and causal:
                pads = [0] * queries.shape[0]
            padded = (
                pads is not None
                and isinstance(mask, mx.array)
                and mask.shape[-1] == keys.shape[2]
            )
            if dbg is not None and dbg[0] > 0:
                _fold_debug(
                    dbg,
                    f"batch claimed={pads is not None and (causal or padded)} "
                    f"q={tuple(queries.shape)} k={tuple(keys.shape)} "
                    f"mask={_mask_desc(mask)} pads={pads}")
            if pads is not None and (causal or padded):
                # An array mask on a batch cache is left-pad + causal by
                # construction (BatchKVCache.make_mask); per-row slices past
                # the pad make each row a plain bottom-right-causal verify.
                rows = [
                    sdpa_orig(
                        queries[r : r + 1],
                        keys[r : r + 1, :, p:, :],
                        values[r : r + 1, :, p:, :],
                        cache=cache,
                        scale=scale,
                        mask="causal",
                    )
                    for r, p in enumerate(pads)
                ]
                return mx.concatenate(rows, axis=0)
        return sdpa_orig(
            queries, keys, values, cache=cache, scale=scale, mask=mask,
            sinks=sinks
        )

    _folded_batch_sdpa._gmlx_orig = sdpa_orig
    q35.scaled_dot_product_attention = _folded_batch_sdpa
    _installed = True
    return True
