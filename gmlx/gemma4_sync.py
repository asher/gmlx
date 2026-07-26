"""Host-sync-free gemma-4 mask building and rope offsets.

Two per-call host round-trips in the upstream gemma4 text path:

- ``Gemma4TextModel._make_masks`` probes the sliding-layer cache with
  ``int(mx.max(mx.array(c.offset)).item()) > 0`` whenever ``qL > 1``. For int
  offsets (B=1 KV/RotatingKV caches) the answer is knowable host-side; for
  array offsets (batched caches) the ``.item()`` forces a command-buffer
  wait that serializes against all in-flight GPU work -- once per MTP verify
  round and once per batched prefill chunk.
- ``Attention.__call__`` wraps ``offset = mx.array(cache.offset)`` on every
  non-shared layer, every call: a device upload per layer per step for int
  offsets, and a full array COPY per layer for batched array offsets.
  ``mx.fast.rope`` accepts both int and array offsets directly.

The replacements keep upstream's decisions everywhere except:

- int offsets compare host-side (identical result, no dispatch);
- array offsets skip the probe and assume a cached prefix exists. The only
  divergent case is a qL>1 forward on an all-zero array offset (first chunk
  of a batched prefill), where upstream picks the "causal" string and this
  patch builds the windowed array mask -- same attention result, built the
  way every later chunk builds it anyway;
- int rope offsets pass through untouched (no per-layer device upload).
  Array offsets keep upstream's snapshot copy: ``update_and_fetch`` runs
  between the key rope and the query rope and advances ``cache.offset``
  with an in-place ``+=`` (mx arrays mutate through every handle under
  augmented assignment), so an aliased offset rotates queries one
  position ahead of keys on every batched decode step and gated B>1
  decode degenerates. The copy is async device work, not a sync.

Install is idempotent; GMLX_G4_NOSYNC=0 disables. Both replaced bodies are
copies of the pinned upstream implementations (seam-fingerprinted in
upstream_seams.json); module-level collaborators (create_attention_mask,
scaled_dot_product_attention, ...) are resolved through the upstream module
at call time so other patches compose.
"""
from __future__ import annotations

import mlx.core as mx

from .envflags import env_bool

_installed = False


def _cache_has_prefix(c) -> bool:
    off = getattr(c, "offset", 0)
    if isinstance(off, mx.array):
        # Batched cache: probing max(offset) is the sync this module exists
        # to remove. Assume a prefix; the array mask is correct at offset 0.
        return True
    try:
        return int(off) > 0
    except (TypeError, ValueError):  # exotic cache: keep upstream's answer
        return True


def install_gemma4_nosync() -> bool:
    """Replace the two host-syncing gemma4 call sites. Idempotent; no-op when
    GMLX_G4_NOSYNC=0 or the gemma4 module is unavailable. Returns True if the
    patch is active."""
    global _installed
    if not env_bool("GMLX_G4_NOSYNC", True):
        return False
    if _installed:
        return True
    try:
        from mlx_vlm.models.gemma4 import language as g4
    except ImportError as e:
        print(f"[g4-nosync] disabled: gemma4 module unavailable ({e})",
              flush=True)
        return False

    masks_orig = getattr(g4.Gemma4TextModel._make_masks, "_gmlx_orig",
                         g4.Gemma4TextModel._make_masks)

    def _make_masks(self, h, cache, mm_token_type_ids=None):
        # Upstream body with the sliding-branch offset probe replaced by
        # _cache_has_prefix (no .item()). Everything else verbatim.
        mask = {}
        masks = []
        has_audio_tokens = (
            mm_token_type_ids is not None
            and int(mx.sum(mm_token_type_ids == 3).item()) > 0
        )
        has_visual_tokens = (
            mm_token_type_ids is not None
            and int(mx.sum(
                (mm_token_type_ids == 1) | (mm_token_type_ids == 2)).item()) > 0
        )
        use_bidirectional_vision = (
            getattr(self.config, "use_bidirectional_attention", None) == "vision"
            and mm_token_type_ids is not None
            and has_visual_tokens
            and not has_audio_tokens
            and h.shape[1] > 1
        )
        for lyr, c in zip(self.layers, cache):
            if lyr.layer_type not in mask:
                if lyr.layer_type == "full_attention":
                    return_array = (
                        use_bidirectional_vision
                        or getattr(c, "left_padding", None) is not None
                    )
                    mask["full_attention"] = g4.create_attention_mask(
                        h, c, return_array=return_array
                    )
                elif lyr.layer_type == "sliding_attention":
                    return_array = (
                        h.shape[1] > 1
                        and c is not None
                        and _cache_has_prefix(c)
                    ) or use_bidirectional_vision
                    mask["sliding_attention"] = g4.create_attention_mask(
                        h, c, window_size=self.window_size,
                        return_array=return_array
                    )
                if (
                    use_bidirectional_vision
                    and isinstance(mask[lyr.layer_type], str)
                    and mask[lyr.layer_type] == "causal"
                ):
                    window = (
                        self.window_size
                        if lyr.layer_type == "sliding_attention"
                        else None
                    )
                    mask[lyr.layer_type] = g4.create_causal_mask(
                        h.shape[1], window_size=window
                    )
                if use_bidirectional_vision and isinstance(
                    mask[lyr.layer_type], mx.array
                ):
                    mask[lyr.layer_type] = self._apply_blockwise_bidirectional_overlay(
                        mask[lyr.layer_type],
                        mm_token_type_ids,
                    )
            masks.append(mask[lyr.layer_type])
        return masks

    _make_masks._gmlx_orig = masks_orig
    g4.Gemma4TextModel._make_masks = _make_masks

    attn_orig = getattr(g4.Attention.__call__, "_gmlx_orig",
                        g4.Attention.__call__)

    def _attn_call(self, x, mask=None, cache=None, shared_kv=None,
                   offset=None):
        # Upstream body with the int-offset wrap removed (no per-layer
        # device upload; rope takes ints directly). Array offsets are still
        # snapshotted: update_and_fetch below advances cache.offset with an
        # in-place += that mutates the aliased array, so the query rope
        # would read positions one step ahead of the key rope and gated
        # B>1 decode degenerates (found by gate-cert 2026-07-25, writer
        # pinned by the offset-hunt probes). Everything else verbatim.
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        queries = self.q_norm(queries)

        if shared_kv is not None:
            keys, values = shared_kv
        else:
            keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

            if self.use_k_eq_v:
                values = keys
            else:
                values = self.v_proj(x).reshape(
                    B, L, self.n_kv_heads, self.head_dim)

            offset = cache.offset if cache is not None else 0
            if isinstance(offset, mx.array):
                offset = mx.array(offset)

            keys = self.k_norm(keys)
            keys = keys.transpose(0, 2, 1, 3)
            keys = self.rope(keys, offset=offset)

            values = self.v_norm(values)
            values = values.transpose(0, 2, 1, 3)

            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)

        queries = queries.transpose(0, 2, 1, 3)
        queries = self.rope(queries, offset=offset)

        output = g4.scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output), (keys, values), offset

    _attn_call._gmlx_orig = attn_orig
    g4.Attention.__call__ = _attn_call

    _installed = True
    return True
