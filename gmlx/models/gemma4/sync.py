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

Scope is the mlx-vlm module only, deliberately (the row route patches
both seams, this module does not): mlx-lm's gemma4_text has no offset
probe, so there is no host sync to remove there. Its only per-layer cost
is the 0-d ``mx.array(cache.offset)`` wrap, an async upload that is also
what keeps rope off the cache's in-place-mutating offset (and off the
upstream scalar-offset rope kernel bug rope_batch_fix covers). Removing
it would buy microseconds against two known hazard classes.

Install is idempotent; GMLX_G4_NOSYNC=0 disables. Both replaced bodies are
copies of the pinned upstream implementations (seam-fingerprinted in
upstream_seams.json); module-level collaborators (create_attention_mask,
scaled_dot_product_attention, ...) are resolved through the upstream module
at call time so other patches compose.

The same install also wraps the gemma4 chunking policies and the mask
builder, correctness fixes that GMLX_G4_NOSYNC=0 leaves in place. Under
bidirectional image attention the image overlay applies with or without
audio, as in transformers, so a prompt holding image or video tokens
prefills in one pass. Stock allows chunking once audio is present. The
installed builder runs the carried body under ``image_masks``, which keeps
full-attention layers causal and puts the image blocks on the sliding
layers only; with GMLX_G4_NOSYNC=0 the stock body runs under it.
"""
from __future__ import annotations

import mlx.core as mx

from gmlx.envflags import env_bool

from .image_masks import image_masks
from .image_masks import visual_tokens as _visual_tokens

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


def _token_types(kwargs: dict):
    types = kwargs.get("mm_token_type_ids", None)
    return kwargs.get("token_type_ids", None) if types is None else types


def _visual_request(config, input_ids, kwargs: dict) -> bool:
    """Image or video tokens in the request, from its token types, else
    from its ids."""
    types = _token_types(kwargs)
    if types is not None:
        return _visual_tokens(types)
    if input_ids is None:
        return False
    ids = [getattr(config, k, None) for k in ("image_token_id", "video_token_id")]
    return any(i is not None and int(mx.sum(input_ids == i).item()) > 0
               for i in ids)


def _install_image_chunking_policy() -> None:
    """Decline chunked prefill for a prompt holding image or video tokens
    under bidirectional image attention, whether or not it holds audio.
    Stock allows chunking once audio is present, while the image overlay
    still needs the whole prompt in one pass from column 0. Idempotent."""
    try:
        from mlx_vlm.models.gemma4 import language as g4
    except ImportError:
        return
    policy = getattr(g4.LanguageModel, "chunked_prefill_policy", None)
    if policy is not None and getattr(policy, "_gmlx_orig", None) is None:
        def chunked_prefill_policy(self, **kwargs):
            if (getattr(self.config, "use_bidirectional_attention", None)
                    == "vision"
                    and _visual_tokens(
                        _token_types(kwargs.get("prefill_kwargs") or {}))):
                return False
            return policy(self, **kwargs)

        chunked_prefill_policy.__dict__["_gmlx_orig"] = policy
        g4.LanguageModel.chunked_prefill_policy = chunked_prefill_policy

    try:
        from mlx_vlm.models.gemma4_unified import gemma4_unified as g4u
    except ImportError:
        return
    decline = getattr(g4u.Model, "_should_disable_chunked_prefill", None)
    if decline is not None and getattr(decline, "_gmlx_orig", None) is None:
        def _should_disable_chunked_prefill(self, input_ids=None, **kwargs):
            if (getattr(self, "_base_no_chunked_prefill", False)
                    and _visual_request(self.config, input_ids, kwargs)):
                return True
            return decline(self, input_ids, **kwargs)

        _should_disable_chunked_prefill.__dict__["_gmlx_orig"] = decline
        g4u.Model._should_disable_chunked_prefill = (
            _should_disable_chunked_prefill)


def _install_stock_image_masks() -> None:
    """Wrap the stock mask builder with ``image_masks`` when no gmlx
    builder is installed, as under GMLX_G4_NOSYNC=0. Idempotent."""
    try:
        from mlx_vlm.models.gemma4 import language as g4
    except ImportError:
        return
    stock = g4.Gemma4TextModel._make_masks
    if getattr(stock, "_gmlx_orig", None) is not None:
        return

    def _make_masks(self, h, cache, mm_token_type_ids=None):
        return image_masks(self, h, cache, mm_token_type_ids, stock)

    _make_masks.__dict__.update(_gmlx_orig=stock, _gmlx_body=stock)
    g4.Gemma4TextModel._make_masks = _make_masks


def install_gemma4_nosync() -> bool:
    """Replace the two host-syncing gemma4 call sites. Idempotent; no-op when
    GMLX_G4_NOSYNC=0 or the gemma4 module is unavailable. Returns True if the
    patch is active. The image chunking policy and the image masks install
    whatever GMLX_G4_NOSYNC says."""
    global _installed
    _install_image_chunking_policy()
    if not env_bool("GMLX_G4_NOSYNC", True):
        _install_stock_image_masks()
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
        has_visual_tokens = (
            mm_token_type_ids is not None
            and int(mx.sum(
                (mm_token_type_ids == 1) | (mm_token_type_ids == 2)).item()) > 0
        )
        use_bidirectional_vision = (
            getattr(self.config, "use_bidirectional_attention", None) == "vision"
            and mm_token_type_ids is not None
            and has_visual_tokens
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

    def _make_image_masks(self, h, cache, mm_token_type_ids=None):
        return image_masks(self, h, cache, mm_token_type_ids, _make_masks)

    _make_image_masks.__dict__.update(_gmlx_orig=masks_orig,
                                      _gmlx_body=_make_masks)
    g4.Gemma4TextModel._make_masks = _make_image_masks

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
