"""Owned model-level forward for qwen3.5/3.6 MTP targets.

Replaces mlx-vlm's ``Qwen3_5Model.__call__`` control flow with an owned
subclass (no monkeypatch): the code that runs at the model level is the
code in this file. Layers stay the stock mlx-vlm classes, so the
layer-level fused kernels and verify seams keep engaging unchanged.

Differences from the stock forward, all deliberate:

- The B=1 single-row batch-cache shortcut (extract row cache, recurse,
  merge back) is removed. This is a correctness fix, not a neutral
  cleanup: the stock shortcut extracts a row cache that drops
  ``left_padding`` while recursing with the full unsliced input, so a
  padded single-row batch attends its pad tokens as content. The direct
  mask path honors the pads and matches an unpadded reference exactly.
  Decode-step cost is zero either way, and the quantized-cache rebuild
  hazard the shortcut had to special-case cannot arise at all.
- The S=0 guard is structural. A fully left-padded prefill row embeds to
  an empty sequence; padding rows produce zero hidden states and the
  caller zero-pads the output, so the forward returns ``norm(empty)``
  without an external wrapper patch.
- The batched left-padded prefill path stays per-row (it is live for
  batched prefill through the stock AR engine) but is owned here. Its
  pad probe folds the any-pads check and the per-row pad list into one
  host read: the no-pad case costs the same two syncs as stock, but a
  chunk with query-side pads costs one instead of stock's up to three.
- The decode-path mask resolution and the decode left-padding walk keep
  upstream's cache-attr protocol exactly (``_qwen3_5_decode_left_padding``
  and the memo attrs), because the stock layers consume it. The helpers
  are called through the upstream module object so their single source
  of truth is preserved until the layer classes are owned too.

``GMLX_QWEN_OWNED=0`` falls back to the stock classes at build time
(loader-level switch; see ``loader._mtp_target_classes``).
"""

from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.models.qwen3_5 import language as _L
from mlx_vlm.models.qwen3_5.config import ModelConfig as _Q35ModelConfig
from mlx_vlm.models.qwen3_5.config import TextConfig as _Q35TextConfig

_OWNED_CALLS = 0


def owned_call_count() -> int:
    """Engagement counter: total owned model-level forwards this process.

    Counts recursive per-row prefill calls individually, so a padded
    B-row prefill chunk adds B+1, not 1. Treat deltas as engagement
    proof, not as a step count.
    """
    return _OWNED_CALLS


def _batched_padded_prefill(self, inputs, h, cache, position_ids):
    """Per-row forward for batched left-padded multi-token prefill.

    Returns None when no row carries padding, letting the caller take the
    plain masked path. Row slices drop each row's query-side padding; a
    row whose padding consumes the whole chunk recurses with S=0 and the
    owned forward returns its empty hidden directly. The pads list is
    read from the device once and reused for both the any-pads check and
    the row loop (stock reads it up to three times).
    """
    fa_cache = cache[self.fa_idx]
    query_left_padding = mx.minimum(mx.maximum(-fa_cache.offset, 0), h.shape[1])
    pads = [int(p) for p in query_left_padding.tolist()]
    if max(pads) <= 0:
        cache_left_padding = getattr(fa_cache, "left_padding", None)
        if not (
            isinstance(cache_left_padding, mx.array)
            and cache_left_padding.ndim > 0
            and int(cache_left_padding.max().item()) > 0
        ):
            return None

    row_outputs = []
    row_caches = [[] for _ in cache]
    for row, pad in enumerate(pads):
        pad = min(max(pad, 0), h.shape[1])
        row_inputs = inputs[row : row + 1, pad:]
        row_embeds = h[row : row + 1, pad:]
        row_position_ids = None
        if position_ids is not None:
            if position_ids.ndim == 2:
                row_position_ids = position_ids[row : row + 1, pad:]
            else:
                row_position_ids = position_ids[:, row : row + 1, pad:]
        current_cache = []
        for cache_entry in cache:
            if cache_entry is None:
                current_cache.append(None)
            else:
                current_cache.append(_L._extract_row_cache(cache_entry, row))

        row_out = self(
            row_inputs,
            inputs_embeds=row_embeds,
            cache=current_cache,
            position_ids=row_position_ids,
        )
        if pad > 0:
            row_out = _L._pad_row_time(row_out, pad, h.shape[1])
        row_outputs.append(row_out)
        for i, cache_entry in enumerate(current_cache):
            row_caches[i].append(cache_entry)

    for i, entries in enumerate(row_caches):
        if cache[i] is None:
            continue
        if hasattr(cache[i].__class__, "merge"):
            cache[i] = cache[i].__class__.merge(entries)
    return mx.concatenate(row_outputs, axis=0)


def _owned_model_call(
    self,
    inputs: mx.array,
    inputs_embeds: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
    cache=None,
    position_ids: Optional[mx.array] = None,
    capture_layer_ids: Optional[List[int]] = None,
    hidden_sink: Optional[list] = None,
    gdn_sink: Optional[list] = None,
):
    # mask is accepted for signature parity with the stock forward, which
    # also ignores it: per-layer masks are always resolved from the cache.
    del mask
    global _OWNED_CALLS
    _OWNED_CALLS += 1

    if inputs_embeds is None:
        h = self.embed_tokens(inputs)
    else:
        h = inputs_embeds

    if h.shape[1] == 0:
        return self.norm(h)

    if cache is None:
        cache = [None] * len(self.layers)

    fa_cache = cache[self.fa_idx]
    if (
        h.shape[0] > 1
        and h.shape[1] > 1
        and hidden_sink is None
        and gdn_sink is None
        and fa_cache is not None
        and hasattr(fa_cache, "extract")
        and hasattr(fa_cache.__class__, "merge")
        and isinstance(getattr(fa_cache, "offset", None), mx.array)
        and fa_cache.offset.ndim > 0
    ):
        out = _batched_padded_prefill(self, inputs, h, cache, position_ids)
        if out is not None:
            return out

    fa_mask = _L._create_qwen3_5_attention_mask(h, cache[self.fa_idx])
    ssm_mask = _L._create_qwen3_5_ssm_mask(h, cache[self.ssm_idx])
    decode_left_padding = (
        getattr(cache[self.fa_idx], "_qwen3_5_decode_left_padding", None)
        if isinstance(fa_mask, str) and fa_mask == "left_padded_decode"
        else None
    )
    _L._set_qwen3_5_decode_left_padding(cache, self.layers, decode_left_padding)

    position_embeddings = None
    if position_ids is not None:
        for layer in self.layers:
            if not layer.is_linear:
                if not layer.self_attn.rotary_emb.fused_apply:
                    position_embeddings = layer.self_attn.rotary_emb(
                        h, position_ids
                    )
                break

    capture_set = set(capture_layer_ids) if capture_layer_ids else set()
    for i, (layer, c) in enumerate(zip(self.layers, cache)):
        layer_mask = ssm_mask if layer.is_linear else fa_mask
        h = layer(
            h,
            mask=layer_mask,
            cache=c,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            gdn_sink=gdn_sink,
            target_verify=gdn_sink is not None,
        )
        if hidden_sink is not None and i in capture_set:
            hidden_sink.append(h)

    return self.norm(h)


class OwnedQwen3_5Model(_L.Qwen3_5Model):
    __call__ = _owned_model_call


class OwnedQwen3_5LanguageModel(_L.LanguageModel):
    """Stock LanguageModel wrapper over the owned model scaffold.

    The wrapper __call__ (mrope position resolution, sinks, head) and all
    speculative_* hooks are inherited stock; only the inner model class
    changes. __init__ mirrors the stock body instead of calling it so the
    stock Qwen3_5Model is never built and thrown away.
    """

    def __init__(self, args: _Q35TextConfig, config: _Q35ModelConfig = None):
        nn.Module.__init__(self)
        self.args = args
        self.config = config
        self.model_type = args.model_type
        self.model = OwnedQwen3_5Model(args)
        self._position_ids = None
        self._rope_deltas = None

        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)


def _moe_classes():
    from mlx_vlm.models.qwen3_5_moe import language as _ML

    class OwnedQwen3_5MoeModel(_ML.Qwen3_5MoeModel):
        __call__ = _owned_model_call

    class OwnedQwen3_5MoeLanguageModel(_ML.LanguageModel):
        def __init__(self, args, config=None):
            nn.Module.__init__(self)
            self.args = args
            self.config = config
            self.model_type = args.model_type
            self.model = OwnedQwen3_5MoeModel(args)
            self._rope_deltas = None
            self._position_ids = None

            if not args.tie_word_embeddings:
                self.lm_head = nn.Linear(
                    args.hidden_size, args.vocab_size, bias=False
                )

    return OwnedQwen3_5MoeModel, OwnedQwen3_5MoeLanguageModel


_MOE_CACHE = None


def language_model_class(model_type: str):
    """Owned LanguageModel class for a qwen3.5 family model_type."""
    global _MOE_CACHE
    if model_type == "qwen3_5":
        return OwnedQwen3_5LanguageModel
    if model_type == "qwen3_5_moe":
        if _MOE_CACHE is None:
            _MOE_CACHE = _moe_classes()
        return _MOE_CACHE[1]
    raise ValueError(f"no owned qwen3.5 forward for model_type {model_type!r}")
