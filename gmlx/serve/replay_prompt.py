"""Prompt kwargs for a requeued row's replay.

A requeued row replays its prompt plus every token it delivered. Its queue
entry needs prompt kwargs for all of those ids: the prefill takes its column
count from ``inputs_embeds``, not from the ids, so prompt-only embeddings
leave the delivered tokens out of the KV.

The delivered tokens are text. Their embeddings come from the outer model's
``get_input_embeddings(ids, None)`` under the row's adapter scales, through
an embedder the server sets on the generator. Each sequence-aligned kwarg
extends by its own rule. A row whose kwargs have no rule raises
``ReplayUnsupported``, and the caller fails that row alone.

Everything here builds an MLX graph. Nothing is evaluated until the
prefill runs.
"""

from __future__ import annotations

import logging

import mlx.core as mx

_log = logging.getLogger(__name__)

# Families whose language model holds a prompt's positions, not the row.
_POSITIONS_ON_MODEL = frozenset({
    "glm4v", "glm4v_moe", "glm_ocr", "paddleocr_vl", "hunyuan_vl",
    "ernie4_5_moe_vl", "minicpmo",
})

# Kwargs that are not sequence-aligned, or that hold image rows only.
_KEEP = frozenset({
    "rope_deltas", "image_grid_thw", "video_grid_thw", "image_position_ids",
    "image_sizes", "input_features", "audio_clips", "attention_mask_4d",
    "target_sizes", "pixel_values", "pixel_values_videos",
    "_apc_semantic_hash", "_apc_tenant", "_apc_image_hash",
})

# Kwargs whose sequence has no rule here.
_REFUSED = {
    "decoder_inputs_embeds": "encoder-decoder row",
    "cross_attention_mask": "cross-attention mask",
    "full_text_row_masked_out_mask": "cross-attention mask",
}

_NGRAM_WARNED = [False]


class ReplayUnsupported(ValueError):
    """The row's prompt kwargs have no replay rule."""


def family(lm) -> str:
    """The model family of a language model, from its class's module."""
    parts = type(lm).__module__.split(".")
    if "models" in parts:
        i = len(parts) - 1 - parts[::-1].index("models")
        if i + 1 < len(parts):
            return parts[i + 1]
    return ""


def embed_text(model, ids, row_settings=None) -> dict:
    """The non-None fields of ``model.get_input_embeddings`` over ``ids``,
    text only, under the row's adapter scales. The language model's
    position state is put back after the call: several text-only calls
    reset it, and a queued media row may still need it."""
    from gmlx import lora_rows

    lm = getattr(model, "language_model", model)
    saved = {a: getattr(lm, a) for a in ("_position_ids", "_rope_deltas")
             if hasattr(lm, a)}
    try:
        with lora_rows.tuple_scope(lora_rows.row_tuple(row_settings), rows=1):
            feats = model.get_input_embeddings(mx.array([list(ids)]), None)
    finally:
        for a, v in saved.items():
            setattr(lm, a, v)
    fields = feats.to_dict() if hasattr(feats, "to_dict") else dict(feats)
    return {k: v for k, v in fields.items() if v is not None}


def _positions(pos: mx.array, n: int) -> mx.array:
    """``pos`` followed by ``n`` text positions: the largest prompt position
    plus 1, 2, ... on every axis."""
    step = mx.arange(1, n + 1, dtype=pos.dtype)
    if pos.ndim == 3:                               # [3, B, L] mrope
        top = pos.max(axis=(0, 2))
        tail = mx.broadcast_to(top[None, :, None] + step[None, None, :],
                               (pos.shape[0], pos.shape[1], n))
    else:                                           # [B, L]
        top = pos.max(axis=-1)
        tail = top[:, None] + step[None, :]
    return mx.concatenate([pos, tail], axis=-1)


def _fill(v: mx.array, n: int, value) -> mx.array:
    shape = (v.shape[0], n) + tuple(v.shape[2:])
    return mx.concatenate([v, mx.full(shape, value, dtype=v.dtype)], axis=1)


def _tail_of(tail: dict, key: str, v: mx.array, n: int) -> mx.array:
    t = tail.get(key)
    if not isinstance(t, mx.array) or t.ndim != v.ndim or t.shape[1] != n:
        raise ReplayUnsupported(f"the text embedding gave no {key} for the "
                                f"delivered tokens")
    return mx.concatenate([v, t.astype(v.dtype)], axis=1)


def _aligned(v, length: int) -> bool:
    return isinstance(v, mx.array) and any(
        d == length for d in v.shape[1:])


def replay_prompt_kwargs(kwargs, prompt_len: int, delivered, embed, lm=None):
    """The prompt kwargs for ``prompt + delivered``. ``embed(ids)`` returns
    the text embedding fields of ``ids``; ``lm`` is the language model the
    row runs on. Returns ``kwargs`` itself when nothing is delivered or the
    row has no ``inputs_embeds``. Raises ``ReplayUnsupported``."""
    n = len(delivered)
    if not n or not kwargs or kwargs.get("inputs_embeds") is None:
        return kwargs
    prompt = kwargs["inputs_embeds"]
    if not isinstance(prompt, mx.array):
        # Rows a model builds a chunk at a time are spent by the prefill.
        raise ReplayUnsupported("unbuilt prompt rows")
    if prompt.shape[1] != prompt_len:
        raise ReplayUnsupported(
            f"inputs_embeds holds {prompt.shape[1]} columns for "
            f"{prompt_len} prompt ids")
    for key, why in _REFUSED.items():
        if key in kwargs:
            raise ReplayUnsupported(f"{why} ({key})")
    fam = family(lm) if lm is not None else ""
    if fam in _POSITIONS_ON_MODEL and "position_ids" not in kwargs:
        raise ReplayUnsupported(f"{fam} keeps the prompt's positions on the "
                                "language model")
    if fam == "falcon_ocr" and "attention_mask_4d" in kwargs:
        raise ReplayUnsupported("falcon_ocr block mask (attention_mask_4d)")
    if fam == "longcat_flash_ngram" and not _NGRAM_WARNED[0]:
        _NGRAM_WARNED[0] = True
        _log.warning("replay: %s embeds the delivered tokens without their "
                     "n-gram context, so a resumed reply is not bit-exact",
                     fam)

    tail = embed(list(delivered))
    out = {}
    for k, v in kwargs.items():
        if k == "inputs_embeds":
            out[k] = _tail_of(tail, k, v, n)
        elif k == "per_layer_inputs" and isinstance(v, mx.array):
            out[k] = _tail_of(tail, k, v, n)
        elif k == "deepstack_visual_embeds" and isinstance(v, mx.array):
            if isinstance(tail.get(k), mx.array):
                out[k] = _tail_of(tail, k, v, n)
            else:
                out[k] = _fill(v, n, 0)
        elif k == "position_ids" and isinstance(v, mx.array):
            out[k] = _positions(v, n)
        elif k == "visual_pos_masks" and isinstance(v, mx.array):
            out[k] = _fill(v, n, False)
        elif k in ("mm_token_type_ids", "token_type_ids") and isinstance(
                v, mx.array):
            out[k] = _fill(v, n, 0)
        elif k == "attention_mask" and isinstance(v, mx.array):
            out[k] = _fill(v, n, 1)
        elif k in _KEEP or k.startswith("_apc_"):
            out[k] = v
        elif _aligned(v, prompt_len):
            raise ReplayUnsupported(f"no replay rule for {k}")
        else:
            out[k] = v
    if fam == "qwen4_exp" and "position_ids" not in kwargs:
        # A text row: plain positions on all three axes, which the model
        # reads before its own state and which equal its text path.
        pos = mx.arange(prompt_len + n, dtype=mx.int32)
        out["position_ids"] = mx.broadcast_to(pos[None, None, :],
                                              (3, 1, prompt_len + n))
        out["rope_deltas"] = mx.zeros((1, 1), dtype=mx.int32)
    return out
