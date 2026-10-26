# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""MiniCPM-V text tower: the owned qwen3.5 dense or MoE language model,
with the MiniCPM-V position rule in place of the Qwen3-VL one.

The owned classes carry the speculative hooks, the DFlash capture and the
GDN replay rollback, so the DSpark drafter runs on this tower as on the
plain qwen3.5 text model.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import numpy as np

from gmlx._vendor.mlx_vlm.models.qwen3_5 import config as _dense_cfg
from gmlx._vendor.mlx_vlm.models.qwen3_5_moe import config as _moe_cfg

from .rope import CanvasTokens, canvas_rope_index, sequential_rope_index

TEXT_MODEL_TYPES = ("qwen3_5", "qwen3_5_moe")


class TextConfig:
    """Builds the qwen3.5 dense or MoE text config named by ``model_type``.
    ``update_module_configs`` calls ``TextConfig.from_dict``."""

    @staticmethod
    def from_dict(params: dict):
        mt = params.get("model_type")
        if mt == "qwen3_5_moe":
            return _moe_cfg.TextConfig.from_dict(params)
        if mt == "qwen3_5":
            return _dense_cfg.TextConfig.from_dict(params)
        raise ValueError(
            f"MiniCPM-V text tower {mt!r} is not supported "
            f"(supported: {', '.join(TEXT_MODEL_TYPES)})")


class MiniCPMVPositions:
    """``get_rope_index`` for MiniCPM-V. 4.7 (``mrope_mode == "canvas"``)
    lays image tokens on the canvas; 4.6 and text-only rows count
    sequentially. Returns ``([3, B, L], rope_deltas [B, 1])``."""

    def get_rope_index(
        self,
        input_ids: mx.array,
        image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        target_sizes=None,
    ):
        del image_grid_thw, video_grid_thw  # Qwen3-VL geometry; unused here
        cfg = self.config  # pyright: ignore[reportAttributeAccessIssue]
        ids = np.asarray(input_ids)
        mask = None if attention_mask is None else np.asarray(attention_mask)
        if mask is not None and mask.shape != ids.shape:
            mask = None
        has_image = bool((ids == cfg.image_token_id).any())
        if has_image and getattr(cfg, "mrope_mode", None) == "canvas":
            if target_sizes is None:
                raise ValueError(
                    "MiniCPM-V 4.7 positions need the image crop sizes "
                    "(target_sizes) of a prompt that holds image tokens")
            pos, deltas = canvas_rope_index(
                ids, np.asarray(target_sizes), canvas_tokens(cfg),
                merge=cfg.vision_config.merge_factor, attention_mask=mask)
        else:
            pos, deltas = sequential_rope_index(ids, mask)
        return mx.array(pos, dtype=mx.int32), mx.array(deltas, dtype=mx.int32)


def canvas_tokens(cfg) -> CanvasTokens:
    return CanvasTokens(
        image_token_id=cfg.image_token_id,
        image_start_id=cfg.image_start_id,
        image_end_id=cfg.image_end_id,
        slice_start_id=cfg.slice_start_id,
        slice_end_id=cfg.slice_end_id,
        newline_id=cfg.newline_id,
    )


_CLASSES: dict[str, type] = {}


def language_model_class(text_model_type: str) -> type:
    """The MiniCPM-V language model over the owned qwen3.5 class."""
    cls = _CLASSES.get(text_model_type)
    if cls is None:
        import gmlx.models.qwen35.owned as qwen35_owned

        base = qwen35_owned.language_model_class(text_model_type)
        cls = type(f"MiniCPMV{base.__name__}", (MiniCPMVPositions, base), {})
        _CLASSES[text_model_type] = cls
    return cls
