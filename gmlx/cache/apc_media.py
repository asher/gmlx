# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Config-declared media token ids for the APC media guard.

mlx-vlm's APC keeps reusable prefixes clear of media token runs
(``media_safe_prefix_min`` and the lookup / checkpoint helpers built on
it): a restored prefix must cover every media placeholder so the suffix
embeds as text, and the exact-tier checkpoint column moves past the last
run. The guard learns the media ids from ``image_token_id`` and
``video_token_id`` on the config, which covers processors that repeat one
placeholder id per patch. A processor that expands a placeholder into
several distinct ids (DeepSeek-V4-Flash-Vision-Exp: five sentinel ids past
the vocab, one per block token type) declares them on the config as
``media_token_ids``; this fold makes the stock reader see them. Without
it the guard sees no media in the expanded stream, and the checkpoint
column or a restored prefix can land inside a block, which the text tower
refuses.

The engine reads the config off the model it was built with (the server
hands it ``model.language_model``), so a container mirrors the field onto
its text config as well.
"""

from __future__ import annotations

import importlib

_FLAG = "_kq_media_token_ids"


def install_media_token_ids() -> bool:
    """Fold ``config.media_token_ids`` into ``multimodal_token_ids_from_config``.
    Idempotent; the callers resolve the function through the ``mlx_vlm.apc``
    module attribute, so rebinding it there covers the batch engine, the
    CLI generate path, and gmlx's own views of the engine helpers."""
    apc = importlib.import_module("mlx_vlm.apc")
    if getattr(apc, _FLAG, False):
        return True
    stock = apc.multimodal_token_ids_from_config

    def multimodal_token_ids_from_config(config):
        ids = set(stock(config))
        extra = getattr(config, "media_token_ids", None)
        if extra is None and isinstance(config, dict):
            extra = config.get("media_token_ids")
        if extra:
            ids.update(int(t) for t in extra)
        return ids

    apc.multimodal_token_ids_from_config = multimodal_token_ids_from_config
    setattr(apc, _FLAG, True)
    return True
