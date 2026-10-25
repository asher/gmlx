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
its text config as well. The VLM loader mirrors a gemma-4 container's
image and video ids the same way when the text config turns on
bidirectional image attention. Only then does a restored prefix that ends
inside an image change the logits, so causal models keep reusing such
prefixes.

For that gemma-4 text config the reader returns ``WholeMediaBlocks``, and
the guard helpers accept any prefix that ends outside every image or
video run: the server embeds the whole prompt and slices each row's
suffix from it, and the prefill loops keep each block inside one chunk
(``gmlx.gen.media_spans``). The container config keeps the stock rule,
since run and chat drop the pixel values when they reuse a prefix.

On the serve engine a Qwen3.5 model reads ``CausalMedia``
(``install_causal_serve_media``), and the guard helpers accept any
prefix: its attention is causal, and the server slices each row's
suffix embeddings and mrope position ids from the whole prompt, so a
suffix may hold images.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Sequence

from gmlx.upstream.patching import wrapped_with

_FLAG = "_kq_media_token_ids"
_CAUSAL_FLAG = "_kq_causal_serve_media"
_WHOLE_BLOCK_TYPES = ("gemma4_text", "gemma4_unified_text")
_CAUSAL_SERVE_TYPES = ("qwen3_5", "qwen3_5_moe", "minicpmv4_6", "minicpmv4_7")


class WholeMediaBlocks(set):
    """Media ids whose runs a restored prefix must not end inside. A prefix
    before, between or after the runs is safe."""


class CausalMedia(set):
    """Media ids of a causal model on the serve engine. Every prefix is
    safe, and the suffix may hold images."""


def _whole_blocks(config) -> bool:
    return (getattr(config, "use_bidirectional_attention", None) == "vision"
            and getattr(config, "model_type", None) in _WHOLE_BLOCK_TYPES)


def safe_prefix(token_ids: Sequence[int],
                media_token_ids: Iterable[int]) -> Callable[[int], bool]:
    """The guard's test for a prefix length of ``token_ids``, with the
    prompt scanned once: outside every run for ``WholeMediaBlocks``, else
    at or past the end of the last run."""
    apc = importlib.import_module("gmlx._vendor.mlx_vlm.apc")
    if isinstance(media_token_ids, CausalMedia):
        return lambda p: True
    if isinstance(media_token_ids, WholeMediaBlocks):
        spans = apc.media_token_spans(token_ids, media_token_ids)
        return lambda p: not any(a < int(p) < b for a, b in spans)
    floor = apc.media_safe_prefix_min(token_ids, media_token_ids)
    return lambda p: int(p) >= floor


def _run_end(token_ids: Sequence[int], prefix_len: int,
             media: Iterable[int]) -> int | None:
    """End of the media run that ``prefix_len`` falls strictly inside."""
    apc = importlib.import_module("gmlx._vendor.mlx_vlm.apc")
    p = int(prefix_len)
    for a, b in apc.media_token_spans(token_ids, media):
        if a < p < b:
            return b
    return None


def install_media_token_ids() -> bool:
    """Fold ``config.media_token_ids`` into ``multimodal_token_ids_from_config``
    and give the guard helpers the whole-block rule for ``WholeMediaBlocks``.
    Idempotent; the callers resolve the functions through the ``mlx_vlm.apc``
    module attributes, so rebinding them there covers the batch engine, the
    APC coordinator, the CLI generate path, and gmlx's own views of the
    engine helpers."""
    apc = importlib.import_module("gmlx._vendor.mlx_vlm.apc")
    if getattr(apc, _FLAG, False):
        return True
    stock = apc.multimodal_token_ids_from_config
    stock_min = apc.media_safe_prefix_min
    stock_leaves = apc.prefix_leaves_text_only_suffix
    stock_adjust = apc.adjust_prefix_to_text_suffix_boundary

    def multimodal_token_ids_from_config(config):
        ids = set(stock(config))
        extra = getattr(config, "media_token_ids", None)
        if extra is None and isinstance(config, dict):
            extra = config.get("media_token_ids")
        if extra:
            ids.update(int(t) for t in extra)
        return WholeMediaBlocks(ids) if _whole_blocks(config) else ids

    def media_safe_prefix_min(token_ids, media_token_ids):
        if isinstance(media_token_ids, (WholeMediaBlocks, CausalMedia)):
            return 0
        return stock_min(token_ids, media_token_ids)

    def prefix_leaves_text_only_suffix(token_ids, prefix_len, media_token_ids):
        if isinstance(media_token_ids, CausalMedia):
            return True
        if isinstance(media_token_ids, WholeMediaBlocks):
            return _run_end(token_ids, prefix_len, media_token_ids) is None
        return stock_leaves(token_ids, prefix_len, media_token_ids)

    def adjust_prefix_to_text_suffix_boundary(token_ids, desired_prefix_len,
                                              media_token_ids, *,
                                              max_prefix_tokens=None):
        if not isinstance(media_token_ids, (WholeMediaBlocks, CausalMedia)):
            return stock_adjust(token_ids, desired_prefix_len,
                                media_token_ids,
                                max_prefix_tokens=max_prefix_tokens)
        max_len = (len(token_ids) - 1 if max_prefix_tokens is None
                   else int(max_prefix_tokens))
        desired = int(desired_prefix_len)
        if max_len <= 0 or desired <= 0:
            return 0
        end = (None if isinstance(media_token_ids, CausalMedia)
               else _run_end(token_ids, desired, media_token_ids))
        prefix_len = desired if end is None else end
        return 0 if prefix_len > max_len else prefix_len

    apc.multimodal_token_ids_from_config = multimodal_token_ids_from_config
    apc.media_safe_prefix_min = media_safe_prefix_min
    apc.prefix_leaves_text_only_suffix = prefix_leaves_text_only_suffix
    apc.adjust_prefix_to_text_suffix_boundary = (
        adjust_prefix_to_text_suffix_boundary)
    setattr(apc, _FLAG, True)
    return True


def install_causal_serve_media() -> bool:
    """Give the serve engine's media guard ``CausalMedia`` for a Qwen3.5
    or MiniCPM-V model (``_CAUSAL_SERVE_TYPES``), so a turn that adds an image restores
    the earlier turns. Wraps ``BatchGenerator._apc_media_token_ids``,
    which the owned MTP path reads through ``_L1View``. Run and chat read
    the config directly and keep the stock floor. Idempotent."""
    ar = importlib.import_module("gmlx._vendor.mlx_vlm.generate.ar")
    stock = ar.BatchGenerator._apc_media_token_ids
    if wrapped_with(stock, _CAUSAL_FLAG):
        return True

    def _apc_media_token_ids(self):
        ids = stock(self)
        config = getattr(getattr(self, "model", None), "config", None)
        if (ids and not isinstance(ids, WholeMediaBlocks)
                and getattr(config, "model_type", None)
                in _CAUSAL_SERVE_TYPES):
            return CausalMedia(ids)
        return ids

    setattr(_apc_media_token_ids, _CAUSAL_FLAG, True)
    _apc_media_token_ids.__wrapped__ = stock  # type: ignore[attr-defined]
    ar.BatchGenerator._apc_media_token_ids = _apc_media_token_ids
    # The owned MTP view grafts the engine's helpers once; a graft taken
    # before this install gets the wrapper too.
    from gmlx.spec import engine

    if engine._L1_BOUND[0]:
        engine._L1View._apc_media_token_ids = _apc_media_token_ids  # type: ignore[attr-defined]
    return True
