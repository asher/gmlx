# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Vendored mlx-vlm model for Meta Muse Glimmer (mmproj projector ``muse-glimmer``).

mlx-vlm has no muse_glimmer package, so this module supplies the vision half of
the pair: a 50-layer ViT, the pixel-shuffle downsample, and the adapter MLP that
lands in the text tower's residual width. The text half is the same vendored
class the text-only path uses (:mod:`gmlx.muse_glimmer_model`), wrapped here in
the ``language_model`` shape mlx-vlm's generate stack expects.

The tower is ported from llama.cpp's ``clip_graph_muse_glimmer::build`` plus the
host-side index math in ``clip.cpp`` (``PROJECTOR_TYPE_MUSE_GLIMMER`` set_input).
Four mechanics are specific to this family:

  1. Patches are reordered into 32x32 windows and 3 of every 4 layers attend
     only within a window; every 4th layer and the last one are global.
  2. 2-D RoPE: the first half of each head's dimensions is rotated by the patch's
     1-indexed column, the second half by its row, both on the same frequency
     ladder (llama.cpp ``build_rope_2d`` with ``interleave_freq`` false).
  3. The learned 32x32 position grid is bilinearly resampled to the image's patch
     grid, matching ggml's non-antialiased ``GGML_SCALE_MODE_BILINEAR``.
  4. Output tokens are pixel-shuffled 2x2 channel-outer (1536 -> 6144) before the
     adapter, so one soft token covers a 28x28 pixel cell.

Q/K pass through un-permuted for the same reason the text tower's do: the
converter emits the interleaved layout llama.cpp's rope mode 0 consumes.
"""

from __future__ import annotations

import importlib
import math
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.models.base import BaseModelConfig, InputEmbeddingsFeatures, LanguageModelOutput
from mlx_vlm.models.cache import KVCache, RotatingKVCache
from mlx_vlm.models.interpolate import bilinear_interpolate

from .muse_glimmer_model import MuseGlimmerModel
from .muse_glimmer_mtp import SpecHooks, _SpecOutput


def ensure_registered() -> None:
    """Make ``mlx_vlm.models.muse_glimmer`` resolve, preferring upstream."""
    if "mlx_vlm.models.muse_glimmer" not in sys.modules:
        try:
            importlib.import_module("mlx_vlm.models.muse_glimmer")  # upstream wins
        except ImportError:
            sys.modules["mlx_vlm.models.muse_glimmer"] = sys.modules[__name__]


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "muse_glimmer"
    hidden_size: int = 6656
    intermediate_size: int = 19968
    num_hidden_layers: int = 52
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    head_dim: int = 128
    vocab_size: int = 202048
    layer_types: List[str] = field(default_factory=list)
    sliding_window: int = 2048
    rms_norm_eps: float = 1e-5
    post_norm_eps: float = 1e-8
    rope_theta: float = 500000.0
    rope_parameters: Optional[dict] = None
    max_position_embeddings: int = 131072
    output_multiplier: float = 1.0
    final_logit_softcapping: float = 0.0
    tie_word_embeddings: bool = False


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "muse_glimmer"
    num_hidden_layers: int = 50
    hidden_size: int = 1536
    intermediate_size: int = 8960
    num_attention_heads: int = 16
    image_size: int = 896
    patch_size: int = 14
    num_channels: int = 3
    projection_dim: int = 6656
    adapter_hidden_size: int = 4096
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    spatial_merge_size: int = 2
    # 3 window layers then 1 global, repeating; the last layer is always global.
    sparse_factor: int = 4
    num_position_embeddings: int = 1024


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = field(default_factory=TextConfig)
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    model_type: str = "muse_glimmer"
    image_token_id: int = 200092
    image_token_index: Optional[int] = None
    vocab_size: int = 202048
    eos_token_id: Optional[List[int]] = None

    def __post_init__(self):
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id


# Grid index math (pure functions of the patch grid; unit-tested)

def window_order(grid_w: int, grid_h: int, window: int) -> tuple[list[int], list[int]]:
    """Patch order that makes window attention block-diagonal.

    Returns ``(perm, segment)``: ``perm[i]`` is the row-major patch index sitting
    at permuted position ``i``, and ``segment[i]`` is its window id, so the
    attention mask is ``segment[:, None] == segment[None, :]``. Windows on the
    right and bottom edges are partial, exactly as llama.cpp builds them.
    """
    perm: list[int] = []
    segment: list[int] = []
    win_id = 0
    for wy in range(0, grid_h, window):
        for wx in range(0, grid_w, window):
            count = 0
            for gy in range(wy, min(wy + window, grid_h)):
                for gx in range(wx, min(wx + window, grid_w)):
                    perm.append(gy * grid_w + gx)
                    segment.append(win_id)
                    count += 1
            if count:
                win_id += 1
    return perm, segment


def window_partition(
    grid_w: int, grid_h: int, window: int
) -> tuple[list[int], list[tuple[int, int, int]]]:
    """Patch order that makes window attention batchable without a mask.

    Same window membership as :func:`window_order`, but windows are laid out
    grouped by size (largest first) instead of row-major, so each group is a
    contiguous run of equal-length windows. Returns ``(perm, groups)`` with
    ``groups`` entries ``(start, n_windows, window_len)``: rows
    ``perm[start : start + n_windows * window_len]`` reshape to
    ``[n_windows, window_len]`` and attend without any mask. A grid has at
    most four sizes (interior, right edge, bottom edge, corner), and window
    order within a group stays row-major. Attention is permutation-invariant
    over its keys, so the layout change cannot alter the math.
    """
    windows: dict[int, list[list[int]]] = {}
    for wy in range(0, grid_h, window):
        for wx in range(0, grid_w, window):
            rows = [
                gy * grid_w + gx
                for gy in range(wy, min(wy + window, grid_h))
                for gx in range(wx, min(wx + window, grid_w))
            ]
            if rows:
                windows.setdefault(len(rows), []).append(rows)
    perm: list[int] = []
    groups: list[tuple[int, int, int]] = []
    for length in sorted(windows, reverse=True):
        group = windows[length]
        groups.append((len(perm), len(group), length))
        for rows in group:
            perm.extend(rows)
    return perm, groups


def pixel_shuffle_order(grid_w: int, grid_h: int, merge: int) -> list[int]:
    """Gather order that groups each ``merge`` x ``merge`` cell contiguously, in
    row-major cell order (llama.cpp's ``ds_perm``)."""
    order: list[int] = []
    for oy in range(grid_h // merge):
        for ox in range(grid_w // merge):
            for ry in range(merge):
                for rx in range(merge):
                    order.append((oy * merge + ry) * grid_w + (ox * merge + rx))
    return order


def _rope_tables(pos: mx.array, half_dim: int, base: float):
    """cos/sin for an interleaved rope over ``half_dim`` dims at the given
    integer positions: pair ``j`` turns at ``base ** (-2j / half_dim)``."""
    n_pair = half_dim // 2
    inv = mx.exp(
        -mx.arange(n_pair, dtype=mx.float32) * (math.log(base) * 2.0 / half_dim))
    theta = pos.astype(mx.float32)[:, None] * inv[None, :]
    return mx.cos(theta), mx.sin(theta)


def _rope_half(v: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Interleaved (pairwise) rotation of ``[B, H, L, D]`` by per-position
    tables of shape ``[L, D // 2]`` (or already broadcast to 4-D, e.g.
    ``[B, 1, L, D // 2]`` for window-batched attention)."""
    B, H, L, D = v.shape
    v = v.reshape(B, H, L, D // 2, 2)
    x0, x1 = v[..., 0], v[..., 1]
    if cos.ndim == 2:
        cos, sin = cos[None, None], sin[None, None]
    c, s = cos.astype(v.dtype), sin.astype(v.dtype)
    return mx.stack([x0 * c - x1 * s, x0 * s + x1 * c], axis=-1).reshape(B, H, L, D)


def _rope_2d(x: mx.array, tables_w, tables_h) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate(
        [_rope_half(x[..., :half], *tables_w), _rope_half(x[..., half:], *tables_h)],
        axis=-1,
    )


# Vision tower

class VisionAttention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_dim = dim // self.n_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.o_proj = nn.Linear(dim, dim, bias=True)

    def __call__(self, x: mx.array, tables_w, tables_h, groups) -> mx.array:
        """``groups`` is None for a global layer (full attention over all
        patches), or the :func:`window_partition` groups for a window layer:
        each group's equal-length windows run as one unmasked batched SDPA,
        skipping the dense scores a block-diagonal mask would compute."""
        B, L, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if groups is None:
            shape = (B, L, self.n_heads, self.head_dim)
            q = _rope_2d(q.reshape(shape).transpose(0, 2, 1, 3), tables_w, tables_h)
            k = _rope_2d(k.reshape(shape).transpose(0, 2, 1, 3), tables_w, tables_h)
            v = v.reshape(shape).transpose(0, 2, 1, 3)
            out = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.scale, mask=None)
            return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))

        outs = []
        for start, n_win, w_len in groups:
            end = start + n_win * w_len
            shape = (n_win, w_len, self.n_heads, self.head_dim)

            def _win(t):
                return t[:, start:end].reshape(shape).transpose(0, 2, 1, 3)

            def _tabs(tables):
                return tuple(t[start:end].reshape(n_win, 1, w_len, -1)
                             for t in tables)

            tw, th = _tabs(tables_w), _tabs(tables_h)
            qg = _rope_2d(_win(q), tw, th)
            kg = _rope_2d(_win(k), tw, th)
            og = mx.fast.scaled_dot_product_attention(
                qg, kg, _win(v), scale=self.scale, mask=None)
            outs.append(og.transpose(0, 2, 1, 3).reshape(1, n_win * w_len, -1))
        out = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=1)
        return self.o_proj(out)


class VisionMLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class VisionLayer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        eps = config.layer_norm_eps
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=eps)
        self.self_attn = VisionAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=eps)
        self.mlp = VisionMLP(config)

    def __call__(self, x: mx.array, tables_w, tables_h, groups) -> mx.array:
        x = x + self.self_attn(self.layer_norm1(x), tables_w, tables_h, groups)
        return x + self.mlp(self.layer_norm2(x))


class VisionModel(nn.Module):
    """The ViT alone: pixels in, post-normed patch features in row-major grid
    order out. The window permutation is applied and undone internally, so the
    caller never sees the sparse-attention ordering."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        patch = config.patch_size
        self.patch_embed = nn.Conv2d(
            config.num_channels, config.hidden_size, kernel_size=patch, stride=patch,
            bias=False)
        self.position_embedding = mx.zeros(
            (config.num_position_embeddings, config.hidden_size))
        self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = [VisionLayer(config) for _ in range(config.num_hidden_layers)]
        self.post_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        # The window side is the learned position grid's side (32), not a
        # separate hyperparameter (clip.cpp derives it the same way).
        self.window = int(round(math.sqrt(config.num_position_embeddings)))

    def _position_embedding(self, grid_w: int, grid_h: int) -> mx.array:
        side = self.window
        if grid_w == side and grid_h == side:
            return self.position_embedding
        grid = self.position_embedding.reshape(side, side, -1)
        # Stays f32: the add promotes anyway, and rounding the interpolated
        # table back to the f16 weight dtype would only lose precision.
        resized = bilinear_interpolate(grid.astype(mx.float32), grid_h, grid_w)
        return resized.reshape(grid_h * grid_w, -1)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        """``pixel_values`` is a single image as ``[1, H, W, C]``."""
        patch = self.config.patch_size
        grid_h = pixel_values.shape[1] // patch
        grid_w = pixel_values.shape[2] // patch

        x = self.patch_embed(pixel_values).reshape(1, grid_h * grid_w, -1)
        x = x + self._position_embedding(grid_w, grid_h)[None]

        perm, groups = window_partition(grid_w, grid_h, self.window)
        perm = mx.array(perm)

        x = self.pre_layernorm(x)
        x = mx.take(x, perm, axis=1)

        # 1-indexed column/row of each patch, in the permuted order.
        pos_w = perm % grid_w + 1
        pos_h = perm // grid_w + 1
        half = (self.config.hidden_size // self.config.num_attention_heads) // 2
        tables_w = _rope_tables(pos_w, half, self.config.rope_theta)
        tables_h = _rope_tables(pos_h, half, self.config.rope_theta)

        n_layer = len(self.layers)
        sf = self.config.sparse_factor
        for idx, layer in enumerate(self.layers):
            is_global = idx == n_layer - 1 or (idx + 1) % sf == 0
            x = layer(x, tables_w, tables_h, None if is_global else groups)

        x = self.post_layernorm(x)
        inverse = mx.zeros(perm.shape, dtype=mx.int32)
        inverse[perm] = mx.arange(perm.size, dtype=mx.int32)
        return mx.take(x, inverse, axis=1)[0]


class VisionAdapter(nn.Module):
    """The mmproj's two-layer adapter; the third linear lives in the LLM as
    ``vision_projection``, matching where the HF checkpoint keeps it."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        merged = config.hidden_size * config.spatial_merge_size**2
        self.fc1 = nn.Linear(merged, config.adapter_hidden_size, bias=False)
        self.fc2 = nn.Linear(
            config.adapter_hidden_size, config.adapter_hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu(self.fc2(nn.gelu(self.fc1(x))))


# Text tower, in the shape mlx-vlm's generate stack expects

class LanguageModel(SpecHooks, nn.Module):
    """The text tower, carrying the same speculative hooks as the text-only
    target so ``--mmproj`` and ``--speculative`` compose on one model."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = MuseGlimmerModel(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        return_hidden: bool = False,
        return_shared_kv: bool = False,
        **kwargs,
    ):
        # The backbone builds its own sliding/full masks from the caches.
        del mask, kwargs
        want_hidden = return_hidden or return_shared_kv
        if want_hidden and self._dflash_capture is not None:
            h, caps = self.model(
                inputs, cache, capture_layers=self._dflash_capture,
                inputs_embeds=inputs_embeds)
            return _SpecOutput(logits=self._spec_logits(h),
                               hidden_states=[self._dflash_pack(h, caps)])
        h = self.model(inputs, cache=cache, inputs_embeds=inputs_embeds)
        logits = self._spec_logits(h)
        if not want_hidden:
            return LanguageModelOutput(logits=logits)
        return _SpecOutput(logits=logits, hidden_states=[h])

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.config.head_dim

    @property
    def n_kv_heads(self):
        return self.config.num_key_value_heads

    def make_cache(self):
        return [
            RotatingKVCache(max_size=self.config.sliding_window, keep=0)
            if layer.use_sliding
            else KVCache()
            for layer in self.model.layers
        ]


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.vision_tower = VisionModel(config.vision_config)
        self.vision_adapter = VisionAdapter(config.vision_config)
        self.vision_projection = nn.Linear(
            config.vision_config.adapter_hidden_size,
            config.vision_config.projection_dim, bias=False)
        self.language_model = LanguageModel(config.text_config)

    def _image_features(self, pixel_values: mx.array, image_sizes) -> mx.array:
        """One padded ``[N, C, H, W]`` batch plus its true ``(h, w)`` sizes ->
        ``[total_soft_tokens, text_hidden]``. Images are run one at a time: the
        patch grid sets the window layout and the rope positions, so a padded
        batch would attend over padding."""
        merge = self.config.vision_config.spatial_merge_size
        patch = self.config.vision_config.patch_size
        feats = []
        for i, (h, w) in enumerate(image_sizes):
            image = pixel_values[i, :, :h, :w].transpose(1, 2, 0)[None]
            # f32 activations against F16 weights (the loader's f16_keep set):
            # dtype promotion computes the whole stack in f32, the oracle's own
            # layout, at half the resident bytes of an fp32 weight pin.
            x = self.vision_tower(image.astype(mx.float32))
            grid_h, grid_w = h // patch, w // patch
            order = mx.array(pixel_shuffle_order(grid_w, grid_h, merge))
            n_out = (grid_h // merge) * (grid_w // merge)
            x = mx.take(x, order, axis=0).reshape(n_out, merge * merge, -1)
            x = x.transpose(0, 2, 1).reshape(n_out, -1)
            feats.append(self.vision_projection(self.vision_adapter(x)))
        return mx.concatenate(feats, axis=0)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        embeds = self.language_model.model.embed_tokens(input_ids)
        if pixel_values is None:
            return InputEmbeddingsFeatures(inputs_embeds=embeds)

        features = kwargs.get("cached_image_features")
        if features is None:
            sizes = kwargs.get("image_sizes")
            if sizes is None:
                sizes = [pixel_values.shape[-2:]] * pixel_values.shape[0]
            features = self._image_features(
                pixel_values, [(int(h), int(w)) for h, w in sizes])
        return InputEmbeddingsFeatures(
            inputs_embeds=self.merge_input_ids_with_image_features(
                self.config.image_token_index, features, embeds, input_ids))

    @staticmethod
    def merge_input_ids_with_image_features(
        image_token_index, image_features, inputs_embeds, input_ids
    ):
        """Scatter ``image_features`` onto the placeholder positions, in order."""
        if image_features.ndim == 3 and image_features.shape[0] == 1:
            image_features = image_features.squeeze(0)
        positions = input_ids == image_token_index
        n_slots = int(mx.sum(positions).item())
        if n_slots != image_features.shape[0]:
            raise ValueError(
                f"{n_slots} image placeholder tokens but "
                f"{image_features.shape[0]} image features")
        if n_slots == 0:
            return inputs_embeds
        features = image_features.astype(inputs_embeds.dtype)
        rank = mx.cumsum(positions.astype(mx.int32).reshape(-1)) - 1
        gathered = mx.take(features, mx.maximum(rank, 0), axis=0)
        gathered = gathered.reshape(inputs_embeds.shape)
        return mx.where(positions[..., None], gathered, inputs_embeds)

    @property
    def layers(self):
        return self.language_model.model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        features = self.get_input_embeddings(input_ids, pixel_values, **kwargs)
        return self.language_model(
            input_ids, cache=cache, inputs_embeds=features.inputs_embeds)
