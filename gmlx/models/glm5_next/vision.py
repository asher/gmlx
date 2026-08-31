# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""GLM-5.3-Flash vision tower + projector (mmproj ``glm5next``).

The ViT is the GLM-OCR encoder (llama.cpp PR 27754, clip_graph_glm4v):
qwen2vl-style dynamic-resolution patches with 2-D interleaved rope (theta
10000, no learned position embeddings, no post-conv norm), RMS pre-norms,
fused qkv with bias, per-head q/k RMSNorm, and a clamped-SwiGLU FFN with
biases. The projector downsamples 2x2 token blocks through a stride-2 conv,
then fc -> LayerNorm(1e-5) -> gelu(erf) -> clamped-SwiGLU MLP.

Patches arrive pre-shuffled by the processor (the h//2,2,w//2,2 block order),
so consecutive groups of four tokens are spatially adjacent 2x2 blocks - the
rope position ids and the merger's conv both assume that order. v1 runs one
image per call (B=1 serve-aligned, like the other vendored towers).
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.models.base import BaseModelConfig

from .model import _limited_swiglu


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "glm5_next"
    depth: int = 24
    hidden_size: int = 1024
    num_heads: int = 16
    intermediate_size: int = 4096
    patch_size: int = 14
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    in_channels: int = 3
    rms_norm_eps: float = 1e-5
    swiglu_limit: float = 10.0
    out_hidden_size: int = 4096
    proj_intermediate_size: int = 10240
    rope_theta: float = 10000.0


def _rotate_half(x: mx.array) -> mx.array:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return mx.concatenate([-x2, x1], axis=-1)


def _apply_vision_rope(tensor: mx.array, freqs: mx.array) -> mx.array:
    """``tensor`` [N, heads, D], ``freqs`` [N, D/2] fp32."""
    orig = tensor.dtype
    cos = mx.tile(mx.cos(freqs)[:, None, :], (1, 1, 2))
    sin = mx.tile(mx.sin(freqs)[:, None, :], (1, 1, 2))
    t = tensor.astype(mx.float32)
    return ((t * cos) + (_rotate_half(t) * sin)).astype(orig)


class Glm5NextVisionAttention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = dim // config.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, rotary_pos_emb: mx.array) -> mx.array:
        N = x.shape[0]
        qkv = self.qkv(x).reshape(N, 3, self.num_heads, self.head_dim)
        q, k, v = (qkv[:, i] for i in range(3))
        q = _apply_vision_rope(self.q_norm(q), rotary_pos_emb)
        k = _apply_vision_rope(self.k_norm(k), rotary_pos_emb)
        # Single image per call: full bidirectional attention over all patches.
        q, k, v = (t.transpose(1, 0, 2)[None] for t in (q, k, v))
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=None)
        return self.proj(out[0].transpose(1, 0, 2).reshape(N, -1))


class Glm5NextVisionMLP(nn.Module):
    def __init__(self, dim: int, hidden: int, limit: float, bias: bool = True):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=bias)
        self.up_proj = nn.Linear(dim, hidden, bias=bias)
        self.down_proj = nn.Linear(hidden, dim, bias=bias)
        self._limit = limit

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(
            _limited_swiglu(self.gate_proj(x), self.up_proj(x), self._limit))


class Glm5NextVisionBlock(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = Glm5NextVisionAttention(config)
        self.mlp = Glm5NextVisionMLP(
            config.hidden_size, config.intermediate_size, config.swiglu_limit)

    def __call__(self, x: mx.array, rotary_pos_emb: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x), rotary_pos_emb)
        return x + self.mlp(self.norm2(x))


class Glm5NextPatchEmbed(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel = [config.temporal_patch_size, config.patch_size,
                  config.patch_size]
        self.proj = nn.Conv3d(config.in_channels, config.hidden_size,
                              kernel_size=kernel, stride=kernel, bias=True)

    def __call__(self, patches: mx.array) -> mx.array:
        x = patches.reshape(
            -1, self.in_channels, self.temporal_patch_size,
            self.patch_size, self.patch_size,
        ).moveaxis(1, 4)
        return self.proj(x).reshape(-1, self.embed_dim)


class Glm5NextPatchMerger(nn.Module):
    """Stride-2 conv over each consecutive-4 token block, then the GLM4V
    projector: fc -> LayerNorm(1e-5, hardcoded upstream) -> gelu(erf) ->
    clamped-SwiGLU MLP (no biases)."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        merge = config.spatial_merge_size
        self.merge = merge
        self.patch_merger = nn.Conv2d(
            config.hidden_size, config.out_hidden_size,
            kernel_size=merge, stride=merge, bias=True)
        self.fc = nn.Linear(
            config.out_hidden_size, config.out_hidden_size, bias=False)
        self.post_norm = nn.LayerNorm(config.out_hidden_size, eps=1e-5)
        self.mlp = Glm5NextVisionMLP(
            config.out_hidden_size, config.proj_intermediate_size,
            config.swiglu_limit, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        n_out = x.shape[0] // (self.merge * self.merge)
        x = x.reshape(n_out, self.merge, self.merge, -1)
        x = self.patch_merger(x).reshape(n_out, -1)
        x = nn.gelu(self.post_norm(self.fc(x)))
        return self.mlp(x)


class Glm5NextVisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = Glm5NextPatchEmbed(config)
        self.blocks = [Glm5NextVisionBlock(config)
                       for _ in range(config.depth)]
        self.post_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.merger = Glm5NextPatchMerger(config)

    def _rot_pos_emb(self, grid_thw) -> mx.array:
        """[N, head_dim/2] fp32 rope angles in the pre-shuffled patch order:
        h and w each contribute head_dim/4 frequencies."""
        merge = self.spatial_merge_size
        dim = (self.config.hidden_size // self.config.num_heads) // 4
        inv_freq = 1.0 / (self.config.rope_theta ** (
            mx.arange(0, dim, dtype=mx.float32) / dim))
        pos_ids = []
        for t, h, w in grid_thw:
            t, h, w = int(t), int(h), int(w)
            hpos = mx.repeat(mx.arange(h)[:, None], w, axis=1)
            wpos = mx.repeat(mx.arange(w)[None, :], h, axis=0)
            ids = []
            for p in (hpos, wpos):
                p = p.reshape(h // merge, merge, w // merge, merge)
                ids.append(p.transpose(0, 2, 1, 3).flatten())
            stacked = mx.stack(ids, axis=-1)  # [h*w, 2]
            pos_ids.append(mx.tile(stacked, (t, 1)))
        pos = mx.concatenate(pos_ids, axis=0)
        freqs = pos[..., None].astype(mx.float32) * inv_freq  # [N, 2, dim]
        return freqs.reshape(pos.shape[0], -1)

    def __call__(self, patches: mx.array, grid_thw) -> mx.array:
        x = self.patch_embed(patches)
        rope = self._rot_pos_emb(grid_thw)
        for block in self.blocks:
            x = block(x, rope)
        return self.merger(self.post_layernorm(x))
