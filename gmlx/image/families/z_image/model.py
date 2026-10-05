# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team
# SPDX-FileCopyrightText: Copyright (c) 2026 Filip Strand
# Ported from diffusers' src/diffusers/models/transformers/transformer_z_image.py
# (the Z-Image forward, its padding, position ids and final layer) at
# e0abab83 and modified for gmlx (Apache-2.0; see licenses/diffusers-LICENSE).
# The module tree follows mflux's Z-Image transformer at e066d98 (MIT; see
# licenses/mflux-LICENSE).
"""Z-Image DiT module tree and forward.

Attribute paths equal the GGUF tensor names, so numbered elements are plain
lists. The fused qkv matrix stays one product whose output splits into q, k
and v. Elementwise and attention steps go through the ``gmlx.image.ops``
seams.

One step runs patchify, ``x_embedder``, the noise refiner, the caption path
through the context refiner, the main layers over ``[x, cap]``, the final
layer, unpatchify and a sign flip, so the result is the flow velocity the
sampler integrates. In float16 each block clamps its attention and
feed-forward outputs to the float16 range and replaces NaN, which costs
nothing in bfloat16. Image and caption rows pad to a multiple of 32 with the
pad tokens.

The forward keeps no caches, but it takes the two parts of a step that do
not depend on the latents from its caller. ``modulation`` derives from
sigma alone the timestep embedding, the four modulation vectors of each of
the 32 modulated blocks and the final layer's scale. ``caption`` derives
from the caption alone its rows after the context refiner and its rotary
tables. A step given either computes the same values as one without.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from gmlx.image import ops
from gmlx.image.ops import RMSNorm


@dataclass(frozen=True)
class ZImageConfig:
    dim: int
    n_heads: int
    head_dim: int
    n_layers: int
    n_refiner_layers: int
    in_channels: int
    patch: int
    cap_feat_dim: int
    ffn_dim: int
    adaln_dim: int
    t_freq_dim: int
    t_mid_dim: int
    norm_eps: float
    final_norm_eps: float
    axes_dims: tuple
    rope_theta: float
    t_scale: float
    seq_multiple: int

    @classmethod
    def from_dict(cls, c: dict) -> "ZImageConfig":
        if c["n_kv_heads"] != c["n_heads"]:
            raise ValueError("Z-Image attention has as many kv heads as heads")
        if c["all_f_patch_size"] != [1]:
            raise ValueError("Z-Image images have a frame patch of 1")
        return cls(
            dim=c["dim"], n_heads=c["n_heads"], head_dim=c["head_dim"],
            n_layers=c["n_layers"], n_refiner_layers=c["n_refiner_layers"],
            in_channels=c["in_channels"], patch=c["all_patch_size"][0],
            cap_feat_dim=c["cap_feat_dim"], ffn_dim=c["ffn_dim"],
            adaln_dim=c["adaln_dim"], t_freq_dim=c["t_freq_dim"],
            t_mid_dim=c["t_mid_dim"], norm_eps=c["norm_eps"],
            final_norm_eps=c["final_norm_eps"],
            axes_dims=tuple(c["axes_dims"]), rope_theta=c["rope_theta"],
            t_scale=c["t_scale"], seq_multiple=c["seq_multiple"])


_FP16_MAX = 65504.0


def fp16_safe(x: mx.array) -> mx.array:
    """In float16 only, clamp ``x`` to the float16 range and replace NaN
    with 0, so an overflow in one block does not reach the latents.
    Other dtypes pass through untouched."""
    if x.dtype != mx.float16:
        return x
    return mx.nan_to_num(mx.clip(x, -_FP16_MAX, _FP16_MAX), nan=0.0,
                         posinf=_FP16_MAX, neginf=-_FP16_MAX)


class Attention(nn.Module):
    def __init__(self, cfg: ZImageConfig):
        super().__init__()
        inner = cfg.n_heads * cfg.head_dim
        self.qkv = nn.Linear(cfg.dim, 3 * inner, bias=False)
        self.out = nn.Linear(inner, cfg.dim, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.eps = cfg.norm_eps

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        b, n, _ = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.n_heads, self.head_dim)
        q, k = ops.qk_norm_rope(qkv[:, :, 0], qkv[:, :, 1],
                                self.q_norm.weight, self.k_norm.weight,
                                cos, sin, self.eps)
        o = ops.sdpa(q, k, qkv[:, :, 2].transpose(0, 2, 1, 3))
        return self.out(o.transpose(0, 2, 1, 3).reshape(b, n, -1))


class FeedForward(nn.Module):
    def __init__(self, cfg: ZImageConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.w2 = nn.Linear(cfg.ffn_dim, cfg.dim, bias=False)
        self.w3 = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(ops.swiglu(self.w1(x), self.w3(x)))


class Block(nn.Module):
    """Joint block. Without modulation it is a context refiner block."""

    def __init__(self, cfg: ZImageConfig, modulated: bool):
        super().__init__()
        self.attention = Attention(cfg)
        self.feed_forward = FeedForward(cfg)
        self.attention_norm1 = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attention_norm2 = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn_norm1 = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn_norm2 = RMSNorm(cfg.dim, cfg.norm_eps)
        if modulated:
            self.adaLN_modulation = [nn.Linear(cfg.adaln_dim, 4 * cfg.dim)]

    def modulation(self, adaln: mx.array) -> tuple:
        """scale_msa, gate_msa, scale_mlp and gate_mlp, each (B, 1, dim)."""
        mod = self.adaLN_modulation[0](adaln)[:, None, :]
        return tuple(mx.split(mod, 4, axis=-1))

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array,
                 adaln: mx.array | None = None,
                 mod: tuple | None = None) -> mx.array:
        if adaln is None and mod is None:
            a = fp16_safe(self.attention(self.attention_norm1(x), cos, sin))
            x = x + self.attention_norm2(a)
            f = fp16_safe(self.feed_forward(self.ffn_norm1(x)))
            return x + self.ffn_norm2(f)
        if mod is None:
            mod = self.modulation(adaln)
        scale_msa, gate_msa, scale_mlp, gate_mlp = mod
        a = fp16_safe(self.attention(
            ops.modulate(self.attention_norm1(x), scale_msa), cos, sin))
        x = ops.gated_residual(x, mx.tanh(gate_msa), self.attention_norm2(a))
        f = fp16_safe(self.feed_forward(
            ops.modulate(self.ffn_norm1(x), scale_mlp)))
        return ops.gated_residual(x, mx.tanh(gate_mlp), self.ffn_norm2(f))


class TimestepEmbedder(nn.Module):
    def __init__(self, cfg: ZImageConfig):
        super().__init__()
        self.mlp = [nn.Linear(cfg.t_freq_dim, cfg.t_mid_dim), nn.SiLU(),
                    nn.Linear(cfg.t_mid_dim, cfg.adaln_dim)]
        self.freq_dim = cfg.t_freq_dim

    def __call__(self, t) -> mx.array:
        freq = ops.timestep_embedding(t, self.freq_dim)
        w = self.mlp[0].weight
        if mx.issubdtype(w.dtype, mx.floating):
            freq = freq.astype(w.dtype)
        return self.mlp[2](self.mlp[1](self.mlp[0](freq)))


class FinalLayer(nn.Module):
    def __init__(self, cfg: ZImageConfig):
        super().__init__()
        out = cfg.patch * cfg.patch * cfg.in_channels
        self.adaLN_modulation = [nn.SiLU(),
                                 nn.Linear(cfg.adaln_dim, cfg.dim)]
        self.linear = nn.Linear(cfg.dim, out)
        self.eps = cfg.final_norm_eps

    def scale(self, adaln: mx.array) -> mx.array:
        return self.adaLN_modulation[1](self.adaLN_modulation[0](adaln))

    def __call__(self, x: mx.array, adaln: mx.array | None = None,
                 scale: mx.array | None = None) -> mx.array:
        if scale is None:
            scale = self.scale(adaln)
        x = ops.modulate(ops.layer_norm(x, self.eps), scale[:, None, :])
        return self.linear(x)


@dataclass(frozen=True)
class Modulation:
    """What one step derives from sigma: the timestep embedding (B, adaln),
    one tuple of four (B, 1, dim) vectors per modulated block, and the
    final layer's scale (B, dim)."""
    adaln: mx.array
    noise_refiner: tuple
    layers: tuple
    final: mx.array

    def arrays(self) -> list:
        out = [self.adaln, self.final]
        for mod in self.noise_refiner + self.layers:
            out.extend(mod)
        return out


@dataclass(frozen=True)
class Caption:
    """What one step derives from the caption: its padded rows after the
    context refiner (B, L, dim), and its rotary tables."""
    rows: mx.array
    cos: mx.array
    sin: mx.array

    def arrays(self) -> list:
        return [self.rows, self.cos, self.sin]


def _ceil(n: int, m: int) -> int:
    return -(-n // m) * m


class ZImageDiT(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        cfg = ZImageConfig.from_dict(config)
        self.cfg = cfg
        self.act_dtype = mx.bfloat16
        patch_dim = cfg.patch * cfg.patch * cfg.in_channels
        self.x_embedder = nn.Linear(patch_dim, cfg.dim)
        self.x_pad_token = mx.zeros((1, cfg.dim))
        self.cap_pad_token = mx.zeros((1, cfg.dim))
        self.t_embedder = TimestepEmbedder(cfg)
        self.cap_embedder = [RMSNorm(cfg.cap_feat_dim, cfg.norm_eps),
                             nn.Linear(cfg.cap_feat_dim, cfg.dim)]
        self.noise_refiner = [Block(cfg, True)
                              for _ in range(cfg.n_refiner_layers)]
        self.context_refiner = [Block(cfg, False)
                                for _ in range(cfg.n_refiner_layers)]
        self.layers = [Block(cfg, True) for _ in range(cfg.n_layers)]
        self.final_layer = FinalLayer(cfg)

    def timesteps(self, sigma, batch: int) -> np.ndarray:
        """Model timesteps for sigmas, in fp32 as the reference pipeline
        forms them: (1000 - 1000 sigma) / 1000, times t_scale."""
        s = np.broadcast_to(np.asarray(sigma, np.float32), (batch,))
        ts = s * np.float32(1000)
        t = (np.float32(1000) - ts) / np.float32(1000)
        return t * np.float32(self.cfg.t_scale)

    def adaln(self, sigma, batch: int) -> mx.array:
        return self.t_embedder(self.timesteps(sigma, batch)).astype(
            self.act_dtype)

    def patchify(self, latents: mx.array) -> mx.array:
        b, c, h, w = latents.shape
        p = self.cfg.patch
        x = latents.reshape(b, c, h // p, p, w // p, p)
        return x.transpose(0, 2, 4, 3, 5, 1).reshape(
            b, (h // p) * (w // p), p * p * c)

    def unpatchify(self, x: mx.array, ht: int, wt: int) -> mx.array:
        b = x.shape[0]
        p, c = self.cfg.patch, self.cfg.in_channels
        x = x.reshape(b, ht, wt, p, p, c).transpose(0, 5, 1, 3, 2, 4)
        return x.reshape(b, c, ht * p, wt * p)

    def _pad(self, x: mx.array, token: mx.array) -> mx.array:
        b, n, d = x.shape
        pad = _ceil(n, self.cfg.seq_multiple) - n
        if not pad:
            return x
        rows = mx.broadcast_to(token.astype(x.dtype), (b, pad, d))
        return mx.concatenate([x, rows], axis=1)

    def embed_image(self, latents: mx.array) -> mx.array:
        x = self.x_embedder(self.patchify(latents).astype(self.act_dtype))
        return self._pad(x, self.x_pad_token)

    def embed_caption(self, cap_feats: mx.array) -> mx.array:
        cap = self.cap_embedder[0](cap_feats.astype(self.act_dtype))
        return self._pad(self.cap_embedder[1](cap), self.cap_pad_token)

    def positions(self, ht: int, wt: int, n_cap: int
                  ) -> tuple[np.ndarray, np.ndarray]:
        """Position ids (L, 3) of the padded image and caption rows. Caption
        row i sits at (1 + i, 0, 0) and image row (h, w) at
        (n_cap + 1, h, w). Image pad rows sit at the origin."""
        n_img = ht * wt
        img = np.zeros((_ceil(n_img, self.cfg.seq_multiple), 3), np.int64)
        hh, ww = np.meshgrid(np.arange(ht), np.arange(wt), indexing="ij")
        img[:n_img, 0] = n_cap + 1
        img[:n_img, 1] = hh.reshape(-1)
        img[:n_img, 2] = ww.reshape(-1)
        cap = np.zeros((n_cap, 3), np.int64)
        cap[:, 0] = np.arange(1, n_cap + 1)
        return img, cap

    def rope(self, ht: int, wt: int, n_cap: int):
        img, cap = self.positions(ht, wt, n_cap)
        c = self.cfg
        return (ops.rope_tables(img, c.axes_dims, c.rope_theta),
                ops.rope_tables(cap, c.axes_dims, c.rope_theta))

    def modulation(self, sigma, batch: int) -> Modulation:
        adaln = self.adaln(sigma, batch)
        return Modulation(
            adaln=adaln,
            noise_refiner=tuple(b.modulation(adaln)
                                for b in self.noise_refiner),
            layers=tuple(b.modulation(adaln) for b in self.layers),
            final=self.final_layer.scale(adaln))

    def caption(self, cap_feats: mx.array) -> Caption:
        cap = self.embed_caption(cap_feats)
        # Caption positions do not depend on the image size.
        _, (cc, cs) = self.rope(1, 1, cap.shape[1])
        for blk in self.context_refiner:
            cap = blk(cap, cc, cs)
        return Caption(cap, cc, cs)

    def __call__(self, latents: mx.array, sigma, cap_feats: mx.array | None,
                 taps: dict | None = None, *, mod: Modulation | None = None,
                 caption: Caption | None = None) -> mx.array:
        """Velocity (B, C, H, W) fp32 for fp32 latents at ``sigma`` with
        caption features (B, L, cap_feat_dim). ``mod`` and ``caption``, from
        ``modulation`` and ``caption``, stand in for the work they hold.
        ``taps`` collects the output of every block by module path."""
        b, _, h, w = latents.shape
        ht, wt = h // self.cfg.patch, w // self.cfg.patch
        # Without the two parts, the step runs in the reference order.
        adaln = self.adaln(sigma, b) if mod is None else None
        x = self.embed_image(latents)
        cap = self.embed_caption(cap_feats) if caption is None else \
            caption.rows
        (xc, xs), (cc, cs) = self.rope(ht, wt, cap.shape[1])
        if caption is not None:
            cc, cs = caption.cos, caption.sin
        for i, blk in enumerate(self.noise_refiner):
            x = blk(x, xc, xs, adaln,
                    None if mod is None else mod.noise_refiner[i])
            if taps is not None:
                taps[f"noise_refiner.{i}"] = x
        if caption is None:
            for i, blk in enumerate(self.context_refiner):
                cap = blk(cap, cc, cs)
                if taps is not None:
                    taps[f"context_refiner.{i}"] = cap
        u = mx.concatenate([x, cap], axis=1)
        cos = mx.concatenate([xc, cc], axis=0)
        sin = mx.concatenate([xs, cs], axis=0)
        for i, blk in enumerate(self.layers):
            u = blk(u, cos, sin, adaln,
                    None if mod is None else mod.layers[i])
            if taps is not None:
                taps[f"layers.{i}"] = u
        out = self.final_layer(u, adaln,
                               None if mod is None else mod.final)
        if taps is not None:
            taps["final_layer"] = out
        v = self.unpatchify(out[:, : ht * wt], ht, wt)
        return -v.astype(mx.float32)
