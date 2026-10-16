# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright 2025 Black Forest Labs, Inc.
# Ported from BFL's FLUX.2 reference at commit 50fe516: the Flux2 forward,
# DoubleStreamBlock, SingleStreamBlock, SelfAttention, SiLUActivation,
# Modulation, LastLayer, MLPEmbedder, QKNorm, timestep_embedding and rope
# of src/flux2/model.py, and the token ids of prc_img, prc_txt and
# encode_image_refs in src/flux2/sampling.py, and modified for gmlx
# (Apache-2.0; see licenses/flux2-LICENSE).
"""FLUX.2 klein DiT module tree and forward.

Attribute paths equal the GGUF tensor names, so numbered elements are
plain lists.

A sample holds the image latents, the latents of any reference images and
the text. The image stream carries the image tokens, then the reference
tokens, all through ``img_in``. The text stream carries the text through
``txt_in``. Each double block runs both streams with weights of their own
and one attention over the text, then the image stream. The two streams
then join, text first, and run the single blocks. The velocity is the
final layer on the image tokens.

One time vector, from the step's time, drives every modulation: one for
the image stream of every double block, one for its text stream, one for
every single block and one for the final layer. The reference tokens take
the image's. Every norm is a LayerNorm without affine parameters, and the
modulated input is ``(1 + scale) * norm(x) + shift``. A double block adds
the attention and then the feed-forward of each stream, each through its
own norm and gate. A single block feeds one modulated norm to one fused
product, of q, k, v and the feed-forward's gate and up halves, and adds
the gated product of the attention rows and the feed-forward rows.

The time embedding follows the reference pipeline, which holds the time
in bf16: the embedder multiplies by 1000 in bf16 and returns the embedding
in bf16, in every activation dtype. The time vector and the modulations
then run in fp32 and round once.

RoPE has four axes of 32 dims at theta 2000. An image token sits at (0,
row, column, 0), a token of reference image i at (10 + 10 i, row, column,
0), and a text token at (0, 0, 0, position). The tables follow the
reference's fp32 arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from gmlx.image import ops, pacing
from gmlx.image.dtypes import ImageDtypeError
from gmlx.image.ops.products import own, precise

# The time axis of reference image i is REF_T * (1 + i).
REF_T = 10


@dataclass(frozen=True)
class Flux2Config:
    dim: int
    heads: int
    head_dim: int
    depth: int
    depth_single: int
    mlp_dim: int
    in_channels: int
    out_channels: int
    text_dim: int
    t_dim: int
    axes_dims: tuple
    theta: float
    eps: float

    @classmethod
    def from_dict(cls, c: dict) -> "Flux2Config":
        head_dim = int(c["attention_head_dim"])
        heads = int(c["num_attention_heads"])
        axes = tuple(int(a) for a in c["axes_dims_rope"])
        if sum(axes) != head_dim or any(a % 2 for a in axes):
            raise ValueError(f"RoPE axes {axes} must be even dims that sum "
                             f"to the head dim {head_dim}")
        if c.get("guidance_embeds"):
            raise ValueError("a FLUX.2 DiT that embeds a guidance is not "
                             "FLUX.2 klein")
        if int(c.get("patch_size", 1)) != 1:
            raise ValueError("FLUX.2 klein takes one latent pixel per token")
        dim = heads * head_dim
        in_ch = int(c["in_channels"])
        out_ch = c.get("out_channels")
        return cls(
            dim=dim, heads=heads, head_dim=head_dim,
            depth=int(c["num_layers"]),
            depth_single=int(c["num_single_layers"]),
            mlp_dim=int(dim * float(c["mlp_ratio"])),
            in_channels=in_ch,
            out_channels=in_ch if out_ch is None else int(out_ch),
            text_dim=int(c["joint_attention_dim"]),
            t_dim=int(c["timestep_guidance_channels"]), axes_dims=axes,
            theta=float(c["rope_theta"]), eps=float(c["eps"]))


def _bf16(a: np.ndarray) -> np.ndarray:
    """float32 values rounded to bfloat16, to nearest even."""
    return np.array(mx.array(np.asarray(a, np.float32)).astype(
        mx.bfloat16).astype(mx.float32))


def time_embedding(t, dim: int) -> mx.array:
    """The embedding (B, dim) of times ``t`` as the reference forms it
    from its bf16 time: 1000 times the time rounded to bf16, the cosines
    and sines in fp32, and the embedding rounded to bf16. Returned as the
    bf16 values in float32."""
    t1000 = _bf16(np.float32(1000) * _bf16(np.reshape(t, (-1,))))
    emb = ops.timestep_embedding(t1000, dim)
    return emb.astype(mx.bfloat16).astype(mx.float32)


def rope_tables(ids: np.ndarray, axes_dims, theta: float) -> tuple:
    """cos and sin tables (L, sum(axes_dims) // 2) for integer position
    ids (L, n_axes). The reference forms each frequency as 1 / theta **
    (2 i / d) and each angle as position times frequency in fp32; the
    cosines and sines here are those of the fp32 angle, rounded once."""
    cos, sin = [], []
    theta32 = np.float32(theta)
    for axis, d in enumerate(axes_dims):
        expo = (np.arange(0, d, 2) / d).astype(np.float32)
        freqs = np.float32(1) / np.power(theta32, expo)
        ang = np.asarray(ids[:, axis], np.float32)[:, None] * freqs[None]
        ang = ang.astype(np.float64)
        cos.append(np.cos(ang).astype(np.float32))
        sin.append(np.sin(ang).astype(np.float32))
    return (mx.array(np.concatenate(cos, axis=1)),
            mx.array(np.concatenate(sin, axis=1)))


def image_ids(h: int, w: int, index: int | None = None) -> np.ndarray:
    """Position ids (h * w, 4) of a latent grid in row-major order: (0,
    row, column, 0) for the image, and the first axis at REF_T * (1 +
    ``index``) for reference image ``index``."""
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    ids = np.zeros((h * w, 4), np.float32)
    ids[:, 1] = rows.reshape(-1)
    ids[:, 2] = cols.reshape(-1)
    if index is not None:
        ids[:, 0] = REF_T * (1 + index)
    return ids


def text_ids(n: int) -> np.ndarray:
    ids = np.zeros((n, 4), np.float32)
    ids[:, 3] = np.arange(n)
    return ids


def tokens(latents: mx.array) -> mx.array:
    """Latents (B, C, h, w) as tokens (B, h * w, C), row-major."""
    b, c, h, w = latents.shape
    return latents.reshape(b, c, h * w).transpose(0, 2, 1)


class RMSNorm(nn.Module):
    """The reference's q and k norm: its scale is the weight itself, and
    it runs in fp32 inside ``ops.qk_norm_rope``."""

    def __init__(self, dims: int):
        super().__init__()
        self.scale = mx.ones((dims,))


class QKNorm(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        self.query_norm = RMSNorm(head_dim)
        self.key_norm = RMSNorm(head_dim)


def adaln(x: mx.array, shift: mx.array, scale: mx.array, eps: float,
          dtype) -> mx.array:
    """``(1 + scale) * LayerNorm(x) + shift`` in fp32 with ``1 + scale``
    in the dtype of ``scale``, rounded once to ``dtype``."""
    y = ops.layer_norm(x.astype(mx.float32), eps)
    return (ops.modulate(y, scale) + shift).astype(dtype)


def heads(attn_qkv: mx.array, n_heads: int, head_dim: int) -> tuple:
    """q, k and v (B, L, H, D) of a fused qkv product (B, L, 3 H D)."""
    b, n, _ = attn_qkv.shape
    qkv = attn_qkv.reshape(b, n, 3, n_heads, head_dim)
    return qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]


def qk(norm: QKNorm, q: mx.array, k: mx.array, cos: mx.array,
       sin: mx.array, eps: float) -> tuple:
    return ops.qk_norm_rope(q, k, norm.query_norm.scale,
                            norm.key_norm.scale, cos, sin, eps)


def gated_out(lin, x: mx.array, gate: mx.array, rows: mx.array) -> mx.array:
    """``x + gate * lin(rows)``, on mlx-kquant's residual epilogue when its
    route takes the leaf."""
    leaf = ops.resid_leaf(lin, rows, x, gate)
    if leaf is not None:
        return ops.resid_fused(rows, leaf, x, gate)
    return ops.gated_residual(x, gate, lin(rows))


class SelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim, bias=False)


class SiLUGate(nn.Module):
    """``silu(a) * b`` of the halves a, b of its input: the reference's
    SiLUActivation, which holds no weight."""

    def __call__(self, x: mx.array) -> mx.array:
        a, b = mx.split(x, 2, axis=-1)
        return ops.swiglu(a, b)


def feed_forward(dim: int, mlp_dim: int) -> list:
    return [nn.Linear(dim, 2 * mlp_dim, bias=False), SiLUGate(),
            nn.Linear(mlp_dim, dim, bias=False)]


class DoubleBlock(nn.Module):
    """One block of both streams: per stream a modulated norm, its q, k
    and v, one attention over the text then the image tokens, the gated
    projection, then a modulated norm and the gated feed-forward."""

    def __init__(self, cfg: Flux2Config):
        super().__init__()
        d = cfg.dim
        self.img_attn = SelfAttention(d, cfg.head_dim)
        self.img_mlp = feed_forward(d, cfg.mlp_dim)
        self.txt_attn = SelfAttention(d, cfg.head_dim)
        self.txt_mlp = feed_forward(d, cfg.mlp_dim)
        self.heads = cfg.heads
        self.head_dim = cfg.head_dim
        self.eps = cfg.eps

    def adaln(self, x: mx.array, shift: mx.array,
              scale: mx.array) -> mx.array:
        return adaln(x, shift, scale, self.eps, x.dtype)

    def qkv(self, attn: SelfAttention, h: mx.array, cos: mx.array,
            sin: mx.array) -> tuple:
        """q and k normed and rotated, and v, each (B, H, L, D)."""
        q, k, v = heads(attn.qkv(h), self.heads, self.head_dim)
        q, k = qk(attn.norm, q, k, cos, sin, self.eps)
        return q, k, v.transpose(0, 2, 1, 3)

    def hidden(self, mlp: list, h: mx.array) -> mx.array:
        """``silu(a) * b`` of the halves a, b of ``mlp[0](h)``: the rows
        that ``mlp[2]`` takes."""
        pair = ops.glu_halves(mlp[0], h)
        if pair is not None:
            return ops.swiglu_fused(h, *pair)
        return mlp[1](mlp[0](h))

    def out(self, lin, x: mx.array, gate: mx.array,
            rows: mx.array) -> mx.array:
        return gated_out(lin, x, gate, rows)

    def __call__(self, img: mx.array, txt: mx.array, mod_img: tuple,
                 mod_txt: tuple, rope_img: tuple, rope_txt: tuple) -> tuple:
        (i_shift, i_scale, i_gate), (i_shift2, i_scale2, i_gate2) = mod_img
        (t_shift, t_scale, t_gate), (t_shift2, t_scale2, t_gate2) = mod_txt
        qi, ki, vi = self.qkv(self.img_attn,
                              self.adaln(img, i_shift, i_scale), *rope_img)
        qt, kt, vt = self.qkv(self.txt_attn,
                              self.adaln(txt, t_shift, t_scale), *rope_txt)
        o = ops.sdpa(mx.concatenate([qt, qi], axis=2),
                     mx.concatenate([kt, ki], axis=2),
                     mx.concatenate([vt, vi], axis=2))
        b, n_txt = txt.shape[0], txt.shape[1]
        o = o.transpose(0, 2, 1, 3).reshape(b, o.shape[2], -1)
        img = self.out(self.img_attn.proj, img, i_gate, o[:, n_txt:])
        txt = self.out(self.txt_attn.proj, txt, t_gate, o[:, :n_txt])
        img = self.out(self.img_mlp[2], img, i_gate2, self.hidden(
            self.img_mlp, self.adaln(img, i_shift2, i_scale2)))
        txt = self.out(self.txt_mlp[2], txt, t_gate2, self.hidden(
            self.txt_mlp, self.adaln(txt, t_shift2, t_scale2)))
        return img, txt


class SingleBlock(nn.Module):
    """One block of the joint tokens: a modulated norm feeds one product of
    q, k, v and the feed-forward halves, and one gated product of the
    attention and feed-forward rows adds to the residual."""

    def __init__(self, cfg: Flux2Config):
        super().__init__()
        d = cfg.dim
        self.linear1 = nn.Linear(d, 3 * d + 2 * cfg.mlp_dim, bias=False)
        self.linear2 = nn.Linear(d + cfg.mlp_dim, d, bias=False)
        self.norm = QKNorm(cfg.head_dim)
        self.dim = d
        self.heads = cfg.heads
        self.head_dim = cfg.head_dim
        self.eps = cfg.eps

    def adaln(self, x: mx.array, shift: mx.array,
              scale: mx.array) -> mx.array:
        return adaln(x, shift, scale, self.eps, x.dtype)

    def out(self, x: mx.array, gate: mx.array, rows: mx.array,
            hid: mx.array) -> mx.array:
        """``x + gate * linear2([rows, hid])``."""
        return gated_out(self.linear2, x, gate,
                         mx.concatenate([rows, hid], axis=-1))

    def __call__(self, x: mx.array, mod: tuple, cos: mx.array,
                 sin: mx.array) -> mx.array:
        shift, scale, gate = mod
        y = self.linear1(self.adaln(x, shift, scale))
        d3 = 3 * self.dim
        q, k, v = heads(y[..., :d3], self.heads, self.head_dim)
        q, k = qk(self.norm, q, k, cos, sin, self.eps)
        o = ops.sdpa(q, k, v.transpose(0, 2, 1, 3))
        b, n = x.shape[0], x.shape[1]
        rows = o.transpose(0, 2, 1, 3).reshape(b, n, -1)
        a, g = mx.split(y[..., d3:], 2, axis=-1)
        return self.out(x, gate, rows, ops.swiglu(a, g))


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, dim, bias=False)
        self.out_layer = nn.Linear(dim, dim, bias=False)


class Modulation(nn.Module):
    """shift, scale and gate of every block of a kind, twice over for a
    double block, from the time vector."""

    def __init__(self, dim: int, n: int):
        super().__init__()
        self.lin = nn.Linear(dim, n * dim, bias=False)


class LastLayer(nn.Module):
    def __init__(self, dim: int, out: int, eps: float):
        super().__init__()
        self.linear = nn.Linear(dim, out, bias=False)
        self.adaLN_modulation = [nn.SiLU(), nn.Linear(dim, 2 * dim,
                                                      bias=False)]
        self.eps = eps

    def __call__(self, x: mx.array, shift: mx.array,
                 scale: mx.array) -> mx.array:
        return self.linear(adaln(x, shift, scale, self.eps, x.dtype))


@dataclass(frozen=True)
class StepModulation:
    """What one step derives from its time: the two (shift, scale, gate)
    triples of the image and of the text stream of the double blocks, the
    triple of the single blocks and the final layer's shift and scale,
    each (B, 1, dim) in the activation dtype."""
    img: tuple
    txt: tuple
    single: tuple
    final: tuple

    def arrays(self) -> list:
        return [*self.img[0], *self.img[1], *self.txt[0], *self.txt[1],
                *self.single, *self.final]


@dataclass(frozen=True)
class Caption:
    """What one step derives from the conditioning: the text tokens after
    ``txt_in``, (B, L, dim). The double blocks read the step's time
    vector, so they run in the step."""
    rows: mx.array

    def arrays(self) -> list:
        return [self.rows]


@dataclass(frozen=True)
class References:
    """The latents of the reference images as tokens (B, L, channels), in
    the order of their images, with their position ids (L, 4)."""
    tokens: mx.array
    ids: np.ndarray

    @classmethod
    def of(cls, latents: list) -> "References":
        """Each reference latent (B, C, h, w) as tokens, with the ids of
        its index in the list."""
        toks = [tokens(lat) for lat in latents]
        ids = [image_ids(lat.shape[2], lat.shape[3], i)
               for i, lat in enumerate(latents)]
        return cls(mx.concatenate(toks, axis=1), np.concatenate(ids))


class Flux2KleinDiT(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        cfg = Flux2Config.from_dict(config)
        self.cfg = cfg
        self.act_dtype = mx.bfloat16
        d = cfg.dim
        self.img_in = nn.Linear(cfg.in_channels, d, bias=False)
        self.txt_in = nn.Linear(cfg.text_dim, d, bias=False)
        self.time_in = MLPEmbedder(cfg.t_dim, d)
        self.double_stream_modulation_img = Modulation(d, 6)
        self.double_stream_modulation_txt = Modulation(d, 6)
        self.single_stream_modulation = Modulation(d, 3)
        self.double_blocks = [DoubleBlock(cfg) for _ in range(cfg.depth)]
        self.single_blocks = [SingleBlock(cfg)
                              for _ in range(cfg.depth_single)]
        self.final_layer = LastLayer(d, cfg.out_channels, cfg.eps)

    @property
    def act_dtype(self) -> mx.Dtype:
        return self._act_dtype

    @act_dtype.setter
    def act_dtype(self, dtype: mx.Dtype) -> None:
        if dtype == mx.float16:
            raise ImageDtypeError("FLUX.2 klein never runs in float16")
        self._act_dtype = dtype

    def _lin(self):
        return precise if self.act_dtype != mx.float32 else own

    def vector(self, sigma, batch: int) -> mx.array:
        """The time vector (B, dim) fp32 of ``sigma``."""
        lin = self._lin()
        e = time_embedding(np.broadcast_to(np.asarray(sigma, np.float32),
                                           (batch,)), self.cfg.t_dim)
        return lin(self.time_in.out_layer,
                   nn.silu(lin(self.time_in.in_layer, e)))

    def modulation(self, sigma, batch: int) -> StepModulation:
        lin = self._lin()
        act = self.act_dtype
        v = nn.silu(self.vector(sigma, batch))

        def chunks(mod, n):
            y = lin(mod, v).astype(act)[:, None, :]
            return tuple(mx.split(y, n, axis=-1))

        img = chunks(self.double_stream_modulation_img.lin, 6)
        txt = chunks(self.double_stream_modulation_txt.lin, 6)
        return StepModulation(
            img=(img[:3], img[3:]), txt=(txt[:3], txt[3:]),
            single=chunks(self.single_stream_modulation.lin, 3),
            final=chunks(self.final_layer.adaLN_modulation[1], 2))

    def references(self, latents) -> References:
        """The reference latents (B, C, h, w), in the order of their
        images, as the tokens and ids the DiT takes."""
        return References.of(list(latents))

    def caption(self, feats: mx.array, taps: dict | None = None) -> Caption:
        """The text tokens of features (B, L, text dim) after txt_in."""
        x = self.txt_in(feats.astype(self.act_dtype))
        if taps is not None:
            taps["txt_in"] = x
        return Caption(x)

    def __call__(self, latents: mx.array, sigma, feats: mx.array | None,
                 taps: dict | None = None, *,
                 refs: References | None = None,
                 mod: StepModulation | None = None,
                 caption: Caption | None = None) -> mx.array:
        """Velocity (B, C, h, w) fp32 for latents (B, C, h, w) at
        ``sigma``, with text features (B, L, text dim) and the tokens of
        any reference images. ``mod`` and ``caption``, from
        ``modulation`` and ``caption``, stand in for the work they hold.
        ``taps`` collects the output of every block by module path, a
        double block's as ``<path>.img`` and ``<path>.txt``."""
        b, _, h, w = latents.shape
        cfg = self.cfg
        act = self.act_dtype
        if mod is None:
            mod = self.modulation(sigma, b)
        if caption is None:
            if feats is None:
                raise ValueError("pass text features or their caption")
            caption = self.caption(feats, taps)
        txt = caption.rows
        if txt.shape[0] != b:
            txt = mx.broadcast_to(txt, (b, *txt.shape[1:]))
        x = tokens(latents)
        ids = image_ids(h, w)
        n_img = x.shape[1]
        if refs is not None:
            cond = refs.tokens
            if cond.shape[0] != b:
                cond = mx.broadcast_to(cond, (b, *cond.shape[1:]))
            x = mx.concatenate([x, cond.astype(x.dtype)], axis=1)
            ids = np.concatenate([ids, np.asarray(refs.ids, np.float32)])
        img = self.img_in(x.astype(act))
        if taps is not None:
            taps["img_in"] = img
        t_ids = text_ids(txt.shape[1])
        rope_img = rope_tables(ids, cfg.axes_dims, cfg.theta)
        rope_txt = rope_tables(t_ids, cfg.axes_dims, cfg.theta)
        for i, blk in enumerate(self.double_blocks):
            img, txt = blk(img, txt, mod.img, mod.txt, rope_img, rope_txt)
            pacing.point(img, txt)
            if taps is not None:
                taps[f"double_blocks.{i}.img"] = img
                taps[f"double_blocks.{i}.txt"] = txt
        n_txt = txt.shape[1]
        joint = mx.concatenate([txt, img], axis=1)
        cos, sin = rope_tables(np.concatenate([t_ids, ids]), cfg.axes_dims,
                               cfg.theta)
        for i, blk in enumerate(self.single_blocks):
            joint = blk(joint, mod.single, cos, sin)
            pacing.point(joint)
            if taps is not None:
                taps[f"single_blocks.{i}"] = joint
        out = self.final_layer(joint[:, n_txt:n_txt + n_img], *mod.final)
        if taps is not None:
            taps["final_layer"] = out
        c = out.shape[-1]
        return out.transpose(0, 2, 1).reshape(b, c, h, w).astype(mx.float32)
