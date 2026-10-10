# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright 2026 Krea AI and The HuggingFace Team
# Ported from diffusers' src/diffusers/models/transformers/transformer_krea2.py
# (the single-stream block, the shared modulation, the text fusion, the
# text and timestep projections and the final layer) at 0.41.0.dev0 and
# modified for gmlx (Apache-2.0; see licenses/diffusers-LICENSE).
"""Krea 2 DiT module tree and forward.

Attribute paths equal the GGUF tensor names, so numbered elements are plain
lists. Every norm stores its scale minus one, as ``scale``, and runs in
fp32. Elementwise and attention steps go through the ``gmlx.image.ops``
seams.

The text conditioning is a stack of twelve encoder layers per token. The
text fusion runs two blocks across the layers of each token, collapses
the layers with ``txtfusion.projector`` and runs two blocks across the
tokens. ``txtmlp`` projects the result to the model width. The text rows
then join the image rows as one sequence, text first, through every
block, and the velocity is the image rows of the final layer, with no
sign flip. Text rows sit at rotary position zero and image rows at
(0, h, w) on the patch grid.

One timestep embedding drives every block: ``tproj`` maps it to six
vectors that each block adds to its own ``mod.lin`` and reads as the
scale, shift and gate of its attention and of its MLP. The final layer
reads the embedding itself, added to ``last.modulation.lin``.

The forward keeps no caches, but it takes the two parts of a step that do
not depend on the latents from its caller. ``modulation`` derives from
sigma alone the timestep embedding and the six vectors of every block,
and ``caption`` derives the text rows from the conditioning alone. A step
given either computes the same values as one without.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from gmlx.image import ops, pacing
from gmlx.image.dtypes import ImageDtypeError


@dataclass(frozen=True)
class Krea2Config:
    dim: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    n_layers: int
    in_channels: int
    out_channels: int
    patch: int
    ffn_dim: int
    t_freq_dim: int
    txt_dim: int
    txt_layers: int
    txt_heads: int
    txt_kv_heads: int
    txt_ffn_dim: int
    n_layerwise: int
    n_refiner: int
    eps: float
    axes_dims: tuple
    rope_theta: float
    train_timesteps: int
    last_mlp: bool

    @classmethod
    def from_dict(cls, c: dict) -> "Krea2Config":
        head_dim = int(c["attention_head_dim"])
        heads = int(c["num_attention_heads"])
        axes = tuple(int(a) for a in c["axes_dims_rope"])
        if len(axes) != 3 or sum(axes) != head_dim or any(a % 2
                                                           for a in axes):
            raise ValueError(f"RoPE axes {axes} must be three even dims "
                             f"that sum to the head dim {head_dim}")
        if heads % int(c["num_key_value_heads"]):
            raise ValueError("Krea 2 query heads are a multiple of its "
                             "key and value heads")
        patch = int(c["patch_size"])
        return cls(
            dim=heads * head_dim, n_heads=heads,
            n_kv_heads=int(c["num_key_value_heads"]), head_dim=head_dim,
            n_layers=int(c["num_layers"]),
            in_channels=int(c["in_channels"]) // (patch * patch),
            out_channels=int(c["out_channels"]), patch=patch,
            ffn_dim=int(c["intermediate_size"]),
            t_freq_dim=int(c["timestep_embed_dim"]),
            txt_dim=int(c["text_hidden_dim"]),
            txt_layers=int(c["num_text_layers"]),
            txt_heads=int(c["text_num_attention_heads"]),
            txt_kv_heads=int(c["text_num_key_value_heads"]),
            txt_ffn_dim=int(c["text_intermediate_size"]),
            n_layerwise=int(c["num_layerwise_text_blocks"]),
            n_refiner=int(c["num_refiner_text_blocks"]),
            eps=float(c["norm_eps"]), axes_dims=axes,
            rope_theta=float(c["rope_theta"]),
            train_timesteps=int(c["num_train_timesteps"]),
            last_mlp=bool(c.get("last_mlp", False)))


def rope_tables(ids: np.ndarray, axes_dims, theta: float) -> tuple:
    """cos and sin tables (L, sum(axes_dims) // 2) for integer position ids
    (L, n_axes). Angles, cos and sin are float64 and round to fp32 once, as
    the reference tables are on the CPU."""
    cos, sin = [], []
    for axis, d in enumerate(axes_dims):
        freqs = 1.0 / theta ** (np.arange(0, d, 2, dtype=np.float64) / d)
        ang = np.outer(ids[:, axis].astype(np.float64), freqs)
        cos.append(np.cos(ang).astype(np.float32))
        sin.append(np.sin(ang).astype(np.float32))
    return (mx.array(np.concatenate(cos, axis=1)),
            mx.array(np.concatenate(sin, axis=1)))


class ZeroCenteredRMSNorm(nn.Module):
    """RMSNorm whose stored ``scale`` is the weight minus one. The norm runs
    in fp32 whatever the input and scale dtypes are, and returns the input
    dtype."""

    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.scale = mx.zeros((dims,))
        self.eps = eps

    def weight(self) -> mx.array:
        return self.scale.astype(mx.float32) + 1

    def __call__(self, x: mx.array) -> mx.array:
        y = mx.fast.rms_norm(x.astype(mx.float32), self.weight(), self.eps)
        return y.astype(x.dtype)


class QKNorm(nn.Module):
    def __init__(self, head_dim: int, eps: float):
        super().__init__()
        self.qnorm = ZeroCenteredRMSNorm(head_dim, eps)
        self.knorm = ZeroCenteredRMSNorm(head_dim, eps)


class Attention(nn.Module):
    """Grouped-query self-attention with q and k norms, optional RoPE and a
    sigmoid gate on its output."""

    def __init__(self, dim: int, heads: int, kv_heads: int, eps: float):
        super().__init__()
        self.head_dim = dim // heads
        self.n_heads = heads
        self.n_kv_heads = kv_heads
        self.eps = eps
        self.wq = nn.Linear(dim, heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=False)
        self.qknorm = QKNorm(self.head_dim, eps)
        self.wo = nn.Linear(dim, dim, bias=False)

    def qkv(self, x: mx.array, cos: mx.array | None,
            sin: mx.array | None) -> tuple:
        """q, k and v as (B, heads, L, D), q and k normed and rotated."""
        b, n, _ = x.shape
        d = self.head_dim
        q = self.wq(x).reshape(b, n, self.n_heads, d)
        k = self.wk(x).reshape(b, n, self.n_kv_heads, d)
        v = self.wv(x).reshape(b, n, self.n_kv_heads, d).transpose(0, 2, 1, 3)
        qn, kn = self.qknorm.qnorm, self.qknorm.knorm
        if cos is None or sin is None:
            q = qn(q).transpose(0, 2, 1, 3)
            k = kn(k).transpose(0, 2, 1, 3)
        elif self.n_kv_heads == self.n_heads:
            q, k = ops.qk_norm_rope(q, k, qn.weight(), kn.weight(), cos, sin,
                                    self.eps)
        else:
            q, _ = ops.qk_norm_rope(q, q, qn.weight(), qn.weight(), cos,
                                    sin, self.eps)
            k, _ = ops.qk_norm_rope(k, k, kn.weight(), kn.weight(), cos,
                                    sin, self.eps)
        return q, k, v

    def gated(self, x: mx.array, o: mx.array) -> mx.array:
        """The attention output (B, H, L, D) as rows, times the sigmoid of
        the gate of the block input ``x``: the rows that ``wo`` takes."""
        b, _, n, _ = o.shape
        o = o.transpose(0, 2, 1, 3).reshape(b, n, -1)
        return o * mx.sigmoid(self.gate(x))

    def __call__(self, x: mx.array, cos: mx.array | None = None,
                 sin: mx.array | None = None) -> mx.array:
        q, k, v = self.qkv(x, cos, sin)
        return self.wo(self.gated(x, ops.sdpa(q, k, v)))


class SwiGLU(nn.Module):
    """down(silu(gate(x)) * up(x)). A gate and up pair that
    ``ops.glu_pair`` accepts runs with its swiglu as one mlx-kquant
    kernel."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def hidden(self, x: mx.array) -> mx.array:
        """The rows that ``down`` takes."""
        pair = ops.glu_pair(self.gate, self.up, x)
        if pair is None:
            return ops.swiglu(self.gate(x), self.up(x))
        return ops.swiglu_fused(x, *pair)

    def __call__(self, x: mx.array, rows=None) -> mx.array:
        return self.down(self.hidden(x) if rows is None else rows)


def gated_projection(x: mx.array, gate: mx.array, lin, h: mx.array,
                     branch) -> mx.array:
    """``x + gate * lin(h)``: on mlx-kquant's residual epilogue when
    ``ops.resid_leaf`` takes the leaf, else ``x + gate * branch()`` on the
    stock ops, where ``branch`` is the module call that gives ``lin(h)``."""
    leaf = ops.resid_leaf(lin, h, x, gate)
    if leaf is not None:
        return ops.resid_fused(h, leaf, x, gate)
    return ops.gated_residual(x, gate, branch())


class TextFusionBlock(nn.Module):
    """A pre-norm block with no modulation and no RoPE."""

    def __init__(self, cfg: Krea2Config):
        super().__init__()
        self.prenorm = ZeroCenteredRMSNorm(cfg.txt_dim, cfg.eps)
        self.postnorm = ZeroCenteredRMSNorm(cfg.txt_dim, cfg.eps)
        self.attn = Attention(cfg.txt_dim, cfg.txt_heads, cfg.txt_kv_heads,
                              cfg.eps)
        self.mlp = SwiGLU(cfg.txt_dim, cfg.txt_ffn_dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.prenorm(x))
        return x + self.mlp(self.postnorm(x))


class TextFusion(nn.Module):
    def __init__(self, cfg: Krea2Config):
        super().__init__()
        self.layerwise_blocks = [TextFusionBlock(cfg)
                                 for _ in range(cfg.n_layerwise)]
        self.projector = nn.Linear(cfg.txt_layers, 1, bias=False)
        self.refiner_blocks = [TextFusionBlock(cfg)
                               for _ in range(cfg.n_refiner)]
        self.txt_layers = cfg.txt_layers
        self.txt_dim = cfg.txt_dim

    def __call__(self, feats: mx.array, taps: dict | None = None) -> mx.array:
        """Features (B, L, layers * dim) to text rows (B, L, dim)."""
        b, n, _ = feats.shape
        x = feats.reshape(b * n, self.txt_layers, self.txt_dim)
        for i, blk in enumerate(self.layerwise_blocks):
            x = blk(x)
            if taps is not None:
                taps[f"txtfusion.layerwise_blocks.{i}"] = x
        x = x.reshape(b, n, self.txt_layers, self.txt_dim).transpose(
            0, 1, 3, 2)
        x = self.projector(x)[..., 0]
        if taps is not None:
            taps["txtfusion.projector"] = x
        for i, blk in enumerate(self.refiner_blocks):
            x = blk(x)
            if taps is not None:
                taps[f"txtfusion.refiner_blocks.{i}"] = x
        return x


class Modulation(nn.Module):
    """A block's six vectors, added to the shared ones from ``tproj``."""

    def __init__(self, dim: int):
        super().__init__()
        self.lin = mx.zeros((6 * dim,))


class Block(nn.Module):
    def __init__(self, cfg: Krea2Config):
        super().__init__()
        self.mod = Modulation(cfg.dim)
        self.prenorm = ZeroCenteredRMSNorm(cfg.dim, cfg.eps)
        self.postnorm = ZeroCenteredRMSNorm(cfg.dim, cfg.eps)
        self.attn = Attention(cfg.dim, cfg.n_heads, cfg.n_kv_heads, cfg.eps)
        self.mlp = SwiGLU(cfg.dim, cfg.ffn_dim)

    def modulation(self, tvec: mx.array) -> tuple:
        """prescale, preshift, pregate, postscale, postshift and postgate,
        each (B, 1, dim), from the shared vectors (B, 1, 6 dim)."""
        rows = tvec + self.mod.lin.astype(tvec.dtype)
        return tuple(mx.split(rows, 6, axis=-1))

    def __call__(self, x: mx.array, mod: tuple, cos: mx.array,
                 sin: mx.array) -> mx.array:
        prescale, preshift, pregate, postscale, postshift, postgate = mod
        h = ops.modulate(self.prenorm(x), prescale) + preshift
        q, k, v = self.attn.qkv(h, cos, sin)
        rows = self.attn.gated(h, ops.sdpa(q, k, v))
        x = gated_projection(x, pregate, self.attn.wo, rows,
                             lambda: self.attn.wo(rows))
        h = ops.modulate(self.postnorm(x), postscale) + postshift
        hid = self.mlp.hidden(h)
        return gated_projection(x, postgate, self.mlp.down, hid,
                                lambda: self.mlp(h, hid))


class SimpleModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = mx.zeros((2, dim))


class LastLayer(nn.Module):
    def __init__(self, cfg: Krea2Config):
        super().__init__()
        self.norm = ZeroCenteredRMSNorm(cfg.dim, cfg.eps)
        self.linear = nn.Linear(cfg.dim, cfg.out_channels)
        self.modulation = SimpleModulation(cfg.dim)
        if cfg.last_mlp:
            # In one converter's files and never read by the reference.
            self.up = nn.Linear(cfg.dim, cfg.dim, bias=False)
            self.down = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def scale_shift(self, temb: mx.array) -> tuple:
        """The scale and the shift (B, 1, dim) of the timestep embedding
        (B, 1, dim)."""
        rows = temb + self.modulation.lin.astype(temb.dtype)
        return rows[:, :1], rows[:, 1:]

    def __call__(self, x: mx.array, scale: mx.array,
                 shift: mx.array) -> mx.array:
        return self.linear(ops.modulate(self.norm(x), scale) + shift)


@dataclass(frozen=True)
class StepModulation:
    """What one step derives from sigma: each block's six vectors and the
    final layer's scale and shift, each (B, 1, dim)."""
    blocks: tuple
    scale: mx.array
    shift: mx.array

    def arrays(self) -> list:
        out = [self.scale, self.shift]
        for mod in self.blocks:
            out.extend(mod)
        return out


@dataclass(frozen=True)
class Caption:
    """What one step derives from the conditioning: its text rows after
    ``txtmlp``, (B, L, dim)."""
    rows: mx.array

    def arrays(self) -> list:
        return [self.rows]


class Krea2DiT(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        cfg = Krea2Config.from_dict(config)
        self.cfg = cfg
        self.act_dtype = mx.bfloat16
        p2 = cfg.patch * cfg.patch
        self.first = nn.Linear(cfg.in_channels * p2, cfg.dim)
        self.blocks = [Block(cfg) for _ in range(cfg.n_layers)]
        self.tmlp = [nn.Linear(cfg.t_freq_dim, cfg.dim),
                     nn.GELU(approx="tanh"), nn.Linear(cfg.dim, cfg.dim)]
        self.txtfusion = TextFusion(cfg)
        self.txtmlp = [ZeroCenteredRMSNorm(cfg.txt_dim, cfg.eps),
                       nn.Linear(cfg.txt_dim, cfg.dim),
                       nn.GELU(approx="tanh"),
                       nn.Linear(cfg.dim, cfg.dim)]
        self.last = LastLayer(cfg)
        self.tproj = [nn.GELU(approx="tanh"),
                      nn.Linear(cfg.dim, 6 * cfg.dim)]

    @property
    def act_dtype(self) -> mx.Dtype:
        return self._act_dtype

    @act_dtype.setter
    def act_dtype(self, dtype: mx.Dtype) -> None:
        if dtype == mx.float16:
            raise ImageDtypeError("Krea 2 never runs in float16")
        self._act_dtype = dtype

    def timesteps(self, sigma, batch: int) -> np.ndarray:
        """The values the timestep embedding takes for sigmas, in fp32 as
        the reference forms them: the scheduler's sigma times its
        training timesteps, the pipeline's division by 1000, and the
        embedder's factor of 1000."""
        s = np.broadcast_to(np.asarray(sigma, np.float32), (batch,))
        passed = ((s * np.float32(self.cfg.train_timesteps))
                  / np.float32(1000))
        return np.float32(1000) * passed

    def temb(self, t) -> mx.array:
        """The timestep embedding (B, 1, dim) of embedder inputs ``t``."""
        freq = ops.timestep_embedding(t, self.cfg.t_freq_dim)
        freq = freq.astype(self.act_dtype)[:, None, :]
        return self.tmlp[2](nn.gelu_approx(self.tmlp[0](freq)))

    def modulation(self, sigma, batch: int) -> StepModulation:
        temb = self.temb(self.timesteps(sigma, batch))
        tvec = self.tproj[1](nn.gelu_approx(temb))
        scale, shift = self.last.scale_shift(temb)
        return StepModulation(
            blocks=tuple(b.modulation(tvec) for b in self.blocks),
            scale=scale, shift=shift)

    def caption(self, feats: mx.array, taps: dict | None = None) -> Caption:
        """The text rows of features (B, L, layers * text dim)."""
        x = self.txtfusion(feats.astype(self.act_dtype), taps)
        x = self.txtmlp[0](x)
        x = self.txtmlp[3](nn.gelu_approx(self.txtmlp[1](x)))
        if taps is not None:
            taps["txtmlp"] = x
        return Caption(x)

    def patchify(self, latents: mx.array) -> mx.array:
        """Latents (B, C, H, W) as tokens (B, H/p * W/p, C p p), each token
        its channels, then its rows and columns within the patch."""
        b, c, h, w = latents.shape
        p = self.cfg.patch
        x = latents.reshape(b, c, h // p, p, w // p, p)
        return x.transpose(0, 2, 4, 1, 3, 5).reshape(
            b, (h // p) * (w // p), c * p * p)

    def unpatchify(self, x: mx.array, ht: int, wt: int) -> mx.array:
        b = x.shape[0]
        p, c = self.cfg.patch, self.cfg.out_channels // (self.cfg.patch ** 2)
        x = x.reshape(b, ht, wt, c, p, p).transpose(0, 3, 1, 4, 2, 5)
        return x.reshape(b, c, ht * p, wt * p)

    def rope(self, n_txt: int, ht: int, wt: int):
        """cos and sin tables of the joint rows: text at zero, image row
        (h, w) at (0, h, w)."""
        ids = np.zeros((n_txt + ht * wt, 3), np.int64)
        hh, ww = np.meshgrid(np.arange(ht), np.arange(wt), indexing="ij")
        ids[n_txt:, 1] = hh.reshape(-1)
        ids[n_txt:, 2] = ww.reshape(-1)
        return rope_tables(ids, self.cfg.axes_dims, self.cfg.rope_theta)

    def __call__(self, latents: mx.array, sigma, feats: mx.array | None,
                 taps: dict | None = None, *,
                 mod: StepModulation | None = None,
                 caption: Caption | None = None) -> mx.array:
        """Velocity (B, C, H, W) fp32 for fp32 latents at ``sigma`` with text
        features (B, L, layers * text dim). ``mod`` and ``caption``, from
        ``modulation`` and ``caption``, stand in for the work they hold.
        ``taps`` collects the output of every block by module path."""
        b, _, h, w = latents.shape
        p = self.cfg.patch
        ht, wt = h // p, w // p
        if mod is None:
            mod = self.modulation(sigma, b)
        if caption is None:
            if feats is None:
                raise ValueError("pass text features or their caption")
            caption = self.caption(feats, taps)
        txt = caption.rows
        if txt.shape[0] != b:
            txt = mx.broadcast_to(txt, (b, *txt.shape[1:]))
        img = self.first(self.patchify(latents).astype(self.act_dtype))
        if taps is not None:
            taps["first"] = img
        n_txt = txt.shape[1]
        x = mx.concatenate([txt, img], axis=1)
        cos, sin = self.rope(n_txt, ht, wt)
        for i, blk in enumerate(self.blocks):
            x = blk(x, mod.blocks[i], cos, sin)
            pacing.point(x)
            if taps is not None:
                taps[f"blocks.{i}"] = x
        out = self.last(x[:, n_txt:], mod.scale, mod.shift)
        if taps is not None:
            taps["last"] = out
        return self.unpatchify(out, ht, wt).astype(mx.float32)
