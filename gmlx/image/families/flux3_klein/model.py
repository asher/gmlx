# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright 2026 Black Forest Labs, Inc.
# Ported from BFL's FLUX.3 klein reference src/flux3/dit.py (the DiT, its
# block, mode modulations, embedders, RoPE and final layer) and the image
# RoPE ids of src/flux3/pipeline.py and src/flux3/autoencoder.py at commit
# 45e57c9 and modified for gmlx (Apache-2.0; see
# licenses/flux-3-image-klein-LICENSE).
"""FLUX.3 klein DiT module tree and forward.

Attribute paths equal the GGUF tensor names, so the per-mode containers
are dicts and numbered elements are plain lists.

A sample holds three modes of tokens: the image latents, the latents of
the reference images (``image_cond``, none for text to image) and the
text. Each mode runs five blocks of its own, ``content_mode_blocks`` for
the two image modes and ``txt_mode_blocks`` for the text, with attention
inside the mode. The tokens of the three modes then join, in the order
image, image_cond, text, and run 28 shared ``single_blocks`` with
attention over all of them. No mask: the text pad tokens are attended. The
velocity is the final layer on the image tokens. Every block is parallel:
one LayerNorm modulation feeds the attention and the feed-forward, and
one gate scales their sum.

Each mode takes a time vector: the image the step's time, the reference
images and the text time zero. The distilled checkpoint is a flow map:
``ref_time_in`` embeds the time the step goes to, and its vector is added
to every mode's. Each mode has its own modulation of the early blocks and
of the single blocks, keyed ``txt`` for the text, and every block of a
kind reads the same one.

The time embedding follows the reference pipeline, which holds every
time in bf16: the embedder multiplies by 1000 in bf16 and returns the
embedding in bf16, in every activation dtype. The time vectors and the
modulations then run in fp32 and round once.

RoPE has four axes of 32 dims. An image token sits at (0, row / 2,
column / 2, 0), a reference token at (0, row / 2, column / 2, -1 - its
image's index), and a text token at (0, 0, 0, position).
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from gmlx.image import ops, pacing
from gmlx.image.dtypes import ImageDtypeError
from gmlx.image.ops.products import own, precise

MODES = ("image", "image_cond", "text")
MOD_KEYS = {"image": "image", "image_cond": "image_cond", "text": "txt"}


@dataclass(frozen=True)
class Flux3Config:
    dim: int
    heads: int
    head_dim: int
    depth: int
    depth_single: int
    mlp_dim: int
    in_channels: int
    cond_channels: int
    text_dim: int
    t_dim: int
    axes_dims: tuple
    theta: float
    eps: float
    flow_map: bool
    image_rope_scale: float
    # A latent pixel is one token: the DiT has no patch of its own.
    patch: int = 1

    @classmethod
    def from_dict(cls, c: dict) -> "Flux3Config":
        head_dim = int(c["attention_head_dim"])
        heads = int(c["num_attention_heads"])
        axes = tuple(int(a) for a in c["axes_dims_rope"])
        if sum(axes) != head_dim or any(a % 2 for a in axes):
            raise ValueError(f"RoPE axes {axes} must be even dims that sum "
                             f"to the head dim {head_dim}")
        dim = heads * head_dim
        return cls(
            dim=dim, heads=heads, head_dim=head_dim,
            depth=int(c["num_layers"]),
            depth_single=int(c["num_single_layers"]),
            mlp_dim=int(dim * float(c["mlp_ratio"])),
            in_channels=int(c["in_channels"]),
            cond_channels=int(c["cond_in_channels"]),
            text_dim=int(c["joint_attention_dim"]),
            t_dim=int(c["timestep_channels"]), axes_dims=axes,
            theta=float(c["rope_theta"]), eps=float(c["eps"]),
            flow_map=bool(c["flow_map"]),
            image_rope_scale=float(c["dit"]["image_rope_scale"]))


def _bf16(a: np.ndarray) -> np.ndarray:
    """float32 values rounded to bfloat16, to nearest even."""
    return np.array(mx.array(np.asarray(a, np.float32)).astype(
        mx.bfloat16).astype(mx.float32))


def time_embedding(t, dim: int) -> mx.array:
    """The embedding (B, dim) of times ``t`` as the reference pipeline
    forms it: the time in bf16, 1000 times it rounded to bf16, the cosines
    and sines in fp32, and the embedding rounded to bf16. Returned as the
    bf16 values in float32."""
    t1000 = _bf16(np.float32(1000) * _bf16(np.reshape(t, (-1,))))
    emb = ops.timestep_embedding(t1000, dim)
    return emb.astype(mx.bfloat16).astype(mx.float32)


def rope_tables(ids: np.ndarray, axes_dims, theta: float) -> tuple:
    """cos and sin tables (L, sum(axes_dims) // 2) for position ids (L,
    n_axes), which may be halves. Angles, cos and sin are float64 and
    round to fp32 once, as the reference's are."""
    cos, sin = [], []
    for axis, d in enumerate(axes_dims):
        freqs = 1.0 / theta ** (np.arange(0, d, 2, dtype=np.float64) / d)
        ang = np.outer(np.asarray(ids[:, axis], np.float64), freqs)
        cos.append(np.cos(ang).astype(np.float32))
        sin.append(np.sin(ang).astype(np.float32))
    return (mx.array(np.concatenate(cos, axis=1)),
            mx.array(np.concatenate(sin, axis=1)))


def image_ids(h: int, w: int, scale: float, index: int | None = None
              ) -> np.ndarray:
    """Position ids (h * w, 4) of a latent grid in row-major order: (0,
    row * scale, column * scale, 0) for the image, and the last axis at
    -1 - ``index`` for reference image ``index``."""
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    ids = np.zeros((h * w, 4), np.float32)
    ids[:, 1] = rows.reshape(-1) * np.float32(scale)
    ids[:, 2] = cols.reshape(-1) * np.float32(scale)
    if index is not None:
        ids[:, 3] = -(1 + index)
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
    in the dtype of ``scale``, as the reference computes it under
    autocast, rounded once to ``dtype``."""
    y = ops.layer_norm(x.astype(mx.float32), eps)
    return (ops.modulate(y, scale) + shift).astype(dtype)


class Block(nn.Module):
    """The parallel block of every stream: one modulated LayerNorm feeds
    the attention and the feed-forward, and one gate scales the sum of
    their outputs."""

    def __init__(self, cfg: Flux3Config):
        super().__init__()
        d = cfg.dim
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.mlp_in = nn.Linear(d, 2 * cfg.mlp_dim, bias=False)
        self.attn_out = nn.Linear(d, d, bias=False)
        self.mlp_out = nn.Linear(cfg.mlp_dim, d, bias=False)
        self.norm = QKNorm(cfg.head_dim)
        self.heads = cfg.heads
        self.head_dim = cfg.head_dim
        self.eps = cfg.eps

    def adaln(self, x: mx.array, shift: mx.array,
              scale: mx.array) -> mx.array:
        return adaln(x, shift, scale, self.eps, x.dtype)

    def attention(self, h: mx.array, cos: mx.array,
                  sin: mx.array) -> mx.array:
        """The attention rows (B, L, dim) that ``attn_out`` takes."""
        b, n, _ = h.shape
        q = self.q_proj(h).reshape(b, n, self.heads, self.head_dim)
        k = self.k_proj(h).reshape(b, n, self.heads, self.head_dim)
        v = self.v_proj(h).reshape(b, n, self.heads, self.head_dim)
        q, k = ops.qk_norm_rope(q, k, self.norm.query_norm.scale,
                                self.norm.key_norm.scale, cos, sin, self.eps)
        o = ops.sdpa(q, k, v.transpose(0, 2, 1, 3))
        return o.transpose(0, 2, 1, 3).reshape(b, n, -1)

    def hidden(self, h: mx.array) -> mx.array:
        """``silu(a) * b`` of the halves a, b of ``mlp_in(h)``: the rows
        that ``mlp_out`` takes."""
        pair = ops.glu_halves(self.mlp_in, h)
        if pair is not None:
            return ops.swiglu_fused(h, *pair)
        a, b = mx.split(self.mlp_in(h), 2, axis=-1)
        return ops.swiglu(a, b)

    def out(self, x: mx.array, gate: mx.array, rows: mx.array,
            hid: mx.array) -> mx.array:
        """``x + gate * (attn_out(rows) + mlp_out(hid))``. When the
        residual route takes both leaves, as it does for a gate of one row
        per sample, the sum runs as two of mlx-kquant's residual
        epilogues, one per leaf."""
        la = ops.resid_leaf(self.attn_out, rows, x, gate)
        lm = ops.resid_leaf(self.mlp_out, hid, x, gate)
        if la is not None and lm is not None:
            return ops.resid_fused(hid, lm, ops.resid_fused(rows, la, x, gate),
                                   gate)
        return ops.gated_residual(x, gate,
                                  self.attn_out(rows) + self.mlp_out(hid))

    def __call__(self, x: mx.array, mod: tuple, cos: mx.array,
                 sin: mx.array) -> mx.array:
        shift, scale, gate = mod
        h = self.adaln(x, shift, scale)
        return self.out(x, gate, self.attention(h, cos, sin), self.hidden(h))


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, dim, bias=False)
        self.out_layer = nn.Linear(dim, dim, bias=False)


class ModeModulation(nn.Module):
    """shift, scale and gate of a mode, from its time vector."""

    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, 3 * dim, bias=False)


class LastLayer(nn.Module):
    def __init__(self, dim: int, out: int):
        super().__init__()
        self.linear = nn.Linear(dim, out, bias=False)
        self.adaLN_modulation = [nn.SiLU(), nn.Linear(dim, 2 * dim,
                                                      bias=False)]
        self.eps = 1e-6

    def __call__(self, x: mx.array, shift: mx.array,
                 scale: mx.array) -> mx.array:
        return self.linear(adaln(x, shift, scale, self.eps, x.dtype))


@dataclass(frozen=True)
class StepModulation:
    """What one step derives from its times: for each mode, the shift,
    scale and gate of the early blocks and of the single blocks, and the
    final layer's shift and scale, each (B, 1, dim) in the activation
    dtype."""
    early: dict
    single: dict
    final: tuple

    def arrays(self) -> list:
        out = list(self.final)
        for m in MODES:
            out.extend(self.early[m])
            out.extend(self.single[m])
        return out


@dataclass(frozen=True)
class Caption:
    """What one step derives from the conditioning: the text tokens after
    ``txt_in``, (B, L, dim). The text blocks read the step's time vector,
    so they run in the step."""
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
    def of(cls, latents: list, scale: float) -> "References":
        """Each reference latent (B, C, h, w) as tokens, with the ids of
        its index in the list."""
        toks = [tokens(lat) for lat in latents]
        ids = [image_ids(lat.shape[2], lat.shape[3], scale, i)
               for i, lat in enumerate(latents)]
        return cls(mx.concatenate(toks, axis=1), np.concatenate(ids))


class Flux3KleinDiT(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        cfg = Flux3Config.from_dict(config)
        self.cfg = cfg
        self.act_dtype = mx.bfloat16
        d = cfg.dim
        self.emb_in = {"image": nn.Linear(cfg.in_channels, d, bias=False),
                       "image_cond": nn.Linear(cfg.cond_channels, d,
                                               bias=False)}
        self.txt_in = nn.Linear(cfg.text_dim, d, bias=False)
        self.time_in = MLPEmbedder(cfg.t_dim, d)
        if cfg.flow_map:
            self.ref_time_in = MLPEmbedder(cfg.t_dim, d)
        self.early_stream_modulations = {k: ModeModulation(d)
                                         for k in MOD_KEYS.values()}
        self.single_stream_modulations = {k: ModeModulation(d)
                                          for k in MOD_KEYS.values()}
        self.content_mode_blocks = {
            m: [Block(cfg) for _ in range(cfg.depth)]
            for m in ("image", "image_cond")}
        self.txt_mode_blocks = [Block(cfg) for _ in range(cfg.depth)]
        self.single_blocks = [Block(cfg) for _ in range(cfg.depth_single)]
        self.final_layer = {"image": LastLayer(d, cfg.in_channels)}

    @property
    def act_dtype(self) -> mx.Dtype:
        return self._act_dtype

    @act_dtype.setter
    def act_dtype(self, dtype: mx.Dtype) -> None:
        if dtype == mx.float16:
            raise ImageDtypeError("FLUX.3 klein never runs in float16")
        self._act_dtype = dtype

    def _lin(self):
        return precise if self.act_dtype != mx.float32 else own

    def _embed(self, emb: MLPEmbedder, t, batch: int) -> mx.array:
        lin = self._lin()
        e = time_embedding(np.broadcast_to(np.asarray(t, np.float32),
                                           (batch,)), self.cfg.t_dim)
        return lin(emb.out_layer, nn.silu(lin(emb.in_layer, e)))

    def vectors(self, sigma, batch: int, target=None) -> dict:
        """The time vector of each mode, (B, dim) fp32: the image at
        ``sigma``, the other modes at zero, each plus the embedding of
        ``target`` for a flow map."""
        if self.cfg.flow_map and target is None:
            raise ValueError("the distilled FLUX.3 klein DiT is a flow map: "
                             "pass the time each step goes to as target")
        if not self.cfg.flow_map and target is not None:
            raise ValueError("the FLUX.3 klein Base DiT takes no target time")
        img = self._embed(self.time_in, sigma, batch)
        zero = self._embed(self.time_in, 0.0, batch)
        if target is not None:
            ref = self._embed(self.ref_time_in, target, batch)
            img, zero = img + ref, zero + ref
        return {"image": img, "image_cond": zero, "text": zero}

    def modulation(self, sigma, batch: int, target=None) -> StepModulation:
        lin = self._lin()
        act = self.act_dtype
        vec = self.vectors(sigma, batch, target)

        def chunks(mod, v, n):
            y = lin(mod, nn.silu(v)).astype(act)[:, None, :]
            return tuple(mx.split(y, n, axis=-1))

        early, single = {}, {}
        for m in MODES:
            k = MOD_KEYS[m]
            early[m] = chunks(self.early_stream_modulations[k].lin, vec[m], 3)
            single[m] = chunks(self.single_stream_modulations[k].lin, vec[m],
                               3)
        final = chunks(self.final_layer["image"].adaLN_modulation[1],
                       vec["image"], 2)
        return StepModulation(early=early, single=single, final=final)

    def references(self, latents) -> References:
        """The reference latents (B, C, h, w), in the order of their
        images, as the tokens and ids the DiT takes, with the image rope
        scale of the config."""
        return References.of(list(latents), self.cfg.image_rope_scale)

    def caption(self, feats: mx.array, taps: dict | None = None) -> Caption:
        """The text tokens of features (B, L, text dim) after txt_in."""
        x = self.txt_in(feats.astype(self.act_dtype))
        if taps is not None:
            taps["txt_in"] = x
        return Caption(x)

    def _stream(self, x: mx.array, blocks: list, mod: tuple, ids, name: str,
                taps: dict | None) -> mx.array:
        cos, sin = rope_tables(ids, self.cfg.axes_dims, self.cfg.theta)
        for i, blk in enumerate(blocks):
            x = blk(x, mod, cos, sin)
            pacing.point(x)
            if taps is not None:
                taps[f"{name}.{i}"] = x
        return x

    def __call__(self, latents: mx.array, sigma, feats: mx.array | None,
                 taps: dict | None = None, *, target=None,
                 refs: References | None = None,
                 mod: StepModulation | None = None,
                 caption: Caption | None = None) -> mx.array:
        """Velocity (B, C, h, w) fp32 for latents (B, C, h, w) at
        ``sigma``, with text features (B, L, text dim) and the tokens of
        any reference images. ``target`` is the time the step goes to,
        which a flow map takes. ``mod`` and ``caption``, from
        ``modulation`` and ``caption``, stand in for the work they hold.
        ``taps`` collects the output of every block by module path."""
        b, _, h, w = latents.shape
        cfg = self.cfg
        act = self.act_dtype
        if mod is None:
            mod = self.modulation(sigma, b, target)
        if caption is None:
            if feats is None:
                raise ValueError("pass text features or their caption")
            caption = self.caption(feats, taps)
        txt = caption.rows
        if txt.shape[0] != b:
            txt = mx.broadcast_to(txt, (b, *txt.shape[1:]))
        img = self.emb_in["image"](tokens(latents).astype(act))
        if taps is not None:
            taps["emb_in.image"] = img
        ids = {"image": image_ids(h, w, cfg.image_rope_scale),
               "text": text_ids(txt.shape[1])}
        x = {"image": img, "text": txt}
        if refs is not None:
            cond = refs.tokens.astype(act)
            if cond.shape[0] != b:
                cond = mx.broadcast_to(cond, (b, *cond.shape[1:]))
            x["image_cond"] = self.emb_in["image_cond"](cond)
            ids["image_cond"] = np.asarray(refs.ids, np.float32)
            if taps is not None:
                taps["emb_in.image_cond"] = x["image_cond"]
        streams = {"image": "content_mode_blocks.image",
                   "image_cond": "content_mode_blocks.image_cond",
                   "text": "txt_mode_blocks"}
        for m in MODES:
            if m not in x:
                continue
            blocks = (self.txt_mode_blocks if m == "text"
                      else self.content_mode_blocks[m])
            x[m] = self._stream(x[m], blocks, mod.early[m], ids[m],
                                streams[m], taps)
        present = [m for m in MODES if m in x]
        joint = mx.concatenate([x[m] for m in present], axis=1)
        smod = tuple(mx.concatenate(
            [mx.broadcast_to(mod.single[m][i], (b, x[m].shape[1], cfg.dim))
             for m in present], axis=1) for i in range(3))
        joint = self._stream(joint, self.single_blocks, smod,
                             np.concatenate([ids[m] for m in present]),
                             "single_blocks", taps)
        n_img = img.shape[1]
        out = self.final_layer["image"](joint[:, :n_img], *mod.final)
        if taps is not None:
            taps["final_layer.image"] = out
        c = out.shape[-1]
        return out.transpose(0, 2, 1).reshape(b, c, h, w).astype(mx.float32)
