# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright 2026 OpenMOSS Team
# Ported from MOSSImageTransformer in OpenMOSS-Team/MOSS-Image-SFT
# pipeline_moss_image.py (its blocks, text projection, RoPE bands and
# coordinates, timestep sinusoid, modulations and output head) at revision
# fc5de0d5 and modified for gmlx (Apache-2.0; see
# licenses/MOSS-Image-LICENSE).
"""MOSS-Image DiT module tree and forward.

Attribute paths equal the GGUF tensor names, which are the release's
training names, so numbered elements are plain lists and ``proj`` of a
block is a dict.

Latents arrive patched, 128 channels a token, and each latent pixel is
one token in row-major order. The text features pass ``text_embedder``:
a projection to the model width and two refiner blocks with biases and no
modulation, over the text alone. The image tokens and then the text rows
run through every block as one sequence, and the velocity is the final
head on the image tokens, with no sign flip. Each sample runs at its own
text length, so no row is padding.

One timestep embedding drives every block: ``shared_block_modulation``
maps it to six vectors (shift, scale and gate of the attention, then of
the feed-forward) that every block reads. ``video_head_modulation`` maps
it to the shift and scale of the head. The sinusoid of the timestep is
formed in float64, cosines first, and rounded to float32 once, as the
release forms it. The time vectors then run in float32 and round once.

The feed-forward is ``ffn_down(situ(hidden, gate))`` over the two halves
of ``ffn_up_gate``, where situ is a soft-capped gated SiLU, computed in
float32 and rounded once.

RoPE has four axes (r, f, h, w) of 16 bands at base 10000. The bands,
the phases and their sines and cosines are float32 as the release builds
them. An image token at (h, w) of an H by W grid sits at (0, L, h - (H -
1) / 2, w - (W - 1) / 2) for a text of L rows, and text row i at (0, i,
0, 0). The rotation pairs dim j with dim j + 64; ``interleave`` puts each
pair side by side, in q, k and their norm weights alike, so the
interleaved rotation of ``ops.qk_norm_rope`` computes the release's and
the attention scores are unchanged.

The forward keeps no caches, but it takes the two parts of a step that do
not depend on the latents from its caller. ``modulation`` derives from
sigma alone the time vectors, and ``caption`` derives the text rows from
the conditioning alone. A step given either computes the same values as
one without.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from gmlx.image import ops, pacing
from gmlx.image.dtypes import ImageDtypeError
from gmlx.image.ops.products import own, precise


@dataclass(frozen=True)
class MossConfig:
    dim: int
    ffn_dim: int
    heads: int
    head_dim: int
    depth: int
    text_dim: int
    text_depth: int
    freq_dim: int
    latent: int
    bands: int
    rope_base: float
    eps: float
    time_period: float
    train_timesteps: int
    # A latent pixel is one token: the DiT has no patch of its own.
    patch: int = 1

    @property
    def in_channels(self) -> int:
        """Channels of a latent token, the noise the sampler draws."""
        return self.latent

    @classmethod
    def from_dict(cls, c: dict) -> "MossConfig":
        dim, heads = int(c["dim"]), int(c["num_heads"])
        head_dim = int(c.get("head_dim", dim // max(heads, 1)))
        bands = int(c.get("num_bands", head_dim // 8))
        if heads * head_dim != dim or head_dim % 8:
            raise ValueError(f"MOSS-Image takes {heads} heads that divide "
                             f"its width {dim} into a head dim that is a "
                             "multiple of 8")
        if 8 * bands != head_dim:
            raise ValueError(f"MOSS-Image takes {head_dim // 8} RoPE bands "
                             f"a head dim of {head_dim}, not {bands}")
        if int(c["freq_dim"]) % 2:
            raise ValueError("the MOSS-Image timestep sinusoid takes an "
                             "even width")
        return cls(
            dim=dim, ffn_dim=int(c["ffn_dim"]), heads=heads,
            head_dim=head_dim, depth=int(c["num_layers"]),
            text_dim=int(c["text_dim"]),
            text_depth=int(c.get("num_text_blocks", 2)),
            freq_dim=int(c["freq_dim"]), latent=int(c["latent_channels"]),
            bands=bands, rope_base=float(c["rope_base"]),
            eps=float(c.get("norm_eps", 1e-6)),
            time_period=float(c.get("time_period", 10000.0)),
            train_timesteps=int(c.get("num_train_timesteps", 1000)))


def rope_bands(n: int, base: float) -> np.ndarray:
    """The release's bands ``1 / base ** (k / n)``, k = 0 to n - 1, each
    op in float32."""
    e = np.arange(n, dtype=np.float32) / np.float32(n)
    return (np.float32(1) / (np.float32(base) ** e)).astype(np.float32)


def positions(h: int, w: int, n_txt: int) -> np.ndarray:
    """The (r, f, h, w) coordinates (h * w + n_txt, 4), float32, of an h
    by w image grid in row-major order and then of ``n_txt`` text rows."""
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    img = np.zeros((h * w, 4), np.float32)
    img[:, 1] = n_txt
    img[:, 2] = rows.reshape(-1) - np.float32((h - 1) / 2)
    img[:, 3] = cols.reshape(-1) - np.float32((w - 1) / 2)
    txt = np.zeros((n_txt, 4), np.float32)
    txt[:, 1] = np.arange(n_txt)
    return np.concatenate([img, txt])


def rope_tables(coords: np.ndarray, bands: np.ndarray) -> tuple:
    """cos and sin tables (L, axes * bands), fp32, of coordinates (L,
    axes): the angle of column j is axis j // bands at band j % bands.
    The phase is a float32 product, as the release forms it, and its
    cosine and sine round to float32 once."""
    phase = (coords[:, :, None] * bands[None, None, :]).reshape(
        coords.shape[0], -1).astype(np.float64)
    return (mx.array(np.cos(phase).astype(np.float32)),
            mx.array(np.sin(phase).astype(np.float32)))


def time_sinusoid(t, dim: int, period: float) -> mx.array:
    """The embedding (B, dim) of timesteps ``t``: cosines then sines of t
    times ``period ** (-k / (dim / 2))``, in float64, rounded to float32
    once."""
    half = dim // 2
    freq = period ** (-np.arange(half, dtype=np.float64) / half)
    phase = np.asarray(t, np.float64).reshape(-1, 1) * freq[None]
    emb = np.concatenate([np.cos(phase), np.sin(phase)], axis=-1)
    return mx.array(emb.astype(np.float32))


def interleave(x: mx.array) -> mx.array:
    """The last dim of ``x`` with dims j and j + D/2 side by side, the
    pairs the release rotates together."""
    *lead, d = x.shape
    return x.reshape(*lead, 2, d // 2).swapaxes(-1, -2).reshape(*lead, d)


def situ(hidden: mx.array, gate: mx.array) -> mx.array:
    """``(4 tanh(gate / 4) sigmoid(gate)) (25 tanh(hidden / 25))`` in
    float32, rounded once to the input dtype."""
    g = gate.astype(mx.float32)
    h = hidden.astype(mx.float32)
    y = (4 * mx.tanh(g / 4) * mx.sigmoid(g)) * (25 * mx.tanh(h / 25))
    return y.astype(hidden.dtype)


def qk_rope(q: mx.array, k: mx.array, q_w: mx.array, k_w: mx.array,
            cos: mx.array, sin: mx.array, eps: float) -> tuple:
    """q and k (B, L, H, D) normed over the head dim and rotated by the
    release's pairs, as (B, H, L, D) in the dims of ``interleave``."""
    return ops.qk_norm_rope(interleave(q), interleave(k), interleave(q_w),
                            interleave(k_w), cos, sin, eps)


def gated_projection(x: mx.array, gate: mx.array, lin, h: mx.array,
                     branch) -> mx.array:
    """``x + gate * lin(h)``: on mlx-kquant's residual epilogue when
    ``ops.resid_leaf`` takes the leaf, else ``x + gate * branch()`` on the
    stock ops, where ``branch`` is the module call that gives ``lin(h)``."""
    leaf = ops.resid_leaf(lin, h, x, gate)
    if leaf is not None:
        return ops.resid_fused(h, leaf, x, gate)
    return ops.gated_residual(x, gate, branch())


class Rope(nn.Module):
    """The file's copy of the bands. The forward builds its own, as the
    release does."""

    def __init__(self, bands: int):
        super().__init__()
        self.bands = mx.zeros((bands,))


class TextBlock(nn.Module):
    """A refiner block of the text projection: pre-norm attention and
    feed-forward with biases, no modulation."""

    def __init__(self, cfg: MossConfig):
        super().__init__()
        d, f = cfg.dim, cfg.ffn_dim
        self.heads, self.head_dim, self.eps = cfg.heads, cfg.head_dim, cfg.eps
        self.norm1 = ops.RMSNorm(d, cfg.eps)
        self.norm2 = ops.RMSNorm(d, cfg.eps)
        self.q_norm = ops.RMSNorm(cfg.head_dim, cfg.eps)
        self.k_norm = ops.RMSNorm(cfg.head_dim, cfg.eps)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ffn_gate = nn.Linear(d, f)
        self.ffn_up = nn.Linear(d, f)
        self.ffn_down = nn.Linear(f, d)

    def attention(self, h: mx.array, cos: mx.array,
                  sin: mx.array) -> mx.array:
        """The attention rows (B, L, dim) that ``proj`` takes."""
        b, n, _ = h.shape
        q, k, v = mx.split(self.qkv(h), 3, axis=-1)
        shape = (b, n, self.heads, self.head_dim)
        q, k = qk_rope(q.reshape(shape), k.reshape(shape), self.q_norm.weight,
                       self.k_norm.weight, cos, sin, self.eps)
        o = ops.sdpa(q, k, v.reshape(shape).transpose(0, 2, 1, 3))
        return o.transpose(0, 2, 1, 3).reshape(b, n, -1)

    def hidden(self, y: mx.array) -> mx.array:
        """The rows that ``ffn_down`` takes."""
        return situ(self.ffn_up(y), self.ffn_gate(y))

    def __call__(self, x: mx.array, cos: mx.array,
                 sin: mx.array) -> mx.array:
        x = x + self.proj(self.attention(self.norm1(x), cos, sin))
        return x + self.ffn_down(self.hidden(self.norm2(x)))


class TextProjection(nn.Module):
    def __init__(self, cfg: MossConfig):
        super().__init__()
        self.proj = nn.Linear(cfg.text_dim, cfg.dim)
        self.blocks = [TextBlock(cfg) for _ in range(cfg.text_depth)]


class Block(nn.Module):
    """A joint block: modulated pre-norm attention and feed-forward over
    the image and text rows, each added under its gate."""

    def __init__(self, cfg: MossConfig):
        super().__init__()
        d = cfg.dim
        self.heads, self.head_dim, self.eps = cfg.heads, cfg.head_dim, cfg.eps
        self.norm1 = ops.RMSNorm(d, cfg.eps)
        self.norm2 = ops.RMSNorm(d, cfg.eps)
        self.q_norm = ops.RMSNorm(cfg.head_dim, cfg.eps)
        self.k_norm = ops.RMSNorm(cfg.head_dim, cfg.eps)
        self.proj = {n: nn.Linear(d, d, bias=False)
                     for n in ("q", "k", "v", "o")}
        self.proj["ffn_up_gate"] = nn.Linear(d, 2 * cfg.ffn_dim, bias=False)
        self.proj["ffn_down"] = nn.Linear(cfg.ffn_dim, d, bias=False)

    def attention(self, h: mx.array, cos: mx.array,
                  sin: mx.array) -> mx.array:
        """The attention rows (B, L, dim) that ``proj.o`` takes."""
        b, n, _ = h.shape
        shape = (b, n, self.heads, self.head_dim)
        p = self.proj
        q, k = qk_rope(p["q"](h).reshape(shape), p["k"](h).reshape(shape),
                       self.q_norm.weight, self.k_norm.weight, cos, sin,
                       self.eps)
        o = ops.sdpa(q, k, p["v"](h).reshape(shape).transpose(0, 2, 1, 3))
        return o.transpose(0, 2, 1, 3).reshape(b, n, -1)

    def hidden(self, h: mx.array) -> mx.array:
        """``situ`` of the two halves of ``ffn_up_gate(h)``, the hidden
        half first: the rows that ``proj.ffn_down`` takes."""
        hid, gate = mx.split(self.proj["ffn_up_gate"](h), 2, axis=-1)
        return situ(hid, gate)

    def __call__(self, x: mx.array, mod: tuple, cos: mx.array,
                 sin: mx.array) -> mx.array:
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod
        p = self.proj
        h = ops.modulate(self.norm1(x), scale_a) + shift_a
        rows = self.attention(h, cos, sin)
        x = gated_projection(x, gate_a, p["o"], rows, lambda: p["o"](rows))
        h = ops.modulate(self.norm2(x), scale_f) + shift_f
        hid = self.hidden(h)
        return gated_projection(x, gate_f, p["ffn_down"], hid,
                                lambda: p["ffn_down"](hid))


@dataclass(frozen=True)
class StepModulation:
    """What one step derives from sigma: the six vectors every block
    reads, then the head's shift and scale, each (B, 1, dim) in the
    activation dtype."""
    blocks: tuple
    shift: mx.array
    scale: mx.array

    def arrays(self) -> list:
        return [*self.blocks, self.shift, self.scale]


@dataclass(frozen=True)
class Caption:
    """What one step derives from the conditioning: the text rows after
    ``text_embedder``, (B, L, dim)."""
    rows: mx.array

    def arrays(self) -> list:
        return [self.rows]


class MossImageDiT(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        cfg = MossConfig.from_dict(config)
        self.cfg = cfg
        self.act_dtype = mx.bfloat16
        d = cfg.dim
        self.video_embedder = nn.Linear(cfg.latent, d)
        self.text_embedder = TextProjection(cfg)
        self.rope = Rope(cfg.bands)
        self.time_embedding = [nn.Linear(cfg.freq_dim, d), nn.SiLU(),
                               nn.Linear(d, d)]
        self.shared_block_modulation = nn.Linear(d, 6 * d)
        self.layers = [Block(cfg) for _ in range(cfg.depth)]
        self.final_norm_video = ops.RMSNorm(d, cfg.eps)
        self.final_linear_video = nn.Linear(d, cfg.latent, bias=False)
        self.video_head_modulation = nn.Linear(d, 2 * d)
        self._bands = rope_bands(cfg.bands, cfg.rope_base)

    @property
    def act_dtype(self) -> mx.Dtype:
        return self._act_dtype

    @act_dtype.setter
    def act_dtype(self, dtype: mx.Dtype) -> None:
        if dtype == mx.float16:
            raise ImageDtypeError("MOSS-Image never runs in float16")
        self._act_dtype = dtype

    def timesteps(self, sigma, batch: int) -> np.ndarray:
        """The timesteps the release's scheduler passes for sigmas: sigma
        times its training timesteps, in float32."""
        s = np.broadcast_to(np.asarray(sigma, np.float32), (batch,))
        return s * np.float32(self.cfg.train_timesteps)

    def time_vectors(self, sigma, batch: int) -> tuple:
        """The timestep embedding (B, dim) and the modulation rows of the
        blocks (B, 6 dim) and of the head (B, 2 dim), in float32. Under
        bf16 activations each leaf runs as a float32 product by
        ``precise``; under fp32 each runs its own call."""
        lin = precise if self.act_dtype != mx.float32 else own
        t = time_sinusoid(self.timesteps(sigma, batch), self.cfg.freq_dim,
                          self.cfg.time_period)
        te = self.time_embedding
        temb = lin(te[2], nn.silu(lin(te[0], t)))
        return (temb, lin(self.shared_block_modulation, temb),
                lin(self.video_head_modulation, temb))

    def modulation(self, sigma, batch: int) -> StepModulation:
        _, blocks, head = self.time_vectors(sigma, batch)
        act = self.act_dtype

        def chunks(rows, n):
            return tuple(mx.split(rows.astype(act)[:, None, :], n, axis=-1))

        shift, scale = chunks(head, 2)
        return StepModulation(blocks=chunks(blocks, 6), shift=shift,
                              scale=scale)

    def text_rope(self, n_txt: int) -> tuple:
        """cos and sin tables of ``n_txt`` text rows alone."""
        return rope_tables(positions(0, 0, n_txt), self._bands)

    def caption(self, feats: mx.array, taps: dict | None = None) -> Caption:
        """The text rows of features (B, L, text dim)."""
        te = self.text_embedder
        x = te.proj(feats.astype(self.act_dtype))
        if taps is not None:
            taps["text_embedder.proj"] = x
        cos, sin = self.text_rope(x.shape[1])
        for i, blk in enumerate(te.blocks):
            x = blk(x, cos, sin)
            if taps is not None:
                taps[f"text_embedder.blocks.{i}"] = x
        return Caption(x)

    def rope_tables(self, h: int, w: int, n_txt: int) -> tuple:
        """cos and sin tables of the joint rows: the image grid, then the
        text."""
        return rope_tables(positions(h, w, n_txt), self._bands)

    def head(self, x: mx.array, shift: mx.array,
             scale: mx.array) -> mx.array:
        """The output head on image rows (B, N, dim)."""
        y = ops.modulate(self.final_norm_video(x), scale) + shift
        return self.final_linear_video(y)

    def __call__(self, latents: mx.array, sigma, feats: mx.array | None,
                 taps: dict | None = None, *,
                 mod: StepModulation | None = None,
                 caption: Caption | None = None) -> mx.array:
        """Velocity (B, C, H, W) fp32 for latents (B, C, H, W) at
        ``sigma``, with text features (B, L, text dim). ``mod`` and
        ``caption``, from ``modulation`` and ``caption``, stand in for
        the work they hold. ``taps`` collects the output of every block by
        module path."""
        b, c, h, w = latents.shape
        if mod is None:
            mod = self.modulation(sigma, b)
        if caption is None:
            if feats is None:
                raise ValueError("pass text features or their caption")
            caption = self.caption(feats, taps)
        txt = caption.rows
        if txt.shape[0] != b:
            txt = mx.broadcast_to(txt, (b, *txt.shape[1:]))
        tokens = latents.reshape(b, c, h * w).transpose(0, 2, 1)
        img = self.video_embedder(tokens.astype(self.act_dtype))
        if taps is not None:
            taps["video_embedder"] = img
        x = mx.concatenate([img, txt], axis=1)
        cos, sin = self.rope_tables(h, w, txt.shape[1])
        for i, blk in enumerate(self.layers):
            x = blk(x, mod.blocks, cos, sin)
            pacing.point(x)
            if taps is not None:
                taps[f"layers.{i}"] = x
        out = self.head(x[:, :h * w], mod.shift, mod.scale)
        if taps is not None:
            taps["final_linear_video"] = out
        return out.transpose(0, 2, 1).reshape(b, c, h, w).astype(mx.float32)
