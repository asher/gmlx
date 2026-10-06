# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright 2026 Qwen-Image Team and The HuggingFace Team
# Ported from diffusers' src/diffusers/models/transformers/transformer_qwenimage21.py
# (the single-stream block, the shared modulation, the t = 0 rows of the
# causal condition, the per-segment prefill and the cached decode) at
# e0abab83 and modified for gmlx (Apache-2.0; see licenses/diffusers-LICENSE).
"""Qwen-Image 2.1 DiT module tree and forward.

Attribute paths equal the GGUF tensor names, so numbered elements are plain
lists, and no tensor is a bias. The image MLP is one fused ``gate_up``
product whose output splits into the gate and the up halves, gate first, or
the ``gate_layer`` and ``proj`` pair, as the file stores it. Elementwise
and attention steps go through the ``gmlx.image.ops`` seams.

The joint sequence is the prefix, which is the text and the reference
image tokens, followed by the target tokens. No prefix token attends to
the target, and prefix tokens modulate from t = 0, so the prefix is a
function of the conditioning alone. The forward is therefore two
functions. ``prefill_prefix`` runs the prefix tokens through the blocks,
one attention call per segment, and keeps each block's K and V after RoPE.
``forward_target`` runs the target tokens at a sigma, with each block's
keys and values the cached prefix followed by the target's own, and no
mask. Every step is a ``forward_target``, the first included.

``modulation`` derives from sigma alone the four modulation vectors that
all blocks share and the final scale. Neither function keeps a cache; a
caller memoizes ``prefill_prefix`` by conditioning and ``modulation`` by
sigma. The model output is the flow velocity, with no sign flip.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from gmlx.image import ops
from gmlx.image.dtypes import ImageDtypeError
from gmlx.image.ops import RMSNorm

from . import layout


@dataclass(frozen=True)
class QwenImage21Config:
    dim: int
    n_heads: int
    head_dim: int
    n_layers: int
    in_channels: int
    out_channels: int
    patch: int
    context_dim: int
    ffn_dim: int
    t_freq_dim: int
    eps: float
    axes_dims: tuple
    rope_theta: float
    fused_mlp: bool
    train_timesteps: int

    @classmethod
    def from_dict(cls, c: dict) -> "QwenImage21Config":
        dim = c["inner_dim"]
        heads, head_dim = c["num_attention_heads"], c["attention_head_dim"]
        axes = tuple(int(a) for a in c["axes_dims_rope"])
        if c["patch_size"] != 1:
            raise ValueError("Qwen-Image 2.1 takes latents unpatched")
        if not c["causal_condition"]:
            raise ValueError("Qwen-Image 2.1 runs with the causal "
                             "condition, which the prefix cache rests on")
        if heads * head_dim != dim or c["modulation_dim"] != 4 * dim:
            raise ValueError("Qwen-Image 2.1 widths disagree: heads times "
                             "head dim is the width, and the modulation "
                             "is four times it")
        if len(axes) != 3 or sum(axes) != head_dim or any(a % 2
                                                           for a in axes):
            raise ValueError(f"RoPE axes {axes} must be three even dims "
                             f"that sum to the head dim {head_dim}")
        return cls(
            dim=dim, n_heads=heads, head_dim=head_dim,
            n_layers=c["num_layers"], in_channels=c["in_channels"],
            out_channels=c["out_channels"], patch=c["patch_size"],
            context_dim=c["context_in_dim"], ffn_dim=c["ffn_dim"],
            t_freq_dim=c["timestep_dim"], eps=c["eps"], axes_dims=axes,
            rope_theta=float(c["rope_theta"]),
            fused_mlp=c["mlp_layout"] == "fused",
            train_timesteps=int(c["num_train_timesteps"]))


def squash(gate: mx.array) -> mx.array:
    """The gate of a residual branch, as the block adds it."""
    return mx.tanh(gate)


def split_modulation(rows: mx.array) -> tuple:
    """scale1, gate1, scale2 and gate2 of the shared modulation rows."""
    return tuple(mx.split(rows, 4, axis=-1))


def split_gate_up(y: mx.array) -> tuple[mx.array, mx.array]:
    """The gate and the up halves of the fused MLP product, gate first."""
    gate, up = mx.split(y, 2, axis=-1)
    return gate, up


def causal_mask(start: int, end: int) -> mx.array:
    """Boolean (end - start, end): a text segment sees every key before
    it and a causal triangle of its own."""
    q = mx.arange(start, end)[:, None]
    return q >= mx.arange(end)[None, :]


def prefix_attention(q: mx.array, k: mx.array, v: mx.array,
                     segments: tuple) -> mx.array:
    """Attention of the prefix rows, one call per segment. A segment's
    keys are everything up to its end. A text segment is causal inside
    itself, and an image block attends fully inside itself."""
    outs = []
    for start, end, text in segments:
        mask = causal_mask(start, end) if text else None
        outs.append(ops.sdpa(q[:, :, start:end], k[:, :, :end],
                             v[:, :, :end], mask))
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)


def join_prefix(prefix_k: mx.array, prefix_v: mx.array, k: mx.array,
                v: mx.array) -> tuple[mx.array, mx.array]:
    """Keys and values of a target step: the cached prefix, then the
    target's own."""
    b = k.shape[0]
    if prefix_k.shape[0] != b:
        prefix_k = mx.broadcast_to(prefix_k, (b, *prefix_k.shape[1:]))
        prefix_v = mx.broadcast_to(prefix_v, (b, *prefix_v.shape[1:]))
    return (mx.concatenate([prefix_k, k], axis=2),
            mx.concatenate([prefix_v, v], axis=2))


def target_attention(q: mx.array, k: mx.array, v: mx.array,
                     prefix_k: mx.array, prefix_v: mx.array) -> mx.array:
    """Attention of the target rows over the prefix and themselves, with
    no mask. This is the one place the two segments join, so a kernel
    that takes them apart replaces this function."""
    k, v = join_prefix(prefix_k, prefix_v, k, v)
    return ops.sdpa(q, k, v)


class ZeroCenteredRMSNorm(nn.Module):
    """RMSNorm whose stored weight is the scale minus one. The norm runs
    in fp32 whatever the input and weight dtypes are."""

    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.zeros((dims,))
        self.eps = eps

    def scale(self) -> mx.array:
        return self.weight.astype(mx.float32) + 1

    def __call__(self, x: mx.array) -> mx.array:
        y = mx.fast.rms_norm(x.astype(mx.float32), self.scale(), self.eps)
        return y.astype(x.dtype)


class TextProjection(nn.Module):
    def __init__(self, cfg: QwenImage21Config):
        super().__init__()
        self.text_norm = ZeroCenteredRMSNorm(cfg.context_dim, cfg.eps)
        self.in_layer = nn.Linear(cfg.context_dim, cfg.dim, bias=False)
        self.out_layer = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.out_layer(nn.gelu_approx(self.in_layer(
            self.text_norm(x))))


class TimestepEmbedder(nn.Module):
    def __init__(self, cfg: QwenImage21Config):
        super().__init__()
        self.linear_1 = nn.Linear(cfg.t_freq_dim, cfg.dim, bias=False)
        self.linear_2 = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def __call__(self, freq: mx.array) -> mx.array:
        return self.linear_2(nn.silu(self.linear_1(freq)))


class TimeTextEmbed(nn.Module):
    def __init__(self, cfg: QwenImage21Config):
        super().__init__()
        self.timestep_embedder = TimestepEmbedder(cfg)


class Attention(nn.Module):
    def __init__(self, cfg: QwenImage21Config):
        super().__init__()
        self.to_q = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.to_k = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.to_v = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.to_out = [nn.Linear(cfg.dim, cfg.dim, bias=False)]
        self.norm_q = RMSNorm(cfg.head_dim, cfg.eps)
        self.norm_k = RMSNorm(cfg.head_dim, cfg.eps)
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.eps = cfg.eps

    def qkv(self, x: mx.array, cos: mx.array, sin: mx.array) -> tuple:
        """q and k after their norm and RoPE, and v, each (B, H, L, D)."""
        b, n, _ = x.shape
        shape = (b, n, self.n_heads, self.head_dim)
        q, k = ops.qk_norm_rope(
            self.to_q(x).reshape(shape), self.to_k(x).reshape(shape),
            self.norm_q.weight, self.norm_k.weight, cos, sin, self.eps)
        return q, k, self.to_v(x).reshape(shape).transpose(0, 2, 1, 3)

    def out(self, o: mx.array) -> mx.array:
        b, _, n, _ = o.shape
        return self.to_out[0](o.transpose(0, 2, 1, 3).reshape(b, n, -1))


class ImageMLP(nn.Module):
    def __init__(self, cfg: QwenImage21Config):
        super().__init__()
        if cfg.fused_mlp:
            self.gate_up = nn.Linear(cfg.dim, 2 * cfg.ffn_dim, bias=False)
        else:
            self.gate_layer = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
            self.proj = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.out = nn.Linear(cfg.ffn_dim, cfg.dim, bias=False)
        self.fused = cfg.fused_mlp

    def __call__(self, x: mx.array) -> mx.array:
        if self.fused:
            gate, up = split_gate_up(self.gate_up(x))
        else:
            gate, up = self.gate_layer(x), self.proj(x)
        return self.out(ops.swiglu(gate, up))


class Block(nn.Module):
    """Single-stream block. ``prefill`` runs prefix rows and returns their
    K and V, and ``target`` runs target rows over a cached prefix."""

    def __init__(self, cfg: QwenImage21Config):
        super().__init__()
        self.attn = Attention(cfg)
        self.img_mlp = ImageMLP(cfg)
        self.eps = cfg.eps

    def _update(self, x: mx.array, mod: "Modulation",
                attn: mx.array) -> mx.array:
        x = ops.gated_residual(x, mod.gate1, attn)
        h = ops.modulate(ops.layer_norm(x, self.eps), mod.scale2)
        return ops.gated_residual(x, mod.gate2, self.img_mlp(h))

    def _qkv(self, x: mx.array, mod: "Modulation", cos: mx.array,
             sin: mx.array) -> tuple:
        h = ops.modulate(ops.layer_norm(x, self.eps), mod.scale1)
        return self.attn.qkv(h, cos, sin)

    def prefill(self, x: mx.array, mod: "Modulation", cos: mx.array,
                sin: mx.array, segments: tuple) -> tuple:
        """The prefix rows after the block, and their K and V."""
        q, k, v = self._qkv(x, mod, cos, sin)
        a = self.attn.out(prefix_attention(q, k, v, segments))
        return self._update(x, mod, a), k, v

    def target(self, x: mx.array, mod: "Modulation", cos: mx.array,
               sin: mx.array, prefix_k: mx.array,
               prefix_v: mx.array) -> mx.array:
        q, k, v = self._qkv(x, mod, cos, sin)
        a = self.attn.out(target_attention(q, k, v, prefix_k, prefix_v))
        return self._update(x, mod, a)


class NormOut(nn.Module):
    """The final adaptive norm, which has a scale and no shift."""

    def __init__(self, cfg: QwenImage21Config):
        super().__init__()
        self.linear = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.eps = cfg.eps

    def scale(self, temb: mx.array) -> mx.array:
        h = nn.silu(temb)
        w = self.linear.weight
        if mx.issubdtype(w.dtype, mx.floating):
            h = h.astype(w.dtype)
        return self.linear(h).astype(temb.dtype)

    def __call__(self, x: mx.array, scale: mx.array) -> mx.array:
        return ops.modulate(ops.layer_norm(x, self.eps), scale)


@dataclass(frozen=True)
class Modulation:
    """What a step derives from a timestep: the two scales and the two
    squashed gates that every block shares, and the final scale, each
    (B, 1, dim)."""
    scale1: mx.array
    gate1: mx.array
    scale2: mx.array
    gate2: mx.array
    final: mx.array

    def arrays(self) -> list:
        return [self.scale1, self.gate1, self.scale2, self.gate2,
                self.final]


@dataclass(frozen=True)
class PrefixCache:
    """What a step derives from the conditioning: each block's K and V of
    the prefix after RoPE, (1, heads, P, head dim), and the position on
    the frame axis where the target block starts. ``built_for`` is the
    ``prefix_tag`` of that conditioning."""
    k: tuple
    v: tuple
    position: int
    built_for: tuple

    def arrays(self) -> list:
        return [*self.k, *self.v]

    @property
    def tokens(self) -> int:
        return int(self.k[0].shape[2])


class QwenImage21DiT(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        cfg = QwenImage21Config.from_dict(config)
        self.cfg = cfg
        self.act_dtype = mx.bfloat16
        self.img_in = nn.Linear(cfg.in_channels, cfg.dim, bias=False)
        self.txt_in = TextProjection(cfg)
        self.time_text_embed = TimeTextEmbed(cfg)
        self.modulation = [nn.SiLU(),
                           nn.Linear(cfg.dim, 4 * cfg.dim, bias=False)]
        self.transformer_blocks = [Block(cfg) for _ in range(cfg.n_layers)]
        self.norm_out = NormOut(cfg)
        self.proj_out = nn.Linear(cfg.dim, cfg.out_channels, bias=False)

    @property
    def act_dtype(self) -> mx.Dtype:
        return self._act_dtype

    @act_dtype.setter
    def act_dtype(self, dtype: mx.Dtype) -> None:
        if dtype == mx.float16:
            raise ImageDtypeError("Qwen-Image 2.1 never runs in float16")
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
        """The timestep embedding (B, dim) of embedder inputs ``t``."""
        freq = ops.timestep_embedding(t, self.cfg.t_freq_dim)
        return self.time_text_embed.timestep_embedder(
            freq.astype(self.act_dtype))

    def modulation_rows(self, temb: mx.array) -> mx.array:
        """The shared modulation (B, 4 dim) of timestep embeddings."""
        return self.modulation[1](self.modulation[0](temb))

    def modulation_at(self, t) -> Modulation:
        temb = self.temb(t)
        rows = self.modulation_rows(temb)[:, None, :]
        scale1, gate1, scale2, gate2 = split_modulation(rows)
        return Modulation(scale1, squash(gate1), scale2, squash(gate2),
                          self.norm_out.scale(temb)[:, None, :])

    def pack(self, latents: mx.array) -> mx.array:
        """Latents (B, C, H, W) as tokens (B, H * W, C), row major."""
        b, c, h, w = latents.shape
        return latents.reshape(b, c, h * w).transpose(0, 2, 1)

    def unpack(self, x: mx.array, h: int, w: int) -> mx.array:
        b, _, c = x.shape
        return x.transpose(0, 2, 1).reshape(b, c, h, w)

    def embed_image(self, latents: mx.array) -> mx.array:
        return self.img_in(self.pack(latents).astype(self.act_dtype))

    def embed_text(self, features: mx.array) -> mx.array:
        return self.txt_in(features.astype(self.act_dtype))

    def embed_prefix(self, cond, lay: layout.PrefixLayout) -> mx.array:
        """The prefix rows (1, P, dim) before the first block: the text
        rows through ``txt_in`` and the reference tokens through
        ``img_in``, in joint order."""
        rows = []
        if lay.text_rows.size:
            rows.append(self.embed_text(
                cond.features[mx.array(lay.text_rows)]))
        for ref in cond.ref_latents:
            if ref.ndim != 3 or ref.shape[0] != self.cfg.in_channels:
                raise ValueError(
                    f"reference latents must be ({self.cfg.in_channels}, "
                    f"h, w), not {tuple(ref.shape)}")
            rows.append(self.embed_image(ref[None])[0])
        x = rows[0] if len(rows) == 1 else mx.concatenate(rows, axis=0)
        return x[mx.array(lay.source)][None]

    def final(self, x: mx.array, mod: Modulation) -> mx.array:
        return self.proj_out(self.norm_out(x, mod.final))


def prefix_key(cond) -> tuple:
    """What the prefix layout of a conditioning depends on: the count of
    its feature rows, its image slots and the shapes of its reference
    latents."""
    refs = tuple((int(r.shape[-2]), int(r.shape[-1]))
                 for r in cond.ref_latents)
    return (int(cond.features.shape[0]),
            tuple(int(s) for s in cond.image_slots), refs)


def prefix_tag(cond) -> tuple:
    """What names the conditioning of a prefix in a target step: its key
    and what its prefix layout depends on."""
    return (cond.key, *prefix_key(cond))


def prefix_bytes(model: QwenImage21DiT, cond) -> int:
    """Bytes of the ``PrefixCache`` of ``cond`` in the model's activation
    dtype, from its layout alone."""
    cfg = model.cfg
    tokens = layout.prefix_layout(*prefix_key(cond)).tokens
    return 2 * cfg.n_layers * tokens * cfg.dim * model.act_dtype.size


def modulation(model: QwenImage21DiT, sigma) -> Modulation:
    """The modulation of target rows at ``sigma``, one value or one per
    row."""
    s = np.asarray(sigma, np.float32).reshape(-1)
    return model.modulation_at(model.timesteps(s, s.size))


def prefix_modulation(model: QwenImage21DiT) -> Modulation:
    """The modulation of every prefix row, which is that of t = 0."""
    return model.modulation_at(np.zeros(1, np.float32))


def prefill_prefix(model: QwenImage21DiT, cond,
                   taps: dict | None = None) -> PrefixCache:
    """Run the prefix of ``cond`` through the blocks and return each
    block's K and V. ``cond.features`` is the encoder sequence (L, D).
    With reference images, ``cond.image_slots`` names its image rows and
    ``cond.ref_latents`` holds the latents that replace them. Each block
    is evaluated as it finishes, so one evaluation holds the work of one
    block. ``taps`` collects the prefix rows after every block."""
    cfg = model.cfg
    key = prefix_key(cond)
    lay = layout.prefix_layout(*key)
    if not lay.tokens:
        raise ValueError("the conditioning has no tokens")
    cos, sin = layout.prefix_tables(cfg.axes_dims, cfg.rope_theta, *key)
    mod = prefix_modulation(model)
    x = model.embed_prefix(cond, lay)
    keys, values = [], []
    for i, blk in enumerate(model.transformer_blocks):
        x, k, v = blk.prefill(x, mod, cos, sin, lay.segments)
        mx.eval(x, k, v)
        keys.append(k)
        values.append(v)
        if taps is not None:
            taps[f"transformer_blocks.{i}"] = x
    return PrefixCache(tuple(keys), tuple(values), lay.position,
                       (cond.key, *key))


def forward_target(model: QwenImage21DiT, latents: mx.array, sigma, cond,
                   prefix: PrefixCache, taps: dict | None = None, *,
                   mod: Modulation | None = None) -> mx.array:
    """Velocity (B, C, H, W) fp32 for fp32 target latents at ``sigma``
    over the prefix of ``cond``. ``prefix`` holds all of ``cond`` that the
    target reads, and one built for another conditioning is refused.
    ``mod``, from ``modulation``, stands in for the work it holds.
    ``taps`` collects the target rows after every block."""
    cfg = model.cfg
    b, _, h, w = latents.shape
    if prefix.built_for != prefix_tag(cond):
        raise ValueError("the prefix was built for another conditioning")
    if len(prefix.k) != cfg.n_layers:
        raise ValueError(f"the prefix holds {len(prefix.k)} blocks, and "
                         f"the model has {cfg.n_layers}")
    if mod is None:
        mod = modulation(model, np.broadcast_to(
            np.asarray(sigma, np.float32).reshape(-1), (b,)))
    cos, sin = layout.target_tables(cfg.axes_dims, cfg.rope_theta,
                                    prefix.position, h, w)
    x = model.embed_image(latents)
    for i, blk in enumerate(model.transformer_blocks):
        x = blk.target(x, mod, cos, sin, prefix.k[i], prefix.v[i])
        if taps is not None:
            taps[f"transformer_blocks.{i}"] = x
    out = model.final(x, mod)
    if taps is not None:
        taps["proj_out"] = out
    return model.unpack(out, h, w).astype(mx.float32)
