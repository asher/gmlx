"""MiniMax-H3's DiT layer, shared by video and image: the blocks, the
token refiner, and what a DiT runs beside them (``H3Core``).

Each block modulates its two pre-norms and gates its two residuals per
segment, with one modality row at the segment's time. Attention norms q
and k per head and rotates the first ``rot`` dims of each head as split
halves (``gmlx.image.h3.rope``). With ``attention`` ``"int8"`` a block
runs its attention as mlx-kquant's ``sdpa_qk8``: int8 Q K^T and bfloat16
P V. The token refiner runs plain blocks over the text.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Self

import mlx.core as mx
import mlx.nn as nn
import mlx_kquant as kq
import numpy as np

from gmlx import mlx_guards
from gmlx.image import ops, pacing
from gmlx.image.h3 import rope
from gmlx.image.int8 import ATTENTION

_F32 = mx.float32
# Rows of one fc1 product on the stock path, which keeps its 2 x ffn wide
# output under 0.5 GiB in bf16.
MLP_ROWS = 8192


class H3DtypeError(ValueError):
    """An activation dtype H3 does not run in."""


@dataclass(frozen=True)
class H3BlockConfig:
    """H3's config less the audio stream and VSA: what the blocks read."""
    dim: int
    heads: int
    head_dim: int
    ffn: int
    depth: int
    refiner_depth: int
    latent: int
    patch: tuple
    text_dim: int
    freqs: int
    eps: float
    qk_eps: float
    final_eps: float
    grid: int
    basis: int

    @property
    def inner(self) -> int:
        return self.heads * self.head_dim

    @property
    def patch_dim(self) -> int:
        return self.latent * math.prod(self.patch)

    @property
    def rot(self) -> int:
        """Rotated dims per head: a pair per frequency of each of the
        three axes."""
        return 2 * 3 * self.freqs

    @classmethod
    def from_dict(cls, c: dict, **extra) -> Self:
        return cls(
            dim=c["hidden_size"], heads=c["num_attention_heads"],
            head_dim=c["attention_head_dim"], ffn=c["ffn_dim"],
            depth=c["num_layers"], refiner_depth=c["num_refiner_layers"],
            latent=c["in_channels"], patch=tuple(c["patch_size"]),
            text_dim=c["text_dim"], freqs=c["rope_freq_dim"],
            eps=c["norm_eps"], qk_eps=c["qk_norm_eps"],
            final_eps=c["final_norm_eps"], grid=c["adaln_grid"],
            basis=c["adaln_basis"], **extra)


def _lerp(a: mx.array, b: mx.array, w: mx.array) -> mx.array:
    """torch.lerp's form: from the nearer end of the two."""
    d = b - a
    return mx.where(w < 0.5, a + w * d, b - d * (1 - w))


def modulated(n: mx.array, shift: mx.array, scale: mx.array,
              segs: tuple) -> mx.array:
    """``n * (1 + scale) + shift`` with each segment's modulation row."""
    parts = [ops.modulate(n[:, a:b], scale[r]) + shift[r]
             for a, b, r in segs]
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)


def gated(x: mx.array, gate: mx.array, y: mx.array, segs: tuple) -> mx.array:
    """``x + gate * y`` with each segment's gate row, rounded once, as the
    reference's addcmul_ rounds. A block can write an update of 2e4 over a
    residual of 40, which a rounded product drops in bfloat16."""
    parts = [ops.gated_residual_once(x[:, a:b], gate[r], y[:, a:b])
             for a, b, r in segs]
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)


def attend(q: mx.array, k: mx.array, v: mx.array,
           chunk: int | None) -> mx.array:
    """sdpa over (B, H, L, D), ``chunk`` query rows a call, with a pause
    point after each. Past 2^31 elements the stock concatenate misplaces
    rows on the GPU, so the chunks join through the guard."""
    n = q.shape[2]
    if not chunk or n <= chunk:
        return ops.sdpa(q, k, v)
    parts = []
    for a in range(0, n, chunk):
        parts.append(ops.sdpa(q[:, :, a:a + chunk], k, v))
        pacing.point(parts[-1])
    return mlx_guards.concatenate(parts, axis=2)


def _kq(name: str):
    """mlx-kquant's op ``name``. The loader refuses int8 attention on a
    build without ``sdpa_qk8`` before any call."""
    return getattr(kq, name)


def attend_int8(q: mx.array, k: mx.array, v: mx.array,
                chunk: int | None) -> mx.array:
    """``attend`` with int8 Q K^T and bfloat16 P V. q and k are quantized
    once a call, with a pause point after; then each chunk of query rows
    reads its rows of the int8 q and their scales, with a pause point
    after each, and the chunks join through the guard."""
    q8, sq, k8, sk = _kq("sdpa_qk8_inputs")(q, k, q.shape[-1] ** -0.5)
    pacing.point(q8, sq, k8, sk)
    attn = _kq("sdpa_qk8")
    n = q.shape[2]
    if not chunk or n <= chunk:
        return attn(q8, sq, k8, sk, v)
    parts = []
    for a in range(0, n, chunk):
        parts.append(attn(q8[:, :, a:a + chunk], sq[:, :, a:a + chunk], k8,
                          sk, v))
        pacing.point(parts[-1])
    return mlx_guards.concatenate(parts, axis=2)


class Attention(nn.Module):
    def __init__(self, cfg: H3BlockConfig):
        super().__init__()
        self.qkv_proj = nn.Linear(cfg.dim, 3 * cfg.inner, bias=False)
        self.q_norm = ops.RMSNorm(cfg.head_dim, cfg.qk_eps)
        self.k_norm = ops.RMSNorm(cfg.head_dim, cfg.qk_eps)
        self.out_proj = nn.Linear(cfg.inner, cfg.dim, bias=False)


class MLP(nn.Module):
    def __init__(self, cfg: H3BlockConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.dim, 2 * cfg.ffn, bias=False)
        self.fc2 = nn.Linear(cfg.ffn, cfg.dim, bias=False)


class AdalnProj(nn.Module):
    def __init__(self, basis: int, width: int):
        super().__init__()
        self.linear = nn.Linear(basis, width)


class _Block(nn.Module):
    def __init__(self, cfg: H3BlockConfig):
        super().__init__()
        self.heads, self.head_dim = cfg.heads, cfg.head_dim
        self.norm1 = ops.RMSNorm(cfg.dim, cfg.eps)
        self.norm2 = ops.RMSNorm(cfg.dim, cfg.eps)
        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)

    def qkv(self, h: mx.array) -> tuple:
        """q, k and v (B, L, H, D) from the fused [q; k; v] product."""
        b, n, _ = h.shape
        shape = (b, n, self.heads, self.head_dim)
        q, k, v = mx.split(self.attn.qkv_proj(h), 3, axis=-1)
        return q.reshape(shape), k.reshape(shape), v.reshape(shape)

    def hidden(self, h: mx.array) -> mx.array:
        """``silu(a) * b`` of the halves a, b of ``fc1(h)``: the rows that
        ``fc2`` takes. The stock path runs ``MLP_ROWS`` rows a product,
        with a pause point after each, so a paced forward holds one
        product at a time."""
        fc1 = self.mlp.fc1
        pair = ops.glu_halves(fc1, h)
        if pair is not None:
            return ops.swiglu_fused(h, *pair)
        n = h.shape[1]
        out = []
        for i in range(0, n, MLP_ROWS):
            a, b = mx.split(fc1(h[:, i:i + MLP_ROWS]), 2, axis=-1)
            out.append(ops.swiglu(a, b))
            pacing.point(out[-1])
        return out[0] if len(out) == 1 else mx.concatenate(out, axis=1)


class RefinerBlock(_Block):
    """Pre-norm attention and feed-forward with plain residuals, no RoPE
    and no modulation."""

    def attention(self, h: mx.array) -> mx.array:
        b, n, _ = h.shape
        q, k, v = self.qkv(h)
        q = self.attn.q_norm(q).transpose(0, 2, 1, 3)
        k = self.attn.k_norm(k).transpose(0, 2, 1, 3)
        o = ops.sdpa(q, k, v.transpose(0, 2, 1, 3))
        return o.transpose(0, 2, 1, 3).reshape(b, n, -1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.attn.out_proj(self.attention(self.norm1(x))) + x
        return self.mlp.fc2(self.hidden(self.norm2(x))) + x


class TokenRefiner(nn.Module):
    def __init__(self, cfg: H3BlockConfig):
        super().__init__()
        self.blocks = [RefinerBlock(cfg) for _ in range(cfg.refiner_depth)]
        self.final_norm = ops.RMSNorm(cfg.dim, cfg.final_eps)


class DiTBlock(_Block):
    def __init__(self, cfg: H3BlockConfig):
        super().__init__(cfg)
        self.dim, self.rot = cfg.dim, cfg.rot
        self.adaln_proj = AdalnProj(cfg.basis, 6 * 3 * cfg.dim)

    def modulation(self, t_emb: mx.array, dtype: mx.Dtype) -> tuple:
        """The block's six vectors (shift, scale and gate of attention,
        then of the feed-forward), each (3 x times, dim): row ``3 i +
        tag`` is time i under modality tag ``tag``."""
        y = self.adaln_proj.linear(t_emb).reshape(-1, 6, self.dim)
        return tuple(y[:, i].astype(dtype) for i in range(6))

    def rotate(self, q: mx.array, k: mx.array, cos: mx.array,
               sin: mx.array) -> tuple:
        """q and k (B, L, H, D) after the qk norms and RoPE, (B, H, L,
        D)."""
        a, r = self.attn, self.rot
        return ops.qk_norm_rope(
            rope.pair_halves(q, r), rope.pair_halves(k, r),
            rope.pair_halves(a.q_norm.weight, r),
            rope.pair_halves(a.k_norm.weight, r), cos, sin, a.q_norm.eps)

    def attention(self, h: mx.array, cos: mx.array, sin: mx.array,
                  chunk: int | None = None,
                  attention: str = "dense") -> mx.array:
        """The attention rows (B, L, inner) that ``out_proj`` takes. A
        pause point after qkv keeps a LoRA delta's second product out of
        the RoPE step's peak. ``attention`` ``"int8"`` runs
        ``attend_int8`` in place of ``attend``."""
        b, n, _ = h.shape
        q, k, v = self.qkv(h)
        pacing.point(q, k, v)
        q, k = self.rotate(q, k, cos, sin)
        run = attend_int8 if attention == "int8" else attend
        o = run(q, k, v.transpose(0, 2, 1, 3), chunk)
        return o.transpose(0, 2, 1, 3).reshape(b, n, -1)

    def __call__(self, x: mx.array, mod: tuple, segs: tuple, cos: mx.array,
                 sin: mx.array, **kw) -> mx.array:
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod
        h = modulated(self.norm1(x), shift_a, scale_a, segs)
        rows = self.attention(h, cos, sin, **kw)
        x = gated(x, gate_a, self.attn.out_proj(rows), segs)
        h = modulated(self.norm2(x), shift_f, scale_f, segs)
        return gated(x, gate_f, self.mlp.fc2(self.hidden(h)), segs)


class Rope(nn.Module):
    def __init__(self, freqs: int):
        super().__init__()
        self.inv_freq = mx.zeros((freqs,))


class H3Core(nn.Module):
    """What an H3 DiT runs beside its blocks: its activation dtype, its
    attention and linears, the text rows and the curve rows. A subclass
    holds ``title``, the refiner, ``condition_proj`` and the curve
    table."""

    title: str

    @property
    def act_dtype(self) -> mx.Dtype:
        return self._act_dtype

    @act_dtype.setter
    def act_dtype(self, dtype: mx.Dtype) -> None:
        if dtype not in (mx.bfloat16, _F32):
            raise H3DtypeError(f"{self.title} runs in bfloat16 or float32, "
                               f"not {dtype}")
        self._act_dtype = dtype

    @property
    def attention(self) -> str:
        """How the DiT blocks run dense attention: ``"dense"``, or
        ``"int8"`` for ``attend_int8``."""
        return self._attention

    @attention.setter
    def attention(self, kind: str) -> None:
        if kind not in ATTENTION:
            raise ValueError(f"{self.title} runs {' or '.join(ATTENTION)} "
                             f"attention, not {kind!r}")
        self._attention = kind

    @property
    def linears(self) -> str:
        """The form of the block linears: ``"file"`` as the GGUF stores
        them, or ``"int8"`` once the loader converted them
        (``gmlx.image.ops.int8_linear``)."""
        return self._linears

    def convert_linears(self, patterns: dict) -> int:
        """Convert the block linears that ``patterns`` names to int8
        (``gmlx.image.ops.int8_linear.convert``). Returns the count."""
        from gmlx.image.ops import int8_linear

        n = int8_linear.convert(self, patterns)
        self._linears = "int8"
        return n

    def caption(self, feats: mx.array, taps: dict | None = None) -> mx.array:
        """The text rows (1, L, dim) of encoder features (1, L, text dim):
        the condition projection, the refiner blocks and its final norm."""
        x = self.condition_proj(feats.astype(self.act_dtype))
        if taps is not None:
            taps["condition_proj"] = x
        tr = self.token_refiner
        for i, blk in enumerate(tr.blocks):
            x = blk(x)
            if taps is not None:
                taps[f"token_refiner.blocks.{i}"] = x
        x = tr.final_norm(x)
        if taps is not None:
            taps["token_refiner.final_norm"] = x
        return x

    def curve(self, times) -> mx.array:
        """(times, basis) float32: each time's blend of two table rows."""
        tab = self.adaln_t_table.astype(_F32)
        g = tab.shape[0] - 1
        pos = mx.clip(mx.array(np.asarray(times, np.float32)), 0.0, 1.0) * g
        i0 = mx.minimum(mx.floor(pos), g - 1)
        w = (pos - i0)[:, None]
        i0 = i0.astype(mx.int32)
        return _lerp(tab[i0], tab[i0 + 1], w)
