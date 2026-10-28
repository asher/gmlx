"""MiniMax-H3 DiT: single-stream blocks over one packed sequence.

The sequence holds text, anchor, audio and video rows (``layout.pack``).
Each block modulates its two pre-norms and gates its two residuals per
segment, with one of three modality rows (video, text, audio) at the
segment's time. A time t in [0, 1] reads two of the curve table's rows
and blends them, and each adaLN linear maps that blend to the rows of all
three tags, in float32. Attention norms q and k per head and rotates the
first 96 dims of each head as split halves (``gmlx.video.rope``). The
token refiner runs plain blocks over the text once per generation. The
model returns the flow velocity, noise minus data, which is the negated
output of its heads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from gmlx.image import ops, pacing
from gmlx.video import rope
from gmlx.video.anchors import pinned
from gmlx.video.layout import Layout

from . import family as fam

_F32 = mx.float32
# Rows of one fc1 product on the stock path, which keeps its 2 x ffn wide
# output under 0.5 GiB in bf16.
MLP_ROWS = 8192


class H3DtypeError(ValueError):
    """An activation dtype H3 does not run in."""


@dataclass(frozen=True)
class H3Config:
    dim: int
    heads: int
    head_dim: int
    ffn: int
    depth: int
    refiner_depth: int
    latent: int
    audio: int
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
    def from_dict(cls, c: dict) -> "H3Config":
        return cls(
            dim=c["hidden_size"], heads=c["num_attention_heads"],
            head_dim=c["attention_head_dim"], ffn=c["ffn_dim"],
            depth=c["num_layers"], refiner_depth=c["num_refiner_layers"],
            latent=c["in_channels"], audio=c["audio_in_channels"],
            patch=tuple(c["patch_size"]), text_dim=c["text_dim"],
            freqs=c["rope_freq_dim"], eps=c["norm_eps"],
            qk_eps=c["qk_norm_eps"], final_eps=c["final_norm_eps"],
            grid=c["adaln_grid"], basis=c["adaln_basis"])


@dataclass(frozen=True)
class StepPlan:
    """What a step derives from its stream times and its layout: the
    distinct times ascending, each segment's (start, stop, modulation
    row), the time index of each head, and the RoPE tables."""
    times: tuple
    segs: tuple
    heads: dict
    cos: mx.array
    sin: mx.array


def patchify(latent: mx.array, patch: tuple) -> mx.array:
    """(1, C, T, H, W) to rows (t, h, w) of features (c, pt, ph, pw)."""
    _, c, t, h, w = latent.shape
    pt, ph, pw = patch
    x = latent[0].reshape(c, t // pt, pt, h // ph, ph, w // pw, pw)
    x = x.transpose(1, 3, 5, 0, 2, 4, 6)
    return x.reshape(-1, c * pt * ph * pw)


def unpatchify(rows: mx.array, shape: tuple, patch: tuple) -> mx.array:
    """Rows of ``patchify`` back to (1, C, T, H, W) for ``shape`` (C, T,
    H, W)."""
    c, t, h, w = shape
    pt, ph, pw = patch
    x = rows.reshape(t // pt, h // ph, w // pw, c, pt, ph, pw)
    x = x.transpose(3, 0, 4, 1, 5, 2, 6)
    return x.reshape(1, c, t, h, w)


def pack_audio(latent: mx.array) -> mx.array:
    """(1, C, 2, T) to rows channel-major: all of channel 0, then 1."""
    _, c, ch, t = latent.shape
    return latent[0].transpose(1, 2, 0).reshape(ch * t, c)


def unpack_audio(rows: mx.array, ch: int = 2) -> mx.array:
    t = rows.shape[0] // ch
    return rows.reshape(ch, t, rows.shape[-1]).transpose(2, 0, 1)[None]


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
    """``x + gate * y`` with each segment's gate row."""
    parts = [ops.gated_residual(x[:, a:b], gate[r], y[:, a:b])
             for a, b, r in segs]
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)


def attend(q: mx.array, k: mx.array, v: mx.array,
           chunk: int | None) -> mx.array:
    """sdpa over (B, H, L, D), ``chunk`` query rows a call, with a pause
    point after each."""
    n = q.shape[2]
    if not chunk or n <= chunk:
        return ops.sdpa(q, k, v)
    parts = []
    for a in range(0, n, chunk):
        parts.append(ops.sdpa(q[:, :, a:a + chunk], k, v))
        pacing.point(parts[-1])
    return mx.concatenate(parts, axis=2)


class Attention(nn.Module):
    def __init__(self, cfg: H3Config):
        super().__init__()
        self.qkv_proj = nn.Linear(cfg.dim, 3 * cfg.inner, bias=False)
        self.q_norm = ops.RMSNorm(cfg.head_dim, cfg.qk_eps)
        self.k_norm = ops.RMSNorm(cfg.head_dim, cfg.qk_eps)
        self.out_proj = nn.Linear(cfg.inner, cfg.dim, bias=False)


class MLP(nn.Module):
    def __init__(self, cfg: H3Config):
        super().__init__()
        self.fc1 = nn.Linear(cfg.dim, 2 * cfg.ffn, bias=False)
        self.fc2 = nn.Linear(cfg.ffn, cfg.dim, bias=False)


class AdalnProj(nn.Module):
    def __init__(self, basis: int, width: int):
        super().__init__()
        self.linear = nn.Linear(basis, width)


class _Block(nn.Module):
    def __init__(self, cfg: H3Config):
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
        ``fc2`` takes. The stock path runs ``MLP_ROWS`` rows a product."""
        fc1 = self.mlp.fc1
        pair = ops.glu_halves(fc1, h)
        if pair is not None:
            return ops.swiglu_fused(h, *pair)
        n = h.shape[1]
        out = []
        for i in range(0, n, MLP_ROWS):
            a, b = mx.split(fc1(h[:, i:i + MLP_ROWS]), 2, axis=-1)
            out.append(ops.swiglu(a, b))
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
    def __init__(self, cfg: H3Config):
        super().__init__()
        self.blocks = [RefinerBlock(cfg) for _ in range(cfg.refiner_depth)]
        self.final_norm = ops.RMSNorm(cfg.dim, cfg.final_eps)


class DiTBlock(_Block):
    def __init__(self, cfg: H3Config):
        super().__init__(cfg)
        self.dim, self.rot = cfg.dim, cfg.rot
        self.adaln_proj = AdalnProj(cfg.basis, 6 * 3 * cfg.dim)

    def modulation(self, t_emb: mx.array, dtype: mx.Dtype) -> tuple:
        """The block's six vectors (shift, scale and gate of attention,
        then of the feed-forward), each (3 x times, dim): row ``3 i +
        tag`` is time i under modality tag ``tag``."""
        y = self.adaln_proj.linear(t_emb).reshape(-1, 6, self.dim)
        return tuple(y[:, i].astype(dtype) for i in range(6))

    def attention(self, h: mx.array, cos: mx.array, sin: mx.array,
                  chunk: int | None = None) -> mx.array:
        """The attention rows (B, L, inner) that ``out_proj`` takes."""
        b, n, _ = h.shape
        q, k, v = self.qkv(h)
        a, r = self.attn, self.rot
        q, k = ops.qk_norm_rope(
            rope.pair_halves(q, r), rope.pair_halves(k, r),
            rope.pair_halves(a.q_norm.weight, r),
            rope.pair_halves(a.k_norm.weight, r), cos, sin, a.q_norm.eps)
        o = attend(q, k, v.transpose(0, 2, 1, 3), chunk)
        return o.transpose(0, 2, 1, 3).reshape(b, n, -1)

    def __call__(self, x: mx.array, mod: tuple, segs: tuple, cos: mx.array,
                 sin: mx.array, chunk: int | None = None) -> mx.array:
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod
        h = modulated(self.norm1(x), shift_a, scale_a, segs)
        rows = self.attention(h, cos, sin, chunk)
        x = gated(x, gate_a, self.attn.out_proj(rows), segs)
        h = modulated(self.norm2(x), shift_f, scale_f, segs)
        return gated(x, gate_f, self.mlp.fc2(self.hidden(h)), segs)


class FinalLayer(nn.Module):
    def __init__(self, cfg: H3Config):
        super().__init__()
        self.norm = ops.RMSNorm(cfg.dim, cfg.final_eps)
        self.adaln_proj = AdalnProj(cfg.basis, 2 * cfg.dim)
        self.video_out = nn.Linear(cfg.dim, cfg.patch_dim)
        self.audio_out = nn.Linear(cfg.dim, cfg.audio)


class Rope(nn.Module):
    def __init__(self, freqs: int):
        super().__init__()
        self.inv_freq = mx.zeros((freqs,))


class H3Transformer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        cfg = H3Config.from_dict(config)
        self.cfg = cfg
        self.act_dtype = mx.bfloat16
        self.video_patch_proj = nn.Linear(cfg.patch_dim, cfg.dim)
        self.audio_patch_proj = nn.Linear(cfg.audio, cfg.dim)
        self.condition_proj = nn.Linear(cfg.text_dim, cfg.dim)
        self.adaln_t_table = mx.zeros((cfg.grid, cfg.basis))
        self.rope = Rope(cfg.freqs)
        self.token_refiner = TokenRefiner(cfg)
        self.blocks = [DiTBlock(cfg) for _ in range(cfg.depth)]
        self.final_layer = FinalLayer(cfg)

    @property
    def act_dtype(self) -> mx.Dtype:
        return self._act_dtype

    @act_dtype.setter
    def act_dtype(self, dtype: mx.Dtype) -> None:
        if dtype not in (mx.bfloat16, _F32):
            raise H3DtypeError(f"{fam.TITLE} runs in bfloat16 or float32, "
                               f"not {dtype}")
        self._act_dtype = dtype

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

    def plan(self, layout: Layout, t_video: float, t_audio: float,
             rules: dict | None = None) -> StepPlan:
        """The step's times, segment rows and RoPE tables. Text runs at the
        video time; anchor rows at their stream's pinned time."""
        rules = fam.ANCHOR_RULES if rules is None else rules
        t = {"text": t_video, "video": t_video, "audio": t_audio,
             "cond": pinned(t_video, rules["video"]),
             "cond_audio": pinned(t_audio, rules["audio"])}
        tag = {"text": fam.TAGS["text"], "video": fam.TAGS["video"],
               "cond": fam.TAGS["video"], "audio": fam.TAGS["audio"],
               "cond_audio": fam.TAGS["audio"]}
        times = tuple(sorted({t_video, t_audio}
                             | {t[s.kind] for s in layout.segments}))
        idx = {v: i for i, v in enumerate(times)}
        segs = tuple((s.start, s.stop, 3 * idx[t[s.kind]] + tag[s.kind])
                     for s in layout.segments if s.rows)
        cos, sin = rope.tables(layout.positions, self.rope.inv_freq,
                               self.cfg.head_dim)
        return StepPlan(times=times, segs=segs,
                        heads={"video": idx[t_video], "audio": idx[t_audio]},
                        cos=cos, sin=sin)

    def embed(self, video: mx.array, audio: mx.array, text: mx.array,
              layout: Layout, cond_video=(), cond_audio=(),
              taps: dict | None = None) -> mx.array:
        """The sequence (1, rows, dim) in segment order. The patch
        projections run in float32 on the anchor rows, then the targets.
        Each segment takes its rows by kind: each cond and cond_audio
        segment the next anchor's rows of its stream, the video and audio
        segments the targets'."""
        act = self.act_dtype
        vrows = mx.concatenate(
            [*cond_video, patchify(video, self.cfg.patch).astype(_F32)])
        arows = mx.concatenate([*cond_audio, pack_audio(audio).astype(_F32)])
        ve = self.video_patch_proj(vrows)
        ae = self.audio_patch_proj(arows)
        if taps is not None:
            taps["video_patch_proj"] = ve
            taps["audio_patch_proj"] = ae
        ve, ae = ve.astype(act), ae.astype(act)
        cv = [c.shape[0] for c in cond_video]
        ca = [c.shape[0] for c in cond_audio]
        # kind: (rows, the row count of each of its segments in turn, the
        # offset of the next)
        take = {"cond": (ve, cv, 0),
                "video": (ve, [ve.shape[0] - sum(cv)], sum(cv)),
                "cond_audio": (ae, ca, 0),
                "audio": (ae, [ae.shape[0] - sum(ca)], sum(ca))}
        parts = []
        for s in layout.segments:
            if s.kind == "text":
                if text.shape[1] != s.rows:
                    raise ValueError(f"{text.shape[1]} text rows for a "
                                     f"layout of {s.rows}")
                parts.append(text.astype(act))
                continue
            rows, sizes, at = take[s.kind]
            if not sizes or sizes[0] != s.rows:
                given = f"{sizes[0]} rows" if sizes else "none"
                raise ValueError(f"the layout's next {s.kind} segment has "
                                 f"{s.rows} rows, the inputs give {given}")
            parts.append(rows[None, at:at + s.rows])
            take[s.kind] = (rows, sizes[1:], at + s.rows)
        left = sorted(k for k, (_, sizes, _) in take.items() if sizes)
        if left:
            raise ValueError(f"the inputs hold rows for {left} segments the "
                             f"layout does not have")
        return mx.concatenate(parts, axis=1)

    def head(self, x: mx.array, t_emb: mx.array, plan: StepPlan,
             layout: Layout, taps: dict | None = None) -> tuple:
        """The raw head rows of the target video and audio, float32."""
        fl = self.final_layer
        y = fl.adaln_proj.linear(t_emb)
        if taps is not None:
            taps["final_layer.adaln_proj.linear"] = y
        y = y.reshape(-1, 2, self.cfg.dim)

        def rows(kind):
            s, i = layout.span(kind), plan.heads[kind]
            n = fl.norm(x[0, s.start:s.stop])
            return (ops.modulate(n, y[i, 1]) + y[i, 0]).astype(_F32)

        v, a = fl.video_out(rows("video")), fl.audio_out(rows("audio"))
        if taps is not None:
            taps["final_layer.video_out"] = v
            taps["final_layer.audio_out"] = a
        return v, a

    def __call__(self, video: mx.array, audio: mx.array, t_video: float,
                 t_audio: float, text: mx.array, layout: Layout, *,
                 plan: StepPlan | None = None, cond_video=(), cond_audio=(),
                 taps: dict | None = None,
                 chunk: int | None = fam.ATTENTION_CHUNK) -> tuple:
        """The velocity of each stream, float32: video (1, C, T, H, W) and
        audio (1, C, 2, T), at stream times ``t_video`` and ``t_audio``
        (1 is clean), with the text rows of ``caption``. ``cond_video``
        and ``cond_audio`` are the anchor rows of the layout's cond and
        cond_audio segments, in order. ``taps`` collects the output of
        every module by path."""
        if video.shape[0] != 1:
            raise ValueError(f"{fam.TITLE} runs a batch of 1")
        if plan is None:
            plan = self.plan(layout, t_video, t_audio)
        x = self.embed(video, audio, text, layout, cond_video, cond_audio,
                       taps)
        t_emb = self.curve(plan.times)
        if taps is not None:
            taps["in.blocks.0"] = x
            taps["t_emb"] = t_emb
        for i, blk in enumerate(self.blocks):
            x = blk(x, blk.modulation(t_emb, self.act_dtype), plan.segs,
                    plan.cos, plan.sin, chunk)
            pacing.point(x)
            if taps is not None:
                taps[f"blocks.{i}"] = x
        v, a = self.head(x, t_emb, plan, layout, taps)
        _, c, t, h, w = video.shape
        return (-unpatchify(v, (c, t, h, w), self.cfg.patch),
                -unpack_audio(a))
