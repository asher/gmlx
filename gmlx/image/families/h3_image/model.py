"""MiniMax-H3-Image DiT: H3's single-stream blocks over text and one
latent frame.

The sequence holds the text rows, then the target image's rows on H3's
clock after them (``plan``). A step runs at time t = 1 - sigma, each
segment with its modality row at t; the blocks, the token refiner and the
curve rows are H3's (``gmlx.image.h3.blocks``). The head runs the whole
``video_out`` bank in float32 on the target rows, and the step reads head
0 alone at the first grid point and head 0 plus head k at grid point k
(``family.step_head``). The model returns the flow velocity, the negated
head.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from gmlx.image import ops, pacing
from gmlx.image.h3 import layout as lay
from gmlx.image.h3 import rope
from gmlx.image.h3.blocks import (_F32, AdalnProj, DiTBlock, H3BlockConfig,
                                  H3Core, Rope, TokenRefiner)
from gmlx.image.families.h3_image import family as fam


@dataclass(frozen=True)
class H3ImageConfig:
    """The block config and what shared image code reads: the patch, the
    latent channels, the text width, the width, and the PDD grid with its
    head count."""
    block: H3BlockConfig
    patch: int
    in_channels: int
    text_dim: int
    dim: int
    pdd_sigmas: tuple
    step_heads: int

    @classmethod
    def from_dict(cls, c: dict) -> H3ImageConfig:
        patch = tuple(c["patch_size"])
        block = H3BlockConfig(
            dim=c["hidden_size"], heads=c["num_attention_heads"],
            head_dim=c["attention_head_dim"], ffn=c["ffn_hidden_size"],
            depth=c["num_layers"], refiner_depth=c["token_refiner_num_layers"],
            latent=c["latents_dim"], patch=patch, text_dim=c["text_dim"],
            freqs=c["rope_inv_freq_len"], eps=c["norm_eps"],
            qk_eps=c["qk_norm_eps"], final_eps=c["final_norm_eps"],
            grid=c["adaln_grid"], basis=c["adaln_basis"])
        if patch[0] != 1 or patch[1] != patch[2]:
            raise ValueError(f"{fam.TITLE} patchifies one frame square, not "
                             f"{patch}")
        return cls(block=block, patch=patch[1], in_channels=block.latent,
                   text_dim=block.text_dim, dim=block.dim,
                   pdd_sigmas=tuple(c["pdd_sigmas"]),
                   step_heads=c["pdd_steps"])


@dataclass(frozen=True)
class StepModulation:
    """What a step derives from sigma: the curve rows (2, basis) at t and
    at a reference image's time, those times, and the PDD head."""
    t_emb: mx.array
    times: tuple
    head: int

    def arrays(self) -> list:
        return [self.t_emb]


@dataclass(frozen=True)
class Caption:
    """The text rows (1, L, dim) after the refiner."""
    rows: mx.array

    def arrays(self) -> list:
        return [self.rows]


@dataclass(frozen=True)
class StepPlan:
    """A step's layout, each segment's (start, stop, modulation row), and
    the target's rows."""
    layout: lay.Layout
    segs: tuple
    target: lay.Segment


def patchify(latents: mx.array, patch: int) -> mx.array:
    """(1, C, H, W) to rows (h, w) of features (c, ph, pw)."""
    _, c, h, w = latents.shape
    x = latents[0].reshape(c, h // patch, patch, w // patch, patch)
    return x.transpose(1, 3, 0, 2, 4).reshape(-1, c * patch * patch)


def unpatchify(rows: mx.array, shape: tuple, patch: int) -> mx.array:
    """Rows of ``patchify`` back to (1, C, H, W) for ``shape`` (C, H, W)."""
    c, h, w = shape
    x = rows.reshape(h // patch, w // patch, c, patch, patch)
    return x.transpose(2, 0, 3, 1, 4).reshape(1, c, h, w)


@functools.lru_cache(maxsize=64)
def layout(text_len: int, lat_h: int, lat_w: int, spans: tuple = ()):
    """The segments and positions of a step: the text group at (i, 0, 0),
    a vision segment over each span, then the target on the clock after
    the text."""
    tl = fam.TIMELINE
    if lat_h % tl.patch or lat_w % tl.patch:
        raise ValueError(f"latent {lat_h}x{lat_w} is not a multiple of the "
                         f"patch {tl.patch}")
    segs, pos = [], []
    for kind, a, b in lay.text_segments(text_len, spans):
        g = np.zeros((b - a, 3), np.float64)
        g[:, 0] = np.arange(a, b, dtype=np.float64)
        segs.append(lay.Segment(a, b, kind))
        pos.append(g)
    frame, _ = lay.frame_grid(tl, lat_h, lat_w)
    p = lay.video_positions(tl, 1, frame, float(text_len))
    segs.append(lay.Segment(text_len, text_len + p.shape[0], "video"))
    pos.append(p)
    return lay.Layout(segments=tuple(segs), positions=np.concatenate(pos),
                      signature=(text_len, lat_h, lat_w, spans))


class FinalLayer(nn.Module):
    def __init__(self, cfg: H3ImageConfig):
        super().__init__()
        b = cfg.block
        self.norm = ops.RMSNorm(b.dim, b.final_eps)
        self.adaln_proj = AdalnProj(b.basis, 2 * b.dim)
        self.video_out = nn.Linear(b.dim, cfg.step_heads * b.patch_dim)


class H3ImageDiT(H3Core):
    title = fam.TITLE

    def __init__(self, config: dict):
        super().__init__()
        cfg = H3ImageConfig.from_dict(config)
        b = cfg.block
        self.cfg = cfg
        self.act_dtype = mx.bfloat16
        self.attention = "dense"
        self._linears = "file"
        self.video_patch_proj = nn.Linear(b.patch_dim, b.dim)
        self.condition_proj = nn.Linear(b.text_dim, b.dim)
        self.adaln_t_table = mx.zeros((b.grid, b.basis))
        self.rope = Rope(b.freqs)
        self.token_refiner = TokenRefiner(b)
        self.blocks = [DiTBlock(b) for _ in range(b.depth)]
        self.final_layer = FinalLayer(cfg)

    def modulation(self, sigma, batch: int) -> StepModulation:
        """The curve rows of a step at ``sigma``: t = 1 - sigma as the
        release's DiT reads sigma (``family.model_sigma``), and its PDD
        head."""
        if batch != 1:
            raise ValueError(f"{fam.TITLE} runs a batch of 1")
        s = float(np.asarray(sigma, np.float32).reshape(-1)[0])
        t = 1.0 - fam.model_sigma(s)
        times = (t, max(t, fam.REF_TIME))
        return StepModulation(self.curve(times), times, fam.step_head(s))

    def caption(self, feats: mx.array, taps: dict | None = None) -> Caption:
        return Caption(super().caption(feats, taps))

    def plan(self, text_len: int, lat_h: int, lat_w: int,
             spans: tuple = ()) -> StepPlan:
        """The step's layout and segment rows: text at tag text, vision
        rows and the target at tag image, each at the step's time."""
        lt = layout(text_len, lat_h, lat_w, tuple(spans))
        tag = {"text": fam.TAGS["text"], "vision": fam.TAGS["video"],
               "video": fam.TAGS["video"]}
        segs = tuple((s.start, s.stop, tag[s.kind]) for s in lt.segments
                     if s.rows)
        return StepPlan(layout=lt, segs=segs, target=lt.span("video"))

    def embed(self, latents: mx.array, text: mx.array, plan: StepPlan,
              taps: dict | None = None) -> mx.array:
        """The sequence (1, rows, dim): the text rows, then the target's
        rows through the float32 patch projection."""
        act = self.act_dtype
        rows = self.video_patch_proj(
            patchify(latents, self.cfg.patch).astype(_F32))
        if taps is not None:
            taps["video_patch_proj"] = rows
        if text.shape[1] != plan.target.start:
            raise ValueError(f"{text.shape[1]} text rows for a layout of "
                             f"{plan.target.start}")
        return mx.concatenate([text.astype(act), rows.astype(act)[None]],
                              axis=1)

    def head(self, x: mx.array, mod: StepModulation, plan: StepPlan,
             taps: dict | None = None) -> mx.array:
        """The step's head rows of the target, float32, at the step's time:
        the bank's head 0, plus head k at grid point k."""
        fl = self.final_layer
        y = fl.adaln_proj.linear(mod.t_emb[:1])
        if taps is not None:
            taps["final_layer.adaln_proj.linear"] = y
        shift, scale = y.reshape(2, self.cfg.dim)
        s = plan.target
        n = fl.norm(x[0, s.start:s.stop])
        bank = fl.video_out((ops.modulate(n, scale) + shift).astype(_F32))
        pd = self.cfg.block.patch_dim
        k = mod.head
        out = bank[:, :pd]
        if k:
            out = out + bank[:, k * pd:(k + 1) * pd]
        if taps is not None:
            taps["final_layer"] = out
        return out

    def __call__(self, latents: mx.array, sigma, feats: mx.array | None,
                 taps: dict | None = None, *,
                 mod: StepModulation | None = None,
                 caption: Caption | None = None, refs=None,
                 chunk: int | None = fam.ATTENTION_CHUNK) -> mx.array:
        """Velocity (1, C, H, W) float32 for latents (1, C, H, W) at
        ``sigma``, with text features (1, L, text dim). ``mod`` and
        ``caption``, from ``modulation`` and ``caption``, stand in for the
        work they hold. ``taps`` collects the output of every module by
        path."""
        if refs is not None:
            raise ValueError(f"{fam.TITLE} takes no reference images yet")
        if latents.shape[0] != 1:
            raise ValueError(f"{fam.TITLE} runs a batch of 1")
        if mod is None:
            mod = self.modulation(sigma, 1)
        if caption is None:
            if feats is None:
                raise ValueError("pass text features or their caption")
            caption = self.caption(feats, taps)
        _, c, h, w = latents.shape
        plan = self.plan(caption.rows.shape[1], h, w)
        x = self.embed(latents, caption.rows, plan, taps)
        cos, sin = rope.tables(plan.layout.positions, self.rope.inv_freq,
                               self.cfg.block.head_dim)
        if taps is not None:
            taps["in.blocks.0"] = x
            taps["t_emb"] = mod.t_emb
        for i, blk in enumerate(self.blocks):
            x = blk(x, blk.modulation(mod.t_emb, self.act_dtype), plan.segs,
                    cos, sin, chunk=chunk, attention=self.attention)
            pacing.point(x)
            if taps is not None:
                taps[f"blocks.{i}"] = x
        v = self.head(x, mod, plan, taps)
        return -unpatchify(v, (c, h, w), self.cfg.patch)
