"""MiniMax-H3-Image DiT: H3's single-stream blocks over text and one
latent frame.

The sequence holds the text rows, then each reference image's rows at its
own time after the text, then the target image's rows on H3's clock after
them (``plan``). A step runs at time t = 1 - sigma, each segment with its
modality row at t, and the reference rows at the family's pinned time;
the blocks, the token refiner and the curve rows are H3's
(``gmlx.image.h3.blocks``). A reference's rows are its patchified latents,
mixed with a little noise by the family's rule (``family.REF_RULE``). The head runs the whole
``video_out`` bank in float32 on the target rows, and a step reads head
0 plus each later head of the grid intervals it spans by its share of
the span (``family.step_heads``): head 0 alone from the first grid point
to the next, head 0 plus head k from grid point k to k + 1. The model
returns the flow velocity, the negated head.
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
from gmlx.image.h3.anchors import augment, pinned
from gmlx.image.h3.pictures import slot_spans
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
    """What a step derives from its sigma and the sigma it goes to: the
    curve rows (2, basis) at t and at a reference image's time, those
    times, and each PDD head past head 0 it reads with its weight."""
    t_emb: mx.array
    times: tuple
    heads: tuple

    def arrays(self) -> list:
        return [self.t_emb]


@dataclass(frozen=True)
class Caption:
    """The text rows (1, L, dim) after the refiner, and the vision spans
    of the prompt's pictures, (start, stop) rows."""
    rows: mx.array
    spans: tuple = ()

    def arrays(self) -> list:
        return [self.rows]


@dataclass(frozen=True)
class References:
    """The reference images of a step: each one's rows (n, patch dim) in
    float32 as the family's rule mixes them, and its latent (h, w)."""
    rows: tuple
    sizes: tuple

    def arrays(self) -> list:
        return list(self.rows)


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
def layout(text_len: int, lat_h: int, lat_w: int, spans: tuple = (),
           refs: tuple = ()):
    """The segments and positions of a step: the text group at (i, 0, 0),
    a vision segment over each span, then each reference of latent (h, w)
    in ``refs`` at time text_len + i on its own frame's grid, then the
    target on the clock after the references (comfy model.py:383-483)."""
    tl = fam.TIMELINE
    for h, w in ((lat_h, lat_w), *refs):
        if h % tl.patch or w % tl.patch:
            raise ValueError(f"latent {h}x{w} is not a multiple of the "
                             f"patch {tl.patch}")
    segs, pos = [], []
    for kind, a, b in lay.text_segments(text_len, spans):
        g = np.zeros((b - a, 3), np.float64)
        g[:, 0] = np.arange(a, b, dtype=np.float64)
        segs.append(lay.Segment(a, b, kind))
        pos.append(g)
    at = text_len
    for i, (h, w) in enumerate(refs):
        grid, _ = lay.frame_grid(tl, h, w)
        g = np.empty((grid.shape[0], 3), np.float64)
        g[:, 0] = float(text_len + i)
        g[:, 1:] = grid
        segs.append(lay.Segment(at, at + g.shape[0], "ref"))
        pos.append(g)
        at += g.shape[0]
    frame, _ = lay.frame_grid(tl, lat_h, lat_w)
    p = lay.video_positions(tl, 1, frame, float(text_len + len(refs)))
    segs.append(lay.Segment(at, at + p.shape[0], "video"))
    pos.append(p)
    positions = np.concatenate(pos)
    positions.setflags(write=False)
    return lay.Layout(segments=tuple(segs), positions=positions,
                      signature=(text_len, lat_h, lat_w, spans, refs))


class FinalLayer(nn.Module):
    def __init__(self, cfg: H3ImageConfig):
        super().__init__()
        b = cfg.block
        self.norm = ops.RMSNorm(b.dim, b.final_eps)
        self.adaln_proj = AdalnProj(b.basis, 2 * b.dim)
        self.video_out = nn.Linear(b.dim, cfg.step_heads * b.patch_dim)


def _head_index(head, bank: int) -> int:
    """``head`` as an index into a bank of ``bank`` heads, refused by
    name off it."""
    if isinstance(head, bool) or not isinstance(head, (int, np.integer)) \
            or not 0 <= int(head) < bank:
        raise ValueError(f"the {fam.TITLE} bank holds heads 0 to {bank - 1}, "
                         f"not {head!r}")
    return int(head)


class H3ImageDiT(H3Core):
    title = fam.TITLE
    # The heads a step reads follow the sigma it goes to, which sample
    # passes as the target, or the head a step names (``head_bank``).
    reads_target = True
    # The caption takes the image slots of the prompt's pictures, and the
    # references take the request's seed or their noise.
    caption_slots = True
    seeded_refs = True

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

    @property
    def head_bank(self) -> int:
        """The PDD heads of the bank, which a step may name by index."""
        return self.cfg.step_heads

    def modulation(self, sigma, batch: int, target=None,
                   head: int | None = None) -> StepModulation:
        """The curve rows of a step from ``sigma`` to ``target``: t = 1 -
        sigma as the release's DiT reads sigma (``family.model_sigma``),
        and the PDD heads the step spans (``family.step_heads``). A
        ``head`` index reads that head alone, head 0 or head 0 plus head
        k, as diffusers reads head i at step i, and takes no target."""
        if batch != 1:
            raise ValueError(f"{fam.TITLE} runs a batch of 1")
        s = float(np.asarray(sigma, np.float32).reshape(-1)[0])
        t = 1.0 - fam.model_sigma(s)
        times = (t, pinned(t, fam.REF_RULE))
        if head is not None:
            k = _head_index(head, self.head_bank)
            return StepModulation(self.curve(times), times,
                                  ((k, 1.0),) if k else ())
        if target is None:
            raise ValueError(f"the {fam.TITLE} heads read the sigma the step "
                             f"goes to: pass target")
        to = float(np.asarray(target, np.float32).reshape(-1)[0])
        start, stop, w = fam.step_heads(s, to)
        first = max(start, 1)
        heads = tuple(zip(range(first, stop), w[first - start:]))
        return StepModulation(self.curve(times), times, heads)

    def caption(self, feats: mx.array, taps: dict | None = None, *,
                slots=()) -> Caption:
        """The text rows of features (1, L, text dim), whose image
        ``slots`` are the vision tokens of the prompt's pictures."""
        return Caption(super().caption(feats, taps), slot_spans(slots))

    def references(self, latents, *, seed: int | None = None,
                   noise=None) -> References:
        """The rows of reference latents, each (1, C, h, w) in model
        space, mixed by the family's rule with noise from ``seed`` or
        ``noise``, one array (n, patch dim) per reference
        (``anchors.augment``)."""
        if noise is None and seed is None:
            raise ValueError(f"{fam.TITLE} references take a seed or their "
                             "noise")
        rows, sizes = [], []
        for z in latents:
            if z.ndim != 4 or z.shape[0] != 1 \
                    or z.shape[1] != self.cfg.in_channels:
                raise ValueError(f"a reference latent is (1, "
                                 f"{self.cfg.in_channels}, h, w), not "
                                 f"{tuple(z.shape)}")
            rows.append(patchify(z.astype(_F32), self.cfg.patch))
            sizes.append((int(z.shape[2]), int(z.shape[3])))
        mixed = augment(rows, fam.REF_RULE, 0 if seed is None else seed,
                        noise)
        return References(tuple(mixed), tuple(sizes))

    def plan(self, text_len: int, lat_h: int, lat_w: int,
             spans: tuple = (), refs: tuple = ()) -> StepPlan:
        """The step's layout and segment rows: text at tag text, vision
        rows and the target at tag image at the step's time, and the
        references of latent (h, w) in ``refs`` at tag image at the
        pinned time, the modulation's second time."""
        lt = layout(text_len, lat_h, lat_w, tuple(spans), tuple(refs))
        row = {"text": fam.TAGS["text"], "vision": fam.TAGS["video"],
               "video": fam.TAGS["video"], "ref": 3 + fam.TAGS["video"]}
        segs = tuple((s.start, s.stop, row[s.kind]) for s in lt.segments
                     if s.rows)
        return StepPlan(layout=lt, segs=segs, target=lt.span("video"))

    def embed(self, latents: mx.array, text: mx.array, plan: StepPlan,
              taps: dict | None = None,
              refs: References | None = None) -> mx.array:
        """The sequence (1, rows, dim): the text rows, then the rows of
        each reference and the target's through the float32 patch
        projection, in one product as ComfyUI's forward runs them."""
        act = self.act_dtype
        rows = patchify(latents, self.cfg.patch).astype(_F32)
        if refs is not None and refs.rows:
            rows = mx.concatenate([*refs.rows, rows], axis=0)
        rows = self.video_patch_proj(rows)
        if taps is not None:
            taps["video_patch_proj"] = rows
        text_len = plan.layout.signature[0]
        if text.shape[1] != text_len \
                or text_len + rows.shape[0] != plan.layout.rows:
            raise ValueError(f"{text.shape[1]} text rows and "
                             f"{rows.shape[0]} image rows for a layout of "
                             f"{text_len} and {plan.layout.rows - text_len}")
        return mx.concatenate([text.astype(act), rows.astype(act)[None]],
                              axis=1)

    def head(self, x: mx.array, mod: StepModulation, plan: StepPlan,
             taps: dict | None = None) -> mx.array:
        """The step's head rows of the target, float32, at the step's time:
        the bank's head 0, plus each head the step spans by its weight."""
        fl = self.final_layer
        y = fl.adaln_proj.linear(mod.t_emb[:1])
        if taps is not None:
            taps["final_layer.adaln_proj.linear"] = y
        shift, scale = y.reshape(2, self.cfg.dim)
        s = plan.target
        n = fl.norm(x[0, s.start:s.stop])
        bank = fl.video_out((ops.modulate(n, scale) + shift).astype(_F32))
        pd = self.cfg.block.patch_dim
        out = bank[:, :pd]
        for k, w in mod.heads:
            out = out + w * bank[:, k * pd:(k + 1) * pd]
        if taps is not None:
            taps["final_layer"] = out
        return out

    def __call__(self, latents: mx.array, sigma, feats: mx.array | None,
                 taps: dict | None = None, *, target=None,
                 mod: StepModulation | None = None,
                 caption: Caption | None = None, refs=None,
                 chunk: int | None = fam.ATTENTION_CHUNK) -> mx.array:
        """Velocity (1, C, H, W) float32 for latents (1, C, H, W) on a
        step from ``sigma`` to ``target``, with text features (1, L, text
        dim) and the rows of reference images ``refs``, from
        ``references``. ``mod`` and ``caption``, from ``modulation`` and
        ``caption``, stand in for the work they hold. ``taps`` collects
        the output of every module by path."""
        if latents.shape[0] != 1:
            raise ValueError(f"{fam.TITLE} runs a batch of 1")
        if mod is None:
            mod = self.modulation(sigma, 1, target)
        if caption is None:
            if feats is None:
                raise ValueError("pass text features or their caption")
            caption = self.caption(feats, taps)
        _, c, h, w = latents.shape
        plan = self.plan(caption.rows.shape[1], h, w, caption.spans,
                         () if refs is None else refs.sizes)
        x = self.embed(latents, caption.rows, plan, taps, refs)
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
