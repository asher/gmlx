"""The memory plan of an image request, from header sizes and the request.

``plan_memory`` is a pure function. Resident bytes are the DiT, the text
encoder, the VAE at its decode dtype, the mmproj for edits, adapter
factors and caches. A request runs in three phases, encode, denoise and
decode, and each phase adds its own activation peak to what is resident
then. The plan fits when the largest phase stays within the working budget
of ``gmlx.serve.capacity``.

A family that caches a prefix per conditioning adds three terms to the
denoise phase: the cache of every conditioning, the K and V of one block
joined over the prefix and the image rows in each step, and the prefill
pass over the prefix rows, which a conditioning runs once. The cache term
also holds the states of other conditionings that stay in the cache
beside the request's own, up to its bound.

A request runs a negative row whenever its guidance is not 1.0, as
``api.sample`` does, and the plan counts the second conditioning then.

Levers apply in order until the plan fits: sequential rows in place of a
batch, release of an owned encoder after encoding, and tiled VAE decode.
A plan that still does not fit raises ``ImageMemoryError`` with each term
in GiB and the largest term named, unless GMLX_OVERCOMMIT=1.

The MLX buffer cache keeps freed buffers for reuse up to its limit, which
defaults to 95 percent of memory. One VAE decode frees and allocates
buffers of many sizes inside one evaluation, so under the default limit
its footprint reaches two to three times its live peak. Image work runs
inside ``bounded_cache``, which lowers the limit to ``CACHE_BOUND`` for
the block, and the step and VAE peaks below are footprints measured that
way. ``vae_tiling`` decides between a whole and a tiled VAE pass from
those peaks and the free working set, and refuses one that does not fit.
"""

from __future__ import annotations

import math
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field

GiB = 1 << 30


@dataclass(frozen=True)
class StepFit:
    """Footprint of one denoise step over the weights: ``base`` bytes plus
    ``k`` times the step formula, the rows times the widest outputs per
    row times the activation itemsize."""
    base: float
    k: float

    def bytes(self, formula: float) -> int:
        return int(self.base + self.k * formula)


# One Z-Image-Turbo Q4_K_M step through ``api.sample`` under bounded_cache
# on an M5 Max, in fresh processes: 0.54, 1.28, 2.19 and 2.24 GiB over the
# settled process at 288, 1056, 4128 and 4608 rows. At 4128 rows the
# active peak is 1.05 GiB above the weights, and the rest is freed buffers
# the cache keeps up to its bound. A line in the formula bytes through the
# 1056 and 4128 row cells has a slope of 5.0. The slope is raised 10
# percent and the base covers every cell.
Z_IMAGE_STEP = StepFit(base=1.0 * GiB, k=5.5)

# Measured footprint of one 512-token Qwen3-4B Q4_K_M encode on an M5 Max:
# 0.35 GiB above the weights, 12.5 times the per-token formula, rounded up.
ENCODE_LIVE = 16.0

# The MLX cache limit inside bounded_cache.
CACHE_BOUND = 2 * GiB

TILE_PX = 512
OVERLAP_PX = 64


@dataclass(frozen=True)
class VaeFit:
    """Footprint of one VAE pass over the weights and inputs: ``base``
    bytes plus ``k`` bytes per pixel, base channel and activation byte."""
    base: float
    k: float

    def bytes(self, pixels: int, channels: int, itemsize: int) -> int:
        return int(self.base + self.k * pixels * channels * itemsize)


# FLUX.1 autoencoder footprints under bounded_cache, by conv path (see
# ``gmlx.image.vae.conv``) and activation itemsize. Whole passes at 256,
# 512, 768 and 1024 pixels on an M5 Max, two runs each in fresh processes,
# fit a line in pixels. The slope is raised 10 percent and the base
# covers every whole pass plus 0.25 GiB. At 1024 the bf16 decode measured
# 4.45 GiB exact and 6.64 GiB on Winograd, against 5.61 and 7.48 here,
# and the fp32 decode 7.01 and 10.42 GiB, against 8.44 and 11.55. The
# Qwen-Image 2.1 VAE uses these rows by its channel count until it is
# measured.
VAE_DECODE = {"exact": {2: VaeFit(base=1.59 * GiB, k=16.1),
                        4: VaeFit(base=2.55 * GiB, k=11.8)},
              "winograd": {2: VaeFit(base=2.35 * GiB, k=20.5),
                           4: VaeFit(base=2.70 * GiB, k=17.7)}}
VAE_ENCODE = {"exact": {2: VaeFit(base=1.53 * GiB, k=10.7),
                        4: VaeFit(base=2.78 * GiB, k=7.3)},
              "winograd": {2: VaeFit(base=2.35 * GiB, k=15.2),
                           4: VaeFit(base=2.50 * GiB, k=12.6)}}



@dataclass(frozen=True)
class TiledFit:
    """Footprint of one tiled VAE pass: ``base`` bytes, plus ``k_tiles``
    bytes per pixel of all tiles together and ``k_image`` per pixel of the
    image, each per base channel and activation byte. The tile term is the
    activations every tile holds at a GroupNorm, and the image term the
    part run whole at latent resolution and the blend buffers."""
    base: float
    k_tiles: float
    k_image: float

    def bytes(self, tile_px: int, image_px: int, channels: int,
              itemsize: int) -> int:
        per = channels * itemsize
        return int(self.base + (self.k_tiles * tile_px
                                + self.k_image * image_px) * per)


# Tiled passes on the exact path, in tiles of 512 pixels and in shrunk
# tiles at 768, 1024 and 2048 pixels, fit base + k_tiles * S + k_image * P.
# The slopes are raised 10 percent and the base covers every pass plus
# 0.25 GiB. A tiled bf16 decode at 1024 in three 384 pixel tiles measured
# 3.20 GiB against 3.45 here, and at 2048 in 512 pixel tiles 8.24 against
# 9.30. A Winograd pass adds the difference of the whole rows at one tile.
VAE_TILED = {"decode": {2: TiledFit(base=1.96 * GiB, k_tiles=4.70,
                                    k_image=0.0),
                        4: TiledFit(base=1.80 * GiB, k_tiles=4.60,
                                    k_image=0.0)},
             "encode": {2: TiledFit(base=1.88 * GiB, k_tiles=1.00,
                                    k_image=3.14),
                        4: TiledFit(base=1.69 * GiB, k_tiles=1.29,
                                    k_image=2.91)}}


@dataclass(frozen=True)
class PrefixRow:
    """The prefix a family caches per conditioning: the K and V of every
    block for the text rows and the reference latent tokens. A step holds
    one block's cached K and V joined with the target's own, and the
    prefill runs the prefix rows through the blocks once per
    conditioning."""
    blocks: int                     # blocks that cache K and V
    dim: int                        # elements of one K or V row
    token_px: int                   # pixels per reference token on each axis

    def rows(self, text: int, refs: int, ref_px: int) -> int:
        """Prefix rows of one conditioning with ``text`` text rows and
        ``refs`` square references of ``ref_px`` pixels."""
        return text + refs * (ref_px // self.token_px) ** 2

    def cache_bytes(self, rows: int, itemsize: int) -> int:
        return 2 * self.blocks * rows * self.dim * itemsize

    def join_bytes(self, rows: int, itemsize: int) -> int:
        """K and V of one block over ``rows`` joined rows."""
        return 2 * rows * self.dim * itemsize


@dataclass(frozen=True)
class PeakRow:
    """How a family's activation peak follows from the request."""
    token_px: int                   # pixels per token along each axis
    row_multiple: int               # image rows pad up to a multiple of it
    caption_rows: int               # caption rows at most, joined to the image
    width: int                      # widest outputs per row, in elements
    step: StepFit                   # step footprint from the formula
    vae_channels: int               # VAE channels at full resolution
    prefix: PrefixRow | None = None
    encode_row_bytes: int = 0       # encoder activation bytes per token,
                                    # times ENCODE_LIVE
    text_rows: int = 512


PEAKS = {
    "z_image": PeakRow(token_px=16, row_multiple=32, caption_rows=512,
                       width=11520 + 2 * 10240, step=Z_IMAGE_STEP,
                       vae_channels=128, encode_row_bytes=3 * 9728 * 2),
    # A step runs the target rows alone: the fused MLP product and its
    # gated half are the widest outputs. The sizes are those the
    # fingerprint holds the file to: 32 blocks, 4096 wide, a 12288 MLP.
    # The step row is Z-Image's until a Qwen-Image 2.1 step is measured.
    # No term counts the attention scores, heads x queries x keys, since
    # the stock attention op was not seen to hold them: a Z-Image step at
    # 2048x2048, 16896 rows and 30 heads, peaks at 9.6 GiB where the scores
    # alone are 16 GiB in bf16. That is one machine and one model. Were
    # the scores held here, a step would add 1.1 GiB at 1024x1024,
    # 16.5 GiB at 2048x2048 and 10.1 GiB with nine 1024 pixel references.
    "qwen_image21": PeakRow(token_px=16, row_multiple=1, caption_rows=0,
                            width=24576 + 12288, step=Z_IMAGE_STEP,
                            vae_channels=96,
                            prefix=PrefixRow(blocks=32, dim=4096,
                                             token_px=16),
                            encode_row_bytes=3 * 12288 * 2),
}

LEVERS = ("sequential rows", "release the encoder", "tiled decode")


class ImageMemoryError(RuntimeError):
    """An image request or VAE pass that does not fit the working budget.
    ``plan`` is the request's MemoryPlan, or None for a VAE pass."""

    def __init__(self, message: str, plan: "MemoryPlan | None" = None):
        super().__init__(message)
        self.plan = plan


@dataclass(frozen=True)
class MemoryPlan:
    terms: dict                     # name -> bytes
    phases: dict                    # phase -> bytes at its peak
    budget: float | None
    rows: int                       # rows evaluated together per step
    release_encoder: bool
    tiled_decode: bool
    levers: tuple                   # applied, in order
    fits: bool
    notes: tuple = field(default=())

    @property
    def peak(self) -> int:
        return max(self.phases.values())

    @property
    def weight_bytes(self) -> int:
        return sum(self.terms[k] for k in
                   ("dit", "encoder", "vae", "mmproj", "adapters"))

    def describe(self) -> str:
        parts = [f"{k} {v / GiB:.2f} GiB" for k, v in self.terms.items() if v]
        phases = ", ".join(f"{k} {v / GiB:.2f}" for k, v in
                           self.phases.items())
        budget = ("unknown" if self.budget is None
                  else f"{self.budget / GiB:.2f} GiB")
        levers = ", ".join(self.levers) or "none"
        return (f"peak {self.peak / GiB:.2f} GiB against a budget of "
                f"{budget} (phases {phases}; levers {levers}): "
                + ", ".join(parts))


def tensor_bytes(shape, codec: str) -> int:
    """Bytes of a GGUF tensor of ``shape`` in ``codec``."""
    from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

    block, size = GGML_QUANT_SIZES[GGMLQuantizationType[codec]]
    return math.prod(int(d) for d in shape) // block * size


def file_bytes(src, *, skip=()) -> int:
    """Tensor bytes of a header view, or ``src`` itself when it is a
    number. Names in ``skip`` are left out."""
    if src is None:
        return 0
    if isinstance(src, (int, float)):
        return int(src)
    return sum(t.nbytes or tensor_bytes(t.shape, t.codec)
               for n, t in src.tensors.items() if n not in skip)


def vae_bytes(src, itemsize: int) -> int:
    """VAE bytes at the decode dtype, from a VAE header or a number."""
    if src is None:
        return 0
    if isinstance(src, (int, float)):
        return int(src)
    shapes = getattr(src, "shapes", None)
    if shapes is None:
        shapes = {n: t.shape for n, t in src.tensors.items()}
    return sum(math.prod(int(d) for d in s) for s in shapes.values()) \
        * itemsize


def image_rows(row: PeakRow, height: int, width: int,
               caption: int | None) -> int:
    """Rows per item of one step: image tokens padded to the family's
    multiple, plus the caption padded the same way when it joins them."""
    tokens = (height // row.token_px) * (width // row.token_px)
    m = row.row_multiple
    cap = row.caption_rows if caption is None else min(caption,
                                                       row.caption_rows)
    return -(-tokens // m) * m + (-(-cap // m) * m if row.caption_rows else 0)


def prefix_row_terms(row: PeakRow, image: int, prefix: int,
                     itemsize: int) -> tuple[int, int, int]:
    """Bytes a conditioning of ``prefix`` prefix rows adds to a request
    with ``image`` image rows: its cache, the step formula of one row with
    the joined K and V of a block, and the step formula of its prefill."""
    step = image * row.width * itemsize
    if row.prefix is None:
        return 0, step, 0
    return (row.prefix.cache_bytes(prefix, itemsize),
            step + row.prefix.join_bytes(prefix + image, itemsize),
            prefix * row.width * itemsize)


def prefix_terms(row: PeakRow, image: int, text: int, refs: int,
                 ref_px: int, itemsize: int) -> tuple[int, int, int]:
    """``prefix_row_terms`` of a conditioning with ``text`` text rows and
    ``refs`` square references of ``ref_px`` pixels."""
    rows = 0 if row.prefix is None else row.prefix.rows(text, refs, ref_px)
    return prefix_row_terms(row, image, rows, itemsize)


def _dtype_name(dtype) -> str:
    return str(dtype).rsplit(".", 1)[-1]


def _itemsize(dtype) -> int:
    return {"bfloat16": 2, "bf16": 2, "float16": 2, "fp16": 2,
            "float32": 4, "fp32": 4}[_dtype_name(dtype)]


def _largest(terms: dict) -> str:
    fixes = {
        "dit": "a smaller DiT quant",
        "encoder": "a smaller encoder quant",
        "step_peak": "a lower resolution",
        "decode_peak": "a lower resolution",
        "prefix_cache": "fewer or smaller references",
        "mmproj": "a smaller mmproj",
        "adapters": "fewer adapters",
        "caches": "a smaller cache",
    }
    name = max((k for k in terms if k in fixes), key=lambda k: terms[k])
    return f"The largest term is {name}; try {fixes[name]}"


def plan_memory(dit, encoder, vae, mmproj=None, adapters=(), *,
                family: str, height: int, width: int, cfg: float = 1.0,
                dtype="bfloat16", vae_dtype=None, batch_rows: bool = True,
                encoder_owned: bool = True, caption_tokens: int | None = None,
                text_tokens: int | None = None, refs: int = 0,
                ref_px: int = 0, prefix_rows=None, prefix_other: int = 0,
                prefix_bound: int = 0, cache_bytes: int = 0,
                tile: int = 512, budget: float | None = None,
                overcommit: bool | None = None) -> MemoryPlan:
    """The memory plan for one image request.

    ``dit``, ``encoder``, ``mmproj`` and ``vae`` are header views or byte
    counts. ``adapters`` is a byte count or a list of them. ``budget``
    defaults to ``capacity.working_budget_bytes()`` and ``overcommit`` to
    GMLX_OVERCOMMIT.

    The prefix rows of a conditioning come from ``text_tokens``, ``refs``
    and ``ref_px``. ``prefix_rows`` gives them for the request's own
    conditionings, the positive first, in place of that count. A cache of
    prefix states that holds ``prefix_other`` bytes of other conditionings
    sheds them to ``prefix_bound`` as the request's states come in, and
    what stays is part of the prefix cache term.
    """
    try:
        row = PEAKS[family]
    except KeyError:
        known = ", ".join(sorted(PEAKS))
        raise ValueError(f"no memory row for image family {family!r}; "
                         f"known: {known}") from None
    if budget is None:
        from gmlx.serve.capacity import working_budget_bytes

        budget = working_budget_bytes()
    if overcommit is None:
        from gmlx.serve.capacity import overcommit as _oc

        overcommit = _oc()

    act = _itemsize(dtype)
    if vae_dtype is None:
        vae_dtype = ("bfloat16" if _dtype_name(dtype) in ("bfloat16", "bf16")
                     else "float32")
    vsize = _itemsize(vae_dtype)
    guided = cfg != 1.0
    conds = 2 if guided else 1
    terms = {
        "dit": file_bytes(dit),
        "encoder": file_bytes(encoder, skip=("output.weight",)),
        "vae": vae_bytes(vae, vsize),
        "mmproj": file_bytes(mmproj),
        "adapters": (int(adapters) if isinstance(adapters, (int, float))
                     else sum(int(a) for a in adapters)),
        "caches": int(cache_bytes),
        "prefix_cache": 0,
    }
    rows = image_rows(row, height, width, caption_tokens)
    text = row.text_rows if text_tokens is None else text_tokens
    counted = 0 if row.prefix is None else row.prefix.rows(text, refs,
                                                           ref_px)
    given = [] if prefix_rows is None else [int(n) for n in prefix_rows]
    per = [prefix_row_terms(row, rows, n, act)
           for n in (given + [counted] * conds)[:conds]]
    own = sum(p[0] for p in per)
    step_one, prefill = max(p[1] for p in per), max(p[2] for p in per)
    terms["prefix_cache"] = own + min(int(prefix_other),
                                      max(0, int(prefix_bound) - own))
    full_decode = vae_peak("decode", height, width, itemsize=vsize,
                           channels=row.vae_channels)
    tiled_decode = vae_peak("decode", height, width, itemsize=vsize,
                            channels=row.vae_channels, tiled=True, tile=tile)
    caption = row.text_rows if caption_tokens is None else caption_tokens
    encode_peak = caption * row.encode_row_bytes * ENCODE_LIVE

    def build(b: int, release: bool, tiled: bool, levers: tuple) -> MemoryPlan:
        t = dict(terms)
        t["step_peak"] = row.step.bytes(max(b * step_one, prefill))
        t["decode_peak"] = int(tiled_decode if tiled else full_decode)
        t["encode_peak"] = int(encode_peak)
        resident = sum(t[n] for n in ("dit", "encoder", "vae", "mmproj",
                                      "adapters", "caches"))
        later = resident - (t["encoder"] if release else 0)
        phases = {
            "encode": resident + t["encode_peak"],
            "denoise": later + t["prefix_cache"] + t["step_peak"],
            "decode": later + t["decode_peak"],
        }
        fits = budget is None or max(phases.values()) <= budget
        return MemoryPlan(terms=t, phases=phases, budget=budget, rows=b,
                          release_encoder=release, tiled_decode=tiled,
                          levers=levers, fits=fits)

    b = 2 if guided and batch_rows else 1
    plan = build(b, False, False, ())
    steps = []
    if b == 2:
        steps.append(("sequential rows", dict(b=1)))
    if encoder_owned and terms["encoder"]:
        steps.append(("release the encoder", dict(release=True)))
    steps.append(("tiled decode", dict(tiled=True)))
    state = dict(b=b, release=False, tiled=False)
    applied: tuple = ()
    for name, change in steps:
        if plan.fits:
            break
        state.update(change)
        applied += (name,)
        plan = build(state["b"], state["release"], state["tiled"], applied)
    if plan.fits or overcommit:
        return plan
    hint = _largest(plan.terms)
    if not encoder_owned and terms["encoder"]:
        hint += ("; a borrowed encoder stays resident, so load the encoder "
                 "on its own")
    if refs:
        hint += _ref_hint(row, plan, rows, text, refs, ref_px, conds, act)
    raise ImageMemoryError(
        f"image request does not fit: {plan.describe()}. {hint}. "
        "GMLX_OVERCOMMIT=1 overrides.", plan)


def _ref_hint(row: PeakRow, plan: MemoryPlan, image: int, text: int,
              refs: int, ref_px: int, conds: int, itemsize: int) -> str:
    """The largest reference count, and the largest reference size at the
    requested count, that fit the denoise phase."""
    if row.prefix is None or plan.budget is None:
        return ""
    room = (plan.budget - plan.phases["denoise"]
            + plan.terms["prefix_cache"] + plan.terms["step_peak"])

    def fits(n: int, px: int) -> bool:
        cache, step, prefill = prefix_terms(row, image, text, n, px,
                                            itemsize)
        return (conds * cache
                + row.step.bytes(max(plan.rows * step, prefill))) <= room

    count = max((n for n in range(refs + 1) if fits(n, ref_px)), default=0)
    side = next((s for s in range(ref_px, 0, -row.prefix.token_px)
                 if fits(refs, s)), 0)
    return (f"; {count} references of {ref_px} px fit, or {refs} of "
            f"{side} px")


def tile_layout(size: int, tile: int = TILE_PX,
                overlap: int = OVERLAP_PX) -> tuple[int, int]:
    """Tiles along one axis of ``size`` pixels and their length, as
    ``tiling.tile_length`` and ``tiling.tile_starts`` lay them out at 8
    pixels per latent pixel."""
    if size <= tile:
        return 1, size
    if overlap >= tile:
        raise ValueError(f"overlap {overlap} must be below the tile {tile}")
    n = math.ceil((size - overlap) / (tile - overlap))
    length = -(-(size + (n - 1) * overlap) // (8 * n)) * 8
    return n, min(tile, length)


def vae_peak(op: str, height: int, width: int, *, itemsize: int,
             channels: int = 128, batch: int = 1, tiled: bool = False,
             tile: int = TILE_PX, overlap: int = OVERLAP_PX,
             conv: str | None = None) -> int:
    """Footprint of one VAE ``op``, "decode" or "encode", of a ``batch``
    of ``height`` x ``width`` images on the ``conv`` path (the default
    path when None), under bounded_cache. A tiled pass follows
    ``VAE_TILED``."""
    if conv is None:
        from gmlx.image.vae.conv import CONV as conv
    fits = {"decode": VAE_DECODE, "encode": VAE_ENCODE}[op]
    fit = fits[conv][itemsize]
    if not tiled or (height <= tile and width <= tile):
        return fit.bytes(batch * height * width, channels, itemsize)
    ny, th = tile_layout(height, tile, overlap)
    nx, tw = tile_layout(width, tile, overlap)
    tiled_fit = VAE_TILED[op][itemsize]
    out = tiled_fit.bytes(batch * ny * nx * th * tw, batch * height * width,
                          channels, itemsize)
    if conv != "exact":
        one = batch * th * tw
        out += max(0, fit.bytes(one, channels, itemsize)
                   - fits["exact"][itemsize].bytes(one, channels, itemsize))
    return out


def free_bytes() -> float | None:
    """Room for new work now: the measured free working set, with the
    buffer cache counted as free, held to the serve ceiling. None when
    the device does not report it."""
    from gmlx.gen.prefill_decay import headroom_bytes
    from gmlx.serve.capacity import working_budget_bytes, working_set_bytes

    head = headroom_bytes()
    budget = working_budget_bytes()
    ws = working_set_bytes()
    if head is not None and budget is not None and ws is not None:
        head -= max(0.0, ws - budget)
    return head


def vae_tiling(op: str, height: int, width: int, *, itemsize: int,
               channels: int = 128, batch: int = 1,
               tiled: bool | None = None, tile: int = TILE_PX,
               overlap: int = OVERLAP_PX, conv: str | None = None,
               free: float | None = None,
               overcommit: bool | None = None) -> bool:
    """Whether a VAE ``op`` runs in tiles. ``tiled=None`` picks a whole
    pass when it fits the free working set and tiles otherwise. A pass
    that does not fit as chosen raises ImageMemoryError, unless
    GMLX_OVERCOMMIT=1. ``free`` defaults to ``free_bytes()``; when the
    device reports none, nothing is refused and None picks a whole pass.
    An image no larger than a tile always runs whole."""
    kw = dict(itemsize=itemsize, channels=channels, batch=batch, tile=tile,
              overlap=overlap, conv=conv)
    whole = vae_peak(op, height, width, **kw)
    can_tile = height > tile or width > tile
    in_tiles = vae_peak(op, height, width, tiled=True, **kw)
    if free is None:
        free = free_bytes()
    if free is None:
        return bool(tiled) and can_tile
    if tiled is None:
        choice = can_tile and whole > free
    else:
        choice = bool(tiled) and can_tile
    need = in_tiles if choice else whole
    if need <= free:
        return choice
    if overcommit is None:
        from gmlx.serve.capacity import overcommit as _oc

        overcommit = _oc()
    if overcommit:
        return choice
    how = "in tiles" if choice else "whole"
    if not choice and can_tile and in_tiles <= free:
        hint = f"a tiled pass needs {in_tiles / GiB:.2f} GiB"
    else:
        hint = "try a smaller image"
    raise ImageMemoryError(
        f"VAE {op} of {batch} x {height}x{width} {how} needs "
        f"{need / GiB:.2f} GiB, and {free / GiB:.2f} GiB is free; {hint}. "
        "GMLX_OVERCOMMIT=1 overrides.")


_cache_lock = threading.Lock()
_cache_depth = 0
_cache_saved: tuple | None = None


@contextmanager
def bounded_cache(limit: int | None = None):
    """Lower the MLX cache limit to ``limit`` (default CACHE_BOUND) for the
    block, so buffers a large evaluation frees go back to the system
    instead of the cache.

    The limit is process wide and a chat model may share the process, so
    the block only ever lowers it, and only for its own span. Nested and
    concurrent blocks share the first block's limit. The last block out
    restores the previous limit, unless other code set a new one while
    the block ran, which then stays."""
    import mlx.core as mx

    global _cache_depth, _cache_saved
    limit = CACHE_BOUND if limit is None else int(limit)
    with _cache_lock:
        if _cache_depth == 0:
            prev = mx.set_cache_limit(limit)
            if prev <= limit:
                mx.set_cache_limit(prev)
                _cache_saved = None
            else:
                _cache_saved = (prev, limit)
        _cache_depth += 1
    try:
        yield
    finally:
        with _cache_lock:
            _cache_depth -= 1
            if _cache_depth == 0 and _cache_saved is not None:
                prev, ours = _cache_saved
                _cache_saved = None
                now = mx.set_cache_limit(prev)
                if now != ours:
                    mx.set_cache_limit(now)


def gate_load(weight_bytes: float, model_id: str, *, what: str,
              hint: str, budget: float | None = None) -> None:
    """Gate a load on its header bytes before any tensor bytes are read.
    Weights above the working budget raise ImageMemoryError with ``hint``.
    Weights within it pass through ``capacity.preload_gate``, which defers
    a load that the measured free working set cannot take beside resident
    models. GMLX_OVERCOMMIT=1 skips both."""
    from gmlx.serve import capacity

    if capacity.overcommit() or weight_bytes <= 0:
        return
    if budget is None:
        budget = capacity.working_budget_bytes()
    if budget is not None and weight_bytes > budget:
        raise ImageMemoryError(
            f"The {what} weights of {weight_bytes / GiB:.2f} GiB exceed the "
            f"working budget of {budget / GiB:.2f} GiB. {hint}. "
            "GMLX_OVERCOMMIT=1 overrides.")
    capacity.preload_gate(float(weight_bytes), model_id)


def gate_weights(plan: MemoryPlan, model_id: str) -> None:
    """Gate the plan's weights with ``gate_load``, so resident text models
    are respected."""
    weights = {k: plan.terms[k] for k in
               ("dit", "encoder", "vae", "mmproj", "adapters")}
    gate_load(plan.weight_bytes, model_id, what="image model",
              hint=_largest(weights), budget=plan.budget)
