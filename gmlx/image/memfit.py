"""The memory plan of an image request, from header sizes and the request.

``plan_memory`` is a pure function. Resident bytes are the DiT, the text
encoder without its output matrix, the VAE at its decode dtype, the mmproj
for edits at the bytes its tensors hold once loaded, adapter factors and
caches. A request runs in three phases, encode, denoise and decode, and
each phase adds its own activation peak to what is resident then. A
request with an init image has a fourth, the VAE encode of that image,
between the encode and the denoise. The plan fits when the largest phase
stays within the working budget of ``gmlx.serve.capacity``.

A family that caches a prefix per conditioning adds three terms to the
denoise phase: the cache of every conditioning, the K and V of one block
joined over the prefix and the image rows in each step, and the prefill
pass over the prefix rows, which a conditioning runs once. The cache term
also holds the states of other conditionings that stay in the cache
beside the request's own, up to its bound. The cache is still held when
the VAE runs, so the VAE phases count it too.

A request runs a negative row whenever its guidance is not 1.0, as
``api.sample`` does, and the plan counts the second conditioning then.

A request with reference images encodes their image rows beside the text
and runs the encoder's vision tower, which the encode phase counts, and
its VAE encodes them one at a time in the phase of the init image. A
request that does not fit is refused with the largest count of its
references and the largest reference area that do.

A plan made before the parts load has a load phase too: what stays
resident and what the load holds beside it, from ``load_transient``.
Such a plan can also load in stages: the encoder loads with its mmproj,
encodes every conditioning of the request and is released, and the DiT
and the VAE load after that. The encoder is then never resident beside
the DiT. The staged order has a phase for the load of the encoder, the
encode phase with the encoder alone, and a load phase for the DiT and
the VAE.

A VAE pass runs whole when its phase fits the budget and in tiles when
it does not. Levers apply in order until the plan fits: sequential rows
in place of a batch, release of an owned encoder after encoding, which
frees its mmproj with it, tiled VAE decode, and last the staged load,
which a plan takes only when the load of every part at once, or the
encode beside the DiT, does not fit. Where the choice of
``vae_tiling`` is tiles on every machine, the decode is in tiles from the
start and tiling is no lever of the plan: on a VAE row that takes tiles
first, and on a measured row past the pixels its whole passes were
measured to. The encode of an init image is in tiles there too, and
otherwise whenever its phase does not fit whole, which is no lever
either, since a tiled encode gives the latents of a whole one. A plan
that still does not fit raises ``ImageMemoryError`` with each term in GiB
and the largest term named, unless GMLX_OVERCOMMIT=1.

The plan decides how a request's VAE passes run. ``MemoryPlan.vae_pass``
gives the choice and the bytes the plan leaves for the pass, and a caller
hands both to the guarded pass, so a pass the plan admits is not priced
again against a free figure measured later.

The MLX buffer cache keeps freed buffers for reuse up to its limit, which
defaults to 95 percent of memory. One VAE decode frees and allocates
buffers of many sizes inside one evaluation, so under the default limit
its footprint reaches two to three times its live peak. Image work runs
inside ``bounded_cache``, which lowers the limit to ``CACHE_BOUND`` for
the block, and the step and VAE peaks below are footprints measured that
way.

A VAE pass is priced by the row of its family in ``VAE_ROWS``, for the
plan and for ``vae.tiling`` alike. A measured row holds footprints fitted
on that VAE. An unmeasured row holds only what arithmetic gives, and
``unmeasured_bytes`` prices its passes by one rule. ``vae_tiling`` decides
between a whole and a tiled pass for a caller without a plan, and
refuses one that does not fit. On a row that takes tiles first, which
every unmeasured row does, and on any row when the device reports no free
working set, it never picks a whole pass by itself for an image larger
than one tile. A row may also bound its whole passes at the pixels the
largest measured one had: above that the whole price is the fit carried
on, and the choice is tiles.
"""

from __future__ import annotations

import math
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

GiB = 1 << 30


@dataclass(frozen=True)
class StepFit:
    """Footprint of one denoise step over the weights: ``base`` bytes plus
    ``k`` times the step formula of one row, the image rows times the
    widest outputs per row times the activation itemsize. Each further row
    the step evaluates with it adds ``row_k`` times the formula, ``k`` if
    None. ``read_k`` times the K and V of one block over the prefix rows
    the step reads adds to it, at most ``read_cap`` bytes."""
    base: float
    k: float
    row_k: float | None = None
    read_k: float = 0.0
    read_cap: float = 0.0

    def bytes(self, formula: float, rows: int = 1, read: float = 0) -> int:
        more = self.k if self.row_k is None else self.row_k
        return int(self.base + self.k * formula
                   + more * (rows - 1) * formula
                   + min(self.read_k * read, self.read_cap))


# One Z-Image-Turbo Q4_K_M step through ``api.sample`` under bounded_cache
# on an M5 Max, in fresh processes: 0.54, 1.28, 2.19 and 2.24 GiB over the
# settled process at 288, 1056, 4128 and 4608 rows. At 4128 rows the
# active peak is 1.05 GiB above the weights, and the rest is freed buffers
# the cache keeps up to its bound. A line in the formula bytes through the
# 1056 and 4128 row cells has a slope of 5.0. The slope is raised 10
# percent. A plan prices the most caption rows, and there the base leaves
# 0.25 GiB above every cell, the one of 1056 rows the closest.
Z_IMAGE_STEP = StepFit(base=1.03 * GiB, k=5.5)

# One Qwen-Image 2.1 Q4_K_M step through ``api.sample`` under bounded_cache
# on an M5 Max, in fresh processes, over the settled process and less the
# prefix states of its conditionings. The cells are in
# tests/fixtures/image/memfit/qwen_image21_cells.json: text requests of
# 256 to 4096 image rows with 36 to 1000 text rows, guided ones to 9216,
# and requests with one to four reference images of 256 to 1024 pixels,
# with and without guidance, on the stock ops and with the kernel routes
# in force.
# Two cells of 16384 image rows read 5.63 and 5.34 GiB. A line in the
# formula bytes through the 4096 and 16384 row cells has a slope of 3.6,
# raised 10 percent to 4.0 rounded up. The base is 0.25 GiB above every
# text cell on both paths.
# The second row of a guided step runs after the first in the same
# evaluation and reuses its freed buffers: one row at a time reads 0.07
# and 0.13 GiB under two rows together, 0.29 and 0.33 of the formula of
# a row. The slope of the second row is 0.5.
# On the stock ops a step that reads a long prefix keeps more freed
# buffers in the cache: up to 0.88 GiB over the line above, at 5096
# prefix rows and 1600 image rows, where it is 11.3 times the K and V of
# one block over the prefix rows. The cache bound holds it past that, and
# at 16896 prefix rows it is gone. The term is 16 times, at
# most 1 GiB, which leaves 0.25 GiB over every cell. The kernel routes
# read at most 0.31 GiB over the line, and the one fit prices both.
QWEN_IMAGE21_STEP = StepFit(base=1.25 * GiB, k=4.0, row_k=0.5, read_k=16.0,
                            read_cap=1.0 * GiB)

# The first step of a Qwen-Image 2.1 request comes after the prefill of
# its conditionings, whose freed buffers the cache still holds. In the
# same cells a prefill alone reads 2.42 GiB above its state at two
# 1024 pixel references on the stock ops and 3.80 GiB at four, and the
# first step after one reads up to 0.3 GiB above a later step. In the formula bytes of the
# prefix rows the tightest cell, one 1024 pixel reference and 512 text
# rows at 4096 image rows on the stock ops, needs a slope of 5.9 over the
# step's base for 0.25 GiB of room. The slope is raised 10 percent.
QWEN_IMAGE21_PREFILL = StepFit(base=1.25 * GiB, k=6.5)

# One Krea 2 Q4_K_M step through ``api.sample`` under bounded_cache on an
# M5 Max, in fresh processes, over the settled process. The cells are in
# tests/fixtures/image/memfit/krea2_cells.json: 256 to 1536 pixels with 36
# or 512 text rows, two of them guided, and two of a Q8_0 DiT. With 512
# text rows the cache reaches its bound in the first step at every size,
# so a 256 pixel step reads 1.94 GiB. The cache term does not grow with
# the size: a 768 or 896 pixel step reads more than a 1024 pixel step,
# 2.62 and 2.76 against 2.36 GiB, as fewer of the cached buffers are of
# a size the step takes again. With the cache off the steps read 1.07
# and 1.11 GiB. A line through the 1280 and 1536 pixel cells has a slope
# of 2.2, raised 10 percent to 2.4. The base is 0.25 GiB above every cell,
# the Q8_0 768 pixel cell the closest. A guided step reads as
# Qwen-Image 2.1's does, and takes its slope of 0.5 for the second row.
KREA2_STEP = StepFit(base=2.35 * GiB, k=2.4, row_k=0.5)

# One FLUX.3 klein Q4_K_M step through ``api.sample`` under bounded_cache
# on an M5 Max, in fresh processes, over the settled process. The cells
# are in tests/fixtures/image/memfit/flux3_klein_cells.json: 256 to 2048
# pixels with 80 text rows, 256 to 1536 with 4096, one with 640, steps
# with one to four reference images, and four guided steps of the Base
# file. The cache term reaches its bound early: a 256 pixel step reads
# 1.42 GiB with 80 text rows and 2.78 with 4096. A line through the 1536
# and 2048 pixel cells has a slope of 2.7, raised 10 percent to 3.0. The
# base is 0.25 GiB above every cell, the 512 pixel cell with 4096 text
# rows the closest. The second row of a guided step reads 0.41 to 0.79
# GiB above one row, so it takes the slope of 1.0, which leaves 0.25 GiB
# above every guided cell where 0.5 does not.
FLUX3_KLEIN_STEP = StepFit(base=2.45 * GiB, k=3.0, row_k=1.0)

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
# and the fp32 decode 7.01 and 10.42 GiB, against 8.44 and 11.55.
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
    part run whole and the buffers of the result."""
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

# Qwen-Image 2.1 autoencoder footprints under bounded_cache on the official
# file, over the loaded weights, by conv path and activation itemsize.
# Whole passes on an M5 Max, in fresh processes with a cold and a warm
# pass, fit a line in pixels by least squares. The slope is raised 10
# percent and the base covers every whole pass plus 0.25 GiB. A pass is
# the larger of the growth of the process footprint and of the MLX peak
# over the active bytes at its start. The footprint is concave in pixels,
# so a line through square passes at 256, 512 and 1024 pixels reads below
# a pass between them: the exact path is fit over every pass measured from
# 256 to 1024 pixels, square and not, 13 bf16 decodes, 11 bf16 encodes,
# and 5 and 4 in fp32. The Winograd path is off by default and has its
# three square passes. At 1024 the bf16 decode added 5.50 GiB exact and
# 7.83 GiB on Winograd, against 6.75 and 8.87 here, and the fp32 decode
# 8.74 and 14.00 GiB, against 10.17 and 15.40. No whole pass above 1024
# pixels has run, so a whole price above that is the line carried on.
QWEN_VAE_DECODE = {"exact": {2: VaeFit(base=1.58 * GiB, k=18.4),
                             4: VaeFit(base=2.69 * GiB, k=13.3)},
                   "winograd": {2: VaeFit(base=2.09 * GiB, k=24.1),
                                4: VaeFit(base=2.29 * GiB, k=23.3)}}
QWEN_VAE_ENCODE = {"exact": {2: VaeFit(base=0.74 * GiB, k=15.5),
                             4: VaeFit(base=1.72 * GiB, k=9.9)},
                   "winograd": {2: VaeFit(base=1.48 * GiB, k=15.5),
                                4: VaeFit(base=1.57 * GiB, k=12.7)}}

# The mid attention of that autoencoder holds its scores whatever chunk
# ``chunked_sdpa`` cuts the rows into: at 16384 rows, a 2048 pixel image,
# the MLX peak on the scores alone is 1.46 times the bytes of every score
# in bf16 and 1.18 times in fp32, with 4 chunks as with 16. A pass is
# priced with 1.5 times rows squared times the itemsize on top of its fit.
# Read at one row count a dtype, and nothing above 2048 pixels has run.
QWEN_VAE_SCORES = 1.5

# Tiled passes of that autoencoder hold one tile at a time beside the part
# run whole. The tile term is the whole fit on the exact path at one tile,
# and the image term is by arithmetic. A decode runs conv_in, the mid block
# and the first up block whole and holds their output while the tiles run,
# 18 values a pixel, beside 36 bytes a pixel for the float32 result on the
# host and on the device: 72 bytes a pixel in bf16 and 108 in fp32. Its
# tiles are cropped and overlap by 224 pixels, so a side of 1024 pixels
# takes three tiles of 496, 1536 five and 2048 seven. With the scores on
# top, a tiled bf16 decode added 2.65 GiB at 1024 pixels, 2.86 at 1536 and
# 3.24 at 2048, against 2.91, 3.19 and 3.83 here, and in fp32 4.36, 4.49
# and 4.55 against 4.65, 5.16 and 6.37: the scores and a tile do not peak
# together, and the sum counts both. An encode holds 36 bytes a pixel, and
# its tiles overlap by 320 pixels, four of 496 pixels a side at 1024 and
# nine of 512 at 2048. In bf16 it added 1.18 GiB at 1024 pixels, 1.48 at
# 1536 and 1.73 at 2048, against 1.50, 1.74 and 2.36 here, and in fp32
# 2.07, 2.70 and 3.39 against 2.72, 3.15 and 4.30. A Winograd pass adds the
# difference of the whole rows at one tile, and no tiled pass has run on
# that path.
QWEN_VAE_TILED = {"decode": {2: TiledFit(base=1.58 * GiB, k_tiles=18.4,
                                         k_image=0.25),
                             4: TiledFit(base=2.69 * GiB, k_tiles=13.3,
                                         k_image=0.19)},
                  "encode": {2: TiledFit(base=0.74 * GiB, k_tiles=15.5,
                                         k_image=0.19),
                             4: TiledFit(base=1.72 * GiB, k_tiles=9.9,
                                         k_image=0.10)}}

# Footprint over op-order floor of a pass on an unmeasured row, by conv
# path. Both numbers are borrowed from another model, the FLUX.1
# autoencoder, because a floor alone is no footprint. Its whole bf16
# decode at 1024 pixels measured 4.45 GiB on the exact path and 6.64 GiB
# on Winograd, against a floor of 1.25 GiB, five of its 128 channel
# activations at full resolution. That is 3.56 and 5.31 times the floor,
# rounded up here. Its fp32 decode gave 2.80 and 4.17 times.
#
# The Qwen-Image 2.1 autoencoder was priced by this rule until its file
# was measured. Its own ratio, the footprint less the weights and the
# cache bound over its floor, came to 2.18 in bf16 and 2.05 in fp32 for
# the whole decode at 1024 pixels on the exact path, and to 3.56 and 3.61
# on Winograd. Over its whole cells from 256 to 1024 pixels the largest
# was 2.93 exact and 4.95 on Winograd, so the borrowed numbers held.
UNMEASURED_RATIO = {"exact": 3.6, "winograd": 5.4}


@dataclass(frozen=True)
class VaeMemoryRow:
    """How the VAE of a family is priced.

    ``measured`` says whether the row holds footprints fitted on this VAE.
    A measured row has ``whole`` and ``tiled``, in bytes per pixel, base
    channel and activation byte. An unmeasured row has ``floor`` alone,
    the bytes a pass holds by arithmetic, and ``unmeasured_bytes`` prices
    the pass from them. The pass that measures a VAE sets its fits and
    ``measured`` together, and may keep the floor beside them.

    ``one_tile`` says that a tiled pass holds one tile at a time, so the
    tile term of its fit counts the pixels of one tile and not of all.
    ``scores`` is what the mid attention holds of its scores, as a factor
    on the bytes of every score of one image, which is the latent pixels
    squared times the itemsize. It is added to a measured price, whole or
    tiled, and is 0 for a VAE whose fits cover its attention.
    ``crop`` gives, for an op and a config, the latent pixels over which a
    tile's result reads its input when the tiles of that op are cropped
    and not blended, and 0 when they are blended. Cropped tiles overlap
    by at least twice that.
    ``overlap`` is the least overlap of blended tiles in pixels, or 0 for
    the overlap asked for. A tiled pass takes the larger of the two.
    ``tiles_first`` makes tiles the automatic choice for every image
    larger than one tile, whatever is free. An unmeasured row always
    takes tiles first.
    ``whole_px`` is the pixel count of the largest whole pass the fits
    were measured on, or 0 for no bound. A pass of more pixels, counted
    over its batch, is in tiles by the automatic choice whatever is free,
    since its whole price is the fit carried past what was measured. A
    caller can still force it whole.
    ``scale`` is the image pixels per latent pixel of the official VAE,
    or 0 when that is the family's ``vae_scale_factor``. A family whose
    model space packs the VAE's latent pixels counts the packing in that
    factor, and a tiled pass lays its tiles out at the VAE's own scale."""
    measured: bool
    channels: dict                  # op -> base channels at full resolution
    whole: dict | None = None       # op -> conv path -> itemsize -> VaeFit
    tiled: dict | None = None       # op -> itemsize -> TiledFit
    floor: Callable | None = None   # see ``_qwen_image21_vae_floor``
    one_tile: bool = False
    tiles_first: bool = False
    crop: Callable | None = None    # see ``_qwen_image21_vae_crop``
    overlap: int = 0
    scores: float = 0.0
    whole_px: int = 0
    scale: int = 0

    @property
    def auto_tiles(self) -> bool:
        """Whether the automatic choice above one tile is tiles on every
        machine, at every size."""
        return self.tiles_first or not self.measured

    def tiles_at(self, pixels: int) -> bool:
        """Whether the automatic choice for a pass of ``pixels`` pixels
        above one tile is tiles on every machine."""
        return self.auto_tiles or bool(self.whole_px
                                       and pixels > self.whole_px)


def _qwen_image21_vae_floor(op: str, height: int, width: int, *,
                            config=None, **kw):
    """The steps of a Qwen-Image 2.1 VAE pass, each (name, device bytes,
    softmax bytes, host bytes), and the weight count, for the VAE of
    ``config`` or the official file's when None. A tiled pass holds one
    tile at a time, in tiles of whole latent pixels."""
    from .vae import qwen_image21_memory as m

    cfg = m.OFFICIAL if config is None else config
    return (m.memory_terms(cfg, op, height, width, **kw),
            m.weight_count(cfg))


def _qwen_image21_vae_crop(op: str, config=None) -> int:
    """The reach of a cut in a tiled pass of the Qwen-Image 2.1 VAE, whose
    tiles are cropped in a decode and in an encode."""
    from .vae import qwen_image21_memory as m
    from .vae.qwen_image21_vae import reach

    return reach(m.OFFICIAL if config is None else config, op)


def _qwen_image_vae_floor(op: str, height: int, width: int, *,
                          config=None, **kw):
    """The steps of a Qwen-Image VAE pass and the weight count, as
    ``_qwen_image21_vae_floor`` gives them."""
    from .vae import qwen_image_memory as m

    cfg = m.OFFICIAL if config is None else config
    return (m.memory_terms(cfg, op, height, width, **kw),
            m.weight_count(cfg))


def _qwen_image_vae_crop(op: str, config=None) -> int:
    """The reach of a cut in a tiled pass of the Qwen-Image VAE, whose
    tiles are cropped in a decode and in an encode."""
    from .vae import qwen_image_memory as m
    from .vae.qwen_image_vae import reach

    return reach(m.OFFICIAL if config is None else config, op)


VAE_ROWS = {
    # The FLUX.1 autoencoder.
    "z_image": VaeMemoryRow(
        measured=True, channels={"decode": 128, "encode": 128},
        whole={"decode": VAE_DECODE, "encode": VAE_ENCODE}, tiled=VAE_TILED),
    # The Qwen-Image 2.1 autoencoder, measured on the official file. A
    # pass runs whole when that fits and in tiles when it does not: a tiled
    # pass gives the result of the whole one, and at 1024 pixels a tiled
    # bf16 decode runs 9 tiles of 496 pixels, 2.1 times the pixels. A
    # whole bf16 decode at 1024 pixels is priced at 6.80 GiB, which a 16
    # GB Mac does not have beside the weights, and a tiled one at 2048
    # pixels at 3.83 GiB, which it has. No whole pass has run above 1024
    # pixels, so above that the automatic choice is tiles.
    "qwen_image21": VaeMemoryRow(
        measured=True, channels={"decode": 144, "encode": 96},
        whole={"decode": QWEN_VAE_DECODE, "encode": QWEN_VAE_ENCODE},
        tiled=QWEN_VAE_TILED, floor=_qwen_image21_vae_floor, one_tile=True,
        crop=_qwen_image21_vae_crop, scores=QWEN_VAE_SCORES,
        whole_px=1024 * 1024),
    # The Qwen-Image autoencoder of Krea 2, priced by the rule for an
    # unmeasured VAE: tiles first, one at a time, cropped.
    "krea2": VaeMemoryRow(
        measured=False, channels={"decode": 96, "encode": 96},
        floor=_qwen_image_vae_floor, one_tile=True,
        crop=_qwen_image_vae_crop),
    # The FLUX.2 autoencoder: the FLUX.1 encoder and decoder at 32 latent
    # channels, which change no activation at full resolution, so it is
    # priced by the FLUX.1 fits. Measured on the official file on an M5
    # Max, each pass grew the footprint by less than its price: a whole
    # bf16 decode at 1024 pixels by 4.49 GiB against 5.61, in fp32 by 7.24
    # against 8.45, a tiled bf16 decode at 2048 pixels by 8.25 against
    # 9.30, a whole bf16 encode at 1024 by 2.93 against 4.20, and a tiled
    # one at 2048 by 5.82 against 6.58. The closest was the whole fp32
    # decode at 512 pixels, 3.73 against 4.02.
    # Its tiles overlap by at least 128 pixels. Its decoder reads farther
    # across a cut than the FLUX.1 one: at 1024 pixels in tiles that
    # overlap by 64, a tiled fp32 decode is 52.0 dB from the whole decode,
    # where the FLUX.1 decoder holds 75.4 dB. At 128 it is 64.8 dB, and a
    # tiled bf16 decode is 55.5 dB from the oracle against 55.9 whole.
    # Its latent pixel is 8 image pixels; the family's 16 counts the 2x2
    # packing of model space.
    "flux3_klein": VaeMemoryRow(
        measured=True, channels={"decode": 128, "encode": 128},
        whole={"decode": VAE_DECODE, "encode": VAE_ENCODE}, tiled=VAE_TILED,
        overlap=128, scale=8),
}


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
    slot_tokens: int = 4            # reference tokens per image row of the
                                    # encoder
    min_ref_px: int = 0             # the smallest reference size, the side
                                    # of the square of a reference's area

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
    prefix: PrefixRow | None = None
    prefill: StepFit | None = None  # prefill footprint, the step's if None
    encode_row_bytes: int = 0       # encoder activation bytes per token,
                                    # times ENCODE_LIVE
    text_rows: int = 512
    encode_tower_bytes: int = 0     # an encode with reference images holds
                                    # this beside its rows, for the tower
    mmproj_fp32: bool = False       # the encoder's load casts the float
                                    # tensors of its mmproj to float32
    encode_base: int = 0            # bytes an encode holds at any length
    ref_token_px: int = 0           # pixels per reference token on each
                                    # axis when the reference tokens join
                                    # the rows of every step, else 0
    prices_prompt: bool = True      # a plan that knows the prompt prices
                                    # its text rows, else the family's most

    def step_peak(self, step: int, prefill: int, rows: int = 1,
                  read: int = 0) -> int:
        """Bytes of the wider of a step and a prefill, from the formula
        bytes of each: ``step`` of one row of the ``rows`` a step
        evaluates together, which read ``read`` bytes of one block's
        prefix K and V."""
        return max(self.step.bytes(step, rows, read),
                   (self.prefill or self.step).bytes(prefill))


PEAKS = {
    # Z-Image prices the most caption rows whatever the prompt. At its own
    # rows a step of a short caption is priced 0.10 GiB above the cell of
    # 1056 rows and 0.19 GiB above the cell of 4128 rows, where the most
    # rows leave 0.25 and 0.35 GiB.
    "z_image": PeakRow(token_px=16, row_multiple=32, caption_rows=512,
                       width=11520 + 2 * 10240, step=Z_IMAGE_STEP,
                       encode_row_bytes=3 * 9728 * 2, prices_prompt=False),
    # A step runs the target rows alone: the fused MLP product and its
    # gated half are the widest outputs. The sizes are those the
    # fingerprint holds the file to: 32 blocks, 4096 wide, a 12288 MLP.
    # A prefix state measures exactly its cache bytes. No term counts the
    # attention scores, heads x queries x keys, since the stock attention
    # op does not hold them on the machine measured: alone, at the shape
    # of a step, it peaks at its output, 0.03 GiB at 1024x1024 and
    # 0.13 GiB at 2048x2048, where the scores are 1.1 and 16.5 GiB in
    # bf16. That is one machine, an M5 Max. With two 1024 pixel references
    # under guidance the scores of a step are 6.3 GiB, and the step reads
    # 3.5 GiB above its states. A reference token holds exactly its cache
    # bytes, 2 x 32 x 4096 elements, per conditioning. An encode with
    # reference images reads up to 0.76 GiB above the encoder at two 512
    # pixel references and 1.31 GiB at two of 1024 pixels with 600 text
    # rows, the vision tower's pass over one image being the peak.
    "qwen_image21": PeakRow(token_px=16, row_multiple=1, caption_rows=0,
                            width=24576 + 12288, step=QWEN_IMAGE21_STEP,
                            prefix=PrefixRow(blocks=32, dim=4096,
                                             token_px=16, min_ref_px=256),
                            prefill=QWEN_IMAGE21_PREFILL,
                            encode_row_bytes=3 * 12288 * 2,
                            encode_tower_bytes=int(1.5 * GiB),
                            mmproj_fp32=True),
    # Krea 2 joins up to 512 text rows to the image rows in every block. The
    # MLP's gate and up products and the attention's query and gate are the
    # widest outputs: 28 blocks, 6144 wide, a 16384 MLP. The encoder's tap
    # runs in float32.
    "krea2": PeakRow(token_px=16, row_multiple=1, caption_rows=512,
                     width=2 * 16384 + 2 * 6144, step=KREA2_STEP,
                     encode_row_bytes=3 * 9728 * 4, text_rows=546),
    # FLUX.3 klein joins up to 4096 text rows, padded to 80, to the image
    # rows in every block, and the tokens of each reference image too.
    # The MLP's fused gate and up product and its gated half are the
    # widest outputs: 3072 wide, a 9216 MLP. A plan that knows the prompt
    # prices its text rows, and one that does not the most. An encode of
    # the encoder alone, Qwen3-VL-4B Q8_0 tapped in float32, reads 0.34
    # GiB over the settled process at 80 rows, 0.50 at 640, 1.19 at 2080
    # and 1.33 at 4096: a base of 0.6 GiB and 28 KiB times ENCODE_LIVE a
    # row leave 0.25 GiB above every cell, where Krea 2's rows alone would
    # price 7.1 GiB at 4096.
    "flux3_klein": PeakRow(token_px=16, row_multiple=1, caption_rows=4096,
                           width=2 * 9216 + 9216, step=FLUX3_KLEIN_STEP,
                           encode_row_bytes=28 * 1024,
                           encode_base=int(0.6 * GiB), text_rows=4096,
                           ref_token_px=16),
}

LEVERS = ("sequential rows", "release the encoder", "tiled decode",
          "staged load")


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
    tiled_encode: bool = False      # the VAE encode of an init image
    staged_load: bool = False       # the encoder goes before the DiT loads

    @property
    def peak(self) -> int:
        return max(self.phases.values())

    @property
    def weight_bytes(self) -> int:
        return sum(self.terms[k] for k in
                   ("dit", "encoder", "vae", "mmproj", "adapters"))

    def vae_pass(self, op: str) -> tuple[bool, float | None]:
        """How the plan runs the VAE ``op`` of its request, "decode" or
        the "encode" of an init image: whether it is in tiles, and the
        bytes the plan leaves for the pass, which are the budget less
        what its phase holds beside the pass. None for no budget."""
        phase, term, tiled = {
            "decode": ("decode", "decode_peak", self.tiled_decode),
            "encode": ("init", "init_peak", self.tiled_encode)}[op]
        if phase not in self.phases:
            raise ValueError(f"this plan has no VAE {op}")
        if self.budget is None:
            return tiled, None
        return tiled, self.budget - (self.phases[phase] - self.terms[term])

    def describe(self) -> str:
        parts = [f"{k} {v / GiB:.2f} GiB" for k, v in self.terms.items() if v]
        phases = ", ".join(f"{k} {v / GiB:.2f}" for k, v in
                           self.phases.items())
        peak, budget = f"{self.peak / GiB:.2f}", "unknown"
        if self.budget is not None:
            budget = f"{self.budget / GiB:.2f} GiB"
            if self.peak > self.budget:
                # Rounded apart, so a refusal never reads as equal.
                peak = f"{math.ceil(self.peak / GiB * 100) / 100:.2f}"
                budget = (f"{math.floor(self.budget / GiB * 100) / 100:.2f}"
                          " GiB")
        levers = ", ".join(self.levers) or "none"
        return (f"peak {peak} GiB against a budget of "
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


FLOAT_CODECS = {"F16": 2, "BF16": 2, "F32": 4}


def mmproj_bytes(src, *, fp32: bool = False) -> int:
    """Bytes an mmproj holds once loaded, from a header view, or ``src``
    itself when it is a number. With ``fp32`` the load casts its float
    tensors to float32, as the Qwen3-VL vision tower's does, so a 16-bit
    tensor is resident at twice its file bytes. Other tensors stay as the
    file stores them."""
    if src is None:
        return 0
    if isinstance(src, (int, float)):
        return int(src)
    total = 0
    for t in src.tensors.values():
        if fp32 and t.codec in FLOAT_CODECS:
            total += math.prod(int(d) for d in t.shape) * 4
        else:
            total += t.nbytes or tensor_bytes(t.shape, t.codec)
    return total


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


# What a load holds over what stays resident, read on the Qwen-Image 2.1
# files on an M5 Max, each part in a fresh process: the encoder's output
# matrix until it is dropped, 0.48 GiB; with an mmproj whose 16-bit
# tensors load as float32, the file's 1.08 GiB beside the cast, 1.55 GiB
# in all; a VAE's file tensors beside their copies in the decode dtype and
# layout, 1.33 GiB from a float32 GGUF to bf16 and 1.20 to float32, and
# 0.71 and 0.77 GiB from safetensors. The bytes named are the output
# matrix, the mmproj file and the VAE's tensors at float32, 0.48, 1.08
# and 1.26 GiB there, and the term is those times this.
LOAD_ROOM = 1.1
# A plan takes the staged order when its budget is nearly full, so each
# of its two load phases holds this over its load term: the 0.25 GiB room
# every measured step, VAE and load cell keeps over its reading. The terms
# alone are 0.04 to 0.06 GiB over the readings of the Qwen-Image 2.1
# encoder, with and without its mmproj, and of the DiT and VAE, loaded in
# that order in one process.
STAGE_ROOM = GiB // 4


_ACT_CODECS = {"bfloat16": "BF16", "bf16": "BF16", "float16": "F16",
               "fp16": "F16"}


def cast_bytes(src, act) -> int:
    """File bytes of the 16-bit float tensors of a DiT that its load casts
    to the activation dtype ``act``, from a header view. Each is held
    beside its copy until the load ends: a Qwen-Image 2.1 F16 DiT under
    bf16 activations peaks at twice its weights. A number or None counts
    nothing."""
    if src is None or isinstance(src, (int, float)):
        return 0
    keep = _ACT_CODECS[_dtype_name(act)]
    return sum(t.nbytes or tensor_bytes(t.shape, t.codec)
               for t in src.tensors.values()
               if t.codec in ("F16", "BF16") and t.codec != keep)


def dit_transient(dit, act) -> int:
    """What the load of a DiT holds over what stays resident, as
    ``load_transient`` counts the other parts: its cast tensors."""
    return int(LOAD_ROOM * cast_bytes(dit, act))


def load_transient(encoder=None, vae=None, mmproj=None) -> int:
    """Bytes the load of an image pipeline holds at its peak over what
    stays resident, from the header views of the encoder and its mmproj
    and the header of the VAE. The parts load one after the other, so
    the figure is the larger of the encoder's and the VAE's. A part given
    as a number, or not at all, counts nothing."""
    head = 0
    if encoder is not None and not isinstance(encoder, (int, float)):
        head = file_bytes(encoder) - file_bytes(encoder,
                                                skip=("output.weight",))
        if mmproj is not None and not isinstance(mmproj, (int, float)):
            head += file_bytes(mmproj)
    copies = 0
    if vae is not None and not isinstance(vae, (int, float)):
        copies = vae_bytes(vae, 4)
    return int(LOAD_ROOM * max(head, copies))


def prices_prompt(family: str) -> bool:
    """Whether a plan of ``family`` that knows the prompt prices its text
    rows, where one that does not prices the family's most."""
    row = PEAKS.get(family)
    return row is not None and row.prices_prompt


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
        "init_peak": "a smaller init image",
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
                ref_px: int = 0, prefix_rows=None,
                prefix_shared: bool = False, prefix_other: int = 0,
                prefix_bound: int = 0, cache_bytes: int = 0,
                tile: int = TILE_PX, init_image: bool = False,
                ref_sizes=(), step_refs=None, vae_price=None,
                load_peak: int = 0, load_parts=None,
                budget: float | None = None, overcommit: bool | None = None,
                _hint: bool = True) -> MemoryPlan:
    """The memory plan for one image request.

    ``dit``, ``encoder``, ``mmproj`` and ``vae`` are header views or byte
    counts. A header view of an mmproj is counted at what it holds once
    loaded, as ``mmproj_bytes`` gives it for the family, and a byte count
    as it stands. The mmproj is part of the encoder: an owned encoder
    that is released frees both. ``adapters`` is a byte count or a list of
    them. ``budget`` defaults to ``capacity.working_budget_bytes()`` and
    ``overcommit`` to GMLX_OVERCOMMIT.

    ``init_image`` says the request encodes an init image of its size
    with the VAE, and ``ref_sizes`` holds the (height, width) in pixels
    of each reference image it encodes. The encodes run one at a time,
    after the encoder's release when that lever is taken, and the init
    phase holds the largest. ``vae_price`` maps "decode" and "encode" to what
    ``vae_peak`` takes of the loaded VAE beside the family and the
    itemsize: its channels, config and conv path, as
    ``tiling.price_args`` gives them. Without it the passes are priced
    for the family's official file on the default conv path.

    The prefix rows of a conditioning come from ``text_tokens``, ``refs``
    and ``ref_px``. With ``refs`` at 0 the count takes the references of
    ``ref_sizes`` at their own sizes. A family without a prefix whose
    reference tokens join the rows of every step counts them there, by
    the same rule, or those of ``step_refs`` when it is given: the
    (height, width) of every reference of one conditioning, an image given
    twice counted twice, where ``ref_sizes`` holds each image the VAE
    encodes once. ``prefix_rows`` gives the rows of the request's own
    conditionings, the positive first, in place of that count.
    ``prefix_shared`` says the two of a guided request are one, whose
    state is held once. A cache of prefix states that holds
    ``prefix_other`` bytes of other conditionings sheds them to
    ``prefix_bound`` as the request's states come in, and what stays is
    part of the prefix cache term.

    ``load_peak`` is what the load of the parts holds over what stays
    resident, as ``load_transient`` gives it from their headers. With it
    the plan has a load phase, which a plan for parts that are loaded
    already leaves out. ``load_parts`` is that figure for the encoder
    with its mmproj and for the VAE, each loaded alone. With it an owned
    encoder can load, encode and be released before the DiT and the VAE
    load, which is the last lever of the plan, and each of the two load
    phases counts STAGE_ROOM over its figure. The adapters load with the
    encoder then, whose halves of them the encode runs under.
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
        "mmproj": mmproj_bytes(mmproj, fp32=row.mmproj_fp32),
        "adapters": (int(adapters) if isinstance(adapters, (int, float))
                     else sum(int(a) for a in adapters)),
        "caches": int(cache_bytes),
        "prefix_cache": 0,
    }
    rows = image_rows(row, height, width, caption_tokens)
    if row.ref_token_px:
        # The reference tokens join the rows of every step.
        p = row.ref_token_px
        rows += sum((int(h) // p) * (int(w) // p) for h, w in (
            [(ref_px, ref_px)] * refs if refs else
            ref_sizes if step_refs is None else step_refs))
    text = row.text_rows if text_tokens is None else text_tokens
    counted = slots = 0
    if row.prefix is not None:
        counted = row.prefix.rows(text, refs, ref_px)
        if not refs:
            # The references of the request by their own sizes.
            counted += sum((int(h) // row.prefix.token_px)
                           * (int(w) // row.prefix.token_px)
                           for h, w in ref_sizes)
        # The image rows the encoder runs beside the text.
        slots = (counted - text) // row.prefix.slot_tokens
    given = [] if prefix_rows is None else [int(n) for n in prefix_rows]
    mine = (given + [counted] * conds)[:conds]
    per = [prefix_row_terms(row, rows, n, act) for n in mine]
    own = per[0][0] if prefix_shared else sum(p[0] for p in per)
    step_one, prefill = max(p[1] for p in per), max(p[2] for p in per)
    read = (0 if row.prefix is None
            else row.prefix.join_bytes(max(mine), act))
    terms["prefix_cache"] = own + min(int(prefix_other),
                                      max(0, int(prefix_bound) - own))
    def price(op: str, tiled: bool, h: int = height, w: int = width) -> int:
        kw = dict(family=family, itemsize=vsize, tile=tile)
        kw.update({k: v for k, v in ((vae_price or {}).get(op) or {}).items()
                   if k in ("channels", "config", "conv")})
        return int(vae_peak(op, h, w, tiled=tiled, **kw))

    # Above one tile a pass can run in tiles. The choice ``vae_tiling``
    # makes by itself is tiles on a row that takes tiles first, and past
    # the pixels a row's whole passes were measured to.
    can_tile = height > tile or width > tile
    auto_tiled = can_tile and vae_row(family).tiles_at(height * width)
    decode = {False: price("decode", False), True: price("decode", True)}
    # The images the VAE encodes: (pixels, tiles of itself, bytes whole,
    # bytes in tiles or None for an image of one tile).
    encodes = []
    for h, w in ([(height, width)] if init_image else []) + [
            (int(h), int(w)) for h, w in ref_sizes]:
        tiles = h > tile or w > tile
        encodes.append((tiles and vae_row(family).tiles_at(h * w),
                        price("encode", False, h, w),
                        price("encode", True, h, w) if tiles else None))
    caption = row.text_rows if caption_tokens is None else caption_tokens
    # An encode with reference images runs their image rows beside the
    # text, and the tower.
    encode_peak = (row.encode_base
                   + (caption + slots) * row.encode_row_bytes * ENCODE_LIVE
                   + (row.encode_tower_bytes if slots else 0))

    def build(b: int, release: bool, tiled: bool, levers: tuple,
              staged: bool = False) -> MemoryPlan:
        t = dict(terms)
        t["step_peak"] = row.step_peak(step_one, prefill, b, read)
        t["decode_peak"] = decode[tiled]
        t["encode_peak"] = int(encode_peak)
        resident = sum(t[n] for n in ("dit", "encoder", "vae", "mmproj",
                                      "adapters", "caches"))
        # An owned encoder that is released takes its mmproj with it.
        later = resident - (t["encoder"] + t["mmproj"] if release else 0)
        # The prefix states stay in their cache until the request ends.
        held = later + t["prefix_cache"]
        phases = {}
        if staged:
            # The encoder loads, encodes and goes, and the DiT and the VAE
            # load after it.
            first = sum(t[n] for n in ("encoder", "mmproj", "adapters",
                                       "caches"))
            t["load_encoder_peak"], t["load_peak"] = (
                int(v) + STAGE_ROOM for v in load_parts)
            phases["load_encoder"] = first + t["load_encoder_peak"]
            phases["encode"] = first + t["encode_peak"]
            phases["load"] = later + t["load_peak"]
        else:
            if load_peak:
                t["load_peak"] = int(load_peak)
                phases["load"] = resident + t["load_peak"]
            phases["encode"] = resident + t["encode_peak"]
        tiled_init = False
        if encodes:
            # A tiled encode is the whole one, so it is a choice by what
            # its phase holds and no lever. Every image above one tile
            # runs in tiles when a whole pass of one does not fit.
            forced = budget is not None and any(
                held + whole > budget for own, whole, cut in encodes
                if cut is not None and not own)
            peaks = [cut if cut is not None and (own or forced) else whole
                     for own, whole, cut in encodes]
            tiled_init = any(cut is not None and (own or forced)
                             for own, _, cut in encodes)
            t["init_peak"] = max(peaks)
            phases["init"] = held + t["init_peak"]
        phases["denoise"] = held + t["step_peak"]
        phases["decode"] = held + t["decode_peak"]
        fits = budget is None or max(phases.values()) <= budget
        return MemoryPlan(terms=t, phases=phases, budget=budget, rows=b,
                          release_encoder=release, tiled_decode=tiled,
                          levers=levers, fits=fits, tiled_encode=tiled_init,
                          staged_load=staged)

    b = 2 if guided and batch_rows else 1
    plan = build(b, False, auto_tiled, ())
    steps = []
    if b == 2:
        steps.append(("sequential rows", dict(b=1)))
    if encoder_owned and (terms["encoder"] or terms["mmproj"]):
        steps.append(("release the encoder", dict(release=True)))
    if can_tile and not auto_tiled:
        steps.append(("tiled decode", dict(tiled=True)))
    if load_parts is not None and encoder_owned and (terms["encoder"]
                                                     or terms["mmproj"]):
        steps.append(("staged load", dict(staged=True, release=True)))
    state = dict(b=b, release=False, tiled=auto_tiled, staged=False)
    applied: tuple = ()
    for name, change in steps:
        if plan.fits:
            break
        state.update(change)
        applied += (name,)
        plan = build(state["b"], state["release"], state["tiled"], applied,
                     state["staged"])
    if plan.fits or overcommit:
        return plan
    hint = _largest(plan.terms)
    if not encoder_owned and (terms["encoder"] or terms["mmproj"]):
        hint += ("; the encoder is not released in this request and stays "
                 "resident")
    sizes = ([(ref_px, ref_px)] * refs if refs
             else [(int(h), int(w)) for h, w in ref_sizes])
    if _hint and sizes and row.prefix is not None and budget is not None:
        def fits(k: int, scale: float) -> bool:
            kw = dict(refs=k, ref_px=_scaled(ref_px, scale)) if refs else dict(
                ref_sizes=[(_scaled(h, scale), _scaled(w, scale))
                           for h, w in sizes[:k]])
            return _replan(
                dit, encoder, vae, mmproj, adapters, family=family,
                height=height, width=width, cfg=cfg, dtype=dtype,
                vae_dtype=vae_dtype, batch_rows=batch_rows,
                encoder_owned=encoder_owned, caption_tokens=caption_tokens,
                text_tokens=text_tokens, prefix_other=prefix_other,
                prefix_bound=prefix_bound, cache_bytes=cache_bytes,
                tile=tile, init_image=init_image, vae_price=vae_price,
                load_peak=load_peak, load_parts=load_parts, budget=budget,
                **kw)

        hint += _ref_hint(fits, sizes, row.prefix.min_ref_px)
    raise ImageMemoryError(
        f"image request does not fit: {plan.describe()}. {hint}. "
        "GMLX_OVERCOMMIT=1 overrides.", plan)


REF_GRID_PX = 32            # reference sides and sizes step by this


def _scaled(px: int, scale: float) -> int:
    """A reference side at ``scale``, on the grid of reference sides."""
    return max(REF_GRID_PX, round(px * scale / REF_GRID_PX) * REF_GRID_PX)


def _replan(*args, **kw) -> bool:
    try:
        return plan_memory(*args, overcommit=False, _hint=False, **kw).fits
    except ImageMemoryError:
        return False


def _ref_hint(fits, sizes: list, least: int) -> str:
    """What of a request with reference images fits its budget: the most
    of its references, first ones first, at their sizes, and the largest
    reference area at which all of them fit, down to a side of ``least``
    pixels. The reference area is that of the largest reference, named by
    the side of its square. ``fits(count, scale)`` plans the request with
    its first ``count`` references at ``scale`` times their sides, every
    phase and every conditioning counted."""
    n = len(sizes)
    count = next((k for k in range(n - 1, -1, -1) if fits(k, 1.0)), None)
    if count is None:
        return "; the request does not fit without reference images either"
    side0 = math.isqrt(max(h * w for h, w in sizes))
    top = (side0 - 1) // REF_GRID_PX * REF_GRID_PX
    side = next((s for s in range(top, max(least, REF_GRID_PX) - 1,
                                  -REF_GRID_PX) if fits(n, s / side0)), 0)
    area = (f"all {n} at a reference area of {side} x {side} px" if side
            else f"all {n} at no smaller reference area")
    return (f"; the largest that fit are {count} of the {n} reference "
            f"images at this reference area, or {area}")


def vae_row(family: str) -> VaeMemoryRow:
    try:
        return VAE_ROWS[family]
    except KeyError:
        known = ", ".join(sorted(VAE_ROWS))
        raise ValueError(f"no VAE memory row for image family {family!r}; "
                         f"known: {known}") from None


def vae_scale(family: str) -> int:
    """Image pixels per latent pixel of ``family``'s VAE, from the
    family's constants row."""
    from .registry import get_family

    return int(get_family(family).constants.values["vae_scale_factor"])


def tile_layout(size: int, tile: int = TILE_PX, overlap: int = OVERLAP_PX,
                scale: int = 8) -> tuple[int, int]:
    """Tiles along one axis of ``size`` pixels and their length, as
    ``tiling.tile_length`` and ``tiling.tile_starts`` lay them out at
    ``scale`` pixels per latent pixel."""
    if size <= tile:
        return 1, size
    if overlap >= tile:
        raise ValueError(f"overlap {overlap} must be below the tile {tile}")
    n = math.ceil((size - overlap) / (tile - overlap))
    length = -(-(size + (n - 1) * overlap) // (scale * n)) * scale
    return n, min(tile, length)


def crop_reach(family: str, op: str, config=None) -> int:
    """Latent pixels a cut reaches in a tiled ``op`` of ``family``'s VAE
    when its tiles are cropped, for the VAE of ``config`` or the official
    file's when None, and 0 when the tiles are blended."""
    crop = vae_row(family).crop
    return 0 if crop is None else int(crop(op, config))


def _latent_px(family: str, config=None) -> int:
    """Image pixels per latent pixel of the VAE of ``config``, or of the
    official VAE of the family's row when None."""
    if config is not None:
        return int(config.scale)
    return vae_row(family).scale or vae_scale(family)


def tile_grid(family: str, op: str, height: int, width: int, *,
              tile: int = TILE_PX, overlap: int = OVERLAP_PX,
              config=None) -> tuple[tuple[int, int], tuple[int, int]]:
    """The tiles of a tiled ``op`` of a ``height`` x ``width`` image, as
    tiling lays them out for the VAE of ``config`` or the official one
    when None: (count, length in pixels) down the rows and across the
    columns."""
    lap = tile_overlap(family, op, tile, overlap, config)
    scale = _latent_px(family, config)
    return (tile_layout(height, tile, lap, scale),
            tile_layout(width, tile, lap, scale))


def croppable(family: str, op: str, tile: int = TILE_PX,
              config=None) -> bool:
    """Whether a tiled ``op`` can run in tiles of ``tile`` pixels. Blended
    tiles always can. Cropped tiles must be longer than twice the reach of
    a cut, since a shorter tile keeps no pixel that a cut does not
    reach."""
    reach = crop_reach(family, op, config)
    return not reach or tile // _latent_px(family, config) > 2 * reach


def tile_overlap(family: str, op: str, tile: int = TILE_PX,
                 overlap: int = OVERLAP_PX, config=None) -> int:
    """The overlap of neighboring tiles of a tiled ``op``, in pixels:
    ``overlap``, or where it is more, the least overlap of the row when the
    tiles are blended and twice the reach of a cut when they are cropped.
    Raises ValueError for a ``tile`` that is not ``croppable``."""
    reach = crop_reach(family, op, config)
    if not reach:
        return max(int(overlap), vae_row(family).overlap)
    scale = _latent_px(family, config)
    if not croppable(family, op, tile, config):
        raise ValueError(
            f"a tile of {tile} pixels is {tile // scale} latent pixels, "
            f"and a tiled {op} of this VAE needs more than {2 * reach}, "
            f"which is {(2 * reach + 1) * scale} pixels")
    return max(int(overlap), 2 * reach * scale)


def _priced_path(conv: str) -> str:
    """The conv path whose fits price ``conv``. The kq path writes no patch
    buffer, and its decodes measured below the exact path's, so it takes
    the exact path's fits."""
    return "exact" if conv == "kq" else conv


def unmeasured_bytes(steps, weights: int, conv: str = "exact") -> int:
    """The price of a VAE pass on an unmeasured row. This is the one rule
    for such a row, whole or in tiles.

    ``steps`` are the steps of the pass, each (name, device bytes, softmax
    bytes, host bytes) by arithmetic, and ``weights`` the weight bytes at
    the pass dtype. The op-order floor of a step is its device bytes, the
    softmax of an attention chunk included. The price is the largest over
    the steps of the floor times ``UNMEASURED_RATIO`` plus the host bytes
    of that step, plus the weights, plus ``CACHE_BOUND``. The weights and
    the cache bound stand where a measured fit has its base. The steps of
    a tiled pass that holds one tile at a time are those of one tile
    beside what stays held, so such a pass is priced on one tile, and its
    host bytes are the blend buffers.

    The ratio is borrowed from the FLUX.1 autoencoder, as its comment
    says, and it is used because no footprint of the row's own VAE exists
    and a floor is not one: lazy evaluation holds tensors past the op
    order, and the buffer cache holds freed ones. A VAE with other blocks
    can hold more than this ratio of its floor, so ``vae_tiling`` never
    picks a whole pass on an unmeasured row for an image larger than one
    tile."""
    ratio = UNMEASURED_RATIO[_priced_path(conv)]
    return int(max(ratio * (device + softmax) + host
                   for _, device, softmax, host in steps)
               + weights + CACHE_BOUND)


def vae_peak(op: str, height: int, width: int, *, family: str,
             itemsize: int, channels: int | None = None, config=None,
             batch: int = 1, tiled: bool = False, tile: int = TILE_PX,
             overlap: int = OVERLAP_PX, conv: str | None = None) -> int:
    """The price of one VAE ``op``, "decode" or "encode", of a ``batch``
    of ``height`` x ``width`` images by the VAE row of ``family``, on the
    ``conv`` path (the default path when None).

    A measured row gives the footprint under bounded_cache over the
    loaded weights from its fits, at ``channels`` base channels, with the
    scores its mid attention holds, and a tiled pass follows its tiled
    fit. An unmeasured row gives ``unmeasured_bytes`` of the steps of a
    VAE with ``config``, which counts the weights. The plan has no VAE at
    hand and passes neither, which prices the official file: the row's
    own channels and config. An image no larger than a tile is priced
    whole."""
    row = vae_row(family)
    if conv is None:
        from gmlx.image.vae.conv import CONV as conv
    conv = _priced_path(conv)
    if tiled and (height > tile or width > tile):
        overlap = tile_overlap(family, op, tile, overlap, config)
    if not row.measured:
        steps, count = row.floor(op, height, width, config=config,
                                 itemsize=itemsize, batch=batch, tiled=tiled,
                                 tile=tile, overlap=overlap)
        return unmeasured_bytes(steps, count * itemsize, conv)
    if channels is None:
        channels = row.channels[op]
    fits = row.whole[op]
    fit = fits[conv][itemsize]
    # The scores count the latent pixels of the VAE at hand, and its tiles
    # are laid out at its scale, as tiling lays them out.
    held = 0
    if row.scores:
        latent = _latent_px(family, config)
        rows = (height // latent) * (width // latent)
        held = int(row.scores * batch * rows * rows * itemsize)
    if not tiled or (height <= tile and width <= tile):
        return fit.bytes(batch * height * width, channels, itemsize) + held
    (ny, th), (nx, tw) = tile_grid(family, op, height, width, tile=tile,
                                   overlap=overlap, config=config)
    tiled_fit = row.tiled[op][itemsize]
    tiles = 1 if row.one_tile else ny * nx
    out = tiled_fit.bytes(batch * tiles * th * tw, batch * height * width,
                          channels, itemsize)
    if conv != "exact":
        one = batch * th * tw
        out += max(0, fit.bytes(one, channels, itemsize)
                   - fits["exact"][itemsize].bytes(one, channels, itemsize))
    return out + held


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


def vae_tiling(op: str, height: int, width: int, *, family: str,
               itemsize: int, channels: int | None = None, config=None,
               batch: int = 1, tiled: bool | None = None,
               tile: int = TILE_PX, overlap: int = OVERLAP_PX,
               conv: str | None = None, free: float | None = None,
               overcommit: bool | None = None) -> bool:
    """Whether a VAE ``op`` runs in tiles, by the VAE row of ``family``.
    An image no larger than a tile always runs whole.

    ``tiled=None`` is the automatic choice, for a caller without a plan.
    It is a whole pass when that fits the free working set and tiles
    otherwise. On a row that takes tiles first, which every unmeasured
    row does, it is tiles for every image larger than one tile, on every
    machine. So it is on any row when the device reports no free working
    set, and on a row that bounds its whole passes for a pass of more
    pixels than the bound. ``tiled=False`` forces a whole pass at the
    row's price. A request with a plan passes the two values of
    ``MemoryPlan.vae_pass`` as ``tiled`` and ``free``.

    A pass that does not fit as chosen raises ImageMemoryError, unless
    GMLX_OVERCOMMIT=1. ``free`` defaults to ``free_bytes()``, and nothing
    is refused when the device reports none."""
    row = vae_row(family)
    kw = dict(family=family, itemsize=itemsize, channels=channels,
              config=config, batch=batch, tile=tile, overlap=overlap,
              conv=conv)
    whole = vae_peak(op, height, width, **kw)
    can_tile = height > tile or width > tile
    # A forced whole pass has no tiled price where no tiled pass exists.
    priced = can_tile and (tiled is None or bool(tiled)
                           or croppable(family, op, tile, config))
    in_tiles = (vae_peak(op, height, width, tiled=True, **kw) if priced
                else whole)
    if free is None:
        free = free_bytes()
    if tiled is not None:
        choice = bool(tiled) and can_tile
    elif free is None or row.tiles_at(batch * height * width):
        choice = can_tile
    else:
        choice = can_tile and whole > free
    if free is None:
        return choice
    need = in_tiles if choice else whole
    if need <= free:
        return choice
    if overcommit is None:
        from gmlx.serve.capacity import overcommit as _oc

        overcommit = _oc()
    if overcommit:
        return choice
    how = "in tiles" if choice else "whole"
    if not choice and priced and in_tiles <= free:
        hint = f"a tiled pass needs {in_tiles / GiB:.2f} GiB"
    else:
        hint = "try a smaller image"
    rule = ""
    if not row.measured:
        from .vae.reader import vae_row as file_row

        title = file_row(family).title
        rule = (f" {title[0].upper()}{title[1:]} is not measured, and this "
                "is the price of the rule for an unmeasured VAE.")
    raise ImageMemoryError(
        f"VAE {op} of {batch} x {height}x{width} {how} needs "
        f"{need / GiB:.2f} GiB, and {free / GiB:.2f} GiB is free; {hint}. "
        f"GMLX_OVERCOMMIT=1 overrides.{rule}")


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
