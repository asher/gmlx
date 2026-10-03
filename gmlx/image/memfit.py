"""The memory plan of an image request, from header sizes and the request.

``plan_memory`` is a pure function. Resident bytes are the DiT, the text
encoder, the VAE at its decode dtype, the mmproj for edits, adapter
factors and caches. A request runs in three phases, encode, denoise and
decode, and each phase adds its own activation peak to what is resident
then. The plan fits when the largest phase stays within the working budget
of ``gmlx.serve.capacity``.

Levers apply in order until the plan fits: sequential rows in place of a
batch, release of an owned encoder after encoding, and tiled VAE decode.
A plan that still does not fit raises ``ImageMemoryError`` with each term
in GiB and the largest term named, unless GMLX_OVERCOMMIT=1.

The MLX buffer cache keeps freed buffers for reuse up to its limit, which
defaults to 95 percent of memory. One VAE decode frees and allocates
buffers of many sizes inside one evaluation, so under the default limit
its footprint reaches two to three times its live peak. Image work runs
inside ``bounded_cache``, which lowers the limit to ``CACHE_BOUND`` for
the block, and the VAE peaks below are footprints measured that way.
``vae_tiling`` decides between a whole and a tiled VAE pass from those
peaks and the free working set, and refuses one that does not fit.
"""

from __future__ import annotations

import math
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

GiB = 1 << 30

# Measured peak of one 1024x1024 Z-Image-Turbo step at 4128 rows: 1.05 GiB
# above the weights on an M5 Max, against 0.246 GiB for the formula.
Z_IMAGE_LIVE = 4.27

# Measured footprint of one 512-token Qwen3-4B Q4_K_M encode on an M5 Max:
# 0.35 GiB above the weights, 12.5 times the per-token formula, rounded up.
ENCODE_LIVE = 16.0

# The MLX cache limit inside bounded_cache.
CACHE_BOUND = 2 * GiB

TILE_PX = 512


@dataclass(frozen=True)
class VaeFit:
    """Footprint of one VAE pass over the weights and inputs: ``base``
    bytes plus ``k`` bytes per pixel, base channel and activation byte."""
    base: float
    k: float

    def bytes(self, pixels: int, channels: int, itemsize: int) -> int:
        return int(self.base + self.k * pixels * channels * itemsize)


# FLUX.1 autoencoder footprints under bounded_cache, by activation
# itemsize. Whole passes at 256, 512, 768 and 1024 pixels on an M5 Max,
# two runs each in fresh processes, fit a line in pixels. The slope is
# raised 10 percent, and the base covers every whole pass and a tiled
# 1024 pass plus 0.25 GiB. At 1024 the bf16 decode measured 6.64 GiB
# and the fp32 decode 10.42 GiB, against 7.48 and 11.55 here. The
# Qwen-Image 2.1 VAE uses these rows by its channel count until it is
# measured.
VAE_DECODE = {2: VaeFit(base=2.35 * GiB, k=20.5),
              4: VaeFit(base=2.70 * GiB, k=17.7)}
VAE_ENCODE = {2: VaeFit(base=2.35 * GiB, k=15.2),
              4: VaeFit(base=2.50 * GiB, k=12.6)}


@dataclass(frozen=True)
class PeakRow:
    """How a family's activation peak follows from the request."""
    token_px: int                   # pixels per token along each axis
    row_multiple: int               # image rows pad up to a multiple of it
    caption_rows: int               # caption rows at most, joined to the image
    width: int                      # widest outputs per row, in elements
    live: float                     # observed peak over the formula
    vae_channels: int               # VAE channels at full resolution
    prefix: Callable | None = None  # (text, refs, ref_px) -> bytes per cond
    encode_row_bytes: int = 0       # encoder activation bytes per token,
                                    # times ENCODE_LIVE
    text_rows: int = 512


def _qwen_prefix(text: int, refs: int, ref_px: int) -> int:
    tokens = text + refs * (ref_px // 16) ** 2
    return 2 * 32 * tokens * 4096 * 2


PEAKS = {
    "z_image": PeakRow(token_px=16, row_multiple=32, caption_rows=512,
                       width=11520 + 2 * 10240, live=Z_IMAGE_LIVE,
                       vae_channels=128, encode_row_bytes=3 * 9728 * 2),
    # The live factor is Z-Image's until a Qwen-Image 2.1 step is measured.
    "qwen_image21": PeakRow(token_px=16, row_multiple=1, caption_rows=0,
                            width=24576 + 12288, live=Z_IMAGE_LIVE,
                            vae_channels=96, prefix=_qwen_prefix,
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
                ref_px: int = 0, cache_bytes: int = 0,
                tile: int = 512, budget: float | None = None,
                overcommit: bool | None = None) -> MemoryPlan:
    """The memory plan for one image request.

    ``dit``, ``encoder``, ``mmproj`` and ``vae`` are header views or byte
    counts. ``adapters`` is a byte count or a list of them. ``budget``
    defaults to ``capacity.working_budget_bytes()`` and ``overcommit`` to
    GMLX_OVERCOMMIT.
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
    guided = cfg > 1.0
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
    if row.prefix is not None:
        text = row.text_rows if text_tokens is None else text_tokens
        terms["prefix_cache"] = conds * row.prefix(text, refs, ref_px)
    rows = image_rows(row, height, width, caption_tokens)
    step_one = rows * row.width * act * row.live
    full_decode = vae_peak("decode", height, width, itemsize=vsize,
                           channels=row.vae_channels)
    tiled_decode = vae_peak("decode", height, width, itemsize=vsize,
                            channels=row.vae_channels, tiled=True, tile=tile)
    caption = row.text_rows if caption_tokens is None else caption_tokens
    encode_peak = caption * row.encode_row_bytes * ENCODE_LIVE

    def build(b: int, release: bool, tiled: bool, levers: tuple) -> MemoryPlan:
        t = dict(terms)
        t["step_peak"] = int(b * step_one)
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
        hint += _ref_hint(row, terms, plan, refs, ref_px, conds, text_tokens)
    raise ImageMemoryError(
        f"image request does not fit: {plan.describe()}. {hint}. "
        "GMLX_OVERCOMMIT=1 overrides.", plan)


def _ref_hint(row: PeakRow, terms: dict, plan: MemoryPlan, refs: int,
              ref_px: int, conds: int, text_tokens: int | None) -> str:
    """The largest reference count, and the largest reference size at the
    requested count, that fit the denoise phase."""
    if row.prefix is None or plan.budget is None:
        return ""
    text = row.text_rows if text_tokens is None else text_tokens
    room = (plan.budget - plan.phases["denoise"]
            + terms["prefix_cache"])

    def fits(n: int, px: int) -> bool:
        return conds * row.prefix(text, n, px) <= room

    count = max((n for n in range(refs + 1) if fits(n, ref_px)), default=0)
    side = next((s for s in range(ref_px, 0, -row.token_px)
                 if fits(refs, s)), 0)
    return (f"; {count} references of {ref_px} px fit, or {refs} of "
            f"{side} px")


def vae_peak(op: str, height: int, width: int, *, itemsize: int,
             channels: int = 128, batch: int = 1, tiled: bool = False,
             tile: int = TILE_PX) -> int:
    """Footprint of one VAE ``op``, "decode" or "encode", of a ``batch``
    of ``height`` x ``width`` images, under bounded_cache. A tiled pass
    costs one tile plus the float32 blend buffers of the whole image."""
    fits = {"decode": VAE_DECODE, "encode": VAE_ENCODE}[op]
    fit = fits[itemsize]
    if not tiled or (height <= tile and width <= tile):
        return fit.bytes(batch * height * width, channels, itemsize)
    th, tw = min(tile, height), min(tile, width)
    blend = batch * height * width * (3 * 4 * 2 + 4)
    return fit.bytes(batch * th * tw, channels, itemsize) + blend


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
               free: float | None = None,
               overcommit: bool | None = None) -> bool:
    """Whether a VAE ``op`` runs in tiles. ``tiled=None`` picks a whole
    pass when it fits the free working set and tiles otherwise. A pass
    that does not fit as chosen raises ImageMemoryError, unless
    GMLX_OVERCOMMIT=1. ``free`` defaults to ``free_bytes()``; when the
    device reports none, nothing is refused and None picks a whole pass.
    An image no larger than a tile always runs whole."""
    kw = dict(itemsize=itemsize, channels=channels, batch=batch, tile=tile)
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


def gate_weights(plan: MemoryPlan, model_id: str) -> None:
    """Pass the plan's weights through ``capacity.preload_gate``, so
    resident text models are respected."""
    from gmlx.serve.capacity import preload_gate

    preload_gate(float(plan.weight_bytes), model_id)
