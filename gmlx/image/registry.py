"""Image model families and their discovery.

Each directory under ``gmlx.image.families`` holds a ``family.py`` that
defines ``FAMILY``. ``families()`` imports them all and sorts by order and
name, so a new family adds one directory and edits no shared file.
"""

from __future__ import annotations

import functools
import importlib
import math
import pkgutil
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class ConstantsRow:
    """Config values no tensor shape determines, with their source."""
    values: dict
    repo: str
    revision: str


@dataclass(frozen=True)
class SamplingRow:
    """A variant's official sampling settings, with their source: steps
    are model evaluations, cfg is gmlx's (diffusers guidance_scale g is
    cfg = 1 + g), and shift is the flow schedule's static shift.
    ``quality`` maps the API quality levels low, medium and high to
    steps, as (level, steps) pairs; the variant's own steps are high.
    ``sigmas`` is a fixed schedule without its final zero, for a variant
    distilled to it: such a variant runs that many steps and no other
    count. ``neg_skip`` holds the blocks, zero-based, that the negative
    pass of a guided step skips. ``size`` is the (width, height) of a
    request that names none."""
    steps: int
    cfg: float
    shift: float
    repo: str
    revision: str
    quality: tuple = ()
    sigmas: tuple = ()
    neg_skip: tuple = ()
    size: tuple = (1024, 1024)


@dataclass(frozen=True)
class StepParts:
    """A family's denoise step in the parts a pipeline caches.

    ``modulation(dit, sigma)`` is what the step derives from one float32
    sigma alone and ``conditioning(dit, cond)`` what it derives from the
    conditioning alone. ``forward(dit, latents, sigma, cond, state, mod)``
    is the rest of the step for one row, given both, and returns the
    float32 velocity. Each is a pure function of its arguments and the DiT
    weights, and each state gives its arrays as ``arrays()``.
    ``state_bytes`` bounds the conditioning states a pipeline holds for
    other requests beside those of the request it runs, by the rule of
    ``gmlx.image.api.state_scope``. ``state_size(dit, cond)`` gives the
    bytes of a conditioning's state before it is built, so the cache makes
    room first.
    """
    modulation: Callable
    conditioning: Callable
    forward: Callable
    state_bytes: int = 256 << 20
    state_size: Callable | None = None


@dataclass(frozen=True)
class LoopRules:
    """Where a family's reference sampling loop rounds, when it does not
    run in float32 throughout. ``noise`` is the dtype the initial noise
    is rounded to. With ``act_update`` the guided velocity and the Euler
    product take the DiT's activation dtype, each op rounded to it as the
    reference's tensors of that dtype are, and the sum with the latents
    runs in float32. The latents stay float32 across the steps, unless
    ``act_latents`` rounds them to the activation dtype after each step,
    as a reference that keeps its latents in that dtype does."""
    noise: str | None = None
    act_update: bool = False
    act_latents: bool = False


@dataclass(frozen=True)
class EditParts:
    """How a family conditions on reference images.

    A request takes ``max_refs`` references at most. ``prepare(image,
    area)`` gives one image as a ``conditioning.Reference``: resized by
    the family's rule to about ``area`` pixels, which is at least
    ``min_area``. ``token_ids(handle, prompt, refs)`` are the ids of the
    prompt in the family's template for references, with the image slots
    of every reference, and ``encode(handle, prompt, refs)`` runs them
    with the images through an encoder that has its vision tower and
    returns the ids, the features ``(L, D)`` and the rows of the features
    that are image slots. ``template_id()`` names the template in a
    conditioning key. ``render(prompt, count)`` is the prompt in that
    template with the slot text of ``count`` references, before their
    slots expand, whose bytes bound its text rows before a tokenizer
    loads. A family whose plan prices the prompt gives it.

    A family whose encoder reads the prompt alone has no ``token_ids``,
    ``encode`` or ``template_id``: its references reach the DiT as VAE
    latents only. A family whose rule for a reference depends on the
    other references of the request has ``prepare_all(images)`` in place
    of ``prepare`` and the areas: it gives the references of a request,
    a tuple with ``crop``, the (left, top, right, bottom) box the output
    is cropped to when the request gives no size, or None. ``size(refs)``
    is then the (width, height) of a request with references and no
    size of its own, which ``edit_size`` scales into its family's limits.
    """
    max_refs: int
    area: int = 0
    min_area: int = 0
    prepare: Callable | None = None
    token_ids: Callable | None = None
    encode: Callable | None = None
    template_id: Callable | None = None
    prepare_all: Callable | None = None
    size: Callable | None = None
    render: Callable | None = None

    @property
    def text_only(self) -> bool:
        """Whether the encoder reads the prompt without the images."""
        return self.encode is None


@dataclass(frozen=True)
class UpsampleParts:
    """How a chat model rewrites a family's typed prompt into the caption
    its DiT was trained on, before the encoder reads it.

    ``prompts`` is a file in ``package``, shipped as its source wrote it,
    with its ``sha256`` pinned: a JSON file whose ``key`` holds the
    system prompt for a prompt without reference images, or a text file
    that is the system prompt when ``key`` is None. ``prompts`` None
    ships no system prompt, and ``needs`` is the refusal that says where
    to get one. The typed prompt is the user turn as it is, or, when the
    system prompt holds ``slot``, takes its place there. The reply is
    decoded with ``temperature``, at most ``max_tokens`` tokens,
    ``presence_penalty`` and the chat model's thinking switch at
    ``thinking``, as ``repo`` at ``revision`` does, and ``parse`` turns
    it into the caption, or None when it holds none. ``suggested`` names
    the chat model to set when none is, and ``expected`` warns when none
    is set.
    """
    prompts: str | None
    sha256: str
    key: str | None
    max_tokens: int
    temperature: float
    presence_penalty: float
    thinking: str
    repo: str
    revision: str
    package: str = ""
    suggested: str = ""
    expected: bool = False
    parse: Callable | None = None
    slot: str = ""
    needs: str = ""


# The image limits of a family without its own: the pixels and the
# longer side of the largest image a served request asks for, and of the
# size an edit without one takes. The CLI and the API hold such a family
# to none.
MAX_PIXELS = 2048 * 2048
MAX_SIDE = 4096


def limited(family) -> bool:
    """Whether ``family`` sets size limits of its own, which hold on every
    path in place of the image limits."""
    return family is not None and any(
        getattr(family, k, None) is not None
        for k in ("max_pixels", "max_side", "min_pixels", "aspect"))


def size_limits(family) -> tuple:
    """(pixels, side) of the largest image of ``family``: its own bounds
    where it is limited, with None for a bound it does not set, else the
    image limits."""
    if not limited(family):
        return MAX_PIXELS, MAX_SIDE
    return family.max_pixels, family.max_side


def _shape(wh: tuple) -> str:
    return f"{wh[0]}:{wh[1]}"


def shape_sizes(width: int, height: int, unit: int, side: int):
    """The sizes of the shape of ``width`` x ``height`` on ``unit`` up to
    ``side``: one side a multiple of ``unit``, the other that side's share
    of the shape rounded down or up to ``unit``. These are the sizes of the
    shape scaled by one factor with both sides rounded down, or both up."""
    for a in range(unit, side + 1, unit):
        for num, den, first in ((height * a, width, True),
                                (width * a, height, False)):
            for b in {num // (den * unit) * unit,
                      -(-num // (den * unit)) * unit}:
                if unit <= b <= side:
                    yield (a, b) if first else (b, a)


def smallest_size(family, width: int, height: int) -> tuple:
    """(width, height), the size of the shape of ``width`` x ``height``
    (``shape_sizes``) with the fewest pixels that ``family`` takes, and of
    sizes with as many pixels the nearest the shape. With none it takes,
    the shape scaled up to its least pixels, each side rounded up to its
    size multiple."""
    unit = int(family.constants.values.get("size_multiple", 16))
    taken = [s for s in shape_sizes(width, height, unit,
                                    size_limits(family)[1] or MAX_SIDE)
             if takes(family, *s)]
    if taken:
        return min(taken, key=lambda s: (
            s[0] * s[1], abs(s[0] * height - s[1] * width) / s[1], s))
    scale = (family.min_pixels / (width * height)) ** 0.5
    return (math.ceil(width * scale / unit) * unit,
            math.ceil(height * scale / unit) * unit)


def takes(family, width: int, height: int) -> bool:
    """Whether ``family`` takes an image of ``width`` x ``height`` pixels
    by its own size limits: ``size_refusal`` without the reason, whose
    least pixels refusal names ``smallest_size``."""
    least = getattr(family, "min_pixels", None)
    if limited(family) and least is not None and width * height < least:
        return False
    return size_refusal(family, width, height) is None


def size_refusal(family, width: int, height: int) -> str | None:
    """Why ``family`` refuses an image of ``width`` x ``height`` pixels by
    its own size limits, or None. A family without limits refuses none:
    serve holds it to the image limits, and the CLI and the API to none."""
    if not limited(family):
        return None
    title, px = family.title, width * height
    if family.max_pixels is not None and px > family.max_pixels:
        return (f"{width}x{height} is {px:,} pixels, over the "
                f"{family.max_pixels:,} that {title} takes")
    if family.max_side is not None and max(width, height) > family.max_side:
        return (f"{width}x{height} has a side over the {family.max_side} "
                f"pixels that {title} takes")
    if family.aspect is not None:
        (lw, lh), (hw, hh) = family.aspect
        if width * lh < lw * height or width * hh > hw * height:
            return (f"{width}x{height} is outside the shapes {title} takes, "
                    f"{_shape(family.aspect[0])} to "
                    f"{_shape(family.aspect[1])}")
    if family.min_pixels is not None and px < family.min_pixels:
        w, h = smallest_size(family, width, height)
        return (f"{width}x{height} is {px:,} pixels, under the "
                f"{family.min_pixels:,} that {title} takes; the smallest "
                f"size of its shape is {w}x{h}")
    return None


def fit_size(width: int, height: int, unit: int, family=None) -> tuple:
    """(width, height) scaled down at its shape to the largest size in the
    limits of ``family`` (the image limits for None or a family without
    its own), each side floored to a multiple of ``unit``. A side that
    floors to nothing is 0. A shape the family takes keeps inside its
    shapes, the longer side floored further where the floors leave it."""
    pixels, side = size_limits(family)
    scale = min(1.0, (pixels / (width * height)) ** 0.5 if pixels else 1.0,
                side / max(width, height) if side else 1.0)
    return in_shape(family, width, height,
                    int(width * scale) // unit * unit,
                    int(height * scale) // unit * unit, unit)


def in_shape(family, width: int, height: int, w: int, h: int,
             unit: int) -> tuple:
    """(w, h), a size scaled from ``width`` x ``height`` and floored to
    ``unit``, with its longer side floored further while the floors leave
    it outside the shapes of ``family`` that ``width`` x ``height`` is
    inside."""
    if limited(family) and family.aspect is not None and min(w, h) > 0:
        (lw, lh), (hw, hh) = family.aspect
        if width * lh >= lw * height and width * hh <= hw * height:
            while h > unit and w * lh < lw * h:
                h -= unit
            while w > unit and w * hh > hw * h:
                w -= unit
    return w, h


def edit_size(family, refs, unit: int) -> tuple | None:
    """(width, height) of an edit of ``family`` with the prepared ``refs``
    and no size of its own: the size the family gives them, scaled down
    into its limits on the ``unit`` grid when it is past them. None for a
    family without that rule. Raises ValueError when a side scales to
    nothing, or when a limited family refuses the size, as for a first
    reference outside its shapes."""
    edit = family.edit
    if not refs or edit is None or edit.size is None:
        return None
    w, h = edit.size(refs)
    pixels, side = size_limits(family)
    fit = w, h
    if (pixels and w * h > pixels) or (side and max(w, h) > side):
        fit = fit_size(w, h, unit, family)
    if min(fit) <= 0:
        raise ValueError(
            f"the first reference sizes the edit at {w}x{h}, which keeps "
            f"no side of {unit} pixels in {limits_text(family)}: give the "
            "edit a size")
    why = size_refusal(family, *fit)
    if why:
        raise ValueError(f"the first reference sizes the edit at {w}x{h}: "
                         f"{why}; give the edit a size")
    return fit


def limits_text(family) -> str:
    """The largest images of ``family`` in words, as refusals name them."""
    if not limited(family):
        return (f"the limit of {MAX_PIXELS} pixels (2048x2048) and a side "
                f"of {MAX_SIDE}")
    parts = [f"{family.max_pixels:,} pixels"] if family.max_pixels else []
    parts += [f"a side of {family.max_side}"] if family.max_side else []
    return f"the {' and '.join(parts)} that {family.title} takes"


@dataclass(frozen=True)
class Family:
    name: str                        # "z_image"
    title: str                       # "Z-Image"
    order: int
    arch_strings: frozenset          # accepted general.architecture values
    strip_prefixes: tuple            # at most one is stripped from every name
    fingerprint: Callable            # HeaderView -> Match | Refusal | None
    synth_config: Callable           # HeaderView -> dict
    constants: ConstantsRow
    build: Callable | None           # config -> nn.Module, None when not built
    fp32_keep: tuple = ()
    identity_tensors: tuple = ()     # tensors hashed for the variant
    variants: dict = field(default_factory=dict)   # hash -> VariantRow
    # hash -> why gmlx image refuses a file of known weights
    refused: dict = field(default_factory=dict)
    sampling: dict = field(default_factory=dict)   # variant -> SamplingRow
    # variant -> the variant that takes guidance, for a variant distilled
    # to sample without it, which then refuses guidance by name; "" when
    # no variant of the family takes guidance.
    unguided: dict = field(default_factory=dict)
    # The sampling row of a variant without one, such as a file whose
    # identity matches no row; None leaves it without defaults.
    unknown_variant: str | None = None
    # A file whose identity matches no row runs only under a variant
    # given by name, as when the schedule depends on the variant.
    needs_variant: bool = False
    lora_map: Callable | None = None   # config -> lora.reader.LoraKeyMap
    # Path patterns (fnmatch) of the linear leaves that see float32 rows
    # under any activation dtype, whose LoRA factors are built and priced
    # in float32.
    lora_f32: tuple = ()
    step: StepParts | None = None      # None runs the DiT's own step
    # () -> text_encoder.EncoderSpec; None reads text_encoder.ENCODERS
    encoder: Callable | None = None
    # (steps, tokens=, variant=) -> float32 sigmas with the final zero
    schedule: Callable | None = None
    edit: EditParts | None = None      # None takes no reference images
    # Reference latents sample with noise_from_seed(seed) draws, fixed per
    # image, as the reference pipeline samples each encode; None encodes
    # them in mode.
    ref_encode_seed: int | None = None
    # None runs the prompt as typed, with no upsampler to rewrite it.
    upsample: UpsampleParts | None = None
    loop: LoopRules | None = None      # None runs the loop in float32
    # Tensors a converter may store flattened with their shape under
    # comfy.gguf.orig_shape, which the load restores.
    reshaped: tuple = ()
    # general.architecture values that converters also write for other
    # models, such as "flux". A file with one loads when its tensors match
    # this family, but the value alone never names the family.
    shared_arch_strings: frozenset = frozenset()
    # The size limits of the family's images on every path, CLI, API and
    # serve: the most pixels, the longest side, the fewest pixels, and the
    # narrowest and widest shapes as ((w, h), (w, h)). All None keeps the
    # image limits in serve and none in the CLI and the API.
    max_pixels: int | None = None
    max_side: int | None = None
    min_pixels: int | None = None
    aspect: tuple | None = None
    # Path patterns (fnmatch) of the DiT linears that int8 linears convert,
    # each to the columns of one activation scale: None for the whole row.
    # Empty runs the file's linears alone.
    int8_linears: dict = field(default_factory=dict)
    # The DiT runs int8 attention in place of dense attention, its
    # ``attention`` setting.
    int8_attention: bool = False

    def variant_names(self) -> tuple[str, ...]:
        return tuple(sorted({v.name for v in self.variants.values()}))

    def variant_needed(self, variant: str) -> str | None:
        """Why a file of ``variant`` cannot run without a variant given
        by name, or None when it can."""
        if not self.needs_variant or variant in self.sampling:
            return None
        names = self.variant_names()
        return (f"this {self.title} GGUF matches no known {self.title} "
                f"weights, and its schedule depends on the variant: set "
                f"`variant: {names[0]}` on its image_models entry, or pass "
                f"--variant {names[0]}, if it is {self.title} "
                f"{names[0]}. Known variants: {', '.join(names)}")

    def sampling_row(self, variant: str):
        """The SamplingRow a variant samples with, or None."""
        row = self.sampling.get(variant)
        if row is None and self.unknown_variant is not None:
            row = self.sampling.get(self.unknown_variant)
        return row


def int8_refusal(family: Family, variant: str, int8: str) -> str | None:
    """Why this install or Mac does not run ``variant`` of ``family`` at
    the int8 setting ``int8`` (``int8.INT8``), or None: a family without
    the attention half runs dense attention alone, then
    ``int8.int8_refusal``. Raises ValueError for a value off INT8."""
    from . import int8 as neutral

    attention, _ = neutral.int8_parts(int8)
    if attention == "int8" and not family.int8_attention:
        return f"gmlx runs {family.title} with dense attention only"
    return neutral.int8_refusal(family.title, variant, int8,
                                family.int8_linears)


def int8_fallback(family: Family) -> str:
    """The int8 settings of ``family`` that keep the file's linears."""
    from . import int8 as neutral

    return neutral.int8_fallback(not family.int8_attention)


@dataclass(frozen=True)
class VariantRow:
    """A known weight identity: the variant and where the hash came from."""
    name: str
    repo: str
    revision: str
    files: tuple


@functools.lru_cache(maxsize=1)
def families() -> tuple[Family, ...]:
    from . import families as pkg

    found: list[Family] = []
    for info in pkgutil.iter_modules(pkg.__path__):
        if not info.ispkg:
            continue
        mod: Any = importlib.import_module(
            f"{pkg.__name__}.{info.name}.family")
        found.append(mod.FAMILY)
    return tuple(sorted(found, key=lambda f: (f.order, f.name)))


def get_family(name: str) -> Family:
    for fam in families():
        if fam.name == name:
            return fam
    known = ", ".join(f.name for f in families())
    raise ValueError(f"unknown image family {name!r}; known: {known}")


def _alnum(text: str) -> str:
    return "".join(c for c in str(text).lower() if c.isalnum())


def names_family(name: str, family: Family, variant: str = "") -> bool:
    """Whether a configured ``family`` value names this family: its name,
    its title, or its name with the variant, compared on letters and
    digits alone, so ``qwen-image-2.1``, ``flux-3-klein`` and
    ``z-image-turbo`` match."""
    want = _alnum(name)
    return want in {_alnum(family.name), _alnum(family.title),
                    _alnum(family.name + variant),
                    _alnum(family.title + variant)}


def family_named(name: str) -> Family | None:
    """The family a configured ``family`` value names, as
    ``names_family`` reads it with any variant the family runs, or None."""
    for fam in families():
        variants = ("", *fam.sampling, *fam.variant_names())
        if any(names_family(name, fam, v) for v in variants):
            return fam
    return None


def is_image_arch(arch: str | None) -> bool:
    return bool(arch) and any(arch in f.arch_strings for f in families())
