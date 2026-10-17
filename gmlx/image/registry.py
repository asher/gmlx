"""Image model families and their discovery.

Each directory under ``gmlx.image.families`` holds a ``family.py`` that
defines ``FAMILY``. ``families()`` imports them all and sorts by order and
name, so a new family adds one directory and edits no shared file.
"""

from __future__ import annotations

import functools
import importlib
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
    count."""
    steps: int
    cfg: float
    shift: float
    repo: str
    revision: str
    quality: tuple = ()
    sigmas: tuple = ()


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
    conditioning key.

    A family whose encoder reads the prompt alone has no ``token_ids``,
    ``encode`` or ``template_id``: its references reach the DiT as VAE
    latents only. A family whose rule for a reference depends on the
    other references of the request has ``prepare_all(images)`` in place
    of ``prepare`` and the areas: it gives the references of a request,
    a tuple with ``crop``, the (left, top, right, bottom) box the output
    is cropped to when the request gives no size, or None. ``size(refs)``
    is then the (width, height) of a request with references and no
    size of its own, which ``edit_size`` scales into the image limits.
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


# The image limits: the pixels and the longer side of the largest image a
# served request asks for, and of the size an edit without one takes.
MAX_PIXELS = 2048 * 2048
MAX_SIDE = 4096


def fit_size(width: int, height: int, unit: int) -> tuple:
    """(width, height) scaled down at its shape to the largest size in the
    image limits, each side floored to a multiple of ``unit``. A side
    that floors to nothing is 0."""
    scale = min(1.0, (MAX_PIXELS / (width * height)) ** 0.5,
                MAX_SIDE / max(width, height))
    return (int(width * scale) // unit * unit,
            int(height * scale) // unit * unit)


def edit_size(family, refs, unit: int) -> tuple | None:
    """(width, height) of an edit of ``family`` with the prepared ``refs``
    and no size of its own: the size the family gives them, scaled into
    the image limits on the ``unit`` grid when it is past them. None for
    a family without that rule. Raises ValueError when a side scales to
    nothing."""
    edit = family.edit
    if not refs or edit is None or edit.size is None:
        return None
    w, h = edit.size(refs)
    if w * h <= MAX_PIXELS and max(w, h) <= MAX_SIDE:
        return w, h
    fit = fit_size(w, h, unit)
    if min(fit) <= 0:
        raise ValueError(
            f"the first reference sizes the edit at {w}x{h}, which keeps "
            f"no side of {unit} pixels in the limit of {MAX_PIXELS} pixels "
            f"(2048x2048) and a side of {MAX_SIDE}: give the edit a size")
    return fit


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
    step: StepParts | None = None      # None runs the DiT's own step
    # () -> text_encoder.EncoderSpec; None reads text_encoder.ENCODERS
    encoder: Callable | None = None
    # (steps, tokens=, variant=) -> float32 sigmas with the final zero
    schedule: Callable | None = None
    edit: EditParts | None = None      # None takes no reference images
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
