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
    is rounded to; the latents stay float32 across the steps. With
    ``act_update`` the guided velocity and the Euler product take the
    DiT's activation dtype, each op rounded to it as the reference's
    tensors of that dtype are, and only the sum with the latents runs in
    float32."""
    noise: str | None = None
    act_update: bool = False


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
    """
    max_refs: int
    area: int
    min_area: int
    prepare: Callable
    token_ids: Callable
    encode: Callable
    template_id: Callable


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


def is_image_arch(arch: str | None) -> bool:
    return bool(arch) and any(arch in f.arch_strings for f in families())
