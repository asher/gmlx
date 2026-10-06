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
    cfg = 1 + g), and shift is the flow schedule's static shift."""
    steps: int
    cfg: float
    shift: float
    repo: str
    revision: str


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
    sampling: dict = field(default_factory=dict)   # variant -> SamplingRow
    lora_map: Callable | None = None   # config -> lora.reader.LoraKeyMap
    step: StepParts | None = None      # None runs the DiT's own step
    # (steps, tokens=, variant=) -> float32 sigmas with the final zero
    schedule: Callable | None = None

    def variant_names(self) -> tuple[str, ...]:
        return tuple(sorted({v.name for v in self.variants.values()}))


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
