"""Video model families and their discovery.

Each directory under ``gmlx.video.families`` holds a ``family.py`` that
defines ``FAMILY``. ``families()`` imports them all and sorts by order and
name. A family carries its own values for the shared mechanisms: the
segment order and clock of the packed sequence (``timeline``, read by
``layout.pack``), each stream's schedule shift (``shifts``) and anchor
rule (``anchor_rules``).
"""

from __future__ import annotations

import functools
import importlib
import pkgutil
from dataclasses import dataclass, field
from typing import Any, Callable

from gmlx.image.registry import ConstantsRow

from .anchors import AnchorRule
from .layout import Timeline


@dataclass(frozen=True)
class VideoVariantRow:
    """A known weight identity: the variant, where its hash came from, and
    the order of the fused qkv rows in its files (``layout``)."""
    name: str
    repo: str
    revision: str
    files: tuple
    layout: str = "qkv"


@dataclass(frozen=True)
class VideoSampling:
    """A recipe's sampling defaults, with their source. ``steps`` counts
    DiT forwards. ``shifts`` maps each stream to its static flow shift."""
    steps: int
    shifts: dict
    repo: str
    revision: str


@dataclass(frozen=True)
class VideoFamily:
    name: str                        # "h3"
    title: str                       # "MiniMax-H3"
    order: int
    arch_strings: frozenset          # accepted general.architecture values
    strip_prefixes: tuple            # at most one is stripped from every name
    fingerprint: Callable            # HeaderView -> Match | Refusal | None
    synth_config: Callable           # HeaderView -> dict
    constants: ConstantsRow
    build: Callable                  # config -> nn.Module
    fp32_keep: tuple = ()
    identity_tensors: tuple = ()     # tensors hashed for the variant
    variants: dict = field(default_factory=dict)   # hash -> VideoVariantRow
    refused: dict = field(default_factory=dict)    # hash -> reason
    # (layout hash) -> VideoVariantRow: files whose qkv rows are known
    layouts: dict = field(default_factory=dict)
    layout_tensor: str = ""          # the fused qkv tensor the hash reads
    layout_rows: int = 0             # rows hashed at each of q, k and v
    sampling: dict = field(default_factory=dict)   # recipe -> VideoSampling
    unknown_variant: str | None = None
    reshaped: tuple = ()
    # The shared mechanisms' values: the clock of the packed sequence,
    # each stream's static shift and anchor rule.
    timeline: Timeline | None = None
    shifts: dict = field(default_factory=dict)
    anchor_rules: dict[str, AnchorRule] = field(default_factory=dict)

    def variant_names(self) -> tuple[str, ...]:
        return tuple(sorted({v.name for v in self.variants.values()}))

    def variant_needed(self, variant: str) -> str | None:
        return None


@functools.lru_cache(maxsize=1)
def families() -> tuple[VideoFamily, ...]:
    from . import families as pkg

    found: list[VideoFamily] = []
    for info in pkgutil.iter_modules(pkg.__path__):
        if not info.ispkg:
            continue
        mod: Any = importlib.import_module(
            f"{pkg.__name__}.{info.name}.family")
        found.append(mod.FAMILY)
    return tuple(sorted(found, key=lambda f: (f.order, f.name)))


def get_family(name: str) -> VideoFamily:
    for fam in families():
        if fam.name == name:
            return fam
    known = ", ".join(f.name for f in families())
    raise ValueError(f"unknown video family {name!r}; known: {known}")
