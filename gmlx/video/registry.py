"""Video model families and their discovery.

Each directory under ``gmlx.video.families`` holds a ``family.py`` that
defines ``FAMILY``. ``families()`` imports them all and sorts by order and
name. A family carries its own values for the shared mechanisms: the
segment order and clock of the packed sequence (``timeline``, read by
``layout.pack``), each stream's schedule shift (``shifts``) and anchor
rule (``anchor_rules``), and its text encoder (``encoder``).
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
    the order of the fused qkv rows in its files (``layout``). A variant
    with a LoRA merged into another's weights names that variant
    (``base``), whose identity tensors it keeps, and the LoRA
    (``merged``)."""
    name: str
    repo: str
    revision: str
    files: tuple
    layout: str = "qkv"
    base: str = ""
    merged: str = ""


@dataclass(frozen=True)
class VideoSampling:
    """A variant's sampling defaults, with their source. ``steps`` counts
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
    sampling: dict = field(default_factory=dict)   # variant -> VideoSampling
    unknown_variant: str | None = None
    reshaped: tuple = ()
    # The shared mechanisms' values: the clock of the packed sequence,
    # each stream's static shift and anchor rule.
    timeline: Timeline | None = None
    shifts: dict = field(default_factory=dict)
    anchor_rules: dict[str, AnchorRule] = field(default_factory=dict)
    # () -> EncoderSpec of the text encoder, built on call so this module
    # loads no mlx.
    encoder: Callable | None = None
    # (frames, width, height) -> the video and audio latent shapes.
    latent_shapes: Callable | None = None
    # config -> LoraKeyMap, and (safetensors header, config) -> (why,
    # kind) for a LoRA file gmlx refuses, or None.
    lora_map: Callable | None = None
    lora_refusal: Callable | None = None

    def variant_names(self) -> tuple[str, ...]:
        return tuple(sorted({v.name for v in (*self.variants.values(),
                                              *self.layouts.values())}))

    def variant_row(self, name: str) -> VideoVariantRow | None:
        return next((v for v in (*self.variants.values(),
                                 *self.layouts.values()) if v.name == name),
                    None)

    def variant_needed(self, variant: str) -> str | None:
        return None

    def sampling_row(self, variant: str) -> VideoSampling | None:
        """The defaults a variant samples with: its own, else its base's,
        else the unknown variant's."""
        row = self.sampling.get(variant)
        v = self.variant_row(variant)
        if row is None and v is not None and v.base:
            row = self.sampling.get(v.base)
        if row is None and self.unknown_variant is not None:
            row = self.sampling.get(self.unknown_variant)
        return row


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
