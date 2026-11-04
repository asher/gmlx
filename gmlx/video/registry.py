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
    (``merged``). ``labels``, on an identity row, are the
    ``general.architecture`` values its files carry beyond the family's
    own. A variant that runs video sparse attention (VSA) gives the
    fraction of video key tiles each video query tile skips
    (``vsa_sparsity``), and its files carry the family's VSA gate in every
    block."""
    name: str
    repo: str
    revision: str
    files: tuple
    layout: str = "qkv"
    base: str = ""
    merged: str = ""
    labels: frozenset = frozenset()
    vsa_sparsity: float = 0.0


@dataclass(frozen=True)
class VideoSampling:
    """A variant's sampling defaults, with their source. ``steps`` counts
    DiT forwards. ``shifts`` maps each stream to its static flow shift.
    ``rungs``, for a variant distilled on fixed timesteps, holds them on
    the 1000-step clock, one per forward: a run of that many steps
    samples them in place of the even grid."""
    steps: int
    shifts: dict
    repo: str
    revision: str
    rungs: tuple = ()


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
    # (frames, width, height) -> the video and audio latent shapes, and
    # (video latent frames) -> the clip's frames, None off the clip grid.
    latent_shapes: Callable | None = None
    clip_frames: Callable | None = None
    # config -> LoraKeyMap, and (safetensors header, config) -> (why,
    # kind) for a LoRA file gmlx refuses, or None.
    lora_map: Callable | None = None
    lora_refusal: Callable | None = None
    # Path patterns (fnmatch) of the linear leaves that see float32 rows
    # under any activation dtype, as the image Family has them.
    lora_f32: tuple = ()
    # The name ending of the VSA gate tensors, which only the files of a
    # VSA variant carry.
    vsa_gate: str = ""

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
        """The defaults a variant samples with: its own, else its base's.
        A file of no known variant takes the unknown variant's. None for
        a known variant gmlx video has no defaults for."""
        row = self.sampling.get(variant)
        v = self.variant_row(variant)
        if row is None and v is not None and v.base:
            row = self.sampling.get(v.base)
        if row is None and v is None and self.unknown_variant is not None:
            row = self.sampling.get(self.unknown_variant)
        return row

    def defaults(self, variant: str) -> VideoSampling:
        """``sampling_row``, or ValueError for a variant without one."""
        row = self.sampling_row(variant)
        if row is None:
            raise ValueError(f"gmlx video has no sampling defaults for "
                             f"{self.title} {variant}, so it does not run "
                             f"it from the command line")
        return row

    def rungs(self, variant: str, steps: int) -> tuple:
        """The trained timesteps a run of ``steps`` forwards samples
        ``variant`` on: its row's when they are ``steps``, else () for
        the even grid."""
        row = self.sampling_row(variant)
        if row is None or len(row.rungs) != steps:
            return ()
        return row.rungs


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


def chip_refusal(fam: VideoFamily) -> str | None:
    """Why this Mac does not run ``fam``, or None. A video DiT runs in
    bfloat16, which an M1 or M2 has only in software."""
    from gmlx.load.dtypes import has_native_bfloat16

    if has_native_bfloat16():
        return None
    from gmlx.image.dtypes import chip_name

    return (f"{fam.title} needs an M3 or newer, and this Mac has an "
            f"{chip_name()} chip")


def vsa_refusal(fam: VideoFamily, variant: str) -> str | None:
    """Why this install or Mac does not run ``variant``'s video sparse
    attention, or None: mlx-kquant's ``vsa_attention`` runs it, on a GPU
    with NAX units (M5 and later)."""
    import mlx_kquant as kq

    what = f"{fam.title} {variant} runs video sparse attention"
    if getattr(kq, "vsa_attention", None) is None:
        return (f"{what}, and the installed mlx-kquant {kq.__version__} "
                f"has no vsa_attention; update mlx-kquant to a release "
                f"that has it")
    if not kq.nax_available():
        return (f"{what}, whose kernel needs a GPU with NAX units (M5 or "
                f"newer), and this Mac's GPU has none")
    return None


def run_refusal(fam: VideoFamily, variant: str) -> str | None:
    """Why this Mac or install does not run ``variant`` of ``fam``, or
    None: its chip, then for a VSA variant the kernel. A run or a pull
    checks it before it loads or downloads anything."""
    why = chip_refusal(fam)
    row = fam.variant_row(variant)
    if why is None and row is not None and row.vsa_sparsity:
        why = vsa_refusal(fam, variant)
    return why


def get_family(name: str) -> VideoFamily:
    for fam in families():
        if fam.name == name:
            return fam
    known = ", ".join(f.name for f in families())
    raise ValueError(f"unknown video family {name!r}; known: {known}")
