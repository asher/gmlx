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
    refused: dict = field(default_factory=dict)    # hash -> what it is
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
    # (spec, tokenizer, prompt, sizes) -> (ids, vision spans) of a prompt
    # with keyframes of sizes (width, height), (width, height) -> a
    # keyframe's vision tokens, and (encoder handle, prompt, images) ->
    # (features (1, L, D), vision spans), for a family that takes
    # keyframes.
    prompt_ids: Callable | None = None
    vision_tokens: Callable | None = None
    keyframe_encode: Callable | None = None
    # How a chat model rewrites a typed prompt (image.registry's
    # UpsampleParts), and (system prompt text, frames, first, last) -> the
    # system prompt of a clip of that length with those keyframes.
    upsample: object = None
    upsample_system: Callable | None = None
    # Path patterns (fnmatch) of the DiT linears that int8 linears convert,
    # each to the columns of one activation scale: None for the whole row.
    int8_linears: dict = field(default_factory=dict)

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


# The DiT's two int8 properties: its attention, a setting of the loaded
# DiT, and the form of its block linears, a property of its weights that
# a load sets (``file`` runs them as the GGUF stores them).
ATTENTION = ("dense", "int8")
LINEARS = ("file", "int8")
# The one int8 setting a run takes: ``--int8``, an entry's
# ``defaults.int8``, and the ``int8`` field of a job or a step. Each value
# maps to the two properties (``int8_parts``).
INT8 = ("off", "attention", "linears", "all")
_PARTS = {"off": ("dense", "file"), "attention": ("int8", "file"),
          "linears": ("dense", "int8"), "all": ("int8", "int8")}


def int8_parts(value: str) -> tuple[str, str]:
    """The DiT's (attention, linears) of an int8 setting. Raises
    ValueError for a value off ``INT8``, a request's list or object
    among them."""
    if not isinstance(value, str) or value not in _PARTS:
        raise ValueError(f"int8 is {', '.join(INT8[:-1])} or {INT8[-1]}, "
                         f"not {value!r}")
    return _PARTS[value]


def int8_served(entry: str) -> tuple[str, str]:
    """The int8 values a request may name on an entry at ``entry``: the
    two whose linears match its loaded DiT's."""
    linears = int8_parts(entry)[1]
    a, b = (v for v in INT8 if _PARTS[v][1] == linears)
    return a, b


def int8_request_refusal(entry: str, value: str) -> str | None:
    """Why a job or a step naming int8 ``value`` does not run on an entry
    loaded at ``entry``, or None. The linears are a property of the loaded
    DiT, so a request may change the attention alone."""
    allowed = int8_served(entry)
    if value in allowed:
        return None
    return (f"this model's entry loads its DiT at int8 {entry}, so a "
            f"request names int8 {allowed[0]} or {allowed[1]}, not "
            f"{value}; set the entry's defaults.int8 to run {value}")


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
                f"has no vsa_attention; update mlx-kquant to 0.5.9 or later")
    if not kq.nax_available():
        return (f"{what}, whose kernel needs a GPU with NAX units (M5 or "
                f"newer), and this Mac's GPU has none")
    return None


def attention_refusal(fam: VideoFamily, variant: str,
                      attention: str) -> str | None:
    """Why this install or Mac does not run ``variant`` with
    ``attention``, or None. Int8 attention replaces dense attention only,
    and mlx-kquant's ``sdpa_qk8`` runs it, on a GPU with NAX units (M5
    and later)."""
    if attention == "dense":
        return None
    import mlx_kquant as kq

    row = fam.variant_row(variant)
    if row is not None and row.vsa_sparsity:
        return (f"{fam.title} {variant} runs video sparse attention, and "
                f"int8 attention applies to dense attention only")
    what = f"{fam.title} {variant} runs with int8 attention"
    if getattr(kq, "sdpa_qk8", None) is None:
        return (f"{what}, and the installed mlx-kquant {kq.__version__} "
                f"has no sdpa_qk8; update mlx-kquant to 0.5.10 or later")
    if not kq.nax_available():
        return (f"{what}, whose kernel needs a GPU with NAX units (M5 or "
                f"newer), and this Mac's GPU has none")
    return None


def int8_fallback(fam: VideoFamily, variant: str) -> str:
    """The int8 settings that keep the file's linears on ``variant``: off
    alone where its sparse attention refuses int8 attention."""
    row = fam.variant_row(variant)
    return "off" if row is not None and row.vsa_sparsity else (
        "off or attention")


def linears_refusal(fam: VideoFamily, variant: str,
                    linears: str) -> str | None:
    """Why this install or Mac does not run ``variant`` with ``linears``,
    or None. mlx-kquant's ``int8_quantize_rows`` and ``int8_matmul`` run
    int8 linears, the matmul on a GPU with NAX units (M5 and later), with
    dense or video sparse attention."""
    if linears == "file":
        return None
    import mlx_kquant as kq

    if not fam.int8_linears:
        return f"gmlx runs {fam.title} with its file's linears only"
    what = f"{fam.title} {variant} runs with int8 linears"
    if (getattr(kq, "int8_matmul", None) is None
            or getattr(kq, "int8_quantize_rows", None) is None):
        return (f"{what}, and the installed mlx-kquant {kq.__version__} "
                f"has no int8_matmul; update mlx-kquant to 0.5.11 or later")
    if not kq.nax_available():
        return (f"{what}, whose kernel needs a GPU with NAX units (M5 or "
                f"newer), and this Mac's GPU has none")
    return None


def int8_refusal(fam: VideoFamily, variant: str, int8: str) -> str | None:
    """Why this install or Mac does not run ``variant`` at the int8
    setting ``int8``, or None: its attention, then its linears."""
    attention, linears = int8_parts(int8)
    return (attention_refusal(fam, variant, attention)
            or linears_refusal(fam, variant, linears))


def run_refusal(fam: VideoFamily, variant: str,
                int8: str = "off") -> str | None:
    """Why this Mac or install does not run ``variant`` of ``fam`` at the
    int8 setting ``int8``, or None: its chip, then for a VSA variant the
    kernel, then the int8 setting. A run or a pull checks it before it
    loads or downloads anything."""
    why = chip_refusal(fam)
    row = fam.variant_row(variant)
    if why is None and row is not None and row.vsa_sparsity:
        why = vsa_refusal(fam, variant)
    if why is None:
        why = int8_refusal(fam, variant, int8)
    return why


def get_family(name: str) -> VideoFamily:
    for fam in families():
        if fam.name == name:
            return fam
    known = ", ".join(f.name for f in families())
    raise ValueError(f"unknown video family {name!r}; known: {known}")
