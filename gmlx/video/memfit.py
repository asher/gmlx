"""The memory plan of a video run, from file sizes and the clip's size.

A run holds one model at a time, in four phases: the encode, the
denoise, the audio decode and the video decode. MLX counts
memory-mapped weights as active memory, so each phase is its files plus
a line fit of what the run adds, both read as MLX's active peak.

- encode: the larger of the load, which maps the text encoder's file
  whole, and the run, which keeps part of the file once the layers past
  the tap drop and adds the prompt's activations by text row.
- denoise: the DiT's file, each LoRA at ``lora_k`` times its file (its
  factors and the table built from them), and one step's activations by
  row of the packed sequence.
- audio: the audio VAE's file and its decode by audio latent.
- decode: the video VAE's file, or TAEH3's for a preview, and one chunk
  of frames by output pixel. A chunk does not grow with the clip.

The plan fits when every phase is within the working budget of
``gmlx.serve.capacity``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

GiB = 1 << 30
PHASES = ("encode", "denoise", "audio", "decode")


@dataclass(frozen=True)
class Line:
    """``base`` bytes and ``k`` bytes per unit."""
    base: float
    k: float

    def bytes(self, n: float) -> float:
        return self.base + self.k * n


@dataclass(frozen=True)
class VideoPeaks:
    """One family's fits, in bytes over its files."""
    encoder_load: float             # the encoder's load, over its file
    encoder_keep: float             # the part of the encoder file a run keeps
    encode: Line                    # per text row, over the kept part
    step: Line                      # per packed row, with a live LoRA
    plain_step: Line                # per packed row, with no LoRA
    audio: Line                     # per audio latent
    decode: Line                    # per output pixel
    tiny_decode: Line               # per output pixel, TAEH3
    lora_k: float = 2.0


# MiniMax-H3 Q4_K, the stock Qwen3-VL-32B Q4_K_M encoder, Comfy-Org's
# VAEs and lightx2v's 8-step LoRA, with its 4-step LoRA stacked on it in
# one cell, on an M5 Max under bounded_cache, one shape per process. The
# cells are in tests/fixtures/video/memfit_h3_cells.json. Each line runs
# through two of its cells, named below, with the slope raised 10 percent
# and the base at least 0.25 GiB above every cell.
#
# The encoder's load peaks 1.2 MB over its file. The run then keeps 75.5
# percent of the file, and a prompt of 26 to 5001 tokens reads 0.13 to
# 1.61 GiB over that, through the 1001 and 5001 token cells. A plan
# counts the prompt's rows by the encoder's own tokenizer.
#
# A step reads 0.88 to 21.41 GiB over the DiT and twice each LoRA from
# 1891 to 109087 rows (15 s at 1344x768), and 0.69 to 6.35 GiB over the
# DiT with no LoRA up to 37735 rows. A second LoRA adds 0.03 GiB to the
# step. Both lines run through the 21751 and 37735 row cells, which puts
# the LoRA line 2.91 GiB over the 109087 row cell.
#
# The audio decode reads 1.30 to 3.59 GiB over its file from 37 to 603
# latents (a 15 s clip has 603), through the 150 and 603 latent cells.
# The video decode reads 1.10 to 1.49 GiB over its file from 512x512 to
# 1344x768, 0.05 GiB more at 362 frames than at 124, and TAEH3 0.90 to
# 3.00 GiB, each through its 512x512 and 1344x768 cells.
PEAKS = {
    "h3": VideoPeaks(
        encoder_load=0.26 * GiB,
        encoder_keep=0.76,
        encode=Line(0.80 * GiB, 255_000),
        step=Line(0.73 * GiB, 232_200),
        plain_step=Line(0.60 * GiB, 195_800),
        audio=Line(1.97 * GiB, 3_980_000),
        decode=Line(1.24 * GiB, 530),
        tiny_decode=Line(0.43 * GiB, 3_200)),
}

# What to change when a phase does not fit.
_HINTS = {"encode": "the text encoder needs more memory than this Mac "
                    "has, and gmlx has no smaller one for this model yet",
          "prompt": "try a shorter prompt",
          "denoise": "try a smaller size or a shorter clip",
          "audio": "try a shorter clip",
          "decode": "try a smaller size, or --preview"}


class VideoMemoryError(RuntimeError):
    """A run with a phase that does not fit the working budget."""


def _gib(b: float, up: bool) -> str:
    x = b / GiB * 100
    return f"{(math.ceil(x) if up else math.floor(x)) / 100:.2f}"


def _and(words: list) -> str:
    return words[0] if len(words) == 1 else (
        ", ".join(words[:-1]) + " and " + words[-1])


@dataclass
class VideoPlan:
    phases: dict                    # phase -> bytes
    terms: dict                     # phase -> {term: bytes}
    budget: float | None
    rows: int
    encoder_load: int = 0           # the encode phase of a short prompt

    @property
    def peak(self) -> int:
        return max(self.phases.values())

    @property
    def largest(self) -> str:
        return max(self.phases, key=self.phases.__getitem__)

    @property
    def over(self) -> list:
        """The phases over the budget, in run order."""
        if self.budget is None:
            return []
        return [p for p, b in self.phases.items() if b > self.budget]

    @property
    def fits(self) -> bool:
        return not self.over

    def hint(self) -> str:
        """What to change: the encoder first, since no clip makes it
        fit, then the largest phase over the budget."""
        over = self.over
        if not over:
            return ""
        if "encode" in over:
            assert self.budget is not None
            return _HINTS["prompt" if self.encoder_load <= self.budget
                          else "encode"]
        return _HINTS[max(over, key=self.phases.__getitem__)]

    def describe(self) -> str:
        """The phases against the budget. Against a budget, a phase over
        it rounds up and the rest round down, so none reads on the wrong
        side of it."""
        if self.budget is None:
            return ", ".join(f"{p} {b / GiB:.2f} GiB"
                             for p, b in self.phases.items())
        over = self.over
        parts = ", ".join(f"{p} {_gib(b, p in over)} GiB"
                          for p, b in self.phases.items())
        text = (f"{parts} against a working budget of "
                f"{_gib(self.budget, False)} GiB")
        if over:
            text += f", with {_and([f'the {p}' for p in over])} over it"
        return text


def plan_memory(family: str, *, dit: int, encoder: int, vae: int,
                audio_vae: int, loras=(), rows: int, text: int,
                audio_latents: int, pixels: int, tiny: bool = False,
                decode: bool = True,
                budget: float | None = None) -> VideoPlan:
    """The plan of one run. File sizes in bytes; ``rows`` is the packed
    sequence's length, ``text`` the prompt's rows, ``audio_latents`` the
    audio stream's length and ``pixels`` a frame's width times height.
    ``vae`` is TAEH3's file when ``tiny``. With ``decode`` False the plan
    has no audio or decode phase. ``budget`` defaults to
    ``capacity.working_budget_bytes()``."""
    p = PEAKS[family]
    load = encoder + p.encoder_load
    kept, prompt = p.encoder_keep * encoder, p.encode.bytes(text)
    lora = sum(int(b) for b in loras)
    step = p.step if lora else p.plain_step
    terms = {
        "encode": ({"encoder": encoder, "load": p.encoder_load}
                   if load >= kept + prompt
                   else {"encoder": kept, "prompt": prompt}),
        "denoise": {"dit": dit, "lora": p.lora_k * lora,
                    "step": step.bytes(rows)},
    }
    if decode:
        terms["audio"] = {"audio_vae": audio_vae,
                          "decode": p.audio.bytes(audio_latents)}
        terms["decode"] = {"vae": vae, "chunk": (
            p.tiny_decode if tiny else p.decode).bytes(pixels)}
    if budget is None:
        from gmlx.serve.capacity import working_budget_bytes

        budget = working_budget_bytes()
    phases = {ph: int(sum(t.values())) for ph, t in terms.items()}
    return VideoPlan(phases=phases, terms=terms, budget=budget, rows=rows,
                     encoder_load=int(load))


def prompt_rows(fam, prompt: str, tokenizer) -> int:
    """The text rows ``prompt`` encodes to: its token ids by the
    encoder's ``tokenizer``, less the dropped ones, plus the pads."""
    from gmlx.image.text_encoder import pad_rows, token_ids

    spec = fam.encoder()
    n = len(token_ids(spec, tokenizer, prompt))
    return n - spec.drop + pad_rows(spec, n)


def clip_rows(fam, *, frames: int, width: int, height: int,
              text: int) -> tuple[int, int]:
    """The packed rows of a clip with ``text`` prompt rows, and its audio
    latents."""
    from .layout import pack

    vshape, ashape = fam.latent_shapes(frames, width, height)
    lay = pack(fam.timeline, text, vshape[2], vshape[3], vshape[4],
               ashape[3])
    return int(lay.rows), int(ashape[3])


def file_bytes(path: str) -> int:
    """A file's bytes, every shard of a split GGUF."""
    from gmlx.load.preflight import find_split_shards

    path = os.path.expanduser(path)
    if path.endswith(".gguf"):
        return sum(os.path.getsize(s) for s in find_split_shards(path))
    return os.path.getsize(path)


def plan_run(fam, *, dit: str, encoder: str, vae: str, audio_vae: str,
             loras=(), frames: int, width: int, height: int, text: int,
             tiny: bool = False, budget: float | None = None) -> VideoPlan:
    """``plan_memory`` from the files at these paths."""
    rows, audio = clip_rows(fam, frames=frames, width=width, height=height,
                            text=text)
    return plan_memory(fam.name, dit=file_bytes(dit),
                       encoder=file_bytes(encoder), vae=file_bytes(vae),
                       audio_vae=file_bytes(audio_vae),
                       loras=[file_bytes(p) for p in loras], rows=rows,
                       text=text, audio_latents=audio,
                       pixels=width * height, tiny=tiny, budget=budget)


def check(plan: VideoPlan, what: str = "this clip") -> None:
    """Raise ``VideoMemoryError`` when the plan does not fit, unless
    GMLX_OVERCOMMIT=1."""
    from gmlx.serve import capacity

    if plan.fits or capacity.overcommit():
        return
    hint = plan.hint()
    raise VideoMemoryError(
        f"{what} does not fit in memory: {plan.describe()}. "
        f"{hint[0].upper()}{hint[1:]}. GMLX_OVERCOMMIT=1 overrides.")
