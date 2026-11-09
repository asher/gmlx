"""The memory plan of a video run, from file sizes and the clip's size.

A run holds one model at a time, in five phases: the encode, the
anchors, the denoise, the audio decode and the video decode. MLX counts
memory-mapped weights as active memory, so each phase is its files plus
a line fit of what the run adds, both read as MLX's active peak.

- encode: the larger of the load, which maps the text encoder's file
  whole, and the run, which keeps part of the file once the layers past
  the tap drop and adds the prompt's activations by text row. A prompt
  with keyframes adds two terms: the attach, which reads the mmproj into
  the vision tower at float32 beside the kept part, and a run that holds
  the tower too and is the larger of the tower's pass over the largest
  keyframe and the language model's pass over the prompt's rows.
- anchors: the keyframes and guides through the video VAE, one at a
  time, a lone frame by pixel and a clip by pixel and frame-pixel, or
  the guide audio through the audio VAE by audio latent, whichever is
  larger. A run with no anchors has no such phase.
- denoise: the DiT's file, its LoRAs, and one step's activations by row
  of the packed sequence, on the variant's own fit where ``PEAKS`` has
  one (FastH3, whose steps run video sparse attention and its gates).
  A lone LoRA in the DiT's activation dtype runs on its file's arrays and
  costs its file once. Any other LoRA costs ``lora_k`` times its file:
  its arrays and the copies the patch table builds from them. A step
  with int8 attention adds the int8 copies of q and k and their scales
  by row. A DiT with int8 linears holds the int8 weights and scales of
  its converted linears in place of their file bytes, counted from the
  header (``int8_delta``), and its step adds the largest int8 copy of a
  linear's input by row.
- audio: the audio VAE's file and its decode by audio latent.
- decode: the video VAE's file, or TAEH3's for a preview, and one chunk
  of frames by output pixel. A chunk does not grow with the clip.

The plan fits when every phase is within the working budget of
``gmlx.serve.capacity``.
"""

from __future__ import annotations

import dataclasses
import math
import os
from dataclasses import dataclass

GiB = 1 << 30
PHASES = ("encode", "anchors", "denoise", "audio", "decode")


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
    # A prompt with keyframes, over the kept part and the vision tower:
    tower: Line                     # per vision token of the largest one
    vision_encode: Line             # per prompt row
    # The anchors, each over its VAE's file:
    frame_encode: Line              # per pixel, a lone frame
    clip_encode: Line               # per pixel, a guide clip
    clip_frame: float               # per frame-pixel of a guide clip
    audio_encode: Line              # per audio latent
    lora_dtype: str = "BF16"        # the DiT's activation dtype
    lora_k: float = 2.0
    int8_attention: float = 0.0     # per packed row, a step's int8 q and k
    int8_linears: float = 0.0       # per packed row, an int8 linear's input


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
# A step reads 0.74 to 16.67 GiB over the DiT and its LoRA from 1891 to
# 109087 rows (15 s at 1344x768), and 0.57 to 5.31 GiB over the DiT with
# no LoRA up to 37735 rows. A second LoRA, at twice each file, adds 0.03
# GiB to the step. Both lines run through the 21751 and 37735 row cells,
# which puts the LoRA line 2.35 GiB over the 109087 row cell.
#
# The audio decode reads 1.30 to 3.59 GiB over its file from 37 to 603
# latents (a 15 s clip has 603), through the 150 and 603 latent cells.
# The video decode reads 1.10 to 1.49 GiB over its file from 512x512 to
# 1344x768, 0.05 GiB more at 362 frames than at 124, and TAEH3 0.90 to
# 3.00 GiB, each through its 512x512 and 1344x768 cells.
#
# Keyframes and anchors, on the same encoder with unsloth's
# mmproj-BF16.gguf and the same VAEs, in
# tests/fixtures/video/memfit_h3_keyframe_cells.json. The encoder loads
# its text part first, at the load's own peak, and the attach then holds
# 0.65 GB over the tower it leaves at float32. Over the kept part and the
# tower, the tower's pass over one keyframe reads 0.12 to 1.31 GB from 60
# to 1008 vision tokens, through the 576 and 1008 token cells, and 0.80
# and 1.38 GB with a second keyframe of 576 and 1008 tokens beside the
# first one's features. A prompt with keyframes reads 0.16 to 2.53 GB
# from 82 to 7283 rows, the tower's pass the larger at one 1344x768
# keyframe; its line runs through the 1182 and 7283 row cells. A lone frame through the
# video VAE reads 0.86 to 0.96 GB over its file from 320x192 to 1344x768,
# through the 768x768 and 1344x768 cells; a guide clip 2.68 to 5.80 GB
# from 320x192 at 22 frames to 1344x768 at 124, through the 672x384 and
# 1344x768 cells at 22 frames by pixel and the 22 and 124 frame cells at
# 1344x768 by frame-pixel. The audio VAE reads 1.10 to 1.87 GB over its
# file from 40 to 600 latents, through the 160 and 600 latent cells.
#
# Int8 attention holds, beside a dense step, the int8 q and k of 56 heads
# of 128 and their float32 scales, one a head: 2 x 56 x 128 + 2 x 56 x 4
# bytes a packed row. The term is counted, not measured.
#
# Int8 linears hold, beside a step on the file's linears, the int8 copy
# of fc2's input, the widest, and its float32 scales, one per 1024
# columns: 14336 + 14 x 4 bytes a packed row. The term is counted, not
# measured.
PEAKS = {
    "h3": VideoPeaks(
        encoder_load=0.26 * GiB,
        encoder_keep=0.76,
        encode=Line(0.80 * GiB, 255_000),
        step=Line(0.69 * GiB, 180_400),
        plain_step=Line(0.54 * GiB, 163_500),
        audio=Line(2.01 * GiB, 3_980_000),
        decode=Line(1.24 * GiB, 530),
        tiny_decode=Line(0.43 * GiB, 3_200),
        tower=Line(0.29 * GiB, 1_378_900),
        vision_encode=Line(1.16 * GiB, 273_000),
        frame_encode=Line(1.12 * GiB, 26),
        clip_encode=Line(3.52 * GiB, 303),
        clip_frame=17,
        audio_encode=Line(1.84 * GiB, 301_400),
        int8_attention=14_784,
        int8_linears=14_392),
}
# FastH3: vantagewithai's Q4_K_M with its VSA gates, measured as the H3
# cells on the same encoder, VAEs and LoRA, in
# tests/fixtures/video/memfit_fasth3_cells.json. Its encode and decodes
# run H3's files, and H3's fits cover them. A step runs video sparse
# attention, whose call holds more than dense attention at a large clip:
# 0.56 to 8.02 GiB over the DiT from 1891 to 37735 rows, against H3's
# 0.57 to 5.31, and 23.41 GiB at 109087 rows. A lone bf16 LoRA adds its
# file and no more, so the two lines are one, fit as H3's are through
# the 21751 and 37735 row cells, which puts it 3.69 GiB over the 109087
# row cell.
PEAKS["fasth3"] = dataclasses.replace(
    PEAKS["h3"], step=Line(0.35 * GiB, 263_300),
    plain_step=Line(0.35 * GiB, 263_300))

# What to change when a phase does not fit.
_HINTS = {"encode": "the text encoder needs more memory than this Mac "
                    "has, and gmlx has no smaller one for this model yet",
          "prompt": "try a shorter prompt",
          "keyframes": "try a shorter prompt or a smaller size, or one "
                       "keyframe",
          "anchors": "try a smaller size or a shorter guide clip",
          "denoise": "try a smaller size or a shorter clip",
          "linears": "run with int8 off or attention, which keep the "
                     "file's linears, or try a smaller size or a shorter "
                     "clip",
          "linears_off": "run with int8 off, which keeps the file's "
                         "linears, or try a smaller size or a shorter clip",
          "audio": "try a shorter clip",
          "decode": "try a smaller size, or --preview"}


class VideoMemoryError(RuntimeError):
    """A run with a phase that does not fit the working budget."""


def gib(b: float, up: bool) -> str:
    """``b`` bytes in GiB at 2 decimals, rounded up when ``up`` and down
    otherwise. A refusal rounds a need up and its budget down, so the need
    never reads at or under the budget."""
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
    encoder_load: int = 0           # the encode's part no prompt changes
    keyframes: bool = False         # the encode runs the vision tower
    linears: str = "file"           # the DiT's block linears
    fallback: str = "off or attention"  # int8 settings that keep them

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

    @property
    def cause(self) -> str | None:
        """The ``_HINTS`` key of what to change, or None when the plan
        fits: the encoder first, since no clip makes it fit, then the
        largest phase over the budget."""
        over = self.over
        if not over:
            return None
        if "encode" in over:
            assert self.budget is not None
            if self.encoder_load > self.budget:
                return "encode"
            return "keyframes" if self.keyframes else "prompt"
        return max(over, key=self.phases.__getitem__)

    def hint(self) -> str:
        """What to change. A denoise over the budget with int8 linears
        names the file's linears first."""
        cause = self.cause
        if cause == "denoise" and self.linears == "int8":
            cause = "linears_off" if self.fallback == "off" else "linears"
        return _HINTS[cause] if cause else ""

    def describe(self) -> str:
        """The phases against the budget. Against a budget, a phase over
        it rounds up and the rest round down, so none reads on the wrong
        side of it."""
        if self.budget is None:
            return ", ".join(f"{p} {b / GiB:.2f} GiB"
                             for p, b in self.phases.items())
        over = self.over
        parts = ", ".join(f"{p} {gib(b, p in over)} GiB"
                          for p, b in self.phases.items())
        text = (f"{parts} against a working budget of "
                f"{gib(self.budget, False)} GiB")
        if over:
            text += f", with {_and([f'the {p}' for p in over])} over it"
        return text


def plan_memory(family: str, *, dit: int, encoder: int, vae: int,
                audio_vae: int, loras=(), lora_dtypes=(), rows: int,
                text: int, audio_latents: int, pixels: int,
                tiny: bool = False, decode: bool = True,
                variant: str | None = None, live_lora: bool = False,
                mmproj: int = 0, tower: int = 0, blocks=(),
                anchor_frames=(), anchor_audio=(),
                anchor_vae: int | None = None, attention: str = "dense",
                linears: str = "file",
                budget: float | None = None) -> VideoPlan:
    """The plan of one run of ``variant``, on its own fits where ``PEAKS``
    has them, else its family's. File sizes in bytes; ``lora_dtypes`` holds
    each LoRA's ``lora_dtype`` result, and a LoRA with none is priced
    with copies. ``rows`` is the packed sequence's length, ``text`` the
    prompt's rows, ``audio_latents`` the audio stream's length and
    ``pixels`` a frame's width times height. ``vae`` is TAEH3's file
    when ``tiny``. With ``decode`` False the plan has no audio or decode
    phase. ``live_lora`` prices the step with a LoRA live when ``loras``
    names none, as a patch table the server was sent runs it.

    ``blocks`` holds each keyframe's vision tokens, whose rows ``text``
    counts; the encode then attaches the vision tower of the ``mmproj``
    file, ``tower`` bytes once loaded (``tower_bytes``). The anchors
    phase encodes ``anchor_frames``, the frames of each keyframe and
    guide at ``pixels``, through the video VAE file ``anchor_vae``
    (``vae`` when None), and ``anchor_audio``, the latents of each guide
    audio. A step with ``attention`` ``"int8"`` holds the int8 copies of
    q and k too, and with ``linears`` ``"int8"`` the int8 copy of a
    linear's input; ``dit`` then counts the int8 weights
    (``dit_bytes``). ``budget`` defaults to
    ``capacity.working_budget_bytes()``."""
    p = PEAKS[variant] if variant in PEAKS else PEAKS[family]
    load = encoder + p.encoder_load
    kept = p.encoder_keep * encoder
    lora = sum(int(b) for b in loras)
    alone = len(loras) == 1 and list(lora_dtypes) == [p.lora_dtype]
    step = p.step if lora or live_lora else p.plain_step
    terms = {"encode": _encode_terms(p, encoder, kept, text, mmproj, tower,
                                     tuple(int(b) for b in blocks))}
    floor = load
    if blocks:
        from gmlx.image.memfit import LOAD_ROOM

        floor = max(load, kept + tower + LOAD_ROOM * mmproj)
    if anchor_frames or anchor_audio:
        if anchor_frames and anchor_vae is None and tiny:
            raise ValueError("a preview's plan needs the video VAE its "
                             "anchors encode through, as anchor_vae")
        terms["anchors"] = _anchor_terms(
            p, pixels, [int(f) for f in anchor_frames],
            [int(n) for n in anchor_audio],
            vae if anchor_vae is None else anchor_vae, audio_vae)
    work = step.bytes(rows)
    if attention == "int8":
        work += p.int8_attention * rows
    if linears == "int8":
        work += p.int8_linears * rows
    terms["denoise"] = {"dit": dit,
                        "lora": lora if alone else p.lora_k * lora,
                        "step": work}
    if decode:
        terms["audio"] = {"audio_vae": audio_vae,
                          "decode": p.audio.bytes(audio_latents)}
        terms["decode"] = {"vae": vae, "chunk": (
            p.tiny_decode if tiny else p.decode).bytes(pixels)}
    if budget is None:
        from gmlx.serve.capacity import working_budget_bytes

        budget = working_budget_bytes()
    phases = {ph: int(sum(t.values())) for ph, t in terms.items()}
    from .registry import get_family, int8_fallback

    return VideoPlan(phases=phases, terms=terms, budget=budget, rows=rows,
                     encoder_load=int(floor), keyframes=bool(blocks),
                     linears=linears, fallback=int8_fallback(
                         get_family(family), variant or ""))


def _encode_terms(p: VideoPeaks, encoder: int, kept: float, text: int,
                  mmproj: int, tower: int, blocks: tuple) -> dict:
    """The encode phase's terms: its load, or a run over the kept part,
    and with keyframes the tower's attach, whichever is the largest."""
    if not blocks:
        prompt = p.encode.bytes(text)
        if encoder + p.encoder_load >= kept + prompt:
            return {"encoder": encoder, "load": p.encoder_load}
        return {"encoder": kept, "prompt": prompt}
    from gmlx.image.memfit import LOAD_ROOM

    run = max(p.vision_encode.bytes(text), p.tower.bytes(max(blocks)))
    parts = ({"encoder": encoder, "load": p.encoder_load},
             {"encoder": kept, "tower": tower,
              "attach": LOAD_ROOM * mmproj},
             {"encoder": kept, "tower": tower, "prompt": run})
    return max(parts, key=lambda t: sum(t.values()))


def _anchor_terms(p: VideoPeaks, pixels: int, frames: list, audio: list,
                  vae: int, audio_vae: int) -> dict:
    """The anchors phase's terms: the larger of the video VAE with its
    largest encode and the audio VAE with its, which run one at a
    time."""
    parts = []
    if frames:
        parts.append({"vae": vae, "encode": max(
            p.frame_encode.bytes(pixels) if f == 1
            else p.clip_encode.bytes(pixels) + p.clip_frame * f * pixels
            for f in frames)})
    if audio:
        parts.append({"audio_vae": audio_vae,
                      "encode": p.audio_encode.bytes(max(audio))})
    return max(parts, key=lambda t: sum(t.values()))


def prompt_rows(fam, prompt: str, tokenizer, sizes=()) -> int:
    """The text rows ``prompt`` encodes to: its token ids by the
    encoder's ``tokenizer``, less the dropped ones, plus the pads. With
    keyframes of ``sizes`` (width, height) the ids are the family's
    prompt with them, their vision rows in it."""
    from gmlx.image.text_encoder import pad_rows, token_ids

    spec = fam.encoder()
    if sizes:
        if fam.prompt_ids is None:
            raise ValueError(f"a {fam.title} prompt takes no keyframes")
        n = len(fam.prompt_ids(spec, tokenizer, prompt, tuple(sizes))[0])
    else:
        n = len(token_ids(spec, tokenizer, prompt))
    return n - spec.drop + pad_rows(spec, n)


def keyframe_blocks(fam, sizes) -> tuple:
    """The vision tokens of each keyframe of ``sizes`` (width, height)."""
    if fam.vision_tokens is None:
        raise ValueError(f"a {fam.title} prompt takes no keyframes")
    return tuple(int(fam.vision_tokens(w, h)) for w, h in sizes)


def clip_rows(fam, *, frames: int, width: int, height: int, text: int,
              anchors=()) -> tuple[int, int]:
    """The packed rows of a clip with ``text`` prompt rows and placed
    ``anchors`` (``layout.AnchorRows``), and its audio latents."""
    from .layout import pack

    vshape, ashape = fam.latent_shapes(frames, width, height)
    lay = pack(fam.timeline, text, vshape[2], vshape[3], vshape[4],
               ashape[3], anchors=tuple(anchors))
    return int(lay.rows), int(ashape[3])


def anchor_frames(fam, anchors) -> list:
    """The frames each anchor with video encodes, from its latent frames:
    one for a lone frame's one, else the clip's."""
    out = []
    for a in anchors:
        if a.video_t:
            f = 1 if a.video_t == 1 else fam.clip_frames(a.video_t)
            if f is None:
                raise ValueError(f"{a.video_t} latent frames are no "
                                 f"{fam.title} guide clip")
            out.append(int(f))
    return out


def tower_bytes(mmproj: str) -> int:
    """Bytes the vision tower of an mmproj holds once attached, its
    16-bit float tensors at float32, from the file's header."""
    from gmlx.image.memfit import mmproj_bytes
    from gmlx.image.text_encoder import _header_view

    return mmproj_bytes(_header_view(os.path.expanduser(mmproj)), fp32=True)


def _int8_matrices(fam, view) -> list:
    """(out, in, file bytes) of each tensor that the family's int8
    linears convert, from a DiT header, its names with or without one of
    the family's prefixes."""
    import fnmatch

    from gmlx.image.memfit import tensor_bytes

    out = []
    for name, t in view.tensors.items():
        for pre in fam.strip_prefixes:
            if name.startswith(pre):
                name = name[len(pre):]
                break
        path, _, kind = name.rpartition(".")
        if kind != "weight" or len(t.shape) != 2 or not any(
                fnmatch.fnmatchcase(path, pat) for pat in fam.int8_linears):
            continue
        k, n = t.shape
        out.append((int(n), int(k), t.nbytes or tensor_bytes(t.shape,
                                                             t.codec)))
    return out


def int8_delta(fam, view) -> int:
    """The bytes a DiT with int8 linears holds over its file: each
    converted matrix's int8 weight and float32 scale a row, less its file
    bytes, from the header ``view``. Exact, not a fit."""
    return sum(n * k + 4 * n - b for n, k, b in _int8_matrices(fam, view))


def int8_transient(fam, view) -> int:
    """The bytes a conversion holds beside the loaded DiT: the largest
    converted matrix at float32, its int8 codes and its scales."""
    return max((4 * n * k + n * k + 4 * n
                for n, k, _ in _int8_matrices(fam, view)), default=0)


def dit_bytes(fam, path: str, linears: str = "file") -> int:
    """The bytes the DiT at ``path`` holds once loaded with ``linears``:
    its file, plus ``int8_delta`` for int8 linears."""
    total = file_bytes(path)
    if linears == "int8":
        from gmlx.image.text_encoder import _header_view

        total += int8_delta(fam, _header_view(os.path.expanduser(path)))
    return total


def file_bytes(path: str) -> int:
    """A file's bytes, every shard of a split GGUF."""
    from gmlx.load.preflight import find_split_shards

    path = os.path.expanduser(path)
    if path.endswith(".gguf"):
        return sum(os.path.getsize(s) for s in find_split_shards(path))
    return os.path.getsize(path)


def lora_dtype(path: str) -> str | None:
    """The one dtype of a safetensors LoRA's matrices, as its header
    names it, or None for a file of mixed dtypes or of another form."""
    from gmlx.image.vae.reader import VAEError, safetensors_header

    try:
        header, _ = safetensors_header(os.path.expanduser(path))
    except (OSError, VAEError):
        return None
    kinds = set()
    for k, v in header.items():
        if k == "__metadata__" or not isinstance(v, dict):
            continue
        shape, kind = v.get("shape"), v.get("dtype")
        if not isinstance(shape, list) or not isinstance(kind, str):
            return None
        # A matrix, or a 1x1 kernel the reader takes as one.
        if len(shape) >= 2 and all(d == 1 for d in shape[2:]):
            kinds.add(kind)
    return kinds.pop() if len(kinds) == 1 else None


def plan_run(fam, *, dit: str, encoder: str, vae: str, audio_vae: str,
             loras=(), frames: int, width: int, height: int, text: int,
             tiny: bool = False, variant: str | None = None,
             mmproj: str | None = None, blocks=(), anchors=(),
             anchor_vae: str | None = None, attention: str = "dense",
             linears: str = "file",
             budget: float | None = None) -> VideoPlan:
    """``plan_memory`` from the files at these paths, with ``blocks``
    keyframe vision tokens through ``mmproj`` and placed ``anchors``
    (``layout.AnchorRows``) encoded through ``anchor_vae`` (``vae``
    when None) and ``audio_vae``, the DiT running ``attention`` and
    ``linears``."""
    if blocks and mmproj is None:
        raise ValueError("keyframes need the encoder's mmproj")
    vision = mmproj if blocks else None
    rows, audio = clip_rows(fam, frames=frames, width=width, height=height,
                            text=text, anchors=anchors)
    return plan_memory(fam.name, dit=dit_bytes(fam, dit, linears),
                       encoder=file_bytes(encoder), vae=file_bytes(vae),
                       audio_vae=file_bytes(audio_vae),
                       loras=[file_bytes(p) for p in loras],
                       lora_dtypes=[lora_dtype(p) for p in loras], rows=rows,
                       text=text, audio_latents=audio,
                       pixels=width * height, tiny=tiny, variant=variant,
                       mmproj=0 if vision is None else file_bytes(vision),
                       tower=0 if vision is None else tower_bytes(vision),
                       blocks=blocks, anchor_frames=anchor_frames(fam,
                                                                  anchors),
                       anchor_audio=[a.audio_t for a in anchors
                                     if a.audio_t],
                       anchor_vae=None if anchor_vae is None
                       else file_bytes(anchor_vae), attention=attention,
                       linears=linears, budget=budget)


def check(plan: VideoPlan, what: str = "this clip", resize=None) -> None:
    """Raise ``VideoMemoryError`` when the plan does not fit, unless
    GMLX_OVERCOMMIT=1. For a plan refused for its denoise, ``resize()``
    names a size or length that fits in place of the plan's own hint,
    or returns None to keep it."""
    from gmlx.serve import capacity

    if plan.fits or capacity.overcommit():
        return
    hint = plan.hint()
    if resize is not None and plan.cause == "denoise":
        sized = resize()
        if sized and plan.linears == "int8":
            hint = f"run with int8 {plan.fallback}, or {sized}"
        elif sized:
            hint = sized
    raise VideoMemoryError(
        f"{what} does not fit in memory: {plan.describe()}. "
        f"{hint[0].upper()}{hint[1:]}. GMLX_OVERCOMMIT=1 overrides.")
