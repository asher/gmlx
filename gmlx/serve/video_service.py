"""Video models in the serve residency pool.

A video entry holds a ``ServedVideo``: the pipeline over the video DiT,
with the configured adapters as its default patch table, the plan it was
priced by and a small cache of refined text rows. It runs on its own
worker thread, ``video-worker``, as an image entry runs on the image
worker, since MLX streams are per thread: a clip's steps never queue
behind an image job, nor an image behind a clip.

The pool counts an entry at the larger of two phases, as the video
command runs them one model at a time: the encode, with the text encoder
alone, and the DiT with its LoRAs and the larger of one step, the video
decode and the audio decode at the entry's default clip. Each phase also
counts the caches the server holds, as an image entry counts them: each
cache takes the room the clip leaves, up to its bound, and the entry
bounds its pipeline's caches at that share. The step routes' tensor
cache keeps room for one uploaded cond of the priced text rows at any
share, and the entry's refined text rows count at their bound of
``CAPTIONS`` prompts of those rows. Each encode lets the DiT and its LoRA
pairs go, loads the encoder, encodes and releases it, and the next call
that reads the DiT loads it and the configured adapters again. An
encode with keyframe images loads the encoder with the entry's mmproj,
whose vision tower reads them. Each decode, and each encode of an
anchor's frames or sound, loads its VAE for the call and lets it go
after.

The step of the ComfyUI pack (``av_step``) takes the two streams as
ComfyUI's sampler carries them and gives back what ComfyUI's MiniMax-H3
forward gives: the audio stream is carried at ``sigma_v / sigma_a`` times
its latent while ``audio_scale`` is not 1, so the step divides that out
before the DiT and returns the velocity of the carried variable. A step
with anchors names each one's frame and the hashes of its latents and of
the noise of its video rows, the draw of ComfyUI's generator at the
payload's seed, and the vision spans of its prompt.

An entry's ``defaults.int8`` (``entry_int8``) sets the DiT's two int8
properties (``registry.int8_parts``). The linears are a property of the
loaded weights: int8 linears convert the block linears at each load, the
first and the one after each encode, and the entry's plan counts the int8
DiT in its ``dit_bytes``. The attention is a setting of the loaded DiT.
A job or a step may name an int8 value whose linears match the entry's
(``registry.int8_request_refusal``), so it changes the attention alone
and loads nothing: the worker sets it before each denoise and each step
(``set_int8``), after the load that follows an encode, and prices the
step at it.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .subservice import SingleWorker

_log = logging.getLogger(__name__)

WORKER = SingleWorker("video-worker")

_F32 = np.float32
# The text rows an entry's encode phase is priced at.
PRICED_TEXT_ROWS = 1024
# Refined text rows kept per entry, by (cond key, patch table): the
# bytes of this many prompts of the priced rows.
CAPTIONS = 8
# ComfyUI's floor on the video sigma of a step (MiniMax-H3 forward).
SIGMA_FLOOR = 1e-6


@dataclass(frozen=True)
class VideoEntryPlan:
    footprint: int                  # the larger phase, in bytes
    family: str
    variant: str
    size: tuple                     # default (width, height)
    frames: int                     # default clip length
    steps: int
    shifts: dict
    dit_bytes: int
    encoder_bytes: int
    vae_bytes: int
    audio_vae_bytes: int
    loras: tuple = ()               # each adapter's file bytes
    lora_dtypes: tuple = ()
    cache_share: float = 1.0        # of each cache's configured bound
    lora_term: int = 0              # what the plan books for the LoRAs
    tensor_floor: int = 0           # step tensor cache bytes kept for a cond
    mmproj_bytes: int = 0           # the encoder's mmproj file, for keyframes
    tower_bytes: int = 0            # its vision tower once attached
    text_rows: int = PRICED_TEXT_ROWS  # the rows its encode is priced at
    int8: str = "off"               # the entry's defaults.int8

    @property
    def attention(self) -> str:
        """The DiT's attention at the entry's int8 setting."""
        from gmlx.video.registry import int8_parts

        return int8_parts(self.int8)[0]

    @property
    def linears(self) -> str:
        """The block linears the entry's DiT loads with."""
        from gmlx.video.registry import int8_parts

        return int8_parts(self.int8)[1]


@dataclass(eq=False)
class ServedVideo:
    """What a video entry holds: the pipeline, the resolved entry, the
    plan it was priced by, its adapter rows, its refined text rows and
    the encoder's tokenizer once a prompt is priced."""
    pipe: Any
    spec: Any
    plan: VideoEntryPlan
    adapters: list = field(default_factory=list)
    captions: OrderedDict = field(default_factory=OrderedDict)
    tokenizer: Any = None
    # Guards captions: the video worker changes it, and a step's plan on
    # the event loop sums it.
    captions_lock: Any = field(default_factory=threading.Lock, repr=False)

    @property
    def lora_own_term(self) -> int:
        """The bytes the footprint books for the entry's own LoRA: its file
        once alone, ``lora_k`` times the files otherwise. Its pairs and
        default table past it are booked at the turn of the job that
        builds them."""
        return int(self.plan.lora_term)


def _log_line(*parts) -> None:
    _log.info(" ".join(str(p) for p in parts))


def clip_defaults(spec, fam, variant: str) -> dict:
    """The default clip, steps and shifts of an entry, as the video
    command takes them from its defaults and the variant's row."""
    from gmlx.commands.video import DEFAULT_SECONDS, DEFAULT_SIZE, frames_of

    d = spec.defaults or {}
    row = fam.defaults(variant)
    if d.get("size"):
        w, _, h = str(d["size"]).partition("x")
        size = (int(w), int(h))
    else:
        size = DEFAULT_SIZE
    shifts = {k: float(v) for k, v in row.shifts.items()}
    if "shift" in d:
        shifts["video"] = float(d["shift"])
    return {"size": size, "frames": frames_of(d.get("seconds",
                                                    DEFAULT_SECONDS)),
            "steps": int(d.get("steps") or row.steps), "shifts": shifts}


def plan_clip(plan: VideoEntryPlan, *, frames: int, width: int, height: int,
              text: int, budget=None, decode: bool = True,
              audio: int | None = None, live_lora: bool = False,
              blocks=(), anchors=(), int8: str | None = None):
    """The memory plan of one clip on the entry's files. ``audio`` is the
    audio stream's latents when they are not the clip's, and
    ``live_lora`` prices the step with a LoRA the server was sent.
    ``blocks`` are the vision tokens of the prompt's keyframes, which
    ``text`` counts, and ``anchors`` the placed ``layout.AnchorRows``.
    ``int8`` is the request's int8 setting, else the entry's; its
    linears are the entry's (``registry.int8_request_refusal``)."""
    from gmlx.video import memfit
    from gmlx.video.registry import get_family, int8_parts

    fam = get_family(plan.family)
    attention = int8_parts(int8 or plan.int8)[0]
    rows, n_audio = memfit.clip_rows(fam, frames=frames, width=width,
                                     height=height, text=text,
                                     anchors=anchors)
    return memfit.plan_memory(
        plan.family, dit=plan.dit_bytes, encoder=plan.encoder_bytes,
        vae=plan.vae_bytes, audio_vae=plan.audio_vae_bytes,
        loras=plan.loras, lora_dtypes=plan.lora_dtypes, rows=rows,
        text=text, audio_latents=n_audio if audio is None else int(audio),
        pixels=width * height, decode=decode, variant=plan.variant,
        live_lora=live_lora, mmproj=plan.mmproj_bytes,
        tower=plan.tower_bytes, blocks=blocks,
        anchor_frames=memfit.anchor_frames(fam, anchors),
        anchor_audio=[a.audio_t for a in anchors if a.audio_t],
        attention=attention, linears=plan.linears, budget=budget)


def parse_int8(value) -> str | None:
    """A request's ``int8`` field: None when it is not sent, else one of
    ``registry.INT8``. Raises ValueError for any other value."""
    from gmlx.video.registry import int8_parts

    if value is not None:
        int8_parts(value)
    return value


def request_int8(plan: VideoEntryPlan, fam, sent: str | None) -> str:
    """The int8 setting a job or a step runs: ``sent``, else the entry's.
    Raises ValueError for a value whose linears differ from the entry's,
    or that this install, Mac or variant does not run."""
    from gmlx.video.registry import int8_refusal, int8_request_refusal

    if sent is None:
        return plan.int8
    why = (int8_request_refusal(plan.int8, sent)
           or int8_refusal(fam, plan.variant, sent))
    if why:
        raise ValueError(why)
    return sent


def set_int8(served: ServedVideo, int8: str | None) -> None:
    """Set the loaded DiT's attention for one job or step from ``int8``,
    else the entry's setting. Each job and step sets it, so none runs at
    the attention the last one left."""
    from gmlx.video.registry import int8_parts

    served.pipe.dit.attention = int8_parts(int8 or served.plan.int8)[0]


def entry_int8(spec) -> str:
    """An entry's int8 setting: its ``defaults.int8``, else off."""
    return (spec.defaults or {}).get("int8") or "off"


def caption_bytes(hidden: int, rows: int = PRICED_TEXT_ROWS) -> int:
    """The bytes of refined text rows an entry keeps: ``CAPTIONS``
    prompts of the entry's priced ``rows`` at the DiT's width ``hidden``
    in bfloat16."""
    return CAPTIONS * int(rows) * int(hidden) * 2


def priced_rows(spec, fam) -> int:
    """The text rows an entry's encode is priced at: PRICED_TEXT_ROWS, or
    for an entry with an upsampler the larger of that and a caption of
    the upsampler's max_tokens. The encoder's tokenizer counts a caption
    about as the chat model's does, so each token is a row, beside the
    most rows of the family's template."""
    from gmlx.image.text_encoder import rows_at_most

    if fam.upsample is None or getattr(spec, "upsampler", None) is None:
        return PRICED_TEXT_ROWS
    caption = fam.upsample.max_tokens + rows_at_most(fam.encoder(), "")
    return max(PRICED_TEXT_ROWS, caption)


def held_bytes(served: ServedVideo) -> tuple:
    """(adapters, caches) a video entry holds beside its DiT: its LoRA
    pairs and patch tables, and the caches the image routes count with
    its refined text rows."""
    from .image_service import held_bytes as image_held

    adapters, caches = image_held(served.pipe)
    with served.captions_lock:
        rows = sum(int(a.nbytes) for a in served.captions.values())
    return adapters, caches + rows


def resident_bytes(phases: dict, terms: dict) -> int:
    """What the entry holds while the DiT is loaded: the DiT, its LoRAs
    and the largest of the step and the two decodes."""
    dit = terms["denoise"]["dit"] + terms["denoise"]["lora"]
    work = [terms["denoise"]["step"]]
    for ph in ("audio", "decode"):
        if ph in phases:
            work.append(phases[ph])
    return int(dit + max(work))


def entry_plan(spec) -> VideoEntryPlan:
    """The footprint of a video entry from the headers of its files.
    Raises ``VideoModelError`` for a file that is no video DiT or an
    adapter the family refuses, and ``VideoMemoryError`` for an entry
    whose default clip does not fit the working budget with its caches
    empty but for one cond of the priced rows and its refined rows."""
    from gmlx.video import memfit, pipeline
    from gmlx.video.fingerprint import VideoModelError
    from gmlx.video.loader import identify_dit
    from gmlx.video.registry import int8_fallback, int8_parts, run_refusal

    from .capacity import overcommit, working_budget_bytes
    from .image_service import _CACHE_BYTES, cache_bytes, fit_floored

    try:
        ident = identify_dit(spec.path, variant=spec.variant)
    except VideoModelError as e:
        raise VideoModelError(f"video model {spec.id!r}: {e}") from None
    fam = ident.family
    int8 = entry_int8(spec)
    linears = int8_parts(int8)[1]
    why = run_refusal(fam, ident.variant, int8)
    if why:
        raise VideoModelError(f"video model {spec.id!r}: {why}")
    for path, _st in spec.adapters:
        pipeline.vet_adapter(fam, dict(ident.config), path)
    try:
        d = clip_defaults(spec, fam, ident.variant)
    except ValueError as e:
        raise VideoModelError(f"video model {spec.id!r}: {e}") from None
    loras = tuple(memfit.file_bytes(p) for p, _ in spec.adapters)
    mmproj = getattr(spec, "encoder_mmproj", None)
    base = VideoEntryPlan(
        footprint=0, family=fam.name, variant=ident.variant, size=d["size"],
        frames=d["frames"], steps=d["steps"], shifts=d["shifts"],
        dit_bytes=memfit.file_bytes(spec.path) + (
            memfit.int8_delta(fam, ident.view) if linears == "int8" else 0),
        encoder_bytes=memfit.file_bytes(spec.encoder),
        vae_bytes=memfit.file_bytes(spec.vae),
        audio_vae_bytes=memfit.file_bytes(spec.audio_vae), loras=loras,
        lora_dtypes=tuple(memfit.lora_dtype(p) for p, _ in spec.adapters),
        mmproj_bytes=memfit.file_bytes(mmproj) if mmproj else 0,
        tower_bytes=memfit.tower_bytes(mmproj) if mmproj else 0,
        int8=int8)
    width, height = d["size"]
    budget = working_budget_bytes()
    rows = priced_rows(spec, fam)
    clip = plan_clip(base, frames=d["frames"], width=width, height=height,
                     text=rows, budget=budget)
    tensor, _cap = cache_bytes()
    steps = _CACHE_BYTES["steps"]
    caches = 2 * tensor + steps
    # The step routes' tensor cache keeps room for one cond of the priced
    # rows, the f32 encoder features a client uploads, at any share. The
    # cond of an upsampler's caption is the server's own, so its room
    # stands past the configured tensor_cache_mb.
    cond = rows * int(ident.config["text_dim"]) * 4
    floor = (cond if rows > PRICED_TEXT_ROWS else min(cond, tensor)
             ) if steps else 0
    # The refined text rows the entry keeps, at any share.
    captions = caption_bytes(ident.config["hidden_size"], rows)

    def price(share: float, kept: int = floor) -> int:
        # The step routes' tensor cache and the refined rows stay through
        # the encode, the tensor cache with room for ``kept`` bytes at any
        # share. A request's LoRA bytes are booked at its job's turns.
        held = (int(caches * share) + max(0, kept - int(tensor * share))
                + captions)
        return max(clip.phases["encode"] + held,
                   resident_bytes(clip.phases, clip.terms) + held)

    # The share the clip leaves the caches, as an image entry's, then the
    # largest one under it that leaves the tensor cache its cond too.
    got = fit_floored(price, budget)
    if got is None and overcommit():
        got = price(1.0), 1.0
    if got is None:
        # An entry is refused only against a budget.
        assert budget is not None
        raise memfit.VideoMemoryError(
            f"video model {spec.id!r} needs {memfit.gib(price(0.0), True)} "
            f"GiB for one {width}x{height} clip of {d['frames']} frames, over "
            f"the working budget of {memfit.gib(budget, False)} GiB. Set a "
            "smaller defaults.size or defaults.seconds"
            + (f", or defaults.int8 {int8_fallback(fam, ident.variant)}"
               if linears == "int8" else "")
            + ". GMLX_OVERCOMMIT=1 overrides.")
    footprint, share = got
    return VideoEntryPlan(**{**base.__dict__, "footprint": int(footprint),
                             "cache_share": share,
                             "lora_term": int(clip.terms["denoise"]["lora"]),
                             "tensor_floor": floor, "text_rows": rows})


def entry_footprint(spec) -> int:
    return entry_plan(spec).footprint


def _load(spec) -> ServedVideo:
    from gmlx.video import pipeline

    from .image_service import bound_caches, set_share

    plan = entry_plan(spec)
    pipe = pipeline.load(spec.path, variant=spec.variant,
                         attention=plan.attention, linears=plan.linears,
                         log=_log_line)
    served = ServedVideo(pipe=pipe, spec=spec, plan=plan)
    try:
        for path, _st in spec.adapters:
            pipeline._vet(pipe, path)
        _load_adapters(served)
        # The pair cache holds the entry's own pairs in the room the plan
        # books for them, beside the share of a request's.
        bound_caches(pipe, plan.lora_term, plan.cache_share)
        set_share(served, plan.cache_share, plan.tensor_floor)
    except BaseException:
        del pipe
        _release([served])
        raise
    return served


def _load_adapters(served: ServedVideo) -> None:
    """Read the entry's configured adapters into its pipeline, as their
    default patch table."""
    from gmlx.image import api, lora

    from .image_service import _pair_keys

    pipe, spec = served.pipe, served.spec
    rows = []
    if spec.adapters:
        lora.fit_cache(pipe, [path for path, _ in spec.adapters])
        for path, strength in spec.adapters:
            ad = lora.load_adapter(pipe, path, hold=rows)
            for w in ad.warnings:
                _log.warning("video model %s: %s", spec.id, w)
            rows.append((ad, float(strength)))
        pipe.default_table = api.define_patch_table(
            pipe, lora.table_entries(rows))
        # A step's table never sheds the entry's own pairs.
        pipe.adapters.keep(_pair_keys(rows))
    served.adapters = rows


def reserved_worker():
    """Hold the video worker for the calling thread, as
    :meth:`SingleWorker.reserved` does."""
    return WORKER.reserved()


def worker_reserved() -> bool:
    """True when the calling thread holds the video worker or is it."""
    return WORKER.held()


def _on_worker(fn, *args):
    """``fn`` on the video worker: inline on it, or in the caller's
    hold."""
    return WORKER.run(fn, *args)


def load_on_worker(spec) -> ServedVideo:
    """Load a video entry on the video worker and wait for it."""
    return _on_worker(_load, spec)


def _release(box: list) -> None:
    from .image_service import drop_share

    served = box.pop() if box else None
    if served is None:
        return
    drop_share(served)
    pipe, served.pipe = served.pipe, None
    with served.captions_lock:
        served.captions.clear()
    if pipe is not None:
        from gmlx.image.lora import install

        install.drop_built(pipe)
        for cache in (pipe.conditionings, pipe.captions, pipe.modulations,
                      pipe.adapters, pipe.tables, pipe.reference_latents):
            cache.clear()
    del pipe, served
    import gc

    gc.collect()
    import mlx.core as mx

    mx.clear_cache()


def release_on_worker(box: list) -> None:
    """Drop the entry in ``box``, a one-item list, on the video worker,
    after the jobs queued before it, and return its memory."""
    _on_worker(_release, box)


# The DiT while the encoder runs.

def drop_dit(served: ServedVideo) -> None:
    """Let the DiT and every LoRA pair go before the encoder loads, as the
    video command encodes with neither held. The patch tables of the
    pairs go with them: the configured adapters load again with the DiT,
    and the step routes answer a table they were sent as missing, which
    the pack sends again."""
    import gc

    import mlx.core as mx

    from gmlx.image import api
    from gmlx.image.lora import install

    pipe = served.pipe
    install.drop_built(pipe)
    pipe.dit = None
    pipe.adapters.keep(())
    pipe.adapters.clear()
    pipe.tables.clear()
    for tid in [k for k, v in pipe.patch_tables.items() if v["entries"]]:
        del pipe.patch_tables[tid]
    pipe.default_table = api.EMPTY_TABLE
    served.adapters = []
    gc.collect()
    mx.clear_cache()


def ensure_dit(served: ServedVideo) -> None:
    """Load the DiT again after an encode, its leaves wrapped for the
    patch tables, and the configured adapters with it. Every reader of
    the DiT calls it: the step and a table's define. An adapter that
    does not load again, as from a file moved since, lets the DiT go
    again before it raises, so the next call loads both."""
    pipe = served.pipe
    if pipe.dit is not None:
        return
    from gmlx.image.lora import install
    from gmlx.video.loader import load_dit

    pipe.dit, _ = load_dit(str(served.spec.path), variant=served.spec.variant,
                           attention=served.plan.attention,
                           linears=served.plan.linears, log=_log_line)
    try:
        install.rewrap(pipe)
        _load_adapters(served)
    except BaseException:
        drop_dit(served)
        raise


def prompt_rows(served: ServedVideo, text: str, sizes=()) -> int:
    """The text rows ``text`` encodes to, with keyframes of ``sizes``
    (width, height), by the encoder's tokenizer from its header, which the
    entry builds once."""
    from gmlx.video import memfit
    from gmlx.video.encoder import header_tokenizer

    fam = served.pipe.video_family
    if served.tokenizer is None:
        served.tokenizer = header_tokenizer(
            fam, os.path.expanduser(served.spec.encoder))
    return memfit.prompt_rows(fam, text, served.tokenizer, sizes)


def anchor_refusal(served: ServedVideo, n: int) -> str | None:
    """Why the entry's DiT runs no request with ``n`` anchors or keyframe
    images, or None: FastH3 runs text to video only."""
    fam = served.pipe.video_family
    row = fam.variant_row(served.plan.variant)
    if not n or row is None or not row.vsa_sparsity:
        return None
    from gmlx.video.families.h3 import vsa

    return vsa.anchor_refusal(n)


def keyframe_refusal(served: ServedVideo, images: list) -> str | None:
    """Why the entry encodes no prompt with keyframe ``images``, each
    (H, W, 3) uint8, or None. A keyframe is the clip's canvas, as
    ComfyUI's node resizes it."""
    if not images:
        return None
    fam = served.pipe.video_family
    if fam.keyframe_encode is None:
        return (f"the gmlx server encodes text alone for {fam.title}; it "
                "takes no images with the prompt")
    why = anchor_refusal(served, len(images))
    if why:
        return why
    if not getattr(served.spec, "encoder_mmproj", None):
        return (f"video model {served.spec.id!r} names no encoder_mmproj, "
                "the vision tower of its text encoder that reads keyframes; "
                "name the encoder's mmproj GGUF as encoder_mmproj in its "
                "entry")
    for i, im in enumerate(images):
        h, w = (int(d) for d in im.shape[:2])
        if not on_canvas(fam, w, h):
            return (f"keyframe image {i} is {w}x{h}, no {fam.title} canvas: "
                    "a keyframe is the clip's canvas, in multiples of 32")
    return None


def on_canvas(fam, width: int, height: int) -> bool:
    """Whether ``width`` x ``height`` pixels are a canvas of the family's
    clips: multiples of its VAE stride times its patch."""
    from .image_steps import COMFY

    lat = COMFY[fam.name]["latent"]
    m = int(lat["stride"]) * int(lat["patch"])
    return width > 0 and height > 0 and not width % m and not height % m


def encode(served: ServedVideo, text: str, images: list) -> tuple:
    """(features, spans): the encoder features (1, L, text dim) f32 of
    ``text``, with keyframe ``images`` (H, W, 3) uint8 at the clip's
    canvas, and their vision spans in the features, on the video worker,
    with the encoder loaded alone, and with the entry's mmproj for
    images. The DiT stays out after it: the next call that reads it
    loads it, so encodes back to back load it once. ``keyframe_refusal``
    comes first."""
    import mlx.core as mx

    from gmlx.video import encoder as video_encoder

    from .image_steps import StepError

    fam = served.pipe.video_family
    drop_dit(served)
    path = os.path.expanduser(served.spec.encoder)
    if images:
        handle = video_encoder.load_encoder(
            fam, path, mmproj=os.path.expanduser(served.spec.encoder_mmproj),
            log=_log_line)
    else:
        handle = video_encoder.load_encoder(fam, path, log=_log_line)
    try:
        spans = ()
        if images:
            try:
                feats, spans = fam.keyframe_encode(handle, text, images)
            except ValueError as e:
                raise StepError(400, str(e)) from None
        else:
            feats = mx.array(handle.encode(text).features)[None]
        feats = feats.astype(mx.float32)
        mx.eval(feats)
        out = np.array(feats, copy=True)
    finally:
        handle.release()
        del handle
        mx.clear_cache()
    return out, tuple(spans)


# One step of the pack.

def stream_sigmas(timestep: float, shifts: dict) -> tuple:
    """The video and audio sigmas of a step at ComfyUI's ``timestep``
    (sigma x 1000), in float32 as ComfyUI's forward takes them."""
    from gmlx.video import schedule

    sv = max(_F32(timestep) / _F32(1000.0), _F32(SIGMA_FLOOR))
    sa = _F32(schedule.time_shift(float(sv), float(shifts["video"]),
                                  float(shifts["audio"])))
    return float(sv), float(sa)


def caption(served: ServedVideo, key: str, feats, table: str):
    """The refined text rows of encoder features ``feats`` under patch
    table ``table``, kept by (key, table) while the entry holds them and
    within ``caption_bytes``, the oldest dropped first. Rows over that
    bound alone are not kept."""
    import mlx.core as mx

    with served.captions_lock:
        held = served.captions.get((key, table)) if key else None
        if held is not None:
            served.captions.move_to_end((key, table))
            return held, True
    text = served.pipe.dit.caption(feats)
    mx.eval(text)
    bound = caption_bytes(served.pipe.dit.cfg.dim, served.plan.text_rows)
    if key and int(text.nbytes) <= bound:
        with served.captions_lock:
            served.captions[(key, table)] = text
            held = sum(int(a.nbytes) for a in served.captions.values())
            while held > bound:
                _k, old = served.captions.popitem(last=False)
                held -= int(old.nbytes)
    return text, False


# The anchors of a step.

@dataclass(frozen=True)
class AnchorRef:
    """An anchor an AV step names: its frame, counted from the end when
    negative, and the tensor hashes of its video latent (1, C, T, h, w),
    the noise of its video rows and its audio latent (1, C, 2, T)."""
    frame: int
    video: str | None = None
    noise: str | None = None
    audio: str | None = None

    def hashes(self) -> list:
        return [h for h in (self.video, self.noise, self.audio) if h]


def _int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def step_anchors(fields: dict) -> tuple:
    """(anchors, spans): the ``AnchorRef`` of each of an AV step's
    ``anchors`` and its ``spans``, the [start, stop) vision rows of its
    cond. A video latent comes with the noise of its rows. Raises
    ValueError naming the field that is wrong."""
    raw = fields.get("anchors")
    raw = [] if raw is None else raw
    if not isinstance(raw, list):
        raise ValueError("an AV step's anchors are a list")
    refs = []
    for i, a in enumerate(raw):
        if not isinstance(a, dict) or not _int(a.get("frame")):
            raise ValueError(f"anchors[{i}] is an object with an integer "
                             "frame")
        got = {}
        for k in ("video", "noise", "audio"):
            if a.get(k) is not None and not isinstance(a[k], str):
                raise ValueError(f"anchors[{i}].{k} is a tensor hash")
            got[k] = a.get(k)
        if got["video"] is None and got["audio"] is None:
            raise ValueError(f"anchors[{i}] names a video or an audio latent")
        if (got["video"] is None) != (got["noise"] is None):
            raise ValueError(f"anchors[{i}] names the noise of its video rows "
                             "with its video latent, and with it alone")
        refs.append(AnchorRef(int(a["frame"]), **got))
    spans = fields.get("spans")
    spans = [] if spans is None else spans
    if not isinstance(spans, list) or not all(
            isinstance(sp, list) and len(sp) == 2 and all(map(_int, sp))
            for sp in spans):
        raise ValueError("an AV step's spans are [start, stop] pairs of "
                         "integers")
    return tuple(refs), tuple((int(a), int(b)) for a, b in spans)


def anchor_rows(served: ServedVideo, refs, held: dict, frames: int,
                vshape, ashape) -> tuple:
    """The ``layout.AnchorRows`` of a step's anchors ``refs``, placed by
    ComfyUI's guide rule on its clip of ``frames`` frames and latents
    ``vshape`` and ``ashape``, from the shapes of their tensors ``held``
    by hash. Raises ValueError naming an anchor that is no part of the
    clip."""
    from gmlx.video import anchors
    from gmlx.video.layout import AnchorRows

    fam = served.pipe.video_family
    vshape, ashape = tuple(vshape), tuple(ashape)
    out = []
    for i, r in enumerate(refs):
        video_t = audio_t = kept = 0
        if r.video is not None:
            got = tuple(held[r.video].shape)
            if len(got) != 5 or got[:2] != vshape[:2] or got[3:] != vshape[3:]:
                raise ValueError(
                    f"anchors[{i}]'s video latent is {list(got)}, not (1, "
                    f"{vshape[1]}, T, {vshape[3]}, {vshape[4]}) on the clip's "
                    "canvas")
            video_t = got[2]
            kept = 1 if video_t == 1 else fam.clip_frames(video_t)
            if kept is None:
                raise ValueError(f"anchors[{i}]'s video latent holds {video_t} "
                                 f"latent frames, no {fam.title} guide clip: "
                                 "1, or 5 k + 2")
        if r.audio is not None:
            got = tuple(held[r.audio].shape)
            if len(got) != 4 or got[:3] != ashape[:3] or got[3] < 1:
                raise ValueError(
                    f"anchors[{i}]'s audio latent is {list(got)}, not (1, "
                    f"{ashape[1]}, {ashape[2]}, T)")
            audio_t = got[3]
        try:
            p = anchors.place(r.frame, frames, ashape[3],
                              fam.timeline.frame_rescale, video_frames=kept,
                              audio=r.audio is not None)
        except ValueError as e:
            raise ValueError(f"anchors[{i}]: {e}") from None
        if p.audio_t is not None and audio_t > p.audio_t:
            raise ValueError(f"anchors[{i}]'s audio holds {audio_t} latent "
                             f"frames, past the {p.audio_t} the clip's audio "
                             f"holds from frame {p.frame}")
        out.append(AnchorRows(p.frame, video_t, audio_t))
    return tuple(out)


def anchor_cond(served: ServedVideo, refs, rows, held: dict) -> tuple:
    """(cond_video, cond_audio): the rows of a step's anchors ``refs``
    placed at ``rows``, from their tensors ``held`` by hash, mixed with
    the noise each video latent came with. On the video worker, with the
    DiT loaded. Raises ValueError for noise of another shape than its
    rows."""
    import mlx.core as mx

    from gmlx.video.anchors import Anchor

    def f32(h):
        return None if h is None else held[h].mx().astype(mx.float32)
    placed = tuple(Anchor(row.frame, f32(r.video), f32(r.audio))
                   for r, row in zip(refs, rows))
    noise = [f32(r.noise) for r in refs if r.noise is not None]
    return served.pipe.dit.cond_rows(placed, 0, noise=noise)


def av_step(served: ServedVideo, video, audio, timestep: float,
            shifts: dict, scale: float, feats, key: str, table: str | None,
            *, anchors=(), cond=((), ()), spans=(),
            int8: str | None = None, pacer=None):
    """One step of the pack's MiniMax-H3 model: the video and audio
    velocity f32 of the latents as ComfyUI's sampler carries them, at its
    ``timestep``, with the audio carried at ``scale`` (the payload's
    audio_scale). ``anchors`` are the placed ``layout.AnchorRows`` of the
    anchor rows ``cond`` (``anchor_cond``) and ``spans`` the vision spans
    of ``feats``. ``int8`` is the step's setting, else the entry's.
    ``pacer``, an ``api.Budget``, runs at every pause point: on the
    caption it evaluates at each point, and in the forward on the step's
    rows. Returns (video, audio, caption cache hit)."""
    import mlx.core as mx

    from gmlx.image import pacing
    from gmlx.image.lora import install
    from gmlx.video import api, layout

    pipe = served.pipe
    ensure_dit(served)
    set_int8(served, int8)
    fam = pipe.video_family
    sv, sa = stream_sigmas(timestep, shifts)
    xv = video.astype(mx.float32)
    xa = audio.astype(mx.float32)
    carry = None
    if scale != 1.0:
        carry = float(_F32(sa) / _F32(sv))
        xa = xa * carry
    tid = pipe.default_table if table is None else table
    if tid not in pipe.patch_tables:
        raise ValueError(f"unknown patch table {table!r}")
    lay = layout.pack(fam.timeline, int(feats.shape[1]), xv.shape[2],
                      xv.shape[3], xv.shape[4], xa.shape[3], tuple(anchors),
                      tuple(spans))
    prev = install.activate(pipe, tid)
    try:
        with pacing.paced(pacer):
            text, hit = caption(served, key, feats, tid)
        with pacing.paced(api.forward_pacer(pacer, lay)):
            vv, va = api.denoise_step(pipe.dit, xv, xa,
                                      {"video": sv, "audio": sa}, text, lay,
                                      cond_video=cond[0], cond_audio=cond[1])
    finally:
        install.restore(pipe, prev)
    vv = vv.astype(mx.float32)
    va = va.astype(mx.float32)
    if carry is not None:
        # d/d(sigma_v) of the carried variable, each factor in float32.
        keep = float(_F32(1.0 - scale))
        rate = float(_F32(1) + _F32(scale - 1.0) * _F32(sa))
        va = keep * xa + rate * va
    return vv, va, hit


# The decodes, and the encodes of anchors.

def decode_video(served: ServedVideo, latents) -> np.ndarray:
    """Frames (F, H, W, 3) f16 in [0, 1] of model-space latents
    (1, 24, T, h, w), on the video worker, with the video VAE loaded for
    the call."""
    import mlx.core as mx

    from gmlx.video.vae.reader import load_vae

    vae = load_vae(os.path.expanduser(served.spec.vae), family="h3",
                   dtype=mx.float16)
    try:
        out = []
        for chunk in vae.decode_frames(latents.astype(mx.float32)):
            out.append(np.array(chunk[0].astype(mx.float16), copy=True))
        return np.concatenate(out, axis=0)
    finally:
        del vae
        mx.clear_cache()


def encoder_frames(frames):
    """Frames (F, H, W, 3) in [0, 1], float or uint8 levels, as the video
    VAE takes them: float32 in [-1, 1], (1, F, H, W, 3). Level k is k / 255
    as torch's ``u8.float() / 255`` gives it, through a table of the same
    float32 operations, one array of the clip."""
    import mlx.core as mx

    if frames.dtype != mx.uint8:
        return frames.astype(mx.float32)[None] * 2.0 - 1.0
    k = np.arange(256, dtype=np.float32) / np.float32(255)
    return mx.take(mx.array(k * np.float32(2) - np.float32(1)), frames)[None]


def encode_video(served: ServedVideo, frames, *, open: bool = False
                 ) -> np.ndarray:
    """Model-space latents (1, 24, T, h, w) f32 of frames (F, H, W, 3) in
    [0, 1], float or uint8 levels, 1 or 17 k + 5 of them, on the video
    worker, with the video VAE loaded for the call. An ``open`` window is
    17 m frames and keeps the latents a clip drops at its end."""
    import mlx.core as mx

    from gmlx.video.vae.reader import load_vae

    vae = load_vae(os.path.expanduser(served.spec.vae), family="h3",
                   dtype=mx.float16)
    try:
        x = encoder_frames(frames)
        z = vae.encode(x, drop=not open).astype(mx.float32)
        mx.eval(z)
        return np.array(z, copy=True)
    finally:
        del vae
        mx.clear_cache()


def encode_audio(served: ServedVideo, waveform) -> np.ndarray:
    """Model-space audio latents (1, 32, 2, T) f32 of a stereo waveform
    (1, 2, samples) at the audio VAE's rate, padded to a whole latent, on
    the video worker, with the audio VAE loaded for the call. Its
    convolutions run in windows, so a long sound encodes as a whole one
    does."""
    import mlx.core as mx

    from gmlx.video.vae.reader import load_vae

    avae = load_vae(os.path.expanduser(served.spec.audio_vae),
                    family="h3_audio", dtype=mx.float32)
    try:
        z = avae.encode(waveform.astype(mx.float32)).astype(mx.float32)
        mx.eval(z)
        return np.array(z, copy=True)
    finally:
        del avae
        mx.clear_cache()


def decode_audio(served: ServedVideo, latents) -> np.ndarray:
    """The waveform (1, 2, samples) f32 of model-space audio latents
    (1, 32, 2, T), on the video worker, with the audio VAE loaded for the
    call."""
    import mlx.core as mx

    from gmlx.video.vae.reader import load_vae

    avae = load_vae(os.path.expanduser(served.spec.audio_vae),
                    family="h3_audio", dtype=mx.float32)
    try:
        wave = avae.decode(latents.astype(mx.float32)).astype(mx.float32)
        mx.eval(wave)
        return np.array(wave, copy=True)
    finally:
        del avae
        mx.clear_cache()
