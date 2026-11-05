"""Video models in the serve residency pool.

A video entry holds a ``ServedVideo``: the pipeline over the video DiT,
with the configured adapters as its default patch table, the plan it was
priced by and a small cache of refined text rows. It runs on the image
worker, as an image entry does, since MLX streams are per thread.

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
that reads the DiT loads it and the configured adapters again. Each
decode loads its VAE for the call and lets it go after.

The step of the ComfyUI pack (``av_step``) takes the two streams as
ComfyUI's sampler carries them and gives back what ComfyUI's MiniMax-H3
forward gives: the audio stream is carried at ``sigma_v / sigma_a`` times
its latent while ``audio_scale`` is not 1, so the step divides that out
before the DiT and returns the velocity of the carried variable.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)

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
              blocks=(), anchors=()):
    """The memory plan of one clip on the entry's files. ``audio`` is the
    audio stream's latents when they are not the clip's, and
    ``live_lora`` prices the step with a LoRA the server was sent.
    ``blocks`` are the vision tokens of the prompt's keyframes, which
    ``text`` counts, and ``anchors`` the placed ``layout.AnchorRows``."""
    from gmlx.video import memfit
    from gmlx.video.registry import get_family

    fam = get_family(plan.family)
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
        budget=budget)


def caption_bytes(hidden: int) -> int:
    """The bytes of refined text rows an entry keeps: ``CAPTIONS``
    prompts of the priced rows at the DiT's width ``hidden`` in
    bfloat16."""
    return CAPTIONS * PRICED_TEXT_ROWS * int(hidden) * 2


def held_bytes(served: ServedVideo) -> tuple:
    """(adapters, caches) a video entry holds beside its DiT: its LoRA
    pairs and patch tables, and the caches the image routes count with
    its refined text rows."""
    from .image_service import held_bytes as image_held

    adapters, caches = image_held(served.pipe)
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
    from gmlx.video.registry import run_refusal

    from .capacity import overcommit, working_budget_bytes
    from .image_service import _CACHE_BYTES, cache_bytes, fit_floored

    try:
        ident = identify_dit(spec.path, variant=spec.variant)
    except VideoModelError as e:
        raise VideoModelError(f"video model {spec.id!r}: {e}") from None
    fam = ident.family
    why = run_refusal(fam, ident.variant)
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
        dit_bytes=memfit.file_bytes(spec.path),
        encoder_bytes=memfit.file_bytes(spec.encoder),
        vae_bytes=memfit.file_bytes(spec.vae),
        audio_vae_bytes=memfit.file_bytes(spec.audio_vae), loras=loras,
        lora_dtypes=tuple(memfit.lora_dtype(p) for p, _ in spec.adapters),
        mmproj_bytes=memfit.file_bytes(mmproj) if mmproj else 0,
        tower_bytes=memfit.tower_bytes(mmproj) if mmproj else 0)
    width, height = d["size"]
    budget = working_budget_bytes()
    clip = plan_clip(base, frames=d["frames"], width=width, height=height,
                     text=PRICED_TEXT_ROWS, budget=budget)
    tensor, _cap = cache_bytes()
    steps = _CACHE_BYTES["steps"]
    caches = 2 * tensor + steps
    # The step routes' tensor cache keeps room for one cond of the priced
    # rows, the f32 encoder features a client uploads, at any share.
    cond = PRICED_TEXT_ROWS * int(ident.config["text_dim"]) * 4
    floor = min(cond, tensor) if steps else 0
    # The refined text rows the entry keeps, at any share.
    captions = caption_bytes(ident.config["hidden_size"])

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
            "smaller defaults.size or defaults.seconds. GMLX_OVERCOMMIT=1 "
            "overrides.")
    footprint, share = got
    return VideoEntryPlan(**{**base.__dict__, "footprint": int(footprint),
                             "cache_share": share,
                             "lora_term": int(clip.terms["denoise"]["lora"]),
                             "tensor_floor": floor})


def entry_footprint(spec) -> int:
    return entry_plan(spec).footprint


def _load(spec) -> ServedVideo:
    from gmlx.video import pipeline

    from .image_service import _SHARES, _scale_step_caches, bound_caches

    plan = entry_plan(spec)
    pipe = pipeline.load(spec.path, variant=spec.variant, log=_log_line)
    served = ServedVideo(pipe=pipe, spec=spec, plan=plan)
    try:
        for path, _st in spec.adapters:
            pipeline._vet(pipe, path)
        _load_adapters(served)
        # The pair cache holds the entry's own pairs in the room the plan
        # books for them, beside the share of a request's.
        bound_caches(pipe, plan.lora_term, plan.cache_share)
        _SHARES[id(served)] = (plan.cache_share, plan.tensor_floor)
        _scale_step_caches()
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


def load_on_worker(spec) -> ServedVideo:
    """Load a video entry on the image worker and wait for it."""
    from .image_service import _on_worker

    return _on_worker(_load, spec)


def _release(box: list) -> None:
    from .image_service import _SHARES, _scale_step_caches

    served = box.pop() if box else None
    if served is None:
        return
    if _SHARES.pop(id(served), None) is not None:
        _scale_step_caches()
    pipe, served.pipe = served.pipe, None
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
    """Drop the entry in ``box``, a one-item list, on the image worker,
    after the jobs queued before it, and return its memory."""
    from .image_service import _on_worker

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
                           log=_log_line)
    try:
        install.rewrap(pipe)
        _load_adapters(served)
    except BaseException:
        drop_dit(served)
        raise


def prompt_rows(served: ServedVideo, text: str) -> int:
    """The text rows ``text`` encodes to, by the encoder's tokenizer from
    its header, which the entry builds once."""
    from gmlx.video import memfit
    from gmlx.video.encoder import header_tokenizer

    fam = served.pipe.video_family
    if served.tokenizer is None:
        served.tokenizer = header_tokenizer(
            fam, os.path.expanduser(served.spec.encoder))
    return memfit.prompt_rows(fam, text, served.tokenizer)


def encode(served: ServedVideo, text: str, images: list) -> np.ndarray:
    """The encoder features (1, L, text dim) f32 of ``text``, on the image
    worker, with the encoder loaded alone. Images are refused. The DiT
    stays out after it: the next call that reads it loads it, so encodes
    back to back load it once."""
    from gmlx.video import encoder as video_encoder

    from .image_steps import StepError

    fam = served.pipe.video_family
    if images:
        raise StepError(400, f"the gmlx server encodes text alone for "
                        f"{fam.title}; it takes no images with the prompt, "
                        "such as a first or last frame")
    import mlx.core as mx

    drop_dit(served)
    handle = video_encoder.load_encoder(
        fam, os.path.expanduser(served.spec.encoder), log=_log_line)
    try:
        feats = mx.array(handle.encode(text).features)[None]
        feats = feats.astype(mx.float32)
        mx.eval(feats)
        out = np.array(feats, copy=True)
    finally:
        handle.release()
        del handle
        mx.clear_cache()
    return out


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

    held = served.captions.get((key, table)) if key else None
    if held is not None:
        served.captions.move_to_end((key, table))
        return held, True
    text = served.pipe.dit.caption(feats)
    mx.eval(text)
    bound = caption_bytes(served.pipe.dit.cfg.dim)
    if key and int(text.nbytes) <= bound:
        served.captions[(key, table)] = text
        held = sum(int(a.nbytes) for a in served.captions.values())
        while held > bound:
            _k, old = served.captions.popitem(last=False)
            held -= int(old.nbytes)
    return text, False


def av_step(served: ServedVideo, video, audio, timestep: float,
            shifts: dict, scale: float, feats, key: str, table: str | None):
    """One step of the pack's MiniMax-H3 model: the video and audio
    velocity f32 of the latents as ComfyUI's sampler carries them, at its
    ``timestep``, with the audio carried at ``scale`` (the payload's
    audio_scale). Returns (video, audio, caption cache hit)."""
    import mlx.core as mx

    from gmlx.image.lora import install
    from gmlx.video import api, layout

    pipe = served.pipe
    ensure_dit(served)
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
                      xv.shape[3], xv.shape[4], xa.shape[3])
    prev = install.activate(pipe, tid)
    try:
        text, hit = caption(served, key, feats, tid)
        vv, va = api.denoise_step(pipe.dit, xv, xa,
                                  {"video": sv, "audio": sa}, text, lay)
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


# The decodes.

def decode_video(served: ServedVideo, latents) -> np.ndarray:
    """Frames (F, H, W, 3) f16 in [0, 1] of model-space latents
    (1, 24, T, h, w), on the image worker, with the video VAE loaded for
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


def decode_audio(served: ServedVideo, latents) -> np.ndarray:
    """The waveform (1, 2, samples) f32 of model-space audio latents
    (1, 32, 2, T), on the image worker, with the audio VAE loaded for the
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
