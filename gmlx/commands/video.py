"""``gmlx video``: video generation from GGUF diffusion models.

``generate`` writes a clip with sound from a prompt: the text encoder,
then the DiT under the recipe's LoRA, then the audio and video VAEs, one
model resident at a time, the frames streamed to ffmpeg as they decode.
``bench`` times the steps of a few shapes and takes their peak memory.
``pull`` fetches a recipe's files, or finds them in the model folders.

Exit codes follow ``gmlx image``: 0 on success, 1 for a refusal or a
runtime failure, 2 for a usage error and 130 for an interrupt.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import secrets
import shutil
import sys
import tempfile
import time

from .image import _adapter

_ACTIONS = ("generate", "bench", "pull")
_ACTION_DESC = {
    "generate": "write a clip with sound from a prompt",
    "bench": "time the steps and take the peak memory at a few shapes",
    "pull": "fetch a video recipe's files, or find them on disk",
}
DEFAULT_SIZE = (1344, 768)
DEFAULT_SECONDS = 5
# The sizes a clip refused for its size tries, in this order, each only
# when no wider and no taller than the clip asked for. A shorter clip at
# the last one, or at the clip's own size when the last is wider or
# taller, is tried when none fits.
HINT_SIZES = ((1344, 768), (768, 1344), (768, 768))
# The recipe that runs the base DiT with no LoRA.
BASE_RECIPE = "h3-base"
# Seeds run from 0 to 2^64 - 1, as a ComfyUI seed does.
SEED_RANGE = 1 << 64
# A guide's sound is read to a second past the clip's audio, more than the
# audio encode's halo (12 latents) past the latents an anchor keeps.
GUIDE_AUDIO_PAST_S = 1
_FRAME_AT = re.compile(r"-?\d+")


class _Usage(Exception):
    """A usage error the parser cannot see, which exits 2."""


def _print_help(prog: str) -> None:
    lines = [f"usage: {prog} <action> [options]", "",
             "video generation from GGUF diffusion models.", "", "actions:"]
    for a in _ACTIONS:
        lines.append(f"  {a:<10} {_ACTION_DESC[a]}")
    lines += ["", f"run `{prog} <action> --help` for an action's options."]
    print("\n".join(lines))


def _family():
    from gmlx.video.families.h3 import family as hf

    return hf


def _size(text: str) -> tuple[int, int]:
    from gmlx.video.families.h3.family import CANVAS_MULTIPLE

    w, sep, h = text.lower().partition("x")
    try:
        width, height = int(w), int(h)
    except ValueError:
        width = height = 0
    if not sep or width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError(f"expected WxH, got {text!r}")
    if width % CANVAS_MULTIPLE or height % CANVAS_MULTIPLE:
        raise argparse.ArgumentTypeError(
            f"{text}: width and height must be multiples of "
            f"{CANVAS_MULTIPLE}")
    return width, height


def _positive(text: str) -> int:
    try:
        n = int(text)
    except ValueError:
        n = 0
    if n <= 0:
        raise argparse.ArgumentTypeError(
            f"a positive integer is required, got {text!r}")
    return n


def _seed(text: str) -> int:
    try:
        n = int(text)
    except ValueError:
        n = -1
    if not 0 <= n < SEED_RANGE:
        raise argparse.ArgumentTypeError(
            f"an integer from 0 to {SEED_RANGE - 1} is required, got "
            f"{text!r}")
    return n


def _at(text: str) -> tuple[str, int]:
    """PATH[@FRAME] as (path, frame): only an integer after the last @ is
    a frame, and a path with none anchors at frame 0."""
    path, sep, frame = text.rpartition("@")
    if sep and path and _FRAME_AT.fullmatch(frame):
        return path, int(frame)
    return text, 0


def _guide(text: str) -> tuple[str, str, int]:
    return ("--guide", *_at(text))


def _guide_audio(text: str) -> tuple[str, str, int]:
    return ("--guide-audio", *_at(text))


def _shift(text: str) -> float:
    try:
        s = float(text)
    except ValueError:
        s = 0.0
    if not (s > 0 and math.isfinite(s)):
        raise argparse.ArgumentTypeError(
            f"a positive number is required, got {text!r}")
    return s


def _whole_frames(seconds: float) -> int:
    # A hair over a whole frame counts as that frame.
    return math.ceil(round(seconds * _family().FPS, 3))


def _clip_seconds(s: float) -> bool:
    hf = _family()
    return (math.isfinite(s * hf.FPS)
            and hf.MIN_FRAMES <= _whole_frames(s) <= hf.MAX_FRAMES)


def _seconds_range() -> str:
    hf = _family()
    return f"{hf.MIN_FRAMES / hf.FPS:g} to {hf.MAX_FRAMES / hf.FPS:g}"


def _seconds(text: str) -> float:
    try:
        s = float(text)
    except ValueError:
        s = math.nan
    if not _clip_seconds(s):
        raise argparse.ArgumentTypeError(
            f"--seconds takes {_seconds_range()}, got {text!r}")
    return s


def _schedules(steps: int, shifts: dict, rungs: tuple = ()) -> bool:
    """Whether every stream gets ``steps`` finite float32 forwards."""
    import numpy as np

    from gmlx.video import schedule

    try:
        with np.errstate(all="ignore"):
            sig = schedule.stream_sigmas(steps, shifts, rungs)
    except ValueError:
        return False
    return all(len(v) == steps + 1 and np.isfinite(v).all()
               for v in sig.values())


def frames_of(seconds: float) -> int:
    """The frames of a clip of ``seconds``: up to the next 17 k + 5."""
    return _family().align_frames(_whole_frames(seconds))


def _frames(text: str) -> int:
    hf = _family()
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"a frame count is required, got {text!r}") from None
    inside = hf.MIN_FRAMES <= n <= hf.MAX_FRAMES
    if inside and n == hf.align_frames(n):
        return n
    msg = (f"{n} frames is not a clip length: a clip has 17 k + 5 frames "
           f"from {hf.MIN_FRAMES} to {hf.MAX_FRAMES}")
    if inside:
        up = hf.align_frames(n)
        msg += f", such as {up - 17} or {up}"
    raise argparse.ArgumentTypeError(msg)


def _seconds_text(frames: int) -> str:
    """``frames`` in seconds, rounded down to 2 decimals, so that
    ``--seconds`` of it makes the same clip."""
    return f"{math.floor(frames / _family().FPS * 100) / 100:.2f}"


def _within(size: tuple, clip: tuple) -> bool:
    return size[0] <= clip[0] and size[1] <= clip[1]


def _size_hint(frames: int, width: int, height: int, plan_at,
               base: bool, lengths: bool = True) -> str | None:
    """What a clip of ``frames`` at ``width`` x ``height`` refused for its
    size runs instead, at no size wider or taller than that: the first of
    ``HINT_SIZES`` whose plan fits at its length, else, when ``lengths``,
    the longest clip that fits at the last one or at the clip's own size.
    A clip whose anchors sit at frames of its length is not ``lengths``.
    When none fits and the smallest clip is over the budget too: with no
    adapters it fits, so the adapters' memory and fewer of them, or
    ``BASE_RECIPE`` when ``base``; with none it is over too, so the DiT
    file's memory and a smaller quant. Else None. ``plan_at(frames,
    width, height, loras=None)`` is the run's memory plan at that clip,
    with ``loras`` in place of the run's adapters when given."""
    hf = _family()
    clip = (width, height)
    tried = [s for s in HINT_SIZES if _within(s, clip) and s != clip]
    for w, h in tried:
        if plan_at(frames, w, h).fits:
            return f"try --size {w}x{h}, which fits at this length"
    w, h = HINT_SIZES[-1] if _within(HINT_SIZES[-1], clip) else clip
    f = hf.align_frames(frames) - 17
    while lengths and f >= hf.MIN_FRAMES:
        if plan_at(f, w, h).fits:
            names = [f"{a}x{b}" for a, b in tried]
            said = ""
            if names:
                listed = (", ".join(names[:-1]) + " or " + names[-1]
                          if len(names) > 1 else names[0])
                said = f"no {listed} clip fits at this length; "
            size = "" if (w, h) == clip else f"--size {w}x{h} "
            return (f"{said}try {size}--frames {f} ({_seconds_text(f)} s), "
                    f"the longest {w}x{h} clip that fits")
        f -= 17
    side = hf.CANVAS_MULTIPLE
    smallest = plan_at(hf.MIN_FRAMES, side, side)
    if smallest.fits:
        return None
    bare = plan_at(hf.MIN_FRAMES, side, side, loras=())
    if bare.fits:
        lora = smallest.terms["denoise"]["lora"]
        return (f"no clip fits at any size with these adapters, which take "
                f"{lora / (1 << 30):.2f} GiB of the denoise; run with fewer "
                f"adapters" + (f", or with {BASE_RECIPE}" if base else ""))
    dit = bare.terms["denoise"]["dit"]
    return (f"no clip fits at any size with this DiT file, which takes "
            f"{dit / (1 << 30):.2f} GiB of the denoise; run a smaller quant "
            f"of it")


def _generate_parser(prog: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description="Write a clip with sound from a prompt. Steps, shift, "
                    "size and length default to the model entry's, else "
                    "the base model's.")
    p.add_argument("model", metavar="MODEL",
                   help="The id of a video_models entry in the config, "
                        "which brings its encoder, VAEs and LoRA, or a DiT "
                        "GGUF.")
    p.add_argument("--config", default=None, metavar="FILE",
                   help="The config whose video_models MODEL names "
                        "(default: the usual config search).")
    p.add_argument("--variant", default=None, metavar="NAME",
                   help="Run the DiT as this variant, such as fl2va-turbo4 "
                        "for a merged Turbo file gmlx does not know.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--prompt", default=None, help="The prompt.")
    src.add_argument("--prompt-file", default=None, metavar="FILE",
                     help="Read the prompt from this file.")
    length = p.add_mutually_exclusive_group()
    length.add_argument("--seconds", type=_seconds, default=None,
                        metavar="S",
                        help="Clip length, rounded up to a clip of 17 k + 5 "
                             "frames at 24 fps: 5 is 124 frames, 10 is 243 "
                             "and 15 is 362 (default 5).")
    length.add_argument("--frames", type=_frames, default=None, metavar="N",
                        help="Exact frame count, 17 k + 5 from 5 to 362.")
    p.add_argument("--size", type=_size, default=None, metavar="WxH",
                   help="Width and height, multiples of 32 (default "
                        "1344x768).")
    p.add_argument("--steps", type=_positive, default=None, metavar="N",
                   help="DiT forwards (default: the entry's, else the "
                        "variant's).")
    p.add_argument("--seed", type=_seed, default=None, metavar="N",
                   help="Seed (default random, and printed).")
    p.add_argument("--shift", type=_shift, default=None, metavar="S",
                   help="The video flow shift (default: the entry's, else "
                        "the variant's).")
    p.add_argument("--adapter", action="append", type=_adapter, default=None,
                   metavar="PATH[:STRENGTH]",
                   help="Apply this LoRA safetensors file at this strength "
                        "(default 1), after the entry's. Repeat the flag to "
                        "stack adapters.")
    p.add_argument("--first-frame", default=None, metavar="IMAGE",
                   help="Start the clip at this image, stretched to the "
                        "canvas. The prompt sees it too, through the "
                        "encoder's vision tower (--mmproj).")
    p.add_argument("--last-frame", default=None, metavar="IMAGE",
                   help="End the clip at this image, cropped about its "
                        "center to the canvas's aspect. The prompt sees "
                        "it too, as it sees --first-frame.")
    p.add_argument("--guide", dest="guides", action="append", type=_guide,
                   default=None, metavar="PATH[@FRAME]",
                   help="Anchor this image, or this video with its sound, "
                        "at FRAME (default 0, and a negative frame counts "
                        "from the end), cropped about its center to the "
                        "canvas's aspect. A video keeps its first 17 k + 5 "
                        "frames, or its first frame when it has fewer "
                        "than 5, and they must fit in the clip from FRAME. "
                        "Repeat the flag for more guides.")
    p.add_argument("--guide-audio", dest="guides", action="append",
                   type=_guide_audio, default=None, metavar="AUDIO[@FRAME]",
                   help="Anchor the sound of this file at FRAME (default "
                        "0), cut to the clip's audio from there. Repeat "
                        "the flag for more.")
    p.add_argument("--guide-no-audio", action="store_true",
                   help="Anchor --guide videos without their sound.")
    p.add_argument("--output", default=None, metavar="PATH",
                   help="The MP4 to write (default ./gmlx-<time>-<seed>.mp4).")
    p.add_argument("--frames-dir", default=None, metavar="DIR",
                   help="Write PNG frames and audio.wav to this folder. "
                        "Without --output, no MP4 is written and ffmpeg is "
                        "not needed, unless a guide video or sound is read.")
    p.add_argument("--preview", nargs="?", const="", default=None,
                   metavar="TAEH3",
                   help="Decode with TAEH3, the small preview decoder, in "
                        "place of the full video VAE: the entry's file, or "
                        "this one.")
    p.add_argument("--save-latent", default=None, metavar="PATH",
                   help="Also write the final latents to this .npz file.")
    p.add_argument("--encoder", default=None, metavar="GGUF",
                   help="The text encoder GGUF.")
    p.add_argument("--mmproj", default=None, metavar="GGUF",
                   help="The text encoder's vision tower, which "
                        "--first-frame and --last-frame need (default: "
                        "the entry's encoder_mmproj).")
    p.add_argument("--vae", default=None, metavar="PATH",
                   help="The video VAE safetensors file.")
    p.add_argument("--audio-vae", default=None, metavar="PATH",
                   help="The audio VAE safetensors file.")
    p.add_argument("--json", action="store_true",
                   help="Print a JSON record on stdout in place of the "
                        "summary line.")
    return p


def _config(a):
    import gmlx.config as cfgmod

    path = (os.path.expanduser(a.config) if a.config
            else next((str(c) for c in cfgmod.default_config_paths()
                       if os.path.exists(c)), None))
    return path


def _from_config(a) -> None:
    """Fill the files and defaults from the video_models entry MODEL
    names, when MODEL is no file. Flags given win."""
    import gmlx.config as cfgmod

    a.entry_defaults, a.entry_adapters, a.tiny_vae = {}, [], None
    if os.path.isfile(os.path.expanduser(a.model)):
        return
    path = _config(a)
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"MODEL {a.model}: no such file")
    try:
        cfg = cfgmod.load_config(path)
        if a.model not in cfg.video_models:
            known = ", ".join(sorted(cfg.video_models)) or "none"
            raise FileNotFoundError(
                f"MODEL {a.model}: no such file, and no video model of that "
                f"id in {path} (it has {known}). gmlx video pull "
                f"{a.model} fetches a recipe of that name")
        rm = cfgmod.resolve_video_model(a.model, cfg)
    except cfgmod.ConfigError as e:
        raise FileNotFoundError(f"MODEL {a.model}: {e}") from None
    a.model = rm.path
    a.encoder = a.encoder or rm.encoder
    a.mmproj = a.mmproj or rm.encoder_mmproj
    a.vae = a.vae or rm.vae
    a.audio_vae = a.audio_vae or rm.audio_vae
    a.variant = a.variant or rm.variant
    a.tiny_vae = rm.tiny_vae
    a.entry_defaults = dict(rm.defaults)
    a.entry_adapters = list(rm.adapters)


def _settings(a, fam, variant: str) -> dict:
    """The clip's settings: the flags, then the entry's defaults, then
    the variant's."""
    d = a.entry_defaults
    row = fam.defaults(variant)
    if a.frames is not None:
        frames = a.frames
    elif a.seconds is not None:
        frames = frames_of(a.seconds)
    else:
        sec = d.get("seconds", DEFAULT_SECONDS)
        if not _clip_seconds(sec):
            raise _Usage(f"the entry's defaults.seconds is {sec:g}; a clip "
                         f"takes {_seconds_range()} seconds")
        frames = frames_of(sec)
    width, height = a.size or (_size(d["size"]) if d.get("size")
                               else DEFAULT_SIZE)
    shifts = dict(row.shifts)
    if a.shift is not None or "shift" in d:
        shifts["video"] = a.shift if a.shift is not None else d["shift"]
    shifts = {k: float(v) for k, v in shifts.items()}
    steps = a.steps or d.get("steps") or row.steps
    if not _schedules(steps, shifts, fam.rungs(variant, steps)):
        src = ("--shift" if a.shift is not None else
               "the entry's defaults.shift" if "shift" in d else
               "the default video shift")
        raise _Usage(f"{src} {shifts['video']:g} gives no {steps}-step "
                     "schedule")
    return {"frames": frames, "width": width, "height": height,
            "steps": steps, "shifts": shifts,
            # A random seed stays under 2^31, short to print and retype.
            "seed": a.seed if a.seed is not None else secrets.randbelow(
                2 ** 31)}


def _prompt(a) -> str:
    if a.prompt is not None:
        return a.prompt
    if a.prompt_file is None:
        raise _Usage("--prompt or --prompt-file is required")
    with open(os.path.expanduser(a.prompt_file), encoding="utf-8") as f:
        return f.read().rstrip("\n")


def _keyframe_flags(a) -> list:
    return [f for f, p in (("--first-frame", a.first_frame),
                           ("--last-frame", a.last_frame)) if p]


def _anchor_flags(a) -> list:
    """(flag, path) of each anchor the flags ask for, keyframes first."""
    return ([(f, getattr(a, f[2:].replace("-", "_")))
             for f in _keyframe_flags(a)]
            + [(f, p) for f, p, _ in a.guides or ()])


def _need(a, preview: str | None) -> None:
    for flag, path in (("--encoder", a.encoder), ("--audio-vae", a.audio_vae),
                       ("--preview", preview) if preview
                       else ("--vae", a.vae)):
        if not path:
            raise _Usage(f"{flag} is required with a DiT path as MODEL")
        if not os.path.isfile(os.path.expanduser(path)):
            raise FileNotFoundError(f"{flag} {path}: no such file")


def _need_anchors(a, preview: str | None) -> None:
    """Refuse an anchor flag the run lacks a file for."""
    keyframes = _keyframe_flags(a)
    if keyframes and not a.mmproj:
        raise _Usage(f"{keyframes[0]} needs the encoder's vision tower: "
                     "pass --mmproj, or name encoder_mmproj in the entry")
    video = [f for f, _ in _anchor_flags(a) if f != "--guide-audio"]
    if preview and video and not a.vae:
        raise _Usage(f"--vae is required with --preview and {video[0]}, "
                     "whose anchor encodes through the video VAE")
    for flag, path in ([("--mmproj", a.mmproj)] if keyframes else []) \
            + _anchor_flags(a):
        if not os.path.isfile(os.path.expanduser(path)):
            raise FileNotFoundError(f"{flag} {path}: no such file")


def _writable(a, out: str | None) -> None:
    """Refuse an output the run could not write, before any model loads."""
    from gmlx.tune.lora import probe_writable

    paths = [p for p in (out, a.save_latent) if p]
    if a.frames_dir is not None:
        d = os.path.expanduser(a.frames_dir)
        if os.path.exists(d) and not os.path.isdir(d):
            raise OSError(f"cannot write frames to {a.frames_dir}: it is a "
                          "file")
        paths.append(os.path.join(d, "audio.wav"))
    for p in paths:
        err = probe_writable(os.path.expanduser(p))
        if err is not None:
            raise OSError(f"cannot write {p}: {err}")


def _makedirs_for(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def _release() -> None:
    import gc

    import mlx.core as mx

    gc.collect()
    mx.clear_cache()


def _progress(log, t0):
    def on_step(i, n):
        el = time.perf_counter() - t0
        est = el / (i + 1) * n
        log(f"step {i + 1} of {n}, {el:.0f} s, about {est - el:.0f} s left")
    return on_step


@dataclasses.dataclass(frozen=True)
class _Anchor:
    """An anchor the flags ask for, read and placed before any model
    loads: its flag and path, its placement and planned rows, and what
    it encodes from, a still at the canvas size, a video file whose
    frames are read when it encodes, its sound, or both."""
    flag: str
    path: str
    placed: object                  # anchors.Placement
    rows: object                    # layout.AnchorRows
    still: object = None            # (H, W, 3) uint8
    video: str | None = None
    wave: object = None             # (2, samples) float32
    ffmpeg: str | None = None


def _still(path: str) -> bool:
    """Whether Pillow opens ``path`` as an image."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path):
            return True
    except UnidentifiedImageError:
        return False


def _read_anchors(a, fam, s: dict) -> tuple[list, list]:
    """The keyframes, (H, W, 3) uint8 at the canvas in prompt order, and
    each anchor of the flags as ``_Anchor``, before any model loads. A
    guide video's frames are counted here and read when it encodes."""
    from gmlx.video import anchors, images, io
    from gmlx.video.layout import AnchorRows
    from gmlx.video.vae.h3_audio import HOP, SAMPLE_RATE

    frames, w, h = s["frames"], s["width"], s["height"]
    audio_t = fam.latent_shapes(frames, w, h)[1][3]
    samples = audio_t * HOP + GUIDE_AUDIO_PAST_S * SAMPLE_RATE
    exe = shutil.which("ffmpeg")
    keyframes: list = []
    out: list = []

    def add(flag, path, frame, *, n=0, wave=None, most=None, **kw):
        try:
            if n == most:
                # Read to its bound, the video holds more than fits.
                raise ValueError(
                    f"a guide clip of {anchors.guide_frames(n)} or more "
                    f"frames at frame {frame} does not fit in the clip's "
                    f"{frames} frames")
            p = anchors.place(frame, frames, audio_t,
                              fam.timeline.frame_rescale, video_frames=n,
                              audio=wave is not None)
        except ValueError as e:
            raise ValueError(f"{flag} {path}: {e}") from None
        video_t = (0 if not p.video_frames else 1 if p.video_frames == 1
                   else fam.latent_shapes(p.video_frames, w, h)[0][2])
        sound = (0 if p.audio_t is None
                 else min(p.audio_t, -(-wave.shape[1] // HOP)))
        out.append(_Anchor(flag, path, p, AnchorRows(p.frame, video_t, sound),
                           wave=wave, **kw))

    for flag, frame, resize in (("--first-frame", 0, images.stretch),
                                ("--last-frame", frames - 1, images.cover)):
        path = getattr(a, flag[2:].replace("-", "_"))
        if path:
            im = resize(images.read(os.path.expanduser(path), exe), w, h)
            keyframes.append(im)
            add(flag, path, frame, n=1, still=im)
    for flag, path, frame in a.guides or ():
        file = os.path.expanduser(path)
        if flag == "--guide" and _still(file):
            add(flag, path, frame, n=1,
                still=images.cover(images.read(file, exe), w, h))
            continue
        if exe is None:
            raise io.FfmpegError(
                f"{flag} {path}: gmlx reads video and audio files through "
                "ffmpeg, which is not on PATH: install ffmpeg (brew install "
                "ffmpeg)")
        at = frame + frames if frame < 0 else frame
        if not 0 <= at < frames:
            add(flag, path, frame, n=1)     # refuses a frame off the clip
        wave = None
        if flag == "--guide-audio" or not a.guide_no_audio:
            wave = io.guide_audio(file, rate=SAMPLE_RATE, samples=samples,
                                  ffmpeg=exe, optional=flag == "--guide")
        if flag == "--guide-audio":
            add(flag, path, frame, wave=wave)
            continue
        # 17 frames past those left from the frame tell a guide that does
        # not fit from one that does.
        most = frames - at + 17
        add(flag, path, frame, n=io.guide_length(file, most=most, ffmpeg=exe),
            most=most, wave=wave, video=file, ffmpeg=exe)
    return keyframes, out


def _encode_anchors(todo: list, s: dict, vae: str, audio_vae: str) -> tuple:
    """The anchors of ``todo``: their frames through the video VAE file
    ``vae``, then their sound through ``audio_vae``, one VAE resident at
    a time."""
    import mlx.core as mx

    from gmlx.video import anchors, io
    from gmlx.video.vae.reader import load_vae

    video, audio = {}, {}
    if any(t.placed.video_frames for t in todo):
        model = load_vae(os.path.expanduser(vae), family="h3",
                         dtype=mx.float16)
        for i, t in enumerate(todo):
            if not t.placed.video_frames:
                continue
            frames = (t.still[None] if t.still is not None else
                      io.guide_frames(t.video, width=s["width"],
                                      height=s["height"],
                                      most=t.placed.video_frames,
                                      ffmpeg=t.ffmpeg))
            video[i] = anchors.encode(
                dataclasses.replace(t.placed, audio_t=None), frames=frames,
                video_vae=model).video
            del frames
        del model
        _release()
    if any(t.wave is not None for t in todo):
        model = load_vae(os.path.expanduser(audio_vae), family="h3_audio",
                         dtype=mx.float32)
        for i, t in enumerate(todo):
            if t.wave is not None:
                audio[i] = anchors.encode(
                    dataclasses.replace(t.placed, video_frames=0),
                    waveform=t.wave, audio_vae=model).audio
        del model
        _release()
    return tuple(anchors.Anchor(t.placed.frame, video.get(i), audio.get(i))
                 for i, t in enumerate(todo))


def _output(a, seed: int) -> str:
    if a.output:
        return os.path.expanduser(a.output)
    return os.path.abspath(f"gmlx-{time.strftime('%Y%m%d-%H%M%S')}-{seed}"
                           ".mp4")


def _generate(a) -> int:
    import mlx.core as mx
    import numpy as np

    from gmlx.video import encoder as video_encoder
    from gmlx.video import catalog, io, memfit, pipeline
    from gmlx.video.loader import identify_dit
    from gmlx.video.registry import run_refusal
    from gmlx.video.vae.h3_audio import SAMPLE_RATE
    from gmlx.video.vae.reader import load_vae

    prompt = _prompt(a)
    _from_config(a)
    preview = a.preview
    if preview == "":
        preview = a.tiny_vae
        if not preview:
            raise _Usage("--preview needs the TAEH3 file with a DiT path as "
                         "MODEL, or an entry that names one")
    _need(a, preview)
    def log(*x):
        print(*x, file=sys.stderr, flush=True)
    ffmpeg = None
    if a.frames_dir is None or a.output:
        ffmpeg = io.probe()
    vid = identify_dit(os.path.expanduser(a.model), variant=a.variant)
    fam = vid.family
    why = run_refusal(fam, vid.variant)
    if why:
        raise RuntimeError(why)
    row = fam.variant_row(vid.variant)
    if _anchor_flags(a) and row is not None and row.vsa_sparsity:
        from gmlx.video.families.h3 import vsa

        raise RuntimeError(vsa.anchor_refusal(len(_anchor_flags(a))))
    _need_anchors(a, preview)
    adapters = list(a.entry_adapters) + list(a.adapter or ())
    # Refused from the headers before the encoder loads.
    for path, _ in adapters:
        pipeline.vet_adapter(fam, dict(vid.config), path)
    if adapters and row is not None and row.merged:
        log(f"[video] warning: this {fam.title} GGUF has {row.merged} "
            f"merged in, and each adapter runs on top of it")
    s = _settings(a, fam, vid.variant)
    out = _output(a, s["seed"]) if ffmpeg is not None else None
    _writable(a, out)
    keyframes, todo = _read_anchors(a, fam, s)
    tokenizer = video_encoder.header_tokenizer(fam, a.encoder)
    texts: dict = {}

    def text_at(width, height):
        # The keyframes' vision rows follow the canvas.
        key = (width, height) if keyframes else None
        if key not in texts:
            texts[key] = memfit.prompt_rows(fam, prompt, tokenizer,
                                            [(width, height)] * len(keyframes))
        return texts[key]
    rows = tuple(t.rows for t in todo)
    anchor_vae = (a.vae if preview and any(t.placed.video_frames
                                           for t in todo) else None)

    def plan_at(frames, width, height, loras=None):
        sizes = [(width, height)] * len(keyframes)
        return memfit.plan_run(
            fam, dit=a.model, encoder=a.encoder, vae=preview or a.vae,
            tiny=bool(preview), audio_vae=a.audio_vae,
            loras=[p for p, _ in adapters] if loras is None else loras,
            frames=frames, width=width, height=height, variant=vid.variant,
            text=text_at(width, height),
            mmproj=a.mmproj if keyframes else None,
            blocks=memfit.keyframe_blocks(fam, sizes) if sizes else (),
            anchors=rows, anchor_vae=anchor_vae)
    # BASE_RECIPE is new to a run of another DiT or with an entry LoRA.
    base = (vid.variant != catalog.RECIPES[BASE_RECIPE].variant
            or bool(a.entry_adapters))
    memfit.check(plan_at(s["frames"], s["width"], s["height"]),
                 resize=lambda: _size_hint(s["frames"], s["width"],
                                           s["height"], plan_at, base,
                                           lengths=not todo))
    log(f"[video] {fam.title} {s['width']}x{s['height']}, {s['frames']} "
        f"frames, {s['steps']} steps at shift {s['shifts']['video']:g}, "
        f"seed {s['seed']}")
    for t in todo:
        what = [f"{t.placed.video_frames} frame"
                + ("s" if t.placed.video_frames > 1 else "")] \
            if t.placed.video_frames else []
        what += [f"{t.rows.audio_t} audio latents"] if t.rows.audio_t else []
        log(f"[video] {t.flag} {t.path} at frame {t.placed.frame}: "
            f"{' and '.join(what)}")
    times = {}

    t0 = time.perf_counter()
    handle = video_encoder.load_encoder(
        fam, os.path.expanduser(a.encoder),
        mmproj=os.path.expanduser(a.mmproj) if keyframes else None, log=log)
    try:
        if keyframes:
            feats, spans = fam.keyframe_encode(handle, prompt, keyframes)
        else:
            feats, spans = mx.array(handle.encode(prompt).features)[None], ()
        mx.eval(feats)
    finally:
        handle.release()
    times["encode_s"] = round(time.perf_counter() - t0, 2)
    placed = ()
    if todo:
        t0 = time.perf_counter()
        placed = _encode_anchors(todo, s, a.vae, a.audio_vae)
        times["anchors_s"] = round(time.perf_counter() - t0, 2)

    t0 = time.perf_counter()
    pipe = pipeline.load(os.path.expanduser(a.model), variant=a.variant,
                         log=log)
    table = pipeline.adapter_table(
        pipe, [(os.path.expanduser(p), st) for p, st in adapters])
    xv, xa = pipeline.generate(
        pipe, feats, frames=s["frames"], width=s["width"],
        height=s["height"], steps=s["steps"], shifts=s["shifts"],
        seed=s["seed"], table=table, anchors=placed, spans=spans,
        on_step=_progress(log, time.perf_counter()))
    mx.eval(xv, xa)
    del pipe
    _release()
    times["denoise_s"] = round(time.perf_counter() - t0, 2)
    if a.save_latent:
        _makedirs_for(os.path.expanduser(a.save_latent))
        np.savez(os.path.expanduser(a.save_latent), video=np.array(xv),
                 audio=np.array(xa), seed=s["seed"], steps=s["steps"])

    t0 = time.perf_counter()
    avae = load_vae(os.path.expanduser(a.audio_vae), family="h3_audio",
                    dtype=mx.float32)
    hf = _family()
    rate = SAMPLE_RATE
    # One length for the MP4 and the frames' audio.wav.
    wave = io.fit_audio(np.array(avae.decode(xa)[0]), frames=s["frames"],
                        fps=hf.FPS, rate=rate)
    del avae
    _release()
    times["audio_s"] = round(time.perf_counter() - t0, 2)

    t0 = time.perf_counter()
    vae = load_vae(os.path.expanduser(preview or a.vae),
                   family="h3_tiny" if preview else "h3", dtype=mx.float16)
    if a.frames_dir is not None:
        os.makedirs(os.path.expanduser(a.frames_dir), exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".gmlx-video-") as tmp:
        wav = io.write_wav(os.path.join(tmp, "audio.wav"), wave, rate)
        mux = None
        if out is not None:
            _makedirs_for(out)
            mux = io.Mux(out, width=s["width"], height=s["height"],
                         fps=hf.FPS, wav=wav, ffmpeg=ffmpeg)
        try:
            n = 0
            for chunk in vae.decode_frames(xv):
                frames = np.array(chunk[0])
                if mux is not None:
                    mux.write(frames)
                if a.frames_dir is not None:
                    n = io.write_frames(os.path.expanduser(a.frames_dir),
                                        frames, n)
                log(f"decoded {mux.frames if mux else n} of {s['frames']} "
                    "frames")
            if mux is not None:
                mux.close()
        except BaseException:
            if mux is not None:
                mux.abort()
            raise
        if a.frames_dir is not None:
            io.write_wav(os.path.join(os.path.expanduser(a.frames_dir),
                                      "audio.wav"), wave, rate)
    times["decode_s"] = round(time.perf_counter() - t0, 2)
    rec = {"output": out, "frames_dir": a.frames_dir, "prompt": prompt,
           "variant": vid.variant, **s, "times": times,
           "adapters": [[p, st] for p, st in adapters],
           "anchors": [{"flag": t.flag, "path": t.path,
                        "frame": t.placed.frame,
                        "frames": t.placed.video_frames,
                        "audio_latents": t.rows.audio_t} for t in todo]}
    if a.json:
        print(json.dumps(rec, indent=2))
    else:
        print(f"wrote {out or a.frames_dir} in "
              f"{sum(times.values()):.0f} s")
    return 0


def cmd_generate(argv: list[str], prog: str = "gmlx video generate") -> int:
    a = _generate_parser(prog).parse_args(argv)
    try:
        from gmlx.image.memfit import bounded_cache

        with bounded_cache():
            return _generate(a)
    except _Usage as e:
        print(f"{prog}: error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as e:  # noqa: BLE001 (refusals and failures exit 1)
        print(f"error: {e}", file=sys.stderr)
        return 1


def cmd_bench(argv: list[str], prog: str = "gmlx video bench") -> int:
    from .video_bench import cmd_bench as run

    return run(argv, prog=prog)


def cmd_pull(argv: list[str], prog: str = "gmlx video pull") -> int:
    from .video_pull import cmd_pull as run

    return run(argv, prog=prog)


HANDLERS = {"generate": cmd_generate, "bench": cmd_bench,
            "pull": cmd_pull}


def cmd_video(argv: list[str], prog: str = "gmlx video") -> int:
    """Dispatch ``gmlx video <action> ...``."""
    if not argv or argv[0] in ("-h", "--help", "help"):
        _print_help(prog)
        return 0
    action, rest = argv[0], argv[1:]
    if action not in HANDLERS:
        print(f"error: unknown action {action!r}; one of "
              f"{', '.join(_ACTIONS)}", file=sys.stderr)
        return 2
    return HANDLERS[action](rest, prog=f"{prog} {action}")
