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
import json
import math
import os
import secrets
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
    if n < 0:
        raise argparse.ArgumentTypeError(
            f"a non-negative integer is required, got {text!r}")
    return n


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


def _schedules(steps: int, shifts: dict) -> bool:
    """Whether every stream gets ``steps`` finite float32 forwards."""
    import numpy as np

    from gmlx.video import schedule

    try:
        with np.errstate(all="ignore"):
            sig = schedule.stream_sigmas(steps, shifts)
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
    p.add_argument("--output", default=None, metavar="PATH",
                   help="The MP4 to write (default ./gmlx-<time>-<seed>.mp4).")
    p.add_argument("--frames-dir", default=None, metavar="DIR",
                   help="Write PNG frames and audio.wav to this folder. "
                        "Without --output, no MP4 is written and ffmpeg is "
                        "not needed.")
    p.add_argument("--preview", nargs="?", const="", default=None,
                   metavar="TAEH3",
                   help="Decode with TAEH3, the small preview decoder, in "
                        "place of the full video VAE: the entry's file, or "
                        "this one.")
    p.add_argument("--save-latent", default=None, metavar="PATH",
                   help="Also write the final latents to this .npz file.")
    p.add_argument("--encoder", default=None, metavar="GGUF",
                   help="The text encoder GGUF.")
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
    row = fam.sampling_row(variant)
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
    if not _schedules(steps, shifts):
        src = ("--shift" if a.shift is not None else
               "the entry's defaults.shift" if "shift" in d else
               "the default video shift")
        raise _Usage(f"{src} {shifts['video']:g} gives no {steps}-step "
                     "schedule")
    return {"frames": frames, "width": width, "height": height,
            "steps": steps, "shifts": shifts,
            "seed": a.seed if a.seed is not None else secrets.randbelow(
                2 ** 31)}


def _prompt(a) -> str:
    if a.prompt is not None:
        return a.prompt
    if a.prompt_file is None:
        raise _Usage("--prompt or --prompt-file is required")
    with open(os.path.expanduser(a.prompt_file), encoding="utf-8") as f:
        return f.read().rstrip("\n")


def _need(a, preview: str | None) -> None:
    for flag, path in (("--encoder", a.encoder), ("--audio-vae", a.audio_vae),
                       ("--preview", preview) if preview
                       else ("--vae", a.vae)):
        if not path:
            raise _Usage(f"{flag} is required with a DiT path as MODEL")
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


def _output(a, seed: int) -> str:
    if a.output:
        return os.path.expanduser(a.output)
    return os.path.abspath(f"gmlx-{time.strftime('%Y%m%d-%H%M%S')}-{seed}"
                           ".mp4")


def _generate(a) -> int:
    import mlx.core as mx
    import numpy as np

    from gmlx.video import encoder as video_encoder
    from gmlx.video import io, memfit, pipeline
    from gmlx.video.loader import identify_dit
    from gmlx.video.registry import chip_refusal
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
    why = chip_refusal(fam)
    if why:
        raise RuntimeError(why)
    adapters = list(a.entry_adapters) + list(a.adapter or ())
    # Refused from the headers before the encoder loads.
    for path, _ in adapters:
        pipeline.vet_adapter(fam, dict(vid.config), path)
    row = fam.variant_row(vid.variant)
    if adapters and row is not None and row.merged:
        log(f"[video] warning: this {fam.title} GGUF has {row.merged} "
            f"merged in, and each adapter runs on top of it")
    s = _settings(a, fam, vid.variant)
    out = _output(a, s["seed"]) if ffmpeg is not None else None
    _writable(a, out)
    plan = memfit.plan_run(
        fam, dit=a.model, encoder=a.encoder, vae=preview or a.vae,
        tiny=bool(preview), audio_vae=a.audio_vae,
        loras=[p for p, _ in adapters], frames=s["frames"],
        width=s["width"], height=s["height"],
        text=memfit.prompt_rows(
            fam, prompt, video_encoder.header_tokenizer(fam, a.encoder)))
    memfit.check(plan)
    log(f"[video] {fam.title} {s['width']}x{s['height']}, {s['frames']} "
        f"frames, {s['steps']} steps at shift {s['shifts']['video']:g}, "
        f"seed {s['seed']}")
    times = {}

    t0 = time.perf_counter()
    handle = video_encoder.load_encoder(fam, os.path.expanduser(a.encoder),
                                        log=log)
    try:
        feats = mx.array(handle.encode(prompt).features)[None]
        mx.eval(feats)
    finally:
        handle.release()
    times["encode_s"] = round(time.perf_counter() - t0, 2)

    t0 = time.perf_counter()
    pipe = pipeline.load(os.path.expanduser(a.model), variant=a.variant,
                         log=log)
    table = pipeline.adapter_table(
        pipe, [(os.path.expanduser(p), st) for p, st in adapters])
    xv, xa = pipeline.generate(
        pipe, feats, frames=s["frames"], width=s["width"],
        height=s["height"], steps=s["steps"], shifts=s["shifts"],
        seed=s["seed"], table=table,
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
           "adapters": [[p, st] for p, st in adapters]}
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
