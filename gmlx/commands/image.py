"""``gmlx image``: image generation from GGUF diffusion models.

``generate`` samples images from a DiT GGUF with its text encoder and VAE
through the ``gmlx.image.api`` primitives, so its latents equal those of
``sample`` called with the same noise. ``bench`` times whole images and
their phases and writes records of schema ``gmlx.image.bench/1``.
``pack-vae`` writes a lossless VAE GGUF from a safetensors VAE and checks
it bit for bit.

Exit codes follow ``gmlx run``: 0 on success, 1 for a refusal or a
runtime failure, 2 for a usage error and 130 for an interrupt.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import secrets
import sys
import tempfile
import time

_ACTIONS = ("generate", "bench", "pack-vae")
_ACTION_DESC = {
    "generate": "generate images from a DiT GGUF, its text encoder and VAE",
    "bench": "time whole images and their phases, with a JSON record",
    "pack-vae": "write a lossless VAE GGUF from a safetensors VAE",
}
_FORMATS = {"png": ".png", "jpeg": ".jpg", "webp": ".webp"}
_EXT_FORMAT = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg",
               ".webp": "webp"}
_BACKGROUNDS = ("opaque", "transparent")
_SAMPLERS = ("euler", "uni_pc_bh2")


class _Usage(Exception):
    """A usage error the parser cannot see, which exits 2."""
_PACK_DTYPES = {"f32": "F32", "f16": "F16", "bf16": "BF16"}


def _print_help(prog: str) -> None:
    lines = [f"usage: {prog} <action> [options]", "",
             "image generation from GGUF diffusion models.", "", "actions:"]
    for a in _ACTIONS:
        lines.append(f"  {a:<10} {_ACTION_DESC[a]}")
    lines += ["", f"run `{prog} <action> --help` for an action's options."]
    print("\n".join(lines))


def _refuse(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


def _size(text: str) -> tuple[int, int]:
    w, sep, h = text.lower().partition("x")
    try:
        size = (int(w), int(h))
    except ValueError:
        size = (0, 0)
    if not sep or min(size) <= 0:
        raise argparse.ArgumentTypeError(
            f"a size such as 1024x1024 is required, got {text!r}")
    return size


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


def _strength(text: str) -> float:
    try:
        x = float(text)
    except ValueError:
        x = -1.0
    if not 0.0 < x <= 1.0:
        raise argparse.ArgumentTypeError(
            f"a strength above 0 and at most 1 is required, got {text!r}")
    return x


def _quality(text: str) -> int:
    try:
        n = int(text)
    except ValueError:
        n = 0
    if not 1 <= n <= 100:
        raise argparse.ArgumentTypeError(
            f"a quality from 1 to 100 is required, got {text!r}")
    return n


def _adapter(text: str) -> tuple[str, float]:
    """PATH or PATH:STRENGTH. The text after the last colon is a strength
    only when it parses as a float, so a path may hold colons."""
    head, sep, tail = text.rpartition(":")
    path, strength = text, 1.0
    if sep and head:
        try:
            path, strength = head, float(tail)
        except ValueError:
            path, strength = text, 1.0
        else:
            # float() reads 1_0 as 10, which nobody means as a strength.
            if "_" in tail:
                raise argparse.ArgumentTypeError(
                    f"a strength is written without underscores, got "
                    f"{tail!r} in {text!r}")
    if not path or not math.isfinite(strength):
        raise argparse.ArgumentTypeError(
            f"a LoRA file with an optional finite strength, as PATH or "
            f"PATH:STRENGTH, is required, got {text!r}")
    return path, strength


def _add_adapter_flag(p: argparse.ArgumentParser) -> None:
    p.add_argument("--adapter", action="append", type=_adapter, default=None,
                   metavar="PATH[:STRENGTH]",
                   help="Apply this LoRA safetensors file at this strength "
                        "(default 1). Repeat the flag to stack adapters.")


def _generate_parser(prog: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description="Generate images from a DiT GGUF with its text encoder "
                    "and VAE. Steps, guidance and size default to the "
                    "model's variant. Images run in sequence at the seed "
                    "plus their index.")
    p.add_argument("model", metavar="MODEL", help="The DiT GGUF.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--prompt", default=None, help="The prompt.")
    src.add_argument("--prompt-file", default=None, metavar="FILE",
                     help="Read the prompt from this file.")
    p.add_argument("--negative-prompt", default=None, metavar="TEXT",
                   help="The negative prompt, used when --guidance is not "
                        "1 (default empty).")
    p.add_argument("--size", type=_size, default=None, metavar="WxH",
                   help="Image width and height, multiples of 16 (default: "
                        "the variant's, or the init image's).")
    p.add_argument("--steps", type=_positive, default=None, metavar="N",
                   help="Sampling steps (default: the variant's).")
    p.add_argument("--seed", type=_seed, default=None, metavar="N",
                   help="Seed of the first image (default random, and "
                        "printed).")
    p.add_argument("--guidance", type=float, default=None, metavar="CFG",
                   help="Classifier-free guidance; 1 turns it off (default: "
                        "the variant's).")
    p.add_argument("--sampler", choices=_SAMPLERS, default="euler",
                   help="The sampler, in ComfyUI's spelling (default "
                        "euler).")
    p.add_argument("--init-image", default=None, metavar="PATH",
                   help="Start from this image (img2img).")
    p.add_argument("--mask", default=None, metavar="PATH",
                   help="Change only where this mask is white (inpainting). "
                        "Needs --init-image.")
    p.add_argument("--strength", type=_strength, default=None, metavar="F",
                   help="How far img2img moves from the init image, above 0 "
                        "and at most 1 (default 0.6, or 1 with --mask).")
    p.add_argument("--count", type=_positive, default=1, metavar="N",
                   help="Generate N images (default 1).")
    p.add_argument("--output", default=None, metavar="PATH",
                   help="A file, a directory, or a template with {seed} and "
                        "{index} (default ./gmlx-<time>-<seed>.<ext>).")
    p.add_argument("--format", choices=sorted(_FORMATS), default=None,
                   help="Image format (default: from --output, else png).")
    p.add_argument("--quality", type=_quality, default=90, metavar="N",
                   help="jpeg and webp quality from 1 to 100 (default 90).")
    p.add_argument("--no-metadata", action="store_true",
                   help="Leave out the PNG text chunk that records the "
                        "settings.")
    p.add_argument("--background", choices=_BACKGROUNDS, default="opaque",
                   help="transparent writes the alpha channel of a model "
                        "that decodes one, as a png or webp (default "
                        "opaque).")
    p.add_argument("--save-latent", default=None, metavar="PATH",
                   help="Also write the final latents and seeds to this "
                        ".npz file.")
    p.add_argument("--encoder", default=None, metavar="GGUF",
                   help="The text encoder GGUF.")
    p.add_argument("--vae", default=None, metavar="PATH",
                   help="The VAE, a GGUF or safetensors file.")
    _add_adapter_flag(p)
    p.add_argument("--encoder-release", choices=("auto", "always", "never"),
                   default="auto",
                   help="Release the text encoder after encoding: when the "
                        "memory plan needs it (auto), always or never.")
    p.add_argument("--json", action="store_true",
                   help="Print a JSON record on stdout, and progress on "
                        "stderr.")
    return p


def _stride(family) -> int:
    """What a size is a multiple of: the family row's ``size_multiple``,
    else one DiT patch in pixels."""
    c = family.constants.values
    if "size_multiple" in c:
        return int(c["size_multiple"])
    return int(c["vae_scale_factor"]) * int(c["all_patch_size"][0])


def _check_release(a, plan) -> None:
    """``--encoder-release never`` on a request that fits only with the
    encoder released is a usage error."""
    if plan is not None and a.encoder_release == "never" \
            and plan.release_encoder:
        raise _Usage("--encoder-release never: this request fits only with "
                     f"the text encoder released; {plan.describe()}")


def _prompt(a) -> str:
    if a.prompt is not None:
        return a.prompt
    if a.prompt_file is None:
        raise _Usage("--prompt or --prompt-file is required")
    with open(os.path.expanduser(a.prompt_file), encoding="utf-8") as f:
        return f.read().rstrip("\n")


def _format(a) -> str:
    if a.format is not None:
        return a.format
    if a.output is not None and "{" not in a.output:
        ext = os.path.splitext(a.output)[1].lower()
        return _EXT_FORMAT.get(ext, "png")
    return "png"


def output_paths(output: str | None, seeds: list, fmt: str,
                 stamp: str) -> list[str]:
    """One path per seed. ``output`` is None, a directory, a template with
    ``{seed}`` and ``{index}``, or a file that gains ``-<seed>`` before its
    extension when there are several images."""
    ext = _FORMATS[fmt]
    names = [f"gmlx-{stamp}-{s}{ext}" for s in seeds]
    if output is None:
        return names
    out = os.path.expanduser(output)
    if "{" in out:
        try:
            paths = [out.format(seed=s, index=i) for i, s in enumerate(seeds)]
        except (KeyError, IndexError, ValueError) as e:
            raise _Usage(f"--output template {output!r}: {e}") from None
        if len(set(paths)) < len(paths):
            raise _Usage(f"--output template {output!r} names one file for "
                         f"{len(paths)} images; add {{seed}} or {{index}}")
        return paths
    if os.path.isdir(out) or out.endswith(os.sep):
        return [os.path.join(out, n) for n in names]
    if len(seeds) == 1:
        return [out]
    stem, e = os.path.splitext(out)
    return [f"{stem}-{s}{e}" for s in seeds]


def read_image(path: str, size: tuple | None, stride: int):
    """An image file as (H, W, 3) float32 in [0, 1] at ``size`` (width,
    height), resized with Lanczos. Without ``size`` the image keeps its
    own, cut down to multiples of ``stride``."""
    import numpy as np
    from PIL import Image

    with Image.open(os.path.expanduser(path)) as im:
        im = im.convert("RGB")
        if size is None:
            size = (im.width // stride * stride, im.height // stride * stride)
            if min(size) <= 0:
                raise ValueError(f"{path} is smaller than {stride} pixels")
        if im.size != tuple(size):
            im = im.resize(tuple(size), Image.Resampling.LANCZOS)
        return np.asarray(im, np.float32) / 255.0


def read_mask(path: str, size: tuple):
    """A mask file as (H, W) float32, 1 where the image changes: white
    pixels, binarized at one half as diffusers does."""
    import numpy as np
    from PIL import Image

    with Image.open(os.path.expanduser(path)) as im:
        im = im.convert("L")
        if im.size != tuple(size):
            im = im.resize(tuple(size), Image.Resampling.BILINEAR)
        m = np.asarray(im, np.float32) / 255.0
    return (m >= 0.5).astype(np.float32)


def _check_background(a, family, fmt: str) -> bool:
    """Whether the request writes an alpha channel. ``--background
    transparent`` is a usage error on a family whose VAE decodes none and
    with a format that holds none."""
    if a.background != "transparent":
        return False
    from gmlx.image.vae.reader import VAES

    row = VAES.get(family.name)
    if row is None or row.image_channels != 4:
        raise _Usage(f"--background transparent: {family.title} decodes no "
                     "alpha channel")
    if fmt == "jpeg":
        raise _Usage("--background transparent: jpeg holds no alpha "
                     "channel; write a png or a webp")
    return True


def to_uint8(image):
    """A decoded (1, H, W, C) image in [0, 1] as uint8, rounded: RGB, or
    RGBA with straight alpha."""
    import mlx.core as mx
    import numpy as np

    x = np.array(image[0].astype(mx.float32))
    return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)


def write_image(path: str, pixels, fmt: str, quality: int,
                metadata: dict | None) -> None:
    from PIL import Image, PngImagePlugin

    im = Image.fromarray(pixels, "RGBA" if pixels.shape[-1] == 4 else "RGB")
    # Written beside the target and renamed, so an interrupt or a failed
    # write leaves no partial image.
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".",
                               prefix=".gmlx-image-", suffix=".part")
    os.close(fd)
    try:
        if fmt == "png":
            info = None
            if metadata is not None:
                info = PngImagePlugin.PngInfo()
                info.add_text("gmlx", json.dumps(metadata, sort_keys=True))
            im.save(tmp, format="PNG", pnginfo=info)
        else:
            im.save(tmp, format=fmt.upper(), quality=quality)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _check_files(a) -> None:
    if not os.path.isfile(os.path.expanduser(a.model)):
        raise FileNotFoundError(f"MODEL {a.model}: no such file")
    if a.encoder is None or a.vae is None:
        raise _Usage("--encoder and --vae are required with a DiT path")
    for flag, path in (("--encoder", a.encoder), ("--vae", a.vae),
                       ("--init-image", a.init_image), ("--mask", a.mask)):
        if path is not None and not os.path.isfile(os.path.expanduser(path)):
            raise FileNotFoundError(f"{flag} {path}: no such file")
    for path, _ in a.adapter or ():
        if not os.path.isfile(os.path.expanduser(path)):
            raise FileNotFoundError(f"--adapter {path}: no such file")


def _vet_adapters(a, ident=None) -> None:
    """Refuse an ``--adapter`` file gmlx does not apply from its header,
    before any model loads. With the identified DiT, also refuse a file
    whose keys name no module of its family or a module the DiT file does
    not hold."""
    if not a.adapter:
        return
    from gmlx.image import lora

    keymap = modules = None
    if ident is not None:
        if ident.family.lora_map is None:
            raise lora.AdapterError(f"{ident.family.title} has no LoRA key "
                                    "map in gmlx")
        keymap = ident.family.lora_map(ident.config)
        modules = {n[:-len(".weight")] for n in ident.view.tensors
                   if n.endswith(".weight")}
    for path, _ in a.adapter:
        lora.vet_header(path, keymap=keymap, modules=modules)


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(os.path.expanduser(path), "rb") as f:
        while chunk := f.read(16 << 20):
            h.update(chunk)
    return h.hexdigest()


def _adapter_bytes(a) -> list:
    """An upper bound on the bytes each adapter holds once loaded: its
    pairs as the file stores them, and their tables in the DiT's dtype."""
    return [2 * os.path.getsize(os.path.expanduser(p))
            for p, _ in a.adapter or ()]


def _load_adapters(a, pipe, log) -> tuple:
    """Load the ``--adapter`` files into the pipeline and apply their text
    encoder halves. Returns the patch table id, or None without adapters,
    and one record per adapter for the metadata: ``sha256`` is the digest
    of the file, and ``rows_sha256`` names the pairs gmlx placed and the
    modules it placed them on."""
    if not a.adapter:
        return None, []
    from gmlx.image import api, lora

    rows = []
    for path, strength in a.adapter:
        ad = lora.load_adapter(pipe, path)
        for w in ad.warnings:
            log(f"warning: {w}")
        rows.append((ad, strength))
    lora.apply_text_adapters(pipe, rows)
    table = api.define_patch_table(pipe, lora.table_entries(rows))
    meta = []
    for ad, strength in rows:
        text = f" and {len(ad.text)} on the text encoder" if ad.text else ""
        log(f"[image] adapter {ad.name} at strength {strength:g}: "
            f"{len(ad.dit)} pairs{text}")
        meta.append({"file": ad.name, "strength": strength,
                     "sha256": _file_sha256(ad.path),
                     "rows_sha256": ad.sha256})
    return table, meta


def _settings(a, ident, defaults):
    """Steps, cfg and strength from the flags and the variant's defaults."""
    if defaults is None and (a.steps is None or a.guidance is None):
        raise ValueError(
            f"{a.model} matches no known {ident.family.title} variant; pass "
            f"--steps and --guidance")
    steps = a.steps if a.steps is not None else defaults.steps
    cfg = a.guidance if a.guidance is not None else defaults.cfg
    strength = None
    if a.init_image is not None:
        strength = a.strength
        if strength is None:
            strength = 1.0 if a.mask is not None else (
                defaults.strength if defaults is not None else 0.6)
    return steps, cfg, strength


def cmd_generate(argv: list[str], prog: str = "gmlx image generate") -> int:
    a = _generate_parser(prog).parse_args(argv)
    try:
        return _generate(a)
    except _Usage as e:
        print(f"{prog}: error: {e}", file=sys.stderr)
        return 2


def _generate(a) -> int:
    import numpy as np

    from gmlx import __version__
    from gmlx.image import api
    from gmlx.tune.lora import probe_writable

    from . import image_preflight

    prompt = _prompt(a)
    if a.mask is not None and a.init_image is None:
        raise _Usage("--mask needs --init-image")
    if a.strength is not None and a.init_image is None:
        raise _Usage("--strength needs --init-image")
    _check_files(a)
    _vet_adapters(a)
    log = (lambda *x: print(*x, file=sys.stderr)) if a.json else print

    ident = image_preflight.identify_dit(a.model)
    _vet_adapters(a, ident)
    fam = ident.family
    defaults = api.sampler_defaults(fam.name, ident.variant)
    steps, cfg, strength = _settings(a, ident, defaults)
    stride = _stride(fam)
    init = None
    if a.init_image is not None:
        init = read_image(a.init_image, a.size, stride)
        width, height = init.shape[1], init.shape[0]
    else:
        width, height = a.size or (defaults.size if defaults else (1024, 1024))
    if width % stride or height % stride:
        raise _Usage(f"--size {width}x{height}: width and height must be "
                     f"multiples of {stride}")
    mask = read_mask(a.mask, (width, height)) if a.mask else None
    if cfg == 1.0 and a.negative_prompt:
        log("warning: --negative-prompt is unused, since --guidance 1 turns "
            "guidance off")

    seed = a.seed
    if seed is None:
        seed = secrets.randbelow(1 << 32)
        log(f"[image] seed {seed}")
    seeds = [seed + i for i in range(a.count)]
    fmt = _format(a)
    alpha = _check_background(a, fam, fmt)
    paths = output_paths(a.output, seeds, fmt, time.strftime("%Y%m%d-%H%M%S"))
    for path in paths + ([a.save_latent] if a.save_latent else []):
        err = probe_writable(path)
        if err is not None:
            raise OSError(f"cannot write {path}: {err}")

    _check_release(a, image_preflight.check(
        a.model, a.encoder, a.vae, height=height, width=width, cfg=cfg,
        adapters=_adapter_bytes(a), init_image=init is not None))
    pipe = api.load(a.model, text_encoder=a.encoder, vae=a.vae, log=log)
    table, adapters = _load_adapters(a, pipe, log)
    t0 = time.perf_counter()
    cond = api.encode_text(pipe, prompt)
    neg = None if cfg == 1.0 else api.encode_text(pipe, a.negative_prompt
                                                   or "")
    encode_s = time.perf_counter() - t0
    plan = api.plan_request(pipe, height=height, width=width, cfg=cfg,
                            conds=[cond] if neg is None else [cond, neg],
                            patch_table=table, init_image=init is not None)
    _check_release(a, plan)
    release = a.encoder_release == "always" or (
        a.encoder_release == "auto" and plan.release_encoder)
    if release:
        pipe.encoder.release()
    init_latents = None
    if init is not None:
        init_latents = api.vae_encode(pipe, init[None], plan=plan)

    files = {"dit": os.path.basename(a.model),
             "encoder": os.path.basename(a.encoder),
             "vae": os.path.basename(a.vae)}
    # scheduler None is the variant's own schedule, as in the bench record.
    settings = {"prompt": prompt, "seed": seed, "steps": steps,
                "size": f"{width}x{height}", "sampler": a.sampler,
                "scheduler": None, "guidance": cfg, "count": a.count,
                "adapters": adapters}
    if cfg != 1.0:
        settings["negative_prompt"] = a.negative_prompt or ""
    if alpha:
        settings["background"] = "transparent"
    if init is not None:
        settings.update(init_image=os.path.basename(a.init_image),
                        strength=strength)
        if mask is not None:
            settings["mask"] = os.path.basename(a.mask)
    images, latents = [], []
    for path, s in zip(paths, seeds):
        t0 = time.perf_counter()
        x = api.sample(pipe, cond, neg=neg, cfg=cfg, height=height,
                       width=width, steps=steps, seed=s, sampler=a.sampler,
                       init_latents=init_latents,
                       denoise=1.0 if strength is None else strength,
                       mask=mask, patch_table=table, plan=plan)
        latents.append(np.array(x))
        pixels = to_uint8(api.vae_decode(pipe, x, alpha=alpha, plan=plan))
        meta = None
        if fmt == "png" and not a.no_metadata:
            meta = {**settings, "seed": s, "model": files["dit"],
                    "files": files, "gmlx": __version__}
            meta.pop("count")
        write_image(path, pixels, fmt, a.quality, meta)
        secs = time.perf_counter() - t0
        images.append({"path": path, "seed": s, "seconds": round(secs, 3)})
        log(f"[image] wrote {path} (seed {s}, {secs:.2f} s)")
    if a.save_latent:
        np.savez(a.save_latent, latents=np.concatenate(latents, axis=0),
                 seeds=np.array(seeds, np.int64))
    if a.json:
        print(json.dumps({
            "schema": "gmlx.image.generate/1", "images": images,
            "settings": settings,
            "model": {"family": fam.name, "variant": pipe.variant, **files},
            "load_s": {k: round(v, 3) for k, v in pipe.load_seconds.items()},
            "encode_s": round(encode_s, 3), "latents": a.save_latent,
            "encoder_released": release, "levers": list(plan.levers)}))
    return 0


BENCH_SCHEMA = "gmlx.image.bench/1"
BENCH_PROMPT = "a red bicycle leaning on a brick wall, morning light"


def _sizes(text: str) -> list:
    return [_size(t) for t in text.split(",") if t.strip()] or [_size(text)]


def _bench_parser(prog: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description="Time whole images from a DiT GGUF with its text "
                    "encoder and VAE: load, encode, each step, decode and "
                    "the whole image, with the peak of MLX memory. Each run "
                    "starts with empty conditioning caches. Prints a "
                    "summary per size, or JSON records of schema "
                    f"{BENCH_SCHEMA}.")
    p.add_argument("model", metavar="MODEL", help="The DiT GGUF.")
    p.add_argument("--encoder", default=None, metavar="GGUF",
                   help="The text encoder GGUF.")
    p.add_argument("--vae", default=None, metavar="PATH",
                   help="The VAE, a GGUF or safetensors file.")
    _add_adapter_flag(p)
    p.add_argument("--prompt", default=BENCH_PROMPT,
                   help="The prompt (default a fixed bench prompt).")
    p.add_argument("--sizes", type=_sizes, default=None, metavar="LIST",
                   help="Comma-separated sizes such as 1024x1024,2048x2048 "
                        "(default: the variant's size).")
    p.add_argument("--steps", type=_positive, default=None, metavar="N",
                   help="Sampling steps (default: the variant's).")
    p.add_argument("--guidance", type=float, default=None, metavar="CFG",
                   help="Classifier-free guidance (default: the variant's).")
    p.add_argument("--sampler", choices=_SAMPLERS, default="euler",
                   help="The sampler (default euler).")
    p.add_argument("--runs", type=_positive, default=2, metavar="N",
                   help="Timed runs per size (default 2).")
    p.add_argument("--warmup", type=_seed, default=1, metavar="N",
                   help="Untimed runs per size before the timed ones "
                        "(default 1).")
    p.add_argument("--seed", type=_seed, default=0, metavar="N",
                   help="Seed of every run (default 0).")
    p.add_argument("--label", default=None, metavar="TEXT",
                   help="Record this label with the results.")
    p.add_argument("--json", action="store_true",
                   help="Print one JSON record per size on stdout.")
    p.add_argument("--json-out", default=None, metavar="FILE",
                   help="Write one JSON record per size to this file.")
    return p


def _median(xs: list) -> float:
    import statistics

    return float(statistics.median(xs))


def _machine() -> dict:
    import platform

    from gmlx.device import metal_info

    info = metal_info() or {}
    return {"chip": info.get("device_name") or platform.machine(),
            "ram_bytes": info.get("memory_size"),
            "os": platform.platform()}


def _quant(ident) -> str | None:
    from collections import Counter

    codecs = Counter(t.codec for t in ident.view.tensors.values()
                     if len(t.shape) == 2)
    return codecs.most_common(1)[0][0] if codecs else None


def _file_bytes(path: str) -> int:
    from gmlx.load.preflight import find_split_shards

    return sum(os.path.getsize(p) for p in find_split_shards(path))


def bench_run(pipe, prompt: str, *, height: int, width: int, steps: int,
              cfg: float, sampler: str, seed: int, patch_table=None,
              clock=time.perf_counter):
    """One timed image from empty conditioning caches: encode, every step,
    decode and the whole image in seconds, with MLX's peak memory. The
    pairs and built tables of ``patch_table`` stay loaded between runs,
    as the weights do."""
    import mlx.core as mx

    from gmlx.image import api

    for cache in (pipe.conditionings, pipe.captions, pipe.modulations):
        cache.clear()
    mx.reset_peak_memory()
    t0 = clock()
    cond = api.encode_text(pipe, prompt)
    neg = None if cfg == 1.0 else api.encode_text(pipe, "")
    mx.eval(cond.features, *(() if neg is None else (neg.features,)))
    t_enc = clock()
    # A bench keeps the encoder, which every run encodes with.
    plan = api.plan_request(pipe, height=height, width=width, cfg=cfg,
                            conds=[cond] if neg is None else [cond, neg],
                            patch_table=patch_table, keep_encoder=True)
    marks = [t_enc]
    x = api.sample(pipe, cond, neg=neg, cfg=cfg, height=height, width=width,
                   steps=steps, seed=seed, sampler=sampler,
                   patch_table=patch_table, plan=plan,
                   on_step=lambda i, _x: marks.append(clock()))
    t_dec = clock()
    img = api.vae_decode(pipe, x, plan=plan)
    mx.eval(img)
    t_end = clock()
    return {"encode_s": t_enc - t0,
            "step_s": [b - a for a, b in zip(marks, marks[1:])],
            "decode_s": t_end - t_dec, "image_s": t_end - t0,
            "peak_bytes": int(mx.get_peak_memory())}


def bench_summary(runs: list, load_s: dict) -> dict:
    firsts = [r["step_s"][0] for r in runs]
    rest = [s for r in runs for s in r["step_s"][1:]] or firsts
    return {"load_s": {**load_s, "total": sum(load_s.values())},
            "encode_s": _median([r["encode_s"] for r in runs]),
            "s_per_step": {"first": _median(firsts), "median": _median(rest),
                           "mean": sum(rest) / len(rest)},
            "decode_s": _median([r["decode_s"] for r in runs]),
            "s_per_image": _median([r["image_s"] for r in runs]),
            "peak_bytes": max(r["peak_bytes"] for r in runs)}


def cmd_bench(argv: list[str], prog: str = "gmlx image bench") -> int:
    a = _bench_parser(prog).parse_args(argv)
    try:
        return _bench(a)
    except _Usage as e:
        print(f"{prog}: error: {e}", file=sys.stderr)
        return 2


def _bench(a) -> int:
    import mlx.core as mx

    from gmlx import __version__
    from gmlx.image import api
    from gmlx.tune.lora import probe_writable

    from . import image_preflight

    a.init_image = a.mask = None
    _check_files(a)
    _vet_adapters(a)
    if a.json_out is not None:
        err = probe_writable(a.json_out)
        if err is not None:
            raise OSError(f"cannot write {a.json_out}: {err}")
    log = (lambda *x: print(*x, file=sys.stderr)) if a.json else print
    ident = image_preflight.identify_dit(a.model)
    _vet_adapters(a, ident)
    fam = ident.family
    defaults = api.sampler_defaults(fam.name, ident.variant)
    a.strength = None
    steps, cfg, _ = _settings(a, ident, defaults)
    sizes = a.sizes or [defaults.size if defaults else (1024, 1024)]
    stride = _stride(fam)
    for w, h in sizes:
        if w % stride or h % stride:
            raise _Usage(f"--sizes {w}x{h}: width and height must be "
                         f"multiples of {stride}")
    big = max(sizes, key=lambda s: s[0] * s[1])
    image_preflight.check(a.model, a.encoder, a.vae, height=big[1],
                          width=big[0], cfg=cfg, adapters=_adapter_bytes(a),
                          keep_encoder=True)
    pipe = api.load(a.model, text_encoder=a.encoder, vae=a.vae, log=log)
    table, adapters = _load_adapters(a, pipe, log)
    model = {
        "id": os.path.basename(a.model), "family": fam.name,
        "variant": pipe.variant,
        "dit": {"file": os.path.basename(a.model),
                "bytes": _file_bytes(a.model), "quant": _quant(ident)},
        "encoder": {"file": os.path.basename(a.encoder),
                    "bytes": _file_bytes(a.encoder), "shared": False},
        "vae": {"file": os.path.basename(a.vae),
                "bytes": os.path.getsize(a.vae),
                "dtype": str(getattr(pipe.vae, "dtype", "")).replace(
                    "mlx.core.", "")}}
    records = []
    for w, h in sizes:
        kw = dict(height=h, width=w, steps=steps, cfg=cfg,
                  sampler=a.sampler, seed=a.seed, patch_table=table)
        for _ in range(a.warmup):
            bench_run(pipe, a.prompt, **kw)
        runs = [bench_run(pipe, a.prompt, **kw) for _ in range(a.runs)]
        rec = {"schema": BENCH_SCHEMA, "gmlx_version": __version__,
               "mlx_version": mx.__version__, "label": a.label,
               "machine": _machine(), "model": model,
               "settings": {"width": w, "height": h, "steps": steps,
                            "sampler": a.sampler, "scheduler": None,
                            "guidance": cfg, "batch": 1,
                            "adapters": adapters,
                            "seed": a.seed, "prompt": a.prompt},
               "runs": runs, "summary": bench_summary(runs,
                                                      pipe.load_seconds),
               "transport": {"mode": "direct", "overhead_ms_median": None}}
        records.append(rec)
        sm = rec["summary"]
        log(f"[bench] {w}x{h} {steps} steps: step first "
            f"{sm['s_per_step']['first']:.3f} s, median "
            f"{sm['s_per_step']['median']:.3f} s; encode "
            f"{sm['encode_s']:.3f} s, decode {sm['decode_s']:.3f} s, image "
            f"{sm['s_per_image']:.2f} s; load "
            f"{sm['load_s']['total']:.1f} s; peak "
            f"{sm['peak_bytes'] / 1e9:.2f} GB")
    lines = "".join(json.dumps(r) + "\n" for r in records)
    if a.json:
        sys.stdout.write(lines)
    if a.json_out is not None:
        with open(a.json_out, "w", encoding="utf-8") as f:
            f.write(lines)
    return 0


def _pack_vae_parser(prog: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description="Pack a safetensors VAE into a VAE GGUF. Tensors keep "
                    "their official names, torch layout and, by default, "
                    "their dtype, so the file is lossless. The GGUF is read "
                    "back and every tensor compared bit for bit before it "
                    "replaces --out.")
    p.add_argument("src", metavar="SRC", help="The safetensors VAE.")
    p.add_argument("--out", required=True, metavar="FILE.gguf",
                   help="The GGUF to write.")
    p.add_argument("--dtype", choices=sorted(_PACK_DTYPES), default=None,
                   help="Store every tensor in this dtype (default: the "
                        "source dtype). A narrower dtype is not lossless, "
                        "and the file records that.")
    p.add_argument("--family", default=None, metavar="NAME",
                   help="The image family whose VAE this is (default: the "
                        "one family whose VAE the file fills).")
    p.add_argument("--source-repo", default="unknown", metavar="ID",
                   help="Record this repository as the source (default "
                        "unknown).")
    p.add_argument("--source-revision", default="unknown", metavar="REV",
                   help="Record this revision of the source repository "
                        "(default unknown).")
    p.add_argument("--source-config", default=None, metavar="FILE",
                   help="Record this VAE config.json (default: config.json "
                        "beside SRC when there is one).")
    p.add_argument("--force", action="store_true",
                   help="Replace --out when it exists.")
    p.add_argument("--json", action="store_true",
                   help="Print a JSON record of the packed file on stdout.")
    return p


def cmd_pack_vae(argv: list[str], prog: str = "gmlx image pack-vae") -> int:
    a = _pack_vae_parser(prog).parse_args(argv)
    from gmlx.image.vae import VAEError, pack_vae

    cfg_path = a.source_config
    if cfg_path is None:
        beside = os.path.join(os.path.dirname(os.path.abspath(a.src)),
                              "config.json")
        cfg_path = beside if os.path.isfile(beside) else None
    config_json = "{}"
    if cfg_path is not None:
        try:
            with open(cfg_path, encoding="utf-8") as f:
                config_json = f.read()
        except OSError as e:
            return _refuse(f"--source-config {cfg_path}: {e.strerror}")
    try:
        rep = pack_vae(a.src, a.out, source_repo=a.source_repo,
                       source_revision=a.source_revision,
                       config_json=config_json, family=a.family,
                       dtype=_PACK_DTYPES.get(a.dtype), overwrite=a.force)
    except VAEError as e:
        return _refuse(str(e))
    record = {"path": rep.path, "family": rep.family, "tensors": rep.tensors,
              "payload_bytes": rep.payload_bytes,
              "source_sha256": rep.source_sha256, "dtype": rep.dtype,
              "lossless": rep.lossless, "verified": True}
    if a.json:
        print(json.dumps(record))
    else:
        print(f"wrote {rep.path}: {rep.tensors} tensors, "
              f"{rep.payload_bytes / 1e6:.1f} MB, {rep.family} VAE, "
              f"verified bit for bit")
    if not rep.lossless:
        print(f"note: {a.dtype} is narrower than the source, so "
              f"{rep.path} is not lossless", file=sys.stderr)
    return 0


HANDLERS = {"generate": cmd_generate, "bench": cmd_bench,
            "pack-vae": cmd_pack_vae}


def cmd_image(argv: list[str], prog: str = "gmlx image") -> int:
    """Dispatch ``gmlx image <action> ...``."""
    if not argv or argv[0] in ("-h", "--help", "help"):
        _print_help(prog)
        return 0
    action, rest = argv[0], argv[1:]
    if action not in HANDLERS:
        print(f"error: unknown action {action!r}; one of "
              f"{', '.join(_ACTIONS)}", file=sys.stderr)
        return 2
    return HANDLERS[action](rest, prog=f"{prog} {action}")
