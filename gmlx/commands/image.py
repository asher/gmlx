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
    # The range the image routes take.
    if not 0.01 <= x <= 1.0:
        raise argparse.ArgumentTypeError(
            f"a strength from 0.01 to 1 is required, got {text!r}")
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
    p.add_argument("model", metavar="MODEL",
                   help="The DiT GGUF, or the id of an image_models entry in "
                        "the config, which brings its encoder and VAE.")
    p.add_argument("--config", default=None, metavar="FILE",
                   help="The config whose image_models MODEL names (default: "
                        "the usual config search).")
    p.add_argument("--variant", default=None, metavar="NAME",
                   help="The DiT's variant, such as turbo, for a file whose "
                        "weights match no known variant (default: from the "
                        "weights, or the entry's variant).")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--prompt", default=None, help="The prompt.")
    src.add_argument("--prompt-file", default=None, metavar="FILE",
                     help="Read the prompt from this file.")
    p.add_argument("--negative-prompt", default=None, metavar="TEXT",
                   help="The negative prompt, used when --guidance is not "
                        "1 (default empty).")
    up = p.add_mutually_exclusive_group()
    up.add_argument("--upsampler", default=None, metavar="GGUF",
                    help="The chat model that rewrites the prompt before it "
                         "is encoded: a GGUF, or a models: id of the config "
                         "(default: the entry's upsampler). A prompt with "
                         "--ref or --init-image runs as typed.")
    up.add_argument("--no-upsample", action="store_true",
                    help="Run the prompt as typed, without the upsampler.")
    p.add_argument("--upsample-prompt", default=None, metavar="FILE",
                   help="A text file whose system prompt the upsampler takes "
                        "in place of the family's (default: the entry's "
                        "upsample_prompt). Qwen-Image 2.1 ships none and "
                        "needs one.")
    p.add_argument("--size", type=_size, default=None, metavar="WxH",
                   help="Image width and height, multiples of 16, or of 32 "
                        "on Qwen-Image 2.1 (default: the init image's, else "
                        "on FLUX.2 klein and FLUX.3 klein the first "
                        "reference's, else the entry's defaults.size, else "
                        "the variant's). On FLUX.3 klein without it, the "
                        "image of one reference is cropped to that "
                        "reference's own aspect.")
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
    p.add_argument("--ref", action="append", default=None, metavar="PATH",
                   help="Edit from this reference image. Repeat the flag for "
                        "several, in the order the prompt names them. Needs "
                        "--encoder-mmproj, except on FLUX.2 klein and FLUX.3 "
                        "klein, whose encoders read the prompt alone.")
    p.add_argument("--ref-size", type=_positive, default=None, metavar="N",
                   help="Resize each reference to about N x N pixels at its "
                        "own aspect ratio (default 1024). FLUX.2 klein and "
                        "FLUX.3 klein size their references by their own "
                        "rules and take no --ref-size.")
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
    p.add_argument("--encoder-mmproj", default=None, metavar="GGUF",
                   help="The mmproj GGUF of the text encoder, which reads "
                        "the --ref images.")
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
        when = (" before the DiT loads"
                if getattr(plan, "staged_load", False) else "")
        raise _Usage("--encoder-release never: this request fits only with "
                     f"the text encoder released{when}; {plan.describe()}")


def _levers(plan, staged: bool) -> list:
    """The levers a run took, in the plan's order. A staged run plans its
    request after the encoder is gone, so that plan holds neither the
    release nor the staged load, which the run took all the same."""
    if not staged:
        return list(plan.levers)
    from gmlx.image.memfit import LEVERS

    took = {*plan.levers, "release the encoder", "staged load"}
    return [name for name in LEVERS if name in took]


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


def open_image(flag: str, path: str):
    """The image file ``path`` given to ``flag``, read whole and turned
    upright by its EXIF orientation, as diffusers' load_image reads an
    image. A file PIL cannot read is refused by the flag and the path."""
    from PIL import Image, ImageOps

    try:
        with Image.open(os.path.expanduser(path)) as im:
            im.load()
            return ImageOps.exif_transpose(im)
    except OSError as e:
        raise OSError(f"{flag} {path}: cannot read it as an image: {e}") \
            from None


def read_image(path: str, size: tuple | None, stride: int,
               flag: str = "--init-image"):
    """An image file as (H, W, 3) float32 in [0, 1] at ``size`` (width,
    height), resized with Lanczos. Without ``size`` the image keeps its
    own, cut down to multiples of ``stride``."""
    import numpy as np
    from PIL import Image

    im = open_image(flag, path).convert("RGB")
    if size is None:
        size = (im.width // stride * stride, im.height // stride * stride)
        if min(size) <= 0:
            raise ValueError(f"{path} is smaller than {stride} pixels")
    if im.size != tuple(size):
        im = im.resize(tuple(size), Image.Resampling.LANCZOS)
    return np.asarray(im, np.float32) / 255.0


def read_mask(path: str, size: tuple, flag: str = "--mask"):
    """A mask file as (H, W) float32, 1 where the image changes: white
    pixels, binarized at one half as diffusers does."""
    import numpy as np
    from PIL import Image

    im = open_image(flag, path).convert("L")
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


def _from_config(a) -> None:
    """Fill the DiT, encoder, VAE, vision tower, variant and image size
    from the image_models entry that MODEL names, when MODEL is no file.
    Flags given win."""
    import gmlx.config as cfgmod

    if os.path.isfile(os.path.expanduser(a.model)):
        return
    path = (os.path.expanduser(a.config) if getattr(a, "config", None)
            else next((str(c) for c in cfgmod.default_config_paths()
                       if os.path.exists(c)), None))
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"MODEL {a.model}: no such file")
    try:
        cfg = cfgmod.load_config(path)
        if a.model not in cfg.image_models:
            known = ", ".join(sorted(cfg.image_models)) or "none"
            raise FileNotFoundError(f"MODEL {a.model}: no such file, and no "
                                    f"image model of that id in {path} (it "
                                    f"has {known})")
        rm = cfgmod.resolve_image_model(a.model, cfg)
        encoder = rm.encoder
        if rm.encoder_model is not None:
            encoder = cfgmod.resolve_path(cfg.models[rm.encoder_model].path,
                                          cfg.model_dirs)
    except cfgmod.ConfigError as e:
        raise FileNotFoundError(f"MODEL {a.model}: {e}") from None
    a.model = rm.path
    a.encoder = a.encoder or encoder
    a.vae = a.vae or rm.vae
    a.variant = a.variant or rm.variant
    if rm.defaults.get("size"):
        a.entry_size = _size(rm.defaults["size"])
    if getattr(a, "ref", None) and a.encoder_mmproj is None:
        a.encoder_mmproj = rm.encoder_mmproj
    if rm.adapters:
        a.adapter = list(rm.adapters) + list(a.adapter or ())
    if hasattr(a, "upsampler") and a.upsampler is None:
        if not rm.upsample:
            a.no_upsample = True
        elif rm.upsampler:
            a.upsampler = rm.upsampler
    if getattr(a, "upsample_prompt", "") is None and rm.upsample_prompt:
        a.upsample_prompt = rm.upsample_prompt


def _upsampler_path(a) -> str:
    """The GGUF of --upsampler: the file, or the models: entry of that id
    in the config."""
    import gmlx.config as cfgmod

    if os.path.isfile(os.path.expanduser(a.upsampler)):
        return os.path.expanduser(a.upsampler)
    path = (os.path.expanduser(a.config) if getattr(a, "config", None)
            else next((str(c) for c in cfgmod.default_config_paths()
                       if os.path.exists(c)), None))
    where = f"in {path}" if path and os.path.exists(path) else "(no config)"
    if path is not None and os.path.exists(path):
        try:
            cfg = cfgmod.load_config(path)
            if a.upsampler in cfg.models:
                return cfgmod.resolve_path(cfg.models[a.upsampler].path,
                                           cfg.model_dirs)
        except cfgmod.ConfigError as e:
            raise FileNotFoundError(f"--upsampler {a.upsampler}: {e}") from None
    raise FileNotFoundError(f"--upsampler {a.upsampler}: no such file, and "
                            f"no models: id of that name {where}")


def _upsample(a, family, prompt: str, refs, log) -> str | None:
    """The caption the upsampler writes for ``prompt``, or None when it
    runs as typed, as a prompt with an image input does."""
    if family.upsample is None:
        if a.upsampler is not None:
            log(f"note: the upsampler is unused, since {family.title} "
                "takes no prompt upsampling, and is not loaded")
        return None
    if a.no_upsample or not prompt.strip():
        return None
    if refs or a.init_image is not None:
        if a.upsampler is not None:
            flag = "--ref" if refs else "--init-image"
            log(f"note: a prompt with {flag} runs as typed, and the "
                "upsampler is not loaded")
        return None
    if a.upsampler is None:
        if family.upsample.expected:
            log(f"warning: {family.title} expects its prompts upsampled; set "
                f"--upsampler to a {family.upsample.suggested} GGUF or its "
                "models: id, or upsampler on the image_models entry, or pass "
                "--no-upsample")
        return None
    from gmlx.image.upsample import UpsampleError, system_prompt

    try:
        system = system_prompt(family, a.upsample_prompt)
    except UpsampleError as e:
        raise _Usage(str(e)) from None
    a.upsampler = _upsampler_path(a)
    return run_upsampler(a.upsampler, family, prompt, log, quiet=a.json,
                         system=system)


def run_upsampler(path: str, family, prompt: str, log,
                  quiet: bool = False, system: str | None = None) -> str:
    """Load the chat model at ``path``, write the caption of ``prompt``,
    and release the model before returning, under ``system``, by default
    the family's shipped system prompt. ``quiet`` keeps the load's lines
    off stdout."""
    import gc

    import mlx.core as mx

    import gmlx.load.loadlog as loadlog
    from gmlx.gen.generation import generate
    from gmlx.gen.thinking_budget import prompt_open_think_tag
    from gmlx.image.upsample import UpsampleError, system_prompt, upsample
    from gmlx.load.loader import load_model
    from gmlx.serve.capacity import working_set_bytes
    from gmlx.serve.memory import resolve_cache_limit
    from gmlx.tui.reasoning import map_thinking_controls

    from .tool_preflight import check_or_exit

    if system is None:
        system = system_prompt(family)
    # About 3 characters a token, which overcounts English.
    ctx = (len(system) + len(prompt)) // 3 + (
        family.upsample.max_tokens)
    def stdout():
        return (contextlib.redirect_stdout(sys.stderr) if quiet
                else contextlib.nullcontext())

    with stdout():
        check_or_exit(path, ctx_tokens=ctx)
    limit, _ = resolve_cache_limit(None, [path], working_set_bytes() or 0.0)
    prev = None if limit is None else mx.set_cache_limit(limit)
    name = os.path.basename(path)
    t0 = time.perf_counter()
    try:
        with stdout(), loadlog.load_ui(False, path):
            model, _config, tok = load_model(path)
        template = getattr(tok, "chat_template", None)
        if template is None:
            raise _Usage(f"--upsampler {name}: the model has no chat "
                         "template, so it cannot take the upsampler's "
                         "system prompt")

        def chat(req):
            kwargs = map_thinking_controls(
                {}, req.thinking, None, template,
                warn=lambda msg: log(f"[upsample] {msg}"))
            reply = generate(model, tok, req.user, system_prompt=req.system,
                             max_tokens=req.max_tokens, temp=req.temperature,
                             presence_penalty=req.presence_penalty,
                             template_kwargs=kwargs, verbose=False)
            # A template that opens the thinking block in the prompt leaves
            # its tag out of the reply, which the parser reads it by.
            rendered = tok.apply_chat_template(
                list(req.messages), tokenize=False, add_generation_prompt=True,
                **kwargs)
            return (prompt_open_think_tag(rendered, tokenizer=tok) or "") + (
                reply)

        try:
            caption = upsample(family, prompt, chat, name=f"--upsampler {name}",
                               system=system)
        except UpsampleError as e:
            raise _Usage(str(e)) from None
    finally:
        model = tok = None
        gc.collect()
        mx.clear_cache()
        if prev is not None:
            mx.set_cache_limit(prev)
    log(f"[image] prompt upsampled by {name} in "
        f"{time.perf_counter() - t0:.1f} s, {len(caption.split())} words")
    return caption


def _check_files(a) -> None:
    _from_config(a)
    if not os.path.isfile(os.path.expanduser(a.model)):
        raise FileNotFoundError(f"MODEL {a.model}: no such file")
    if a.encoder is None or a.vae is None:
        raise _Usage("--encoder and --vae are required with a DiT path")
    if a.ref_size is not None and not a.ref:
        raise _Usage("--ref-size needs --ref")
    for flag, path in (("--encoder", a.encoder), ("--vae", a.vae),
                       ("--encoder-mmproj", a.encoder_mmproj),
                       ("--init-image", a.init_image), ("--mask", a.mask),
                       *(("--ref", r) for r in a.ref or ())):
        if path is not None and not os.path.isfile(os.path.expanduser(path)):
            raise FileNotFoundError(f"{flag} {path}: no such file")
    for path, _ in a.adapter or ():
        if not os.path.isfile(os.path.expanduser(path)):
            raise FileNotFoundError(f"--adapter {path}: no such file")


def _references(a, family) -> tuple:
    """The ``--ref`` images as the family runs them, in order. A family
    that takes none, a count over its cap and a size outside what it takes
    are usage errors. A family that prepares the references of a request
    together gives them with ``crop``, and sizes them by its own rule."""
    if not a.ref:
        return ()
    edit = family.edit
    if edit is None:
        raise _Usage(f"--ref: {family.title} takes no reference images")
    if len(a.ref) > edit.max_refs:
        raise _Usage(f"--ref: {family.title} takes at most {edit.max_refs} "
                     f"reference images, and the request has {len(a.ref)}")
    if a.encoder_mmproj is None and not edit.text_only:
        raise _Usage("--ref needs --encoder-mmproj with a DiT path")
    if edit.prepare_all is not None:
        if a.ref_size is not None:
            raise _Usage(f"--ref-size: {family.title} sizes its references "
                         "by its own rule")
        images = [open_image("--ref", path) for path in a.ref]
        try:
            return edit.prepare_all(images)
        except ValueError as e:
            i = getattr(e, "index", None)
            where = "--ref" if i is None else f"--ref {a.ref[i]}"
            raise _Usage(f"{where}: {e}") from None
    low, high = math.isqrt(edit.min_area - 1) + 1, math.isqrt(edit.area)
    side = high if a.ref_size is None else a.ref_size
    if not low <= side <= high:
        raise _Usage(f"--ref-size {side}: {family.title} takes a reference "
                     f"size from {low} to {high}")
    out = []
    for path in a.ref:
        im = open_image("--ref", path)
        try:
            out.append(edit.prepare(im, side * side))
        except ValueError as e:
            least = getattr(e, "side", None)
            if least is not None and least <= high:
                raise _Usage(f"--ref {path}: {e} (--ref-size {least})") \
                    from None
            raise _Usage(f"--ref {path}: {e}") from None
    return tuple(out)


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


def _read_adapters(a, pipe, log) -> list:
    """Load the ``--adapter`` files into the pipeline and apply their text
    encoder halves. Returns one (adapter, strength) row per file."""
    if not a.adapter:
        return []
    from gmlx.image import lora

    rows = []
    for path, strength in a.adapter:
        ad = lora.load_adapter(pipe, path)
        for w in ad.warnings:
            log(f"warning: {w}")
        rows.append((ad, strength))
    lora.apply_text_adapters(pipe, rows)
    return rows


def _adapter_table(pipe, rows, log) -> tuple:
    """The patch table id of the adapter rows, or None without adapters,
    and one record per adapter for the metadata: ``sha256`` is the digest
    of the file, and ``rows_sha256`` names the pairs gmlx placed and the
    modules it placed them on."""
    if not rows:
        return None, []
    from gmlx.image import api, lora

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


def _load_adapters(a, pipe, log) -> tuple:
    """``_read_adapters`` and ``_adapter_table`` on a loaded pipeline."""
    return _adapter_table(pipe, _read_adapters(a, pipe, log), log)


def _settings(a, ident, defaults):
    """Steps, cfg and strength from the flags and the variant's defaults."""
    why = ident.family.variant_needed(ident.variant)
    if why:
        raise ValueError(why)
    if defaults is None and (a.steps is None or a.guidance is None):
        raise ValueError(
            f"{a.model} matches no known {ident.family.title} variant; pass "
            f"--steps and --guidance")
    steps = a.steps if a.steps is not None else defaults.steps
    from gmlx.image.api import check_sampling, check_steps

    check_steps(ident.family, ident.variant, steps)
    cfg = a.guidance if a.guidance is not None else defaults.cfg
    # Before any file loads.
    check_sampling(ident.family, ident.variant,
                   bool(ident.config.get("flow_map")), cfg, a.sampler)
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

    ident = image_preflight.identify_dit(a.model, variant=a.variant)
    _vet_adapters(a, ident)
    fam = ident.family
    defaults = api.sampler_defaults(fam.name, ident.variant)
    steps, cfg, strength = _settings(a, ident, defaults)
    stride = _stride(fam)
    refs = _references(a, fam)
    edit = fam.edit
    # A family that sizes a request by its references names that size,
    # scaled into the image limits.
    ref_size = None
    if a.size is None and a.init_image is None:
        from gmlx.image.registry import edit_size

        ref_size = edit_size(fam, refs, stride)
    init = crop = None
    if a.init_image is not None:
        init = read_image(a.init_image, a.size, stride)
        width, height = init.shape[1], init.shape[0]
    elif a.size is None and ref_size is not None:
        width, height = ref_size
        # The output of a padded reference is cropped back to its box.
        crop = getattr(refs, "crop", None)
    else:
        width, height = a.size or getattr(a, "entry_size", None) or (
            defaults.size if defaults else (1024, 1024))
    if width % stride or height % stride:
        raise _Usage(f"--size {width}x{height}: width and height must be "
                     f"multiples of {stride}")
    mask = read_mask(a.mask, (width, height)) if a.mask else None
    text_only = edit is not None and edit.text_only
    revised = _upsample(a, fam, prompt, refs, log)
    text = revised or prompt
    mmproj = a.encoder_mmproj if refs and not text_only else None
    if a.encoder_mmproj is not None and not refs:
        log("note: --encoder-mmproj is unused without --ref, and is not "
            "loaded")
    elif a.encoder_mmproj is not None and mmproj is None:
        log(f"note: --encoder-mmproj is unused, since the {fam.title} "
            "encoder reads the prompt alone, and is not loaded")
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

    before = image_preflight.check(
        a.model, a.encoder, a.vae, height=height, width=width, cfg=cfg,
        adapters=_adapter_bytes(a), init_image=init is not None,
        mmproj=mmproj, refs=[(r.height, r.width) for r in refs],
        variant=a.variant, prompts=() if refs and not text_only else (
            [text] + ([] if cfg == 1.0 else [a.negative_prompt or ""])))
    _check_release(a, before)
    staged = bool(getattr(before, "staged_load", False))

    def encode(pipe):
        # The VAE encodes the references after the encoder's release.
        t0 = time.perf_counter()
        cond = api.encode_text(pipe, text, images=refs, ref_latents=False)
        neg = None if cfg == 1.0 else api.encode_text(
            pipe, a.negative_prompt or "", images=refs, ref_latents=False)
        return cond, neg, time.perf_counter() - t0

    if staged:
        # The plan fits only with the encoder gone before the DiT loads.
        log("[image] loading in stages: the text encoder encodes and is "
            "released before the DiT loads")
        pipe, (rows, (cond, neg, encode_s)) = api.load_staged(
            a.model, text_encoder=a.encoder, vae=a.vae, mmproj=mmproj,
            variant=a.variant, log=log,
            encode=lambda pipe: (_read_adapters(a, pipe, log), encode(pipe)))
        table, adapters = _adapter_table(pipe, rows, log)
    else:
        pipe = api.load(a.model, text_encoder=a.encoder, vae=a.vae,
                        mmproj=mmproj, variant=a.variant, log=log)
        table, adapters = _load_adapters(a, pipe, log)
        cond, neg, encode_s = encode(pipe)
    plan = api.plan_request(pipe, height=height, width=width, cfg=cfg,
                            conds=[cond] if neg is None else [cond, neg],
                            patch_table=table, init_image=init is not None)
    _check_release(a, plan)
    release = staged or a.encoder_release == "always" or (
        a.encoder_release == "auto" and plan.release_encoder)
    if release:
        pipe.encoder.release()
    init_latents = None
    if init is not None:
        init_latents = api.vae_encode(pipe, init[None], plan=plan)
    if refs:
        cond = api.encode_references(pipe, cond, plan=plan)
        if neg is not None:
            neg = api.encode_references(pipe, neg, plan=plan)

    files = {"dit": os.path.basename(a.model),
             "encoder": os.path.basename(a.encoder),
             "vae": os.path.basename(a.vae)}
    if mmproj is not None:
        files["mmproj"] = os.path.basename(mmproj)
    if revised is not None:
        files["upsampler"] = os.path.basename(a.upsampler)
    # scheduler None is the variant's own schedule, as in the bench record.
    settings = {"prompt": prompt, "seed": seed, "steps": steps,
                "size": f"{width}x{height}", "sampler": a.sampler,
                "scheduler": None, "guidance": cfg, "count": a.count,
                "adapters": adapters}
    if revised is not None:
        settings["revised_prompt"] = revised
    if cfg != 1.0:
        settings["negative_prompt"] = a.negative_prompt or ""
    if alpha:
        settings["background"] = "transparent"
    if refs:
        settings["refs"] = [
            {"file": os.path.basename(p), "size": f"{r.width}x{r.height}",
             "sha256": _file_sha256(p)} for p, r in zip(a.ref, refs)]
    if crop is not None:
        # The size written, the size sampled, and the (left, top, right,
        # bottom) box of the written image in the sampled one.
        left, top, right, bottom = (int(v) for v in crop)
        settings.update(size=f"{right - left}x{bottom - top}",
                        canvas=f"{width}x{height}",
                        crop=[left, top, right, bottom])
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
        if crop is not None:
            left, top, right, bottom = crop
            pixels = np.ascontiguousarray(pixels[top:bottom, left:right])
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
            "encoder_released": release, "levers": _levers(plan, staged)}))
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
    p.add_argument("model", metavar="MODEL",
                   help="The DiT GGUF, or the id of an image_models entry in "
                        "the config, which brings its encoder and VAE.")
    p.add_argument("--config", default=None, metavar="FILE",
                   help="The config whose image_models MODEL names (default: "
                        "the usual config search).")
    p.add_argument("--variant", default=None, metavar="NAME",
                   help="The DiT's variant, such as turbo, for a file whose "
                        "weights match no known variant (default: from the "
                        "weights, or the entry's variant).")
    p.add_argument("--encoder", default=None, metavar="GGUF",
                   help="The text encoder GGUF.")
    p.add_argument("--vae", default=None, metavar="PATH",
                   help="The VAE, a GGUF or safetensors file.")
    _add_adapter_flag(p)
    p.add_argument("--prompt", default=BENCH_PROMPT,
                   help="The prompt (default a fixed bench prompt).")
    p.add_argument("--ref", action="append", default=None, metavar="PATH",
                   help="Time an edit from this reference image. Repeat the "
                        "flag for several. On FLUX.2 klein and FLUX.3 klein, "
                        "whose encoders read the prompt alone.")
    p.add_argument("--sizes", type=_sizes, default=None, metavar="LIST",
                   help="Comma-separated sizes such as 1024x1024,2048x2048 "
                        "(default: the entry's defaults.size, else the "
                        "variant's size, or the size the references give an "
                        "edit).")
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
    p.add_argument("--via-steps", action="store_true",
                   help="Run each step through the image step routes on a "
                        "loopback port of this process, and record the "
                        "transport time per step and whether the latents "
                        "equal those of a direct run.")
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
              refs=(), clock=time.perf_counter):
    """One timed image from empty conditioning caches: encode, every step,
    decode and the whole image in seconds, with MLX's peak memory. An edit
    encodes its prepared ``refs`` with the VAE after the prompt, and the
    encode time counts both. The pairs and built tables of
    ``patch_table`` stay loaded between runs, as the weights do."""
    import mlx.core as mx

    from gmlx.image import api

    for cache in (pipe.conditionings, pipe.captions, pipe.modulations,
                  pipe.reference_latents):
        cache.clear()
    mx.reset_peak_memory()
    t0 = clock()
    cond = api.encode_text(pipe, prompt, images=refs, ref_latents=False)
    neg = None if cfg == 1.0 else api.encode_text(pipe, "", images=refs,
                                                  ref_latents=False)
    mx.eval(cond.features, *(() if neg is None else (neg.features,)))
    text_s = clock() - t0
    # A bench keeps the encoder, which every run encodes with.
    plan = api.plan_request(pipe, height=height, width=width, cfg=cfg,
                            conds=[cond] if neg is None else [cond, neg],
                            patch_table=patch_table, keep_encoder=True)
    t_ref = clock()
    if refs:
        cond = api.encode_references(pipe, cond, plan=plan)
        neg = None if neg is None else api.encode_references(pipe, neg,
                                                             plan=plan)
    t_enc = clock()
    marks = [t_enc]
    x = api.sample(pipe, cond, neg=neg, cfg=cfg, height=height, width=width,
                   steps=steps, seed=seed, sampler=sampler,
                   patch_table=patch_table, plan=plan,
                   on_step=lambda i, _x: marks.append(clock()))
    t_dec = clock()
    img = api.vae_decode(pipe, x, plan=plan)
    mx.eval(img)
    t_end = clock()
    return {"encode_s": text_s + t_enc - t_ref,
            "step_s": [b - a for a, b in zip(marks, marks[1:])],
            "decode_s": t_end - t_dec, "image_s": t_end - t0,
            "peak_bytes": int(mx.get_peak_memory())}


def bench_run_steps(pipe, client, model: str, prompt: str, *, height: int,
                    width: int, steps: int, cfg: float, sampler: str,
                    seed: int, clock=time.perf_counter):
    """``bench_run`` with every step a request of the step contract to
    ``client``. The prompt is encoded in process and uploaded by hash, the
    client moves the latents between steps with the bench's sampler, and
    the decode runs in process. Returns the run record, with each step's
    transport time, and the final latents."""
    import mlx.core as mx
    import numpy as np

    from gmlx.image import api
    from gmlx.image.samplers import make_sampler, noise_from_seed
    from gmlx.serve.image_steps_loopback import work

    def start():
        for cache in (pipe.conditionings, pipe.captions, pipe.modulations):
            cache.clear()
        mx.reset_peak_memory()

    def encode():
        cond = api.encode_text(pipe, prompt)
        neg = None if cfg == 1.0 else api.encode_text(pipe, "")
        feats = [c.features.astype(mx.float32)[None]
                 for c in (cond, neg) if c is not None]
        mx.eval(*feats)
        return [np.array(f) for f in feats]

    # The loop rounds where the family's reference loop does, as sample
    # does: the noise, the guided velocity and the Euler product, and the
    # latents held between steps.
    rules = pipe.family.loop
    act = (pipe.dit.act_dtype
           if rules is not None and rules.act_update else None)
    held = (pipe.dit.act_dtype
            if rules is not None and rules.act_latents else mx.float32)

    def schedule():
        s = api.image_sigmas(pipe, steps, height, width)
        shape = (1, pipe.dit.cfg.in_channels, height // pipe.vae_scale,
                 width // pipe.vae_scale)
        noise = mx.array(noise_from_seed(seed, shape))
        if rules is not None and rules.noise is not None:
            noise = noise.astype(getattr(mx, rules.noise)).astype(mx.float32)
        smp = (make_sampler(sampler, s, product=act)
               if act is not None and sampler == "euler"
               else make_sampler(sampler, s))
        return np.asarray(s, np.float32), smp, np.array(noise)

    work(start)
    t0 = clock()
    feats = work(encode)
    rows = [[{"cond": client.put(f), "num_tokens": int(f.shape[1])}]
            for f in feats]
    t_enc = clock()
    sigmas, smp, noise = work(schedule)

    def update(x, v, i):
        def run():
            y = smp.step(mx.array(x), mx.array(v), i)
            return np.array(y.astype(held).astype(mx.float32))
        return work(run)

    def combine(pos, neg, g):
        return work(lambda: np.array(api.guide(mx.array(pos), mx.array(neg),
                                               g, act)))

    marks = [t_enc]
    first = len(client.records)
    run = client.begin(model)
    try:
        x = client.sample(model, noise, sigmas, rows[0], cfg=cfg,
                          neg_rows=rows[1] if len(rows) > 1 else None,
                          run=run, update=update, combine=combine,
                          on_step=lambda i, _x: marks.append(clock()))
    finally:
        client.end(run)
    t_dec = clock()

    def decode():
        img = api.vae_decode(pipe, mx.array(x))
        mx.eval(img)
        return int(mx.get_peak_memory())

    peak = work(decode)
    t_end = clock()
    return {"encode_s": t_enc - t0,
            "step_s": [b - a for a, b in zip(marks, marks[1:])],
            "decode_s": t_end - t_dec, "image_s": t_end - t0,
            "peak_bytes": peak,
            "transport_ms": [r.overhead_ms for r in client.records[first:]]
            }, x


def _direct_latents(pipe, prompt: str, *, height: int, width: int,
                    steps: int, cfg: float, sampler: str, seed: int):
    import mlx.core as mx
    import numpy as np

    from gmlx.image import api

    cond = api.encode_text(pipe, prompt)
    neg = None if cfg == 1.0 else api.encode_text(pipe, "")
    x = api.sample(pipe, cond, neg=neg, cfg=cfg, height=height, width=width,
                   steps=steps, seed=seed, sampler=sampler)
    mx.eval(x)
    return np.array(x)


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
    from gmlx.image import api
    from gmlx.tune.lora import probe_writable

    from . import image_preflight

    if a.via_steps and a.adapter:
        raise _Usage("--via-steps times the base model; drop --adapter")
    if a.via_steps and a.ref:
        raise _Usage("--via-steps times text to image; drop --ref")
    a.init_image = a.mask = None
    a.ref_size = a.encoder_mmproj = None
    _check_files(a)
    _vet_adapters(a)
    if a.json_out is not None:
        err = probe_writable(a.json_out)
        if err is not None:
            raise OSError(f"cannot write {a.json_out}: {err}")
    log = (lambda *x: print(*x, file=sys.stderr)) if a.json else print
    ident = image_preflight.identify_dit(a.model, variant=a.variant)
    _vet_adapters(a, ident)
    fam = ident.family
    defaults = api.sampler_defaults(fam.name, ident.variant)
    a.strength = None
    steps, cfg, _ = _settings(a, ident, defaults)
    stride = _stride(fam)
    if a.ref and fam.edit is not None and not fam.edit.text_only:
        raise _Usage(f"--ref: the {fam.title} encoder reads the references "
                     "with its vision tower, which bench does not load")
    refs = _references(a, fam)
    if refs and not a.sizes:
        from gmlx.image.registry import edit_size

        a.sizes = [s for s in [edit_size(fam, refs, stride)] if s]
    sizes = a.sizes or [getattr(a, "entry_size", None) or (
        defaults.size if defaults else (1024, 1024))]
    for w, h in sizes:
        if w % stride or h % stride:
            raise _Usage(f"--sizes {w}x{h}: width and height must be "
                         f"multiples of {stride}")
    big = max(sizes, key=lambda s: s[0] * s[1])
    image_preflight.check(a.model, a.encoder, a.vae, height=big[1],
                          width=big[0], cfg=cfg, adapters=_adapter_bytes(a),
                          keep_encoder=True, variant=a.variant,
                          refs=[(r.height, r.width) for r in refs],
                          prompts=[a.prompt] + ([] if cfg == 1.0 else [""]))
    if a.via_steps:
        from gmlx.serve.image_steps_loopback import work

        # The step routes run the pipeline on the image worker, so it
        # loads there.
        pipe = work(lambda: api.load(a.model, text_encoder=a.encoder,
                                     vae=a.vae, variant=a.variant, log=log))
        table, adapters = None, []
        with _steps_client(a, pipe) as (client, mid):
            records = _bench_sizes(a, pipe, ident, fam, sizes, steps, cfg,
                                   table, adapters, log, client, mid)
    else:
        pipe = api.load(a.model, text_encoder=a.encoder, vae=a.vae,
                        variant=a.variant, log=log)
        table, adapters = _load_adapters(a, pipe, log)
        records = _bench_sizes(a, pipe, ident, fam, sizes, steps, cfg, table,
                               adapters, log, refs=refs)
    lines = "".join(json.dumps(r) + "\n" for r in records)
    if a.json:
        sys.stdout.write(lines)
    if a.json_out is not None:
        with open(a.json_out, "w", encoding="utf-8") as f:
            f.write(lines)
    return 0


@contextlib.contextmanager
def _steps_client(a, pipe):
    from gmlx.serve import image_steps_loopback as loop
    from gmlx.serve.image_steps_client import StepsClient, http_transport

    mid = loop.model_id(a.model)
    host = loop.OneModelHost(mid, pipe, {"dit": a.model, "encoder": a.encoder,
                                         "vae": a.vae})
    with loop.serve(host) as base:
        # The reference convention embeds the time as the direct run does,
        # so the latents of the two compare equal.
        yield StepsClient(http_transport(base), convention="reference"), mid


def _bench_sizes(a, pipe, ident, fam, sizes, steps, cfg, table, adapters,
                 log, client=None, mid: str = "", refs=()) -> list:
    import mlx.core as mx
    import numpy as np

    from gmlx import __version__

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
                  sampler=a.sampler, seed=a.seed)
        transport = {"mode": "direct", "overhead_ms_median": None}
        if client is None:
            kw.update(patch_table=table, refs=refs)
            for _ in range(a.warmup):
                bench_run(pipe, a.prompt, **kw)
            runs = [bench_run(pipe, a.prompt, **kw) for _ in range(a.runs)]
        else:
            from gmlx.serve.image_steps_loopback import work

            for _ in range(a.warmup):
                bench_run_steps(pipe, client, mid, a.prompt, **kw)
            runs, x = [], None
            for _ in range(a.runs):
                rec, x = bench_run_steps(pipe, client, mid, a.prompt, **kw)
                runs.append(rec)
            direct = work(lambda: _direct_latents(pipe, a.prompt, **kw))
            ms = [t for r in runs for t in r["transport_ms"]]
            transport = {"mode": "steps",
                         "overhead_ms_median": _median(ms),
                         "equal_to_direct": bool(np.array_equal(
                             direct, x, equal_nan=True))}
        rec = {"schema": BENCH_SCHEMA, "gmlx_version": __version__,
               "mlx_version": mx.__version__, "label": a.label,
               "machine": _machine(), "model": model,
               "settings": {"width": w, "height": h, "steps": steps,
                            "sampler": a.sampler, "scheduler": None,
                            "guidance": cfg, "batch": 1,
                            "adapters": adapters,
                            "seed": a.seed, "prompt": a.prompt,
                            **({"refs": [
                                {"file": os.path.basename(p),
                                 "size": f"{r.width}x{r.height}",
                                 "sha256": _file_sha256(p)}
                                for p, r in zip(a.ref, refs)]}
                               if refs else {})},
               "runs": runs, "summary": bench_summary(runs,
                                                      pipe.load_seconds),
               "transport": transport}
        records.append(rec)
        sm = rec["summary"]
        log(f"[bench] {w}x{h} {steps} steps: step first "
            f"{sm['s_per_step']['first']:.3f} s, median "
            f"{sm['s_per_step']['median']:.3f} s; encode "
            f"{sm['encode_s']:.3f} s, decode {sm['decode_s']:.3f} s, image "
            f"{sm['s_per_image']:.2f} s; load "
            f"{sm['load_s']['total']:.1f} s; peak "
            f"{sm['peak_bytes'] / 1e9:.2f} GB")
        if client is not None:
            log(f"[bench] {w}x{h} via the step routes: transport "
                f"{transport['overhead_ms_median']:.2f} ms per step "
                f"(median); its latents "
                f"{'equal' if transport['equal_to_direct'] else 'differ from'}"
                " those of a direct run")
    return records


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
                   help="The image family whose VAE this is, such as "
                        "z-image, qwen-image-2.1, krea-2, flux-3-klein or "
                        "flux-2-klein (default: the family whose VAE the "
                        "file fills, and flux3_klein for the FLUX.2 "
                        "autoencoder, whose pack serves every klein "
                        "family).")
    p.add_argument("--source-repo", default="unknown", metavar="ID",
                   help="Record this repository as the source (default "
                        "unknown).")
    p.add_argument("--source-revision", default="unknown", metavar="REV",
                   help="Record this revision of the source repository "
                        "(default unknown).")
    p.add_argument("--source-config", default=None, metavar="FILE",
                   help="Record this VAE config.json, which names its "
                        "Autoencoder class in _class_name (default: the "
                        "config.json beside SRC when it is a VAE config).")
    p.add_argument("--force", action="store_true",
                   help="Replace --out when it exists.")
    p.add_argument("--json", action="store_true",
                   help="Print a JSON record of the packed file on stdout.")
    return p


def _not_a_vae_config(text: str) -> str | None:
    """Why a config.json text is not a VAE's, or None. A VAE config names
    its Autoencoder class in ``_class_name``, as diffusers writes it."""
    try:
        config = json.loads(text)
    except ValueError:
        return "is not JSON"
    name = config.get("_class_name") if isinstance(config, dict) else None
    if isinstance(name, str) and name.startswith("Autoencoder"):
        return None
    if name is None:
        return "names no Autoencoder class in _class_name"
    return f"is the config of a {name}, not of a VAE"


def cmd_pack_vae(argv: list[str], prog: str = "gmlx image pack-vae") -> int:
    a = _pack_vae_parser(prog).parse_args(argv)
    from gmlx.image.vae import VAEError, pack_vae

    config_json = "{}"
    if a.source_config is not None:
        try:
            with open(a.source_config, encoding="utf-8") as f:
                config_json = f.read()
        except OSError as e:
            return _refuse(f"--source-config {a.source_config}: {e.strerror}")
        why = _not_a_vae_config(config_json)
        if why:
            return _refuse(f"--source-config {a.source_config}: {why}")
    else:
        beside = os.path.join(os.path.dirname(os.path.abspath(a.src)),
                              "config.json")
        try:
            with open(beside, encoding="utf-8") as f:
                text = f.read()
        except OSError:
            text = None
        why = text is not None and _not_a_vae_config(text)
        if why:
            # A model folder can hold the VAE beside its pipeline config.
            print(f"note: {beside} {why}, so the GGUF records no config; "
                  "--source-config names one", file=sys.stderr)
        elif text is not None:
            config_json = text
    family = a.family
    if family is not None:
        from gmlx.image.registry import family_named

        # Spelled as the server reads a family: flux-3-klein is flux3_klein.
        named = family_named(family)
        family = named.name if named else family
    try:
        rep = pack_vae(a.src, a.out, source_repo=a.source_repo,
                       source_revision=a.source_revision,
                       config_json=config_json, family=family,
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
