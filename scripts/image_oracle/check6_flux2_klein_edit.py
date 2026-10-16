#!/usr/bin/env python3
"""Check 6 oracle for FLUX.2 klein edits: a whole sampling run with one
reference image, on injected noise, to the final latents, and the
preparation of reference images of several sizes.

The references are prepared by the reference's own encode_image_refs in
src/flux2/sampling.py, with BFL's AutoEncoder in float32 from
--autoencoder: default_prep under the pixel limit of the reference count,
the encode, and the ids of each reference's index. The run is
check6_flux2_klein's text to image path with those tokens and ids as
img_cond_seq and img_cond_seq_ids, the size the first prepared reference's
(gmlx's default; the CLI's match_image_size takes the source's own size,
and its default is 1360 x 768). The DiT's input is kept at every call, so
the fixture holds the reference tokens and ids the DiT read, and the
encode is kept in float32 before encode_image_refs rounds it to bf16.

The oracle environment has no torchvision. default_prep reads it for
ToTensor alone, so a stand-in gives ToTensor's result for an RGB image:
the uint8 pixels, channels first, divided by 255 in float32. The
preparations of the sources of PREPARE are the PIL images default_prep
makes before ToTensor, kept as uint8, and each is checked against
default_prep's tensor. A source the rules refuse is kept with its error.

  python scripts/image_oracle/run.py --python <oracle python> --threads 16 \\
      check6_flux2_klein_edit.py --reference <flux2>/src \\
      --reference-commit <sha> --dit flux-2-klein-4b-Q4_K_M.gguf \\
      --variant distilled --autoencoder ae.safetensors \\
      --features check2_flux2_klein.npz
"""

from __future__ import annotations

import argparse
import sys
import time
import types
from pathlib import Path

import numpy as np

import check3_flux2_klein as c3
import check6_flux2_klein as c6
import fixtures
import oracle

NAME = "check6_flux2_klein_edit"
SCRIPT = "scripts/image_oracle/check6_flux2_klein_edit.py"
SEED = 6
# Synthetic sources, (width, height, channels).
SOURCES = {"wide": (600, 400, 3), "square": (400, 400, 4),
           "large": (2400, 1800, 3), "many": (1600, 1000, 3),
           "odd": (1031, 777, 3), "narrow": (512, 64, 3),
           "small": (64, 40, 3), "long": (513, 64, 3)}
# The edit the DiT runs: its sources, in order.
CASE = ("wide",)
# Sources prepared as one reference and as one of several.
PREPARE = {"one": ("wide", "square", "large", "odd", "narrow", "small",
                   "long"),
           "several": ("wide", "large", "many", "odd")}
RULE = ("latents after a kept step and after the last: relative L2 at "
        "twice the oracle's own bf16 against fp32 relative L2 after that "
        "step, on the run's own size")


def source(rng, width: int, height: int, channels: int) -> np.ndarray:
    """A picture of gradients, rectangles and a little noise, (H, W, C)
    uint8."""
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    img = np.stack([x / max(1, width - 1) * 255,
                    y / max(1, height - 1) * 255,
                    (x + y) / max(1, width + height - 2) * 255], axis=-1)
    for _ in range(12):
        x0, x1 = sorted(rng.integers(0, width, 2))
        y0, y1 = sorted(rng.integers(0, height, 2))
        img[y0:y1 + 1, x0:x1 + 1] = rng.integers(0, 256, 3)
    img += rng.normal(0, 6, img.shape)
    img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    if channels == 4:
        alpha = rng.integers(0, 256, (height, width, 1), dtype=np.uint8)
        img = np.concatenate([img, alpha], axis=-1)
    return img


def pil(a: np.ndarray):
    from PIL import Image

    return Image.fromarray(a, "RGBA" if a.shape[2] == 4 else "RGB")


def to_tensor_module(torch):
    """A torchvision stand-in whose transforms.ToTensor gives torchvision's
    result for an RGB PIL image."""
    def to_tensor(pic):
        if pic.mode != "RGB":
            raise SystemExit(f"error: ToTensor of a {pic.mode} image")
        a = np.array(pic, np.uint8, copy=True)
        return torch.from_numpy(a).permute(2, 0, 1).contiguous().to(
            torch.float32).div(255)

    tv = types.ModuleType("torchvision")
    tv.transforms = types.SimpleNamespace(ToTensor=lambda: to_tensor)
    return tv


def prepared(sampling, image, limit: int):
    """default_prep's PIL steps for ``image`` under ``limit`` pixels: the
    uint8 pixels before ToTensor, or the error the rules raise."""
    try:
        img = sampling.to_rgb(image)
        img = sampling.cap_min_pixels(img)
        img = sampling.cap_pixels(img, limit)
        img = sampling.center_crop_to_multiple_of_x(img, 16)
    except ValueError as e:
        return None, str(e)
    out = np.asarray(img, np.uint8)
    t = oracle.to_numpy(sampling.default_prep(img=image, limit_pixels=limit))
    want = out.transpose(2, 0, 1).astype(np.float32) / np.float32(255)
    if not np.array_equal(t, 2 * want - 1):
        raise SystemExit("error: default_prep is not its PIL steps")
    return out, None


class Encodes:
    """BFL's AutoEncoder whose encode also keeps each output in float32,
    before encode_image_refs rounds its tokens to bf16."""

    def __init__(self, ae):
        self.ae, self.kept = ae, []

    def encode(self, x):
        z = self.ae.encode(x)
        self.kept.append(oracle.to_numpy(z.float()))
        return z


class OnCPU:
    """encode_image_refs moves each prepared image with .cuda(); on the
    CPU that is the image itself."""

    def __init__(self, torch):
        self.torch = torch

    def __enter__(self):
        self.real = self.torch.Tensor.cuda
        self.torch.Tensor.cuda = lambda t, *a, **k: t

    def __exit__(self, *exc):
        self.torch.Tensor.cuda = self.real


class Kept(c6.Recorded):
    """c6.Recorded that also keeps the reference rows of the first call."""

    def __init__(self, torch, model, n_img: int):
        super().__init__(torch, model, n_img)
        self.ref_tokens = self.ref_ids = None

    def __call__(self, **kw):
        if self.ref_tokens is None:
            self.ref_tokens = oracle.to_numpy(kw["x"][0, self.n_img:].float())
            self.ref_ids = oracle.to_numpy(kw["x_ids"][0, self.n_img:])
        return super().__call__(**kw)


def generate(torch, sampling, model, variant: str, feats: dict,
             noise: np.ndarray, steps: int, guidance: float, dtype,
             ref_tokens, ref_ids):
    """c6.generate with the references after the image tokens."""
    _, _, h, w = noise.shape
    prompt = torch.from_numpy(feats[c6.PROMPT][None].copy()).to(
        torch.bfloat16)
    if variant == "distilled":
        ctx = prompt
    else:
        empty = torch.from_numpy(feats[""][None].copy()).to(torch.bfloat16)
        ctx = torch.cat([empty, prompt], dim=0)
    ctx, ctx_ids = sampling.batched_prc_txt(ctx)
    randn = torch.from_numpy(noise.copy()).to(torch.bfloat16)
    x, x_ids = sampling.batched_prc_img(randn)
    timesteps = sampling.get_schedule(steps, x.shape[1])
    rec = Kept(torch, model, x.shape[1])
    run = sampling.denoise if variant == "distilled" else sampling.denoise_cfg
    t0 = time.perf_counter()
    out = run(rec, x.to(dtype), x_ids, ctx.to(dtype), ctx_ids,
              timesteps=timesteps, guidance=guidance,
              img_cond_seq=ref_tokens.to(dtype), img_cond_seq_ids=ref_ids)
    end = time.perf_counter()
    if len(rec.latents) != steps:
        raise SystemExit(f"error: {len(rec.latents)} DiT calls for {steps} "
                         f"steps")
    if not np.array_equal(c6.to_latents(rec.latents[0], h, w),
                          oracle.to_numpy(randn)):
        raise SystemExit("error: the first latents are not the noise "
                         "rounded to bf16")
    after = [c6.to_latents(t, h, w)
             for t in rec.latents[1:] + [oracle.to_numpy(out[0])]]
    secs = np.diff(np.array([t0, *rec.marks[1:], end]))
    return (after, np.float32(rec.times), secs,
            np.asarray(timesteps, np.float32), rec)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reference", type=Path, required=True,
                    help="src directory of BFL's flux2")
    ap.add_argument("--reference-commit", required=True,
                    help="the commit of --reference, for the manifest")
    ap.add_argument("--dit", type=Path, required=True,
                    help="FLUX.2 klein DiT GGUF")
    ap.add_argument("--variant", choices=("distilled", "base"),
                    required=True, help="the checkpoint --dit holds")
    ap.add_argument("--autoencoder", type=Path, required=True,
                    help="BFL's ae.safetensors")
    ap.add_argument("--features", type=Path, required=True,
                    help="check 2 .npz with features_plain and "
                         "features_empty")
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--steps", type=int,
                    help="steps, FLUX2_MODEL_INFO's for the variant when "
                         "unset")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dequant", choices=oracle.DEQUANT, default="exact")
    ap.add_argument("--name", default=NAME)
    ap.add_argument("--out", type=Path,
                    help="fixture directory (default $GMLX_IMAGE_FIXTURES)")
    a = ap.parse_args(argv)
    torch = oracle.setup()
    oracle.set_dequant(a.dequant)
    fm, sampling, util = c6.reference(a.reference.expanduser())
    # sampling imported c6.reference's empty stand-in under its own name.
    sampling.torchvision = to_tensor_module(torch)
    from flux2 import autoencoder
    from safetensors.torch import load_file

    t0 = time.perf_counter()
    ae = autoencoder.AutoEncoder(autoencoder.AutoEncoderParams())
    ae.load_state_dict(load_file(str(a.autoencoder)), strict=True)
    ae = ae.float().eval()
    state, census = c3.gguf_state(torch, a.dit)
    info = c6.model_info(util, fm, state, a.variant)
    modes = ("fp32",) if a.no_bf16 else ("fp32", "bf16")
    models = {m: c3.build(torch, fm, state, m) for m in modes}
    print(f"[edit] loaded {a.dit.name} ({a.variant}, {info['key']}) and the "
          f"autoencoder in {time.perf_counter() - t0:.1f}s, codecs {census}",
          flush=True)
    steps = a.steps or info["num_steps"]
    guidance = info["guidance"]
    keep = [k for k in c6.KEEP[a.variant] if k < steps]
    feats = c6.load_features(torch, a.features)
    fx = fixtures.FixtureWriter(a.name, check=6, family="flux2_klein",
                                script=SCRIPT)
    fx.add_file("dit", a.dit)
    fx.add_file("autoencoder", a.autoencoder)
    fx.add_file("features", a.features)
    fx.add("features", feats[c6.PROMPT])
    if a.variant == "base":
        fx.add("features_empty", feats[""])
    rng = np.random.Generator(np.random.PCG64(a.seed))
    srcs = {k: source(rng, *v) for k, v in SOURCES.items()}
    for k, img in srcs.items():
        fx.add(f"source_{k}", img)
    limits = {"one": 2024 ** 2, "several": 1024 ** 2}
    preps = {}
    for rule, names in PREPARE.items():
        for k in names:
            px, err = prepared(sampling, pil(srcs[k]), limits[rule])
            if px is not None:
                fx.add(f"prepared_{rule}_{k}", px)
            preps[f"{rule}_{k}"] = ({"size": [px.shape[1], px.shape[0]]}
                                    if px is not None else {"error": err})
    fx.inputs["prepared"] = preps
    enc = Encodes(ae)
    with OnCPU(torch), torch.no_grad():
        ref_tokens, ref_ids = sampling.encode_image_refs(
            enc, [pil(srcs[k]) for k in CASE])
    if len(enc.kept) != len(CASE):
        raise SystemExit(f"error: {len(enc.kept)} reference encodes")
    first = preps[f"one_{CASE[0]}"]["size"] if len(CASE) == 1 else None
    width, height = first
    h, w = height // 16, width // 16
    c = c3.params_of(fm, state).in_channels
    noise = np.random.Generator(np.random.PCG64(a.seed)).standard_normal(
        (1, c, h, w), dtype=np.float32)
    fx.add("noise", noise)
    fx.add("ref_tokens", oracle.to_numpy(ref_tokens[0].float()))
    fx.add("ref_ids", oracle.to_numpy(ref_ids[0]))
    lat = enc.kept[0]
    if lat.shape != (1, c, h, w):
        raise SystemExit(f"error: reference latents {lat.shape}")
    rounded = oracle.to_numpy(torch.from_numpy(lat).to(torch.bfloat16).float())
    if not np.array_equal(c6.to_latents(oracle.to_numpy(
            ref_tokens[0].float()), h, w), rounded):
        raise SystemExit("error: the tokens are not the encode in bf16")
    fx.add("ref_latents_0", lat)
    oracle.emulate_bf16_matmul(False)
    after, times, secs, schedule, rec = generate(
        torch, sampling, models["fp32"], a.variant, feats, noise, steps,
        guidance, torch.float32, ref_tokens, ref_ids)
    c6.check_times(torch, schedule, times)
    if not (np.array_equal(rec.ref_tokens, oracle.to_numpy(
            ref_tokens[0].float()))
            and np.array_equal(rec.ref_ids, oracle.to_numpy(ref_ids[0]))):
        raise SystemExit("error: the DiT read other reference rows")
    fx.add("schedule", schedule)
    fx.add("times", times)
    fx.add("latents", after[-1])
    for k in keep:
        fx.add(f"latents_step{k}", after[k - 1])
    row = {"sources": list(CASE), "size": [width, height],
           "fp32_s": round(float(secs.sum()), 1)}
    print(f"[edit] {width}x{height}, fp32 {row['fp32_s']} s", flush=True)
    if "bf16" in models:
        oracle.emulate_bf16_matmul(True)
        try:
            after16, times16, secs16, _, _ = generate(
                torch, sampling, models["bf16"], a.variant, feats, noise,
                steps, guidance, torch.bfloat16, ref_tokens, ref_ids)
        finally:
            oracle.emulate_bf16_matmul(False)
        if not np.array_equal(times16, times):
            raise SystemExit("error: the bf16 run saw other times")
        fx.add("latents_bf16", after16[-1])
        gap = fixtures.compare(after16[-1], after[-1])
        by_step = [fixtures.rel_l2(x, y) for x, y in zip(after16, after)]
        row.update(bf16_s=round(float(secs16.sum()), 1),
                   latents_bf16_vs_fp32=gap, bf16_vs_fp32_by_step=by_step)
        fx.tolerances["by_step_rel_l2"] = [2 * g for g in by_step]
        print(f"[edit] bf16 {row['bf16_s']} s, final rel L2 "
              f"{gap['rel_l2']:.3e}, by step "
              f"{[f'{v:.2e}' for v in by_step]}", flush=True)
    fx.results.update(
        oracle={"device": "cpu", "threads": torch.get_num_threads(),
                "dequant": oracle.dequant_mode(),
                "bf16_matmul": "bf16 operands, float32 GEMM, one bf16 "
                               "rounding",
                "autoencoder": "float32"},
        run=row)
    fx.inputs.update(
        variant=a.variant, model_info=info["key"], steps=steps,
        guidance=guidance, seed=a.seed, kept_steps=keep, case=list(CASE),
        sources=SOURCES, limits=limits, bf16="bf16" in models,
        noise="numpy Generator(PCG64(seed)).standard_normal(shape, "
              "dtype=float32), NCHW, (1, 128, height / 16, width / 16) of "
              "the first prepared reference; the CLI's noise is bf16, so "
              "it is rounded",
        sources_rule="numpy Generator(PCG64(seed)) in SOURCES order: "
                     "gradients, 12 rectangles, normal noise of 6, and a "
                     "random alpha for 4 channels",
        ref_tokens="encode_image_refs' bf16 tokens as float32, (L, 128)",
        ref_ids="encode_image_refs' ids, (L, 4)",
        ref_latents="the float32 encode of each reference, (1, 128, "
                    "height / 16, width / 16), before encode_image_refs "
                    "rounds its tokens to bf16",
        reference_commit=a.reference_commit, gguf_codecs=census)
    fx.tolerances["rule"] = RULE
    path = fx.write(a.out)
    print(f"[edit] wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
