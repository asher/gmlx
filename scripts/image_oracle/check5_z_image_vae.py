#!/usr/bin/env python3
"""Check 5 oracle for Z-Image: the FLUX.1 VAE decode and encode.

diffusers' AutoencoderKL loads the official VAE on the CPU in fp32 and in
bf16. For each size it encodes a seeded synthetic image to the mean of the
latent distribution, maps the mean to model space as the img2img pipeline
does, (mean - shift) * scaling, and decodes those latents back as
ZImagePipeline does, latents / scaling + shift. Realistic latents come
from the encoder, since seeded normal latents decode to noise that hides
grouping and border errors.

The fixture holds, per size, the image, the encoder mean, the model space
latents, the raw fp32 decode and the raw bf16 decode. The manifest records
the bf16 against fp32 gap and the tolerances: fp32 decode max abs at the
plan's 1e-3, and bf16 PSNR on the [0, 1] image at twice the oracle's own
error, which is 6.02 dB below its PSNR. The fp32 encode is held to twice
the oracle's own fp32 error, its max abs against a float64 encode, since
the encoder's large activations put that error above 1e-3.

The bf16 run computes each convolution and linear product in float32 over
the bfloat16 operands, rounded once to bfloat16, as a bfloat16 GEMM with
float32 accumulation would. Its decode input is the fp32 VAE space latent
cast to bfloat16, as gmlx computes it.

  python scripts/image_oracle/run.py --python <oracle python> --threads 16 \\
      check5_z_image_vae.py --config <official repo dir>
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from pathlib import Path

import numpy as np

import fixtures
import oracle

NAME = "check5_z_image_vae"
SIZES = (512, 1024)
SEED = 5
STARTING = {"decode_fp32_max_abs": 1e-3, "decode_bf16_psnr_db": 45.0,
            "encode_fp32_max_abs": 1e-3}
RULE = ("fp32 decode max abs at the plan's starting value; fp32 encode max "
        "abs at twice the oracle's own fp32 against float64 max abs; bf16 "
        "PSNR at twice the oracle's own bf16 against fp32 RMS error, which "
        "is its PSNR less 20 log10(2) dB; bf16 encode relative L2 at twice "
        "the oracle's own")


def synthetic_image(size: int, seed: int) -> np.ndarray:
    """A (1, 3, size, size) float32 image in [-1, 1]: smooth gradients,
    hard edged shapes, a checkerboard, fine stripes, thin lines and a
    seeded texture, laid out in fractions of the size."""
    rng = np.random.default_rng(seed)
    y, x = np.meshgrid(np.linspace(0, 1, size, dtype=np.float64),
                       np.linspace(0, 1, size, dtype=np.float64),
                       indexing="ij")
    img = np.stack([0.5 + 0.4 * np.sin(2 * np.pi * (1.3 * x + 0.2)),
                    0.2 + 0.6 * y,
                    0.8 - 0.6 * x * y])
    for cx, cy, r, col in ((0.3, 0.3, 0.15, (0.9, 0.2, 0.1)),
                           (0.7, 0.35, 0.1, (0.1, 0.5, 0.9)),
                           (0.55, 0.75, 0.18, (0.95, 0.85, 0.2))):
        disk = (x - cx) ** 2 + (y - cy) ** 2 < r * r
        img[:, disk] = np.asarray(col)[:, None]
    box = (x > 0.05) & (x < 0.25) & (y > 0.6) & (y < 0.9)
    img[:, box] = 0.15
    cell = max(size // 64, 2)
    iy, ix = np.indices((size, size))
    checker = ((iy // cell + ix // cell) % 2).astype(np.float64)
    patch = (x > 0.75) & (y > 0.7)
    img[:, patch] = checker[patch]
    stripes = (x > 0.8) & (y < 0.25)
    img[:, stripes] = 0.5 + 0.5 * np.sin(2 * np.pi * iy[stripes] / 3.0)
    for k in range(6):
        row = int(size * (0.45 + 0.03 * k))
        img[:, row, int(0.05 * size):int(0.45 * size)] = 0.0
    tex = rng.standard_normal((3, size // 8, size // 8))
    tex = np.kron(tex, np.ones((8, 8)))
    zone = (x > 0.3) & (x < 0.7) & (y > 0.05) & (y < 0.2)
    img[:, zone] = 0.5 + 0.15 * tex[:, zone]
    img = np.clip(img, 0.0, 1.0) * 2 - 1
    return img[None].astype(np.float32)


def emulate_bf16(torch, enable: bool) -> None:
    """Swap F.conv2d and F.linear for float32 products over bfloat16
    operands with one bfloat16 rounding of the result, or restore them."""
    import torch.nn.functional as F

    native = getattr(F, "_check5_native", None)
    if native is None:
        native = F._check5_native = {"conv2d": F.conv2d, "linear": F.linear}
    if not enable:
        F.conv2d, F.linear = native["conv2d"], native["linear"]
        return

    def wrap(fn):
        def run(x, w, b=None, *args, **kw):
            if x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16:
                y = fn(x.float(), w.float(),
                       None if b is None else b.float(), *args, **kw)
                return y.to(torch.bfloat16)
            return fn(x, w, b, *args, **kw)
        return run

    F.conv2d = wrap(native["conv2d"])
    F.linear = wrap(native["linear"])


def to_unit(img: np.ndarray) -> np.ndarray:
    """The pipeline's postprocess: x / 2 + 0.5, clamped to [0, 1]."""
    return np.clip(img.astype(np.float64) / 2 + 0.5, 0.0, 1.0)


def run_size(torch, vaes, vae64, size: int, fx, results: dict,
             tol: dict) -> None:
    cfg = vaes["fp32"].config
    scaling, shift = float(cfg.scaling_factor), float(cfg.shift_factor)
    image = synthetic_image(size, SEED)
    x32 = torch.from_numpy(image)
    t = {}
    out = {}
    for mode, vae in vaes.items():
        emulate_bf16(torch, mode == "bf16")
        dt = oracle.torch_dtype(mode)
        t0 = time.perf_counter()
        mean = vae.encode(x32.to(dt)).latent_dist.mean
        t[f"{mode}_encode_s"] = round(time.perf_counter() - t0, 1)
        out[f"mean_{mode}"] = oracle.to_numpy(mean)
    emulate_bf16(torch, False)
    mean64 = vae64.encode(x32.double()).latent_dist.mean.numpy()
    mean32 = torch.from_numpy(out["mean_fp32"])
    latents = (mean32 - shift) * scaling
    z = latents / scaling + shift
    for mode, vae in vaes.items():
        emulate_bf16(torch, mode == "bf16")
        dt = oracle.torch_dtype(mode)
        t0 = time.perf_counter()
        dec = vae.decode(z.to(dt), return_dict=False)[0]
        t[f"{mode}_decode_s"] = round(time.perf_counter() - t0, 1)
        out[f"decoded_{mode}"] = oracle.to_numpy(dec)
    emulate_bf16(torch, False)

    fx.add(f"image_{size}", image)
    fx.add(f"encode_mean_{size}", out["mean_fp32"])
    fx.add(f"encode_mean_bf16_{size}", out["mean_bf16"])
    fx.add(f"latents_{size}", oracle.to_numpy(latents))
    fx.add(f"decoded_{size}", out["decoded_fp32"])
    fx.add(f"decoded_bf16_{size}", out["decoded_bf16"])

    d32, d16 = out["decoded_fp32"], out["decoded_bf16"]
    gap = fixtures.compare(d16, d32)
    gap["psnr_db"] = fixtures.psnr(to_unit(d16), to_unit(d32), 1.0)
    enc = fixtures.compare(out["mean_bf16"], out["mean_fp32"])
    enc64 = fixtures.compare(out["mean_fp32"], mean64)
    rt = fixtures.psnr(to_unit(d32), to_unit(image), 1.0)
    results[str(size)] = {
        "decode_bf16_vs_fp32": gap, "encode_bf16_vs_fp32": enc,
        "encode_fp32_vs_fp64": enc64,
        "round_trip_fp32_psnr_db": rt,
        "latents_std": float(np.std(oracle.to_numpy(latents))),
        "seconds": t}
    tol[str(size)] = {
        "decode_fp32_max_abs": STARTING["decode_fp32_max_abs"],
        "decode_bf16_psnr_db": gap["psnr_db"] - 20 * math.log10(2),
        "encode_fp32_max_abs": 2 * enc64["max_abs"],
        "encode_bf16_rel_l2": 2 * enc["rel_l2"]}
    print(f"[check5] {size}: bf16 decode PSNR {gap['psnr_db']:.2f} dB, max "
          f"abs {gap['max_abs']:.3e}; bf16 encode rel L2 "
          f"{enc['rel_l2']:.3e}; fp32 encode max abs against float64 "
          f"{enc64['max_abs']:.3e}; fp32 round trip {rt:.2f} dB; {t}",
          flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, type=Path,
                    help="local copy of Tongyi-MAI/Z-Image-Turbo with "
                         "vae/config.json and its safetensors")
    ap.add_argument("--config-revision",
                    default="f332072aa78be7aecdf3ee76d5c247082da564a6",
                    help="revision of the --config copy, for the manifest")
    ap.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    ap.add_argument("--out", type=Path,
                    help="fixture directory (default $GMLX_IMAGE_FIXTURES)")
    a = ap.parse_args(argv)
    if any(s % 16 for s in a.sizes):
        ap.error("sizes must be multiples of 16")
    torch = oracle.setup()
    from diffusers import AutoencoderKL

    vae32 = AutoencoderKL.from_pretrained(
        str(a.config / "vae"), torch_dtype=torch.float32,
        local_files_only=True).eval()
    vae16 = copy.deepcopy(vae32).to(torch.bfloat16).eval()
    vaes = {"fp32": vae32, "bf16": vae16}
    vae64 = copy.deepcopy(vae32).double().eval()
    for mode, vae in vaes.items():
        where = {str(p.device) for p in vae.parameters()}
        kinds = {p.dtype for p in vae.parameters()}
        if where != {"cpu"} or kinds != {oracle.torch_dtype(mode)}:
            raise SystemExit(f"error: {mode} VAE on {where} in {kinds}")

    fx = fixtures.FixtureWriter(
        NAME, check=5, family="z_image",
        script=f"scripts/image_oracle/{Path(__file__).name}",
        argv=[Path(__file__).name, "--config", a.config.name, "--sizes",
              *map(str, a.sizes)])
    fx.add_file("vae", a.config / "vae" / "diffusion_pytorch_model.safetensors")
    fx.add_file("vae_config", a.config / "vae" / "config.json")
    results, tol = {}, {}
    for size in a.sizes:
        run_size(torch, vaes, vae64, size, fx, results, tol)
    cfg = vae32.config
    fx.inputs.update(
        sizes=list(a.sizes), seed=SEED, config_revision=a.config_revision,
        image="synthetic_image(size, seed) in the oracle script, NCHW in "
        "[-1, 1]", scaling_factor=float(cfg.scaling_factor),
        shift_factor=float(cfg.shift_factor),
        latents="(encoder mean - shift) * scaling, from the fp32 VAE",
        decode_input="latents / scaling + shift in float32, cast to the "
        "VAE dtype", psnr="on x / 2 + 0.5 clamped to [0, 1], peak 1")
    fx.results.update(
        oracle={"device": "cpu", "threads": torch.get_num_threads(),
                "bf16_products": "bf16 operands, float32 conv and GEMM, "
                "one bf16 rounding"}, sizes=results)
    fx.tolerances.update(rule=RULE, starting=STARTING, sizes=tol)
    fx.note("The bf16 VAE is a bfloat16 copy of the fp32 one. The official "
            "weights are bfloat16, so both hold the same values.")
    path = fx.write(a.out)
    print(f"[check5] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
