#!/usr/bin/env python3
"""Check 5 oracle for Qwen-Image 2.1: the VAE decode and encode.

diffusers' AutoencoderKLQwenImage21 runs on the CPU in fp32 and in bf16.
For each size it encodes a seeded synthetic RGB image, with the alpha of
ones that the pipeline's RGBA conversion gives it, to the mean of the
latent distribution, maps the mean to model space as the pipeline does,
(mean - latents_mean) / latents_std, and decodes those latents back as
the pipeline does, latents * latents_std + latents_mean. It also encodes
the image with a seeded alpha channel, so the fourth input channel is
read as data.

The fixture holds, per size, the RGB image, the alpha, the two encoder
means, the model space latents, the fp32 decode and the bf16 decode, all
RGBA and clamped to [-1, 1] as the VAE returns them. The manifest records
the bf16 against fp32 gap, the share of the fp32 decode that the clamp
holds at -1 or 1, and the tolerances: fp32 decode max abs at 2.5e-4,
which is 5 times the largest reading of gmlx on the official file, fp32
encode max abs at its starting value of 1e-3 or at twice the oracle's own
fp32 error against a float64 encode where that is larger, and bf16 PSNR on
the [0, 1] image at twice the oracle's own error, which is 6.02 dB below
its PSNR.

The bf16 run computes each convolution in float32 over the bfloat16
operands, rounded once to bfloat16, as a bfloat16 conv with float32
accumulation would. Its decode input is the fp32 VAE space latent cast to
bfloat16, as gmlx computes it.

--self-test builds the class from the tiny config of
qwen_image21_vae_tiny.py with that module's seeded weights, written as a
safetensors file in the layout of the official one, and runs one 48x64
image through it. It then builds each of that module's further configs,
with other widths, levels and time flags, and writes a second fixture with
the fp32 decode of seeded latents and the encoder mean and log variance of
a seeded RGBA image for each. The test tree keeps both fixtures' arrays
beside their manifests. Their fp32 tolerance is 5e-5 max abs on the CPU:
about 7 times the largest reading of a right model over these configs,
6.9e-6, where the smallest reading of a wrong model is 0.33.

  python scripts/image_oracle/run.py --python <oracle python> --threads 16 \\
      check5_qwen_image21_vae.py --config <official repo dir>
  python scripts/image_oracle/run.py --python <oracle python> \\
      check5_qwen_image21_vae.py --self-test
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import fixtures
import oracle
import qwen_image21_vae_tiny as tiny

NAME = "check5_qwen_image21_vae"
TINY_NAME = "selftest_qwen_image21_vae_tiny"
CASES_NAME = "selftest_qwen_image21_vae_cases"
SIZES = (512, 1024)
TINY_SIZE = (48, 64)
SEED = 5
REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
STARTING = 1e-3
# The first readings of gmlx on the official file, in float32 with exact
# float32 products on the default conv path, were 3.04e-5 max abs at 512
# pixels and 4.89e-5 at 1024. The decode gate is 5 times the larger.
DECODE_FP32_MAX_ABS = 2.5e-4
TINY_FP32_MAX_ABS = 5e-5
BF16_RULE = ("bf16 PSNR at twice the oracle's own bf16 against fp32 RMS "
             "error, which is its PSNR less 20 log10(2) dB; bf16 encode "
             "relative L2 at twice the oracle's own")
RULE = ("fp32 decode max abs at 2.5e-4: 5 times the largest reading of a "
        "right model on the official file, 4.89e-5 at 1024 pixels; fp32 "
        "encode max abs at its starting value, 1e-3, or at twice the "
        "oracle's own fp32 against float64 max abs where that is larger; "
        + BF16_RULE)
TINY_RULE = ("fp32 max abs at 5e-5 on the CPU: about 7 times the largest "
             "reading of a right model over the self-test configs, 6.9e-6, "
             "where the smallest reading of a wrong model is 0.33")


def synthetic_image(height: int, width: int, seed: int) -> np.ndarray:
    """A (1, 3, height, width) float32 image in [-1, 1]: smooth gradients,
    hard edged shapes, a checkerboard, fine stripes, thin lines and a
    seeded texture, laid out in fractions of the size."""
    rng = np.random.default_rng(seed)
    y, x = np.meshgrid(np.linspace(0, 1, height, dtype=np.float64),
                       np.linspace(0, 1, width, dtype=np.float64),
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
    cell = max(min(height, width) // 64, 2)
    iy, ix = np.indices((height, width))
    checker = ((iy // cell + ix // cell) % 2).astype(np.float64)
    patch = (x > 0.75) & (y > 0.7)
    img[:, patch] = checker[patch]
    stripes = (x > 0.8) & (y < 0.25)
    img[:, stripes] = 0.5 + 0.5 * np.sin(2 * np.pi * iy[stripes] / 3.0)
    for k in range(6):
        row = int(height * (0.45 + 0.03 * k))
        img[:, row, int(0.05 * width):int(0.45 * width)] = 0.0
    tex = rng.standard_normal((3, height, width))
    zone = (x > 0.3) & (x < 0.7) & (y > 0.05) & (y < 0.2)
    img[:, zone] = 0.5 + 0.15 * tex[:, zone]
    img = np.clip(img, 0.0, 1.0) * 2 - 1
    return img[None].astype(np.float32)


def synthetic_alpha(height: int, width: int, seed: int) -> np.ndarray:
    """A (1, 1, height, width) float32 alpha in [-1, 1]: opaque, with a
    ramp to transparent on the right, a transparent disk and a seeded
    band."""
    rng = np.random.default_rng(seed + 1)
    y, x = np.meshgrid(np.linspace(0, 1, height, dtype=np.float64),
                       np.linspace(0, 1, width, dtype=np.float64),
                       indexing="ij")
    a = np.clip((1.0 - x) / 0.4, 0.0, 1.0)
    a[(x - 0.35) ** 2 + (y - 0.6) ** 2 < 0.15 ** 2] = 0.0
    band = (y > 0.85)
    a[band] = rng.uniform(0, 1, (height, width))[band]
    return (a * 2 - 1)[None, None].astype(np.float32)


def emulate_bf16(torch, enable: bool) -> None:
    """Swap F.conv2d for a float32 conv over bfloat16 operands with one
    bfloat16 rounding of the result, or restore it."""
    import torch.nn.functional as F

    native = getattr(F, "_check5_native", None)
    if native is None:
        native = F._check5_native = {"conv2d": F.conv2d}
    if not enable:
        F.conv2d = native["conv2d"]
        return

    def run(x, w, b=None, *args, **kw):
        if x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16:
            y = native["conv2d"](x.float(), w.float(),
                                 None if b is None else b.float(), *args,
                                 **kw)
            return y.to(torch.bfloat16)
        return native["conv2d"](x, w, b, *args, **kw)

    F.conv2d = run


def to_unit(img: np.ndarray) -> np.ndarray:
    """The pipeline's postprocess: x / 2 + 0.5, clamped to [0, 1]."""
    return np.clip(img.astype(np.float64) / 2 + 0.5, 0.0, 1.0)


def tag(size) -> str:
    return f"{size[0]}x{size[1]}"


def _mean(vae, x):
    """The latent mean (1, C, h, w) of an RGBA image (1, 4, H, W)."""
    return vae.encode(x.unsqueeze(2)).latent_dist.mean[:, :, 0]


def run_size(torch, vaes, vae64, size, fx, results: dict, tol: dict,
             fp32_max_abs: float, decode_max_abs: float) -> None:
    cfg = vaes["fp32"].config
    z_dim = cfg.z_dim
    l_mean = torch.tensor(cfg.latents_mean).view(1, z_dim, 1, 1)
    l_std = torch.tensor(cfg.latents_std).view(1, z_dim, 1, 1)
    h, w = size
    image = synthetic_image(h, w, SEED)
    alpha = synthetic_alpha(h, w, SEED)
    rgb = torch.from_numpy(image)
    opaque = torch.cat([rgb, torch.ones_like(rgb[:, :1])], dim=1)
    cut = torch.cat([rgb, torch.from_numpy(alpha)], dim=1)
    t, out = {}, {}
    for mode, vae in vaes.items():
        emulate_bf16(torch, mode == "bf16")
        dt = oracle.torch_dtype(mode)
        t0 = time.perf_counter()
        out[f"mean_{mode}"] = oracle.to_numpy(_mean(vae, opaque.to(dt)))
        t[f"{mode}_encode_s"] = round(time.perf_counter() - t0, 1)
        out[f"mean_rgba_{mode}"] = oracle.to_numpy(_mean(vae, cut.to(dt)))
    emulate_bf16(torch, False)
    mean64 = _mean(vae64, opaque.double()).numpy()
    mean32 = torch.from_numpy(out["mean_fp32"])
    latents = (mean32 - l_mean) / l_std
    z = latents * l_std + l_mean
    for mode, vae in vaes.items():
        emulate_bf16(torch, mode == "bf16")
        dt = oracle.torch_dtype(mode)
        t0 = time.perf_counter()
        dec = vae.decode(z.to(dt).unsqueeze(2), return_dict=False)[0][:, :, 0]
        t[f"{mode}_decode_s"] = round(time.perf_counter() - t0, 1)
        out[f"decoded_{mode}"] = oracle.to_numpy(dec)
    emulate_bf16(torch, False)

    s = tag(size)
    fx.add(f"image_{s}", image)
    fx.add(f"alpha_{s}", alpha)
    fx.add(f"encode_mean_{s}", out["mean_fp32"])
    fx.add(f"encode_mean_bf16_{s}", out["mean_bf16"])
    fx.add(f"encode_mean_rgba_{s}", out["mean_rgba_fp32"])
    fx.add(f"latents_{s}", oracle.to_numpy(latents))
    fx.add(f"decoded_{s}", out["decoded_fp32"])
    fx.add(f"decoded_bf16_{s}", out["decoded_bf16"])

    d32, d16 = out["decoded_fp32"], out["decoded_bf16"]
    gap = fixtures.compare(d16, d32)
    gap["psnr_db"] = fixtures.psnr(to_unit(d16), to_unit(d32), 1.0)
    enc = fixtures.compare(out["mean_bf16"], out["mean_fp32"])
    enc64 = fixtures.compare(out["mean_fp32"], mean64)
    rt = fixtures.psnr(to_unit(d32[:, :3]), to_unit(image), 1.0)
    held = float(np.mean(np.abs(d32) == 1.0))
    results[s] = {
        "decode_bf16_vs_fp32": gap, "encode_bf16_vs_fp32": enc,
        "encode_fp32_vs_fp64": enc64,
        "encode_rgba_vs_opaque": fixtures.compare(out["mean_rgba_fp32"],
                                                  out["mean_fp32"]),
        "decode_clamped_share": held,
        "decode_alpha_range": [float(d32[:, 3].min()),
                               float(d32[:, 3].max())],
        "round_trip_fp32_psnr_db": rt,
        "latents_std": float(np.std(oracle.to_numpy(latents))),
        "seconds": t}
    tol[s] = {
        "decode_fp32_max_abs": decode_max_abs,
        "decode_bf16_psnr_db": gap["psnr_db"] - 20 * math.log10(2),
        "encode_fp32_max_abs": max(fp32_max_abs, 2 * enc64["max_abs"]),
        "encode_bf16_rel_l2": 2 * enc["rel_l2"]}
    print(f"[check5] {s}: bf16 decode PSNR {gap['psnr_db']:.2f} dB, max "
          f"abs {gap['max_abs']:.3e}; bf16 encode rel L2 "
          f"{enc['rel_l2']:.3e}; fp32 encode max abs against float64 "
          f"{enc64['max_abs']:.3e}; clamp holds {held:.4f} of the fp32 "
          f"decode; fp32 round trip {rt:.2f} dB; {t}", flush=True)


def _modes(torch, vae32) -> tuple[dict, object]:
    vae16 = copy.deepcopy(vae32).to(torch.bfloat16).eval()
    vaes = {"fp32": vae32, "bf16": vae16}
    for mode, vae in vaes.items():
        where = {str(p.device) for p in vae.parameters()}
        kinds = {p.dtype for p in vae.parameters()}
        if where != {"cpu"} or kinds != {oracle.torch_dtype(mode)}:
            raise SystemExit(f"error: {mode} VAE on {where} in {kinds}")
    return vaes, copy.deepcopy(vae32).double().eval()


def _write(torch, a, name: str, vae32, sizes, files: dict, argv: list,
           extra: dict, notes: list, fp32_max_abs: float, rule: str,
           decode_max_abs: float | None = None) -> int:
    """``fp32_max_abs`` gates the fp32 encode, and the fp32 decode too
    unless ``decode_max_abs`` gives the decode a value of its own."""
    if decode_max_abs is None:
        decode_max_abs = fp32_max_abs
    vaes, vae64 = _modes(torch, vae32)
    fx = fixtures.FixtureWriter(
        name, check=5, family="qwen_image21",
        script=f"scripts/image_oracle/{Path(__file__).name}", argv=argv)
    for role, path in files.items():
        fx.add_file(role, path)
    results, tol = {}, {}
    for size in sizes:
        run_size(torch, vaes, vae64, size, fx, results, tol, fp32_max_abs,
                 decode_max_abs)
    cfg = vae32.config
    fx.inputs.update(
        sizes=[list(s) for s in sizes], seed=SEED,
        image="synthetic_image(height, width, seed) in the oracle script, "
        "RGB, NCHW in [-1, 1], encoded with an alpha of ones",
        alpha="synthetic_alpha(height, width, seed), which replaces the "
        "ones for encode_mean_rgba",
        latents_mean=[float(v) for v in cfg.latents_mean],
        latents_std=[float(v) for v in cfg.latents_std],
        latents="(encoder mean - latents_mean) / latents_std, from the "
        "fp32 VAE",
        decode_input="latents * latents_std + latents_mean in float32, "
        "cast to the VAE dtype",
        decoded="RGBA, NCHW, clamped to [-1, 1] by the VAE",
        psnr="on x / 2 + 0.5 clamped to [0, 1], peak 1", **extra)
    fx.results.update(
        oracle={"device": "cpu", "threads": torch.get_num_threads(),
                "bf16_products": "bf16 operands, float32 conv, one bf16 "
                "rounding"}, sizes=results)
    fx.tolerances.update(rule=rule, fp32_max_abs=fp32_max_abs,
                         decode_fp32_max_abs=decode_max_abs, sizes=tol)
    for n in notes:
        fx.note(n)
    path = fx.write(a.out)
    print(f"[check5] wrote {path}")
    return 0


def _case(torch, name: str, fx, results: dict) -> None:
    """One further config: the fp32 decode and encoder moments, and the
    oracle's own fp32 error against float64."""
    from diffusers import AutoencoderKLQwenImage21

    config = tiny.case_config(name)
    vae = AutoencoderKLQwenImage21(**config).eval()
    want = {k: tuple(v.shape) for k, v in vae.state_dict().items()}
    if want != tiny.shapes(config):
        raise SystemExit(f"error: the {name} shapes differ from the class")
    weights = tiny.weights(config)
    vae.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()},
                        strict=True)
    vae64 = copy.deepcopy(vae).double().eval()
    inputs = tiny.case_inputs(name)
    lat, image = map(torch.from_numpy, inputs)
    z_dim = config["z_dim"]
    l_mean = torch.tensor(vae.config.latents_mean).view(1, z_dim, 1, 1)
    l_std = torch.tensor(vae.config.latents_std).view(1, z_dim, 1, 1)
    z = lat * l_std + l_mean

    def run(v, dt):
        dec = v.decode(z.to(dt).unsqueeze(2), return_dict=False)[0]
        dist = v.encode(image.to(dt).unsqueeze(2)).latent_dist
        return [t[:, :, 0].numpy()
                for t in (dec, dist.mean, dist.logvar)]

    got, ref = run(vae, torch.float32), run(vae64, torch.float64)
    for key, arr in zip(("decoded", "mean", "logvar"), got):
        fx.add(f"{key}_{name}", arr)
    results[name] = {
        "config": config, "size": list(tiny.CASES[name][1]),
        "tensors": len(want),
        "weights_sha256": tiny.digest(weights[k] for k in sorted(weights)),
        "inputs_sha256": tiny.digest(inputs),
        "fp32_vs_fp64_max_abs": {
            k: fixtures.max_abs(g, r)
            for k, g, r in zip(("decoded", "mean", "logvar"), got, ref)},
        "decode_clamped_share": float(np.mean(np.abs(got[0]) == 1.0)),
        "logvar_range": [float(got[2].min()), float(got[2].max())]}
    print(f"[check5] case {name}: {results[name]}", flush=True)


def self_test_cases(torch, a) -> int:
    fx = fixtures.FixtureWriter(
        CASES_NAME, check=5, family="qwen_image21",
        script=f"scripts/image_oracle/{Path(__file__).name}",
        argv=[Path(__file__).name, "--self-test"])
    results = {}
    for name in tiny.CASES:
        _case(torch, name, fx, results)
    fx.inputs.update(
        cases=list(tiny.CASES), weights_seed=tiny.SEED,
        out_gain=tiny.OUT_GAIN,
        weights="qwen_image21_vae_tiny.weights(config) per case",
        inputs="qwen_image21_vae_tiny.case_inputs(name): model space "
        "latents and an RGBA image, NCHW",
        decoded="the fp32 decode of latents * latents_std + latents_mean, "
        "RGBA, NCHW, clamped to [-1, 1] by the VAE",
        moments="the mean and the log variance of the latent distribution "
        "of the RGBA image, the log variance clamped to [-30, 20]")
    fx.results.update(
        oracle={"device": "cpu", "threads": torch.get_num_threads()},
        cases=results)
    fx.tolerances.update(rule=TINY_RULE, fp32_max_abs=TINY_FP32_MAX_ABS)
    fx.note("Each model is the official class at a tiny config with seeded "
            "random weights, so the numbers say nothing about image "
            "quality.")
    path = fx.write(a.out)
    print(f"[check5] wrote {path}")
    return 0


def self_test(torch, a) -> int:
    from diffusers import AutoencoderKLQwenImage21
    from safetensors.torch import load_file

    self_test_cases(torch, a)
    vae32 = AutoencoderKLQwenImage21(**tiny.TINY).eval()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "diffusion_pytorch_model.safetensors"
        tiny.write_safetensors(path, tiny.weights())
        state = load_file(str(path))
        want = {k: tuple(v.shape) for k, v in vae32.state_dict().items()}
        if want != tiny.shapes(tiny.TINY):
            raise SystemExit("error: the tiny shapes differ from the class")
        vae32.load_state_dict(state, strict=True)
        vae32 = vae32.to(torch.float32).eval()
        return _write(
            torch, a, TINY_NAME, vae32, [TINY_SIZE], {"vae": path},
            [Path(__file__).name, "--self-test"],
            {"config": tiny.TINY, "weights_seed": tiny.SEED,
             "weights": "qwen_image21_vae_tiny.weights(), written by its "
             "write_safetensors", "out_gain": tiny.OUT_GAIN},
            ["The model is the official class at the tiny config with "
             "seeded random weights, so the numbers say nothing about "
             "image quality. The weights are float32, and the bf16 VAE is "
             "a bfloat16 copy."],
            TINY_FP32_MAX_ABS, TINY_RULE + "; " + BF16_RULE)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path,
                    help="local copy of Qwen/Qwen-Image-2.1 with "
                         "vae/config.json and its safetensors")
    ap.add_argument("--config-revision", default=REVISION,
                    help="revision of the --config copy, for the manifest")
    ap.add_argument("--sizes", type=int, nargs="+", default=list(SIZES),
                    help="square sizes in pixels")
    ap.add_argument("--self-test", action="store_true",
                    help="write the tiny random model fixture and stop")
    ap.add_argument("--out", type=Path,
                    help="fixture directory (default $GMLX_IMAGE_FIXTURES)")
    a = ap.parse_args(argv)
    torch = oracle.setup()
    if a.self_test:
        return self_test(torch, a)
    if a.config is None:
        ap.error("--config is required without --self-test")
    if any(s % 16 for s in a.sizes):
        ap.error("sizes must be multiples of 16")
    from diffusers import AutoencoderKLQwenImage21

    vae32 = AutoencoderKLQwenImage21.from_pretrained(
        str(a.config / "vae"), torch_dtype=torch.float32,
        local_files_only=True).eval()
    return _write(
        torch, a, NAME, vae32, [(s, s) for s in a.sizes],
        {"vae": a.config / "vae" / "diffusion_pytorch_model.safetensors",
         "vae_config": a.config / "vae" / "config.json"},
        [Path(__file__).name, "--config", a.config.name, "--sizes",
         *map(str, a.sizes)],
        {"config_revision": a.config_revision},
        ["The official weights are float32, and the bf16 VAE is a "
         "bfloat16 copy of the fp32 one."], STARTING, RULE,
        DECODE_FP32_MAX_ABS)


if __name__ == "__main__":
    sys.exit(main())
