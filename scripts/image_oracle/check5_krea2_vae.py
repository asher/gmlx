#!/usr/bin/env python3
"""Check 5 oracle for Krea 2: the Qwen-Image VAE decode and encode.

diffusers' AutoencoderKLQwenImage runs on the CPU in fp32, in bf16 and in
float64, on the weights of a VAE file in the original Wan 2.1 layout that
diffusers' own convert_wan_vae_to_diffusers renames, loaded strict. For
each size it encodes the seeded synthetic RGB image of
check5_qwen_image21_vae.py to the mean of the latent distribution, maps
the mean to model space as the pipeline does, (mean - latents_mean) /
latents_std, and decodes those latents back as the pipeline does, latents
* latents_std + latents_mean, clamped to [-1, 1] by the VAE.

The fixture holds, per size, the image, the fp32 and bf16 encoder means,
the model space latents and the fp32, bf16 and float64 decodes. The
tolerances follow the rule of check5_qwen_image21_vae.py: the fp32 decode
max abs at twice the oracle's own fp32 against float64 max abs, the fp32
encode max abs at 1e-3 or twice the oracle's own where that is larger,
the bf16 decode PSNR at twice the oracle's own error and the bf16 encode
relative L2 at twice the oracle's own. The bf16 run computes each conv, 2D
and causal 3D, in float32 over bfloat16 operands, rounded once.

--self-test builds the class at a tiny config with seeded weights, which
the fixture holds by diffusers name, and runs one 48x64 image through it.

  python scripts/image_oracle/run.py --python <oracle python> --threads 16 \\
      check5_krea2_vae.py --vae qwen_image_vae.safetensors --sizes 512 1024
  python scripts/image_oracle/run.py --python <oracle python> \\
      check5_krea2_vae.py --self-test
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
from check5_qwen_image21_vae import (STARTING, _rss_gib, synthetic_image,
                                     tag, to_unit)

NAME = "check5_krea2_vae"
TINY_NAME = "selftest_krea2_vae_tiny"
SEED = 5
TINY_SIZE = (48, 64)
# Four levels as the official file has, at widths of a few channels.
TINY = {"base_dim": 4, "z_dim": 16, "dim_mult": [1, 2, 2, 2],
        "num_res_blocks": 1}
TINY_SEED = 23
TINY_FP32_MAX_ABS = 5e-5
RULE = ("fp32 decode max abs at twice the oracle's own fp32 against "
        "float64 max abs; fp32 encode max abs at 1e-3, or at twice the "
        "oracle's own fp32 against float64 max abs where that is larger; "
        "bf16 decode PSNR at twice the oracle's own bf16 against fp32 RMS "
        "error, its PSNR less 20 log10(2) dB; bf16 encode relative L2 at "
        "twice the oracle's own")
TINY_RULE = ("fp32 max abs at 5e-5 on the CPU, for the decode and the "
             "encode of the tiny model; " + RULE.split("; ", 2)[2])


def emulate_bf16(torch, enable: bool) -> None:
    """Swap F.conv2d and F.conv3d for a float32 conv over bfloat16
    operands with one bfloat16 rounding of the result, or restore them."""
    import torch.nn.functional as F

    native = getattr(F, "_check5_krea2_native", None)
    if native is None:
        native = F._check5_krea2_native = {"conv2d": F.conv2d,
                                           "conv3d": F.conv3d}
    for name, op in native.items():
        if not enable:
            setattr(F, name, op)
            continue

        def run(x, w, b=None, *args, _op=op, **kw):
            if x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16:
                y = _op(x.float(), w.float(),
                        None if b is None else b.float(), *args, **kw)
                return y.to(torch.bfloat16)
            return _op(x, w, b, *args, **kw)

        setattr(F, name, run)


def _mean(vae, x):
    return vae.encode(x.unsqueeze(2)).latent_dist.mean[:, :, 0]


def _decode(vae, z, dt):
    return vae.decode(z.to(dt).unsqueeze(2), return_dict=False)[0][:, :, 0]


def run_size(torch, vaes, vae64, size, fx, results, tol, fp32_max_abs,
             decode_max_abs=None) -> None:
    cfg = vaes["fp32"].config
    z_dim = cfg.z_dim
    l_mean = torch.tensor(cfg.latents_mean).view(1, z_dim, 1, 1)
    l_std = torch.tensor(cfg.latents_std).view(1, z_dim, 1, 1)
    h, w = size
    image = synthetic_image(h, w, SEED)
    rgb = torch.from_numpy(image)
    out, t, rss = {}, {}, {}
    for mode, vae in vaes.items():
        emulate_bf16(torch, mode == "bf16")
        t0 = time.perf_counter()
        out[f"mean_{mode}"] = oracle.to_numpy(
            _mean(vae, rgb.to(oracle.torch_dtype(mode))))
        t[f"{mode}_encode_s"] = round(time.perf_counter() - t0, 1)
    emulate_bf16(torch, False)
    mean64 = _mean(vae64, rgb.double()).numpy()
    rss["encode"] = _rss_gib()
    latents = (torch.from_numpy(out["mean_fp32"]) - l_mean) / l_std
    z = latents * l_std + l_mean
    for mode, vae in vaes.items():
        emulate_bf16(torch, mode == "bf16")
        t0 = time.perf_counter()
        out[f"decoded_{mode}"] = oracle.to_numpy(
            _decode(vae, z, oracle.torch_dtype(mode)))
        t[f"{mode}_decode_s"] = round(time.perf_counter() - t0, 1)
    emulate_bf16(torch, False)
    t0 = time.perf_counter()
    dec64 = _decode(vae64, z, torch.float64).numpy()
    t["float64_decode_s"] = round(time.perf_counter() - t0, 1)
    rss["decode"] = _rss_gib()

    s = tag(size)
    fx.add(f"image_{s}", image)
    fx.add(f"encode_mean_{s}", out["mean_fp32"])
    fx.add(f"encode_mean_bf16_{s}", out["mean_bf16"])
    fx.add(f"latents_{s}", oracle.to_numpy(latents))
    fx.add(f"decoded_{s}", out["decoded_fp32"])
    fx.add(f"decoded_bf16_{s}", out["decoded_bf16"])
    d32, d16 = out["decoded_fp32"], out["decoded_bf16"]
    gap = fixtures.compare(d16, d32)
    gap["psnr_db"] = fixtures.psnr(to_unit(d16), to_unit(d32), 1.0)
    enc = fixtures.compare(out["mean_bf16"], out["mean_fp32"])
    enc64 = fixtures.compare(out["mean_fp32"], mean64)
    dec64g = fixtures.compare(d32, dec64)
    rt = fixtures.psnr(to_unit(d32), to_unit(image), 1.0)
    results[s] = {
        "decode_bf16_vs_fp32": gap, "encode_bf16_vs_fp32": enc,
        "encode_fp32_vs_fp64": enc64, "decode_fp32_vs_fp64": dec64g,
        "decode_clamped_share": float(np.mean(np.abs(d32) == 1.0)),
        "round_trip_fp32_psnr_db": rt,
        "latents_std": float(np.std(oracle.to_numpy(latents))),
        "peak_rss_gib": rss, "seconds": t}
    tol[s] = {
        "decode_fp32_max_abs": (decode_max_abs if decode_max_abs is not None
                                else 2 * dec64g["max_abs"]),
        "decode_bf16_psnr_db": gap["psnr_db"] - 20 * math.log10(2),
        "encode_fp32_max_abs": max(fp32_max_abs, 2 * enc64["max_abs"]),
        "encode_bf16_rel_l2": 2 * enc["rel_l2"]}
    print(f"[check5] {s}: bf16 decode PSNR {gap['psnr_db']:.2f} dB; bf16 "
          f"encode rel L2 {enc['rel_l2']:.3e}; fp32 against float64: "
          f"decode max abs {dec64g['max_abs']:.3e}, encode max abs "
          f"{enc64['max_abs']:.3e}; fp32 round trip {rt:.2f} dB; resident "
          f"set {rss} GiB; {t}", flush=True)
    print(f"[check5] {s}: tolerances {tol[s]}", flush=True)


def _modes(torch, vae32):
    vaes = {"fp32": vae32,
            "bf16": copy.deepcopy(vae32).to(torch.bfloat16).eval()}
    return vaes, copy.deepcopy(vae32).double().eval()


def write(torch, a, name, vae32, sizes, files, argv, extra, notes,
          fp32_max_abs, rule, decode_max_abs=None, weights=None) -> int:
    vaes, vae64 = _modes(torch, vae32)
    fx = fixtures.FixtureWriter(
        name, check=5, family="krea2",
        script=f"scripts/image_oracle/{Path(__file__).name}", argv=argv)
    for role, path in files.items():
        fx.add_file(role, path)
    for k, v in (weights or {}).items():
        fx.add(f"weight.{k}", v)
    results, tol = {}, {}
    for size in sizes:
        run_size(torch, vaes, vae64, size, fx, results, tol, fp32_max_abs,
                 decode_max_abs)
    cfg = vae32.config
    fx.inputs.update(
        sizes=[list(s) for s in sizes], seed=SEED,
        image="synthetic_image(height, width, seed) of "
        "check5_qwen_image21_vae.py, RGB, NCHW in [-1, 1]",
        latents_mean=[float(v) for v in cfg.latents_mean],
        latents_std=[float(v) for v in cfg.latents_std],
        latents="(encoder mean - latents_mean) / latents_std, from the "
        "fp32 VAE",
        decode_input="latents * latents_std + latents_mean in float32, "
        "cast to the VAE dtype",
        decoded="RGB, NCHW, clamped to [-1, 1] by the VAE",
        psnr="on x / 2 + 0.5 clamped to [0, 1], peak 1", **extra)
    fx.results.update(
        oracle={"device": "cpu", "threads": torch.get_num_threads(),
                "bf16_products": "bf16 operands, float32 conv, one bf16 "
                "rounding"}, sizes=results)
    fx.tolerances.update(rule=rule, fp32_max_abs=fp32_max_abs, sizes=tol)
    for n in notes:
        fx.note(n)
    print(f"[check5] wrote {fx.write(a.out)}", flush=True)
    return 0


def tiny_weights(torch, vae) -> dict:
    """Seeded float32 weights for every tensor of ``vae``: gammas near 1,
    biases small, kernels scaled by their fan in."""
    rng = np.random.default_rng(TINY_SEED)
    out = {}
    for k, p in sorted(vae.state_dict().items()):
        shape = tuple(p.shape)
        if k.endswith(".gamma"):
            v = 1 + 0.2 * rng.standard_normal(shape)
        elif k.endswith(".bias"):
            v = 0.05 * rng.standard_normal(shape)
        else:
            fan = int(np.prod(shape[1:]))
            v = rng.standard_normal(shape) / math.sqrt(fan)
        out[k] = v.astype(np.float32)
    return out


def self_test(torch, a) -> int:
    from diffusers import AutoencoderKLQwenImage

    vae32 = AutoencoderKLQwenImage(**TINY).eval()
    weights = tiny_weights(torch, vae32)
    vae32.load_state_dict({k: torch.from_numpy(v)
                           for k, v in weights.items()}, strict=True)
    return write(
        torch, a, TINY_NAME, vae32.float().eval(), [TINY_SIZE], {},
        [Path(__file__).name, "--self-test"],
        {"config": TINY, "weights_seed": TINY_SEED,
         "weights": "every tensor by its diffusers name, float32, under "
         "weight.<name>"},
        ["The model is the official class at a tiny config with seeded "
         "random weights, so the numbers say nothing about image quality."],
        TINY_FP32_MAX_ABS, TINY_RULE, decode_max_abs=TINY_FP32_MAX_ABS,
        weights=weights)


def load_vae(torch, path: Path):
    from diffusers import AutoencoderKLQwenImage
    from diffusers.loaders.single_file_utils import \
        convert_wan_vae_to_diffusers
    from safetensors.torch import load_file

    state = load_file(str(path))
    dtypes = sorted({str(v.dtype) for v in state.values()})
    state = convert_wan_vae_to_diffusers(state)
    vae = AutoencoderKLQwenImage()
    vae.load_state_dict({k: v.float() for k, v in state.items()},
                        strict=True)
    return vae.float().eval(), dtypes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--vae", type=Path,
                    help="the VAE safetensors in the original Wan 2.1 "
                         "layout")
    ap.add_argument("--sizes", type=int, nargs="+", default=[512, 1024],
                    help="square sizes in pixels")
    ap.add_argument("--name", default=NAME)
    ap.add_argument("--self-test", action="store_true",
                    help="write the tiny random model fixture and stop")
    ap.add_argument("--out", type=Path,
                    help="fixture directory (default $GMLX_IMAGE_FIXTURES)")
    a = ap.parse_args(argv)
    torch = oracle.setup()
    if a.self_test:
        return self_test(torch, a)
    if a.vae is None:
        ap.error("--vae is required without --self-test")
    if any(s % 8 for s in a.sizes):
        ap.error("sizes must be multiples of 8")
    vae32, dtypes = load_vae(torch, a.vae)
    print(f"[check5] loaded {a.vae.name}, source dtypes {dtypes}, resident "
          f"set {_rss_gib()} GiB", flush=True)
    return write(
        torch, a, a.name, vae32, [(s, s) for s in a.sizes], {"vae": a.vae},
        [Path(__file__).name, "--vae", a.vae.name, "--sizes",
         *map(str, a.sizes)],
        {"source_dtypes": dtypes,
         "load": "convert_wan_vae_to_diffusers, then "
         "AutoencoderKLQwenImage() strict, in float32"},
        ["The file holds bfloat16 weights. The fp32 VAE holds them "
         "exactly in float32, and the bf16 VAE is a bfloat16 copy, which "
         "is the file's own values."], STARTING, RULE)


if __name__ == "__main__":
    sys.exit(main())
