#!/usr/bin/env python3
"""Check 6 oracle for Z-Image: a whole sampling run on injected noise.

The run follows ZImagePipeline.__call__ at guidance 1: the prompt through
the text encoder, every step of the DiT with the scheduler's Euler update
in float32, and the VAE decode of the final latents. It runs once in fp32
and once in bf16, each with its own text encoder, DiT and VAE dtype, from
the same noise. The noise is numpy's PCG64 standard normal for the seed,
which is how gmlx makes noise from a seed, so the gmlx side can check
that its seed gives the same noise.

For each size the fixture holds the noise, the caption features of both
modes, the final latents of both modes and both decoded images as
(H, W, 3) float32 in [0, 1]. The manifest records the bf16 against fp32
gap of the image and sets the PSNR gate at twice the oracle's own RMS
error, which is its PSNR less 20 log10(2) dB.

The text encoder loads from its GGUF through transformers, which
dequantizes with gguf-py's float32 reference, and the DiT through
diffusers' from_single_file with float32 block arithmetic. In bf16 the
linear and conv products run as float32 products over bfloat16 operands
with one bfloat16 rounding.

  python scripts/image_oracle/run.py --python <oracle python> --threads 16 \\
      check6_z_image.py --dit z-image-turbo-Q4_K_M.gguf \\
      --encoder Qwen3-4B-Q4_K_M.gguf --config <official repo dir>
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

NAME = "check6_z_image_turbo"
PROMPT = ("A red bicycle leaning against a weathered brick wall at sunset, "
          "a cat asleep in its basket, warm light, 35mm photograph")
SIZES = (512, 1024)
STEPS = 8
SEED = 6
STARTING = {"image_psnr_db": 35.0}
RULE = ("image PSNR at twice the oracle's own bf16 against fp32 RMS error, "
        "which is its PSNR less 20 log10(2) dB, on (H, W, 3) images in "
        "[0, 1] with peak 1")


def emulate_bf16(torch, enable: bool) -> None:
    """Swap F.conv2d and F.linear for float32 products over bfloat16
    operands with one bfloat16 rounding of the result, or restore them."""
    import torch.nn.functional as F

    native = getattr(F, "_check6_native", None)
    if native is None:
        native = F._check6_native = {"conv2d": F.conv2d, "linear": F.linear}
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


def encode(torch, a, mode: str) -> np.ndarray:
    """Caption features (L, 2560) as float32, from the text encoder run in
    ``mode`` through the pipeline's encode_prompt."""
    from diffusers import FlowMatchEulerDiscreteScheduler, ZImagePipeline
    from transformers import AutoConfig, AutoTokenizer, Qwen3Model

    dt = oracle.torch_dtype(mode)
    tok = AutoTokenizer.from_pretrained(str(a.config / "tokenizer"),
                                        local_files_only=True)
    cfg = AutoConfig.from_pretrained(str(a.config / "text_encoder"),
                                     local_files_only=True)
    te = Qwen3Model.from_pretrained(
        str(a.encoder.parent), gguf_file=a.encoder.name, config=cfg,
        torch_dtype=dt, device_map="cpu", local_files_only=True).eval()
    where = {str(p.device) for p in te.parameters()}
    kinds = {p.dtype for p in te.parameters()}
    if where != {"cpu"} or kinds != {dt}:
        raise SystemExit(f"error: {mode} text encoder on {where} in {kinds}")
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(a.config), subfolder="scheduler", local_files_only=True)
    pipe = ZImagePipeline(scheduler=sched, vae=None, text_encoder=te,
                          tokenizer=tok, transformer=None)
    emulate_bf16(torch, mode == "bf16")
    embeds, _ = pipe.encode_prompt(a.prompt, device=torch.device("cpu"),
                                   do_classifier_free_guidance=False)
    emulate_bf16(torch, False)
    return oracle.to_numpy(embeds[0])


def scheduler(torch, config: Path, steps: int, image_seq_len: int):
    """The Euler scheduler as ZImagePipeline sets it for ``steps``."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.z_image.pipeline_z_image import (
        calculate_shift, get_default_z_image_sigmas, retrieve_timesteps)

    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(config), subfolder="scheduler", local_files_only=True)
    c = sched.config
    mu = calculate_shift(image_seq_len, c.get("base_image_seq_len", 256),
                         c.get("max_image_seq_len", 4096),
                         c.get("base_shift", 0.5), c.get("max_shift", 1.15))
    timesteps, _ = retrieve_timesteps(
        sched, steps, torch.device("cpu"),
        sigmas=get_default_z_image_sigmas(steps), mu=mu)
    sched.set_begin_index(0)
    return sched, timesteps


def sample(torch, a, model, noise: np.ndarray, cap: np.ndarray,
           mode: str) -> np.ndarray:
    """The pipeline's denoise loop at guidance 1, float32 latents."""
    dt = oracle.torch_dtype(mode)
    h, w = noise.shape[2:]
    sched, timesteps = scheduler(torch, a.config, a.steps,
                                 (h // 2) * (w // 2))
    lat = torch.from_numpy(noise.copy())
    caps = [torch.from_numpy(cap).to(dt)]
    emulate_bf16(torch, mode == "bf16")
    for t in timesteps:
        t_model = (1000 - t.expand(1)) / 1000
        x = list(lat.to(dt).unsqueeze(2).unbind(dim=0))
        out = model(x, t_model, caps, return_dict=False)[0]
        v = -torch.stack([o.float() for o in out], dim=0).squeeze(2)
        lat = sched.step(v.to(torch.float32), t, lat, return_dict=False)[0]
    emulate_bf16(torch, False)
    return oracle.to_numpy(lat)


def decode(torch, vae, latents: np.ndarray, mode: str) -> np.ndarray:
    """The pipeline's decode and postprocess, as (H, W, 3) in [0, 1]."""
    cfg = vae.config
    emulate_bf16(torch, mode == "bf16")
    z = torch.from_numpy(latents).to(oracle.torch_dtype(mode))
    z = z / cfg.scaling_factor + cfg.shift_factor
    img = vae.decode(z, return_dict=False)[0]
    emulate_bf16(torch, False)
    img = oracle.to_numpy(img)[0].transpose(1, 2, 0)
    return np.clip(img / 2 + 0.5, 0.0, 1.0).astype(np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dit", required=True, type=Path,
                    help="Z-Image-Turbo DiT GGUF")
    ap.add_argument("--encoder", required=True, type=Path,
                    help="Qwen3-4B text GGUF")
    ap.add_argument("--config", required=True, type=Path,
                    help="local copy of Tongyi-MAI/Z-Image-Turbo")
    ap.add_argument("--config-revision",
                    default="f332072aa78be7aecdf3ee76d5c247082da564a6",
                    help="revision of the --config copy, for the manifest")
    ap.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--name", default=NAME,
                    help="fixture name, for a ladder run apart from the "
                         "production fixture")
    ap.add_argument("--out", type=Path,
                    help="fixture directory (default $GMLX_IMAGE_FIXTURES)")
    a = ap.parse_args(argv)
    if any(s % 16 for s in a.sizes):
        ap.error("sizes must be multiples of 16")
    torch = oracle.setup()
    from diffusers import AutoencoderKL

    t0 = time.perf_counter()
    caps = {mode: encode(torch, a, mode) for mode in oracle.MODES}
    encode_s = round(time.perf_counter() - t0, 1)
    models = {mode: oracle.load_dit(a.dit, a.config, mode)
              for mode in oracle.MODES}
    vae32 = AutoencoderKL.from_pretrained(
        str(a.config / "vae"), torch_dtype=torch.float32,
        local_files_only=True).eval()
    vaes = {"fp32": vae32,
            "bf16": copy.deepcopy(vae32).to(torch.bfloat16).eval()}
    c_in = models["fp32"].config.in_channels

    fx = fixtures.FixtureWriter(
        a.name, check=6, family="z_image",
        script=f"scripts/image_oracle/{Path(__file__).name}",
        argv=[Path(__file__).name, "--dit", a.dit.name, "--encoder",
              a.encoder.name, "--config", a.config.name, "--sizes",
              *map(str, a.sizes)])
    fx.add_file("dit", a.dit)
    fx.add_file("encoder", a.encoder)
    fx.add_file("vae", a.config / "vae" / "diffusion_pytorch_model.safetensors")
    for mode, cap in caps.items():
        fx.add(f"cap_feats_{mode}", cap)
    results, tol = {}, {}
    for size in a.sizes:
        rng = np.random.Generator(np.random.PCG64(a.seed))
        noise = rng.standard_normal((1, c_in, size // 8, size // 8),
                                    dtype=np.float32)
        lat, img, secs = {}, {}, {}
        for mode in oracle.MODES:
            t0 = time.perf_counter()
            lat[mode] = sample(torch, a, models[mode], noise, caps[mode],
                               mode)
            secs[f"{mode}_sample_s"] = round(time.perf_counter() - t0, 1)
            t0 = time.perf_counter()
            img[mode] = decode(torch, vaes[mode], lat[mode], mode)
            secs[f"{mode}_decode_s"] = round(time.perf_counter() - t0, 1)
        fx.add(f"noise_{size}", noise)
        for mode in oracle.MODES:
            sfx = "" if mode == "fp32" else "_bf16"
            fx.add(f"latents_{size}{sfx}", lat[mode])
            fx.add(f"image_{size}{sfx}", img[mode])
        gap = fixtures.compare(img["bf16"], img["fp32"])
        gap["psnr_db"] = fixtures.psnr(img["bf16"], img["fp32"], 1.0)
        results[str(size)] = {
            "image_bf16_vs_fp32": gap,
            "latents_bf16_vs_fp32": fixtures.compare(lat["bf16"],
                                                     lat["fp32"]),
            "seconds": secs}
        tol[str(size)] = {"image_psnr_db": gap["psnr_db"]
                          - 20 * math.log10(2)}
        print(f"[check6] {size}: bf16 image PSNR {gap['psnr_db']:.2f} dB, "
              f"latents rel L2 "
              f"{results[str(size)]['latents_bf16_vs_fp32']['rel_l2']:.3e}; "
              f"{secs}", flush=True)

    cap_gap = fixtures.compare(caps["bf16"], caps["fp32"])
    fx.inputs.update(
        sizes=list(a.sizes), steps=a.steps, seed=a.seed, prompt=a.prompt,
        guidance=1.0, config_revision=a.config_revision,
        noise="numpy Generator(PCG64(seed)).standard_normal(shape, "
              "dtype=float32), NCHW, shape (1, 16, size / 8, size / 8)",
        caption="pipeline encode_prompt, hidden_states[-2], masked rows",
        image="vae.decode((latents / scaling) + shift) in the mode's "
              "dtype, x / 2 + 0.5 clamped to [0, 1], (H, W, 3)")
    fx.results.update(
        oracle={"device": "cpu", "threads": torch.get_num_threads(),
                "dequant": oracle.dequant_mode(),
                "bf16_products": "bf16 operands, float32 conv and GEMM, "
                "one bf16 rounding"},
        encode_seconds=encode_s, caption_bf16_vs_fp32=cap_gap,
        sizes=results)
    fx.tolerances.update(rule=RULE, starting=STARTING, sizes=tol)
    path = fx.write(a.out)
    print(f"[check6] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
