#!/usr/bin/env python3
"""Check 5 oracle for MiniMax-H3-Image: the image VAE's decode and encode.

ComfyUI's MiniMaxH3ImageVAE runs on the CPU on Comfy-Org's file as
published, which holds no latent statistics, so the class takes ComfyUI's
constants as sd.py does. It runs in fp32, in float16 and in float64, each
mode its own model loaded from the file, one after another. The float16
pass computes every linear, convolution, norm and attention in float32
over the float16 operands with one float16 rounding, as gmlx computes them
on the GPU (scripts/video_oracle/check5_h3.py, whose helpers this reuses).

The cases, named width x height: 256x256, one tile; 512x320, 3x2 tiles;
1024x1024, 5x5 tiles, a row of five a decoder call of up to four. Each
image is h3_vae_tiny.clip(1, height, width, SEED). The class always
tiles: 256 px tiles overlapping by at least 64, decoded tiles blended
with the tiles above and to the left as those were blended, encoded tiles
blended as encoded.

Every encode is the class's: the moments of its tiled encode, captured as
it runs, give the mean and the log variance clamped to [-30, 20], and its
output is the posterior sample of those with its own draws, seed 42 over
the whole latent, normalized by the latent statistics. The fixture holds
the draws. Every decode takes the fp32 pass's sampled latents in the
mode's dtype and returns float32 pixels in [0, 1].

Every gate reads against the float64 result at twice the oracle's own
error there: fp32 and a float16 encode by relative L2, a float16 decode by
PSNR at the oracle's own PSNR less 20 log10(2) dB.

--self-test runs the same passes on ComfyUI's class at a tiny size with
seeded weights written in the published layout (h3_vae_tiny), the
encoder at h3_vae_tiny.VIDEO and a 2D ViT decoder of 2 layers and 2
heads, on a 128x128 image, one tile, and a 464x32 image, three tiles
wide with crossing overlaps, and on seeded latents of 17x17, 2x2 tiles, where each tile
blends with neighbors already blended.

  python scripts/image_oracle/run.py --python <ComfyUI venv python> \\
      --threads 16 check5_h3_image_vae.py --comfy-dir <ComfyUI checkout> \\
      --vae minimax_h3_image_vae_fp16.safetensors
  python scripts/image_oracle/run.py --python <ComfyUI venv python> \\
      check5_h3_image_vae.py --comfy-dir <ComfyUI checkout> --self-test
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import fixtures
import oracle

ORACLE_NEEDS = ("torch", "einops", "comfy_kitchen", "safetensors")

FAMILY = "h3_image"
SCRIPT = "scripts/image_oracle/check5_h3_image_vae.py"
NAME = "check5_h3_image_vae"
TINY_NAME = "selftest_h3_image_vae"
SEED = 5
ENCODE_SEED = 42
LOGVAR_RANGE = (-30.0, 20.0)
TWICE_DB = 20 * math.log10(2)
STATS = ("latents_mean", "latents_std")
RULE = ("against the float64 result: fp32 relative L2 at twice the "
        "oracle's own fp32 relative L2; float16 decode PSNR at the "
        "oracle's own float16 PSNR less 20 log10(2) dB; float16 encode "
        "relative L2 at twice the oracle's own float16 relative L2")
OFFICIAL = {"images": {"256x256": (256, 256), "512x320": (320, 512),
                       "1024x1024": (1024, 1024)},
            "latents": {}}
SELF_TEST = {"images": {"128x128": (128, 128), "464x32": (32, 464)},
             "latents": {"17x17": (17, 17)}}
TINY_DECODER = {"num_layers": 2, "heads": 2, "dim_head": 64}
_VIDEO = Path(__file__).resolve().parent.parent / "video_oracle"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if str(_VIDEO) not in sys.path:
    sys.path.append(str(_VIDEO))
import h3_vae_tiny as tiny  # noqa: E402

video = _module("_h3_check5_video", _VIDEO / "check5_h3.py")
comfy_oracle = _module("_h3_comfy_oracle", _VIDEO / "oracle.py")
MODES = video.MODES


def load(torch, build, path: Path, mode: str):
    """``build()`` in ``mode`` with every tensor of the file copied in, and
    the latent statistics, which the file lacks, from ComfyUI's
    constants as sd.py sets them."""
    from comfy.ldm.minimax import vae as cv
    from safetensors import safe_open

    dt = getattr(torch, MODES[mode])
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dt)
    try:
        model = build()
    finally:
        torch.set_default_dtype(prev)
    model = model.to(dt).eval()
    sd = model.state_dict()
    with safe_open(str(path), "pt") as f:
        names = set(f.keys())
        if names != set(sd) - set(STATS):
            raise SystemExit(f"error: {path.name} and the class differ in "
                             f"{sorted(names ^ (set(sd) - set(STATS)))[:8]}")
        for k in names:
            sd[k].copy_(f.get_tensor(k))
    sd["latents_mean"].copy_(torch.tensor(cv.LATENTS_MEAN))
    sd["latents_std"].copy_(torch.tensor(cv.LATENTS_STD))
    return model


def plan(vae, height: int, width: int) -> dict:
    return {"tiles_y": dict(zip(("starts", "lengths", "overlaps"),
                                vae.split_tiles(height))),
            "tiles_x": dict(zip(("starts", "lengths", "overlaps"),
                                vae.split_tiles(width)))}


def run_vae(torch, build, path, cases, out: dict, times: dict) -> dict:
    plans = {}
    for mode in MODES:
        video.functional(torch, mode)
        dt = getattr(torch, MODES[mode])
        vae = load(torch, build, path, mode)
        real = vae._adaptive_encode
        held = {}

        def spy(x):
            held["moments"] = real(x)
            return held["moments"]
        vae._adaptive_encode = spy
        for tag, (h, w) in cases["images"].items():
            if mode == "fp32":
                plans[tag] = plan(vae, h, w)
            x = torch.from_numpy(out[f"input_{tag}"]).to(dt)
            t0 = time.perf_counter()
            out[f"latents_{mode}_{tag}"] = video._np(vae.encode(x), mode)
            times[f"{mode} encode {tag}"] = round(time.perf_counter() - t0, 1)
            mean, logvar = held.pop("moments").chunk(2, dim=1)
            out[f"mean_{mode}_{tag}"] = video._np(mean, mode)
            out[f"logvar_{mode}_{tag}"] = video._np(
                logvar.clamp(*LOGVAR_RANGE), mode)
            if mode == "fp32":
                out[f"noise_{tag}"] = torch.randn(
                    mean.shape, generator=torch.Generator().manual_seed(
                        ENCODE_SEED)).numpy()
        for tag in [*cases["images"], *cases["latents"]]:
            z = torch.from_numpy(out[f"latents_fp32_{tag}"]).to(dt)
            t0 = time.perf_counter()
            out[f"decoded_{mode}_{tag}"] = video._np(vae.decode(z), mode)
            times[f"{mode} decode {tag}"] = round(time.perf_counter() - t0, 1)
        times[f"{mode} peak rss gib"] = video._rss_gib()
        print(f"[check5] {mode}: "
              f"{ {k: v for k, v in times.items() if k.startswith(mode)} }",
              flush=True)
        del vae
    video.functional(torch, "fp32")
    for tag, (h, w) in cases["latents"].items():
        plans[tag] = {"seeded": True, "latent_shape": [h, w]}
    return plans


def gates(cases, out: dict) -> tuple[dict, dict]:
    res, tol = {}, {}

    def item(base, tag, psnr):
        a32, a16, a64 = (out[f"{base}_{m}_{tag}"] for m in MODES)
        g = {"fp32_vs_f64": fixtures.compare(a32, a64),
             "f16_vs_f64": fixtures.compare(a16, a64)}
        t = {"fp32_rel_l2": 2 * g["fp32_vs_f64"]["rel_l2"]}
        if psnr:
            db = fixtures.psnr(a16, a64, 1.0)
            g["f16_vs_f64"]["psnr_db"] = db
            t["f16_psnr_db"] = db - TWICE_DB
        else:
            t["f16_rel_l2"] = 2 * g["f16_vs_f64"]["rel_l2"]
        return g, t

    for tag in cases["images"]:
        res[f"encode {tag}"], tol[f"encode {tag}"] = item("latents", tag,
                                                          False)
        res[f"mean {tag}"], tol[f"mean {tag}"] = item("mean", tag, False)
    for tag in [*cases["images"], *cases["latents"]]:
        res[f"decode {tag}"], tol[f"decode {tag}"] = item("decoded", tag,
                                                          True)
    for k, g in res.items():
        parts = [f"{n} rel L2 {v['rel_l2']:.3e} max abs {v['max_abs']:.3e}"
                 + (f" PSNR {v['psnr_db']:.2f} dB" if "psnr_db" in v else "")
                 for n, v in g.items()]
        print(f"[check5] {k}: {'; '.join(parts)}", flush=True)
    return res, tol


def run(torch, a, name: str, cases: dict, path: Path, build, extra: dict,
        notes: list) -> int:
    out: dict = {}
    for tag, (h, w) in cases["images"].items():
        out[f"input_{tag}"] = tiny.clip(1, h, w, SEED)
    rng = np.random.default_rng(SEED)
    for tag, (h, w) in cases["latents"].items():
        out[f"latents_fp32_{tag}"] = rng.standard_normal(
            (1, 24, 1, h, w)).astype(np.float32)
    times: dict = {}
    plans = run_vae(torch, build, path, cases, out, times)
    res, tol = gates(cases, out)

    from comfy.ldm.minimax import vae as cv

    fx = fixtures.FixtureWriter(name, check=5, family=FAMILY, script=SCRIPT)
    fx.add_file("vae", path)
    # The float64 results are the reference of every gate; the fp32
    # latents are the decode inputs.
    keep = [k for k in out if "_f16_" not in k
            and not ("_fp32_" in k and not k.startswith("latents_"))]
    for k in keep:
        fx.add(k, out[k])
    fx.inputs.update(
        cases={k: {t: list(v) for t, v in c.items()}
               for k, c in cases.items()},
        seed=SEED, encode_seed=ENCODE_SEED, plans=plans,
        latent_stats={k: torch.tensor(getattr(cv, k.upper())).tolist()
                      for k in STATS},
        image="h3_vae_tiny.clip(1, height, width, seed): (1, 3, 1, H, W) "
        "in [-1, 1]",
        seeded_latents="numpy default_rng(seed).standard_normal((1, 24, 1, "
        "h, w)) in float32, drawn in the order of cases['latents']",
        latents="the class's encode: the posterior sample mean + exp(0.5 "
        "logvar) noise, normalized, (1, 24, 1, h, w)",
        mean="the mean of the class's moments, (1, 24, 1, h, w)",
        logvar="the log variance of its moments clamped to [-30, 20]",
        noise="torch.randn(mean shape, generator=torch.Generator()"
        ".manual_seed(42)), the class's draws",
        decoded="the class's decode: (1, 3, 1, H, W) float32 in [0, 1]",
        **extra)
    fx.results.update(
        oracle={"device": "cpu", "threads": torch.get_num_threads(),
                "comfyui": a.comfy_head, "f16_math": video.F16_MATH},
        gaps=res, times=times)
    fx.tolerances.update(rule=RULE, items=tol)
    for n in notes:
        fx.note(n)
    p = fx.write(a.out, compressed=a.self_test)
    print(f"[check5] wrote {p}")
    return 0


def _build(tiny_model: bool):
    import comfy.ops
    from comfy.ldm.minimax import vae as cv

    ops = comfy.ops.disable_weight_init
    if not tiny_model:
        return lambda: cv.MiniMaxH3ImageVAE(operations=ops)

    class TinyImageVAE(cv.MiniMaxH3ImageVAE):
        def __init__(self):
            cv.MiniMaxH3VideoVAE.__init__(self, still_frame=0, decoder=False,
                                          operations=ops, **tiny.VIDEO)
            self.post_quant_conv = ops.Conv2d(24, 24, 1)
            self.decoder = cv.ViT3DDecoder(
                patch_size=self.vae_ratio, patch_size_t=1, rope_dim_ratio=0.5,
                rope_dims=2, operations=ops, **TINY_DECODER)
    return TinyImageVAE


def self_test(torch, a) -> int:
    build = _build(True)
    shapes = {k: list(v.shape) for k, v in build().state_dict().items()
              if k not in STATS}
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "vae.safetensors"
        digest = tiny.write_safetensors(path, tiny.weights(shapes))
        return run(torch, a, TINY_NAME, SELF_TEST, path, build,
                   {"model": {**tiny.VIDEO, "decoder": TINY_DECODER},
                    "weights_seed": tiny.SEED, "shapes": shapes,
                    "file_sha256": digest,
                    "weights": "h3_vae_tiny.weights(shapes) written by "
                    "h3_vae_tiny.write_safetensors as F32"},
                   ["The model is ComfyUI's class at a tiny size with "
                    "seeded weights, so the numbers say nothing about "
                    "image quality."])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--comfy-dir", type=Path, required=True,
                    help="ComfyUI checkout")
    ap.add_argument("--vae", type=Path,
                    help="minimax_h3_image_vae_fp16.safetensors")
    ap.add_argument("--self-test", action="store_true",
                    help="write the tiny seeded model fixture and stop")
    ap.add_argument("--out", type=Path,
                    help="fixture directory (default $GMLX_IMAGE_FIXTURES)")
    a = ap.parse_args(argv)
    if not a.self_test and a.vae is None:
        ap.error("--vae is required without --self-test")
    torch = oracle.setup()
    a.comfy_head = comfy_oracle.start_comfy(a.comfy_dir)
    torch.set_grad_enabled(False)
    video.eager_rope()
    if a.self_test:
        return self_test(torch, a)
    return run(torch, a, NAME, OFFICIAL, a.vae, _build(False), {},
               ["The file is float16; each mode's model takes its values in "
                "its own dtype."])


if __name__ == "__main__":
    sys.exit(main())
