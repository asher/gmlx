#!/usr/bin/env python3
"""Check 5 oracle for FLUX.3 klein: the FLUX.2 VAE decode and encode.

diffusers' AutoencoderKLFlux2 runs on the CPU in fp32, in bf16 and in
float64, built from the official vae/config.json and loaded strict from
the Comfy-Org flux2-vae.safetensors, in the diffusers layout. For each
size it encodes the seeded synthetic RGB image of
check5_qwen_image21_vae.py to the mean of the latent distribution, maps
the mean to model space as Flux2KleinPipeline does, its 2x2 patchify and
then (z - running_mean) / sqrt(running_var + eps), and decodes those
latents back as the pipeline does, the inverse of both, through the VAE.

BFL's AutoEncoder of src/flux3/autoencoder.py, loaded strict from
ae.safetensors in the original layout, encodes and decodes the same
image and latents in fp32, and the manifest records its gap to the
diffusers path. The weights of the two files are compared tensor by
tensor after diffusers' convert_ldm_vae_checkpoint renames the original
layout, with the quant convs moved out of the encoder and the decoder.

The fixture holds, per size, the image, the fp32 and bf16 encoder means,
the model space latents of each, and the fp32 and bf16 decodes of the fp32
latents. The tolerances follow the rule of check5_krea2_vae.py, on the
model space latents for the encode: the fp32 decode max abs at twice the
oracle's own fp32 against float64 max abs, the fp32 encode at 1e-3 or
twice the oracle's own where that is larger, the bf16 decode PSNR at twice
the oracle's own error and the bf16 encode relative L2 at twice the
oracle's own. The bf16 run computes each conv and linear product in
float32 over bfloat16 operands, rounded once.

The torch CPU decode grows the process by about 46 times the bytes of one
full resolution activation and keeps them, 5.7 GiB at 512 pixels in fp32,
so a run at 512 and 1024 pixels with float64 peaked at 32.8 GiB on an M5
Max, and the oracle runs no larger size.

MOSS-Image reads the FLUX.2 autoencoder with a narrower decoder, from
OpenMOSS-Team/MOSS-Image-SFT: ``--family moss_image``, ``--config`` a
copy of that repository and ``--name check5_moss_image_vae`` write its
fixture. Its pipeline decodes through the same BatchNorm inverse and 2x2
unpack, which the script checks.

FLUX.2 klein reads the same autoencoder, which gmlx pulls from BFL's
FLUX.2 klein 4B repository in BF16. ``--vae`` that file, ``--config`` a
copy of that repository, ``--name check5_flux2_klein_vae`` and
``--family flux2_klein`` write its fixture, the fp32 and float64 runs
reading the bf16 weights upcast.

--self-test builds the class at a tiny config with seeded weights and
BatchNorm statistics, which the fixture holds by diffusers name, and runs
one 48x64 image through it; with ``--family moss_image`` the tiny decoder
is narrower than its encoder.

  python scripts/image_oracle/run.py --python <oracle python> --threads 16 \\
      check5_flux3_klein_vae.py --config <repo dir> --vae flux2-vae.safetensors \\
      --original ae.safetensors --reference <flux-3-image-klein>/src \\
      --sizes 512 1024
  python scripts/image_oracle/run.py --python <oracle python> \\
      check5_flux3_klein_vae.py --self-test
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import fixtures
import oracle
from check5_qwen_image21_vae import (STARTING, _rss_gib, synthetic_image,
                                     tag, to_unit)
from check5_z_image_vae import emulate_bf16

NAME = "check5_flux3_klein_vae"
TINY_NAME = "selftest_flux3_klein_vae_tiny"
SEED = 5
TINY_SIZE = (48, 64)
# Three levels at widths of 32 and 64, and the official 32 latent channels,
# which the family's VAE row requires.
TINY = {"block_out_channels": [32, 32, 64], "layers_per_block": 1,
        "latent_channels": 32, "norm_num_groups": 32,
        "down_block_types": ["DownEncoderBlock2D"] * 3,
        "up_block_types": ["UpDecoderBlock2D"] * 3}
# MOSS-Image's autoencoder: the same tree with a decoder narrower than the
# encoder at every level.
TINY_MOSS = {**TINY, "block_out_channels": [64, 64, 96],
             "decoder_block_out_channels": [32, 32, 64]}
TINY_NAMES = {"flux3_klein": TINY_NAME,
              "moss_image": "selftest_moss_image_vae_tiny"}
TINY_SEED = 29
# The BatchNorm's batch count, an int64 training counter no forward reads.
COUNT = "bn.num_batches_tracked"
TINY_FP32_MAX_ABS = 5e-5
RULE = ("fp32 decode max abs at twice the oracle's own fp32 against "
        "float64 max abs; fp32 encode max abs, on the model space latents, "
        "at 1e-3, or at twice the oracle's own fp32 against float64 max abs "
        "where that is larger; bf16 decode PSNR at twice the oracle's own "
        "bf16 against fp32 RMS error, its PSNR less 20 log10(2) dB; bf16 "
        "encode relative L2, on the model space latents, at twice the "
        "oracle's own")
TINY_RULE = ("fp32 max abs at 5e-5 on the CPU, for the decode and the "
             "model space latents of the encode of the tiny model; "
             + RULE.split("; ", 2)[2])


def pipeline_maps():
    """The pipeline's own patchify and unpatchify."""
    from diffusers.pipelines.flux2.pipeline_flux2_klein import \
        Flux2KleinPipeline as P

    return P._patchify_latents, P._unpatchify_latents


def same_maps_as_moss(torch) -> None:
    """MOSSImagePipeline decodes the inverse of the BatchNorm and then
    unpacks channel c * 4 + dy * 2 + dx to pixel (2y + dy, 2x + dx), which
    is a pixel shuffle by 2. The FLUX.2 klein pipeline's unpatchify must
    be that map, so the fixture holds for MOSS-Image."""
    _, unpatchify = pipeline_maps()
    z = torch.randn(2, 128, 3, 5, dtype=torch.float64)
    if not torch.equal(unpatchify(z), torch.nn.functional.pixel_shuffle(z, 2)):
        raise SystemExit("error: the FLUX.2 klein unpatchify is not "
                         "MOSS-Image's pixel shuffle")


def _stats(torch, vae):
    """The pipeline's BatchNorm mean and std, (1, 4C, 1, 1)."""
    mean = vae.bn.running_mean.view(1, -1, 1, 1)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1)
                     + vae.config.batch_norm_eps)
    return mean, std


def to_model(torch, vae, mean):
    """Model space latents of an encoder mean, as _encode_vae_image."""
    patchify, _ = pipeline_maps()
    m, s = _stats(torch, vae)
    return (patchify(mean) - m) / s


def to_vae(torch, vae, latents):
    """VAE space latents of model space ones, as the pipeline's decode."""
    _, unpatchify = pipeline_maps()
    m, s = _stats(torch, vae)
    return unpatchify(latents * s + m)


def _mean(vae, x):
    return vae.encode(x).latent_dist.mean


def _decode(vae, z, dt):
    return vae.decode(z.to(dt), return_dict=False)[0]


def run_size(torch, vaes, vae64, size, fx, results, tol, fp32_max_abs,
             decode_max_abs=None, bfl=None) -> None:
    """One size: encode and decode in each mode and in float64."""
    h, w = size
    image = synthetic_image(h, w, SEED)
    rgb = torch.from_numpy(image)
    out, t, rss = {}, {}, {}
    for mode, vae in vaes.items():
        emulate_bf16(torch, mode == "bf16")
        t0 = time.perf_counter()
        mean = _mean(vae, rgb.to(oracle.torch_dtype(mode)))
        out[f"mean_{mode}"] = oracle.to_numpy(mean)
        out[f"latents_{mode}"] = oracle.to_numpy(
            to_model(torch, vaes["fp32"], mean.float()))
        t[f"{mode}_encode_s"] = round(time.perf_counter() - t0, 1)
    emulate_bf16(torch, False)
    mean64 = _mean(vae64, rgb.double())
    lat64 = to_model(torch, vae64, mean64).numpy()
    rss["encode"] = _rss_gib()
    latents = torch.from_numpy(out["latents_fp32"])
    for mode, vae in vaes.items():
        emulate_bf16(torch, mode == "bf16")
        t0 = time.perf_counter()
        z = to_vae(torch, vaes["fp32"], latents)
        out[f"decoded_{mode}"] = oracle.to_numpy(
            _decode(vae, z, oracle.torch_dtype(mode)))
        t[f"{mode}_decode_s"] = round(time.perf_counter() - t0, 1)
    emulate_bf16(torch, False)
    t0 = time.perf_counter()
    dec64 = _decode(vae64, to_vae(torch, vae64, latents.double()),
                    torch.float64).numpy()
    t["float64_decode_s"] = round(time.perf_counter() - t0, 1)
    rss["decode"] = _rss_gib()

    s = tag(size)
    fx.add(f"image_{s}", image)
    fx.add(f"encode_mean_{s}", out["mean_fp32"])
    fx.add(f"encode_mean_bf16_{s}", out["mean_bf16"])
    fx.add(f"latents_{s}", out["latents_fp32"])
    fx.add(f"latents_bf16_{s}", out["latents_bf16"])
    fx.add(f"decoded_{s}", out["decoded_fp32"])
    fx.add(f"decoded_bf16_{s}", out["decoded_bf16"])
    d32, d16 = out["decoded_fp32"], out["decoded_bf16"]
    gap = fixtures.compare(d16, d32)
    gap["psnr_db"] = fixtures.psnr(to_unit(d16), to_unit(d32), 1.0)
    enc = fixtures.compare(out["latents_bf16"], out["latents_fp32"])
    enc64 = fixtures.compare(out["latents_fp32"], lat64)
    mean64g = fixtures.compare(out["mean_fp32"], mean64.numpy())
    dec64g = fixtures.compare(d32, dec64)
    rt = fixtures.psnr(to_unit(d32), to_unit(image), 1.0)
    results[s] = {
        "decode_bf16_vs_fp32": gap, "encode_bf16_vs_fp32": enc,
        "encode_fp32_vs_fp64": enc64, "encode_mean_fp32_vs_fp64": mean64g,
        "decode_fp32_vs_fp64": dec64g, "round_trip_fp32_psnr_db": rt,
        "latents_std": float(np.std(out["latents_fp32"])),
        "peak_rss_gib": rss, "seconds": t}
    if bfl is not None:
        results[s]["bfl"] = bfl_gap(torch, bfl, rgb, out)
        rss["bfl"] = _rss_gib()
    tol[s] = {
        "decode_fp32_max_abs": (decode_max_abs if decode_max_abs is not None
                                else 2 * dec64g["max_abs"]),
        "decode_bf16_psnr_db": gap["psnr_db"] - 20 * math.log10(2),
        "encode_fp32_max_abs": max(fp32_max_abs, 2 * enc64["max_abs"]),
        "encode_bf16_rel_l2": 2 * enc["rel_l2"]}
    print(f"[check5] {s}: bf16 decode PSNR {gap['psnr_db']:.2f} dB; bf16 "
          f"encode rel L2 {enc['rel_l2']:.3e}; fp32 against float64: "
          f"decode max abs {dec64g['max_abs']:.3e}, latents max abs "
          f"{enc64['max_abs']:.3e}, mean max abs {mean64g['max_abs']:.3e}; "
          f"fp32 round trip {rt:.2f} dB; resident set {rss} GiB; {t}",
          flush=True)
    print(f"[check5] {s}: tolerances {tol[s]}", flush=True)


def bfl_gap(torch, bfl, rgb, out) -> dict:
    """The fp32 encode tokens and decode of BFL's AutoEncoder against the
    diffusers path."""
    from flux3.autoencoder import latent_ids

    lat = out["latents_fp32"]
    _, c, h, w = lat.shape
    tokens, _ = bfl.encode(rgb[0])
    got = oracle.to_numpy(tokens).reshape(h, w, c).transpose(2, 0, 1)[None]
    toks = torch.from_numpy(lat[0].transpose(1, 2, 0).reshape(h * w, c))
    dec = oracle.to_numpy(bfl.decode(toks, latent_ids(h, w)))[None]
    return {"latents": fixtures.compare(got, lat),
            "decoded": fixtures.compare(dec, out["decoded_fp32"])}


def _modes(torch, vae32):
    vaes = {"fp32": vae32,
            "bf16": copy.deepcopy(vae32).to(torch.bfloat16).eval()}
    # The BatchNorm statistics stay float32, as the pipeline casts them to
    # its float32 latents.
    vaes["bf16"].bn.float()
    return vaes, copy.deepcopy(vae32).double().eval()


def write(torch, a, name, vae32, sizes, files, argv, extra, notes,
          fp32_max_abs, rule, decode_max_abs=None, weights=None, bfl=None,
          results_extra=None) -> int:
    vaes, vae64 = _modes(torch, vae32)
    fx = fixtures.FixtureWriter(
        name, check=5, family=a.family,
        script=f"scripts/image_oracle/{Path(__file__).name}", argv=argv)
    for role, path in files.items():
        fx.add_file(role, path)
    for k, v in (weights or {}).items():
        fx.add(f"weight.{k}", v)
    fx.add("bn_running_mean", oracle.to_numpy(vae32.bn.running_mean))
    fx.add("bn_running_var", oracle.to_numpy(vae32.bn.running_var))
    results, tol = {}, {}
    for size in sizes:
        run_size(torch, vaes, vae64, size, fx, results, tol, fp32_max_abs,
                 decode_max_abs, bfl)
    cfg = vae32.config
    fx.inputs.update(
        sizes=[list(s) for s in sizes], seed=SEED,
        image="synthetic_image(height, width, seed) of "
        "check5_qwen_image21_vae.py, RGB, NCHW in [-1, 1]",
        batch_norm_eps=float(cfg.batch_norm_eps),
        patch_size=list(cfg.patch_size),
        latents="Flux2KleinPipeline._patchify_latents of the encoder mean, "
        "then (z - running_mean) / sqrt(running_var + batch_norm_eps), in "
        "float32 from each mode's mean",
        decode_input="latents * sqrt(running_var + eps) + running_mean, "
        "then Flux2KleinPipeline._unpatchify_latents, in float32 from the "
        "fp32 latents, cast to the VAE dtype",
        decoded="RGB, NCHW, unclamped as the VAE returns it",
        psnr="on x / 2 + 0.5 clamped to [0, 1], peak 1", **extra)
    fx.results.update(
        oracle={"device": "cpu", "threads": torch.get_num_threads(),
                "bf16_products": "bf16 operands, float32 conv and linear, "
                "one bf16 rounding"}, sizes=results, **(results_extra or {}))
    fx.tolerances.update(rule=rule, fp32_max_abs=fp32_max_abs, sizes=tol)
    for n in notes:
        fx.note(n)
    print(f"[check5] wrote {fx.write(a.out)}", flush=True)
    return 0


def tiny_weights(torch, vae) -> dict:
    """Seeded values for every float tensor of ``vae``: norm gammas near 1,
    biases small, kernels scaled by their fan in, BatchNorm means near 0
    and variances in [0.25, 2]. float16 holds each exactly, so the fixture
    stores them in half the bytes."""
    rng = np.random.default_rng(TINY_SEED)
    out = {}
    for k, p in sorted(vae.state_dict().items()):
        if not p.is_floating_point():
            continue
        shape = tuple(p.shape)
        if k == "bn.running_mean":
            v = 0.5 * rng.standard_normal(shape)
        elif k == "bn.running_var":
            v = rng.uniform(0.25, 2.0, shape)
        elif k.endswith(".bias"):
            v = 0.05 * rng.standard_normal(shape)
        elif len(shape) == 1:
            v = 1 + 0.2 * rng.standard_normal(shape)
        else:
            fan = int(np.prod(shape[1:]))
            v = rng.standard_normal(shape) / math.sqrt(fan)
        out[k] = v.astype(np.float16)
    return out


def self_test(torch, a) -> int:
    from diffusers import AutoencoderKLFlux2

    if a.family not in TINY_NAMES:
        raise SystemExit(f"error: no tiny model for {a.family}")
    tiny = TINY_MOSS if a.family == "moss_image" else TINY
    vae32 = AutoencoderKLFlux2(**tiny).eval()
    weights = tiny_weights(torch, vae32)
    state = vae32.state_dict()
    state.update({k: torch.from_numpy(v.astype(np.float32))
                  for k, v in weights.items()})
    vae32.load_state_dict(state, strict=True)
    tracked = int(vae32.bn.num_batches_tracked)
    return write(
        torch, a, TINY_NAMES[a.family], vae32.float().eval(), [TINY_SIZE],
        {}, [Path(__file__).name, "--self-test"]
        + (["--family", a.family] if a.family != "flux3_klein" else []),
        {"config": tiny, "weights_seed": TINY_SEED,
         "weights": "every float tensor by its diffusers name, float16 "
         "values the model holds in float32, under weight.<name>",
         "num_batches_tracked": tracked},
        ["The model is the official class at a tiny config with seeded "
         "random weights and statistics, so the numbers say nothing about "
         "image quality.",
         "bn.num_batches_tracked is an int64 scalar the fixture does not "
         "hold; its value is inputs.num_batches_tracked."],
        TINY_FP32_MAX_ABS, TINY_RULE, decode_max_abs=TINY_FP32_MAX_ABS,
        weights=weights)


def load_vae(torch, config: Path, path: Path):
    from diffusers import AutoencoderKLFlux2
    from safetensors.torch import load_file

    cfg = json.loads((config / "vae" / "config.json").read_text())
    vae = AutoencoderKLFlux2.from_config(cfg)
    state = load_file(str(path))
    dtypes = sorted({str(v.dtype) for v in state.values()})
    vae.load_state_dict(state, strict=True)
    return vae.float().eval(), dtypes, state


def reference(path: Path):
    """BFL's autoencoder module and the reference's git commit."""
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(path))
    from flux3 import autoencoder

    try:
        commit = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return autoencoder, commit


def load_bfl(torch, autoencoder, path: Path):
    from safetensors.torch import load_file

    ae = autoencoder.AutoEncoder()
    state = load_file(str(path))
    ae.load_state_dict(state, strict=True)
    return ae.float().eval(), state


def same_weights(cfg: dict, original: dict, diffusers_state: dict) -> dict:
    """The original layout renamed by diffusers' convert_ldm_vae_checkpoint,
    with the quant convs moved out of the encoder and the decoder, against
    the diffusers file, tensor by tensor and bit for bit. The batch count
    of the BatchNorm, which no forward reads, is reported apart."""
    import torch
    from diffusers.loaders.single_file_utils import \
        convert_ldm_vae_checkpoint

    moved = {"encoder.quant_conv.": "quant_conv.",
             "decoder.post_quant_conv.": "post_quant_conv."}
    renamed = convert_ldm_vae_checkpoint(original, cfg)
    for k, v in original.items():
        for old, new in moved.items():
            if k.startswith(old):
                renamed[new + k[len(old):]] = v
        if k.startswith("bn."):
            renamed[k] = v

    def same(a, b) -> bool:
        a = a.reshape(b.shape).contiguous().reshape(-1)
        b = b.contiguous().reshape(-1)
        return a.dtype == b.dtype and bool(
            (a.view(torch.uint8) == b.view(torch.uint8)).all())

    differ = sorted(k for k in diffusers_state if k != COUNT and (
        k not in renamed or not same(renamed[k], diffusers_state[k])))
    return {"tensors": len(diffusers_state), "original_tensors":
            len(original), "renamed": len(renamed), "differ": differ,
            "unmatched": sorted(set(renamed) - set(diffusers_state)),
            "num_batches_tracked": {
                "vae": int(diffusers_state[COUNT]),
                "vae_original": int(renamed[COUNT])}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path,
                    help="local copy of black-forest-labs/flux-3-image-klein "
                         "with vae/config.json")
    ap.add_argument("--config-revision",
                    default="8454017f9ebaada30c998fccd9f67a80c7580f68",
                    help="revision of the --config copy, for the manifest")
    ap.add_argument("--vae", type=Path,
                    help="the VAE safetensors in the diffusers layout")
    ap.add_argument("--original", type=Path,
                    help="the same VAE in the original layout, ae.safetensors")
    ap.add_argument("--reference", type=Path,
                    help="src directory of BFL's flux-3-image-klein")
    ap.add_argument("--sizes", type=int, nargs="*", default=[512, 1024],
                    help="square sizes in pixels, encoded and decoded")
    ap.add_argument("--name", default=NAME)
    ap.add_argument("--family",
                    choices=("flux3_klein", "flux2_klein", "moss_image"),
                    default="flux3_klein",
                    help="the family the manifest records, which reads the "
                         "autoencoder")
    ap.add_argument("--self-test", action="store_true",
                    help="write the tiny random model fixture and stop")
    ap.add_argument("--out", type=Path,
                    help="fixture directory (default $GMLX_IMAGE_FIXTURES)")
    a = ap.parse_args(argv)
    torch = oracle.setup()
    if a.family == "moss_image":
        same_maps_as_moss(torch)
    if a.self_test:
        return self_test(torch, a)
    if None in (a.config, a.vae):
        ap.error("--config and --vae are required without --self-test")
    if not a.sizes or any(s % 16 for s in a.sizes):
        ap.error("--sizes needs a size, and sizes are multiples of 16")
    vae32, dtypes, state = load_vae(torch, a.config, a.vae)
    cfg = json.loads((a.config / "vae" / "config.json").read_text())
    print(f"[check5] loaded {a.vae.name}, source dtypes {dtypes}, resident "
          f"set {_rss_gib()} GiB", flush=True)
    files = {"vae": a.vae, "vae_config": a.config / "vae" / "config.json"}
    extra = {"source_dtypes": dtypes, "config_revision": a.config_revision,
             "load": "AutoencoderKLFlux2.from_config(vae/config.json), "
             "then the file strict, in float32"}
    held = ("bfloat16 weights, which the fp32 and float64 runs read "
            "upcast" if "torch.bfloat16" in dtypes else "float32 weights")
    results_extra, bfl, notes = {}, None, [
        f"The file holds {held}. The bf16 VAE is a bfloat16 copy with its "
        "BatchNorm statistics kept float32."]
    if a.original is not None:
        files["vae_original"] = a.original
        bfl_state = None
        if a.reference is not None:
            autoencoder, commit = reference(a.reference.expanduser())
            bfl, bfl_state = load_bfl(torch, autoencoder, a.original)
            extra["reference_commit"] = commit
        from safetensors.torch import load_file

        layouts = same_weights(cfg, bfl_state or load_file(str(a.original)),
                               state)
        del bfl_state
        results_extra["layouts"] = layouts
        print(f"[check5] layouts: {layouts}", flush=True)
        if layouts["differ"] or layouts["unmatched"]:
            raise SystemExit("error: the two layouts hold other weights")
        counts = layouts["num_batches_tracked"]
        notes.append("The original layout holds the same weights bit for "
                     "bit, so the fixture holds for both files. The batch "
                     "count of the BatchNorm, which no forward reads, is "
                     f"{counts['vae']} in the diffusers file and "
                     f"{counts['vae_original']} in the original one.")
    del state
    return write(
        torch, a, a.name, vae32, [(s, s) for s in a.sizes], files,
        [Path(__file__).name, "--vae", a.vae.name, "--family", a.family,
         "--sizes", *map(str, a.sizes)],
        extra, notes, STARTING, RULE, bfl=bfl, results_extra=results_extra)


if __name__ == "__main__":
    sys.exit(main())
