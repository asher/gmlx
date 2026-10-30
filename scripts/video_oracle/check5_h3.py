#!/usr/bin/env python3
"""Check 5 oracle for MiniMax-H3: the video VAE, TAEH3 and the audio VAE.

ComfyUI's classes run on the CPU, each on its file as published:
MiniMaxH3VideoVAE on Comfy-Org's video VAE, TAEHV on madebyollin's
TAEH3 and MiniMaxH3AudioVAE on Comfy-Org's audio VAE. The video VAE and
TAEH3 run in fp32, in float16 and in float64, and the audio VAE, which
ComfyUI runs in float32 only, in fp32 and float64. Each mode is its own
model, loaded from the file, and the modes run one after another, so one
copy of the weights is resident at a time.

The cases, named width x height x frames:

- 256x256x22: one tile, two encode pieces, one decode run.
- 512x320x39: 3x2 tiles, three encode pieces, 12 latents in two decode
  runs. The manifest records the tile starts and overlaps and the chunk
  plan as the class computes them.
- 256x256: the first frame of the first clip encoded alone to one latent,
  and that latent decoded.
- TAEH3 decodes of the clips' and the image's latents. The clips give
  it two and three runs, and an image its single-latent path.
- 2 s of stereo audio plus 123 samples, encoded, and its latents decoded.

Every encode takes the synthetic input of h3_vae_tiny. Every decode takes
the fp32 pass's latents, in the mode's dtype, as ComfyUI's VAE wrapper
casts them. The class returns its encode in float32 and writes a clip's
decode into a float32 buffer, so the float64 pass's latents and clip
decodes carry one float32 rounding.

The float16 pass computes every linear, convolution, norm and attention
in float32 over the float16 operands with one float16 rounding, as gmlx
computes them on the GPU. torch's CPU conv3d unfolds its input, so every
pass runs conv3d over pieces of its output frames, each under 2 GiB.

Every gate reads against the float64 result, at twice the oracle's own
error there: fp32 and a float16 encode by relative L2, a float16 decode
by PSNR, at the oracle's own PSNR less 20 log10(2) dB. The fixture holds
the float64 results, and the fp32 latents that every decode takes.

--self-test runs the same passes on the tiny models of h3_vae_tiny with
seeded weights, written as safetensors files in the published layout.
The manifest records each model's names and shapes and each file's
sha256, so the gmlx tests write the same files.

  python scripts/video_oracle/run.py --python <comfyui venv python> \\
      --threads 16 check5_h3.py --comfy-dir <ComfyUI checkout> \\
      --vae minimax_h3_video_vae_fp16.safetensors \\
      --audio-vae minimax_h3_audio_vae_fp32.safetensors \\
      --tiny taeh3.safetensors
  python scripts/video_oracle/run.py --python <comfyui venv python> \\
      check5_h3.py --comfy-dir <ComfyUI checkout> --self-test
"""

from __future__ import annotations

import argparse
import math
import os
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import fixtures
import h3_vae_tiny as tiny
import oracle

ORACLE_NEEDS = ("torch", "einops", "comfy_kitchen", "safetensors")

FAMILY = "h3"
SCRIPT = "scripts/video_oracle/check5_h3.py"
NAME = "check5_h3"
TINY_NAME = "selftest_h3_vae"
SEED = 5
CONV_BUDGET = 2 << 30
TWICE_DB = 20 * math.log10(2)
RULE = ("against the float64 result: fp32 relative L2 at twice the "
        "oracle's own fp32 relative L2; float16 decode PSNR at the "
        "oracle's own float16 PSNR less 20 log10(2) dB; float16 encode "
        "relative L2 at twice the oracle's own float16 relative L2")
F16_MATH = ("each linear, convolution, norm and attention in float32 over "
            "float16 operands, one float16 rounding")
MODES = {"fp32": "float32", "f16": "float16", "f64": "float64"}

# What each run covers: clips as (frames, height, width), images as
# (height, width), seeded latents decoded alone as (frames, height,
# width), and the audio length in samples.
OFFICIAL = {"clips": {"256x256x22": (22, 256, 256),
                      "512x320x39": (39, 320, 512)},
            "images": {"256x256": (256, 256)},
            "latents": {}, "audio_samples": 2 * 32000 + 123}
SELF_TEST = {"clips": {"32x32x22": (22, 32, 32)},
             # Three tiles wide on encode and decode, overlaps 160 and 144
             # px, so the middle tile's overlaps cross.
             "images": {"464x32": (32, 464)},
             # Two video decode runs at 12 and 11 latents and one at 5;
             # TAEH3 runs of 5, 5 and 2 latents, of 5, 5 and 1, and of 5.
             # TAEH3 decodes every case here of more than one latent.
             # One latent at 2x2 tiles, where each tile blends with
             # neighbors already blended.
             "latents": {"12x1x1": (12, 1, 1), "11x1x1": (11, 1, 1),
                         "5x1x1": (5, 1, 1), "1x17x17": (1, 17, 17)},
             "audio_samples": 8000 + 123}

_NATIVE: dict = {}
_PATCHED = ("linear", "conv1d", "conv2d", "conv3d", "conv_transpose1d",
            "group_norm", "layer_norm", "rms_norm",
            "scaled_dot_product_attention", "pad", "interpolate")


def _rss_now() -> float:
    """This process's resident set now, in GiB."""
    import subprocess

    kib = subprocess.run(["/bin/ps", "-o", "rss=", "-p", str(os.getpid())],
                         capture_output=True, text=True).stdout
    return round(int(kib) / 2 ** 20, 2)


def _rss_gib() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                 / 2 ** 30, 2)


def _pieces(torch, native):
    """conv3d over pieces of its output frames, so torch's unfold of the
    input stays under CONV_BUDGET. Each output is the same sum."""
    def conv3d(x, w, b=None, stride=1, padding=0, dilation=1, groups=1):
        def flat(v):
            return tuple(v) if isinstance(v, (tuple, list)) else (v,) * 3
        if (x.dim() != 5 or groups != 1 or set(flat(dilation)) != {1}
                or isinstance(padding, str) or set(flat(padding)) != {0}):
            return native(x, w, b, stride, padding, dilation, groups)
        st = flat(stride)
        kt, kh, kw = w.shape[2:]
        t_out = (x.shape[2] - kt) // st[0] + 1
        hw = (((x.shape[3] - kh) // st[1] + 1)
              * ((x.shape[4] - kw) // st[2] + 1))
        step = max(1, CONV_BUDGET // (w[0].numel() * hw * x.element_size()))
        if step >= t_out:
            return native(x, w, b, stride, 0, 1, 1)
        return torch.cat([
            native(x[:, :, i * st[0]:(min(t_out, i + step) - 1) * st[0] + kt],
                   w, b, stride, 0, 1, 1)
            for i in range(0, t_out, step)], 2)
    return conv3d


def _in_float32(torch, fn):
    def up(a):
        return a.float() if isinstance(a, torch.Tensor) and \
            a.dtype == torch.float16 else a

    def run(*args, **kw):
        if not any(isinstance(a, torch.Tensor) and a.dtype == torch.float16
                   for a in (*args, *kw.values())):
            return fn(*args, **kw)
        return fn(*map(up, args), **{k: up(v) for k, v in kw.items()}).half()
    return run


def functional(torch, mode: str) -> None:
    """Set torch.nn.functional for ``mode``: conv3d in pieces always, and
    in f16 each patched op in float32 over float16 operands."""
    import torch.nn.functional as F

    if not _NATIVE:
        _NATIVE.update({n: getattr(F, n) for n in _PATCHED})
    for n in _PATCHED:
        setattr(F, n, _NATIVE[n])
    F.conv3d = _pieces(torch, _NATIVE["conv3d"])
    if mode == "f16":
        for n in _PATCHED:
            setattr(F, n, _in_float32(torch, getattr(F, n)))


def eager_rope() -> None:
    """Call comfy_kitchen's eager norm and rope directly. On the CPU its
    registry picks that function for float32 and float16 and refuses
    float64, so every mode runs the same code."""
    import comfy.quant_ops
    from comfy_kitchen.backends.eager import rope

    def rms_rope_split_half_(q, k, freqs_cis, q_scale, k_scale=None,
                             epsilon=1e-6, rot_dim=0):
        oq, ok = rope.rms_rope_split_half(q, k, freqs_cis, q_scale, k_scale,
                                          epsilon, rot_dim)
        q.copy_(oq)
        k.copy_(ok)
        return q, k

    comfy.quant_ops.ck.rms_rope_split_half_ = rms_rope_split_half_
    comfy.quant_ops.ck.rms_rope_split_half = rope.rms_rope_split_half


def load(torch, build, path: Path, mode: str):
    """``build()`` in ``mode`` with every tensor of the file at ``path``
    copied in, which must name exactly the class's state. The class
    builds in the mode's dtype, so no second copy of the weights exists."""
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
        if names != set(sd):
            raise SystemExit(f"error: {path.name} and the class differ in "
                             f"{sorted(names ^ set(sd))[:8]}")
        for k in names:
            sd[k].copy_(f.get_tensor(k))
    return model


def _np(t, mode: str) -> np.ndarray:
    t = t.detach()
    return (t.double() if mode == "f64" else t.float()).numpy()


def video_plan(vae, frames: int, height: int, width: int) -> dict:
    """The tile schedule of both axes and the chunk plan of the decode,
    as the class computes them."""
    lat = 1 if frames == 1 else (
        math.ceil(frames / vae.clip_length) * vae.tokens_chunk_size
        - vae.token_drop)
    plan = {"latents": lat,
            "tiles_y": dict(zip(("starts", "lengths", "overlaps"),
                                vae.split_tiles(height))),
            "tiles_x": dict(zip(("starts", "lengths", "overlaps"),
                                vae.split_tiles(width)))}
    if frames > 1:
        pad, runs = vae._decode_temporal_chunks(lat)
        plan.update(encode_pieces=math.ceil(frames / vae.clip_length),
                    decode_pad_latents=pad, decode_runs=runs,
                    tokens_chunk_size=vae.tokens_chunk_size,
                    token_overlap=vae.token_overlap,
                    token_drop=vae.token_drop,
                    frame_pre_padding=vae.frame_pre_padding,
                    frame_overlap=vae.frame_overlap)
    return plan


def run_video(torch, build, path, cases, out: dict, times: dict) -> dict:
    """Every mode of the video VAE: encodes of the clips and images, and
    decodes of the fp32 latents and the seeded latents."""
    plans = {}
    for mode in MODES:
        functional(torch, mode)
        dt = getattr(torch, MODES[mode])
        vae = load(torch, build, path, mode)
        times[f"{mode} video loaded rss gib"] = _rss_now()
        for tag, (f, h, w) in {**cases["clips"],
                               **{k: (1, *v)
                                  for k, v in cases["images"].items()}}.items():
            if mode == "fp32":
                plans[tag] = video_plan(vae, f, h, w)
            x = torch.from_numpy(out[f"input_{tag}"]).to(dt)
            t0 = time.perf_counter()
            out[f"latents_{mode}_{tag}"] = _np(vae.encode(x), mode)
            times[f"{mode} encode {tag}"] = round(time.perf_counter() - t0, 1)
        times[f"{mode} video encoded rss gib"] = _rss_now()
        for tag in [*cases["clips"], *cases["images"], *cases["latents"]]:
            z = torch.from_numpy(out[f"latents_fp32_{tag}"]).to(dt)
            t0 = time.perf_counter()
            out[f"decoded_{mode}_{tag}"] = _np(vae.decode(z), mode)
            times[f"{mode} decode {tag}"] = round(time.perf_counter() - t0, 1)
        times[f"{mode} video peak rss gib"] = _rss_gib()
        print(f"[check5] video {mode}: "
              f"{ {k: v for k, v in times.items() if k.startswith(mode)} }",
              flush=True)
        del vae
    functional(torch, "fp32")
    for tag, shape in cases["latents"].items():
        plans[tag] = video_plan_latents(shape)
    return plans


def video_plan_latents(shape) -> dict:
    return {"latents": shape[0], "seeded": True}


def tiny_tags(cases: dict) -> list:
    """The cases TAEH3 decodes: every clip and image, and the seeded
    latents of more than one frame. TAEH3 has no tiles, so the images
    reach its single latent."""
    return [*cases["clips"], *cases["images"],
            *(t for t, (f, _, _) in cases["latents"].items() if f > 1)]


def run_tiny(torch, build, path, cases, out: dict, times: dict) -> None:
    tags = tiny_tags(cases)
    for mode in MODES:
        functional(torch, mode)
        dt = getattr(torch, MODES[mode])
        tae = load(torch, build, path, mode)
        t0 = time.perf_counter()
        for tag in tags:
            z = torch.from_numpy(out[f"latents_fp32_{tag}"]).to(dt)
            out[f"tiny_{mode}_{tag}"] = _np(tae.decode(z), mode)
        times[f"{mode} tiny decodes"] = round(time.perf_counter() - t0, 1)
        del tae
    functional(torch, "fp32")


def run_audio(torch, build, path, out: dict, times: dict) -> None:
    for mode in ("fp32", "f64"):
        functional(torch, mode)
        dt = getattr(torch, MODES[mode])
        aud = load(torch, build, path, mode)
        t0 = time.perf_counter()
        out[f"wave_latents_{mode}"] = _np(
            aud.encode(torch.from_numpy(out["wave"]).to(dt)), mode)
        out[f"wave_decoded_{mode}"] = _np(
            aud.decode(torch.from_numpy(out["wave_latents_fp32"]).to(dt)),
            mode)
        times[f"{mode} audio"] = round(time.perf_counter() - t0, 1)
        del aud
    functional(torch, "fp32")


def _rel(a, b) -> float:
    return fixtures.compare(a, b)["rel_l2"]


def gates(cases, out: dict) -> tuple[dict, dict]:
    """The oracle's own gaps and the tolerances of every tensor."""
    res, tol = {}, {}

    def item(key, base, f16: str | None):
        g = {"fp32_vs_f64": fixtures.compare(out[f"{base}_fp32{key}"],
                                             out[f"{base}_f64{key}"])}
        t = {"fp32_rel_l2": 2 * g["fp32_vs_f64"]["rel_l2"]}
        if f16 is not None:
            a, b = out[f"{base}_f16{key}"], out[f"{base}_f64{key}"]
            g["f16_vs_f64"] = fixtures.compare(a, b)
            if f16 == "psnr":
                db = fixtures.psnr(a, b, 1.0)
                g["f16_vs_f64"]["psnr_db"] = db
                t["f16_psnr_db"] = db - TWICE_DB
            else:
                t["f16_rel_l2"] = 2 * g["f16_vs_f64"]["rel_l2"]
        return g, t

    for tag in [*cases["clips"], *cases["images"]]:
        res[f"encode {tag}"], tol[f"encode {tag}"] = item(
            f"_{tag}", "latents", "rel")
    for tag in [*cases["clips"], *cases["images"], *cases["latents"]]:
        res[f"decode {tag}"], tol[f"decode {tag}"] = item(
            f"_{tag}", "decoded", "psnr")
    for tag in tiny_tags(cases):
        res[f"tiny {tag}"], tol[f"tiny {tag}"] = item(f"_{tag}", "tiny",
                                                      "psnr")
    res["audio encode"], tol["audio encode"] = item("", "wave_latents", None)
    res["audio decode"], tol["audio decode"] = item("", "wave_decoded", None)
    for k, g in res.items():
        parts = [f"{n} rel L2 {v['rel_l2']:.3e} max abs {v['max_abs']:.3e}"
                 + (f" PSNR {v['psnr_db']:.2f} dB" if "psnr_db" in v else "")
                 for n, v in g.items()]
        print(f"[check5] {k}: {'; '.join(parts)}", flush=True)
    return res, tol


def run(torch, a, name: str, cases: dict, files: dict, builds: dict,
        extra: dict, notes: list) -> int:
    out: dict = {}
    for tag, (f, h, w) in cases["clips"].items():
        out[f"input_{tag}"] = tiny.clip(f, h, w, SEED)
    for tag, (h, w) in cases["images"].items():
        out[f"input_{tag}"] = tiny.clip(1, h, w, SEED)
    rng = np.random.default_rng(SEED)
    for tag, (f, h, w) in cases["latents"].items():
        out[f"latents_fp32_{tag}"] = rng.standard_normal(
            (1, 24, f, h, w)).astype(np.float32)
    out["wave"] = tiny.wave(cases["audio_samples"], SEED)
    times: dict = {}
    plans = run_video(torch, builds["video"], files["vae"], cases, out, times)
    run_tiny(torch, builds["tiny"], files["tiny"], cases, out, times)
    run_audio(torch, builds["audio"], files["audio_vae"], out, times)
    res, tol = gates(cases, out)

    fx = fixtures.FixtureWriter(name, check=5, family=FAMILY, script=SCRIPT)
    for role, path in files.items():
        fx.add_file(role, path)
    # The float64 results are the reference of every gate; the fp32
    # latents are also the decode inputs.
    keep = [k for k in out if "_f16" not in k and not (
        "_fp32" in k and not k.startswith(("latents_", "wave_latents")))]
    for k in keep:
        fx.add(k, out[k])
    fx.inputs.update(
        cases={k: {t: list(v) if isinstance(v, tuple) else v
                   for t, v in c.items()} if isinstance(c, dict) else c
               for k, c in cases.items()},
        seed=SEED, plans=plans,
        clip="h3_vae_tiny.clip(frames, height, width, seed): (1, 3, T, H, "
        "W) in [-1, 1]; an image is clip(1, height, width, seed)",
        wave="h3_vae_tiny.wave(samples, seed): (1, 2, n) in [-1, 1]",
        seeded_latents="numpy default_rng(seed).standard_normal((1, 24, T, "
        "H, W)) in float32, drawn in the order of cases['latents']",
        latents="the class's encode: (mean - latents_mean) / latents_std, "
        "(1, 24, T, H, W)",
        decoded="the class's decode: (1, 3, T, H, W) float32 in [0, 1]",
        tiny="TAEHV(latent_channels=24).decode of each fp32 latent: (1, 3, "
        "T, H, W) in [0, 1]",
        wave_latents="(1, 32, 2, T)", wave_decoded="(1, 2, T * 800)",
        **extra)
    fx.results.update(
        oracle={"device": "cpu", "threads": torch.get_num_threads(),
                "comfyui": a.comfy_head, "f16_math": F16_MATH},
        gaps=res, times=times)
    fx.tolerances.update(rule=RULE, items=tol)
    for n in notes:
        fx.note(n)
    path = fx.write(a.out)
    print(f"[check5] wrote {path}")
    return 0


def scaled(cases: dict, n: int) -> dict:
    """``cases`` with every side divided by ``n``, to a multiple of 16 and
    at least 32, and the audio length divided by ``n``."""
    def side(v):
        return max(32, v // n // 16 * 16)

    clips = {f"{side(w)}x{side(h)}x{f}": (f, side(h), side(w))
             for f, h, w in cases["clips"].values()}
    images = {f"{side(w)}x{side(h)}": (side(h), side(w))
              for h, w in cases["images"].values()}
    return {"clips": clips, "images": images, "latents": {},
            "audio_samples": cases["audio_samples"] // n}


def _builds(torch, tiny_models: bool) -> dict:
    import comfy.ops
    from comfy.ldm.minimax.audio_vae import MiniMaxH3AudioVAE
    from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE, ViT3DDecoder
    from comfy.taesd.taehv import TAEHV

    ops = comfy.ops.disable_weight_init

    def video():
        if not tiny_models:
            return MiniMaxH3VideoVAE(operations=ops)
        vae = MiniMaxH3VideoVAE(**tiny.VIDEO, operations=ops)
        vae.decoder = ViT3DDecoder(**tiny.VIDEO_DECODER, operations=ops)
        return vae

    def audio():
        return MiniMaxH3AudioVAE(**(tiny.AUDIO if tiny_models else {}))

    return {"video": video, "audio": audio,
            "tiny": lambda: TAEHV(latent_channels=24)}


def plan_tables(vae) -> dict:
    """The class's tile schedule for every side from 16 to 2048 pixels in
    steps of 16, and its decode chunk plan for 1 to 92 latents: the pad
    latents, the runs and the frames out."""
    tiles = {str(n): dict(zip(("starts", "lengths", "overlaps"),
                              vae.split_tiles(n)))
             for n in range(16, 2049, 16)}
    chunks = {}
    for n in range(1, 93):
        pad, runs = vae._decode_temporal_chunks(n) if n > 1 else (0, 1)
        chunks[str(n)] = {"pad": pad, "runs": runs,
                          "frames": vae.decode_output_shape(
                              (1, 24, n, 1, 1))[2]}
    return {"tiles": tiles, "chunks": chunks}


def self_test(torch, a) -> int:
    builds = _builds(torch, True)
    files, shapes, digests = {}, {}, {}
    with tempfile.TemporaryDirectory() as d:
        for role, key in (("vae", "video"), ("tiny", "tiny"),
                          ("audio_vae", "audio")):
            sd = builds[key]().state_dict()
            shapes[role] = {k: list(v.shape) for k, v in sd.items()}
            files[role] = Path(d) / f"{role}.safetensors"
            digests[role] = tiny.write_safetensors(
                files[role], tiny.weights(shapes[role]))
        return run(torch, a, TINY_NAME, SELF_TEST, files, builds,
                   {"models": {"video": {**tiny.VIDEO,
                                         "decoder": tiny.VIDEO_DECODER},
                               "audio": tiny.AUDIO,
                               "tiny": "TAEHV(latent_channels=24)"},
                    "weights_seed": tiny.SEED, "shapes": shapes,
                    "plan_tables": plan_tables(builds["video"]()),
                    "files_sha256": digests,
                    "weights": "h3_vae_tiny.weights(shapes[role]) written "
                    "by h3_vae_tiny.write_safetensors as F32"},
                   ["Each model is a ComfyUI class at a tiny size with "
                    "seeded weights, so the numbers say nothing about image "
                    "or audio quality."])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--comfy-dir", type=Path, required=True,
                    help="ComfyUI checkout")
    ap.add_argument("--vae", type=Path,
                    help="minimax_h3_video_vae_fp16.safetensors")
    ap.add_argument("--audio-vae", type=Path,
                    help="minimax_h3_audio_vae_fp32.safetensors")
    ap.add_argument("--tiny", type=Path, help="taeh3.safetensors")
    ap.add_argument("--self-test", action="store_true",
                    help="write the tiny seeded model fixture and stop")
    ap.add_argument("--ladder", type=int, default=0, metavar="N",
                    help="memory ladder: every side and the audio length "
                    "divided by N, written under a name ending _ladderN")
    ap.add_argument("--out", type=Path,
                    help="fixture directory (default $GMLX_VIDEO_FIXTURES)")
    a = ap.parse_args(argv)
    if not a.self_test and None in (a.vae, a.audio_vae, a.tiny):
        ap.error("--vae, --audio-vae and --tiny are required without "
                 "--self-test")
    torch = oracle.setup()
    a.comfy_head = oracle.start_comfy(a.comfy_dir)
    torch.set_grad_enabled(False)
    eager_rope()
    if a.self_test:
        return self_test(torch, a)
    name, cases = NAME, OFFICIAL
    if a.ladder:
        name, cases = f"{NAME}_ladder{a.ladder}", scaled(OFFICIAL, a.ladder)
    return run(torch, a, name, cases,
               {"vae": a.vae, "tiny": a.tiny, "audio_vae": a.audio_vae},
               _builds(torch, False), {},
               ["The video VAE file is float16 and the audio VAE file "
                "float32; each mode's model takes the file's values in its "
                "own dtype."])


if __name__ == "__main__":
    sys.exit(main())
