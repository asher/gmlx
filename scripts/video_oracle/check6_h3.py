#!/usr/bin/env python3
"""Check 6 oracle for MiniMax-H3: a whole run of ComfyUI's MiniMaxH3Model
from seeded noise to the final latents, text to video or with keyframes
and guides, with or without a LoRA, on the weights of the DiT GGUF, and
its own bf16 step at each of the run's fp32 latents.

The model loads as check3_h3.py loads it: ComfyUI-GGUF's UnetLoaderGGUF
with dequant_dtype float32 and torch's SDPA. A LoRA loads through
ComfyUI's own LoRA loader, as a ComfyUI user runs it. ComfyUI-GGUF
patches the K-quant and BF16 weights on every call: each patched weight
is dequantized and cast to the pass's dtype, and the LoRA delta,
computed in float32, is cast to that dtype and added. The fp32 pass adds
in float32 and the bf16 pass in bfloat16. A pair on an F32 or F16 weight
(the heads and the patch projections, F32; the adaLN linears, F16) it
merges once at load in float32 and stores in the weight's dtype. The
8-step LoRA has pairs on the K-quant block leaves only.

--lora-merge float32 merges as ComfyUI core merges a LoRA into a bf16
checkpoint on the CPU, where the oracle runs: each weight patched per
call is dequantized and patched in float32 and rounded to the pass's
dtype once. ComfyUI-GGUF's own add in bf16 drops every part of the delta
under half a bf16 step, which is most of a Turbo LoRA's. Its bf16 pass
then keeps little of the LoRA, and a gate set from that pass's step gap
is wider than the LoRA's whole effect. The fp32 pass is the same under
both.

Each step is one forward at the video sigma, the timestep sigma x 1000 in
float32, with the shifts in transformer_options as ModelSamplingMiniMaxH3
sets them; the forward derives the audio time from them. Each stream then
takes an Euler step in float32 on its own sigmas, the audio sigmas the
time shift of the video ones. ComfyUI's sampler steps the audio carried
as (sigma_v / sigma_a) x audio on the video sigmas; the carry's x0 is the
audio's times shift_v / shift_a at every sigma, so its Euler step is the
audio stream's own.

The fp32 pass reads float32 text features and the bf16 passes bfloat16
ones, by default check 2's oracle taps of one prompt in each mode. The
fixture holds the fp32 latents and velocity of every step. The
bf16_forced pass takes each step from the fp32 pass's latents before it
and keeps its velocity, so the manifest sets each step's velocity
tolerance to twice the oracle's own bf16 against fp32 relative L2 at that
step, video and audio apart. A free bf16 pass (bf16) drifts from the
fp32 run as a different sample would, so its final gap is reported and
gates nothing. The bf16 passes keep the latents float32 at the model
boundary, as the fp32 pass does. --modes fp32 writes the fp32 run alone,
a reference without a tolerance.

  python scripts/video_oracle/run.py --python <comfyui venv python> \\
      --threads 16 check6_h3.py --comfy-dir <ComfyUI checkout> \\
      --dit minimax_h3_fl2va_pruned-Q4_K.gguf --features check2_h3.npz \\
      --lora <8-step LoRA> --lora-merge float32 --shift-video 6 \\
      --shape 512x512x22

--cases adds the anchor cases (ANCHOR_CASES) beside the text to video
run, each a run of its own. ComfyUI's nodes place each anchor and encode
it on the CPU before the DiT loads: MiniMaxH3ImageToVideo the keyframes,
with a stand-in text encoder that hands on check 2's tap of the keyframe
prompt (--keyframes) and checks the node gives it check 2's images, and
MiniMaxH3AddGuide the guides, with the text prompt. Both VAEs load as
VAELoader loads them, in float32 on the CPU. The first frame is a PNG of
check 2's 640x400 source image through LoadImage; the guide audio a
32 kHz stereo 16-bit WAV through LoadAudio's reader. The payload is what
the checkout's MiniMaxH3.extra_conds writes for the keyframes, the tags
and the run's seed, in both passes. Each fixture holds the anchors'
latents, the noise ComfyUI's forward mixes into their video rows and the
prompt's vision spans, and the first frame case the PNG and its canvas
image, for the gmlx side to read the PNG itself.

  python scripts/video_oracle/run.py --python <comfyui venv python> \\
      --threads 16 check6_h3.py --comfy-dir <ComfyUI checkout> \\
      --dit minimax_h3_fl2va_pruned-Q4_K.gguf --features check2_h3.npz \\
      --keyframes check2_h3_keyframes.npz \\
      --vae minimax_h3_video_vae_fp16.safetensors \\
      --audio-vae minimax_h3_audio_vae_fp32.safetensors \\
      --lora <8-step LoRA> --lora-merge float32 --shift-video 6 \\
      --shape 512x512x22 --cases first last both mid_audio clip17
"""

from __future__ import annotations

import argparse
import gc
import resource
import sys
import time
import wave
from pathlib import Path

import numpy as np

import check2_h3 as c2
import check3_h3 as c3
import fixtures
import oracle

ORACLE_NEEDS = ("torch", "gguf", "einops", "comfy_kitchen")

SCRIPT = "scripts/video_oracle/check6_h3.py"
RULE = ("steps: the velocity of each step from the fp32 latents before "
        "it, tolerance = 2 x the oracle's own bf16 against fp32 relative "
        "L2 at that step, video and audio apart; the exact gates are the "
        "gmlx test's, from the floors it records; the LoRA merged as "
        "inputs.lora_merge records")
LORA_MERGE = {
    "gguf": ("as ComfyUI-GGUF runs a LoRA: on each call the weight and the "
             "float32 delta cast to the pass dtype and added there"),
    "float32": ("each weight patched per call dequantized and patched in "
                "float32, then cast to the pass dtype once, as ComfyUI "
                "core merges a LoRA on the CPU"),
}
EULER = ("Euler in float32 per stream; the audio sigmas the time shift of "
         "the video sigmas, as the forward derives the audio time")
_F32 = np.float32
# The anchor cases: the keyframe prompt of check 2 a keyframe case takes
# ("first" the PNG of its first source image, an integer the canvas image
# of that index, or "source" and the index for the source image), or the
# guides as (frame, image frames, audio) with "mid" the clip's middle
# frame.
ANCHOR_CASES = {
    "first": {"prompt": "keyframe", "first": "png"},
    "last": {"prompt": "keyframe", "last": ("image", 0)},
    "both": {"prompt": "keyframes", "first": ("source", 0),
             "last": ("source", 1)},
    "mid_audio": {"guides": [("mid", 1, True)]},
    "clip17": {"guides": [(17, 5, False)]},
}
CASES = ("t2v", *ANCHOR_CASES)
GUIDE_SOURCE = (640, 400)       # each guide picture, cropped to the canvas
AUDIO_RATE, AUDIO_S = 32000, 1.0


def video_sigmas(steps: int, shift: float) -> np.ndarray:
    """float32 shifted linspace(1, 0, steps + 1), each op in float32, as
    the reference scheduler builds it."""
    import torch

    s = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32).numpy()
    return (_F32(shift) * s / (_F32(1) + _F32(shift - 1) * s)).astype(_F32)


def audio_sigmas(torch, sv: np.ndarray, shift_v: float,
                 shift_a: float) -> np.ndarray:
    """The audio sigma at each video sigma, by the forward's time shift."""
    from comfy.ldm.minimax.model import time_shift_sigma

    return np.array([float(time_shift_sigma(
        torch.tensor(s, dtype=torch.float32), shift_v, shift_a))
        for s in sv], _F32)


def add_lora(path: Path, patcher, strength: float):
    """The patcher with the LoRA at ``path``, every key placed."""
    import comfy.model_management
    import comfy.sd
    import comfy.utils

    sd = comfy.utils.load_torch_file(str(path), safe_load=True)
    pairs = {k.rsplit(".lora_A.weight", 1)[0] for k in sd
             if k.endswith(".lora_A.weight")}
    patched, _ = comfy.sd.load_lora_for_models(patcher, None, sd, strength,
                                               0)
    if len(patched.patches) != len(pairs):
        raise SystemExit(f"error: ComfyUI placed {len(patched.patches)} of "
                         f"the LoRA's {len(pairs)} pairs")
    comfy.model_management.load_models_gpu([patched])
    return patched, patched.model.diffusion_model, len(pairs)


def merge_in_float32(torch) -> None:
    """Patch ComfyUI-GGUF so a weight patched per call is dequantized and
    patched in float32 and cast to the pass dtype once."""
    cls = sys.modules["ComfyUI_GGUF.ops"].GGMLLayer
    stock = cls.get_weight

    def get_weight(self, tensor, dtype):
        if tensor is None or not getattr(tensor, "patches", None) \
                or dtype == torch.float32:
            return stock(self, tensor, dtype)
        return stock(self, tensor, torch.float32).to(dtype)

    cls.get_weight = get_weight


def guide_wav(path: Path, seed: int) -> None:
    """A seeded 32 kHz stereo 16-bit WAV of AUDIO_S seconds: a tone apart
    in each channel under mild noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(AUDIO_RATE * AUDIO_S)) / AUDIO_RATE
    x = np.stack([0.3 * np.sin(2 * np.pi * 220 * t),
                  0.25 * np.sin(2 * np.pi * 330 * t + 1)])
    x = x + rng.normal(0, 0.02, x.shape)
    pcm = np.clip(np.round(x * 32767), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(AUDIO_RATE)
        w.writeframes(pcm.T.tobytes())


class _Clip:
    """The text encoder MiniMaxH3ImageToVideo calls: check 2 ran the
    keyframe prompt, so this one checks the node hands it check 2's
    images and returns an empty conditioning for the node to fill."""

    def __init__(self, torch, images):
        self.torch, self.images = torch, images

    def tokenize(self, prompt, images=()):
        got = [np.round(im[0].numpy() * 255.0).astype(np.uint8)
               for im in images]
        if prompt != c2.KEYFRAME_TEXT or len(got) != len(self.images) or \
                not all(np.array_equal(g, w)
                        for g, w in zip(got, self.images)):
            raise SystemExit("error: the node's prompt or keyframes are not "
                             "check 2's")
        return {}

    def encode_from_tokens_scheduled(self, _tokens):
        return [[self.torch.zeros(1, 1, 1), {}]]


def _image(torch, im: np.ndarray):
    """An (H, W, 3) uint8 image as a ComfyUI IMAGE batch."""
    return torch.from_numpy(im.astype(np.float32) / 255.0)[None]


def zero_runs(tags: np.ndarray) -> list:
    """The [start, stop) runs of the vision tags (0) of a prompt."""
    out, start = [], None
    for i, t in enumerate([*tags.tolist(), 1]):
        if t == 0 and start is None:
            start = i
        elif t != 0 and start is not None:
            out.append([start, i])
            start = None
    return out


def anchor_cases(torch, a, shape, base: str) -> dict:
    """Each anchor case of ``a.cases`` as ComfyUI's nodes write it, run
    before the DiT loads: case: (keyframes, tags, prompt key, arrays,
    inputs, files). The PNG and the WAV are written beside the fixtures
    under names from ``base``."""
    import comfy.model_management
    import comfy.sd
    import comfy.utils
    import folder_paths
    import nodes
    from comfy_extras import nodes_audio
    from comfy_extras.nodes_minimax_h3 import (MiniMaxH3AddGuide,
                                               MiniMaxH3ImageToVideo,
                                               _empty_av_latent)
    from PIL import Image

    w, h, f = shape
    todo = [c for c in a.cases if c != "t2v"]
    specs = {c: ANCHOR_CASES[c] for c in todo}
    root = fixtures.fixture_root(a.out, create=True)
    png = root / f"{base}_first_frame.png"
    wav = root / f"{base}_guide_audio.wav"
    z = np.load(a.keyframes) if any("prompt" in s for s in specs.values()) \
        else None
    sounds = [c for c, s in specs.items()
              if any(g[2] for g in s.get("guides", ()))]
    if any(s.get("first") == "png" for s in specs.values()):
        Image.fromarray(z["source_keyframe_0"]).save(png)
        folder_paths.set_input_directory(str(root))
    if sounds:
        guide_wav(wav, a.seed)

    def load_vae(path):
        sd, meta = comfy.utils.load_torch_file(str(path),
                                               return_metadata=True)
        vae = comfy.sd.VAE(sd=sd, metadata=meta)
        vae.throw_exception_if_invalid()
        return vae

    vae = load_vae(a.vae)
    audio_vae = load_vae(a.audio_vae) if sounds else None
    print(f"[check6] VAEs loaded, max rss {_max_rss():.1f} GiB", flush=True)
    out = {}
    for c, spec in specs.items():
        extra, files, tags, key = {}, {}, None, spec.get("prompt")
        if key is not None:
            if (w, h) != c2.CANVAS:
                raise SystemExit(f"error: {c} runs at check 2's canvas "
                                 f"{c2.CANVAS}")
            n = sum(1 for k in z.files if k.startswith(f"image_{key}_"))
            want = [z[f"image_{key}_{i}"] for i in range(n)]
            given = {}
            for side in ("first", "last"):
                src = spec.get(side)
                if src == "png":
                    given[side] = nodes.LoadImage().load_image(png.name)[0]
                    extra["first_frame_read"] = np.round(
                        given[side][0].numpy() * 255.0).astype(np.uint8)
                    extra["first_frame_canvas"] = want[0]
                    files["first_frame"] = png
                elif src is not None:
                    given[side] = _image(torch, z[f"{src[0]}_{key}_{src[1]}"])
            cond = MiniMaxH3ImageToVideo.execute(
                _Clip(torch, want), vae, c2.KEYFRAME_TEXT, w, h, f,
                first_frame=given.get("first"),
                last_frame=given.get("last")).args[0]
            tags = torch.from_numpy(z[f"tags_{key}"].astype(np.int64))
        else:
            cond = [[torch.zeros(1, 1, 1), {}]]
            latent, _ = _empty_av_latent(w, h, f)
            for g, (frame, frames, sound) in enumerate(spec["guides"]):
                pics = np.stack([c2.keyframe_image(20 + 10 * g + t,
                                                   *GUIDE_SOURCE)
                                 for t in range(frames)])
                extra[f"guide_{g}_frames"] = pics
                audio = None
                if sound:
                    wave_t, sr = nodes_audio.load(str(wav))
                    audio = {"waveform": wave_t[None], "sample_rate": sr}
                    extra[f"guide_{g}_waveform"] = oracle.to_numpy(wave_t)
                    files["guide_audio"] = wav
                cond = MiniMaxH3AddGuide.execute(
                    cond, latent, (f - 1) // 2 if frame == "mid" else frame,
                    vae=vae, audio_vae=audio_vae,
                    image=torch.from_numpy(pics.astype(np.float32) / 255.0),
                    audio=audio).args[0]
        out[c] = (cond[0][1]["minimax_keyframes"], tags, key, extra, files)
        print(f"[check6] {c} anchors encoded, max rss {_max_rss():.1f} GiB",
              flush=True)
    del vae, audio_vae
    comfy.model_management.unload_all_models()
    gc.collect()
    cases = {}
    for c, (keyframes, tags, key, arrays, files) in out.items():
        placed = []
        for i, kf in enumerate(keyframes):
            arrays[f"anchor.{i}.frame"] = np.int64(kf["resolved_frame_index"])
            v, au = kf.get("latent"), kf.get("audio_latent")
            if v is not None:
                arrays[f"anchor.{i}.video"] = oracle.to_numpy(v).astype(_F32)
            if au is not None:
                arrays[f"anchor.{i}.audio"] = oracle.to_numpy(au).astype(_F32)
            placed.append({"frame": int(kf["resolved_frame_index"]),
                           "video_t": 0 if v is None else int(v.shape[2]),
                           "audio_t": 0 if au is None else int(au.shape[-1])})
        runs = []
        if tags is not None:
            arrays["text_token_tags"] = tags.numpy()
            runs = zero_runs(tags.numpy())
            arrays["spans"] = np.array(runs, np.int64)
        arrays["anchor_seed"] = np.int64(a.seed)
        inputs = {"case": c, "anchors": placed, "spans": runs,
                  "anchor_seed": a.seed, "anchor_spec": specs[c]}
        cases[c] = (keyframes, tags, key, arrays, inputs, files)
    return cases


def _max_rss() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30


def run(torch, dm, video, audio, feats, sv, sa, shifts, mode, tag,
        forced=None, payload=None) -> tuple:
    """The final latents of one pass, and each step's (video, audio)
    velocity and latents after it. A pass ``forced`` with each step's
    input latents takes every step from them; ``payload`` is each
    forward's minimax_payload."""
    ctx = torch.from_numpy(feats[None]).to(
        oracle.torch_dtype("fp32" if mode == "fp32" else "bf16"))
    xv, xa = video.copy(), audio.copy()
    vel, lat = [], []
    oracle.emulate_bf16_matmul(mode != "fp32")
    try:
        for i in range(len(sv) - 1):
            t0 = time.perf_counter()
            if forced is not None:
                xv, xa = (a.copy() for a in forced[i])
            opts = {"minimax_h3_sigma_shift_video": shifts["video"],
                    "minimax_h3_sigma_shift_audio": shifts["audio"]}
            with torch.no_grad():
                out = dm([torch.from_numpy(xv), torch.from_numpy(xa)],
                         c3.timestep(torch, float(sv[i])), ctx,
                         transformer_options=opts, minimax_payload=payload)
            vv = oracle.to_numpy(out[0]).astype(_F32)
            va = oracle.to_numpy(out[1]).astype(_F32)
            vel.append((vv, va))
            xv = (xv + (sv[i + 1] - sv[i]) * vv).astype(_F32)
            xa = (xa + (sa[i + 1] - sa[i]) * va).astype(_F32)
            lat.append((xv.copy(), xa.copy()))
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
            print(f"[check6] {tag} {mode} step {i + 1}/{len(sv) - 1} "
                  f"{time.perf_counter() - t0:.1f}s, max rss {rss:.1f} GiB",
                  flush=True)
    finally:
        oracle.emulate_bf16_matmul(False)
    return xv, xa, vel, lat


def run_case(torch, dm, patcher, a, name: str, base: dict, feats: dict,
             ids, files: dict, anchored=None) -> None:
    """The passes of one case, written as fixture ``name``. ``base`` holds
    the inputs every case shares, ``anchored`` an anchor case's
    (keyframes, tags, arrays, inputs) from anchor_cases."""
    shape = tuple(base["inputs"]["shape"])
    sv = video_sigmas(a.steps, a.shift_video)
    sa = audio_sigmas(torch, sv, a.shift_video, a.shift_audio)
    if a.stop_after is not None:
        sv, sa = sv[:a.stop_after + 1], sa[:a.stop_after + 1]
    video, audio = c3.latents(shape, a.seed)
    fx = fixtures.FixtureWriter(name, check=6, family=c3.FAMILY,
                                script=SCRIPT)
    for role, path in files.items():
        fx.add_file(role, path)
    for n in base["notes"]:
        fx.note(n)
    fx.inputs.update(base["inputs"])
    fx.inputs.update(forwards=len(sv) - 1, text_rows=int(ids.shape[0]),
                     video_shape=list(video.shape),
                     audio_shape=list(audio.shape))
    payload = None
    if anchored is not None:
        keyframes, tags, arrays, more = anchored
        c3.check_payload(torch, patcher, keyframes, tags, a.seed)
        payload = c3.payload_of(keyframes, tags, a.seed)
        fx.inputs.update(more)
        for k, arr in arrays.items():
            fx.add(k, arr)
        for i, n in enumerate(c3.anchor_noise(torch, dm, payload)):
            fx.add(f"anchor_noise.{i}", n)
    fx.add("noise_video", video)
    fx.add("noise_audio", audio)
    fx.add("ids", ids)
    fx.add("sigmas_video", sv)
    fx.add("sigmas_audio", sa)
    fx.add("features", feats["fp32"])
    final, times, vels = {}, {}, {}
    inputs = None
    for mode in a.modes:
        t0 = time.perf_counter()
        xv, xa, vel, lat = run(torch, dm, video, audio, feats[mode], sv, sa,
                               base["inputs"]["shifts"], mode, name,
                               inputs if mode == "bf16_forced" else None,
                               payload)
        times[f"{mode}_s"] = round(time.perf_counter() - t0, 1)
        vels[mode] = vel
        if mode == "fp32":
            inputs = [(video, audio)] + lat[:-1]
            for i, ((vv, va), (xv_i, xa_i)) in enumerate(zip(vel, lat)):
                fx.add(f"video_step{i + 1}", xv_i)
                fx.add(f"audio_step{i + 1}", xa_i)
                fx.add(f"velocity_video_step{i + 1}", vv)
                fx.add(f"velocity_audio_step{i + 1}", va)
        if mode == "bf16_forced":
            for i, (vv, va) in enumerate(vel):
                fx.add(f"velocity_bf16_video_step{i + 1}", vv)
                fx.add(f"velocity_bf16_audio_step{i + 1}", va)
        else:
            final[mode] = {"video": xv, "audio": xa}
            sfx = "" if mode == "fp32" else "_bf16"
            fx.add(f"video{sfx}", xv)
            fx.add(f"audio{sfx}", xa)
    times["max_rss_gib"] = round(_max_rss(), 2)
    fx.results["times"] = times
    if "bf16" in final and "fp32" in final:
        gaps = {}
        for stream in ("video", "audio"):
            g = fixtures.compare(final["bf16"][stream],
                                 final["fp32"][stream])
            g["db"] = float(-20 * np.log10(g["rel_l2"]))
            gaps[stream] = g
        fx.results["final_bf16_vs_fp32"] = gaps
    if "bf16_forced" in vels:
        gaps, tols = [], []
        for (bv, ba), (fv, fa) in zip(vels["bf16_forced"], vels["fp32"]):
            g = {"video": fixtures.compare(bv, fv),
                 "audio": fixtures.compare(ba, fa)}
            gaps.append(g)
            tols.append({s: {"rel_l2": 2 * g[s]["rel_l2"]} for s in g})
        fx.results["step_bf16_vs_fp32"] = gaps
        fx.tolerances["steps"] = tols
        fx.tolerances["rule"] = RULE
    else:
        fx.note("no bf16_forced pass: a reference with no tolerance")
    path = fx.write(a.out)
    print(f"[check6] {name}: {times}", flush=True)
    for i, g in enumerate(fx.results.get("step_bf16_vs_fp32", ())):
        print(f"[check6] step {i + 1} bf16 vs fp32 velocity rel L2 video "
              f"{g['video']['rel_l2']:.3e} audio {g['audio']['rel_l2']:.3e}",
              flush=True)
    gap = fx.results.get("final_bf16_vs_fp32")
    if gap:
        print(f"[check6] free bf16 final rel L2 video "
              f"{gap['video']['rel_l2']:.3e} audio "
              f"{gap['audio']['rel_l2']:.3e}", flush=True)
    print(f"[check6] wrote {path}", flush=True)


def _feats(z, key: str) -> dict:
    bf16 = z[f"tap_bf16_{key}"].astype(_F32)
    return {"fp32": z[f"tap_{key}"].astype(_F32), "bf16": bf16,
            "bf16_forced": bf16}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--comfy-dir", type=Path, required=True,
                    help="ComfyUI checkout with custom_nodes/ComfyUI-GGUF")
    ap.add_argument("--dit", type=Path, required=True,
                    help="MiniMax-H3 DiT GGUF")
    ap.add_argument("--features", type=Path, required=True,
                    help="check 2 fixture .npz with tap_<prompt> and "
                         "tap_bf16_<prompt>")
    ap.add_argument("--prompt", default="dialogue",
                    help="prompt key of --features")
    ap.add_argument("--cases", nargs="+", default=["t2v"], choices=CASES,
                    help="t2v and the anchor cases (ANCHOR_CASES)")
    ap.add_argument("--keyframes", type=Path,
                    help="check 2 keyframe fixture .npz, for a keyframe "
                         "case")
    ap.add_argument("--vae", type=Path,
                    help="video VAE safetensors, for an anchor case")
    ap.add_argument("--audio-vae", type=Path,
                    help="audio VAE safetensors, for a guide with audio")
    ap.add_argument("--lora", type=Path, help="LoRA safetensors file")
    ap.add_argument("--lora-strength", type=float, default=1.0)
    ap.add_argument("--lora-merge", choices=sorted(LORA_MERGE),
                    default="gguf",
                    help="gguf: " + LORA_MERGE["gguf"] + "; float32: "
                         + LORA_MERGE["float32"] + ". The fp32 pass is "
                         "the same under both")
    ap.add_argument("--shape", default="512x512x22",
                    help="clip shape WxHxFRAMES")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--stop-after", type=int,
                    help="stop each pass after N steps (a memory ladder)")
    ap.add_argument("--shift-video", type=float, default=12.0)
    ap.add_argument("--shift-audio", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=6,
                    help="the seed of the noise and of the anchors' noise")
    ap.add_argument("--modes", nargs="+", default=["fp32", "bf16_forced"],
                    choices=["fp32", "bf16", "bf16_forced"])
    ap.add_argument("--name", help="fixture name of the text to video run "
                    "(default from the run); an anchor case adds _<case> "
                    "before the shape")
    ap.add_argument("--out", type=Path,
                    help="fixture directory (default $GMLX_VIDEO_FIXTURES)")
    a = ap.parse_args(argv)
    if "bf16_forced" in a.modes and a.modes[0] != "fp32":
        ap.error("bf16_forced takes its steps from the fp32 pass: put fp32 "
                 "first in --modes")
    if a.lora_merge == "float32" and a.lora is None:
        ap.error("--lora-merge float32 merges a LoRA: give --lora")
    anchors = [c for c in a.cases if c != "t2v"]
    if anchors and a.vae is None:
        ap.error("an anchor case needs --vae")
    if any("prompt" in ANCHOR_CASES[c] for c in anchors) \
            and a.keyframes is None:
        ap.error("a keyframe case needs --keyframes")
    if any(g[2] for c in anchors for g in ANCHOR_CASES[c].get("guides", ())) \
            and a.audio_vae is None:
        ap.error("a guide with audio needs --audio-vae")
    torch = oracle.setup()
    a.comfy_head = oracle.start_comfy(a.comfy_dir)
    gguf_head = oracle.load_comfy_gguf(a.comfy_dir)
    if a.lora_merge == "float32":
        merge_in_float32(torch)
    for k in ("dit", "features", "keyframes", "vae", "audio_vae", "lora"):
        if getattr(a, k) is not None:
            setattr(a, k, getattr(a, k).expanduser().resolve())
    shape = c3.parse_shape(a.shape)
    w, h, f = shape
    lora_tag = "_lora" if a.lora is not None else ""
    stop = f"_{a.stop_after}s" if a.stop_after is not None else ""
    stem = a.name.rsplit(f"_{w}x{h}x{f}", 1)[0] if a.name \
        else f"check6_h3{lora_tag}"
    t0 = time.perf_counter()
    anchored = anchor_cases(torch, a, shape, f"{stem}{stop}") if anchors \
        else {}
    if anchors:
        print(f"[check6] anchors of {anchors} in "
              f"{time.perf_counter() - t0:.1f}s", flush=True)
    t0 = time.perf_counter()
    patcher, dm = c3.load_gguf_dit(torch, a.dit, None)
    pairs = 0
    if a.lora is not None:
        patcher, dm, pairs = add_lora(a.lora, patcher, a.lora_strength)
    print(f"[check6] loaded {a.dit.name} in {time.perf_counter() - t0:.1f}s"
          + (f", {pairs} LoRA pairs from {a.lora.name}" if pairs else ""),
          flush=True)
    import comfy.ldm.modules.attention as cattn

    shifts = {"video": a.shift_video, "audio": a.shift_audio}
    notes = [f"oracle: ComfyUI {a.comfy_head}, ComfyUI-GGUF {gguf_head}, "
             "UnetLoaderGGUF with dequant_dtype float32"
             + (", LoRA through comfy.sd.load_lora_for_models, merge "
                f"{a.lora_merge}: {LORA_MERGE[a.lora_merge]}" if pairs
                else "")]
    inputs = dict(
        shape=list(shape), steps=a.steps, seed=a.seed, shifts=shifts,
        lora=a.lora.name if pairs else None,
        lora_strength=a.lora_strength if pairs else None, lora_pairs=pairs,
        lora_merge=a.lora_merge if pairs else None,
        lora_merge_means=LORA_MERGE[a.lora_merge] if pairs else None,
        dequant=c3.DEQUANT, matmul_bf16=c3.MATMUL, euler=EULER,
        timestep="sigma x 1000 in float32",
        attention=cattn.optimized_attention.__name__, comfy=a.comfy_head,
        comfy_gguf=gguf_head, modes=a.modes)
    files = {"dit": a.dit}
    if a.lora is not None:
        files["lora"] = a.lora
    z = np.load(a.features)
    zk = np.load(a.keyframes) if a.keyframes is not None else None
    for c in a.cases:
        if c == "t2v":
            run_case(torch, dm, patcher, a, f"{stem}_{w}x{h}x{f}{stop}",
                     {"inputs": {**inputs, "prompt": a.prompt},
                      "notes": notes},
                     _feats(z, a.prompt), z[f"ids_{a.prompt}"],
                     {**files, "features": a.features})
            continue
        keyframes, tags, key, arrays, more, made = anchored[c]
        src, prompt = (zk, key) if key is not None else (z, a.prompt)
        role = {"features": a.keyframes if key is not None else a.features,
                "vae": a.vae}
        if any(kf.get("audio_latent") is not None for kf in keyframes):
            role["audio_vae"] = a.audio_vae
        run_case(torch, dm, patcher, a, f"{stem}_{c}_{w}x{h}x{f}{stop}",
                 {"inputs": {**inputs, "prompt": prompt}, "notes": notes},
                 _feats(src, prompt), src[f"ids_{prompt}"],
                 {**files, **role, **made},
                 (keyframes, tags, arrays, more))
    return 0


if __name__ == "__main__":
    sys.exit(main())
