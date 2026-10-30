#!/usr/bin/env python3
"""Check 6 oracle for MiniMax-H3: a whole text to video run of ComfyUI's
MiniMaxH3Model from seeded noise to the final latents, with or without a
LoRA, on the weights of the DiT GGUF, and its own bf16 step at each of
the run's fp32 latents.

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
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np

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


def run(torch, dm, video, audio, feats, sv, sa, shifts, mode, tag,
        forced=None) -> tuple:
    """The final latents of one pass, and each step's (video, audio)
    velocity and latents after it. A pass ``forced`` with each step's
    input latents takes every step from them."""
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
                         transformer_options=opts)
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
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--modes", nargs="+", default=["fp32", "bf16_forced"],
                    choices=["fp32", "bf16", "bf16_forced"])
    ap.add_argument("--name", help="fixture name (default from the run)")
    a = ap.parse_args(argv)
    if "bf16_forced" in a.modes and a.modes[0] != "fp32":
        ap.error("bf16_forced takes its steps from the fp32 pass: put fp32 "
                 "first in --modes")
    if a.lora_merge == "float32" and a.lora is None:
        ap.error("--lora-merge float32 merges a LoRA: give --lora")
    torch = oracle.setup()
    a.comfy_head = oracle.start_comfy(a.comfy_dir)
    gguf_head = oracle.load_comfy_gguf(a.comfy_dir)
    if a.lora_merge == "float32":
        merge_in_float32(torch)
    a.dit = a.dit.expanduser().resolve()
    shape = c3.parse_shape(a.shape)
    w, h, f = shape
    t0 = time.perf_counter()
    patcher, dm = c3.load_gguf_dit(torch, a.dit, None)
    pairs = 0
    if a.lora is not None:
        a.lora = a.lora.expanduser().resolve()
        patcher, dm, pairs = add_lora(a.lora, patcher, a.lora_strength)
    print(f"[check6] loaded {a.dit.name} in {time.perf_counter() - t0:.1f}s"
          + (f", {pairs} LoRA pairs from {a.lora.name}" if pairs else ""),
          flush=True)
    import comfy.ldm.modules.attention as cattn

    z = np.load(a.features)
    bf16 = z[f"tap_bf16_{a.prompt}"].astype(_F32)
    feats = {"fp32": z[f"tap_{a.prompt}"].astype(_F32), "bf16": bf16,
             "bf16_forced": bf16}
    ids = z[f"ids_{a.prompt}"]
    shifts = {"video": a.shift_video, "audio": a.shift_audio}
    sv = video_sigmas(a.steps, a.shift_video)
    sa = audio_sigmas(torch, sv, a.shift_video, a.shift_audio)
    if a.stop_after is not None:
        sv, sa = sv[:a.stop_after + 1], sa[:a.stop_after + 1]
    video, audio = c3.latents(shape, a.seed)
    lora_tag = "_lora" if a.lora is not None else ""
    stop = f"_{a.stop_after}s" if a.stop_after is not None else ""
    name = a.name or f"check6_h3{lora_tag}_{w}x{h}x{f}{stop}"
    fx = fixtures.FixtureWriter(name, check=6, family=c3.FAMILY,
                                script=SCRIPT)
    fx.add_file("dit", a.dit)
    fx.add_file("features", a.features)
    if a.lora is not None:
        fx.add_file("lora", a.lora)
    fx.note(f"oracle: ComfyUI {a.comfy_head}, ComfyUI-GGUF {gguf_head}, "
            "UnetLoaderGGUF with dequant_dtype float32"
            + (", LoRA through comfy.sd.load_lora_for_models, merge "
               f"{a.lora_merge}: {LORA_MERGE[a.lora_merge]}" if pairs
               else ""))
    fx.inputs.update(
        shape=list(shape), steps=a.steps, forwards=len(sv) - 1,
        seed=a.seed, prompt=a.prompt, text_rows=int(ids.shape[0]),
        shifts=shifts, lora=a.lora.name if pairs else None,
        lora_strength=a.lora_strength if pairs else None, lora_pairs=pairs,
        lora_merge=a.lora_merge if pairs else None,
        lora_merge_means=LORA_MERGE[a.lora_merge] if pairs else None,
        video_shape=list(video.shape), audio_shape=list(audio.shape),
        dequant=c3.DEQUANT, matmul_bf16=c3.MATMUL, euler=EULER,
        timestep="sigma x 1000 in float32",
        attention=cattn.optimized_attention.__name__, comfy=a.comfy_head,
        comfy_gguf=gguf_head, modes=a.modes)
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
                               shifts, mode, name,
                               inputs if mode == "bf16_forced" else None)
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
    path = fx.write()
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
