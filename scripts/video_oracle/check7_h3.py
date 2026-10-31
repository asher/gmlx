#!/usr/bin/env python3
"""Check 7 oracle for MiniMax-H3: check 3's denoise step of ComfyUI's
MiniMaxH3Model, block by block, under a LoRA merged into the weights.

The model loads as check3_h3.py loads it, and the LoRA through ComfyUI's
own LoRA loader, as check6_h3.py loads it. Each weight the LoRA patches
is dequantized and patched in float32 on every call, then cast to the
pass dtype once, as ComfyUI core merges a LoRA on the CPU (check6_h3.py
--lora-merge float32). The step, its rows, its passes and their
tolerances are check 3's: one forward at --sigma on seeded latents and
seeded text features, in fp32 and in bf16, each refiner block, block and
head also teacher-forced, each row's tolerance twice its bf16 against
fp32 gap.

  python scripts/video_oracle/run.py --python <comfyui venv python> \\
      --threads 16 check7_h3.py --comfy-dir <ComfyUI checkout> \\
      --dit minimax_h3_fl2va_pruned-Q4_K.gguf --lora <8-step LoRA> \\
      --shape 512x512x22
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import check3_h3 as c3
import check6_h3 as c6
import oracle

ORACLE_NEEDS = ("torch", "gguf", "einops", "comfy_kitchen")

SCRIPT = "scripts/video_oracle/check7_h3.py"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--comfy-dir", type=Path, required=True,
                    help="ComfyUI checkout with custom_nodes/ComfyUI-GGUF")
    ap.add_argument("--dit", type=Path, required=True,
                    help="MiniMax-H3 DiT GGUF")
    ap.add_argument("--lora", type=Path, required=True,
                    help="LoRA safetensors file")
    ap.add_argument("--lora-strength", type=float, default=1.0)
    ap.add_argument("--shape", default="512x512x22",
                    help="clip shape WxHxFRAMES")
    ap.add_argument("--text-len", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", help="fixture name (default from the run)")
    a = ap.parse_args(argv)
    torch = oracle.setup()
    a.comfy_head = oracle.start_comfy(a.comfy_dir)
    gguf_head = oracle.load_comfy_gguf(a.comfy_dir)
    c6.merge_in_float32(torch)
    a.dit = a.dit.expanduser().resolve()
    a.lora = a.lora.expanduser().resolve()
    shape = c3.parse_shape(a.shape)
    w, h, f = shape
    t0 = time.perf_counter()
    patcher, _ = c3.load_gguf_dit(torch, a.dit, None)
    _, dm, pairs = c6.add_lora(a.lora, patcher, a.lora_strength)
    print(f"[check7] loaded {a.dit.name} in {time.perf_counter() - t0:.1f}s,"
          f" {pairs} LoRA pairs from {a.lora.name}", flush=True)
    import comfy.ldm.modules.attention as cattn

    rng = np.random.default_rng(a.seed)
    feats = rng.standard_normal((a.text_len, 5120)).astype(np.float32)
    video, audio = c3.latents(shape, a.seed)
    inputs = {"shape": list(shape), "blocks": len(dm.blocks),
              "sigma": a.sigma, "seed": a.seed,
              "features": "seeded normal, std 1.0",
              "text_rows": int(feats.shape[0]),
              "video_shape": list(video.shape),
              "audio_shape": list(audio.shape), "dequant": c3.DEQUANT,
              "matmul_bf16": c3.MATMUL,
              "timestep": "sigma x 1000 in float32 in both modes",
              "attention": cattn.optimized_attention.__name__,
              "comfy": a.comfy_head, "comfy_gguf": gguf_head,
              "lora": a.lora.name, "lora_strength": a.lora_strength,
              "lora_pairs": pairs, "lora_merge": "float32",
              "lora_merge_means": c6.LORA_MERGE["float32"]}
    notes = [f"oracle: ComfyUI {a.comfy_head}, ComfyUI-GGUF {gguf_head}, "
             "UnetLoaderGGUF with dequant_dtype float32, LoRA through "
             "comfy.sd.load_lora_for_models, merge float32: "
             + c6.LORA_MERGE["float32"]]
    name = a.name or f"check7_h3_lora_{w}x{h}x{f}"
    c3.run_case(torch, dm, video, audio, feats, a.sigma, name, inputs,
                {"dit": a.dit, "lora": a.lora}, notes, check=7,
                script=SCRIPT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
