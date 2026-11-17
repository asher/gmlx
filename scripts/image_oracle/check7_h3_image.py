#!/usr/bin/env python3
"""Check 7 oracle for MiniMax-H3-Image: check 3's denoise step of
ComfyUI's MiniMaxH3ImageModel, block by block, under LoRA files merged
into the weights.

The model loads as check3_h3_image.py loads it. Each LoRA goes through
ComfyUI's own LoRA loader, comfy.lora.load_lora on ComfyUI's key map,
and each patch is added at the file's strength, as LoraLoader adds it.
A file whose every pair stock ComfyUI's key map places goes through it
unchanged. A file in diffusers' module names, which ComfyUI's key map
does not read for MiniMaxH3 (comfy/lora.py:421-428), goes through the key
map with those names added as diffusers 0.43.0 defines them
(_convert_non_diffusers_minimax_h3_lora_to_diffusers,
loaders/lora_conversion_utils.py:3342), read backward:

    transformer_blocks.N.        blocks.N.
    token_refiner.refiner_blocks.N.  token_refiner.blocks.N.
    attn.to_q, to_k, to_v        rows 0, 1 and 2 of attn.qkv_proj in thirds
    attn.to_out.0                attn.out_proj
    ff.net.0.proj                mlp.fc1, its two output halves swapped
    ff.net.2                     mlp.fc2
    adaln_proj.linear            adaln_proj.linear
    context_embedder             condition_proj
    proj_in                      video_patch_proj
    norm_out.linear              final_layer.adaln_proj.linear
    proj_out                     final_layer.video_out, placed as ComfyUI
                                 places the head's own name

each with no prefix or under ``transformer.``. A band is ComfyUI's offset
key and the swap its key function, comfy.utils.swap_scale_shift, as
ComfyUI writes diffusers q, k, v and norm_out for Flux, so ComfyUI's
arithmetic merges every pair. Every pair of every file must be placed.
Each --converted file is diffusers' conversion of a --lora file, named
by its source_file metadata: gmlx reads it in place of that file, so the
stock merge of the original is the oracle of gmlx's reading of the
conversion.
Each weight the LoRAs patch is dequantized and patched in float32 on every
call, then cast to the pass dtype once, as ComfyUI core merges a LoRA on
the CPU (scripts/video_oracle/check6_h3.py --lora-merge float32). The
step, its rows, its passes and their tolerances are check 3's.

  python scripts/image_oracle/run.py --python <ComfyUI venv python> \\
      --threads 16 check7_h3_image.py --comfy-dir <ComfyUI checkout> \\
      --dit <copy with the config key> --source <release GGUF> \\
      --lora character_swap.safetensors:1.0 \\
      --lora pytorch_lora_weights.safetensors:0.5 --case 512x512:0.75 \\
      [--converted h3_image_character_swap_diffusers.safetensors]

--heads-in-order departs from ComfyUI on one line: step k of
--sample-sigmas reads head k of the bank (head 0, or head 0 plus the
bank's block k), as diffusers' MiniMax-H3-Image pipeline reads pdd head
k at step k, in place of the dt-weighted heads of the grid intervals the
step spans (comfy/ldm/minimax/model.py:349-381). The fixture records the
head as step_head.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

import check3_h3_image as c3
import oracle

ORACLE_NEEDS = ("torch", "gguf", "einops", "comfy_kitchen")

SCRIPT = "scripts/image_oracle/check7_h3_image.py"
PREFIXES = ("", "transformer.")
# The diffusers name of each GGUF module outside the blocks.
STANDALONE = {"condition_proj": "context_embedder",
              "video_patch_proj": "proj_in",
              "final_layer.adaln_proj.linear": "norm_out.linear",
              "final_layer.video_out": "proj_out"}
BLOCKS = (("blocks.", "transformer_blocks."),
          ("token_refiner.blocks.", "token_refiner.refiner_blocks."))
LEAVES = {"attn.out_proj": "attn.to_out.0", "mlp.fc2": "ff.net.2",
          "adaln_proj.linear": "adaln_proj.linear"}
QKV = ("attn.to_q", "attn.to_k", "attn.to_v")
MERGE = ("each weight patched per call dequantized and patched in float32, "
         "then cast to the pass dtype once, as ComfyUI core merges a LoRA "
         "on the CPU")
KEYS = ("ComfyUI's key map for the model, with the diffusers names of "
        "diffusers 0.43.0's MiniMax-H3 LoRA converter added: q, k, v as "
        "offset keys on thirds of qkv_proj's rows, ff.net.0.proj with "
        "comfy.utils.swap_scale_shift as its key function")


def diffusers_names(name: str) -> list:
    """(diffusers name, part) for the GGUF module ``name``: part None for
    the whole weight, 0 to 2 for a third of the fused qkv rows, "swap" for
    fc1's swapped halves."""
    if name in STANDALONE:
        return [(STANDALONE[name], None)]
    for src, dst in BLOCKS:
        if not name.startswith(src):
            continue
        idx, _, leaf = name[len(src):].partition(".")
        base = f"{dst}{idx}."
        if leaf == "attn.qkv_proj":
            return [(base + q, i) for i, q in enumerate(QKV)]
        if leaf == "mlp.fc1":
            return [(base + "ff.net.0.proj", "swap")]
        if leaf in LEAVES:
            return [(base + LEAVES[leaf], None)]
    return []


def extend_key_map(key_map: dict, sd: dict) -> int:
    """Add the diffusers names of every weight of the model's state dict
    ``sd`` to ComfyUI's ``key_map``; the count of names added."""
    import comfy.utils

    added = 0
    for k in sd:
        if not (k.startswith("diffusion_model.") and k.endswith(".weight")):
            continue
        name = k[len("diffusion_model."):-len(".weight")]
        for dn, part in diffusers_names(name):
            if part is None:
                # The head keeps ComfyUI's own placement of its name.
                to = key_map.get(name, k)
            elif part == "swap":
                to = (k, None, comfy.utils.swap_scale_shift)
            else:
                n = int(sd[k].shape[0]) // 3
                to = (k, (0, part * n, n))
            for p in PREFIXES:
                if p + dn in key_map:
                    raise SystemExit(f"error: ComfyUI's key map has {p}{dn}")
                key_map[p + dn] = to
                added += 1
    return added


def merge_in_float32(torch) -> None:
    """Patch ComfyUI-GGUF so a weight patched per call is dequantized and
    patched in float32 and cast to the pass dtype once, as
    scripts/video_oracle/check6_h3.py merge_in_float32 does."""
    cls = sys.modules["ComfyUI_GGUF.ops"].GGMLLayer
    stock = cls.get_weight

    def get_weight(self, tensor, dtype):
        if tensor is None or not getattr(tensor, "patches", None) \
                or dtype == torch.float32:
            return stock(self, tensor, dtype)
        return stock(self, tensor, torch.float32).to(dtype)

    cls.get_weight = get_weight


def parse_lora(s: str) -> tuple[Path, float]:
    path, sep, strength = s.rpartition(":")
    try:
        if not sep:
            raise ValueError
        return Path(path).expanduser().resolve(), float(strength)
    except ValueError:
        raise SystemExit(f"error: --lora {s!r} is not PATH:STRENGTH") \
            from None


def source_file(path: Path) -> str | None:
    """The source_file of a safetensors file's metadata, or None."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n)).get("__metadata__") or {}
    return meta.get("source_file")


def add_lora(patcher, path: Path, strength: float) -> tuple:
    """A clone of ``patcher`` with the LoRA at ``path`` added at
    ``strength``, every pair placed, on stock ComfyUI's key map when it
    places every pair and else on it with diffusers' names added, and the
    file's row for the manifest."""
    import comfy.lora
    import comfy.lora_convert
    import comfy.utils

    sd = comfy.utils.load_torch_file(str(path), safe_load=True)
    stems = sorted(k[:-len(".lora_A.weight")] for k in sd
                   if k.endswith(".lora_A.weight"))
    model_sd = patcher.model.state_dict()
    key_map = comfy.lora.model_lora_keys_unet(patcher.model, {})
    stock = comfy.lora.load_lora(comfy.lora_convert.convert_lora(sd),
                                 dict(key_map), log_missing=False)
    keys = "stock"
    if len(stock) != len(stems):
        keys = "extended"
        extend_key_map(key_map, model_sd)
    loaded = comfy.lora.load_lora(comfy.lora_convert.convert_lora(sd),
                                  key_map)
    patched = patcher.clone()
    placed = patched.add_patches(loaded, strength)
    if len(loaded) != len(stems) or len(set(placed)) != len(stems):
        raise SystemExit(f"error: ComfyUI placed {len(set(placed))} of the "
                         f"{len(stems)} pairs of {path.name}")
    ranks = sorted({int(sd[s + ".lora_A.weight"].shape[0]) for s in stems})
    prefix = next((p for p in ("transformer.", "diffusion_model.")
                   if all(s.startswith(p) for s in stems)), "")
    row = {"file": path.name, "strength": strength, "pairs": len(stems),
           "ranks": ranks, "prefix": prefix,
           "alphas": sum(s + ".alpha" in sd for s in stems),
           "stock_key_map_pairs": len(stock), "key_map": keys}
    return patched, row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--comfy-dir", type=Path, required=True,
                    help="ComfyUI checkout with custom_nodes/ComfyUI-GGUF")
    ap.add_argument("--dit", type=Path, required=True,
                    help="H3-Image DiT GGUF with the config key")
    ap.add_argument("--source", type=Path, required=True,
                    help="the release GGUF that --dit copies")
    ap.add_argument("--lora", action="append", required=True,
                    help="PATH:STRENGTH of a LoRA file; repeat to stack")
    ap.add_argument("--converted", type=Path, action="append", default=[],
                    help="diffusers' conversion of a --lora file, which "
                         "gmlx reads in its place; repeat for each")
    ap.add_argument("--case", default="512x512:0.75",
                    help="the step as WxH:SIGMA")
    ap.add_argument("--sample-sigmas", default=None, metavar="S,S,...",
                    help="the sampler's schedule the step is part of, "
                         "ending at 0; the PDD grid by default")
    ap.add_argument("--heads-in-order", action="store_true",
                    help="step k of --sample-sigmas reads PDD head k, as "
                         "diffusers samples, in place of ComfyUI's heads of "
                         "the grid intervals it spans")
    ap.add_argument("--text-len", type=int, default=8)
    ap.add_argument("--feature-std", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="check7_h3_image_lora")
    a = ap.parse_args(argv)
    loras = [parse_lora(s) for s in a.lora]
    converted = {}
    for c in a.converted:
        c = c.expanduser().resolve()
        src = source_file(c)
        hit = [i for i, (p, _) in enumerate(loras) if p.name == src]
        if len(hit) != 1:
            raise SystemExit(f"error: --converted {c.name} names source "
                             f"file {src!r}, which is not one --lora file")
        converted[hit[0]] = c
    w, h, sigma = c3.parse_case(a.case)
    torch = oracle.setup()
    co = c3.comfy_oracle()
    comfy_head = co.start_comfy(a.comfy_dir)
    gguf_head = co.load_comfy_gguf(a.comfy_dir)
    merge_in_float32(torch)
    a.dit = a.dit.expanduser().resolve()
    t0 = time.perf_counter()
    patcher, _ = c3.load_gguf_dit(torch, a.dit)
    rows = []
    for i, (path, strength) in enumerate(loras):
        patcher, row = add_lora(patcher, path, strength)
        if i in converted:
            row["gmlx_file"] = converted[i].name
        rows.append(row)
    import comfy.ldm.modules.attention as cattn
    import comfy.model_management

    comfy.model_management.load_models_gpu([patcher])
    dm = patcher.model.diffusion_model
    print(f"[check7] loaded {a.dit.name} and {len(rows)} LoRA files in "
          f"{time.perf_counter() - t0:.1f}s: {rows}", flush=True)
    rng = np.random.default_rng(a.seed)
    feats = (rng.standard_normal((a.text_len, 5120))
             * a.feature_std).astype(np.float32)
    lat = c3.latents(w, h, a.seed)
    sched = c3.PDD_SIGMAS if a.sample_sigmas is None else tuple(
        float(v) for v in a.sample_sigmas.split(","))
    if sched[-1] != 0.0 or list(sched) != sorted(sched, reverse=True):
        raise SystemExit("error: --sample-sigmas must decrease to 0")
    k = min(range(len(sched)), key=lambda i: abs(sched[i] - sigma))
    if a.heads_in_order:
        import comfy.ldm.minimax.model as cmm

        if k >= dm.final_layer.pdd_steps:
            raise SystemExit(f"error: step {k} has no head of its own")
        span = cmm._pdd_head
        # The one departure: head k alone, in place of the span's heads.
        cmm._pdd_head = lambda head, h, grid, start, stop: span(
            head, h, grid, k, k + 1)
    inputs = {"shape": [w, h], "blocks": len(dm.blocks), "seed": a.seed,
              "features": f"seeded normal, std {a.feature_std}",
              "text_rows": int(feats.shape[0]),
              "latents_shape": list(lat.shape), "dequant": c3.DEQUANT,
              "matmul_bf16": c3.MATMUL,
              "timestep": "sigma x 1000 in float32 in both modes",
              "attention": cattn.optimized_attention.__name__,
              "comfy": comfy_head, "comfy_gguf": gguf_head,
              "adapters": rows, "lora_merge": "float32",
              "lora_merge_means": MERGE, "lora_keys": KEYS}
    files = {"dit": a.source, "oracle_dit": a.dit}
    for i, (path, _) in enumerate(loras):
        files[f"lora_{i}"] = path
        if i in converted:
            files[f"gmlx_lora_{i}"] = converted[i]
    notes = [f"oracle: ComfyUI {comfy_head}, ComfyUI-GGUF {gguf_head}, "
             "UnetLoaderGGUF with dequant_dtype float32, each LoRA through "
             "comfy.lora.load_lora and add_patches at its strength, merge "
             "float32: " + MERGE,
             "lora keys: a file stock ComfyUI's key map places in full goes "
             "through it unchanged (key_map stock), else " + KEYS
             + " (key_map extended)",
             "gmlx_lora_N, where present, is diffusers' conversion of lora_N, "
             "named by its source_file metadata, which gmlx reads in its "
             "place",
             "oracle_dit is dit with the config key {\"transformer\": "
             "{\"pdd_steps\": 6, \"pdd_sigmas\": [...]}} and every tensor "
             "unchanged"]
    if a.heads_in_order:
        notes.append("heads in order: step k of sample_sigmas reads PDD head "
                     "k of the bank, head 0 or head 0 plus block k, as "
                     "diffusers samples; ComfyUI's _pdd_head is called "
                     "with start k and stop k + 1 in place of the span of "
                     "grid intervals the step covers (one departure)")
    c3.run_case(torch, dm, lat, feats, sigma, f"{a.name}_{w}x{h}_s{k}",
                inputs, files, notes, check=7, script=SCRIPT,
                sample_sigmas=None if a.sample_sigmas is None else sched,
                step_head=k if a.heads_in_order else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
