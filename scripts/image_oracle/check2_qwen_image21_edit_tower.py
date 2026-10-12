#!/usr/bin/env python3
"""Check 2 oracle for Qwen-Image 2.1 with reference images: the vision
tower alone, in float32 and in float64.

The tower is set from the mmproj GGUF by the name map of
check2_qwen_image21_edit.py and runs on the patches that fixture holds for
each image, the patches the reference pipeline handed the tower. It runs
twice, in float32 and with every parameter and buffer in float64, on the
CPU. The fixture holds the merger output and each deepstack output of the
float64 run, rounded to float32. The tolerance of each is twice the
oracle's own float32 against float64 relative L2 on it, as check 5 sets
the gate of a float32 VAE pass.

  python scripts/image_oracle/run.py --python <oracle python> \\
      check2_qwen_image21_edit_tower.py --mmproj mmproj-BF16.gguf \\
      --config <official repo dir>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import check2_qwen_image21_edit as c2e
import fixtures
import oracle

NAME = "check2_qwen_image21_edit_tower"
RULE = ("relative L2 against the float64 run at twice the oracle's own "
        "float32 against float64 relative L2 on the same output")


def run(torch, visual, patches: np.ndarray, grid, dtype) -> list:
    out = visual(torch.from_numpy(patches).to(dtype),
                 grid_thw=torch.tensor([grid]))
    return [t.detach().double().numpy()
            for t in (out.pooler_output, *out.deepstack_features)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mmproj", required=True, type=Path,
                    help="Qwen3-VL-8B mmproj GGUF (BF16)")
    ap.add_argument("--config", required=True, type=Path,
                    help="local copy of Qwen/Qwen-Image-2.1 (text_encoder "
                         "config)")
    ap.add_argument("--config-revision",
                    default="790c92633540aa0cb11d9abf19eb46d861714758",
                    help="revision of the --config copy, for the manifest")
    ap.add_argument("--check2", default=c2e.NAME,
                    help="the fixture whose patches the tower runs on")
    ap.add_argument("--name", default=NAME)
    ap.add_argument("--out", type=Path,
                    help="fixture directory (default $GMLX_IMAGE_FIXTURES)")
    a = ap.parse_args(argv)
    torch = oracle.setup()

    from accelerate import init_empty_weights
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    arrays, m2 = fixtures.load(a.check2, root=a.out)
    cfg = AutoConfig.from_pretrained(str(a.config / "text_encoder"),
                                     local_files_only=True)
    with init_empty_weights():
        te = Qwen3VLForConditionalGeneration(cfg)
    census = c2e.set_vision(torch, te, a.mmproj)
    visual = te.model.visual.eval()
    patch = int(cfg.vision_config.patch_size)
    temporal = int(cfg.vision_config.temporal_patch_size)

    fx = fixtures.FixtureWriter(
        a.name, check=2, family="qwen_image21",
        script=f"scripts/image_oracle/{Path(__file__).name}",
        argv=[Path(__file__).name, "--mmproj", a.mmproj.name, "--config",
              a.config.name])
    fx.add_file("mmproj", a.mmproj)
    fx.add_file("text_encoder_config", a.config / "text_encoder/config.json")
    names = sorted(k.removeprefix("patches_") for k in arrays
                   if k.startswith("patches_"))
    runs: dict = {}
    for dtype, key in ((torch.float32, "fp32"), (torch.float64, "fp64")):
        visual.to(dtype)
        for n in names:
            h, w = arrays[f"resized_{n}"].shape[:2]
            grid = [1, h // patch, w // patch]
            px = arrays[f"patches_{n}"]
            if px.shape[0] != grid[1] * grid[2] or px.shape[1] != (
                    3 * temporal * patch * patch):
                raise SystemExit(f"error: patches_{n} {px.shape} for a "
                                 f"grid of {grid}")
            t0 = time.perf_counter()
            runs.setdefault(n, {})[key] = run(torch, visual, px, grid,
                                              dtype)
            runs[n]["grid"] = grid
            print(f"[check2t] {key} {n}: grid {grid}, "
                  f"{time.perf_counter() - t0:.1f} s", flush=True)
    outputs = ["merger"] + [f"deepstack_{i}" for i in
                            range(len(runs[names[0]]["fp64"]) - 1)]
    gaps, tol = {}, {}
    for n in names:
        gaps[n], tol[n] = {}, {}
        for k, f32, f64 in zip(outputs, runs[n]["fp32"], runs[n]["fp64"]):
            fx.add(f"{k}_f64_{n}", f64.astype(np.float32))
            g = fixtures.rel_l2(f32, f64)
            gaps[n][k] = {"rel_l2": g, "cosine": fixtures.cosine(f32, f64)}
            tol[n][k] = 2 * g
        shown = ", ".join(f"{k} {v['rel_l2']:.2e}"
                          for k, v in gaps[n].items())
        print(f"[check2t] {n}: fp32 against fp64 {shown}", flush=True)
    fx.inputs.update(
        source=a.check2, source_npz_sha256={
            f"patches_{n}": m2["tensors"][f"patches_{n}"]["sha256"]
            for n in names},
        grids={n: runs[n]["grid"] for n in names}, outputs=outputs,
        config_revision=a.config_revision,
        tower="Qwen3VLForConditionalGeneration.model.visual set by name "
        "from the mmproj GGUF as check2_qwen_image21_edit.py sets it, its "
        "BF16 and F32 tensors read as float32, then every parameter and "
        "buffer cast to float64 for the second run")
    fx.results.update(oracle={"device": "cpu",
                              "threads": torch.get_num_threads()},
                      fp32_vs_fp64=gaps, gguf_codecs=census)
    fx.tolerances.update(rule=RULE, rel_l2=tol)
    path = fx.write(a.out)
    print(f"[check2t] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
