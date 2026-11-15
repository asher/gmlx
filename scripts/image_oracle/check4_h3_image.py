#!/usr/bin/env python3
"""Check 4 oracle for MiniMax-H3-Image: the sigmas ComfyUI samples and the
PDD head each step reads.

ComfyUI builds the model's sampling from the GGUF's `config` metadata
(pdd_steps and pdd_sigmas, which ComfyUI-GGUF passes on) as
MiniMaxH3ImageModelSampling, whose sigma is linear in the index of the
PDD grid. The fixture holds that grid, the `simple` schedule at its step
count, which the stock workflow samples, the timestep the DiT reads at
each step (sigma x 1000 in float32), and the heads each step of the
schedule and of the grid itself reads: the grid points (start, stop) the
final layer's PDD rule picks, and their dt weights. The arrays are small,
so the fixture's .npz is committed beside its manifest.

  python scripts/image_oracle/run.py --python <ComfyUI venv python> \\
      check4_h3_image.py --comfy-dir <ComfyUI checkout> \\
      --dit <lab copy with the config key> --pdd-config pdd_config.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

import fixtures
import oracle

ORACLE_NEEDS = ("torch", "gguf", "einops", "comfy_kitchen")

NAME = "check4_h3_image"
FAMILY = "h3_image"


def comfy_oracle():
    """scripts/video_oracle/oracle.py: ComfyUI and ComfyUI-GGUF on the
    CPU."""
    path = Path(__file__).resolve().parent.parent / "video_oracle" / \
        "oracle.py"
    spec = importlib.util.spec_from_file_location("_h3_comfy_oracle", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def head_picks(torch, steps: int, grid: list, sample_sigmas):
    """The (start, stop) grid points and dt weights ComfyUI's final layer
    takes at each sigma of ``sample_sigmas`` but the last: a tiny final
    layer with the release's head count, whose PDD rule is recorded."""
    import comfy.ldm.minimax.model as cm
    import comfy.ops

    picks = []
    real = cm._pdd_head

    def spy(head, h, g, start, stop):
        dt = [g[k] - g[k + 1] for k in range(start, stop)]
        picks.append((start, stop, [d / sum(dt) for d in dt]))
        return real(head, h, g, start, stop)

    fl = cm.FinalLayer(8, 8, 4, None, 1e-5, apply_silu=False, pdd_steps=steps,
                       dtype=torch.float32, device="cpu",
                       operations=comfy.ops.manual_cast)
    x, t_emb = torch.zeros(2, 8), torch.zeros(1, 8)
    cm._pdd_head = spy
    try:
        for s in sample_sigmas[:-1]:
            ts = s.reshape(1).float() * 1000
            sigma = float(ts.flatten()[0]) / 1000.0
            fl(x, t_emb, (0, 2, 0), None, sigma, sample_sigmas, (grid,))
    finally:
        cm._pdd_head = real
    return picks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--comfy-dir", type=Path, required=True,
                    help="ComfyUI checkout with custom_nodes/ComfyUI-GGUF")
    ap.add_argument("--dit", type=Path, required=True,
                    help="H3-Image DiT GGUF with the config key")
    ap.add_argument("--pdd-config", type=Path, required=True,
                    help="the release's pdd_config.json")
    ap.add_argument("--out", type=Path,
                    help="fixture directory (default $GMLX_IMAGE_FIXTURES)")
    a = ap.parse_args(argv)
    torch = oracle.setup()
    co = comfy_oracle()
    comfy_head = co.start_comfy(a.comfy_dir)
    gguf_head = co.load_comfy_gguf(a.comfy_dir)
    import comfy.model_detection
    import comfy.samplers
    from comfy.ldm.minimax.sampling import MiniMaxH3ImageModelSampling
    from ComfyUI_GGUF.loader import gguf_sd_loader

    sd, extra = gguf_sd_loader(str(a.dit.expanduser().resolve()))
    config = comfy.model_detection.model_config_from_unet(
        sd, "", metadata=extra.get("metadata"))
    if type(config).__name__ != "MiniMaxH3Image":
        raise SystemExit(f"error: ComfyUI detects {type(config).__name__}")
    unet = config.unet_config
    # model_base.MiniMaxH3Image's sampling and its model's grid, built
    # without the model's weights.
    ms = MiniMaxH3ImageModelSampling(config)
    pdd_sigmas = [float(s) for s in unet["pdd_sigmas"]]
    steps = int(unet["pdd_steps"])
    pdd = json.loads(a.pdd_config.read_text())
    grid = ms.grid.flip(0)
    if grid.tolist() != torch.tensor(pdd["sigmas"], dtype=torch.float32
                                     ).tolist():
        raise SystemExit("error: ComfyUI's grid is not pdd_config.json's")
    simple = comfy.samplers.simple_scheduler(ms, steps)
    fx = fixtures.FixtureWriter(
        NAME, check=4, family=FAMILY,
        script=f"scripts/image_oracle/{Path(__file__).name}")
    fx.add_file("pdd_config", a.pdd_config)
    fx.add_file("oracle_dit", a.dit)
    fx.add("grid", grid.numpy().astype(np.float32))
    fx.add(f"simple_{steps}", simple.numpy().astype(np.float32))
    fx.add(f"timesteps_simple_{steps}",
           ms.timestep(simple[:-1]).float().numpy())
    fx.add("timesteps_grid", ms.timestep(grid[:-1]).float().numpy())
    picks = {}
    for name, sched in ((f"simple_{steps}", simple), ("grid", grid)):
        got = head_picks(torch, steps, pdd_sigmas, sched)
        fx.add(f"heads_{name}", np.array([(s, e) for s, e, _ in got],
                                         np.int64))
        picks[name] = [w for _, _, w in got]
    off = float((simple - grid).abs().max())
    fx.inputs.update(steps=steps, pdd_sigmas=unet["pdd_sigmas"],
                     scheduler="simple", comfy=comfy_head,
                     comfy_gguf=gguf_head,
                     model_sampling=type(ms).__name__,
                     config_key=extra["metadata"].get("config"))
    fx.results.update(simple_off_grid=off, head_weights=picks)
    fx.tolerances.update(
        grid="bitwise", heads="exact",
        rule="the sampling row's sigmas and final zero equal ComfyUI's PDD "
             "grid bit for bit; each step reads the head ComfyUI's PDD rule "
             "picks, one grid interval with weight 1, on the simple "
             "schedule and on the grid")
    fx.note(f"oracle: ComfyUI {comfy_head}, ComfyUI-GGUF {gguf_head}; "
            "the model config from the GGUF's config key")
    print(f"wrote {fx.write(a.out, compressed=True)}; simple off the grid by "
          f"{off:.3e}; heads {picks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
