#!/usr/bin/env python3
"""Check 4 oracle for MiniMax-H3-Image: the sigmas ComfyUI samples and the
PDD head each step reads.

ComfyUI builds the model's sampling from the GGUF's `config` metadata
(pdd_steps and pdd_sigmas, which ComfyUI-GGUF passes on) as
MiniMaxH3ImageModelSampling, whose sigma is linear in the index of the
PDD grid. The fixture holds that grid, the `simple` schedule at its step
count, which the stock workflow samples, the `simple` schedule at
OFF_GRID_STEPS, whose steps span more than one grid interval, the
timestep the DiT reads at each step (sigma x 1000 in float32), and the
heads each step of those schedules and of the grid itself reads: the
grid points (start, stop) the final layer's PDD rule picks, and the
float32 weights its _pdd_head gives the heads past head 0. For the steps
at OFF_GRID_STEPS it also holds the input and output of the real
_pdd_head on a seeded tiny head bank. The arrays are small, so the
fixture's .npz is committed beside its manifest.

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
# A step count whose `simple` steps each span two grid intervals.
OFF_GRID_STEPS = 3
# The seeded head bank: inputs, outputs a head (the tiny DiT's patch
# rows) and rows read.
HEAD_IN, HEAD_OUT, HEAD_ROWS = 16, 4, 5


def comfy_oracle():
    """scripts/video_oracle/oracle.py: ComfyUI and ComfyUI-GGUF on the
    CPU."""
    path = Path(__file__).resolve().parent.parent / "video_oracle" / \
        "oracle.py"
    spec = importlib.util.spec_from_file_location("_h3_comfy_oracle", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pdd_calls(torch, fl, grid: list, sample_sigmas, x, t_emb) -> list:
    """The input, output, start and stop of each call ComfyUI's final
    layer ``fl`` makes to the real _pdd_head, one a sigma of
    ``sample_sigmas`` but the last."""
    import comfy.ldm.minimax.model as cm

    calls = []
    real = cm._pdd_head

    def spy(head, h, g, start, stop):
        y = real(head, h, g, start, stop)
        calls.append((h.numpy().copy(), y.numpy().copy(), start, stop))
        return y

    cm._pdd_head = spy
    try:
        with torch.no_grad():
            for s in sample_sigmas[:-1]:
                ts = s.reshape(1).float() * 1000
                sigma = float(ts.flatten()[0]) / 1000.0
                fl(x, t_emb, (0, x.shape[0], 0), None, sigma, sample_sigmas,
                   (grid,))
    finally:
        cm._pdd_head = real
    return calls


def final_layer(torch, hidden: int, out: int, steps: int):
    import comfy.ldm.minimax.model as cm
    import comfy.ops

    return cm.FinalLayer(hidden, 8, out, None, 1e-5, apply_silu=False,
                         pdd_steps=steps, dtype=torch.float32, device="cpu",
                         operations=comfy.ops.manual_cast)


def head_picks(torch, steps: int, grid: list, sample_sigmas) -> list:
    """The (start, stop) grid points ComfyUI's final layer picks at each
    sigma of ``sample_sigmas`` but the last, and the float32 weights its
    _pdd_head gives the heads past head 0: read off a bank whose weights
    are zero and whose bias block k is the unit vector k, so entry k of
    the output is head k's weight, exactly."""
    fl = final_layer(torch, 2, steps, steps)
    with torch.no_grad():
        fl.video_out.weight.zero_()
        fl.video_out.bias.zero_()
        for k in range(1, steps):
            fl.video_out.bias[k * steps + k] = 1.0
    picks = []
    for _, y, start, stop in pdd_calls(torch, fl, grid, sample_sigmas,
                                       torch.zeros(1, 2), torch.zeros(1, 8)):
        first = max(start, 1)
        if y[0, :first].any() or y[0, stop:].any():
            raise SystemExit(f"error: the head bank read outside {start} to "
                             f"{stop}")
        picks.append((start, stop, y[0, first:stop].tolist()))
    return picks


def head_record(torch, steps: int, grid: list, sample_sigmas) -> dict:
    """The real _pdd_head's float32 input and output at each step of
    ``sample_sigmas`` on a seeded bank of ``steps`` heads of HEAD_OUT
    outputs over HEAD_IN inputs, read on HEAD_ROWS rows."""
    torch.manual_seed(0)
    fl = final_layer(torch, HEAD_IN, HEAD_OUT, steps)
    with torch.no_grad():
        for p in fl.parameters():
            p.copy_(torch.randn_like(p))
    x, t_emb = torch.randn(HEAD_ROWS, HEAD_IN), torch.randn(1, 8)
    calls = pdd_calls(torch, fl, grid, sample_sigmas, x, t_emb)
    return dict(weight=fl.video_out.weight.detach().numpy(),
                bias=fl.video_out.bias.detach().numpy(),
                h=np.stack([h for h, *_ in calls]),
                y=np.stack([y for _, y, *_ in calls]),
                spans=[(a, b) for *_, a, b in calls])


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
    off_grid = comfy.samplers.simple_scheduler(ms, OFF_GRID_STEPS)
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
    fx.add(f"simple_{OFF_GRID_STEPS}", off_grid.numpy().astype(np.float32))
    fx.add(f"timesteps_simple_{OFF_GRID_STEPS}",
           ms.timestep(off_grid[:-1]).float().numpy())
    picks = {}
    for name, sched in ((f"simple_{steps}", simple), ("grid", grid),
                        (f"simple_{OFF_GRID_STEPS}", off_grid)):
        got = head_picks(torch, steps, pdd_sigmas, sched)
        fx.add(f"heads_{name}", np.array([(s, e) for s, e, _ in got],
                                         np.int64))
        picks[name] = [w for _, _, w in got]
    rec = head_record(torch, steps, pdd_sigmas, off_grid)
    if rec["spans"] != [(s, e) for s, e, _ in head_picks(
            torch, steps, pdd_sigmas, off_grid)]:
        raise SystemExit("error: the seeded bank read other spans")
    fx.add("head_bank_weight", rec["weight"])
    fx.add("head_bank_bias", rec["bias"])
    fx.add(f"head_in_simple_{OFF_GRID_STEPS}", rec["h"])
    fx.add(f"head_out_simple_{OFF_GRID_STEPS}", rec["y"])
    off = float((simple - grid).abs().max())
    fx.inputs.update(steps=steps, pdd_sigmas=unet["pdd_sigmas"],
                     scheduler="simple", off_grid_steps=OFF_GRID_STEPS,
                     comfy=comfy_head,
                     comfy_gguf=gguf_head,
                     model_sampling=type(ms).__name__,
                     config_key=extra["metadata"].get("config"),
                     head_bank=dict(seed=0, inputs=HEAD_IN, outputs=HEAD_OUT,
                                    rows=HEAD_ROWS))
    fx.results.update(simple_off_grid=off, head_weights=picks)
    fx.tolerances.update(
        grid="bitwise", heads="exact", head_out_rel=1e-6,
        rule="the sampling row's sigmas and final zero equal ComfyUI's PDD "
             "grid bit for bit; each step reads the heads ComfyUI's PDD rule "
             "picks, its start and stop grid points and the float32 weights "
             "of the heads past head 0, on the simple schedule at the step "
             f"count, on the grid, and on the simple schedule at "
             f"{OFF_GRID_STEPS} steps; on that last schedule the head's output "
             "on the seeded bank is within float32 rounding (relative L2) of "
             "the real _pdd_head's")
    fx.note(f"oracle: ComfyUI {comfy_head}, ComfyUI-GGUF {gguf_head}; "
            "the model config from the GGUF's config key")
    print(f"wrote {fx.write(a.out, compressed=True)}; simple off the grid by "
          f"{off:.3e}; heads {picks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
