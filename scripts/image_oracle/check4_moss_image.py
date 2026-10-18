#!/usr/bin/env python3
"""Check 4 oracle for MOSS-Image: sigmas and timesteps of the release's
MOSSImageScheduler, as MOSSImagePipeline sets them.

The class lives in the release's pipeline_moss_image.py, which the script
loads by path from --pipeline, with the official scheduler config. For
each step count the fixture holds the scheduler's float32 sigmas with the
final zero and its timesteps, and once the training grid the sigmas snap
to. gmlx's sigmas must equal these bit for bit, with the same count. The
grid is a float32 linspace that differs from k / 1000 rounded once by one
ulp at 386 points, and on that other grid a tie can go to the other
point, a whole grid step off, with the count unchanged: at 822 steps one
sigma does.

  python scripts/image_oracle/run.py --python <oracle python> \\
      check4_moss_image.py --pipeline <dir>/pipeline_moss_image.py \\
      --config <dir>/scheduler/scheduler_config.json
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

NAME = "check4_moss_image"
# 250 and 1000 steps snap some sigmas to one grid point, so the release
# runs fewer steps than asked for. At 822 a sigma ties between two grid
# points, which a grid of k / 1000 rounded once breaks the other way.
STEPS = (50, 28, 8, 250, 1000, 822)


def release(path: Path):
    spec = importlib.util.spec_from_file_location("moss_release", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"error: cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pipeline", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True,
                    help="the release's scheduler/scheduler_config.json")
    ap.add_argument("--out", type=Path,
                    help="fixture directory (default $GMLX_IMAGE_FIXTURES)")
    a = ap.parse_args(argv)
    torch = oracle.setup()
    rel = release(a.pipeline)
    cfg = {k: v for k, v in json.loads(a.config.read_text()).items()
           if not k.startswith("_")}
    fx = fixtures.FixtureWriter(
        NAME, check=4, family="moss_image",
        script=f"scripts/image_oracle/{Path(__file__).name}",
        argv=[Path(__file__).name])
    fx.add_file("pipeline", a.pipeline)
    fx.add_file("scheduler_config", a.config)
    counts = {}
    for n in STEPS:
        sched = rel.MOSSImageScheduler(
            num_train_timesteps=cfg["num_train_timesteps"],
            shift=cfg["shift"], training_shift=cfg["training_shift"])
        sched.set_timesteps(n)
        sig = sched.sigmas.numpy().astype(np.float32)
        fx.add(f"sigmas_{n}", sig)
        fx.add(f"timesteps_{n}", sched.timesteps.numpy().astype(np.float32))
        counts[n] = int(sig.size - 1)
    total = cfg["num_train_timesteps"]
    grid = (torch.linspace(1, 1 / total, total).flip(0)).numpy()
    fx.add("training_grid", grid.astype(np.float32))
    exact = (np.arange(1, total + 1) / total).astype(np.float32)
    off = int(np.count_nonzero(grid != exact))
    fx.inputs.update(steps=list(STEPS), scheduler=cfg)
    fx.results.update(steps_run=counts, grid_points_off_k_over_total=off,
                      grid_max_ulps=int(np.max(np.abs(
                          grid.view(np.int32) - exact.view(np.int32)))))
    fx.tolerances.update(
        sigmas="bitwise", count="exact",
        timesteps="sigma x 1000 in float32",
        rule="each sigma equal to the release's bit for bit and the same "
             "number of steps")
    print(f"wrote {fx.write(a.out)}; steps {counts}; grid points off "
          f"k / {total}: {off}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
