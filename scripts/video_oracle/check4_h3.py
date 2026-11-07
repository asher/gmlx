#!/usr/bin/env python3
"""Check 4 oracle for MiniMax-H3: sigmas and timesteps of diffusers'
MiniMaxH3Scheduler at the video shift 12, the audio shift 3 and the 768p
Turbo LoRAs' video shift 6.

The scheduler counts grid points, the final zero included, so K forwards
take set_timesteps(K + 1). For each shift and forward count the fixture
holds the float32 sigmas and the scheduler's timesteps, 1 - sigma in
float32, which is the time the DiT runs at. gmlx's schedule must equal
both bit for bit, with the same count. The arrays are small, so the
fixture's .npz is committed beside its manifest.

  python scripts/video_oracle/run.py --python <image-oracle python> \\
      check4_h3.py --config <MiniMax-H3>/scheduler/scheduler_config.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import fixtures
import oracle

ORACLE_NEEDS = ("torch", "diffusers")

NAME = "check4_h3"
SHIFTS = (12.0, 3.0, 6.0)
FORWARDS = (*range(1, 17), 20, 24, 30, 40, 50, 100)


def key(shift: float, n: int) -> str:
    return f"{shift:g}_{n}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, required=True,
                    help="the release's scheduler/scheduler_config.json")
    ap.add_argument("--out", type=Path,
                    help="fixture directory (default $GMLX_VIDEO_FIXTURES)")
    a = ap.parse_args(argv)
    oracle.setup()
    from diffusers import MiniMaxH3Scheduler

    cfg = {k: v for k, v in json.loads(a.config.read_text()).items()
           if not k.startswith("_")}
    fx = fixtures.FixtureWriter(
        NAME, check=4, family="h3",
        script=f"scripts/video_oracle/{Path(__file__).name}",
        argv=[Path(__file__).name])
    fx.add_file("scheduler_config", a.config)
    counts = {}
    for shift in SHIFTS:
        for n in FORWARDS:
            sched = MiniMaxH3Scheduler(**cfg)
            sched.set_shift(shift)
            sched.set_timesteps(n + 1)
            sig = sched.sigmas.numpy().astype(np.float32)
            fx.add(f"sigmas_{key(shift, n)}", sig)
            fx.add(f"timesteps_{key(shift, n)}",
                   sched.timesteps.numpy().astype(np.float32))
            counts[key(shift, n)] = int(sched.timesteps.numel())
    fx.inputs.update(shifts=list(SHIFTS), forwards=list(FORWARDS),
                     scheduler=cfg,
                     grid_points="forwards + 1, the final zero included")
    fx.results.update(forwards_run=counts)
    fx.tolerances.update(
        sigmas="bitwise", timesteps="bitwise", count="exact",
        rule="each sigma and each 1 - sigma equal to the scheduler's bit "
             "for bit, with the same number of forwards")
    short = {k: v for k, v in counts.items() if v != int(k.split("_")[1])}
    print(f"wrote {fx.write(a.out)}; forwards that a repeat dropped: "
          f"{short or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
