#!/usr/bin/env python3
"""Check 4 oracle for Z-Image: sigmas, timesteps and an Euler trajectory
from diffusers, as ZImagePipeline sets its scheduler.

For each step count the fixture holds the scheduler's float32 sigmas with
the final zero, its timesteps and the model timesteps (1000 - t) / 1000.
It also holds float32 torch.linspace(1, 1/n, n) for n from 1 to 100, and
an 8 step trajectory of FlowMatchEulerDiscreteScheduler.step on a constant
velocity, whose exact solution is known. gmlx must equal the sigmas bit
for bit and the trajectory within 1e-6.

The fixture is small, so its .npz is committed beside the manifest under
tests/fixtures/image/oracle/ and the CPU tier reads it there.

  python scripts/image_oracle/run.py --python <oracle python> \\
      check4_z_image.py --config <official repo dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import fixtures
import oracle

NAME = "check4_z_image"
STEPS = (1, 4, 8, 9, 20, 28, 40, 50)
PROBE_MAX = 100
TRAJECTORY_STEPS = 8
SHAPE = (1, 4, 8, 8)
SEED = 4
IMAGE_SEQ_LEN = 4096        # 1024x1024; the static shift ignores it


def _scheduler(config: Path):
    from diffusers import FlowMatchEulerDiscreteScheduler

    return FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(config), subfolder="scheduler", local_files_only=True)


def _set(torch, sched, steps: int):
    from diffusers.pipelines.z_image.pipeline_z_image import (
        calculate_shift, get_default_z_image_sigmas, retrieve_timesteps)

    c = sched.config
    mu = calculate_shift(IMAGE_SEQ_LEN, c.get("base_image_seq_len", 256),
                         c.get("max_image_seq_len", 4096),
                         c.get("base_shift", 0.5), c.get("max_shift", 1.15))
    ts, _ = retrieve_timesteps(sched, steps, torch.device("cpu"),
                               sigmas=get_default_z_image_sigmas(steps),
                               mu=mu)
    sched.set_begin_index(0)
    return ts, mu


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, type=Path,
                    help="local copy of the official repository")
    ap.add_argument("--out", type=Path,
                    help="fixture directory (default $GMLX_IMAGE_FIXTURES)")
    a = ap.parse_args(argv)
    torch = oracle.setup()
    if torch.get_default_dtype() != torch.float32:
        raise SystemExit("error: torch default dtype is not float32")

    fx = fixtures.FixtureWriter(
        NAME, check=4, family="z_image",
        script=f"scripts/image_oracle/{Path(__file__).name}",
        argv=[Path(__file__).name, "--config", a.config.name])
    fx.add_file("scheduler_config",
                a.config / "scheduler" / "scheduler_config.json")
    mu = None
    for n in STEPS:
        sched = _scheduler(a.config)
        ts, mu = _set(torch, sched, n)
        fx.add(f"sigmas_{n}", sched.sigmas.numpy().astype(np.float32))
        fx.add(f"timesteps_{n}", ts.numpy().astype(np.float32))
        fx.add(f"t_model_{n}", ((1000 - ts.float()) / 1000).numpy())

    fx.add("linspace_probe", np.concatenate(
        [torch.linspace(1.0, 1 / n, n).numpy() for n in
         range(1, PROBE_MAX + 1)]).astype(np.float32))

    rng = np.random.default_rng(SEED)
    x0 = rng.standard_normal(SHAPE).astype(np.float32)
    noise = rng.standard_normal(SHAPE).astype(np.float32)
    v = (noise - x0).astype(np.float32)
    sched = _scheduler(a.config)
    ts, _ = _set(torch, sched, TRAJECTORY_STEPS)
    lat = torch.from_numpy(noise.copy())
    traj = [lat.numpy().copy()]
    for t in ts:
        lat = sched.step(torch.from_numpy(v), t, lat, return_dict=False)[0]
        traj.append(lat.numpy().copy())
    traj = np.stack(traj).astype(np.float32)
    fx.add("x0", x0)
    fx.add("noise", noise)
    fx.add(f"trajectory_{TRAJECTORY_STEPS}", traj)

    sig = sched.sigmas.numpy().astype(np.float32)
    exact = noise[None] + (sig[:, None, None, None, None] - 1) * v[None]
    fx.results["trajectory_vs_exact_max_abs"] = fixtures.max_abs(traj, exact)
    fx.results["trajectory_end_vs_x0_max_abs"] = fixtures.max_abs(traj[-1],
                                                                  x0)
    fx.inputs.update(
        steps=list(STEPS), image_seq_len=IMAGE_SEQ_LEN, mu=mu,
        shift=float(sched.config.shift),
        use_dynamic_shifting=bool(sched.config.use_dynamic_shifting),
        linspace_probe=f"torch.linspace(1.0, 1/n, n) float32, n = 1 to "
                       f"{PROBE_MAX}, concatenated",
        trajectory_steps=TRAJECTORY_STEPS, shape=list(SHAPE), seed=SEED,
        noise="numpy default_rng(seed) standard_normal float32: x0, then "
              "noise", velocity="noise - x0, constant")
    fx.tolerances.update(sigmas="bitwise", linspace_probe="bitwise",
                         trajectory_max_abs=1e-6, exact_max_abs=1e-6,
                         rule="plan starting values; the schedule is exact")
    fx.note("The trajectory starts at the noise and steps with the "
            "constant velocity noise - x0, so its exact value at sigma s is "
            "noise + (s - 1) * (noise - x0).")
    path = fx.write(a.out)
    print(f"wrote {path}")
    print(f"trajectory vs exact max abs "
          f"{fx.results['trajectory_vs_exact_max_abs']:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
