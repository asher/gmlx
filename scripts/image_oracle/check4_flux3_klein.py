#!/usr/bin/env python3
"""Check 4 oracle for FLUX.3 klein: the schedule of BFL's reference.

timesteps of src/flux3/sampling.py forms torch.linspace(1.0, 0.0, n + 1)
in float32 and shifts it by alpha, alpha * t / (1 + (alpha - 1) * t), with
the alpha that Pipeline.generate defaults to. The schedule depends on
neither the image size nor the variant. For each step count the fixture
holds the float32 schedule, and for the default step counts of both
variants the values the samplers read from it: each step's time as the
reference fills its bf16 time tensor, 1000 times that in bf16 as the
embedder reads it, the target time of the flow map in bf16, and the step
dt, the float32 difference of two times. gmlx must equal all of them bit
for bit.

  python scripts/image_oracle/run.py --python <oracle python> \\
      check4_flux3_klein.py --reference <flux-3-image-klein>/src
"""

from __future__ import annotations

import argparse
import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np

import fixtures
import oracle

NAME = "check4_flux3_klein"
# Every count to 64, then the counts a user may ask for beyond.
STEPS = (*range(1, 65), 80, 100, 128, 200)


def reference(path: Path):
    """The reference's pipeline and sampling modules, and its git
    commit."""
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(path))
    from flux3 import pipeline, sampling

    try:
        commit = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return pipeline, sampling, commit


def chain(torch, schedule) -> dict:
    """What the samplers of src/flux3/sampling.py read from ``schedule``,
    step by step, as float32 arrays."""
    t_bf16, t1000, target, dt = [], [], [], []
    filled = torch.zeros(1, dtype=torch.bfloat16)
    for step in range(len(schedule) - 1):
        t = schedule[step].item()
        d = (schedule[step + 1] - schedule[step]).item()
        filled.fill_(t)
        t_bf16.append(filled.float().item())
        # timestep_embedding of src/flux3/dit.py multiplies first.
        t1000.append((1000.0 * filled).float().item())
        target.append(schedule[step + 1].reshape(1).to(torch.bfloat16)
                      .float().item())
        dt.append(d)
    return {k: np.asarray(v, np.float32) for k, v in (
        ("t_bf16", t_bf16), ("t1000", t1000), ("target_bf16", target),
        ("dt", dt))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reference", type=Path, required=True,
                    help="src directory of BFL's flux-3-image-klein")
    ap.add_argument("--out", type=Path,
                    help="fixture directory (default $GMLX_IMAGE_FIXTURES)")
    a = ap.parse_args(argv)
    torch = oracle.setup()
    pipeline, sampling, commit = reference(a.reference.expanduser())
    alpha = inspect.signature(pipeline.Pipeline.generate).parameters[
        "alpha"].default
    models = {k: dict(v) for k, v in pipeline.MODELS.items()}
    fx = fixtures.FixtureWriter(
        NAME, check=4, family="flux3_klein",
        script=f"scripts/image_oracle/{Path(__file__).name}",
        argv=[Path(__file__).name])
    for n in STEPS:
        s = sampling.timesteps(n, alpha)
        assert s.dtype == torch.float32 and s.shape == (n + 1,)
        fx.add(f"sigmas_{n}", s.numpy())
    chains = sorted({v["steps"] for v in models.values()})
    for n in chains:
        for k, v in chain(torch, sampling.timesteps(n, alpha)).items():
            fx.add(f"{k}_{n}", v)
    fx.inputs.update(
        steps=list(STEPS), alpha=alpha, chain_steps=chains, models=models,
        reference_commit=commit,
        schedule="sampling.timesteps(steps, alpha), float32, length "
                 "steps + 1 ending at zero",
        chain="for each step of sampling.denoise and denoise_flow_map: "
              "t_bf16 the step's time filled into a bf16 tensor, t1000 "
              "1000 times that in bf16, target_bf16 the next time cast to "
              "bf16, dt the float32 difference of the next time and this "
              "one")
    fx.tolerances.update(sigmas="bitwise", chain="bitwise",
                         rule="the schedule is exact")
    print(f"wrote {fx.write(a.out)}; alpha {alpha}, chains {chains}, "
          f"reference {commit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
