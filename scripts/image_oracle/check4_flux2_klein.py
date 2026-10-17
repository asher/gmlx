#!/usr/bin/env python3
"""Check 4 oracle for FLUX.2 klein: the schedule of BFL's reference.

get_schedule of src/flux2/sampling.py forms torch.linspace(1, 0, n + 1) in
float32 and shifts it by the mu of compute_empirical_mu, which reads the
image token count and the step count, and both variants take it. For each
step count and token count the fixture holds the float32 schedule, and at
the default step counts of both variants, for the token counts of a 512
and a 1024 image, the values the samplers read from it: each step's time
as denoise fills its bf16 time tensor, 1000 times that in bf16 as the
embedder reads it, and the step dt, the float32 of the difference of two
times as Python floats, as the bf16 product takes it. gmlx must equal all
of them bit for bit.

  python scripts/image_oracle/run.py --python <oracle python> \\
      check4_flux2_klein.py --reference <flux2>/src \\
      --reference-commit <sha>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import check3_flux2_klein as c3
import fixtures
import oracle

NAME = "check4_flux2_klein"
# Every count to 16, the counts users ask for beyond, and the defaults.
STEPS = (*range(1, 17), 20, 24, 28, 32, 40, 50, 64, 100)
# Image token counts: one token, squares of 256 to 4096 pixels, 768 x
# 1344, and 4300 and 4301, where the rule for mu changes, with a 1056
# pixel square just past it.
TOKENS = (1, 256, 1024, 2304, 4032, 4096, 4300, 4301, 4356, 9216, 16384,
          65536)
CHAIN_STEPS = (4, 50)
CHAIN_TOKENS = (1024, 4096)


def chain(torch, schedule: list) -> dict:
    """What denoise and denoise_cfg read from ``schedule``, step by step,
    as float32 arrays."""
    t_bf16, t1000, dt = [], [], []
    for t_curr, t_prev in zip(schedule[:-1], schedule[1:]):
        filled = torch.full((1,), t_curr, dtype=torch.bfloat16)
        t_bf16.append(filled.float().item())
        # timestep_embedding of src/flux2/model.py multiplies first.
        t1000.append((1000.0 * filled).float().item())
        dt.append(t_prev - t_curr)
    return {k: np.asarray(v, np.float32) for k, v in (
        ("t_bf16", t_bf16), ("t1000", t1000), ("dt", dt))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reference", type=Path, required=True,
                    help="src directory of BFL's flux2")
    ap.add_argument("--reference-commit", required=True,
                    help="the commit the reference directory holds")
    ap.add_argument("--out", type=Path,
                    help="fixture directory (default $GMLX_IMAGE_FIXTURES)")
    a = ap.parse_args(argv)
    torch = oracle.setup()
    _, sampling = c3.reference(a.reference.expanduser())
    fx = fixtures.FixtureWriter(
        NAME, check=4, family="flux2_klein",
        script=f"scripts/image_oracle/{Path(__file__).name}",
        argv=[Path(__file__).name])
    mus = {}
    for n in STEPS:
        rows = []
        for tok in TOKENS:
            s = sampling.get_schedule(n, tok)
            a32 = np.asarray(s, np.float32)
            assert len(s) == n + 1 and np.array_equal(
                a32.astype(np.float64), np.asarray(s, np.float64))
            rows.append(a32)
        fx.add(f"sigmas_{n}", np.stack(rows))
        mus[str(n)] = [sampling.compute_empirical_mu(t, n) for t in TOKENS]
    for n in CHAIN_STEPS:
        for tok in CHAIN_TOKENS:
            for k, v in chain(torch, sampling.get_schedule(n, tok)).items():
                fx.add(f"{k}_{n}_{tok}", v)
    fx.inputs.update(
        steps=list(STEPS), tokens=list(TOKENS), mu=mus,
        chain_steps=list(CHAIN_STEPS), chain_tokens=list(CHAIN_TOKENS),
        reference_commit=a.reference_commit,
        schedule="sigmas_<steps> (len(tokens), steps + 1): row i is "
                 "sampling.get_schedule(steps, tokens[i]), float32 values as "
                 "the Python floats it returns, ending at zero",
        mu_rule="mu[steps][i] is compute_empirical_mu(tokens[i], steps)",
        chain="for each step of sampling.denoise and denoise_cfg: t_bf16 "
              "the step's time filled into a bf16 tensor, t1000 1000 times "
              "that in bf16, dt the float32 of the next time less this one "
              "in Python floats")
    fx.tolerances.update(sigmas="bitwise", chain="bitwise",
                         rule="the schedule is exact")
    print(f"wrote {fx.write(a.out)}; {len(STEPS) * len(TOKENS)} schedules, "
          f"reference {a.reference_commit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
