#!/usr/bin/env python3
"""Check 4 oracle for Krea 2: sigmas and timesteps from diffusers, as
Krea2Pipeline sets its scheduler.

The pipeline passes np.linspace(1.0, 1/n, n) and a mu to
FlowMatchEulerDiscreteScheduler: 1.15 for the distilled Turbo checkpoint,
and for Raw the mu of calculate_shift over the count of image tokens.
The scheduler config is the one the pipeline documents, since the
official one is gated: FlowMatchEulerDiscreteScheduler's defaults with
use_dynamic_shifting and the Krea 2 shift parameters. For each variant,
size and step count the fixture holds the scheduler's float32 sigmas with
the final zero and its timesteps, and for one cell the timestep chain:
the timestep the pipeline passes in float32 latents, t / 1000, and the
value the embedder reads, 1000 times that. gmlx must equal all of them
bit for bit.

  python scripts/image_oracle/run.py --python <oracle python> \\
      check4_krea2.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import fixtures
import oracle

NAME = "check4_krea2"
STEPS = {"turbo": (4, 6, 8, 12), "raw": (20, 28, 36, 52)}
# width, height. Squares of 512, 1024 and 2048 pixels, and two sizes that
# are not square.
SIZES = ((512, 512), (1024, 1024), (2048, 2048), (1344, 768), (832, 1248))
TOKEN_PX = 16
CHAIN = ("raw", 28, (1024, 1024))
CONFIG = {"num_train_timesteps": 1000, "use_dynamic_shifting": True,
          "base_shift": 0.5, "max_shift": 1.15, "base_image_seq_len": 256,
          "max_image_seq_len": 6400}


def tokens(width: int, height: int) -> int:
    return (height // TOKEN_PX) * (width // TOKEN_PX)


def _set(torch, variant: str, steps: int, n_tokens: int):
    """Timesteps, sigmas and mu as Krea2Pipeline.__call__ sets them."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.krea2.pipeline_krea2 import (calculate_shift,
                                                          retrieve_timesteps)

    sched = FlowMatchEulerDiscreteScheduler(**CONFIG)
    c = sched.config
    if variant == "turbo":
        mu = 1.15
    else:
        mu = calculate_shift(n_tokens, c.get("base_image_seq_len", 256),
                             c.get("max_image_seq_len", 6400),
                             c.get("base_shift", 0.5),
                             c.get("max_shift", 1.15))
    sigmas = np.linspace(1.0, 1 / steps, steps)
    ts, _ = retrieve_timesteps(sched, steps, torch.device("cpu"),
                               sigmas=sigmas, mu=mu)
    return ts, sched.sigmas, mu, dict(sched.config)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path,
                    help="fixture directory (default $GMLX_IMAGE_FIXTURES)")
    a = ap.parse_args(argv)
    torch = oracle.setup()
    fx = fixtures.FixtureWriter(
        NAME, check=4, family="krea2",
        script=f"scripts/image_oracle/{Path(__file__).name}",
        argv=[Path(__file__).name])
    mus, config = {}, None
    for variant, steps in STEPS.items():
        for w, h in SIZES:
            n_tokens = tokens(w, h)
            for n in steps:
                ts, sig, mu, config = _set(torch, variant, n, n_tokens)
                mus[f"{variant}_{w}x{h}"] = mu
                fx.add(f"sigmas_{variant}_{n}_{w}x{h}",
                       sig.numpy().astype(np.float32))
                fx.add(f"timesteps_{variant}_{n}_{w}x{h}",
                       ts.numpy().astype(np.float32))
                if (variant, n, (w, h)) == CHAIN:
                    passed = (ts / 1000).to(torch.float32)
                    fx.add("t_passed", passed.numpy())
                    fx.add("t_embedded", (passed.float() * 1e3).numpy())
    fx.inputs.update(
        steps={k: list(v) for k, v in STEPS.items()},
        sizes=[list(s) for s in SIZES],
        tokens={f"{w}x{h}": tokens(w, h) for w, h in SIZES}, mu=mus,
        scheduler={k: v for k, v in config.items()
                   if not k.startswith("_")},
        chain={"variant": CHAIN[0], "steps": CHAIN[1],
               "size": list(CHAIN[2]), "latents": "float32"},
        base_sigmas="np.linspace(1.0, 1 / n, n) in float64, as the "
                    "pipeline passes them")
    fx.tolerances.update(sigmas="bitwise", timesteps="bitwise",
                         chain="bitwise", rule="the schedule is exact")
    fx.note("The official scheduler config is gated. The scheduler here is "
            "the one Krea2Pipeline's docstring asks for: "
            "use_dynamic_shifting with base_shift 0.5, max_shift 1.15, "
            "base_image_seq_len 256 and max_image_seq_len 6400, and the "
            "class defaults for the rest.")
    print(f"wrote {fx.write(a.out)}; mu {mus}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
