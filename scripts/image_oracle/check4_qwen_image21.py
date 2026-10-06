#!/usr/bin/env python3
"""Check 4 oracle for Qwen-Image 2.1: sigmas and timesteps from diffusers,
as QwenImage21Pipeline sets its scheduler.

The pipeline passes np.linspace(1.0, 1/n, n) and a mu from the count of
target latent tokens to FlowMatchEulerDiscreteScheduler, which shifts the
sigmas exponentially and stretches them to its terminal value. For each
size and step count the fixture holds the scheduler's float32 sigmas with
the final zero and its timesteps. For one cell it also holds the model
timestep chain: the timestep the pipeline passes, t / 1000, and the value
the transformer embeds, 1000 * that. gmlx must equal all of them bit for
bit.

The fixture is small, so its .npz is committed beside the manifest under
tests/fixtures/image/oracle/ and the CPU tier reads it there.

  python scripts/image_oracle/run.py --python <oracle python> \\
      check4_qwen_image21.py --scheduler-config <scheduler_config.json>
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

import fixtures
import oracle

NAME = "check4_qwen_image21"
STEPS = (2, 4, 8, 9, 20, 28, 40, 50)
# width, height. The plan's three squares and two that are not square.
SIZES = ((512, 512), (1024, 1024), (2048, 2048), (1344, 768), (832, 1248))
VAE_SCALE = 16
CHAIN = (40, (1024, 1024))


def _scheduler(config: dict):
    from diffusers import FlowMatchEulerDiscreteScheduler

    return FlowMatchEulerDiscreteScheduler.from_config(config)


def tokens(width: int, height: int) -> int:
    """Target latent tokens, as prepare_latents sizes them."""
    h = 2 * (int(height) // (VAE_SCALE * 2))
    w = 2 * (int(width) // (VAE_SCALE * 2))
    return h * w


def _set(torch, sched, steps: int, n_tokens: int):
    """Timesteps and mu as QwenImage21Pipeline.__call__ sets them."""
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import (
        calculate_shift, retrieve_timesteps)

    c = sched.config
    sigmas = np.linspace(1.0, 1 / steps, steps)
    mu = calculate_shift(n_tokens, c.get("base_image_seq_len", 256),
                         c.get("max_image_seq_len", 4096),
                         c.get("base_shift", 0.5), c.get("max_shift", 1.15))
    ts, _ = retrieve_timesteps(sched, steps, torch.device("cpu"),
                               sigmas=sigmas, mu=mu)
    return ts, mu


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scheduler-config", required=True, type=Path,
                    help="scheduler/scheduler_config.json of the official "
                         "repository")
    ap.add_argument("--config-revision",
                    default="790c92633540aa0cb11d9abf19eb46d861714758",
                    help="revision of the scheduler config, for the manifest")
    ap.add_argument("--out", type=Path,
                    help="fixture directory (default $GMLX_IMAGE_FIXTURES)")
    a = ap.parse_args(argv)
    torch = oracle.setup()
    if torch.get_default_dtype() != torch.float32:
        raise SystemExit("error: torch default dtype is not float32")
    config = {k: v for k, v in
              json.loads(a.scheduler_config.read_text()).items()
              if not k.startswith("_")}

    fx = fixtures.FixtureWriter(
        NAME, check=4, family="qwen_image21",
        script=f"scripts/image_oracle/{Path(__file__).name}",
        argv=[Path(__file__).name, "--scheduler-config",
              a.scheduler_config.name])
    fx.add_file("scheduler_config", a.scheduler_config)
    mus = {}
    for w, h in SIZES:
        n_tokens = tokens(w, h)
        for n in STEPS:
            sched = _scheduler(config)
            ts, mu = _set(torch, sched, n, n_tokens)
            mus[f"{w}x{h}"] = mu
            fx.add(f"sigmas_{n}_{w}x{h}",
                   sched.sigmas.numpy().astype(np.float32))
            fx.add(f"timesteps_{n}_{w}x{h}", ts.numpy().astype(np.float32))
            if (n, (w, h)) == CHAIN:
                # QwenImage21Pipeline: timestep = t.expand(B) / 1000, and
                # QwenImage21TemporalTimesteps: 1000.0 * timestep.float().
                passed = ts / 1000
                fx.add("t_passed", passed.numpy().astype(np.float32))
                fx.add("t_embedded",
                       (1000.0 * passed.float()).numpy().astype(np.float32))

    # One step: the stretch divides zero by zero.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sched = _scheduler(config)
        _set(torch, sched, 1, tokens(1024, 1024))
    one = sched.sigmas.numpy().astype(np.float32)
    fx.results["one_step_sigmas_are_nan"] = bool(np.isnan(one[0]))

    fx.inputs.update(
        steps=list(STEPS), sizes=[list(s) for s in SIZES],
        tokens={f"{w}x{h}": tokens(w, h) for w, h in SIZES}, mu=mus,
        scheduler=config, config_revision=a.config_revision,
        chain={"steps": CHAIN[0], "size": list(CHAIN[1])},
        base_sigmas="np.linspace(1.0, 1 / n, n) in float64, as the "
                    "pipeline passes them")
    fx.tolerances.update(sigmas="bitwise", timesteps="bitwise",
                         chain="bitwise",
                         rule="the schedule is exact")
    fx.note("NumPy 2 rounds a Python scalar to the array's dtype, so every "
            "scheduler operation after the float32 cast is in float32. "
            "Under NumPy 1 the stretch's scale factor is a float64.")
    path = fx.write(a.out)
    print(f"wrote {path}")
    print(f"mu {mus}; one step NaN "
          f"{fx.results['one_step_sigmas_are_nan']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
