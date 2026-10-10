"""Check 6 of the image acceptance gate for Krea 2 Turbo, on latents: a
whole sampling run on injected noise. gmlx takes the oracle's
conditioning, samples from the seed's noise with the Turbo defaults, and
the final latents and the latents after each step the oracle kept must
stay within twice the oracle's own bf16 against fp32 gap after that step,
as the manifest records it. The seed's noise must equal the oracle's,
and the schedule of the request must equal the schedule of the
pipeline's own call bit for bit. A schedule shifted by the image token
count at the same size fails the schedule row and the kept steps.

Skipped unless the manifest of scripts/image_oracle/check6_krea2.py is
committed, GMLX_IMAGE_FIXTURES holds its arrays and
KQUANT_TEST_GGUF_DIR the DiT GGUF the oracle read."""
from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from image_fixtures import image_dits
from oracle_fixture import ORACLE, committed, fixture, fx

pytestmark = pytest.mark.integration

NAME = "check6_krea2"


def _manifest() -> dict:
    if not (ORACLE / f"{NAME}.manifest.json").is_file():
        pytest.skip(f"{NAME}.manifest.json is not committed: run "
                    "scripts/image_oracle/check6_krea2.py on the Krea 2 "
                    "Turbo GGUF and commit the manifest it writes")
    return committed(NAME)


@pytest.fixture(scope="module")
def check6():
    _manifest()
    return fixture(NAME)


@pytest.fixture(scope="module")
def pipe(check6, gguf_dir):
    from gmlx.image import api

    want = check6[1]["files"]["dit"]
    for p in image_dits(gguf_dir, "krea2"):
        if Path(p).name == want["name"] and os.path.getsize(p) == want["size"]:
            break
    else:
        pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")
    if fx.file_identity(p)["sha256"] != want["sha256"]:
        pytest.fail(f"{p} is not the file the oracle read")
    return api.load(p, log=lambda *_: None)


def _bits(a) -> list:
    return np.asarray(a, np.float32).view(np.uint32).tolist()


def _sizes(m: dict) -> list:
    return [int(s) for s in m["inputs"]["sizes"]]


def _schedule_misses(got, arrays: dict, size: int) -> list[str]:
    got = np.asarray(got, np.float32)
    out = []
    if _bits(got) != _bits(arrays[f"sigmas_{size}"]):
        out.append(f"sigmas of {size}")
    if _bits(got[:-1] * np.float32(1000)) != _bits(
            arrays[f"timesteps_{size}"]):
        out.append(f"timesteps of {size}")
    return out


def _gates(m: dict, size: int) -> dict:
    """The gate of the latents after each kept step and after the last,
    by the manifest's rule."""
    sizes = m["results"]["sizes"]
    src = size if "bf16_vs_fp32_by_step" in sizes[str(size)] else max(
        int(s) for s, r in sizes.items() if "bf16_vs_fp32_by_step" in r)
    by_step = sizes[str(src)]["bf16_vs_fp32_by_step"]
    steps = m["inputs"]["steps"]
    return {k: 2 * by_step[k - 1] for k in m["inputs"]["kept_steps"]
            + [steps]}


def _run(pipe, arrays: dict, m: dict, size: int, sigmas=None,
         stop: int | None = None) -> dict:
    """The relative L2 of the latents after each kept step and after the
    last, against the oracle's."""
    from gmlx.image import api
    from gmlx.image.conditioning import Conditioning

    steps = m["inputs"]["steps"]
    want = {k: arrays[f"latents_{size}_step{k}"]
            for k in m["inputs"]["kept_steps"]}
    want[steps] = arrays[f"latents_{size}"]
    cond = Conditioning(mx.array(arrays["features"]), f"check6 {size}")
    got: dict = {}

    def on_step(i, x):
        if i + 1 in want:
            got[i + 1] = fx.rel_l2(np.array(x), want[i + 1])

    if sigmas is not None and stop is not None:
        sigmas = sigmas[:stop + 1]
    api.sample(pipe, cond, height=size, width=size,
               noise=mx.array(arrays[f"noise_{size}"]), sigmas=sigmas,
               on_step=on_step)
    return got


def test_the_seed_gives_the_noise_of_the_oracle(check6):
    from gmlx.image.samplers import noise_from_seed

    arrays, m = check6
    for size in _sizes(m):
        noise = arrays[f"noise_{size}"]
        assert noise.shape == (1, 16, size // 8, size // 8)
        assert np.array_equal(
            noise_from_seed(m["inputs"]["seed"], noise.shape), noise)


def test_the_schedule_of_a_request_is_the_schedule_of_the_pipeline(check6,
                                                                   pipe):
    arrays, m = check6
    steps = m["inputs"]["steps"]
    assert pipe.defaults.steps == steps and pipe.defaults.cfg == 1.0
    for size in _sizes(m):
        assert not _schedule_misses(pipe.sigmas(steps, (size // 16) ** 2),
                                    arrays, size)


def test_a_token_count_shift_fails_the_turbo_rows(check6, pipe):
    from gmlx.image.schedule import calculate_shift, krea2_sigmas

    arrays, m = check6
    size = _sizes(m)[0]
    steps = m["inputs"]["steps"]
    tokens = (size // 16) ** 2
    # The dynamic shift the Krea2Pipeline docstring gives.
    raw = krea2_sigmas(steps, calculate_shift(tokens, 256, 6400, 0.5, 1.15))
    assert _schedule_misses(raw, arrays, size)
    gates = _gates(m, size)
    got = _run(pipe, arrays, m, size, sigmas=raw, stop=2)
    share = {k: v / gates[k] for k, v in got.items()}
    print(f"krea2 check6 token count shift {size}: kept steps "
          + ", ".join(f"{k} {v:.2f} of its gate" for k, v in share.items()))
    assert max(share.values()) > 1.0, share


@pytest.mark.parametrize("size", [512, 1024])
def test_check6_final_latents(check6, pipe, size):
    arrays, m = check6
    if size not in _sizes(m):
        pytest.skip(f"the oracle did not run {size}")
    steps = m["inputs"]["steps"]
    gates = _gates(m, size)
    got = _run(pipe, arrays, m, size)
    tol = m["tolerances"]["sizes"][str(size)]["latents_rel_l2"]
    assert gates[steps] == pytest.approx(tol)
    print(f"krea2 check6 latents {size}: "
          + ", ".join(f"step {k} {v:.2e} ({v / gates[k]:.2f} of its gate)"
                      for k, v in sorted(got.items())))
    assert sorted(got) == sorted(gates)
    over = {k: (v, gates[k]) for k, v in got.items() if v > gates[k]}
    assert not over, over
