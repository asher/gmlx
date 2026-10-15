"""Check 6 of the image acceptance gate for Qwen-Image 2.1, on latents: a
whole sampling run on injected noise. gmlx takes the oracle's
conditioning, samples from the oracle's noise with its default steps and
guidance, and the final latents must stay within twice the oracle's own
bf16 against fp32 gap, as the manifest records it. The latents after each
step the oracle kept must stay within twice that gap after that step. The
seed's noise must equal the oracle's noise, and the schedule of the
request must equal the schedule of the pipeline's own call bit for bit.

The final latents do not see a wrong schedule: a run with no shift at all
reads 0.83 of their gate at 512. Two other rows hold the schedule. The
schedule row compares the sigmas with the oracle's in bits, so it fails
on any other schedule. The kept steps hold the loop on the real model
from its first step. ``test_each_wrong_schedule_fails_its_rows`` makes
each schedule below and asserts both readings. A kept number is the
larger of steps 1 and 2 over its gate as one run read it, and a number at
or under 1 passes; a right run reads 0.43 at 512 and 0.36 at 1024.

  wrong schedule                     schedule row  kept, 512   kept, 1024
  max_shift x 1.1                    fails         0.75        2.90
  base_shift x 1.1                   fails         3.30        1.73
  mu x 1.1                           fails         3.92        4.53
  mu x 0.9                           fails         4.25        4.80
  39 steps                           fails         2.32        2.03
  the last step left out             fails         0.43        0.36
  the first step left out            fails         23.9        15.0
  the stretch to the terminal off    fails         1.81        2.17
  terminal 0.03                      fails         0.82        0.81
  the tokens of four times the area  fails         10.7        31.2
  the tokens of the other size       fails         10.7        11.2
  no shift                           fails         51.5        65.1

The first step left out also fails the final latents at 512 (2.70 of
their gate). A schedule whose last step is left out has no kept step
after the fault, and the kept steps pass max_shift x 1.1 at 512 and a
terminal of 0.03: the schedule row alone holds those three. A loop that
takes the right schedule and leaves its last step out is held by the
tiny runs of test_image_api.py and test_image_qwen_model.py, which need
no model file.

Skipped unless the manifest of
scripts/image_oracle/check6_qwen_image21.py is committed,
GMLX_IMAGE_FIXTURES holds its arrays and KQUANT_TEST_GGUF_DIR the DiT
GGUF the oracle read."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from image_fixtures import image_dits
from oracle_fixture import ORACLE, committed, fixture, fx

import mlx.core as mx

# A Mac of 16 GB, as most users run (tests/host_box.py).
pytestmark = [pytest.mark.integration, pytest.mark.host_gb(16)]

NAME = "check6_qwen_image21"


def _manifest() -> dict:
    if not (ORACLE / f"{NAME}.manifest.json").is_file():
        pytest.skip(f"{NAME}.manifest.json is not committed: run "
                    "scripts/image_oracle/check6_qwen_image21.py on the "
                    "Qwen-Image 2.1 GGUF and commit the manifest it writes")
    return committed(NAME)


@pytest.fixture(scope="module")
def check6():
    _manifest()
    return fixture(NAME)


@pytest.fixture(scope="module")
def pipe(check6, gguf_dir):
    from gmlx.image import api

    want = check6[1]["files"]["dit"]
    for p in image_dits(gguf_dir, "qwen_image21"):
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
    """What the schedule row holds of one size: the sigmas and the
    timesteps of the oracle's own call, in bits."""
    got = np.asarray(got, np.float32)
    out = []
    if _bits(got) != _bits(arrays[f"sigmas_{size}"]):
        out.append(f"sigmas of {size}")
    if _bits(got[:-1] * np.float32(1000)) != _bits(
            arrays[f"timesteps_{size}"]):
        out.append(f"timesteps of {size}")
    return out


def _kept_gates(m: dict, size: int) -> dict:
    """The gate of the latents after each kept step, by the manifest's
    rule: twice the oracle's own bf16 against fp32 relative L2 after that
    step, and a size without a bf16 run takes the gates of the largest
    size that has one."""
    sizes = m["results"]["sizes"]
    src = size if "bf16_vs_fp32_by_step" in sizes[str(size)] else max(
        int(s) for s, r in sizes.items() if "bf16_vs_fp32_by_step" in r)
    by_step = sizes[str(src)]["bf16_vs_fp32_by_step"]
    return {k: 2 * by_step[k - 1] for k in m["inputs"]["kept_steps"]}


def _wrong_schedules(steps: int, tokens: int) -> dict:
    """Each wrong schedule of the module docstring, by name."""
    from gmlx.image import schedule as sc

    f32 = np.float32
    right = sc.qwen21_sigmas(steps, tokens)
    base = np.linspace(1.0, 1.0 / steps, steps).astype(f32)
    mu = sc.calculate_shift(tokens)

    def ended(s):
        return np.append(s, f32(0)).astype(f32)

    def at_mu(value):
        return ended(sc.stretch_to_terminal(
            sc.shift_exponential(base, value), 0.02))

    return {
        "max_shift x 1.1": sc.qwen21_sigmas(steps, tokens, max_shift=0.99),
        "base_shift x 1.1": sc.qwen21_sigmas(steps, tokens, base_shift=0.55),
        "mu x 1.1": at_mu(1.1 * mu),
        "mu x 0.9": at_mu(0.9 * mu),
        "39 steps": sc.qwen21_sigmas(steps - 1, tokens),
        "the last step left out": ended(right[:-2]),
        "the first step left out": right[1:],
        "the stretch to the terminal off": ended(
            sc.shift_exponential(base, mu)),
        "terminal 0.03": sc.qwen21_sigmas(steps, tokens, terminal=0.03),
        "the tokens of four times the area": sc.qwen21_sigmas(
            steps, 4 * tokens),
        "the tokens of the other size": sc.qwen21_sigmas(
            steps, 4096 if tokens == 1024 else 1024),
        "no shift": ended(base),
    }


# The wrong schedules whose first two steps the kept steps do not
# separate from a right run, by size. The schedule row holds them.
KEPT_PASSES = {
    512: {"max_shift x 1.1", "the last step left out", "terminal 0.03"},
    1024: {"the last step left out", "terminal 0.03"},
}


def test_the_seed_gives_the_noise_of_the_oracle(check6):
    from gmlx.image.samplers import noise_from_seed

    arrays, m = check6
    for size in _sizes(m):
        noise = arrays[f"noise_{size}"]
        assert noise.shape == (1, 64, size // 16, size // 16)
        assert np.array_equal(
            noise_from_seed(m["inputs"]["seed"], noise.shape), noise)


def test_the_schedule_of_a_request_is_the_schedule_of_the_pipeline(check6,
                                                                   pipe):
    """The sigmas and the timesteps of the pipeline's own call, at the
    token count of each size it ran."""
    arrays, m = check6
    if m["inputs"]["sigmas_passed"] is not None:
        pytest.skip("the oracle ran explicit sigmas")
    steps = m["inputs"]["steps"]
    assert pipe.defaults.steps == steps and pipe.defaults.cfg == 1.0
    for size in _sizes(m):
        got = pipe.sigmas(steps, (size // 16) ** 2)
        assert not _schedule_misses(got, arrays, size)


@pytest.mark.parametrize("size", [512, 1024])
def test_each_wrong_schedule_fails_its_rows(check6, pipe, size):
    """Each wrong schedule of the module docstring fails the schedule
    row, and all but those of KEPT_PASSES fail the kept steps in their
    first two steps on the real model."""
    from gmlx.image import api
    from gmlx.image.conditioning import Conditioning

    arrays, m = check6
    ins = m["inputs"]
    if size not in _sizes(m):
        pytest.skip(f"the oracle did not run {size}")
    first = [k for k in ins["kept_steps"] if k <= 2]
    assert first == [1, 2]
    gates = _kept_gates(m, size)
    tokens = (size // 16) ** 2
    right = pipe.sigmas(ins["steps"], tokens)
    assert not _schedule_misses(right, arrays, size)
    wrong = _wrong_schedules(ins["steps"], tokens)
    assert len(wrong) == 12

    def share(sigmas) -> float:
        cond = Conditioning(mx.array(arrays["features"]),
                            f"check6 {size} schedules")
        kept: dict = {}

        def on_step(i, x):
            kept[i + 1] = fx.rel_l2(
                np.array(x), arrays[f"latents_{size}_step{i + 1}"])

        api.sample(pipe, cond, height=size, width=size,
                   noise=mx.array(arrays[f"noise_{size}"]),
                   sigmas=sigmas[:3], on_step=on_step)
        return max(kept[k] / gates[k] for k in first)

    control = share(right)
    reads = {}
    for name, sigmas in wrong.items():
        assert _schedule_misses(sigmas, arrays, size), name
        reads[name] = share(sigmas)
    print(f"qwen check6 wrong schedules {size}: a right run reads "
          f"{control:.2f} of the kept step gates over steps 1 and 2; "
          + ", ".join(f"{n} {v:.2f}" for n, v in reads.items()))
    assert control <= 1.0
    passed = {n for n, v in reads.items() if v <= 1.0}
    assert passed == KEPT_PASSES[size], reads


@pytest.mark.parametrize("size", [512, 1024])
def test_check6_final_latents(check6, pipe, size):
    from gmlx.image import api
    from gmlx.image.conditioning import Conditioning

    arrays, m = check6
    if size not in _sizes(m):
        pytest.skip(f"the oracle did not run {size}")
    ins = m["inputs"]
    cond = Conditioning(mx.array(arrays["features"]), f"check6 {size}")
    kept: dict = {}

    def on_step(i, x):
        if i + 1 in ins["kept_steps"]:
            kept[i + 1] = fx.rel_l2(np.array(x),
                                    arrays[f"latents_{size}_step{i + 1}"])

    explicit = ins["sigmas_passed"] is not None
    x = api.sample(pipe, cond, height=size, width=size, seed=ins["seed"],
                   sigmas=arrays[f"sigmas_{size}"] if explicit else None,
                   on_step=on_step)
    got = fx.compare(np.array(x), arrays[f"latents_{size}"])
    tol = m["tolerances"]["sizes"][str(size)]
    gates = _kept_gates(m, size)
    print(f"qwen check6 latents {size}: rel L2 {got['rel_l2']:.4e} of "
          f"{tol['latents_rel_l2']:.4e}"
          f"{' (the gap of another size)' if tol.get('from') else ''}, "
          f"cosine {got['cosine']:.6f}; after steps "
          + ", ".join(f"{k} {v:.2e} ({v / gates[k]:.2f} of its gate)"
                      for k, v in sorted(kept.items())))
    assert x.shape == arrays[f"latents_{size}"].shape
    assert got["rel_l2"] <= tol["latents_rel_l2"], got
    assert sorted(kept) == ins["kept_steps"]
    over = {k: (v, gates[k]) for k, v in kept.items() if v > gates[k]}
    assert not over, over
