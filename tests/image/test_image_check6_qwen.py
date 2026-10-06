"""Check 6 of the image acceptance gate for Qwen-Image 2.1, on latents: a
whole sampling run on injected noise. gmlx takes the oracle's
conditioning, samples from the oracle's noise with its default steps and
guidance, and the final latents must stay within twice the oracle's own
bf16 against fp32 gap, as the manifest records it. The seed's noise must
equal the oracle's noise, and the schedule of the request must equal the
schedule of the pipeline's own call bit for bit.

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

pytestmark = pytest.mark.integration

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
        assert _bits(got) == _bits(arrays[f"sigmas_{size}"]), size
        assert _bits(got[:-1] * np.float32(1000)) == _bits(
            arrays[f"timesteps_{size}"]), size


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
    print(f"qwen check6 latents {size}: rel L2 {got['rel_l2']:.4e} of "
          f"{tol['latents_rel_l2']:.4e}"
          f"{' (the gap of another size)' if tol.get('from') else ''}, "
          f"cosine {got['cosine']:.6f}; after steps "
          + ", ".join(f"{k} {v:.2e}" for k, v in sorted(kept.items())))
    assert x.shape == arrays[f"latents_{size}"].shape
    assert got["rel_l2"] <= tol["latents_rel_l2"], got
