"""Check 4 of the image acceptance gate for FLUX.3 klein: the family's
schedule equals that of BFL's reference bit for bit at every step count
to 64 and four more, for both variants, and so do the values the samplers
read from it at the default step counts: each step's time in bf16, 1000
times that as the embedder reads it, the target time of the flow map and
the step dt. The arrays are small and committed, so this runs on the CPU
tier. scripts/image_oracle/check4_flux3_klein.py writes them."""
from __future__ import annotations

import numpy as np
import pytest
from oracle_fixture import committed_fixture

from gmlx.image.registry import get_family

NAME = "check4_flux3_klein"


@pytest.fixture(scope="module")
def check4():
    return committed_fixture(NAME)


def _bits(a) -> list:
    return np.asarray(a, np.float32).view(np.uint32).tolist()


def test_the_schedule_is_bitwise_equal_for_both_variants(check4):
    arrays, m = check4
    fam = get_family("flux3_klein")
    assert m["tolerances"]["sigmas"] == "bitwise"
    assert m["inputs"]["alpha"] == fam.constants.values["alpha"]
    assert len(m["inputs"]["steps"]) == 68
    for n in m["inputs"]["steps"]:
        want = _bits(arrays[f"sigmas_{n}"])
        for variant in ("distilled", "base"):
            for tokens in (1024, 4096, 16384):
                got = fam.schedule(n, tokens=tokens, variant=variant)
                assert got.dtype == np.float32 and got.shape == (n + 1,)
                assert _bits(got) == want, (n, variant, tokens)


def test_the_default_step_counts_are_the_reference_ones(check4):
    _, m = check4
    models = m["inputs"]["models"]
    assert (models["distilled"]["steps"], models["distilled"]["guidance"]) \
        == (16, None)
    assert (models["base"]["steps"], models["base"]["guidance"]) == (50, 3.5)
    assert m["inputs"]["chain_steps"] == [16, 50]


@pytest.mark.parametrize("steps", [16, 50])
def test_the_values_the_samplers_read_are_the_reference_ones(check4, steps):
    """At every step the model's time embedding of the schedule's time is
    the embedding of the reference's 1000 times the time, in bf16, and
    the Euler step moves by the reference's dt. The time and the target
    in bf16 are the reference's too."""
    import mlx.core as mx

    from gmlx.image import ops
    from gmlx.image.families.flux3_klein import model as fm
    from gmlx.image.samplers import Euler

    arrays, _ = check4
    sig = get_family("flux3_klein").schedule(steps, tokens=None)
    assert _bits(fm._bf16(sig[:-1])) == _bits(arrays[f"t_bf16_{steps}"])
    assert _bits(fm._bf16(sig[1:])) == _bits(arrays[f"target_bf16_{steps}"])
    t1000 = arrays[f"t1000_{steps}"]
    euler = Euler(sig)
    zero, one = mx.zeros((1,), mx.float32), mx.ones((1,), mx.float32)
    dt = []
    for i in range(steps):
        got = fm.time_embedding(sig[i], 256)
        want = ops.timestep_embedding(t1000[i:i + 1], 256)
        want = want.astype(mx.bfloat16).astype(mx.float32)
        assert mx.array_equal(got, want).item(), i
        dt.append(np.array(euler.step(zero, one, i))[0])
    assert _bits(dt) == _bits(arrays[f"dt_{steps}"])


@pytest.mark.parametrize("variant", ["distilled", "base"])
def test_the_pipeline_takes_the_family_schedule(variant):
    from gmlx.image.api import ImagePipeline
    from gmlx.image.schedule import flux3_sigmas

    pipe = ImagePipeline(dit=None, family=get_family("flux3_klein"),
                         variant=variant)
    got = pipe.sigmas(16, tokens=4096)
    assert _bits(got) == _bits(flux3_sigmas(16, 6.93))
    assert got[-1] == 0 and got[0] <= 1


def test_flux3_sigmas_refuses_zero_steps():
    from gmlx.image.schedule import flux3_sigmas

    with pytest.raises(ValueError, match="at least 1"):
        flux3_sigmas(0, 6.93)
