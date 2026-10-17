"""Check 4 of the image acceptance gate for FLUX.2 klein: the family's
schedule equals that of BFL's reference bit for bit at 24 step counts and
12 image token counts, for both variants, with each mu equal to the
reference's, and so do the values the samplers read from it at the
default step counts for a 512 and a 1024 image: each step's time in bf16,
1000 times that as the embedder reads it and the step dt. The arrays are
small and committed, so this runs on the CPU tier.
scripts/image_oracle/check4_flux2_klein.py writes them."""
from __future__ import annotations

import numpy as np
import pytest
from oracle_fixture import committed_fixture

from gmlx.image.registry import get_family

NAME = "check4_flux2_klein"
REFERENCE_COMMIT = "50fe5162777813d869182b139e83b10743caef15"


@pytest.fixture(scope="module")
def check4():
    return committed_fixture(NAME)


def _bits(a) -> list:
    return np.asarray(a, np.float32).view(np.uint32).tolist()


def test_the_schedule_is_bitwise_equal_for_both_variants(check4):
    arrays, m = check4
    fam = get_family("flux2_klein")
    i = m["inputs"]
    assert m["tolerances"]["sigmas"] == "bitwise"
    assert i["reference_commit"] == REFERENCE_COMMIT
    assert len(i["steps"]) == 24 and len(i["tokens"]) == 12
    assert {4300, 4301} <= set(i["tokens"])
    for n in i["steps"]:
        rows = arrays[f"sigmas_{n}"]
        assert rows.shape == (len(i["tokens"]), n + 1)
        for want, tok in zip(rows, i["tokens"]):
            for variant in ("distilled", "base"):
                got = fam.schedule(n, tokens=tok, variant=variant)
                assert got.dtype == np.float32 and got.shape == (n + 1,)
                assert _bits(got) == _bits(want), (n, tok, variant)


def test_mu_is_the_reference_mu(check4):
    from gmlx.image.schedule import flux2_mu

    _, m = check4
    i = m["inputs"]
    for n in i["steps"]:
        got = [flux2_mu(t, n) for t in i["tokens"]]
        assert got == i["mu"][str(n)], n


def test_the_ends_of_every_schedule(check4):
    """The last sigma is zero and the first is 1 or the float32 below it,
    as the reference's reciprocal leaves it; both occur. No sigma rises,
    and at 65536 tokens and many steps the shift leaves the first two at
    1, so the first step does not move, in the reference and in gmlx."""
    from gmlx.image.schedule import check_sigmas

    arrays, m = check4
    below = np.nextafter(np.float32(1), np.float32(0))
    firsts, flat = set(), set()
    for n in m["inputs"]["steps"]:
        rows = arrays[f"sigmas_{n}"]
        assert (rows[:, -1] == 0).all()
        assert np.isin(rows[:, 0], [np.float32(1), below]).all()
        assert (np.diff(rows, axis=1) <= 0).all()
        firsts.update(rows[:, 0].tolist())
        for want, tok in zip(rows, m["inputs"]["tokens"]):
            assert _bits(check_sigmas(want)) == _bits(want)
            if (np.diff(want) == 0).any():
                flat.add(tok)
    assert firsts == {1.0, float(below)}
    assert flat == {65536}


def test_one_quotient_misses_the_reference(check4):
    """e / (e + (1 / t - 1)) with one rounding per quotient is not the
    reference's schedule: the reciprocal and its product are."""
    import math

    from gmlx.image.schedule import flux2_mu, linspace_f32

    arrays, m = check4
    i = m["inputs"]
    f = np.float32
    miss = 0
    for n in i["steps"]:
        for want, tok in zip(arrays[f"sigmas_{n}"], i["tokens"]):
            t = linspace_f32(1.0, 0.0, n + 1)
            e = f(math.exp(flux2_mu(tok, n)))
            with np.errstate(divide="ignore"):
                one = (e / (e + (f(1) / t - f(1)))).astype(f)
            miss += _bits(one) != _bits(want)
    assert miss > len(i["steps"]) * len(i["tokens"]) // 2


def test_the_default_step_counts_are_the_reference_ones(check4):
    _, m = check4
    fam = get_family("flux2_klein")
    assert (fam.sampling["distilled"].steps, fam.sampling["distilled"].cfg) \
        == (4, 1.0)
    assert (fam.sampling["base"].steps, fam.sampling["base"].cfg) \
        == (50, 4.0)
    assert m["inputs"]["chain_steps"] == [4, 50]
    assert m["inputs"]["chain_tokens"] == [1024, 4096]


@pytest.mark.parametrize("steps", [4, 50])
@pytest.mark.parametrize("tokens", [1024, 4096])
def test_the_values_the_samplers_read_are_the_reference_ones(check4, steps,
                                                             tokens):
    """At every step the model's time embedding of the schedule's time is
    the embedding of the reference's 1000 times the time, in bf16, and
    the Euler step moves by the reference's dt. The time in bf16 is the
    reference's too."""
    import mlx.core as mx

    from gmlx.image import ops
    from gmlx.image.families.flux2_klein import model as fm
    from gmlx.image.samplers import Euler

    arrays, _ = check4
    key = f"{steps}_{tokens}"
    sig = get_family("flux2_klein").schedule(steps, tokens=tokens)
    assert _bits(fm._bf16(sig[:-1])) == _bits(arrays[f"t_bf16_{key}"])
    t1000 = arrays[f"t1000_{key}"]
    euler = Euler(sig, product=mx.bfloat16)
    zero, one = mx.zeros((1,), mx.float32), mx.ones((1,), mx.float32)
    dt = []
    with mx.stream(mx.cpu):
        for i in range(steps):
            got = fm.time_embedding(sig[i], 256)
            want = ops.timestep_embedding(t1000[i:i + 1], 256)
            want = want.astype(mx.bfloat16).astype(mx.float32)
            assert mx.array_equal(got, want).item(), i
            dt.append(np.array(Euler(sig).step(zero, one, i))[0])
            # The bf16 product of a unit velocity is dt in bf16.
            p = np.array(euler.step(zero, one, i))[0]
            assert p == float(fm._bf16(np.float32(dt[-1]))[()]), i
    assert _bits(dt) == _bits(arrays[f"dt_{key}"])


@pytest.mark.parametrize("variant", ["distilled", "base"])
def test_the_pipeline_takes_the_family_schedule(variant):
    from gmlx.image.api import ImagePipeline
    from gmlx.image.schedule import flux2_sigmas

    pipe = ImagePipeline(dit=None, family=get_family("flux2_klein"),
                         variant=variant)
    for tokens in (1024, 4096, 16384):
        got = pipe.sigmas(4, tokens=tokens)
        assert _bits(got) == _bits(flux2_sigmas(4, tokens))
    assert _bits(pipe.sigmas(4, 1024)) != _bits(pipe.sigmas(4, 4096))
    with pytest.raises(ValueError, match="pass tokens"):
        pipe.sigmas(4)


def test_flux2_sigmas_refuses_zero_steps():
    from gmlx.image.schedule import flux2_sigmas

    with pytest.raises(ValueError, match="at least 1"):
        flux2_sigmas(0, 1024)
