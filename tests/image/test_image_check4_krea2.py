"""Check 4 of the image acceptance gate for Krea 2: the family's schedule
equals diffusers' bit for bit for both variants at five sizes and four
step counts each, with the scheduler config Krea2Pipeline documents, and
the timestep the embedder reads is the oracle's. The arrays are small and
committed, so this runs on the CPU tier.
scripts/image_oracle/check4_krea2.py writes them."""
from __future__ import annotations

import numpy as np
import pytest
from oracle_fixture import committed_fixture

from gmlx.image.registry import get_family

NAME = "check4_krea2"


@pytest.fixture(scope="module")
def check4():
    return committed_fixture(NAME)


def _bits(a) -> list:
    return np.asarray(a, np.float32).view(np.uint32).tolist()


def test_sigmas_and_timesteps_are_bitwise_equal(check4):
    arrays, m = check4
    fam = get_family("krea2")
    assert m["tolerances"]["sigmas"] == "bitwise"
    sizes = [tuple(s) for s in m["inputs"]["sizes"]]
    cells = 0
    for variant, steps in m["inputs"]["steps"].items():
        for w, h in sizes:
            tokens = (h // 16) * (w // 16)
            assert tokens == m["inputs"]["tokens"][f"{w}x{h}"]
            for n in steps:
                got = fam.schedule(n, tokens=tokens, variant=variant)
                want = arrays[f"sigmas_{variant}_{n}_{w}x{h}"]
                assert got.dtype == np.float32 and got.shape == (n + 1,)
                assert _bits(got) == _bits(want), (variant, n, w, h)
                ts = got[:-1] * np.float32(1000)
                assert _bits(ts) == _bits(
                    arrays[f"timesteps_{variant}_{n}_{w}x{h}"])
                cells += 1
    assert cells == 40


def test_the_scheduler_config_is_the_family_constants(check4):
    _, m = check4
    c = get_family("krea2").constants.values
    s = m["inputs"]["scheduler"]
    assert s["use_dynamic_shifting"] and s["time_shift_type"] == \
        "exponential" and s["shift_terminal"] is None
    for key in ("base_shift", "max_shift", "base_image_seq_len",
                "max_image_seq_len"):
        assert s[key] == c[key], key
    assert s["num_train_timesteps"] == c["num_train_timesteps"]
    assert all(v == c["turbo_mu"] for k, v in m["inputs"]["mu"].items()
               if k.startswith("turbo_"))


def test_the_timestep_chain_is_the_oracle_one(check4):
    from gmlx.image.families.krea2.model import Krea2DiT

    arrays, m = check4
    chain = m["inputs"]["chain"]
    w, h = chain["size"]
    sig = get_family("krea2").schedule(
        chain["steps"], tokens=(h // 16) * (w // 16),
        variant=chain["variant"])[:-1]
    cfg = type("C", (), {"train_timesteps": 1000})()
    got = Krea2DiT.timesteps(type("D", (), {"cfg": cfg})(), sig, len(sig))
    assert _bits(got) == _bits(arrays["t_embedded"])
