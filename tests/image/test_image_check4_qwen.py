"""Check 4 of the image acceptance gate for Qwen-Image 2.1: the sigma
schedule equals diffusers' bit for bit at five sizes and eight step
counts, and its constants are the official scheduler config's. The arrays
are small and committed, so this runs on the CPU tier.
scripts/image_oracle/check4_qwen_image21.py writes them. The sigmas the
oracle recorded for its calls on the real file are held to the schedule
too, from the manifests of check 3."""
from __future__ import annotations

import hashlib
import inspect

import numpy as np
import pytest
from image_fixtures import FIXTURES, load_json
from oracle_fixture import committed_fixture

from gmlx.image import schedule

NAME = "check4_qwen_image21"
CONFIG = "qwen_image21_scheduler_config.json"


@pytest.fixture(scope="module")
def check4():
    return committed_fixture(NAME)


def _bits(a) -> list:
    return np.asarray(a, np.float32).view(np.uint32).tolist()


def _tokens(width: int, height: int) -> int:
    return (height // 16) * (width // 16)


def test_sigmas_and_timesteps_are_bitwise_equal(check4):
    arrays, m = check4
    assert m["tolerances"]["sigmas"] == "bitwise"
    assert m["oracle"]["packages"]["diffusers"]["version"]
    sizes = [tuple(s) for s in m["inputs"]["sizes"]]
    assert {(512, 512), (1024, 1024), (2048, 2048)} <= set(sizes)
    assert sum(1 for w, h in sizes if w != h) >= 2
    cells = 0
    for w, h in sizes:
        assert _tokens(w, h) == m["inputs"]["tokens"][f"{w}x{h}"]
        for n in m["inputs"]["steps"]:
            want = arrays[f"sigmas_{n}_{w}x{h}"]
            got = schedule.qwen21_sigmas(n, _tokens(w, h))
            assert got.dtype == np.float32 and got.shape == (n + 1,)
            assert _bits(got) == _bits(want), (n, w, h)
            ts = got[:-1] * np.float32(1000)
            assert _bits(ts) == _bits(arrays[f"timesteps_{n}_{w}x{h}"])
            assert got[-1] == 0 and np.all(np.diff(got) < 0)
            cells += 1
    assert cells == 40


def test_mu_follows_the_token_count(check4):
    _, m = check4
    for size, mu in m["inputs"]["mu"].items():
        assert schedule.calculate_shift(m["inputs"]["tokens"][size]) == mu
    # The line passes through its two anchors and is not clamped past them.
    assert schedule.calculate_shift(256) == pytest.approx(0.5, abs=1e-15)
    assert schedule.calculate_shift(8192) == pytest.approx(0.9, abs=1e-15)
    assert schedule.calculate_shift(16384) > 0.9


@pytest.mark.parametrize("name", ["check3_qwen_image21_512",
                                  "check3_qwen_image21_1024"])
def test_the_sigmas_the_oracle_recorded_are_the_schedule(name):
    """The schedule the oracle set for its calls on the real DiT, at the
    token count of each size. It reads the committed manifest alone and
    opens no model file."""
    from oracle_fixture import ORACLE, committed

    if not (ORACLE / f"{name}.manifest.json").is_file():
        pytest.skip(f"{name}.manifest.json is not committed")
    ins = committed(name)["inputs"]
    h, w = ins["target_tokens"]
    assert h * w == _tokens(16 * w, 16 * h)
    want = np.asarray(ins["sigmas"], np.float64)
    assert np.array_equal(want, want.astype(np.float32)), "not float32"
    got = schedule.qwen21_sigmas(ins["steps"], h * w)
    assert _bits(got) == _bits(want), name
    for tag in "ab":
        assert got[ins[f"step_{tag}"]] == np.float32(ins[f"sigma_{tag}"])
    # The two sizes differ in every sigma but the ends.
    other = schedule.qwen21_sigmas(ins["steps"], 4 * h * w)
    assert np.all(other[1:-2] != got[1:-2])


def test_the_defaults_are_the_official_scheduler_config(check4):
    _, m = check4
    raw = (FIXTURES / CONFIG).read_bytes()
    row = load_json("manifest.json")[CONFIG]
    assert hashlib.sha256(raw).hexdigest() == row["sha256"]
    assert m["files"]["scheduler_config"]["sha256"] == row["sha256"]
    assert row["revision"] == m["inputs"]["config_revision"]
    cfg = load_json(CONFIG)
    assert cfg["use_dynamic_shifting"] and \
        cfg["time_shift_type"] == "exponential"
    shift = inspect.signature(schedule.calculate_shift).parameters
    sig = inspect.signature(schedule.qwen21_sigmas).parameters
    for params in (shift, sig):
        assert params["base_len"].default == cfg["base_image_seq_len"]
        assert params["max_len"].default == cfg["max_image_seq_len"]
        assert params["base_shift"].default == cfg["base_shift"]
        assert params["max_shift"].default == cfg["max_shift"]
    assert sig["terminal"].default == cfg["shift_terminal"]


def test_the_shift_and_the_stretch_do_what_they_say():
    base = np.linspace(1.0, 1 / 8, 8).astype(np.float32)
    s = schedule.shift_exponential(base, 0.7)
    assert s.dtype == np.float32 and s[0] == 1.0
    # A positive mu moves every sigma below 1 up.
    assert np.all(s[1:] > base[1:])
    assert _bits(schedule.shift_exponential(base, 0.0)) != _bits(s)
    t = schedule.stretch_to_terminal(s, 0.02)
    assert t.dtype == np.float32 and t[0] == 1.0
    assert abs(float(t[-1]) - 0.02) < 1e-7
    # Another terminal, or another constant, is another schedule.
    a = schedule.qwen21_sigmas(8, 4096)
    assert _bits(schedule.qwen21_sigmas(8, 4096, terminal=0.03)) != _bits(a)
    assert _bits(schedule.qwen21_sigmas(8, 4096, max_len=4096)) != _bits(a)
    assert _bits(schedule.qwen21_sigmas(8, 1024)) != _bits(a)


def test_one_step_is_refused_where_the_reference_gives_nan(check4):
    _, m = check4
    assert m["results"]["one_step_sigmas_are_nan"] is True
    with pytest.raises(ValueError, match="at least 2 steps, not 1"):
        schedule.qwen21_sigmas(1, 4096)
