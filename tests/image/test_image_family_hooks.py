"""The hooks a family's registry row gives the shared image code, on toy
families: the step in parts under the pipeline's two memos, the schedule
and the size multiple. CPU."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from gmlx.image import api
from gmlx.image.registry import ConstantsRow, StepParts, get_family


class _State:
    def __init__(self, *arrays):
        self._arrays = list(arrays)

    def arrays(self) -> list:
        return self._arrays


def _parts_pipe(state_bytes: int = 1 << 20, rows: int = 4):
    """A pipeline whose family gives step parts, and their call counts.
    v = latents * 2 sigma + mean of the features."""
    calls = {"modulation": 0, "conditioning": 0, "forward": 0}

    def modulation(dit, sigma):
        calls["modulation"] += 1
        assert isinstance(sigma, np.float32)
        return _State(mx.array(sigma) * 2)

    def conditioning(dit, cond):
        calls["conditioning"] += 1
        return _State(mx.zeros((rows,), mx.float32) + cond.features.mean())

    def forward(dit, x, sigma, cond, state, mod):
        calls["forward"] += 1
        return x * mod.arrays()[0] + state.arrays()[0][0]

    fam = replace(get_family("z_image"), step=StepParts(
        modulation, conditioning, forward, state_bytes))
    dit = SimpleNamespace(cfg=SimpleNamespace(in_channels=4, patch=2),
                          act_dtype=mx.bfloat16)
    return api.ImagePipeline(dit=dit, family=fam, variant="turbo"), calls


def _cond(value: float, key: str = "") -> api.Conditioning:
    return api.Conditioning(mx.full((3, 5), value, mx.float32), key)


def test_step_parts_run_a_row_under_the_two_memos():
    pipe, calls = _parts_pipe()
    x = mx.arange(32, dtype=mx.float32).reshape(1, 4, 2, 4)
    v = api.denoise_step(pipe, x, 0.5, _cond(0.25, "a"))
    assert np.array_equal(np.array(v), np.array(x) * 1.0 + 0.25)
    api.denoise_step(pipe, x, 0.5, _cond(0.25, "a"))
    assert calls == {"modulation": 1, "conditioning": 1, "forward": 2}
    # Another sigma is another modulation, another key another state.
    api.denoise_step(pipe, x, 0.25, _cond(0.25, "a"))
    api.denoise_step(pipe, x, 0.25, _cond(0.5, "b"))
    assert calls == {"modulation": 2, "conditioning": 2, "forward": 4}
    # A conditioning without a key is never cached.
    api.denoise_step(pipe, x, 0.25, _cond(0.5))
    api.denoise_step(pipe, x, 0.25, _cond(0.5))
    assert calls["conditioning"] == 4 and len(pipe.captions) == 2
    # Both memo keys hold the activation dtype.
    pipe.dit.act_dtype = mx.float32
    api.denoise_step(pipe, x, 0.25, _cond(0.5, "b"))
    assert calls["modulation"] == 3 and calls["conditioning"] == 5
    # Cleared caches change nothing but the work.
    pipe.captions.clear()
    pipe.modulations.clear()
    again = api.denoise_step(pipe, x, 0.5, _cond(0.25, "a"))
    assert np.array_equal(np.array(again), np.array(v))


def test_step_parts_rows_take_their_own_sigma_and_conditioning():
    pipe, calls = _parts_pipe()
    x = mx.arange(64, dtype=mx.float32).reshape(2, 4, 2, 4)
    conds = [_cond(0.25, "a"), _cond(0.5, "b")]
    v = api.denoise_step(pipe, x, [0.5, 0.25], conds)
    for i, s in enumerate((0.5, 0.25)):
        one = api.denoise_step(pipe, x[i:i + 1], s, conds[i])
        assert np.array_equal(np.array(v[i:i + 1]), np.array(one))
    assert calls["modulation"] == 2 and calls["conditioning"] == 2


def test_the_state_cache_keeps_two_states_of_the_newest_size():
    # A state of 64 rows is 256 bytes, over half the bound of 300.
    pipe, calls = _parts_pipe(state_bytes=300, rows=64)
    x = mx.zeros((1, 4, 2, 4), mx.float32)
    for _ in range(2):
        api.denoise_step(pipe, x, 0.5, _cond(1.0, "pos"))
        api.denoise_step(pipe, x, 0.5, _cond(0.0, "neg"))
    assert calls["conditioning"] == 2 and pipe.captions.bytes == 512
    # A third state of that size evicts the least recent one.
    api.denoise_step(pipe, x, 0.5, _cond(2.0, "other"))
    assert len(pipe.captions) == 2 and pipe.captions.get(
        ("pos", pipe.default_table, str(mx.bfloat16))) is None
    # A small state brings the bound back, and the cache sheds down to it.
    small, _ = _parts_pipe(state_bytes=300, rows=4)
    step = small.family.step
    pipe.family = replace(pipe.family, step=step)
    api.denoise_step(pipe, x, 0.5, _cond(3.0, "small"))
    assert pipe.captions.max_bytes == 300
    assert pipe.captions.bytes <= 300 and len(pipe.captions) == 2


def test_a_family_schedule_reads_the_token_count_of_the_request():
    seen = []

    def schedule(steps, *, tokens, variant):
        seen.append((steps, tokens, variant))
        return np.linspace(1.0, 0.0, steps + 1, dtype=np.float32)

    pipe, calls = _parts_pipe()
    pipe.family = replace(pipe.family, schedule=schedule)
    assert np.array_equal(pipe.sigmas(4, 96), schedule(4, tokens=96,
                                                       variant="turbo"))
    # Z-Image tokens are 16 pixels a side: 64x48 is 4 x 3 of them.
    x = api.sample(pipe, _cond(0.25, "a"), height=48, width=64, seed=3,
                   steps=2)
    assert seen[-1] == (2, 12, "turbo") and calls["forward"] == 2
    assert x.shape == (1, 4, 6, 8)
    # An outside list still replaces the family's schedule.
    n = len(seen)
    api.sample(pipe, _cond(0.25, "a"), height=48, width=64, seed=3,
               sigmas=[1.0, 0.5, 0.0])
    assert len(seen) == n and calls["forward"] == 4


def test_the_size_multiple_comes_from_the_family_row_when_it_has_one():
    pipe, _ = _parts_pipe()
    assert pipe.size_multiple == 16
    row = pipe.family.constants
    pipe.family = replace(pipe.family, constants=ConstantsRow(
        {**row.values, "size_multiple": 32}, row.repo, row.revision))
    assert pipe.size_multiple == 32
    with pytest.raises(ValueError, match="multiples of 32, not 48x64"):
        api.sample(pipe, _cond(0.25), height=48, width=64, seed=0, steps=1)
    x = api.sample(pipe, _cond(0.25), height=64, width=32, seed=0, steps=1)
    assert x.shape == (1, 4, 8, 4)
