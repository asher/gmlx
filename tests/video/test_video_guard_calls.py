"""The video concatenates that can pass 2^31 elements call
gmlx.mlx_guards.

Past 2^31 output elements the stock concatenate misplaces rows on a GPU
stream (docs/upstream-defects.md), so pair_halves, the chunked dense
attention of H3 and the TAEH3 memory block go through the guard. Each
test runs a site at a few elements on the CPU and checks the guard calls
it makes, so a site that calls the stock op fails on every run.
"""
from __future__ import annotations

import mlx.core as mx
import pytest

from gmlx import mlx_guards
from gmlx.video import rope
from gmlx.video.families.h3 import model as hm
from gmlx.video.vae.tiny import MemBlock


@pytest.fixture(autouse=True)
def cpu():
    with mx.stream(mx.cpu):
        yield


@pytest.fixture
def calls(monkeypatch) -> list:
    """("concatenate", its axis) for each call of the guard from now on,
    before it goes on to the guard. The sites look the guard up on the
    module when they run, so the patch reaches them."""
    seen = []
    guard = mlx_guards.concatenate

    def spy(arrays, axis=0):
        arrays = list(arrays)
        seen.append(("concatenate", axis % arrays[0].ndim))
        return guard(arrays, axis)

    monkeypatch.setattr(mlx_guards, "concatenate", spy)
    return seen


def test_pair_halves_concatenates_through_the_guard(calls):
    """The paired rotated dims join the rest of the head on the last
    axis."""
    mx.eval(rope.pair_halves(mx.ones((1, 3, 2, 8)), 6))
    assert calls == [("concatenate", 3)]


def test_chunked_dense_attention_concatenates_through_the_guard(calls):
    """Three chunks of query rows join on the row axis."""
    q = mx.ones((1, 2, 5, 4))
    mx.eval(hm.attend(q, q, q, 2))
    assert calls == [("concatenate", 2)]


def test_a_memory_block_concatenates_through_the_guard(calls):
    """A memory block joins each frame and the frame before on the
    channel axis."""
    mx.eval(MemBlock(4, 4)(mx.ones((2, 3, 3, 4)), 2))
    assert calls == [("concatenate", 3)]
