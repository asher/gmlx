"""The FLUX.2 klein single block and the FLUX.1 VAE downsample call
gmlx.mlx_guards.

Past 2^31 output elements the stock pad and concatenate misplace rows on
a GPU stream (docs/upstream-defects.md), so these sites go through the
guards. Each test runs a site at a few elements on the CPU and checks the
guard calls it makes, so a site that calls the stock op fails on every
run. tests/image/test_image_past_int32.py runs the sites past the real
limit.
"""
from __future__ import annotations

import mlx.core as mx
import pytest

from gmlx.image.families.flux2_klein.model import Flux2Config, SingleBlock
from gmlx.image.vae.flux1_ae import Downsample
from helpers import spy_guards
from oracle_fixture import committed


@pytest.fixture(autouse=True)
def cpu():
    with mx.stream(mx.cpu):
        yield


def test_a_single_block_concatenates_through_the_guard(monkeypatch):
    """The tiny klein's single block joins the attention and feed-forward
    rows on the last axis."""
    cfg = Flux2Config.from_dict(committed("selftest_flux2_klein_tiny_weights")["inputs"]["config"])
    block = SingleBlock(cfg)
    x = mx.ones((1, 3, cfg.dim))
    calls = spy_guards(monkeypatch)
    mx.eval(block.out(x, x, x, mx.ones((1, 3, cfg.mlp_dim))))
    assert calls == [("concatenate", 2)]


def test_a_vae_downsample_pads_through_the_guard(monkeypatch):
    """The downsample pads the bottom and the right of an activation."""
    calls = spy_guards(monkeypatch)
    mx.eval(Downsample(4)(mx.ones((1, 5, 7, 4))))
    assert calls == [("pad", (1, 2))]
