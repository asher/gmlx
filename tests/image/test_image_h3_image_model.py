"""The MiniMax-H3-Image DiT's module tree against the release file's
tensors, and the step plan's rows. CPU-only; nothing is evaluated at the
release size."""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from image_fixtures import fixture_view
from mlx.utils import tree_flatten

from gmlx.image.families.h3_image import family as hi
from gmlx.image.families.h3_image import model as him

HEADER = "h3_image_pruned_ud_q4_k_xl.header.json"


@pytest.fixture(scope="module")
def config():
    return hi.synth_config(fixture_view(HEADER))


def test_the_module_tree_is_the_files(config):
    """Every parameter of the built model is a tensor of the file at its
    shape, and every tensor of the file a parameter."""
    view = fixture_view(HEADER)
    model = hi.build(config)
    got = {k: tuple(a.shape) for k, a in tree_flatten(model.parameters())}
    want = {n: tuple(reversed(t.shape)) for n, t in view.tensors.items()}
    assert got == want
    assert model.cfg.step_heads == hi.STEPS
    assert model.cfg.pdd_sigmas == hi.PDD_SIGMAS
    assert (model.cfg.patch, model.cfg.in_channels, model.cfg.text_dim) == \
        (2, 24, 5120)


def test_patchify_round_trips():
    lat = mx.array(np.random.default_rng(0).standard_normal(
        (1, 24, 6, 8)).astype(np.float32))
    rows = him.patchify(lat, 2)
    assert rows.shape == (12, 96)
    assert np.array_equal(np.array(rows[1]),
                          np.array(lat[0, :, 0:2, 2:4]).reshape(-1))
    assert np.array_equal(np.array(him.unpatchify(rows, (24, 6, 8), 2)),
                          np.array(lat))


def test_the_plan_tags_text_one_and_the_image_zero():
    """Text rows at (i, 0, 0) under tag 1, a vision span of the prompt and
    the target under tag 0, the target on the clock after the text."""
    lay = him.layout(10, 4, 6, ((2, 5),))
    kinds = [(s.kind, s.start, s.stop) for s in lay.segments]
    assert kinds == [("text", 0, 2), ("vision", 2, 5), ("text", 5, 10),
                     ("video", 10, 16)]
    assert np.array_equal(lay.positions[:10, 0], np.arange(10))
    assert not lay.positions[:10, 1:].any()
    assert (lay.positions[10:, 0] == 10).all()
    assert him.layout(10, 4, 6, ((2, 5),)) is lay
    assert not lay.positions.flags.writeable
    tags = [r for *_, r in _plan(10, 4, 6, ((2, 5),)).segs]
    assert tags == [1, 0, 1, 0]


def _plan(text_len, h, w, spans=()):
    cfg = {**hi.synth_config(fixture_view(HEADER)), "num_layers": 0,
           "token_refiner_num_layers": 0}
    return him.H3ImageDiT(cfg).plan(text_len, h, w, spans)


@pytest.mark.parametrize("sigma, head", [(1.0, 0), (0.9675236940383911, 1),
                                          (0.75, 3), (0.23200887441635132, 5),
                                          (0.0005, 5)])
def test_a_step_reads_its_grid_points_head(config, sigma, head):
    cfg = {**config, "num_layers": 0, "token_refiner_num_layers": 0}
    mod = him.H3ImageDiT(cfg).modulation(sigma, 1)
    t = 1.0 - hi.model_sigma(sigma)
    assert mod.head == head
    assert mod.times == (t, max(t, hi.REF_TIME))
    assert mod.t_emb.shape == (2, 8)


def test_what_the_model_does_not_run_is_refused(config):
    cfg = {**config, "num_layers": 0, "token_refiner_num_layers": 0}
    model = him.H3ImageDiT(cfg)
    with pytest.raises(ValueError, match="batch of 1"):
        model.modulation(1.0, 2)
    lat = mx.zeros((1, 24, 4, 4))
    with pytest.raises(ValueError, match="no reference images"):
        model(lat, 1.0, mx.zeros((1, 2, 5120)), refs=[lat])
    with pytest.raises(ValueError, match="one frame square"):
        him.H3ImageConfig.from_dict({**config, "patch_size": [2, 2, 2]})
