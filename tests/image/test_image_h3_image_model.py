"""The MiniMax-H3-Image DiT's module tree against the release file's
tensors, and the step plan's rows. CPU-only; nothing is evaluated at the
release size."""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from image_fixtures import fixture_view
from mlx.utils import tree_flatten

from gmlx.image import ops
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


G = hi.PDD_SIGMAS


@pytest.mark.parametrize("sigma, target, heads", [
    (G[0], G[1], ()), (G[1], G[2], ((1, 1.0),)), (G[3], G[4], ((3, 1.0),)),
    (G[5], 0.0, ((5, 1.0),)), (0.0005, 0.0, ((5, 1.0),)),
    (G[0], G[2], ((1, 0.7111971974372864),)),
    (G[4], 0.0, ((4, 0.5645265579223633), (5, 0.43547341227531433)))])
def test_a_step_reads_the_heads_it_spans(config, sigma, target, heads):
    """Head 0 alone, or beside head k, on one grid interval; past head 0
    each head of a wider step by its share of the span."""
    cfg = {**config, "num_layers": 0, "token_refiner_num_layers": 0}
    mod = him.H3ImageDiT(cfg).modulation(sigma, 1, target)
    t = 1.0 - hi.model_sigma(sigma)
    assert mod.heads == heads
    assert mod.times == (t, max(t, hi.REF_TIME))
    assert mod.t_emb.shape == (2, 8)


def test_the_head_on_the_grid_is_head_0_plus_head_k_bit_for_bit(config):
    """On the six steps of the default schedule the head equals head 0, or
    head 0 plus head k, as the head read it before steps spanned grid
    intervals."""
    cfg = {**config, "num_layers": 0, "token_refiner_num_layers": 0,
           "hidden_size": 64, "num_attention_heads": 1,
           "attention_head_dim": 64, "ffn_hidden_size": 128}
    model = him.H3ImageDiT(cfg)
    fl = model.final_layer
    pd = model.cfg.block.patch_dim
    plan = model.plan(3, 4, 6)
    x = mx.random.normal((1, plan.layout.rows, model.cfg.dim),
                         key=mx.random.key(0))
    for k in range(hi.STEPS):
        mod = model.modulation(G[k], 1, G[k + 1])
        y = fl.adaln_proj.linear(mod.t_emb[:1])
        shift, scale = y.reshape(2, model.cfg.dim)
        s = plan.target
        n = fl.norm(x[0, s.start:s.stop])
        bank = fl.video_out((ops.modulate(n, scale) + shift).astype(
            mx.float32))
        want = bank[:, :pd]
        if k:
            want = want + bank[:, k * pd:(k + 1) * pd]
        got = model.head(x, mod, plan)
        assert np.array(got).tobytes() == np.array(want).tobytes(), k


def test_what_the_model_does_not_run_is_refused(config):
    cfg = {**config, "num_layers": 0, "token_refiner_num_layers": 0}
    model = him.H3ImageDiT(cfg)
    with pytest.raises(ValueError, match="batch of 1"):
        model.modulation(1.0, 2, G[1])
    with pytest.raises(ValueError, match="pass target"):
        model.modulation(1.0, 1)
    lat = mx.zeros((1, 24, 4, 4))
    with pytest.raises(ValueError, match="no reference images"):
        model(lat, 1.0, mx.zeros((1, 2, 5120)), refs=[lat])
    with pytest.raises(ValueError, match="one frame square"):
        him.H3ImageConfig.from_dict({**config, "patch_size": [2, 2, 2]})


def test_sample_refuses_a_size_outside_the_family_limits():
    """The API holds a sample to MiniMax-H3-Image's limits, by name,
    before any step runs."""
    from types import SimpleNamespace

    from gmlx.image import api

    pipe = SimpleNamespace(
        family=hi.FAMILY, variant="pruned", default_table="", patch_tables={
            "": None}, defaults=SimpleNamespace(cfg=1.0, steps=6),
        dit=SimpleNamespace(cfg=SimpleNamespace(), reads_target=True),
        size_multiple=32)
    for (w, h), why in (((512, 512), "the smallest size of its shape is "
                                     "832x832"),
                        ((7744, 1024), "a side over the 7712"),
                        ((2048, 640), "1:3 to 3:1")):
        with pytest.raises(api.RequestRefusal, match=why):
            api.sample(pipe, None, height=h, width=w)
