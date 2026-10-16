"""The FLUX.2 klein DiT on the CPU: the module tree against the tensor
names of the 4B and 9B files, the RoPE tables against the reference's,
and a tiny model against BFL's own class from the committed fixtures of
scripts/image_oracle/check3_flux2_klein.py --self-test, with two
reference images and with text alone, in fp32 at a fixed bound and in
bf16 at twice the oracle's own gap. The step parts that a pipeline
memoizes give the step they stand in for. Each wrong model of FAULTS
misses the fp32 bound.

The fp32 rows read the reference's time embedding, so they measure the
model. gmlx's own embedding is held apart: the reference's fp32 exp is an
ulp off the correctly rounded one at some frequencies, which moves one
bf16 entry of the embedding at some times, 0.73 among them."""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

import flux2_gates as fg
from image_fixtures import fixture_view
from oracle_fixture import committed_fixture

from gmlx.image import fingerprint as fp
from gmlx.image import ops
from gmlx.image.dtypes import ImageDtypeError
from gmlx.image.families.flux2_klein import model as fm
from gmlx.image.loader import check_names
from gmlx.image.registry import get_family

TINY = "selftest_flux2_klein_tiny"
NOREFS = "selftest_flux2_klein_tiny_norefs"
# The tiny model in fp32 on the CPU against the oracle in fp32. The
# weakest wrong model of FAULTS moves its most moved row far past it.
TOL = 4e-6
# The fp32 step on gmlx's own time embedding, one entry one bf16 ulp off
# the reference's.
OWN_TOL = 2e-4


@pytest.fixture(scope="module")
def tiny():
    weights, wm = committed_fixture(TINY + "_weights")
    return weights, wm["inputs"]["config"]


@pytest.mark.parametrize("name", ["flux2_klein_4b_q4_k_m.header.json",
                                  "flux2_klein_9b_q4_k_m.header.json"])
def test_the_module_tree_names_every_tensor_of_the_file(name):
    """Strict name equality and the torch shape of every tensor, on the
    real config. Nothing is evaluated."""
    got = fp.identify(fixture_view(name))
    model = get_family("flux2_klein").build(got.config)
    check_names(model, got.view.tensors, "FLUX.2 klein")
    params = dict(tree_flatten(model.parameters()))
    for n, t in got.view.tensors.items():
        assert params[n].shape == tuple(reversed(t.shape)), n


def test_the_rope_tables_are_the_references():
    """The reference forms each angle in fp32, so at the positions of
    the fixture its cosines differ from those of a float64 angle by about
    6e-5. The tables here are within two fp32 ulps of the reference's at
    every position, and the float64 form of ops.rope_tables is not."""
    arrays, m = committed_fixture("selftest_flux2_klein_rope")
    ids = arrays["ids"].astype(np.float32)
    cos, sin = fm.rope_tables(ids, m["inputs"]["axes_dims"],
                              m["inputs"]["theta"])
    for got, want in ((cos, arrays["cos"]), (sin, arrays["sin"])):
        err = np.abs(np.array(got) - want).max()
        assert err <= 2.4e-7, err
    c64, _ = ops.rope_tables(ids, m["inputs"]["axes_dims"],
                             m["inputs"]["theta"])
    assert np.abs(np.array(c64) - arrays["cos"]).max() > 1e-5


@pytest.mark.parametrize("name", [TINY, NOREFS])
def test_the_tiny_model_matches_the_oracle_in_fp32(tiny, name):
    weights, cfg = tiny
    arrays, _ = committed_fixture(name)
    model = fg.tiny_model(weights, cfg)
    # The CPU stream keeps fp32 products exact; the GPU's TF32 would not.
    with mx.stream(mx.cpu):
        got = fg.measures(model, arrays)
    worst = max(got, key=got.get)
    print(f"flux2 tiny fp32 {name}: {len(got)} measures, worst "
          f"{got[worst]:.2e} at {worst}")
    assert got[worst] < TOL, got


@pytest.mark.parametrize("name", [TINY, NOREFS])
def test_the_tiny_model_in_bf16_is_inside_the_bf16_tolerances(tiny, name):
    """On gmlx's own time embedding, as a request runs."""
    weights, cfg = tiny
    arrays, m = committed_fixture(name)
    model = fg.tiny_model(weights, cfg, act=mx.bfloat16)
    got = fg.measures(model, arrays, own_time=True)
    out = fg.missed(got, m)
    assert not out, out


def test_the_own_time_embedding_is_the_references_to_one_ulp(tiny):
    """At 0.73 one entry of gmlx's embedding is one bf16 ulp off the
    reference's, and the fp32 step on it stays within OWN_TOL."""
    weights, cfg = tiny
    arrays, _ = committed_fixture(TINY)
    n, ulps = fg.own_time_gap(arrays)
    assert n <= 2 and ulps <= 1.0, (n, ulps)
    model = fg.tiny_model(weights, cfg)
    with mx.stream(mx.cpu):
        got = fg.measures(model, arrays, own_time=True)
    worst = max(got, key=got.get)
    print(f"flux2 tiny fp32, own embedding: {n} entries off by up to "
          f"{ulps:.0f} bf16 ulp; worst {got[worst]:.2e} at {worst}")
    assert got[worst] < OWN_TOL, got


def test_the_position_ids_are_the_references(tiny):
    """Image tokens at (0, row, column, 0), the tokens of reference image
    i after them at a first axis of 10 + 10 i, and text at (0, 0, 0,
    position)."""
    weights, cfg = tiny
    arrays, _ = committed_fixture(TINY)
    model = fg.tiny_model(weights, cfg)
    img, txt = fg.ids_of(model, arrays)
    assert np.array_equal(img, arrays["ids.image"])
    assert np.array_equal(txt, arrays["ids.text"])
    n = arrays["latents"].shape[2] * arrays["latents"].shape[3]
    assert set(np.unique(img[:n, 0])) == {0.0}
    assert set(np.unique(img[n:, 0])) == {10.0, 20.0}


def _embedded(t1000: float, dim: int = 8) -> np.ndarray:
    want = ops.timestep_embedding(np.array([t1000], np.float32), dim)
    return np.array(want.astype(mx.bfloat16).astype(mx.float32))


@pytest.mark.parametrize("t, t1000", [(1.0, 1000.0), (0.958, 956.0),
                                      (0.73, 732.0)])
def test_the_time_embedding_follows_the_bf16_time(t, t1000):
    """The time rounds to bf16, then 1000 times it rounds to bf16 before
    the sinusoid, as the reference's embedder does on the bf16 time of its
    denoise. 0.958 is 0.95703125 in bf16 and gives 956, where 1000 times
    the time itself gives 960; 0.73 is 0.73046875 and gives 732, not
    728."""
    got = np.array(fm.time_embedding(np.float32(t), 8))
    assert np.array_equal(got, _embedded(t1000))
    if t != 1.0:
        unrounded = float(fm._bf16(np.float32(1000) * np.float32(t))[()])
        assert unrounded != t1000
        assert not np.array_equal(got, _embedded(unrounded))


def test_the_reference_tokens_run_in_the_activation_dtype(tiny):
    """float32 reference latents enter the image stream as the bf16 of
    the activations, as the image's latents do."""
    weights, cfg = tiny
    arrays, _ = committed_fixture(TINY)
    model = fg.tiny_model(weights, cfg, act=mx.bfloat16)
    lat, feats, refs, sigma = fg.inputs(model, arrays)
    refs = fm.References(refs.tokens * (1 + 2.0 ** -12) + 1e-3, refs.ids)
    rounded = fm.References(refs.tokens.astype(mx.bfloat16), refs.ids)
    assert not mx.array_equal(rounded.tokens.astype(mx.float32),
                              refs.tokens).item()
    got = model(lat, sigma, feats, refs=refs)
    want = model(lat, sigma, feats, refs=rounded)
    assert mx.array_equal(got, want).item()


def test_the_step_parts_give_the_step(tiny):
    weights, cfg = tiny
    arrays, _ = committed_fixture(TINY)
    model = fg.tiny_model(weights, cfg, act=mx.bfloat16)
    lat, feats, refs, sigma = fg.inputs(model, arrays)
    whole = model(lat, sigma, feats, refs=refs)
    mod = model.modulation(sigma, 1)
    cap = model.caption(feats)
    parts = model(lat, sigma, None, refs=refs, mod=mod, caption=cap)
    assert mx.array_equal(whole, parts).item()
    assert len(mod.arrays()) == 6 + 6 + 3 + 2
    assert cap.arrays()[0].shape == (1, feats.shape[1], model.cfg.dim)
    with pytest.raises(ValueError, match="text features"):
        model(lat, sigma, None, mod=mod)


def test_a_batch_of_two_gives_each_row_its_own_velocity(tiny):
    weights, cfg = tiny
    arrays, _ = committed_fixture(TINY)
    model = fg.tiny_model(weights, cfg)
    lat, feats, refs, sigma = fg.inputs(model, arrays)
    with mx.stream(mx.cpu):
        one = model(lat, sigma, feats, refs=refs)
        other = model(lat * 0.5, sigma, feats * 0.5, refs=refs)
        two = model(mx.concatenate([lat, lat * 0.5]), np.repeat(sigma, 2),
                    mx.concatenate([feats, feats * 0.5]), refs=refs)
        assert mx.allclose(two[:1], one, atol=1e-5).item()
        assert mx.allclose(two[1:], other, atol=1e-5).item()


def test_a_pipeline_samples_on_the_grid_of_one_latent_pixel(tiny):
    """A latent pixel is one token, so the DiT has no patch of its own:
    sizes are multiples of the VAE's 16 pixels, and a 32 x 48 request
    samples a 2 x 3 latent grid."""
    from gmlx.image import api
    from gmlx.image.conditioning import Conditioning

    weights, cfg = tiny
    arrays, _ = committed_fixture(TINY)
    model = fg.tiny_model(weights, cfg, act=mx.bfloat16)
    pipe = api.ImagePipeline(dit=model, family=get_family("flux2_klein"),
                             variant="distilled")
    assert model.cfg.patch == 1 and pipe.size_multiple == 16
    cond = Conditioning(mx.array(arrays["features"]), "tiny")
    x = api.sample(pipe, cond, height=32, width=48, seed=0, steps=2)
    assert x.shape == (1, model.cfg.in_channels, 2, 3)
    assert bool(mx.isfinite(x).all().item())
    with pytest.raises(ValueError, match="multiples of 16"):
        api.sample(pipe, cond, height=40, width=48, seed=0, steps=2)


def test_float16_is_refused(tiny):
    weights, cfg = tiny
    model = fg.tiny_model(weights, cfg)
    with pytest.raises(ImageDtypeError, match="float16"):
        model.act_dtype = mx.float16


def test_wrong_configs_are_refused(tiny):
    _, cfg = tiny
    with pytest.raises(ValueError, match="sum to the head dim"):
        fm.Flux2KleinDiT(dict(cfg, axes_dims_rope=[4, 4, 4, 2]))
    with pytest.raises(ValueError, match="guidance"):
        fm.Flux2KleinDiT(dict(cfg, guidance_embeds=True))
    with pytest.raises(ValueError, match="one latent pixel"):
        fm.Flux2KleinDiT(dict(cfg, patch_size=2))


# Wrong models: each changes one convention of the forward.

def _double(mp, change):
    real = fm.DoubleBlock.__call__
    mp.setattr(fm.DoubleBlock, "__call__",
               lambda self, img, txt, mi, mt, ri, rt: real(
                   self, img, txt, *change(mi, mt), ri, rt))


def _shift_scale_swapped(mp, model):
    _double(mp, lambda mi, mt: (((mi[0][1], mi[0][0], mi[0][2]), mi[1]), mt))


def _norm_mods_swapped(mp, model):
    """The modulation of the attention's norm on the feed-forward's and
    the other way round."""
    _double(mp, lambda mi, mt: ((mi[1], mi[0]), mt))


def _stream_mods_crossed(mp, model):
    _double(mp, lambda mi, mt: (mt, mi))


def _glu_halves_swapped(mp, model):
    def hidden(self, mlp, h):
        a, b = mx.split(mlp[0](h), 2, axis=-1)
        return ops.swiglu(b, a)
    mp.setattr(fm.DoubleBlock, "hidden", hidden)


def _single_split_swapped(mp, model):
    """The single block's feed-forward halves in the other order."""
    real = ops.swiglu
    mp.setattr(ops, "swiglu", lambda a, b: real(b, a))
    mp.setattr(fm.DoubleBlock, "hidden",
               lambda self, mlp, h: real(*mx.split(mlp[0](h), 2, axis=-1)))


def _joint_image_first(mp, model):
    """The joint tokens image first against the position ids text first,
    so each token runs at another token's position."""
    real = fm.SingleBlock.__call__

    def call(self, x, mod, cos, sin):
        n = x.shape[1] - model._n_txt
        roll = mx.concatenate([x[:, -n:], x[:, :-n]], axis=1)
        y = real(self, roll, mod, cos, sin)
        return mx.concatenate([y[:, n:], y[:, :n]], axis=1)
    mp.setattr(fm.SingleBlock, "__call__", call)


def _refs_at_the_image(mp, model):
    real = fm.image_ids
    mp.setattr(fm, "image_ids", lambda h, w, index=None: real(h, w))


def _refs_unscaled(mp, model):
    real = fm.image_ids

    def ids(h, w, index=None):
        out = real(h, w)
        if index is not None:
            out[:, 0] = 1 + index
        return out
    mp.setattr(fm, "image_ids", ids)


def _text_at_one_axis_early(mp, model):
    def text_ids(n):
        ids = np.zeros((n, 4), np.float32)
        ids[:, 2] = np.arange(n)
        return ids
    mp.setattr(fm, "text_ids", text_ids)


def _time_unrounded(mp, model):
    def emb(t, dim):
        return ops.timestep_embedding(
            np.float32(1000) * np.reshape(t, (-1,)).astype(np.float32), dim)
    mp.setattr(fm, "time_embedding", emb)


def _final_shift_scale_swapped(mp, model):
    real = fm.LastLayer.__call__
    mp.setattr(fm.LastLayer, "__call__",
               lambda self, x, shift, scale: real(self, x, scale, shift))


def _qk_scales_swapped(mp, model):
    mp.setattr(fm, "qk", lambda norm, q, k, cos, sin, eps: ops.qk_norm_rope(
        q, k, norm.key_norm.scale, norm.query_norm.scale, cos, sin, eps))


FAULTS = {"shift and scale swapped": _shift_scale_swapped,
          "attention and feed-forward modulations swapped":
              _norm_mods_swapped,
          "image and text modulations crossed": _stream_mods_crossed,
          "glu halves swapped": _glu_halves_swapped,
          "single block halves swapped": _single_split_swapped,
          "joint tokens image first": _joint_image_first,
          "references at the image's positions": _refs_at_the_image,
          "reference time axis unscaled": _refs_unscaled,
          "text position on another axis": _text_at_one_axis_early,
          "time embedding in fp32": _time_unrounded,
          "final shift and scale swapped": _final_shift_scale_swapped,
          "q and k norm scales swapped": _qk_scales_swapped}


@pytest.fixture(scope="module")
def clean_velocity(tiny):
    weights, cfg = tiny
    arrays, _ = committed_fixture(TINY)
    with mx.stream(mx.cpu):
        return fg.measures(fg.tiny_model(weights, cfg), arrays)["velocity"]


@pytest.mark.parametrize("name", list(FAULTS))
def test_each_wrong_model_misses_the_fp32_bound(tiny, clean_velocity,
                                                monkeypatch, name):
    """Each wrong model moves its most moved row past 100 times the
    bound of the step on gmlx's own embedding, and moves the velocity
    past 100 times the clean run's, so a fault that leaves the forward as
    it was fails here."""
    weights, cfg = tiny
    arrays, _ = committed_fixture(TINY)
    model = fg.tiny_model(weights, cfg)
    model._n_txt = arrays["features"].shape[0]
    with monkeypatch.context() as mp:
        FAULTS[name](mp, model)
        with mx.stream(mx.cpu):
            got = fg.measures(model, arrays,
                              own_time=name == "time embedding in fp32")
    worst = max(got, key=got.get)
    print(f"flux2 tiny, {name}: worst {got[worst]:.2e} at {worst}, "
          f"velocity {got['velocity']:.2e} against {clean_velocity:.2e} "
          "clean")
    assert got[worst] > 100 * OWN_TOL, (name, worst, got[worst])
    assert got["velocity"] > 100 * clean_velocity, (name, got["velocity"])
