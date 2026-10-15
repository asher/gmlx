"""The FLUX.3 klein DiT on the CPU: the module tree against the tensor
names of the distilled and Base files, and a tiny model against BFL's own
class from the committed fixtures of
scripts/image_oracle/check3_flux3_klein.py --self-test, with two
reference images and the flow-map time, and as Base with text alone, in
fp32 at a fixed bound and in bf16 at twice the oracle's own gap. The step
parts that a pipeline memoizes give the step they stand in for. Each
wrong model of FAULTS misses the fp32 bound."""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

import flux3_gates as fg
from image_fixtures import fixture_view
from oracle_fixture import committed_fixture

from gmlx.image import fingerprint as fp
from gmlx.image import ops
from gmlx.image.dtypes import ImageDtypeError
from gmlx.image.families.flux3_klein import model as fm
from gmlx.image.loader import check_names
from gmlx.image.registry import get_family

TINY = "selftest_flux3_klein_tiny"
BASE = "selftest_flux3_klein_tiny_base"
# The tiny model in fp32 on the CPU against the oracle in fp32. The
# weakest wrong model of FAULTS moves its most moved row far past it.
TOL = 4e-6
# Times 4 and 5 of sampling.timesteps(16, 6.93), in float32.
STEP4 = 0.954107403755188
STEP4_TARGET = 0.9384464025497437


@pytest.fixture(scope="module")
def tiny():
    weights, wm = committed_fixture(TINY + "_weights")
    return weights, wm["inputs"]["config"]


def _case(name):
    arrays, m = committed_fixture(name)
    return arrays, m, m["inputs"]["flow_map"]


@pytest.mark.parametrize("name", ["flux3_klein_q4_k_m.header.json",
                                  "flux3_klein_abiray_q4_k_m.header.json",
                                  "flux3_klein_base_q4_k_m.header.json"])
def test_the_module_tree_names_every_tensor_of_the_file(name):
    """Strict name equality and the torch shape of every tensor, on the
    real config. Nothing is evaluated."""
    got = fp.identify(fixture_view(name))
    model = get_family("flux3_klein").build(got.config)
    check_names(model, got.view.tensors, "FLUX.3 klein")
    params = dict(tree_flatten(model.parameters()))
    for n, t in got.view.tensors.items():
        assert params[n].shape == tuple(reversed(t.shape)), n
    assert hasattr(model, "ref_time_in") == got.config["flow_map"]


@pytest.mark.parametrize("name", [TINY, BASE])
def test_the_tiny_model_matches_the_oracle_in_fp32(tiny, name):
    weights, cfg = tiny
    arrays, _, flow = _case(name)
    model = fg.tiny_model(weights, cfg, flow_map=flow)
    # The CPU stream keeps fp32 products exact; the GPU's TF32 would not.
    with mx.stream(mx.cpu):
        got = fg.measures(model, arrays)
    worst = max(got, key=got.get)
    print(f"flux3 tiny fp32 {name}: {len(got)} measures, worst "
          f"{got[worst]:.2e} at {worst}")
    assert got[worst] < TOL, got


@pytest.mark.parametrize("name", [TINY, BASE])
def test_the_tiny_model_in_bf16_is_inside_the_bf16_tolerances(tiny, name):
    weights, cfg = tiny
    arrays, m, flow = _case(name)
    model = fg.tiny_model(weights, cfg, act=mx.bfloat16, flow_map=flow)
    got = fg.measures(model, arrays)
    out = fg.missed(got, m["tolerances"])
    assert not out, out


def test_the_position_ids_are_the_references(tiny):
    weights, cfg = tiny
    arrays, _, _ = _case(TINY)
    model = fg.tiny_model(weights, cfg)
    ids = fg._ids(model, arrays)
    for m in fm.MODES:
        assert np.array_equal(ids[m], arrays[f"ids.{m}"]), m
    assert set(np.unique(ids["image_cond"][:, 3])) == {-1.0, -2.0}


def _embedded(t1000: float, dim: int = 8) -> np.ndarray:
    want = ops.timestep_embedding(np.array([t1000], np.float32), dim)
    return np.array(want.astype(mx.bfloat16).astype(mx.float32))


@pytest.mark.parametrize("t, t1000", [(0.9905, 992.0), (STEP4, 952.0),
                                      (STEP4_TARGET, 936.0)])
def test_the_time_embedding_follows_the_bf16_pipeline(t, t1000):
    """The time rounds to bf16, then 1000 times it rounds to bf16 before
    the sinusoid. 0.9905 is 0.9921875 in bf16, and 1000 times that is 992
    with or without the first rounding. The time of step 4 of the 16-step
    schedule, 0.954107, is 0.953125 in bf16 and gives 952, where 1000
    times the time itself gives 956; the time of step 5 gives 936, not
    940."""
    got = np.array(fm.time_embedding(np.float32(t), 8))
    assert np.array_equal(got, _embedded(t1000))
    unrounded = float(fm._bf16(np.float32(1000) * np.float32(t))[()])
    if t != 0.9905:
        assert unrounded == t1000 + 4.0
        assert not np.array_equal(got, _embedded(unrounded))


def test_the_step_parts_give_the_step(tiny):
    weights, cfg = tiny
    arrays, _, _ = _case(TINY)
    model = fg.tiny_model(weights, cfg, act=mx.bfloat16)
    lat, feats, refs, sigma, target = fg.inputs(model, arrays)
    whole = model(lat, sigma, feats, target=target, refs=refs)
    mod = model.modulation(sigma, 1, target)
    cap = model.caption(feats)
    parts = model(lat, sigma, None, refs=refs, mod=mod, caption=cap)
    assert mx.array_equal(whole, parts).item()
    assert len(mod.arrays()) == 2 + 3 * 2 * 3
    assert cap.arrays()[0].shape == (1, feats.shape[1], model.cfg.dim)
    with pytest.raises(ValueError, match="text features"):
        model(lat, sigma, None, mod=mod)


def test_the_flow_map_needs_its_target_and_base_takes_none(tiny):
    weights, cfg = tiny
    sigma = np.ones((1,), np.float32)
    with pytest.raises(ValueError, match="flow map"):
        fg.tiny_model(weights, cfg).modulation(sigma, 1)
    with pytest.raises(ValueError, match="no target"):
        fg.tiny_model(weights, cfg, flow_map=False).modulation(
            sigma, 1, target=sigma)


def test_a_batch_of_two_gives_each_row_its_own_velocity(tiny):
    weights, cfg = tiny
    arrays, _, _ = _case(TINY)
    model = fg.tiny_model(weights, cfg)
    lat, feats, refs, sigma, target = fg.inputs(model, arrays)
    with mx.stream(mx.cpu):
        one = model(lat, sigma, feats, target=target, refs=refs)
        other = model(lat * 0.5, sigma, feats, target=target, refs=refs)
        two = model(mx.concatenate([lat, lat * 0.5]),
                    np.repeat(sigma, 2), feats, target=np.repeat(target, 2),
                    refs=refs)
        assert mx.allclose(two[:1], one, atol=1e-5).item()
        assert mx.allclose(two[1:], other, atol=1e-5).item()


def test_float16_is_refused(tiny):
    weights, cfg = tiny
    model = fg.tiny_model(weights, cfg)
    with pytest.raises(ImageDtypeError, match="float16"):
        model.act_dtype = mx.float16


def test_wrong_configs_are_refused(tiny):
    _, cfg = tiny
    with pytest.raises(ValueError, match="sum to the head dim"):
        fm.Flux3KleinDiT(dict(cfg, axes_dims_rope=[4, 4, 4, 2]))


# Wrong models: each changes one convention of the forward.

def _shift_scale_swapped(mp, model):
    real = fm.Block.__call__
    mp.setattr(fm.Block, "__call__", lambda self, x, mod, cos, sin: real(
        self, x, (mod[1], mod[0], mod[2]), cos, sin))


def _gate_on_attention_only(mp, model):
    def out(self, x, gate, rows, hid):
        return x + gate * self.attn_out(rows) + self.mlp_out(hid)
    mp.setattr(fm.Block, "out", out)


def _glu_halves_swapped(mp, model):
    def hidden(self, h):
        a, b = mx.split(self.mlp_in(h), 2, axis=-1)
        return ops.swiglu(b, a)
    mp.setattr(fm.Block, "hidden", hidden)


def _rope_unscaled(mp, model):
    real = fm.image_ids
    mp.setattr(fm, "image_ids", lambda h, w, scale, index=None: real(
        h, w, 1.0, index))


def _text_at_one_axis_early(mp, model):
    def text_ids(n):
        ids = np.zeros((n, 4), np.float32)
        ids[:, 2] = np.arange(n)
        return ids
    mp.setattr(fm, "text_ids", text_ids)


def _single_order(mp, model):
    real = fm.Flux3KleinDiT._stream

    def stream(self, x, blocks, mod, ids, name, taps):
        if name == "single_blocks":
            x = x[:, ::-1]
            ids = ids[::-1].copy()
            mod = tuple(t[:, ::-1] for t in mod)
            return real(self, x, blocks, mod, ids, name, taps)[:, ::-1]
        return real(self, x, blocks, mod, ids, name, taps)
    mp.setattr(fm.Flux3KleinDiT, "_stream", stream)


def _no_ref_time(mp, model):
    real = fm.Flux3KleinDiT.vectors

    def vectors(self, sigma, batch, target=None):
        v = real(self, sigma, batch, target)
        if target is None:
            return v
        ref = self._embed(self.ref_time_in, target, batch)
        return {"image": v["image"] - ref, "image_cond": v["image_cond"],
                "text": v["text"]}
    mp.setattr(fm.Flux3KleinDiT, "vectors", vectors)


def _time_unrounded(mp, model):
    def emb(t, dim):
        return ops.timestep_embedding(
            np.float32(1000) * np.reshape(t, (-1,)).astype(np.float32), dim)
    mp.setattr(fm, "time_embedding", emb)


def _final_shift_scale_swapped(mp, model):
    real = fm.LastLayer.__call__
    mp.setattr(fm.LastLayer, "__call__",
               lambda self, x, shift, scale: real(self, x, scale, shift))


FAULTS = {"shift and scale swapped": _shift_scale_swapped,
          "gate on the attention alone": _gate_on_attention_only,
          "glu halves swapped": _glu_halves_swapped,
          "image ids unscaled": _rope_unscaled,
          "text position on another axis": _text_at_one_axis_early,
          "single stream token order": _single_order,
          "target time left out of the image": _no_ref_time,
          "time embedding in fp32": _time_unrounded,
          "final shift and scale swapped": _final_shift_scale_swapped}


@pytest.mark.parametrize("name", list(FAULTS))
def test_each_wrong_model_misses_the_fp32_bound(tiny, monkeypatch, name):
    weights, cfg = tiny
    arrays, _, _ = _case(TINY)
    model = fg.tiny_model(weights, cfg)
    with monkeypatch.context() as mp:
        FAULTS[name](mp, model)
        with mx.stream(mx.cpu):
            got = fg.measures(model, arrays)
    worst = max(got, key=got.get)
    print(f"flux3 tiny, {name}: worst {got[worst]:.2e} at {worst}")
    assert got[worst] > 100 * TOL, (name, worst, got[worst])
