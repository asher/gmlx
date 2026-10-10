"""The Krea 2 DiT on the CPU: the module tree against both converters'
tensor names, and a tiny model against diffusers' own class from the
committed fixtures of scripts/image_oracle/check3_krea2.py --self-test,
in fp32 at a fixed bound and in bf16 at twice the oracle's own gap. The
step parts that the pipeline memoizes give the step they stand in for.
Each wrong model of FAULTS misses the fp32 bound."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

import krea2_gates as kg
from image_fixtures import fixture_view
from oracle_fixture import committed_fixture

from gmlx.image import fingerprint as fp
from gmlx.image import ops
from gmlx.image.dtypes import ImageDtypeError
from gmlx.image.families.krea2 import model as km
from gmlx.image.loader import check_names
from gmlx.image.registry import get_family

TINY = "selftest_krea2_tiny"
# The tiny model in fp32 on the CPU: the right model measures within
# 5.4e-7 of the oracle on every row, at the timestep embedding, whose
# sinusoid differs from torch's in the last bit. The weakest wrong model
# of FAULTS, the erf gelu, moves its most moved row by 2.9e-4.
TOL = 4e-6


@pytest.fixture(scope="module")
def tiny():
    arrays, m = committed_fixture(TINY)
    weights, wm = committed_fixture(TINY + "_weights")
    cfg = wm["inputs"]["config"]
    return arrays, m, weights, cfg


@pytest.mark.parametrize("name", ["krea2_turbo_q4_k_m.header.json",
                                  "krea2_raw_q4_k_m.header.json"])
@pytest.mark.parametrize("plain", [False, True], ids=["last_mlp", "plain"])
def test_the_module_tree_names_every_tensor_of_the_file(name, plain):
    """Strict name equality and the torch shape of every tensor, on the
    real config. Nothing is evaluated."""
    drop = ("last.up.", "last.down.") if plain else ()
    got = fp.identify(fixture_view(name, drop=drop))
    model = get_family("krea2").build(got.config)
    check_names(model, got.view.tensors, "Krea 2")
    params = dict(tree_flatten(model.parameters()))
    for n, t in got.view.tensors.items():
        want = tuple(reversed(t.shape))
        if n == "txtfusion.projector.weight":
            want = (1, 12)
        assert params[n].shape == want, n


def test_the_tiny_model_matches_the_oracle_in_fp32(tiny):
    arrays, _, weights, cfg = tiny
    model = kg.tiny_model(weights, cfg)
    # The CPU stream keeps fp32 products exact; the GPU's TF32 would not.
    with mx.stream(mx.cpu):
        got = kg.measures(model, arrays)
    worst = max(got, key=got.get)
    print(f"krea2 tiny fp32: {len(got)} measures, worst {got[worst]:.2e} "
          f"at {worst}")
    assert got[worst] < TOL, got


def test_the_tiny_model_in_bf16_is_inside_the_bf16_tolerances(tiny):
    arrays, m, weights, cfg = tiny
    model = kg.tiny_model(weights, cfg, act=mx.bfloat16)
    got = kg.measures(model, arrays)
    out = kg.missed(got, m["tolerances"])
    assert not out, out


def test_the_step_parts_give_the_step(tiny):
    arrays, _, weights, cfg = tiny
    model = kg.tiny_model(weights, cfg, act=mx.bfloat16)
    lat = mx.array(arrays["latents"])
    feats = mx.array(arrays["features"])[None]
    sigma = np.reshape(np.float32(arrays["sigma"]), (1,))
    whole = model(lat, sigma, feats)
    mod = model.modulation(sigma, 1)
    cap = model.caption(feats)
    parts = model(lat, sigma, None, mod=mod, caption=cap)
    assert mx.array_equal(whole, parts).item()
    assert len(mod.arrays()) == 2 + 6 * len(model.blocks)
    assert cap.arrays()[0].shape == (1, feats.shape[1], model.cfg.dim)
    with pytest.raises(ValueError, match="text features"):
        model(lat, sigma, None)


def test_a_batch_of_two_gives_each_row_its_own_velocity(tiny):
    arrays, _, weights, cfg = tiny
    model = kg.tiny_model(weights, cfg)
    lat = mx.array(arrays["latents"])
    feats = mx.array(arrays["features"])[None]
    sigma = np.float32(arrays["sigma"])
    with mx.stream(mx.cpu):
        one = model(lat, np.reshape(sigma, (1,)), feats)
        other = model(lat * 0.5, np.reshape(sigma, (1,)), feats)
        two = model(mx.concatenate([lat, lat * 0.5]),
                    np.array([sigma, sigma]), feats)
        assert mx.allclose(two[:1], one, atol=1e-5).item()
        assert mx.allclose(two[1:], other, atol=1e-5).item()


def test_the_timestep_chain_equals_the_reference_bit_for_bit():
    model = km.Krea2DiT(kg.config_of(
        committed_fixture(TINY + "_weights")[1]["inputs"]["config"]))
    s = np.float32(0.6875)
    t = model.timesteps(np.reshape(s, (1,)), 1)
    passed = (s * np.float32(1000)) / np.float32(1000)
    assert t.dtype == np.float32
    assert t.tobytes() == (np.float32(1000) * passed).tobytes()


def test_float16_is_refused(tiny):
    _, _, weights, cfg = tiny
    model = kg.tiny_model(weights, cfg)
    with pytest.raises(ImageDtypeError, match="float16"):
        model.act_dtype = mx.float16


def test_wrong_configs_are_refused():
    cfg = kg.config_of(committed_fixture(TINY + "_weights")[1]["inputs"][
        "config"])
    with pytest.raises(ValueError, match="sum to the head dim"):
        km.Krea2DiT(dict(cfg, axes_dims_rope=[4, 6, 4]))
    with pytest.raises(ValueError, match="multiple"):
        km.Krea2DiT(dict(cfg, num_key_value_heads=3))


# Wrong models: each changes one convention of the forward.

def _norm_not_centered(mp, model):
    mp.setattr(km.ZeroCenteredRMSNorm, "weight",
               lambda self: self.scale.astype(mx.float32))


def _gate_on_output(mp, model):
    def gated(self, x, o):
        b, _, n, _ = o.shape
        o = o.transpose(0, 2, 1, 3).reshape(b, n, -1)
        return o * mx.sigmoid(self.gate(o))
    mp.setattr(km.Attention, "gated", gated)


def _mod_chunks_swapped(mp, model):
    real = km.Block.modulation

    def swapped(self, tvec):
        a = real(self, tvec)
        return (a[1], a[0], *a[2:])
    mp.setattr(km.Block, "modulation", swapped)


def _final_reads_tvec(mp, model):
    real = km.LastLayer.scale_shift

    def shifted(self, temb):
        scale, shift = real(self, temb)
        return shift, scale
    mp.setattr(km.LastLayer, "scale_shift", shifted)


def _rope_halves(mp, model):
    real = km.rope_tables

    def tables(ids, axes, theta):
        return real(ids[:, ::-1].copy(), axes, theta)
    mp.setattr(km, "rope_tables", tables)


def _text_rows_last(mp, model):
    real = km.Krea2DiT.__call__

    def call(self, latents, sigma, feats, taps=None, *, mod=None,
             caption=None):
        cap = self.caption(feats) if caption is None else caption
        rows = cap.rows[:, ::-1]
        return real(self, latents, sigma, None, taps, mod=mod,
                    caption=km.Caption(rows))
    mp.setattr(km.Krea2DiT, "__call__", call)


def _patch_order(mp, model):
    def patchify(self, latents):
        b, c, h, w = latents.shape
        p = self.cfg.patch
        x = latents.reshape(b, c, h // p, p, w // p, p)
        return x.transpose(0, 2, 4, 3, 5, 1).reshape(
            b, (h // p) * (w // p), c * p * p)
    mp.setattr(km.Krea2DiT, "patchify", patchify)


def _gelu_exact(mp, model):
    mp.setattr(km.nn, "gelu_approx", nn.gelu)


def _projector_mean(mp, model):
    def call(self, feats, taps=None):
        b, n, _ = feats.shape
        x = feats.reshape(b * n, self.txt_layers, self.txt_dim)
        for blk in self.layerwise_blocks:
            x = blk(x)
        x = x.reshape(b, n, self.txt_layers, self.txt_dim).mean(axis=2)
        for blk in self.refiner_blocks:
            x = blk(x)
        return x
    mp.setattr(km.TextFusion, "__call__", call)


FAULTS = {
    "norm not centered": _norm_not_centered,
    "gate on the attention output": _gate_on_output,
    "scale and shift swapped": _mod_chunks_swapped,
    "final scale and shift swapped": _final_reads_tvec,
    "rope axes reversed": _rope_halves,
    "text rows reversed": _text_rows_last,
    "patch channels last": _patch_order,
    "exact gelu": _gelu_exact,
    "projector a mean": _projector_mean,
}


@pytest.mark.parametrize("name", list(FAULTS))
def test_each_wrong_model_fails_the_tiny_gate(tiny, monkeypatch, name):
    arrays, _, weights, cfg = tiny
    model = kg.tiny_model(weights, cfg)
    with monkeypatch.context() as mp:
        FAULTS[name](mp, model)
        with mx.stream(mx.cpu):
            got = kg.measures(model, arrays)
    worst = max(got, key=got.get)
    print(f"krea2 tiny, {name}: worst {got[worst]:.2e} at {worst}")
    assert got[worst] > 25 * TOL


def test_the_ops_seams_stay_in_force():
    """The model calls each seam through gmlx.image.ops, so a kernel
    install reaches it."""
    import inspect

    src = inspect.getsource(km)
    for seam in ("ops.sdpa", "ops.qk_norm_rope", "ops.modulate",
                 "ops.gated_residual", "ops.swiglu"):
        assert seam in src, seam
    assert ops.SEAMS
