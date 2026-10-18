"""The MOSS-Image DiT on the CPU: the module tree against the file's
tensor names, the RoPE bands against the file's stored copy, and a tiny
model in fp32 against a float64 numpy forward written from the release's
MOSSImageTransformer, with its rotate-half RoPE, its six shared vectors
and its head. The step parts give the step they stand in for. Each wrong
model of FAULTS misses the fp32 bound."""
from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

import moss_gates as mg
from image_fixtures import fixture_view, load_json
from oracle_fixture import committed_fixture

from gmlx.image import fingerprint as fp
from gmlx.image.dtypes import ImageDtypeError
from gmlx.image.families.moss_image import model as mm
from gmlx.image.loader import check_names
from gmlx.image.registry import get_family

TINY = {"dim": 64, "ffn_dim": 48, "freq_dim": 16, "latent_channels": 8,
        "num_heads": 2, "num_layers": 2, "rope_base": 10000.0,
        "text_dim": 24, "num_text_blocks": 2}
H, W, L = 3, 4, 5
SIGMA = np.float32(0.6875)
# The tiny model in fp32 on the CPU against the float64 forward: the
# right model lands within 1e-6 relative; the weakest wrong model of
# FAULTS moves the velocity by more than 25 times that.
TOL = 4e-6


def _tiny(seed: int = 0, act=mx.float32):
    m = mm.MossImageDiT(TINY)
    rng = np.random.default_rng(seed)
    params = []
    for k, v in tree_flatten(m.parameters()):
        a = rng.standard_normal(v.shape) * 0.3
        if k.endswith("norm1.weight") or k.endswith("norm2.weight"):
            a += 1.0
        params.append((k, mx.array(a.astype(np.float32))))
    m.load_weights(params)
    m.act_dtype = act
    m.eval()
    return m, {k: np.array(v, np.float64) for k, v in params}


def _inputs(seed: int = 1, batch: int = 1):
    rng = np.random.default_rng(seed)
    lat = rng.standard_normal((batch, 8, H, W)).astype(np.float32)
    feats = rng.standard_normal((1, L, 24)).astype(np.float32)
    return lat, feats


# The float64 forward, written from the release.

def _lin(p, name, x):
    y = x @ p[name + ".weight"].T
    return y + p[name + ".bias"] if name + ".bias" in p else y


def _rms(x, w):
    return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + 1e-6) * w


def _situ(hid, gate):
    sig = 1 / (1 + np.exp(-gate))
    return 4 * np.tanh(gate / 4) * sig * (25 * np.tanh(hid / 25))


def _rope(h, w, n):
    nb = 4
    bands = (np.float32(1) / np.float32(10000) ** (
        np.arange(nb, dtype=np.float32) / np.float32(nb))).astype(np.float32)
    r, c = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    img = np.stack([np.zeros(h * w), np.full(h * w, n), r.reshape(-1),
                    c.reshape(-1)], -1).astype(np.float32)
    img[:, 2] -= (h - 1) / 2
    img[:, 3] -= (w - 1) / 2
    txt = np.zeros((n, 4), np.float32)
    txt[:, 1] = np.arange(n)
    phase = (np.concatenate([img, txt])[:, :, None] * bands).reshape(
        h * w + n, -1).astype(np.float64)
    return np.sin(phase), np.cos(phase)


def _rotate(x, sin, cos):
    sin = np.concatenate([sin, sin], -1)[:, None]
    cos = np.concatenate([cos, cos], -1)[:, None]
    a, b = np.split(x, 2, -1)
    return x * cos + np.concatenate([-b, a], -1) * sin


def _attend(q, k, v, rope, qn, kn, heads):
    n = q.shape[0]
    q = _rotate(_rms(q.reshape(n, heads, -1), qn), *rope)
    k = _rotate(_rms(k.reshape(n, heads, -1), kn), *rope)
    v = v.reshape(n, heads, -1)
    s = np.einsum("qhd,khd->hqk", q, k) / np.sqrt(q.shape[-1])
    s = np.exp(s - s.max(-1, keepdims=True))
    s /= s.sum(-1, keepdims=True)
    return np.einsum("hqk,khd->qhd", s, v).reshape(n, -1)


def reference(p, lat, sigma, feats, heads=2):
    """Velocity (B, C, H, W) float64 of the release's forward."""
    b, c, h, w = lat.shape
    t = np.float64(np.float32(sigma) * np.float32(1000))
    half = 8
    phase = t * 10000.0 ** (-np.arange(half, dtype=np.float64) / half)
    emb = np.concatenate([np.cos(phase), np.sin(phase)]).astype(
        np.float32).astype(np.float64)
    hid = _lin(p, "time_embedding.0", emb)
    time = _lin(p, "time_embedding.2", hid / (1 + np.exp(-hid)))
    mod = _lin(p, "shared_block_modulation", time).reshape(6, -1)
    head = _lin(p, "video_head_modulation", time).reshape(2, -1)
    n = feats.shape[1]
    sin, cos = _rope(h, w, n)
    txt_rope = (sin[h * w:], cos[h * w:])
    x = _lin(p, "text_embedder.proj", feats[0].astype(np.float64))
    for i in range(2):
        pre = f"text_embedder.blocks.{i}."
        q, k, v = np.split(_lin(p, pre + "qkv", _rms(x, p[pre + "norm1.weight"])),
                           3, -1)
        x = x + _lin(p, pre + "proj", _attend(
            q, k, v, txt_rope, p[pre + "q_norm.weight"],
            p[pre + "k_norm.weight"], heads))
        y = _rms(x, p[pre + "norm2.weight"])
        x = x + _lin(p, pre + "ffn_down", _situ(_lin(p, pre + "ffn_up", y),
                                                _lin(p, pre + "ffn_gate", y)))
    out = []
    for j in range(b):
        tok = lat[j].reshape(c, h * w).T.astype(np.float64)
        z = np.concatenate([_lin(p, "video_embedder", tok), x])
        for i in range(len([k for k in p if k.endswith("proj.o.weight")])):
            pre = f"layers.{i}."
            sa, ca, ga, sf, cf, gf = mod
            y = _rms(z, p[pre + "norm1.weight"]) * (1 + ca) + sa
            o = _attend(_lin(p, pre + "proj.q", y), _lin(p, pre + "proj.k", y),
                        _lin(p, pre + "proj.v", y), (sin, cos),
                        p[pre + "q_norm.weight"], p[pre + "k_norm.weight"],
                        heads)
            z = z + _lin(p, pre + "proj.o", o) * ga
            y = _rms(z, p[pre + "norm2.weight"]) * (1 + cf) + sf
            hg = _lin(p, pre + "proj.ffn_up_gate", y)
            z = z + _lin(p, pre + "proj.ffn_down", _situ(*np.split(hg, 2, -1))) * gf
        y = _rms(z[:h * w], p["final_norm_video.weight"]) * (1 + head[1]) + head[0]
        out.append(_lin(p, "final_linear_video", y).T.reshape(c, h, w))
    return np.stack(out)


def _gap(model, p, lat, feats) -> float:
    with mx.stream(mx.cpu):
        got = np.array(model(mx.array(lat), np.reshape(SIGMA, (1,)),
                             mx.array(feats)), np.float64)
    want = reference(p, lat, SIGMA, feats)
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


def test_the_module_tree_names_every_tensor_of_the_file():
    """Strict name equality and the torch shape of every tensor, on the
    real config. Nothing is evaluated."""
    got = fp.identify(fixture_view("moss_image_sft_q4_k_m.header.json"))
    model = get_family("moss_image").build(got.config)
    check_names(model, got.view.tensors, "MOSS-Image")
    params = dict(tree_flatten(model.parameters()))
    for n, t in got.view.tensors.items():
        assert params[n].shape == tuple(reversed(t.shape)), n


def test_the_bands_equal_the_files_copy_bit_for_bit():
    want = np.array(load_json("moss_image_identity.json")["rope_bands"]["bits"],
                    np.uint32)
    assert mm.rope_bands(16, 10000.0).view(np.uint32).tolist() == \
        want.tolist()


def test_the_coordinates_center_the_image_and_count_the_text():
    got = mm.positions(2, 3, 4)
    assert got.dtype == np.float32 and got.shape == (10, 4)
    assert got[:6].tolist() == [[0, 4, -0.5, -1], [0, 4, -0.5, 0],
                                [0, 4, -0.5, 1], [0, 4, 0.5, -1],
                                [0, 4, 0.5, 0], [0, 4, 0.5, 1]]
    assert got[6:].tolist() == [[0, i, 0, 0] for i in range(4)]


def test_the_sinusoid_rounds_float64_phases_once():
    """Each entry is the double-precision cosine or sine of t times its
    frequency, rounded once to float32. The same phases in float32 miss."""
    t = 300.0
    freqs = [10000.0 ** (-k / 8) for k in range(8)]
    want = ([math.cos(t * f) for f in freqs]
            + [math.sin(t * f) for f in freqs])
    got = np.array(mm.time_sinusoid(np.array([t], np.float32), 16, 10000.0))
    assert got[0].tolist() == np.float32(want).tolist()
    single = np.float32(t) * np.float32(freqs)
    assert np.cos(single).tolist() != got[0, :8].tolist()


def test_the_tiny_model_matches_the_release_forward_in_fp32():
    model, p = _tiny()
    lat, feats = _inputs()
    gap = _gap(model, p, lat, feats)
    print(f"moss tiny fp32: rel L2 {gap:.2e}")
    assert gap < TOL


@pytest.fixture(scope="module")
def release_tiny():
    """The release's class on random tiny weights, from the oracle's
    --self-test."""
    arrays, m = committed_fixture("selftest_moss_image_tiny")
    weights, wm = committed_fixture("selftest_moss_image_tiny_weights")
    bands = weights["rope.bands"]
    return arrays, m, weights, wm["inputs"]["config"], bands


def test_the_tiny_model_matches_the_release_class_in_fp32(release_tiny):
    arrays, _, weights, cfg, bands = release_tiny
    model = mg.tiny_model(weights, cfg)
    assert mm.rope_bands(len(bands), cfg["rope_base"]).tobytes() == \
        bands.tobytes()
    # The CPU stream keeps fp32 products exact; the GPU's TF32 would not.
    with mx.stream(mx.cpu):
        got = mg.measures(model, arrays)
    worst = max(got, key=got.get)
    print(f"moss release tiny fp32: {len(got)} measures, worst "
          f"{got[worst]:.2e} at {worst}")
    assert got[worst] < TOL, got


def test_the_tiny_model_in_bf16_is_inside_the_bf16_tolerances(release_tiny):
    arrays, m, weights, cfg, _ = release_tiny
    model = mg.tiny_model(weights, cfg, act=mx.bfloat16)
    out = mg.missed(mg.measures(model, arrays), m["tolerances"])
    assert not out, out


def test_the_step_parts_give_the_step():
    model, _ = _tiny(act=mx.bfloat16)
    lat, feats = (mx.array(a) for a in _inputs())
    sigma = np.reshape(SIGMA, (1,))
    whole = model(lat, sigma, feats)
    mod = model.modulation(sigma, 1)
    cap = model.caption(feats)
    parts = model(lat, sigma, None, mod=mod, caption=cap)
    assert mx.array_equal(whole, parts).item()
    assert len(mod.arrays()) == 8
    assert cap.arrays()[0].shape == (1, L, 64)
    with pytest.raises(ValueError, match="text features"):
        model(lat, sigma, None)


def test_a_batch_of_two_gives_each_row_its_own_velocity():
    model, _ = _tiny()
    lat, feats = _inputs(batch=2)
    lat, feats = mx.array(lat), mx.array(feats)
    sigma = np.reshape(SIGMA, (1,))
    with mx.stream(mx.cpu):
        two = model(lat, np.array([SIGMA, SIGMA]), feats)
        for i in range(2):
            one = model(lat[i:i + 1], sigma, feats)
            assert mx.allclose(two[i:i + 1], one, atol=1e-5).item()


def test_float16_is_refused():
    model, _ = _tiny()
    with pytest.raises(ImageDtypeError, match="float16"):
        model.act_dtype = mx.float16


def test_wrong_configs_are_refused():
    with pytest.raises(ValueError, match="head dim"):
        mm.MossImageDiT(dict(TINY, num_heads=3))
    with pytest.raises(ValueError, match="RoPE bands"):
        mm.MossImageDiT(dict(TINY, num_bands=8))
    with pytest.raises(ValueError, match="even width"):
        mm.MossImageDiT(dict(TINY, freq_dim=15))


# Wrong models: each changes one convention of the forward.

def _rope_adjacent(mp):
    mp.setattr(mm, "interleave", lambda x: x)


def _shift_scale_swapped(mp):
    real = mm.MossImageDiT.modulation

    def swapped(self, sigma, batch):
        got = real(self, sigma, batch)
        b = got.blocks
        return mm.StepModulation((b[1], b[0], *b[2:]), got.shift, got.scale)
    mp.setattr(mm.MossImageDiT, "modulation", swapped)


def _head_swapped(mp):
    real = mm.MossImageDiT.modulation

    def swapped(self, sigma, batch):
        got = real(self, sigma, batch)
        return mm.StepModulation(got.blocks, got.scale, got.shift)
    mp.setattr(mm.MossImageDiT, "modulation", swapped)


def _situ_halves_swapped(mp):
    real = mm.situ
    mp.setattr(mm, "situ", lambda hid, gate: real(gate, hid))


def _grid_not_centered(mp):
    real = mm.positions

    def pos(h, w, n):
        got = real(h, w, n)
        got[:h * w, 2:] += np.float32([(h - 1) / 2, (w - 1) / 2])
        return got
    mp.setattr(mm, "positions", pos)


def _image_at_text_zero(mp):
    real = mm.positions

    def pos(h, w, n):
        got = real(h, w, n)
        got[:h * w, 1] = 0
        return got
    mp.setattr(mm, "positions", pos)


def _sines_first(mp):
    def sinusoid(t, dim, period):
        got = np.array(mm_real_sinusoid(t, dim, period))
        return mx.array(np.roll(got, dim // 2, axis=-1))
    mm_real_sinusoid = mm.time_sinusoid
    mp.setattr(mm, "time_sinusoid", sinusoid)


FAULTS = {
    "rope pairs adjacent dims": _rope_adjacent,
    "block shift and scale swapped": _shift_scale_swapped,
    "head shift and scale swapped": _head_swapped,
    "situ halves swapped": _situ_halves_swapped,
    "grid not centered": _grid_not_centered,
    "image rows at text position 0": _image_at_text_zero,
    "sines first": _sines_first,
}


@pytest.mark.parametrize("name", list(FAULTS))
def test_each_wrong_model_fails_the_tiny_gate(monkeypatch, name):
    model, p = _tiny()
    lat, feats = _inputs()
    with monkeypatch.context() as mp:
        FAULTS[name](mp)
        gap = _gap(model, p, lat, feats)
    print(f"moss tiny, {name}: rel L2 {gap:.2e}")
    assert gap > 25 * TOL


@pytest.mark.parametrize("name", list(FAULTS))
def test_each_wrong_model_misses_the_release_class(release_tiny, monkeypatch,
                                                   name):
    arrays, _, weights, cfg, _ = release_tiny
    model = mg.tiny_model(weights, cfg)
    with monkeypatch.context() as mp, mx.stream(mx.cpu):
        FAULTS[name](mp)
        gap = mg.rel(mg.free_rows(model, arrays)["velocity"],
                     arrays["velocity"])
    print(f"moss release tiny, {name}: velocity rel L2 {gap:.2e}")
    assert gap > 25 * TOL


def test_every_op_of_a_tiny_bf16_step_is_within_its_rounding_bound():
    """The site gate on the tiny model in bf16: each op of each block and
    each call of a linear leaf against the stock op in float32 on the same
    input, within 2 n u, and no op or leaf of the step left unread."""
    import moss_gates as mg
    import qwen_gates as qg

    model, _ = _tiny(act=mx.bfloat16)
    model.set_dtype(mx.bfloat16)
    lat, feats = (mx.array(a) for a in _inputs())
    sites: dict = {}
    skipped: set = set()
    with mg.step_sites(model, sites, skipped=skipped):
        mx.eval(model(lat, np.reshape(SIGMA, (1,)), feats))
    assert not mg.missing_ops(model, sites, skipped=skipped)
    assert not qg.over_gate(sites), qg.over_gate(sites)
    assert len(qg.op_sites(sites)) == 2 * 9 + 2 * 5
