"""The Z-Image module tree and forward on a tiny config in fp32, against an
independent numpy reference of the diffusers semantics: patch order, pad
rows, position ids, the fused qkv split, interleaved RoPE, modulation and
the sign of the output. The tree's names equal the header fixture's."""
from __future__ import annotations

import math

import numpy as np
import pytest

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map

from image_fixtures import TINY_Z_CONFIG, Z_HEADER, fixture_view, tiny_z_model

from gmlx.image import fingerprint as fp

def test_the_real_config_builds_the_header_names():
    from gmlx.image.registry import get_family

    fam = get_family("z_image")
    got = fp.identify(fixture_view(Z_HEADER))
    model = fam.build(got.config)
    names = {k for k, _ in tree_flatten(model.parameters())}
    assert names == set(got.view.tensors)


def test_patchify_round_trips_and_follows_the_reference_order():
    m, _ = tiny_z_model()
    lat = mx.arange(2 * 4 * 6 * 8, dtype=mx.float32).reshape(2, 4, 6, 8)
    patches = m.patchify(lat)
    assert patches.shape == (2, 12, 16)
    # Feature order is (ph, pw, c), row order is (h, w).
    ref = np.array(lat).reshape(2, 4, 3, 2, 4, 2).transpose(0, 2, 4, 3, 5, 1)
    assert np.array_equal(np.array(patches), ref.reshape(2, 12, 16))
    assert np.array_equal(np.array(m.unpatchify(patches, 3, 4)),
                          np.array(lat))


def test_positions_put_the_caption_first_on_axis_zero():
    m, _ = tiny_z_model()
    img, cap = m.positions(3, 4, 32)
    assert img.shape == (32, 3) and cap.shape == (32, 3)
    assert cap[:3].tolist() == [[1, 0, 0], [2, 0, 0], [3, 0, 0]]
    assert cap[-1].tolist() == [32, 0, 0]
    assert img[0].tolist() == [33, 0, 0]
    assert img[5].tolist() == [33, 1, 1]
    assert img[11].tolist() == [33, 2, 3]
    assert (img[12:] == 0).all()


def _rms(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def _rope_ref(ids, dims, theta):
    parts = []
    for a, d in enumerate(dims):
        f = 1.0 / theta ** (np.arange(0, d, 2) / d)
        parts.append(np.exp(1j * np.outer(ids[:, a], f)))
    return np.concatenate(parts, axis=1)


def _ref_forward(p, lat, sigma, cap):
    c = TINY_Z_CONFIG
    nh, hd, eps = c["n_heads"], c["head_dim"], c["norm_eps"]
    silu = lambda z: z / (1 + np.exp(-z))  # noqa: E731

    def lin(name, x, bias=True):
        y = x @ p[name + ".weight"].T
        return y + p[name + ".bias"] if bias else y

    def attn(pre, x, fc):
        n = x.shape[0]
        qkv = lin(pre + ".attention.qkv", x, False).reshape(n, 3, nh, hd)
        q = _rms(qkv[:, 0], p[pre + ".attention.q_norm.weight"], eps)
        k = _rms(qkv[:, 1], p[pre + ".attention.k_norm.weight"], eps)
        v = qkv[:, 2]

        def rot(z):
            zc = z.reshape(n, nh, hd // 2, 2)
            zc = (zc[..., 0] + 1j * zc[..., 1]) * fc[:, None, :]
            return np.stack([zc.real, zc.imag], -1).reshape(n, nh, hd)

        q, k = rot(q), rot(k)
        s = np.einsum("qhd,khd->hqk", q, k) / math.sqrt(hd)
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        o = np.einsum("hqk,khd->qhd", s, v).reshape(n, nh * hd)
        return lin(pre + ".attention.out", o, False)

    def ffn(pre, x):
        a = lin(pre + ".feed_forward.w1", x, False)
        b = lin(pre + ".feed_forward.w3", x, False)
        return lin(pre + ".feed_forward.w2", silu(a) * b, False)

    def block(pre, x, fc, t=None):
        n1 = lambda z, k: _rms(z, p[f"{pre}.{k}.weight"], eps)  # noqa: E731
        if t is None:
            x = x + n1(attn(pre, n1(x, "attention_norm1"), fc),
                       "attention_norm2")
            return x + n1(ffn(pre, n1(x, "ffn_norm1")), "ffn_norm2")
        mod = lin(pre + ".adaLN_modulation.0", t)
        sm, gm, sf, gf = np.split(mod, 4)
        x = x + np.tanh(gm) * n1(attn(pre, n1(x, "attention_norm1")
                                      * (1 + sm), fc), "attention_norm2")
        return x + np.tanh(gf) * n1(ffn(pre, n1(x, "ffn_norm1") * (1 + sf)),
                                    "ffn_norm2")

    # Timestep embedding.
    t = (1000 - sigma * 1000) / 1000 * c["t_scale"]
    half = c["t_freq_dim"] // 2
    freqs = np.exp(-math.log(10000) * np.arange(half) / half)
    tf = np.concatenate([np.cos(t * freqs), np.sin(t * freqs)])
    temb = lin("t_embedder.mlp.2", silu(lin("t_embedder.mlp.0", tf)))

    # Image patches: (c, h, w) -> rows (h, w), features (ph, pw, c).
    ch, h, w = lat.shape
    ht, wt = h // 2, w // 2
    x = lat.reshape(ch, ht, 2, wt, 2).transpose(1, 3, 2, 4, 0)
    x = lin("x_embedder", x.reshape(ht * wt, 4 * ch))
    pad = (-len(x)) % 32
    x = np.concatenate([x, np.repeat(p["x_pad_token"], pad, 0)])
    cp = lin("cap_embedder.1", _rms(cap, p["cap_embedder.0.weight"], eps))
    cpad = (-len(cp)) % 32
    cp = np.concatenate([cp, np.repeat(p["cap_pad_token"], cpad, 0)])
    ncap = len(cp)
    cap_ids = np.stack([np.arange(1, ncap + 1), np.zeros(ncap),
                        np.zeros(ncap)], 1)
    hh, ww = np.meshgrid(np.arange(ht), np.arange(wt), indexing="ij")
    img_ids = np.stack([np.full(ht * wt, ncap + 1), hh.ravel(), ww.ravel()],
                       1)
    img_ids = np.concatenate([img_ids, np.zeros((pad, 3))])
    fx = _rope_ref(img_ids, c["axes_dims"], c["rope_theta"])
    fcap = _rope_ref(cap_ids, c["axes_dims"], c["rope_theta"])

    x = block("noise_refiner.0", x, fx, temb)
    cp = block("context_refiner.0", cp, fcap)
    u = np.concatenate([x, cp])
    fu = np.concatenate([fx, fcap])
    for i in range(c["n_layers"]):
        u = block(f"layers.{i}", u, fu, temb)
    mu = u.mean(-1, keepdims=True)
    var = ((u - mu) ** 2).mean(-1, keepdims=True)
    scale = 1 + lin("final_layer.adaLN_modulation.1", silu(temb))
    out = lin("final_layer.linear", (u - mu) / np.sqrt(var + 1e-6) * scale)
    out = out[: ht * wt].reshape(ht, wt, 2, 2, ch).transpose(4, 0, 2, 1, 3)
    return -out.reshape(ch, h, w)


@pytest.mark.parametrize("hw,n_cap", [((12, 10), 5), ((16, 16), 32)])
def test_forward_matches_the_numpy_reference(hw, n_cap):
    m, p = tiny_z_model()
    rng = np.random.default_rng(1)
    lat = rng.standard_normal((1, 4, *hw)).astype(np.float32)
    cap = rng.standard_normal((1, n_cap, 16)).astype(np.float32)
    taps: dict = {}
    # The CPU stream keeps fp32 matmuls exact. On the GPU they run at TF32,
    # whose 1e-3 error hides a position id, RoPE theta, SDPA scale or norm
    # eps mistake, each 5e-5 to 1e-2 here.
    with mx.stream(mx.cpu):
        got = m(mx.array(lat), 0.6, mx.array(cap), taps=taps)
        mx.eval(got)
    assert got.dtype == mx.float32 and got.shape == (1, 4, *hw)
    assert sorted(taps) == ["context_refiner.0", "final_layer", "layers.0",
                            "layers.1", "noise_refiner.0"]
    ref = _ref_forward(p, lat[0].astype(np.float64), 0.6,
                       cap[0].astype(np.float64))
    err = np.linalg.norm(np.array(got[0]) - ref) / np.linalg.norm(ref)
    assert err < 1e-5


def test_rope_tables_round_the_angles_to_fp32():
    """Angles are formed in float64 and rounded to fp32 before cos and sin.
    Large ids tell this apart from cos and sin taken in float64."""
    from gmlx.image import ops

    dims, theta = (32, 48, 48), 256.0
    ids = np.stack([np.arange(1000, 1536), np.arange(536) % 512,
                    np.arange(536)[::-1] % 512], 1)
    cos, sin = ops.rope_tables(ids, dims, theta)
    ang = np.concatenate(
        [np.outer(ids[:, a].astype(np.float64),
                  1.0 / theta ** (np.arange(0, d, 2, dtype=np.float64) / d))
         for a, d in enumerate(dims)], 1)
    ang32 = ang.astype(np.float32)
    assert np.array_equal(np.array(cos), np.cos(ang32))
    assert np.array_equal(np.array(sin), np.sin(ang32))
    assert not np.array_equal(np.cos(ang32), np.cos(ang).astype(np.float32))


def test_bf16_forward_stays_bf16_inside_and_returns_fp32():
    m, _ = tiny_z_model()
    m.update(tree_map(lambda a: a.astype(mx.bfloat16), m.parameters()))
    m.act_dtype = mx.bfloat16
    taps: dict = {}
    out = m(mx.zeros((1, 4, 8, 8)), 0.5, mx.zeros((1, 3, 16)), taps=taps)
    assert out.dtype == mx.float32
    assert taps["layers.1"].dtype == mx.bfloat16
    assert bool(mx.all(mx.isfinite(out)))
