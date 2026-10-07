"""The Qwen-Image 2.1 VAE on tiny synthetic files: the official header and
config fixtures, the module tree against the header, the shortcut, resample,
norm and attention semantics against numpy, the reader's strict loading and
refusals, the lossless packer round trip through the CLI, and tiled decode
and encode at 16 pixels per latent pixel with four channels. CPU."""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest
from image_fixtures import FIXTURES, load_json, write_gguf
from qwen_vae_fixtures import (FAMILY, ST_NAME, build, gguf_with_config,
                               module_config, pruned, tiny, tiny_config,
                               tiny_file)
from test_image_vae import TINY as FLUX_TINY
from test_image_vae import _np_conv, _payload_digests, tiny_tensors
from test_image_vae import write_safetensors as write_bf16

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from gmlx.image.vae import qwen_image21_vae as q
from gmlx.image.vae import reader, tiling

HEADER = "qwen_image21_vae.header.json"
CONFIG = "qwen_image21_vae_config.json"
REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"


@pytest.fixture(autouse=True)
def _on_the_cpu():
    with mx.stream(mx.cpu):
        yield


@pytest.fixture(scope="module")
def st_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("qwen_vae")
    tiny_file(d)
    (d / "config.json").write_text((FIXTURES / CONFIG).read_text())
    return d


@pytest.fixture
def st_file(st_dir):
    return str(st_dir / ST_NAME)


@pytest.fixture
def vae(st_file):
    return reader.load_vae(st_file, family=FAMILY, dtype=mx.float32)


# A tiled encode crops its tiles at the reach of a cut, which is 10 latent
# pixels of 16 pixels in the tiny VAE as in the official one. A tile is
# therefore longer than 20 latent pixels, and 336 pixels is the shortest.
# An image of 352x368 pixels is 22x23 latent pixels in 2x3 such tiles.
ENC_TILES = dict(tiled=True, tile=336, overlap=0)
ENC_SIDES = (352, 368)
ENC_LATENT = (22, 23)

# A pass does not round a value the same way at every input shape. On the
# CPU, where this module runs, a cropped head differs from the whole one by
# up to 2.9e-6 over the layouts of CROPPED and the tiny VAE at ENC_SIDES,
# on values of 1.8 rms, and the latents by 3e-7. A pixel kept one latent
# pixel within what a cut reaches is off by 0.2, and blended tiles by 0.07
# or more.
ROUNDING = 1e-4


def _header_shapes() -> dict:
    return {n: tuple(s) for n, s, _ in load_json(HEADER)["tensors"]}


def _official() -> dict:
    return {k: v for k, v in load_json(CONFIG).items()
            if not k.startswith("_")}


# Fixtures: the official header and config


def test_the_header_and_config_fixtures_are_pinned():
    manifest = load_json("manifest.json")
    head, cfg = manifest[HEADER], manifest[CONFIG]
    for row in (head, cfg):
        assert (row["repo"], row["revision"]) == ("Qwen/Qwen-Image-2.1",
                                                  REVISION)
        assert len(row["sha256"]) == 64
    assert head["file"] == "vae/diffusion_pytorch_model.safetensors"
    assert cfg["file"] == "vae/config.json"
    # The safetensors sha256 is the Hub's listing and the hash of the
    # downloaded file, and the manifest says so.
    assert head["sha256_source"].startswith("The oid that the Hub lists")
    assert "the sha256 of the downloaded file" in head["sha256_source"]
    raw = (FIXTURES / CONFIG).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == cfg["sha256"]
    d = load_json(HEADER)
    assert d["n_tensors"] == len(d["tensors"]) == 238
    assert {t[2] for t in d["tensors"]} == {"F32"}
    payload = sum(4 * int(np.prod(t[1])) for t in d["tensors"])
    assert payload == d["payload_bytes"] == 1350961616
    assert head["size"] == 8 + d["header_bytes"] + payload == 1350989512
    assert max(len(t[0]) for t in d["tensors"]) == 52


def test_the_constants_are_the_official_config_at_its_revision():
    manifest = load_json("manifest.json")[CONFIG]
    assert q.CONSTANTS_SOURCE == (manifest["repo"], manifest["revision"])
    cfg = load_json(CONFIG)
    assert cfg["_class_name"] == "AutoencoderKLQwenImage21"
    assert list(q.CONSTANTS) == ["temperal_downsample", "latents_mean",
                                 "latents_std"]
    for key, value in q.CONSTANTS.items():
        assert list(value) == cfg[key], key
    assert q.config_disagreements(cfg) == []
    assert len(cfg["latents_mean"]) == len(cfg["latents_std"]) == 64
    assert (q.LATENT_CHANNELS, q.IMAGE_CHANNELS) == (cfg["z_dim"],
                                                    cfg["in_channels"])


def test_the_shape_formula_gives_the_official_header():
    assert tiny.shapes(_official()) == _header_shapes()


def test_the_header_gives_the_official_config():
    off = _official()
    cfg = q.config_from_shapes(_header_shapes())
    assert cfg is not None
    assert (cfg.base_dim, cfg.decoder_base_dim) == (off["base_dim"],
                                                    off["decoder_base_dim"])
    assert list(cfg.dim_mult) == off["dim_mult"]
    assert cfg.num_res_blocks == off["num_res_blocks"]
    assert list(cfg.temporal) == off["temperal_downsample"]
    assert cfg.latent_channels == off["z_dim"] == 64
    assert (cfg.in_channels, cfg.out_channels) == (4, 4)
    assert cfg.scale == off["scale_factor_spatial"] == 16
    assert cfg.encoder_dims == (96, 96, 192, 384, 768, 768)
    assert cfg.decoder_dims == (1152, 1152, 1152, 576, 288, 144)
    assert (q.reach(cfg, "decode"), q.reach(cfg, "encode")) == (13, 10)


def _torch_shape(shape) -> tuple:
    return (shape[0], shape[3], shape[1], shape[2]) if len(shape) == 4 \
        else tuple(shape)


# Tiny configs with the official levels and time flags, which the reader
# takes from a file's shapes, and tiny configs with other levels or time
# flags, which only a config gives.
OFFICIAL_FLAGS = (False, True, True, True)
ALL = {"tiny": tiny.TINY, **{n: tiny.case_config(n) for n in tiny.CASES}}
OTHER_WIDTHS = {n: c for n, c in ALL.items()
                if tuple(c["temperal_downsample"]) == OFFICIAL_FLAGS}
OTHER_FLAGS = {n: c for n, c in ALL.items() if n not in OTHER_WIDTHS}


def _tree(cfg) -> dict:
    return {n: _torch_shape(p.shape) for n, p in tree_flatten(
        q.QwenImage21VAE(cfg).parameters())}


@pytest.mark.parametrize("name", sorted(OTHER_WIDTHS))
def test_the_module_tree_is_the_shape_formula_less_time_conv(name):
    config = OTHER_WIDTHS[name]
    shapes = tiny.shapes(config)
    cfg = q.config_from_shapes(shapes)
    assert cfg == module_config(config)
    unused = {n for n in shapes if ".time_conv." in n}
    assert unused == set(q.expected_unused(cfg)) and len(unused) == 12
    assert _tree(cfg) == {n: s for n, s in shapes.items()
                          if n not in unused}


@pytest.mark.parametrize("name", sorted(OTHER_FLAGS))
def test_a_config_of_its_own_gives_the_module_its_levels_and_flags(name):
    config = OTHER_FLAGS[name]
    shapes = tiny.shapes(config)
    cfg = module_config(config)
    unused = {n for n in shapes if ".time_conv." in n}
    assert unused == set(q.expected_unused(cfg))
    assert len(unused) == 4 * sum(config["temperal_downsample"])
    assert _tree(cfg) == {n: s for n, s in shapes.items()
                          if n not in unused}


def test_a_file_has_the_official_time_flags_whatever_time_conv_it_holds():
    assert sorted(OTHER_WIDTHS) == [
        "first_multiplier_2", "last_level_narrows", "one_resnet",
        "step_of_8", "three_resnets", "tiny"]
    assert sorted(OTHER_FLAGS) == [
        "all_time_flags", "no_time_flags", "other_time_flags",
        "three_levels", "two_levels"]
    for name in ("other_time_flags", "no_time_flags", "all_time_flags"):
        cfg = q.config_from_shapes(tiny.shapes(OTHER_FLAGS[name]))
        assert cfg is not None and cfg.temporal == OFFICIAL_FLAGS, name
        assert cfg == module_config(tiny.TINY)
    # Another number of levels has no official flags.
    for name in ("three_levels", "two_levels"):
        assert q.config_from_shapes(tiny.shapes(OTHER_FLAGS[name])) is None
    assert q.expected_unused(module_config(tiny.TINY)) == tuple(
        f"{at}.time_conv.{part}" for at in (
            "encoder.down_blocks.1.downsampler",
            "encoder.down_blocks.2.downsampler",
            "encoder.down_blocks.3.downsampler",
            "decoder.up_blocks.0.upsampler",
            "decoder.up_blocks.1.upsampler",
            "decoder.up_blocks.2.upsampler") for part in ("weight", "bias"))


def test_widths_that_no_shortcut_can_map_are_not_this_vae():
    def cfg(mult):
        return q.config_from_shapes(tiny.shapes(tiny_config(dim_mult=mult)))

    # 4 channels into 32 at a block that does not halve time: half a
    # channel per group.
    assert cfg([8, 8, 8, 8, 8]) is None
    # The encoder divides and the decoder does not: 48 channels up into 6
    # at the block that does not double time is half a repeat.
    assert cfg([1, 1, 8, 8, 8]) is None
    # A step of 8 where time halves on the way down is one channel per
    # group, and where time doubles on the way up one repeat.
    assert cfg([1, 8, 8, 8, 8]) is not None
    assert cfg([1, 1, 1, 8, 8]) is not None
    assert cfg([2, 2, 4, 8, 8]) is not None


def test_the_official_header_fills_the_module_tree_without_its_weights():
    # Parameters are lazy, so the tree at the official widths is compared
    # by name and shape and no weight is ever allocated.
    before = mx.get_active_memory()
    shapes = _header_shapes()
    head = reader.VaeHeader("official", "safetensors", "diffusers",
                            {n: n for n in shapes}, shapes,
                            {n: "F32" for n in shapes}, {})
    cfg = reader.identify(head, FAMILY)
    params = dict(tree_flatten(q.QwenImage21VAE(cfg).parameters()))
    reader.check_names(head, params, reader.VAES[FAMILY], cfg)
    assert len(params) == 226 and len(shapes) - len(params) == 12
    assert sum(int(np.prod(p.shape)) for p in params.values()) * 4 \
        + sum(4 * int(np.prod(s)) for n, s in shapes.items()
              if n not in params) == load_json(HEADER)["payload_bytes"]
    assert mx.get_active_memory() - before < 1 << 20


# Module semantics against numpy


def _ref_avg_down(x, cout, factor_t, factor_s):
    """AvgDown3D of the reference on one frame, (B, C, 1, H, W)."""
    pad_t = (factor_t - x.shape[2] % factor_t) % factor_t
    x = np.pad(x, ((0, 0), (0, 0), (pad_t, 0), (0, 0), (0, 0)))
    b, c, t, h, w = x.shape
    f = factor_t * factor_s * factor_s
    x = x.reshape(b, c, t // factor_t, factor_t, h // factor_s, factor_s,
                  w // factor_s, factor_s)
    x = x.transpose(0, 1, 3, 5, 7, 2, 4, 6)
    x = x.reshape(b, cout, c * f // cout, t // factor_t, h // factor_s,
                  w // factor_s)
    return x.mean(axis=2)


def _ref_dup_up(x, cout, factor_t, factor_s):
    """DupUp3D of the reference on the first chunk, (B, C, 1, H, W)."""
    b, c, t, h, w = x.shape
    f = factor_t * factor_s * factor_s
    x = np.repeat(x, cout * f // c, axis=1)
    x = x.reshape(b, cout, factor_t, factor_s, factor_s, t, h, w)
    x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)
    x = x.reshape(b, cout, t * factor_t, h * factor_s, w * factor_s)
    return x[:, :, factor_t - 1:]


def _frame(x):
    """Channels last (B, H, W, C) as one frame (B, C, 1, H, W)."""
    return x.transpose(0, 3, 1, 2)[:, :, None]


def _unframe(x):
    assert x.shape[2] == 1
    return x[:, :, 0].transpose(0, 2, 3, 1)


# (C in, C out, factor_t, factor_s) at the official ratios.
DOWN = [(4, 4, 1, 2), (4, 8, 2, 2), (8, 16, 2, 2), (16, 32, 2, 2),
        (32, 32, 1, 1)]
UP = [(8, 8, 2, 2), (8, 4, 2, 2), (4, 2, 1, 2)]


@pytest.mark.parametrize("cin,cout,ft,fs", DOWN)
def test_the_down_shortcut_is_the_reference_on_one_frame(cin, cout, ft, fs):
    rng = np.random.default_rng(cin + cout)
    x = rng.standard_normal((2, 6, 4, cin)).astype(np.float32)
    got = np.array(q.avg_down(mx.array(x), cout, ft, fs))
    want = _unframe(_ref_avg_down(_frame(x), cout, ft, fs))
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)


def test_the_down_shortcut_zeroes_even_channels_where_time_halves():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, 4, 6, 3)).astype(np.float32)
    mean = x.reshape(1, 2, 2, 3, 2, 3).mean(axis=(2, 4))
    plain = np.array(q.avg_down(mx.array(x), 3, 1, 2))
    np.testing.assert_allclose(plain, mean, rtol=1e-6, atol=1e-6)
    got = np.array(q.avg_down(mx.array(x), 6, 2, 2))
    assert not got[..., 0::2].any()
    np.testing.assert_allclose(got[..., 1::2], mean, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("cin,cout,ft,fs", UP)
def test_the_up_shortcut_is_the_reference_on_the_first_chunk(cin, cout, ft,
                                                             fs):
    rng = np.random.default_rng(cin + cout)
    x = rng.standard_normal((2, 3, 5, cin)).astype(np.float32)
    got = np.array(q.dup_up(mx.array(x), cout, ft, fs))
    want = _unframe(_ref_dup_up(_frame(x), cout, ft, fs))
    assert np.array_equal(got, want)


def test_the_up_shortcut_reads_these_channels():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((1, 2, 3, 8)).astype(np.float32)
    near = x.repeat(2, axis=1).repeat(2, axis=2)
    assert np.array_equal(np.array(q.dup_up(mx.array(x), 8, 2, 2)), near)
    assert np.array_equal(np.array(q.dup_up(mx.array(x), 4, 2, 2)),
                          near[..., 1::2])
    got = np.array(q.dup_up(mx.array(x), 4, 1, 2))
    assert np.array_equal(got[:, 0::2], near[:, 0::2, :, 0::2])
    assert np.array_equal(got[:, 1::2], near[:, 1::2, :, 1::2])


def test_the_norm_divides_by_the_l2_norm_and_scales_by_sqrt_c():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 3, 4, 12)).astype(np.float32) * 5
    x[0, 0, 0] = 0
    gamma = (1 + 0.1 * rng.standard_normal(12)).astype(np.float32)
    norm = np.sqrt((x.astype(np.float64) ** 2).sum(-1, keepdims=True))
    want = x / np.maximum(norm, 1e-12) * np.sqrt(12) * gamma
    for shape in ((12, 1, 1, 1), (12, 1, 1)):
        got = np.array(q.rms_norm(mx.array(x), mx.array(gamma.reshape(shape))))
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
        assert not got[0, 0, 0].any()
    # A bf16 stream keeps its dtype and the sums run in float32: 4096 ones
    # have the norm 64 exactly.
    x16 = mx.array(x).astype(mx.bfloat16)
    g16 = mx.array(gamma.reshape(12, 1, 1, 1)).astype(mx.bfloat16)
    got = q.rms_norm(x16, g16)
    assert got.dtype == mx.bfloat16
    ref = q.rms_norm(x16.astype(mx.float32), g16.astype(mx.float32))
    np.testing.assert_allclose(np.array(got.astype(mx.float32)),
                               np.array(ref), rtol=2.0 ** -6, atol=1e-6)
    ones = mx.ones((1, 1, 1, 4096), dtype=mx.bfloat16)
    assert mx.array_equal(
        q.rms_norm(ones, mx.ones((4096, 1, 1), dtype=mx.bfloat16)),
        ones).item()


def test_the_norm_floor_is_the_reference_one_on_the_mean_of_squares():
    c = 96
    sign = np.where(np.arange(c) % 2, 1.0, -1.0)

    def pixel(value, dtype=mx.float32):
        """The largest output of a pixel whose channels are +-value, from
        the module and from the reference's x / max(|x|, 1e-12)."""
        x = (value * sign).astype(np.float32).reshape(1, 1, 1, c)
        got = q.rms_norm(mx.array(x).astype(dtype),
                         mx.ones((c, 1, 1, 1), dtype=dtype))
        norm = np.sqrt((x.astype(np.float64) ** 2).sum())
        want = np.abs(x / max(norm, 1e-12) * np.sqrt(c)).max()
        return float(mx.abs(got.astype(mx.float32)).max()), float(want)

    # |x| is 9.8 times the channel value. Above the floor the reference
    # gives 1, and the floor under the mean of squares takes
    # 1e-24 / (2 |x|^2) from it.
    got, want = pixel(1e-11)
    assert want == pytest.approx(1.0, abs=1e-6)
    assert got == pytest.approx(1 / math.sqrt(1 + 1 / 9600), abs=1e-5)
    assert got == pytest.approx(0.99995, abs=1e-5)
    got, want = pixel(1e-12)
    assert want == pytest.approx(1.0, abs=1e-6)
    assert got == pytest.approx(1 / math.sqrt(1 + 1 / 96), abs=1e-5)
    assert got == pytest.approx(0.99483, abs=1e-5)
    # Far below the floor both divide by the floor.
    got, want = pixel(1e-15)
    assert want == pytest.approx(1e-15 * math.sqrt(c) / 1e-12, rel=1e-5)
    assert got == pytest.approx(want, rel=1e-4)
    # A pixel of zeros stays zeros.
    assert pixel(0.0) == (0.0, 0.0)
    # A bf16 stream has the same floor.
    assert pixel(1e-11, mx.bfloat16)[0] == pytest.approx(1.0, abs=0.01)
    assert pixel(0.0, mx.bfloat16)[0] == 0.0
    assert q.NORM_FLOOR == 1e-12


def test_gamma_keeps_the_shape_of_the_file(vae):
    r = vae.decoder.up_blocks[0].resnets[0]
    assert r.norm1.gamma.shape == r.norm2.gamma.shape == (48, 1, 1, 1)
    assert vae.decoder.norm_out.gamma.shape == (6, 1, 1, 1)
    assert vae.encoder.mid_block.attentions[0].norm.gamma.shape == (32, 1, 1)


def test_downsample_pads_bottom_and_right_then_strides(vae):
    t = tiny.weights()
    name = "encoder.down_blocks.1.downsampler.resample.1"
    down = vae.encoder.down_blocks[1].downsampler
    x = np.random.default_rng(4).standard_normal((1, 6, 8, 8)).astype(
        np.float32)
    got = np.array(down(mx.array(x)))
    want = _np_conv(x, t[name + ".weight"], t[name + ".bias"], stride=2,
                    pad=(0, 1, 0, 1))
    assert got.shape == (1, 3, 4, 8)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)
    assert not hasattr(down, "time_conv")


def test_upsample_is_nearest_then_conv(vae):
    t = tiny.weights()
    name = "decoder.up_blocks.3.upsampler.resample.1"
    up = vae.decoder.up_blocks[3].upsampler
    x = np.random.default_rng(5).standard_normal((1, 3, 4, 12)).astype(
        np.float32)
    got = np.array(up(mx.array(x)))
    near = x.repeat(2, axis=1).repeat(2, axis=2)
    want = _np_conv(near, t[name + ".weight"], t[name + ".bias"])
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)
    assert not hasattr(up, "time_conv")


def test_the_last_blocks_have_no_resample(vae):
    assert vae.encoder.down_blocks[4].downsampler is None
    assert vae.decoder.up_blocks[4].upsampler is None
    x = mx.array(np.random.default_rng(6).standard_normal(
        (1, 2, 2, 12)).astype(np.float32))
    blk = vae.decoder.up_blocks[4]
    h = x
    for r in blk.resnets:
        h = r(h)
    assert mx.array_equal(blk(x), h).item()


def test_mid_attention_is_one_head_over_pixels_with_a_residual(vae):
    t = tiny.weights()
    name = "decoder.mid_block.attentions.0"
    attn = vae.decoder.mid_block.attentions[0]
    c = 48
    x = np.random.default_rng(7).standard_normal((1, 3, 4, c)).astype(
        np.float32)
    got = np.array(attn(mx.array(x)))
    xf = x.astype(np.float64)
    y = xf / np.sqrt((xf ** 2).sum(-1, keepdims=True)) * np.sqrt(c) \
        * t[name + ".norm.gamma"].reshape(-1)
    qkv = y.reshape(-1, c) @ t[name + ".to_qkv.weight"].reshape(3 * c, c).T \
        + t[name + ".to_qkv.bias"]
    qq, kk, vv = qkv[:, :c], qkv[:, c:2 * c], qkv[:, 2 * c:]
    s = qq @ kk.T / np.sqrt(c)
    p = np.exp(s - s.max(-1, keepdims=True))
    o = (p / p.sum(-1, keepdims=True)) @ vv
    o = o @ t[name + ".proj.weight"].reshape(c, c).T + t[name + ".proj.bias"]
    np.testing.assert_allclose(got, x + o.reshape(x.shape), rtol=1e-4,
                               atol=1e-4)


def test_encode_adds_an_alpha_of_ones_and_takes_the_first_half(vae):
    rng = np.random.default_rng(8)
    rgb = mx.array(rng.uniform(-1, 1, (1, 32, 48, 3)).astype(np.float32))
    rgba = mx.concatenate([rgb, mx.ones((1, 32, 48, 1))], axis=-1)
    mean, logvar = vae.encode(rgb)
    assert mean.shape == logvar.shape == (1, 2, 3, 64)
    assert mx.array_equal(mean, vae.encode(rgba)[0]).item()
    full = vae.quant_conv(vae.encoder(rgba))
    assert full.shape == (1, 2, 3, 128)
    assert mx.array_equal(mean, full[..., :64]).item()
    assert mx.array_equal(logvar, mx.clip(full[..., 64:], -30, 20)).item()
    clear = mx.concatenate([rgb, -mx.ones((1, 32, 48, 1))], axis=-1)
    assert not mx.allclose(vae.encode(clear)[0], mean, atol=1e-3).item()
    with pytest.raises(ValueError, match="2 channels.*encodes 3 or 4"):
        vae.encode(rgb[..., :2])


def test_the_log_variance_is_clipped_to_minus_30_and_20(vae):
    m = np.zeros((1, 1, 2, 128), np.float32)
    m[0, 0, 0, 64:68] = (-40.0, -30.5, 20.5, 40.0)
    m[0, 0, 1, 64:68] = (-29.5, -1.0, 1.0, 19.5)
    mean, logvar = vae.split_moments(mx.array(m))
    assert mean.shape == logvar.shape == (1, 1, 2, 64)
    assert not np.array(mean).any()
    lv = np.array(logvar)
    assert lv[0, 0, 0, :4].tolist() == [-30.0, -30.0, 20.0, 20.0]
    assert lv[0, 0, 1, :4].tolist() == [-29.5, -1.0, 1.0, 19.5]

    # Through a pass: a quant_conv bias of 60 on the first 32 log variance
    # channels and of -60 on the others.
    bias = np.array(vae.quant_conv.bias)
    bias[64:96] += 60
    bias[96:] -= 60
    vae.quant_conv.bias = mx.array(bias)
    rng = np.random.default_rng(20)
    img = rng.uniform(-1, 1, (1, 64, 48, 3)).astype(np.float32)
    lv = np.array(vae.encode(mx.array(img))[1])
    assert (lv[..., :32] == 20.0).all() and (lv[..., 32:] == -30.0).all()
    # A sampled encode scales its noise by exp(log variance / 2), which is
    # exp(10) at the upper bound.
    std = np.array(vae.config.latents_std, np.float32)[:, None, None]
    big = rng.uniform(-1, 1, (1, *ENC_SIDES, 3)).astype(np.float32)
    for x, kw in ((img, {}), (big, ENC_TILES)):
        mode = np.array(tiling.encode(vae, x, free=1e12, **kw))
        got = np.array(tiling.encode(vae, x, mode="sample", free=1e12,
                                     noise=np.ones(mode.shape, np.float32),
                                     **kw))
        step = (got - mode) * std
        np.testing.assert_allclose(step[:, :32], math.exp(10), rtol=1e-4)
        assert np.abs(step[:, 32:]).max() < 1e-4


def test_decode_is_rgba_in_float32_and_clamped(vae, monkeypatch):
    z = mx.array(np.random.default_rng(9).standard_normal(
        (1, 2, 3, 64)).astype(np.float32))
    img = vae.decode(z)
    assert img.shape == (1, 32, 48, 4) and img.dtype == mx.float32
    assert float(mx.abs(img).max()) == 1.0
    monkeypatch.setattr(q, "clamp", lambda x: x.astype(mx.float32))
    raw = vae.decode(z)
    assert float(mx.abs(raw).max()) > 1.0
    assert mx.array_equal(img, mx.clip(raw, -1.0, 1.0)).item()


def test_latent_space_is_z_times_std_plus_mean(vae):
    cfg = load_json(CONFIG)
    mean = np.array(cfg["latents_mean"], np.float32)
    std = np.array(cfg["latents_std"], np.float32)
    lat = np.random.default_rng(10).standard_normal((1, 64, 2, 3)).astype(
        np.float32)
    z = vae.to_vae_space(mx.array(lat))
    assert z.shape == (1, 2, 3, 64) and z.dtype == mx.float32
    np.testing.assert_allclose(
        np.array(z), lat.transpose(0, 2, 3, 1) * std + mean, rtol=1e-6)
    back = vae.to_model_space(z)
    np.testing.assert_allclose(np.array(back), lat, atol=1e-5)
    with pytest.raises(ValueError, match="63 means and 64 deviations"):
        q.QwenImage21VAE(q.QwenImage21VAEConfig(
            4, 4, 64, 4, 6, (1, 2), 1, (False,),
            latents_mean=tuple(mean[:63])))


def test_latents_of_another_channel_count_are_refused(vae):
    for shape in ((1, 16, 2, 3), (1, 1, 2, 3), (1, 2, 3, 64), (64, 2, 3)):
        with pytest.raises(ValueError, match=r"are not \(B, 64, h, w\)"):
            vae.to_vae_space(mx.zeros(shape))
        with pytest.raises(ValueError, match=r"are not \(B, 64, h, w\)"):
            tiling.decode(vae, np.zeros(shape, np.float32), free=1e12)


# Reader


def test_the_reader_identifies_the_tiny_file_and_strips_prefixes(
        st_file, tmp_path):
    plain = reader.load_vae(st_file, family=FAMILY, dtype="fp32")
    cfg = plain.config
    assert (cfg.base_dim, cfg.decoder_base_dim, cfg.scale) == (4, 6, 16)
    assert cfg.temporal == (False, True, True, True)
    assert plain.source_format == "safetensors" and plain.family == FAMILY
    assert plain.lockstep is False
    assert (plain.channels("decode"), plain.channels("encode")) == (6, 4)
    pre = reader.load_vae(tiny_file(
        tmp_path, {"vae." + k: v for k, v in tiny.weights().items()}),
        family=FAMILY, dtype="fp32")
    a = dict(tree_flatten(plain.parameters()))
    b = dict(tree_flatten(pre.parameters()))
    assert a.keys() == b.keys() and len(a) == 226
    assert all(mx.array_equal(a[k], b[k]).item() for k in a)


def test_time_conv_tensors_are_left_unread(st_file):
    head = reader.read_header(st_file)
    unread = sorted(n for n in head.shapes if ".time_conv." in n)
    assert len(unread) == 12 and len(head.shapes) == 238
    vae = reader.load_vae(st_file, family=FAMILY, dtype="fp32")
    names = {n for n, _ in tree_flatten(vae.parameters())}
    assert names == set(head.shapes) - set(unread)
    assert not [n for n, _ in vae.named_modules() if "time_conv" in n]


def test_weights_are_cast_to_the_decode_dtype(st_file):
    v16 = reader.load_vae(st_file, family=FAMILY, dtype="bf16")
    assert {p.dtype for _, p in tree_flatten(v16.parameters())} == {
        mx.bfloat16}
    assert v16.dtype == mx.bfloat16
    z = mx.zeros((1, 1, 1, 64))
    assert v16.decode(z).dtype == mx.float32
    assert v16.decode_tail(v16.decode_head(z.astype(mx.bfloat16))).dtype \
        == mx.bfloat16
    mean, logvar = v16.encode(mx.zeros((1, 16, 16, 3)))
    assert mean.dtype == logvar.dtype == mx.bfloat16
    src = tiny.weights()["decoder.conv_out.weight"]
    got = np.array(v16.decoder.conv_out.weight.astype(mx.float32))
    want = np.array(mx.array(src.transpose(0, 2, 3, 1)).astype(mx.bfloat16)
                    .astype(mx.float32))
    assert np.array_equal(got, want) and not np.array_equal(
        got, src.transpose(0, 2, 3, 1))


def test_load_vae_puts_every_conv_on_the_chosen_path(st_file):
    from gmlx.image.vae.conv import CONV, Conv2d

    for conv, exact in ((None, CONV == "exact"), ("exact", True),
                        ("winograd", False)):
        vae = reader.load_vae(st_file, family=FAMILY, dtype=mx.float32,
                              conv=conv)
        convs = [m for _, m in vae.named_modules()
                 if isinstance(m, nn.Conv2d)]
        assert vae.conv == (conv or CONV)
        # Every conv weight of the file but the six of time_conv.
        assert len(convs) == 82 == sum(
            n.endswith(".weight") and ".time_conv." not in n
            for n in tiny.shapes(tiny.TINY))
        assert all(isinstance(m, Conv2d) and m.exact == exact
                   for m in convs)
    with pytest.raises(ValueError, match="conv must be one of"):
        reader.load_vae(st_file, family=FAMILY, conv="fast")


def test_fp16_is_refused(st_file):
    for dt in (mx.float16, "fp16", "float16"):
        with pytest.raises(reader.VAEError, match="never runs in float16"):
            reader.load_vae(st_file, family=FAMILY, dtype=dt)
    with pytest.raises(reader.VAEError, match="not bfloat16 or float32"):
        reader.load_vae(st_file, family=FAMILY, dtype=mx.int8)


def test_loading_is_strict_both_ways(tmp_path):
    t = tiny.weights()
    miss = dict(t)
    del miss["decoder.mid_block.resnets.1.norm2.gamma"]
    with pytest.raises(reader.VAEError, match="1 missing.*norm2.gamma"):
        reader.load_vae(tiny_file(tmp_path, miss, "m.safetensors"),
                        family=FAMILY, dtype=mx.float32)
    extra = dict(t)
    extra["decoder.up_blocks.0.attentions.0.proj.weight"] = np.zeros(
        (4, 4, 1, 1), np.float32)
    with pytest.raises(reader.VAEError, match="1 unexpected.*attentions"):
        reader.load_vae(tiny_file(tmp_path, extra, "e.safetensors"),
                        family=FAMILY, dtype=mx.float32)
    wrong = dict(t)
    wrong["decoder.norm_out.gamma"] = np.ones((6,), np.float32)
    with pytest.raises(reader.VAEError, match="another shape.*norm_out"):
        reader.load_vae(tiny_file(tmp_path, wrong, "w.safetensors"),
                        family=FAMILY, dtype=mx.float32)
    # Only the time_conv weights and biases of the file are left unread.
    for i, name in enumerate((
            "decoder.up_blocks.0.time_conv.weight",
            "decoder.up_blocks.0.upsampler.time_conv.gamma",
            "encoder.down_blocks.1.downsampler.time_conv.weight.1")):
        stray = dict(t)
        stray[name] = np.zeros((4, 4, 1, 1), np.float32)
        with pytest.raises(reader.VAEError,
                           match="1 unexpected.*time_conv"):
            reader.load_vae(tiny_file(tmp_path, stray, f"s{i}.safetensors"),
                            family=FAMILY, dtype=mx.float32)


# time_conv at a block where the official config halves or doubles no time:
# the first downsample, its mirror the last upsample, and the last block of
# each side, which has no resample.
STRAY = ("encoder.down_blocks.0.downsampler.time_conv.weight",
         "encoder.down_blocks.0.downsampler.time_conv.bias",
         "decoder.up_blocks.3.upsampler.time_conv.bias",
         "encoder.down_blocks.4.downsampler.time_conv.weight",
         "decoder.up_blocks.4.upsampler.time_conv.bias")


@pytest.mark.parametrize("form,held", [("all_12", 226), ("the_weights", 232),
                                       ("the_decoder_side", 232),
                                       ("one_weight", 237)])
def test_a_file_without_time_conv_loads_as_the_same_model(vae, tmp_path,
                                                          form, held):
    from gmlx.image import preflight
    from gmlx.image.vae import pack_vae

    t = pruned()[form]
    assert len(t) == held
    st = tiny_file(tmp_path, t, "pruned.safetensors")
    rep = pack_vae(st, tmp_path / "pruned.gguf", source_repo="r",
                   source_revision="v", config_json=json.dumps(tiny.TINY))
    assert (rep.family, rep.tensors, rep.lossless) == (FAMILY, held, True)
    assert preflight.classify(rep.path) == "vae"
    raw = write_gguf(tmp_path / "raw.gguf",
                     {k: (v, "F32") for k, v in t.items()})
    assert preflight.classify(raw) == "vae"
    want = dict(tree_flatten(vae.parameters()))
    z = mx.array(np.random.default_rng(19).standard_normal(
        (1, 2, 3, 64)).astype(np.float32))
    for path in (st, rep.path, raw):
        got = reader.load_vae(path, family=FAMILY, dtype=mx.float32)
        assert got.config == vae.config
        assert got.config.temporal == (False, True, True, True)
        have = dict(tree_flatten(got.parameters()))
        assert have.keys() == want.keys() and len(have) == 226
        assert all(mx.array_equal(have[k], want[k]).item() for k in have)
        assert mx.array_equal(got.decode(z), vae.decode(z)).item()


@pytest.mark.parametrize("name", STRAY)
def test_time_conv_where_no_time_flag_is_set_is_refused_by_name(
        tmp_path, name):
    from gmlx.image import preflight
    from gmlx.image.vae import pack_vae

    assert name not in q.expected_unused(module_config(tiny.TINY))
    shape = (4, 4, 1, 1) if name.endswith("weight") else (4,)
    for base in (tiny.weights(), pruned()["all_12"]):
        t = {**base, name: np.zeros(shape, np.float32)}
        st = tiny_file(tmp_path, t, "stray.safetensors")
        told = f"1 unexpected ({name})"
        with pytest.raises(reader.VAEError) as e:
            reader.load_vae(st, family=FAMILY, dtype=mx.float32)
        assert told in str(e.value)
        for family in (None, FAMILY):
            with pytest.raises(reader.VAEError) as e:
                pack_vae(st, tmp_path / "stray.gguf", source_repo="r",
                         source_revision="v", config_json="{}",
                         family=family)
            assert told in str(e.value)
        assert not (tmp_path / "stray.gguf").exists()
        raw = write_gguf(tmp_path / "raw.gguf",
                         {k: (v, "F32") for k, v in t.items()})
        with pytest.raises(reader.VAEError) as e:
            reader.load_vae(raw, family=FAMILY, dtype=mx.float32)
        assert told in str(e.value)
        # preflight gives the kind of a file, not whether it loads.
        assert preflight.classify(raw) == "vae"


def test_a_packed_config_that_differs_in_a_constant_is_refused_by_name(
        st_file, tmp_path):
    from gmlx.image.vae import pack_vae

    official = _official()
    assert q.config_disagreements(official) == []
    assert q.config_disagreements({}) == []
    assert q.config_disagreements(tiny.TINY) == []
    flags = {**tiny.TINY, "temperal_downsample": [False] * 4}
    std = {**official, "latents_std": [1.0] * 64}
    mean = {**official, "latents_mean": official["latents_mean"][:63]}
    both = {**flags, "latents_mean": None}
    cases = ((flags, "temperal_downsample"), (std, "latents_std"),
             (mean, "latents_mean"),
             (both, "temperal_downsample, latents_mean"))
    t = tiny.weights()
    for i, (config, fields) in enumerate(cases):
        assert q.config_disagreements(config) == fields.split(", ")
        told = ("differs from the Qwen-Image 2.1 autoencoder in "
                f"{fields}")
        dst = tmp_path / f"p{i}.gguf"
        with pytest.raises(reader.VAEError) as e:
            pack_vae(st_file, dst, source_repo="r", source_revision="v",
                     config_json=json.dumps(config))
        assert str(e.value) == f"{st_file}: its VAE config {told}"
        assert not dst.exists()
        gg = gguf_with_config(tmp_path / f"g{i}.gguf", t, json.dumps(config))
        with pytest.raises(reader.VAEError) as e:
            reader.load_vae(gg, family=FAMILY, dtype=mx.float32)
        assert str(e.value) == f"{gg}: its VAE config {told}"
    # A config that leaves the fields out, or no config, is no disagreement.
    for i, text in enumerate(("{}", json.dumps({"z_dim": 64}), None)):
        gg = gguf_with_config(tmp_path / f"ok{i}.gguf", t, text)
        assert reader.load_vae(gg, family=FAMILY, dtype=mx.float32
                               ).config.temporal == (False, True, True, True)
    for i, text in enumerate(("[1]", "null", "{")):
        gg = gguf_with_config(tmp_path / f"bad{i}.gguf", t, text)
        with pytest.raises(reader.VAEError, match="is not a JSON object"):
            reader.load_vae(gg, family=FAMILY, dtype=mx.float32)
    with pytest.raises(reader.VAEError, match="is not a JSON object"):
        pack_vae(st_file, tmp_path / "list.gguf", source_repo="r",
                 source_revision="v", config_json="[1]")
    # The FLUX.1 autoencoder takes its statistics from the DiT family's
    # constants, and its config text is not read.
    flux = write_bf16(tmp_path / "ae.safetensors", tiny_tensors())
    assert pack_vae(flux, tmp_path / "ae.gguf", source_repo="r",
                    source_revision="v", config_json="[1]").family \
        == "z_image"


def test_the_two_vaes_are_never_read_as_each_other(st_file, tmp_path):
    flux = write_bf16(tmp_path / "ae.safetensors", tiny_tensors())
    with pytest.raises(reader.VAEError) as e:
        reader.load_vae(flux, family=FAMILY, dtype=mx.float32)
    assert ("holds the FLUX.1 autoencoder, the z_image VAE, and "
            "qwen_image21 decodes with the Qwen-Image 2.1 autoencoder"
            ) in str(e.value)
    with pytest.raises(reader.VAEError) as e:
        reader.load_vae(st_file, family="z_image", dtype=mx.float32)
    assert ("holds the Qwen-Image 2.1 autoencoder, the qwen_image21 VAE, "
            "and z_image decodes with the FLUX.1 autoencoder") in str(e.value)
    assert q.config_from_shapes(reader.read_header(flux).shapes) is None
    from gmlx.image.vae import flux1_ae

    assert flux1_ae.config_from_shapes(
        reader.read_header(st_file).shapes) is None


def test_relatives_are_refused_by_what_they_hold(tmp_path):
    z48 = tiny_file(tmp_path, tiny.weights(tiny_config(z_dim=48)),
                    "z48.safetensors")
    with pytest.raises(reader.VAEError, match="48 latent channels, and "
                       "qwen_image21 decodes with the Qwen-Image 2.1 "
                       "autoencoder at 64"):
        reader.load_vae(z48, family=FAMILY, dtype=mx.float32)
    rgb = tiny_file(tmp_path, tiny.weights(tiny_config(
        in_channels=3, out_channels=3)), "rgb.safetensors")
    with pytest.raises(reader.VAEError, match="3 image channels in and 3 "
                       "out, and qwen_image21 decodes with the Qwen-Image "
                       "2.1 autoencoder at 4"):
        reader.load_vae(rgb, family=FAMILY, dtype=mx.float32)

    def video(t):
        return {k: (v[:, :, None] if k.endswith(".weight") else v)
                for k, v in t.items()}

    wan22 = tiny_file(tmp_path, video(tiny.weights(tiny_config(z_dim=48))),
                      "wan22.safetensors")
    for family, title in ((FAMILY, "the Qwen-Image 2.1 autoencoder"),
                          ("z_image", "the FLUX.1 autoencoder")):
        with pytest.raises(reader.VAEError, match="holds a video VAE in "
                           "the Wan 2.2 layout, 48 latent channels, and "
                           f"{family} decodes with {title}"):
            reader.load_vae(wan22, family=family, dtype=mx.float32)
    flat = {k: v for k, v in video(tiny.weights(tiny_config(z_dim=16)))
            .items() if ".down_blocks." not in k}
    flat["encoder.down_blocks.0.conv1.weight"] = np.zeros(
        (4, 4, 3, 3, 3), np.float32)
    wan21 = tiny_file(tmp_path, flat, "wan21.safetensors")
    with pytest.raises(reader.VAEError, match="holds a video VAE in the "
                       "Wan 2.1 and Qwen-Image 20B layout, 16 latent "
                       "channels"):
        reader.load_vae(wan21, family=FAMILY, dtype=mx.float32)
    # A rank 5 conv_in alone names no video VAE: the decoder's norm_out
    # must be there too.
    shapes = {k: v.shape for k, v in flat.items()}
    assert q.relative(shapes).startswith("a video VAE in the Wan 2.1")
    del shapes["decoder.norm_out.gamma"]
    assert q.relative(shapes) is None
    bare = tiny_file(tmp_path, {k: v for k, v in flat.items()
                                if k != "decoder.norm_out.gamma"},
                     "bare.safetensors")
    with pytest.raises(reader.VAEError, match="not a VAE gmlx image reads"):
        reader.load_vae(bare, family=FAMILY, dtype=mx.float32)


def test_other_files_and_families_are_refused(tmp_path, st_file):
    path = tiny_file(tmp_path, {"model.layers.0.weight": np.zeros(
        (4, 4), np.float32)}, "x.safetensors")
    with pytest.raises(reader.VAEError, match="not a VAE gmlx image reads"):
        reader.load_vae(path, family=FAMILY, dtype=mx.float32)
    with pytest.raises(reader.VAEError,
                       match="known: qwen_image21, z_image"):
        reader.load_vae(st_file, family="sd15", dtype=mx.float32)
    # A file with another number of levels has no official time flags.
    for i, config in enumerate((OTHER_FLAGS["three_levels"],
                                OTHER_FLAGS["two_levels"])):
        with pytest.raises(reader.VAEError,
                           match="not a VAE gmlx image reads"):
            reader.load_vae(tiny_file(tmp_path, tiny.weights(config),
                                      f"n{i}.safetensors"),
                            family=FAMILY, dtype=mx.float32)


def test_preflight_classifies_the_names_as_a_vae(tmp_path):
    from gmlx.image import preflight

    t = {k: (v, "F32") for k, v in tiny.weights().items()}
    assert preflight.classify(write_gguf(tmp_path / "n.gguf", t)) == "vae"


# Packer


def test_the_packer_round_trip_is_bit_for_bit_in_f32(st_dir, st_file,
                                                    tmp_path):
    import gguf

    from gmlx.image.vae import pack_vae

    text = (st_dir / "config.json").read_text()
    dst = tmp_path / "vae.gguf"
    rep = pack_vae(st_file, dst, source_repo="Qwen/Qwen-Image-2.1",
                   source_revision=REVISION, config_json=text)
    assert rep.family == FAMILY and rep.lossless and rep.tensors == 238
    assert rep.dtype == "source"
    assert rep.digests == _payload_digests(st_file)

    r = gguf.GGUFReader(str(dst))
    got = {t.name: hashlib.sha256(t.data.tobytes()).hexdigest()
           for t in r.tensors}
    assert got == rep.digests
    assert {t.tensor_type.name for t in r.tensors} == {"F32"}
    assert max(len(t.name) for t in r.tensors) == 52
    src = tiny.weights()
    for t in r.tensors:
        assert tuple(int(d) for d in reversed(t.shape)) == src[t.name].shape
    keys = {k: r.fields[k].contents() for k in r.fields
            if not k.startswith("GGUF.")}
    with open(st_file, "rb") as f:
        src_sha = hashlib.sha256(f.read()).hexdigest()
    assert keys == {
        "general.architecture": "autoencoder",
        "general.type": "vae",
        "gmlx.vae.family": FAMILY,
        "gmlx.vae.source_repo": "Qwen/Qwen-Image-2.1",
        "gmlx.vae.source_revision": REVISION,
        "gmlx.vae.source_file": ST_NAME,
        "gmlx.vae.source_sha256": src_sha,
        "gmlx.vae.layout": "torch-oihw",
        "gmlx.vae.config_json": text,
        "gmlx.packer.version": "1",
    }
    with pytest.raises(reader.VAEError, match="holds the Qwen-Image 2.1"):
        pack_vae(st_file, tmp_path / "z.gguf", source_repo="r",
                 source_revision="v", config_json="{}", family="z_image")


def test_the_packer_gives_a_refusal_once_when_the_families_share_it(
        tmp_path):
    from gmlx.image.vae import pack_vae

    kw = dict(source_repo="r", source_revision="v", config_json="{}")
    junk = tiny_file(tmp_path, {"model.layers.0.weight": np.zeros(
        (4, 4), np.float32)}, "junk.safetensors")
    with pytest.raises(reader.VAEError) as e:
        pack_vae(junk, tmp_path / "junk.gguf", **kw)
    assert str(e.value) == (f"{junk} is not a VAE gmlx image reads: 1 "
                            "tensors, first model.layers.0.weight")
    # Refusals that differ by family are all given.
    video = tiny_file(tmp_path, {
        k: (v[:, :, None] if k.endswith(".weight") else v)
        for k, v in tiny.weights(tiny_config(z_dim=48)).items()},
        "video.safetensors")
    with pytest.raises(reader.VAEError) as e:
        pack_vae(video, tmp_path / "video.gguf", **kw)
    text = str(e.value)
    assert text.startswith(f"{video} matches no image family's VAE: ")
    assert text.count("holds a video VAE in the Wan 2.2 layout") == 2
    assert "qwen_image21 decodes with" in text
    assert "z_image decodes with" in text


def test_pack_vae_on_the_command_line_picks_the_family(st_dir, st_file,
                                                       tmp_path, capsys):
    import gguf

    from gmlx.commands.cli import umbrella_main

    out = tmp_path / "vae.gguf"
    rc = umbrella_main(["image", "pack-vae", st_file, "--out", str(out),
                        "--source-repo", "Qwen/Qwen-Image-2.1",
                        "--source-revision", REVISION, "--json"])
    assert rc == 0
    rec = json.loads(capsys.readouterr().out)
    assert rec["family"] == FAMILY and rec["tensors"] == 238
    assert rec["lossless"] and rec["verified"] and rec["dtype"] == "source"
    r = gguf.GGUFReader(str(out))
    assert r.fields["gmlx.vae.config_json"].contents() == (
        st_dir / "config.json").read_text()
    assert r.fields["gmlx.vae.family"].contents() == FAMILY
    assert umbrella_main(["image", "pack-vae", st_file, "--out",
                          str(tmp_path / "z.gguf"), "--family",
                          "z_image"]) == 1
    assert "holds the Qwen-Image 2.1 autoencoder" in capsys.readouterr().err


def test_a_packed_gguf_refuses_the_same_things(st_file, tmp_path):
    from gmlx.image.vae import pack_vae

    gg = pack_vae(st_file, tmp_path / "vae.gguf", source_repo="r",
                  source_revision="v", config_json="{}").path
    head = reader.read_header(gg)
    assert head.format == "gguf" and len(head.shapes) == 238
    assert head.shapes["decoder.norm_out.gamma"] == (6, 1, 1, 1)
    assert head.shapes["decoder.mid_block.attentions.0.norm.gamma"] == (
        48, 1, 1)
    with pytest.raises(reader.VAEError, match="never runs in float16"):
        reader.load_vae(gg, family=FAMILY, dtype="fp16")
    with pytest.raises(reader.VAEError, match="holds the Qwen-Image 2.1"):
        reader.load_vae(gg, family="z_image", dtype=mx.float32)
    t = {k: (v, "F32") for k, v in tiny.weights().items()}
    extra = dict(t)
    extra["stray.weight"] = (np.zeros((2, 2), np.float32), "F32")
    with pytest.raises(reader.VAEError, match="1 unexpected.*stray"):
        reader.load_vae(write_gguf(tmp_path / "e.gguf", extra,
                                   arch="autoencoder"),
                        family=FAMILY, dtype=mx.float32)
    miss = dict(t)
    del miss["quant_conv.bias"]
    with pytest.raises(reader.VAEError, match="1 missing.*quant_conv.bias"):
        reader.load_vae(write_gguf(tmp_path / "m.gguf", miss,
                                   arch="autoencoder"),
                        family=FAMILY, dtype=mx.float32)


# Tiling


def test_the_tiled_identity_decoder_reproduces_its_input_at_16x():
    rng = np.random.default_rng(11)
    z = rng.standard_normal((2, 4, 3, 4)).astype(np.float32)

    def up16(t):
        return mx.repeat(mx.repeat(t, 16, axis=1), 16, axis=2)

    got = tiling.run_tiled(up16, mx.array(z), in_scale=1, out_scale=16,
                           tile=2, overlap=1)
    want = z.repeat(16, axis=1).repeat(16, axis=2)
    assert got.shape == (2, 64, 48, 4)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)

    x = rng.standard_normal((1, 64, 48, 4)).astype(np.float32)
    got = tiling.run_tiled(lambda t: t[:, ::16, ::16], mx.array(x),
                           in_scale=16, out_scale=1, tile=2, overlap=1)
    np.testing.assert_allclose(got, x[:, ::16, ::16], rtol=1e-6, atol=1e-6)


def test_decode_drops_the_alpha_unless_asked(vae):
    lat = np.random.default_rng(12).standard_normal((1, 64, 2, 3)).astype(
        np.float32)
    rgba = tiling.decode(vae, lat, alpha=True, free=1e12)
    rgb = tiling.decode(vae, lat, free=1e12)
    assert rgba.shape == (1, 32, 48, 4) and rgb.shape == (1, 32, 48, 3)
    assert rgba.dtype == rgb.dtype == mx.float32
    assert mx.array_equal(rgb, rgba[..., :3]).item()
    tiles = tiling.decode(vae, lat, alpha=True, tiled=True, tile=32,
                          overlap=16, free=1e12)
    assert tiles.shape == (1, 32, 48, 4)
    assert mx.array_equal(
        tiling.decode(vae, lat, tiled=True, tile=32, overlap=16, free=1e12),
        tiles[..., :3]).item()
    from gmlx.image.vae import flux1_ae

    flux = flux1_ae.Flux1AE(FLUX_TINY)
    with pytest.raises(ValueError, match="decodes no alpha channel"):
        tiling.decode(flux, np.zeros((1, 16, 2, 2), np.float32), alpha=True)


def _pointwise(vae):
    """The VAE with every 3x3 kernel cut to its center tap, so no conv mixes
    neighboring pixels. The shortcuts, the norms and the resamples stay,
    and only the mid attention sees the whole image."""
    for _, m in vae.named_modules():
        if isinstance(m, nn.Conv2d) and m.weight.shape[1] == 3:
            keep = np.zeros((1, 3, 3, 1), np.float32)
            keep[0, 1, 1, 0] = 1
            m.weight = m.weight * mx.array(keep)
    return vae


def test_blended_tiles_of_a_pointwise_vae_equal_the_whole_decode(vae):
    vae = _pointwise(vae)
    rng = np.random.default_rng(13)
    lat = rng.standard_normal((1, 64, 4, 3)).astype(np.float32)
    whole = np.array(tiling.decode(vae, lat, alpha=True, tiled=False,
                                   free=1e12))
    tiles = np.array(tiling.decode(vae, lat, alpha=True, tiled=True,
                                   tile=32, overlap=16, free=1e12))
    assert whole.shape == (1, 64, 48, 4)
    np.testing.assert_allclose(tiles, whole, rtol=1e-5, atol=1e-5)


def _two_levels():
    """Two levels, so a latent pixel is 2 pixels and a 64 pixel side holds
    more latent pixels than the convs reach."""
    return build(tiny_config(dim_mult=[2, 2], temperal_downsample=[True],
                             num_res_blocks=1))


def _beyond(reach: int) -> list:
    """Latent columns of a 32 column side in two tiles of 28, at 0 and 4,
    that are farther than ``reach`` from each cut edge of a tile that
    covers them. The first tile is cut at column 28, the second at 4."""
    assert tiling.tile_length(32, 28, 24) == 28
    assert tiling.tile_starts(32, 28, 24) == [0, 4]
    return [c for c in range(32) if (c >= 28 or c + reach < 28)
            and (c < 4 or c - reach >= 4)]


def test_tiles_equal_the_whole_pass_beyond_the_reach_of_a_cut_edge():
    vae = _two_levels()
    cfg = vae.config
    assert cfg.scale == 2 and cfg.temporal == (True,)
    reach = q.reach(cfg, "decode")
    assert reach == 7
    rng = np.random.default_rng(14)
    lat = rng.standard_normal((1, 64, 4, 32)).astype(np.float32)
    whole = np.array(tiling.decode(vae, lat, alpha=True, tiled=False,
                                   free=1e12))
    tiles = np.array(tiling.decode(vae, lat, alpha=True, tiled=True,
                                   tile=56, overlap=48, free=1e12))
    assert whole.shape == tiles.shape == (1, 8, 64, 4)
    far = _beyond(reach)
    assert far == [*range(4), *range(11, 21), *range(28, 32)]
    px = np.repeat(np.isin(np.arange(32), far), 2)
    err = np.abs(tiles - whole).max(axis=(0, 1, 3))
    print(f"[tiling] decode: max abs {err[px].max():.3e} beyond the reach "
          f"of a cut edge, {err[~px].max():.3e} within it")
    assert err[px].max() <= 1e-5
    assert 1e-3 < err[~px].max() <= 2.0

    # The encoder's mid attention mixes every latent pixel, so the part
    # run in tiles is compared: the output of the down blocks, blended.
    reach = q.reach(cfg, "encode")
    assert reach == 5
    img = rng.uniform(-1, 1, (1, 8, 64, 3)).astype(np.float32)
    head = np.array(vae.encode_head(mx.array(img)))
    cut = tiling.run_tiled(vae.encode_head, mx.array(img), in_scale=2,
                           out_scale=1, tile=28, overlap=24)
    far = np.isin(np.arange(32), _beyond(reach))
    assert far.sum() == 22
    err = np.abs(cut - head).max(axis=(0, 1, 3))
    print(f"[tiling] encode head: max abs {err[far].max():.3e} beyond the "
          f"reach of a cut edge, {err[~far].max():.3e} within it")
    assert err[far].max() <= 1e-5 and err[~far].max() > 1e-3
    # An encode keeps of each tile the columns beyond the reach and no
    # other, so it is the whole encode.
    a = np.array(tiling.encode(vae, img, tiled=False, free=1e12))
    b = np.array(tiling.encode(vae, img, tiled=True, tile=56, overlap=48,
                               free=1e12))
    assert a.shape == b.shape == (1, 64, 4, 32)
    assert np.abs(b - a).max() <= ROUNDING


def test_a_tiled_encode_runs_the_head_in_tiles_cropped_at_the_reach():
    vae = _two_levels()
    img = np.random.default_rng(21).uniform(-1, 1, (1, 8, 64, 3)).astype(
        np.float32)
    # 56 pixel tiles with 48 pixels of overlap are 28 latent pixels with
    # 24, which is more than twice the reach and stands. The rest of the
    # encoder runs whole on the cropped head.
    assert tiling.encode_layout(vae, 56, 48) == (28, 24, 5)
    tiles = dict(in_scale=2, out_scale=1, tile=28, overlap=24)
    head = tiling.run_tiled(vae.encode_head, mx.array(img), reach=5, **tiles)
    mean, _ = vae.split_moments(vae.encode_tail(mx.array(head)))
    want = np.array(vae.to_model_space(mean))
    got = np.array(tiling.encode(vae, img, tiled=True, tile=56, overlap=48,
                                 free=1e12))
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-6)
    # The same tiles blended differ from it within the reach of the cut
    # edges, and so does the encode of that head.
    blend = tiling.run_tiled(vae.encode_head, mx.array(img), **tiles)
    assert np.abs(blend - head).max() > 1e-3
    mean, _ = vae.split_moments(vae.encode_tail(mx.array(blend)))
    assert np.abs(np.array(vae.to_model_space(mean)) - got).max() > 1e-4


def _window_sum(reach: int):
    """A stand-in for the pixel-space part of an encode at 2 pixels a
    latent pixel, in integer arithmetic that is the same at every input
    shape and on every device: a latent pixel is the sum of its own 2x2
    pixels and of the ``reach`` latent pixels around them on every side,
    with zeros beyond the edge of the input."""
    def fn(x):
        a = np.array(x).astype(np.int64)
        _, h, w, _ = a.shape
        p = 2 * reach
        a = np.pad(a, ((0, 0), (p, p), (p, p), (0, 0)))
        c = np.pad(a.cumsum(1).cumsum(2), ((0, 0), (1, 0), (1, 0), (0, 0)))
        n = 2 * p + 2           # the side of a window in pixels
        i, j = np.arange(0, h, 2)[:, None], np.arange(0, w, 2)[None, :]
        out = (c[:, i + n, j + n] - c[:, i, j + n] - c[:, i + n, j]
               + c[:, i, j])
        return mx.array(out.astype(np.float32))
    return fn


# (pixels high, pixels wide, tile in pixels) at 2 pixels a latent pixel and
# a reach of 5 latent pixels. Every layout has neighbors that overlap by
# 10 or 11 latent pixels, so a kept pixel is exactly 5 from a cut edge.
CROPPED = [(64, 96, 32), (32, 200, 32), (72, 40, 32), (64, 96, 44),
           (26, 130, 24), (64, 96, 48), (64, 64, 36)]


def _layout(height: int, width: int, t: int, lap: int) -> tuple:
    """Tile starts along the rows and the columns of an image at 2 pixels
    a latent pixel, for tiles of ``t`` latent pixels."""
    return tuple(tiling.tile_starts(n // 2, tiling.tile_length(n // 2, t, lap),
                                    lap) for n in (height, width))


@pytest.mark.parametrize("height,width,tile", CROPPED)
def test_cropped_tiles_of_exact_arithmetic_are_the_whole_pass_bit_for_bit(
        height, width, tile):
    reach = 5
    ys, xs = _layout(height, width, tile // 2, 2 * reach)
    # More than one tile, so there is a cut to be wrong at.
    assert len(ys) * len(xs) > 1
    rng = np.random.default_rng(43)
    x = mx.array(rng.integers(1, 10, (1, height, width, 3)).astype(
        np.float32))
    tiles = dict(in_scale=2, out_scale=1, tile=tile // 2, overlap=2 * reach)
    fn = _window_sum(reach)
    whole = np.array(fn(x))
    cut = tiling.run_tiled(fn, x, reach=reach, **tiles)
    assert cut.dtype == np.float32 and cut.shape == whole.shape
    assert np.array_equal(cut, whole)
    # A function that reads one latent pixel farther reads past a cut at
    # a pixel this crop keeps.
    wide = _window_sum(reach + 1)
    assert not np.array_equal(tiling.run_tiled(wide, x, reach=reach, **tiles),
                              np.array(wide(x)))
    # The same tiles blended are not the whole pass.
    assert not np.array_equal(tiling.run_tiled(fn, x, **tiles), whole)


@pytest.mark.parametrize("height,width,tile", CROPPED)
def test_a_tiled_encode_is_the_whole_encode(height, width, tile):
    """To the rounding of the device. A conv does not round a value the
    same way at every input shape, so a tile and the whole pass can differ
    in the last bits where no cut reaches."""
    vae = _two_levels()
    assert vae.config.scale == 2 and q.reach(vae.config, "encode") == 5
    rng = np.random.default_rng(41)
    img = rng.uniform(-1, 1, (1, height, width, 3)).astype(np.float32)
    # The caller asks for no overlap, and the encode takes twice the reach.
    t, lap, r = tiling.encode_layout(vae, tile, 0)
    assert (t, lap, r) == (tile // 2, 10, 5)
    ys, xs = _layout(height, width, t, lap)
    assert len(ys) * len(xs) > 1
    tiles = dict(in_scale=2, out_scale=1, tile=t, overlap=lap)
    head = np.array(vae.encode_head(mx.array(img)))
    cut = tiling.run_tiled(vae.encode_head, mx.array(img), reach=r, **tiles)
    blend = tiling.run_tiled(vae.encode_head, mx.array(img), **tiles)
    whole = np.array(tiling.encode(vae, img, tiled=False, free=1e12))
    got = np.array(tiling.encode(vae, img, tiled=True, tile=tile, overlap=0,
                                 free=1e12))
    print(f"[tiling] encode {height}x{width}, tile {tile}: {len(ys)} x "
          f"{len(xs)} tiles, head max abs {np.abs(cut - head).max():.1e} "
          f"cropped and {np.abs(blend - head).max():.1e} blended, latents "
          f"max abs {np.abs(got - whole).max():.1e}")
    assert np.abs(cut - head).max() <= ROUNDING
    assert got.shape == whole.shape
    assert np.abs(got - whole).max() <= ROUNDING
    # The same tiles blended over the same overlap are not the whole head.
    assert np.abs(blend - head).max() > 0.05


def test_cropped_tiles_keep_twice_the_reach_at_every_size():
    for tile, reach in ((32, 10), (48, 10), (16, 5), (22, 5), (32, 13)):
        lap = 2 * reach
        for size in range(1, 513):
            t = tiling.tile_length(size, tile, lap)
            starts = tiling.tile_starts(size, t, lap)
            assert starts[0] == 0 and starts[-1] + t == size
            rows = np.zeros(size, int)
            for i, s0 in enumerate(starts):
                k = tiling.kept(starts, i, t, size, 1, reach)
                rows[s0 + k.start:s0 + k.stop] += 1
                # A kept pixel is at least the reach from a cut edge.
                assert i == 0 or k.start >= reach
                assert i + 1 == len(starts) or t - k.stop >= reach
            # The kept pixels of all tiles are every pixel once.
            assert (rows == 1).all(), (tile, reach, size)
    # The layout of the official file: 16, 49 and 81 tiles.
    for px, n, t in ((1024, 4, 31), (1536, 7, 31), (2048, 9, 32)):
        length = tiling.tile_length(px // 16, 32, 20)
        assert (len(tiling.tile_starts(px // 16, length, 20)),
                length) == (n, t)


def test_the_price_of_a_tiled_encode_lays_out_the_tiles_that_run(vae):
    """The pass takes its tiles from ``encode_layout`` and the price takes
    its overlap from ``memfit.tile_overlap``. Both give the overlap of the
    crop, whatever overlap the caller names below it."""
    from gmlx.image import memfit

    t, lap, r = tiling.encode_layout(vae, memfit.TILE_PX, memfit.OVERLAP_PX)
    assert (t, lap, r) == (32, 20, 10)
    assert memfit.tile_overlap(FAMILY, "encode", config=vae.config) == 320
    for px in range(528, 2049, 16):
        length = tiling.tile_length(px // 16, t, lap)
        n = len(tiling.tile_starts(px // 16, length, lap))
        assert memfit.tile_layout(px, 512, 320, 16) == (n, length * 16), px
    kw = dict(family=FAMILY, itemsize=2, tiled=True)
    price = memfit.vae_peak("encode", 1024, 1024, **kw)
    assert memfit.vae_peak("encode", 1024, 1024, overlap=0, **kw) == price
    assert memfit.vae_peak("encode", 1024, 1024, overlap=320, **kw) == price


def test_a_crop_refuses_what_cannot_be_exact():
    vae = _two_levels()
    # An overlap under twice the reach.
    with pytest.raises(ValueError, match="overlap by 9 latent pixels, and "
                       "a crop at a reach of 5 needs 10"):
        tiling.kept([0, 11], 0, 20, 31, 1, 5)
    # A tile no longer than twice the reach keeps no exact pixel.
    with pytest.raises(ValueError, match="a tile of 20 pixels is 10 latent "
                       "pixels, and a tiled encode of this VAE needs more "
                       "than 10, which is 22 pixels"):
        tiling.encode_layout(vae, 20, 0)
    img = np.zeros((1, 64, 96, 3), np.float32)
    # The price of the pass refuses it first, in the same words.
    with pytest.raises(ValueError, match="a tiled encode of this VAE needs "
                       "more than 10, which is 22 pixels"):
        tiling.encode(vae, img, tiled=True, tile=20, overlap=0, free=1e12)
    # An image of one tile runs whole, whatever the tile is.
    small = np.zeros((1, 16, 20, 3), np.float32)
    assert tiling.encode(vae, small, tiled=True, tile=20, overlap=0,
                         free=1e12).shape == (1, 64, 8, 10)
    # A forced whole pass has no tiles, so it takes any tile.
    assert tiling.encode(vae, img, tiled=False, tile=20, overlap=0,
                         free=1e12).shape == (1, 64, 32, 48)
    # A caller's overlap above twice the reach stands.
    assert tiling.encode_layout(vae, 44, 30) == (22, 15, 5)
    # The official file: 336 pixels is the shortest tile of an encode, and
    # a decode, which blends, takes any.
    from gmlx.image import memfit

    assert memfit.croppable(FAMILY, "encode", 336)
    assert not memfit.croppable(FAMILY, "encode", 335)
    assert memfit.croppable(FAMILY, "decode", 16)
    assert memfit.tile_overlap(FAMILY, "encode") == 320
    assert memfit.tile_overlap(FAMILY, "decode") == memfit.OVERLAP_PX
    assert tiling.encode_layout(vae, 44, 4) == (22, 10, 5)


def test_tiled_decode_and_encode_run_on_the_tiny_vae(vae, monkeypatch):
    rng = np.random.default_rng(15)
    lat = rng.standard_normal((1, 64, 4, 3)).astype(np.float32)
    whole = tiling.decode(vae, lat, free=1e12)
    tiles = tiling.decode(vae, lat, tiled=True, tile=32, overlap=16,
                          free=1e12)
    assert tiles.shape == whole.shape == (1, 64, 48, 3)
    assert float(mx.abs(whole).max()) == float(mx.abs(tiles).max()) == 1.0
    # The clamp follows the blend.
    monkeypatch.setattr(q, "clamp", lambda x: x.astype(mx.float32))
    raw = tiling.decode(vae, lat, tiled=True, tile=32, overlap=16,
                        free=1e12)
    assert float(mx.abs(raw).max()) > 1.0
    assert mx.array_equal(tiles, mx.clip(raw, -1.0, 1.0)).item()
    monkeypatch.undo()
    img = rng.uniform(-1, 1, (1, *ENC_SIDES, 3)).astype(np.float32)
    a = tiling.encode(vae, img, tiled=False, free=1e12)
    b = tiling.encode(vae, img, free=1e12, **ENC_TILES)
    assert a.shape == b.shape == (1, 64, *ENC_LATENT)
    assert a.dtype == b.dtype == mx.float32
    # The tiles are cropped, so they are the whole encode.
    assert float(mx.abs(a - b).max()) <= ROUNDING
    noise = rng.standard_normal(a.shape).astype(np.float32)
    c = tiling.encode(vae, img, mode="sample", noise=noise, tiled=False,
                      free=1e12)
    assert not mx.array_equal(a, c).item()


def test_a_bf16_vae_tiles_in_bf16_and_returns_float32(st_file):
    vae = reader.load_vae(st_file, family=FAMILY, dtype="bf16")
    ref = reader.load_vae(st_file, family=FAMILY, dtype="fp32")
    rng = np.random.default_rng(18)
    lat = rng.standard_normal((1, 64, 4, 3)).astype(np.float32)
    kw = dict(tiled=True, tile=32, overlap=16, free=1e12)
    got = tiling.decode(vae, lat, alpha=True, **kw)
    want = tiling.decode(ref, lat, alpha=True, **kw)
    assert got.shape == (1, 64, 48, 4) and got.dtype == mx.float32
    assert float(mx.abs(got).max()) == 1.0
    err = float(mx.abs(got - want).mean())
    assert 0.0 < err < 0.02
    img = rng.uniform(-1, 1, (1, *ENC_SIDES, 3)).astype(np.float32)
    a = tiling.encode(vae, img, free=1e12, **ENC_TILES)
    b = tiling.encode(ref, img, free=1e12, **ENC_TILES)
    assert a.shape == (1, 64, *ENC_LATENT) and a.dtype == mx.float32
    rel = float(mx.sqrt(((a - b) ** 2).sum() / (b ** 2).sum()))
    print(f"[tiling] bf16 against fp32, tiled: decode mean abs {err:.3e}, "
          f"encode rel L2 {rel:.3e}")
    assert 0.0 < rel < 0.05
    with pytest.raises(ValueError, match="not a multiple of 16"):
        tiling.encode(vae, img[:, :60], free=1e12)


def _dtypes_inside(monkeypatch) -> list:
    """Records the dtype that each norm and each conv of a pass reads."""
    from gmlx.image.vae.conv import Conv2d

    seen = []
    norm, conv = q.rms_norm, Conv2d.__call__

    def spy_norm(x, gamma):
        seen.append(("norm", x.dtype))
        return norm(x, gamma)

    def spy_conv(self, x):
        seen.append(("conv", x.dtype))
        return conv(self, x)

    monkeypatch.setattr(q, "rms_norm", spy_norm)
    monkeypatch.setattr(Conv2d, "__call__", spy_conv)
    return seen


def _ops(*modules) -> int:
    """Norms and convs in ``modules``."""
    return sum(isinstance(m, (q.RMSNorm, nn.Conv2d))
               for mod in modules for _, m in mod.named_modules())


@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
def test_every_norm_and_conv_of_a_pass_reads_the_vae_dtype(st_file, dtype,
                                                           monkeypatch):
    vae = reader.load_vae(st_file, family=FAMILY, dtype=dtype)
    want = {"bf16": mx.bfloat16, "fp32": mx.float32}[dtype]
    assert vae.dtype == want
    dec = _ops(vae.decoder, vae.post_quant_conv)
    enc = _ops(vae.encoder, vae.quant_conv)
    assert (dec, enc) == (82, 62)
    seen = _dtypes_inside(monkeypatch)
    rng = np.random.default_rng(22)
    lat = rng.standard_normal((1, 64, 4, 3)).astype(np.float32)
    img = rng.uniform(-1, 1, (1, 64, 48, 3)).astype(np.float32)
    big = rng.uniform(-1, 1, (1, *ENC_SIDES, 3)).astype(np.float32)
    noise = np.zeros((1, 64, *ENC_LATENT), np.float32)
    tiles = dict(tiled=True, tile=32, overlap=16, free=1e12)
    cut = dict(free=1e12, **ENC_TILES)
    # (the pass, its ops when whole, whether the part in tiles runs more
    # than once, the dtype of the result)
    f32 = mx.float32
    passes = [
        (lambda: vae.decode(vae.to_vae_space(mx.array(lat))), dec, False,
         f32),
        (lambda: tiling.decode(vae, lat, tiled=False, free=1e12), dec, False,
         f32),
        (lambda: tiling.decode(vae, lat, **tiles), dec, True, f32),
        (lambda: vae.encode(mx.array(img))[0], enc, False, want),
        (lambda: tiling.encode(vae, img, tiled=False, free=1e12), enc, False,
         f32),
        (lambda: tiling.encode(vae, big, **cut), enc, True, f32),
        (lambda: tiling.encode(vae, big, mode="sample", noise=noise, **cut),
         enc, True, f32),
    ]
    for run, ops, in_tiles, result in passes:
        seen.clear()
        out = run()
        mx.eval(out)
        assert out.dtype == result
        assert {d for _, d in seen} == {want}
        assert len(seen) > ops if in_tiles else len(seen) == ops
        assert {k for k, _ in seen} == {"norm", "conv"}


def _pipe(vae, title: str):
    from types import SimpleNamespace

    return SimpleNamespace(vae=vae, family=SimpleNamespace(title=title))


def test_the_pipeline_entry_points_serve_this_vae(vae, monkeypatch):
    from gmlx.image import api, memfit

    monkeypatch.setattr(memfit, "free_bytes", lambda: 1e12)
    pipe = _pipe(vae, "Qwen-Image 2.1")
    rng = np.random.default_rng(16)
    lat = rng.standard_normal((1, 64, 2, 3)).astype(np.float32)
    rgb = api.vae_decode(pipe, lat)
    assert rgb.shape == (1, 32, 48, 3) and rgb.dtype == mx.float32
    raw = tiling.decode(vae, lat, alpha=True, free=1e12)
    assert mx.array_equal(rgb, raw[..., :3] / 2 + 0.5).item()
    assert float(rgb.min()) >= 0.0 and float(rgb.max()) == 1.0

    back = api.vae_encode(pipe, rgb)
    assert back.shape == (1, 64, 2, 3) and back.dtype == mx.float32
    assert mx.array_equal(
        back, tiling.encode(vae, rgb * 2 - 1, free=1e12)).item()
    opaque = mx.concatenate([rgb, mx.ones((1, 32, 48, 1))], axis=-1)
    assert mx.array_equal(api.vae_encode(pipe, opaque), back).item()


def test_a_pass_is_priced_by_the_family_row_and_the_config_of_the_vae(
        vae, st_file, monkeypatch):
    from gmlx.image import memfit
    from gmlx.image.vae import flux1_ae

    seen = []

    def spy(op, height, width, **kw):
        seen.append((op, height, width, kw["family"], kw["config"],
                     kw["channels"], kw["itemsize"], kw["conv"]))
        return False

    monkeypatch.setattr(memfit, "vae_tiling", spy)
    tiling.decode(vae, np.zeros((1, 64, 2, 3), np.float32))
    tiling.encode(vae, np.zeros((1, 32, 48, 3), np.float32))
    assert seen == [("decode", 32, 48, FAMILY, vae.config, 6, 4, "exact"),
                    ("encode", 32, 48, FAMILY, vae.config, 4, 4, "exact")]
    other = reader.load_vae(st_file, family=FAMILY, dtype="bf16",
                            conv="winograd")
    tiling.decode(other, np.zeros((1, 64, 2, 3), np.float32))
    assert seen[-1][-2:] == (2, "winograd")
    # A VAE built from a config was loaded for no family, and takes the
    # row of the family that decodes with its class.
    assert tiling.row_family(build(tiny_config())) == FAMILY
    assert tiling.row_family(flux1_ae.Flux1AE(FLUX_TINY)) == "z_image"
    with pytest.raises(ValueError, match="no image family decodes with"):
        tiling.row_family(object())


def test_above_one_tile_the_automatic_choice_is_tiles_whatever_is_free(
        vae, monkeypatch):
    """The row of this VAE takes tiles first. A whole pass above one tile
    runs only when it is forced, at the price of its fit, and is refused
    one byte under it."""
    from gmlx.image import memfit
    from gmlx.image.memfit import GiB, ImageMemoryError

    row = memfit.VAE_ROWS[FAMILY]
    assert row.measured is True and row.tiles_first is True
    monkeypatch.delenv("GMLX_OVERCOMMIT", raising=False)
    rng = np.random.default_rng(31)
    lat = rng.standard_normal((1, 64, 4, 3)).astype(np.float32)
    img = rng.uniform(-1, 1, (1, *ENC_SIDES, 3)).astype(np.float32)
    # A cropped encode is the whole encode, so the arrays do not say which
    # pass ran. The calls of run_tiled do.
    calls, real = [], tiling.run_tiled
    monkeypatch.setattr(tiling, "run_tiled",
                        lambda *a, **kw: calls.append(1) or real(*a, **kw))

    def ran(run, x, in_tiles: bool, want=None, **kw):
        calls.clear()
        out = run(vae, x, **kw)
        assert bool(calls) is in_tiles
        assert want is None or mx.array_equal(out, want).item()
        return out

    for run, x, op, (high, wide), cut in (
            (tiling.decode, lat, "decode", (64, 48),
             dict(tile=32, overlap=16)),
            (tiling.encode, img, "encode", ENC_SIDES,
             dict(tile=ENC_TILES["tile"], overlap=0))):
        whole = ran(run, x, False, tiled=False, free=1e12, **cut)
        tiles = ran(run, x, True, tiled=True, free=1e12, **cut)
        assert whole.shape == tiles.shape
        kw = dict(tiling.price_args(vae, op, 1), **cut)
        need_whole = memfit.vae_peak(op, high, wide, **kw)
        need_tiles = memfit.vae_peak(op, high, wide, tiled=True, **kw)
        # Room for either pass, room for the tiles alone, and no figure.
        for free in (1e12, need_tiles):
            ran(run, x, True, tiles, free=free, **cut)
        with monkeypatch.context() as mp:
            mp.setattr(memfit, "free_bytes", lambda: None)
            ran(run, x, True, tiles, **cut)
            ran(run, x, False, whole, tiled=False, **cut)
        # An image no larger than one tile runs whole.
        ran(run, x, False, whole, free=1e12)
        ran(run, x, False, whole, tiled=False, free=need_whole, **cut)
        with pytest.raises(ImageMemoryError) as e:
            run(vae, x, tiled=False, free=need_whole - 1, **cut)
        msg = str(e.value)
        assert (f"VAE {op} of 1 x {high}x{wide} whole needs "
                f"{need_whole / GiB:.2f} GiB") in msg
        assert msg.endswith("GMLX_OVERCOMMIT=1 overrides.")
        with pytest.raises(ImageMemoryError, match="in tiles needs"):
            run(vae, x, free=need_tiles - 1, **cut)
        with monkeypatch.context() as mp:
            mp.setenv("GMLX_OVERCOMMIT", "1")
            ran(run, x, False, whole, tiled=False, free=0, **cut)


def test_the_pipeline_decode_keeps_the_alpha_when_asked(vae, monkeypatch):
    from gmlx.image import api, memfit
    from gmlx.image.vae import flux1_ae

    monkeypatch.setattr(memfit, "free_bytes", lambda: 1e12)
    pipe = _pipe(vae, "Qwen-Image 2.1")
    lat = np.random.default_rng(17).standard_normal((1, 64, 2, 3)).astype(
        np.float32)
    rgba = api.vae_decode(pipe, lat, alpha=True)
    assert rgba.shape == (1, 32, 48, 4) and rgba.dtype == mx.float32
    raw = tiling.decode(vae, lat, alpha=True, free=1e12)
    assert mx.array_equal(rgba, raw / 2 + 0.5).item()
    assert mx.array_equal(rgba[..., :3], api.vae_decode(pipe, lat)).item()
    flux = _pipe(flux1_ae.Flux1AE(FLUX_TINY), "Z-Image")
    with pytest.raises(ValueError, match="Z-Image VAE has no alpha channel"):
        api.vae_decode(flux, np.zeros((1, 16, 2, 2), np.float32),
                       alpha=True)
