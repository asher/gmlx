"""The FLUX.1 VAE on a tiny synthetic file: contiguous GroupNorm, the conv
layout transform and the resample and attention semantics against numpy,
the original-layout name map, strict loading, refusals, the lossless packer
round trip and tiled decode and encode. CPU."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import struct

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from gmlx.image.vae import flux1_ae, reader, tiling
from gmlx.image.vae.group_norm import GroupNorm, group_norm

# A Mac of 16 GB, as most users run (tests/host_box.py).
pytestmark = pytest.mark.host_gb(16)

TINY = flux1_ae.Flux1AEConfig(
    in_channels=3, out_channels=3, latent_channels=16,
    block_out_channels=(32, 64), layers_per_block=1, norm_num_groups=32,
    eps=1e-6)
CONFIG_JSON = json.dumps({"_class_name": "AutoencoderKL",
                          "block_out_channels": [32, 64],
                          "latent_channels": 16})


def _bf16(x: np.ndarray) -> np.ndarray:
    """float32 values rounded down to bf16, so bf16 holds them exactly."""
    u = np.asarray(x, np.float32).view(np.uint32) & np.uint32(0xFFFF0000)
    return u.view(np.float32)


def _torch_shape(shape):
    return (shape[0], shape[3], shape[1], shape[2]) if len(shape) == 4 \
        else tuple(shape)


def tiny_tensors(seed: int = 0) -> dict:
    """Diffusers names to float32 torch-layout values that bf16 holds."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, p in tree_flatten(flux1_ae.Flux1AE(TINY).parameters()):
        shape = _torch_shape(p.shape)
        fan = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        v = rng.standard_normal(shape) / np.sqrt(fan)
        if name.endswith("norm1.weight") or name.endswith("norm2.weight") \
                or "norm.weight" in name or "norm_out.weight" in name:
            v = 1.0 + 0.1 * rng.standard_normal(shape)
        out[name] = _bf16(v)
    return out


def to_original(name: str, levels: int) -> str:
    """The original FLUX.1 name of a diffusers name, written out apart from
    the reader's map."""
    swaps = [
        ("mid_block.resnets.0.", "mid.block_1."),
        ("mid_block.resnets.1.", "mid.block_2."),
        ("mid_block.attentions.0.group_norm.", "mid.attn_1.norm."),
        ("mid_block.attentions.0.to_q.", "mid.attn_1.q."),
        ("mid_block.attentions.0.to_k.", "mid.attn_1.k."),
        ("mid_block.attentions.0.to_v.", "mid.attn_1.v."),
        ("mid_block.attentions.0.to_out.0.", "mid.attn_1.proj_out."),
        (".resnets.", ".block."),
        ("conv_shortcut.", "nin_shortcut."),
        ("downsamplers.0.", "downsample."),
        ("upsamplers.0.", "upsample."),
        ("conv_norm_out.", "norm_out."),
        ("encoder.down_blocks.", "encoder.down."),
    ]
    for old, new in swaps:
        name = name.replace(old, new)
    if name.startswith("decoder.up_blocks."):
        k, rest = name[len("decoder.up_blocks."):].split(".", 1)
        name = f"decoder.up.{levels - 1 - int(k)}.{rest}"
    return name


def write_safetensors(path, tensors: dict, dtype: str = "BF16") -> str:
    header, blobs, off = {"__metadata__": {"format": "pt"}}, [], 0
    for name, v in tensors.items():
        v = np.ascontiguousarray(v, np.float32)
        if dtype == "BF16":
            raw = (v.view(np.uint32) >> 16).astype(np.uint16).tobytes()
        elif dtype == "F16":
            raw = v.astype(np.float16).tobytes()
        else:
            raw = v.tobytes()
        header[name] = {"dtype": dtype, "shape": list(v.shape),
                        "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))
    return str(path)


def _weights(vae) -> dict:
    return {k: np.array(v.astype(mx.float32))
            for k, v in tree_flatten(vae.parameters())}


@pytest.fixture
def st_file(tmp_path):
    return write_safetensors(tmp_path / "ae.safetensors", tiny_tensors())


# GroupNorm


def _np_group_norm(x, w, b, groups, eps):
    x = np.asarray(x, np.float64)
    bsz, c = x.shape[0], x.shape[-1]
    g = x.reshape(bsz, -1, groups, c // groups)
    mean = g.mean(axis=(1, 3), keepdims=True)
    var = g.var(axis=(1, 3), keepdims=True)
    y = ((g - mean) / np.sqrt(var + eps)).reshape(x.shape)
    return y * w + b


def test_group_norm_groups_contiguous_channels_and_keeps_the_dtype():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 5, 7, 64)).astype(np.float32)
    x[..., :32] *= 4.0
    x[..., 32:] += 3.0
    w = (1 + 0.1 * rng.standard_normal(64)).astype(np.float32)
    b = (0.1 * rng.standard_normal(64)).astype(np.float32)
    want = _np_group_norm(x, w, b, 8, 1e-6)

    got32 = group_norm(mx.array(x), mx.array(w), mx.array(b), 8, 1e-6)
    assert got32.dtype == mx.float32
    np.testing.assert_allclose(np.array(got32), want, rtol=1e-4, atol=1e-5)

    norm = GroupNorm(8, 64, 1e-6)
    norm.weight, norm.bias = mx.array(w), mx.array(b)
    got16 = norm(mx.array(x).astype(mx.bfloat16))
    assert got16.dtype == mx.bfloat16
    err = np.abs(np.array(got16.astype(mx.float32)) - want).max()
    assert err < 0.05

    stock = nn.GroupNorm(8, 64, eps=1e-6)
    stock.weight, stock.bias = mx.array(w), mx.array(b)
    off = np.abs(np.array(stock(mx.array(x))) - want).max()
    assert off > 0.5, "the stock default groups interleaved channels"


def test_group_norm_refuses_channels_off_the_group_count():
    with pytest.raises(ValueError, match="do not split"):
        group_norm(mx.zeros((1, 2, 2, 30)), None, None, 32, 1e-6)


# Module semantics against numpy


def _np_conv(x, w, b, stride=1, pad=(1, 1, 1, 1)):
    """NHWC cross-correlation with torch weights (O, I, kH, kW). ``pad`` is
    (top, bottom, left, right)."""
    x = np.pad(np.asarray(x, np.float64),
               [(0, 0), pad[:2], pad[2:], (0, 0)])
    o, i, kh, kw = w.shape
    hh = (x.shape[1] - kh) // stride + 1
    ww = (x.shape[2] - kw) // stride + 1
    out = np.zeros((x.shape[0], hh, ww, o))
    for dy in range(kh):
        for dx in range(kw):
            patch = x[:, dy:dy + stride * hh:stride, dx:dx + stride * ww:stride]
            out += patch @ w[:, :, dy, dx].T.astype(np.float64)
    return out + b


@pytest.fixture
def tiny_vae(st_file):
    return reader.load_vae(st_file, family="z_image", dtype=mx.float32)


def test_conv_layout_transform_matches_a_numpy_conv(tiny_vae):
    t = tiny_tensors()
    rng = np.random.default_rng(1)
    z = rng.standard_normal((1, 5, 6, 16)).astype(np.float32)
    got = np.array(tiny_vae.decoder.conv_in(mx.array(z)))
    want = _np_conv(z, t["decoder.conv_in.weight"], t["decoder.conv_in.bias"])
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)

    x = rng.standard_normal((1, 6, 7, 64)).astype(np.float32)
    assert tiny_vae.encoder.down_blocks[0].resnets[0].conv_shortcut is None
    sc = tiny_vae.decoder.up_blocks[1].resnets[0].conv_shortcut
    name = "decoder.up_blocks.1.resnets.0.conv_shortcut"
    got = np.array(sc(mx.array(x)))
    want = _np_conv(x, t[name + ".weight"], t[name + ".bias"],
                    pad=(0, 0, 0, 0))
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


def test_downsample_pads_bottom_and_right_then_strides(tiny_vae):
    t = tiny_tensors()
    x = np.random.default_rng(2).standard_normal((1, 6, 8, 32))
    x = x.astype(np.float32)
    got = np.array(tiny_vae.encoder.down_blocks[0].downsamplers[0](
        mx.array(x)))
    name = "encoder.down_blocks.0.downsamplers.0.conv"
    want = _np_conv(x, t[name + ".weight"], t[name + ".bias"], stride=2,
                    pad=(0, 1, 0, 1))
    assert got.shape == (1, 3, 4, 32)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


def test_upsample_is_nearest_then_conv(tiny_vae):
    t = tiny_tensors()
    x = np.random.default_rng(3).standard_normal((1, 3, 4, 64))
    x = x.astype(np.float32)
    got = np.array(tiny_vae.decoder.up_blocks[0].upsamplers[0](mx.array(x)))
    near = x.repeat(2, axis=1).repeat(2, axis=2)
    name = "decoder.up_blocks.0.upsamplers.0.conv"
    want = _np_conv(near, t[name + ".weight"], t[name + ".bias"])
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


def test_mid_attention_is_one_head_over_pixels_with_a_residual(tiny_vae):
    t = tiny_tensors()
    p = "decoder.mid_block.attentions.0."
    x = np.random.default_rng(4).standard_normal((2, 3, 5, 64))
    x = x.astype(np.float32)
    got = np.array(tiny_vae.decoder.mid_block.attentions[0](mx.array(x)))
    h = _np_group_norm(x, t[p + "group_norm.weight"], t[p + "group_norm.bias"],
                       32, 1e-6).reshape(2, 15, 64)

    def lin(n):
        return h @ t[p + n + ".weight"].T + t[p + n + ".bias"]

    q, k, v = lin("to_q"), lin("to_k"), lin("to_v")
    s = q @ k.transpose(0, 2, 1) / np.sqrt(64)
    a = np.exp(s - s.max(-1, keepdims=True))
    a /= a.sum(-1, keepdims=True)
    o = (a @ v) @ t[p + "to_out.0.weight"].T + t[p + "to_out.0.bias"]
    # The projections are float32 matmuls, which the GPU runs at TF32
    # precision, 6.5e-3 off float64 at most here.
    np.testing.assert_allclose(got, x + o.reshape(x.shape), rtol=1e-2,
                               atol=5e-2)


def test_chunked_sdpa_equals_one_call():
    rng = np.random.default_rng(5)
    q, k, v = (mx.array(rng.standard_normal((1, 1, 37, 16)).astype(
        np.float32)) for _ in range(3))
    whole = flux1_ae.chunked_sdpa(q, k, v)
    parts = flux1_ae.chunked_sdpa(q, k, v, score_bytes=37 * 4 * 5)
    np.testing.assert_allclose(np.array(parts), np.array(whole), rtol=1e-6,
                               atol=1e-6)


def test_encode_splits_mean_and_clipped_logvar(tiny_vae):
    x = mx.array(np.random.default_rng(6).uniform(-1, 1, (1, 8, 6, 3))
                 .astype(np.float32))
    mean, logvar = tiny_vae.encode(x)
    assert mean.shape == logvar.shape == (1, 4, 3, 16)
    full = tiny_vae.encoder(x)
    assert mx.array_equal(mean, full[..., :16]).item()
    assert mx.array_equal(logvar, mx.clip(full[..., 16:], -30, 20)).item()


def test_latent_space_round_trip_and_decode_shape(tiny_vae):
    assert tiny_vae.config.scaling_factor == pytest.approx(0.3611)
    assert tiny_vae.config.shift_factor == pytest.approx(0.1159)
    lat = mx.array(np.random.default_rng(7).standard_normal((1, 16, 3, 5))
                   .astype(np.float32))
    z = tiny_vae.to_vae_space(lat)
    assert z.shape == (1, 3, 5, 16)
    np.testing.assert_allclose(np.array(z),
                               np.array(lat).transpose(0, 2, 3, 1) / 0.3611
                               + 0.1159, rtol=1e-6)
    back = tiny_vae.to_model_space(z)
    np.testing.assert_allclose(np.array(back), np.array(lat), atol=1e-5)
    img = tiling.decode(tiny_vae, lat)
    assert img.shape == (1, 6, 10, 3) and img.dtype == mx.float32


# Reader


def test_reader_identifies_the_tiny_file_and_strips_prefixes(tmp_path):
    t = tiny_tensors()
    plain = reader.load_vae(write_safetensors(tmp_path / "a.safetensors", t),
                            family="z_image", dtype="fp32")
    pre = reader.load_vae(write_safetensors(
        tmp_path / "b.safetensors",
        {"first_stage_model." + k: v for k, v in t.items()}),
        family="z_image", dtype="fp32")
    assert plain.config == pre.config
    assert plain.config.block_out_channels == (32, 64)
    assert plain.source_format == "safetensors"
    a, b = _weights(plain), _weights(pre)
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_original_layout_loads_the_same_weights(tmp_path):
    t = tiny_tensors()
    orig = {}
    for k, v in t.items():
        name = to_original(k, levels=2)
        if ".attn_1." in name and v.ndim == 2:
            v = v.reshape(*v.shape, 1, 1)
        orig[name] = v
    assert "decoder.up.1.upsample.conv.weight" in orig
    assert "decoder.up.0.block.0.nin_shortcut.weight" in orig
    h = reader.read_header(write_safetensors(tmp_path / "o.safetensors",
                                             orig))
    assert h.layout == "original"
    assert set(h.shapes) == set(t)
    a = _weights(reader.load_vae(tmp_path / "o.safetensors",
                                 family="z_image", dtype=mx.float32))
    b = _weights(reader.load_vae(write_safetensors(
        tmp_path / "d.safetensors", t), family="z_image", dtype=mx.float32))
    assert all(np.array_equal(a[k], b[k]) for k in b)


def test_weights_are_cast_to_the_decode_dtype(st_file):
    vae = reader.load_vae(st_file, family="z_image", dtype="bf16")
    dts = {v.dtype for _, v in tree_flatten(vae.parameters())}
    assert dts == {mx.bfloat16}
    assert vae.dtype == mx.bfloat16
    vae32 = reader.load_vae(st_file, family="z_image", dtype=mx.float32)
    a, b = _weights(vae), _weights(vae32)
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_fp16_is_refused(st_file):
    for dt in (mx.float16, "fp16", "float16"):
        with pytest.raises(reader.VAEError, match="never runs in float16"):
            reader.load_vae(st_file, family="z_image", dtype=dt)
    with pytest.raises(reader.VAEError, match="not bfloat16 or float32"):
        reader.load_vae(st_file, family="z_image", dtype=mx.int8)


def test_latent_channels_off_the_family_are_refused(tmp_path):
    t = tiny_tensors()
    bad = {}
    for k, v in t.items():
        if k == "decoder.conv_in.weight":
            v = v[:, :4]
        elif k == "encoder.conv_out.weight":
            v = v[:8]
        elif k == "encoder.conv_out.bias":
            v = v[:8]
        bad[k] = v
    path = write_safetensors(tmp_path / "sd.safetensors", bad)
    with pytest.raises(reader.VAEError,
                       match="4 latent channels.*FLUX.1 autoencoder at 16"):
        reader.load_vae(path, family="z_image", dtype=mx.float32)


def test_load_vae_puts_every_conv_on_the_chosen_path(st_file):
    from gmlx.image.vae.conv import CONV, Conv2d

    for conv in (None, "kq", "exact", "winograd"):
        vae = reader.load_vae(st_file, family="z_image", dtype=mx.float32,
                              conv=conv)
        convs = [m for _, m in vae.named_modules() if isinstance(m, Conv2d)]
        assert vae.conv == (conv or CONV)
        assert convs and all(m.path == (conv or CONV) for m in convs)
    with pytest.raises(ValueError, match="conv must be one of"):
        reader.load_vae(st_file, family="z_image", conv="fast")


def test_loading_is_strict_both_ways(tmp_path):
    t = tiny_tensors()
    miss = dict(t)
    del miss["decoder.mid_block.resnets.1.norm2.bias"]
    with pytest.raises(reader.VAEError, match="1 missing.*norm2.bias"):
        reader.load_vae(write_safetensors(tmp_path / "m.safetensors", miss),
                        family="z_image", dtype=mx.float32)
    extra = dict(t)
    extra["quant_conv.weight"] = np.zeros((32, 32, 1, 1), np.float32)
    with pytest.raises(reader.VAEError, match="1 unexpected.*quant_conv"):
        reader.load_vae(write_safetensors(tmp_path / "e.safetensors", extra),
                        family="z_image", dtype=mx.float32)
    wrong = dict(t)
    wrong["decoder.conv_out.weight"] = np.zeros((3, 32, 1, 1), np.float32)
    with pytest.raises(reader.VAEError, match="another shape.*conv_out"):
        reader.load_vae(write_safetensors(tmp_path / "w.safetensors", wrong),
                        family="z_image", dtype=mx.float32)


def test_other_files_and_families_are_refused(tmp_path):
    path = write_safetensors(tmp_path / "x.safetensors",
                             {"model.layers.0.weight": np.zeros((4, 4))})
    with pytest.raises(reader.VAEError, match="not a VAE gmlx image reads"):
        reader.load_vae(path, family="z_image", dtype=mx.float32)
    junk = tmp_path / "junk.bin"
    junk.write_bytes(b"\xff" * 64)
    with pytest.raises(reader.VAEError, match="neither a GGUF"):
        reader.load_vae(junk, family="z_image", dtype=mx.float32)
    with pytest.raises(reader.VAEError, match="known: flux3_klein, krea2, "
                       "qwen_image21, z_image"):
        reader.load_vae(path, family="sd15", dtype=mx.float32)


# Packer


def _payload_digests(path) -> dict:
    header, start = reader.safetensors_header(str(path))
    header.pop("__metadata__", None)
    raw = open(path, "rb").read()
    return {n: hashlib.sha256(raw[start + v["data_offsets"][0]:
                                  start + v["data_offsets"][1]]).hexdigest()
            for n, v in header.items()}


def test_packer_round_trip_is_bit_for_bit(tmp_path, st_file):
    import gguf

    from gmlx.image.vae import pack_vae

    dst = tmp_path / "ae.gguf"
    rep = pack_vae(st_file, dst, source_repo="org/repo",
                   source_revision="abc123", config_json=CONFIG_JSON)
    assert rep.family == "z_image" and rep.lossless and rep.tensors == 120
    assert rep.digests == _payload_digests(st_file)

    r = gguf.GGUFReader(str(dst))
    got = {t.name: hashlib.sha256(t.data.tobytes()).hexdigest()
           for t in r.tensors}
    assert got == rep.digests
    assert {t.tensor_type.name for t in r.tensors} == {"BF16"}
    keys = {k: r.fields[k].contents() for k in r.fields
            if not k.startswith("GGUF.")}
    src_sha = hashlib.sha256(open(st_file, "rb").read()).hexdigest()
    assert keys == {
        "general.architecture": "autoencoder",
        "general.type": "vae",
        "gmlx.vae.family": "z_image",
        "gmlx.vae.source_repo": "org/repo",
        "gmlx.vae.source_revision": "abc123",
        "gmlx.vae.source_file": "ae.safetensors",
        "gmlx.vae.source_sha256": src_sha,
        "gmlx.vae.layout": "torch-oihw",
        "gmlx.vae.config_json": CONFIG_JSON,
        "gmlx.packer.version": "1",
    }
    assert not (tmp_path / "ae.gguf.partial").exists()


def test_gguf_and_safetensors_loads_are_bit_identical(tmp_path, st_file):
    from gmlx.image.vae import pack_vae

    dst = pack_vae(st_file, tmp_path / "ae.gguf", source_repo="r",
                   source_revision="v", config_json="{}").path
    for dt in (mx.bfloat16, mx.float32):
        g = reader.load_vae(dst, family="z_image", dtype=dt)
        s = reader.load_vae(st_file, family="z_image", dtype=dt)
        assert g.source_format == "gguf" and s.source_format == "safetensors"
        a, b = _weights(g), _weights(s)
        assert a.keys() == b.keys()
        assert all(np.array_equal(a[k], b[k]) for k in a)


def test_packer_keeps_original_names_and_refuses_misuse(tmp_path):
    from gmlx.image.vae import pack_vae

    t = tiny_tensors()
    orig = {}
    for k, v in t.items():
        name = to_original(k, levels=2)
        orig[name] = v.reshape(*v.shape, 1, 1) if (
            ".attn_1." in name and v.ndim == 2) else v
    src = write_safetensors(tmp_path / "o.safetensors", orig)
    rep = pack_vae(src, tmp_path / "o.gguf", source_repo="r",
                   source_revision="v", config_json="{}")
    assert set(rep.digests) == set(orig)
    with pytest.raises(reader.VAEError, match="exists"):
        pack_vae(src, tmp_path / "o.gguf", source_repo="r",
                 source_revision="v", config_json="{}")
    with pytest.raises(reader.VAEError, match="already a GGUF"):
        pack_vae(tmp_path / "o.gguf", tmp_path / "p.gguf", source_repo="r",
                 source_revision="v", config_json="{}")
    with pytest.raises(reader.VAEError, match="not JSON"):
        pack_vae(src, tmp_path / "q.gguf", source_repo="r",
                 source_revision="v", config_json="{")
    bad = write_safetensors(tmp_path / "x.safetensors",
                            {"model.w": np.zeros((2, 2))})
    with pytest.raises(reader.VAEError, match="not a VAE"):
        pack_vae(bad, tmp_path / "x.gguf", source_repo="r",
                 source_revision="v", config_json="{}")


def test_a_downcast_is_recorded_as_lossy(tmp_path):
    import gguf

    from gmlx.image.vae import pack_vae

    t = {k: v + np.float32(2 ** -12) * (v != 0) for k, v in
         tiny_tensors().items()}
    src = write_safetensors(tmp_path / "f.safetensors", t, dtype="F32")
    rep = pack_vae(src, tmp_path / "f.gguf", source_repo="r",
                   source_revision="v", config_json="{}", dtype="BF16")
    assert not rep.lossless
    r = gguf.GGUFReader(str(tmp_path / "f.gguf"))
    assert r.fields["gmlx.vae.lossless"].contents() is False
    up = pack_vae(src, tmp_path / "g.gguf", source_repo="r",
                  source_revision="v", config_json="{}", dtype="F32")
    assert up.lossless and up.digests == _payload_digests(src)


# Tiling


def test_tile_starts_cover_evenly_with_the_overlap():
    assert tiling.tile_starts(40, 64, 8) == [0]
    assert tiling.tile_starts(64, 64, 8) == [0]
    for size in (65, 100, 120, 121, 128, 200, 256, 333):
        s = tiling.tile_starts(size, 64, 8)
        assert s[0] == 0 and s[-1] + 64 == size
        gaps = np.diff(s)
        assert all(64 - g >= 8 for g in gaps)
        assert gaps.max() - gaps.min() <= 1
    assert tiling.tile_starts(128, 64, 8) == [0, 32, 64]


def test_tiled_identity_decoder_reproduces_the_input():
    rng = np.random.default_rng(8)
    z = rng.standard_normal((2, 21, 30, 3)).astype(np.float32)

    def up8(t):
        return mx.repeat(mx.repeat(t, 8, axis=1), 8, axis=2)

    got = tiling.run_tiled(up8, mx.array(z), in_scale=1, out_scale=8,
                           tile=8, overlap=3)
    want = z.repeat(8, axis=1).repeat(8, axis=2)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)

    x = rng.standard_normal((1, 21 * 8, 30 * 8, 3)).astype(np.float32)
    got = tiling.run_tiled(lambda t: t[:, ::8, ::8], mx.array(x),
                           in_scale=8, out_scale=1, tile=8, overlap=3)
    np.testing.assert_allclose(got, x[:, ::8, ::8], rtol=1e-6, atol=1e-6)


def test_blend_weights_ramp_across_each_overlap():
    starts = tiling.tile_starts(20, 8, 2)
    total = np.zeros(20 * 4)
    for i, s in enumerate(starts):
        w = tiling.axis_weights(starts, i, 8, 4)
        assert w.min() > 0 and w.max() <= 1
        total[s * 4:(s + 8) * 4] += w
    np.testing.assert_allclose(total, 1.0, atol=1e-6)


def test_tiled_decode_and_encode_run_on_the_tiny_vae(tiny_vae):
    rng = np.random.default_rng(9)
    lat = rng.standard_normal((1, 16, 12, 10)).astype(np.float32)
    whole = tiling.decode(tiny_vae, lat)
    tiles = tiling.decode(tiny_vae, lat, tiled=True, tile=16, overlap=4)
    assert tiles.shape == whole.shape == (1, 24, 20, 3)
    assert bool(mx.all(mx.isfinite(tiles)))
    img = rng.uniform(-1, 1, (1, 24, 20, 3)).astype(np.float32)
    a = tiling.encode(tiny_vae, img)
    b = tiling.encode(tiny_vae, img, tiled=True, tile=16, overlap=4)
    assert a.shape == b.shape == (1, 16, 12, 10)
    noise = rng.standard_normal((1, 16, 12, 10)).astype(np.float32)
    c = tiling.encode(tiny_vae, img, mode="sample", noise=noise)
    assert not mx.array_equal(a, c).item()
    with pytest.raises(ValueError, match="needs noise"):
        tiling.encode(tiny_vae, img, mode="sample")
    with pytest.raises(ValueError, match="not a multiple"):
        tiling.encode(tiny_vae, img[:, :23])


def _pointwise_vae(tiny_vae):
    """The tiny VAE with every 3x3 kernel cut to its center tap, so no conv
    mixes neighboring pixels and a tile border changes no value. Only the
    GroupNorm statistics and the mid attention see the whole image."""
    for _, m in tiny_vae.named_modules():
        if isinstance(m, nn.Conv2d) and m.weight.shape[1] == 3:
            keep = np.zeros((1, 3, 3, 1), np.float32)
            keep[0, 1, 1, 0] = 1
            m.weight = m.weight * mx.array(keep)
    return tiny_vae


def test_tiles_normalize_with_the_statistics_of_the_whole_image(tiny_vae):
    vae = _pointwise_vae(tiny_vae)
    rng = np.random.default_rng(12)
    ramp = np.linspace(-2, 2, 40, dtype=np.float32)
    lat = rng.standard_normal((1, 16, 40, 36)).astype(np.float32)
    lat += ramp[None, None, :, None] + 2 * ramp[None, None, None, :36]
    whole = np.array(tiling.decode(vae, lat, tiled=False, free=1e12))
    tiles = np.array(tiling.decode(vae, lat, tiled=True, tile=32,
                                   overlap=8, free=1e12))
    np.testing.assert_allclose(tiles, whole, rtol=1e-4, atol=1e-4)
    img = np.clip(whole, -1, 1)
    a = np.array(tiling.encode(vae, img, tiled=False, free=1e12))
    b = np.array(tiling.encode(vae, img, tiled=True, tile=32, overlap=8,
                               free=1e12))
    # The encoder's tiled convs round at TF32 on a GPU that multiplies
    # float32 that way: about 1.2e-4 from the whole pass here. Tiles
    # normalized with their own statistics land 6.6e-2 off.
    np.testing.assert_allclose(b, a, rtol=1e-3, atol=1e-3)


def test_owned_regions_partition_the_image():
    for size, tile, overlap in ((40, 16, 4), (128, 64, 8), (77, 20, 6)):
        starts = tiling.tile_starts(size, tile, overlap)
        for r in (1, 2, 8):
            rows = np.concatenate([
                np.arange(round(s * r), round((s + tile) * r))[
                    tiling._owned(starts, i, tile, size, r)]
                for i, s in enumerate(starts)])
            np.testing.assert_array_equal(rows, np.arange(size * r))


# Memory: the tiling choice, the refusal and the cache bound


def _unit_fits(monkeypatch):
    from gmlx.image import memfit

    unit = {2: memfit.VaeFit(0.0, 1.0), 4: memfit.VaeFit(0.0, 1.0)}
    rows = {"exact": unit, "winograd": unit}
    tiled = {2: memfit.TiledFit(0.0, 0.25, 0.0),
             4: memfit.TiledFit(0.0, 0.25, 0.0)}
    monkeypatch.setitem(memfit.VAE_ROWS, "z_image", dataclasses.replace(
        memfit.VAE_ROWS["z_image"], whole={"decode": rows, "encode": rows},
        tiled={"decode": tiled, "encode": tiled}))


# With k = 1 the tiny VAE (32 base channels, fp32) at 24x20 needs 61440
# bytes whole, and with k_tiles = 0.25 32768 bytes in four 16 pixel tiles.
WHOLE, TILES = 24 * 20 * 32 * 4, 4 * 16 * 16 * 32 * 4 // 4


def test_decode_and_encode_tile_only_when_a_whole_pass_does_not_fit(
        tiny_vae, monkeypatch):
    _unit_fits(monkeypatch)
    rng = np.random.default_rng(11)
    lat = rng.standard_normal((1, 16, 12, 10)).astype(np.float32)
    whole = tiling.decode(tiny_vae, lat, tiled=False, free=1e12)
    tiles = tiling.decode(tiny_vae, lat, tiled=True, tile=16, overlap=4,
                          free=1e12)
    assert not mx.array_equal(whole, tiles).item()
    roomy = tiling.decode(tiny_vae, lat, tile=16, overlap=4, free=WHOLE)
    tight = tiling.decode(tiny_vae, lat, tile=16, overlap=4, free=WHOLE - 1)
    assert mx.array_equal(roomy, whole).item()
    assert mx.array_equal(tight, tiles).item()
    img = rng.uniform(-1, 1, (1, 24, 20, 3)).astype(np.float32)
    a = tiling.encode(tiny_vae, img, tiled=False, free=1e12)
    b = tiling.encode(tiny_vae, img, tiled=True, tile=16, overlap=4,
                      free=1e12)
    assert mx.array_equal(tiling.encode(tiny_vae, img, tile=16, overlap=4,
                                        free=WHOLE), a).item()
    assert mx.array_equal(tiling.encode(tiny_vae, img, tile=16, overlap=4,
                                        free=TILES), b).item()


def test_without_a_free_figure_an_image_above_one_tile_runs_in_tiles(
        tiny_vae, monkeypatch):
    from gmlx.image import memfit

    monkeypatch.setattr(memfit, "free_bytes", lambda: None)
    rng = np.random.default_rng(13)
    lat = rng.standard_normal((1, 16, 12, 10)).astype(np.float32)
    img = rng.uniform(-1, 1, (1, 24, 20, 3)).astype(np.float32)
    cut = dict(tile=16, overlap=4)
    for run, x in ((tiling.decode, lat), (tiling.encode, img)):
        whole = run(tiny_vae, x, tiled=False, **cut)
        tiles = run(tiny_vae, x, tiled=True, **cut)
        assert not mx.array_equal(whole, tiles).item()
        assert mx.array_equal(run(tiny_vae, x, **cut), tiles).item()
        # An image no larger than one tile runs whole.
        assert mx.array_equal(run(tiny_vae, x), whole).item()


def test_a_pass_that_does_not_fit_is_refused_before_it_runs(
        tiny_vae, monkeypatch):
    from gmlx.image.memfit import ImageMemoryError

    _unit_fits(monkeypatch)
    monkeypatch.delenv("GMLX_OVERCOMMIT", raising=False)

    def boom(*_):
        raise AssertionError("ran a pass that was refused")

    monkeypatch.setattr(tiny_vae, "decode", boom)
    lat = np.zeros((1, 16, 12, 10), np.float32)
    with pytest.raises(ImageMemoryError, match="a tiled pass needs") as e:
        tiling.decode(tiny_vae, lat, tiled=False, tile=16, overlap=4,
                      free=TILES)
    assert e.value.plan is None
    assert "GMLX_OVERCOMMIT=1 overrides" in str(e.value)
    with pytest.raises(ImageMemoryError, match="try a smaller image"):
        tiling.decode(tiny_vae, lat, tile=16, overlap=4, free=TILES - 1)
    with pytest.raises(ImageMemoryError, match="VAE encode of 1 x 24x20"):
        tiling.encode(tiny_vae, np.zeros((1, 24, 20, 3), np.float32),
                      tile=16, overlap=4, free=TILES - 1)
    monkeypatch.setenv("GMLX_OVERCOMMIT", "1")
    with pytest.raises(AssertionError, match="was refused"):
        tiling.decode(tiny_vae, lat, tiled=False, tile=16, overlap=4,
                      free=0)


def test_decode_and_encode_run_inside_the_cache_bound(tiny_vae, monkeypatch):
    from gmlx.image import memfit

    seen = []

    def limit_now():
        lim = mx.set_cache_limit(0)
        mx.set_cache_limit(lim)
        return lim

    def spy(fn):
        def run(*a):
            seen.append(limit_now())
            return fn(*a)
        return run

    monkeypatch.setattr(memfit, "CACHE_BOUND", 3 << 20)
    monkeypatch.setattr(tiny_vae, "decode", spy(tiny_vae.decode))
    monkeypatch.setattr(tiny_vae, "encode", spy(tiny_vae.encode))
    before = mx.set_cache_limit(1 << 40)
    try:
        tiling.decode(tiny_vae, np.zeros((1, 16, 12, 10), np.float32),
                      free=1e12)
        tiling.encode(tiny_vae, np.zeros((1, 24, 20, 3), np.float32),
                      tiled=True, tile=16, overlap=4, free=1e12)
        assert seen and set(seen) == {3 << 20}
        assert limit_now() == 1 << 40
    finally:
        mx.set_cache_limit(before)


def test_a_truncated_vae_file_is_refused_before_any_bytes_load(tmp_path,
                                                                st_file):
    from gmlx.image.vae import pack_vae

    gg = pack_vae(st_file, tmp_path / "ae.gguf", source_repo="r",
                  source_revision="v", config_json="{}").path
    for src, kind in ((gg, "GGUF"), (st_file, "safetensors")):
        data = open(src, "rb").read()
        cut = tmp_path / f"cut-{kind}"
        cut.write_bytes(data[:-1024])
        with pytest.raises(reader.VAEError,
                           match=rf"truncated {kind} \(.*expects "
                                 rf"\d+ bytes, file has "
                                 rf"{len(data) - 1024}\)"):
            reader.read_header(str(cut))
        with pytest.raises(reader.VAEError, match=f"truncated {kind}"):
            reader.load_vae(str(cut), family="z_image", dtype=mx.bfloat16)
    head = tmp_path / "head.gguf"
    head.write_bytes(open(gg, "rb").read()[:40])
    with pytest.raises(reader.VAEError,
                       match=r"ends inside its header, at 40 bytes"):
        reader.read_header(str(head))
