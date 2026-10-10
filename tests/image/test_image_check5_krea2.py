"""Check 5 of the image acceptance gate for Krea 2: the Qwen-Image VAE.

The tiny self-test runs on the MLX CPU stream: diffusers'
AutoencoderKLQwenImage at a tiny config with seeded weights, which the
committed fixture holds by diffusers name, written here as a safetensors
file with the video layout's rank 5 kernels. The fp32 decode and encoder
mean are held to 5e-5 max abs, each wrong model below misses that, and a
GGUF packed from the file loads the same weights bit for bit.

The official rows read the Comfy-Org file, in the original Wan 2.1
layout, under GMLX_TEST_IMAGE_DIR, against the oracle fixture at 512
pixels: the fp32 decode to twice the oracle's own fp32 against float64
max abs, the bf16 decode PSNR and the encoder means to the manifest's
gates, and a tiled decode at 1024 pixels to the whole decode.
scripts/image_oracle/check5_krea2_vae.py writes the fixtures.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest
from image_dir import sha256, vae_files
from image_fixtures import load_json
from oracle_fixture import ORACLE, committed_fixture, fixture, fx

import mlx.core as mx
from mlx.utils import tree_flatten

FAMILY = "krea2"
NAME = "check5_krea2_vae"
TINY = "selftest_krea2_vae_tiny"
TINY_FP32_MAX_ABS = 5e-5
HEADER = "krea2_vae.header.json"


def to_unit(img: np.ndarray) -> np.ndarray:
    return np.clip(img.astype(np.float64) / 2 + 0.5, 0.0, 1.0)


def _write_st(path, tensors: dict) -> str:
    """``tensors`` as an F32 safetensors file, names sorted."""
    head, blobs, off = {}, [], 0
    for k in sorted(tensors):
        a = np.ascontiguousarray(tensors[k], np.float32)
        head[k] = {"dtype": "F32", "shape": list(a.shape),
                   "data_offsets": [off, off + a.nbytes]}
        blobs.append(a.tobytes())
        off += a.nbytes
    text = json.dumps(head, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    with open(path, "wb") as f:
        f.write(len(text).to_bytes(8, "little") + text + b"".join(blobs))
    return str(path)


@pytest.fixture(scope="module")
def tiny():
    return committed_fixture(TINY)


@pytest.fixture
def tiny_file(tiny, tmp_path):
    arrays, _ = tiny
    w = {k[len("weight."):]: v for k, v in arrays.items()
         if k.startswith("weight.")}
    return _write_st(tmp_path / "vae.safetensors", w)


def _decode(vae, latents, **kw) -> np.ndarray:
    from gmlx.image.vae import tiling

    return np.array(tiling.decode(vae, latents, **kw)).transpose(0, 3, 1, 2)


def _mean(vae, image) -> np.ndarray:
    mean, _ = vae.encode(mx.array(image.transpose(0, 2, 3, 1)))
    return np.array(mean.astype(mx.float32)).transpose(0, 3, 1, 2)


def _tiny_readings(vae, arrays) -> dict:
    s = "48x64"
    with mx.stream(mx.cpu):
        dec = _decode(vae, arrays[f"latents_{s}"], tiled=False)
        mean = _mean(vae, arrays[f"image_{s}"])
    return {"decode": fx.max_abs(dec, arrays[f"decoded_{s}"]),
            "encode": fx.max_abs(mean, arrays[f"encode_mean_{s}"])}


def test_the_tiny_fp32_decode_and_encode_are_within_max_abs(tiny,
                                                           tiny_file):
    from gmlx.image.vae import load_vae, read_header

    arrays, m = tiny
    assert read_header(tiny_file).layout == "diffusers"
    vae = load_vae(tiny_file, family=FAMILY, dtype="fp32", conv="exact")
    got = _tiny_readings(vae, arrays)
    print(f"[check5] krea2 tiny fp32: {got}, gate {TINY_FP32_MAX_ABS}")
    assert m["tolerances"]["fp32_max_abs"] == TINY_FP32_MAX_ABS
    assert max(got.values()) <= TINY_FP32_MAX_ABS


def _first_slice(monkeypatch):
    """A causal kernel read at its first time slice, which one frame
    multiplies by the zero padding."""
    from gmlx.image.vae import reader

    monkeypatch.setattr(reader, "frame_kernel", lambda a: a[:, :, 0])


def _no_sqrt_c(monkeypatch):
    from gmlx.image.vae import qwen_image21_vae as q

    real = q.rms_norm
    monkeypatch.setattr(q, "rms_norm", lambda x, g: real(x, g) / (
        x.shape[-1] ** 0.5))


def _pad_top_left(monkeypatch):
    from gmlx.image.vae import qwen_image21_vae as q

    monkeypatch.setattr(q, "DOWN_PAD", ((0, 0), (1, 0), (1, 0), (0, 0)))


def _latents_reversed(monkeypatch):
    """The latent statistics applied to the channels in reverse."""
    from gmlx.image.vae import qwen_image_vae as q

    real = q.QwenImageVAE.to_vae_space
    monkeypatch.setattr(q.QwenImageVAE, "to_vae_space",
                        lambda self, z: real(self, z)[..., ::-1])


WRONG = {"first time slice": _first_slice, "norm without sqrt(C)": _no_sqrt_c,
         "pad on the top and left": _pad_top_left,
         "latent channels reversed": _latents_reversed}


@pytest.mark.parametrize("name", list(WRONG),
                         ids=[n.replace(" ", "_") for n in WRONG])
def test_a_wrong_model_misses_the_tiny_gate(tiny, tiny_file, monkeypatch,
                                            name):
    from gmlx.image.vae import load_vae

    arrays, _ = tiny
    WRONG[name](monkeypatch)
    vae = load_vae(tiny_file, family=FAMILY, dtype="fp32", conv="exact")
    got = _tiny_readings(vae, arrays)
    print(f"[check5] krea2 tiny, {name}: {got}")
    assert max(got.values()) > 25 * TINY_FP32_MAX_ABS


def test_a_packed_gguf_loads_the_same_weights(tiny_file, tmp_path):
    import gguf

    from gmlx.image import preflight
    from gmlx.image.vae import load_vae, pack_vae, read_header

    dst = tmp_path / "vae.gguf"
    rep = pack_vae(tiny_file, dst, source_repo="r", source_revision="v",
                   config_json="{}")
    assert rep.family == FAMILY and rep.lossless
    r = gguf.GGUFReader(str(dst))
    stored = {k[len("comfy.gguf.orig_shape."):]: r.fields[k].contents()
              for k in r.fields if k.startswith("comfy.gguf.orig_shape.")}
    st = read_header(tiny_file)
    five = {n: list(s) for n, s in st.shapes.items() if len(s) == 5}
    assert stored == five and len(five) > 20
    assert max(len(t.shape) for t in r.tensors) == 4
    assert read_header(str(dst)).shapes == st.shapes
    assert preflight.classify(str(dst)) == "vae"
    a = dict(tree_flatten(load_vae(tiny_file, family=FAMILY,
                                   dtype="fp32").parameters()))
    b = dict(tree_flatten(load_vae(str(dst), family=FAMILY,
                                   dtype="fp32").parameters()))
    assert a.keys() == b.keys()
    assert [k for k in a if not mx.array_equal(a[k], b[k]).item()] == []


def test_the_wan_layout_fills_the_module_tree():
    """The Comfy-Org file's names, in the original Wan 2.1 layout, map
    onto every parameter of the module, and every other tensor is a
    time_conv one frame leaves unread, at its shape."""
    from gmlx.image.vae import reader
    from gmlx.image.vae import qwen_image_vae as q

    head = load_json(HEADER)
    shapes = {n: tuple(s) for n, s, _ in head["tensors"]}
    layout, names = reader.diffusers_names({n: n for n in shapes})
    assert layout == "wan" and len(names) == len(shapes) == 194
    mapped = {n: shapes[f] for n, f in names.items()}
    cfg = q.config_from_shapes(mapped)
    assert cfg is not None and cfg.dim_mult == (1, 2, 4, 4)
    header = reader.VaeHeader("x", "safetensors", layout, names, mapped,
                              {n: "BF16" for n in names}, {})
    params = dict(tree_flatten(q.QwenImageVAE(cfg).parameters()))
    reader.check_names(header, params, reader.VAES[FAMILY], cfg)
    unread = q.expected_unused(cfg)
    assert set(mapped) - set(params) == set(unread) and len(unread) == 8


def test_other_families_refuse_the_qwen_image_vae(tiny_file):
    from gmlx.image.vae import load_vae, reader

    for family, title in (("qwen_image21", "the Qwen-Image 2.1 autoencoder"),
                          ("z_image", "the FLUX.1 autoencoder")):
        with pytest.raises(reader.VAEError, match="holds the Qwen-Image "
                           f"autoencoder, the krea2 VAE, and {family} "
                           f"decodes with {title}"):
            load_vae(tiny_file, family=family, dtype="fp32")


def test_the_memory_row_prices_tiles_first_with_a_crop():
    from gmlx.image import memfit
    from gmlx.image.vae import qwen_image_memory as m
    from gmlx.image.vae import qwen_image_vae as q

    row = memfit.vae_row(FAMILY)
    assert not row.measured and row.auto_tiles and row.one_tile
    assert memfit.crop_reach(FAMILY, "decode") == q.reach(m.OFFICIAL,
                                                          "decode") == 6
    assert memfit.crop_reach(FAMILY, "encode") == 10
    # Each step of a whole decode at 512 pixels gives the shape the module
    # computes.
    shapes = m.output_shapes(m.OFFICIAL, "decode", 512, 512)
    assert shapes["decoder.mid_block.resnets.1.add"] == (384, 64, 64)
    assert shapes["decoder.up_blocks.2.upsamplers.0.resample.1"] == (
        96, 512, 512)
    assert shapes["clamp"] == (3, 512, 512)
    whole = memfit.vae_peak("decode", 1024, 1024, family=FAMILY, itemsize=2)
    tiled = memfit.vae_peak("decode", 1024, 1024, family=FAMILY, itemsize=2,
                            tiled=True)
    assert tiled < whole
    params = tree_flatten(q.QwenImageVAE(m.OFFICIAL).parameters())
    assert m.weight_count(m.OFFICIAL) == sum(v.size for _, v in params)


# The official file: needs it under GMLX_TEST_IMAGE_DIR and the fixture of
# the committed manifest.

official = pytest.mark.integration


@pytest.fixture(scope="module")
def check5():
    if not (ORACLE / f"{NAME}.manifest.json").is_file():
        pytest.skip(f"no committed {NAME} manifest")
    return fixture(NAME)


@pytest.fixture(scope="module")
def vaes(check5):
    from gmlx.image.vae import load_vae

    want = check5[1]["files"]["vae"]["sha256"]
    path = next((p for p in vae_files(FAMILY) if p.suffix == ".safetensors"
                 and sha256(p) == want), None)
    if path is None:
        pytest.skip("the fixture's VAE is not under GMLX_TEST_IMAGE_DIR")
    cache = {}

    def get(dtype):
        if dtype not in cache:
            cache[dtype] = load_vae(path, family=FAMILY, dtype=dtype)
        return cache[dtype]

    return get


def _tf32() -> bool:
    from block_gates import tf32

    return tf32()


def _gate_without_tf32(request) -> None:
    env = dict(os.environ, MLX_ENABLE_TF32="0")
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-s", "-p", "no:cacheprovider",
         request.node.nodeid], cwd=request.config.rootpath, env=env,
        capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if "[check5]" in line:
            print(line[line.index("[check5]"):])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]


@official
def test_the_file_is_the_inventoried_one(vaes):
    from gmlx.image.vae import read_header

    path = vaes("bf16").source
    want = load_json("manifest.json")[HEADER]
    head = read_header(path)
    assert os.path.getsize(path) == want["size"] and head.layout == "wan"
    assert sorted((head.names[n], list(s)) for n, s in head.shapes.items()) \
        == sorted((n, s) for n, s, _ in load_json(HEADER)["tensors"])


@official
def test_fp32_decode_and_encode_are_within_the_gates(check5, vaes, request):
    arrays, m = check5
    vae = vaes("fp32")
    for s, t in m["tolerances"]["sizes"].items():
        dec = _decode(vae, arrays[f"latents_{s}"], tiled=False)
        mean = _mean(vae, arrays[f"image_{s}"])
        d = fx.max_abs(dec, arrays[f"decoded_{s}"])
        e = fx.max_abs(mean, arrays[f"encode_mean_{s}"])
        print(f"[check5] krea2 {s} fp32, TF32 {'on' if _tf32() else 'off'}: "
              f"decode max abs {d:.3e} of gate "
              f"{t['decode_fp32_max_abs']:.3e}, encode mean max abs "
              f"{e:.3e} of gate {t['encode_fp32_max_abs']:.3e}")
        if not _tf32():
            assert d <= t["decode_fp32_max_abs"], s
            assert e <= t["encode_fp32_max_abs"], s
    if _tf32():
        _gate_without_tf32(request)


@official
def test_bf16_decode_psnr_and_encode(check5, vaes):
    arrays, m = check5
    vae = vaes("bf16")
    for s, t in m["tolerances"]["sizes"].items():
        dec = _decode(vae, arrays[f"latents_{s}"], tiled=False)
        db = fx.psnr(to_unit(dec), to_unit(arrays[f"decoded_{s}"]), 1.0)
        rel = fx.rel_l2(_mean(vae, arrays[f"image_{s}"]),
                        arrays[f"encode_mean_{s}"])
        print(f"[check5] krea2 {s} bf16: decode PSNR {db:.2f} dB, gate "
              f"{t['decode_bf16_psnr_db']:.2f} dB; encode mean rel L2 "
              f"{rel:.3e}, gate {t['encode_bf16_rel_l2']:.3e}")
        assert db >= t["decode_bf16_psnr_db"], s
        assert rel <= t["encode_bf16_rel_l2"], s


@official
@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
def test_a_tiled_decode_is_the_whole_decode_at_1024(check5, vaes, dtype):
    """The 512 latents of the fixture, tiled two by two to 1024 pixels. The
    part of the decoder that reads the whole latent runs once and the
    tiles of the rest are cropped at the reach of a cut, so a tiled decode
    is the whole one: equal in bits in fp32, and within 2e-3 max abs in
    bf16."""
    from gmlx.image import memfit

    arrays, m = check5
    lat = np.tile(arrays["latents_512x512"], (1, 1, 2, 2))
    vae = vaes(dtype)
    assert memfit.vae_row(FAMILY).tiles_at(1024 * 1024)
    tiled = _decode(vae, lat, tiled=True)
    whole = _decode(vae, lat, tiled=False)
    err = fx.max_abs(tiled, whole)
    print(f"[check5] krea2 1024x1024 {dtype} tiled decode: max abs {err:.3e} "
          "against the whole decode")
    if dtype == "fp32":
        assert np.array_equal(tiled, whole)
    else:
        assert err <= 2e-3


@official
def test_a_tiled_encode_is_the_whole_encode_at_1024(check5, vaes):
    from gmlx.image.vae import tiling

    arrays, _ = check5
    image = np.tile(arrays["image_512x512"], (1, 1, 2, 2)).transpose(
        0, 2, 3, 1)
    vae = vaes("fp32")
    tiled = np.array(tiling.encode(vae, image, tiled=True))
    whole = np.array(tiling.encode(vae, image, tiled=False))
    rel = fx.rel_l2(tiled, whole)
    print(f"[check5] krea2 1024x1024 fp32 tiled encode: rel L2 {rel:.3e} "
          "against the whole encode")
    assert rel <= 1e-5
