"""Check 5 of the image acceptance gate for FLUX.3 klein: the FLUX.2 VAE.

The tiny self-test runs on the MLX CPU stream: diffusers'
AutoencoderKLFlux2 at a tiny config with seeded weights and BatchNorm
statistics, which the committed fixture holds by diffusers name, written
here as a safetensors file in each layout with the int64 batch count. The
fp32 decode, the encoder mean and the model space latents are held to
5e-5 max abs, each wrong model below misses that, and a GGUF packed from
either file loads the same weights bit for bit.

The official rows read both files under GMLX_TEST_IMAGE_DIR, the
Comfy-Org file in the diffusers layout and BFL's ae.safetensors in the
original one, against the oracle fixture at 512 and 1024 pixels: the fp32
decode and model space latents to the manifest's max abs, the bf16 decode
PSNR and latents relative L2 to its gates, and a tiled bf16 decode at 1024
pixels to the bf16 PSNR gate of that size. At 2048 pixels, which the
oracle does not decode, a tiled bf16 decode holds the gate of 1024 against
the tiled fp32 decode. A tiled bf16 encode at 1024 stays within 0.05
relative L2 of the whole encode, and its round trip holds 50 dB PSNR
against the whole round trip. scripts/image_oracle/check5_flux3_klein_vae.py
writes the fixtures.
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

# A Mac of 16 GB, as most users run (tests/host_box.py).
pytestmark = pytest.mark.host_gb(16)

FAMILY = "flux3_klein"
NAME = "check5_flux3_klein_vae"
TINY = "selftest_flux3_klein_vae_tiny"
TINY_FP32_MAX_ABS = 5e-5
HEADERS = {"diffusers": "flux3_klein_vae.header.json",
           "original": "flux3_klein_ae.header.json"}
COUNT = "bn.num_batches_tracked"
TILED_ENCODE_REL_L2 = 0.05
TILED_ROUND_TRIP_DB = 50.0


def to_unit(img: np.ndarray) -> np.ndarray:
    return np.clip(img.astype(np.float64) / 2 + 0.5, 0.0, 1.0)


def _write_st(path, tensors: dict) -> str:
    """``tensors`` as a safetensors file, names sorted: int64 arrays as
    I64, others as F32."""
    head, blobs, off = {}, [], 0
    for k in sorted(tensors):
        a = np.asarray(tensors[k])
        if a.dtype == np.int64:
            kind = "I64"
        else:
            a, kind = np.asarray(a, np.float32), "F32"
        raw = np.ascontiguousarray(a).tobytes()
        head[k] = {"dtype": kind, "shape": list(a.shape),
                   "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    text = json.dumps(head, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    with open(path, "wb") as f:
        f.write(len(text).to_bytes(8, "little") + text + b"".join(blobs))
    return str(path)


def to_original(name: str, levels: int) -> str:
    """The name of a diffusers tensor in BFL's original layout, written out
    apart from the reader's map."""
    for old, new in (("quant_conv.", "encoder.quant_conv."),
                     ("post_quant_conv.", "decoder.post_quant_conv.")):
        if name.startswith(old):
            return new + name[len(old):]
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


@pytest.fixture(scope="module")
def tiny():
    return committed_fixture(TINY)


def _tiny_tensors(tiny) -> dict:
    arrays, m = tiny
    w = {k[len("weight."):]: v for k, v in arrays.items()
         if k.startswith("weight.")}
    w[COUNT] = np.array(m["inputs"]["num_batches_tracked"], np.int64)
    return w


@pytest.fixture
def tiny_file(tiny, tmp_path):
    return _write_st(tmp_path / "vae.safetensors", _tiny_tensors(tiny))


@pytest.fixture
def tiny_original(tiny, tmp_path):
    levels = len(tiny[1]["inputs"]["config"]["block_out_channels"])
    out = {}
    for k, v in _tiny_tensors(tiny).items():
        name = to_original(k, levels)
        if ".attn_1." in name and v.ndim == 2:
            v = v.reshape(*v.shape, 1, 1)
        out[name] = v
    return _write_st(tmp_path / "ae.safetensors", out)


def _decode(vae, latents, **kw) -> np.ndarray:
    from gmlx.image.vae import tiling

    return np.array(tiling.decode(vae, latents, **kw)).transpose(0, 3, 1, 2)


def _latents(vae, image, **kw) -> np.ndarray:
    from gmlx.image.vae import tiling

    return np.array(tiling.encode(vae, image.transpose(0, 2, 3, 1), **kw))


def _mean(vae, image) -> np.ndarray:
    mean, _ = vae.encode(mx.array(image.transpose(0, 2, 3, 1)))
    return np.array(mean.astype(mx.float32)).transpose(0, 3, 1, 2)


def _tiny_readings(vae, arrays) -> dict:
    s = "48x64"
    with mx.stream(mx.cpu):
        dec = _decode(vae, arrays[f"latents_{s}"], tiled=False)
        lat = _latents(vae, arrays[f"image_{s}"], tiled=False)
        mean = _mean(vae, arrays[f"image_{s}"])
    return {"decode": fx.max_abs(dec, arrays[f"decoded_{s}"]),
            "latents": fx.max_abs(lat, arrays[f"latents_{s}"]),
            "mean": fx.max_abs(mean, arrays[f"encode_mean_{s}"])}


@pytest.mark.parametrize("layout", ["diffusers", "original"])
def test_the_tiny_fp32_decode_and_encode_are_within_max_abs(
        tiny, tiny_file, tiny_original, layout):
    from gmlx.image.vae import load_vae, read_header

    arrays, m = tiny
    path = tiny_file if layout == "diffusers" else tiny_original
    assert read_header(path).layout == layout
    vae = load_vae(path, family=FAMILY, dtype="fp32", conv="exact")
    assert vae.bn.running_var.dtype == mx.float32
    got = _tiny_readings(vae, arrays)
    print(f"[check5] flux3_klein tiny {layout} fp32: {got}, gate "
          f"{TINY_FP32_MAX_ABS}")
    assert m["tolerances"]["fp32_max_abs"] == TINY_FP32_MAX_ABS
    assert max(got.values()) <= TINY_FP32_MAX_ABS


def _bn_skipped(monkeypatch):
    from gmlx.image.vae import flux2_ae

    monkeypatch.setattr(flux2_ae.Flux2AE, "stats", lambda self, n: (
        mx.zeros((1, n, 1, 1)), mx.ones((1, n, 1, 1))))


def _dy_dx_swapped(monkeypatch):
    """The 2x2 patch read column first."""
    from gmlx.image.vae import flux2_ae

    def unpatchify(z, p):
        b, n, h, w = z.shape
        c = n // (p * p)
        z = z.reshape(b, c, p, p, h, w).transpose(0, 4, 3, 5, 2, 1)
        return z.reshape(b, h * p, w * p, c)

    def patchify(z, p):
        b, hh, ww, c = z.shape
        z = z.reshape(b, hh // p, p, ww // p, p, c).transpose(0, 5, 4, 2, 1, 3)
        return z.reshape(b, c * p * p, hh // p, ww // p)

    monkeypatch.setattr(flux2_ae, "unpatchify", unpatchify)
    monkeypatch.setattr(flux2_ae, "patchify", patchify)


def _patch_major(monkeypatch):
    """Channel ``(dy * 2 + dx) * C + c`` in place of ``c * 4 + dy * 2 +
    dx``."""
    from gmlx.image.vae import flux2_ae

    def unpatchify(z, p):
        b, n, h, w = z.shape
        c = n // (p * p)
        z = z.reshape(b, p, p, c, h, w).transpose(0, 4, 1, 5, 2, 3)
        return z.reshape(b, h * p, w * p, c)

    def patchify(z, p):
        b, hh, ww, c = z.shape
        z = z.reshape(b, hh // p, p, ww // p, p, c).transpose(0, 2, 4, 5, 1, 3)
        return z.reshape(b, c * p * p, hh // p, ww // p)

    monkeypatch.setattr(flux2_ae, "unpatchify", unpatchify)
    monkeypatch.setattr(flux2_ae, "patchify", patchify)


def _no_post_quant_conv(monkeypatch):
    from gmlx.image.vae import flux2_ae

    monkeypatch.setattr(flux2_ae.Flux2AE, "decode", lambda self, z: (
        self.decoder(z.astype(self.dtype))))


def _no_quant_conv(monkeypatch):
    from gmlx.image.vae import flux1_ae, flux2_ae

    monkeypatch.setattr(flux2_ae.Flux2AE, "split_moments",
                        flux1_ae.Flux1AE.split_moments)


WRONG = {"BatchNorm skipped": _bn_skipped, "dy and dx swapped": _dy_dx_swapped,
         "patch major channels": _patch_major,
         "post quant conv skipped": _no_post_quant_conv,
         "quant conv skipped": _no_quant_conv}


@pytest.mark.parametrize("name", list(WRONG),
                         ids=[n.replace(" ", "_") for n in WRONG])
def test_a_wrong_model_misses_the_tiny_gate(tiny, tiny_file, monkeypatch,
                                            name):
    from gmlx.image.vae import load_vae

    arrays, _ = tiny
    WRONG[name](monkeypatch)
    vae = load_vae(tiny_file, family=FAMILY, dtype="fp32", conv="exact")
    got = _tiny_readings(vae, arrays)
    print(f"[check5] flux3_klein tiny, {name}: {got}")
    assert max(got.values()) > 25 * TINY_FP32_MAX_ABS


def test_model_space_round_trips_and_refuses_odd_latents(tiny_file):
    from gmlx.image.vae import load_vae

    vae = load_vae(tiny_file, family=FAMILY, dtype="fp32")
    z = mx.random.normal((2, 6, 10, 32), key=mx.random.key(1))
    lat = vae.to_model_space(z)
    assert lat.shape == (2, 128, 3, 5)
    assert mx.allclose(vae.to_vae_space(lat), z, atol=1e-5).item()
    with pytest.raises(ValueError, match="do not patchify 2x2"):
        vae.to_model_space(z[:, :5])
    with pytest.raises(ValueError, match="this VAE takes 128"):
        vae.to_vae_space(lat[:, :32])


@pytest.mark.parametrize("layout", ["diffusers", "original"])
def test_a_packed_gguf_loads_the_same_weights(tiny_file, tiny_original,
                                              tmp_path, layout):
    import gguf

    from gmlx.image import preflight
    from gmlx.image.vae import load_vae, pack_vae, read_header

    src = tiny_file if layout == "diffusers" else tiny_original
    dst = tmp_path / "vae.gguf"
    rep = pack_vae(src, dst, source_repo="r", source_revision="v",
                   config_json="{}")
    assert rep.family == FAMILY and rep.lossless
    r = gguf.GGUFReader(str(dst))
    count = next(t for t in r.tensors if t.name.endswith(COUNT))
    assert count.tensor_type.name == "I64" and list(count.shape) == [1]
    head = read_header(str(dst))
    assert head.layout == layout and head.dtypes[COUNT] == "I64"
    assert preflight.classify(str(dst)) == "vae"
    a = dict(tree_flatten(load_vae(src, family=FAMILY,
                                   dtype="fp32").parameters()))
    b = dict(tree_flatten(load_vae(str(dst), family=FAMILY,
                                   dtype="fp32").parameters()))
    assert a.keys() == b.keys()
    assert [k for k in a if not mx.array_equal(a[k], b[k]).item()] == []
    # A downcast leaves the batch count an integer and is not lossless.
    rep16 = pack_vae(src, tmp_path / "bf16.gguf", source_repo="r",
                     source_revision="v", config_json="{}", dtype="BF16")
    assert not rep16.lossless
    assert read_header(str(tmp_path / "bf16.gguf")).dtypes[COUNT] == "I64"


def test_the_bf16_vae_keeps_its_batchnorm_statistics_in_float32(tiny_file):
    from gmlx.image.vae import load_vae

    vae = load_vae(tiny_file, family=FAMILY, dtype="bf16")
    kinds = {k: v.dtype for k, v in tree_flatten(vae.parameters())}
    assert kinds.pop("bn.running_mean") == kinds.pop("bn.running_var") \
        == mx.float32
    assert set(kinds.values()) == {mx.bfloat16}


@pytest.mark.parametrize("layout", ["diffusers", "original"])
def test_the_official_layouts_fill_the_module_tree(layout):
    """The names of each official file map onto every parameter of the
    module, the batch count is the one tensor left unread, and both files
    hold the same tensors at the same shapes."""
    from gmlx.image.vae import flux2_ae, reader

    head = load_json(HEADERS[layout])
    shapes = {n: tuple(s) for n, s, _ in head["tensors"]}
    dtypes = {n: d for n, _, d in head["tensors"]}
    got, names = reader.diffusers_names({n: n for n in shapes})
    assert got == layout and len(names) == len(shapes) == 251
    mapped = {}
    for n, f in names.items():
        s = shapes[f]
        if layout == "original" and ".attentions.0.to_" in n and len(s) == 4:
            s = s[:2]
        mapped[n] = s
    other = load_json(HEADERS["diffusers"])
    assert mapped == {n: tuple(s) for n, s, _ in other["tensors"]}
    header = reader.VaeHeader("x", "safetensors", layout, names, mapped,
                              {n: dtypes[f] for n, f in names.items()}, {})
    cfg = reader.identify(header, FAMILY)
    assert cfg.block_out_channels == (128, 256, 512, 512)
    assert (cfg.latent_channels, cfg.layers_per_block, cfg.model_channels) \
        == (32, 2, 128)
    params = dict(tree_flatten(flux2_ae.Flux2AE(cfg).parameters()))
    reader.check_names(header, params, reader.VAES[FAMILY], cfg)
    assert set(mapped) - set(params) == {COUNT}


def test_the_official_config_agrees_with_the_constants():
    from gmlx.image.vae import flux2_ae

    config = load_json("flux3_klein_vae_config.json")
    assert flux2_ae.config_disagreements(config) == []
    assert config["batch_norm_eps"] == flux2_ae.CONSTANTS["batch_norm_eps"]
    bad = dict(config, patch_size=[1, 1], batch_norm_eps=1e-5)
    assert flux2_ae.config_disagreements(bad) == ["batch_norm_eps",
                                                  "patch_size"]


def test_other_families_refuse_the_flux2_vae(tiny_file):
    from gmlx.image.vae import load_vae, reader

    for family, title in (("z_image", "the FLUX.1 autoencoder"),
                          ("krea2", "the Qwen-Image autoencoder"),
                          ("qwen_image21", "the Qwen-Image 2.1 autoencoder")):
        with pytest.raises(reader.VAEError, match="holds the FLUX.2 "
                           f"autoencoder, the flux3_klein VAE, and {family} "
                           f"decodes with {title}"):
            load_vae(tiny_file, family=family, dtype="fp32")


def test_flux3_klein_refuses_the_flux1_vae(tiny, tmp_path):
    """The tiny tree without its quant convs and BatchNorm is a FLUX.1
    autoencoder."""
    from gmlx.image.vae import flux1_ae, load_vae, reader

    flux1 = {k: v for k, v in _tiny_tensors(tiny).items()
             if not k.startswith(("quant_conv.", "post_quant_conv.", "bn."))}
    assert flux1_ae.config_from_shapes(
        {k: v.shape for k, v in flux1.items()}) is not None
    path = _write_st(tmp_path / "flux1.safetensors", flux1)
    with pytest.raises(reader.VAEError, match="holds the FLUX.1 autoencoder, "
                       "the z_image VAE, and flux3_klein decodes with the "
                       "FLUX.2 autoencoder"):
        load_vae(path, family=FAMILY, dtype="fp32")


def test_the_memory_row_prices_the_flux1_tree():
    from gmlx.image import memfit

    row = memfit.vae_row(FAMILY)
    z = memfit.vae_row("z_image")
    assert row.measured and row.whole == z.whole and row.tiled == z.tiled
    assert row.channels == {"decode": 128, "encode": 128}
    # Its tiles overlap by 128 pixels and are priced so.
    for op in ("decode", "encode"):
        assert memfit.tile_overlap(FAMILY, op) == 128
        assert memfit.tile_overlap(FAMILY, op, overlap=192) == 192
        assert memfit.tile_overlap("z_image", op) == 64
        for tiled in (False, True):
            assert memfit.vae_peak(op, 1024, 1024, family=FAMILY, itemsize=2,
                                   tiled=tiled) == memfit.vae_peak(
                op, 1024, 1024, family="z_image", itemsize=2, tiled=tiled,
                overlap=128)
        assert memfit.vae_peak(op, 1024, 1024, family=FAMILY, itemsize=2,
                               tiled=True) > memfit.vae_peak(
            op, 1024, 1024, family="z_image", itemsize=2, tiled=True)


@pytest.mark.parametrize("op", ["decode", "encode"])
def test_a_tiled_pass_takes_the_least_overlap_of_the_row(tiny, tiny_file,
                                                         monkeypatch, op):
    """The tiles of a tiled pass overlap by the row's least overlap where
    the overlap asked for is less. The tiny image is smaller than the 128
    pixels of the official row, so the row here asks for 24."""
    import dataclasses

    from gmlx.image import memfit
    from gmlx.image.vae import load_vae, tiling

    class Stop(Exception):
        pass

    seen = []

    def run_lockstep(steps, x, *, in_scale, out_scale, tile, overlap):
        seen.append((tile, overlap))
        raise Stop

    monkeypatch.setattr(tiling, "run_lockstep", run_lockstep)
    monkeypatch.setitem(memfit.VAE_ROWS, FAMILY, dataclasses.replace(
        memfit.vae_row(FAMILY), overlap=24))
    arrays, _ = tiny
    vae = load_vae(tiny_file, family=FAMILY, dtype="fp32")
    s = vae.config.scale
    with mx.stream(mx.cpu), pytest.raises(Stop):
        if op == "decode":
            _decode(vae, arrays["latents_48x64"], tiled=True, tile=32,
                    overlap=8, free=float(1 << 40))
        else:
            _latents(vae, arrays["image_48x64"], tiled=True, tile=32,
                     overlap=8, free=float(1 << 40))
    assert seen == [(32 // s, 24 // s)]


# The official files: need them under GMLX_TEST_IMAGE_DIR and the fixture of
# the committed manifest.

official = pytest.mark.integration
ROLES = {"diffusers": "vae", "original": "vae_original"}


@pytest.fixture(scope="module")
def check5():
    if not (ORACLE / f"{NAME}.manifest.json").is_file():
        pytest.skip(f"no committed {NAME} manifest")
    return fixture(NAME)


@pytest.fixture(scope="module")
def vaes(check5):
    """The official VAE of a layout in a dtype. One is held at a time, so a
    pass is priced against the memory of one VAE."""
    import gc

    from gmlx.image.vae import load_vae

    files = [p for p in vae_files(FAMILY) if p.suffix == ".safetensors"]
    sums = {sha256(p): p for p in files}
    held = {}

    def get(layout, dtype):
        want = check5[1]["files"][ROLES[layout]]["sha256"]
        if want not in sums:
            pytest.skip(f"the fixture's {layout} VAE is not under "
                        "GMLX_TEST_IMAGE_DIR")
        if (layout, dtype) not in held:
            held.clear()
            gc.collect()
            mx.clear_cache()
            held[layout, dtype] = load_vae(sums[want], family=FAMILY,
                                           dtype=dtype)
        return held[layout, dtype]

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


def _full(m) -> list:
    return [f"{h}x{w}" for h, w in m["inputs"]["sizes"]]


@official
@pytest.mark.parametrize("layout", ["diffusers", "original"])
def test_the_file_is_the_inventoried_one(check5, vaes, layout):
    from gmlx.image.vae import read_header, reader

    path = vaes(layout, "bf16").source
    want = load_json("manifest.json")[HEADERS[layout]]
    assert want["sha256"] == check5[1]["files"][ROLES[layout]]["sha256"]
    assert os.path.getsize(path) == want["size"]
    assert read_header(path).layout == layout
    raw, _ = reader.safetensors_header(path)
    raw.pop("__metadata__", None)
    assert sorted([n, v["shape"], v["dtype"]] for n, v in raw.items()) \
        == load_json(HEADERS[layout])["tensors"]


@official
def test_the_two_files_load_the_same_weights(vaes):
    a = {k: np.array(v) for k, v in
         tree_flatten(vaes("diffusers", "fp32").parameters())}
    b = dict(tree_flatten(vaes("original", "fp32").parameters()))
    assert a.keys() == b.keys()
    assert [k for k in a if not np.array_equal(a[k], np.array(b[k]))] == []


# The fp32 decode at 1024 pixels is priced at 8.45 GiB, more than a 16 GB
# Mac has free beside the weights; users decode in bf16.
@official
@pytest.mark.host_gb(24)
@pytest.mark.parametrize("layout", ["diffusers", "original"])
def test_fp32_decode_and_encode_are_within_the_gates(check5, vaes, request,
                                                     layout):
    arrays, m = check5
    vae = vaes(layout, "fp32")
    failed = []
    for s in _full(m):
        t = m["tolerances"]["sizes"][s]
        dec = _decode(vae, arrays[f"latents_{s}"], tiled=False)
        lat = _latents(vae, arrays[f"image_{s}"], tiled=False)
        d = fx.max_abs(dec, arrays[f"decoded_{s}"])
        e = fx.max_abs(lat, arrays[f"latents_{s}"])
        print(f"[check5] flux3_klein {layout} {s} fp32, TF32 "
              f"{'on' if _tf32() else 'off'}: decode max abs {d:.3e} of "
              f"gate {t['decode_fp32_max_abs']:.3e}, latents max abs "
              f"{e:.3e} of gate {t['encode_fp32_max_abs']:.3e}")
        if d > t["decode_fp32_max_abs"] or e > t["encode_fp32_max_abs"]:
            failed.append(s)
    if _tf32():
        _gate_without_tf32(request)
    else:
        assert not failed


@official
@pytest.mark.parametrize("layout", ["diffusers", "original"])
def test_bf16_decode_psnr_and_encode(check5, vaes, layout):
    arrays, m = check5
    vae = vaes(layout, "bf16")
    for s in _full(m):
        t = m["tolerances"]["sizes"][s]
        dec = _decode(vae, arrays[f"latents_{s}"], tiled=False)
        db = fx.psnr(to_unit(dec), to_unit(arrays[f"decoded_{s}"]), 1.0)
        rel = fx.rel_l2(_latents(vae, arrays[f"image_{s}"], tiled=False),
                        arrays[f"latents_{s}"])
        print(f"[check5] flux3_klein {layout} {s} bf16: decode PSNR "
              f"{db:.2f} dB, gate {t['decode_bf16_psnr_db']:.2f} dB; latents "
              f"rel L2 {rel:.3e}, gate {t['encode_bf16_rel_l2']:.3e}")
        assert db >= t["decode_bf16_psnr_db"], s
        assert rel <= t["encode_bf16_rel_l2"], s


@official
@pytest.mark.parametrize("layout", ["diffusers", "original"])
def test_a_tiled_bf16_decode_at_the_largest_size(check5, vaes, layout):
    arrays, m = check5
    s = _full(m)[-1]
    dec = _decode(vaes(layout, "bf16"), arrays[f"latents_{s}"], tiled=True)
    db = fx.psnr(to_unit(dec), to_unit(arrays[f"decoded_{s}"]), 1.0)
    tol = m["tolerances"]["sizes"][s]["decode_bf16_psnr_db"]
    print(f"[check5] flux3_klein {layout} {s} bf16 tiled decode PSNR "
          f"{db:.2f} dB, gate {tol:.2f} dB")
    assert s == "1024x1024"
    assert db >= tol


# A tiled fp32 decode at 2048 pixels is priced at 16.17 GiB, more than a
# 16 GB Mac has free.
@official
@pytest.mark.host_gb(24)
@pytest.mark.parametrize("layout", ["diffusers", "original"])
def test_a_tiled_bf16_decode_at_2048_holds_the_gate_of_1024(check5, vaes,
                                                            layout):
    """The 1024 latents of the fixture tiled two by two to 2048 pixels. The
    oracle decodes nothing that large, so the tiled bf16 decode is held to
    the bf16 PSNR gate of 1024 against the tiled fp32 decode."""
    arrays, m = check5
    s = _full(m)[-1]
    lat = np.tile(arrays[f"latents_{s}"], (1, 1, 2, 2))
    assert lat.shape[2:] == (128, 128)
    ref = _decode(vaes(layout, "fp32"), lat, tiled=True)
    got = _decode(vaes(layout, "bf16"), lat, tiled=True)
    db = fx.psnr(to_unit(got), to_unit(ref), 1.0)
    tol = m["tolerances"]["sizes"][s]["decode_bf16_psnr_db"]
    print(f"[check5] flux3_klein {layout} 2048x2048 bf16 tiled decode PSNR "
          f"{db:.2f} dB against the tiled fp32 decode, gate {tol:.2f} dB")
    assert db >= tol


@official
@pytest.mark.parametrize("layout", ["diffusers", "original"])
def test_a_tiled_bf16_encode_at_the_largest_full_size(check5, vaes, layout):
    arrays, m = check5
    s = _full(m)[-1]
    vae = vaes(layout, "bf16")
    whole = _latents(vae, arrays[f"image_{s}"], tiled=False)
    tiled = _latents(vae, arrays[f"image_{s}"], tiled=True)
    rel = fx.rel_l2(tiled, whole)
    a = _decode(vae, whole, tiled=False)
    b = _decode(vae, tiled, tiled=False)
    db = fx.psnr(to_unit(b), to_unit(a), 1.0)
    print(f"[check5] flux3_klein {layout} {s} bf16 tiled encode rel L2 "
          f"{rel:.3e} against the whole encode, round trip {db:.2f} dB "
          "against the whole round trip")
    assert s == "1024x1024"
    assert rel < TILED_ENCODE_REL_L2
    assert db >= TILED_ROUND_TRIP_DB
