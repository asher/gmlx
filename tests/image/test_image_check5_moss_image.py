"""Check 5 of the image acceptance gate for MOSS-Image: its VAE.

MOSS-Image decodes with the FLUX.2 autoencoder at narrower decoder widths,
[96, 192, 384, 384] where the encoder runs at [128, 256, 512, 512]. The
module, its wrong models and the helpers are those of
test_image_check5_flux3_klein.py.

The tiny self-test runs on the MLX CPU stream: diffusers'
AutoencoderKLFlux2 at a tiny config whose decoder is narrower than its
encoder, with seeded weights and BatchNorm statistics. The tiny rows read
it as MOSS-Image's VAE at the tiny decoder widths. Its fp32 decode, encoder
mean and model space latents are held to 5e-5 max abs, each wrong model
misses that, and a GGUF packed from it loads the same weights bit for bit.
The FLUX.2 families refuse it, MOSS-Image refuses the FLUX.2 autoencoder,
and a tree at other decoder widths is refused by its widths.

The official rows read the file under GMLX_TEST_IMAGE_DIR against the
oracle fixture at 512 and 1024 pixels: the fp32 decode and model space
latents to the manifest's max abs, the bf16 decode PSNR and latents
relative L2 to its gates, and a tiled bf16 decode at 1024 pixels to the
bf16 PSNR gate of that size. At 2048 pixels a tiled bf16 decode holds the
gate of 1024 against the tiled fp32 decode. A tiled bf16 encode at 1024
stays within 0.05 relative L2 of the whole encode, with its round trip at
50 dB PSNR against the whole round trip.
scripts/image_oracle/check5_flux3_klein_vae.py --family moss_image writes
the fixtures.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pytest
from image_dir import sha256, vae_files
from image_fixtures import load_json
from oracle_fixture import ORACLE, committed, committed_fixture, fixture, fx
from test_image_check5_flux3_klein import (
    TILED_ENCODE_REL_L2,
    TILED_ROUND_TRIP_DB,
    WRONG,
    _decode,
    _gate_without_tf32,
    _latents,
    _tf32,
    _tiny_readings,
    _write_st,
    to_unit,
)

import mlx.core as mx
from mlx.utils import tree_flatten

# The tiny rows run on a Mac of 16 GB, as most users run, and the rows on
# local files on this Mac (tests/host_box.py).

FAMILY = "moss_image"
KLEIN = ("flux3_klein", "flux2_klein", "flux2_klein_9b", "flux2_klein_9b_kv")
NAME = "check5_moss_image_vae"
TINY = "selftest_moss_image_vae_tiny"
FLUX2_TINY = "selftest_flux3_klein_vae_tiny"
TINY_FP32_MAX_ABS = 5e-5
HEADER = "moss_image_vae.header.json"
CONFIG = "moss_image_vae_config.json"
COUNT = "bn.num_batches_tracked"


def _tensors(fixture_) -> dict:
    arrays, m = fixture_
    w = {k[len("weight."):]: v for k, v in arrays.items()
         if k.startswith("weight.")}
    w[COUNT] = np.array(m["inputs"]["num_batches_tracked"], np.int64)
    return w


@pytest.fixture(scope="module")
def tiny():
    return committed_fixture(TINY)


@pytest.fixture
def tiny_file(tiny, tmp_path):
    return _write_st(tmp_path / "vae.safetensors", _tensors(tiny))


@pytest.fixture
def tiny_widths(tiny, monkeypatch):
    """MOSS-Image's VAE read at the decoder widths of the tiny tree."""
    from gmlx.image.vae import moss_image_ae

    widths = tuple(tiny[1]["inputs"]["config"]["decoder_block_out_channels"])
    monkeypatch.setattr(moss_image_ae, "DECODER", widths)
    monkeypatch.setitem(moss_image_ae.CONSTANTS, "decoder_block_out_channels",
                        widths)
    return widths


@pytest.mark.host_gb(16)
def test_the_tiny_fp32_decode_and_encode_are_within_max_abs(
        tiny, tiny_file, tiny_widths):
    from gmlx.image.vae import load_vae, read_header

    arrays, m = tiny
    cfg = m["inputs"]["config"]
    assert cfg["decoder_block_out_channels"] != cfg["block_out_channels"]
    assert read_header(tiny_file).layout == "diffusers"
    vae = load_vae(tiny_file, family=FAMILY, dtype="fp32", conv="exact")
    assert vae.config.decoder_channels == tiny_widths
    assert vae.bn.running_var.dtype == mx.float32
    got = _tiny_readings(vae, arrays)
    print(f"[check5] moss_image tiny fp32: {got}, gate {TINY_FP32_MAX_ABS}")
    assert m["tolerances"]["fp32_max_abs"] == TINY_FP32_MAX_ABS
    assert max(got.values()) <= TINY_FP32_MAX_ABS


@pytest.mark.host_gb(16)
@pytest.mark.parametrize("name", list(WRONG),
                         ids=[n.replace(" ", "_") for n in WRONG])
def test_a_wrong_model_misses_the_tiny_gate(tiny, tiny_file, tiny_widths,
                                            monkeypatch, name):
    from gmlx.image.vae import load_vae

    arrays, _ = tiny
    WRONG[name](monkeypatch)
    vae = load_vae(tiny_file, family=FAMILY, dtype="fp32", conv="exact")
    got = _tiny_readings(vae, arrays)
    print(f"[check5] moss_image tiny, {name}: {got}")
    assert max(got.values()) > 25 * TINY_FP32_MAX_ABS


@pytest.mark.host_gb(16)
def test_a_packed_gguf_loads_the_same_weights(tiny_file, tiny_widths,
                                              tmp_path, capsys):
    """pack-vae labels the file moss_image without --family, packs it under
    --family moss-image, and the pack loads the same weights. A pack whose
    config names other decoder widths is refused."""
    import json

    from gmlx.commands.image import cmd_pack_vae
    from gmlx.image.vae import VAEError, load_vae, pack_vae

    bare = pack_vae(tiny_file, tmp_path / "bare.gguf", source_repo="r",
                    source_revision="v", config_json="{}")
    assert bare.family == FAMILY and bare.lossless
    config = dict(load_json(CONFIG),
                  decoder_block_out_channels=list(tiny_widths))
    src_config = tmp_path / "vae-config.json"
    src_config.write_text(json.dumps(config))
    dst = tmp_path / "vae.gguf"
    assert cmd_pack_vae([tiny_file, "--out", str(dst), "--family",
                         "moss-image", "--source-config", str(src_config),
                         "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["family"] == FAMILY
    a = dict(tree_flatten(load_vae(tiny_file, family=FAMILY,
                                   dtype="fp32").parameters()))
    b = dict(tree_flatten(load_vae(str(dst), family=FAMILY,
                                   dtype="fp32").parameters()))
    assert a.keys() == b.keys()
    assert [k for k in a if not mx.array_equal(a[k], b[k]).item()] == []
    with pytest.raises(VAEError, match="its VAE config differs from the "
                       "MOSS-Image autoencoder in decoder_block_out_channels"):
        pack_vae(tiny_file, tmp_path / "other.gguf", source_repo="r",
                 source_revision="v", config_json=json.dumps(dict(
                     config, decoder_block_out_channels=[32, 32, 32])),
                 family=FAMILY)


@pytest.mark.host_gb(16)
def test_the_official_config_agrees_with_the_constants():
    from gmlx.image.vae import flux2_ae, moss_image_ae

    config = load_json(CONFIG)
    assert moss_image_ae.config_disagreements(config) == []
    assert moss_image_ae.CONSTANTS["decoder_block_out_channels"] \
        == moss_image_ae.DECODER == tuple(config["decoder_block_out_channels"])
    assert flux2_ae.config_disagreements(config) == [
        "decoder_block_out_channels"]
    bad = dict(config, decoder_block_out_channels=config["block_out_channels"])
    assert moss_image_ae.config_disagreements(bad) == [
        "decoder_block_out_channels"]
    assert flux2_ae.config_disagreements(bad) == []


@pytest.mark.host_gb(16)
def test_the_official_header_fills_the_module_tree():
    from gmlx.image.vae import flux2_ae, moss_image_ae, reader

    head = load_json(HEADER)
    shapes = {n: tuple(s) for n, s, _ in head["tensors"]}
    dtypes = {n: d for n, _, d in head["tensors"]}
    got, names = reader.diffusers_names({n: n for n in shapes})
    assert got == "diffusers" and len(names) == len(shapes) == 251
    header = reader.VaeHeader("x", "safetensors", got, names, shapes, dtypes,
                              {})
    cfg = reader.identify(header, FAMILY)
    assert cfg.block_out_channels == (128, 256, 512, 512)
    assert cfg.decoder_channels == moss_image_ae.DECODER
    assert (cfg.latent_channels, cfg.layers_per_block, cfg.model_channels) \
        == (32, 2, 128)
    params = dict(tree_flatten(flux2_ae.Flux2AE(cfg).parameters()))
    reader.check_names(header, params, reader.VAES[FAMILY], cfg)
    assert set(shapes) - set(params) == {COUNT}
    # The encoder side and the BatchNorm are the FLUX.2 autoencoder's.
    flux2 = {n: tuple(s) for n, s, _ in
             load_json("flux3_klein_vae.header.json")["tensors"]}
    assert flux2.keys() == shapes.keys()
    assert [n for n in shapes if shapes[n] != flux2[n]
            and not n.startswith(("decoder.", "post_quant_conv."))] == []


@pytest.mark.host_gb(16)
def test_the_flux2_families_refuse_the_moss_vae(tiny_file, tiny_widths):
    from gmlx.image.vae import load_vae, reader

    for family in KLEIN:
        with pytest.raises(reader.VAEError, match="holds the MOSS-Image "
                           f"autoencoder, the moss_image VAE, and {family} "
                           "decodes with the FLUX.2 autoencoder"):
            load_vae(tiny_file, family=family, dtype="fp32")


@pytest.mark.host_gb(16)
def test_moss_image_refuses_the_flux2_vae(tmp_path):
    from gmlx.image.vae import load_vae, reader

    path = _write_st(tmp_path / "flux2.safetensors",
                     _tensors(committed_fixture(FLUX2_TINY)))
    with pytest.raises(reader.VAEError, match="holds the FLUX.2 autoencoder, "
                       "the flux3_klein VAE, and moss_image decodes with the "
                       "MOSS-Image autoencoder"):
        load_vae(path, family=FAMILY, dtype="fp32")


@pytest.mark.host_gb(16)
@pytest.mark.parametrize("family", [FAMILY, "flux3_klein", "z_image"])
def test_a_tree_at_other_decoder_widths_is_refused_by_its_widths(
        tiny, tiny_file, family):
    """The tiny tree at MOSS-Image's own widths reads as a relative, and
    without its quant convs and BatchNorm the FLUX.1 tree refuses it."""
    from gmlx.image.vae import flux1_ae, load_vae, reader

    widths = tiny[1]["inputs"]["config"]["decoder_block_out_channels"]
    with pytest.raises(reader.VAEError, match=re.escape(
            f"holds a FLUX.2 autoencoder with decoder widths {widths}, and "
            f"{family} decodes with")):
        load_vae(tiny_file, family=family, dtype="fp32")
    flux1 = {k: v.shape for k, v in _tensors(tiny).items()
             if not k.startswith(("quant_conv.", "post_quant_conv.", "bn."))}
    assert flux1_ae.config_from_shapes(flux1) is None


@pytest.mark.host_gb(16)
def test_the_memory_row_prices_its_own_decoder():
    """The encode and the tiles are priced as the FLUX.2 autoencoder's, the
    tiles at the decoder's 96 channels, and the whole decode by its own
    fits, which price the fp32 decode at 512 pixels that the FLUX.1 fits
    at 96 channels price under its reading of 4.08 GiB."""
    import dataclasses

    from gmlx.image import memfit

    GiB = 1 << 30
    row = memfit.vae_row(FAMILY)
    assert row.measured and row.channels == {"decode": 96, "encode": 128}
    assert row.whole["decode"] is memfit.MOSS_VAE_DECODE
    assert row.whole["encode"] is memfit.VAE_ENCODE
    assert row.tiled is memfit.VAE_TILED
    for op in ("decode", "encode"):
        assert memfit.tile_overlap(FAMILY, op) == 128
    for tiled in (False, True):
        assert memfit.vae_peak("encode", 1024, 1024, family=FAMILY,
                               itemsize=2, tiled=tiled) == memfit.vae_peak(
            "encode", 1024, 1024, family="flux3_klein", itemsize=2,
            tiled=tiled)
    assert memfit.vae_peak("decode", 2048, 2048, family=FAMILY, itemsize=2,
                           tiled=True) < memfit.vae_peak(
        "decode", 2048, 2048, family="flux3_klein", itemsize=2, tiled=True)
    for conv in ("kq", "exact"):
        assert memfit.vae_peak("decode", 512, 512, family=FAMILY, itemsize=4,
                               conv=conv) >= (4.08 + 0.25) * GiB
    flux1 = dataclasses.replace(row, whole={"decode": memfit.VAE_DECODE,
                                            "encode": memfit.VAE_ENCODE})
    fit = flux1.whole["decode"]["exact"][4]
    assert fit.bytes(512 * 512, 96, 4) < 4.08 * GiB


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
    """The official VAE in a dtype. One is held at a time, so a pass is
    priced against the memory of one VAE."""
    import gc

    from gmlx.image.vae import load_vae

    sums = {sha256(p): p for p in vae_files(FAMILY)
            if p.suffix == ".safetensors"}
    held = {}

    def get(dtype):
        want = check5[1]["files"]["vae"]["sha256"]
        if want not in sums:
            pytest.skip("the fixture's VAE is not under GMLX_TEST_IMAGE_DIR")
        if dtype not in held:
            held.clear()
            gc.collect()
            mx.clear_cache()
            held[dtype] = load_vae(sums[want], family=FAMILY, dtype=dtype)
        return held[dtype]

    return get


def _full(m) -> list:
    return [f"{h}x{w}" for h, w in m["inputs"]["sizes"]]


@pytest.mark.host_gb(16)
def test_the_manifest_is_the_moss_families():
    m = committed(NAME)
    assert m["family"] == FAMILY
    assert m["argv"][m["argv"].index("--family") + 1] == FAMILY
    assert m["files"]["vae"]["sha256"] == load_json("manifest.json")[
        HEADER]["sha256"]
    assert m["files"]["vae_config"]["sha256"] == load_json("manifest.json")[
        CONFIG]["sha256"]


@official
def test_the_file_is_the_inventoried_one(check5, vaes):
    from gmlx.image.vae import read_header, reader

    path = vaes("bf16").source
    want = load_json("manifest.json")[HEADER]
    assert want["sha256"] == check5[1]["files"]["vae"]["sha256"]
    assert os.path.getsize(path) == want["size"]
    assert read_header(path).layout == "diffusers"
    raw, _ = reader.safetensors_header(path)
    raw.pop("__metadata__", None)
    assert sorted([n, v["shape"], v["dtype"]] for n, v in raw.items()) \
        == load_json(HEADER)["tensors"]
    config = Path(path).with_name("config.json")
    if config.is_file():
        assert sha256(config) == check5[1]["files"]["vae_config"]["sha256"]


@official
def test_fp32_decode_and_encode_are_within_the_gates(check5, vaes, request):
    arrays, m = check5
    vae = vaes("fp32")
    failed = []
    for s in _full(m):
        t = m["tolerances"]["sizes"][s]
        dec = _decode(vae, arrays[f"latents_{s}"], tiled=False)
        lat = _latents(vae, arrays[f"image_{s}"], tiled=False)
        d = fx.max_abs(dec, arrays[f"decoded_{s}"])
        e = fx.max_abs(lat, arrays[f"latents_{s}"])
        print(f"[check5] moss_image {s} fp32, TF32 "
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
def test_bf16_decode_psnr_and_encode(check5, vaes):
    arrays, m = check5
    vae = vaes("bf16")
    for s in _full(m):
        t = m["tolerances"]["sizes"][s]
        dec = _decode(vae, arrays[f"latents_{s}"], tiled=False)
        db = fx.psnr(to_unit(dec), to_unit(arrays[f"decoded_{s}"]), 1.0)
        rel = fx.rel_l2(_latents(vae, arrays[f"image_{s}"], tiled=False),
                        arrays[f"latents_{s}"])
        print(f"[check5] moss_image {s} bf16: decode PSNR {db:.2f} dB, gate "
              f"{t['decode_bf16_psnr_db']:.2f} dB; latents rel L2 "
              f"{rel:.3e}, gate {t['encode_bf16_rel_l2']:.3e}")
        assert db >= t["decode_bf16_psnr_db"], s
        assert rel <= t["encode_bf16_rel_l2"], s


@official
def test_a_tiled_bf16_decode_at_the_largest_size(check5, vaes):
    arrays, m = check5
    s = _full(m)[-1]
    dec = _decode(vaes("bf16"), arrays[f"latents_{s}"], tiled=True)
    db = fx.psnr(to_unit(dec), to_unit(arrays[f"decoded_{s}"]), 1.0)
    tol = m["tolerances"]["sizes"][s]["decode_bf16_psnr_db"]
    print(f"[check5] moss_image {s} bf16 tiled decode PSNR {db:.2f} dB, "
          f"gate {tol:.2f} dB")
    assert s == "1024x1024"
    assert db >= tol


@official
def test_a_tiled_bf16_decode_at_2048_holds_the_gate_of_1024(check5, vaes):
    """The 1024 latents of the fixture tiled two by two to 2048 pixels,
    held to the bf16 PSNR gate of 1024 against the tiled fp32 decode."""
    arrays, m = check5
    s = _full(m)[-1]
    lat = np.tile(arrays[f"latents_{s}"], (1, 1, 2, 2))
    assert lat.shape[2:] == (128, 128)
    ref = _decode(vaes("fp32"), lat, tiled=True)
    got = _decode(vaes("bf16"), lat, tiled=True)
    db = fx.psnr(to_unit(got), to_unit(ref), 1.0)
    tol = m["tolerances"]["sizes"][s]["decode_bf16_psnr_db"]
    print(f"[check5] moss_image 2048x2048 bf16 tiled decode PSNR {db:.2f} "
          f"dB against the tiled fp32 decode, gate {tol:.2f} dB")
    assert db >= tol


@official
def test_a_tiled_bf16_encode_at_the_largest_full_size(check5, vaes):
    arrays, m = check5
    s = _full(m)[-1]
    vae = vaes("bf16")
    whole = _latents(vae, arrays[f"image_{s}"], tiled=False)
    tiled = _latents(vae, arrays[f"image_{s}"], tiled=True)
    rel = fx.rel_l2(tiled, whole)
    a = _decode(vae, whole, tiled=False)
    b = _decode(vae, tiled, tiled=False)
    db = fx.psnr(to_unit(b), to_unit(a), 1.0)
    print(f"[check5] moss_image {s} bf16 tiled encode rel L2 {rel:.3e} "
          f"against the whole encode, round trip {db:.2f} dB against the "
          "whole round trip")
    assert s == "1024x1024"
    assert rel < TILED_ENCODE_REL_L2
    assert db >= TILED_ROUND_TRIP_DB


@official
def test_the_gguf_pack_loads_the_same_weights_and_decodes_the_same(
        check5, vaes, tmp_path):
    """The F32 file packs losslessly with its config, and its pack decodes
    the 512 latents bit for bit as the file does."""
    from gmlx.image.vae import load_vae, pack_vae

    arrays, m = check5
    a = vaes("bf16")
    dst = tmp_path / "moss-image-vae-F32.gguf"
    rep = pack_vae(a.source, dst, source_repo="OpenMOSS-Team/MOSS-Image-SFT",
                   source_revision=m["inputs"]["config_revision"],
                   config_json=Path(ORACLE.parent / CONFIG).read_text())
    assert rep.lossless and rep.family == FAMILY
    b = load_vae(str(dst), family=FAMILY, dtype="bf16")
    pa = dict(tree_flatten(a.parameters()))
    pb = dict(tree_flatten(b.parameters()))
    assert pa.keys() == pb.keys()
    assert [k for k in pa if not mx.array_equal(pa[k], pb[k]).item()] == []
    s = _full(m)[0]
    assert np.array_equal(_decode(a, arrays[f"latents_{s}"], tiled=False),
                          _decode(b, arrays[f"latents_{s}"], tiled=False))
