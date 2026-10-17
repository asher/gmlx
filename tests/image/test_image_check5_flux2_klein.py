"""Check 5 of the image acceptance gate for FLUX.2 klein: BFL's BF16 VAE.

FLUX.2 klein and FLUX.3 klein decode with one autoencoder, whose module,
tiny fixture and wrong models test_image_check5_flux3_klein.py holds. gmlx
pulls it for both from BFL's FLUX.2 klein 4B repository, in BF16. Here the
tiny fixture written in BF16 packs losslessly in BF16 for either family and
loads for both, and its F32 file and BF16 rounding load the same bf16 VAE
but for the BatchNorm statistics, which a bf16 VAE keeps float32 from an F32
file.

The official rows read BFL's file under GMLX_TEST_IMAGE_DIR against the
oracle fixture at 512 and 1024 pixels, which the oracle wrote from that
file read upcast: the fp32 decode and model space latents to the
manifest's max abs, the bf16 decode PSNR and latents relative L2 to its
gates, a tiled bf16 decode at 1024 pixels to the bf16 PSNR gate of that
size, and a tiled bf16 encode at 1024 within 0.05 relative L2 of the whole
encode with a round trip at 50 dB PSNR against the whole round trip. A
GGUF packed from the file loads the same weights bit for bit and decodes
the same. Beside the Comfy-Org F32 file of FLUX.3's fixture, BFL's file is
its bf16 rounding, and the bf16 decodes of the two differ through the
BatchNorm statistics alone. scripts/image_oracle/check5_flux3_klein_vae.py
writes the fixture.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from image_dir import sha256, vae_files
from image_fixtures import load_json, official_config
from oracle_fixture import ORACLE, committed, committed_fixture, fixture, fx

import mlx.core as mx
from mlx.utils import tree_flatten

# A Mac of 16 GB, as most users run (tests/host_box.py).
pytestmark = pytest.mark.host_gb(16)

FAMILY = "flux2_klein"
OTHER = "flux3_klein"
# Every family that decodes with the FLUX.2 autoencoder.
KLEIN = (FAMILY, OTHER, "flux2_klein_9b")
NAME = "check5_flux2_klein_vae"
F32_NAME = "check5_flux3_klein_vae"
TINY = "selftest_flux3_klein_vae_tiny"
HEADER = "flux2_klein_vae.header.json"
COUNT = "bn.num_batches_tracked"
BN = ("bn.running_mean", "bn.running_var")
TILED_ENCODE_REL_L2 = 0.05
TILED_ROUND_TRIP_DB = 50.0


def to_unit(img: np.ndarray) -> np.ndarray:
    return np.clip(img.astype(np.float64) / 2 + 0.5, 0.0, 1.0)


def bf16_bits(a) -> np.ndarray:
    """The bfloat16 bits of ``a`` rounded to nearest even, as uint16."""
    u = np.ascontiguousarray(a, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def _write_st(path, tensors: dict, *, bf16: bool) -> str:
    """``tensors`` as a safetensors file, names sorted: int64 arrays as
    I64, others as BF16 when ``bf16``, else F32."""
    head, blobs, off = {}, [], 0
    for k in sorted(tensors):
        a = np.asarray(tensors[k])
        if a.dtype == np.int64:
            kind = "I64"
        elif bf16:
            a, kind = bf16_bits(a), "BF16"
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
def tiny_f32(tiny, tmp_path):
    return _write_st(tmp_path / "f32.safetensors", _tiny_tensors(tiny),
                     bf16=False)


@pytest.fixture
def tiny_bf16(tiny, tmp_path):
    return _write_st(tmp_path / "bf16.safetensors", _tiny_tensors(tiny),
                     bf16=True)


def _decode(vae, latents, **kw) -> np.ndarray:
    from gmlx.image.vae import tiling

    return np.array(tiling.decode(vae, latents, **kw)).transpose(0, 3, 1, 2)


def _latents(vae, image, **kw) -> np.ndarray:
    from gmlx.image.vae import tiling

    return np.array(tiling.encode(vae, image.transpose(0, 2, 3, 1), **kw))


def _params(vae) -> dict:
    return dict(tree_flatten(vae.parameters()))


def _differ(a: dict, b: dict) -> list:
    assert a.keys() == b.keys()
    return [k for k in a if not mx.array_equal(a[k], b[k]).item()]


def _bn_rounded(vae) -> None:
    """The BatchNorm statistics of ``vae`` rounded to bfloat16, in float32,
    as a bf16 VAE holds them from a BF16 file."""
    for n in ("running_mean", "running_var"):
        v = getattr(vae.bn, n)
        setattr(vae.bn, n, v.astype(mx.bfloat16).astype(mx.float32))


def test_the_tiny_bf16_file_packs_losslessly_and_serves_both_families(
        tiny_bf16, tmp_path):
    """BF16 stays BF16 and the batch count I64. A pack for either klein
    family loads for every klein family, bit for bit with the file."""
    import gguf

    from gmlx.image import catalog, preflight
    from gmlx.image.vae import load_vae, pack_vae, read_header

    assert read_header(tiny_bf16).dtypes["bn.running_var"] == "BF16"
    src = {d: _params(load_vae(tiny_bf16, family=FAMILY, dtype=d))
           for d in ("fp32", "bf16")}
    for fam in (FAMILY, OTHER):
        dst = tmp_path / f"{fam}.gguf"
        rep = pack_vae(tiny_bf16, dst, source_repo="r", source_revision="v",
                       config_json="{}", family=fam)
        assert rep.family == fam and rep.lossless and rep.dtype == "source"
        r = gguf.GGUFReader(str(dst))
        kinds = {t.name: t.tensor_type.name for t in r.tensors}
        assert kinds.pop(COUNT) == "I64" and set(kinds.values()) == {"BF16"}
        assert r.fields["gmlx.vae.family"].contents() == fam
        assert "gmlx.vae.lossless" not in r.fields
        assert preflight.classify(str(dst)) == "vae"
        assert catalog.vae_families(fam) == (
            fam, *(f for f in KLEIN if f != fam))
        for use in KLEIN:
            for d in ("fp32", "bf16"):
                got = _params(load_vae(str(dst), family=use, dtype=d))
                assert _differ(src[d], got) == [], (fam, use, d)


def test_a_bare_pack_takes_the_label_the_help_and_the_cli_page_name(
        tiny_bf16, tmp_path):
    """Without --family the FLUX.2 autoencoder packs for the label that
    the pack-vae help and docs/cli.md name."""
    from gmlx.commands.image import _pack_vae_parser
    from gmlx.image.vae import pack_vae

    label = pack_vae(tiny_bf16, tmp_path / "bare.gguf", source_repo="r",
                     source_revision="v", config_json="{}").family
    assert label == OTHER
    help_text = " ".join(_pack_vae_parser("p").format_help().split())
    assert f"and {label} for the FLUX.2 autoencoder" in help_text
    page = (Path(__file__).resolve().parents[2] / "docs" / "cli.md").read_text()
    section = page.split("### image pack-vae", 1)[1].split("\n## ", 1)[0]
    assert f"without `--family` is labeled `{label}`" in " ".join(
        section.split())
    row = [ln for ln in section.splitlines() if ln.startswith("| `--family")]
    assert f"`{label}` for the FLUX.2 autoencoder" in row[0]
    assert "`flux-2-klein`" in row[0]


def test_the_manifest_names_the_family_it_was_written_for():
    """FLUX.3's oracle script wrote FLUX.2's fixture before it took
    --family, and the manifest notes the family corrected after the run."""
    m = committed(NAME)
    assert m["family"] == FAMILY
    assert "--family" not in m["argv"]
    assert any("corrected to flux2_klein" in n for n in m["notes"])
    assert committed(F32_NAME)["family"] == OTHER


def test_the_bf16_vae_of_an_f32_file_and_of_its_bf16_rounding(
        tiny, tiny_f32, tiny_bf16):
    """The bf16 VAE of each holds the same weights, the BatchNorm statistics
    apart: full float32 from the F32 file, bf16 values from the BF16 one.
    With those rounded the two decode bit for bit the same."""
    from gmlx.image.vae import load_vae

    arrays, _ = tiny
    a = load_vae(tiny_f32, family=FAMILY, dtype="bf16")
    b = load_vae(tiny_bf16, family=FAMILY, dtype="bf16")
    pa, pb = _params(a), _params(b)
    assert pa[BN[1]].dtype == pb[BN[1]].dtype == mx.float32
    assert sorted(_differ(pa, pb)) == list(BN)
    with mx.stream(mx.cpu):
        x = arrays["latents_48x64"]
        before = _decode(a, x, tiled=False)
        _bn_rounded(a)
        assert _differ(_params(a), pb) == []
        after = _decode(a, x, tiled=False)
        want = _decode(b, x, tiled=False)
    assert not np.array_equal(before, want)
    assert np.array_equal(after, want)


def test_bfls_config_agrees_with_the_constants():
    from gmlx.image.vae import flux2_ae

    config = load_json("flux2_klein_vae_config.json")
    assert flux2_ae.config_disagreements(config) == []
    # FLUX.3 klein's copy, saved by a later diffusers, adds one key unset.
    ours, other = (official_config(n) for n in (
        "flux2_klein_vae_config.json", "flux3_klein_vae_config.json"))
    assert {k: other[k] for k in ours} == ours
    assert {k: v for k, v in other.items() if k not in ours} == {
        "decoder_block_out_channels": None}


def test_bfls_header_holds_the_flux3_files_tensors_in_bf16():
    """The names and shapes of the Comfy-Org F32 file, every float tensor
    BF16, and every parameter of the module filled."""
    from gmlx.image.vae import flux2_ae, reader

    head = load_json(HEADER)
    shapes = {n: tuple(s) for n, s, _ in head["tensors"]}
    dtypes = {n: d for n, _, d in head["tensors"]}
    other = load_json("flux3_klein_vae.header.json")
    assert shapes == {n: tuple(s) for n, s, _ in other["tensors"]}
    floats = {n: d for n, d in dtypes.items() if n != COUNT}
    assert dtypes[COUNT] == "I64" and set(floats.values()) == {"BF16"}
    got, names = reader.diffusers_names({n: n for n in shapes})
    assert got == "diffusers" and len(names) == 251
    header = reader.VaeHeader("x", "safetensors", got, names, shapes, dtypes,
                              {})
    cfg = reader.identify(header, FAMILY)
    params = dict(tree_flatten(flux2_ae.Flux2AE(cfg).parameters()))
    reader.check_names(header, params, reader.VAES[FAMILY], cfg)
    assert set(shapes) - set(params) == {COUNT}


# The official files: need them under GMLX_TEST_IMAGE_DIR and the fixture of
# the committed manifest.

official = pytest.mark.integration


@pytest.fixture(scope="module")
def check5():
    if not (ORACLE / f"{NAME}.manifest.json").is_file():
        pytest.skip(f"no committed {NAME} manifest")
    return fixture(NAME)


@pytest.fixture(scope="module")
def sources():
    """The safetensors VAE files under GMLX_TEST_IMAGE_DIR by sha256."""
    return {sha256(p): p for p in vae_files(FAMILY)
            if p.suffix == ".safetensors"}


def _source(sources, sha: str, what: str):
    if sha not in sources:
        pytest.skip(f"{what} is not under GMLX_TEST_IMAGE_DIR")
    return sources[sha]


@pytest.fixture(scope="module")
def vaes(check5, sources):
    """BFL's VAE in a dtype. One is held at a time, so a pass is priced
    against the memory of one VAE."""
    import gc

    from gmlx.image.vae import load_vae

    held = {}

    def get(dtype):
        path = _source(sources, check5[1]["files"]["vae"]["sha256"],
                       "the fixture's VAE")
        if dtype not in held:
            held.clear()
            gc.collect()
            mx.clear_cache()
            held[dtype] = load_vae(path, family=FAMILY, dtype=dtype)
        return held[dtype]

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
def test_the_file_is_the_inventoried_one(check5, vaes):
    from gmlx.image import catalog
    from gmlx.image.vae import read_header, reader

    path = vaes("bf16").source
    want = load_json("manifest.json")[HEADER]
    m = check5[1]
    assert m["family"] == FAMILY
    assert OTHER in catalog.vae_families(FAMILY)
    assert want["sha256"] == m["files"]["vae"]["sha256"]
    assert m["files"]["vae_config"]["sha256"] == load_json("manifest.json")[
        "flux2_klein_vae_config.json"]["sha256"]
    assert os.path.getsize(path) == want["size"]
    assert m["inputs"]["config_revision"] == want["revision"]
    assert m["inputs"]["source_dtypes"] == ["torch.bfloat16", "torch.int64"]
    assert read_header(path).layout == "diffusers"
    raw, _ = reader.safetensors_header(path)
    raw.pop("__metadata__", None)
    assert sorted([n, v["shape"], v["dtype"]] for n, v in raw.items()) \
        == load_json(HEADER)["tensors"]


# The fp32 decode at 1024 pixels is priced at 8.45 GiB, more than a 16 GB
# Mac has free beside the weights; users decode in bf16.
@official
@pytest.mark.host_gb(24)
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
        print(f"[check5] flux2_klein {s} fp32, TF32 "
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
        print(f"[check5] flux2_klein {s} bf16: decode PSNR {db:.2f} dB, "
              f"gate {t['decode_bf16_psnr_db']:.2f} dB; latents rel L2 "
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
    print(f"[check5] flux2_klein {s} bf16 tiled decode PSNR {db:.2f} dB, "
          f"gate {tol:.2f} dB")
    assert s == "1024x1024"
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
    print(f"[check5] flux2_klein {s} bf16 tiled encode rel L2 {rel:.3e} "
          f"against the whole encode, round trip {db:.2f} dB against the "
          "whole round trip")
    assert s == "1024x1024"
    assert rel < TILED_ENCODE_REL_L2
    assert db >= TILED_ROUND_TRIP_DB


@official
def test_the_gguf_pack_loads_the_same_weights_and_decodes_the_same(
        check5, vaes, tmp_path):
    """BFL's file packs in BF16, lossless, and its pack decodes the 512
    latents bit for bit as the file does."""
    from gmlx.image.vae import load_vae, pack_vae

    arrays, m = check5
    a = vaes("bf16")
    dst = tmp_path / "flux2-vae-BF16.gguf"
    rep = pack_vae(a.source, dst, source_repo="black-forest-labs/"
                   "FLUX.2-klein-4B", source_revision=m["inputs"][
                       "config_revision"], config_json="{}", family=FAMILY)
    assert rep.lossless and rep.family == FAMILY
    b = load_vae(str(dst), family=FAMILY, dtype="bf16")
    assert _differ(_params(a), _params(b)) == []
    s = _full(m)[0]
    assert np.array_equal(_decode(a, arrays[f"latents_{s}"], tiled=False),
                          _decode(b, arrays[f"latents_{s}"], tiled=False))


@official
def test_the_bf16_file_is_the_f32_files_rounding_and_decodes_apart_by_bn(
        check5, sources):
    """BFL's BF16 file holds the bf16 rounding of each float tensor of the
    Comfy-Org F32 file. The bf16 VAEs of the two hold the same weights but
    the BatchNorm statistics, which stay full float32 from the F32 file, so
    their bf16 decodes are not bit for bit the same; with the F32 file's
    statistics rounded they are, at each size of the fixture."""
    import gc

    from gmlx.image.vae import load_vae

    arrays, m = check5
    want = committed(F32_NAME)["files"]["vae"]
    f32 = _source(sources, want["sha256"], "the Comfy-Org F32 VAE")
    bf16 = _source(sources, m["files"]["vae"]["sha256"], "BFL's BF16 VAE")
    a = mx.load(str(f32))
    b = mx.load(str(bf16))
    assert a.keys() == b.keys()
    floats = [k for k in a if a[k].dtype == mx.float32]
    assert len(floats) == 250 and all(b[k].dtype == mx.bfloat16
                                      for k in floats)
    off = [k for k in floats if not np.array_equal(
        bf16_bits(np.array(a[k])), np.array(b[k].view(mx.uint16)))]
    assert off == []
    del a, b
    gc.collect()
    va = load_vae(str(f32), family=FAMILY, dtype="bf16")
    vb = load_vae(str(bf16), family=FAMILY, dtype="bf16")
    assert sorted(_differ(_params(va), _params(vb))) == list(BN)
    for s in _full(m):
        x = arrays[f"latents_{s}"]
        da = _decode(va, x, tiled=False)
        db = _decode(vb, x, tiled=False)
        psnr = fx.psnr(to_unit(da), to_unit(db), 1.0)
        print(f"[check5] flux2_klein {s} bf16 decode, F32 file against BF16 "
              f"file: max abs {fx.max_abs(da, db):.3e}, PSNR {psnr:.2f} dB, "
              f"bit for bit {np.array_equal(da, db)}")
        assert not np.array_equal(da, db), s
    _bn_rounded(va)
    assert _differ(_params(va), _params(vb)) == []
    for s in _full(m):
        x = arrays[f"latents_{s}"]
        assert np.array_equal(_decode(va, x, tiled=False),
                              _decode(vb, x, tiled=False)), s
