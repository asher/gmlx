"""Check 5 of the image acceptance gate for Z-Image: the FLUX.1 VAE.

gmlx decodes the oracle's latents and encodes its image with the official
VAE, read from a GGUF packed from the oracle's safetensors when one is
found, else from that safetensors file. The fp32 decode must stay within
the manifest's max abs and the bf16 decode above its PSNR, both against
the oracle's fp32 decode. Decodes run on both conv paths, and the gates
hold the default path. Each decode is forced whole or in tiles, so a gate
reads the pass its tolerance was set on. The automatic choice tiles a
decode that does not fit, and a tiled fp32 decode at 1024 reads 2.0e-2
max abs against the 1e-3 gate of the whole decode. The encoder mean is
held to the calibrated max abs in fp32 and relative L2 in bf16. A tiled
bf16 decode at the largest size must stay above the calibrated bf16
decode PSNR of that size. A tiled bf16 encode at that size stays within
0.05 relative L2 of the whole encode, and its round trip holds at least
50 dB PSNR against the whole round trip.
On the oracle's synthetic 1024 image the bf16 VAE measured 0.0307 and
55.44 dB; a natural 1024 image measured 0.0103 and 53.46 dB.

The fp32 VAE runs only on chips without TF32, so the fp32 items are gated
with MLX_ENABLE_TF32=0. MLX reads it once per process, so with TF32 on the
fp32 items report their numbers as attribution and rerun in a child
process with TF32 off, which holds the gate. The bf16 items run on the
session's numerics. Skipped unless GMLX_TEST_IMAGE_DIR holds the VAE and
GMLX_IMAGE_FIXTURES the fixture. scripts/image_oracle/check5_z_image_vae.py
writes it."""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest
from image_dir import sha256, vae_files
from oracle_fixture import fixture, fx

# Each row loads local model files, so it runs on this Mac
# (tests/host_box.py).
pytestmark = pytest.mark.integration

NAME = "check5_z_image_vae"
FAMILY = "z_image"
TILED_ENCODE_REL_L2 = 0.05
TILED_ROUND_TRIP_DB = 50.0


def to_unit(img: np.ndarray) -> np.ndarray:
    return np.clip(img.astype(np.float64) / 2 + 0.5, 0.0, 1.0)


@pytest.fixture(scope="module")
def check5():
    return fixture(NAME)


@pytest.fixture(scope="module")
def vae_path(check5):
    from gmlx.image.vae import read_header

    want = check5[1]["files"]["vae"]["sha256"]
    files = vae_files(FAMILY)
    for p in files:
        if p.suffix == ".gguf" and read_header(str(p)).metadata.get(
                "gmlx.vae.source_sha256") == want:
            return p
    for p in files:
        if p.suffix == ".safetensors" and sha256(p) == want:
            return p
    pytest.skip("the fixture's VAE is not under GMLX_TEST_IMAGE_DIR")


@pytest.fixture(scope="module")
def vaes(vae_path):
    from gmlx.image.vae import load_vae

    cache = {}

    def get(dtype, conv=None):
        if (dtype, conv) not in cache:
            cache[dtype, conv] = load_vae(vae_path, family=FAMILY,
                                          dtype=dtype, conv=conv)
        return cache[dtype, conv]

    return get


def _decode(vae, latents, **kw) -> np.ndarray:
    from gmlx.image.vae import tiling

    out = tiling.decode(vae, latents, **kw)
    return np.array(out).transpose(0, 3, 1, 2)


def _sizes(m) -> list:
    return [str(s) for s in m["inputs"]["sizes"]]


def _tf32() -> bool:
    from block_gates import tf32

    return tf32()


def _numerics() -> str:
    return "TF32 on" if _tf32() else "TF32 off"


def _gate_without_tf32(request) -> None:
    """Rerun this test in a child process with TF32 off, where it gates."""
    env = dict(os.environ, MLX_ENABLE_TF32="0")
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-s", "-p", "no:cacheprovider",
         request.node.nodeid], cwd=request.config.rootpath, env=env,
        capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if "[check5]" in line:
            print(line[line.index("[check5]"):])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]


def test_latent_space_constants_are_the_oracle_ones(check5, vaes):
    _, m = check5
    cfg = vaes("fp32").config
    assert cfg.scaling_factor == m["inputs"]["scaling_factor"]
    assert cfg.shift_factor == m["inputs"]["shift_factor"]


# Both conv paths are reported, and the gates hold the default one.


def test_fp32_decode_is_within_max_abs(check5, vaes, vae_path, request):
    from gmlx.image.vae.conv import CONV, CONV_PATHS

    arrays, m = check5
    failed = []
    for conv in CONV_PATHS:
        vae = vaes("fp32", conv)
        for s in _sizes(m):
            got = _decode(vae, arrays[f"latents_{s}"], tiled=False)
            ref = arrays[f"decoded_{s}"]
            err = fx.max_abs(got, ref)
            tol = m["tolerances"]["sizes"][s]["decode_fp32_max_abs"]
            print(f"[check5] {vae_path.name} {s} fp32 {conv} decode, "
                  f"{_numerics()}, max abs {err:.3e}, rel L2 "
                  f"{fx.rel_l2(got, ref):.3e}, tolerance {tol:.1e}")
            if conv == CONV and err > tol:
                failed.append(s)
    if _tf32():
        _gate_without_tf32(request)
    else:
        assert not failed


def test_bf16_decode_psnr(check5, vaes):
    from gmlx.image.vae.conv import CONV, CONV_PATHS

    arrays, m = check5
    failed = []
    for conv in CONV_PATHS:
        vae = vaes("bf16", conv)
        for s in _sizes(m):
            got = _decode(vae, arrays[f"latents_{s}"], tiled=False)
            ref = arrays[f"decoded_{s}"]
            db = fx.psnr(to_unit(got), to_unit(ref), 1.0)
            tol = m["tolerances"]["sizes"][s]["decode_bf16_psnr_db"]
            print(f"[check5] {s} bf16 {conv} decode PSNR {db:.2f} dB, max "
                  f"abs {fx.max_abs(got, ref):.3e}, gate {tol:.2f} dB")
            if conv == CONV and db < tol:
                failed.append(s)
    assert not failed


def _encode_mean(vae, image) -> np.ndarray:
    import mlx.core as mx

    from gmlx.image import memfit

    x = mx.array(image.transpose(0, 2, 3, 1))
    # The raw mean comes from the module, so the cache bound that
    # tiling.encode applies is applied here.
    with memfit.bounded_cache():
        mean, _ = vae.encode(x)
        return np.array(mean.astype(mx.float32)).transpose(0, 3, 1, 2)


def test_fp32_encoder_mean(check5, vaes, request):
    arrays, m = check5
    failed = []
    for s in _sizes(m):
        got = _encode_mean(vaes("fp32"), arrays[f"image_{s}"])
        ref = arrays[f"encode_mean_{s}"]
        tol = m["tolerances"]["sizes"][s]["encode_fp32_max_abs"]
        err = fx.max_abs(got, ref)
        print(f"[check5] {s} fp32 encode mean, {_numerics()}, max abs "
              f"{err:.3e}, rel L2 {fx.rel_l2(got, ref):.3e}, tolerance "
              f"{tol:.3e}")
        if err > tol:
            failed.append(s)
    if _tf32():
        _gate_without_tf32(request)
    else:
        assert not failed


def test_bf16_encoder_mean(check5, vaes):
    arrays, m = check5
    for s in _sizes(m):
        got = _encode_mean(vaes("bf16"), arrays[f"image_{s}"])
        ref = arrays[f"encode_mean_{s}"]
        tol = m["tolerances"]["sizes"][s]["encode_bf16_rel_l2"]
        rel = fx.rel_l2(got, ref)
        print(f"[check5] {s} bf16 encode mean max abs "
              f"{fx.max_abs(got, ref):.3e}, rel L2 {rel:.3e}, tolerance "
              f"{tol:.3e}")
        assert rel <= tol, s


def test_tiled_bf16_decode_at_the_largest_size(check5, vaes):
    arrays, m = check5
    s = _sizes(m)[-1]
    lat = arrays[f"latents_{s}"]
    vae = vaes("bf16")
    tiled = _decode(vae, lat, tiled=True)
    ref = arrays[f"decoded_{s}"]
    db = fx.psnr(to_unit(tiled), to_unit(ref), 1.0)
    whole = fx.psnr(to_unit(_decode(vae, lat, tiled=False)), to_unit(ref),
                    1.0)
    tol = m["tolerances"]["sizes"][s]["decode_bf16_psnr_db"]
    print(f"[check5] {s} bf16 tiled decode PSNR {db:.2f} dB against "
          f"{whole:.2f} dB whole, tolerance {tol:.2f} dB")
    assert int(s) > 512
    assert db >= tol


def test_tiled_bf16_encode_at_the_largest_size(check5, vaes):
    import mlx.core as mx

    from gmlx.image.vae import tiling

    arrays, m = check5
    s = _sizes(m)[-1]
    vae = vaes("bf16")
    x = mx.array(arrays[f"image_{s}"].transpose(0, 2, 3, 1))
    whole = tiling.encode(vae, x, tiled=False)
    tiled = tiling.encode(vae, x, tiled=True)
    rel = fx.rel_l2(np.array(tiled), np.array(whole))
    a = _decode(vae, whole, tiled=False)
    b = _decode(vae, tiled, tiled=False)
    db = fx.psnr(to_unit(b), to_unit(a), 1.0)
    print(f"[check5] {s} bf16 tiled encode rel L2 {rel:.3e} against the "
          f"whole encode, round trip {db:.2f} dB against the whole round "
          f"trip")
    assert int(s) > 512
    assert rel < TILED_ENCODE_REL_L2
    assert db >= TILED_ROUND_TRIP_DB
