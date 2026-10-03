"""Check 5 of the image acceptance gate for Z-Image: the FLUX.1 VAE.

gmlx decodes the oracle's latents and encodes its image with the official
VAE, read from a GGUF packed from the oracle's safetensors when one is
found, else from that safetensors file. The fp32 decode must stay within
the manifest's max abs and the bf16 decode above its PSNR, both against
the oracle's fp32 decode. Decodes run on both conv paths, and the gates
hold the default path. The encoder mean is held to the manifest's max abs
in fp32 and relative L2 in bf16. A tiled bf16 decode at the largest size
must stay above 35 dB, the image floor of check 6. Skipped
unless GMLX_TEST_IMAGE_DIR holds the VAE and GMLX_IMAGE_FIXTURES the
fixture. scripts/image_oracle/check5_z_image_vae.py writes it."""
from __future__ import annotations

import numpy as np
import pytest
from image_dir import sha256, vae_files
from oracle_fixture import fixture, fx

pytestmark = pytest.mark.integration

NAME = "check5_z_image_vae"
FAMILY = "z_image"
TILED_PSNR_MIN = 35.0


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


def test_latent_space_constants_are_the_oracle_ones(check5, vaes):
    _, m = check5
    cfg = vaes("fp32").config
    assert cfg.scaling_factor == m["inputs"]["scaling_factor"]
    assert cfg.shift_factor == m["inputs"]["shift_factor"]


# Both conv paths are reported, and the gates hold the default one.


def test_fp32_decode_is_within_max_abs(check5, vaes, vae_path):
    from gmlx.image.vae.conv import CONV, CONV_PATHS

    arrays, m = check5
    failed = []
    for conv in CONV_PATHS:
        vae = vaes("fp32", conv)
        for s in _sizes(m):
            got = _decode(vae, arrays[f"latents_{s}"])
            ref = arrays[f"decoded_{s}"]
            err = fx.max_abs(got, ref)
            tol = m["tolerances"]["sizes"][s]["decode_fp32_max_abs"]
            print(f"[check5] {vae_path.name} {s} fp32 {conv} decode max abs "
                  f"{err:.3e}, rel L2 {fx.rel_l2(got, ref):.3e}, tolerance "
                  f"{tol:.1e}")
            if conv == CONV and err > tol:
                failed.append(s)
    assert not failed


def test_bf16_decode_psnr(check5, vaes):
    from gmlx.image.vae.conv import CONV, CONV_PATHS

    arrays, m = check5
    failed = []
    for conv in CONV_PATHS:
        vae = vaes("bf16", conv)
        for s in _sizes(m):
            got = _decode(vae, arrays[f"latents_{s}"])
            ref = arrays[f"decoded_{s}"]
            db = fx.psnr(to_unit(got), to_unit(ref), 1.0)
            tol = m["tolerances"]["sizes"][s]["decode_bf16_psnr_db"]
            print(f"[check5] {s} bf16 {conv} decode PSNR {db:.2f} dB, max "
                  f"abs {fx.max_abs(got, ref):.3e}, gate {tol:.2f} dB")
            if conv == CONV and db < tol:
                failed.append(s)
    assert not failed


def test_encoder_mean(check5, vaes):
    import mlx.core as mx

    from gmlx.image import memfit

    arrays, m = check5
    for dtype in ("fp32", "bf16"):
        vae = vaes(dtype)
        for s in _sizes(m):
            x = mx.array(arrays[f"image_{s}"].transpose(0, 2, 3, 1))
            # The raw mean comes from the module, so the cache bound that
            # tiling.encode applies is applied here.
            with memfit.bounded_cache():
                mean, _ = vae.encode(x)
                got = np.array(mean.astype(mx.float32)).transpose(0, 3, 1, 2)
            ref = arrays[f"encode_mean_{s}"]
            tol = m["tolerances"]["sizes"][s]
            err, rel = fx.max_abs(got, ref), fx.rel_l2(got, ref)
            print(f"[check5] {s} {dtype} encode mean max abs {err:.3e}, "
                  f"rel L2 {rel:.3e}")
            if dtype == "fp32":
                assert err <= tol["encode_fp32_max_abs"], s
            else:
                assert rel <= tol["encode_bf16_rel_l2"], s


def test_tiled_bf16_decode_at_the_largest_size(check5, vaes):
    arrays, m = check5
    s = _sizes(m)[-1]
    lat = arrays[f"latents_{s}"]
    vae = vaes("bf16")
    tiled = _decode(vae, lat, tiled=True)
    ref = arrays[f"decoded_{s}"]
    db = fx.psnr(to_unit(tiled), to_unit(ref), 1.0)
    whole = fx.psnr(to_unit(_decode(vae, lat)), to_unit(ref), 1.0)
    print(f"[check5] {s} bf16 tiled decode PSNR {db:.2f} dB against "
          f"{whole:.2f} dB whole")
    assert int(s) > 512
    assert db >= TILED_PSNR_MIN
