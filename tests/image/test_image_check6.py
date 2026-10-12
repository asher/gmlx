"""Check 6 of the image acceptance gate for Z-Image-Turbo: a whole sampling
run on injected noise. gmlx encodes the oracle's prompt, samples from the
oracle's noise with its default steps and guidance, and decodes, and the
image must stay within the PSNR gate of the oracle's fp32 image. The gate
is twice the oracle's own bf16 against fp32 RMS error, as the manifest
records it. The seed's noise must equal the oracle's noise.

Skipped unless GMLX_IMAGE_FIXTURES holds the fixture that
scripts/image_oracle/check6_z_image.py writes, KQUANT_TEST_GGUF_DIR the
DiT and text encoder GGUFs the oracle read, and GMLX_TEST_IMAGE_DIR the
VAE."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from image_dir import sha256, vae_files
from oracle_fixture import fixture, fx

import mlx.core as mx

pytestmark = pytest.mark.integration

NAME = "check6_z_image_turbo"


def _find(gguf_index: dict, arch: str, want: dict) -> str:
    for p in gguf_index.get(arch, []):
        if Path(p).name == want["name"] and os.path.getsize(p) == want["size"]:
            if fx.file_identity(p)["sha256"] != want["sha256"]:
                pytest.fail(f"{p} is not the file the oracle read")
            return p
    pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")


def _vae(want: str) -> str:
    from gmlx.image.vae import read_header

    files = vae_files("z_image")
    for p in files:
        if p.suffix == ".gguf" and read_header(str(p)).metadata.get(
                "gmlx.vae.source_sha256") == want:
            return str(p)
    for p in files:
        if p.suffix == ".safetensors" and sha256(p) == want:
            return str(p)
    pytest.skip("the fixture's VAE is not under GMLX_TEST_IMAGE_DIR")


@pytest.fixture(scope="module")
def check6():
    return fixture(NAME)


@pytest.fixture(scope="module")
def pipe(check6, gguf_index):
    from gmlx.image import api
    from gmlx.image.text_encoder import Z_IMAGE

    files = check6[1]["files"]
    dit = _find(gguf_index, "lumina2", files["dit"])
    enc = _find(gguf_index, Z_IMAGE.arch, files["encoder"])
    p = api.load(dit, text_encoder=enc, vae=_vae(files["vae"]["sha256"]),
                 log=lambda *_: None)
    assert p.variant == "turbo"
    return p


def _sizes(m) -> list:
    return [int(s) for s in m["inputs"]["sizes"]]


def test_the_seed_noise_is_the_oracle_noise(check6):
    from gmlx.image.samplers import noise_from_seed

    arrays, m = check6
    for size in _sizes(m):
        want = arrays[f"noise_{size}"]
        got = noise_from_seed(m["inputs"]["seed"], want.shape)
        assert np.array_equal(got, want), size


def test_check6_full_run_psnr(check6, pipe):
    from gmlx.image import api
    from gmlx.image.samplers import noise_from_seed

    arrays, m = check6
    seed = m["inputs"]["seed"]
    assert m["inputs"]["steps"] == pipe.defaults.steps
    assert m["inputs"]["guidance"] == pipe.defaults.cfg
    cond = api.encode_text(pipe, m["inputs"]["prompt"])
    cap = arrays["cap_feats_fp32"]
    got_cap = np.array(cond.features.astype(mx.float32))
    assert got_cap.shape == cap.shape
    misses = []
    for size in _sizes(m):
        ref = arrays[f"image_{size}"]
        noise = noise_from_seed(seed, arrays[f"noise_{size}"].shape)
        x = api.sample(pipe, cond, height=size, width=size, noise=noise)
        img = np.array(api.vae_decode(pipe, x)[0].astype(mx.float32))
        lat = np.array(x)
        err = np.mean((img.astype(np.float64) - ref) ** 2)
        psnr = float("inf") if err == 0 else float(10 * np.log10(1 / err))
        gate = m["tolerances"]["sizes"][str(size)]["image_psnr_db"]
        lat_ref = arrays[f"latents_{size}"]
        rel = float(np.linalg.norm(lat - lat_ref) / np.linalg.norm(lat_ref))
        oracle = m["results"]["sizes"][str(size)]["image_bf16_vs_fp32"]
        print(f"[check6] {size}: PSNR {psnr:.2f} dB against the gate "
              f"{gate:.2f} (oracle bf16 {oracle['psnr_db']:.2f}); latents "
              f"rel L2 {rel:.3e}")
        if not psnr >= gate:
            misses.append((size, psnr, gate))
    assert not misses, misses
