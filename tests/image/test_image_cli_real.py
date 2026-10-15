"""``gmlx image generate`` on real Z-Image weights writes the latents that
``sample`` returns for the noise of the same seed, bit for bit, and a PNG
of the requested size. A step split into its cached parts equals the plain
forward bit for bit. Skipped unless KQUANT_TEST_GGUF_DIR holds a Z-Image
DiT GGUF and its text encoder, and GMLX_TEST_IMAGE_DIR the Z-Image VAE."""
from __future__ import annotations

import gc

import numpy as np
import pytest
from image_dir import vae_files

import mlx.core as mx

# A Mac of 16 GB, as most users run (tests/host_box.py).
pytestmark = [pytest.mark.integration, pytest.mark.host_gb(16)]

PROMPT = "a red bicycle leaning on a brick wall"
SEED = 7
SIZE = 512


def _files(gguf_index: dict) -> tuple[str, str, str]:
    from gmlx.commands.image_preflight import identify_dit
    from gmlx.image import text_encoder as te

    dits = []
    for p in gguf_index.get("lumina2", []):
        try:
            if identify_dit(p).family.name == "z_image":
                dits.append(p)
        except ValueError:
            continue
    encs = []
    for p in gguf_index.get(te.Z_IMAGE.arch, []):
        try:
            te.check_encoder_file(te._header_view(p), te.Z_IMAGE)
        except te.EncoderMismatchError:
            continue
        encs.append(p)
    vaes = sorted(vae_files("z_image"), key=lambda p: p.suffix != ".gguf")
    if not (dits and encs and vaes):
        pytest.skip("needs a Z-Image DiT and text encoder under "
                    "KQUANT_TEST_GGUF_DIR and its VAE under "
                    "GMLX_TEST_IMAGE_DIR")
    return dits[0], encs[0], str(vaes[0])


def test_generate_latents_equal_sample_with_the_seed_noise(gguf_index,
                                                           tmp_path):
    from PIL import Image

    from gmlx.commands.cli import umbrella_main
    from gmlx.image import api
    from gmlx.image.samplers import noise_from_seed

    dit, enc, vae = _files(gguf_index)
    lat, png = tmp_path / "lat.npz", tmp_path / "a.png"
    assert umbrella_main([
        "image", "generate", dit, "--encoder", enc, "--vae", vae,
        "--prompt", PROMPT, "--size", f"{SIZE}x{SIZE}", "--seed", str(SEED),
        "--output", str(png), "--save-latent", str(lat)]) == 0
    got = np.load(lat)["latents"]
    with Image.open(png) as im:
        assert im.size == (SIZE, SIZE)
    gc.collect()
    mx.clear_cache()

    pipe = api.load(dit, text_encoder=enc, vae=vae, log=lambda *_: None)
    cond = api.encode_text(pipe, PROMPT)
    shape = (1, pipe.dit.cfg.in_channels, SIZE // 8, SIZE // 8)
    want = np.array(api.sample(pipe, cond, height=SIZE, width=SIZE,
                               noise=noise_from_seed(SEED, shape)))
    assert got.shape == want.shape
    assert np.isfinite(got).all()
    assert np.array_equal(got, want)


def test_a_split_step_equals_the_plain_forward(gguf_index):
    from gmlx.image import api
    from gmlx.image.samplers import noise_from_seed

    dit, enc, _ = _files(gguf_index)
    gc.collect()
    mx.clear_cache()
    pipe = api.load(dit, text_encoder=enc, log=lambda *_: None)
    cond = api.encode_text(pipe, PROMPT)
    shape = (1, pipe.dit.cfg.in_channels, SIZE // 8, SIZE // 8)
    x = mx.array(noise_from_seed(SEED, shape))
    for sigma in pipe.sigmas(pipe.defaults.steps)[:2]:
        s = np.reshape(np.float32(sigma), (1,))
        plain = pipe.dit(x, s, cond.features[None])
        split = api.denoise_step(pipe, x, sigma, cond)
        cached = api.denoise_step(pipe, x, sigma, cond)
        assert mx.array_equal(split, plain).item()
        assert mx.array_equal(cached, plain).item()
    assert len(pipe.modulations) == 2 and len(pipe.captions) == 1
