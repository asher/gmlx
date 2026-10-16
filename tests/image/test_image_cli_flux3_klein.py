"""``gmlx image generate`` on the real FLUX.3 klein files writes the
latents that ``sample`` returns for the noise of the same seed, bit for
bit, for the distilled flow map and for Base, whose unconditional row is
the empty prompt's, and a PNG of the requested size. Skipped unless
KQUANT_TEST_GGUF_DIR holds the FLUX.3 klein DiT GGUFs and the
Qwen3-VL-4B-Instruct encoder, and GMLX_TEST_IMAGE_DIR the FLUX.2 VAE."""
from __future__ import annotations

import gc

import mlx.core as mx
import numpy as np
import pytest
from image_dir import vae_files
from image_fixtures import image_dits

# A Mac of 16 GB, as most users run (tests/host_box.py).
pytestmark = [pytest.mark.integration, pytest.mark.host_gb(16)]

FAMILY = "flux3_klein"
PROMPT = "a red bicycle leaning on a brick wall"
SEED = 7
SIZE = 512


def _files(gguf_dir, gguf_index: dict, variant: str) -> tuple:
    from gmlx.commands.image_preflight import identify_dit
    from gmlx.image import text_encoder as te

    dit = None
    for p in image_dits(gguf_dir, FAMILY):
        if identify_dit(p).variant == variant:
            dit = p
            break
    spec = te.encoder_spec(FAMILY)
    encs = []
    for p in gguf_index.get(spec.arch, []):
        try:
            te.check_encoder_file(te._header_view(p), spec)
        except te.EncoderMismatchError:
            continue
        encs.append(p)
    vaes = sorted(vae_files(FAMILY), key=lambda p: p.suffix != ".gguf")
    if not (dit and encs and vaes):
        pytest.skip(f"needs a FLUX.3 klein {variant} DiT and its text "
                    "encoder under KQUANT_TEST_GGUF_DIR and the FLUX.2 VAE "
                    "under GMLX_TEST_IMAGE_DIR")
    return dit, encs[0], str(vaes[0])


def _direct(dit, enc, vae, *, refs=(), size=(SIZE, SIZE)):
    """The latents of ``sample`` on the noise of SEED, by the variant's
    defaults, the empty prompt's conditioning for Base's unconditional
    row."""
    from gmlx.image import api
    from gmlx.image.samplers import noise_from_seed

    pipe = api.load(dit, text_encoder=enc, vae=vae, log=lambda *_: None)
    cfg = pipe.defaults.cfg
    cond = api.encode_text(pipe, PROMPT, images=refs)
    neg = None if cfg == 1.0 else api.encode_text(pipe, "", images=refs)
    width, height = size
    shape = (1, pipe.dit.cfg.in_channels, height // 16, width // 16)
    return np.array(api.sample(pipe, cond, neg=neg, cfg=cfg, height=height,
                               width=width,
                               noise=noise_from_seed(SEED, shape)))


@pytest.mark.parametrize("variant", ["distilled", "base"])
def test_generate_latents_equal_sample_with_the_seed_noise(
        gguf_dir, gguf_index, tmp_path, variant):
    from PIL import Image

    from gmlx.commands.cli import umbrella_main

    dit, enc, vae = _files(gguf_dir, gguf_index, variant)
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
    want = _direct(dit, enc, vae)
    assert got.shape == want.shape == (1, 128, SIZE // 16, SIZE // 16)
    assert np.isfinite(got).all()
    assert np.array_equal(got, want)
