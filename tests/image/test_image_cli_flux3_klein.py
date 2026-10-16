"""``gmlx image generate`` on the real FLUX.3 klein files writes the
latents that ``sample`` returns for the noise of the same seed, bit for
bit, for the distilled flow map and for Base, whose unconditional row is
the empty prompt's, and a PNG of the requested size. With ``--ref`` and
no size it runs at the size of the first prepared reference, and crops
the image of one reference to the box of its fitted image. Skipped unless
KQUANT_TEST_GGUF_DIR holds the FLUX.3 klein DiT GGUFs and the
Qwen3-VL-4B-Instruct encoder, and GMLX_TEST_IMAGE_DIR the FLUX.2 VAE."""
from __future__ import annotations

import gc
import json

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
    cond = api.encode_text(pipe, PROMPT, images=refs, ref_latents=False)
    neg = None if cfg == 1.0 else api.encode_text(pipe, "", images=refs,
                                                  ref_latents=False)
    # The VAE encodes the references after the encoder's release, as the
    # CLI does on a Mac where the three files leave too little room.
    pipe.encoder.release()
    cond = api.encode_references(pipe, cond)
    neg = None if neg is None else api.encode_references(pipe, neg)
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


def _source(w: int, h: int, seed: int, channels: int = 3) -> np.ndarray:
    """A picture with structure: two gradients and a few rectangles."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([255 * x / w, 255 * y / h, 128 + 0 * x], axis=-1)
    for _ in range(6):
        x0, y0 = rng.integers(0, w - 8), rng.integers(0, h - 8)
        img[y0:y0 + h // 4, x0:x0 + w // 4] = rng.integers(0, 256, 3)
    img = np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    if channels == 4:
        alpha = rng.integers(0, 256, (h, w, 1)).astype(np.uint8)
        img = np.concatenate([img, alpha], axis=-1)
    return img


@pytest.mark.parametrize("case", ["one", "two"])
def test_generate_with_references_equals_sample_and_crops_its_image(
        gguf_dir, gguf_index, tmp_path, case):
    """``--ref`` with no size: the size of the first prepared reference,
    the latents of ``sample`` on the conditioning of the prepared
    references, and for one reference the PNG cropped to the box of the
    fitted image."""
    from PIL import Image

    from gmlx.commands.cli import umbrella_main
    from gmlx.image.families.flux3_klein import edit

    dit, enc, vae = _files(gguf_dir, gguf_index, "distilled")
    sources = [_source(600, 400, 1)]
    if case == "two":
        sources.append(_source(400, 400, 2, channels=4))
    paths = []
    for i, src in enumerate(sources):
        p = tmp_path / f"ref{i}.png"
        Image.fromarray(src).save(p)
        paths += ["--ref", str(p)]
    lat, png = tmp_path / "lat.npz", tmp_path / "a.png"
    assert umbrella_main([
        "image", "generate", dit, "--encoder", enc, "--vae", vae,
        "--prompt", PROMPT, "--seed", str(SEED), "--output", str(png),
        "--save-latent", str(lat)] + paths) == 0
    got = np.load(lat)["latents"]
    refs = edit.prepare(sources)
    size = edit.default_size(refs)
    want_size = {"one": (608, 400), "two": (592, 400)}[case]
    assert size == want_size
    with Image.open(png) as im:
        crop = refs.crop
        assert im.size == (size if crop is None else
                           (crop[2] - crop[0], crop[3] - crop[1]))
        assert json.loads(im.text["gmlx"]).get("crop") == (
            None if crop is None else list(crop))
    gc.collect()
    mx.clear_cache()
    want = _direct(dit, enc, vae, refs=refs, size=size)
    assert got.shape == want.shape == (1, 128, size[1] // 16, size[0] // 16)
    assert np.isfinite(got).all()
    assert np.array_equal(got, want)
