"""``gmlx image generate`` on the real FLUX.3 klein files writes the
latents that ``sample`` returns for the noise of the same seed, bit for
bit, for the distilled flow map and for Base, whose unconditional row is
the empty prompt's, and a PNG of the requested size. With ``--ref`` and
no size it runs at the size of the first prepared reference, and crops
the image of one reference to the box of its fitted image. With two
``--adapter`` files it writes the latents of ``sample`` under the patch
table of the two. Skipped unless KQUANT_TEST_GGUF_DIR holds the FLUX.3
klein DiT GGUFs and the Qwen3-VL-4B-Instruct encoder, and
GMLX_TEST_IMAGE_DIR the FLUX.2 VAE."""
from __future__ import annotations

import gc
import json

import mlx.core as mx
import numpy as np
import pytest
from image_dir import vae_files
from image_fixtures import image_dits
from lora_fixtures import write_adapter

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


def _direct(dit, enc, vae, *, refs=(), size=(SIZE, SIZE), adapters=()):
    """The latents of ``sample`` on the noise of SEED, by the variant's
    defaults, the empty prompt's conditioning for Base's unconditional
    row, under the patch table of the (file, strength) rows of
    ``adapters``."""
    from gmlx.image import api, lora
    from gmlx.image.samplers import noise_from_seed

    pipe = api.load(dit, text_encoder=enc, vae=vae, log=lambda *_: None)
    table = None
    if adapters:
        rows = [(lora.load_adapter(pipe, p), s) for p, s in adapters]
        table = api.define_patch_table(pipe, lora.table_entries(rows))
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
                               width=width, patch_table=table,
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


def _pairs(seed: int, dims: dict, gain: float = 0.5) -> dict:
    """Target name to (down, up, alpha): rank 4 and alpha 2, down rows at
    unit spread over the input and up rows at ``gain`` over the rank."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, (o, i) in dims.items():
        down = rng.standard_normal((4, i)) / np.sqrt(i)
        up = rng.standard_normal((o, 4)) * gain / 2
        out[name] = (down.astype(np.float32), up.astype(np.float32), 2.0)
    return out


def test_generate_with_adapters_equals_sample_under_their_table(
        gguf_dir, gguf_index, tmp_path, capsys):
    """Two stacked ``--adapter`` files on the distilled DiT, one in the
    diffusers names as OneTrainer spells them, the final modulation among
    them, and one in the GGUF names as kohya spells them, the target time
    embedder among them: the latents of ``sample`` under the table of the
    two at their strengths, other latents than the base model's, and the
    files and strengths in the PNG's metadata."""
    from PIL import Image

    from gmlx.commands.cli import umbrella_main

    dit, enc, vae = _files(gguf_dir, gguf_index, "distilled")
    d, mlp = 3072, 9216
    a = write_adapter(tmp_path / "a.safetensors", "onetrainer", _pairs(1, {
        "single_transformer_blocks.0.attn.to_q": (d, d),
        "single_transformer_blocks.27.ff.linear_in": (2 * mlp, d),
        "mode_blocks.image.0.attn.to_out.0": (d, d),
        "timestep_embedder.linear_2": (d, d),
        "early_stream_modulations.image.linear": (3 * d, d),
        "norm_out.linear": (2 * d, d)}))
    b = write_adapter(tmp_path / "b.safetensors", "kohya", _pairs(2, {
        "txt_mode_blocks.2.mlp_out": (d, mlp),
        "content_mode_blocks.image.4.v_proj": (d, d),
        "ref_time_in.in_layer": (d, 256)}))
    lat, png = tmp_path / "lat.npz", tmp_path / "a.png"
    assert umbrella_main([
        "image", "generate", dit, "--encoder", enc, "--vae", vae,
        "--prompt", PROMPT, "--size", f"{SIZE}x{SIZE}", "--seed", str(SEED),
        "--output", str(png), "--save-latent", str(lat),
        "--adapter", f"{a}:0.5", "--adapter", b]) == 0
    out = capsys.readouterr().out
    assert "adapter a.safetensors at strength 0.5: 6 pairs" in out
    assert "adapter b.safetensors at strength 1: 3 pairs" in out
    got = np.load(lat)["latents"]
    with Image.open(png) as im:
        meta = json.loads(im.text["gmlx"])["adapters"]
    assert [(r["file"], r["strength"]) for r in meta] == [
        ("a.safetensors", 0.5), ("b.safetensors", 1.0)]
    gc.collect()
    mx.clear_cache()
    want = _direct(dit, enc, vae, adapters=[(a, 0.5), (b, 1.0)])
    gc.collect()
    mx.clear_cache()
    base = _direct(dit, enc, vae)
    assert np.isfinite(got).all()
    assert np.array_equal(got, want)
    moved = float(np.linalg.norm(got - base) / np.linalg.norm(base))
    print(f"flux3 cli adapters: the latents move {moved:.3e} from the base "
          "model's")
    assert moved > 1e-2


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
