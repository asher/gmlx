"""Qwen-Image 2.1 from a prompt to an image, on the real files.

``gmlx image generate`` with a DiT path, ``--encoder`` and ``--vae`` writes
the latents that ``sample`` returns for the noise of the same seed, bit for
bit. Under the working budget of a 16 GiB Mac the same run loads in stages
and writes the same latents. Check 6 from a prompt: the run starts from the
prompt text through gmlx's own encoder, at the family defaults, and its
final latents stay within the check 6 tolerance of the oracle's run, which
took the oracle's own float32 features. The image gmlx decodes from its latents stays at or
above 30 dB of the float32 decode of the oracle's latents. The oracle's
fixture holds latents and no image, so the reference image is gmlx's own
float32 VAE on them, a whole pass.

Skipped unless KQUANT_TEST_GGUF_DIR holds the DiT GGUF the check 6 fixture
names and the Qwen3-VL-8B text GGUF, GMLX_TEST_IMAGE_DIR the Qwen-Image 2.1
VAE, and GMLX_IMAGE_FIXTURES the check 6 fixture."""
from __future__ import annotations

import contextlib
import gc
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest
from image_dir import vae_files
from image_fixtures import image_dits
from oracle_fixture import ORACLE, fixture, fx

import mlx.core as mx

# Each row loads local model files, so it runs on this Mac
# (tests/host_box.py).
pytestmark = pytest.mark.integration

NAME = "check6_qwen_image21"
FAMILY = "qwen_image21"
SIZE = 512
PSNR_DB = 30.0


@pytest.fixture(scope="module")
def check6():
    if not (ORACLE / f"{NAME}.manifest.json").is_file():
        pytest.skip(f"{NAME}.manifest.json is not committed")
    arrays, m = fixture(NAME)
    if SIZE not in m["inputs"]["sizes"]:
        pytest.skip(f"the oracle did not run {SIZE}")
    return arrays, m


@pytest.fixture(scope="module")
def files(check6, gguf_dir, gguf_index):
    from gmlx.image import text_encoder as te

    want = check6[1]["files"]["dit"]
    dits = [p for p in image_dits(gguf_dir, FAMILY)
            if Path(p).name == want["name"]
            and os.path.getsize(p) == want["size"]]
    spec = te.encoder_spec(FAMILY)
    encs = []
    for p in gguf_index.get(spec.arch, []):
        try:
            te.check_encoder_file(te._header_view(p), spec)
        except te.EncoderMismatchError:
            continue
        encs.append(p)
    vaes = sorted(vae_files(FAMILY), key=lambda p: p.suffix != ".gguf")
    if not (dits and encs and vaes):
        pytest.skip("needs the Qwen-Image 2.1 DiT of the fixture and the "
                    "Qwen3-VL-8B text GGUF under KQUANT_TEST_GGUF_DIR, and "
                    "the Qwen-Image 2.1 VAE under GMLX_TEST_IMAGE_DIR")
    return dits[0], encs[0], str(vaes[0])


@pytest.fixture(scope="module")
def generated(check6, files, tmp_path_factory):
    """One ``gmlx image generate`` run at the family defaults: the saved
    latents, the PNG pixels and the JSON record."""
    from PIL import Image

    from gmlx.commands.cli import umbrella_main

    dit, enc, vae = files
    ins = check6[1]["inputs"]
    d = tmp_path_factory.mktemp("qwen_t2i")
    lat, png = d / "lat.npz", d / "a.png"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = umbrella_main([
            "image", "generate", dit, "--encoder", enc, "--vae", vae,
            "--prompt", ins["prompt"], "--size", f"{SIZE}x{SIZE}",
            "--seed", str(ins["seed"]), "--output", str(png),
            "--save-latent", str(lat), "--json"])
    assert rc == 0, buf.getvalue()
    record = json.loads(buf.getvalue().strip().splitlines()[-1])
    with Image.open(png) as im:
        pixels = np.asarray(im.convert("RGB"))
        meta = json.loads(im.info["gmlx"])
    gc.collect()
    mx.clear_cache()
    return {"latents": np.load(lat)["latents"], "pixels": pixels,
            "record": record, "meta": meta}


@pytest.fixture(scope="module")
def pipe(files, generated):
    from gmlx.image import api

    dit, enc, vae = files
    return api.load(dit, text_encoder=enc, vae=vae, log=lambda *_: None)


@pytest.fixture(scope="module")
def sampled(check6, pipe):
    """The run through the primitives: the conditioning of the prompt and
    the final latents from the noise of the oracle's seed."""
    from gmlx.image import api
    from gmlx.image.samplers import noise_from_seed

    arrays, m = check6
    ins = m["inputs"]
    cond = api.encode_text(pipe, ins["prompt"])
    noise = noise_from_seed(ins["seed"], arrays[f"noise_{SIZE}"].shape)
    assert np.array_equal(noise, arrays[f"noise_{SIZE}"])
    x = api.sample(pipe, cond, height=SIZE, width=SIZE, noise=noise)
    return cond, x


def test_generate_runs_the_family_defaults(check6, generated):
    ins = check6[1]["inputs"]
    rec = generated["record"]
    s = rec["settings"]
    assert rec["model"]["family"] == FAMILY
    assert (s["steps"], s["guidance"], s["sampler"]) == (
        ins["steps"], ins["guidance"], "euler")
    assert "negative_prompt" not in s
    assert generated["pixels"].shape == (SIZE, SIZE, 3)
    assert generated["meta"]["prompt"] == ins["prompt"]


BUDGET_16 = 17_179_869_184 - 8e9    # the working budget of a 16 GiB Mac


def test_a_16_gib_budget_stages_the_load_and_writes_the_same_latents(
        check6, files, generated, tmp_path, monkeypatch):
    """The run of ``generated`` again under the working budget of a 16 GiB
    Mac. The encoder encodes with no DiT and no VAE beside it and is gone
    when the DiT loads, and the latents are those of the run that loaded
    every part at once."""
    from gmlx.commands.cli import umbrella_main
    from gmlx.device import metal_value
    from gmlx.image import api, loader
    from gmlx.image import text_encoder as te

    mem = float(metal_value("memory_size"))
    monkeypatch.setenv("GMLX_GOV_RESERVE_GB", repr((mem - BUDGET_16) / 1e9))
    seen: dict = {"encode": []}
    load_encoder, encode_text, load_dit = (te.load_encoder, api.encode_text,
                                           loader.load_dit)

    def encoder(*a, **kw):
        seen["handle"] = load_encoder(*a, **kw)
        return seen["handle"]

    def encode(pipe, *a, **kw):
        seen["encode"].append((pipe.dit, pipe.vae, pipe.encoder.released))
        return encode_text(pipe, *a, **kw)

    def dit(*a, **kw):
        seen["dit"] = seen["handle"].released
        return load_dit(*a, **kw)

    monkeypatch.setattr(te, "load_encoder", encoder)
    monkeypatch.setattr(api, "encode_text", encode)
    monkeypatch.setattr(loader, "load_dit", dit)
    path, enc, vae = files
    ins = check6[1]["inputs"]
    lat = tmp_path / "lat.npz"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = umbrella_main([
            "image", "generate", path, "--encoder", enc, "--vae", vae,
            "--prompt", ins["prompt"], "--size", f"{SIZE}x{SIZE}",
            "--seed", str(ins["seed"]), "--output", str(tmp_path / "a.png"),
            "--save-latent", str(lat), "--json"])
    assert rc == 0, buf.getvalue()
    record = json.loads(buf.getvalue().strip().splitlines()[-1])
    gc.collect()
    mx.clear_cache()
    assert record["levers"][-1] == "staged load"
    assert record["encoder_released"] is True
    assert seen["encode"] and all(
        state == (None, None, False) for state in seen["encode"])
    assert seen["dit"] is True
    assert generated["record"]["levers"][-1:] != ["staged load"]
    assert np.array_equal(np.load(lat)["latents"], generated["latents"])


def test_generate_latents_equal_sample_with_the_seed_noise(generated,
                                                           sampled):
    got, want = generated["latents"], np.array(sampled[1])
    assert got.shape == want.shape and np.isfinite(got).all()
    assert np.array_equal(got, want)


def test_the_conditioning_of_the_prompt_is_near_the_features_of_the_oracle(
        check6, sampled):
    """The oracle sampled from its own float32 features of this prompt."""
    arrays, _ = check6
    got = np.array(sampled[0].features.astype(mx.float32))
    ref = arrays["features"]
    assert got.shape == ref.shape
    print(f"qwen t2i conditioning: rel L2 {fx.rel_l2(got, ref):.4e} of the "
          f"oracle's float32 features, {got.shape[0]} rows")


def test_check6_final_latents_from_the_prompt(check6, sampled):
    arrays, m = check6
    got = fx.compare(np.array(sampled[1]), arrays[f"latents_{SIZE}"])
    tol = m["tolerances"]["sizes"][str(SIZE)]["latents_rel_l2"]
    print(f"qwen check6 latents {SIZE} from the prompt: rel L2 "
          f"{got['rel_l2']:.4e}, {got['rel_l2'] / tol:.2f} of its gate "
          f"{tol:.4e}, cosine {got['cosine']:.6f}")
    assert got["rel_l2"] <= tol, got


def test_the_image_of_the_prompt(check6, files, pipe, sampled, generated):
    """The decode of gmlx's latents against the float32 decode of the
    oracle's latents, both by gmlx's VAE."""
    from gmlx.commands.image import to_uint8
    from gmlx.image import api
    from gmlx.image.vae import load_vae, tiling

    arrays, _ = check6
    got = np.array(api.vae_decode(pipe, sampled[1]))
    assert got.shape == (1, SIZE, SIZE, 3)
    pixels = to_uint8(mx.array(got))
    fp32 = load_vae(files[2], family=FAMILY, dtype=mx.float32)
    ref = tiling.decode(fp32, mx.array(arrays[f"latents_{SIZE}"]),
                        tiled=False)
    ref = np.clip(np.array(ref) / 2 + 0.5, 0.0, 1.0)
    own = tiling.decode(fp32, sampled[1], tiled=False)
    own = np.clip(np.array(own) / 2 + 0.5, 0.0, 1.0)
    del fp32
    db = fx.psnr(got, ref, 1.0)
    db_png = fx.psnr(generated["pixels"].astype(np.float64) / 255.0, ref[0],
                     1.0)
    print(f"qwen t2i image {SIZE}: {db:.2f} dB against the float32 decode "
          f"of the oracle's latents, bar {PSNR_DB:.0f}; the written PNG "
          f"{db_png:.2f} dB; this pipeline's decode against the float32 "
          f"decode of its own latents {fx.psnr(got, own, 1.0):.2f} dB")
    assert np.array_equal(generated["pixels"], pixels)
    assert db >= PSNR_DB and db_png >= PSNR_DB
