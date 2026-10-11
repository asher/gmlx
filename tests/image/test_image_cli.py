"""``gmlx image generate`` and ``bench`` over a toy pipeline: generate's
latents equal those of ``sample`` with the seed's noise, the output naming,
the PNG text chunk, img2img and inpainting, encoder release, the bench
record, and the refusals with their exit codes. The loader, the VAE and the
fit check are replaced. CPU."""
from __future__ import annotations

import json
import os
import re
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from test_image_api import _Dit, _Encoder

from gmlx.commands import image as cmd
from gmlx.commands import image_preflight
from gmlx.commands.cli import umbrella_main
from gmlx.image import api
from gmlx.image.registry import get_family
from gmlx.image.samplers import noise_from_seed
from gmlx.image.text_encoder import TextTap

PIL = pytest.importorskip("PIL")


class _OwnedEncoder(_Encoder):
    owned = True

    @property
    def released(self):
        return self.model is None

    def release(self):
        self.model = None
        return True


def _decode(pipe, x, tiled=None, alpha=False, plan=None):
    """Gray pixels from the latents, and with ``alpha`` a fourth channel
    that runs from clear at the left to opaque at the right."""
    pipe.decoded.append((tiled, plan))
    b, _, h, w = x.shape
    v = mx.sigmoid(x.mean(axis=1))[..., None]
    img = mx.broadcast_to(v, (b, h, w, 3))
    if alpha:
        ramp = mx.broadcast_to(mx.linspace(0.0, 1.0, w)[None, None, :, None],
                               (b, h, w, 1))
        img = mx.concatenate([img, ramp], axis=-1)
    return mx.repeat(mx.repeat(img, 8, axis=1), 8, axis=2)


def _encode(pipe, image, tiled=None, mode="mode", noise=None, plan=None):
    pipe.encoded_images.append((tiled, plan))
    x = mx.array(image).mean(axis=-1)[:, None]
    pooled = x.reshape(x.shape[0], 1, x.shape[2] // 8, 8, x.shape[3] // 8,
                       8).mean(axis=(3, 5))
    return mx.broadcast_to(pooled, (x.shape[0], 4) + pooled.shape[2:])


@pytest.fixture
def env(tmp_path, monkeypatch):
    files = {}
    for name in ("dit.gguf", "enc.gguf", "vae.gguf"):
        p = tmp_path / name
        p.write_bytes(b"x")
        files[name] = str(p)
    state = SimpleNamespace(pipes=[], plan=SimpleNamespace(
        release_encoder=False, tiled_decode=False, levers=(), rows=2,
        describe=lambda: "peak 9.00 GiB against a budget of 8.00 GiB"),
        variant="turbo", checks=[], checked=None)
    fam = get_family("z_image")

    def load(dit, *, text_encoder=None, vae=None, log=print, **kw):
        pipe = api.ImagePipeline(dit=_Dit(), family=fam,
                                 variant=state.variant,
                                 encoder=_OwnedEncoder(), vae=object(),
                                 load_seconds={"dit": 0.0}, log=log)
        pipe.decoded, pipe.encoded_images = [], []
        state.pipes.append(pipe)
        return pipe

    monkeypatch.setattr(api, "load", load)
    monkeypatch.setattr(api, "vae_decode", _decode)
    monkeypatch.setattr(api, "vae_encode", _encode)
    monkeypatch.setattr(api, "plan_request",
                        lambda *a, **k: state.planned.append(k) or state.plan)
    state.planned = []
    view = SimpleNamespace(tensors={
        "a": SimpleNamespace(shape=(4, 4), codec="Q4_K"),
        "b": SimpleNamespace(shape=(4, 4), codec="Q4_K"),
        "c": SimpleNamespace(shape=(4, 4), codec="BF16"),
        "d": SimpleNamespace(shape=(4,), codec="F32")})
    monkeypatch.setattr(image_preflight, "identify_dit",
                        lambda p, **k: SimpleNamespace(family=fam, view=view,
                                                  variant=state.variant))
    monkeypatch.setattr(image_preflight, "check",
                        lambda *a, **k: state.checks.append(k)
                        or state.checked)
    monkeypatch.chdir(tmp_path)
    state.files = files
    state.base = ["image", "generate", files["dit.gguf"], "--encoder",
                  files["enc.gguf"], "--vae", files["vae.gguf"]]
    return state


def _direct(pipe, prompt, seed, shape=(1, 4, 4, 6), **kw):
    """sample called directly with the seed's noise on a fresh pipeline."""
    fresh = api.ImagePipeline(dit=_Dit(), family=pipe.family,
                              variant=pipe.variant, encoder=_OwnedEncoder())
    cond = api.encode_text(fresh, prompt)
    return np.array(api.sample(fresh, cond, height=shape[2] * 8,
                               width=shape[3] * 8,
                               noise=noise_from_seed(seed, shape), **kw))


def test_cli_latents_equal_sample_with_the_seed_noise(env, tmp_path, capsys):
    lat = tmp_path / "lat.npz"
    rc = umbrella_main(env.base + ["--prompt", "a red bicycle", "--size",
                                   "48x32", "--seed", "11", "--count", "2",
                                   "--output", str(tmp_path / "out.png"),
                                   "--save-latent", str(lat), "--json"])
    assert rc == 0
    rec = json.loads(capsys.readouterr().out)
    assert [i["seed"] for i in rec["images"]] == [11, 12]
    assert [i["path"] for i in rec["images"]] == [
        str(tmp_path / "out-11.png"), str(tmp_path / "out-12.png")]
    saved = np.load(lat)
    assert saved["seeds"].tolist() == [11, 12]
    for i, seed in enumerate((11, 12)):
        want = _direct(env.pipes[0], "a red bicycle", seed)
        assert np.array_equal(saved["latents"][i:i + 1], want)
    assert env.checks[0]["height"] == 32 and env.checks[0]["width"] == 48


def test_a_variant_with_fixed_sigmas_runs_them_and_no_other_count(
        env, tmp_path, capsys, monkeypatch):
    import dataclasses

    fam = get_family("z_image")
    sigmas = (1.0, 0.9, 0.6, 0.3)
    monkeypatch.setitem(fam.sampling, "turbo", dataclasses.replace(
        fam.sampling["turbo"], steps=4, sigmas=sigmas))
    lat = tmp_path / "lat.npz"
    assert umbrella_main(env.base + ["--prompt", "p", "--size", "48x32",
                                     "--seed", "5", "--output",
                                     str(tmp_path / "a.png"),
                                     "--save-latent", str(lat)]) == 0
    assert env.pipes[-1].dit.calls == 4
    want = _direct(env.pipes[-1], "p", 5, sigmas=[*sigmas, 0.0])
    assert np.array_equal(np.load(lat)["latents"], want)
    assert umbrella_main(env.base + ["--prompt", "p", "--size", "32x32",
                                     "--steps", "6", "--output",
                                     str(tmp_path / "b.png")]) == 1
    assert "Z-Image turbo runs 4 steps" in capsys.readouterr().err


def test_model_may_name_an_image_models_entry(env, tmp_path, monkeypatch,
                                             capsys):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"server:\n  model_dirs:\n    - {tmp_path}\n"
                   "models:\n  qwen3-4b:\n    path: enc.gguf\n"
                   "image_models:\n  zit:\n    path: dit.gguf\n"
                   "    encoder: qwen3-4b\n    vae: vae.gguf\n")
    loads = []
    real = api.load
    monkeypatch.setattr(api, "load", lambda dit, **kw: loads.append(
        (dit, kw.get("text_encoder"), kw.get("vae"))) or real(dit, **kw))
    rc = umbrella_main(["image", "generate", "zit", "--config", str(cfg),
                        "--prompt", "p", "--size", "48x32", "--seed", "1",
                        "--output", str(tmp_path / "o.png"), "--json"])
    assert rc == 0
    # The entry brings the DiT, the shared chat model's GGUF and the VAE.
    f = env.files
    assert loads == [(f["dit.gguf"], f["enc.gguf"], f["vae.gguf"])]
    capsys.readouterr()
    rc = umbrella_main(["image", "generate", "nope", "--config", str(cfg),
                        "--prompt", "p"])
    assert rc != 0
    assert "no image model of that id" in capsys.readouterr().err


def test_an_entry_defaults_size_is_the_size_without_size(env, tmp_path):
    from PIL import Image

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"server:\n  model_dirs:\n    - {tmp_path}\n"
                   "image_models:\n  zit:\n    path: dit.gguf\n"
                   "    encoder: enc.gguf\n    vae: vae.gguf\n"
                   "    defaults: {size: 48x32}\n")
    base = ["image", "generate", "zit", "--config", str(cfg), "--prompt",
            "p", "--seed", "1", "--output"]
    assert umbrella_main(base + [str(tmp_path / "a.png")]) == 0
    assert umbrella_main(base + [str(tmp_path / "b.png"), "--size",
                                 "32x32"]) == 0
    with Image.open(tmp_path / "a.png") as a, \
            Image.open(tmp_path / "b.png") as b:
        assert (a.size, b.size) == ((48, 32), (32, 32))


def test_the_png_carries_the_settings_unless_told_not_to(env, tmp_path):
    from PIL import Image

    out = tmp_path / "a.png"
    assert umbrella_main(env.base + ["--prompt", "p", "--size", "32x32",
                                     "--seed", "3", "--output",
                                     str(out)]) == 0
    with Image.open(out) as im:
        meta = json.loads(im.text["gmlx"])
        assert im.size == (32, 32)
    assert meta["seed"] == 3 and meta["steps"] == 8
    assert meta["guidance"] == 1.0 and meta["sampler"] == "euler"
    assert "scheduler" in meta and meta["scheduler"] is None
    assert meta["files"]["dit"] == "dit.gguf" and meta["adapters"] == []
    assert "negative_prompt" not in meta
    out2 = tmp_path / "b.png"
    assert umbrella_main(env.base + ["--prompt", "p", "--size", "32x32",
                                     "--output", str(out2),
                                     "--no-metadata"]) == 0
    with Image.open(out2) as im:
        assert "gmlx" not in im.text


def test_output_naming():
    seeds = [5, 6]
    assert cmd.output_paths(None, [5], "png", "T") == ["gmlx-T-5.png"]
    assert cmd.output_paths("d/", seeds, "jpeg", "T") == [
        "d/gmlx-T-5.jpg", "d/gmlx-T-6.jpg"]
    assert cmd.output_paths("x_{index}_{seed}.webp", seeds, "webp", "T") == [
        "x_0_5.webp", "x_1_6.webp"]
    assert cmd.output_paths("f.png", [5], "png", "T") == ["f.png"]
    assert cmd.output_paths("f.png", seeds, "png", "T") == [
        "f-5.png", "f-6.png"]
    with pytest.raises(cmd._Usage, match="one file for 2 images"):
        cmd.output_paths("a{{b}}.png", seeds, "png", "T")
    with pytest.raises(cmd._Usage, match="template"):
        cmd.output_paths("{nope}.png", seeds, "png", "T")


def test_format_follows_the_output_and_the_flag(env, tmp_path):
    from PIL import Image

    assert umbrella_main(env.base + ["--prompt", "p", "--size", "32x32",
                                     "--output", str(tmp_path / "a.jpg")]) == 0
    with Image.open(tmp_path / "a.jpg") as im:
        assert im.format == "JPEG"
    assert umbrella_main(env.base + ["--prompt", "p", "--size", "32x32",
                                     "--format", "webp", "--output",
                                     str(tmp_path / "w")]) == 0
    with Image.open(tmp_path / "w") as im:
        assert im.format == "WEBP"


@pytest.fixture
def rgba(env, monkeypatch):
    """The env with a VAE row that decodes four channels, as the row of
    Qwen-Image 2.1 does."""
    import dataclasses

    from gmlx.image.vae import reader

    assert reader.VAES["qwen_image21"].image_channels == 4
    assert reader.VAES["z_image"].image_channels is None
    monkeypatch.setitem(reader.VAES, "z_image", dataclasses.replace(
        reader.VAES["z_image"], image_channels=4))
    return env


@pytest.mark.parametrize("fmt", ["png", "webp"])
def test_a_transparent_background_writes_the_alpha_channel(rgba, tmp_path,
                                                           capsys, fmt):
    from PIL import Image

    out = tmp_path / f"a.{fmt}"
    args = rgba.base + ["--prompt", "p", "--size", "32x32", "--seed", "3",
                        "--output", str(out), "--json"]
    assert umbrella_main(args + ["--background", "transparent"]) == 0
    rec = json.loads(capsys.readouterr().out)
    assert rec["settings"]["background"] == "transparent"
    with Image.open(out) as im:
        assert im.mode == "RGBA" and im.size == (32, 32)
        clear = np.asarray(im)
        if fmt == "png":
            assert json.loads(im.text["gmlx"])["background"] == "transparent"
    # The alpha of the decode, clear at the left and opaque at the right,
    # as straight alpha, exact in a png and in a lossy webp.
    assert clear[:, :8, 3].max() == 0 and clear[:, -8:, 3].min() == 255
    assert 0 < clear[0, 8, 3] < clear[0, 16, 3] < 255
    assert umbrella_main(args) == 0
    rec = json.loads(capsys.readouterr().out)
    assert "background" not in rec["settings"]
    with Image.open(out) as im:
        assert im.mode == "RGB"
        opaque = np.asarray(im)
    if fmt == "png":
        assert np.array_equal(clear[..., :3], opaque)
    # The decode ran with the alpha kept in the first run alone.
    assert [p.decoded for p in rgba.pipes] == [[(None, rgba.plan)]] * 2


def test_a_transparent_background_is_refused_as_a_jpeg(rgba, tmp_path,
                                                       capsys):
    for flags in (["--output", str(tmp_path / "a.jpg")],
                  ["--format", "jpeg", "--output", str(tmp_path / "a.png")]):
        assert umbrella_main(rgba.base + [
            "--prompt", "p", "--size", "32x32", "--background",
            "transparent"] + flags) == 2
        assert ("--background transparent: jpeg holds no alpha channel; "
                "write a png or a webp") in capsys.readouterr().err
    # Refused before any model loads, and nothing is written.
    assert rgba.pipes == [] and list(tmp_path.glob("a.*")) == []


def test_the_uint8_image_keeps_four_channels():
    from gmlx.commands.image import to_uint8

    x = mx.array(np.array([[[[0.0, 0.5, 1.0, 0.25]]]], np.float32))
    assert to_uint8(x).tolist() == [[[0, 128, 255, 64]]]
    assert to_uint8(x[..., :3]).shape == (1, 1, 3)


def test_guidance_encodes_the_negative_prompt(env, tmp_path, capsys):
    assert umbrella_main(env.base + ["--prompt", "p", "--size", "32x32",
                                     "--guidance", "3", "--negative-prompt",
                                     "blur", "--output",
                                     str(tmp_path / "a.png")]) == 0
    assert env.pipes[0].encoder.encoded == ["p", "blur"]
    assert env.pipes[0].dit.calls == 16
    assert umbrella_main(env.base + ["--prompt", "p", "--size", "32x32",
                                     "--negative-prompt", "blur", "--output",
                                     str(tmp_path / "b.png")]) == 0
    assert "--negative-prompt is unused" in capsys.readouterr().out
    assert env.pipes[1].encoder.encoded == ["p"]


def test_img2img_and_inpainting_reach_sample(env, tmp_path, monkeypatch):
    from PIL import Image

    Image.new("RGB", (70, 40), (200, 10, 10)).save(tmp_path / "in.png")
    mask = np.zeros((32, 64), np.uint8)
    mask[:, 32:] = 255
    Image.fromarray(mask, "L").save(tmp_path / "m.png")
    seen = []
    real = api.sample
    monkeypatch.setattr(api, "sample",
                        lambda *a, **k: seen.append(k) or real(*a, **k))
    assert umbrella_main(env.base + ["--prompt", "p", "--init-image",
                                     str(tmp_path / "in.png"), "--output",
                                     str(tmp_path / "a.png")]) == 0
    k = seen[-1]
    assert (k["width"], k["height"]) == (64, 32)
    assert k["denoise"] == 0.6 and k["mask"] is None
    assert k["init_latents"].shape == (1, 4, 4, 8)
    # The plan counts the encode of the init image, before the load and
    # after it, and the encode and the decode run as that plan says.
    assert env.checks[-1]["init_image"] is True
    assert env.planned[-1]["init_image"] is True
    pipe = env.pipes[-1]
    assert pipe.encoded_images == [(None, env.plan)]
    assert pipe.decoded == [(None, env.plan)]
    assert umbrella_main(env.base + ["--prompt", "p", "--init-image",
                                     str(tmp_path / "in.png"), "--mask",
                                     str(tmp_path / "m.png"), "--output",
                                     str(tmp_path / "b.png")]) == 0
    k = seen[-1]
    assert k["denoise"] == 1.0
    assert k["mask"].shape == (32, 64) and k["mask"][:, 40:].min() == 1.0
    assert k["mask"][:, :24].max() == 0.0


def _turned(tmp_path, name: str, upright, mode: str = "RGB"):
    """``upright`` saved on its side with EXIF orientation 6, as a phone
    camera stores a photo held upright."""
    from PIL import Image

    im = Image.fromarray(upright, mode).transpose(Image.Transpose.ROTATE_90)
    exif = Image.Exif()
    exif[0x0112] = 6
    im.save(tmp_path / name, exif=exif)
    return str(tmp_path / name)


def test_images_are_read_upright_by_their_exif_orientation(env, tmp_path,
                                                           monkeypatch):
    up = np.zeros((70, 40, 3), np.uint8)
    up[:35] = 255
    path = _turned(tmp_path, "in.png", up)
    got = cmd.read_image(path, None, 16)
    assert got.shape == (64, 32, 3)
    assert got[:30].min() == 1.0 and got[38:].max() == 0.0
    mask = np.zeros((32, 64), np.uint8)
    mask[:, 32:] = 255
    m = cmd.read_mask(_turned(tmp_path, "m.png", mask, "L"), (64, 32))
    assert np.array_equal(m, (mask > 0).astype(np.float32))
    # A JPEG init image through the command: its size is the upright one.
    seen = []
    real = api.sample
    monkeypatch.setattr(api, "sample",
                        lambda *a, **k: seen.append(k) or real(*a, **k))
    assert umbrella_main(env.base + ["--prompt", "p", "--init-image",
                                     _turned(tmp_path, "in.jpg", up),
                                     "--output",
                                     str(tmp_path / "a.png")]) == 0
    assert (seen[-1]["width"], seen[-1]["height"]) == (32, 64)


def test_an_unreadable_image_is_refused_by_its_flag(env, tmp_path, capsys):
    from PIL import Image

    junk, ok = tmp_path / "junk.png", tmp_path / "ok.png"
    junk.write_bytes(b"not an image")
    Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(ok)
    for flag, flags in (("--init-image", ["--init-image", str(junk)]),
                        ("--mask", ["--init-image", str(ok), "--mask",
                                    str(junk)])):
        assert umbrella_main(env.base + ["--prompt", "p"] + flags) == 1
        assert f"{flag} {junk}: cannot read it as an image" in \
            capsys.readouterr().err
    assert env.pipes == []


def test_encoder_release_follows_the_policy_and_the_plan(env, tmp_path):
    def run(*flags):
        assert umbrella_main(env.base + ["--prompt", "p", "--size", "32x32",
                                         "--output", str(tmp_path / "a.png"),
                                         *flags]) == 0
        return env.pipes[-1]

    assert not run().encoder.released
    assert run("--encoder-release", "always").encoder.released
    assert env.checks[-1]["init_image"] is False
    assert env.planned[-1]["init_image"] is False
    env.plan.release_encoder = True
    pipe = run()
    # The decode is handed the plan, which says how it runs, and no
    # choice of the command's own.
    assert pipe.encoder.released and pipe.decoded == [(None, env.plan)]


def test_never_releasing_an_encoder_the_plan_must_release_exits_2(
        env, tmp_path, capsys):
    args = env.base + ["--prompt", "p", "--size", "32x32", "--output",
                       str(tmp_path / "a.png"), "--encoder-release", "never"]
    assert umbrella_main(args) == 0
    env.plan.release_encoder = True
    assert umbrella_main(args) == 2
    err = capsys.readouterr().err
    assert "--encoder-release never: this request fits only with the text " \
           "encoder released; peak 9.00 GiB" in err
    n = len(env.pipes)
    env.checked = env.plan
    assert umbrella_main(args) == 2
    assert len(env.pipes) == n


def test_an_unknown_variant_needs_steps_and_guidance(env, tmp_path, capsys):
    env.variant = "unknown"
    out = str(tmp_path / "a.png")
    assert umbrella_main(env.base + ["--prompt", "p", "--size", "32x32",
                                     "--output", out]) == 1
    assert "pass --steps and --guidance" in capsys.readouterr().err
    assert umbrella_main(env.base + ["--prompt", "p", "--size", "32x32",
                                     "--steps", "2", "--guidance", "1",
                                     "--output", out]) == 0
    assert env.pipes[-1].dit.calls == 2


def test_a_krea_2_file_of_unknown_weights_runs_under_a_variant_by_name(
        env, tmp_path, monkeypatch, capsys):
    asked = []
    k2 = get_family("krea2")
    monkeypatch.setattr(image_preflight, "identify_dit",
                        lambda p, variant=None: asked.append(variant) or
                        SimpleNamespace(family=k2, view=SimpleNamespace(
                            tensors={}), variant=variant or "unknown"))
    out = str(tmp_path / "a.png")
    args = ["--prompt", "p", "--size", "32x32", "--output", out]
    assert umbrella_main(env.base + args) == 1
    err = capsys.readouterr().err
    assert "--variant turbo" in err and "`variant: turbo`" in err
    assert umbrella_main(env.base + args + ["--variant", "turbo"]) == 0
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"server:\n  model_dirs:\n    - {tmp_path}\n"
                   "image_models:\n  k2:\n    path: dit.gguf\n"
                   "    encoder: enc.gguf\n    vae: vae.gguf\n"
                   "    variant: turbo\n")
    assert umbrella_main(["image", "generate", "k2", "--config", str(cfg)]
                         + args) == 0
    assert asked == [None, "turbo", "turbo"]


@pytest.mark.parametrize("flags, code, text", [
    ([], 2, "--prompt or --prompt-file is required"),
    (["--prompt", "p", "--size", "40x32"], 2, "multiples of 16"),
    (["--prompt", "p", "--size", "big"], 2, "a size such as"),
    (["--prompt", "p", "--mask", "m.png"], 2, "--mask needs --init-image"),
    (["--prompt", "p", "--strength", "0.5"], 2, "needs --init-image"),
    (["--prompt", "p", "--strength", "1.5"], 2, "from 0.01 to 1"),
    (["--prompt", "p", "--strength", "0.005"], 2, "from 0.01 to 1"),
    (["--prompt", "p", "--quality", "0"], 2, "from 1 to 100"),
    (["--prompt", "p", "--sampler", "dpm"], 2, "invalid choice"),
    (["--prompt", "p", "--init-image", "missing.png"], 1, "no such file"),
    (["--prompt", "p", "--count", "2", "--output", "a{{b}}.png"], 2,
     "one file for 2 images"),
    (["--prompt", "p", "--background", "clear"], 2, "invalid choice"),
    (["--prompt", "p", "--background", "transparent"], 2,
     "--background transparent: Z-Image decodes no alpha channel"),
])
def test_refusals_and_usage_errors(env, flags, code, text, capsys):
    assert umbrella_main(env.base + flags) == code
    assert text in capsys.readouterr().err


def test_companions_are_required_and_must_exist(env, capsys):
    dit = env.files["dit.gguf"]
    assert umbrella_main(["image", "generate", dit, "--prompt", "p"]) == 2
    assert "--encoder and --vae are required" in capsys.readouterr().err
    assert umbrella_main(["image", "generate", "nope.gguf", "--prompt",
                          "p"]) == 1
    assert umbrella_main(["image", "generate", dit, "--prompt", "p",
                          "--encoder", "x.gguf", "--vae",
                          env.files["vae.gguf"]]) == 1
    assert "--encoder x.gguf: no such file" in capsys.readouterr().err


def test_a_prompt_file_and_an_unwritable_output(env, tmp_path, capsys):
    (tmp_path / "p.txt").write_text("from a file\n")
    assert umbrella_main(env.base + ["--prompt-file", str(tmp_path / "p.txt"),
                                     "--size", "32x32", "--output",
                                     str(tmp_path / "a.png")]) == 0
    assert env.pipes[-1].encoder.encoded == ["from a file"]
    (tmp_path / "f").write_text("")
    assert umbrella_main(env.base + ["--prompt", "p", "--output",
                                     str(tmp_path / "f" / "a.png")]) == 1
    assert "cannot write" in capsys.readouterr().err
    assert len(env.pipes) == 1


def test_a_random_seed_is_printed(env, tmp_path, capsys):
    assert umbrella_main(env.base + ["--prompt", "p", "--size", "32x32",
                                     "--output", str(tmp_path / "a.png"),
                                     "--json"]) == 0
    io = capsys.readouterr()
    seed = json.loads(io.out)["settings"]["seed"]
    assert f"[image] seed {seed}" in io.err


def test_an_interrupt_exits_130_and_leaves_no_partial_image(
        env, tmp_path, monkeypatch):
    from PIL import Image

    out = tmp_path / "a.png"
    args = env.base + ["--prompt", "p", "--size", "32x32", "--output",
                       str(out)]

    def stop(*a, **k):
        raise KeyboardInterrupt

    with monkeypatch.context() as m:
        m.setattr(api, "sample", stop)
        assert umbrella_main(args) == 130
    assert not out.exists()

    def half_written(self, fp, *a, **k):
        with open(fp, "wb") as f:
            f.write(b"\x89PNG partial")
        raise KeyboardInterrupt

    monkeypatch.setattr(Image.Image, "save", half_written)
    assert umbrella_main(args) == 130
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "dit.gguf", "enc.gguf", "vae.gguf"]


def test_bench_records_each_size_from_empty_caches(env, tmp_path, capsys):
    out = tmp_path / "bench.jsonl"
    rc = umbrella_main(["image", "bench"] + env.base[2:] + [
        "--sizes", "32x32,48x32", "--steps", "3", "--runs", "2",
        "--warmup", "1", "--label", "toy", "--json", "--json-out",
        str(out)])
    assert rc == 0
    io = capsys.readouterr()
    recs = [json.loads(line) for line in io.out.splitlines()]
    assert recs == [json.loads(line) for line in out.read_text().splitlines()]
    assert [(r["settings"]["width"], r["settings"]["height"]) for r in recs] \
        == [(32, 32), (48, 32)]
    pipe = env.pipes[0]
    assert len(env.pipes) == 1 and pipe.encoder.encoded == ["a red bicycle "
        "leaning on a brick wall, morning light"] * 6
    assert pipe.dit.calls == 18
    # A bench keeps its encoder, so no plan of it counts a release, and
    # every decode runs as the plan of its run says.
    assert env.checks[0]["keep_encoder"] is True
    assert len(env.planned) == 6
    assert all(k["keep_encoder"] is True for k in env.planned)
    assert [(k["width"], k["height"]) for k in env.planned] == [
        (32, 32)] * 3 + [(48, 32)] * 3
    assert pipe.decoded == [(None, env.plan)] * 6
    assert not pipe.encoder.released
    for r in recs:
        assert r["schema"] == "gmlx.image.bench/1" and r["label"] == "toy"
        assert set(r) >= {"gmlx_version", "mlx_version", "machine", "model",
                          "settings", "runs", "summary", "transport"}
        assert r["model"]["dit"] == {"file": "dit.gguf", "bytes": 1,
                                     "quant": "Q4_K"}
        assert r["model"]["encoder"]["shared"] is False
        assert r["settings"]["steps"] == 3 and r["settings"]["guidance"] == 1
        assert r["transport"]["mode"] == "direct"
        assert len(r["runs"]) == 2
        for run in r["runs"]:
            assert len(run["step_s"]) == 3 and run["peak_bytes"] >= 0
            assert run["image_s"] >= run["encode_s"] + sum(run["step_s"])
        sm = r["summary"]
        assert set(sm["load_s"]) == {"dit", "total"}
        assert set(sm["s_per_step"]) == {"first", "median", "mean"}
    assert "[bench] 48x32 3 steps" in io.err


def test_bench_summary_takes_medians_and_the_first_step_apart():
    from gmlx.commands.image import bench_summary

    runs = [{"encode_s": 1.0, "step_s": [5.0, 1.0, 3.0], "decode_s": 2.0,
             "image_s": 12.0, "peak_bytes": 7},
            {"encode_s": 3.0, "step_s": [7.0, 2.0, 2.0], "decode_s": 4.0,
             "image_s": 18.0, "peak_bytes": 9}]
    sm = bench_summary(runs, {"dit": 1.0, "encoder": 2.0, "vae": 0.5})
    assert sm["load_s"]["total"] == 3.5
    assert sm["encode_s"] == 2.0 and sm["decode_s"] == 3.0
    assert sm["s_per_step"] == {"first": 6.0, "median": 2.0, "mean": 2.0}
    assert sm["s_per_image"] == 15.0 and sm["peak_bytes"] == 9


@pytest.mark.parametrize("guidance", [[], ["--guidance", "3"]])
def test_bench_via_steps_gives_the_latents_of_sample(env, capsys, guidance,
                                                     monkeypatch):
    latents = []
    real = cmd.bench_run_steps

    def spy(*a, **k):
        rec, x = real(*a, **k)
        latents.append(x)
        return rec, x

    monkeypatch.setattr(cmd, "bench_run_steps", spy)
    # A real encoder's template gives the empty negative prompt rows. With
    # none, the guided velocity is NaN, and the CPU stream crashes on the
    # mean.
    monkeypatch.setattr(_OwnedEncoder, "token_ids",
                        lambda self, prompt: [0] + [ord(c) for c in prompt])
    rc = umbrella_main(["image", "bench"] + env.base[2:] + [
        "--sizes", "48x32", "--steps", "3", "--runs", "2", "--warmup", "1",
        "--seed", "5", "--via-steps", "--json", *guidance])
    io = capsys.readouterr()
    assert rc == 0, io.err
    rec = json.loads(io.out)
    t = rec["transport"]
    assert t["mode"] == "steps" and t["equal_to_direct"] is True
    assert t["overhead_ms_median"] >= 0
    assert [len(r["transport_ms"]) for r in rec["runs"]] == [3, 3]
    assert all(len(r["step_s"]) == 3 for r in rec["runs"])
    fresh = api.ImagePipeline(dit=_Dit(), family=env.pipes[0].family,
                              variant="turbo", encoder=_OwnedEncoder())
    kw = {"cfg": 3.0, "neg": api.encode_text(fresh, "")} if guidance else {}
    want = _direct(env.pipes[0], rec["settings"]["prompt"], 5, steps=3, **kw)
    assert len(latents) == 3
    assert np.isfinite(want).all()
    assert all(np.array_equal(x, want) for x in latents)
    assert "via the step routes: transport" in io.err
    # The bench's runs and routes end with it.
    from gmlx.serve.image_steps import STATE
    assert STATE.runs == {}


def test_bench_via_steps_refuses_adapters(env, capsys):
    assert umbrella_main(["image", "bench"] + env.base[2:] + [
        "--via-steps", "--adapter", "a.safetensors"]) == 2
    assert "drop --adapter" in capsys.readouterr().err
    assert env.pipes == []


def test_bench_refuses_sizes_off_the_stride(env, capsys):
    assert umbrella_main(["image", "bench"] + env.base[2:] + [
        "--sizes", "1024x1000"]) == 2
    assert "multiples of 16" in capsys.readouterr().err
    assert env.pipes == []


def test_the_preflight_plans_from_headers_and_refuses_what_cannot_fit(
        monkeypatch):
    from gmlx.image import memfit
    from gmlx.serve import capacity

    pf = image_preflight
    fam = get_family("z_image")
    monkeypatch.setattr(pf, "identify_dit", lambda p, **k: SimpleNamespace(
        family=fam, variant="turbo"))
    sizes = {"dit": 5 << 30, "encoder": 2 << 30, "vae": 160 << 20}
    monkeypatch.setattr(api, "_header_bytes",
                        lambda path, kind: 0 if path is None else sizes[kind])
    seen = []
    monkeypatch.setattr(pf, "dit_transient", lambda dit, act: 0)
    monkeypatch.setattr(pf, "load_transient",
                        lambda *parts: seen.append(parts) or 300 << 20)
    monkeypatch.delenv("GMLX_OVERCOMMIT", raising=False)
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: 100 << 30)
    plan = pf.check("d", "e", "v", height=1024, width=1024, cfg=None)
    assert plan.fits and plan.terms["dit"] == sizes["dit"]
    assert plan.terms["encoder"] == sizes["encoder"]
    # The load is a phase of the plan the command makes before it loads:
    # every part resident, and what the load holds beside them.
    assert list(plan.phases) == ["load", "encode", "denoise", "decode"]
    # The load of every part together, then of each part alone, which the
    # staged order reads.
    assert seen == [("e", "v", None), ("e", None, None), (None, "v", None)]
    assert not plan.staged_load
    assert plan.phases["load"] == sum(sizes.values()) + (300 << 20)
    # An init image adds the phase of its encode.
    init = pf.check("d", "e", "v", height=1024, width=1024, cfg=None,
                    init_image=True)
    assert list(init.phases) == ["load", "encode", "init", "denoise",
                                 "decode"]
    assert init.terms["init_peak"] == memfit.vae_peak(
        "encode", 1024, 1024, family="z_image", itemsize=2)
    # At 12 GiB the request fits with the encoder released. A caller
    # that keeps the encoder has no such lever, and tiles the decode.
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: 12 << 30)
    freed = pf.check("d", "e", "v", height=1024, width=1024, cfg=None)
    assert freed.levers == ("release the encoder",)
    kept = pf.check("d", "e", "v", height=1024, width=1024, cfg=None,
                    keep_encoder=True)
    assert kept.levers == ("tiled decode",) and not kept.release_encoder
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: 6 << 30)
    with pytest.raises(memfit.ImageMemoryError):
        pf.check("d", "e", "v", height=1024, width=1024, cfg=1.0)
    monkeypatch.setenv("GMLX_TOOL_PREFLIGHT", "0")
    assert pf.check("d", "e", "v", height=1024, width=1024, cfg=1.0) is None
    monkeypatch.delenv("GMLX_TOOL_PREFLIGHT")
    monkeypatch.setattr(pf, "identify_dit", lambda p, **k: SimpleNamespace(
        family=fam, variant="unknown"))
    assert pf.check("d", "e", "v", height=1024, width=1024, cfg=None) is None


def test_the_preflight_counts_the_dit_tensors_its_load_casts(monkeypatch):
    """A Qwen-Image 2.1 F16 DiT loads under bf16 activations, so each of
    its tensors is held beside its cast copy: on a 32 GB Mac the load
    peaked at 28.2 GiB against a budget of 20.3 GiB. The plan counts the
    cast in each load phase and refuses the file, as it would a quant too
    large."""
    from gmlx.image import memfit
    from gmlx.image import text_encoder as te
    from gmlx.serve import capacity

    pf = image_preflight
    fam = get_family("qwen_image21")
    monkeypatch.setattr(pf, "identify_dit", lambda p, **k: SimpleNamespace(
        family=fam, variant="turbo"))
    weights = int(13.25 * memfit.GiB)
    sizes = {"dit": weights, "encoder": 4 << 30, "vae": 1 << 30}
    monkeypatch.setattr(api, "_header_bytes",
                        lambda path, kind: 0 if path is None else sizes[kind])
    monkeypatch.setattr(pf, "load_transient", lambda e, v, m: 300 << 20)

    def view(codec):
        return SimpleNamespace(tensors={"w": SimpleNamespace(
            shape=(), codec=codec, nbytes=weights)})

    monkeypatch.delenv("GMLX_OVERCOMMIT", raising=False)
    monkeypatch.setattr(capacity, "working_budget_bytes",
                        lambda: int(20.27 * memfit.GiB))
    kw = dict(height=1024, width=1024, cfg=None)
    monkeypatch.setattr(te, "_header_view", lambda path: view("BF16"))
    assert pf.check("d", "e", "v", **kw).fits
    monkeypatch.setattr(te, "_header_view", lambda path: view("F16"))
    with pytest.raises(memfit.ImageMemoryError) as e:
        pf.check("d", "e", "v", **kw)
    assert e.value.plan.terms["load_peak"] >= weights
    assert "smaller DiT quant" in str(e.value)


def test_the_preflight_stages_the_load_when_no_other_lever_fits(
        monkeypatch):
    from gmlx.image import memfit
    from gmlx.serve import capacity

    pf = image_preflight
    fam = get_family("z_image")
    monkeypatch.setattr(pf, "identify_dit", lambda p, **k: SimpleNamespace(
        family=fam, variant="turbo"))
    sizes = {"dit": 3 << 30, "encoder": 4 << 30, "vae": 160 << 20}
    monkeypatch.setattr(api, "_header_bytes",
                        lambda path, kind: 0 if path is None else sizes[kind])
    monkeypatch.setattr(pf, "dit_transient", lambda dit, act: 0)
    monkeypatch.setattr(pf, "load_transient", lambda e, v, m: (
        (500 << 20 if e else 0) if v is None else 300 << 20 if e is None
        else 500 << 20))
    monkeypatch.delenv("GMLX_OVERCOMMIT", raising=False)
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: 7 << 30)
    kw = dict(height=512, width=512, cfg=None)
    plan = pf.check("d", "e", "v", **kw)
    assert plan.fits and plan.staged_load and plan.release_encoder
    assert plan.levers == ("release the encoder", "staged load")
    room = memfit.STAGE_ROOM
    assert plan.phases["load_encoder"] == (4 << 30) + (500 << 20) + room
    assert plan.phases["load"] == ((3 << 30) + (160 << 20) + (300 << 20)
                                   + room)
    # A caller that keeps its encoder, as bench does, cannot stage it.
    with pytest.raises(memfit.ImageMemoryError) as e:
        pf.check("d", "e", "v", keep_encoder=True, **kw)
    assert not e.value.plan.staged_load
    assert "the encoder is not released in this request" in str(e.value)
    # With every part in the budget the plan is the one without stages.
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: 12 << 30)
    whole = pf.check("d", "e", "v", **kw)
    assert whole.fits and not whole.staged_load and not whole.levers


class _TinyEncoder(_OwnedEncoder):
    """Features of the tiny Z-Image DiT's caption width."""

    def encode(self, prompt):
        self.encoded.append(prompt)
        ids = np.array(self.token_ids(prompt), np.float32)[:, None]
        h = (ids % 7 - 3) * np.linspace(0.1, 1.0, 16, dtype=np.float32)
        return TextTap(ids=tuple(self.token_ids(prompt)),
                       hidden=(mx.array(h)[None],), template_id="",
                       tap_layers=())


def _tiny_pipe(log=print):
    from image_fixtures import tiny_z_model

    m, _ = tiny_z_model()
    pipe = api.ImagePipeline(dit=m, family=get_family("z_image"),
                             variant="turbo", encoder=_TinyEncoder(),
                             vae=object(), load_seconds={"dit": 0.0}, log=log)
    pipe.decoded, pipe.encoded_images = [], []
    return pipe


@pytest.fixture
def tiny(env, monkeypatch, tmp_path):
    """The env with a tiny Z-Image DiT behind the loader, and two adapter
    files beside the model."""
    from lora_fixtures import tiny_z_adapter, write_adapter

    def load(dit, *, text_encoder=None, vae=None, log=print, **kw):
        env.pipes.append(_tiny_pipe(log))
        return env.pipes[-1]

    monkeypatch.setattr(api, "load", load)
    # The header of the tiny DiT's file: its config and its tensor names.
    from image_fixtures import TINY_Z_CONFIG, tiny_z_model
    from mlx.utils import tree_flatten

    names = {k: SimpleNamespace(shape=tuple(v.shape), codec="Q4_K")
             for k, v in tree_flatten(tiny_z_model()[0].parameters())}
    monkeypatch.setattr(image_preflight, "identify_dit",
                        lambda p, **k: SimpleNamespace(
                            family=get_family("z_image"), variant=env.variant,
                            config=dict(TINY_Z_CONFIG),
                            view=SimpleNamespace(tensors=names)))
    env.a = write_adapter(tmp_path / "a.safetensors", "kohya",
                          tiny_z_adapter(1, 4.0, wide=True))
    env.b = write_adapter(tmp_path / "b.safetensors", "diffusers",
                          tiny_z_adapter(2))
    return env


def _staged_plan(env):
    """A plan made before the load that takes the staged load."""
    return SimpleNamespace(
        release_encoder=True, staged_load=True, tiled_decode=False, rows=1,
        levers=("release the encoder", "staged load"),
        describe=lambda: "peak 7.00 GiB against a budget of 8.00 GiB")


@pytest.fixture
def staged(env, monkeypatch):
    """The env with the real ``api.load_staged`` over stub loaders, and a
    plan before the load that takes the staged load. ``env.events`` holds
    each load and encode with (the DiT is loaded, the VAE is loaded, the
    encoder is resident) as the pipeline had them then."""
    from gmlx.image import loader, text_encoder
    from gmlx.image import vae as vae_pkg

    fam = get_family("z_image")
    ident = SimpleNamespace(family=fam, variant="turbo", config={},
                            view=SimpleNamespace(tensors={}))
    env.events, env.make_dit, env.encoder = [], _Dit, _OwnedEncoder
    env.ident = ident

    def state(pipe):
        return (pipe.dit is not None, pipe.vae is not None,
                not pipe.encoder.released)

    def load_encoder(path, spec, log=print, mmproj=None):
        env.events.append(("load_encoder", mmproj))
        return env.encoder()

    def load_dit(path, **kw):
        env.events.append(("load_dit", state(env.pipes[-1])))
        return env.make_dit(), SimpleNamespace(identified=env.ident)

    def load_vae(path, family, dtype):
        env.events.append(("load_vae", state(env.pipes[-1])))
        return object()

    real = api.encode_text

    def encode_text(pipe, prompt, **kw):
        if not env.pipes or env.pipes[-1] is not pipe:
            pipe.decoded, pipe.encoded_images = [], []
            env.pipes.append(pipe)
        env.events.append(("encode", prompt, state(pipe)))
        return real(pipe, prompt, **kw)

    monkeypatch.setattr(loader, "identify_dit",
                        lambda path, variant=None: env.ident)
    monkeypatch.setattr(loader, "load_dit", load_dit)
    monkeypatch.setattr(text_encoder, "load_encoder", load_encoder)
    monkeypatch.setattr(vae_pkg, "load_vae", load_vae)
    monkeypatch.setattr(api, "_header_bytes", lambda path, kind: 0)
    monkeypatch.setattr(api, "encode_text", encode_text)
    env.checked = _staged_plan(env)
    return env


def test_a_staged_plan_loads_the_dit_after_the_encoder_is_gone(
        staged, tmp_path, capsys):
    lat = tmp_path / "lat.npz"
    args = ["--prompt", "a red bicycle", "--negative-prompt", "blur",
            "--guidance", "3", "--size", "48x32", "--seed", "7",
            "--save-latent", str(lat), "--output", str(tmp_path / "a.png"),
            "--json"]
    assert umbrella_main(staged.base + args) == 0
    io = capsys.readouterr()
    rec = json.loads(io.out)
    # The encoder loads and encodes with no DiT and no VAE beside it, and
    # is gone when they load.
    assert staged.events == [
        ("load_encoder", None),
        ("encode", "a red bicycle", (False, False, True)),
        ("encode", "blur", (False, False, True)),
        ("load_dit", (False, False, False)),
        ("load_vae", (True, False, False))]
    pipe = staged.pipes[-1]
    assert pipe.encoder.released and pipe.dit.calls > 0
    assert "loading in stages" in io.err
    # The request plan is made after the release, and the record holds
    # the levers of the run.
    assert rec["encoder_released"] is True
    assert rec["levers"] == ["release the encoder", "staged load"]
    staged.plan.levers = ("sequential rows", "tiled decode")
    assert umbrella_main(staged.base + args) == 0
    assert json.loads(capsys.readouterr().out)["levers"] == [
        "sequential rows", "release the encoder", "tiled decode",
        "staged load"]
    staged_latents = np.load(lat)["latents"]
    # The latents are those of the same request loaded whole.
    staged.checked = None
    monkeypatch_events = len(staged.events)
    assert umbrella_main(staged.base + args) == 0
    rec = json.loads(capsys.readouterr().out)
    assert rec["levers"] == ["sequential rows", "tiled decode"]
    assert rec["encoder_released"] is False
    # The whole load went through api.load: no stub loader ran.
    assert [e[0] for e in staged.events[monkeypatch_events:]] == [
        "encode", "encode"]
    assert np.array_equal(np.load(lat)["latents"], staged_latents)


def test_never_releasing_the_encoder_of_a_staged_plan_exits_2(
        staged, tmp_path, capsys):
    args = staged.base + ["--prompt", "p", "--size", "32x32", "--output",
                          str(tmp_path / "a.png"), "--encoder-release",
                          "never"]
    assert umbrella_main(args) == 2
    err = capsys.readouterr().err
    assert "--encoder-release never: this request fits only with the text " \
           "encoder released before the DiT loads; peak 7.00 GiB" in err
    assert staged.events == [] and staged.pipes == []
    # always and auto run it staged.
    for policy in ("always", "auto"):
        staged.events.clear()
        assert umbrella_main(args[:-1] + [policy]) == 0
        assert [e[0] for e in staged.events] == [
            "load_encoder", "encode", "load_dit", "load_vae"]


def test_a_staged_plan_reads_adapters_before_the_encode(staged, tiny,
                                                        tmp_path, capsys,
                                                        monkeypatch):
    """Under the staged order the adapters are read in the encoder stage,
    where their text encoder halves go on before the encode, from the
    DiT's header. The patch table is defined after the DiT loads, and the
    latents are those of the request loaded whole."""
    from image_fixtures import TINY_Z_CONFIG, tiny_z_model
    from mlx.utils import tree_flatten

    from gmlx.image import lora

    # The header of the tiny DiT, with shapes in the GGUF's order.
    names = {k: SimpleNamespace(shape=tuple(reversed(v.shape)), codec="F32")
             for k, v in tree_flatten(tiny_z_model()[0].parameters())}
    staged.ident = SimpleNamespace(
        family=get_family("z_image"), variant="turbo",
        config=dict(TINY_Z_CONFIG), view=SimpleNamespace(tensors=names))
    staged.make_dit = lambda: tiny_z_model()[0]
    staged.encoder = _TinyEncoder
    real_text, real_table = lora.apply_text_adapters, api.define_patch_table

    def text(pipe, rows):
        staged.events.append(("text halves", pipe.dit is not None))
        return real_text(pipe, rows)

    def table(pipe, entries):
        staged.events.append(("table", pipe.dit is not None,
                              pipe.encoder.released))
        return real_table(pipe, entries)

    monkeypatch.setattr(lora, "apply_text_adapters", text)
    monkeypatch.setattr(api, "define_patch_table", table)
    lat = tmp_path / "lat.npz"
    args = ["--prompt", "a red bicycle", "--size", "32x32", "--seed", "5",
            "--steps", "3", "--output", str(tmp_path / "o.png"),
            "--save-latent", str(lat), "--json", "--adapter",
            f"{tiny.a}:0.5", "--adapter", tiny.b]
    assert umbrella_main(staged.base + args) == 0
    rec = json.loads(capsys.readouterr().out)
    assert [e[0] for e in staged.events] == [
        "load_encoder", "text halves", "encode", "load_dit", "load_vae",
        "table"]
    assert staged.events[1] == ("text halves", False)
    assert staged.events[-1] == ("table", True, True)
    want, tid, shas = _direct_adapters("a red bicycle", 5,
                                       [(tiny.a, 0.5), (tiny.b, 1.0)],
                                       steps=3)
    assert np.array_equal(np.load(lat)["latents"], want)
    assert tid in staged.pipes[-1].patch_tables
    assert [r["rows_sha256"] for r in rec["settings"]["adapters"]] == shas
    assert rec["levers"] == ["release the encoder", "staged load"]


def _direct_adapters(prompt, seed, rows, shape=(1, 4, 4, 4), **kw):
    """sample on a fresh tiny pipeline under the table of (path, strength)
    rows, and the table's id."""
    from gmlx.image import lora

    fresh = _tiny_pipe()
    loaded = [(lora.load_adapter(fresh, p), s) for p, s in rows]
    tid = api.define_patch_table(fresh, lora.table_entries(loaded))
    x = api.sample(fresh, api.encode_text(fresh, prompt),
                   height=shape[2] * 8, width=shape[3] * 8,
                   noise=noise_from_seed(seed, shape), patch_table=tid, **kw)
    return np.array(x), tid, [ad.sha256 for ad, _ in loaded]


def _digest(path) -> str:
    import hashlib

    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@pytest.mark.parametrize("text, want", [
    ("a.safetensors", ("a.safetensors", 1.0)),
    ("a.safetensors:0.5", ("a.safetensors", 0.5)),
    ("a.safetensors:-1", ("a.safetensors", -1.0)),
    ("a.safetensors:1e-1", ("a.safetensors", 0.1)),
    ("my:lora.safetensors", ("my:lora.safetensors", 1.0)),
    ("my:lora.safetensors:2", ("my:lora.safetensors", 2.0)),
    ("dir/0.5", ("dir/0.5", 1.0)),
    ("a.safetensors:strong", ("a.safetensors:strong", 1.0)),
    (":0.5", (":0.5", 1.0)),
])
def test_the_adapter_tail_is_a_strength_only_when_it_is_a_float(text, want):
    assert cmd._adapter(text) == want


@pytest.mark.parametrize("tail", ["1_0", "1_000.5", "1e1_0", "-0_5"])
def test_a_strength_with_an_underscore_is_refused(tail):
    import argparse

    assert float(tail)      # Python reads it, as 10 for the first one
    with pytest.raises(argparse.ArgumentTypeError,
                       match="without underscores"):
        cmd._adapter(f"a.safetensors:{tail}")
    # An underscore in the path, or in a tail that is no number, is fine.
    assert cmd._adapter("my_lora.safetensors:0.5") == \
        ("my_lora.safetensors", 0.5)
    assert cmd._adapter("dir:my_lora_1") == ("dir:my_lora_1", 1.0)


def test_adapters_reach_sample_as_one_patch_table(tiny, tmp_path, capsys,
                                                  monkeypatch):
    from PIL import Image

    from gmlx.image import lora

    texts = []
    real = lora.apply_text_adapters
    monkeypatch.setattr(lora, "apply_text_adapters",
                        lambda pipe, rows: texts.append(rows)
                        or real(pipe, rows))
    lat, out = tmp_path / "lat.npz", tmp_path / "out.png"
    args = ["--prompt", "a red bicycle", "--size", "32x32", "--seed", "5",
            "--steps", "3", "--output", str(out), "--save-latent", str(lat),
            "--json"]
    rc = umbrella_main(tiny.base + args + ["--adapter", f"{tiny.a}:0.5",
                                           "--adapter", tiny.b])
    assert rc == 0
    io = capsys.readouterr()
    rec = json.loads(io.out)
    want, tid, shas = _direct_adapters("a red bicycle", 5,
                                       [(tiny.a, 0.5), (tiny.b, 1.0)],
                                       steps=3)
    assert np.array_equal(np.load(lat)["latents"], want)
    pipe = tiny.pipes[0]
    assert tid in pipe.patch_tables and len(pipe.adapter_files) == 2
    # sha256 is the digest of the file, and the hash of the placed rows
    # has its own name.
    adapters = [{"file": "a.safetensors", "strength": 0.5,
                 "sha256": _digest(tiny.a), "rows_sha256": shas[0]},
                {"file": "b.safetensors", "strength": 1.0,
                 "sha256": _digest(tiny.b), "rows_sha256": shas[1]}]
    assert _digest(tiny.a) != shas[0]
    assert rec["settings"]["adapters"] == adapters
    with Image.open(out) as im:
        assert json.loads(im.text["gmlx"])["adapters"] == adapters
    assert "adapter a.safetensors at strength 0.5: 13 pairs" in io.err
    assert "warning: " in io.err and "underscore keys spell diffusers " \
        "names" in io.err
    # The text halves went to the encoder, and the fit check saw the files.
    assert [[s for _, s in rows] for rows in texts] == [[0.5, 1.0]]
    assert tiny.checks[0]["adapters"] == [2 * os.path.getsize(tiny.a),
                                          2 * os.path.getsize(tiny.b)]
    # Without the flag the latents are those of the base model.
    assert umbrella_main(tiny.base + args) == 0
    capsys.readouterr()
    base = np.load(lat)["latents"]
    assert not np.array_equal(base, want)
    assert tiny.pipes[1].lora is None and tiny.checks[1]["adapters"] == []
    # A strength of zero is the base model too.
    assert umbrella_main(tiny.base + args + ["--adapter",
                                             f"{tiny.a}:0"]) == 0
    capsys.readouterr()
    assert np.array_equal(np.load(lat)["latents"], base)


def test_the_command_plans_the_conditionings_and_the_table_it_samples(
        tiny, tmp_path, capsys, monkeypatch):
    """The plan of a request counts the states that its conditionings
    hold under its patch table, so the command hands the plan what it
    hands ``sample``: the positive conditioning, the negative one when
    guided, and the table."""
    planned, sampled = [], []
    real = api.sample
    monkeypatch.setattr(api, "plan_request",
                        lambda pipe, **kw: planned.append(kw) or tiny.plan)
    monkeypatch.setattr(api, "sample",
                        lambda pipe, cond, **kw: sampled.append((cond, kw))
                        or real(pipe, cond, **kw))
    args = ["--prompt", "p", "--size", "32x32", "--seed", "5", "--steps",
            "2", "--output", str(tmp_path / "o.png")]
    guide = ["--guidance", "3", "--negative-prompt", "blur"]
    for extra, guided, adapted in (
            ([], False, False), (guide, True, False),
            (["--adapter", tiny.a], False, True),
            ([*guide, "--adapter", f"{tiny.b}:0.5"], True, True)):
        assert umbrella_main(tiny.base + args + extra) == 0
        plan, (cond, kw), pipe = planned[-1], sampled[-1], tiny.pipes[-1]
        assert (kw["neg"] is not None) == guided
        want = [cond, kw["neg"]] if guided else [cond]
        assert len(plan["conds"]) == len(want)
        assert all(a is b for a, b in zip(plan["conds"], want))
        assert plan["patch_table"] == kw["patch_table"]
        # The run follows the plan it was admitted under.
        assert kw["plan"] is tiny.plan
        assert (plan["patch_table"] not in (None, pipe.default_table)) \
            == adapted
        assert (plan["height"], plan["width"], plan["cfg"]) == (
            32, 32, 3.0 if guided else 1.0)
    assert len(planned) == len(sampled) == 4
    capsys.readouterr()


def test_adapter_refusals_and_usage_errors(tiny, tmp_path, capsys):
    from lora_fixtures import make_pair, tiny_z_adapter, write_adapter, \
        write_keys

    rng = np.random.default_rng(0)
    run = ["--prompt", "p", "--size", "32x32", "--seed", "1"]
    assert umbrella_main(tiny.base + run + ["--adapter",
                                            "nope.safetensors:0.5"]) == 1
    assert "--adapter nope.safetensors: no such file" in \
        capsys.readouterr().err
    assert tiny.pipes == []
    assert umbrella_main(tiny.base + run + ["--adapter",
                                            f"{tiny.a}:nan"]) == 2
    assert "finite strength" in capsys.readouterr().err
    flux = write_adapter(tmp_path / "flux.safetensors", "kohya", {
        "double_blocks.0.img_attn.qkv": make_pair(rng, 8, 8, 2)})
    extra = dict(tiny_z_adapter(1))
    extra["layers.9.attention.to_q"] = make_pair(rng, 64, 64, 2)
    extra = write_adapter(tmp_path / "extra.safetensors", "diffusers", extra)
    d, u, _ = make_pair(rng, 64, 64, 2)
    stem = "lora_unet_layers_0_attention_out"
    loha = write_keys(tmp_path / "loha.safetensors", {
        f"{stem}.hada_w1_a": u, f"{stem}.hada_w1_b": d,
        f"{stem}.hada_w2_a": u, f"{stem}.hada_w2_b": d})
    dora = write_keys(tmp_path / "dora.safetensors", {
        f"{stem}.lora_down.weight": d, f"{stem}.lora_up.weight": u,
        f"{stem}.dora_scale": np.ones((64, 1), np.float32)})
    stray = write_keys(tmp_path / "stray.safetensors", {
        f"{stem}.lora_down.weight": d, f"{stem}.lora_up.weight": u,
        f"{stem}.mystery": d})
    junk = tmp_path / "junk.safetensors"
    junk.write_bytes(b"not a safetensors file at all")
    half = write_keys(tmp_path / "half.safetensors", {
        f"{stem}.lora_down.weight": d})
    alone = write_keys(tmp_path / "alone.safetensors", {
        f"{stem}.lora_down.weight": d, f"{stem}.lora_up.weight": u,
        "lora_unet_layers_1_attention_out.alpha": np.float32(4.0)})
    twice = write_keys(tmp_path / "twice.safetensors", {
        f"{stem}.lora_down.weight": d, f"{stem}.lora_up.weight": u,
        "transformer.layers.0.attention.to_out.0.lora_A.weight": d,
        "transformer.layers.0.attention.to_out.0.lora_B.weight": u})
    # A file that is refused by its key names, or is not safetensors, is
    # refused from its header, before any model loads. So is a file whose
    # keys name no module of this DiT: its GGUF header gives the family
    # and the modules it holds. So is a file with half a pair, an alpha
    # alone or two stems on one module, which the header's names show.
    for path, text in ((half, f"half.safetensors: {stem} has no up tensor"),
                       (alone, "lora_unet_layers_1_attention_out has an "
                               "alpha and no pair"),
                       (twice, "both target layers.0.attention.out"),
                       (loha, "loha.safetensors is a LoHa adapter"),
                       (dora, "dora.safetensors is a DoRA adapter"),
                       (stray, "1 of 3 keys have no known LoRA suffix"),
                       (str(junk), "is not a safetensors file"),
                       (flux, "flux.safetensors is not a Z-Image adapter"),
                       (extra, "1 of 9 targets name no Z-Image module")):
        for action in (tiny.base + run, [
                "image", "bench", tiny.files["dit.gguf"], "--encoder",
                tiny.files["enc.gguf"], "--vae", tiny.files["vae.gguf"],
                "--sizes", "32x32", "--steps", "1", "--runs", "1"]):
            assert umbrella_main(action + ["--adapter", tiny.a,
                                           "--adapter", path]) == 1
            assert text in capsys.readouterr().err
            assert tiny.pipes == []
    assert not list(tmp_path.glob("gmlx-*.png"))


def test_an_adapter_for_a_module_the_dit_file_lacks_is_refused_unloaded(
        tiny, monkeypatch, capsys):
    """The vet gets the module paths the DiT file holds a weight for. A
    file that targets a module of the family which this DiT file lacks is
    refused before any model loads, by generate and by bench."""
    ident = image_preflight.identify_dit(tiny.files["dit.gguf"])
    gone = "layers.0.feed_forward.w1"
    held = {k: v for k, v in ident.view.tensors.items()
            if k != f"{gone}.weight"}
    assert len(held) == len(ident.view.tensors) - 1
    monkeypatch.setattr(image_preflight, "identify_dit",
                        lambda p, **k: SimpleNamespace(
                            family=ident.family, variant=ident.variant,
                            config=ident.config,
                            view=SimpleNamespace(tensors=held)))
    for action in (tiny.base + ["--prompt", "p", "--size", "32x32",
                                "--seed", "1"],
                   ["image", "bench", tiny.files["dit.gguf"], "--encoder",
                    tiny.files["enc.gguf"], "--vae", tiny.files["vae.gguf"],
                    "--sizes", "32x32", "--steps", "1", "--runs", "1"]):
        assert umbrella_main(action + ["--adapter", tiny.b,
                                       "--adapter", tiny.a]) == 1
        err = capsys.readouterr().err
        assert f"targets {gone}, which this DiT does not have" in err
        assert "b.safetensors" in err
        assert tiny.pipes == []


def test_bench_runs_under_the_adapters_and_records_them(tiny, capsys,
                                                        monkeypatch):
    tables = []
    real = api.sample
    monkeypatch.setattr(api, "sample", lambda *a, **k: tables.append(
        k.get("patch_table")) or real(*a, **k))
    rc = umbrella_main(["image", "bench", tiny.files["dit.gguf"],
                        "--encoder", tiny.files["enc.gguf"], "--vae",
                        tiny.files["vae.gguf"], "--sizes", "32x32",
                        "--steps", "2", "--runs", "2", "--json",
                        "--adapter", f"{tiny.a}:0.8"])
    assert rc == 0
    rec = json.loads(capsys.readouterr().out)
    seen = list(tables)
    _, tid, shas = _direct_adapters("p", 0, [(tiny.a, 0.8)], steps=1)
    assert rec["settings"]["adapters"] == [
        {"file": "a.safetensors", "strength": 0.8,
         "sha256": _digest(tiny.a), "rows_sha256": shas[0]}]
    # One warm-up and two timed runs, each under the table.
    assert seen == [tid] * 3
    pipe = tiny.pipes[0]
    assert len(pipe.adapter_files) == 1 and len(pipe.tables) >= 1


def test_the_adapter_flag_completes_as_a_file():
    from gmlx.commands.completion import _complete

    for action in ("generate", "bench"):
        flags = [c.split("\t")[0] for c in _complete(["image", action, "-"])]
        assert "--adapter" in flags
        assert _complete(["image", action, "x.gguf", "--adapter", ""]) == \
            ["::files"]


# Reference images.

@pytest.fixture
def edit(env, monkeypatch, tmp_path):
    """The env on a family that takes reference images: the toy pipeline
    under the reference rule of Qwen-Image 2.1. The encoder and the VAE
    of the references are replaced, and their calls are kept."""
    import dataclasses

    from PIL import Image

    fam = dataclasses.replace(get_family("z_image"),
                              edit=get_family("qwen_image21").edit)
    state = env
    state.calls, state.loads = [], []

    def load(dit, *, text_encoder=None, vae=None, mmproj=None, log=print,
             **kw):
        state.loads.append(mmproj)
        pipe = api.ImagePipeline(dit=_Dit(), family=fam,
                                 variant=state.variant,
                                 encoder=_OwnedEncoder(), vae=object(),
                                 load_seconds={"dit": 0.0}, log=log)
        pipe.decoded, pipe.encoded_images = [], []
        state.pipes.append(pipe)
        return pipe

    text = api.encode_text

    def encode_text(pipe, prompt, *, images=(), ref_latents=True):
        state.calls.append(("encode", prompt, tuple(images), ref_latents,
                            pipe.encoder.released))
        cond = text(pipe, prompt)
        return dataclasses.replace(cond, refs=tuple(images))

    def encode_references(pipe, cond, *, plan=None):
        state.calls.append(("vae", cond.refs, plan, pipe.encoder.released))
        return cond

    monkeypatch.setattr(api, "load", load)
    monkeypatch.setattr(api, "encode_text", encode_text)
    monkeypatch.setattr(api, "encode_references", encode_references)
    view = image_preflight.identify_dit("x").view
    monkeypatch.setattr(image_preflight, "identify_dit",
                        lambda p, **k: SimpleNamespace(family=fam, view=view,
                                                  variant=state.variant))
    monkeypatch.setattr(os, "environ", {
        k: v for k, v in os.environ.items() if k != "MLX_ENABLE_TF32"})
    rng = np.random.default_rng(0)
    state.refs = []
    for name, (w, h), mode in (("a.png", (300, 260), "RGB"),
                               ("b.png", (200, 320), "RGBA")):
        px = rng.integers(0, 256, (h, w, len(mode))).astype(np.uint8)
        Image.fromarray(px, mode).save(tmp_path / name)
        state.refs.append(str(tmp_path / name))
    mm = tmp_path / "mmproj.gguf"
    mm.write_bytes(b"x")
    state.mmproj = str(mm)
    return state


def test_references_reach_the_encoder_in_order_and_the_vae_after_it(
        edit, tmp_path, capsys):
    args = edit.base + [
        "--prompt", "swap them", "--size", "32x32", "--seed", "3",
        "--guidance", "3", "--negative-prompt", "blurry", "--ref",
        edit.refs[0], "--ref", edit.refs[1], "--ref-size", "320",
        "--encoder-mmproj", edit.mmproj, "--encoder-release", "always",
        "--output", str(tmp_path / "o.png"), "--json"]
    assert umbrella_main(args) == 0
    rec = json.loads(capsys.readouterr().out)
    assert edit.loads == [edit.mmproj]
    # The tower splits its own products: the process keeps its TF32 state.
    assert "MLX_ENABLE_TF32" not in os.environ
    kinds = [c[0] for c in edit.calls]
    assert kinds == ["encode", "encode", "vae", "vae"]
    for call, prompt in zip(edit.calls[:2], ("swap them", "blurry")):
        _, got, refs, with_latents, released = call
        assert got == prompt and not with_latents and not released
        assert [(r.width, r.height) for r in refs] == [(352, 288),
                                                      (256, 416)]
        assert bool((refs[0].pixels[..., 3] == 255).all())
        assert len(set(refs[1].pixels[..., 3].ravel().tolist())) > 2
    for call in edit.calls[2:]:
        assert call[1] == edit.calls[0][2] and call[2] is edit.plan
        assert call[3] is True
    # The fit check before the load counts the mmproj and the references.
    assert edit.checks[0]["mmproj"] == edit.mmproj
    assert edit.checks[0]["refs"] == [(288, 352), (416, 256)]
    assert [c.refs for c in edit.planned[0]["conds"]] == [
        edit.calls[0][2]] * 2
    assert rec["model"]["mmproj"] == "mmproj.gguf"
    # The record names each reference file by its own sha256, as it names
    # an adapter file.
    import hashlib

    files = [hashlib.sha256(open(p, "rb").read()).hexdigest()
             for p in edit.refs]
    assert files[0] != edit.calls[0][2][0].sha256
    assert rec["settings"]["refs"] == [
        {"file": "a.png", "size": "352x288", "sha256": files[0]},
        {"file": "b.png", "size": "256x416", "sha256": files[1]}]
    from PIL import Image

    with Image.open(tmp_path / "o.png") as im:
        assert json.loads(im.text["gmlx"])["refs"] == rec["settings"]["refs"]
        assert json.loads(im.text["gmlx"])["files"]["mmproj"] == "mmproj.gguf"


def test_references_default_to_the_family_size_and_keep_the_user_env(
        edit, tmp_path):
    os.environ["MLX_ENABLE_TF32"] = "1"
    assert umbrella_main(edit.base + [
        "--prompt", "p", "--size", "32x32", "--ref", edit.refs[0],
        "--encoder-mmproj", edit.mmproj, "--output",
        str(tmp_path / "o.png")]) == 0
    (ref,) = edit.calls[0][2]
    assert (ref.width, ref.height) == (1088, 960)
    assert os.environ["MLX_ENABLE_TF32"] == "1"
    # One conditioning, so one encode of each kind.
    assert [c[0] for c in edit.calls] == ["encode", "vae"]


def test_a_text_request_never_loads_the_mmproj(edit, tmp_path, capsys):
    assert umbrella_main(edit.base + [
        "--prompt", "p", "--size", "32x32", "--encoder-mmproj", edit.mmproj,
        "--output", str(tmp_path / "o.png"), "--json"]) == 0
    out = capsys.readouterr()
    assert edit.loads == [None]
    assert "--encoder-mmproj is unused without --ref" in out.err
    rec = json.loads(out.out)
    assert "mmproj" not in rec["model"] and "refs" not in rec["settings"]
    assert "MLX_ENABLE_TF32" not in os.environ
    assert edit.calls == [("encode", "p", (), False, False)]
    assert "mmproj" not in edit.checks[0] or edit.checks[0]["mmproj"] is None
    assert edit.checks[0]["refs"] == []


def test_reference_refusals_and_usage_errors(edit, env, tmp_path, capsys,
                                             monkeypatch):
    from PIL import Image

    base = edit.base + ["--prompt", "p", "--size", "32x32"]
    ref = ["--ref", edit.refs[0]]
    mm = ["--encoder-mmproj", edit.mmproj]
    long = tmp_path / "long.png"
    Image.fromarray(np.zeros((8, 4000, 3), np.uint8)).save(long)
    wide = tmp_path / "wide.png"
    Image.fromarray(np.zeros((480, 640, 3), np.uint8)).save(wide)
    junk = tmp_path / "junk.png"
    junk.write_bytes(b"not an image")
    for flags, code, text in (
            (ref, 2, "--ref needs --encoder-mmproj with a DiT path"),
            (mm + ["--ref-size", "512"], 2, "--ref-size needs --ref"),
            (ref * 11 + mm, 2, "--ref: Z-Image takes at most 10 reference "
                               "images, and the request has 11"),
            (ref + mm + ["--ref-size", "128"], 2,
             "--ref-size 128: Z-Image takes a reference size from 256 to "
             "1024"),
            (ref + mm + ["--ref-size", "2048"], 2, "from 256 to 1024"),
            (["--ref", "missing.png"] + mm, 1,
             "--ref missing.png: no such file"),
            (ref + ["--encoder-mmproj", "missing.gguf"], 1,
             "--encoder-mmproj missing.gguf: no such file"),
            (["--ref", str(long)] + mm, 2, "is too long for its area"),
            (["--ref", str(wide), "--ref-size", "256"] + mm, 2,
             f"--ref {wide}: a reference of 288x224 pixels is under the "
             "65536 pixels the encoder takes; use a reference area of 264 x "
             "264 pixels or more for this aspect ratio (--ref-size 264)"),
            (["--ref", str(junk)] + mm, 1,
             f"--ref {junk}: cannot read it as an image")):
        assert umbrella_main(base + flags) == code, flags
        assert text in capsys.readouterr().err
    # A family without reference images refuses the flag by name.
    plain = get_family("z_image")
    view = image_preflight.identify_dit("x").view
    monkeypatch.setattr(image_preflight, "identify_dit",
                        lambda p, **k: SimpleNamespace(family=plain, view=view,
                                                  variant="turbo"))
    assert umbrella_main(base + ref + mm) == 2
    assert "--ref: Z-Image takes no reference images" in \
        capsys.readouterr().err
    # All refused before any model loads.
    assert edit.pipes == [] and edit.loads == []


def test_the_reference_flags_complete_as_files():
    from gmlx.commands.completion import _complete

    flags = [c.split("\t")[0] for c in _complete(["image", "generate", "-"])]
    for flag in ("--ref", "--encoder-mmproj"):
        assert flag in flags
        assert _complete(["image", "generate", "x.gguf", flag, ""]) == \
            ["::files"]
    assert "--ref-size" in flags
    assert "--ref" not in [c.split("\t")[0]
                           for c in _complete(["image", "bench", "-"])]


def test_the_preflight_counts_the_mmproj_and_the_references(monkeypatch):
    from gmlx.image import memfit
    from gmlx.image import text_encoder as te
    from gmlx.serve import capacity

    pf = image_preflight
    fam = get_family("qwen_image21")
    monkeypatch.setattr(pf, "identify_dit", lambda p, **k: SimpleNamespace(
        family=fam, variant="base"))
    sizes = {"dit": 12 << 30, "encoder": 5 << 30, "vae": 400 << 20}
    monkeypatch.setattr(api, "_header_bytes",
                        lambda path, kind: 0 if path is None else sizes[kind])
    monkeypatch.setattr(te, "_header_view", lambda path: 2 << 30)
    seen = []
    monkeypatch.setattr(pf, "dit_transient", lambda dit, act: 0)
    monkeypatch.setattr(pf, "load_transient",
                        lambda *parts: seen.append(parts) or 1 << 30)
    monkeypatch.delenv("GMLX_OVERCOMMIT", raising=False)
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: 200 << 30)
    kw = dict(height=512, width=512, cfg=1.0)
    text = pf.check("d", "e", "v", **kw)
    assert text.terms["mmproj"] == 0 and "init" not in text.phases
    assert seen == [("e", "v", None), ("e", None, None), (None, "v", None)]
    refs = [(1024, 1024), (512, 768)]
    plan = pf.check("d", "e", "v", mmproj="m", refs=refs, **kw)
    assert plan.terms["mmproj"] == 2 << 30
    row = memfit.PEAKS["qwen_image21"]
    tokens = 64 * 64 + 32 * 48
    assert plan.terms["prefix_cache"] - text.terms["prefix_cache"] == \
        row.prefix.cache_bytes(tokens, 2)
    assert plan.terms["init_peak"] == memfit.vae_peak(
        "encode", 1024, 1024, family="qwen_image21", itemsize=2)
    assert list(plan.phases) == ["load", "encode", "init", "denoise",
                                 "decode"]
    assert seen[-3:] == [("e", "v", "m"), ("e", None, "m"),
                         (None, "v", None)]
    assert plan.phases["load"] == (sum(sizes.values()) + (2 << 30)
                                   + (1 << 30))
    # The encode runs the image rows of the references beside the text,
    # and the tower.
    assert plan.terms["encode_peak"] - text.terms["encode_peak"] == int(
        tokens // 4 * row.encode_row_bytes * memfit.ENCODE_LIVE
        + row.encode_tower_bytes)
    # Guidance holds the references in both conditionings.
    both = pf.check("d", "e", "v", mmproj="m", refs=refs,
                    **{**kw, "cfg": 4.0})
    assert both.terms["prefix_cache"] == 2 * plan.terms["prefix_cache"]
    # A budget the guided request passes: the refusal names the most of
    # its references and the largest reference area that fit.
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: 22 << 30)
    assert pf.check("d", "e", "v", mmproj="m", **{**kw, "cfg": 4.0}).fits
    with pytest.raises(memfit.ImageMemoryError) as e:
        pf.check("d", "e", "v", mmproj="m", refs=refs, **{**kw, "cfg": 4.0})
    assert re.search(r"; the largest that fit are \d of the 2 reference "
                     r"images at this reference area, or all 2 at a "
                     r"reference area of \d+ x \d+ px\. ", str(e.value))
