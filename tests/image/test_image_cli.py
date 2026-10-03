"""``gmlx image generate`` and ``bench`` over a toy pipeline: generate's
latents equal those of ``sample`` with the seed's noise, the output naming,
the PNG text chunk, img2img and inpainting, encoder release, the bench
record, and the refusals with their exit codes. The loader, the VAE and the
fit check are replaced. CPU."""
from __future__ import annotations

import json
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

PIL = pytest.importorskip("PIL")


class _OwnedEncoder(_Encoder):
    owned = True

    @property
    def released(self):
        return self.model is None

    def release(self):
        self.model = None
        return True


def _decode(pipe, x, tiled=None):
    pipe.decoded.append(tiled)
    b, _, h, w = x.shape
    v = mx.sigmoid(x.mean(axis=1))[..., None]
    img = mx.broadcast_to(v, (b, h, w, 3))
    return mx.repeat(mx.repeat(img, 8, axis=1), 8, axis=2)


def _encode(pipe, image, tiled=None, mode="mode", noise=None):
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
        release_encoder=False, tiled_decode=False, levers=()),
        variant="turbo", checks=[])
    fam = get_family("z_image")

    def load(dit, *, text_encoder=None, vae=None, log=print, **kw):
        pipe = api.ImagePipeline(dit=_Dit(), family=fam,
                                 variant=state.variant,
                                 encoder=_OwnedEncoder(), vae=object(),
                                 load_seconds={"dit": 0.0}, log=log)
        pipe.decoded = []
        state.pipes.append(pipe)
        return pipe

    monkeypatch.setattr(api, "load", load)
    monkeypatch.setattr(api, "vae_decode", _decode)
    monkeypatch.setattr(api, "vae_encode", _encode)
    monkeypatch.setattr(api, "plan_request", lambda *a, **k: state.plan)
    view = SimpleNamespace(tensors={
        "a": SimpleNamespace(shape=(4, 4), codec="Q4_K"),
        "b": SimpleNamespace(shape=(4, 4), codec="Q4_K"),
        "c": SimpleNamespace(shape=(4, 4), codec="BF16"),
        "d": SimpleNamespace(shape=(4,), codec="F32")})
    monkeypatch.setattr(image_preflight, "identify_dit",
                        lambda p: SimpleNamespace(family=fam, view=view,
                                                  variant=state.variant))
    monkeypatch.setattr(image_preflight, "check",
                        lambda *a, **k: state.checks.append(k))
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
    assert umbrella_main(env.base + ["--prompt", "p", "--init-image",
                                     str(tmp_path / "in.png"), "--mask",
                                     str(tmp_path / "m.png"), "--output",
                                     str(tmp_path / "b.png")]) == 0
    k = seen[-1]
    assert k["denoise"] == 1.0
    assert k["mask"].shape == (32, 64) and k["mask"][:, 40:].min() == 1.0
    assert k["mask"][:, :24].max() == 0.0


def test_encoder_release_follows_the_policy_and_the_plan(env, tmp_path):
    def run(*flags):
        assert umbrella_main(env.base + ["--prompt", "p", "--size", "32x32",
                                         "--output", str(tmp_path / "a.png"),
                                         *flags]) == 0
        return env.pipes[-1]

    assert not run().encoder.released
    assert run("--encoder-release", "always").encoder.released
    env.plan.release_encoder = True
    env.plan.tiled_decode = True
    pipe = run()
    assert pipe.encoder.released and pipe.decoded == [True]
    assert not run("--encoder-release", "never").encoder.released


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


@pytest.mark.parametrize("flags, code, text", [
    ([], 2, "--prompt or --prompt-file is required"),
    (["--prompt", "p", "--size", "40x32"], 2, "multiples of 16"),
    (["--prompt", "p", "--size", "big"], 2, "a size such as"),
    (["--prompt", "p", "--mask", "m.png"], 2, "--mask needs --init-image"),
    (["--prompt", "p", "--strength", "0.5"], 2, "needs --init-image"),
    (["--prompt", "p", "--strength", "1.5"], 2, "at most 1"),
    (["--prompt", "p", "--quality", "0"], 2, "from 1 to 100"),
    (["--prompt", "p", "--sampler", "dpm"], 2, "invalid choice"),
    (["--prompt", "p", "--init-image", "missing.png"], 1, "no such file"),
    (["--prompt", "p", "--count", "2", "--output", "a{{b}}.png"], 2,
     "one file for 2 images"),
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


def test_bench_refuses_sizes_off_the_stride(env, capsys):
    assert umbrella_main(["image", "bench"] + env.base[2:] + [
        "--sizes", "1024x1000"]) == 2
    assert "multiples of 16" in capsys.readouterr().err
    assert env.pipes == []
