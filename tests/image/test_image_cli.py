"""``gmlx image generate`` and ``bench`` over a toy pipeline: generate's
latents equal those of ``sample`` with the seed's noise, the output naming,
the PNG text chunk, img2img and inpainting, encoder release, the bench
record, and the refusals with their exit codes. The loader, the VAE and the
fit check are replaced. CPU."""
from __future__ import annotations

import json
import os
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
        release_encoder=False, tiled_decode=False, levers=(),
        describe=lambda: "peak 9.00 GiB against a budget of 8.00 GiB"),
        variant="turbo", checks=[], checked=None)
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


def test_the_preflight_plans_from_headers_and_refuses_what_cannot_fit(
        monkeypatch):
    from gmlx.image import memfit
    from gmlx.serve import capacity

    pf = image_preflight
    fam = get_family("z_image")
    monkeypatch.setattr(pf, "identify_dit", lambda p: SimpleNamespace(
        family=fam, variant="turbo"))
    sizes = {"dit": 5 << 30, "encoder": 2 << 30, "vae": 160 << 20}
    monkeypatch.setattr(api, "_header_bytes",
                        lambda path, kind: 0 if path is None else sizes[kind])
    monkeypatch.delenv("GMLX_OVERCOMMIT", raising=False)
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: 100 << 30)
    plan = pf.check("d", "e", "v", height=1024, width=1024, cfg=None)
    assert plan.fits and plan.terms["dit"] == sizes["dit"]
    assert plan.terms["encoder"] == sizes["encoder"]
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: 6 << 30)
    with pytest.raises(memfit.ImageMemoryError):
        pf.check("d", "e", "v", height=1024, width=1024, cfg=1.0)
    monkeypatch.setenv("GMLX_TOOL_PREFLIGHT", "0")
    assert pf.check("d", "e", "v", height=1024, width=1024, cfg=1.0) is None
    monkeypatch.delenv("GMLX_TOOL_PREFLIGHT")
    monkeypatch.setattr(pf, "identify_dit", lambda p: SimpleNamespace(
        family=fam, variant="unknown"))
    assert pf.check("d", "e", "v", height=1024, width=1024, cfg=None) is None


class _TinyEncoder(_OwnedEncoder):
    """Features of the tiny Z-Image DiT's caption width."""

    def encode(self, prompt):
        self.encoded.append(prompt)
        ids = np.array(self.token_ids(prompt), np.float32)[:, None]
        h = (ids % 7 - 3) * np.linspace(0.1, 1.0, 16, dtype=np.float32)
        return SimpleNamespace(hidden=(mx.array(h)[None],))


def _tiny_pipe(log=print):
    from image_fixtures import tiny_z_model

    m, _ = tiny_z_model()
    pipe = api.ImagePipeline(dit=m, family=get_family("z_image"),
                             variant="turbo", encoder=_TinyEncoder(),
                             vae=object(), load_seconds={"dit": 0.0}, log=log)
    pipe.decoded = []
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
    env.a = write_adapter(tmp_path / "a.safetensors", "kohya",
                          tiny_z_adapter(1, 4.0, wide=True))
    env.b = write_adapter(tmp_path / "b.safetensors", "diffusers",
                          tiny_z_adapter(2))
    return env


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
    adapters = [{"file": "a.safetensors", "strength": 0.5, "sha256": shas[0]},
                {"file": "b.safetensors", "strength": 1.0, "sha256": shas[1]}]
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
    for path, text in ((flux, "flux.safetensors is not a Z-Image adapter"),
                       (extra, "1 of 9 targets name no Z-Image module"),
                       (loha, "loha.safetensors is a LoHa adapter"),
                       (dora, "dora.safetensors is a DoRA adapter")):
        assert umbrella_main(tiny.base + run + ["--adapter", path]) == 1
        assert text in capsys.readouterr().err
    assert not list(tmp_path.glob("gmlx-*.png"))


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
        {"file": "a.safetensors", "strength": 0.8, "sha256": shas[0]}]
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
