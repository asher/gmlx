"""``gmlx video generate``: the clip length, size, shift and seed
parsers, the settings from the flags, the entry and the variant, the
config entry, the refusals that come before the text encoder loads, and
the decode and mux of a run with every model replaced. CPU."""
from __future__ import annotations

import argparse
import dataclasses
import json
import re

import numpy as np
import pytest
from video_fixtures import H3_HEADER, fixture_view, gguf_tokenizer, wait_for

from gmlx.commands import video as cmd
from gmlx.video import fingerprint as vfp
from gmlx.video.families.h3 import family as hf


@pytest.fixture(autouse=True)
def _room(monkeypatch):
    """An M3 with room for every clip here, and the stub tokenizer in
    place of the encoder file's. Returns the family and path each
    tokenizer was built for. The memory and chip tests set their own."""
    from gmlx.load import dtypes
    from gmlx.serve import capacity
    from gmlx.video import encoder

    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: 100 << 30)
    monkeypatch.setattr(dtypes, "_arch_gen", 15)
    asked: list = []
    monkeypatch.setattr(encoder, "header_tokenizer", lambda fam, path: (
        asked.append((fam.name, path)) or gguf_tokenizer()))
    return asked


class _Stop(BaseException):
    """Raised by the replaced encoder load: the run got that far. Not an
    Exception, so the command's error handler lets it through."""


@pytest.mark.parametrize("seconds,frames", [
    (5, 124), (10, 243), (15, 362), (0.25, 22), (15.08, 362),
    (0.208333, 5), (15.0833, 362), (15.08334, 362), (5.16667, 124)])
def test_seconds_round_up_to_a_clip(seconds, frames):
    assert cmd.frames_of(cmd._seconds(str(seconds))) == frames


@pytest.mark.parametrize("text", ["0", "-1", "nan", "inf", "x"])
def test_a_shift_is_a_positive_number(text):
    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        cmd._shift(text)
    assert cmd._shift("6") == 6.0


@pytest.mark.parametrize("text", ["0.1", "16", "five", "nan", "1e308",
                                  "-1e308"])
def test_seconds_outside_a_clip_are_refused(text):
    with pytest.raises(argparse.ArgumentTypeError,
                       match="--seconds takes 0.208333 to 15.0833"):
        cmd._seconds(text)


def test_a_frame_count_is_a_clip_length():
    assert cmd._frames("124") == 124
    with pytest.raises(argparse.ArgumentTypeError) as ei:
        cmd._frames("125")
    assert str(ei.value) == ("125 frames is not a clip length: a clip has "
                             "17 k + 5 frames from 5 to 362, such as 124 "
                             "or 141")
    with pytest.raises(argparse.ArgumentTypeError) as ei:
        cmd._frames("400")
    assert str(ei.value).endswith("from 5 to 362")


def test_a_size_is_in_multiples_of_32():
    assert cmd._size("1344X768") == (1344, 768)
    with pytest.raises(argparse.ArgumentTypeError, match="multiples of 32"):
        cmd._size("1000x768")
    with pytest.raises(argparse.ArgumentTypeError, match="expected WxH"):
        cmd._size("768")


def _args(*argv, defaults=None):
    a = cmd._generate_parser("gmlx video generate").parse_args(
        ["h3.gguf", "--seed", "7", *argv])
    a.entry_defaults = dict(defaults or {})
    return a


@pytest.mark.parametrize("variant,steps,shift", [
    ("fl2va", 20, 12.0), (hf.TURBO4, 4, 6.0), ("unknown", 20, 12.0)])
def test_the_variant_sets_steps_and_shift(variant, steps, shift):
    s = cmd._settings(_args(), hf.FAMILY, variant)
    assert (s["steps"], s["shifts"]) == (steps, {"video": shift,
                                                 "audio": 3.0})
    assert (s["frames"], s["width"], s["height"], s["seed"]) == (
        124, 1344, 768, 7)


def test_flags_win_over_the_entry_and_the_entry_over_the_variant():
    entry = {"steps": 8, "shift": 6.0, "size": "768x768", "seconds": 10.0}
    s = cmd._settings(_args(defaults=entry), hf.FAMILY, "fl2va")
    assert (s["steps"], s["shifts"]["video"], s["frames"], s["width"]) == (
        8, 6.0, 243, 768)
    s = cmd._settings(_args("--steps", "4", "--shift", "5", "--frames", "22",
                            "--size", "512x512", defaults=entry),
                      hf.FAMILY, "fl2va")
    assert (s["steps"], s["shifts"], s["frames"], s["width"]) == (
        4, {"video": 5.0, "audio": 3.0}, 22, 512)


@pytest.mark.parametrize("seconds", [20.0, 1e308])
def test_an_entry_clip_too_long_is_a_usage_error(seconds):
    with pytest.raises(cmd._Usage) as ei:
        cmd._settings(_args(defaults={"seconds": seconds}), hf.FAMILY,
                      "fl2va")
    assert str(ei.value) == (f"the entry's defaults.seconds is {seconds:g}; "
                             "a clip takes 0.208333 to 15.0833 seconds")


@pytest.mark.parametrize("flags,defaults,match", [
    (["--shift", "1e-300"], {}, "--shift 1e-300 gives no 20-step schedule"),
    ([], {"shift": 1e39}, "the entry's defaults.shift 1e+39 gives no"),
    (["--steps", "1048577"], {},
     "the default video shift 12 gives no 1048577-step schedule"),
])
def test_a_shift_with_no_schedule_is_a_usage_error(flags, defaults, match):
    with pytest.raises(cmd._Usage, match=re.escape(match)):
        cmd._settings(_args(*flags, defaults=defaults), hf.FAMILY, "fl2va")


@pytest.fixture
def files(tmp_path):
    out = {}
    for flag, name in (("--encoder", "enc.gguf"), ("--vae", "v.safetensors"),
                       ("--audio-vae", "a.safetensors")):
        (tmp_path / name).write_bytes(b"x")
        out[flag] = str(tmp_path / name)
    (tmp_path / "h3.gguf").write_bytes(b"x")
    out["model"] = str(tmp_path / "h3.gguf")
    return out


@pytest.fixture
def run(monkeypatch, files, tmp_path):
    """``cmd_generate`` on a DiT that identifies as the header fixture
    (as ``variant``), with the encoder load replaced by a stop."""
    from gmlx.video import encoder, loader

    loads: list = []

    def load_encoder(*a, **k):
        loads.append(a)
        raise _Stop

    monkeypatch.setattr(encoder, "load_encoder", load_encoder)

    def go(*argv, variant=None):
        def identify(path, **kw):
            # The flag or the entry's variant reaches the weights check.
            vid = vfp.identify(fixture_view(H3_HEADER), **kw)
            return (vid if variant is None
                    else dataclasses.replace(vid, variant=variant))
        monkeypatch.setattr(loader, "identify_dit", identify)
        flags = [x for f in ("--encoder", "--vae", "--audio-vae")
                 for x in (f, files[f])]
        try:
            return cmd.cmd_generate([files["model"], "--prompt", "a cat",
                                     "--frames-dir", str(tmp_path / "out"),
                                     *flags, *argv])
        except _Stop:
            return "encoder"
    go.loads = loads
    return go


def test_a_bad_adapter_is_refused_before_the_encoder_loads(run, tmp_path,
                                                           capsys):
    bad = tmp_path / "lora.safetensors"
    bad.write_bytes(b"not a safetensors file")
    assert run("--adapter", str(bad)) == 1
    assert run.loads == []
    assert "lora.safetensors" in capsys.readouterr().err


@pytest.mark.parametrize("gen, chip", [(13, "M1"), (14, "M2"), (15, None)])
def test_an_m1_or_m2_is_refused_before_the_encoder_loads(run, monkeypatch,
                                                         capsys, gen, chip):
    from gmlx.load import dtypes

    monkeypatch.setattr(dtypes, "_arch_gen", gen)
    if chip is None:
        assert run() == "encoder"
        return
    assert run() == 1
    assert run.loads == []
    assert capsys.readouterr().err.splitlines()[-1] == (
        f"error: MiniMax-H3 needs an M3 or newer, and this Mac has an "
        f"{chip} chip")


def test_an_adapter_on_a_merged_file_warns(run, monkeypatch, tmp_path,
                                           capsys):
    from gmlx.video import pipeline

    monkeypatch.setattr(pipeline, "vet_adapter", lambda *a, **k: None)
    lora = tmp_path / "lora.safetensors"
    lora.write_bytes(b"x")
    assert run("--adapter", str(lora), variant=hf.TURBO4) == "encoder"
    err = capsys.readouterr().err
    assert ("[video] warning: this MiniMax-H3 GGUF has lightx2v's Turbo "
            "4-step v1.0 768p LoRA merged in, and each adapter runs on top "
            "of it") in err
    assert "4 steps at shift 6, seed" in err


def test_a_merged_file_alone_runs_its_own_recipe(run, capsys):
    assert run(variant=hf.TURBO4) == "encoder"
    err = capsys.readouterr().err
    assert "warning" not in err and "4 steps at shift 6, seed" in err


def test_an_adapter_on_a_base_file_runs_without_a_warning(run, monkeypatch,
                                                         tmp_path, capsys):
    from gmlx.video import pipeline

    monkeypatch.setattr(pipeline, "vet_adapter", lambda *a, **k: None)
    lora = tmp_path / "lora.safetensors"
    lora.write_bytes(b"x")
    assert run("--adapter", str(lora)) == "encoder"
    err = capsys.readouterr().err
    assert "warning" not in err and "20 steps at shift 12, seed" in err


def test_the_variant_flag_reaches_the_weights_check(run, capsys):
    assert run("--variant", hf.TURBO4) == "encoder"
    assert "4 steps at shift 6, seed" in capsys.readouterr().err
    assert run("--variant", "fl2va-turbo9") == 1
    assert "is not a MiniMax-H3 variant" in capsys.readouterr().err
    assert len(run.loads) == 1


@pytest.mark.parametrize("flags,match", [
    (["--output", "{tmp}"], "it is a directory"),
    (["--output", "{tmp}/file/clip.mp4"], "cannot write"),
    (["--save-latent", "{tmp}/file/x.npz"], "cannot write"),
    (["--frames-dir", "{tmp}/file"], "it is a file"),
    (["--frames-dir", "{tmp}/file/sub"], "cannot write"),
    (["--preview", "{tmp}/missing.safetensors"], "--preview"),
])
def test_an_output_the_run_cannot_write_is_refused_first(run, monkeypatch,
                                                         tmp_path, capsys,
                                                         flags, match):
    from gmlx.video import io

    monkeypatch.setattr(io, "probe", lambda *a: "ffmpeg")
    (tmp_path / "file").write_bytes(b"x")
    assert run(*[f.format(tmp=tmp_path) for f in flags]) == 1
    assert match in capsys.readouterr().err
    assert run.loads == []


def test_no_ffmpeg_stops_before_any_file_is_read(monkeypatch, files,
                                                 capsys):
    from gmlx.video import loader

    def never(*a, **k):
        raise AssertionError("identify ran")

    monkeypatch.setattr(loader, "identify_dit", never)
    monkeypatch.setenv("PATH", "")
    flags = [x for f in ("--encoder", "--vae", "--audio-vae")
             for x in (f, files[f])]
    assert cmd.cmd_generate([files["model"], "--prompt", "a cat",
                             *flags]) == 1
    assert "pass --frames-dir" in capsys.readouterr().err


def test_a_missing_prompt_is_a_usage_error(files, capsys):
    assert cmd.cmd_generate([files["model"]]) == 2
    assert "--prompt or --prompt-file is required" in capsys.readouterr().err


def test_a_config_entry_brings_its_files_and_defaults(tmp_path):
    import os

    for name in ("h3.gguf", "enc.gguf", "v.safetensors", "a.safetensors",
                 "lora.safetensors"):
        (tmp_path / name).write_bytes(b"x")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"""server:
  model_dirs: [{tmp_path}]
video_models:
  clip:
    path: h3.gguf
    encoder: enc.gguf
    vae: v.safetensors
    audio_vae: a.safetensors
    adapters: [lora.safetensors]
    defaults: {{steps: 8, shift: 6}}
""")
    a = cmd._generate_parser("g").parse_args(["clip", "--config", str(cfg)])
    cmd._from_config(a)
    assert a.model == os.path.join(tmp_path, "h3.gguf")
    assert a.audio_vae == os.path.join(tmp_path, "a.safetensors")
    assert a.entry_adapters == [(os.path.join(tmp_path, "lora.safetensors"),
                                 1.0)]
    assert a.entry_defaults == {"steps": 8, "shift": 6.0}
    a = cmd._generate_parser("g").parse_args(["other", "--config", str(cfg)])
    with pytest.raises(FileNotFoundError, match="it has clip"):
        cmd._from_config(a)


def test_an_entry_variant_reaches_the_weights_check(monkeypatch, tmp_path,
                                                    capsys):
    from gmlx.video import encoder, loader

    for name in ("h3.gguf", "enc.gguf", "v.safetensors", "a.safetensors"):
        (tmp_path / name).write_bytes(b"x")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"""server:
  model_dirs: [{tmp_path}]
video_models:
  clip:
    path: h3.gguf
    encoder: enc.gguf
    vae: v.safetensors
    audio_vae: a.safetensors
    variant: {hf.TURBO4}
""")
    seen = []

    def identify(path, **kw):
        seen.append(kw.get("variant"))
        return vfp.identify(fixture_view(H3_HEADER), **kw)

    def stop(*a, **k):
        raise _Stop

    monkeypatch.setattr(loader, "identify_dit", identify)
    monkeypatch.setattr(encoder, "load_encoder", stop)
    with pytest.raises(_Stop):
        cmd.cmd_generate(["clip", "--config", str(cfg), "--prompt", "a cat",
                          "--frames-dir", str(tmp_path / "out")])
    assert seen == [hf.TURBO4]
    assert "4 steps at shift 6, seed" in capsys.readouterr().err


class _Handle:
    def encode(self, prompt):
        return type("E", (), {"features": np.zeros((3, 8), np.float32)})()

    def release(self):
        pass


class _VAE:
    def __init__(self, fail_after=None, wait=None, samples=800):
        self.fail_after, self.wait = fail_after, wait
        self.samples = samples

    def decode(self, xa):
        import mlx.core as mx

        return mx.zeros((1, 2, self.samples))

    def decode_frames(self, xv):
        import mlx.core as mx

        for i in range(2):
            if self.fail_after == i:
                # Once ffmpeg has opened the clip, so a missed abort shows.
                assert self.wait is None or wait_for(self.wait)
                raise RuntimeError("the decode failed")
            yield mx.full((1, 3 - i, 32, 32, 3), 0.5)


@pytest.fixture
def stubbed(monkeypatch, files, tmp_path):
    """``cmd_generate`` with every model replaced: an encoder handle, a
    pipeline whose run returns zeros, VAEs that decode 5 frames of 32x32
    in two chunks, and a stand-in ffmpeg that writes its input. Returns
    the arguments ``pipeline.load`` got."""
    import stat

    import mlx.core as mx

    from gmlx.video import encoder, io, loader, pipeline
    from gmlx.video.vae import reader

    ff = tmp_path / "ffmpeg"
    ff.write_text('#!/bin/sh\nfor last; do :; done\ncat > "$last"\n')
    ff.chmod(ff.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(io, "probe", lambda *a: str(ff))
    monkeypatch.setattr(loader, "identify_dit", lambda path, **kw: (
        vfp.identify(fixture_view(H3_HEADER), **kw)))
    monkeypatch.setattr(encoder, "load_encoder",
                        lambda *a, **k: _Handle())
    loads = []
    monkeypatch.setattr(pipeline, "load",
                        lambda path, **kw: loads.append(kw) or object())
    monkeypatch.setattr(pipeline, "adapter_table", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "generate", lambda *a, **k: (
        mx.zeros((1, 24, 2, 2, 2)), mx.zeros((1, 32, 2, 1))))
    state = {"vae": _VAE(), "audio": _VAE()}
    monkeypatch.setattr(reader, "load_vae", lambda path, family, dtype: (
        state["audio"] if family == "h3_audio" else state["vae"]))

    def go(*argv):
        return cmd.cmd_generate([
            files["model"], "--prompt", "a cat", "--size", "32x32",
            "--frames", "5", "--seed", "3",
            *[x for f in ("--encoder", "--vae", "--audio-vae")
              for x in (f, files[f])], *argv])
    go.loads, go.state = loads, state
    return go


def test_a_run_writes_the_mp4_and_the_frames(stubbed, tmp_path, capsys):
    out, frames = tmp_path / "clips" / "a.mp4", tmp_path / "frames"
    lat = tmp_path / "lat" / "x.npz"
    assert stubbed("--output", str(out), "--frames-dir", str(frames),
                   "--variant", "fl2va", "--json", "--save-latent",
                   str(lat)) == 0
    assert out.stat().st_size == 5 * 32 * 32 * 3
    assert sorted(np.load(lat).files) == ["audio", "seed", "steps", "video"]
    assert sorted(p.name for p in frames.iterdir()) == [
        "audio.wav", *(f"frame_{i:06d}.png" for i in range(1, 6))]
    assert stubbed.loads == [{"variant": "fl2va", "log": stubbed.loads[0][
        "log"]}]
    rec = json.loads(capsys.readouterr().out)
    assert (rec["output"], rec["variant"], rec["frames"]) == (
        str(out), "fl2va", 5)


@pytest.mark.parametrize("decoded", [7200, 6000])
def test_the_mp4_and_the_frames_get_audio_of_the_clips_length(
        stubbed, monkeypatch, tmp_path, decoded):
    """Audio decoded past the last frame or short of it: both WAVs hold 5
    frames at 24 fps, 6667 samples."""
    import wave

    from gmlx.video import io

    lengths = []
    write_wav = io.write_wav

    def spy(path, waveform, rate):
        lengths.append(waveform.shape)
        return write_wav(path, waveform, rate)

    monkeypatch.setattr(io, "write_wav", spy)
    stubbed.state["audio"] = _VAE(samples=decoded)
    frames = tmp_path / "frames"
    assert stubbed("--output", str(tmp_path / "a.mp4"), "--frames-dir",
                   str(frames)) == 0
    assert lengths == [(2, 6667), (2, 6667)]
    with wave.open(str(frames / "audio.wav"), "rb") as w:
        assert w.getnframes() == 6667


def test_a_decode_error_leaves_no_clip(stubbed, tmp_path, capsys):
    out = tmp_path / "a.mp4"
    stubbed.state["vae"] = _VAE(fail_after=1, wait=str(out) + ".part")
    assert stubbed("--output", str(out)) == 1
    assert "the decode failed" in capsys.readouterr().err
    assert list(tmp_path.glob("a.mp4*")) == []


def test_completion_offers_recipes_and_video_ids(tmp_path):
    from gmlx.commands.completion import _complete

    assert [c.split("\t")[0] for c in _complete(["video", "pull", ""])] == [
        "h3", "h3-fast", "h3-base"]
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("""video_models:
  clip: {path: h3.gguf, encoder: e.gguf, vae: v.safetensors,
         audio_vae: a.safetensors}
""")
    assert _complete(["video", "generate", "--config", str(cfg), ""]) == [
        "::files", "clip\th3.gguf"]


def test_a_clip_over_the_memory_budget_is_refused_first(run, monkeypatch,
                                                        capsys):
    from gmlx.serve import capacity

    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: 1 << 20)
    monkeypatch.setattr(capacity, "overcommit", lambda: False)
    assert run() == 1
    assert "this clip does not fit in memory" in capsys.readouterr().err
    assert run.loads == []


def _cache_limit() -> int:
    import mlx.core as mx

    prev = mx.set_cache_limit(0)
    mx.set_cache_limit(prev)
    return prev


@pytest.fixture
def planned(monkeypatch, tmp_path):
    """``cmd_generate`` with ``~`` at ``tmp_path``, the memory plan's
    arguments and the order of the write check, the memory check and the
    encoder load recorded, and the encoder load replaced by a stop."""
    from gmlx.video import encoder, loader, memfit, pipeline

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("GMLX_OVERCOMMIT", raising=False)
    seen: dict = {"order": [], "raise": None}
    plan, check, writable = memfit.plan_memory, memfit.check, cmd._writable

    def spy_plan(*a, **k):
        seen["plan"] = k
        return plan(*a, **k)

    def spy_check(p, **k):
        seen["order"].append("check")
        return check(p, **k)

    def spy_writable(a, out):
        seen["order"].append("writable")
        return writable(a, out)

    def load_encoder(fam, path, **k):
        seen["order"].append("encoder")
        seen["encoder"] = path
        seen["cache_limit"] = _cache_limit()
        if seen["raise"] is not None:
            raise seen["raise"]
        raise _Stop

    monkeypatch.setattr(memfit, "plan_memory", spy_plan)
    monkeypatch.setattr(memfit, "check", spy_check)
    monkeypatch.setattr(cmd, "_writable", spy_writable)
    monkeypatch.setattr(encoder, "load_encoder", load_encoder)
    monkeypatch.setattr(loader, "identify_dit", lambda path, **kw: (
        vfp.identify(fixture_view(H3_HEADER), **kw)))
    monkeypatch.setattr(pipeline, "vet_adapter", lambda *a, **k: None)
    sizes = {"h3.gguf": 11, "enc.gguf": 19, "v.safetensors": 5,
             "a.safetensors": 3, "tiny.safetensors": 1,
             "entry.safetensors": 2, "flag.safetensors": 7}
    for name, mib in sizes.items():
        with open(tmp_path / name, "wb") as f:
            f.truncate(mib << 20)
    files = ["~/h3.gguf", "--encoder", "~/enc.gguf", "--vae",
             "~/v.safetensors", "--audio-vae", "~/a.safetensors",
             "--frames-dir", str(tmp_path / "out")]

    def go(*argv, budget=100 << 30):
        from gmlx.serve import capacity

        monkeypatch.setattr(capacity, "working_budget_bytes", lambda: budget)
        seen["order"].clear()
        try:
            return cmd.cmd_generate(list(argv))
        except _Stop:
            return "encoder"
    go.seen, go.files = seen, files
    return go


def test_the_memory_plan_prices_the_files_the_run_loads(planned, tmp_path,
                                                        _room):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"""server:
  model_dirs: [{tmp_path}]
video_models:
  clip:
    path: h3.gguf
    encoder: enc.gguf
    vae: v.safetensors
    audio_vae: a.safetensors
    tiny_vae: tiny.safetensors
    adapters: [entry.safetensors]
""")
    # 22 tokens, 23 characters and 24 bytes.
    prompt = "He met Hera at the caf\u00e9"
    assert planned("clip", "--config", str(cfg), "--prompt", prompt,
                   "--frames-dir", str(tmp_path / "out"), "--preview",
                   "--adapter", "~/flag.safetensors:0.5", "--frames", "22",
                   "--size", "512x320") == "encoder"
    k = planned.seen["plan"]
    assert (k["dit"], k["encoder"], k["vae"], k["audio_vae"]) == (
        11 << 20, 19 << 20, 1 << 20, 3 << 20)
    assert k["tiny"] is True and k["loras"] == [2 << 20, 7 << 20]
    assert k["text"] == 22
    assert planned.seen["encoder"] == str(tmp_path / "enc.gguf")
    assert _room == [("h3", planned.seen["encoder"])]
    assert k["pixels"] == 512 * 320
    assert planned.seen["order"] == ["writable", "check", "encoder"]


def test_a_write_refusal_comes_before_the_memory_refusal(planned, tmp_path,
                                                         capsys):
    (tmp_path / "afile").write_text("x")
    assert planned(*planned.files, "--prompt", "a cat", "--frames-dir",
                   str(tmp_path / "afile"), budget=1 << 20) == 1
    assert "it is a file" in capsys.readouterr().err
    assert planned.seen["order"] == ["writable"]


def test_overcommit_or_no_budget_lets_the_run_through(planned, monkeypatch):
    run = [*planned.files, "--prompt", "a cat"]
    assert planned(*run, budget=None) == "encoder"
    monkeypatch.setenv("GMLX_OVERCOMMIT", "1")
    assert planned(*run, budget=1 << 20) == "encoder"


def test_the_cache_bound_holds_for_the_run_and_is_restored(planned):
    import mlx.core as mx

    from gmlx.image.memfit import CACHE_BOUND

    prior = 50 << 30
    first = mx.set_cache_limit(prior)
    run = [*planned.files, "--prompt", "a cat"]
    try:
        assert planned(*run, budget=1 << 20) == 1       # refused
        assert _cache_limit() == prior
        assert planned(*run) == "encoder"               # stopped
        assert planned.seen["cache_limit"] == CACHE_BOUND
        assert _cache_limit() == prior
        planned.seen["raise"] = RuntimeError("the load failed")
        assert planned(*run) == 1
        assert _cache_limit() == prior
        assert planned(*planned.files) == 2             # no prompt
        assert _cache_limit() == prior
    finally:
        mx.set_cache_limit(first)
