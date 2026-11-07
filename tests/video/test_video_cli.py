"""``gmlx video generate``: the clip length, size, shift and seed
parsers, the settings from the flags, the entry and the variant, the
config entry, the refusals that come before the text encoder loads, and
the decode and mux of a run with every model replaced. Keyframes and
guides: their flags, their refusals before any load, their memory plan,
and a run with them through the encoder, the VAEs and the DiT, in that
order, the guide files read by the system ffmpeg (skipped without it).
CPU."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shutil
import struct
import subprocess
import wave

import numpy as np
import pytest
from video_fixtures import (H3_HEADER, fixture_view, gguf_tokenizer,
                            vsa_kernel, wait_for)

from gmlx.commands import video as cmd
from gmlx.video import catalog
from gmlx.video import fingerprint as vfp
from gmlx.video.families.h3 import family as hf


@pytest.fixture(autouse=True)
def _room(monkeypatch):
    """An M3, with no NAX units, and room for every clip here, and the
    stub tokenizer in place of the encoder file's. Returns the family and
    path each tokenizer was built for. The memory, chip and NAX tests set
    their own."""
    from gmlx.load import dtypes
    from gmlx.serve import capacity
    from gmlx.video import encoder

    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: 100 << 30)
    monkeypatch.setattr(dtypes, "_arch_gen", 15)
    vsa_kernel(monkeypatch, nax=False)
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
    ("fl2va", 20, 12.0), (hf.TURBO4, 4, 6.0), ("unknown", 20, 12.0),
    (hf.FASTH3, 8, 10.0)])
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


def test_fasth3_runs_its_own_schedule(run, monkeypatch, capsys):
    vsa_kernel(monkeypatch)
    assert run(variant=hf.FASTH3) == "encoder"
    assert "8 steps at shift 10, seed" in capsys.readouterr().err


def test_fasth3_without_nax_is_refused_before_the_encoder_loads(
        run, monkeypatch, capsys):
    vsa_kernel(monkeypatch, nax=False)
    assert run(variant=hf.FASTH3) == 1
    assert run.loads == []
    assert capsys.readouterr().err.splitlines()[-1] == (
        "error: MiniMax-H3 fasth3 runs video sparse attention, whose "
        "kernel needs a GPU with NAX units (M5 or newer), and this Mac's "
        "GPU has none")


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
def stubbed(monkeypatch, files, tmp_path, stand_in_ffmpeg):
    """``cmd_generate`` with every model replaced: an encoder handle, a
    pipeline whose run returns zeros, VAEs that decode 5 frames of 32x32
    in two chunks, and a stand-in ffmpeg that writes its input. Returns
    the arguments ``pipeline.load`` got."""
    import mlx.core as mx

    from gmlx.video import encoder, io, loader, pipeline
    from gmlx.video.vae import reader

    ff = stand_in_ffmpeg('cat > "$last"')
    monkeypatch.setattr(io, "probe", lambda *a: ff)
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


def test_a_shorter_clip_leaves_no_frames_of_the_one_before(stubbed,
                                                           tmp_path):
    """A frames folder that holds a longer earlier clip holds this clip's
    5 frames and audio.wav after the run, and every other file it held:
    a frame name in digits other than ASCII is no name it writes, and a
    folder under a frame's name stays. A link under a frame's name goes,
    and the file it points to stays."""
    frames = tmp_path / "frames"
    frames.mkdir()
    other = "frame_\u0661\u0662\u0663\u0664\u0665\u0666.png"
    for name in [*(f"frame_{i:06d}.png" for i in range(1, 8)),
                 "frame_000009.png.part", "audio.wav", "notes.txt",
                 "frame_7.png", other]:
        (frames / name).write_bytes(b"earlier")
    (frames / "frame_000008.png").mkdir()
    outside = tmp_path / "keep.png"
    outside.write_bytes(b"kept")
    (frames / "frame_000009.png").symlink_to(outside)
    assert stubbed("--frames-dir", str(frames)) == 0
    assert sorted(p.name for p in frames.iterdir()) == sorted([
        "audio.wav", *(f"frame_{i:06d}.png" for i in range(1, 6)),
        "frame_000008.png", "frame_7.png", "notes.txt", other])
    assert (frames / "notes.txt").read_bytes() == b"earlier"
    assert (frames / "frame_000008.png").is_dir()
    assert outside.read_bytes() == b"kept"


def test_the_latents_land_at_the_path_given(stubbed, tmp_path):
    """A --save-latent path without .npz is written as given, not beside
    it with .npz added."""
    lat = tmp_path / "lat" / "run1"
    assert stubbed("--frames-dir", str(tmp_path / "f"), "--save-latent",
                   str(lat)) == 0
    assert sorted(np.load(lat).files) == ["audio", "seed", "steps", "video"]
    assert [p.name for p in lat.parent.iterdir()] == ["run1"]


def test_an_interrupted_latent_write_keeps_the_file_there(stubbed,
                                                          monkeypatch,
                                                          tmp_path):
    """A latent write stopped part way leaves the file an earlier run
    wrote, and no part file."""
    lat = tmp_path / "lat" / "x.npz"
    lat.parent.mkdir()
    np.savez(lat, video=np.ones(3))
    before = lat.read_bytes()
    savez = np.savez

    def stopped(file, **arrays):
        f = open(file, "wb") if isinstance(file, (str, os.PathLike)) \
            else file
        f.write(b"PK\x03\x04 part of a zip")
        f.flush()
        raise KeyboardInterrupt

    monkeypatch.setattr(np, "savez", stopped)
    assert stubbed("--frames-dir", str(tmp_path / "f"), "--save-latent",
                   str(lat)) == 130
    monkeypatch.setattr(np, "savez", savez)
    assert lat.read_bytes() == before
    assert [p.name for p in lat.parent.iterdir()] == ["x.npz"]


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
        "h3", "h3-fast", "h3-base", "fasth3"]
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


def _pinned(files, recipe, monkeypatch, gb):
    """The run's files at the sizes of ``recipe``'s pins, sparse, on a
    Mac of ``gb`` GB."""
    from gmlx.serve import capacity

    pins = catalog.RECIPES[recipe].pins()
    for key, role in (("model", "dit"), ("--encoder", "encoder"),
                      ("--vae", "vae"), ("--audio-vae", "audio_vae")):
        os.truncate(files[key], pins[role].size)
    monkeypatch.setattr(capacity, "working_budget_bytes",
                        lambda: capacity.ram_budget_bytes(gb << 30))
    monkeypatch.setattr(capacity, "overcommit", lambda: False)


def _hint(capsys) -> str:
    err = capsys.readouterr().err.splitlines()[-1]
    m = re.fullmatch(r"error: this clip does not fit in memory: .*? over "
                     r"it\. (.*)\. GMLX_OVERCOMMIT=1 overrides\.", err)
    assert m, err
    return m[1]


@pytest.mark.parametrize("recipe, variant, argv, size", [
    ("fasth3", hf.FASTH3, (), "768x768"),
    ("h3-base", None, ("--size", "1344x1344"), "1344x768"),
    ("h3-base", None, ("--size", "1088x1920"), "768x1344")])
def test_a_clip_too_large_for_the_mac_names_the_largest_size_inside_it(
        run, monkeypatch, files, capsys, recipe, variant, argv, size):
    """On a 32 GB Mac: FastH3's default clip, a square clip, and a
    portrait clip, which no 1344 wide size fits inside."""
    if variant == hf.FASTH3:
        vsa_kernel(monkeypatch)
    _pinned(files, recipe, monkeypatch, 32)
    assert run(*argv, variant=variant) == 1
    assert run.loads == []
    assert _hint(capsys) == f"Try --size {size}, which fits at this length"
    assert run("--size", size, variant=variant) == "encoder"


@pytest.mark.parametrize("argv", [("--seconds", "10"), ("--frames", "175")])
def test_a_length_no_size_fits_names_the_longest_768x768_clip(
        run, monkeypatch, files, capsys, argv):
    """FastH3 on a 32 GB Mac fits 768x768 up to 158 frames."""
    vsa_kernel(monkeypatch)
    _pinned(files, "fasth3", monkeypatch, 32)
    assert run(*argv, variant=hf.FASTH3) == 1
    assert run.loads == []
    assert _hint(capsys) == (
        "No 768x768 clip fits at this length; try --size 768x768 --frames "
        "158 (6.58 s), the longest 768x768 clip that fits")
    assert run("--size", "768x768", "--frames", "158",
               variant=hf.FASTH3) == "encoder"


def test_a_clip_under_768x768_names_its_own_longest_clip(run, monkeypatch,
                                                         files, capsys):
    """FastH3 at 640x640 on a 32 GB Mac fits up to 226 frames."""
    vsa_kernel(monkeypatch)
    _pinned(files, "fasth3", monkeypatch, 32)
    assert run("--size", "640x640", "--seconds", "15",
               variant=hf.FASTH3) == 1
    assert run.loads == []
    assert _hint(capsys) == (
        "Try --frames 226 (9.41 s), the longest 640x640 clip that fits")
    assert run("--size", "640x640", "--frames", "226",
               variant=hf.FASTH3) == "encoder"


def _bf16_lora(path, size: int) -> str:
    """A bf16 LoRA of ``size`` bytes: its header, then a sparse body."""
    head = json.dumps({"blocks.0.attn.q.lora_A.weight": {
        "dtype": "BF16", "shape": [16, 5120],
        "data_offsets": [0, 16 * 5120 * 2]}}).encode()
    path.write_bytes(struct.pack("<Q", len(head)) + head)
    os.truncate(path, size)
    return str(path)


def test_a_large_lora_names_the_107_frame_768x768_clip(
        run, monkeypatch, files, tmp_path, capsys):
    """FastH3 with a 2.2 GB bf16 LoRA on a 32 GB Mac: a 768x768 clip
    fits at 107 frames and not at 124."""
    from gmlx.video import pipeline

    monkeypatch.setattr(pipeline, "vet_adapter", lambda *a, **k: None)
    vsa_kernel(monkeypatch)
    _pinned(files, "fasth3", monkeypatch, 32)
    lora = _bf16_lora(tmp_path / "style.safetensors", 2_200_000_000)
    assert run("--adapter", str(lora), variant=hf.FASTH3) == 1
    assert run.loads == []
    assert _hint(capsys) == (
        "No 768x768 clip fits at this length; try --size 768x768 --frames "
        "107 (4.45 s), the longest 768x768 clip that fits")
    assert run("--adapter", str(lora), "--size", "768x768", "--frames",
               "107", variant=hf.FASTH3) == "encoder"


@pytest.mark.parametrize("recipe, variant, gb, base", [
    ("fasth3", hf.FASTH3, 2.0, ", or with h3-base"),
    ("h3-base", catalog.RECIPES["h3-base"].variant, 2.5, "")])
def test_stacked_loras_that_leave_no_clip_name_their_memory(
        run, monkeypatch, files, tmp_path, capsys, recipe, variant, gb,
        base):
    """Two bf16 LoRAs on a 32 GB Mac. Stacked, each costs twice its file,
    and the smallest clip is over the budget. h3-base is named unless
    the run is already its DiT with no entry LoRA."""
    from gmlx.video import pipeline

    monkeypatch.setattr(pipeline, "vet_adapter", lambda *a, **k: None)
    vsa_kernel(monkeypatch)
    _pinned(files, recipe, monkeypatch, 32)
    loras = [_bf16_lora(tmp_path / f"{name}.safetensors", int(gb * 1e9))
             for name in ("style", "subject")]
    argv = [x for p in loras for x in ("--adapter", p)]
    assert run(*argv, variant=variant) == 1
    assert run.loads == []
    lora = 2 * 2 * int(gb * 1e9) / (1 << 30)
    assert _hint(capsys) == (
        f"No clip fits at any size with these adapters, which take "
        f"{lora:.2f} GiB of the denoise; run with fewer adapters{base}")
    assert run("--adapter", loras[0], "--size", "768x768",
               variant=variant) == "encoder"


def test_stacked_loras_that_leave_a_clip_under_768x768_keep_the_hint(
        run, monkeypatch, files, tmp_path, capsys):
    """FastH3 with two 1.9 GB bf16 LoRAs on a 32 GB Mac: no 768x768 clip
    fits down to 5 frames, and a 512x512 one does."""
    from gmlx.video import pipeline

    monkeypatch.setattr(pipeline, "vet_adapter", lambda *a, **k: None)
    vsa_kernel(monkeypatch)
    _pinned(files, "fasth3", monkeypatch, 32)
    argv = [x for name in ("style", "subject") for x in (
        "--adapter", _bf16_lora(tmp_path / f"{name}.safetensors",
                                1_900_000_000))]
    assert run(*argv, variant=hf.FASTH3) == 1
    assert run.loads == []
    assert _hint(capsys) == "Try a smaller size or a shorter clip"
    assert run(*argv, "--size", "512x512", "--frames", "5",
               variant=hf.FASTH3) == "encoder"


@pytest.mark.parametrize("adapter", [False, True])
def test_a_dit_file_that_fits_no_clip_alone_names_its_memory(
        run, monkeypatch, files, tmp_path, capsys, adapter):
    """unsloth's fl2va Q8_0 DiT on a 32 GB Mac, alone or with the Turbo
    LoRA: the DiT alone leaves even the smallest clip over the budget."""
    from gmlx.video import pipeline

    monkeypatch.setattr(pipeline, "vet_adapter", lambda *a, **k: None)
    _pinned(files, "h3-base", monkeypatch, 32)
    os.truncate(files["model"], 21_437_786_208)
    argv = (["--adapter", _bf16_lora(tmp_path / "turbo.safetensors",
                                     catalog.RECIPES["h3"].lora.size)]
            if adapter else [])
    variant = catalog.RECIPES["h3-base"].variant
    assert run(*argv, variant=variant) == 1
    assert run.loads == []
    assert _hint(capsys) == (
        "No clip fits at any size with this DiT file, which takes 19.97 GiB "
        "of the denoise; run a smaller quant of it")
    assert run("--size", "32x32", "--frames", "5", variant=variant) == 1


def test_an_entry_lora_and_one_more_name_h3_base(monkeypatch, tmp_path,
                                                 capsys):
    """An h3 entry carries the Turbo LoRA, so one more adapter stacks on
    the base DiT and h3-base, with no LoRA, is new to the run."""
    from gmlx.serve import capacity
    from gmlx.video import encoder, loader, pipeline

    h3 = catalog.RECIPES["h3"]
    pins = h3.pins()
    for name, role in (("h3.gguf", "dit"), ("enc.gguf", "encoder"),
                       ("v.safetensors", "vae"), ("a.safetensors",
                                                  "audio_vae")):
        (tmp_path / name).write_bytes(b"")
        os.truncate(tmp_path / name, pins[role].size)
    _bf16_lora(tmp_path / "turbo.safetensors", pins["lora"].size)
    mine = _bf16_lora(tmp_path / "mine.safetensors", 3_000_000_000)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"""server:
  model_dirs: [{tmp_path}]
video_models:
  h3:
    path: h3.gguf
    encoder: enc.gguf
    vae: v.safetensors
    audio_vae: a.safetensors
    adapters: [turbo.safetensors]
""")
    monkeypatch.setattr(loader, "identify_dit", lambda path, **kw: (
        dataclasses.replace(vfp.identify(fixture_view(H3_HEADER), **kw),
                            variant=h3.variant)))
    monkeypatch.setattr(encoder, "load_encoder", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "vet_adapter", lambda *a, **k: None)
    monkeypatch.setattr(capacity, "working_budget_bytes",
                        lambda: capacity.ram_budget_bytes(32 << 30))
    monkeypatch.setattr(capacity, "overcommit", lambda: False)
    assert cmd.cmd_generate(["h3", "--config", str(cfg), "--prompt", "a cat",
                             "--frames-dir", str(tmp_path / "out"),
                             "--adapter", mine]) == 1
    assert _hint(capsys).endswith(
        "GiB of the denoise; run with fewer adapters, or with h3-base")


def test_the_hints_seconds_make_its_clip():
    """``--seconds`` of the seconds the size hint prints for a clip makes
    that clip, at every clip length."""
    hf = cmd._family()
    for f in range(hf.MIN_FRAMES, hf.MAX_FRAMES + 1, 17):
        assert cmd.frames_of(cmd._seconds(cmd._seconds_text(f))) == f


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
    assert k["variant"] == "unknown"
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


@pytest.mark.parametrize("text, ok", [
    ("18446744073709551615", True), ("0", True),
    ("18446744073709551616", False), ("-1", False), ("seven", False)])
def test_a_seed_is_a_64_bit_integer_as_comfyuis_is(text, ok):
    if ok:
        assert cmd._seed(text) == int(text)
        return
    with pytest.raises(argparse.ArgumentTypeError, match=re.escape(
            f"an integer from 0 to 18446744073709551615 is required, got "
            f"{text!r}")):
        cmd._seed(text)


@pytest.mark.parametrize("text, want", [
    ("clip.mp4@12", ("clip.mp4", 12)), ("clip.mp4@-17", ("clip.mp4", -17)),
    ("clip.mp4", ("clip.mp4", 0)), ("me@home.png", ("me@home.png", 0)),
    ("take@2/clip.mp4@3", ("take@2/clip.mp4", 3)),
    ("clip.mp4@+3", ("clip.mp4@+3", 0))])
def test_only_an_integer_after_the_last_at_is_a_frame(text, want):
    assert cmd._at(text) == want


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        pytest.skip("no ffmpeg on PATH")
    return exe


def _png(path, w, h, seed=0) -> str:
    from PIL import Image

    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(
        path)
    return str(path)


def _wav(path, seconds, channels=1, rate=32000) -> str:
    t = np.arange(int(rate * seconds)) / rate
    x = np.stack([0.4 * np.sin(2 * np.pi * (330 + 110 * c) * t)
                  for c in range(channels)], -1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((x * 32767).astype("<i2").tobytes())
    return str(path)


def _clip(path, frames, sound=True) -> str:
    """A 64x48 clip of ``frames`` frames at 24 fps, with ``frames`` / 24 s
    of 16-bit stereo sound when ``sound``, in a MOV so its length is
    exact."""
    ff = _ffmpeg()
    sec = f"{frames / 24:.6f}"
    args = ["-f", "lavfi", "-i", "testsrc=size=64x48:rate=24"]
    if sound:
        args += ["-f", "lavfi", "-i", "sine=frequency=330:sample_rate=32000",
                 "-ac", "2", "-c:a", "pcm_s16le"]
    subprocess.run([ff, "-v", "error", "-nostdin", "-y", *args, "-t", sec,
                    "-frames:v", str(frames), "-pix_fmt", "yuv420p",
                    str(path)], check=True)
    return str(path)


def test_fasth3_refuses_anchors_before_any_load(run, monkeypatch, tmp_path,
                                                capsys):
    vsa_kernel(monkeypatch)
    sound = _wav(tmp_path / "a.wav", 0.1)
    assert run("--guide-audio", sound, variant=hf.FASTH3) == 1
    assert run.loads == []
    assert capsys.readouterr().err.splitlines()[-1] == (
        "error: MiniMax-H3 fasth3 runs text to video only, and this request "
        "has 1 anchor; run it without keyframes, guides or audio anchors")


def test_a_keyframe_needs_the_mmproj_and_a_preview_the_vae(
        files, tmp_path, monkeypatch, capsys):
    from gmlx.video import loader

    monkeypatch.setattr(loader, "identify_dit", lambda path, **kw: (
        vfp.identify(fixture_view(H3_HEADER), **kw)))
    still = _png(tmp_path / "f.png", 64, 48)
    base = [files["model"], "--prompt", "a cat", "--frames-dir",
            str(tmp_path / "out"), "--encoder", files["--encoder"],
            "--audio-vae", files["--audio-vae"]]
    assert cmd.cmd_generate([*base, "--vae", files["--vae"],
                             "--first-frame", still]) == 2
    assert capsys.readouterr().err.splitlines()[-1].endswith(
        "error: --first-frame needs the encoder's vision tower: pass "
        "--mmproj, or name encoder_mmproj in the entry")
    assert cmd.cmd_generate([*base, "--preview", files["--vae"],
                             "--guide", still]) == 2
    assert capsys.readouterr().err.splitlines()[-1].endswith(
        "error: --vae is required with --preview and --guide, whose anchor "
        "encodes through the video VAE")


@pytest.mark.parametrize("argv, why", [
    (["--guide", "{clip}@20"], "--guide {clip}: a 22-frame guide clip at "
                               "frame 20 does not fit in the clip's 39 "
                               "frames"),
    (["--guide", "{long}"], "--guide {long}: a guide clip of 56 or more "
                            "frames at frame 0 does not fit in the clip's "
                            "39 frames"),
    (["--guide", "{clip}@39"], "--guide {clip}: frame 39 is outside the "
                               "clip's 39 frames"),
    (["--guide-audio", "{clip}@-39", "--guide-audio", "{still}"],
     "{still} has no audio stream")])
def test_a_guide_that_does_not_fit_is_refused_before_any_load(
        run, tmp_path, capsys, argv, why):
    """ComfyUI's guide bound on a 39-frame clip: a 30-frame guide keeps 22
    frames, a guide read to its bound holds more than fits, and a frame
    off the clip; and a file without sound as an audio guide."""
    paths = {"clip": _clip(tmp_path / "clip.mov", 30),
             "long": _clip(tmp_path / "long.mov", 80, sound=False),
             "still": _png(tmp_path / "s.png", 64, 48)}
    argv = [x.format(**paths) for x in argv]
    assert run("--frames", "39", *argv) == 1
    assert run.loads == []
    err = capsys.readouterr().err.splitlines()[-1]
    assert err.startswith("error: " + why.format(**paths)), err


class _Recorded(_VAE):
    """A VAE that records its loads and encodes: frames to one latent
    frame for a still and the clip grid's for a clip, a waveform to a
    latent per 800 samples, read with a 12 latent halo."""
    halo = 12

    def __init__(self, family, order):
        super().__init__()
        self.family, self.order, self.encoded = family, order, []
        self.waves: list = []

    def samples_for(self, latents):
        return (latents + self.halo) * 800

    def encode(self, x):
        import mlx.core as mx

        self.encoded.append(tuple(x.shape))
        if self.family == "h3_audio":
            self.waves.append(np.array(x))
            return mx.zeros((1, 32, 2, -(-x.shape[-1] // 800)))
        t = x.shape[1]
        return mx.zeros((1, 24, 1 if t == 1 else hf.latent_frames(t),
                         x.shape[2] // 16, x.shape[3] // 16))


@pytest.fixture
def anchored(stubbed, monkeypatch, tmp_path):
    """``stubbed`` with the loads, the releases and the encodes recorded
    in order, the keyframe encode replaced, and a 39-frame 32x32 clip."""
    import mlx.core as mx

    from gmlx.video import encoder, memfit, pipeline
    from gmlx.video.families.h3 import keyframes
    from gmlx.video.vae import reader

    order: list = []
    seen: dict = {"vae": {}}

    class Handle(_Handle):
        def release(self):
            order.append("release")

    def load_encoder(fam, path, **kw):
        order.append("encoder")
        seen["mmproj"] = kw.get("mmproj")
        return Handle()

    def keyframe_encode(handle, prompt, images):
        seen["keyframes"], seen["prompt"] = images, prompt
        return mx.zeros((1, 9, 8)), ((0, 4),)

    def load_vae(path, family, dtype):
        order.append(family)
        return seen["vae"].setdefault(family, _Recorded(family, order))

    def generate(pipe, feats, **kw):
        seen["generate"] = kw
        return mx.zeros((1, 24, 2, 2, 2)), mx.zeros((1, 32, 2, 1))

    monkeypatch.setattr(encoder, "load_encoder", load_encoder)
    monkeypatch.setattr(keyframes, "encode", keyframe_encode)
    monkeypatch.setattr(reader, "load_vae", load_vae)
    monkeypatch.setattr(pipeline, "load", lambda path, **kw: (
        order.append("dit") or object()))
    monkeypatch.setattr(pipeline, "generate", generate)
    monkeypatch.setattr(memfit, "tower_bytes", lambda p: 1 << 20)
    mm = tmp_path / "mm.gguf"
    mm.write_bytes(b"x")

    def go(*argv):
        return stubbed("--frames", "39", "--frames-dir",
                       str(tmp_path / "out"), "--mmproj", str(mm), *argv)
    go.order, go.seen = order, seen
    return go


@pytest.mark.parametrize("quiet", [False, True], ids=["sound", "no-sound"])
def test_a_run_encodes_its_anchors_between_the_encoder_and_the_dit(
        anchored, tmp_path, capsys, quiet):
    """Both keyframes, a guide clip with its sound and an audio guide at a
    frame from the end: the encoder loads with the mmproj and sees the
    keyframes resized as ComfyUI resizes them, the video VAE then the
    audio VAE encode the anchors after the encoder's release and before
    the DiT loads, the audio guide is read past its kept latents' halo,
    and the DiT gets the anchors in the flags' order with the spans and
    the seed."""
    from gmlx.video import images

    first = _png(tmp_path / "first.png", 64, 48, seed=1)
    last = _png(tmp_path / "last.png", 96, 48, seed=2)
    clip = _clip(tmp_path / "clip.mov", 30)
    sound = _wav(tmp_path / "long.wav", 3.0)
    argv = ["--first-frame", first, "--last-frame", last, "--guide",
            f"{clip}@4", "--guide-audio", f"{sound}@-35", "--json"]
    assert anchored(*argv, *(["--guide-no-audio"] if quiet else [])) == 0
    assert anchored.order == ["encoder", "release", "h3", "h3_audio", "dit",
                              "h3_audio", "h3"]
    seen = anchored.seen
    assert seen["mmproj"] == str(tmp_path / "mm.gguf")
    np.testing.assert_array_equal(seen["keyframes"][0], images.stretch(
        images.read(first), 32, 32))
    np.testing.assert_array_equal(seen["keyframes"][1], images.cover(
        images.read(last), 32, 32))
    video = seen["vae"]["h3"].encoded
    assert video == [(1, 1, 32, 32, 3), (1, 1, 32, 32, 3),
                     (1, 22, 32, 32, 3)]
    # The clip's 1.25 s of sound is short of its kept latents; the audio
    # guide at frame 4 keeps 58 latents and is read to their halo.
    sounds = seen["vae"]["h3_audio"].encoded
    assert sounds == ([] if quiet else [(1, 2, 40000)]) + [
        (1, 2, (58 + 12) * 800)]
    gen = seen["generate"]
    assert gen["seed"] == 3 and gen["spans"] == ((0, 4),)
    got = [(x.frame, None if x.video is None else x.video.shape[2],
            None if x.audio is None else x.audio.shape[-1])
           for x in gen["anchors"]]
    assert got == [(0, 1, None), (38, 1, None),
                   (4, 7, None if quiet else 50), (4, None, 58)]
    rec = json.loads(capsys.readouterr().out)
    assert [(r["flag"], r["frame"], r["frames"], r["audio_latents"])
            for r in rec["anchors"]] == [
        ("--first-frame", 0, 1, 0), ("--last-frame", 38, 1, 0),
        ("--guide", 4, 22, 0 if quiet else 50), ("--guide-audio", 4, 0, 58)]



@pytest.mark.parametrize("guided", [True, False], ids=["guide", "text"])
def test_a_run_without_keyframes_loads_no_mmproj(anchored, tmp_path, guided):
    """The vision tower reads keyframes alone: a run with an image guide
    and a run from text alone, each given --mmproj, load the encoder
    without it."""
    argv = ["--guide", f"{_png(tmp_path / 'g.png', 64, 48)}@10"] \
        if guided else []
    assert anchored(*argv) == 0
    assert anchored.order[0] == "encoder"
    assert anchored.seen["mmproj"] is None
    assert "keyframes" not in anchored.seen


def test_a_guide_sound_is_cut_as_comfyuis_vae_encode_cuts_it(anchored,
                                                             tmp_path):
    """3000 samples of guide sound give 3 latents, from samples 300 to
    2699, as ComfyUI's VAE encode crops a waveform to whole latents about
    its middle; the CLI once started at 0 and padded a fourth latent."""
    src = tmp_path / "ramp.wav"
    ramp = (np.arange(3000) * 7 - 10000).astype("<i2")
    with wave.open(str(src), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(32000)
        w.writeframes(ramp.tobytes())
    _ffmpeg()
    assert anchored("--guide-audio", f"{src}@4") == 0
    rec = anchored.seen["vae"]["h3_audio"]
    want = np.repeat(ramp[None, 300:2700] / 32768.0, 2, 0)[None]
    assert rec.encoded == [(1, 2, 2400)]
    np.testing.assert_array_equal(rec.waves[0], want.astype(np.float32))
    assert [x.audio.shape[-1] for x in anchored.seen["generate"]["anchors"]
            ] == [3]

def test_the_memory_plan_prices_the_anchors_the_run_encodes(
        planned, tmp_path, monkeypatch):
    """A first frame at 512x320 adds its vision rows to the prompt and its
    block to the encode, through the mmproj; a guide image is one more
    frame to encode, and a half-second audio guide its 20 latents, not
    the 207 the clip leaves it."""
    from gmlx.video import memfit

    monkeypatch.setattr(memfit, "tower_bytes", lambda p: 5 << 20)
    with open(tmp_path / "mm.gguf", "wb") as f:
        f.truncate(3 << 20)
    still = _png(tmp_path / "f.png", 64, 48)
    _ffmpeg()
    sound = _wav(tmp_path / "a.wav", 0.5)
    assert planned(*planned.files, "--prompt", "a cat", "--size", "512x320",
                   "--mmproj", "~/mm.gguf", "--first-frame", still,
                   "--guide", f"{still}@60", "--guide-audio",
                   sound) == "encoder"
    k = planned.seen["plan"]
    tok = hf.vision_tokens(512, 320)
    assert k["blocks"] == (tok,)
    assert (k["mmproj"], k["tower"]) == (3 << 20, 5 << 20)
    assert k["text"] == memfit.prompt_rows(hf.FAMILY, "a cat",
                                           gguf_tokenizer(), [(512, 320)])
    assert k["text"] >= memfit.prompt_rows(hf.FAMILY, "a cat",
                                           gguf_tokenizer()) + tok + 8
    assert k["anchor_frames"] == [1, 1] and k["anchor_audio"] == [20]
    assert planned.seen["order"] == ["writable", "check", "encoder"]


def test_a_run_with_anchors_hints_no_shorter_clip(run, monkeypatch, files,
                                                  tmp_path, capsys):
    """A shorter clip moves the last frame and can drop a guide, so the
    hint of a run with anchors names no shorter clip: with a 7 GB LoRA on
    a 32 GB Mac a 768x768 clip fits at 73 frames and not at 124. A LoRA
    that leaves no clip at all is named with anchors as without."""
    from gmlx.video import pipeline

    monkeypatch.setattr(pipeline, "vet_adapter", lambda *a, **k: None)
    _pinned(files, "h3", monkeypatch, 32)
    still = _png(tmp_path / "g.png", 64, 48)
    lora = _bf16_lora(tmp_path / "style.safetensors", 7_000_000_000)
    assert run("--adapter", lora) == 1
    assert _hint(capsys) == (
        "No 768x768 clip fits at this length; try --size 768x768 --frames "
        "73 (3.04 s), the longest 768x768 clip that fits")
    assert run("--adapter", lora, "--guide", f"{still}@60") == 1
    assert _hint(capsys) == "Try a smaller size or a shorter clip"
    lora = _bf16_lora(tmp_path / "style.safetensors", 20_000_000_000)
    assert run("--adapter", lora, "--guide", f"{still}@60") == 1
    assert _hint(capsys) == (
        "No clip fits at any size with these adapters, which take 18.63 GiB "
        "of the denoise; run with fewer adapters, or with h3-base")
    assert run.loads == []


def test_an_upsampler_rewrites_the_prompt_before_the_encoder_reads_it(
        stubbed, monkeypatch, tmp_path, capsys):
    """--upsampler loads the chat model before the encoder, under the
    guide's system prompt for the clip's task and length, and the encoder
    reads its caption, priced again at its rows; the record keeps both
    prompts."""
    from gmlx.commands import image as image_cmd
    from gmlx.video import encoder, memfit

    guide = tmp_path / "guide.md"
    guide.write_text("the guide")
    seen = []
    monkeypatch.setattr(image_cmd, "_upsampler_path", lambda a: "/up.gguf")

    def run_upsampler(path, fam, prompt, log, quiet=False, system=None,
                      tag="image"):
        seen.append(("upsample", path, prompt, system, tag))
        return "the caption"
    monkeypatch.setattr(image_cmd, "run_upsampler", run_upsampler)
    rows = memfit.prompt_rows

    def priced(fam, text, *a, **k):
        if not seen or seen[-1] != ("price", text):
            seen.append(("price", text))
        return rows(fam, text, *a, **k)
    monkeypatch.setattr(memfit, "prompt_rows", priced)

    class Handle(_Handle):
        def encode(self, prompt):
            seen.append(("encode", prompt))
            return super().encode(prompt)
    monkeypatch.setattr(encoder, "load_encoder", lambda *a, **k: Handle())
    assert stubbed("--upsampler", "chat", "--upsample-prompt", str(guide),
                   "--frames-dir", str(tmp_path / "f"), "--json") == 0
    first, (_, path, prompt, system, tag), *rest = seen
    assert first == ("price", "a cat")
    assert (path, prompt, tag) == ("/up.gguf", "a cat", "video")
    assert rest == [("price", "the caption"), ("encode", "the caption")]
    assert "a T2VA video of 0.21 seconds" in system
    assert system.endswith("The guide:\n\nthe guide\n")
    rec = json.loads(capsys.readouterr().out)
    assert (rec["prompt"], rec["revised_prompt"]) == ("a cat", "the caption")


def test_no_upsample_or_no_guide_keeps_the_typed_prompt_or_says_why(
        stubbed, monkeypatch, tmp_path, capsys):
    from gmlx.commands import image as image_cmd

    monkeypatch.setattr(image_cmd, "run_upsampler", lambda *a, **k: (
        pytest.fail("the upsampler ran")))
    out = str(tmp_path / "f")
    assert stubbed("--no-upsample", "--frames-dir", out) == 0
    assert stubbed("--upsampler", "chat", "--frames-dir", out) == 2
    assert "gmlx video pull h3` fetches it" in capsys.readouterr().err


def test_a_keyframe_run_asks_the_upsampler_for_the_guides_keyframe_task(
        anchored, monkeypatch, tmp_path):
    """A first and a last frame make the clip the guide's FL2VA task, of
    the clip's length; the upsampler sees no picture, and its caption
    joins the keyframes in the encode."""
    from gmlx.commands import image as image_cmd

    guide = tmp_path / "guide.md"
    guide.write_text("the guide")
    systems = []
    monkeypatch.setattr(image_cmd, "_upsampler_path", lambda a: "/up.gguf")
    monkeypatch.setattr(image_cmd, "run_upsampler", lambda *a, system=None,
                        **k: (systems.append(system), "the caption")[1])
    first = _png(tmp_path / "first.png", 64, 48, seed=1)
    last = _png(tmp_path / "last.png", 64, 48, seed=2)
    assert anchored("--first-frame", first, "--last-frame", last,
                    "--upsampler", "chat", "--upsample-prompt", str(guide),
                    "--json") == 0
    assert "an FL2VA video of 1.62 seconds, where its first and last" in (
        systems[0])
    assert "You see no picture" in systems[0]
    assert anchored.seen["prompt"] == "the caption"
