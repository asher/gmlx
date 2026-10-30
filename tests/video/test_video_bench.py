"""``gmlx video bench`` with every model replaced by stubs that hold
known amounts of memory: its records, the peak of each phase, the memory
check, the flags and the exit codes. CPU, no weights."""
from __future__ import annotations

import argparse
import gc
import json
import os
import re

import numpy as np
import pytest
from video_fixtures import H3_HEADER, fixture_view, gguf_tokenizer

from gmlx.commands import video as cmd
from gmlx.commands import video_bench as vb
from gmlx.video import fingerprint as vfp
from gmlx.video import memfit
from gmlx.video.families.h3 import family as hf

MiB = 1 << 20
GiB = 1 << 30
TEXT = 25
# What each stub holds, and what its load or run adds for a moment, MiB.
ENC = 64
DIT, STEP = 128, 32
AVAE, AVAE_LOAD, AVAE_RUN = 8, 256, 16
VAE, VAE_LOAD, VAE_RUN = 8, 256, 24


def _alloc(mib):
    import mlx.core as mx

    a = mx.zeros((mib * MiB // 4,), mx.float32)
    mx.eval(a)
    return a


def _cache_limit() -> int:
    import mlx.core as mx

    prev = mx.set_cache_limit(0)
    mx.set_cache_limit(prev)
    return prev


@pytest.fixture(autouse=True)
def _room(monkeypatch):
    """An M3 with room for every shape here, and the stub tokenizer in
    place of the encoder file's. Returns the family and path each
    tokenizer was built for."""
    from gmlx.load import dtypes
    from gmlx.serve import capacity
    from gmlx.video import encoder

    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: 100 << 30)
    monkeypatch.setattr(dtypes, "_arch_gen", 15)
    asked: list = []
    monkeypatch.setattr(encoder, "header_tokenizer", lambda fam, path: (
        asked.append((fam.name, path)) or gguf_tokenizer()))
    return asked


class _Handle:
    def __init__(self):
        self.w = _alloc(ENC)

    def encode(self, prompt):
        return type("E", (), {"features": np.zeros((TEXT, 8), np.float32)})()

    def release(self):
        self.w = None


class _Pipe:
    def __init__(self, load):
        if load:
            _alloc(load)
        self.w = _alloc(DIT)


class _VAE:
    def __init__(self, res, load, run, fail=None):
        _alloc(load)
        self.w, self.run, self.fail, self.n = _alloc(res), run, fail, 0

    def decode(self, xa):
        import mlx.core as mx

        return mx.zeros((1, 2, 800)) + _alloc(self.run)[:1].sum()

    def decode_frames(self, xv):
        import mlx.core as mx

        self.n += 1
        if self.fail is not None and self.fail[0] == self.n - 1:
            raise self.fail[1]
        for i in range(2):
            yield mx.full((1, 3 - i, 32, 32, 3), 0.5) + _alloc(
                self.run)[:1].sum()


@pytest.fixture
def bench(monkeypatch, tmp_path):
    """``gmlx video bench`` on stub files, recording each load and run."""
    import mlx.core as mx

    from gmlx.video import encoder, loader, pipeline
    from gmlx.video.vae import reader

    # A failed run's exception keeps its stubs in a cycle, and MLX would
    # count them in this test's peaks.
    gc.collect()
    ev: list = []
    cfg = {"dit_load": 0, "fail_at": None, "fail": None, "variant": None,
           "decode_fail": None}
    files = {}
    for flag, name in (("--encoder", "enc.gguf"), ("--vae", "v.safetensors"),
                       ("--audio-vae", "a.safetensors")):
        (tmp_path / name).write_bytes(b"x")
        files[flag] = str(tmp_path / name)
    (tmp_path / "h3.gguf").write_bytes(b"x")
    files["model"] = str(tmp_path / "h3.gguf")

    def identify(path, **kw):
        ev.append(("identify", kw.get("variant")))
        vid = vfp.identify(fixture_view(H3_HEADER), **kw)
        if cfg["variant"] is not None:
            import dataclasses

            vid = dataclasses.replace(vid, variant=cfg["variant"])
        return vid

    def load_encoder(fam, path, **k):
        ev.append(("load_encoder", path))
        return _Handle()

    def load(path, **kw):
        ev.append(("load", kw.get("variant")))
        return _Pipe(cfg["dit_load"])

    def generate(pipe, feats, *, frames, width, height, steps, shifts, seed,
                 table, on_step):
        n = len([e for e in ev if e[0] == "generate"])
        ev.append(("generate", (width, height, frames), steps, dict(shifts),
                   seed, _cache_limit()))
        if cfg["fail_at"] == n:
            raise cfg["fail"]
        vshape, ashape = hf.latent_shapes(frames, width, height)
        for i in range(steps):
            _alloc(STEP)
            on_step(i, steps)
        return mx.zeros(vshape), mx.zeros(ashape)

    def load_vae(path, family, dtype):
        ev.append(("load_vae", family))
        return (_VAE(AVAE, AVAE_LOAD, AVAE_RUN) if family == "h3_audio"
                else _VAE(VAE, VAE_LOAD, VAE_RUN, cfg["decode_fail"]))

    monkeypatch.setattr(loader, "identify_dit", identify)
    monkeypatch.setattr(encoder, "load_encoder", load_encoder)
    monkeypatch.setattr(pipeline, "load", load)
    monkeypatch.setattr(pipeline, "adapter_table", lambda pipe, ads: (
        ev.append(("adapter_table", list(ads))) or None))
    monkeypatch.setattr(pipeline, "vet_adapter", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "generate", generate)
    monkeypatch.setattr(reader, "load_vae", load_vae)

    def go(*argv, model=None, flags=True):
        f = [x for k in ("--encoder", "--vae", "--audio-vae")
             for x in (k, files[k])] if flags else []
        return cmd.cmd_video(["bench", model or files["model"], *f, *argv])
    go.ev, go.cfg, go.files = ev, cfg, files
    return go


def _kinds(ev, kind):
    return [e for e in ev if e[0] == kind]


def _records(text):
    return [json.loads(x) for x in text.splitlines()]


def test_shapes_and_shift():
    assert vb._shapes("512x512x22, 1344x768x124") == [
        (512, 512, 22), (1344, 768, 124)]
    with pytest.raises(argparse.ArgumentTypeError, match="not a clip"):
        vb._shape("512x512x23")
    a = vb._parser("b").parse_args(["h3.gguf"])
    a.entry_defaults = {"shift": 5.0}
    assert vb._shifts(a, hf.FAMILY, "fl2va") == {"video": 5.0, "audio": 3.0}
    a.shift = 7.0
    assert vb._shifts(a, hf.FAMILY, hf.TURBO4)["video"] == 7.0
    a.entry_defaults, a.shift = {}, None
    assert vb._shifts(a, hf.FAMILY, hf.TURBO4)["video"] == 6.0


def test_one_record_per_shape_from_one_load_of_each_model(bench, tmp_path,
                                                          capsys, _room):
    out = tmp_path / "b.jsonl"
    assert bench("--shapes", "64x64x5,96x64x22", "--steps", "3", "--json",
                 "--json-out", str(out), "--label", "L", "--seed", "4") == 0
    printed = capsys.readouterr().out
    assert out.read_text() == printed
    recs = _records(printed)
    assert len(recs) == 2
    for r, (w, h, f) in zip(recs, [(64, 64, 5), (96, 64, 22)]):
        assert r["schema"] == "gmlx.video.bench/1" and r["label"] == "L"
        s = r["settings"]
        assert (s["width"], s["height"], s["frames"], s["steps"],
                s["seed"]) == (w, h, f, 3, 4)
        assert len(r["step_s"]) == 3
        assert set(r["peak"]) == set(r["predicted"]) == set(memfit.PHASES)
        rows, _ = memfit.clip_rows(hf.FAMILY, frames=f, width=w, height=h,
                                   text=TEXT)
        assert r["rows"]["total"] == rows and r["rows"]["text"] == TEXT
        assert r["summary"]["s_per_step"]["first"] == r["step_s"][0]
        assert r["summary"]["s_per_step"]["median"] == float(
            np.median(r["step_s"][1:]))
        assert r["model"]["dit"] == {"file": "h3.gguf", "bytes": 1}
        assert r["predicted"] == vb._plan(hf.FAMILY, r["model"], (w, h, f),
                                          TEXT, True).phases
        assert "error" not in r
    ev = bench.ev
    assert len(_kinds(ev, "load_encoder")) == len(_kinds(ev, "load")) == 1
    # The prompt is priced by the tokenizer of the encoder the run loads.
    assert _room == [("h3", bench.files["--encoder"])]
    assert _kinds(ev, "load_encoder") == [("load_encoder",
                                           bench.files["--encoder"])]
    assert [e[1] for e in _kinds(ev, "load_vae")] == ["h3_audio", "h3"]
    assert len(_kinds(ev, "generate")) == 2


def test_each_peak_holds_its_model_and_its_load(bench, capsys):
    from gmlx.image.memfit import CACHE_BOUND

    bench.cfg["dit_load"] = 512
    assert bench("--shapes", "64x64x5,64x64x5", "--steps", "2",
                 "--json") == 0
    out, err = capsys.readouterr()
    logged = [float(x) for x in re.findall(r"denoise peak ([\d.]+) GiB",
                                           err)]
    assert logged == [round(r["peak"]["denoise"] / GiB, 2)
                      for r in _records(out)]
    for r in _records(out):
        p = {k: v / MiB for k, v in r["peak"].items()}
        # Each load's moment is larger than its model and its run.
        assert p["encode"] >= ENC
        assert p["denoise"] >= 512
        assert p["audio"] >= AVAE_LOAD
        assert p["decode"] >= VAE_LOAD
    assert all(g[5] == CACHE_BOUND for g in _kinds(bench.ev, "generate"))


def test_a_step_peak_holds_the_dit_not_the_encoder(bench, capsys):
    _alloc(512)                     # an earlier peak, which each phase resets
    assert bench("--shapes", "64x64x5", "--steps", "2", "--json",
                 "--no-decode") == 0
    peak = _records(capsys.readouterr().out)[0]["peak"]
    assert ENC <= peak["encode"] / MiB < 2 * ENC
    assert DIT + STEP <= peak["denoise"] / MiB < DIT + STEP + ENC


def test_no_decode_reads_no_vae(bench, monkeypatch, tmp_path, capsys):
    from gmlx.serve import capacity

    # Under the audio decode's plan, over the encode's and the denoise's.
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: 1 << 30)
    monkeypatch.setenv("GMLX_OVERCOMMIT", "0")
    assert bench("--shapes", "64x64x5", "--steps", "2", "--json",
                 "--no-decode", "--encoder", bench.files["--encoder"],
                 "--vae", str(tmp_path / "missing.safetensors"),
                 flags=False) == 0
    r = _records(capsys.readouterr().out)[0]
    assert set(r["peak"]) == set(r["predicted"]) == {"encode", "denoise"}
    assert "vae" not in r["model"] and "audio_vae" not in r["model"]
    assert _kinds(bench.ev, "load_vae") == []


def test_a_split_dit_counts_every_shard(bench, tmp_path, capsys):
    first = tmp_path / "h3-00001-of-00002.gguf"
    first.write_bytes(b"x" * 3)
    (tmp_path / "h3-00002-of-00002.gguf").write_bytes(b"x" * 5)
    assert bench("--shapes", "64x64x5", "--steps", "1", "--json",
                 "--no-decode", model=str(first)) == 0
    dit = _records(capsys.readouterr().out)[0]["model"]["dit"]
    assert dit == {"file": "h3-00001-of-00002.gguf", "bytes": 8}


def test_json_out_goes_into_a_new_folder(bench, tmp_path):
    out = tmp_path / "new" / "sub" / "b.jsonl"
    assert bench("--shapes", "64x64x5", "--steps", "1", "--json-out",
                 str(out)) == 0
    assert len(_records(out.read_text())) == 1


def test_a_json_out_folder_is_refused_before_any_load(bench, tmp_path,
                                                      capsys):
    assert bench("--json-out", str(tmp_path)) == 1
    assert "it is a directory" in capsys.readouterr().err
    assert bench.ev == []


def test_an_m1_is_refused_before_any_load(bench, monkeypatch, capsys):
    from gmlx.load import dtypes

    monkeypatch.setattr(dtypes, "_arch_gen", 13)
    assert bench("--steps", "1") == 1
    assert capsys.readouterr().err.splitlines()[-1] == (
        "error: MiniMax-H3 needs an M3 or newer, and this Mac has an M1 chip")
    assert _kinds(bench.ev, "identify") and len(bench.ev) == 1


@pytest.mark.parametrize("shapes", ["1344x768x124,64x64x5",
                                    "64x64x5,1344x768x124",
                                    "64x64x5,1344x768x124,96x64x22"])
def test_a_shape_over_the_budget_is_refused_before_any_load(
        bench, monkeypatch, capsys, shapes):
    """A budget the small shapes fit and 1344x768x124 does not, wherever
    it is in the list."""
    from gmlx.serve import capacity

    files = {k: {"bytes": 1} for k in ("dit", "encoder", "vae",
                                       "audio_vae")}
    files["adapters"] = []
    text = memfit.prompt_rows(hf.FAMILY, vb.BENCH_PROMPT, gguf_tokenizer())
    small = max(vb._plan(hf.FAMILY, files, s, text, False).peak
                for s in ((64, 64, 5), (96, 64, 22)))
    big = vb._plan(hf.FAMILY, files, (1344, 768, 124), text, False).peak
    monkeypatch.setattr(capacity, "working_budget_bytes",
                        lambda: (small + big) / 2)
    monkeypatch.setenv("GMLX_OVERCOMMIT", "0")
    argv = ["--shapes", shapes, "--steps", "1", "--no-decode"]
    assert bench(*argv) == 1
    assert ("error: the 1344x768x124 shape does not fit in memory: encode "
            in capsys.readouterr().err)
    assert [e[0] for e in bench.ev] == ["identify"]
    monkeypatch.setenv("GMLX_OVERCOMMIT", "1")
    assert bench(*argv) == 0
    assert len(_kinds(bench.ev, "generate")) == len(shapes.split(","))


# The sizes of the measured files and the 8-step LoRA.
SIZES = {"model": 11420663904, "--encoder": 19762151200,
         "--vae": 5207808496, "--audio-vae": 605254808}
LORA = 1956193000
ENCODER_HINT = "The text encoder needs more memory than this Mac has"
SIZE_HINT = "Try a smaller size or a shorter clip"


@pytest.mark.parametrize("ram,lora,shapes,named,over,hint", [
    # No LoRA: the encode is the peak at every default shape, and
    # 768x768x124 and 1344x768x124 have the denoise over too.
    (24, False, None, "the 1344x768x124 shape",
     "the encode and the denoise", ENCODER_HINT),
    (24, False, "512x512x22", "this bench", "the encode", ENCODER_HINT),
    # The shape with the larger peak has the smaller decode.
    (32, True, "1024x768x124,1344x768x56", "the 1024x768x124 shape",
     "the denoise", SIZE_HINT),
    (32, True, "1344x768x56,1024x768x124", "the 1024x768x124 shape",
     "the denoise", SIZE_HINT),
    (32, True, "1024x768x107,768x768x124", "the 1024x768x107 shape",
     "the denoise", SIZE_HINT),
])
def test_the_largest_phases_name_the_refused_shape(
        bench, monkeypatch, tmp_path, capsys, ram, lora, shapes, named,
        over, hint):
    """The measured file sizes, as sparse files."""
    from gmlx.serve import capacity

    for k, n in SIZES.items():
        os.truncate(bench.files[k], n)
    argv = ["--shapes", shapes] if shapes else []
    if lora:
        path = tmp_path / "lora.safetensors"
        with open(path, "wb") as f:
            f.truncate(LORA)
        argv += ["--adapter", str(path)]
    monkeypatch.delenv("GMLX_GOV_MARGIN", raising=False)
    monkeypatch.delenv("GMLX_GOV_RESERVE_GB", raising=False)
    budget = capacity.ram_budget_bytes(ram * GiB)
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: budget)
    monkeypatch.setenv("GMLX_OVERCOMMIT", "0")
    assert bench(*argv) == 1
    err = capsys.readouterr().err
    assert f"error: {named} does not fit in memory: encode " in err
    assert f", with {over} over it. {hint}" in err
    assert _kinds(bench.ev, "load_encoder") == []


def test_a_long_prompt_can_stop_the_bench_alone(bench, monkeypatch,
                                                capsys):
    """The check prices the prompt's rows: a budget between this
    prompt's denoise and its encode refuses on the encode, which is the
    same at every shape."""
    from gmlx.serve import capacity

    prompt = "x" * 30_000
    files = {k: {"bytes": 1} for k in ("dit", "encoder", "vae",
                                       "audio_vae")}
    files["adapters"] = []
    plan = vb._plan(hf.FAMILY, files, (64, 64, 5),
                    memfit.prompt_rows(hf.FAMILY, prompt, gguf_tokenizer()),
                    False)
    budget = (plan.phases["denoise"] + plan.phases["encode"]) / 2
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: budget)
    monkeypatch.setenv("GMLX_OVERCOMMIT", "0")
    assert bench("--shapes", "64x64x5", "--no-decode", "--prompt",
                 prompt) == 1
    err = capsys.readouterr().err
    assert "error: this bench does not fit in memory: encode " in err
    assert "with the encode over it. Try a shorter prompt." in err
    assert _kinds(bench.ev, "load_encoder") == []


def test_the_shapes_that_ran_keep_their_records(bench, tmp_path, capsys):
    out = tmp_path / "b.jsonl"
    bench.cfg.update(fail_at=2, fail=RuntimeError("out of memory"))
    assert bench("--shapes", "64x64x5,96x64x22,64x96x5", "--steps", "1",
                 "--json-out", str(out)) == 1
    assert "error: out of memory" in capsys.readouterr().err
    recs = _records(out.read_text())
    assert [r["settings"]["width"] for r in recs] == [64, 96]
    for r in recs:
        assert r["error"] == "RuntimeError: out of memory"
        assert list(r["peak"]) == list(r["predicted"]) == ["encode",
                                                           "denoise"]


def test_an_interrupt_keeps_the_records_too(bench, tmp_path):
    out = tmp_path / "b.jsonl"
    bench.cfg.update(fail_at=1, fail=KeyboardInterrupt())
    assert bench("--shapes", "64x64x5,96x64x22", "--steps", "1",
                 "--json-out", str(out)) == 130
    (r,) = _records(out.read_text())
    assert r["error"] == "KeyboardInterrupt"


def test_a_decode_that_fails_keeps_what_each_shape_measured(bench,
                                                            tmp_path):
    out = tmp_path / "b.jsonl"
    bench.cfg["decode_fail"] = (1, RuntimeError("decode"))
    assert bench("--shapes", "64x64x5,96x64x22", "--steps", "1",
                 "--json-out", str(out)) == 1
    first, second = _records(out.read_text())
    assert list(first["peak"]) == list(first["predicted"]) == list(
        memfit.PHASES)
    assert list(second["peak"]) == ["encode", "denoise", "audio"]
    assert "decode_s" in first and "decode_s" not in second
    assert first["error"] == second["error"] == "RuntimeError: decode"


@pytest.mark.parametrize("argv,match", [
    (["--shapes", ""], "expected one or more WxHxF shapes"),
    (["--shapes", "512x512x23"], "not a clip length"),
    (["--shapes", "512x512"], "expected WxHxF, got '512x512'"),
])
def test_a_bad_shape_list_is_a_usage_error(bench, capsys, argv, match):
    with pytest.raises(SystemExit) as e:
        bench(*argv)
    assert e.value.code == 2 and match in capsys.readouterr().err
    assert bench.ev == []


def test_usage_errors_exit_2_before_any_load(bench, capsys):
    assert bench("--shift", "1e-300") == 2
    assert "gives no 3-step schedule" in capsys.readouterr().err
    assert bench("--vae", bench.files["--vae"], flags=False) == 2
    assert "--encoder is required" in capsys.readouterr().err
    assert bench("--encoder", bench.files["--encoder"], flags=False) == 2
    assert "--vae is required" in capsys.readouterr().err
    assert _kinds(bench.ev, "load_encoder") == []


@pytest.mark.parametrize("fail,code", [(KeyboardInterrupt, 130),
                                       (RuntimeError, 1)])
def test_failures_exit_as_generate_does(bench, fail, code):
    bench.cfg.update(fail_at=0, fail=fail("boom"))
    assert bench("--shapes", "64x64x5") == code


def test_an_entry_brings_its_adapters_before_the_flags(bench, tmp_path,
                                                       capsys):
    for name in ("lora.safetensors", "extra.safetensors"):
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
    variant: fl2va
    adapters: [lora.safetensors]
    defaults: {{steps: 8, shift: 5}}
""")
    assert bench("--config", str(cfg), "--adapter",
                 str(tmp_path / "extra.safetensors") + ":0.5", "--shapes",
                 "64x64x5", "--json", model="clip", flags=False) == 0
    r = _records(capsys.readouterr().out)[0]
    ads = _kinds(bench.ev, "adapter_table")[0][1]
    assert [(os.path.basename(p), st) for p, st in ads] == [
        ("lora.safetensors", 1.0), ("extra.safetensors", 0.5)]
    assert [a["file"] for a in r["model"]["adapters"]] == [
        "lora.safetensors", "extra.safetensors"]
    assert r["settings"]["shifts"] == {"video": 5.0, "audio": 3.0}
    assert r["settings"]["steps"] == 3
    assert _kinds(bench.ev, "load") == [("load", "fl2va")]


def test_an_adapter_on_a_merged_file_warns(bench, tmp_path, capsys):
    lora = tmp_path / "lora.safetensors"
    lora.write_bytes(b"x")
    bench.cfg["variant"] = hf.TURBO4
    assert bench("--shapes", "64x64x5", "--steps", "1", "--no-decode",
                 "--adapter", str(lora)) == 0
    assert ("[video] warning: this MiniMax-H3 GGUF has lightx2v's Turbo "
            "4-step v1.0 768p LoRA merged in") in capsys.readouterr().err
