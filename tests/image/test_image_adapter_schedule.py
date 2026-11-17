"""Adapter schedules: pDMD-2steps' schedule from gmlx's table by the
sha256 of its file, a sidecar over the table and a config entry over the
sidecar, the request's own settings over all of them, adapters that
disagree refused, a bad sidecar refused with its path, and the sidecar
--save-adapter-schedule writes; the CLI's plan and sample under each;
and heads in order on the tiny MiniMax-H3-Image DiT, step i reading head
i. CPU."""
from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import h3_image_gates as hg
from image_fixtures import fixture_view
from lora_fixtures import write_keys
from oracle_fixture import committed_fixture

from gmlx.commands import image_preflight
from gmlx.commands.cli import umbrella_main
from gmlx.image import api
from gmlx.image.conditioning import Conditioning
from gmlx.image.families.h3_image import family as hi
from gmlx.image.lora import schedule as sc
from gmlx.image.samplers import make_sampler

HEADER = "h3_image_pruned_ud_q4_k_xl.header.json"
PDMD = sc.AdapterSchedule(sigmas=(1.0, 12 / 13, 0.0), heads="in_order",
                          size=(1024, 1024))
LEAF = "blocks.0.attn.out_proj"


@pytest.fixture(scope="module")
def view():
    return fixture_view(HEADER)


def _lora(path, view, seed=0) -> str:
    """A rank 1 LoRA on one leaf of the release file, in its names."""
    out, inp = reversed(view.tensors[f"{LEAF}.weight"].shape)
    rng = np.random.default_rng(seed)
    stem = f"diffusion_model.{LEAF}"
    return write_keys(path, {
        f"{stem}.lora_A.weight": rng.standard_normal((1, inp)).astype(
            np.float32) * 1e-3,
        f"{stem}.lora_B.weight": rng.standard_normal((out, 1)).astype(
            np.float32) * 1e-3})


@pytest.fixture
def known(tmp_path, view, monkeypatch):
    """A LoRA file standing in for pDMD's in gmlx's table, by its sha256,
    and a second file the table does not know."""
    path = _lora(tmp_path / "pdmd.safetensors", view)
    other = _lora(tmp_path / "other.safetensors", view, seed=1)
    monkeypatch.setattr(sc, "KNOWN", {sc.file_sha256(path): sc.Known(
        "pDMD-2steps", "linoyts/MiniMax-H3-Image-pDMD-2steps", PDMD)})
    return SimpleNamespace(path=path, other=other)


def test_the_table_applies_pdmds_schedule_and_the_request_wins(known):
    d = sc.declared(known.path)
    assert (d.name, d.schedule) == ("pDMD-2steps", PDMD)
    assert sc.declared(known.other) is None
    got = sc.resolve([d, None])
    assert (got.sigmas, got.heads, got.size) == (
        (1.0, 12 / 13, 0.0), "in_order", (1024, 1024))
    assert got.line == (
        "adapter schedule of pDMD-2steps (gmlx's table of known adapters, "
        "linoyts/MiniMax-H3-Image-pDMD-2steps): sigmas 1, 0.923077, 0, "
        "heads in order, default size 1024x1024")
    # The request's sigmas read heads in order; its step count runs the
    # family's schedule; its size is the size.
    got = sc.resolve([d], sigmas=(1.0, 0.5, 0.0), size=(2048, 2048))
    assert (got.sigmas, got.heads, got.size) == (
        (1.0, 0.5, 0.0), "in_order", (2048, 2048))
    assert got.line.endswith("nothing; sigmas and size given, which win")
    got = sc.resolve([d], steps=6)
    assert (got.sigmas, got.heads, got.size) == (None, None, (1024, 1024))
    assert sc.resolve([None, None]) == sc.Resolved()


def test_adapters_that_disagree_are_refused_unless_the_request_chooses(
        known):
    pdmd = sc.declared(known.path)
    four = sc.Declared("four.safetensors", "its sidecar four.gmlx.json",
                       dataclasses.replace(PDMD, sigmas=(1.0, 0.5, 0.0)))
    with pytest.raises(sc.ScheduleError, match=(
            r"adapters pDMD-2steps \(gmlx's table .*\) and four.safetensors "
            r"\(its sidecar four.gmlx.json\) declare different sigmas: "
            r"1, 0.923077, 0 and 1, 0.5, 0; give --sigmas to choose")):
        sc.resolve([pdmd, four], names={"sigmas": "--sigmas"})
    assert sc.resolve([pdmd, four], sigmas=(1.0, 0.0)).sigmas == (1.0, 0.0)
    big = dataclasses.replace(pdmd, schedule=sc.AdapterSchedule(
        size=(2048, 2048)))
    with pytest.raises(sc.ScheduleError, match="different size"):
        sc.resolve([pdmd, big])
    assert sc.resolve([pdmd, big], size=(1024, 1024)).size == (1024, 1024)


def test_a_sidecar_wins_over_the_table_and_an_entry_over_the_sidecar(known):
    side = sc.write_sidecar(known.path, sc.AdapterSchedule(
        sigmas=(1.0, 0.75, 0.0)))
    assert side == known.path + ".gmlx.json"
    d = sc.declared(known.path)
    assert (d.name, d.source, d.schedule) == (
        "pdmd.safetensors", f"its sidecar {side}",
        sc.AdapterSchedule(sigmas=(1.0, 0.75, 0.0)))
    # A declared schedule without heads reads them in order.
    assert sc.resolve([d]).heads == "in_order"
    d = sc.declared(known.path, {"heads": "span", "size": "1280x1280"})
    assert (d.source, d.schedule) == ("its entry in the config",
                                      sc.AdapterSchedule(heads="span",
                                                         size=(1280, 1280)))
    assert sc.resolve([d]).heads is None


@pytest.mark.parametrize("body, why", [
    ("{", "does not read as JSON"),
    ('["sigmas"]', "is not a JSON object"),
    ('{"sigmas": [1, 0]}', "has schema None"),
    ('{"schema": "gmlx.adapter_schedule/1", "sigma": [1, 0]}',
     "unknown key 'sigma'"),
    ('{"schema": "gmlx.adapter_schedule/1", "sigmas": [0, 1]}',
     "sigmas must not increase"),
    ('{"schema": "gmlx.adapter_schedule/1", "sigmas": [1]}',
     "two values or more"),
    ('{"schema": "gmlx.adapter_schedule/1", "heads": "first"}',
     "heads is one of in_order, span"),
    ('{"schema": "gmlx.adapter_schedule/1", "size": "big"}',
     "size is WxH"),
    ('{"schema": "gmlx.adapter_schedule/1"}', "declares none of"),
    ('{"schema": "gmlx.adapter_schedule/1", "sha256": "ab", "size": '
     '"1024x1024"}', "was written for another file")])
def test_a_bad_sidecar_is_refused_with_its_path(known, body, why):
    side = known.other + ".gmlx.json"
    with open(side, "w") as f:
        f.write(body)
    with pytest.raises(sc.ScheduleError) as e:
        sc.declared(known.other)
    assert side in str(e.value) and why in str(e.value)


def test_a_sidecar_is_bound_to_its_file_and_needs_a_writable_folder(
        known, tmp_path, view):
    sched = sc.AdapterSchedule(sigmas=(1.0, 0.5, 0.0), heads="in_order",
                               size=(1024, 1280))
    side = sc.write_sidecar(known.other, sched)
    assert json.load(open(side)) == {
        "schema": sc.SCHEMA, "sha256": sc.file_sha256(known.other),
        "sigmas": [1.0, 0.5, 0.0], "heads": "in_order", "size": "1024x1280"}
    assert sc.read_sidecar(known.other) == sched
    # A new version of the file under the same name reads no stale
    # schedule.
    _lora(known.other, view, seed=7)
    with pytest.raises(sc.ScheduleError, match="written for another file"):
        sc.read_sidecar(known.other)
    ro = tmp_path / "ro"
    ro.mkdir()
    path = _lora(ro / "a.safetensors", view)
    ro.chmod(0o500)
    try:
        with pytest.raises(sc.ScheduleError, match=(
                f"cannot write the adapter schedule {path}.gmlx.json: the "
                f"folder {ro} is not writable")):
            sc.write_sidecar(path, sched)
    finally:
        ro.chmod(0o700)


@pytest.fixture
def h3(tmp_path, view, monkeypatch):
    """The CLI on a MiniMax-H3-Image DiT by path. Its preflight records the
    size it plans and stops the run there unless ``go`` is set; past it
    the pipeline is a stand-in whose sample records what it is given."""
    files = {}
    for name in ("dit.gguf", "enc.gguf", "vae.gguf"):
        (tmp_path / name).write_bytes(b"x")
        files[name] = str(tmp_path / name)
    state = SimpleNamespace(planned=[], sampled=[], go=False)
    monkeypatch.setattr(image_preflight, "identify_dit",
                        lambda p, **k: SimpleNamespace(
                            family=hi.FAMILY, view=view, variant="pruned",
                            config=hi.synth_config(view)))

    def check(*a, **k):
        state.planned.append((k["width"], k["height"]))
        if not state.go:
            raise RuntimeError("planned")

    monkeypatch.setattr(image_preflight, "check", check)

    def load(dit, **kw):
        return SimpleNamespace(load_seconds={}, variant="pruned",
                               encoder=SimpleNamespace(release=lambda: None))

    def sample(pipe, cond, **kw):
        state.sampled.append(kw)
        return mx.zeros((1, 24, kw["height"] // 16, kw["width"] // 16))

    from gmlx.commands import image as cmd

    monkeypatch.setattr(api, "load", load)
    monkeypatch.setattr(cmd, "_read_adapters", lambda a, pipe, log: [])
    monkeypatch.setattr(api, "encode_text", lambda *a, **k: None)
    monkeypatch.setattr(api, "plan_request", lambda *a, **k: SimpleNamespace(
        release_encoder=False, levers=(), rows=1))
    monkeypatch.setattr(api, "sample", sample)
    monkeypatch.setattr(api, "vae_decode", lambda pipe, x, **k: mx.zeros(
        (1, 64, 64, 3)))
    monkeypatch.chdir(tmp_path)
    state.base = ["image", "generate", files["dit.gguf"], "--encoder",
                  files["enc.gguf"], "--vae", files["vae.gguf"], "--prompt",
                  "p", "--no-upsample", "--seed", "1", "--output",
                  str(tmp_path / "a.png"), "--json"]
    return state


def test_the_cli_plans_pdmd_at_its_size_and_a_size_given_wins(
        h3, known, capsys):
    assert umbrella_main(h3.base + ["--adapter", known.path]) != 0
    assert h3.planned == [(1024, 1024)]
    err = capsys.readouterr().err
    assert "[image] adapter schedule of pDMD-2steps" in err
    assert umbrella_main(h3.base + ["--adapter", known.path, "--size",
                                    "2048x2048"]) != 0
    assert h3.planned[-1] == (2048, 2048)
    assert "--size given, which wins" in capsys.readouterr().err
    # At strength 0 the adapter sets nothing.
    assert umbrella_main(h3.base + ["--adapter", f"{known.path}:0"]) != 0
    assert h3.planned[-1] == (2048, 2048)


def test_the_cli_samples_on_the_schedule_with_heads_in_order(
        h3, known, tmp_path, capsys):
    h3.go = True
    runs = [([], ((1.0, 12 / 13, 0.0), "in_order", 2)),
            (["--sigmas", "1,1/2,0"], ((1.0, 0.5, 0.0), "in_order", 2)),
            (["--steps", "6"], (None, None, 6))]
    for extra, want in runs:
        assert umbrella_main(h3.base + ["--adapter", known.path] + extra) \
            == 0, capsys.readouterr().err
        kw = h3.sampled[-1]
        assert (kw["sigmas"], kw["heads"], kw["steps"]) == want
        rec = json.loads(capsys.readouterr().out)["settings"]
        assert (rec.get("sigmas"), rec.get("heads"), rec["steps"]) == (
            None if want[0] is None else list(want[0]), want[1], want[2])
    # A schedule of more steps than the bank's six heads is refused
    # before any file loads.
    n = len(h3.sampled)
    assert umbrella_main(h3.base + ["--sigmas", "1,.9,.8,.7,.6,.5,.4,0"]) \
        == 2
    assert "bank holds 6 heads, not the 7 steps" in capsys.readouterr().err
    assert len(h3.sampled) == n
    assert umbrella_main(h3.base + ["--sigmas", "1,1/2,0", "--steps",
                                    "6"]) == 2
    assert "--steps 6 disagrees with the 2 steps of --sigmas" in \
        capsys.readouterr().err


def test_the_cli_saves_a_schedule_that_its_later_runs_read(
        h3, known, tmp_path, capsys):
    other = known.other
    assert umbrella_main(h3.base + [
        "--adapter", other, "--sigmas", "1,3/4,0", "--size", "1280x1280",
        "--save-adapter-schedule"]) != 0
    side = other + ".gmlx.json"
    assert f"[image] wrote the adapter schedule {side}" in \
        capsys.readouterr().err
    assert sc.read_sidecar(other) == sc.AdapterSchedule(
        sigmas=(1.0, 0.75, 0.0), heads="in_order", size=(1280, 1280))
    assert umbrella_main(h3.base + ["--adapter", other]) != 0
    assert h3.planned[-1] == (1280, 1280)
    assert f"(its sidecar {side})" in capsys.readouterr().err
    # Stacked with pDMD, the two schedules disagree.
    assert umbrella_main(h3.base + ["--adapter", other, "--adapter",
                                    known.path]) == 2
    assert "declare different sigmas" in capsys.readouterr().err
    assert umbrella_main(h3.base + ["--adapter", other, "--adapter",
                                    known.path, "--sigmas", "1,0",
                                    "--size", "1024x1024",
                                    "--save-adapter-schedule"]) == 2
    assert "one --adapter file, and the command gives 2" in \
        capsys.readouterr().err
    with open(side, "w") as f:
        f.write("{")
    assert umbrella_main(h3.base + ["--adapter", other]) == 2
    assert f"the adapter schedule {side} does not read as JSON" in \
        capsys.readouterr().err


@pytest.fixture(scope="module")
def tiny():
    weights, wm = committed_fixture("selftest_h3_image_tiny_weights")
    arrays, _ = committed_fixture("selftest_h3_image_tiny_s0")
    weights = {k: np.asarray(v, np.float32) for k, v in weights.items()}
    pipe = api.ImagePipeline(
        dit=hg.tiny_model(weights, wm["inputs"]["config"]),
        family=dataclasses.replace(hi.FAMILY, min_pixels=None),
        variant="pruned")
    return pipe, arrays


def _sample(pipe, arrays, sigmas, heads=None, **kw):
    lat = np.asarray(arrays["latents"], np.float32)
    cond = Conditioning(mx.array(arrays["features"]), "h3-heads")
    return np.array(api.sample(pipe, cond, height=lat.shape[2] * 16,
                               width=lat.shape[3] * 16, noise=mx.array(lat),
                               sigmas=sigmas, heads=heads, **kw))


def test_heads_in_order_read_head_i_at_step_i(tiny):
    """On pDMD's two steps, step 0 reads head 0 and step 1 head 1, as a
    loop of steps that name them gives, which the heads of each step's
    span do not; on the grid of the default schedule the two rules are
    one."""
    pipe, arrays = tiny
    s = np.asarray((1.0, 12 / 13, 0.0), np.float32)
    got = _sample(pipe, arrays, s, heads="in_order")
    cond = Conditioning(mx.array(arrays["features"]), "h3-heads")
    x = mx.array(np.asarray(arrays["latents"], np.float32))
    smp = make_sampler("euler", s)
    for i in range(2):
        x = smp.step(x, api.denoise_step(pipe, x, s[i], cond, head=i), i)
    assert np.array_equal(got, np.array(x))
    assert not np.array_equal(got, _sample(pipe, arrays, s))
    grid = np.asarray(hi.PDD_SIGMAS, np.float32)
    assert np.array_equal(_sample(pipe, arrays, grid, heads="in_order"),
                          _sample(pipe, arrays, grid))
    # An img2img run from step 3 of the grid reads heads 3 to 5.
    init = mx.array(np.ascontiguousarray(np.flip(arrays["latents"], -1)))
    assert np.array_equal(
        _sample(pipe, arrays, grid, "in_order", init_latents=init, denoise=0.5),
        _sample(pipe, arrays, grid, init_latents=init, denoise=0.5))
    with pytest.raises(api.RequestRefusal, match="bank holds 6 heads, not "
                       "the 7 steps"):
        _sample(pipe, arrays, np.linspace(1, 0, 8), heads="in_order")
    with pytest.raises(ValueError, match="bank holds heads 0 to 5, not 6"):
        api.denoise_step(pipe, x, s[0], cond, head=6)
