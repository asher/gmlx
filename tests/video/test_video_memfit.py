"""The video memory plan against measured peaks: each fit leaves at
least 0.25 GiB over every measured cell, its closest cell within 0.35
GiB and none more than 1.5 GiB over past its slope's 10 percent; the
rows of a clip; what each file costs; the refusal and its message. CPU,
no weights."""
from __future__ import annotations

import re

import pytest
from video_fixtures import load_json

from gmlx.serve import capacity
from gmlx.video import memfit
from gmlx.video.families.h3 import family as hf

GiB = 1 << 30
ROOM = 0.25 * GiB               # what a fit leaves over every cell
DOC = load_json("memfit_h3_cells.json")
FILES = DOC["files"]
ROW = memfit.PEAKS["h3"]


def _plan(cell, files):
    w, h, _ = cell["shape"]
    rows = cell["rows"]
    return memfit.plan_memory(
        "h3", dit=files["dit"], encoder=files["encoder"], vae=files["vae"],
        audio_vae=files["audio_vae"],
        loras=[files["lora"]] * cell["loras"], rows=rows["total"],
        text=rows["text"], audio_latents=rows["audio"] // 2, pixels=w * h,
        budget=None)


def _rooms() -> dict:
    """Bytes each fit leaves over each of its cells, by fit, with the
    part of it that is the slope's 10 percent at that cell."""
    out: dict = {k: [] for k in ("encode", "step", "plain_step", "audio",
                                 "decode", "tiny_decode")}

    def add(fit, room, units):
        out[fit].append((room, getattr(ROW, fit).k / 11 * units))
    for c in DOC["cells"]:
        plan = _plan(c, FILES)
        w, h, _ = c["shape"]
        units = {"encode": 0, "denoise": c["rows"]["total"],
                 "audio": c["rows"]["audio"] // 2, "decode": w * h}
        for phase, measured in c["peak"].items():
            fit = ("step" if c["loras"] else "plain_step") \
                if phase == "denoise" else phase
            add(fit, plan.phases[phase] - measured, units[phase])
    enc = DOC["encode"]
    add("encode", enc["file"] + ROW.encoder_load - enc["load_peak"], 0)
    for r in enc["rows"]:
        add("encode", ROW.encoder_keep * enc["file"]
            + ROW.encode.bytes(r["tokens"]) - r["peak"], r["tokens"])
    aud = DOC["audio"]
    for r in aud["rows"]:
        add("audio", aud["file"] + ROW.audio.bytes(r["latents"])
            - r["peak"], r["latents"])
    tiny = DOC["tiny"]
    for r in tiny["rows"]:
        w, h, _ = r["shape"]
        add("tiny_decode", tiny["file"] + ROW.tiny_decode.bytes(w * h)
            - r["peak"], w * h)
    return out


def test_the_cells_cover_the_shapes_and_both_recipes():
    shapes = {(tuple(c["shape"]), c["loras"]) for c in DOC["cells"]}
    for s in ((512, 512, 22), (768, 768, 56), (768, 768, 124),
              (1344, 768, 124)):
        assert (s, 1) in shapes and (s, 0) in shapes
    # Two stacked LoRAs, and a 15 s clip at the default size.
    assert ((1344, 768, 124), 2) in shapes
    assert ((1344, 768, 362), 1) in shapes
    assert max(r["latents"] for r in DOC["audio"]["rows"]) == 603
    assert max(r["tokens"] for r in DOC["encode"]["rows"]) == 5001


@pytest.mark.parametrize("fit", ["encode", "step", "plain_step", "audio",
                                 "decode", "tiny_decode"])
def test_each_fit_covers_its_cells_and_stays_close(fit):
    cells = _rooms()[fit]
    rooms = [r for r, _ in cells]
    assert len(rooms) >= 3
    assert min(rooms) >= ROOM, [r / GiB for r in rooms]
    # A fit, not a ceiling far above the readings.
    assert min(rooms) <= 0.35 * GiB
    assert max(r - slope for r, slope in cells) <= 1.5 * GiB


def test_the_encoder_run_keeps_less_than_the_plan_counts():
    enc = DOC["encode"]
    assert enc["resident"] <= ROW.encoder_keep * enc["file"]
    for r in enc["rows"]:
        assert r["peak"] >= enc["resident"]


def test_a_long_prompt_is_priced_near_what_it_measured():
    """A plan counts a prompt's tokens as its rows. At the measured
    prompts the encode stays within 1.5 GiB of the larger of the load and
    the run."""
    enc = DOC["encode"]
    for r in enc["rows"]:
        plan = memfit.plan_memory(
            "h3", dit=0, encoder=enc["file"], vae=0, audio_vae=0, rows=1,
            text=r["tokens"], audio_latents=1, pixels=1, budget=None)
        measured = max(enc["load_peak"], r["peak"])
        assert ROOM <= plan.phases["encode"] - measured <= 1.5 * GiB


def test_the_encoder_load_is_priced_close():
    enc = DOC["encode"]
    room = enc["file"] + ROW.encoder_load - enc["load_peak"]
    assert ROOM <= room <= 0.35 * GiB


def test_the_encode_is_the_load_or_the_kept_part_and_the_prompt():
    kw = dict(dit=10 * GiB, encoder=20 * GiB, vae=5 * GiB, audio_vae=GiB,
              rows=20_000, audio_latents=200, pixels=1 << 20, budget=None)
    short = memfit.plan_memory("h3", text=25, **kw)
    assert short.terms["encode"] == {"encoder": 20 * GiB,
                                     "load": ROW.encoder_load}
    long = memfit.plan_memory("h3", text=30_000, **kw)
    assert long.terms["encode"] == {"encoder": ROW.encoder_keep * 20 * GiB,
                                    "prompt": ROW.encode.bytes(30_000)}
    assert long.phases["encode"] > short.phases["encode"]
    assert short.encoder_load == long.encoder_load == short.phases["encode"]


def test_the_rows_are_the_layouts():
    assert memfit.clip_rows(hf.FAMILY, frames=124, width=1344, height=768,
                            text=25) == (37_735, 207)
    for c in DOC["cells"]:
        w, h, f = c["shape"]
        assert memfit.clip_rows(hf.FAMILY, frames=f, width=w, height=h,
                                text=c["rows"]["text"]) == (
            c["rows"]["total"], c["rows"]["audio"] // 2)


def test_each_lora_costs_twice_its_file_and_its_own_step():
    kw = dict(dit=10 * GiB, encoder=18 * GiB, vae=5 * GiB, audio_vae=GiB,
              rows=20_000, text=25, audio_latents=200, pixels=1 << 20,
              budget=None)
    plain = memfit.plan_memory("h3", **kw)
    one = memfit.plan_memory("h3", loras=[2 * GiB], **kw)
    two = memfit.plan_memory("h3", loras=[2 * GiB, GiB], **kw)
    assert one.terms["denoise"]["lora"] == 4 * GiB
    assert two.terms["denoise"]["lora"] == 6 * GiB
    assert one.terms["denoise"]["step"] == ROW.step.bytes(20_000)
    assert plain.terms["denoise"]["step"] == ROW.plain_step.bytes(20_000)
    assert plain.phases["encode"] == one.phases["encode"]


def test_a_preview_prices_taeh3():
    kw = dict(dit=10 * GiB, encoder=18 * GiB, audio_vae=GiB, rows=20_000,
              text=25, audio_latents=200, pixels=1 << 20, budget=None)
    full = memfit.plan_memory("h3", vae=5 * GiB, **kw)
    tiny = memfit.plan_memory("h3", vae=22 << 20, tiny=True, **kw)
    assert tiny.terms["decode"]["chunk"] == ROW.tiny_decode.bytes(1 << 20)
    assert full.terms["decode"]["chunk"] == ROW.decode.bytes(1 << 20)


def test_a_plan_with_no_decode_has_no_vae_phase():
    plan = memfit.plan_memory("h3", dit=10 * GiB, encoder=18 * GiB, vae=0,
                              audio_vae=0, rows=20_000, text=25,
                              audio_latents=200, pixels=1 << 20,
                              decode=False, budget=None)
    assert list(plan.phases) == ["encode", "denoise"]


def test_a_run_plan_reads_every_file_and_shard(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    sizes = {"h3-00001-of-00002.gguf": 3, "h3-00002-of-00002.gguf": 5,
             "enc.gguf": 7, "v.safetensors": 11, "a.safetensors": 13,
             "l1.safetensors": 17, "l2.safetensors": 19}
    for name, mib in sizes.items():
        with open(tmp_path / name, "wb") as f:
            f.truncate(mib << 20)
    kw = dict(dit="~/h3-00001-of-00002.gguf", encoder="~/enc.gguf",
              loras=["~/l1.safetensors", "~/l2.safetensors"], frames=22,
              width=512, height=320, text=25)
    plan = memfit.plan_run(hf.FAMILY, vae="~/v.safetensors",
                           audio_vae="~/a.safetensors", budget=123.0, **kw)
    assert plan.budget == 123.0
    assert plan.terms["denoise"]["dit"] == 8 << 20
    assert plan.terms["denoise"]["lora"] == ROW.lora_k * (36 << 20)
    assert plan.encoder_load == int((7 << 20) + ROW.encoder_load)
    assert plan.terms["audio"]["audio_vae"] == 13 << 20
    assert plan.terms["decode"] == {"vae": 11 << 20,
                                    "chunk": ROW.decode.bytes(512 * 320)}
    assert plan.rows == memfit.clip_rows(hf.FAMILY, frames=22, width=512,
                                         height=320, text=25)[0]


# Refusals, on the measured files with the LoRA.

def _h3(w, h, f, text=25, budget=None):
    """A plan with ``text`` prompt rows; the cells' prompt has 25."""
    rows, audio = memfit.clip_rows(hf.FAMILY, frames=f, width=w, height=h,
                                   text=text)
    return memfit.plan_memory(
        "h3", dit=FILES["dit"], encoder=FILES["encoder"], vae=FILES["vae"],
        audio_vae=FILES["audio_vae"], loras=[FILES["lora"]], rows=rows,
        text=text, audio_latents=audio, pixels=w * h, budget=budget)


@pytest.fixture
def no_overcommit(monkeypatch):
    """The stock margin and reserve and no override, whatever the caller
    set."""
    monkeypatch.delenv("GMLX_GOV_MARGIN", raising=False)
    monkeypatch.delenv("GMLX_GOV_RESERVE_GB", raising=False)
    monkeypatch.setattr(capacity, "overcommit", lambda: False)


def _refusal(plan) -> str:
    with pytest.raises(memfit.VideoMemoryError) as e:
        memfit.check(plan)
    return str(e.value)


def test_on_24_gb_every_clip_names_the_encoder(no_overcommit):
    budget = capacity.ram_budget_bytes(24 * GiB)
    for shape in ((1344, 768, 124), (768, 768, 124), (512, 512, 22)):
        msg = _refusal(_h3(*shape, budget=budget))
        assert msg.startswith("this clip does not fit in memory: encode ")
        assert ("The text encoder needs more memory than this Mac has, and "
                "gmlx has no smaller one for this model yet. "
                "GMLX_OVERCOMMIT=1 overrides.") in msg
    assert "with the encode and the denoise over it." in _refusal(
        _h3(1344, 768, 124, budget=budget))


def test_on_32_gb_768_fits_and_a_prompt_can_be_too_long(no_overcommit):
    budget = capacity.ram_budget_bytes(32 * GiB)
    assert _h3(768, 768, 124, budget=budget).fits
    # About 2.6 KB of English, priced by its tokens.
    assert _h3(768, 768, 124, text=600, budget=budget).fits
    assert _h3(512, 512, 22, text=7000, budget=budget).fits
    msg = _refusal(_h3(1344, 768, 124, budget=budget))
    assert "with the denoise over it. Try a smaller size or a shorter clip" \
        in msg
    msg = _refusal(_h3(512, 512, 22, text=30_000, budget=budget))
    assert "Try a shorter prompt." in msg


@pytest.mark.parametrize("gap", [1, 1 << 20, 4 << 20, 9 << 20])
def test_a_refusal_never_prints_a_phase_on_the_wrong_side(no_overcommit,
                                                          gap):
    plan = _h3(512, 512, 22, budget=1.0)
    plan.budget = plan.peak - gap
    msg = _refusal(plan)
    budget = float(re.search(r"working budget of ([\d.]+) GiB", msg)[1])
    for phase, b in plan.phases.items():
        shown = float(re.search(rf"{phase} ([\d.]+) GiB", msg)[1])
        assert shown > budget if b > plan.budget else shown <= budget


@pytest.mark.parametrize("encode,shown", [(15.206, "15.21"),
                                           (15.212, "15.22")])
def test_a_refusal_rounds_the_phase_over_up_and_the_rest_down(
        no_overcommit, encode, shown):
    plan = memfit.VideoPlan(
        phases={"encode": int(encode * GiB), "denoise": int(15.199 * GiB)},
        terms={}, budget=15.2049 * GiB, rows=1)
    assert plan.describe() == (
        f"encode {shown} GiB, denoise 15.19 GiB against a working budget "
        "of 15.20 GiB, with the encode over it")


def test_a_plan_at_its_budget_fits():
    plan = _h3(512, 512, 22, budget=1.0)
    plan.budget = plan.peak
    assert plan.fits and plan.over == []
    plan.budget = plan.peak - 1
    assert plan.over == [plan.largest]


def test_no_budget_never_refuses(monkeypatch, no_overcommit):
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: None)
    plan = memfit.plan_memory("h3", dit=1 << 50, encoder=1, vae=1,
                              audio_vae=1, rows=1, text=1, audio_latents=1,
                              pixels=1)
    assert plan.budget is None and plan.fits
    memfit.check(plan)
    assert plan.describe().startswith("encode 0.80 GiB, denoise ")
    assert "budget" not in plan.describe()


def test_gmlx_overcommit_lets_a_plan_through(monkeypatch):
    plan = _h3(512, 512, 22, budget=GiB)
    monkeypatch.setenv("GMLX_OVERCOMMIT", "1")
    memfit.check(plan)
    monkeypatch.setenv("GMLX_OVERCOMMIT", "0")
    with pytest.raises(memfit.VideoMemoryError,
                       match="GMLX_OVERCOMMIT=1 overrides") as e:
        memfit.check(plan)
    assert ("with the encode, the denoise, the audio and the decode over "
            "it") in str(e.value)
