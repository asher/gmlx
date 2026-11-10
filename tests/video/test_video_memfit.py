"""The video memory plan against measured peaks: each fit leaves at
least 0.25 GiB over every measured cell, its closest cell within 0.35
GiB and none more than 1.5 GiB over past its slope's 10 percent; the
same for FastH3's own step fits and for the fits of keyframes and
anchors; the rows of a clip; what each file costs; the refusal and its
message; the admission table and the int8 table of docs/video.md. CPU,
no weights."""
from __future__ import annotations

import dataclasses
import json
import os
import re
import struct

import pytest
from video_fixtures import load_json

from gmlx.serve import capacity
from gmlx.video import memfit
from gmlx.video.families.h3 import family as hf
from gmlx.video.layout import AnchorRows

GiB = 1 << 30
ROOM = 0.25 * GiB               # what a fit leaves over every cell
DOC = load_json("memfit_h3_cells.json")
FILES = DOC["files"]
ROW = memfit.PEAKS["h3"]
FAST = load_json("memfit_fasth3_cells.json")
FAST_ROW = memfit.PEAKS[hf.FASTH3]


def _plan(cell, files, variant=None):
    w, h, _ = cell["shape"]
    rows = cell["rows"]
    return memfit.plan_memory(
        "h3", dit=files["dit"], encoder=files["encoder"], vae=files["vae"],
        audio_vae=files["audio_vae"],
        loras=[files["lora"]] * cell["loras"],
        lora_dtypes=["BF16"] * cell["loras"], rows=rows["total"],
        text=rows["text"], audio_latents=rows["audio"] // 2, pixels=w * h,
        variant=variant, budget=None)


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


def _fast_rooms() -> dict:
    """``_rooms`` over the FastH3 cells, each phase of each cell against
    its FastH3 plan: its own step fits, and H3's fits for the encode and
    the decodes, whose files are H3's."""
    out: dict = {k: [] for k in ("encode", "step", "plain_step", "audio",
                                 "decode")}
    for c in FAST["cells"]:
        plan = _plan(c, FAST["files"], hf.FASTH3)
        w, h, _ = c["shape"]
        units = {"encode": 0, "denoise": c["rows"]["total"],
                 "audio": c["rows"]["audio"] // 2, "decode": w * h}
        for phase, measured in c["peak"].items():
            fit = ("step" if c["loras"] else "plain_step") \
                if phase == "denoise" else phase
            out[fit].append((plan.phases[phase] - measured,
                             getattr(FAST_ROW, fit).k / 11 * units[phase]))
    return out


def test_fasth3_has_its_own_steps_and_h3s_other_fits():
    assert FAST["variant"] == hf.FASTH3 and FAST["vsa_sparsity"] == 0.8
    assert dataclasses.replace(FAST_ROW, step=ROW.step,
                               plain_step=ROW.plain_step) == ROW
    assert FAST_ROW.step != ROW.step and FAST_ROW.plain_step != ROW.plain_step


@pytest.mark.parametrize("fit", ["encode", "step", "plain_step", "audio",
                                 "decode"])
def test_each_fasth3_fit_covers_its_cells(fit):
    """Its step fits as close as H3's are; H3's other fits over FastH3's
    cells, which run the same files, keep their room."""
    cells = _fast_rooms()[fit]
    rooms = [r for r, _ in cells]
    assert len(rooms) >= 4
    assert min(rooms) >= ROOM, [r / GiB for r in rooms]
    if fit in ("step", "plain_step"):
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


def test_a_lone_lora_in_the_dit_dtype_costs_its_file_and_others_twice():
    kw = dict(dit=10 * GiB, encoder=18 * GiB, vae=5 * GiB, audio_vae=GiB,
              rows=20_000, text=25, audio_latents=200, pixels=1 << 20,
              budget=None)

    def lora(loras, dtypes):
        return memfit.plan_memory("h3", loras=loras, lora_dtypes=dtypes,
                                  **kw).terms["denoise"]["lora"]
    assert lora([2 * GiB], ["BF16"]) == 2 * GiB
    assert lora([2 * GiB], ["F32"]) == 4 * GiB
    assert lora([2 * GiB], [None]) == 4 * GiB
    assert lora([2 * GiB, GiB], ["BF16", "BF16"]) == 6 * GiB
    plain = memfit.plan_memory("h3", **kw)
    one = memfit.plan_memory("h3", loras=[2 * GiB], lora_dtypes=["BF16"],
                             **kw)
    assert one.terms["denoise"]["step"] == ROW.step.bytes(20_000)
    assert plain.terms["denoise"]["step"] == ROW.plain_step.bytes(20_000)
    assert plain.phases["encode"] == one.phases["encode"]


def _safetensors(path, tensors: dict) -> None:
    head = json.dumps({k: {"dtype": d, "shape": s, "data_offsets": [0, 0]}
                       for k, (d, s) in tensors.items()}).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(head)) + head)


def test_a_lora_file_names_its_matrices_dtype(tmp_path):
    """lightx2v's files hold BF16 matrices and an F32 alpha per module."""
    key = "diffusion_model.blocks.0.attn.out_proj"
    _safetensors(tmp_path / "turbo.safetensors", {
        f"{key}.lora_down.weight": ("BF16", [128, 7168]),
        f"{key}.lora_up.weight": ("BF16", [5376, 128]),
        f"{key}.alpha": ("F32", [])})
    assert memfit.lora_dtype(str(tmp_path / "turbo.safetensors")) == "BF16"
    _safetensors(tmp_path / "mixed.safetensors", {
        f"{key}.lora_down.weight": ("F32", [128, 7168]),
        f"{key}.lora_up.weight": ("BF16", [5376, 128])})
    assert memfit.lora_dtype(str(tmp_path / "mixed.safetensors")) is None
    (tmp_path / "lora.gguf").write_bytes(b"GGUF" + bytes(60))
    assert memfit.lora_dtype(str(tmp_path / "lora.gguf")) is None
    # A 1x1 kernel the reader takes as a matrix counts with the matrices.
    _safetensors(tmp_path / "conv.safetensors", {
        f"{key}.lora_down.weight": ("BF16", [128, 7168]),
        f"{key}.lora_up.weight": ("BF16", [5376, 128]),
        "conv.lora_down.weight": ("F32", [8, 64, 1, 1])})
    assert memfit.lora_dtype(str(tmp_path / "conv.safetensors")) is None


@pytest.mark.parametrize("entry", [("BF16", 5), ("BF16", None),
                                   (["BF16"], [128, 7168])],
                         ids=["int shape", "null shape", "list dtype"])
def test_a_malformed_lora_header_names_no_dtype(tmp_path, entry):
    """The reader refuses such a file with its own message, so pricing
    it raises nothing."""
    _safetensors(tmp_path / "bad.safetensors", {
        "blocks.0.attn.out_proj.lora_up.weight": ("BF16", [5376, 128]),
        "blocks.0.attn.out_proj.lora_down.weight": entry})
    assert memfit.lora_dtype(str(tmp_path / "bad.safetensors")) is None


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
    # A lone bf16 LoRA, as the recipes pull, costs its file once.
    _safetensors(tmp_path / "l1.safetensors", {
        "x.lora_down.weight": ("BF16", [4, 64]),
        "x.lora_up.weight": ("BF16", [64, 4])})
    os.truncate(tmp_path / "l1.safetensors", 17 << 20)
    kw["loras"] = ["~/l1.safetensors"]
    plan = memfit.plan_run(hf.FAMILY, vae="~/v.safetensors",
                           audio_vae="~/a.safetensors", **kw)
    assert plan.terms["denoise"]["lora"] == 17 << 20


# Refusals, on the measured files with the LoRA.

def _h3(w, h, f, text=25, budget=None):
    """A plan with ``text`` prompt rows; the cells' prompt has 25."""
    rows, audio = memfit.clip_rows(hf.FAMILY, frames=f, width=w, height=h,
                                   text=text)
    return memfit.plan_memory(
        "h3", dit=FILES["dit"], encoder=FILES["encoder"], vae=FILES["vae"],
        audio_vae=FILES["audio_vae"], loras=[FILES["lora"]],
        lora_dtypes=["BF16"], rows=rows,
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


def test_on_32_gb_the_default_clip_fits_and_a_prompt_can_be_too_long(
        no_overcommit):
    budget = capacity.ram_budget_bytes(32 * GiB)
    assert _h3(1344, 768, 124, budget=budget).fits
    # About 2.6 KB of English, priced by its tokens.
    assert _h3(1344, 768, 124, text=600, budget=budget).fits
    assert _h3(512, 512, 22, text=7000, budget=budget).fits
    msg = _refusal(_h3(1344, 768, 243, budget=budget))
    assert "with the denoise over it. Try a smaller size or a shorter clip" \
        in msg
    msg = _refusal(_h3(512, 512, 22, text=30_000, budget=budget))
    assert "Try a shorter prompt." in msg


@pytest.mark.parametrize("own, extra", [(0, 5 * 10**9), (1, 800 * 10**6)])
def test_on_32_gb_a_768x768_clip_has_the_lora_room_the_video_page_gives(
        no_overcommit, own, extra):
    """docs/video.md: on a 32 GB Mac a 768x768 clip of 5 s has room for a
    bf16 LoRA file of up to 5 GB beside h3-base, and up to 800 MB beside
    h3 on its recipe's LoRA."""
    rows, audio = memfit.clip_rows(hf.FAMILY, frames=124, width=768,
                                   height=768, text=25)
    loras = [FILES["lora"]] * own + [extra]
    assert memfit.plan_memory(
        "h3", dit=FILES["dit"], encoder=FILES["encoder"], vae=FILES["vae"],
        audio_vae=FILES["audio_vae"], loras=loras,
        lora_dtypes=["BF16"] * len(loras), rows=rows, text=25,
        audio_latents=audio, pixels=768 * 768,
        budget=capacity.ram_budget_bytes(32 * GiB)).fits


def _fasth3(w, h, f, budget):
    """A FastH3 plan on its measured files, with the cells' prompt."""
    rows, audio = memfit.clip_rows(hf.FAMILY, frames=f, width=w, height=h,
                                   text=25)
    files = FAST["files"]
    return memfit.plan_memory(
        "h3", dit=files["dit"], encoder=files["encoder"], vae=files["vae"],
        audio_vae=files["audio_vae"], rows=rows, text=25,
        audio_latents=audio, pixels=w * h, variant=hf.FASTH3, budget=budget)


def test_on_32_gb_fasth3_makes_768x768_clips_and_not_the_default(
        no_overcommit):
    """FastH3 alone at 1344x768x124 measured a denoise peak over a 32 GB
    Mac's budget, so its plan refuses the default clip there."""
    budget = capacity.ram_budget_bytes(32 * GiB)
    cell = next(c for c in FAST["cells"]
                if c["shape"] == [1344, 768, 124] and not c["loras"])
    assert cell["peak"]["denoise"] > budget
    assert "with the denoise over it. Try a smaller size" in _refusal(
        _fasth3(1344, 768, 124, budget))
    assert _fasth3(768, 768, 124, budget).fits
    assert _fasth3(1344, 768, 124, capacity.ram_budget_bytes(36 * GiB)).fits


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


# Keyframes and anchors, against memfit_h3_keyframe_cells.json.

KF = load_json("memfit_h3_keyframe_cells.json")
# What the attach leaves over the kept part: the tower at float32.
TOWER = KF["encode"]["resident"] - KF["encode"]["kept"]


def _part(**kw) -> memfit.VideoPlan:
    """A plan of no files but those given, with no decode."""
    args = dict(dit=0, encoder=0, vae=0, audio_vae=0, rows=1, text=1,
                audio_latents=1, pixels=1, decode=False, budget=None)
    return memfit.plan_memory("h3", **{**args, **kw})


def _vision_run(rows: int, blocks) -> float:
    """The run of an encode with keyframes over the kept part and the
    tower, as the plan prices it: with no encoder or mmproj bytes the
    plan's encode is the tower and that run."""
    terms = _part(text=rows, tower=TOWER, blocks=blocks).terms["encode"]
    assert terms["encoder"] == 0 and terms["tower"] == TOWER
    return terms["prompt"]


def _kf_rooms() -> dict:
    """Bytes each keyframe and anchor fit leaves over each of its cells,
    by fit, with the part of it that is the slope's 10 percent there."""
    out: dict = {k: [] for k in ("tower", "vision_encode", "frame_encode",
                                 "clip_encode", "audio_encode")}
    for c in KF["tower"]:
        # A second keyframe runs beside the first one's features.
        out["tower"].append((ROW.tower.bytes(c["tokens"])
                             - (c["peak"] - c["start"]),
                             ROW.tower.k / 11 * c["tokens"]))
    for c in KF["encode"]["cells"]:
        run = _vision_run(c["rows"], c["blocks"])
        line = ROW.vision_encode
        slope = (line.k * c["rows"] if run == line.bytes(c["rows"])
                 else ROW.tower.k * max(c["blocks"])) / 11
        out["vision_encode"].append((run - (c["peak"] - c["start"]), slope))
    vae = KF["files"]["vae"]
    for c in KF["video_encode"]:
        w, h, f = c["shape"]
        fit = "frame_encode" if f == 1 else "clip_encode"
        enc = _part(vae=vae, pixels=w * h, anchor_frames=[f]).terms[
            "anchors"]
        slope = (ROW.frame_encode.k if f == 1
                 else ROW.clip_encode.k + ROW.clip_frame * f) * w * h / 11
        out[fit].append((vae + enc["encode"] - c["peak"], slope))
    audio = KF["files"]["audio_vae"]
    for c in KF["audio_encode"]:
        enc = _part(audio_vae=audio, anchor_audio=[c["latents"]]).terms[
            "anchors"]
        out["audio_encode"].append((audio + enc["encode"] - c["peak"],
                                    ROW.audio_encode.k / 11 * c["latents"]))
    return out


def test_the_keyframe_cells_cover_the_table_shapes():
    enc = {(tuple(c["shape"]), len(c["blocks"])) for c in KF["encode"][
        "cells"]}
    for size in ((768, 768), (1344, 768)):
        assert (size, 1) in enc and (size, 2) in enc
    # The table's prompt, 1,024 text rows and two keyframes, and longer.
    assert max(c["rows"] for c in KF["encode"]["cells"]) > 7000
    clips = {tuple(c["shape"]) for c in KF["video_encode"]}
    for w, h in ((768, 768), (1344, 768)):
        assert {(w, h, 1), (w, h, 22)} <= clips
    assert (1344, 768, 124) in clips
    assert [c["seconds"] for c in KF["audio_encode"]] == [1.0, 4.0, 15.0]
    assert KF["files"]["encoder"] == FILES["encoder"]
    assert (KF["files"]["vae"], KF["files"]["audio_vae"]) == (
        FILES["vae"], FILES["audio_vae"])


def test_the_servers_encode_window_is_the_longest_measured_clip():
    """The server takes a guide clip in windows of at most encode_pieces
    pieces of 17 frames, 17 p + 5 frames at the clip's end: the longest
    clip the video encode cells measure at 1344x768, so a window is priced
    inside them."""
    from gmlx.serve.image_steps import COMFY

    p = COMFY["h3"]["vae"]["encode_pieces"]
    longest = max(c["shape"][2] for c in KF["video_encode"]
                  if c["shape"][:2] == [1344, 768])
    assert 17 * p + 5 == longest == 124


@pytest.mark.parametrize("fit", ["tower", "vision_encode", "frame_encode",
                                 "clip_encode", "audio_encode"])
def test_each_keyframe_and_anchor_fit_covers_its_cells(fit):
    cells = _kf_rooms()[fit]
    rooms = [r for r, _ in cells]
    assert len(rooms) >= 3
    assert min(rooms) >= ROOM, [r / GiB for r in rooms]
    assert min(rooms) <= 0.35 * GiB
    assert max(r - slope for r, slope in cells) <= 1.5 * GiB


def test_the_text_load_of_a_keyframe_encode_is_the_t2v_load():
    """The encoder loads its text part as text to video does, and the
    attach then holds less: the load stays the encode's floor and the
    attach is priced over its reading."""
    from gmlx.image.memfit import LOAD_ROOM

    enc = KF["encode"]
    assert enc["load_peak"] == DOC["encode"]["load_peak"]
    assert enc["kept"] == DOC["encode"]["resident"]
    attach = (ROW.encoder_keep * KF["files"]["encoder"] + TOWER
              + LOAD_ROOM * KF["files"]["mmproj"])
    assert ROOM <= attach - enc["attach_peak"] <= 1.5 * GiB
    plan = _part(encoder=KF["files"]["encoder"], mmproj=KF["files"][
        "mmproj"], tower=TOWER, text=3083, blocks=(1008, 1008))
    assert plan.terms["encode"] == {"encoder": KF["files"]["encoder"],
                                    "load": ROW.encoder_load}
    assert plan.encoder_load == plan.phases["encode"] > attach


def test_an_encode_with_keyframes_is_its_load_its_attach_or_its_run():
    from gmlx.image.memfit import LOAD_ROOM

    kw = dict(encoder=20 * GiB, mmproj=GiB, tower=2 * GiB)
    t2v = _part(encoder=20 * GiB, text=1024)
    one = _part(text=2040, blocks=(1008,), **kw)
    assert one.terms["encode"] == t2v.terms["encode"]
    assert one.keyframes and not t2v.keyframes
    kept = ROW.encoder_keep * 20 * GiB
    long = _part(text=30_000, blocks=(1008, 1008), **kw)
    assert long.terms["encode"] == {
        "encoder": kept, "tower": 2 * GiB,
        "prompt": ROW.vision_encode.bytes(30_000)}
    assert long.encoder_load == one.encoder_load == int(
        20 * GiB + ROW.encoder_load)
    big = _part(text=100, blocks=(60,), **{**kw, "mmproj": 8 * GiB})
    assert big.terms["encode"] == {"encoder": kept, "tower": 2 * GiB,
                                   "attach": LOAD_ROOM * 8 * GiB}
    assert big.encoder_load == big.phases["encode"]
    # The tower runs one keyframe at a time, the larger one the peak.
    for blocks in ((1008, 60), (60, 1008)):
        assert _vision_run(100, blocks) == ROW.tower.bytes(1008)


def test_the_anchors_phase_is_the_larger_vae_with_its_largest_encode():
    px = 1344 * 768
    kw = dict(dit=10 * GiB, encoder=20 * GiB, vae=5 * GiB, audio_vae=GiB,
              rows=20_000, text=25, audio_latents=200, pixels=px,
              budget=None)
    assert "anchors" not in memfit.plan_memory("h3", **kw).phases
    clip = memfit.plan_memory("h3", anchor_frames=[1, 22, 5], **kw)
    assert list(clip.phases) == list(memfit.PHASES)
    assert clip.terms["anchors"] == {
        "vae": 5 * GiB,
        "encode": ROW.clip_encode.bytes(px) + ROW.clip_frame * 22 * px}
    one = memfit.plan_memory("h3", anchor_frames=[1], **kw)
    assert one.terms["anchors"] == {"vae": 5 * GiB,
                                    "encode": ROW.frame_encode.bytes(px)}
    sound = memfit.plan_memory("h3", anchor_audio=[40, 600], **kw)
    assert sound.terms["anchors"] == {
        "audio_vae": GiB, "encode": ROW.audio_encode.bytes(600)}
    # The VAEs run one at a time.
    both = memfit.plan_memory("h3", anchor_frames=[1], anchor_audio=[600],
                              **kw)
    assert both.terms["anchors"] == one.terms["anchors"]
    both = memfit.plan_memory("h3", anchor_frames=[1], anchor_audio=[600],
                              **{**kw, "audio_vae": 8 * GiB})
    assert both.terms["anchors"]["audio_vae"] == 8 * GiB
    # A preview decodes through TAEH3 and encodes through the video VAE.
    tiny = {**kw, "vae": 22 << 20, "tiny": True}
    with pytest.raises(ValueError, match="the video VAE its anchors "
                                         "encode through, as anchor_vae"):
        memfit.plan_memory("h3", anchor_frames=[1], **tiny)
    got = memfit.plan_memory("h3", anchor_frames=[1], anchor_vae=5 * GiB,
                             **tiny)
    assert got.terms["anchors"]["vae"] == 5 * GiB
    assert got.terms["decode"]["vae"] == 22 << 20
    assert "anchors" in memfit.plan_memory("h3", anchor_audio=[40],
                                           **tiny).phases


def test_the_rows_count_each_anchors_latents():
    """A 1344x768 latent frame packs 1008 rows, and an audio latent two,
    one a channel: docs/video.md's rows."""
    base, audio = memfit.clip_rows(hf.FAMILY, frames=124, width=1344,
                                   height=768, text=1024)
    for anchors, extra in [
            ((AnchorRows(0, 1, 0),), 1008),
            ((AnchorRows(0, 1, 0), AnchorRows(123, 1, 0)), 2016),
            ((AnchorRows(0, 2, hf.audio_frames(5)),), 2016 + 16),
            ((AnchorRows(0, 7, hf.audio_frames(22)),), 7056 + 74),
            ((AnchorRows(0, 0, hf.audio_frames(124)),), 414)]:
        assert memfit.clip_rows(hf.FAMILY, frames=124, width=1344,
                                height=768, text=1024, anchors=anchors) \
            == (base + extra, audio)
    assert memfit.anchor_frames(hf.FAMILY, [
        AnchorRows(0, 1, 0), AnchorRows(0, 2, 8), AnchorRows(9, 7, 37),
        AnchorRows(0, 0, 207)]) == [1, 5, 22]
    with pytest.raises(ValueError, match="3 latent frames are no "
                                         "MiniMax-H3 guide clip"):
        memfit.anchor_frames(hf.FAMILY, [AnchorRows(0, 3, 0)])


def test_a_prompt_with_keyframes_counts_comfyuis_rows():
    from test_video_keyframes import CASES, _Pieces

    for c in CASES.values():
        sizes = [tuple(s) for s in c["sizes"]]
        tok = _Pieces(c["pieces"])
        assert memfit.prompt_rows(hf.FAMILY, c["prompt"], tok, sizes) == \
            len(c["ids"])
        assert sum(memfit.keyframe_blocks(hf.FAMILY, sizes)) == \
            c["ids"].count(hf.IMAGE_PAD)


# The CLI's admission table in docs/video.md: each mode's smallest Mac
# at 1344x768 and 768x768, for h3 on its 8-step LoRA, 5 s unless _FRAMES
# names the mode, the prompt at 1,024 text rows plus each keyframe's,
# caches empty. mode: (keyframes, anchors).
_MODES = {
    "text to video": (0, ()),
    "first frame": (1, (AnchorRows(0, 1, 0),)),
    "last frame": (1, (AnchorRows(123, 1, 0),)),
    "both keyframes": (2, (AnchorRows(0, 1, 0), AnchorRows(123, 1, 0))),
    "image guide": (0, (AnchorRows(60, 1, 0),)),
    "5-frame guide clip with sound": (
        0, (AnchorRows(0, 2, hf.audio_frames(5)),)),
    "22-frame guide clip with sound": (
        0, (AnchorRows(0, 7, hf.audio_frames(22)),)),
    "guide audio, 5 s": (0, (AnchorRows(0, 0, hf.audio_frames(124)),)),
    "362-frame guide clip with sound, 15 s": (
        0, (AnchorRows(0, hf.latent_frames(362), hf.audio_frames(362)),)),
}
_FRAMES = {"362-frame guide clip with sound, 15 s": 362}
_TABLE = {"text to video": (32, 32), "first frame": (32, 32),
          "last frame": (32, 32), "both keyframes": (36, 32),
          "image guide": (32, 32), "5-frame guide clip with sound": (32, 32),
          "22-frame guide clip with sound": (36, 32),
          "guide audio, 5 s": (32, 32),
          "362-frame guide clip with sound, 15 s": (96, 64)}
_SMALLER = {32: 24, 36: 32, 64: 48, 96: 64}


def _mode(monkeypatch, mode, w, h, mac, frames=124):
    files = {"dit": FILES["dit"], "enc": FILES["encoder"],
             "vae": FILES["vae"], "audio": FILES["audio_vae"],
             "lora": FILES["lora"], "mmproj": KF["files"]["mmproj"]}
    monkeypatch.setattr(memfit, "file_bytes", lambda p: files[p])
    monkeypatch.setattr(memfit, "lora_dtype", lambda p: "BF16")
    monkeypatch.setattr(memfit, "tower_bytes", lambda p: TOWER)
    keyframes, anchors = _MODES[mode]
    blocks = memfit.keyframe_blocks(hf.FAMILY, [(w, h)] * keyframes)
    return memfit.plan_run(
        hf.FAMILY, dit="dit", encoder="enc", vae="vae", audio_vae="audio",
        loras=["lora"], frames=frames, width=w, height=h,
        text=1024 + sum(b + 8 for b in blocks), mmproj="mmproj",
        blocks=blocks, anchors=anchors,
        budget=capacity.ram_budget_bytes(mac * GiB))


@pytest.mark.parametrize("size", [(1344, 768), (768, 768)],
                         ids=["1344x768", "768x768"])
@pytest.mark.parametrize("mode", list(_MODES))
def test_each_mode_runs_on_the_mac_the_table_names(monkeypatch,
                                                   no_overcommit, mode,
                                                   size):
    mac = _TABLE[mode][size == (768, 768)]
    frames = _FRAMES.get(mode, 124)
    assert _mode(monkeypatch, mode, *size, mac, frames).fits
    msg = _refusal(_mode(monkeypatch, mode, *size, _SMALLER[mac], frames))
    if mac == 32:
        assert "The text encoder needs more memory than this Mac has" in msg
    else:
        assert "with the denoise over it. Try a smaller size or a shorter " \
               "clip." in msg


@pytest.mark.parametrize("mode", ["first frame",
                                  "5-frame guide clip with sound"])
def test_on_32_gb_the_default_clip_has_the_room_the_table_gives(
        monkeypatch, no_overcommit, mode):
    """A first frame or a 5-frame guide clip at 1344x768 leaves 0.27 GiB
    of a 32 GB Mac's budget, to 2 decimals."""
    plan = _mode(monkeypatch, mode, 1344, 768, 32)
    assert plan.largest == "denoise"
    assert f"{(plan.budget - plan.peak) / GiB:.2f}" == "0.27"


def test_on_32_gb_a_long_prompt_with_keyframes_names_them(monkeypatch,
                                                          no_overcommit):
    """Two 512x512 keyframes with a prompt of 15,000 text rows run the
    encode over a 32 GB Mac's budget, where the prompt alone runs."""
    budget = capacity.ram_budget_bytes(32 * GiB)
    text = 15_000
    rows, audio = memfit.clip_rows(hf.FAMILY, frames=22, width=512,
                                   height=512, text=text + 2 * 264)
    kw = dict(dit=FILES["dit"], encoder=FILES["encoder"], vae=FILES["vae"],
              audio_vae=FILES["audio_vae"], loras=[FILES["lora"]],
              lora_dtypes=["BF16"], rows=rows, audio_latents=audio,
              pixels=512 * 512, budget=budget)
    assert memfit.plan_memory("h3", text=text, **kw).fits
    msg = _refusal(memfit.plan_memory(
        "h3", text=text + 2 * 264, mmproj=KF["files"]["mmproj"],
        tower=TOWER, blocks=(256, 256), **kw))
    assert msg.startswith("this clip does not fit in memory: encode ")
    assert msg.endswith("with the encode over it. Try a shorter prompt or "
                        "a smaller size, or one keyframe. GMLX_OVERCOMMIT=1 "
                        "overrides.")


# Int8 linears: H3's 200 block linears at one int8 byte a weight and one
# float32 scale an output row in place of their Q4_K bytes, and the
# widest int8 input of a step, fc2's, beside it (design 3.5).
H3_INT8_DELTA = 8_441_753_600
# fc1, N 28672 and K 5376: the float32 weight, its codes and its scales.
H3_INT8_TRANSIENT = 5 * 28672 * 5376 + 4 * 28672


def test_the_int8_dit_counts_each_block_linear_from_the_header():
    """At the published file's header the 200 converted matrices hold
    8,441,753,600 bytes over their Q4_K bytes, and the largest conversion
    holds fc1 at float32 with its codes and scales beside the DiT. The
    token refiner's linears, though Q4_K, are not counted."""
    from video_fixtures import H3_HEADER, fixture_view

    view = fixture_view(H3_HEADER)
    mats = memfit._int8_matrices(hf.FAMILY, view)
    assert len(mats) == 4 * 50
    assert memfit.int8_delta(hf.FAMILY, view) == H3_INT8_DELTA
    assert memfit.int8_transient(hf.FAMILY, view) == H3_INT8_TRANSIENT
    # A prefixed file holds the same matrices.
    pre = fixture_view(H3_HEADER, prefix=hf.FAMILY.strip_prefixes[0])
    assert memfit.int8_delta(hf.FAMILY, pre) == H3_INT8_DELTA


def test_the_fasth3_file_converts_its_block_linears_and_not_its_gates():
    """The FastH3 file carries the wan label and a VSA gate in each block.
    Its 200 block linears convert and its gates do not, so it holds H3's
    delta."""
    from video_fixtures import fasth3_view

    view = fasth3_view()
    assert len(memfit._int8_matrices(hf.FAMILY, view)) == 4 * 50
    assert memfit.int8_delta(hf.FAMILY, view) == H3_INT8_DELTA


def test_the_dit_bytes_of_int8_linears_are_the_file_and_the_delta(
        monkeypatch):
    from gmlx.image import text_encoder
    from video_fixtures import H3_HEADER, fixture_view

    monkeypatch.setattr(memfit, "file_bytes", lambda p: FILES["dit"])
    monkeypatch.setattr(text_encoder, "_header_view",
                        lambda p: fixture_view(H3_HEADER))
    assert memfit.dit_bytes(hf.FAMILY, "dit.gguf") == FILES["dit"]
    assert memfit.dit_bytes(hf.FAMILY, "dit.gguf", "int8") == (
        FILES["dit"] + H3_INT8_DELTA)


def test_an_int8_linears_step_holds_fc2s_int8_input_beside_the_files():
    """fc2's int8 input of 14336 columns and its float32 scales, one each
    1024 columns: 14,392 bytes a packed row over the step on the file's
    linears, and no other term moves."""
    kw = dict(dit=12 << 30, encoder=20 << 30, vae=1 << 30,
              audio_vae=1 << 30, rows=37735, text=1024, audio_latents=150,
              pixels=1344 * 768, budget=100 << 30)
    file = memfit.plan_memory("h3", **kw)
    int8 = memfit.plan_memory("h3", linears="int8", **kw)
    step = int8.terms["denoise"].pop("step") - file.terms["denoise"].pop(
        "step")
    assert step == (14336 + 14336 // 1024 * 4) * 37735 == 14_392 * 37735
    assert int8.terms == file.terms
    both = memfit.plan_memory("h3", attention="int8", linears="int8", **kw)
    assert both.phases["denoise"] - file.phases["denoise"] == (
        (14_392 + 14_784) * 37735)


@pytest.mark.parametrize("variant, keep, fallback", [
    (None, "off or attention, which keep", "off or attention"),
    ("fl2va", "off or attention, which keep", "off or attention"),
    ("fasth3", "off, which keeps", "off")])
def test_a_denoise_over_the_budget_with_int8_linears_names_the_setting(
        no_overcommit, variant, keep, fallback):
    """The hint of a clip whose denoise does not fit at int8 linears
    names the int8 settings that keep the file's linears first, before a
    size, as the resize the CLI finds does: off or attention, and off
    alone on FastH3, whose sparse attention refuses int8 attention."""
    rows, audio = memfit.clip_rows(hf.FAMILY, frames=124, width=1344,
                                   height=768, text=25)
    plan = memfit.plan_memory(
        "h3", dit=FILES["dit"] + H3_INT8_DELTA, encoder=1, vae=1,
        audio_vae=1, rows=rows, text=25, audio_latents=audio,
        pixels=1344 * 768, variant=variant, linears="int8",
        budget=float(FILES["dit"]))
    assert plan.cause == "denoise"
    assert _refusal(plan).endswith(
        f"Run with int8 {keep} the file's linears, or try a smaller size or "
        "a shorter clip. GMLX_OVERCOMMIT=1 overrides.")
    with pytest.raises(memfit.VideoMemoryError) as e:
        memfit.check(plan, resize=lambda: "try 768x768")
    assert str(e.value).endswith(
        f"Run with int8 {fallback}, or try 768x768. GMLX_OVERCOMMIT=1 "
        "overrides.")


# docs/video.md's int8 table: the largest clip of h3 on its 8-step LoRA at
# int8 linears or all on each Mac, with the 1,024-row prompt of the
# keyframe table. Mac: (clips that fit, the next clips up that do not).
# On 32 GB the int8 DiT and its LoRA alone are over the budget, so even
# the smallest clip is refused. On 36 GB smaller clips fit.
_INT8_MACS = {
    32: ((), ((32, 32, hf.MIN_FRAMES),)),
    36: (((512, 288, 124),), ((1344, 768, 124), (768, 768, 124))),
    48: (((1344, 768, 124), (768, 768, 243)),
         ((1344, 768, 243), (768, 768, 362))),
    64: (((1344, 768, 362), (768, 768, 362)), ()),
}


@pytest.mark.parametrize("attention", ["dense", "int8"],
                         ids=["linears", "all"])
@pytest.mark.parametrize("mac", list(_INT8_MACS))
def test_int8_linears_make_the_clips_the_video_page_gives(
        monkeypatch, no_overcommit, mac, attention):
    from gmlx.image import text_encoder
    from video_fixtures import H3_HEADER, fixture_view

    monkeypatch.setattr(memfit, "file_bytes", lambda p: FILES[p])
    monkeypatch.setattr(memfit, "lora_dtype", lambda p: "BF16")
    monkeypatch.setattr(text_encoder, "_header_view",
                        lambda p: fixture_view(H3_HEADER))

    def plan(w, h, f):
        return memfit.plan_run(
            hf.FAMILY, dit="dit", encoder="encoder", vae="vae",
            audio_vae="audio_vae", loras=["lora"], frames=f, width=w,
            height=h, text=1024, attention=attention, linears="int8",
            budget=capacity.ram_budget_bytes(mac * GiB))

    fits, over = _INT8_MACS[mac]
    assert all(plan(*shape).fits for shape in fits)
    assert not any(plan(*shape).fits for shape in over)
