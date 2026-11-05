"""Check 6 of the video acceptance gate for MiniMax-H3: a whole text to
video run against ComfyUI's, with the 8-step Turbo LoRA and without, and
the LoRA runs with keyframes and guides.

scripts/video_oracle/check6_h3.py writes the fixtures: ComfyUI's
MiniMaxH3Model on the same GGUF, the LoRA through ComfyUI's own loader,
run in fp32 from seeded noise with every step's latents and velocity,
and its own bf16 velocity at each step from the fp32 latents before it.
Three gates:

- exact steps: gmlx in the oracle's arithmetic, each step from the
  oracle's fp32 latents on the oracle's features and sigma tables, its
  velocity within EXACT_TOL, or, for a step and stream that misses it,
  within the larger of EXACT_TOL and twice the floor the manifest records
  for it: the largest relative L2 between that step's velocity and the
  same step on the CPU stream or from latents and features nudged by one
  float32 step;
- exact run: the same arithmetic from the oracle's noise to the final
  latents, within the larger of EXACT_TOL and twice the floor the
  manifest records, the largest relative L2 between gmlx's exact run on
  the GPU and the same run on the CPU stream or from inputs nudged by one
  float32 step;
- shipping steps: gmlx's default path with its own encoder's features,
  its own schedule and the LoRA live, each step from the oracle's fp32
  latents, its velocity within twice the oracle's own bf16 gap at that
  step. On the LoRA case that bf16 pass merges the LoRA in float32 and
  rounds once (--lora-merge float32): ComfyUI-GGUF's own bf16 add keeps
  too little of the LoRA for the gap to tell a run without it.

The same run without the LoRA must miss both exact gates on the LoRA
case. Over eight steps a free run drifts from the fp32 run as a
different sample would, the oracle's own bf16 run too, so the default
path's free run is printed and gates nothing.

The anchor cases (ANCHOR_CASES) run each gate with ComfyUI's anchors:
its nodes encode them, and the exact gates and every shipping step but
the first frame case's take the oracle's latents and the noise ComfyUI
mixes into their rows. The keyframe cases take check 2's keyframe
prompts, the guides its text prompt. The first frame case's shipping
steps run gmlx's own path from the PNG, as the CLI reads it: the reader,
the stretch, the vision tower and the vision spans, the placement and the
video VAE encode, with the oracle's noise.

The runs need the DiT, the encoder, its mmproj, the video VAE and the
LoRA under KQUANT_TEST_GGUF_DIR and the fixtures under
GMLX_VIDEO_FIXTURES.
"""
from __future__ import annotations

import json
import os
import types
from collections import Counter
from pathlib import Path

import block_gates as bg
import h3_gates as hg
import mlx.core as mx
import numpy as np
import pytest
from video_fixtures import video_dits
from video_oracle_fixture import ORACLE, committed, fixture, fixture_present, fx

from gmlx.image import pacing
from gmlx.image.lora import install as lora_install
from gmlx.image.ops import kq_routes
from gmlx.video import anchors as va
from gmlx.video import api, encoder, images, layout, pipeline, schedule
from gmlx.video.families.h3 import family as hf

LORA = "check6_h3_lora_512x512x22"
BASE = "check6_h3_512x512x22"
CASES = (LORA, BASE)
STREAMS = ("video", "audio")
LORA_FILE = "minimax_h3_fl2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors"
VAE_FILE = "minimax_h3_video_vae_fp16.safetensors"
MMPROJ = "mmproj-BF16.gguf"
KEYFRAMES = "check2_h3_keyframes"
# Each anchor case: its anchors as (frame, video latent frames, audio
# latent frames), and the check 2 prompt of its features.
ANCHOR_CASES = {
    "first": ([(0, 1, 0)], "keyframe"),
    "last": ([(21, 1, 0)], "keyframe"),
    "both": ([(0, 1, 0), (21, 1, 0)], "keyframes"),
    "mid_audio": ([(10, 1, 20)], "dialogue"),
    "clip17": ([(17, 2, 0)], "dialogue"),
}
ANCHORED = tuple(f"check6_h3_lora_{c}_512x512x22" for c in ANCHOR_CASES)
ALL = (*CASES, *ANCHORED)
FIRST = ANCHORED[0]
NO_ANCHORS = ((), [], ())


def _floor(m: dict) -> dict:
    assert "exact_free_floor" in m["results"], (
        f"{m['name']} records no exact_free_floor: run "
        f"test_check6_exact_gates with {hg.RECORD}=1")
    return m["results"]["exact_free_floor"]


def _manifest(name: str) -> dict:
    if not (ORACLE / f"{name}.manifest.json").is_file():
        pytest.skip(f"{name}.manifest.json is not committed: run "
                    "scripts/video_oracle/check6_h3.py and commit the "
                    "manifest it writes")
    return committed(name)


def _assert_step_floors(m: dict) -> None:
    """The step floors a manifest records, if any: each for a step and
    stream whose velocity missed EXACT_TOL when it was recorded, with its
    gate the larger of EXACT_TOL and twice the floor, and no gate without
    a floor."""
    rec = m["results"].get("exact_step_floor")
    tols = m["tolerances"].get("exact_steps")
    assert (rec is None) == (tols is None), "a step floor without its gate"
    if rec is None:
        return
    steps = rec["steps"]
    assert steps and set(steps) == set(tols)
    assert set(steps) <= {str(i + 1) for i in range(m["inputs"]["steps"])}
    for k, v in steps.items():
        assert v and set(v) == set(tols[k]) and set(v) <= set(STREAMS), k
        for s, f in v.items():
            assert f["base"] > bg.EXACT_TOL, (k, s, "a floor for a step "
                                              "under EXACT_TOL")
            assert f["rel_l2"] > 0, (k, s)
            assert tols[k][s] == {
                "rel_l2": max(bg.EXACT_TOL, 2 * f["rel_l2"])}, (k, s)


@pytest.mark.parametrize("name", ALL)
def test_the_manifests_follow_the_rule(name):
    """Each step's gate is twice the oracle's bf16 gap at that step, each
    exact run gate follows its floor, the run is the recipe's, and every
    case ran on one DiT. An anchor case ran its anchors on check 2's
    features of its prompt."""
    m = _manifest(name)
    inp = m["inputs"]
    case = inp.get("case")
    prompt = ANCHOR_CASES[case][1] if case else "dialogue"
    assert inp["forwards"] == inp["steps"] == 8
    assert inp["shape"] == [512, 512, 22]
    assert inp["prompt"] == prompt
    assert inp["shifts"]["audio"] == hf.SHIFTS["audio"]
    assert inp["modes"][:1] == ["fp32"] and "bf16_forced" in inp["modes"]
    gaps = m["results"]["step_bf16_vs_fp32"]
    assert len(gaps) == len(m["tolerances"]["steps"]) == inp["steps"]
    for g, tol in zip(gaps, m["tolerances"]["steps"]):
        assert tol == {s: {"rel_l2": 2 * g[s]["rel_l2"]} for s in STREAMS}
    floor = _floor(m)
    assert m["tolerances"]["exact_free"] == {
        s: {"rel_l2": max(bg.EXACT_TOL, 2 * floor[s]["rel_l2"])}
        for s in STREAMS}
    _assert_step_floors(m)
    assert m["files"]["dit"] == _manifest(LORA)["files"]["dit"]
    keyed = prompt.startswith("keyframe")
    tap = committed(KEYFRAMES if keyed else "check2_h3")["tensors"][
        f"tap_{prompt}"]
    assert m["tensors"]["features"]["sha256"] == tap["sha256"]
    if case is not None:
        assert [(a["frame"], a["video_t"], a["audio_t"])
                for a in inp["anchors"]] == ANCHOR_CASES[case][0]
        assert inp["anchor_seed"] == inp["seed"]
        assert len(inp["spans"]) == (len(inp["anchors"]) if keyed else 0)
        assert m["files"]["vae"]["name"] == VAE_FILE
        assert ("first_frame" in m["files"]) == (case == "first")
        assert ("guide_audio" in m["files"]) == (case == "mid_audio")
    if name != BASE:
        assert inp["shifts"]["video"] == hf.TURBO_768_SHIFT
        assert inp["lora"] == m["files"]["lora"]["name"] == LORA_FILE
        assert inp["lora_pairs"] == 208 and inp["lora_strength"] == 1.0
        assert inp["lora_merge"] == "float32"
    else:
        assert inp["shifts"]["video"] == hf.SHIFTS["video"]
        assert inp["lora"] is None and "lora" not in m["files"]
        assert inp.get("lora_merge") is None


FLOOR8 = 1.5e-04


def _floored() -> dict:
    """A manifest whose step 8 video velocity missed EXACT_TOL and records
    its floor."""
    return {"name": "floored", "inputs": {"steps": 8},
            "results": {"exact_step_floor": {
                "steps": {"8": {"video": {"rel_l2": FLOOR8,
                                          "base": 3e-04}}},
                "measured": "test"}},
            "tolerances": {"exact_steps": {
                "8": {"video": {"rel_l2": 2 * FLOOR8}}}}}


def _steps_with(video8: float = 3e-04, audio8: float = 1e-05) -> list:
    out = [{"video": 2e-05, "audio": 1e-05} for _ in range(8)]
    out[7] = {"video": video8, "audio": audio8}
    return out


def _gate_widened(m):
    m["tolerances"]["exact_steps"]["8"]["video"]["rel_l2"] = 3 * FLOOR8


def _floor_under_tol(m):
    m["results"]["exact_step_floor"]["steps"]["8"]["video"]["base"] = 4e-05


def _gate_alone(m):
    del m["results"]["exact_step_floor"]


def _floor_alone(m):
    del m["tolerances"]["exact_steps"]


def _past_the_last(m):
    for d in (m["results"]["exact_step_floor"]["steps"],
              m["tolerances"]["exact_steps"]):
        d["9"] = d.pop("8")


def _stream_unfloored(m):
    m["tolerances"]["exact_steps"]["8"]["audio"] = {"rel_l2": bg.EXACT_TOL}


@pytest.mark.parametrize("wrong", [_gate_widened, _floor_under_tol,
                                   _gate_alone, _floor_alone, _past_the_last,
                                   _stream_unfloored])
def test_a_step_floor_follows_its_rule(wrong):
    """A recorded step floor is for a step and stream that missed
    EXACT_TOL, inside the run, and its gate is the larger of EXACT_TOL and
    twice it; a manifest with none passes."""
    _assert_step_floors({"inputs": {"steps": 8}, "results": {},
                         "tolerances": {}})
    m = _floored()
    _assert_step_floors(m)
    wrong(m)
    with pytest.raises(AssertionError):
        _assert_step_floors(m)


def test_a_step_over_exact_tol_takes_its_floor_gate():
    """Only the steps and streams that miss EXACT_TOL measure a floor; each
    takes its recorded gate, the rest keep EXACT_TOL."""
    asked = []

    def measure(miss):
        asked.append(miss)
        return {"8": {"video": 1.1 * FLOOR8}}

    gates = _step_gates(_floored(), _steps_with(), measure)
    assert asked == [{"8": ["video"]}]
    assert gates[7] == {"video": 2 * FLOOR8, "audio": bg.EXACT_TOL}
    assert all(g == {s: bg.EXACT_TOL for s in STREAMS} for g in gates[:7])


def test_steps_under_exact_tol_measure_nothing():
    def measure(miss):
        raise AssertionError(f"measured {miss}")

    gates = _step_gates({"name": "bare", "results": {}, "tolerances": {}},
                        _steps_with(video8=4e-05), measure)
    assert all(g == {s: bg.EXACT_TOL for s in STREAMS} for g in gates)


@pytest.mark.parametrize("video8,audio8", [(3e-04, 1e-05), (3e-04, 6e-05)])
def test_a_miss_without_a_recorded_floor_fails_loud(video8, audio8):
    """A step or stream over EXACT_TOL with no floor on record fails with
    the RECORD hint before anything is measured."""
    m = ({"name": "bare", "results": {}, "tolerances": {}}
         if audio8 < bg.EXACT_TOL else _floored())
    with pytest.raises(AssertionError, match=hg.RECORD):
        _step_gates(m, _steps_with(video8, audio8),
                    lambda miss: pytest.fail(f"measured {miss}"))


@pytest.mark.parametrize("scale", [1.3, 1 / 1.3])
def test_a_step_floor_off_its_record_fails_loud(scale):
    with pytest.raises(AssertionError, match="within 1.25x"):
        _step_gates(_floored(), _steps_with(),
                    lambda miss: {"8": {"video": scale * FLOOR8}})


@pytest.mark.parametrize("name", ALL)
def test_the_oracle_ran_gmlx_schedule_and_shapes(name):
    """The sigma tables gmlx samples on, and the latent shapes of the
    clip. Video sigmas are the oracle's bit for bit. The oracle takes each
    audio sigma by the time shift of the video one, where gmlx shifts the
    grid as check 4 pins, so audio sigmas differ by up to 2 float32 ulps;
    the exact gates run the oracle's tables."""
    arrays, m = fixture(_manifest(name)["name"])
    inp = m["inputs"]
    sig = schedule.stream_sigmas(inp["steps"], inp["shifts"])
    np.testing.assert_array_equal(np.asarray(sig["video"], np.float32),
                                  arrays["sigmas_video"])
    np.testing.assert_array_max_ulp(np.asarray(sig["audio"], np.float32),
                                    arrays["sigmas_audio"], maxulp=2)
    w, h, f = inp["shape"]
    vs, aus = hf.latent_shapes(f, w, h)
    assert arrays["noise_video"].shape == vs == tuple(inp["video_shape"])
    assert arrays["noise_audio"].shape == aus == tuple(inp["audio_shape"])


def _find(root: Path, want: dict) -> str:
    for p in sorted(root.rglob(want["name"])):
        if p.is_file() and os.path.getsize(p) == want["size"]:
            if fx.file_identity(str(p))["sha256"] != want["sha256"]:
                pytest.fail(f"{p} is not the file the oracle read")
            return str(p)
    pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")


def _load(gguf_dir: Path, m: dict) -> pipeline.VideoPipeline:
    want = m["files"]["dit"]
    for p in video_dits(gguf_dir, "h3"):
        if Path(p).name == want["name"] and os.path.getsize(p) == want["size"]:
            break
    else:
        pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")
    if fx.file_identity(p)["sha256"] != want["sha256"]:
        pytest.fail(f"{p} is not the file the oracle read")
    return pipeline.load(p, log=lambda *_: None)


def _table(gguf_dir: Path, pipe, m: dict) -> str | None:
    if m["inputs"]["lora"] is None:
        return None
    path = _find(gguf_dir, m["files"]["lora"])
    return pipeline.adapter_table(pipe, [(path,
                                          m["inputs"]["lora_strength"])])


def _rel(got, want) -> float:
    return fx.rel_l2(np.asarray(got, np.float64), want)


def _clip(m: dict) -> dict:
    inp = m["inputs"]
    w, h, f = inp["shape"]
    return dict(frames=f, width=w, height=h, steps=inp["steps"],
                shifts=inp["shifts"])


def _oracle_sigmas(arrays: dict) -> dict:
    return {s: arrays[f"sigmas_{s}"] for s in STREAMS}


def _layout(pipe, feats: mx.array, m: dict, anc: tuple = NO_ANCHORS):
    clip = _clip(m)
    fam = pipe.video_family
    vs, aus = fam.latent_shapes(clip["frames"], clip["width"],
                                clip["height"])
    return layout.pack(fam.timeline, int(feats.shape[1]), vs[2], vs[3],
                       vs[4], aus[3], anchors=va.layout_rows(anc[0]),
                       spans=anc[2])


def _step_input(arrays: dict, i: int) -> tuple:
    """The oracle's fp32 latents before step ``i`` (0-based)."""
    src = "noise_{}" if i == 0 else f"{{}}_step{i}"
    return tuple(arrays[src.format(s)] for s in STREAMS)


def _velocities(pipe, feats: mx.array, arrays: dict, m: dict,
                table: str | None, sig: dict, anc: tuple = NO_ANCHORS,
                steps=None, seed: int | None = None) -> dict:
    """Each step's velocity (video, audio) by 0-based step, for ``steps``
    (all), from the oracle's fp32 latents before the step at the sigmas
    ``sig``, with the placed anchors, their noise and the vision spans of
    ``anc``. With ``seed`` each step's latents and the features are nudged
    one float32 step as h3_gates.nudged nudges a check 3 step's."""
    n = _clip(m)["steps"]
    lay = _layout(pipe, feats, m, anc)
    model = pipe.dit
    cv, ca = (model.cond_rows(anc[0], m["inputs"]["seed"], noise=anc[1])
              if anc[0] else ((), ()))
    out = {}
    prev = lora_install.activate(pipe, pipe.default_table if table is None
                                 else table)
    try:
        text = None if seed is not None else model.caption(feats)
        with pacing.paced(lambda a: mx.eval(*a)):
            for i in range(n) if steps is None else steps:
                xv, xa = _step_input(arrays, i)
                f = feats
                if seed is not None:
                    x = hg.nudged({"video": xv, "audio": xa,
                                   "features": np.array(feats)}, seed)
                    xv, xa, f = x["video"], x["audio"], mx.array(
                        x["features"])
                t = model.caption(f) if text is None else text
                sv, sa = sig["video"][i], sig["audio"][i]
                plan = model.plan(lay, api.stream_time(sv),
                                  api.stream_time(sa))
                v = api.denoise_step(model, mx.array(xv), mx.array(xa),
                                     {"video": sv, "audio": sa}, t, lay,
                                     cond_video=cv, cond_audio=ca, plan=plan)
                mx.eval(*v)
                out[i] = tuple(np.array(g) for g in v)
    finally:
        lora_install.restore(pipe, prev)
    return out


def _steps(pipe, feats: mx.array, arrays: dict, m: dict,
           table: str | None, sig: dict, anc: tuple = NO_ANCHORS) -> tuple:
    """Relative L2 of each step's velocity, video and audio, against the
    oracle's (``_velocities``), and the velocities."""
    vel = _velocities(pipe, feats, arrays, m, table, sig, anc)
    return [{s: _rel(g, arrays[f"velocity_{s}_step{i + 1}"])
             for s, g in zip(STREAMS, vel[i])} for i in sorted(vel)], vel


def _misses(steps: list) -> dict:
    """The streams of each step, by 1-based step, whose velocity misses
    EXACT_TOL."""
    out = {}
    for i, r in enumerate(steps):
        over = [s for s in STREAMS if r[s] > bg.EXACT_TOL]
        if over:
            out[str(i + 1)] = over
    return out


def _step_floor(pipe, feats, arrays, m, table, sig, anc, vel,
                miss: dict) -> dict:
    """The largest relative L2 of each step and stream of ``miss`` between
    its velocity ``vel`` and the same step on the CPU stream and from
    latents and features nudged one float32 step (NUDGE_SEEDS)."""
    at = sorted(int(k) - 1 for k in miss)
    others = [_velocities(pipe, feats, arrays, m, table, sig, anc, at, seed)
              for seed in hg.NUDGE_SEEDS]
    with mx.stream(mx.cpu):
        others.append(_velocities(pipe, feats, arrays, m, table, sig, anc,
                                  at))
    return {k: {s: max(_rel(o[int(k) - 1][STREAMS.index(s)],
                            vel[int(k) - 1][STREAMS.index(s)])
                       for o in others) for s in streams}
            for k, streams in miss.items()}


def _step_gates(m: dict, steps: list, measure) -> list:
    """Each step's gate by stream: EXACT_TOL, or for a step and stream
    whose velocity misses it, the larger of EXACT_TOL and twice the floor
    the manifest records, after ``measure(miss)`` finds the floor within
    1.25x of it now."""
    gates = [{s: bg.EXACT_TOL for s in STREAMS} for _ in steps]
    miss = _misses(steps)
    if not miss:
        return gates
    rec = m["results"].get("exact_step_floor", {}).get("steps", {})
    tols = m["tolerances"].get("exact_steps", {})
    hint = (f"on another machine, or after an mlx, mlx-kquant or macOS "
            f"update, run with {hg.RECORD}=1")
    for k, streams in miss.items():
        for s in streams:
            assert s in rec.get(k, {}), (
                f"step {k} {s} misses EXACT_TOL at {steps[int(k) - 1][s]:.3e} "
                f"and {m['name']} records no floor for it; {hint}")
    now = measure(miss)
    for k, streams in miss.items():
        for s in streams:
            r = rec[k][s]["rel_l2"]
            assert r / 1.25 <= now[k][s] <= 1.25 * r, (
                f"step {k} {s} floor {now[k][s]:.3e} is not within 1.25x of "
                f"the recorded {r:.3e}; {hint}")
            gates[int(k) - 1][s] = tols[k][s]["rel_l2"]
    return gates


def _run(pipe, feats: mx.array, start: tuple, m: dict,
         table: str | None, sig: dict | None = None,
         anc: tuple = NO_ANCHORS) -> dict:
    xv, xa = pipeline.generate(pipe, feats, start=tuple(map(mx.array, start)),
                               table=table, sigmas=sig, anchors=anc[0],
                               spans=anc[2], anchor_noise=anc[1] or None,
                               **_clip(m))
    return {"video": np.array(xv), "audio": np.array(xa)}


def _start(arrays: dict) -> tuple:
    return arrays["noise_video"], arrays["noise_audio"]


def _nudged(arrays: dict, seed: int) -> tuple:
    """The start noise and the features, each element moved one float32
    step up, down or not at all, at random."""
    rng = np.random.default_rng(seed)

    def nudge(a):
        a = np.asarray(a, np.float32)
        step = rng.integers(-1, 2, a.shape)
        return np.where(step == 0, a, np.nextafter(
            a, np.where(step > 0, np.inf, -np.inf).astype(np.float32)))

    return (nudge(arrays["noise_video"]), nudge(arrays["noise_audio"])), \
        mx.array(nudge(arrays["features"]))[None]


def _free_floor(pipe, feats, arrays, m, table, got, anc) -> dict:
    """The largest relative L2 of the exact run on the GPU against the
    same run on the CPU stream and from nudged inputs."""
    sig = _oracle_sigmas(arrays)
    others = []
    for seed in hg.NUDGE_SEEDS:
        start, f = _nudged(arrays, seed)
        others.append(_run(pipe, f, start, m, table, sig, anc))
    with mx.stream(mx.cpu):
        others.append(_run(pipe, feats, _start(arrays), m, table, sig, anc))
    return {s: max(_rel(o[s], got[s]) for o in others) for s in STREAMS}


def _record(m: dict, floor: dict, step_floor: dict, steps: list,
            paths) -> dict:
    """Write the run floor measured now, the step floors of the steps and
    streams that miss EXACT_TOL (none, the key dropped), and the gates
    they give, into each manifest at ``paths``. Returns the new
    manifest."""
    files = [m["files"][k] for k in ("dit", "lora") if k in m["files"]]
    named = " with ".join(f"{f['name']} {f['sha256'][:8]}" for f in files)
    how = (f"the largest relative L2 between gmlx's exact {{}} on the GPU "
           f"and the same {{}} on the CPU stream or from inputs nudged by "
           f"one float32 step ({len(hg.NUDGE_SEEDS)} seeds); "
           f"tests/video/test_video_check6_h3.py on {named}, "
           f"{mx.device_info()['device_name']}")
    res = {"exact_free_floor": {s: {"rel_l2": floor[s],
                                    "measured": how.format("run", "run")}
                                for s in STREAMS}}
    tols = {"exact_free": {s: {"rel_l2": max(bg.EXACT_TOL, 2 * floor[s])}
                           for s in STREAMS}}
    if step_floor:
        res["exact_step_floor"] = {
            "steps": {k: {s: {"rel_l2": f, "base": steps[int(k) - 1][s]}
                          for s, f in v.items()}
                      for k, v in step_floor.items()},
            "measured": how.format("step", "step") + (
                ", for each step and stream whose velocity misses "
                "EXACT_TOL")}
        tols["exact_steps"] = {k: {s: {"rel_l2": max(bg.EXACT_TOL, 2 * f)}
                                   for s, f in v.items()}
                               for k, v in step_floor.items()}
    for path in paths:
        doc = json.loads(path.read_text())
        assert doc["npz"] == m["npz"], path
        doc["results"].pop("exact_step_floor", None)
        doc["tolerances"].pop("exact_steps", None)
        doc["results"].update(res)
        doc["tolerances"].update(tols)
        tmp = path.with_name(path.name + ".part")
        tmp.write_text(json.dumps(doc, indent=1) + "\n")
        tmp.replace(path)
    results = {k: v for k, v in m["results"].items()
               if k != "exact_step_floor"}
    tolerances = {k: v for k, v in m["tolerances"].items()
                  if k != "exact_steps"}
    return {**m, "results": {**results, **res},
            "tolerances": {**tolerances, **tols}}


def _line(label: str, steps: list, run: dict, tols: dict) -> str:
    worst = {s: max(r[s] for r in steps) for s in STREAMS}
    return (f"{label}: steps within video {worst['video']:.2e}, audio "
            f"{worst['audio']:.2e}, gate {bg.EXACT_TOL:.0e}; run video "
            f"{run['video']:.2e} of {tols['video']['rel_l2']:.2e}, audio "
            f"{run['audio']:.2e} of {tols['audio']['rel_l2']:.2e}")


@pytest.mark.integration
@pytest.mark.parametrize("name", ALL)
def test_check6_exact_gates(gguf_dir, request, name):
    """Each step and the whole run in the oracle's arithmetic within the
    exact gates, with the oracle's anchors on an anchor case, and, on the
    LoRA case, the same run without the LoRA outside both."""
    fixture_present(_manifest(name)["name"])
    label = f"h3 check6 exact {name}"
    if not bg.exact_process():
        return bg.rerun_exact(request, label)
    arrays, m = fixture(_manifest(name)["name"])
    record = os.environ.get(hg.RECORD) == "1"
    if not record:
        _floor(m)
    pipe = _load(gguf_dir, m)
    table = _table(gguf_dir, pipe, m)
    feats = mx.array(arrays["features"])[None]
    sig = _oracle_sigmas(arrays)
    anc = hg.anchored(arrays)
    pipe.dit.act_dtype = mx.float32
    with hg.exact_floats(pipe.dit), bg.exact_products(pipe.dit) as used:
        steps, vel = _steps(pipe, feats, arrays, m, table, sig, anc)
        got = _run(pipe, feats, _start(arrays), m, table, sig, anc)
        floor = _free_floor(pipe, feats, arrays, m, table, got, anc)

        def measure(miss):
            return _step_floor(pipe, feats, arrays, m, table, sig, anc,
                               vel, miss)

        miss = _misses(steps)
        step_floor = measure(miss) if record and miss else {}
        if record:
            m = _record(m, floor, step_floor, steps,
                        (ORACLE / f"{name}.manifest.json",
                         fx.fixture_root() / f"{name}.manifest.json"))
        gates = _step_gates(m, steps, (lambda _: step_floor) if record
                            else measure)
        bare = None
        if name == LORA:
            bare = (_steps(pipe, feats, arrays, m, None, sig)[0],
                    _run(pipe, feats, _start(arrays), m, None, sig))
    bg.assert_all_dense(pipe.dit, used)
    rec = _floor(m)
    tols = m["tolerances"]["exact_free"]
    run = {s: _rel(got[s], arrays[s]) for s in STREAMS}
    for s in STREAMS:
        print(f"{label}: {s} floor {floor[s]:.2e}, recorded "
              f"{rec[s]['rel_l2']:.2e}")
    print(_line(label, steps, run, tols))
    for i, (r, g) in enumerate(zip(steps, gates)):
        for s in STREAMS:
            if g[s] != bg.EXACT_TOL:
                print(f"{label}: step {i + 1} {s} {r[s]:.2e} of its floor "
                      f"gate {g[s]:.2e}, {r[s] / g[s]:.2f} of it")
    if bare is not None:
        b_steps, b_got = bare
        b_run = {s: _rel(b_got[s], arrays[s]) for s in STREAMS}
        print(_line(f"{label} without the LoRA", b_steps, b_run, tols))
    for s in STREAMS:
        assert rec[s]["rel_l2"] / 1.25 <= floor[s] <= \
            1.25 * rec[s]["rel_l2"], (
            f"{s} floor {floor[s]:.3e} is not within 1.25x of the recorded "
            f"{rec[s]['rel_l2']:.3e}; on another machine, or after an mlx, "
            f"mlx-kquant or macOS update, run with {hg.RECORD}=1")
    over = [(i + 1, s, r[s], g[s]) for i, (r, g) in enumerate(zip(steps, gates))
            for s in STREAMS if not r[s] <= g[s]]
    assert not over, over
    assert all(run[s] <= tols[s]["rel_l2"] for s in STREAMS), run
    if bare is not None:
        assert all(r[s] > g[s] for r, g in zip(b_steps, gates)
                   for s in STREAMS)
        assert all(b_run[s] > tols[s]["rel_l2"] for s in STREAMS), b_run


def _census(pipe, table, lay, steps: int, ran: list) -> tuple:
    """The kernel calls the shipping steps made, the calls the routes in
    force must make, and the blocks whose fc1 takes the GLU kernel (None
    when that route is out of force). The q and k kernel runs in every
    block of every step, and the GLU kernel in every block whose fc1 has
    no live LoRA delta, from its row floor on a GPU with tensor ops. The
    refiner's rows are below the floor."""
    on = kq_routes.active()
    blocks = pipe.dit.blocks
    want: Counter = Counter()
    plain = None
    if "qk" in on:
        want["qk"] = steps * len(blocks)
    if ("glu" in on and kq_routes._nax()
            and lay.rows >= kq_routes._GLU_MIN_ROWS):
        prev = lora_install.activate(pipe, pipe.default_table
                                     if table is None else table)
        try:
            plain = sum(kq_routes.kquant_leaf(b.mlp.fc1, mx.bfloat16)
                        is not None for b in blocks)
        finally:
            lora_install.restore(pipe, prev)
        if plain:
            want["glu"] = steps * plain
    return Counter(ran), want, plain


def _fixture_file(want: dict) -> Path:
    """A file the oracle wrote beside its fixtures, by name, size and
    sha256."""
    p = fx.fixture_root() / want["name"]
    if not p.is_file() or os.path.getsize(p) != want["size"]:
        pytest.skip(f"{want['name']} is not under {fx.ENV}")
    if fx.file_identity(str(p))["sha256"] != want["sha256"]:
        pytest.fail(f"{p} is not the file the oracle wrote")
    return p


def _first_frame(m: dict) -> tuple:
    """The first frame case's PNG through the CLI's anchor reader: the
    keyframes at the canvas and the anchors it will encode."""
    from gmlx.commands import video as cli

    png = _fixture_file(m["files"]["first_frame"])
    w, h, f = m["inputs"]["shape"]
    flags = types.SimpleNamespace(first_frame=str(png), last_frame=None,
                                  guides=None, guide_no_audio=False)
    return cli._read_anchors(flags, hf.FAMILY,
                             {"frames": f, "width": w, "height": h})


def test_the_first_frame_reads_and_lands_as_comfyuis():
    """The CLI reads the PNG to LoadImage's pixels, stretches it to the
    canvas image MiniMaxH3ImageToVideo encoded and places it at frame 0
    as one latent frame without audio."""
    _manifest(FIRST)
    arrays, m = fixture(FIRST)
    png = _fixture_file(m["files"]["first_frame"])
    np.testing.assert_array_equal(images.read(str(png)),
                                  arrays["first_frame_read"])
    keyframes, todo = _first_frame(m)
    assert len(keyframes) == 1
    np.testing.assert_array_equal(keyframes[0], arrays["first_frame_canvas"])
    assert [(t.placed.frame, t.placed.video_frames, t.placed.audio_t)
            for t in todo] == [(int(arrays["anchor.0.frame"]), 1, None)]


def _keyframe_images(arrays: dict, key: str) -> list:
    n = sum(1 for k in arrays if k.startswith(f"image_{key}_"))
    return [arrays[f"image_{key}_{i}"] for i in range(n)]


@pytest.fixture(scope="module")
def cases(gguf_dir):
    """Each case's arrays and manifest, gmlx's features of its prompt and
    the anchors its shipping steps take: the oracle's latents and noise,
    but on the first frame case gmlx's own encode from the PNG. The
    encoder the check 2 oracle read loads for the text prompts, then with
    its mmproj for the keyframe prompts, then the video VAE for the
    first frame, each released before the next and before the DiT."""
    from gmlx.commands import video as cli

    loaded = {}
    for name in ALL:
        fixture_present(_manifest(name)["name"])
        loaded[name] = fixture(_manifest(name)["name"])
    c2 = committed("check2_h3")
    prompts = c2["inputs"]["prompts"]
    out = {}
    handle = encoder.load_encoder(hf.FAMILY, _find(gguf_dir,
                                                   c2["files"]["encoder"]),
                                  log=lambda *_: None)
    try:
        for name, (arrays, m) in loaded.items():
            if m["inputs"]["prompt"] not in prompts:
                continue
            prompt = prompts[m["inputs"]["prompt"]]
            assert handle.token_ids(prompt) == arrays["ids"].tolist()
            feats = np.array(handle.encode(prompt).features, np.float32)
            out[name] = (arrays, m, feats, hg.anchored(arrays))
    finally:
        handle.release()
    keyed = [n for n in loaded if n not in out]
    kf_arrays, kf_m = fixture(KEYFRAMES)
    first = _first_frame(loaded[FIRST][1])
    handle = encoder.load_encoder(
        hf.FAMILY, _find(gguf_dir, kf_m["files"]["encoder"]),
        mmproj=_find(gguf_dir, kf_m["files"]["mmproj"]), log=lambda *_: None)
    try:
        for name in keyed:
            arrays, m = loaded[name]
            key = m["inputs"]["prompt"]
            ims = first[0] if name == FIRST else _keyframe_images(kf_arrays,
                                                                  key)
            feats, spans = hf.FAMILY.keyframe_encode(
                handle, kf_m["inputs"]["prompts"][key], ims)
            anchors, noise, want = hg.anchored(arrays)
            assert tuple(map(tuple, spans)) == want, (name, spans, want)
            out[name] = (arrays, m, np.array(feats[0], np.float32),
                         (anchors, noise, want))
    finally:
        handle.release()
    arrays, m, feats, (_, noise, spans) = out[FIRST]
    w, h, f = m["inputs"]["shape"]
    placed = cli._encode_anchors(first[1], {"frames": f, "width": w,
                                            "height": h},
                                 _find(gguf_dir, m["files"]["vae"]), None)
    print(f"[check6] first frame: gmlx's float16 VAE encode of the PNG "
          f"against the oracle's float32, rel L2 "
          f"{_rel(placed[0].video, arrays['anchor.0.video']):.2e}; gmlx's "
          f"features against the oracle's "
          f"{_rel(feats, arrays['features']):.2e} (not gated)")
    out[FIRST] = (arrays, m, feats, (placed, noise, spans))
    return out


@pytest.fixture(scope="module")
def pipe(gguf_dir, cases):
    return _load(gguf_dir, cases[LORA][1])


@pytest.mark.integration
@pytest.mark.parametrize("name", ALL)
def test_check6_shipping_steps(gguf_dir, pipe, cases, name):
    """Each step of the default path, on gmlx's own encoder features and
    the LoRA live, within twice the oracle's own bf16 gap at that step
    (the LoRA merged in float32 in that pass), with the kernels in
    force, and on an anchor case its anchors. The LoRA runs live on stock
    ops, since every H3 linear is wider than the fused LoRA route takes.
    The free run is printed."""
    assert not bg.exact_process()
    assert pipe.dit.act_dtype == mx.bfloat16
    arrays, m, feats, anc = cases[name]
    table = _table(gguf_dir, pipe, m)
    f = mx.array(feats)[None]
    sig = schedule.stream_sigmas(m["inputs"]["steps"], m["inputs"]["shifts"])
    ran: list = []
    with bg.kernel_calls(ran.append):
        steps, _ = _steps(pipe, f, arrays, m, table, sig, anc)
    calls, want, plain = _census(pipe, table, _layout(pipe, f, m, anc),
                                 len(steps), ran)
    print(f"[check6] {name} kernel calls {dict(calls)}, routes in force "
          f"{sorted(kq_routes.active())}")
    tols = m["tolerances"]["steps"]
    for i, (r, t) in enumerate(zip(steps, tols)):
        print(f"[check6] {name} step {i + 1}: " + "; ".join(
            f"{s} {r[s]:.3e}, {r[s] / t[s]['rel_l2']:.2f} of its gate"
            for s in STREAMS))
    got = _run(pipe, f, _start(arrays), m, table, anc=anc)
    print(f"[check6] {name} free run, not gated: " + "; ".join(
        f"{s} rel L2 {_rel(got[s], arrays[s]):.3e}" for s in STREAMS))
    # The 8-step LoRA has a pair on every fc1.
    if plain is not None:
        assert plain == (0 if table is not None
                         else len(pipe.dit.blocks)), plain
    assert calls == want
    over = [(i + 1, s, r[s], t[s]["rel_l2"])
            for i, (r, t) in enumerate(zip(steps, tols)) for s in STREAMS
            if not r[s] <= t[s]["rel_l2"]]
    assert not over, over
