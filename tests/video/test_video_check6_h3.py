"""Check 6 of the video acceptance gate for MiniMax-H3: a whole text to
video run against ComfyUI's, with the 8-step Turbo LoRA and without.

scripts/video_oracle/check6_h3.py writes the fixtures: ComfyUI's
MiniMaxH3Model on the same GGUF, the LoRA through ComfyUI's own loader,
run in fp32 from seeded noise with every step's latents and velocity,
and its own bf16 velocity at each step from the fp32 latents before it.
Three gates:

- exact steps: gmlx in the oracle's arithmetic, each step from the
  oracle's fp32 latents on the oracle's features and sigma tables, its
  velocity within EXACT_TOL;
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

The runs need the DiT, the encoder and the LoRA under
KQUANT_TEST_GGUF_DIR and the fixtures under GMLX_VIDEO_FIXTURES.
"""
from __future__ import annotations

import json
import os
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
from gmlx.video import api, encoder, layout, pipeline, schedule
from gmlx.video.families.h3 import family as hf

LORA = "check6_h3_lora_512x512x22"
BASE = "check6_h3_512x512x22"
CASES = (LORA, BASE)
STREAMS = ("video", "audio")
LORA_FILE = "minimax_h3_fl2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors"


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


@pytest.mark.parametrize("name", CASES)
def test_the_manifests_follow_the_rule(name):
    """Each step's gate is twice the oracle's bf16 gap at that step, each
    exact run gate follows its floor, the run is the recipe's, and the
    two cases ran on one DiT."""
    m = _manifest(name)
    inp = m["inputs"]
    assert inp["forwards"] == inp["steps"] == 8
    assert inp["shape"] == [512, 512, 22]
    assert inp["prompt"] == "dialogue"
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
    assert m["files"]["dit"] == _manifest(LORA)["files"]["dit"]
    if name == LORA:
        assert inp["shifts"]["video"] == hf.TURBO_768_SHIFT
        assert inp["lora"] == m["files"]["lora"]["name"] == LORA_FILE
        assert inp["lora_pairs"] == 208 and inp["lora_strength"] == 1.0
        assert inp["lora_merge"] == "float32"
    else:
        assert inp["shifts"]["video"] == hf.SHIFTS["video"]
        assert inp["lora"] is None and "lora" not in m["files"]
        assert inp.get("lora_merge") is None


@pytest.mark.parametrize("name", CASES)
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


def _layout(pipe, feats: mx.array, m: dict):
    clip = _clip(m)
    fam = pipe.video_family
    vs, aus = fam.latent_shapes(clip["frames"], clip["width"],
                                clip["height"])
    return layout.pack(fam.timeline, int(feats.shape[1]), vs[2], vs[3],
                       vs[4], aus[3])


def _steps(pipe, feats: mx.array, arrays: dict, m: dict,
           table: str | None, sig: dict) -> list:
    """Relative L2 of each step's velocity, video and audio, from the
    oracle's fp32 latents before the step, against the oracle's, at the
    sigmas ``sig``."""
    n = _clip(m)["steps"]
    lay = _layout(pipe, feats, m)
    model = pipe.dit
    out = []
    prev = lora_install.activate(pipe, pipe.default_table if table is None
                                 else table)
    try:
        text = model.caption(feats)
        mx.eval(text)
        with pacing.paced(lambda a: mx.eval(*a)):
            for i in range(n):
                src = ("noise_{}" if i == 0 else f"{{}}_step{i}")
                xv, xa = (mx.array(arrays[src.format(s)]) for s in STREAMS)
                sv, sa = sig["video"][i], sig["audio"][i]
                plan = model.plan(lay, api.stream_time(sv),
                                  api.stream_time(sa))
                v = api.denoise_step(model, xv, xa,
                                     {"video": sv, "audio": sa}, text, lay,
                                     plan=plan)
                mx.eval(*v)
                out.append({s: _rel(g, arrays[f"velocity_{s}_step{i + 1}"])
                            for s, g in zip(STREAMS, v)})
    finally:
        lora_install.restore(pipe, prev)
    return out


def _run(pipe, feats: mx.array, start: tuple, m: dict,
         table: str | None, sig: dict | None = None) -> dict:
    xv, xa = pipeline.generate(pipe, feats, start=tuple(map(mx.array, start)),
                               table=table, sigmas=sig, **_clip(m))
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


def _free_floor(pipe, feats, arrays, m, table, got) -> dict:
    """The largest relative L2 of the exact run on the GPU against the
    same run on the CPU stream and from nudged inputs."""
    sig = _oracle_sigmas(arrays)
    others = []
    for seed in hg.NUDGE_SEEDS:
        start, f = _nudged(arrays, seed)
        others.append(_run(pipe, f, start, m, table, sig))
    with mx.stream(mx.cpu):
        others.append(_run(pipe, feats, _start(arrays), m, table, sig))
    return {s: max(_rel(o[s], got[s]) for o in others) for s in STREAMS}


def _record(m: dict, floor: dict, paths) -> dict:
    """Write the floor measured now, and the gates it gives, into each
    manifest at ``paths``. Returns the new manifest."""
    files = [m["files"][k] for k in ("dit", "lora") if k in m["files"]]
    named = " with ".join(f"{f['name']} {f['sha256'][:8]}" for f in files)
    res = {"exact_free_floor": {s: {
        "rel_l2": floor[s],
        "measured": (f"the largest relative L2 between gmlx's exact run on "
                     f"the GPU and the same run on the CPU stream or from "
                     f"inputs nudged by one float32 step "
                     f"({len(hg.NUDGE_SEEDS)} seeds); "
                     f"tests/video/test_video_check6_h3.py on {named}, "
                     f"{mx.device_info()['device_name']}")}
        for s in STREAMS}}
    tols = {"exact_free": {s: {"rel_l2": max(bg.EXACT_TOL, 2 * floor[s])}
                           for s in STREAMS}}
    for path in paths:
        doc = json.loads(path.read_text())
        assert doc["npz"] == m["npz"], path
        doc["results"].update(res)
        doc["tolerances"].update(tols)
        tmp = path.with_name(path.name + ".part")
        tmp.write_text(json.dumps(doc, indent=1) + "\n")
        tmp.replace(path)
    return {**m, "results": {**m["results"], **res},
            "tolerances": {**m["tolerances"], **tols}}


def _line(label: str, steps: list, run: dict, tols: dict) -> str:
    worst = {s: max(r[s] for r in steps) for s in STREAMS}
    return (f"{label}: steps within video {worst['video']:.2e}, audio "
            f"{worst['audio']:.2e}, gate {bg.EXACT_TOL:.0e}; run video "
            f"{run['video']:.2e} of {tols['video']['rel_l2']:.2e}, audio "
            f"{run['audio']:.2e} of {tols['audio']['rel_l2']:.2e}")


@pytest.mark.integration
@pytest.mark.parametrize("name", CASES)
def test_check6_exact_gates(gguf_dir, request, name):
    """Each step and the whole run in the oracle's arithmetic within the
    exact gates, and, on the LoRA case, the same run without the LoRA
    outside both."""
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
    pipe.dit.act_dtype = mx.float32
    with hg.exact_floats(pipe.dit), bg.exact_products(pipe.dit) as used:
        steps = _steps(pipe, feats, arrays, m, table, sig)
        got = _run(pipe, feats, _start(arrays), m, table, sig)
        floor = _free_floor(pipe, feats, arrays, m, table, got)
        bare = None
        if table is not None:
            bare = (_steps(pipe, feats, arrays, m, None, sig),
                    _run(pipe, feats, _start(arrays), m, None, sig))
    bg.assert_all_dense(pipe.dit, used)
    if record:
        m = _record(m, floor, (ORACLE / f"{name}.manifest.json",
                               fx.fixture_root() / f"{name}.manifest.json"))
    rec = _floor(m)
    tols = m["tolerances"]["exact_free"]
    run = {s: _rel(got[s], arrays[s]) for s in STREAMS}
    for s in STREAMS:
        print(f"{label}: {s} floor {floor[s]:.2e}, recorded "
              f"{rec[s]['rel_l2']:.2e}")
    print(_line(label, steps, run, tols))
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
    assert all(r[s] <= bg.EXACT_TOL for r in steps for s in STREAMS), steps
    assert all(run[s] <= tols[s]["rel_l2"] for s in STREAMS), run
    if bare is not None:
        assert all(r[s] > bg.EXACT_TOL for r in b_steps for s in STREAMS)
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


@pytest.fixture(scope="module")
def cases(gguf_dir):
    """Each case's arrays and manifest, with gmlx's features of its
    prompt from the encoder the check 2 oracle read, which is released
    before the DiT loads."""
    for name in CASES:
        fixture_present(_manifest(name)["name"])
    path = _find(gguf_dir, committed("check2_h3")["files"]["encoder"])
    prompts = committed("check2_h3")["inputs"]["prompts"]
    out = {}
    handle = encoder.load_encoder(hf.FAMILY, path, log=lambda *_: None)
    try:
        for name in CASES:
            arrays, m = fixture(_manifest(name)["name"])
            prompt = prompts[m["inputs"]["prompt"]]
            assert handle.token_ids(prompt) == arrays["ids"].tolist()
            feats = np.array(handle.encode(prompt).features, np.float32)
            out[name] = (arrays, m, feats)
    finally:
        handle.release()
    return out


@pytest.fixture(scope="module")
def pipe(gguf_dir, cases):
    return _load(gguf_dir, cases[LORA][1])


@pytest.mark.integration
@pytest.mark.parametrize("name", CASES)
def test_check6_shipping_steps(gguf_dir, pipe, cases, name):
    """Each step of the default path, on gmlx's own encoder features and
    the LoRA live, within twice the oracle's own bf16 gap at that step
    (the LoRA merged in float32 in that pass), with the kernels in
    force. The LoRA runs live on stock ops, since every H3 linear is
    wider than the fused LoRA route takes. The free run is printed."""
    assert not bg.exact_process()
    assert pipe.dit.act_dtype == mx.bfloat16
    arrays, m, feats = cases[name]
    table = _table(gguf_dir, pipe, m)
    f = mx.array(feats)[None]
    sig = schedule.stream_sigmas(m["inputs"]["steps"], m["inputs"]["shifts"])
    ran: list = []
    with bg.kernel_calls(ran.append):
        steps = _steps(pipe, f, arrays, m, table, sig)
    calls, want, plain = _census(pipe, table, _layout(pipe, f, m),
                                 len(steps), ran)
    print(f"[check6] {name} kernel calls {dict(calls)}, routes in force "
          f"{sorted(kq_routes.active())}")
    tols = m["tolerances"]["steps"]
    for i, (r, t) in enumerate(zip(steps, tols)):
        print(f"[check6] {name} step {i + 1}: " + "; ".join(
            f"{s} {r[s]:.3e}, {r[s] / t[s]['rel_l2']:.2f} of its gate"
            for s in STREAMS))
    got = _run(pipe, f, _start(arrays), m, table)
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
