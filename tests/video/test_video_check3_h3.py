"""Check 3 of the video acceptance gate for MiniMax-H3: one denoise step
of the gmlx DiT, row by row, against ComfyUI's MiniMaxH3Model on the same
GGUF, under three gates.

scripts/video_oracle/check3_h3.py writes the fixtures: every row of a
text to video step in fp32 at 64x64x5 and 512x512x22, with the oracle's
own bf16 against fp32 gap of each. The step's positions, RoPE angles,
stream times, segment rows and curve rows equal the oracle's.

The product gate runs the model as it ships, in bf16 with the kernel
routes at their default, and holds each row to twice the oracle's gap:
teacher-forced for each refiner block, the refiner's norm, each block and
the heads, and free running for the other rows, for each of those rows
run free and for the velocities. Where gmlx's free run misses twice the
oracle's free gap at a block row while the teacher-forced rows, the
velocities and the exact gates pass, the free block rows from that row
to the last block take the larger of twice the gap and the oracle's
restart spread: its own largest gap at the row over its bf16 blocks
restarted from the fixture's state at 13 blocks (4 to 30), as is and with
3 nudges of one float32 step. scripts/video_oracle/check3_h3_spread.py
measures the spread once and records it in the manifest; the tests read
it. The kernel routes in force must run where they apply. The exact gate runs the model in the oracle's fp32
arithmetic, every K-quant product dense on its dequantized weight and
every float tensor of the file in float32. It holds the teacher-forced
rows to EXACT_TOL, and the free rows and each velocity to the larger of
EXACT_TOL and twice gmlx's own fp32 floor that the manifest records;
GMLX_VIDEO_RECORD_FLOORS=1 records the floors measured now. The
site gate holds each op of every block and each linear leaf on the path a
request takes to the stock op in float32 on the same input, within 2 n u
(h3_gates.py).

The anchor cases of merge 3 run the same step with keyframes and guides:
the first frame, the last, both, a mid-clip image with an audio anchor,
and at 512x512x22 a 5-frame guide clip at frame 17. Each carries its
anchors' latents, the oracle's noise of their video rows, and the
prompt's tags with a vision run where the case has keyframes.

The tiny cases run ComfyUI's class at a tiny size on random weights. Its
fixtures are committed, so the whole forward is checked on the CPU tier:
in fp32 within FP32_TOL of the oracle, and in bf16 under the product
gate. Each anchor case moves the velocities, and a tag run one row off,
an anchor one frame off or its noise dropped misses the fp32 gate.

The rows on the real file need the MiniMax-H3 GGUF under
KQUANT_TEST_GGUF_DIR and the fixtures under GMLX_VIDEO_FIXTURES. They
skip until both are there.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import block_gates as bg
from block_gates import exact_process, kernel_calls, rerun_exact
from video_fixtures import video_dits
from video_oracle_fixture import (ORACLE, committed, committed_fixture,
                                  fixture, fixture_present, fx)

import h3_gates as hg
import qwen_gates as qg

from gmlx.image import ops
from gmlx.image.ops import kq_routes
from gmlx.video import loader as video_loader
from gmlx.video.families.h3 import model as hm

TINY = "selftest_h3_tiny"
ANCHOR_CASES = ("first", "last", "both", "mid_audio", "clip17")
TINY_ANCHORS = tuple(f"{TINY}_{c}" for c in ANCHOR_CASES)
REAL = ("check3_h3_64x64x5", "check3_h3_512x512x22")
REAL_ANCHORS = tuple(f"check3_h3_{c}_{shape}" for shape in
                     ("64x64x5", "512x512x22") for c in ANCHOR_CASES
                     if not (c == "clip17" and shape == "64x64x5"))
# A float32 step against the oracle's float32 step: the gap is summation
# order alone, about a hundred float32 roundings.
FP32_TOL = 1e-5


def _manifest(name: str) -> dict:
    if not (ORACLE / f"{name}.manifest.json").is_file():
        pytest.skip(f"{name}.manifest.json is not committed: run "
                    "scripts/video_oracle/check3_h3.py on the MiniMax-H3 "
                    "GGUF and commit the manifests it writes")
    return committed(name)


@pytest.mark.parametrize("name", (TINY, *TINY_ANCHORS, *REAL,
                                  *REAL_ANCHORS))
def test_the_manifests_follow_the_rule(name):
    """Every row tolerance is twice the gap beside it, or the float32
    floor for a row the oracle runs in float32 in both modes, each
    teacher-forced row run free is twice its free gap, the velocity
    floors follow their gaps, the fixture holds every row of its depth
    with a gap for each, gmlx's stream times are the oracle's, and on the
    real file the exact gate's tolerances follow their floors."""
    m = _manifest(name)
    hg.assert_manifest_rule(m)
    if name in REAL or name in REAL_ANCHORS:
        hg.exact_tols(m)
        assert m["files"]["dit"] == committed(REAL[0])["files"]["dit"]


def _with_spread(m: dict, first: int) -> dict:
    """``m`` with a restart spread recorded from block row ``first`` to the
    last block, each row's spread four times its free gap, and the gates
    it gives."""
    m = copy.deepcopy(m)
    free = m["results"]["free_bf16_vs_fp32"]
    rows = {f"blocks.{b}": {"rel_l2": 4 * free[f"blocks.{b}"]["rel_l2"],
                            "run": "blocks.8 nudge1"}
            for b in range(first, m["inputs"]["blocks"])}
    m["results"]["free_restart_spread"] = {
        "starts": list(hg.RESTART_STARTS), "seeds": list(hg.RESTART_SEEDS),
        "first": first, "rows": rows}
    for k, r in rows.items():
        m["tolerances"][f"free {k}"] = {"rel_l2": r["rel_l2"]}
    return m


def _gate_left(m, row="blocks.40"):
    gap = m["results"]["free_bf16_vs_fp32"][row]["rel_l2"]
    m["tolerances"][f"free {row}"] = {"rel_l2": 2 * gap}


def _forced_widened(m):
    m["tolerances"]["blocks.40"]["rel_l2"] *= 4


def _start_dropped(m):
    m["results"]["free_restart_spread"]["starts"].pop()


def _from_a_start(m):
    wider = _with_spread(m, hg.RESTART_STARTS[-1])
    m["results"] = wider["results"]
    m["tolerances"] = wider["tolerances"]


def _row_dropped(m):
    del m["results"]["free_restart_spread"]["rows"]["blocks.49"]
    _gate_left(m, "blocks.49")


@pytest.mark.parametrize("wrong", (_gate_left, _forced_widened,
                                   _start_dropped, _from_a_start,
                                   _row_dropped))
def test_a_restart_spread_widens_free_block_rows_alone(wrong):
    """A recorded restart spread sets the gate of each free block row from
    its first to the last block, over the fixed restarts. It never widens
    a teacher-forced row, and a spread off that shape fails the rule."""
    m = _with_spread(_manifest(REAL[1]), 36)
    hg.assert_manifest_rule(m)
    wrong(m)
    with pytest.raises(AssertionError):
        hg.assert_manifest_rule(m)


@pytest.fixture(scope="module")
def tiny():
    weights, wm = committed_fixture(TINY + "_weights")
    arrays, m = committed_fixture(TINY)
    assert wm["inputs"]["config"] == m["inputs"]["config"]
    return weights, arrays, m


def test_the_tiny_step_lays_out_as_the_oracle(tiny):
    weights, arrays, m = tiny
    model = hg.tiny_model(weights, m["inputs"]["config"])
    assert not hg.layout_misses(model, arrays, m)


def test_the_tiny_step_in_float32_is_the_oracle_step(tiny, request):
    """In float32 products with TF32 off and the stock ops, as the oracle
    multiplies."""
    if not exact_process():
        return rerun_exact(request, "h3 check3 tiny fp32")
    weights, arrays, m = tiny
    model = hg.tiny_model(weights, m["inputs"]["config"])
    got = hg.measures(model, arrays, m)
    assert not hg.unmeasured(got, m)
    rows = {k: v for k, v in got.items() if not k.startswith("cosine ")}
    top = max(rows, key=rows.get)
    print(f"h3 check3 tiny fp32: worst {rows[top]:.2e} at {top}")
    over = {k: v for k, v in rows.items() if not v <= FP32_TOL}
    assert not over, over


@pytest.fixture(scope="module")
def tiny_anchors(tiny):
    weights, t2v, _ = tiny
    return weights, t2v, {n: committed_fixture(n) for n in TINY_ANCHORS}


def test_each_tiny_anchor_case_is_its_keyframes_or_guides(tiny_anchors):
    """The anchors sit where ComfyUI's nodes put them on the 22-frame
    clip, with vision runs in the prompt where the case has keyframes."""
    _, _, cases = tiny_anchors
    got = {n.removeprefix(TINY + "_"): (
        [(a["frame"], a["video_t"], a["audio_t"])
         for a in m["inputs"]["anchors"]], m["inputs"]["spans"])
        for n, (_, m) in cases.items()}
    assert got == {"first": ([(0, 1, 0)], [[2, 6]]),
                   "last": ([(21, 1, 0)], [[2, 6]]),
                   "both": ([(0, 1, 0), (21, 1, 0)], [[1, 3], [4, 7]]),
                   "mid_audio": ([(10, 1, 8)], []),
                   "clip17": ([(17, 2, 0)], [])}


@pytest.mark.parametrize("name", TINY_ANCHORS)
def test_the_tiny_anchor_step_in_float32_is_the_oracle_step(
        tiny_anchors, request, name):
    """Each anchor case lays out as the oracle and runs its step within
    FP32_TOL, and its anchors move the velocities from the text to video
    step's by far more than the gate."""
    if not exact_process():
        return rerun_exact(request, f"h3 check3 tiny fp32 {name}")
    weights, t2v, cases = tiny_anchors
    arrays, m = cases[name]
    model = hg.tiny_model(weights, m["inputs"]["config"])
    assert not hg.layout_misses(model, arrays, m)
    got = hg.measures(model, arrays, m)
    assert not hg.unmeasured(got, m)
    rows = {k: v for k, v in got.items() if not k.startswith("cosine ")}
    top = max(rows, key=rows.get)
    moved = {v: hg.rel(arrays[v], t2v[v]) for v in hg.VELOCITIES}
    print(f"h3 check3 tiny fp32 {name}: worst {rows[top]:.2e} at {top}; "
          "the velocities move from text to video by "
          + ", ".join(f"{k} {v:.2e}" for k, v in moved.items()))
    over = {k: v for k, v in rows.items() if not v <= FP32_TOL}
    assert not over, over
    assert all(v > 100 * FP32_TOL for v in moved.values()), moved


@pytest.mark.parametrize("name", TINY_ANCHORS)
def test_the_tiny_anchor_step_in_bfloat16_passes_the_product_gate(
        tiny_anchors, name):
    weights, _, cases = tiny_anchors
    arrays, m = cases[name]
    model = hg.tiny_model(weights, m["inputs"]["config"], act=mx.bfloat16)
    got = hg.measures(model, arrays, m)
    assert not hg.unmeasured(got, m)
    print(hg.worst_line(f"h3 check3 tiny bf16 {name}", got, m))
    out = hg.missed(got, m["tolerances"])
    assert not out, out


def _spans_off(arrays):
    return {**arrays, "spans": arrays["spans"] + 1}


def _frame_off(arrays):
    return {**arrays, "anchor.0.frame": arrays["anchor.0.frame"] - 1}


def _noise_dropped(arrays):
    return {**arrays, "anchor_noise.0": np.zeros_like(
        arrays["anchor_noise.0"])}


# A wrong anchor the float32 gate must fail, and the case that shows it:
# the vision run one row late, the guide a frame early, the noise of its
# video rows not drawn.
ANCHOR_FAULTS = {"a tag run one row off": (_spans_off, "first"),
                 "an anchor one frame off": (_frame_off, "clip17"),
                 "the noise draw dropped": (_noise_dropped, "both")}


@pytest.mark.parametrize("fault", ANCHOR_FAULTS)
def test_the_tiny_float32_gate_fails_a_wrong_anchor(tiny_anchors, request,
                                                    fault):
    if not exact_process():
        return rerun_exact(request, "h3 check3 tiny anchor fault")
    weights, _, cases = tiny_anchors
    wrong, case = ANCHOR_FAULTS[fault]
    arrays, m = cases[f"{TINY}_{case}"]
    model = hg.tiny_model(weights, m["inputs"]["config"])
    got = hg.measures(model, wrong(arrays), m)
    rows = {k: v for k, v in got.items() if not k.startswith("cosine ")}
    top = max(rows, key=rows.get)
    vel = {v: rows[v] for v in hg.VELOCITIES}
    print(f"h3 check3 tiny anchor fault {fault}: {rows[top]:.2e} at {top}; "
          + ", ".join(f"{k} {v:.2e}" for k, v in vel.items()))
    assert rows[top] > 10 * FP32_TOL, (top, rows[top])


def _glu_halves_swapped(mp):
    real_swiglu, real_fused = ops.swiglu, ops.swiglu_fused
    mp.setattr(ops, "swiglu", lambda a, b: real_swiglu(b, a))
    mp.setattr(ops, "swiglu_fused",
               lambda x, g, u, codec: real_fused(x, u, g, codec))


def _rope_pairs_adjacent(mp):
    mp.setattr(hm.rope, "pair_halves", lambda x, rot: x)


def _audio_frame_major(mp):
    mp.setattr(hm, "pack_audio", lambda lat: lat[0].transpose(2, 1, 0)
               .reshape(-1, lat.shape[1]))
    mp.setattr(hm, "unpack_audio", lambda rows, ch=2: rows.reshape(
        -1, ch, rows.shape[-1]).transpose(2, 1, 0)[None])


# Wrong models the float32 gates must fail: each a convention on which the
# references differ (the diffusers port stores the GLU halves the other
# way round) or one easy to get wrong.
FAULTS = {"glu halves swapped": _glu_halves_swapped,
          "rope pairs adjacent": _rope_pairs_adjacent,
          "stereo audio frame-major": _audio_frame_major}


@pytest.mark.parametrize("fault", FAULTS)
def test_the_tiny_float32_gate_fails_a_wrong_model(tiny, request,
                                                    monkeypatch, fault):
    if not exact_process():
        return rerun_exact(request, "h3 check3 tiny fault")
    weights, arrays, m = tiny
    model = hg.tiny_model(weights, m["inputs"]["config"])
    FAULTS[fault](monkeypatch)
    got = hg.measures(model, arrays, m)
    rows = {k: v for k, v in got.items() if not k.startswith("cosine ")}
    top = max(rows, key=rows.get)
    print(f"h3 check3 tiny fault {fault}: {rows[top]:.2e} at {top}")
    assert rows[top] > 100 * FP32_TOL


def test_the_tiny_step_in_bfloat16_passes_the_product_gate(tiny):
    """On the weights in the dtypes the loader gives them."""
    weights, arrays, m = tiny
    model = hg.tiny_model(weights, m["inputs"]["config"], act=mx.bfloat16)
    assert model.blocks[0].norm1.weight.dtype == mx.bfloat16
    assert model.video_patch_proj.weight.dtype == mx.float32
    got = hg.measures(model, arrays, m)
    assert not hg.unmeasured(got, m)
    print(hg.worst_line("h3 check3 tiny bf16", got, m))
    out = hg.missed(got, m["tolerances"])
    assert not out, out


@pytest.fixture(scope="module")
def real(gguf_dir):
    for name in REAL:
        fixture_present(_manifest(name)["name"])
    want = _manifest(REAL[0])["files"]["dit"]
    for p in video_dits(gguf_dir, "h3"):
        if Path(p).name == want["name"] and os.path.getsize(p) == want["size"]:
            break
    else:
        pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")
    if fx.file_identity(p)["sha256"] != want["sha256"]:
        pytest.fail(f"{p} is not the file the oracle read")
    model, vid = video_loader.load_dit(p, log=lambda *_: None)
    assert (vid.variant, vid.layout) == ("fl2va", "qkv")
    return model, list(model.blocks)


def _depth(real, m: dict, blocks: int | None = None):
    """The model with the blocks the fixture holds, or the first
    ``blocks``."""
    model, all_blocks = real
    n = m["inputs"]["blocks"] if blocks is None else blocks
    assert n <= len(all_blocks)
    model.blocks = all_blocks[:n]
    return model


# The exact gate. Each test reruns in a child process, which loads its
# own DiT. The parent takes none, and these tests come before the first
# test that holds the module's DiT, so a run of the whole file starts each
# child with no model loaded beside it.

@pytest.mark.integration
@pytest.mark.parametrize("name", (*REAL, *REAL_ANCHORS))
def test_check3_exact_gate(request, name):
    """Every teacher-forced row within EXACT_TOL of the oracle in its
    arithmetic, and the free rows and the velocities within the fixture's
    gates from the floors measured now."""
    fixture_present(_manifest(name)["name"])
    if not exact_process():
        return rerun_exact(request, f"h3 check3 exact {name}")
    arrays, m = fixture(name)
    model = _depth(request.getfixturevalue("real"), m)
    label = f"h3 check3 exact {name}"
    with hg.exact_floats(model) as n:
        floor = hg.exact_free_floor(model, arrays, m)
        if os.environ.get(hg.RECORD) == "1":
            m = hg.record_floors(m, floor, (
                ORACLE / f"{name}.manifest.json",
                fx.fixture_root() / f"{name}.manifest.json"))
        tol = hg.check_floor(m, floor, label)
        got = hg.exact_measures(model, arrays, m)
    print(hg.exact_line(f"{label}, {n} float tensors cast to fp32", got,
                        tol))
    over = hg.exact_over(got, tol)
    assert not over, over


@pytest.mark.integration
@pytest.mark.parametrize("fault", FAULTS)
def test_check3_exact_gate_fails_a_wrong_model(request, monkeypatch, fault):
    fixture_present(_manifest(REAL[0])["name"])
    if not exact_process():
        return rerun_exact(request, "h3 check3 exact fault")
    arrays, m = fixture(REAL[0])
    model = _depth(request.getfixturevalue("real"), m)
    tol = hg.exact_tols(m)
    with hg.exact_floats(model), monkeypatch.context() as mp:
        FAULTS[fault](mp)
        bad = hg.exact_measures(model, arrays, m)
    line, top = hg.fault_line(f"h3 check3 exact fault {fault}", bad, tol)
    print(line)
    assert top > 10, (fault, top)


def _text_rows_rounded(mp):
    """The text rows of the packed sequence rounded to bfloat16."""
    real_embed = hm.H3Transformer.embed

    def embed(self, video, audio, text, layout, *a, **k):
        x = real_embed(self, video, audio, text, layout, *a, **k)
        s = layout.span("text")
        t = x[:, s.start:s.stop].astype(mx.bfloat16).astype(x.dtype)
        return mx.concatenate([x[:, :s.start], t, x[:, s.stop:]], axis=1)
    mp.setattr(hm.H3Transformer, "embed", embed)


@pytest.mark.integration
def test_check3_exact_gate_fails_a_wrong_sequence(request, monkeypatch):
    """A slip in the sequence between the blocks, which the teacher-forced
    rows cannot see since each block takes the oracle's input there, misses
    the free rows' and the velocities' gates from the floors."""
    fixture_present(_manifest(REAL[0])["name"])
    if not exact_process():
        return rerun_exact(request, "h3 check3 exact sequence fault")
    arrays, m = fixture(REAL[0])
    model = _depth(request.getfixturevalue("real"), m)
    tol = hg.exact_tols(m)
    with hg.exact_floats(model), monkeypatch.context() as mp:
        _text_rows_rounded(mp)
        bad = hg.exact_measures(model, arrays, m)
    line, _ = hg.fault_line("h3 check3 exact sequence fault text rows "
                              "in bfloat16", bad, tol)
    print(line)
    assert not [k for k in hg.exact_over(bad, tol) if hg.forced_row(k)]
    ratio = {k: v / hg.exact_gate(k, tol) for k, v in bad.items()}
    free = max(r for k, r in ratio.items() if hg.free_row(k))
    vel = {v: ratio[v] for v in hg.VELOCITIES}
    assert free > 10, free
    assert all(r > 1 for r in vel.values()) and max(vel.values()) > 10, vel


@pytest.mark.integration
def test_the_loaded_dit_keeps_its_float32_leaves(real):
    """Every float tensor under the family's fp32_keep loads in float32,
    and every other one in the activation dtype."""
    from mlx.utils import tree_flatten

    model, _ = real
    quant = set(bg.kquant_leaves(model).values())
    wrong = {k: a.dtype for k, a in tree_flatten(model.parameters())
             if k.rsplit(".", 1)[0] not in quant
             and a.dtype != (mx.float32 if hg.keeps_fp32(k)
                             else model.act_dtype)}
    assert not wrong, wrong


@pytest.mark.integration
@pytest.mark.parametrize("name", (*REAL, *REAL_ANCHORS))
def test_check3_rows(real, name):
    arrays, m = fixture(_manifest(name)["name"])
    model = _depth(real, m)
    assert not hg.layout_misses(model, arrays, m)
    ran: list = []
    with kernel_calls(ran.append):
        got = hg.measures(model, arrays, m)
    *_, lay = hg.case(arrays)
    calls, want = hg.kernel_census(model, lay, ran)
    print(f"h3 check3 {name}: kernel calls {dict(calls)}, routes in force "
          f"{sorted(kq_routes.active())}")
    assert calls == want
    assert not hg.unmeasured(got, m)
    print(hg.worst_line(f"h3 check3 {name}", got, m))
    out = hg.missed(got, m["tolerances"])
    assert not out, out


# The site gate.

def _sites(model, arrays: dict) -> tuple:
    """The sites of one free-running production step on the fixture's
    inputs, the kernel sites, the head rows, the leaves called on a
    float32 input and the velocities."""
    sites, routed, heads, skipped, by_row = {}, {}, {}, set(), {}
    with hg.step_sites(model, sites, routed=routed, heads=heads,
                       skipped=skipped, by_row=by_row):
        rows = hg.free_rows(model, arrays)
    return sites, routed, heads, skipped, rows, by_row


def site_line(label: str, sites: dict, routed: dict) -> str:
    worst = qg.worst_by_op(sites)
    kinds: dict = {}
    for op in routed.values():
        kinds[op] = kinds.get(op, 0) + 1
    return (f"{label}: {len(sites)} readings, the largest over its gate "
            "by op: "
            + ", ".join(f"{k} {v:.2f}" for k, v in sorted(worst.items()))
            + "; sites on mlx-kquant: "
            + (", ".join(f"{k} {v}" for k, v in sorted(kinds.items()))
               or "none"))


@pytest.mark.integration
@pytest.mark.parametrize("name", REAL)
def test_check3_site_gate(real, name):
    """Every op of every block and every linear leaf a step calls on the
    path a request takes is within its site gate of the stock op in
    float32 on the same input, the leaves on a float32 input are the ones
    the family keeps in float32, and that run's velocities pass the
    product gate."""
    from gmlx.image.lora import install

    arrays, m = fixture(_manifest(name)["name"])
    model = _depth(real, m)
    *_, lay = hg.case(arrays)
    sites, routed, heads, skipped, rows, by_row = _sites(model, arrays)
    assert skipped == {p for p in install.linear_leaves(model)
                       if hg.keeps_fp32(p)}
    missing = hg.missing_ops(model, sites, lay, routed, skipped)
    assert not missing, missing
    qk = {k for k in qg.op_sites(sites) if k[2] == "qk"}
    on = "qk" in kq_routes.active()
    assert {k for k, v in routed.items() if v == "qk_norm_rope"} == (
        qk if on else set())
    assert set(qg.head_sites(sites)) == qk
    assert set(by_row) == {k for k, v in routed.items()
                           if v == bg.KERNEL_OPS["glu"]}
    print(site_line(f"h3 check3 sites {name}", sites, routed))
    print(f"h3 check3 sites {name}, q and k: "
          + qg.qk_reading(sites, routed, heads, hg.qk_stock_n(model)))
    if by_row:
        worst = max(r / g for r, g in by_row.values())
        print(f"h3 check3 sites {name}, glu rows: {len(by_row)} sites, the "
              f"worst token row {worst:.2f} of its gate")
    over = {**qg.over_gate(sites), **_rows_over(by_row)}
    assert not over, over
    for v in hg.VELOCITIES:
        rel = hg.rel(rows[v], arrays[v])
        assert rel <= m["tolerances"][v]["rel_l2"], (v, rel)


def _rows_over(by_row: dict) -> dict:
    """The token row readings over their gate, keyed apart from the site's
    own reading."""
    return {(*k, "row"): v for k, v in qg.over_gate(by_row).items()}


def _swiglu_high(mp):
    real_hidden = hm._Block.hidden
    mp.setattr(hm._Block, "hidden",
               lambda self, h: real_hidden(self, h) * 1.05)


def _a_head_off(mp):
    """Head 1 of the seam's output a twentieth off, on the kernel when the
    qk route is in force."""
    real_qk = ops.qk_norm_rope

    def off(q, k, qw, kw, cos, sin, eps):
        q2, k2 = real_qk(q, k, qw, kw, cos, sin, eps)
        scale = mx.array([1.05 if h == 1 else 1.0
                          for h in range(q2.shape[1])]).astype(q2.dtype)
        return q2 * scale[:, None, None], k2
    mp.setattr(ops, "qk_norm_rope", off)


def _glu_tail_high(mp):
    """The last 16 rows of the MLP's hidden a tenth high, as a GLU kernel
    with a wrong partial last tile would write them, wherever the kernel
    takes the rows."""
    real_hidden = hm._Block.hidden

    def hidden(self, h):
        y = real_hidden(self, h)
        n = y.shape[1]
        if n < kq_routes._GLU_MIN_ROWS:
            return y
        scale = mx.concatenate([mx.ones((n - 16,)), mx.full((16,), 1.1)])
        return y * scale.astype(y.dtype)[None, :, None]
    mp.setattr(hm._Block, "hidden", hidden)


def _modulate_high(mp):
    real_mod = ops.modulate
    mp.setattr(ops, "modulate", lambda x, s: real_mod(x, s) * 1.05)


def _gated_residual_high(mp):
    real_res = ops.gated_residual_once
    mp.setattr(ops, "gated_residual_once",
               lambda x, g, y: real_res(x, g * 1.05, y))


# name: (the wrong production op, the ops whose sites must miss)
SITE_FAULTS = {"swiglu a twentieth high": (_swiglu_high, {"swiglu"}),
               "the GLU kernel's last 16 rows a tenth high": (
                   _glu_tail_high, {"swiglu"}),
               "one head of q a twentieth off": (_a_head_off, {"qk"}),
               "modulation a twentieth high": (_modulate_high,
                                               {"modulate"}),
               "gated residual a twentieth high": (
                   _gated_residual_high, {"gated_residual_once"})}


@pytest.mark.integration
@pytest.mark.parametrize("name", list(SITE_FAULTS))
def test_check3_each_wrong_op_fails_the_site_gate(real, monkeypatch, name):
    """A site reads an op on the op's own input, so a wrong op is over the
    gate at its own sites. Two blocks at 512x512x22 are enough: each block
    reads its own sites, and the kernel routes run there."""
    arrays, m = fixture(_manifest(REAL[1])["name"])
    model = _depth(real, m, blocks=2)
    apply, hit = SITE_FAULTS[name]
    try:
        with monkeypatch.context() as mp:
            apply(mp)
            sites, _, _, _, _, by_row = _sites(model, arrays)
    finally:
        _depth(real, m)
    if apply is _glu_tail_high and not by_row:
        pytest.skip("the GLU kernel route does not run here")
    over = {**qg.over_gate(sites), **_rows_over(by_row)}
    worst = qg.worst_by_op(sites)
    print(f"h3 check3 sites, {name}: {len(over)} readings over the gate; "
          "the largest by op "
          + ", ".join(f"{k} {v:.2f}" for k, v in sorted(worst.items())))
    missed = {qg.site_op(k) for k in over if len(k) != 2}
    assert hit <= missed, (name, sorted(missed))
