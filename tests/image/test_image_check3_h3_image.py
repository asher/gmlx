"""Check 3 of the image acceptance gate for MiniMax-H3-Image: one denoise
step of the gmlx DiT, row by row, against ComfyUI's MiniMaxH3ImageModel
on the same weights.

scripts/image_oracle/check3_h3_image.py writes the fixtures: every row of
a text to image step in fp32, with the oracle's own bf16 against fp32 gap
of each. The step's positions, RoPE angles, segment rows, time, curve
rows and head segment equal the oracle's.

The tiny cases run ComfyUI's class at a tiny size on random weights at
sigma 1.0, which reads head 0 alone, and at 0.75, which adds head 3 to
it. Their fixtures are committed, so the whole forward is checked on the
CPU tier: in fp32 within FP32_TOL of the oracle, and in bf16 under the
product gate, twice the oracle's gap per row (h3_image_gates.py). A head
one grid point off, head k without head 0, RoPE pairs taken adjacent or
the GLU halves swapped miss the fp32 gate.

On the release file the fixtures hold the step at 512x512 at sigma 1.0
and 0.75, at 768x512 at 0.5327739119529724 and at 1024x1024 at
0.23200887441635132, the last past the GLU kernel route's row floor. The
oracle reads a copy of the file with the `config` key ComfyUI needs for
the PDD heads (the manifest's oracle_dit), every tensor unchanged; gmlx
reads the release file (dit), found under KQUANT_TEST_GGUF_DIR by the
manifest's name, size and sha256 and loaded by the family's loader, as
the family is not discovered yet. The product gate runs the model as it
ships and holds each row to twice the oracle's gap; the kernel routes in
force must run where they apply. The exact gate runs the model in the
oracle's fp32 arithmetic, every K-quant product dense on its dequantized
weight and every float tensor in float32: the teacher-forced rows within
EXACT_TOL, the free rows and the velocity within the larger of EXACT_TOL
and twice gmlx's own fp32 floor that the manifest records
(GMLX_IMAGE_RECORD_FLOORS=<dir> writes manifests with the floors
measured now). The rows on the release file need it under
KQUANT_TEST_GGUF_DIR and the fixtures under GMLX_IMAGE_FIXTURES.
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import mlx.core as mx
import pytest

import block_gates as bg
import h3_image_gates as hg
from block_gates import exact_process, kernel_calls, rerun_exact
from image_fixtures import fixture_view
from oracle_fixture import ORACLE, committed, committed_fixture, fixture, fx

from gmlx.image import ops
from gmlx.image.families.h3_image import family as hi
from gmlx.image.families.h3_image import loader as hl
from gmlx.image.families.h3_image import model as him
from gmlx.image.h3 import blocks
from gmlx.image.ops import kq_routes

TINY = "selftest_h3_image_tiny"
TINY_CASES = (f"{TINY}_s0", f"{TINY}_s3")
REAL = ("check3_h3_image_512x512_s0", "check3_h3_image_512x512_s3",
        "check3_h3_image_768x512_s4", "check3_h3_image_1024x1024_s5")
HEADER = "h3_image_pruned_ud_q4_k_xl.header.json"
# A float32 step against the oracle's float32 step: the gap is summation
# order alone.
FP32_TOL = 1e-5


@pytest.fixture(scope="module")
def tiny():
    weights, wm = committed_fixture(TINY + "_weights")
    out = {}
    for name in TINY_CASES:
        arrays, m = committed_fixture(name)
        assert wm["inputs"]["config"] == m["inputs"]["config"]
        out[name] = (arrays, m)
    return weights, wm["inputs"]["config"], out


@pytest.mark.parametrize("name", TINY_CASES)
def test_the_tiny_manifests_follow_the_rule(tiny, name):
    hg.assert_manifest_rule(tiny[2][name][1])


def test_the_tiny_cases_read_head_0_alone_and_head_3_beside_it(tiny):
    _, config, cases = tiny
    model = hg.tiny_model(tiny[0], config)
    heads = [model.modulation(float(cases[n][0]["sigma"]), 1).head
             for n in TINY_CASES]
    assert heads == [0, 3]


@pytest.mark.parametrize("name", TINY_CASES)
def test_the_tiny_step_lays_out_as_the_oracle(tiny, name):
    weights, config, cases = tiny
    arrays, m = cases[name]
    assert not hg.layout_misses(hg.tiny_model(weights, config), arrays, m)


@pytest.mark.parametrize("name", TINY_CASES)
def test_the_tiny_step_in_float32_is_the_oracle_step(tiny, name, request):
    """In float32 products with TF32 off and the stock ops, as the oracle
    multiplies."""
    if not exact_process():
        return rerun_exact(request, f"h3 image check3 tiny fp32 {name}")
    weights, config, cases = tiny
    arrays, m = cases[name]
    got = hg.measures(hg.tiny_model(weights, config), arrays, m)
    assert not hg.unmeasured(got, m)
    rows = {k: v for k, v in got.items() if not k.startswith("cosine ")}
    top = max(rows, key=rows.get)
    print(f"h3 image check3 tiny fp32 {name}: worst {rows[top]:.2e} at "
          f"{top}")
    over = {k: v for k, v in rows.items() if not v <= FP32_TOL}
    assert not over, over


@pytest.mark.parametrize("name", TINY_CASES)
def test_the_tiny_step_in_bfloat16_passes_the_product_gate(tiny, name):
    """On the weights in the dtypes the loader gives them."""
    weights, config, cases = tiny
    arrays, m = cases[name]
    model = hg.tiny_model(weights, config, act=mx.bfloat16)
    assert model.blocks[0].norm1.weight.dtype == mx.bfloat16
    assert model.video_patch_proj.weight.dtype == mx.float32
    assert model.final_layer.video_out.weight.dtype == mx.float32
    got = hg.measures(model, arrays, m)
    assert not hg.unmeasured(got, m)
    print(hg.worst_line(f"h3 image check3 tiny bf16 {name}", got, m))
    out = hg.missed(got, m["tolerances"])
    assert not out, out


def _glu_halves_swapped(mp):
    real_swiglu, real_fused = ops.swiglu, ops.swiglu_fused
    mp.setattr(ops, "swiglu", lambda a, b: real_swiglu(b, a))
    mp.setattr(ops, "swiglu_fused",
               lambda x, g, u, codec: real_fused(x, u, g, codec))


def _rope_pairs_adjacent(mp):
    mp.setattr(blocks.rope, "pair_halves", lambda x, rot: x)


def _head_one_off(mp):
    real = hi.step_head
    mp.setattr(hi, "step_head", lambda s: min(real(s) + 1, hi.STEPS - 1))


def _head_alone(mp):
    """Head k without head 0, as a slice of the bank reads it."""
    real = him.H3ImageDiT.head

    def head(self, x, mod, plan, taps=None):
        out = real(self, x, mod, plan, taps)
        if not mod.head:
            return out
        return out - real(self, x, dataclasses.replace(mod, head=0), plan)
    mp.setattr(him.H3ImageDiT, "head", head)


# Wrong models the float32 gate must fail, with the case each shows on:
# the head rule at a step past the first, the rest on both.
FAULTS = {"glu halves swapped": (_glu_halves_swapped, TINY_CASES),
          "rope pairs adjacent": (_rope_pairs_adjacent, TINY_CASES),
          "head one grid point off": (_head_one_off, TINY_CASES),
          "head k without head 0": (_head_alone, TINY_CASES[1:])}


@pytest.mark.parametrize("fault", FAULTS)
def test_the_tiny_float32_gate_fails_a_wrong_model(tiny, request,
                                                    monkeypatch, fault):
    if not exact_process():
        return rerun_exact(request, f"h3 image check3 tiny fault {fault}")
    weights, config, cases = tiny
    apply, names = FAULTS[fault]
    apply(monkeypatch)
    for name in names:
        arrays, m = cases[name]
        got = hg.measures(hg.tiny_model(weights, config), arrays, m)
        print(f"h3 image check3 tiny fault {fault} {name}: velocity "
              f"{got[hg.VELOCITY]:.2e}")
        assert got[hg.VELOCITY] > 100 * FP32_TOL, (name, got[hg.VELOCITY])


def _manifest(name: str) -> dict:
    if not (ORACLE / f"{name}.manifest.json").is_file():
        pytest.skip(f"{name}.manifest.json is not committed: run "
                    "scripts/image_oracle/check3_h3_image.py and commit the "
                    "manifests it writes")
    return committed(name)


@pytest.mark.parametrize("name", REAL)
def test_the_real_manifests_follow_the_rule(name):
    """As the tiny ones, and every fixture read the same files, with the
    exact gate's tolerances following their floors."""
    m = _manifest(name)
    hg.assert_manifest_rule(m)
    hg.exact_tols(m)
    first = committed(REAL[0])["files"]
    assert m["files"] == first
    assert first["oracle_dit"] == committed("check4_h3_image")["files"][
        "oracle_dit"]
    assert first["dit"]["name"] in hi.VARIANTS[next(iter(hi.VARIANTS))].files


def _find(root: Path, want: dict) -> str:
    for p in sorted(root.rglob(want["name"])):
        if p.is_file() and os.path.getsize(p) == want["size"]:
            if fx.file_identity(str(p))["sha256"] != want["sha256"]:
                pytest.fail(f"{p} is not the file the oracle read")
            return str(p)
    pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")


@pytest.fixture(scope="module")
def real(gguf_dir):
    p = _find(gguf_dir, _manifest(REAL[0])["files"]["dit"])
    model, info = hl.load(p, log=lambda *_: None)
    return model, info


# The exact gate. Each test reruns in a child process, which loads its
# own DiT. These tests come before the first test that holds the module's
# DiT, so a run of the whole file starts each child with none beside it.

@pytest.mark.integration
@pytest.mark.parametrize("name", REAL)
def test_check3_exact_gate(request, name):
    """Every teacher-forced row within EXACT_TOL of the oracle in its
    arithmetic, and the free rows and the velocity within the fixture's
    gates from the floors measured now."""
    fixture(name)
    record = hg.record_dir() is not None
    if not exact_process():
        return rerun_exact(request, f"h3 image check3 exact {name}")
    arrays, m = fixture(name)
    model, _ = request.getfixturevalue("real")
    label = f"h3 image check3 exact {name}"
    with hg.exact_floats(model) as n:
        floor = hg.exact_free_floor(model, arrays, m)
        if record:
            m = hg.record_floors(m, floor,
                                 (ORACLE / f"{name}.manifest.json",))
        tol = hg.check_floor(m, floor, label)
        got = hg.exact_measures(model, arrays, m)
    print(hg.exact_line(f"{label}, {n} float tensors cast to fp32", got,
                        tol))
    over = hg.exact_over(got, tol)
    assert not over, over


# Wrong models the exact gate must fail at a step that adds a head.
EXACT_FAULTS = {k: FAULTS[k][0] for k in ("rope pairs adjacent",
                                          "head one grid point off",
                                          "head k without head 0")}


@pytest.mark.integration
@pytest.mark.parametrize("fault", EXACT_FAULTS)
def test_check3_exact_gate_fails_a_wrong_model(request, monkeypatch, fault):
    fixture(REAL[1])
    if not exact_process():
        return rerun_exact(request, f"h3 image check3 exact fault {fault}")
    arrays, m = fixture(REAL[1])
    model, _ = request.getfixturevalue("real")
    tol = hg.exact_tols(m)
    with hg.exact_floats(model), monkeypatch.context() as mp:
        EXACT_FAULTS[fault](mp)
        bad = hg.exact_measures(model, arrays, m)
    line, top = hg.fault_line(f"h3 image check3 exact fault {fault}", bad,
                              tol)
    print(line)
    assert top > 10, (fault, top)


@pytest.mark.integration
def test_the_release_file_is_the_pruned_variant(real):
    """The loader names the release file by its weights, and its config is
    the one check 1 holds to the official transformer/config.json."""
    _, info = real
    ident = info.identified
    assert ident.family is hi.FAMILY
    assert (ident.variant, ident.layout, ident.prefix) == ("pruned",
                                                           "default", "")
    assert ident.identity_hash in hi.VARIANTS
    assert ident.config == hi.synth_config(fixture_view(HEADER))


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
@pytest.mark.parametrize("name", REAL)
def test_check3_rows(real, name):
    arrays, m = fixture(_manifest(name)["name"])
    model, _ = real
    assert not hg.layout_misses(model, arrays, m)
    ran: list = []
    with kernel_calls(ran.append):
        got = hg.measures(model, arrays, m)
    _, plan = hg.plan_of(model, arrays)
    calls, want = hg.kernel_census(model, plan.layout.rows, ran)
    print(f"h3 image check3 {name}: kernel calls {dict(calls)}, routes in "
          f"force {sorted(kq_routes.active())}")
    assert calls == want
    assert not hg.unmeasured(got, m)
    print(hg.worst_line(f"h3 image check3 {name}", got, m))
    out = hg.missed(got, m["tolerances"])
    assert not out, out
