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
"""
from __future__ import annotations

import dataclasses

import mlx.core as mx
import pytest

import h3_image_gates as hg
from block_gates import exact_process, rerun_exact
from oracle_fixture import committed_fixture

from gmlx.image import ops
from gmlx.image.families.h3_image import family as hi
from gmlx.image.families.h3_image import model as him
from gmlx.image.h3 import blocks

TINY = "selftest_h3_image_tiny"
TINY_CASES = (f"{TINY}_s0", f"{TINY}_s3")
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
