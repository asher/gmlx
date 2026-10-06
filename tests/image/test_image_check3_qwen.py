"""Check 3 of the image acceptance gate for Qwen-Image 2.1: one denoise
step of the gmlx DiT, block by block, against the diffusers oracle in both
of its modes, the joint forward and the cached decode.

qwen_gates.py describes the rows. The tiny rows run on the CPU from the
fixtures committed under tests/fixtures/image/oracle/, which
scripts/image_oracle/check3_qwen_image21.py --self-test writes with
diffusers' own class on the tiny model of qwen_image21_tiny.py. gmlx
loads the same weights from a GGUF in GGUF tensor names and runs in the
oracle's fp32 arithmetic, so every row is held to one fixed bound, and
each wrong model of FAULTS must miss it.

The real rows need the Qwen-Image 2.1 GGUF under KQUANT_TEST_GGUF_DIR
and the fixtures the same script writes from it. They skip until both
are there.
"""
from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import qwen_gates as qg
import qwen_tiny as qt
from block_gates import (EXACT_TOL, assert_all_dense, exact_process,
                         exact_products, rerun_exact)
from oracle_fixture import ORACLE, committed, committed_fixture, fixture, fx

from gmlx.image import loader as image_loader
from gmlx.image import ops, registry
from gmlx.image.families.qwen_image21 import layout
from gmlx.image.families.qwen_image21 import model as qm

BLOCKS = "selftest_qwen_image21_tiny_ref2"
TINY = {c: f"selftest_qwen_image21_tiny_{c}" for c in ("text", "ref1", "ref2")}

# The tiny model in fp32 on the CPU, with the oracle's timestep
# embedding: the right model measures 1.5e-7 at most on a teacher-forced
# row, 3.5e-7 on a branch row and 5.3e-7 on a free-running measure, the
# joint velocity under two reference images. The weakest wrong model, the
# erf gelu in the text projection, moves its most moved row to 1.9e-4,
# and the weakest of the others, the gate without its tanh, to 1.1e-1.
TOL = 4e-6
# gmlx's own sinusoid differs from torch's by one ulp at some
# frequencies. At a timestep of 375 that is 2.9e-6 in the modulation, and
# the tiny random blocks carry it to 4.1e-6 in the velocity.
TEMB_TOL = 1e-5
OWN_TOL = 4e-5
# A wrong model is seen when it moves a measure this many times the gate.
SEEN = 2.0


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """The oracle's tiny model, read from a GGUF in GGUF tensor names,
    with fp32 activations."""
    w = qt.weights()
    path = qt.write(tmp_path_factory.mktemp("qwen") / "tiny.gguf", w)
    fam = qt.tiny_family()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(registry, "families", lambda: (fam,))
        model, info = image_loader.load_dit(path, log=lambda *_: None)
    assert info.identified.layout == "fused"
    model.act_dtype = mx.float32
    return model


@pytest.mark.parametrize("case", list(TINY))
def test_tiny_weights_and_inputs_follow_the_shared_rule(case):
    arrays, m = committed_fixture(TINY[case])
    assert m["inputs"]["config"] == qt.TINY and m["inputs"]["case"] == case
    assert qt.rule.sha256(qt.weights()) == m["inputs"]["weights_sha256"]
    assert np.array_equal(arrays["features"], qt.rule.features(case))
    assert np.array_equal(arrays["img_mask"], qt.rule.image_mask(case))
    refs = qt.rule.ref_latents(case)
    assert len(refs) == len(m["inputs"]["ref_shapes"])
    for i, ref in enumerate(refs):
        assert np.array_equal(arrays[f"ref_latents.{i}"], ref)
    size = m["inputs"]["target_tokens"]
    assert np.array_equal(arrays["latents_a"], qt.rule.latents(0, 1, size))
    assert np.array_equal(arrays["latents_b"], qt.rule.latents(1, 1, size))
    # The oracle read these weights from a GGUF, and its extract call
    # gives the joint call's output.
    assert "GGUF tensor names" in m["inputs"]["loaded"]
    assert m["results"]["extract_against_joint_max_abs_fp32"] == 0.0


def test_tiny_model_matches_the_diffusers_oracle_block_by_block(tiny):
    arrays, m = committed_fixture(BLOCKS)
    n = len(tiny.transformer_blocks)
    # The CPU stream keeps fp32 products exact; the GPU's TF32 would not.
    with mx.stream(mx.cpu):
        for tag in "ab":
            assert np.array_equal(
                tiny.timesteps(float(arrays[f"sigma_{tag}"]), 1),
                arrays[f"t_embedded_{tag}"])
        branches: dict = {}
        rows = qg.check3_rows(tiny, arrays, branches)
        free = qg.free_run(tiny, arrays)
        times = qg.timestep_rows(tiny, arrays)
        own = qg.free_run(tiny, arrays, own_temb=True)
    assert sorted(branches) == sorted(m["inputs"]["branches"])
    assert set(m["inputs"]["rows"]) <= set(rows)
    assert sorted([*free, *times]) == sorted(
        [*m["inputs"]["free_rows"], "velocity_a", "velocity_b"])
    assert len(rows) == 8 + 4 * n + 2 and len(branches) == 4 * n

    def worst(d):
        k = max(d, key=lambda r: d[r]["rel_l2"])
        return f"{d[k]['rel_l2']:.2e} at {k}"

    print(f"qwen check3 tiny: {len(rows)} teacher-forced rows within "
          f"{worst(rows)}, {len(branches)} branch rows within "
          f"{worst(branches)}, {len(free)} free-running measures within "
          f"{worst(free)}, gate {TOL:.0e}; own sinusoid: embedding and "
          f"modulation within {worst(times)}, gate {TEMB_TOL:.0e}, free "
          f"run within {worst(own)}, gate {OWN_TOL:.0e}")
    for measured, tol in ((rows, TOL), (branches, TOL), (free, TOL),
                          (times, TEMB_TOL), (own, OWN_TOL)):
        assert not qg.missed(measured, tol), qg.missed(measured, tol)


@pytest.mark.parametrize("case", list(TINY))
def test_tiny_velocities_match_the_diffusers_oracle(tiny, case):
    """Both of the oracle's modes: a is its joint forward, and b its
    cached decode at another sigma on other latents."""
    arrays, m = committed_fixture(TINY[case])
    assert m["inputs"]["sigma_a"] != m["inputs"]["sigma_b"]
    with mx.stream(mx.cpu):
        got = qg.free_run(tiny, arrays)
        own = qg.free_run(tiny, arrays, own_temb=True)
    print(f"qwen check3 tiny {case}: velocity within "
          f"{got['velocity_a']['rel_l2']:.2e} of the joint call and "
          f"{got['velocity_b']['rel_l2']:.2e} of the cached call, gate "
          f"{TOL:.0e}; own sinusoid {own['velocity_a']['rel_l2']:.2e} and "
          f"{own['velocity_b']['rel_l2']:.2e}, gate {OWN_TOL:.0e}")
    for tag in "ab":
        assert got[f"velocity_{tag}"]["rel_l2"] < TOL, (tag, got)
        assert own[f"velocity_{tag}"]["rel_l2"] < OWN_TOL, (tag, own)
        # The oracle's own bf16 run sits far outside the fp32 gate.
        gap = fx.rel_l2(arrays[f"velocity_{tag}_bf16"],
                        arrays[f"velocity_{tag}"])
        assert gap > 100 * TOL


# Wrong models.

def halves_swapped(mp, model, arrays):
    """The fused product read as up first, gate second."""
    mp.setattr(qm, "split_gate_up",
               lambda y: tuple(reversed(mx.split(y, 2, axis=-1))))


def gelu_for_silu(mp, model, arrays):
    mp.setattr(ops, "swiglu", lambda gate, up: nn.gelu(gate) * up)


def tanh_dropped(mp, model, arrays):
    mp.setattr(qm, "squash", lambda gate: gate)


def text_norm_not_centered(mp, model, arrays):
    """scale = weight, where it is weight + 1."""
    mp.setattr(qm.ZeroCenteredRMSNorm, "scale",
               lambda self: self.weight.astype(mx.float32))


def rope_axes_permuted(mp, model, arrays):
    """Height on the width axis and width on the height axis."""
    real = layout.rope_angles
    mp.setattr(layout, "rope_angles",
               lambda pos, axes, theta: real(pos[:, [0, 2, 1]], axes, theta))


def rope_frame_axis_last(mp, model, arrays):
    """The frame position on the last axis."""
    real = layout.rope_angles
    mp.setattr(layout, "rope_angles",
               lambda pos, axes, theta: real(pos[:, [1, 2, 0]], axes, theta))


def timestep_rows_swapped(mp, model, arrays):
    """The prefix reads the sampled row and the target the t = 0 row."""
    sampled, zero = qm.modulation, qm.prefix_modulation
    sigma = float(arrays["sigma_a"])
    mp.setattr(qm, "modulation", lambda model, s: zero(model))
    mp.setattr(qm, "prefix_modulation", lambda model: sampled(model, sigma))


def scales_and_gates_grouped(mp, model, arrays):
    """The modulation read as scale1, scale2, gate1, gate2."""
    def split(rows):
        s1, s2, g1, g2 = mx.split(rows, 4, axis=-1)
        return s1, g1, s2, g2

    mp.setattr(qm, "split_modulation", split)


def gates_before_scales(mp, model, arrays):
    """The modulation read as gate1, scale1, gate2, scale2."""
    def split(rows):
        g1, s1, g2, s2 = mx.split(rows, 4, axis=-1)
        return s1, g1, s2, g2

    mp.setattr(qm, "split_modulation", split)


def text_gelu_exact(mp, model, arrays):
    """The text projection with the erf gelu for the tanh one."""
    mp.setattr(nn, "gelu_approx", nn.gelu)


def latents_column_major(mp, model, arrays):
    def pack(latents):
        b, c, h, w = latents.shape
        return latents.transpose(0, 3, 2, 1).reshape(b, h * w, c)

    mp.setattr(model, "pack", pack)


def _block(r: str) -> bool:
    return "transformer_blocks." in r


def _block_out(r: str) -> bool:
    return _block(r) and not r.endswith(qg.BRANCHES)


def _mlp(r: str) -> bool:
    return _block_out(r) or r.endswith(".img_mlp")


def _after_scale1(r: str) -> bool:
    return _block(r) or r.startswith("prefill.")


TIMES = ("joint.time_text_embed", "cached.time_text_embed",
         "joint.modulation", "cached.modulation")
FIRST_KV = ("cache.k.0", "cache.v.0")

# name: (the wrong model, the teacher-forced and branch rows it can
# reach, the free-running measures it cannot reach).
FAULTS = {
    "gate and up halves swapped": (halves_swapped, _mlp, FIRST_KV),
    "gelu for silu": (gelu_for_silu, _mlp, FIRST_KV),
    "tanh on the gate dropped": (tanh_dropped, _mlp, FIRST_KV),
    "text_norm not zero-centered": (
        text_norm_not_centered, lambda r: r in ("txt_in", "joint.in"), ()),
    "rope axes permuted": (
        rope_axes_permuted,
        lambda r: _block(r) or r.startswith(("rope_", "prefill.k.")),
        ("cache.v.0",)),
    "rope frame axis last": (
        rope_frame_axis_last,
        lambda r: _block(r) or r.startswith(("rope_", "prefill.k.")),
        ("cache.v.0",)),
    "t = 0 row and sampled row swapped": (
        timestep_rows_swapped,
        lambda r: _after_scale1(r) or r.endswith(".proj_out"), ()),
    "scales and gates grouped": (scales_and_gates_grouped, _after_scale1, ()),
    "gates before scales": (gates_before_scales, _after_scale1, ()),
    "text gelu exact": (
        text_gelu_exact, lambda r: r in ("txt_in", "joint.in"), ()),
    "latents column major": (
        latents_column_major,
        lambda r: r in ("img_in", "joint.in", "cached.in"), ()),
}


def fault_misses(monkeypatch, model, arrays: dict, name: str, tol: float
                 ) -> dict:
    """The measures the wrong model ``name`` misses. It must miss one by
    SEEN times the gate at least, and none outside the rows its fault
    reaches."""
    apply, reach, clean = FAULTS[name]
    layout.clear_caches()
    try:
        with monkeypatch.context() as mp:
            apply(mp, model, arrays)
            missed = qg.missed(qg.step(model, arrays), tol)
    finally:
        layout.clear_caches()
    assert missed and max(missed.values()) >= SEEN * tol, (name, missed)
    stray = sorted(r for r in missed if (
        r.removeprefix("free ") in clean if r.startswith("free ")
        else not reach(r)))
    assert not stray, (name, stray)
    return missed


@pytest.mark.parametrize("name", list(FAULTS))
def test_each_wrong_model_fails_the_tiny_gate(tiny, monkeypatch, name):
    arrays, _ = committed_fixture(BLOCKS)
    with mx.stream(mx.cpu):
        missed = fault_misses(monkeypatch, tiny, arrays, name, TOL)
        # The same model passes once the fault is gone.
        assert not qg.missed(qg.step(tiny, arrays), TOL)
    forced = {r: v for r, v in missed.items() if not r.startswith("free ")}
    low = min(forced, key=forced.get) if forced else None
    high = max(missed, key=missed.get)
    print(f"qwen check3 tiny, {name}: {len(missed)} measures miss, "
          f"{len(forced)} of them teacher-forced"
          + (f", from {forced[low]:.2e} at {low}" if low else "")
          + f", up to {missed[high]:.2e} at {high}; velocities "
          f"{missed.get('free velocity_a', 0.0):.2e} and "
          f"{missed.get('free velocity_b', 0.0):.2e}")
    assert "free velocity_a" in missed and "free velocity_b" in missed
    assert len(FAULTS) >= 7


@pytest.mark.parametrize("case", ["text", "ref1"])
def test_each_wrong_model_fails_on_the_velocities_alone(tiny, monkeypatch,
                                                        case):
    """The fixtures without block rows see every wrong model in both
    velocities."""
    arrays, _ = committed_fixture(TINY[case])
    seen = {}
    with mx.stream(mx.cpu):
        for name, (apply, _, _) in FAULTS.items():
            layout.clear_caches()
            with monkeypatch.context() as mp:
                apply(mp, tiny, arrays)
                got = qg.free_run(tiny, arrays)
            layout.clear_caches()
            seen[name] = min(got[f"velocity_{t}"]["rel_l2"] for t in "ab")
    print(f"qwen check3 tiny {case}, wrong models on the velocities: "
          + ", ".join(f"{k} {v:.2e}" for k, v in seen.items()))
    for name, miss in seen.items():
        assert miss >= SEEN * TOL, (name, miss)


# Qwen-Image 2.1 Q4_K_M, when the file and its fixtures are there.

REAL = {512: "check3_qwen_image21_512", 1024: "check3_qwen_image21_1024"}


def _real_manifest(name: str) -> dict:
    if not (ORACLE / f"{name}.manifest.json").is_file():
        pytest.skip(f"{name}.manifest.json is not committed: run "
                    "scripts/image_oracle/check3_qwen_image21.py on the "
                    "Qwen-Image 2.1 GGUF and commit the manifest it writes")
    return committed(name)


@pytest.fixture(scope="module")
def real(gguf_index):
    m = _real_manifest(REAL[512])
    want = m["files"]["dit"]
    for p in gguf_index.get("qwen_image", []):
        if Path(p).name == want["name"] and os.path.getsize(p) == want["size"]:
            break
    else:
        pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")
    if fx.file_identity(p)["sha256"] != want["sha256"]:
        pytest.fail(f"{p} is not the file the oracle read")
    model, _ = image_loader.load_dit(p, log=lambda *_: None)
    return model


def _product_misses(m: dict, rows: dict, free: dict) -> list[str]:
    """Measures of the product path over the manifest's tolerances, each
    twice the oracle's own bf16 against fp32 gap at that point."""
    tol = m["tolerances"]
    out = [f"{r} rel L2 {g['rel_l2']:.3e} > {tol['rows_rel_l2'][r]:.3e}"
           for r, g in rows.items() if r in tol.get("rows_rel_l2", {})
           and not g["rel_l2"] <= tol["rows_rel_l2"][r]]
    for r, g in free.items():
        if r.startswith("velocity_"):
            if not g["rel_l2"] <= tol[f"{r}_rel_l2"]:
                out.append(f"{r} rel L2 {g['rel_l2']:.3e} > "
                           f"{tol[f'{r}_rel_l2']:.3e}")
            if not g["cosine"] >= tol[f"{r}_cosine_min"]:
                out.append(f"{r} cosine {g['cosine']:.6f} < "
                           f"{tol[f'{r}_cosine_min']}")
        elif not g["rel_l2"] <= tol["free_rel_l2"][r]:
            out.append(f"free {r} rel L2 {g['rel_l2']:.3e} > "
                       f"{tol['free_rel_l2'][r]:.3e}")
    return out


@pytest.mark.integration
def test_check3_one_step_block_by_block_at_512(real):
    """The product gate, in both of the oracle's modes."""
    arrays, m = fixture(_real_manifest(REAL[512])["name"])
    for tag in "ab":
        assert np.array_equal(
            real.timesteps(float(arrays[f"sigma_{tag}"]), 1),
            arrays[f"t_embedded_{tag}"])
    rows = qg.check3_rows(real, arrays)
    free = {**qg.free_run(real, arrays, own_temb=True),
            **qg.timestep_rows(real, arrays)}
    misses = _product_misses(m, rows, free)
    assert not misses, misses


@pytest.mark.integration
def test_check3_final_velocities_at_1024(real):
    arrays, m = fixture(_real_manifest(REAL[1024])["name"])
    misses = _product_misses(m, {}, qg.free_run(real, arrays, own_temb=True))
    assert not misses, misses


@pytest.mark.integration
def test_check3_exact_gate_at_512(real, request, monkeypatch):
    """The model matches the oracle on every row in the oracle's
    arithmetic, and each wrong model of FAULTS fails."""
    arrays, m = fixture(_real_manifest(REAL[512])["name"])
    if not exact_process():
        return rerun_exact(request, "qwen check3 exact")
    with exact_products(real) as used:
        got = qg.step(real, arrays)
        assert_all_dense(real, used)
        top = max(got, key=lambda r: got[r]["rel_l2"])
        print(f"qwen check3 exact 512: {len(got)} measures within "
              f"{got[top]['rel_l2']:.2e} of the oracle, at {top}, gate "
              f"{EXACT_TOL:.0e}")
        assert not qg.missed(got, EXACT_TOL), qg.missed(got, EXACT_TOL)
        for name in FAULTS:
            missed = fault_misses(monkeypatch, real, arrays, name, EXACT_TOL)
            high = max(missed, key=missed.get)
            print(f"qwen check3 exact 512, {name}: {len(missed)} measures "
                  f"miss, up to {missed[high]:.2e} at {high}")


@pytest.mark.integration
def test_check3_exact_velocities_at_1024(real, request):
    arrays, m = fixture(_real_manifest(REAL[1024])["name"])
    if not exact_process():
        return rerun_exact(request, "qwen check3 exact")
    with exact_products(real):
        got = qg.free_run(real, arrays)
    print(f"qwen check3 exact 1024: velocities within "
          f"{got['velocity_a']['rel_l2']:.2e} and "
          f"{got['velocity_b']['rel_l2']:.2e}, gate {EXACT_TOL:.0e}")
    assert not qg.missed(got, EXACT_TOL), qg.missed(got, EXACT_TOL)
