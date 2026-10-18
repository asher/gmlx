"""Check 3 of the image acceptance gate for Qwen-Image 2.1: one denoise
step of the gmlx DiT, block by block, against the diffusers oracle in both
of its modes, the joint forward and the cached decode.

qwen_gates.py describes the rows. The tiny rows run on the CPU from the
fixtures committed under tests/fixtures/image/oracle/, which
scripts/image_oracle/check3_qwen_image21.py --self-test writes with
diffusers' own class on the tiny model of qwen_image21_tiny.py. gmlx
loads the same weights from a GGUF in GGUF tensor names and runs in the
oracle's fp32 arithmetic, so every row is held to one fixed bound, and
each wrong model of FAULTS must miss it. The site gate reads the model
in bf16: each op of a block and each linear leaf on its own input against
the stock op in float32, and each wrong op of SITE_FAULTS must miss it at
its own sites.

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

import block_gates as bg
import qwen_gates as qg
import qwen_tiny as qt
from block_gates import (EXACT_TOL, QWEN_EXACT_FREE_TOL, assert_all_dense,
                         exact_process, exact_products, free_running,
                         rerun_exact)
from image_fixtures import image_dits
from oracle_fixture import ORACLE, committed, committed_fixture, fixture, fx

from gmlx.image import api, ops, registry
from gmlx.image import loader as image_loader
from gmlx.image.families.qwen_image21 import layout
from gmlx.image.families.qwen_image21 import model as qm
from gmlx.image.lora import install
from gmlx.image.ops import attention, kq_routes, norms

# The tiny rows run on a Mac of 16 GB, as most users run, and the rows on
# local files on this Mac (tests/host_box.py).

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


@pytest.mark.host_gb(16)
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


@pytest.mark.host_gb(16)
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


@pytest.mark.host_gb(16)
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


@pytest.fixture(scope="module")
def tiny_bf16(tmp_path_factory):
    """The tiny model as ``api.load`` gives it: bf16 activations."""
    path = qt.write(tmp_path_factory.mktemp("qwen16") / "tiny.gguf",
                    qt.weights())
    fam = qt.tiny_family()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(registry, "families", lambda: (fam,))
        return api.load(path, log=lambda *_: None)


@pytest.mark.host_gb(16)
@pytest.mark.parametrize("case", list(TINY))
def test_the_tiny_model_in_bf16_is_inside_the_bf16_tolerances(tiny_bf16,
                                                              case):
    """The product dtype on the tiny model: every state and tap of the
    prefill and of a target step is bf16, the velocity is fp32, and it
    sits inside twice the oracle's own bf16 against fp32 gap."""
    arrays, m = committed_fixture(TINY[case])
    tol = m["tolerances"]
    model = tiny_bf16.dit
    assert model.act_dtype == mx.bfloat16
    cond = qg.conditioning(arrays, key="")
    seen = []
    with mx.stream(mx.cpu):
        for tag in "ab":
            lat = mx.array(arrays[f"latents_{tag}"])
            sigma = float(arrays[f"sigma_{tag}"])
            ptaps, ttaps = {}, {}
            prefix = qm.prefill_prefix(model, cond, ptaps)
            v = qm.forward_target(model, lat, sigma, cond, prefix, ttaps)
            states = {**ptaps, **{f"target.{k}": a for k, a in ttaps.items()}}
            for i, a in enumerate(prefix.arrays()):
                states[f"prefix.{i}"] = a
            for i, a in enumerate(qm.modulation(model, sigma).arrays()):
                states[f"modulation.{i}"] = a
            off = {k: str(a.dtype) for k, a in states.items()
                   if a.dtype != mx.bfloat16}
            assert not off, off
            assert v.dtype == mx.float32
            got, want = np.array(v), arrays[f"velocity_{tag}"]
            assert np.array_equal(got, np.array(api.denoise_step(
                tiny_bf16, lat, sigma, cond)))
            rel, cos = fx.rel_l2(got, want), fx.cosine(got, want)
            seen.append(f"{rel:.2e} of {tol[f'velocity_{tag}_rel_l2']:.2e}")
            assert rel <= tol[f"velocity_{tag}_rel_l2"], (tag, rel)
            assert cos >= tol[f"velocity_{tag}_cosine_min"], (tag, cos)
            # Not the fp32 run under another name.
            assert rel > 100 * TOL
    print(f"qwen check3 tiny {case} in bf16: velocities at "
          + " and ".join(seen) + f" of the oracle's fp32, {len(states)} "
          "states and taps bf16")


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


def fault_misses(monkeypatch, model, arrays: dict, name: str, tol) -> dict:
    """The measures the wrong model ``name`` misses. It must miss one by
    SEEN times its gate at least, and none outside the rows its fault
    reaches. ``tol`` is one gate or a function of a measure's name."""
    apply, reach, clean = FAULTS[name]
    gate = tol if callable(tol) else (lambda r: tol)
    layout.clear_caches()
    try:
        with monkeypatch.context() as mp:
            apply(mp, model, arrays)
            missed = qg.missed(qg.step(model, arrays), tol)
    finally:
        layout.clear_caches()
    assert any(v >= SEEN * gate(r) for r, v in missed.items()), (name,
                                                                 missed)
    stray = sorted(r for r in missed if (
        r.removeprefix("free ") in clean if r.startswith("free ")
        else not reach(r)))
    assert not stray, (name, stray)
    return missed


def _fault_id(name: str) -> str:
    """A wrong model's name as a test id, which holds no space."""
    return name.replace(" = ", "").replace(" ", "_")


@pytest.mark.host_gb(16)
@pytest.mark.parametrize("name", list(FAULTS),
                         ids=[_fault_id(n) for n in FAULTS])
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


@pytest.mark.host_gb(16)
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


# The site gate on the tiny model in bf16.

def _tiny_sites(pipe, arrays: dict, routed: dict | None = None
                ) -> tuple[dict, int]:
    """The sites of one prefill and two target steps on the path a
    request takes, and the count of the prefix's segments."""
    model = pipe.dit
    cond = qg.conditioning(arrays, key="")
    sites: dict = {}
    with mx.stream(mx.cpu), qg.block_sites(model, sites, leaves=True,
                                           routed=routed):
        prefix = qm.prefill_prefix(model, cond)
        for tag in "ab":
            mx.eval(qm.forward_target(
                model, mx.array(arrays[f"latents_{tag}"]),
                float(arrays[f"sigma_{tag}"]), cond, prefix))
    return sites, len(layout.prefix_layout(*qm.prefix_key(cond)).segments)


@pytest.mark.host_gb(16)
@pytest.mark.parametrize("case", list(TINY))
def test_the_sites_of_a_tiny_step_are_within_their_gates(tiny_bf16, case):
    """Each op of each block and each linear leaf on the path a request
    takes in bf16 is within 2 n u of the stock op in float32 on the same
    input, and every op a block calls is read."""
    arrays, _ = committed_fixture(TINY[case])
    model = tiny_bf16.dit
    routed: dict = {}
    sites, segments = _tiny_sites(tiny_bf16, arrays, routed)
    assert not qg.missing_ops(model, sites, 1, 2, segments)
    # No kernel takes a head of 16, so every q and k site is the stock op
    # at the stock count, and each has its head row reading.
    qk = {k: v for k, v in qg.op_sites(sites).items() if k[2] == "qk"}
    assert not routed and qk
    assert {g for _, g in qk.values()} == {
        2 * qg.OP_ROUNDINGS["qk"] * qg.SITE_UNIT}
    assert qg.head_sites(sites).keys() == qk.keys()
    assert {k[0] for k in sites if len(k) == 2} == set(
        install.linear_leaves(model))
    worst = qg.worst_by_op(sites)
    print(f"qwen check3 tiny {case} sites: {len(sites)} readings, the "
          "largest over its gate by op: "
          + ", ".join(f"{k} {v:.2f}" for k, v in sorted(worst.items())))
    assert not qg.over_gate(sites), qg.over_gate(sites)
    # Not float32 under another name: a bf16 op rounds.
    assert max(worst.values()) > 0.01


def _a_leaf_off(mp, model):
    leaf = model.transformer_blocks[1].attn.to_v
    real = type(leaf).__call__
    mp.setattr(type(leaf), "__call__", lambda self, x: (
        real(self, x) * 1.05 if self is leaf else real(self, x)))


def _a_head_off(mp, _):
    """The seam's own op, a kernel when a route is in force, with head 1
    of q a twentieth off. The seam returns q as (B, heads, L, D)."""
    real = ops.qk_norm_rope

    def off(q, k, qw, kw, cos, sin, eps):
        q2, k2 = real(q, k, qw, kw, cos, sin, eps)
        scale = mx.array([1.05 if h == 1 else 1.0
                          for h in range(q2.shape[1])]).astype(q2.dtype)
        return q2 * scale[:, None, None], k2

    mp.setattr(ops, "qk_norm_rope", off)


# name: (the wrong production op, the op whose sites must miss)
SITE_FAULTS = {
    "swiglu without its sigmoid": (
        lambda mp, _: mp.setattr(ops, "swiglu", lambda a, b: a * b),
        "swiglu"),
    "modulate without its one": (
        lambda mp, _: mp.setattr(ops, "modulate", lambda x, s: x * s),
        "modulate"),
    "a residual without its gate": (
        lambda mp, _: mp.setattr(ops, "gated_residual",
                                 lambda x, g, y: x + y), "gated_residual"),
    "sdpa without its mask": (
        lambda mp, _: mp.setattr(
            ops, "sdpa", lambda q, k, v, mask=None: attention.sdpa(q, k, v)),
        "sdpa"),
    "rope with its tables swapped": (
        lambda mp, _: mp.setattr(
            ops, "qk_norm_rope", lambda q, k, qw, kw, cos, sin, eps:
            attention.qk_norm_rope(q, k, qw, kw, sin, cos, eps)), "qk"),
    "one head of q a twentieth off": (_a_head_off, "qk"),
    "layer norm at another eps": (
        lambda mp, _: mp.setattr(
            ops, "layer_norm", lambda x, eps: norms.layer_norm(x, 0.5)),
        "layer_norm"),
    "a leaf a twentieth off": (_a_leaf_off, "leaf"),
}


@pytest.mark.host_gb(16)
@pytest.mark.parametrize("name", list(SITE_FAULTS),
                         ids=[_fault_id(n) for n in SITE_FAULTS])
def test_each_wrong_op_fails_the_tiny_site_gate_at_its_own_site(
        tiny_bf16, monkeypatch, name):
    """A site reads an op on the op's own input, so a wrong op is over
    the gate at its own sites and at no other."""
    arrays, _ = committed_fixture(BLOCKS)
    apply, op = SITE_FAULTS[name]
    with monkeypatch.context() as mp:
        apply(mp, tiny_bf16.dit)
        sites, _ = _tiny_sites(tiny_bf16, arrays)
    over = qg.over_gate(sites)
    assert over, name
    assert {qg.site_op(k) for k in over} == {op}, over
    if op == "leaf":
        assert {k[0] for k in over} == {"transformer_blocks.1.attn.to_v"}
    if op == "sdpa":
        # The target rows take no mask.
        assert {k[1] for k in over} == {"prefill"}
    if name.startswith("one head"):
        # Every q and k site misses at its worst head row.
        assert {k[:4] for k in over if len(k) == 5} == {
            k for k in qg.op_sites(sites) if k[2] == "qk"}
    sites, _ = _tiny_sites(tiny_bf16, arrays)
    assert not qg.over_gate(sites)


# Qwen-Image 2.1 Q4_K_M, when the file and its fixtures are there.

REAL = {512: "check3_qwen_image21_512", 1024: "check3_qwen_image21_1024"}
# One step at 512 with one 512 reference in the prefix.
REF = "check3_qwen_image21_ref_512"
# The wrong models whose fault a reference in the prefix widens: its rows
# sit on a frame of their own and read the t = 0 row.
REF_FAULTS = ("rope axes permuted", "rope frame axis last",
              "t = 0 row and sampled row swapped")


def _real_manifest(name: str) -> dict:
    if not (ORACLE / f"{name}.manifest.json").is_file():
        pytest.skip(f"{name}.manifest.json is not committed: run "
                    "scripts/image_oracle/check3_qwen_image21.py on the "
                    "Qwen-Image 2.1 GGUF and commit the manifest it writes")
    return committed(name)


@pytest.fixture(scope="module")
def real(gguf_dir):
    m = _real_manifest(REAL[512])
    want = m["files"]["dit"]
    for p in image_dits(gguf_dir, "qwen_image21"):
        if Path(p).name == want["name"] and os.path.getsize(p) == want["size"]:
            break
    else:
        pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")
    if fx.file_identity(p)["sha256"] != want["sha256"]:
        pytest.fail(f"{p} is not the file the oracle read")
    model, _ = image_loader.load_dit(p, log=lambda *_: None)
    return model


def _row_names(blocks: int) -> list[str]:
    """The names ``check3_rows`` gives its rows for a model of ``blocks``
    blocks."""
    names = ["rope_cos", "rope_sin", "txt_in", "img_in", "joint.in",
             "cached.in", "joint.modulation", "cached.modulation",
             "joint.proj_out", "cached.proj_out"]
    for i in range(blocks):
        names += [f"joint.transformer_blocks.{i}", f"prefill.k.{i}",
                  f"prefill.v.{i}", f"cached.transformer_blocks.{i}"]
    return names


@pytest.mark.host_gb(16)
def test_a_row_with_no_tolerance_is_an_error_of_the_product_gate(tiny):
    """The product gate reads every teacher-forced row the manifest has
    a tolerance for, leaves the rows NOT_HELD names to their other gates,
    and refuses any other row with no tolerance. The names are those
    ``check3_rows`` gives, and the rows left out of the real manifest are
    the 72 of NOT_HELD."""
    arrays, _ = committed_fixture(BLOCKS)
    with mx.stream(mx.cpu):
        tiny_rows = qg.check3_rows(tiny, arrays)
    assert sorted(tiny_rows) == sorted(_row_names(
        len(tiny.transformer_blocks)))
    m = committed(REAL[512])
    gates = m["tolerances"]["rows_rel_l2"]
    names = _row_names(32)
    left = [r for r in names if r not in gates]
    assert len(names) == 138 and len(left) == 72
    assert all(qg.not_held(r) for r in left)
    assert not [r for r in gates if qg.not_held(r)]
    assert set(gates) <= set(names)
    zero = {"rel_l2": 0.0, "cosine": 1.0}
    assert qg.product_misses(m, dict.fromkeys(names, zero), {}) == []
    # A row the manifest lost its tolerance for, and a row of a name the
    # gate does not know.
    row = "cached.transformer_blocks.7"
    thin = {**m, "tolerances": {**m["tolerances"], "rows_rel_l2": {
        r: g for r, g in gates.items() if r != row}}}
    with pytest.raises(LookupError, match="1 teacher-forced rows have no"):
        qg.product_misses(thin, dict.fromkeys(names, zero), {})
    for name in ("prefill.q.3", "prefill.k.x", "joint.out"):
        with pytest.raises(LookupError, match="no tolerance"):
            qg.product_misses(m, {name: zero}, {})
    with pytest.raises(LookupError, match="free-running measure"):
        qg.product_misses(m, {}, {"cache.q.3": zero})
    # A row over its tolerance is a miss, not an error.
    over = {"rel_l2": 2 * gates[row], "cosine": 1.0}
    assert len(qg.product_misses(m, {row: over}, {})) == 1


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
    print("qwen check3 product 512: " + qg.product_reading(m, rows, free))
    misses = qg.product_misses(m, rows, free)
    assert not misses, misses


@pytest.mark.integration
def test_check3_site_gate_at_512(real):
    """Every op of every block and every linear leaf on the path a
    request takes is within its site gate of the stock op in float32 on
    the same input, and that run's velocities pass the product gate."""
    arrays, m = fixture(_real_manifest(REAL[512])["name"])
    sites, routed, heads = {}, {}, {}
    with qg.block_sites(real, sites, leaves=True, routed=routed,
                        heads=heads):
        free = qg.free_run(real, arrays, own_temb=True)
    cond = qg.conditioning(arrays)
    segments = len(layout.prefix_layout(*qm.prefix_key(cond)).segments)
    assert not qg.missing_ops(real, sites, 1, 2, segments)
    # With the qk route in force every q and k site of the real model ran
    # on the kernel and is held to its count, and with the routes off
    # none did.
    qk = {k: v for k, v in qg.op_sites(sites).items() if k[2] == "qk"}
    on = "qk" in kq_routes.active()
    assert routed.keys() == (qk.keys() if on else set()), (
        on, len(routed), len(qk))
    n = bg.KERNEL_ROUNDINGS["qk"] if on else qg.OP_ROUNDINGS["qk"]
    assert {g for _, g in qk.values()} == {2 * n * qg.SITE_UNIT}
    assert qg.head_sites(sites).keys() == qk.keys()
    print("qwen check3 sites 512, q and k: "
          + qg.qk_reading(sites, routed, heads))
    assert {k[0] for k in sites if len(k) == 2} == set(
        install.linear_leaves(real))
    worst = qg.worst_by_op(sites)
    print(f"qwen check3 sites 512: {len(sites)} readings, the largest over "
          "its gate by op: "
          + ", ".join(f"{k} {v:.2f}" for k, v in sorted(worst.items())))
    print("qwen check3 sites 512, that run: "
          + qg.product_reading(m, {}, free))
    assert not qg.over_gate(sites), qg.over_gate(sites)
    assert not qg.product_misses(m, {}, free)


@pytest.mark.integration
def test_check3_one_head_of_q_off_fails_its_head_row_at_512(real,
                                                           monkeypatch):
    """One head of q a twentieth off at the seam, on the kernel when the
    qk route is in force: every q and k site misses at its head row. The
    reading over all rows of a site is printed beside it: one head is a
    thirty-second of q."""
    arrays, _ = fixture(_real_manifest(REAL[512])["name"])
    sites, routed = {}, {}
    with monkeypatch.context() as mp:
        _a_head_off(mp, real)
        with qg.block_sites(real, sites, routed=routed):
            qg.free_run(real, arrays, own_temb=True)
    qk = {k: v for k, v in qg.op_sites(sites).items() if k[2] == "qk"}
    on = "qk" in kq_routes.active()
    assert routed.keys() == (qk.keys() if on else set())
    by_head = qg.head_sites(sites)
    low = min(r / g for r, g in by_head.values())
    top = max(r / g for r, g in qk.values())
    print(f"qwen check3 sites 512, one head of q a twentieth off, "
          f"{len(routed)} of {len(qk)} sites on the kernel: the smallest "
          f"worst head row {low:.2f} of its gate, the largest reading over "
          f"all rows {top:.2f}")
    assert low > 1, low
    assert {qg.site_op(k) for k in qg.over_gate(sites)} == {"qk"}


@pytest.mark.integration
def test_check3_final_velocities_at_1024(real):
    arrays, m = fixture(_real_manifest(REAL[1024])["name"])
    free = qg.free_run(real, arrays, own_temb=True)
    print("qwen check3 product 1024: " + qg.product_reading(m, {}, free))
    misses = qg.product_misses(m, {}, free)
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
        # A step takes the oracle's timestep embedding. The embedder of
        # the real file is two K-quant products, held here from the
        # model's own sinusoid.
        own = qg.timestep_rows(real, arrays)
        assert_all_dense(real, used)
        top = max(got, key=lambda r: got[r]["rel_l2"])
        high = max(own, key=lambda r: own[r]["rel_l2"])
        forced = max((r for r in got if not free_running(r)),
                     key=lambda r: got[r]["rel_l2"])
        print(f"qwen check3 exact 512: {len(got)} measures within "
              f"{got[top]['rel_l2']:.2e} of the oracle, at {top}, the "
              f"teacher-forced rows within {got[forced]['rel_l2']:.2e}, "
              f"at {forced}, and the model's own timestep rows within "
              f"{own[high]['rel_l2']:.2e}, at {high}; gates "
              f"{EXACT_TOL:.0e} and {QWEN_EXACT_FREE_TOL:.0e} free-running")
        assert not qg.exact_missed(got), qg.exact_missed(got)
        assert not qg.missed(own, EXACT_TOL), qg.missed(own, EXACT_TOL)
        for name in FAULTS:
            missed = fault_misses(monkeypatch, real, arrays, name,
                                  qg.exact_gate)
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
          f"{got['velocity_b']['rel_l2']:.2e}, gate "
          f"{QWEN_EXACT_FREE_TOL:.0e}")
    # Both are free-running measures.
    assert not qg.missed(got, QWEN_EXACT_FREE_TOL), got


def _reference(arrays: dict, m: dict):
    """The conditioning of the reference fixture: one reference of the
    target's size, in a prefix longer than the target."""
    cond = qg.conditioning(arrays)
    (ref,) = cond.ref_latents
    assert m["inputs"]["ref_shapes"] == [list(ref.shape[1:])]
    assert list(ref.shape[1:]) == m["inputs"]["target_tokens"]
    assert tuple(cond.image_slots) == tuple(m["inputs"]["image_slots"])
    assert 4 * len(cond.image_slots) == ref.shape[1] * ref.shape[2]
    return cond


def _transposed(arrays: dict) -> dict:
    """The fixture with its reference read with height and width
    swapped."""
    return {**arrays, "ref_latents.0": np.ascontiguousarray(
        arrays["ref_latents.0"].transpose(0, 2, 1))}


@pytest.mark.integration
def test_check3_one_step_with_a_reference_at_512(real):
    """The product gate with one 512 reference in the prefix: one prefill
    and a target step over its K and V, against the oracle's joint
    forward of the whole sequence on the first latents and against its
    cached forward on the second, on every block row, every free-running
    block, the kept K and V and both velocities."""
    arrays, m = fixture(_real_manifest(REF)["name"])
    cond = _reference(arrays, m)
    assert arrays["joint.in"].shape[1] == (
        arrays["features"].shape[0] - len(cond.image_slots)
        + 2 * arrays["cached.in"].shape[1])
    rows = qg.check3_rows(real, arrays)
    free = {**qg.free_run(real, arrays, own_temb=True),
            **qg.timestep_rows(real, arrays)}
    print("qwen check3 product 512 with a reference: "
          + qg.product_reading(m, rows, free))
    misses = qg.product_misses(m, rows, free)
    assert not misses, misses
    # A control: the reference read with its sides swapped misses both
    # velocities.
    wrong = qg.free_run(real, _transposed(arrays))
    vel = {k: v for k, v in wrong.items() if k.startswith("velocity_")}
    tol = m["tolerances"]
    print("qwen check3 product 512 with a reference, the reference "
          "transposed: " + ", ".join(
              f"{k} {g['rel_l2'] / tol[k + '_rel_l2']:.1f} of its gate"
              for k, g in vel.items()))
    assert all(g["rel_l2"] > tol[f"{k}_rel_l2"] for k, g in vel.items()), vel


@pytest.mark.integration
def test_check3_exact_gate_with_a_reference_at_512(real, request,
                                                   monkeypatch):
    """The model with a reference in its prefix matches the oracle on
    every row in the oracle's arithmetic, and the wrong models of
    REF_FAULTS and a transposed reference fail."""
    arrays, m = fixture(_real_manifest(REF)["name"])
    if not exact_process():
        return rerun_exact(request, "qwen check3 exact")
    _reference(arrays, m)
    with exact_products(real) as used:
        got = qg.step(real, arrays)
        own = qg.timestep_rows(real, arrays)
        assert_all_dense(real, used)
        top = max(got, key=lambda r: got[r]["rel_l2"])
        forced = max((r for r in got if not free_running(r)),
                     key=lambda r: got[r]["rel_l2"])
        print(f"qwen check3 exact 512 with a reference: {len(got)} "
              f"measures within {got[top]['rel_l2']:.2e} of the oracle, "
              f"at {top}, the teacher-forced rows within "
              f"{got[forced]['rel_l2']:.2e}, at {forced}; gates "
              f"{EXACT_TOL:.0e} and {QWEN_EXACT_FREE_TOL:.0e} free-running")
        assert not qg.exact_missed(got), qg.exact_missed(got)
        assert not qg.missed(own, EXACT_TOL), qg.missed(own, EXACT_TOL)
        for name in REF_FAULTS:
            missed = fault_misses(monkeypatch, real, arrays, name,
                                  qg.exact_gate)
            high = max(missed, key=missed.get)
            print(f"qwen check3 exact 512 with a reference, {name}: "
                  f"{len(missed)} measures miss, up to {missed[high]:.2e} "
                  f"at {high}")
        missed = qg.exact_missed(qg.step(real, _transposed(arrays)))
        assert {"img_in", "joint.in", "free velocity_a",
                "free velocity_b"} <= set(missed), sorted(missed)[:8]
        assert "txt_in" not in missed and "cached.in" not in missed
        print("qwen check3 exact 512 with a reference, the reference "
              f"transposed: {len(missed)} measures miss, the velocities by "
              f"{missed['free velocity_a']:.2e} and "
              f"{missed['free velocity_b']:.2e}")
