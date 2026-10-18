"""Check 3 of the image acceptance gate for MOSS-Image: one denoise step of
the gmlx DiT, row by row, against the release's MOSSImageTransformer on
the same GGUF, under the plan's three gates.

scripts/image_oracle/check3_moss_image.py writes the fixtures: every row
of the step at 512 and the velocity at 1024 in fp32, with the oracle's
own bf16 against fp32 gap of each.

The product gate runs the model as it ships, in bf16 with the kernel
routes at their default, and holds each row to twice the oracle's gap:
teacher-forced for each text block, each block and the head, free
running for the rest. The exact gate runs the model in the oracle's fp32
arithmetic, every K-quant product dense on its dequantized weight and
every float tensor of the file in float32. It holds the teacher-forced
rows to EXACT_TOL, and the free rows and the velocity each to the larger
of EXACT_TOL and twice gmlx's own fp32 floor that the manifest records.
The site gate holds each op of every block and each linear leaf on the
path a request takes to the stock op in float32 on the same input,
within 2 n u (moss_gates.py).

The rows need the MOSS-Image GGUF under KQUANT_TEST_GGUF_DIR and the
fixtures under GMLX_IMAGE_FIXTURES. They skip until both are there and
the manifests are committed.
"""
from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import pytest

import moss_gates as mg
import qwen_gates as qg
from block_gates import exact_process, rerun_exact
from image_fixtures import image_dits
from oracle_fixture import ORACLE, committed, fixture, fx

from gmlx.image import loader as image_loader
from gmlx.image import ops
from gmlx.image.families.moss_image import model as mm
from gmlx.image.ops import kq_routes

ROWS = "check3_moss_image_512"
VELOCITY = "check3_moss_image_1024"


def _manifest(name: str) -> dict:
    if not (ORACLE / f"{name}.manifest.json").is_file():
        pytest.skip(f"{name}.manifest.json is not committed: run "
                    "scripts/image_oracle/check3_moss_image.py on the "
                    "MOSS-Image GGUF and commit the manifest it writes")
    return committed(name)


@pytest.fixture(scope="module")
def real(gguf_dir):
    m = _manifest(ROWS)
    want = m["files"]["dit"]
    for p in image_dits(gguf_dir, "moss_image"):
        if Path(p).name == want["name"] and os.path.getsize(p) == want["size"]:
            break
    else:
        pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")
    if fx.file_identity(p)["sha256"] != want["sha256"]:
        pytest.fail(f"{p} is not the file the oracle read")
    model, _ = image_loader.load_dit(p, log=lambda *_: None)
    return model, p, list(model.layers)


def _depth(real, m: dict, blocks: int | None = None):
    """The model with the blocks the fixture holds, or the first
    ``blocks``."""
    model, _, all_blocks = real
    n = m["inputs"]["blocks"] if blocks is None else blocks
    assert n <= len(all_blocks)
    model.layers = all_blocks[:n]
    return model


def test_the_manifests_follow_the_rule():
    """The committed manifests alone: every row tolerance is twice the
    gap beside it, the velocity's cosine floor follows its gap, and the
    exact gate's free-row and velocity tolerances follow their floors."""
    for name in (ROWS, VELOCITY):
        m = _manifest(name)
        gap = m["results"]["velocity_bf16_vs_fp32"]
        tol = m["tolerances"]["velocity"]
        assert tol["rel_l2"] == 2 * gap["rel_l2"]
        assert tol["cosine_min"] == pytest.approx(
            1 - 4 * (1 - gap["cosine"]), abs=1e-12)
        assert m["inputs"]["dequant"] == "exact"
    rows = _manifest(ROWS)
    for k, g in rows["results"]["rows_bf16_vs_fp32"].items():
        assert rows["tolerances"][k]["rel_l2"] == 2 * g["rel_l2"], k
        assert g["teacher_forced"] == mg.forced_row(k), k
    mg.exact_tols(rows)


@pytest.mark.integration
def test_check3_rows_at_512(real):
    arrays, m = fixture(_manifest(ROWS)["name"])
    model = _depth(real, m)
    got = mg.measures(model, arrays)
    out = mg.missed(got, m["tolerances"])
    ratio = {k: v / m["tolerances"][k]["rel_l2"] for k, v in got.items()
             if not k.startswith("free ")}
    top = max(ratio, key=ratio.get)
    print(f"moss check3 512: {len(ratio)} rows, worst {got[top]:.2e} at "
          f"{top}, {ratio[top]:.2f} of its gate; velocity "
          f"{got['velocity']:.2e} of gate "
          f"{m['tolerances']['velocity']['rel_l2']:.2e}")
    assert not out, out


@pytest.mark.integration
def test_check3_velocity_at_1024(real):
    arrays, m = fixture(_manifest(VELOCITY)["name"])
    model = _depth(real, m)
    got = mg.free_rows(model, arrays)["velocity"]
    rel = mg.rel(got, arrays["velocity"])
    cos = fx.cosine(got, arrays["velocity"])
    tol = m["tolerances"]["velocity"]
    print(f"moss check3 1024: velocity {rel:.2e} of gate "
          f"{tol['rel_l2']:.2e}, cosine {cos:.6f} floor "
          f"{tol['cosine_min']:.6f}")
    assert rel <= tol["rel_l2"] and cos >= tol["cosine_min"]


# The exact gate.

def _situ_halves_swapped(mp, model):
    real_situ = mm.situ
    mp.setattr(mm, "situ", lambda hid, gate: real_situ(gate, hid))


def _shift_dropped(mp, model):
    real_mod = mm.MossImageDiT.modulation

    def mod(self, sigma, batch):
        got = real_mod(self, sigma, batch)
        b = got.blocks
        return mm.StepModulation((mx.zeros_like(b[0]), *b[1:]), got.shift,
                                 got.scale)
    mp.setattr(mm.MossImageDiT, "modulation", mod)


def _rope_adjacent(mp, model):
    mp.setattr(mm, "interleave", lambda x: x)


# Wrong models the exact gate must fail.
EXACT_FAULTS = {"situ halves swapped": _situ_halves_swapped,
                "pre-attention shift dropped": _shift_dropped,
                "rope pairs adjacent dims": _rope_adjacent}


@pytest.mark.integration
def test_check3_exact_gate_at_512(real, request, monkeypatch):
    """Every teacher-forced row within EXACT_TOL of the oracle in its
    arithmetic, the free rows and the velocity within the fixture's gates
    from the floors measured now, and each wrong model of EXACT_FAULTS
    over the gate."""
    arrays, m = fixture(_manifest(ROWS)["name"])
    if not exact_process():
        return rerun_exact(request, "moss check3 exact")
    model = _depth(real, m)
    n = mg.exact_floats(model, real[1])
    label = "moss check3 exact 512"
    tol = mg.check_floor(m, mg.exact_free_floor(model, arrays), label)
    got = mg.exact_measures(model, arrays)
    print(mg.exact_line(f"{label}, {n} float tensors in fp32", got, tol))
    over = mg.exact_over(got, tol)
    for name, apply in EXACT_FAULTS.items():
        with monkeypatch.context() as mp:
            apply(mp, model)
            bad = mg.exact_measures(model, arrays)
        line, top = mg.fault_line(f"{label}, {name}", bad, tol)
        print(line)
        assert top > 10, (name, top)
    assert not over, over


# The site gate.

def _sites(model, arrays: dict) -> tuple:
    """The sites of one free-running production step on the fixture's
    inputs, the kernel sites, the head rows, the leaves called on a
    float32 input and the velocity."""
    sites, routed, heads, skipped = {}, {}, {}, set()
    with mg.step_sites(model, sites, routed=routed, heads=heads,
                       skipped=skipped):
        vel = mg.free_rows(model, arrays)["velocity"]
    return sites, routed, heads, skipped, vel


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
def test_check3_site_gate_at_512(real):
    """Every op of every block and every linear leaf a step calls on the
    path a request takes is within its site gate of the stock op in
    float32 on the same input, and that run's velocity passes the
    product gate."""
    arrays, m = fixture(_manifest(ROWS)["name"])
    model = _depth(real, m)
    sites, routed, heads, skipped, vel = _sites(model, arrays)
    missing = mg.missing_ops(model, sites, skipped=skipped)
    assert not missing, missing
    qk = {k for k in qg.op_sites(sites) if k[2] == "qk"}
    on = "qk" in kq_routes.active()
    assert {k for k, v in routed.items() if v == "qk_norm_rope"} == (
        qk if on else set())
    assert set(qg.head_sites(sites)) == qk
    print(site_line("moss check3 sites 512", sites, routed))
    print("moss check3 sites 512, q and k: "
          + qg.qk_reading(sites, routed, heads))
    over = qg.over_gate(sites)
    assert not over, over
    rel = mg.rel(vel, arrays["velocity"])
    assert rel <= m["tolerances"]["velocity"]["rel_l2"], rel


def _situ_high(mp, model):
    for cls in (mm.Block, mm.TextBlock):
        real_hidden = cls.hidden
        mp.setattr(cls, "hidden",
                   lambda self, h, f=real_hidden: f(self, h) * 1.05)


def _a_head_off(mp, model):
    """Head 1 of the seam's output a twentieth off, on the kernel when the
    qk route is in force."""
    real_qk = ops.qk_norm_rope

    def off(q, k, qw, kw, cos, sin, eps):
        q2, k2 = real_qk(q, k, qw, kw, cos, sin, eps)
        scale = mx.array([1.05 if h == 1 else 1.0
                          for h in range(q2.shape[1])]).astype(q2.dtype)
        return q2 * scale[:, None, None], k2
    mp.setattr(ops, "qk_norm_rope", off)


def _modulate_high(mp, model):
    real_mod = ops.modulate
    mp.setattr(ops, "modulate", lambda x, s: real_mod(x, s) * 1.05)


def _gated_residual_high(mp, model):
    real_resid, real_res = ops.resid_fused, ops.gated_residual
    mp.setattr(ops, "resid_fused", lambda x, leaf, r, g:
               real_resid(x, leaf, r, g * 1.05))
    mp.setattr(ops, "gated_residual",
               lambda x, g, y: real_res(x, g * 1.05, y))


# name: (the wrong production op, the ops whose sites must miss)
SITE_FAULTS = {"situ a twentieth high": (_situ_high, {"situ"}),
               "one head of q a twentieth off": (_a_head_off, {"qk"}),
               "modulation a twentieth high": (_modulate_high,
                                               {"modulate"}),
               "gated residual a twentieth high": (_gated_residual_high,
                                                   {"gated_residual"})}


@pytest.mark.integration
@pytest.mark.parametrize("name", list(SITE_FAULTS))
def test_check3_each_wrong_op_fails_the_site_gate(real, monkeypatch, name):
    """A site reads an op on the op's own input, so a wrong op is over the
    gate at its own sites. Two blocks are enough: each block reads its
    own sites."""
    arrays, m = fixture(_manifest(ROWS)["name"])
    model = _depth(real, m, blocks=2)
    apply, hit = SITE_FAULTS[name]
    try:
        with monkeypatch.context() as mp:
            apply(mp, model)
            sites, _, _, _, _ = _sites(model, arrays)
    finally:
        _depth(real, m)
    over = qg.over_gate(sites)
    worst = qg.worst_by_op(sites)
    print(f"moss check3 sites, {name}: {len(over)} readings over the "
          "gate; the largest by op "
          + ", ".join(f"{k} {v:.2f}" for k, v in sorted(worst.items())))
    missed = {qg.site_op(k) for k in over if len(k) != 2}
    assert hit <= missed, (name, sorted(missed))
