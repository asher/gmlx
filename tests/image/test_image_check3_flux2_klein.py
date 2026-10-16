"""Check 3 of the image acceptance gate for FLUX.2 klein: one denoise step
of the gmlx DiT, row by row, against BFL's reference DiT on the same GGUF,
under three gates, for the distilled and the Base 4B checkpoints.

scripts/image_oracle/check3_flux2_klein.py writes the fixtures. Each step
is the first of the sampler, at time 1, where seeded normal latents are on
every trajectory, with 512 rows of seeded normal text features and, in the
r1 fixtures, one reference image at the output size. One more distilled
step is the second of the 4-step schedule at 512 pixels, time 0.958 on
seeded normal latents, where 1000 times the time rounded to bf16 differs
from 1000 times the time itself, in bf16, so the step holds the rounding
the reference makes first. Every row of the step at 512 and the velocity
at 1024 are in fp32, with the oracle's own bf16 against fp32 gap of each.

The product gate runs the model as it ships, in bf16 with the kernel
routes at their default and its own time embedding, and holds each row to
twice the oracle's gap: teacher-forced for each block and the final layer,
free running for the rest. The exact gate runs the model in the oracle's
fp32 arithmetic on the reference's time embedding, every K-quant product
dense on its dequantized weight and every float tensor of the file in
float32. It holds the teacher-forced rows to EXACT_TOL, and the free rows
and the velocity each to the larger of EXACT_TOL and twice gmlx's own fp32
floor that the manifest records. The site gate holds each op of every
block and each linear leaf on the path a request takes to the stock op in
float32 on the same input, within 2 n u (flux2_gates.py).

The rows need the FLUX.2 klein 4B GGUFs under KQUANT_TEST_GGUF_DIR and the
fixtures under GMLX_IMAGE_FIXTURES. They skip until both are there.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import flux2_gates as fg
import qwen_gates as qg
from block_gates import exact_process, rerun_exact
from image_fixtures import image_dits
from oracle_fixture import ORACLE, committed, fixture, fx

from gmlx.image import loader as image_loader
from gmlx.image import ops
from gmlx.image.families.flux2_klein import model as fm
from gmlx.image.ops import kq_routes

# A Mac of 16 GB, as most users run (tests/host_box.py).
pytestmark = pytest.mark.host_gb(16)

ROWS = {"distilled": "check3_flux2_klein_512",
        "distilled r1": "check3_flux2_klein_r1_512",
        "base": "check3_flux2_klein_base_512",
        "base r1": "check3_flux2_klein_base_r1_512",
        "distilled step 1": "check3_flux2_klein_s1_512"}
VELOCITY = {"distilled": "check3_flux2_klein_1024",
            "distilled r1": "check3_flux2_klein_r1_1024",
            "base": "check3_flux2_klein_base_1024"}
DITS = {"distilled": "flux-2-klein-4b-Q4_K_M.gguf",
        "base": "flux-2-klein-base-4b-Q4_K_M.gguf"}
STEPS = {"distilled": 4, "base": 50}
REFERENCE_COMMIT = "50fe5162777813d869182b139e83b10743caef15"
# The case that runs every stream.
FULL = "distilled r1"


def schedule(steps: int, tokens: int) -> list[float]:
    """get_schedule of BFL's flux2 sampling.py at REFERENCE_COMMIT."""
    a1, b1, a2, b2 = 8.73809524e-05, 1.89833333, 0.00016927, 0.45666666
    if tokens > 4300:
        mu = a2 * tokens + b2
    else:
        m200, m10 = a2 * tokens + b2, a1 * tokens + b1
        a = (m200 - m10) / 190.0
        mu = a * steps + m200 - 200.0 * a
    out = []
    for i in range(steps + 1):
        t = 1 - i / steps
        out.append(1.0 if t == 1 else 0.0 if t == 0 else
                   math.exp(mu) / (math.exp(mu) + (1 / t - 1)))
    return out


def _manifest(name: str) -> dict:
    if not (ORACLE / f"{name}.manifest.json").is_file():
        pytest.skip(f"{name}.manifest.json is not committed: run "
                    "scripts/image_oracle/check3_flux2_klein.py on the "
                    "FLUX.2 klein GGUF and commit the manifest it writes")
    return committed(name)


class _Dits:
    """The DiT of a fixture's file, one loaded at a time."""

    def __init__(self, root):
        self.root, self.path, self.model = root, None, None

    def __call__(self, m: dict):
        want = m["files"]["dit"]
        if self.path is not None and Path(self.path).name == want["name"]:
            return self.model, self.path
        self.model = self.path = None
        mx.clear_cache()
        for p in image_dits(self.root, "flux2_klein"):
            if Path(p).name == want["name"] and \
                    os.path.getsize(p) == want["size"]:
                break
        else:
            pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")
        if fx.file_identity(p)["sha256"] != want["sha256"]:
            pytest.fail(f"{p} is not the file the oracle read")
        self.model, _ = image_loader.load_dit(p, log=lambda *_: None)
        self.path = p
        return self.model, p


@pytest.fixture(scope="module")
def dits(gguf_dir):
    return _Dits(gguf_dir)


def _rows(case: str) -> tuple[dict, dict]:
    return fixture(_manifest(ROWS[case])["name"])


def test_the_manifests_follow_the_rule():
    """The committed manifests alone: each step is the first of the
    sampler on the reference's commit, or the second of the 4-step
    schedule for the step 1 case, every row tolerance is twice the gap
    beside it, and the exact gate's free-row and velocity tolerances
    follow their floors."""
    for case, name in [*ROWS.items(), *VELOCITY.items()]:
        m = _manifest(name)
        ins = m["inputs"]
        variant = case.split()[0]
        assert m["files"]["dit"]["name"] == DITS[variant], name
        assert ins["reference_commit"] == REFERENCE_COMMIT, name
        step = 1 if case.endswith("step 1") else 0
        n = (ins["size"] // 16) ** 2
        assert ins["step"] == step, name
        assert ins["schedule"] == f"get_schedule({STEPS[variant]}, {n})"
        assert ins["sigma"] == pytest.approx(
            schedule(STEPS[variant], n)[step], abs=1e-6)
        assert ins["text_rows"] == 512, name
        assert ins["refs"] == (1 if case.endswith("r1") else 0), name
        assert ins["refs"] == 0 or ins["ref_size"] == ins["size"], name
        gap = m["results"]["velocity_bf16_vs_fp32"]
        tol = m["tolerances"]["velocity"]
        assert tol["rel_l2"] == 2 * gap["rel_l2"]
        assert tol["cosine_min"] == pytest.approx(
            1 - 4 * (1 - gap["cosine"]), abs=1e-12)
        # The oracle's own gap is the size of one step's rounding.
        assert gap["rel_l2"] < 0.06, (name, gap)
    for name in ROWS.values():
        rows = _manifest(name)
        for k, g in rows["results"]["rows_bf16_vs_fp32"].items():
            assert rows["tolerances"][k]["rel_l2"] == 2 * g["rel_l2"], k
            assert g["teacher_forced"] == fg.forced_row(k), k
        fg.exact_tols(rows)


@pytest.mark.integration
@pytest.mark.parametrize("case", list(ROWS))
def test_check3_rows_at_512(dits, case):
    arrays, m = _rows(case)
    model, _ = dits(m)
    got = fg.measures(model, arrays, own_time=True)
    out = fg.missed(got, m)
    ratio = {k: v / fg.gate(k, m) for k, v in got.items()}
    top = max(ratio, key=ratio.get)
    free = {k: r for k, r in ratio.items() if k.startswith("free ")}
    ftop = max(free, key=free.get)
    print(f"flux2 check3 512 {case}: {len(ratio)} measures, worst "
          f"{got[top]:.2e} at {top}, {ratio[top]:.2f} of its gate; of the "
          f"{len(free)} blocks run free, {free[ftop]:.2f} of its gate at "
          f"{ftop[len('free '):]}; velocity {got['velocity']:.2e} of gate "
          f"{m['tolerances']['velocity']['rel_l2']:.2e}")
    assert not out, out


@pytest.mark.integration
@pytest.mark.parametrize("case", list(VELOCITY))
def test_check3_velocity_at_1024(dits, case):
    arrays, m = fixture(_manifest(VELOCITY[case])["name"])
    model, _ = dits(m)
    got = fg.free_rows(model, arrays)["velocity"]
    rel = fg.rel(got, arrays["velocity"])
    cos = fx.cosine(got, arrays["velocity"])
    tol = m["tolerances"]["velocity"]
    print(f"flux2 check3 1024 {case}: velocity {rel:.2e} of gate "
          f"{tol['rel_l2']:.2e}, cosine {cos:.6f} floor "
          f"{tol['cosine_min']:.6f}")
    assert rel <= tol["rel_l2"] and cos >= tol["cosine_min"]


@pytest.mark.integration
@pytest.mark.parametrize("case", list(ROWS))
def test_check3_own_time_embedding(case):
    """gmlx's own time embedding against the reference's: equal, or one
    entry one bf16 ulp off where the reference's fp32 exp is an ulp off
    the correctly rounded one."""
    arrays, _ = _rows(case)
    n, ulps = fg.own_time_gap(arrays)
    print(f"flux2 check3 {case}: own time embedding {n} entries off, up "
          f"to {ulps:.0f} bf16 ulp")
    assert n <= 2 and ulps <= 1.0, (n, ulps)


# The exact gate.

def _shift_scale_swapped(mp, model):
    real = fm.DoubleBlock.__call__
    mp.setattr(fm.DoubleBlock, "__call__",
               lambda self, img, txt, mi, mt, ri, rt: real(
                   self, img, txt, ((mi[0][1], mi[0][0], mi[0][2]), mi[1]),
                   mt, ri, rt))


def _glu_halves_swapped(mp, model):
    def hidden(self, mlp, h):
        a, b = mx.split(mlp[0](h), 2, axis=-1)
        return ops.swiglu(b, a)
    mp.setattr(fm.DoubleBlock, "hidden", hidden)


def _reference_at_the_image(mp, model):
    """The reference image's tokens at the image's own positions, where
    the reference puts them at 10 on the first axis."""
    real = fm.image_ids
    mp.setattr(fm, "image_ids", lambda h, w, index=None: real(h, w))


def _text_on_the_first_axis(mp, model):
    """The text tokens' positions on the first axis, where the reference
    puts them on the last."""
    def ids(n):
        out = np.zeros((n, 4), np.float32)
        out[:, 0] = np.arange(n)
        return out
    mp.setattr(fm, "text_ids", ids)


# Wrong models the exact gate must fail. RoPE angles formed in float64,
# as ops.rope_tables forms them, are not one: at the positions of these
# steps, at most 511, they move the rows by less than the gate, and
# test_image_flux2_model.py holds the tables to the reference's at
# positions up to 4095.
EXACT_FAULTS = {"shift and scale swapped": _shift_scale_swapped,
                "glu halves swapped": _glu_halves_swapped,
                "reference at the image's positions": _reference_at_the_image,
                "text on the first axis": _text_on_the_first_axis}


@pytest.mark.integration
@pytest.mark.parametrize("case", list(ROWS))
def test_check3_exact_gate_at_512(dits, request, monkeypatch, case):
    """Every teacher-forced row within EXACT_TOL of the oracle in its
    arithmetic, the free rows and the velocity within the fixture's gates
    from the floors measured now, and each wrong model of EXACT_FAULTS
    over the gate."""
    arrays, m = _rows(case)
    if not exact_process():
        return rerun_exact(request, "flux2 check3 exact")
    model, path = dits(m)
    n = fg.exact_floats(model, path)
    label = f"flux2 check3 exact 512 {case}"
    tol = fg.check_floor(m, fg.exact_free_floor(model, arrays), label)
    got = fg.exact_measures(model, arrays)
    print(fg.exact_line(f"{label}, {n} float tensors in fp32", got, tol))
    over = fg.exact_over(got, tol)
    for name, apply in EXACT_FAULTS.items() if case == FULL else ():
        with monkeypatch.context() as mp:
            apply(mp, model)
            bad = fg.exact_measures(model, arrays)
        line, top = fg.fault_line(f"flux2 check3 exact 512, {name}", bad,
                                  tol)
        print(line)
        assert top > 10, (name, top)
    assert not over, over


# The site gate.

def _sites(model, arrays: dict) -> tuple:
    """The sites of one free-running production step on the fixture's
    inputs, the kernel sites, the head rows, the leaves called on a
    float32 input and the velocity."""
    sites, routed, heads, skipped = {}, {}, {}, set()
    with fg.step_sites(model, sites, routed=routed, heads=heads,
                       skipped=skipped):
        vel = fg.free_rows(model, arrays)["velocity"]
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
def test_check3_site_gate_at_512(dits):
    """Every op of every block and every linear leaf a step calls on the
    path a request takes is within its site gate of the stock op in
    float32 on the same input, and that run's velocity passes the
    product gate."""
    arrays, m = _rows(FULL)
    model, _ = dits(m)
    sites, routed, heads, skipped, vel = _sites(model, arrays)
    missing = fg.missing_ops(model, sites, routed, skipped=skipped)
    assert not missing, missing
    # With the qk route in force every q and k site ran on the kernel and
    # is held to its count, and with the routes off none did.
    qk = {k for k in qg.op_sites(sites) if k[2] == "qk"}
    on = "qk" in kq_routes.active()
    assert {k for k, v in routed.items() if v == "qk_norm_rope"} == (
        qk if on else set())
    assert set(qg.head_sites(sites)) == qk
    print(site_line("flux2 check3 sites 512", sites, routed))
    print("flux2 check3 sites 512, q and k: "
          + qg.qk_reading(sites, routed, heads))
    over = qg.over_gate(sites)
    assert not over, over
    rel = fg.rel(vel, arrays["velocity"])
    assert rel <= m["tolerances"]["velocity"]["rel_l2"], rel


def _adaln_high(mp, model):
    for cls in (fm.DoubleBlock, fm.SingleBlock):
        real = cls.adaln
        mp.setattr(cls, "adaln",
                   lambda self, x, shift, scale, real=real:
                   real(self, x, shift, scale) * 1.05)


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


def _attention_high(mp, model):
    real = ops.sdpa
    mp.setattr(ops, "sdpa", lambda q, k, v, mask=None: real(q, k, v, mask)
               * 1.05)


def _hidden_high(mp, model):
    real = fm.DoubleBlock.hidden
    mp.setattr(fm.DoubleBlock, "hidden",
               lambda self, mlp, h: real(self, mlp, h) * 1.05)


def _swiglu_high(mp, model):
    real = ops.swiglu
    mp.setattr(ops, "swiglu", lambda a, b: real(a, b) * 1.05)


def _gated_residual_high(mp, model):
    real_resid, real_res = ops.resid_fused, ops.gated_residual
    mp.setattr(ops, "resid_fused", lambda x, leaf, r, g:
               real_resid(x, leaf, r, g * 1.05))
    mp.setattr(ops, "gated_residual",
               lambda x, g, y: real_res(x, g * 1.05, y))


# name: (the wrong production op, the ops whose sites must miss)
SITE_FAULTS = {"modulated norm a twentieth high": (_adaln_high, {"adaln"}),
               "one head of q a twentieth off": (_a_head_off, {"qk"}),
               "attention a twentieth high": (_attention_high, {"sdpa"}),
               "feed-forward hidden a twentieth high": (_hidden_high,
                                                        {"glu"}),
               "single block activation a twentieth high": (_swiglu_high,
                                                            {"swiglu"}),
               "gated residual a twentieth high": (_gated_residual_high,
                                                   {"out"})}


@pytest.mark.integration
@pytest.mark.parametrize("name", list(SITE_FAULTS))
def test_check3_each_wrong_op_fails_the_site_gate(dits, monkeypatch, name):
    """A site reads an op on the op's own input, so a wrong op is over the
    gate at its own sites. Each block reads its own sites, so the first
    double block and two single blocks are enough."""
    arrays, m = _rows(FULL)
    model, _ = dits(m)
    keep_double, keep_single = list(model.double_blocks), \
        list(model.single_blocks)
    apply, hit = SITE_FAULTS[name]
    try:
        model.double_blocks = keep_double[:1]
        model.single_blocks = keep_single[:2]
        with monkeypatch.context() as mp:
            apply(mp, model)
            sites, _, _, _, _ = _sites(model, arrays)
    finally:
        model.double_blocks = keep_double
        model.single_blocks = keep_single
    over = qg.over_gate(sites)
    worst = qg.worst_by_op(sites)
    print(f"flux2 check3 sites, {name}: {len(over)} readings over the "
          "gate; the largest by op "
          + ", ".join(f"{k} {v:.2f}" for k, v in sorted(worst.items())))
    missed = {qg.site_op(k) for k in over if len(k) != 2}
    assert hit <= missed, (name, sorted(missed))
