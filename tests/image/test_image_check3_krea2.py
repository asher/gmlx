"""Check 3 of the image acceptance gate for Krea 2: one denoise step of the
gmlx DiT, row by row, against the diffusers oracle on the same GGUF,
under the plan's three gates.

scripts/image_oracle/check3_krea2.py writes the fixtures. Each takes its
latent and sigma from a sampling trajectory: x_i of check 8's stock bf16
ComfyUI run on the same GGUF, at the step nearest sigma 0.75 (step 4,
sigma 0.7595) and at a late step (step 7, sigma 0.3109), with the
oracle's features of check 8's prompt. A random latent at a middle sigma
is off every trajectory: block 0 grows outlier channels on it, and the
oracle's own bf16 against fp32 velocity gap reads 0.19 at 512 and 0.26
at 1024 there, against 0.012 to 0.04 on the trajectory. Every row of the
step at 512 and the velocity at 1024 are in fp32, with the oracle's own
bf16 against fp32 gap of each.

The product gate runs the model as it ships, in bf16 with the kernel
routes at their default, and holds each row to twice the oracle's gap:
teacher-forced for each block and the final layer, free running for the
rest. The exact gate runs the model in the oracle's fp32 arithmetic,
every K-quant product dense on its dequantized weight and every float
tensor of the file in float32. It holds the teacher-forced rows to
EXACT_TOL, and the free rows and the velocity each to the larger of
EXACT_TOL and twice gmlx's own fp32 floor that the manifest records. The
site gate holds each op of every block and each linear leaf on the path
a request takes to the stock op in float32 on the same input, within 2 n
u (krea2_gates.py).

The rows need the Krea 2 GGUF under KQUANT_TEST_GGUF_DIR and the
fixtures under GMLX_IMAGE_FIXTURES. They skip until both are there.
"""
from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import pytest

import krea2_gates as kg
import qwen_gates as qg
from block_gates import exact_process, rerun_exact
from image_fixtures import image_dits
from oracle_fixture import ORACLE, committed, fixture, fx

from gmlx.image import loader as image_loader
from gmlx.image import ops
from gmlx.image.families.krea2 import model as km
from gmlx.image.ops import kq_routes

# A Mac of 24 GB: the module peaks at 14.3 GiB, past the 14 GiB working
# set of a 16 GB Mac (tests/host_box.py).
pytestmark = pytest.mark.host_gb(24)

# The step nearest sigma 0.75 and a late step of check 8's trajectory.
STEPS = (4, 7)
ROWS = {s: f"check3_krea2_s{s}_512" for s in STEPS}
VELOCITY = {s: f"check3_krea2_s{s}_1024" for s in STEPS}


def _manifest(name: str) -> dict:
    if not (ORACLE / f"{name}.manifest.json").is_file():
        pytest.skip(f"{name}.manifest.json is not committed: run "
                    "scripts/image_oracle/check3_krea2.py on the Krea 2 "
                    "GGUF and commit the manifest it writes")
    return committed(name)


@pytest.fixture(scope="module")
def real(gguf_dir):
    m = _manifest(ROWS[STEPS[0]])
    want = m["files"]["dit"]
    for p in image_dits(gguf_dir, "krea2"):
        if Path(p).name == want["name"] and os.path.getsize(p) == want["size"]:
            break
    else:
        pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")
    if fx.file_identity(p)["sha256"] != want["sha256"]:
        pytest.fail(f"{p} is not the file the oracle read")
    model, _ = image_loader.load_dit(p, log=lambda *_: None)
    return model, p, list(model.blocks)


def _depth(real, m: dict, blocks: int | None = None):
    """The model with the blocks the fixture holds, or the first
    ``blocks``."""
    model, _, all_blocks = real
    n = m["inputs"]["blocks"] if blocks is None else blocks
    assert n <= len(all_blocks)
    model.blocks = all_blocks[:n]
    return model


def test_the_manifests_take_a_trajectory_and_follow_the_rule():
    """The committed manifests alone: each fixture takes x_i and sigma_i
    of check 8's trajectory at its step, with the features of check 8's
    prompt, every row tolerance is twice the gap beside it, and the exact
    gate's free-row and velocity tolerances follow their floors."""
    for step in STEPS:
        for name in (ROWS[step], VELOCITY[step]):
            m = _manifest(name)
            ins = m["inputs"]
            assert ins["step"] == step and "seed" not in ins, name
            assert m["files"]["latents"]["name"].startswith(
                "check8_stock16_"), name
            assert m["files"]["features"]["name"] == \
                "check2_krea2_check8.npz", name
            gap = m["results"]["velocity_bf16_vs_fp32"]
            tol = m["tolerances"]["velocity"]
            assert tol["rel_l2"] == 2 * gap["rel_l2"]
            assert tol["cosine_min"] == pytest.approx(
                1 - 4 * (1 - gap["cosine"]), abs=1e-12)
            # On the trajectory the oracle's own gap is the size of one
            # step's rounding, not that of the outlier channels a random
            # latent grows.
            assert gap["rel_l2"] < 0.06, (name, gap)
        rows = _manifest(ROWS[step])
        for k, g in rows["results"]["rows_bf16_vs_fp32"].items():
            assert rows["tolerances"][k]["rel_l2"] == 2 * g["rel_l2"], k
        kg.exact_tols(rows)
        assert (rows["inputs"]["sigma"]
                == _manifest(VELOCITY[step])["inputs"]["sigma"])


@pytest.mark.integration
@pytest.mark.parametrize("step", STEPS)
def test_check3_rows_at_512(real, step):
    arrays, m = fixture(_manifest(ROWS[step])["name"])
    model = _depth(real, m)
    got = kg.measures(model, arrays)
    out = kg.missed(got, m["tolerances"])
    ratio = {k: v / m["tolerances"][k]["rel_l2"] for k, v in got.items()
             if not k.startswith("free ")}
    top = max(ratio, key=ratio.get)
    print(f"krea2 check3 512 step {step}: {len(ratio)} rows, worst "
          f"{got[top]:.2e} at {top}, {ratio[top]:.2f} of its gate; velocity "
          f"{got['velocity']:.2e} of gate "
          f"{m['tolerances']['velocity']['rel_l2']:.2e}")
    assert not out, out


@pytest.mark.integration
@pytest.mark.parametrize("step", STEPS)
def test_check3_velocity_at_1024(real, step):
    arrays, m = fixture(_manifest(VELOCITY[step])["name"])
    model = _depth(real, m)
    got = kg.free_rows(model, arrays)["velocity"]
    rel = kg.rel(got, arrays["velocity"])
    cos = fx.cosine(got, arrays["velocity"])
    tol = m["tolerances"]["velocity"]
    print(f"krea2 check3 1024 step {step}: velocity {rel:.2e} of gate "
          f"{tol['rel_l2']:.2e}, cosine {cos:.6f} floor "
          f"{tol['cosine_min']:.6f}")
    assert rel <= tol["rel_l2"] and cos >= tol["cosine_min"]


# The exact gate.

def _gate_and_up_swapped(mp, model):
    real_hidden = km.SwiGLU.hidden

    def hidden(self, x):
        self.gate, self.up = self.up, self.gate
        try:
            return real_hidden(self, x)
        finally:
            self.gate, self.up = self.up, self.gate
    mp.setattr(km.SwiGLU, "hidden", hidden)


def _shift_dropped(mp, model):
    real_mod = km.Block.modulation

    def mod(self, tvec):
        a = real_mod(self, tvec)
        return (a[0], mx.zeros_like(a[1]), *a[2:])
    mp.setattr(km.Block, "modulation", mod)


def _gate_on_rows(mp, model):
    """The attention output gate read from the attention rows instead of
    the block input."""
    def gated(self, x, o):
        b, _, n, _ = o.shape
        rows = o.transpose(0, 2, 1, 3).reshape(b, n, -1)
        return rows * mx.sigmoid(self.gate(rows))
    mp.setattr(km.Attention, "gated", gated)


# Wrong models the exact gate must fail.
EXACT_FAULTS = {"gate and up swapped": _gate_and_up_swapped,
                "pre-attention shift dropped": _shift_dropped,
                "output gate on the attention rows": _gate_on_rows}


# Free rows and the velocity take max(EXACT_TOL, 2 x gmlx's own fp32
# floor), krea2_gates.py: block 0's input carries channels 122 times its
# rms, and blocks 12 to 25 amplify a summation-order change at its output
# about 50 times.
@pytest.mark.integration
@pytest.mark.parametrize("step", STEPS)
def test_check3_exact_gate_at_512(real, request, monkeypatch, step):
    """Every teacher-forced row within EXACT_TOL of the oracle in its
    arithmetic, the free rows and the velocity within the fixture's gates
    from the floors measured now, and each wrong model of EXACT_FAULTS
    over the gate."""
    arrays, m = fixture(_manifest(ROWS[step])["name"])
    if not exact_process():
        return rerun_exact(request, "krea2 check3 exact")
    model = _depth(real, m)
    n = kg.exact_floats(model, real[1])
    label = f"krea2 check3 exact 512 step {step}"
    tol = kg.check_floor(m, kg.exact_free_floor(model, arrays), label)
    got = kg.exact_measures(model, arrays)
    print(kg.exact_line(f"{label}, {n} float tensors in fp32", got, tol))
    over = kg.exact_over(got, tol)
    for name, apply in EXACT_FAULTS.items() if step == STEPS[0] else ():
        with monkeypatch.context() as mp:
            apply(mp, model)
            bad = kg.exact_measures(model, arrays)
        line, top = kg.fault_line(f"krea2 check3 exact 512, {name}", bad,
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
    with kg.step_sites(model, sites, routed=routed, heads=heads,
                       skipped=skipped):
        vel = kg.free_rows(model, arrays)["velocity"]
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
    arrays, m = fixture(_manifest(ROWS[STEPS[0]])["name"])
    model = _depth(real, m)
    sites, routed, heads, skipped, vel = _sites(model, arrays)
    missing = kg.missing_ops(model, sites, routed, skipped=skipped)
    assert not missing, missing
    # With the qk route in force every q and k site ran on the kernel and
    # is held to its count, and with the routes off none did.
    qk = {k for k in qg.op_sites(sites) if k[2] == "qk"}
    on = "qk" in kq_routes.active()
    assert {k for k, v in routed.items() if v == "qk_norm_rope"} == (
        qk if on else set())
    assert set(qg.head_sites(sites)) == qk
    print(site_line("krea2 check3 sites 512", sites, routed))
    print("krea2 check3 sites 512, q and k: "
          + qg.qk_reading(sites, routed, heads))
    over = qg.over_gate(sites)
    assert not over, over
    rel = kg.rel(vel, arrays["velocity"])
    assert rel <= m["tolerances"]["velocity"]["rel_l2"], rel


def _gate_scaled(mp, model):
    real_gated = km.Attention.gated
    mp.setattr(km.Attention, "gated",
               lambda self, x, o: real_gated(self, x, o) * 1.05)


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


def _swiglu_without_sigmoid(mp, model):
    real_hidden = km.SwiGLU.hidden

    def hidden(self, x):
        return real_hidden(self, x) / mx.sigmoid(self.gate(x))
    mp.setattr(km.SwiGLU, "hidden", hidden)


def _gated_residual_high(mp, model):
    real_resid, real_res = ops.resid_fused, ops.gated_residual
    mp.setattr(ops, "resid_fused", lambda x, leaf, r, g:
               real_resid(x, leaf, r, g * 1.05))
    mp.setattr(ops, "gated_residual",
               lambda x, g, y: real_res(x, g * 1.05, y))


# name: (the wrong production op, the ops whose sites must miss)
SITE_FAULTS = {"attention output gate scaled": (_gate_scaled, {"gated"}),
               "one head of q a twentieth off": (_a_head_off, {"qk"}),
               "swiglu without its sigmoid": (_swiglu_without_sigmoid,
                                              {"glu"}),
               "gated residual a twentieth high": (_gated_residual_high,
                                                   {"gated_residual"})}


@pytest.mark.integration
@pytest.mark.parametrize("name", list(SITE_FAULTS))
def test_check3_each_wrong_op_fails_the_site_gate(real, monkeypatch, name):
    """A site reads an op on the op's own input, so a wrong op is over the
    gate at its own sites. Two blocks are enough: each block reads its
    own sites."""
    arrays, m = fixture(_manifest(ROWS[STEPS[0]])["name"])
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
    print(f"krea2 check3 sites, {name}: {len(over)} readings over the "
          "gate; the largest by op "
          + ", ".join(f"{k} {v:.2f}" for k, v in sorted(worst.items())))
    missed = {qg.site_op(k) for k in over if len(k) != 2}
    assert hit <= missed, (name, sorted(missed))
