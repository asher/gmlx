"""Check 3 of the image acceptance gate for MiniMax-H3-Image: one denoise
step of the gmlx DiT, row by row, against ComfyUI's MiniMaxH3ImageModel
on the same weights.

scripts/image_oracle/check3_h3_image.py writes the fixtures: every row of
a text to image step in fp32, with the oracle's own bf16 against fp32 gap
of each. The step's positions, RoPE angles, segment rows, time, curve
rows and head segment equal the oracle's.

The tiny cases run ComfyUI's class at a tiny size on random weights at
sigma 1.0, which reads head 0 alone, and at 0.75, which adds head 3 to
it, each as text to image and as an edit with two references of other
sizes than the target and a vision span in the prompt. Their fixtures are
committed, so the whole forward is checked on the CPU tier: in fp32
within FP32_TOL of the oracle, and in bf16 under the product gate, twice
the oracle's gap per row (h3_image_gates.py). A head one grid point off,
head k without head 0, RoPE pairs taken adjacent or the GLU halves
swapped miss the fp32 gate, and on an edit so do reference rows at the
step's time or left unmixed.

On the release file the fixtures hold the step at 512x512 at sigma 1.0
and 0.75, at 768x512 at 0.5327739119529724 and at 1024x1024 at
0.23200887441635132, the last past the GLU kernel route's row floor. The
oracle reads a copy of the file with the `config` key ComfyUI needs for
the PDD heads (the manifest's oracle_dit), every tensor unchanged; gmlx
reads the release file (dit), found under KQUANT_TEST_GGUF_DIR by the
manifest's name, size and sha256 and loaded by the image loader, which
identifies it by discovery. The product gate runs the model as it ships
and holds each row to twice the oracle's gap; the kernel routes in
force must run where they apply. The exact gate runs the model in the
oracle's fp32 arithmetic, every K-quant product dense on its dequantized
weight and every float tensor in float32: the teacher-forced rows within
EXACT_TOL, the free rows and the velocity within the larger of EXACT_TOL
and twice gmlx's own fp32 floor that the manifest records
(GMLX_IMAGE_RECORD_FLOORS=<dir> writes manifests with the floors
measured now). The site gate reads each op of every block and each
linear leaf of the shipped model against the stock op in float32 on the
same input (h3_sites.py), at 512x512 and at 1024x1024 on the GLU kernel
route. Each int8 setting loads the release file through the loader and
runs every real case on the int8 ops of each block, its rows printed
against the product gate and held to a sanity bound (INT8_SANITY). The
rows on the release file need it under KQUANT_TEST_GGUF_DIR and the
fixtures under GMLX_IMAGE_FIXTURES.
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import block_gates as bg
import h3_image_gates as hg
import h3_sites as hs
import qwen_gates as qg
from block_gates import exact_process, kernel_calls, rerun_exact
from image_fixtures import fixture_view
from oracle_fixture import ORACLE, committed, committed_fixture, fixture, fx

from gmlx.image import api, ops
from gmlx.image import loader as image_loader
from gmlx.image.conditioning import Conditioning
from gmlx.image.families.h3_image import family as hi
from gmlx.image.families.h3_image import model as him
from gmlx.image.h3 import blocks
from gmlx.image.ops import kq_routes

TINY = "selftest_h3_image_tiny"
TINY_CASES = (f"{TINY}_s0", f"{TINY}_s3")
TINY_EDITS = (f"{TINY}_edit_s0", f"{TINY}_edit_s3")
TINY_ALL = TINY_CASES + TINY_EDITS
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
    for name in TINY_ALL:
        arrays, m = committed_fixture(name)
        assert wm["inputs"]["config"] == m["inputs"]["config"]
        out[name] = (arrays, m)
    return weights, wm["inputs"]["config"], out


@pytest.mark.parametrize("name", TINY_ALL)
def test_the_tiny_manifests_follow_the_rule(tiny, name):
    hg.assert_manifest_rule(tiny[2][name][1])


def test_the_tiny_cases_read_head_0_alone_and_head_3_beside_it(tiny):
    _, config, cases = tiny
    model = hg.tiny_model(tiny[0], config)
    heads = [model.modulation(float(cases[n][0]["sigma"]), 1,
                              hg.step_target(cases[n][0])).heads
             for n in TINY_CASES]
    assert heads == [(), ((3, 1.0),)]


def test_sample_gives_each_step_the_sigma_it_goes_to(tiny, monkeypatch):
    """A run on ComfyUI's simple schedule at 3 steps, off the grid, hands
    each step the next sigma of the schedule, so each step reads the
    heads and weights ComfyUI reads on it (check 4's fixture)."""
    weights, config, cases = tiny
    arrays, _ = cases[TINY_CASES[0]]
    c4, m4 = committed_fixture("check4_h3_image")
    s = c4["simple_3"]
    # The tiny model runs at 64x64, under the family's least pixels.
    pipe = api.ImagePipeline(dit=hg.tiny_model(weights, config),
                             family=dataclasses.replace(hi.FAMILY,
                                                        min_pixels=None),
                             variant="pruned")
    real = him.H3ImageDiT.modulation
    seen = []

    def spy(self, sigma, batch, target=None):
        mod = real(self, sigma, batch, target)
        seen.append((np.float32(np.reshape(sigma, -1)[0]),
                     np.float32(np.reshape(target, -1)[0]), mod.heads))
        return mod
    monkeypatch.setattr(him.H3ImageDiT, "modulation", spy)
    feats = mx.array(arrays["features"])
    with mx.stream(mx.cpu):
        api.sample(pipe, Conditioning(feats), height=64, width=64,
                   sigmas=s, seed=0, cfg=1.0, neg_skip=())
    # The fixture's weights are those of the heads past head 0.
    want = [tuple(zip(range(max(start, 1), stop), w)) for (start, stop), w
            in zip(c4["heads_simple_3"].tolist(),
                   m4["results"]["head_weights"]["simple_3"])]
    assert [(a, b) for a, b, _ in seen] == list(zip(s[:-1], s[1:]))
    assert [h for *_, h in seen] == want


def test_sample_mixes_the_references_by_the_request_seed(tiny, monkeypatch):
    """An edit's reference rows take the noise of the request's seed, or
    the noise the conditioning holds; noise given without a seed and no
    reference noise is refused. The caption reads the image slots."""
    from gmlx.image.h3.pictures import span_slots, spans_of

    weights, config, cases = tiny
    arrays, _ = cases[TINY_EDITS[0]]
    pipe = api.ImagePipeline(dit=hg.tiny_model(weights, config),
                             family=dataclasses.replace(hi.FAMILY,
                                                        min_pixels=None),
                             variant="pruned")
    seen, slots_seen = [], []
    real_refs, real_cap = him.H3ImageDiT.references, him.H3ImageDiT.caption

    def refs(self, latents, *, seed=None, noise=None):
        seen.append((seed, None if noise is None else
                     [np.array(n) for n in noise]))
        return real_refs(self, latents, seed=seed, noise=noise)

    def cap(self, feats, taps=None, *, slots=()):
        slots_seen.append(tuple(slots))
        return real_cap(self, feats, taps, slots=slots)
    monkeypatch.setattr(him.H3ImageDiT, "references", refs)
    monkeypatch.setattr(him.H3ImageDiT, "caption", cap)
    slots = span_slots(spans_of(arrays["text_tags"]))
    z = tuple(mx.array(arrays[f"ref_latents_{i}"][0]) for i in range(2))
    cond = Conditioning(mx.array(arrays["features"]), ref_latents=z,
                        image_slots=slots)
    _, _, h, w = arrays["latents"].shape
    grid = list(hi.PDD_SIGMAS[:3])
    with mx.stream(mx.cpu):
        api.sample(pipe, cond, height=h * 16, width=w * 16, sigmas=grid,
                   seed=5, cfg=1.0, neg_skip=())
        assert [s for s, _ in seen] == [5, 5] and set(slots_seen) == {slots}
        seen.clear()
        noise = tuple(mx.array(arrays[f"ref_noise_{i}"]) for i in range(2))
        api.sample(pipe, dataclasses.replace(cond, ref_noise=noise),
                   height=h * 16, width=w * 16, sigmas=grid, seed=5,
                   cfg=1.0, neg_skip=())
        assert all(s is None and np.array_equal(n[1], arrays["ref_noise_1"])
                   for s, n in seen)
        with pytest.raises(ValueError, match="pass a seed"):
            api.sample(pipe, cond, height=h * 16, width=w * 16, sigmas=grid,
                       noise=np.zeros((1, 24, h, w), np.float32), cfg=1.0,
                       neg_skip=())


@pytest.mark.parametrize("name", TINY_ALL)
def test_the_tiny_step_lays_out_as_the_oracle(tiny, name):
    weights, config, cases = tiny
    arrays, m = cases[name]
    assert not hg.layout_misses(hg.tiny_model(weights, config), arrays, m)


@pytest.mark.parametrize("name", TINY_ALL)
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


@pytest.mark.parametrize("name", TINY_ALL)
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
    real = hi.grid_point
    mp.setattr(hi, "grid_point",
               lambda s: min(real(s) + 1, len(hi.PDD_SIGMAS) - 1))


def _head_alone(mp):
    """Head k without head 0, as a slice of the bank reads it."""
    real = him.H3ImageDiT.head

    def head(self, x, mod, plan, taps=None):
        out = real(self, x, mod, plan, taps)
        if not mod.heads:
            return out
        return out - real(self, x, dataclasses.replace(mod, heads=()), plan)
    mp.setattr(him.H3ImageDiT, "head", head)


def _refs_at_step_time(mp):
    """Reference rows on the step's time row, not the pinned one."""
    real = him.H3ImageDiT.plan

    def plan(self, *a, **k):
        p = real(self, *a, **k)
        segs = tuple((s, e, r - 3 if r >= 3 else r) for s, e, r in p.segs)
        return dataclasses.replace(p, segs=segs)
    mp.setattr(him.H3ImageDiT, "plan", plan)


def _refs_unmixed(mp):
    """Reference rows without the rule's noise."""
    mp.setattr(him, "augment", lambda rows, rule, seed, noise=None: [
        r.astype(mx.float32) for r in rows])


# Wrong models the float32 gate must fail, with the case each shows on:
# the head rule at a step past the first, the references' rules on the
# edits, the rest on all. Each misses its gate a hundredfold at the
# velocity, and unmixed references fiftyfold at the patch projection's
# rows, where their 0.001 of noise enters the step (83 times its gate
# there and 12 times at the velocity when booked).
FAULTS = {"glu halves swapped": (_glu_halves_swapped, TINY_ALL),
          "rope pairs adjacent": (_rope_pairs_adjacent, TINY_ALL),
          "head one grid point off": (_head_one_off, TINY_ALL),
          "head k without head 0": (_head_alone, TINY_CASES[1:]
                                    + TINY_EDITS[1:]),
          "references at the step's time": (_refs_at_step_time, TINY_EDITS),
          "references unmixed": (_refs_unmixed, TINY_EDITS,
                                 "video_patch_proj", 50)}


@pytest.mark.parametrize("fault", FAULTS)
def test_the_tiny_float32_gate_fails_a_wrong_model(tiny, request,
                                                    monkeypatch, fault):
    if not exact_process():
        return rerun_exact(request, f"h3 image check3 tiny fault {fault}")
    weights, config, cases = tiny
    apply, names, *at = FAULTS[fault]
    row, times = at if at else (hg.VELOCITY, 100)
    apply(monkeypatch)
    for name in names:
        arrays, m = cases[name]
        got = hg.measures(hg.tiny_model(weights, config), arrays, m)
        print(f"h3 image check3 tiny fault {fault} {name}: {row} "
              f"{got[row]:.2e}")
        assert got[row] > times * FP32_TOL, (name, got[row])


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
    model, info = image_loader.load_dit(p, log=lambda *_: None)
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


# Int8 (design section 5 item 5). Each setting loads through the loader,
# which sets the attention and converts the family's 200 block linears,
# and every real case runs the int8 ops in each block of both passes, the
# teacher-forced one too. H3 video ships its int8 with no numeric per-step
# gate; here each case's rows print against the product gate as a record,
# held to INT8_SANITY times the worst row's ratio to its gate measured in
# batch 5, which a dropped scale or a wrong conversion misses by far. Off
# is test_check3_rows.
INT8_WORST = {"attention": 1.36, "linears": 3.31, "all": 3.32}
INT8_SANITY = 1.5


@pytest.mark.integration
@pytest.mark.parametrize("int8", tuple(INT8_WORST))
def test_check3_int8(gguf_dir, monkeypatch, int8):
    import gc
    from collections import Counter

    import mlx_kquant as kq

    from gmlx.image.int8 import int8_parts
    from gmlx.image.ops.int8_linear import Int8Linear

    if not kq.nax_available():
        pytest.skip("int8 runs on a GPU with NAX units")
    attention, linears = int8_parts(int8)
    p = _find(gguf_dir, _manifest(REAL[0])["files"]["dit"])
    counted = []
    gate = image_loader.gate_load
    monkeypatch.setattr(image_loader, "gate_load", lambda n, *a, **k: (
        counted.append(n), gate(n, *a, **k)))
    gc.collect()
    mx.clear_cache()
    before = mx.get_active_memory()
    mx.reset_peak_memory()
    model, _ = image_loader.load_dit(p, int8=int8, log=lambda *_: None)
    load_peak = mx.get_peak_memory() - before
    try:
        # A load that converts its linears peaks within what its gate
        # counts: each K-quant leaf frees as its int8 copy lands.
        print(f"h3 image check3 int8 {int8}: load peak "
              f"{load_peak / 2**30:.2f} GiB of {counted[0] / 2**30:.2f} "
              "counted")
        assert linears != "int8" or load_peak <= counted[0]
        assert (model.attention, model.linears) == (attention, linears)
        assert sum(isinstance(m, Int8Linear) for _, m in
                   model.named_modules()) == (200 if linears == "int8" else 0)
        calls: Counter = Counter()
        for op in ("sdpa_qk8_inputs", "int8_matmul"):
            monkeypatch.setattr(kq, op, lambda *a, _op=getattr(kq, op),
                                _n=op, **k: (calls.update([_n]),
                                             _op(*a, **k))[1])
        passes = 2 * len(model.blocks)
        for name in REAL:
            arrays, m = fixture(_manifest(name)["name"])
            calls.clear()
            got = hg.measures(model, arrays, m)
            assert calls["sdpa_qk8_inputs"] == (
                passes if attention == "int8" else 0)
            if linears == "int8":
                assert calls["int8_matmul"] >= 4 * passes
            else:
                assert calls["int8_matmul"] == 0
            tols = m["tolerances"]
            worst = max(v / hg.gate(tols[k]) for k, v in got.items()
                        if not k.startswith("cosine "))
            print(hg.worst_line(f"h3 image check3 int8 {int8} {name}", got,
                                m) + f", {len(hg.missed(got, tols))} rows "
                  f"over the product gate")
            assert worst <= INT8_SANITY * INT8_WORST[int8], (name, worst)
            del arrays
    finally:
        del model
        gc.collect()
        mx.clear_cache()


# The site gate (h3_sites.py): each op of every block and each linear
# leaf on the path a request takes, against the stock op in float32 on
# the same input, within 2 n u, as every image family's check 3 holds
# it. 512x512 reads every block on the stock feed-forward, and
# 1024x1024 at 1032 rows on the GLU kernel route.
SITE_CASES = (REAL[1], REAL[3])


def _sites(model, arrays: dict) -> tuple:
    """The sites of one free-running production step on the fixture's
    inputs, the kernel sites, the head rows, the leaves called on a
    float32 input, the rows and the GLU sites' worst token rows."""
    sites, routed, heads, skipped, by_row = {}, {}, {}, set(), {}
    with hs.step_sites(model, sites, routed=routed, heads=heads,
                       skipped=skipped, by_row=by_row):
        rows = hg.free_rows(model, arrays)
    return sites, routed, heads, skipped, rows, by_row


def _rows_over(by_row: dict) -> dict:
    """The token row readings over their gate, keyed apart from the site's
    own reading."""
    return {(*k, "row"): v for k, v in qg.over_gate(by_row).items()}


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
@pytest.mark.parametrize("name", SITE_CASES)
def test_check3_site_gate(real, name):
    """Every op of every block and every linear leaf a step calls on the
    path a request takes is within its site gate of the stock op in
    float32 on the same input, the leaves on a float32 input are the ones
    the family keeps in float32, and that run's velocity passes the
    product gate."""
    from gmlx.image.lora import install

    arrays, m = fixture(_manifest(name)["name"])
    model, _ = real
    _, plan = hg.plan_of(model, arrays)
    sites, routed, heads, skipped, rows, by_row = _sites(model, arrays)
    assert skipped == {p for p in install.linear_leaves(model)
                       if hg.keeps_fp32(p)}
    missing = hs.missing_ops(model, sites, plan.layout, routed, skipped,
                             hi.ATTENTION_CHUNK)
    assert not missing, missing
    qk = {k for k in qg.op_sites(sites) if k[2] == "qk"}
    on = "qk" in kq_routes.active()
    assert {k for k, v in routed.items() if v == "qk_norm_rope"} == (
        qk if on else set())
    assert set(qg.head_sites(sites)) == qk
    assert set(by_row) == {k for k, v in routed.items()
                           if v == bg.KERNEL_OPS["glu"]}
    if plan.layout.rows >= kq_routes._GLU_MIN_ROWS and \
            "glu" in kq_routes.active():
        assert by_row, "the GLU route did not run"
    print(site_line(f"h3 image check3 sites {name}", sites, routed))
    print(f"h3 image check3 sites {name}, q and k: "
          + qg.qk_reading(sites, routed, heads, hs.qk_stock_n(model)))
    if by_row:
        worst = max(r / g for r, g in by_row.values())
        print(f"h3 image check3 sites {name}, glu rows: {len(by_row)} "
              f"sites, the worst token row {worst:.2f} of its gate")
    over = {**qg.over_gate(sites), **_rows_over(by_row)}
    assert not over, over
    v = hg.VELOCITY
    rel = hg.rel(rows[v], arrays[v])
    assert rel <= m["tolerances"][v]["rel_l2"], rel


def _swiglu_high(mp):
    real_hidden = blocks._Block.hidden
    mp.setattr(blocks._Block, "hidden",
               lambda self, h: real_hidden(self, h) * 1.05)


def _glu_tail_high(mp):
    """The last 16 rows of the MLP's hidden a tenth high, as a GLU kernel
    with a wrong partial last tile would write them, wherever the kernel
    takes the rows."""
    real_hidden = blocks._Block.hidden

    def hidden(self, h):
        y = real_hidden(self, h)
        n = y.shape[1]
        if n < kq_routes._GLU_MIN_ROWS:
            return y
        scale = mx.concatenate([mx.ones((n - 16,)), mx.full((16,), 1.1)])
        return y * scale.astype(y.dtype)[None, :, None]
    mp.setattr(blocks._Block, "hidden", hidden)


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
    gate at its own sites. Two blocks at 1024x1024 are enough: each block
    reads its own sites, and the kernel routes, the GLU route among them,
    run there."""
    arrays, _ = fixture(_manifest(REAL[3])["name"])
    model, _ = real
    keep = list(model.blocks)
    apply, hit = SITE_FAULTS[name]
    try:
        model.blocks = keep[:2]
        with monkeypatch.context() as mp:
            apply(mp)
            sites, _, _, _, _, by_row = _sites(model, arrays)
    finally:
        model.blocks = keep
    if apply is _glu_tail_high and not by_row:
        assert "glu" not in kq_routes.active(), "the GLU route did not run"
        pytest.skip("the GLU kernel route is not in force")
    over = {**qg.over_gate(sites), **_rows_over(by_row)}
    worst = qg.worst_by_op(sites)
    rows = (f"; glu token rows {max(r / g for r, g in by_row.values()):.2f}"
            if by_row else "")
    print(f"h3 image check3 sites, {name}: {len(over)} readings over the "
          "gate; the largest by op "
          + ", ".join(f"{k} {v:.2f}" for k, v in sorted(worst.items()))
          + rows)
    missed = {qg.site_op(k) for k in over if len(k) != 2}
    assert hit <= missed, (name, sorted(missed))
