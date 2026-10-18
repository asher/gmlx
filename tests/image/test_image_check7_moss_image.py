"""Check 7 of the image acceptance gate for MOSS-Image, step half: check
3's step at 512 with a synthetic adapter applied live under a patch
table, against the release's MOSSImageTransformer with the same adapter
applied as PEFT applies it, under the three gates of check 3.

No community MOSS-Image adapter is known.
scripts/image_oracle/check7_moss_image_lora.py writes the adapter from a
seed: one pair on every linear layer of blocks 0, 21 and 41, the fused
proj.ffn_up_gate among them, on every linear layer of text refiner block
1, its fused qkv among them, and on six layers outside the blocks, at
ranks 4 to 16 with an alpha of half the rank, each moving its leaf by
about 8 percent. The fixture holds the pairs, and the test writes them
in a dialect. Every row is in fp32, with the oracle's own bf16 against
fp32 gap of each and check 3's rule.

The product gate holds each row to twice the oracle's gap, and the base
model and the adapter at strength 0.8 miss it. The velocity of the
request path, api.denoise_step with the kernel routes at their default,
passes its gate too. The exact gate runs the step in the oracle's fp32
arithmetic in a child process and holds the teacher-forced rows to
EXACT_TOL, and the free rows and the velocity to the gates the manifest
records from gmlx's floors under the table; each wrong table of
``wrong_tables`` must miss it. The site gate holds each op of every
block and each linear leaf, the adapted leaves included, to the stock op
in float32 on the same input with the table's delta. The delta half runs
on the CPU in test_image_moss_lora_map.py.

Skipped unless the manifest is committed, GMLX_IMAGE_FIXTURES holds its
arrays and KQUANT_TEST_GGUF_DIR the DiT GGUF."""
from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import moss_gates as mg
import qwen_gates as qg
from block_gates import exact_process, rerun_exact
from image_fixtures import image_dits
from lora_fixtures import DIALECTS, write_adapter
from oracle_fixture import ORACLE, committed, fixture, fx

SYNTHETIC = "check7_moss_image_synthetic_512"
SOURCE = "check3_moss_image_512"
# The dialect each gate reads the adapter in: one that stores the alpha,
# and one that does not, with the alpha in its up rows.
EXACT_DIALECT, SITE_DIALECT = "kohya", "diffusers"
FUSED = ("layers.21.proj.ffn_up_gate", "text_embedder.blocks.1.qkv")


def _manifest() -> dict:
    if not (ORACLE / f"{SYNTHETIC}.manifest.json").is_file():
        pytest.skip(f"{SYNTHETIC}.manifest.json is not committed: run "
                    "scripts/image_oracle/check7_moss_image_lora.py and "
                    "commit the manifest it writes")
    return committed(SYNTHETIC)


@pytest.fixture(scope="module")
def synthetic():
    _manifest()
    return fixture(SYNTHETIC)


@pytest.fixture(scope="module")
def dit_path(gguf_dir):
    want = _manifest()["files"]["dit"]
    for p in image_dits(gguf_dir, "moss_image"):
        if Path(p).name == want["name"] and os.path.getsize(p) == want["size"]:
            if fx.file_identity(p)["sha256"] != want["sha256"]:
                pytest.fail(f"{p} is not the file the oracle read")
            return p
    pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")


# The float tensors the exact gate's process put in float32.
_EXACT: dict = {}


@pytest.fixture(scope="module")
def pipe(dit_path):
    from gmlx.image import api

    p = api.load(dit_path, log=lambda *_: None)
    if exact_process():
        # Before a table wraps the leaves, so the names are the file's.
        _EXACT["floats"] = mg.exact_floats(p.dit, dit_path)
    return p


def synthetic_adapter(arrays: dict, m: dict, dialect: str) -> dict:
    """The fixture's pairs as an adapter in the GGUF names, for
    ``dialect``: a dialect that stores no alpha takes it in its up rows."""
    keeps = DIALECTS[dialect][4]
    out = {}
    for name, alpha in m["inputs"]["lora_pair_alphas"].items():
        down, up = arrays[f"down.{name}"], arrays[f"up.{name}"]
        if not keeps:
            up = (up * np.float32(alpha / down.shape[0])).astype(np.float32)
            alpha = None
        out[name] = (down, up, alpha)
    return out


def _adapter(pipe, tmp, dialect: str):
    from gmlx.image import lora

    arrays, m = fixture(SYNTHETIC)
    path = write_adapter(Path(tmp) / f"synthetic-{dialect}.safetensors",
                         dialect, synthetic_adapter(arrays, m, dialect))
    ad = lora.load_adapter(pipe, path)
    assert ad.text == () and len(ad.dit) == m["inputs"]["lora_pairs"]
    return ad


def _entries(ad, strength: float) -> list:
    from gmlx.image import lora

    return lora.table_entries([(ad, strength)])


def _under(pipe, tid: str | None, read, arrays: dict):
    """``read(dit, arrays)`` with the table ``tid`` active."""
    from gmlx.image.lora import install

    if tid is None:
        return read(pipe.dit, arrays)
    prev = install.activate(pipe, tid)
    try:
        return read(pipe.dit, arrays)
    finally:
        install.restore(pipe, prev)


def test_the_manifest_reads_check3s_step_and_follows_the_rule():
    """The committed manifest alone: the fixture takes check 3's step,
    holds a pair for every target it names, the fused weights, the
    refiner and the layers outside the blocks among them, every row
    tolerance is twice the gap beside it, and the exact gate's free-row
    and velocity tolerances follow their floors."""
    src = committed(SOURCE)
    m = _manifest()
    assert m["inputs"]["source"] == SOURCE
    for k in ("sigma", "seed", "size", "blocks", "text_rows"):
        assert m["inputs"][k] == src["inputs"][k], k
    assert m["files"]["dit"] == src["files"]["dit"]
    for k, g in m["results"]["rows_bf16_vs_fp32"].items():
        assert m["tolerances"][k]["rel_l2"] == 2 * g["rel_l2"], k
    mg.exact_tols(m)
    targets = m["inputs"]["targets"]
    assert sorted(m["inputs"]["lora_pair_alphas"]) == targets
    assert len(targets) == m["inputs"]["lora_pairs"] == 29
    assert set(FUSED) | {"final_linear_video", "video_embedder",
                         "shared_block_modulation"} <= set(targets)
    assert {f"{h}.{t}" for t in targets for h in ("down", "up")} <= set(
        m["tensors"])
    assert "lora" not in m["files"]


# The product gate.

@pytest.mark.integration
def test_check7_adapter_step_rows_at_512(synthetic, pipe, tmp_path):
    from gmlx.image import api

    arrays, m = synthetic
    ad = _adapter(pipe, tmp_path, EXACT_DIALECT)
    tid = api.define_patch_table(pipe, _entries(ad, m["inputs"]["strength"]))
    got = _under(pipe, tid, mg.measures, arrays)
    tol = m["tolerances"]
    ratio = {k: v / tol[k]["rel_l2"] for k, v in got.items()
             if not k.startswith("free ")}
    top = max(ratio, key=ratio.get)
    print(f"moss check7 512: {len(ad.dit)} pairs, {len(ratio)} rows, worst "
          f"{got[top]:.2e} at {top}, {ratio[top]:.2f} of its gate; "
          f"velocity {got['velocity']:.2e} of gate "
          f"{tol['velocity']['rel_l2']:.2e}")
    out = mg.missed(got, tol)
    assert not out, out


@pytest.mark.integration
@pytest.mark.parametrize("wrong", ["base", "strength 0.8"])
def test_check7_rows_see_the_adapter_and_its_strength(synthetic, pipe,
                                                      tmp_path, wrong):
    from gmlx.image import api

    arrays, m = synthetic
    tid = None
    if wrong != "base":
        ad = _adapter(pipe, tmp_path, EXACT_DIALECT)
        tid = api.define_patch_table(
            pipe, _entries(ad, 0.8 * m["inputs"]["strength"]))
    out = mg.missed(_under(pipe, tid, mg.measures, arrays), m["tolerances"])
    worst = max(out, key=lambda k: out[k][0] / out[k][1]) if out else None
    print(f"moss check7 {wrong}: {len(out)} rows miss"
          + (f", worst {worst} at {out[worst][0] / out[worst][1]:.1f} of "
             "its gate" if worst else ""))
    assert out


@pytest.mark.integration
def test_check7_velocity_of_the_request_path(synthetic, pipe, tmp_path):
    from gmlx.image import api

    arrays, m = synthetic
    ad = _adapter(pipe, tmp_path, SITE_DIALECT)
    tid = api.define_patch_table(pipe, _entries(ad, m["inputs"]["strength"]))
    v = api.denoise_step(pipe, mx.array(arrays["latents"]),
                         float(arrays["sigma"]),
                         api.Conditioning(mx.array(arrays["features"])),
                         patch_table=tid)
    got = fx.compare(np.array(v), arrays["velocity"])
    tol = m["tolerances"]["velocity"]
    print(f"moss check7 request velocity: rel L2 {got['rel_l2']:.3e} of "
          f"{tol['rel_l2']:.3e}, cosine {got['cosine']:.6f} of "
          f"{tol['cosine_min']:.6f}")
    assert got["rel_l2"] <= tol["rel_l2"]
    assert got["cosine"] >= tol["cosine_min"]


# The exact gate.

def _block(path: str):
    parts = path.split(".")
    return int(parts[1]) if parts[0] == "layers" else None


def wrong_tables(entries: list, strength: float) -> dict:
    """Wrong tables of an adapter's ``entries`` at ``strength``: the
    strength a tenth low, the pairs of the middle adapted block left out
    and of the last, and the middle block's pairs on the block before it,
    which the adapter leaves alone."""
    blocks = sorted({b for p, _, _ in entries
                     if (b := _block(p)) is not None})
    mid, last = blocks[len(blocks) // 2], blocks[-1]
    assert mid - 1 not in blocks

    def scaled(k):
        return [(p, band, [(key, s * k) for key, s in patches])
                for p, band, patches in entries]

    return {
        "strength a tenth low": scaled(0.9),
        f"block {mid} left out": [e for e in entries if _block(e[0]) != mid],
        f"block {last} left out": [e for e in entries
                                    if _block(e[0]) != last],
        f"block {mid} on block {mid - 1}": [
            (p.replace(f"layers.{mid}.", f"layers.{mid - 1}.", 1)
             if _block(p) == mid else p, band, patches)
            for p, band, patches in entries]}


@pytest.mark.integration
def test_check7_exact_gate_at_512(pipe, tmp_path, request):
    """Every measure within its exact gate under the right table, the
    floors measured now within the manifest's, and each wrong table of
    ``wrong_tables`` over the gate."""
    from gmlx.image import api

    arrays, m = fixture(_manifest()["name"])
    if not exact_process():
        return rerun_exact(request, "moss check7 exact")
    assert _EXACT.get("floats", 0) > 0
    ad = _adapter(pipe, tmp_path, EXACT_DIALECT)
    strength = m["inputs"]["strength"]
    entries = _entries(ad, strength)
    tid = api.define_patch_table(pipe, entries)
    label = "moss check7 exact"
    tol = mg.check_floor(m, _under(pipe, tid, mg.exact_free_floor, arrays),
                         label)
    got = _under(pipe, tid, mg.exact_measures, arrays)
    print(mg.exact_line(f"{label}, {len(ad.dit)} pairs", got, tol))
    over = mg.exact_over(got, tol)
    weakest = None
    for fault, table in wrong_tables(entries, strength).items():
        bad = _under(pipe, api.define_patch_table(pipe, table),
                     mg.exact_measures, arrays)
        line, top = mg.fault_line(f"{label}, {fault}", bad, tol)
        print(line)
        assert top > 10, (fault, top)
        if weakest is None or top < weakest[0]:
            weakest = (top, fault)
    print(f"{label}: the weakest wrong table is {weakest[1]} at "
          f"{weakest[0]:.0f} of the gate")
    assert not over, over


# The site gate.

def table_sites(pipe, arrays: dict, tid: str) -> tuple:
    """The sites of one free-running production step under the table
    ``tid``, read against the exact ops with the table's delta."""
    from gmlx.image.lora import install

    built = install.build_table(pipe, tid, pipe.dit.act_dtype, "block")
    sites, routed, heads, skipped = {}, {}, {}, set()
    prev = install.activate(pipe, tid)
    try:
        with mg.step_sites(pipe.dit, sites, routed=routed, heads=heads,
                           built=built, skipped=skipped):
            vel = mg.free_rows(pipe.dit, arrays)["velocity"]
    finally:
        install.restore(pipe, prev)
    return sites, routed, skipped, vel


def _site_setup(pipe, tmp):
    from gmlx.image import api

    arrays, m = fixture(_manifest()["name"])
    entries = _entries(_adapter(pipe, tmp, SITE_DIALECT),
                       m["inputs"]["strength"])
    return arrays, m, entries, api.define_patch_table(pipe, entries)


@pytest.mark.integration
def test_check7_site_gate_at_512(pipe, tmp_path):
    """Every op of every block and every linear leaf a step calls under
    the table, each adapted leaf with its delta, is within its site gate
    of the stock op in float32 on the same input, and that run's velocity
    passes the product gate."""
    arrays, m, entries, tid = _site_setup(pipe, tmp_path)
    sites, routed, skipped, vel = table_sites(pipe, arrays, tid)
    missing = mg.missing_ops(pipe.dit, sites, skipped=skipped)
    assert not missing, missing
    adapted = {p for p, _, _ in entries}
    # An adapted leaf is held to the count of a leaf under its adapter, or
    # to the lora op's own. The time vectors are read past their call,
    # and the exact gate holds them.
    held = {k[0] for k, (_, g) in sites.items() if len(k) == 2
            and g >= 2 * 3 * qg.SITE_UNIT}
    called = adapted - skipped - set(mg.NOT_CALLED)
    assert called <= held, sorted(called - held)[:4]
    assert set(FUSED) <= called
    lora_on = sum(v == "quantized_matmul_lora" for v in routed.values())
    print(f"moss check7 sites: {len(adapted)} adapted leaves, {lora_on} "
          f"calls on the lora op, leaves on a float32 input "
          f"{sorted(skipped)}")
    worst = qg.worst_by_op(sites)
    print(f"moss check7 sites: {len(sites)} readings, the largest over its "
          "gate by op: "
          + ", ".join(f"{k} {v:.2f}" for k, v in sorted(worst.items())))
    over = qg.over_gate(sites)
    assert not over, over
    rel = mg.rel(vel, arrays["velocity"])
    assert rel <= m["tolerances"]["velocity"]["rel_l2"], rel


@pytest.mark.integration
def test_check7_a_leaf_that_drops_its_delta_fails_the_site_gate(
        pipe, tmp_path, monkeypatch):
    """The adapter's leaves of its middle block run their base product
    alone: their sites read the table's delta as error, and no other leaf
    moves past its gate."""
    from gmlx.image.lora import install

    arrays, _, entries, tid = _site_setup(pipe, tmp_path)
    blocks = sorted({b for p, _, _ in entries
                     if (b := _block(p)) is not None})
    mid = blocks[len(blocks) // 2]
    dropped = {p for p, _, _ in entries if _block(p) == mid}
    real = install.ImageLoRALinear.__call__

    def call(self, x):
        return self.base(x) if self._path in dropped else real(self, x)

    monkeypatch.setattr(install.ImageLoRALinear, "__call__", call)
    sites, _, _, _ = table_sites(pipe, arrays, tid)
    over = qg.over_gate(sites)
    leaves = {k[0] for k in over if len(k) == 2}
    low = min(sites[k][0] / sites[k][1] for k in sites
              if len(k) == 2 and k[0] in dropped)
    print(f"moss check7 sites, block {mid}'s deltas dropped: {len(over)} "
          f"readings over the gate, {len(leaves)} of {len(dropped)} dropped "
          f"leaves, the least at {low:.1f} of its gate")
    assert leaves == dropped, sorted(dropped - leaves)
