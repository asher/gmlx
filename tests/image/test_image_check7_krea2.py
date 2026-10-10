"""Check 7 of the image acceptance gate for Krea 2, step half: check 3's
step at 512 with an adapter applied live under a patch table, against
the diffusers oracle with the same adapter applied as PEFT applies it,
under the plan's three gates.

scripts/image_oracle/check7_krea2_lora.py writes two fixtures on check
3's on-trajectory step at 512 (step 4 of check 8's run): one with a
Comfy-Org Krea 2 LoRA, and one with a synthetic adapter that the script
writes from a seed, one pair on every read linear layer of blocks 0, 13
and 27, of a layerwise and a refiner text block and of three layers
outside the blocks, at ranks 4 to 16 with an alpha of half the rank,
each moving its leaf by about 8 percent. The synthetic fixture holds its pairs, and the test writes them in each
dialect. Every row is in fp32, with the oracle's own bf16 against fp32
gap of each and check 3's rule.

The product gate holds each row to twice the oracle's gap, and the base
model and the adapter at strength 0.8 miss it. The velocity of the
request path, api.denoise_step with the kernel routes at their default,
passes its gate too. The exact gate runs the step in the oracle's fp32
arithmetic in a child process and holds the teacher-forced rows and the
velocity to EXACT_TOL and the free rows to the gate of check 3's rule
from the floor each manifest records; it is the gate that holds the table, and each wrong table of
``wrong_tables`` must miss it. The site gate holds each op of every
block and each linear leaf, the leaves under the adapter included, to
the stock op in float32 on the same input with the table's delta. The
delta half runs on the CPU in test_image_krea2_lora_map.py.

Skipped unless the manifests are committed, GMLX_IMAGE_FIXTURES holds
their arrays, KQUANT_TEST_GGUF_DIR the DiT GGUF and, for the Comfy-Org
fixture, GMLX_TEST_LORA_DIR the adapter the oracle read."""
from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import krea2_gates as kg
import qwen_gates as qg
from block_gates import exact_process, rerun_exact
from image_fixtures import image_dits
from lora_fixtures import DIALECTS, write_adapter
from oracle_fixture import ORACLE, committed, fixture, fx

pytestmark = pytest.mark.integration

COMFY = "check7_krea2_comfy_512"
SYNTHETIC = "check7_krea2_synthetic_512"
SOURCE = "check3_krea2_s4_512"
LORA_ENV = "GMLX_TEST_LORA_DIR"
# The dialect each gate reads the synthetic adapter in: one that stores
# the alpha, and one that does not, with the alpha in its up rows.
EXACT_DIALECT, SITE_DIALECT = "kohya", "diffusers"


def _manifest(name: str) -> dict:
    if not (ORACLE / f"{name}.manifest.json").is_file():
        pytest.skip(f"{name}.manifest.json is not committed: run "
                    "scripts/image_oracle/check7_krea2_lora.py and commit "
                    "the manifest it writes")
    return committed(name)


def _identified(p, want: dict) -> bool:
    if Path(p).name != want["name"] or os.path.getsize(p) != want["size"]:
        return False
    if fx.file_identity(p)["sha256"] != want["sha256"]:
        pytest.fail(f"{p} is not the file the oracle read")
    return True


@pytest.fixture(scope="module")
def check7():
    _manifest(COMFY)
    return fixture(COMFY)


@pytest.fixture(scope="module")
def synthetic():
    _manifest(SYNTHETIC)
    return fixture(SYNTHETIC)


@pytest.fixture(scope="module")
def lora_path(check7):
    want = check7[1]["files"]["lora"]
    root = os.environ.get(LORA_ENV)
    if not root:
        pytest.skip(f"set {LORA_ENV} to the folder of {want['name']}")
    for p in sorted(Path(root).expanduser().rglob(want["name"])):
        if _identified(p, want):
            return str(p)
    pytest.skip(f"{want['name']} is not under {LORA_ENV}")


@pytest.fixture(scope="module")
def dit_path(gguf_dir):
    want = _manifest(SYNTHETIC)["files"]["dit"]
    for p in image_dits(gguf_dir, "krea2"):
        if _identified(p, want):
            return p
    pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")


# The float tensors the exact gate's process put in float32.
_EXACT: dict = {}


@pytest.fixture(scope="module")
def pipe(dit_path):
    from gmlx.image import api

    p = api.load(dit_path, log=lambda *_: None)
    if exact_process():
        # The exact gate's process: the file's float tensors in float32,
        # before a table wraps the leaves.
        _EXACT["floats"] = kg.exact_floats(p.dit, dit_path)
    return p


def synthetic_adapter(arrays: dict, m: dict, dialect: str) -> dict:
    """The fixture's pairs as an adapter in the diffusers names, for
    ``dialect``: a dialect that stores no alpha takes it in its up rows."""
    alphas = m["inputs"]["lora_pair_alphas"]
    keeps = DIALECTS[dialect][4]
    out = {}
    for name, alpha in alphas.items():
        down, up = arrays[f"down.{name}"], arrays[f"up.{name}"]
        if not keeps:
            up = (up * np.float32(alpha / down.shape[0])).astype(np.float32)
            alpha = None
        out[name] = (down, up, alpha)
    return out


def _adapter(pipe, kind: str, tmp, lora_path=None, dialect=None):
    from gmlx.image import lora

    if kind == "comfy":
        return lora.load_adapter(pipe, lora_path)
    arrays, m = fixture(SYNTHETIC)
    path = write_adapter(Path(tmp) / f"synthetic-{dialect}.safetensors",
                         dialect, synthetic_adapter(arrays, m, dialect))
    return lora.load_adapter(pipe, path)


def _entries(ad, strength: float) -> list:
    from gmlx.image import lora

    return lora.table_entries([(ad, strength)])


def _table(pipe, path: str, strength: float) -> str:
    from gmlx.image import api, lora

    ad = lora.load_adapter(pipe, path)
    assert ad.text == ()
    return api.define_patch_table(pipe, _entries(ad, strength))


def _rows(pipe, arrays: dict, tid: str | None) -> dict:
    from gmlx.image.lora import install

    prev = install.activate(pipe, tid) if tid is not None else None
    try:
        return kg.measures(pipe.dit, arrays)
    finally:
        if tid is not None:
            install.restore(pipe, prev)


def test_the_manifests_read_the_trajectory_step_and_follow_the_rule():
    """The committed manifests alone: both fixtures take check 3's
    on-trajectory step, the synthetic one holds a pair for every target
    it names, every row tolerance is twice the gap beside it, and the
    exact gate's free-row tolerance follows its floor."""
    src = committed(SOURCE)
    for name in (COMFY, SYNTHETIC):
        m = _manifest(name)
        assert m["inputs"]["source"] == SOURCE
        assert m["inputs"]["step"] == src["inputs"]["step"]
        assert m["inputs"]["sigma"] == src["inputs"]["sigma"]
        assert m["files"]["dit"] == src["files"]["dit"]
        for k, g in m["results"]["rows_bf16_vs_fp32"].items():
            assert m["tolerances"][k]["rel_l2"] == 2 * g["rel_l2"], k
        kg.exact_free_tol(m)
    m = _manifest(SYNTHETIC)
    targets = m["inputs"]["targets"]
    assert sorted(m["inputs"]["lora_pair_alphas"]) == targets
    assert {f"{h}.{t}" for t in targets for h in ("down", "up")} <= set(
        m["tensors"])
    assert "lora" not in m["files"]


# The product gate.

def test_check7_adapter_step_rows_at_512(check7, pipe, lora_path):
    arrays, m = check7
    tol = m["tolerances"]
    ad_n = m["inputs"]["lora_pairs"]
    from gmlx.image import lora

    assert len(lora.load_adapter(pipe, lora_path).dit) == ad_n
    got = _rows(pipe, arrays, _table(pipe, lora_path,
                                     m["inputs"]["strength"]))
    out = kg.missed(got, tol)
    ratio = {k: v / tol[k]["rel_l2"] for k, v in got.items()
             if not k.startswith("free ")}
    top = max(ratio, key=ratio.get)
    print(f"krea2 check7 512: {ad_n} pairs, {len(ratio)} rows, worst "
          f"{got[top]:.2e} at {top}, {ratio[top]:.2f} of its gate; "
          f"velocity {got['velocity']:.2e} of gate "
          f"{tol['velocity']['rel_l2']:.2e}")
    assert not out, out


@pytest.mark.parametrize("wrong", ["base", "strength 0.8"])
def test_check7_rows_see_the_adapter_and_its_strength(check7, pipe,
                                                      lora_path, wrong):
    arrays, m = check7
    tid = None if wrong == "base" else _table(
        pipe, lora_path, 0.8 * m["inputs"]["strength"])
    got = _rows(pipe, arrays, tid)
    out = kg.missed(got, m["tolerances"])
    worst = max(out, key=lambda k: out[k][0] / out[k][1]) if out else None
    print(f"krea2 check7 {wrong}: {len(out)} rows miss"
          + (f", worst {worst} at {out[worst][0] / out[worst][1]:.1f} of "
             "its gate" if worst else ""))
    assert out


def test_check7_velocity_of_the_request_path(check7, pipe, lora_path):
    from gmlx.image import api

    arrays, m = check7
    tid = _table(pipe, lora_path, m["inputs"]["strength"])
    v = api.denoise_step(pipe, mx.array(arrays["latents"]),
                         float(arrays["sigma"]),
                         api.Conditioning(mx.array(arrays["features"])),
                         patch_table=tid)
    got = fx.compare(np.array(v), arrays["velocity"])
    tol = m["tolerances"]["velocity"]
    print(f"krea2 check7 request velocity: rel L2 {got['rel_l2']:.3e} of "
          f"{tol['rel_l2']:.3e}, cosine {got['cosine']:.6f} of "
          f"{tol['cosine_min']:.6f}")
    assert got["rel_l2"] <= tol["rel_l2"]
    assert got["cosine"] >= tol["cosine_min"]


def test_check7_synthetic_adapter_step_rows_at_512(synthetic, pipe,
                                                   tmp_path):
    """The product gate on the synthetic adapter, written in a dialect
    that stores its alpha."""
    from gmlx.image import api

    arrays, m = synthetic
    ad = _adapter(pipe, "synthetic", tmp_path, dialect=EXACT_DIALECT)
    assert len(ad.dit) == m["inputs"]["lora_pairs"]
    tid = api.define_patch_table(pipe, _entries(ad, m["inputs"]["strength"]))
    got = _rows(pipe, arrays, tid)
    tol = m["tolerances"]
    ratio = {k: v / tol[k]["rel_l2"] for k, v in got.items()
             if not k.startswith("free ")}
    top = max(ratio, key=ratio.get)
    print(f"krea2 check7 synthetic 512: {len(ad.dit)} pairs, {len(ratio)} "
          f"rows, worst {got[top]:.2e} at {top}, {ratio[top]:.2f} of its "
          f"gate; velocity {got['velocity']:.2e} of gate "
          f"{tol['velocity']['rel_l2']:.2e}")
    out = kg.missed(got, tol)
    assert not out, out


# The exact gate.

def _block(path: str):
    parts = path.split(".")
    return int(parts[1]) if parts[0] == "blocks" else None


def wrong_tables(entries: list, strength: float, moved: bool) -> dict:
    """Wrong tables of an adapter's ``entries`` at ``strength``: the
    strength a tenth low, the pairs of the middle adapted block left out
    and of the last, and with ``moved`` the middle block's pairs on the
    block before it, which the adapter leaves alone."""
    blocks = sorted({b for p, _, _ in entries
                     if (b := _block(p)) is not None})
    mid, last = blocks[len(blocks) // 2], blocks[-1]

    def scaled(k):
        return [(p, band, [(key, s * k) for key, s in patches])
                for p, band, patches in entries]

    out = {"strength a tenth low": scaled(0.9),
           f"block {mid} left out": [e for e in entries
                                      if _block(e[0]) != mid],
           f"block {last} left out": [e for e in entries
                                       if _block(e[0]) != last]}
    if moved:
        assert mid - 1 not in blocks
        out[f"block {mid} on block {mid - 1}"] = [
            (p.replace(f"blocks.{mid}.", f"blocks.{mid - 1}.", 1)
             if _block(p) == mid else p, band, patches)
            for p, band, patches in entries]
    return out


@pytest.mark.parametrize("kind", ["comfy", "synthetic"])
def test_check7_exact_gate_at_512(pipe, tmp_path, request, kind):
    """Every measure within its exact gate under the right table, the
    free-row gate from the floor measured now, and each wrong table of
    ``wrong_tables`` over the gate."""
    from gmlx.image import api

    name = COMFY if kind == "comfy" else SYNTHETIC
    arrays, m = fixture(_manifest(name)["name"])
    lp = request.getfixturevalue("lora_path") if kind == "comfy" else None
    if not exact_process():
        return rerun_exact(request, "krea2 check7 exact")
    assert _EXACT.get("floats", 0) > 0
    ad = _adapter(pipe, kind, tmp_path, lp, EXACT_DIALECT)
    strength = m["inputs"]["strength"]
    entries = _entries(ad, strength)
    tid = api.define_patch_table(pipe, entries)
    label = f"krea2 check7 exact {kind}"
    kg.check_floor(m, _under(pipe, tid, kg.exact_free_floor, arrays), label)
    tol = kg.exact_free_tol(m)
    got = _under(pipe, tid, kg.exact_measures, arrays)
    print(kg.exact_line(f"{label}, {len(ad.dit)} pairs", got, tol))
    over = kg.exact_over(got, tol)
    weakest = None
    for fault, table in wrong_tables(entries, strength,
                                     moved=kind == "synthetic").items():
        bad = _under(pipe, api.define_patch_table(pipe, table),
                     kg.exact_measures, arrays)
        line, top = kg.fault_line(f"{label}, {fault}", bad, tol)
        print(line)
        assert top > 10, (fault, top)
        if weakest is None or top < weakest[0]:
            weakest = (top, fault)
    print(f"{label}: the weakest wrong table is {weakest[1]} at "
          f"{weakest[0]:.0f} of the gate")
    assert not over, over


def _under(pipe, tid: str, read, arrays: dict):
    """``read(dit, arrays)`` with the table ``tid`` active."""
    from gmlx.image.lora import install

    prev = install.activate(pipe, tid)
    try:
        return read(pipe.dit, arrays)
    finally:
        install.restore(pipe, prev)


# The site gate.

def table_sites(pipe, arrays: dict, tid: str) -> tuple:
    """The sites of one free-running production step under the table
    ``tid``, read against the exact ops with the table's delta."""
    from gmlx.image.lora import install

    built = install.build_table(pipe, tid, pipe.dit.act_dtype, "block")
    sites, routed, heads, skipped = {}, {}, {}, set()
    prev = install.activate(pipe, tid)
    try:
        with kg.step_sites(pipe.dit, sites, routed=routed, heads=heads,
                           built=built, skipped=skipped):
            vel = kg.free_rows(pipe.dit, arrays)["velocity"]
    finally:
        install.restore(pipe, prev)
    return sites, routed, skipped, vel


def _site_setup(pipe, tmp, kind: str, lp):
    from gmlx.image import api

    name = COMFY if kind == "comfy" else SYNTHETIC
    arrays, m = fixture(_manifest(name)["name"])
    ad = _adapter(pipe, kind, tmp, lp, SITE_DIALECT)
    entries = _entries(ad, m["inputs"]["strength"])
    return arrays, m, entries, api.define_patch_table(pipe, entries)


@pytest.mark.parametrize("kind", ["comfy", "synthetic"])
def test_check7_site_gate_at_512(pipe, tmp_path, request, kind):
    """Every op of every block and every linear leaf a step calls under
    the table, each adapted leaf with its delta, is within its site gate
    of the stock op in float32 on the same input, and that run's velocity
    passes the product gate."""
    lp = request.getfixturevalue("lora_path") if kind == "comfy" else None
    arrays, m, entries, tid = _site_setup(pipe, tmp_path, kind, lp)
    sites, routed, skipped, vel = table_sites(pipe, arrays, tid)
    missing = kg.missing_ops(pipe.dit, sites, routed, skipped=skipped)
    assert not missing, missing
    adapted = {p for p, _, _ in entries}
    # An adapted leaf is held to the count of a leaf under its adapter, or
    # to the lora op's own.
    held = {k[0] for k, (_, g) in sites.items() if len(k) == 2
            and g >= 2 * 3 * qg.SITE_UNIT}
    assert adapted - skipped <= held, sorted(adapted - skipped - held)[:4]
    lora_on = sum(v == "quantized_matmul_lora" for v in routed.values())
    print(f"krea2 check7 sites {kind}: {len(adapted)} adapted leaves, "
          f"{lora_on} calls on the lora op, leaves on a float32 input "
          f"{sorted(skipped)}")
    worst = qg.worst_by_op(sites)
    print(f"krea2 check7 sites {kind}: {len(sites)} readings, the largest "
          "over its gate by op: "
          + ", ".join(f"{k} {v:.2f}" for k, v in sorted(worst.items())))
    over = qg.over_gate(sites)
    assert not over, over
    rel = kg.rel(vel, arrays["velocity"])
    assert rel <= m["tolerances"]["velocity"]["rel_l2"], rel


@pytest.mark.parametrize("kind", ["comfy", "synthetic"])
def test_check7_a_leaf_that_drops_its_delta_fails_the_site_gate(
        pipe, tmp_path, request, monkeypatch, kind):
    """The adapter's leaves of its middle block run their base product
    alone: their sites read the table's delta as error, and no other leaf
    moves past its gate."""
    from gmlx.image.lora import install

    lp = request.getfixturevalue("lora_path") if kind == "comfy" else None
    arrays, _, entries, tid = _site_setup(pipe, tmp_path, kind, lp)
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
    print(f"krea2 check7 sites {kind}, block {mid}'s deltas dropped: "
          f"{len(over)} readings over the gate, {len(leaves)} of "
          f"{len(dropped)} dropped leaves, the least at {low:.1f} of its "
          "gate")
    assert leaves == dropped, sorted(dropped - leaves)
