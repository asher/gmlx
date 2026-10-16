"""Check 7 of the image acceptance gate for FLUX.3 klein, step half: check
3's distilled step at 512 with one reference image, with a synthetic
adapter applied live under a patch table, against BFL's reference DiT
with the same adapter applied as PEFT applies it, under check 3's gates.

scripts/image_oracle/check7_flux3_klein_lora.py writes the fixture on
check3_flux3_klein_r1_512: one pair on every linear layer of blocks 0 and
4 of the image mode, block 2 of the reference mode, blocks 0 and 4 of the
text mode and single blocks 0, 13 and 27, and on nine layers outside the
blocks, the time, target time and modulation layers among them, at ranks
4 to 16 with an alpha of half the rank. The fixture holds its pairs in
the GGUF names, and the test writes them as an adapter in a dialect,
under the GGUF names or the diffusers names. Every row is in fp32, with
the oracle's own bf16 against fp32 gap of each and check 3's rule.

The product gate holds each row to twice the oracle's gap, in the kohya
dialect under the GGUF names and in the diffusers dialect under the
diffusers names, whose final modulation the reader swaps back. The base
model and the adapter at strength 0.8 miss it. The velocity of the
request path passes its gate too, after a step of the base model on the
same conditioning. The exact gate runs the step in the oracle's fp32
arithmetic in a child process and holds the teacher-forced rows to
EXACT_TOL, and the free rows and the velocity to the gates from the
floors the manifest records; each wrong table of ``wrong_tables`` must
miss it. The delta half runs on the CPU in test_image_flux3_lora_map.py.

Skipped unless the manifest is committed, GMLX_IMAGE_FIXTURES holds its
arrays and KQUANT_TEST_GGUF_DIR the DiT GGUF."""
from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import flux3_gates as fg
from block_gates import exact_process, rerun_exact
from image_fixtures import fixture_view, image_dits
from lora_fixtures import DIALECTS, write_adapter
from oracle_fixture import ORACLE, committed, fixture, fx

# A Mac of 16 GB, as most users run (tests/host_box.py).
pytestmark = pytest.mark.host_gb(16)

SYNTHETIC = "check7_flux3_klein_synthetic_512"
SOURCE = "check3_flux3_klein_r1_512"
# The spellings the product gate reads the adapter in: one that stores
# the alpha, under the GGUF names, and one that does not, with the alpha
# in its up rows, under the diffusers names. The exact gate reads the
# first.
SPELLINGS = {"kohya gguf": ("kohya", "gguf"),
             "diffusers diffusers": ("diffusers", "diffusers")}
EXACT = "kohya gguf"
ADALN = fg.ADALN
_TIME = ("time_in.out_layer", "ref_time_in.in_layer",
         "early_stream_modulations.image.lin")


def _manifest(name: str) -> dict:
    if not (ORACLE / f"{name}.manifest.json").is_file():
        pytest.skip(f"{name}.manifest.json is not committed: run "
                    "scripts/image_oracle/check7_flux3_klein_lora.py and "
                    "commit the manifest it writes")
    return committed(name)


@pytest.fixture(scope="module")
def synthetic():
    _manifest(SYNTHETIC)
    return fixture(SYNTHETIC)


@pytest.fixture(scope="module")
def dit_path(gguf_dir):
    want = _manifest(SYNTHETIC)["files"]["dit"]
    for p in image_dits(gguf_dir, "flux3_klein"):
        if Path(p).name == want["name"] and \
                os.path.getsize(p) == want["size"]:
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
        # The exact gate's process: the file's float tensors in float32,
        # before a table wraps the leaves.
        _EXACT["floats"] = fg.exact_floats(p.dit, dit_path)
    return p


def synthetic_adapter(arrays: dict, m: dict, spelling: str) -> dict:
    """The fixture's pairs as an adapter in the dialect and the names of
    ``spelling``. A dialect that stores no alpha takes it in its up rows.
    Under the diffusers names the final modulation's up rows are in
    diffusers' order, the scale half first."""
    dialect, names = SPELLINGS[spelling]
    keeps = DIALECTS[dialect][4]
    out = {}
    for name, alpha in m["inputs"]["lora_pair_alphas"].items():
        down, up = arrays[f"down.{name}"], arrays[f"up.{name}"]
        if not keeps:
            up = (up * np.float32(alpha / down.shape[0])).astype(np.float32)
            alpha = None
        if names == "diffusers":
            if name == ADALN:
                up = fg.swap_halves(up)
            name = fg.diffusers_name(name)
        out[name] = (down, up, alpha)
    return out


def _adapter(pipe, tmp, spelling: str):
    from gmlx.image import lora

    arrays, m = fixture(SYNTHETIC)
    dialect = SPELLINGS[spelling][0]
    path = write_adapter(Path(tmp) / f"synthetic-{dialect}.safetensors",
                         dialect, synthetic_adapter(arrays, m, spelling))
    ad = lora.load_adapter(pipe, path)
    assert ad.text == ()
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


def test_the_manifest_reads_the_step_and_follows_the_rule():
    """The committed manifest alone: the fixture takes check 3's step with
    one reference image on the same file, holds a pair for every target
    it names, on every leaf kind of every group of blocks and on the
    time and modulation layers, every row tolerance is twice the gap
    beside it, and the exact gate's tolerances follow their floors."""
    from gmlx.image import fingerprint as fp
    from gmlx.image.families.flux3_klein.lora_map import LEAVES, lora_map

    src, m = committed(SOURCE), _manifest(SYNTHETIC)
    assert m["inputs"]["source"] == SOURCE
    for k in ("step", "sigma", "target", "refs", "seed", "size",
              "reference_commit"):
        assert m["inputs"][k] == src["inputs"][k], k
    assert m["files"]["dit"] == src["files"]["dit"]
    assert m["check"] == 7 and m["inputs"]["refs"] == 1
    for k, g in m["results"]["rows_bf16_vs_fp32"].items():
        assert m["tolerances"][k]["rel_l2"] == 2 * g["rel_l2"], k
    fg.exact_tols(m)
    targets = m["inputs"]["targets"]
    assert sorted(m["inputs"]["lora_pair_alphas"]) == targets
    assert len(targets) == m["inputs"]["lora_pairs"] == 57
    assert {f"{h}.{t}" for t in targets for h in ("down", "up")} <= set(
        m["tensors"])
    km = lora_map(fp.identify(fixture_view(
        "flux3_klein_q4_k_m.header.json")).config)
    assert all(km.targets[t] == (t, None) for t in targets)
    for group in ("content_mode_blocks.image",
                  "content_mode_blocks.image_cond", "txt_mode_blocks",
                  "single_blocks"):
        got = {t.rsplit(".", 1)[1] for t in targets
               if t.startswith(group + ".")}
        assert got == set(LEAVES), group
    assert set(_TIME) | {ADALN} <= set(targets)
    effect = m["results"]["adapter_effect"]
    assert min(effect.values()) > 1e-2 and effect["velocity"] > 0.1


# The product gate.

def _product(pipe, arrays, m, tid, label: str) -> dict:
    got = _under(pipe, tid, fg.measures, arrays)
    ratio = {k: v / fg.gate(k, m) for k, v in got.items()}
    top = max(ratio, key=ratio.get)
    print(f"{label}: {len(ratio)} measures, worst {got[top]:.2e} at {top}, "
          f"{ratio[top]:.2f} of its gate; velocity {got['velocity']:.2e} of "
          f"gate {m['tolerances']['velocity']['rel_l2']:.2e}")
    return got


@pytest.mark.integration
@pytest.mark.parametrize("spelling", list(SPELLINGS))
def test_check7_adapter_step_rows_at_512(synthetic, pipe, tmp_path,
                                         spelling):
    from gmlx.image import api

    arrays, m = synthetic
    ad = _adapter(pipe, tmp_path, spelling)
    assert len(ad.dit) == m["inputs"]["lora_pairs"]
    tid = api.define_patch_table(pipe, _entries(ad, m["inputs"]["strength"]))
    got = _product(pipe, arrays, m, tid, f"flux3 check7 512 {spelling}, "
                   f"{len(ad.dit)} pairs")
    out = fg.missed(got, m)
    assert not out, out


@pytest.mark.integration
@pytest.mark.parametrize("wrong", ["base", "strength 0.8"])
def test_check7_rows_see_the_adapter_and_its_strength(synthetic, pipe,
                                                      tmp_path, wrong):
    from gmlx.image import api

    arrays, m = synthetic
    tid = None
    if wrong != "base":
        ad = _adapter(pipe, tmp_path, EXACT)
        tid = api.define_patch_table(
            pipe, _entries(ad, 0.8 * m["inputs"]["strength"]))
    out = fg.missed(_under(pipe, tid, fg.measures, arrays), m)
    worst = max(out, key=lambda k: out[k][0] / out[k][1]) if out else None
    print(f"flux3 check7 {wrong}: {len(out)} measures miss"
          + (f", worst {worst} at {out[worst][0] / out[worst][1]:.1f} of "
             "its gate" if worst else ""))
    assert out


@pytest.mark.integration
def test_check7_velocity_of_the_request_path(synthetic, pipe, tmp_path):
    """api.denoise_step with the kernel routes at their default, after a
    step of the base model on the same conditioning, whose time vectors,
    modulations and caption rows the pipeline keeps per table."""
    from gmlx.image import api

    arrays, m = synthetic
    ad = _adapter(pipe, tmp_path, "diffusers diffusers")
    tid = api.define_patch_table(pipe, _entries(ad, m["inputs"]["strength"]))
    cond = api.Conditioning(mx.array(arrays["features"]), "check7",
                            ref_latents=(mx.array(arrays["refs.0"])[0],))

    def step(table):
        return np.array(api.denoise_step(
            pipe, mx.array(arrays["latents"]), float(arrays["sigma"]), cond,
            patch_table=table, target=float(arrays["target"])))

    base = step(None)
    v = step(tid)
    got = fx.compare(v, arrays["velocity"])
    tol = m["tolerances"]["velocity"]
    moved = fg.rel(base, arrays["velocity"])
    print(f"flux3 check7 request velocity: rel L2 {got['rel_l2']:.3e} of "
          f"{tol['rel_l2']:.3e}, cosine {got['cosine']:.6f} of "
          f"{tol['cosine_min']:.6f}; the base step is {moved:.2e} away")
    assert got["rel_l2"] <= tol["rel_l2"]
    assert got["cosine"] >= tol["cosine_min"]
    assert moved > 3 * tol["rel_l2"]


# The exact gate.

def _single(path: str):
    parts = path.split(".")
    return int(parts[1]) if parts[0] == "single_blocks" else None


def wrong_tables(entries: list, strength: float) -> dict:
    """Wrong tables of an adapter's ``entries`` at ``strength``: the
    strength a tenth low, the pairs of the middle and of the last adapted
    single block left out, the middle one's pairs on the single block
    before it, which the adapter leaves alone, and the pairs of the time,
    target time and early image modulation layers left out."""
    blocks = sorted({b for p, _, _ in entries
                     if (b := _single(p)) is not None})
    mid, last = blocks[len(blocks) // 2], blocks[-1]
    assert mid - 1 not in blocks

    def scaled(k):
        return [(p, band, [(key, s * k) for key, s in patches])
                for p, band, patches in entries]

    return {"strength a tenth low": scaled(0.9),
            f"single block {mid} left out": [
                e for e in entries if _single(e[0]) != mid],
            f"single block {last} left out": [
                e for e in entries if _single(e[0]) != last],
            f"single block {mid} on single block {mid - 1}": [
                (p.replace(f"single_blocks.{mid}.",
                           f"single_blocks.{mid - 1}.", 1)
                 if _single(p) == mid else p, band, patches)
                for p, band, patches in entries],
            "time and modulation pairs left out": [
                e for e in entries if e[0] not in _TIME]}


@pytest.mark.integration
def test_check7_exact_gate_at_512(synthetic, pipe, tmp_path, request):
    """Every measure within its exact gate under the right table, the
    free-row and velocity gates from the floors measured now, and each
    wrong table of ``wrong_tables`` over the gate."""
    from gmlx.image import api

    arrays, m = synthetic
    if not exact_process():
        return rerun_exact(request, "flux3 check7 exact")
    assert _EXACT.get("floats", 0) > 0
    ad = _adapter(pipe, tmp_path, EXACT)
    strength = m["inputs"]["strength"]
    entries = _entries(ad, strength)
    tid = api.define_patch_table(pipe, entries)
    label = "flux3 check7 exact"
    tol = fg.check_floor(m, _under(pipe, tid, fg.exact_free_floor, arrays),
                         label)
    got = _under(pipe, tid, fg.exact_measures, arrays)
    print(fg.exact_line(f"{label}, {len(ad.dit)} pairs, {_EXACT['floats']} "
                        "float tensors in fp32", got, tol))
    over = fg.exact_over(got, tol)
    weakest = None
    for fault, table in wrong_tables(entries, strength).items():
        bad = _under(pipe, api.define_patch_table(pipe, table),
                     fg.exact_measures, arrays)
        line, top = fg.fault_line(f"{label}, {fault}", bad, tol)
        print(line)
        assert top > 10, (fault, top)
        if weakest is None or top < weakest[0]:
            weakest = (top, fault)
    print(f"{label}: the weakest wrong table is {weakest[1]} at "
          f"{weakest[0]:.0f} of the gate")
    assert not over, over
