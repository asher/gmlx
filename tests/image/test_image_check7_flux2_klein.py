"""Check 7 of the image acceptance gate for FLUX.2 klein on community LoRA
files, step half: check 3's step at 512 with a file applied live under a
patch table, against BFL's reference DiT with the same file applied as
PEFT applies it, under check 3's gates.

scripts/image_oracle/check7_flux2_klein_community.py writes one fixture
per file, on the check 3 fixture of the checkpoint the file is for: the
distilled 4B step for the pixel-art file, in the diffusers spelling with
a rank-stabilized PEFT header, and for the mikkoani file, in the kohya
spelling with an alpha, and Base 4B's step with one reference image for
the sketch file, in AI-Toolkit's reference names without an alpha. The
oracle reads each file with its own parser and takes a PEFT header's
scale from peft itself. Every row is in fp32, with the oracle's own bf16
against fp32 gap of each and check 3's rule.

The files are public adapters that stay outside the repository. The
committed manifest names each by file name, size, sha256, repository,
revision and license, and the tests find the file under
GMLX_TEST_LORA_DIR and skip without it. The reader runs with its
defaults, so a key the FLUX.2 map does not cover fails the test.

The product gate holds each row to twice the oracle's gap, on the model
as it ships with its own time embedding. The base model misses it, and
so does the file at 0.8 of its strength where a fifth of the file's move
of the velocity is over twice the gate. The velocity of the request path passes its gate too,
after a step of the base model on the same conditioning. The exact gate
runs the step in the oracle's fp32 arithmetic in a child process and
holds the teacher-forced rows to EXACT_TOL, and the free rows and the
velocity to the gates from the floors the manifest records; each wrong
table of ``wrong_tables`` must miss it.

Skipped unless the manifests are committed, GMLX_IMAGE_FIXTURES holds
their arrays, KQUANT_TEST_GGUF_DIR the DiT GGUFs and GMLX_TEST_LORA_DIR
the files."""
from __future__ import annotations

import functools
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import flux2_gates as fg
from block_gates import exact_process, rerun_exact
from image_fixtures import image_dits
from oracle_fixture import ORACLE, committed, fixture, fx

# A Mac of 16 GB, as most users run (tests/host_box.py).
pytestmark = pytest.mark.host_gb(16)

ENV = "GMLX_TEST_LORA_DIR"
# id: (the check 3 fixture of the file's checkpoint, the pairs it holds,
# its spelling)
ADAPTERS = {"pixel_art": ("check3_flux2_klein_512", 86, "diffusers"),
            "mikkoani": ("check3_flux2_klein_512", 80, "kohya"),
            "sketch": ("check3_flux2_klein_base_r1_512", 80, "reference")}
# The pixel-art file's three modulation pairs, which ComfyUI does not read.
MODULATIONS = ("double_stream_modulation_img.lin",
               "double_stream_modulation_txt.lin",
               "single_stream_modulation.lin")


def _name(aid: str) -> str:
    return f"check7_flux2_klein_{aid}_512"


def _manifest(aid: str) -> dict:
    name = _name(aid)
    if not (ORACLE / f"{name}.manifest.json").is_file():
        pytest.skip(f"{name}.manifest.json is not committed: run "
                    "scripts/image_oracle/check7_flux2_klein_community.py "
                    "and commit the manifest it writes")
    return committed(name)


@functools.lru_cache(maxsize=None)
def _file(aid: str) -> str:
    """The adapter file under GMLX_TEST_LORA_DIR, checked by sha256."""
    root = os.environ.get(ENV)
    if not root:
        pytest.skip(f"set {ENV} to a directory of the community LoRA files")
    want = _manifest(aid)["files"]["lora"]
    for p in sorted(Path(root).expanduser().rglob(want["name"])):
        if p.stat().st_size == want["size"]:
            if fx.file_identity(p)["sha256"] != want["sha256"]:
                pytest.fail(f"{p} is not the file the oracle read")
            return str(p)
    pytest.skip(f"{want['name']} is not under {ENV}")


# The float tensors the exact gate's process put in float32, by DiT path.
_EXACT: dict = {}


class _Pipes:
    """The pipeline of a fixture's DiT, one loaded at a time."""

    def __init__(self, root):
        self.root, self.path, self.pipe = root, None, None

    def __call__(self, m: dict):
        from gmlx.image import api

        want = m["files"]["dit"]
        if self.path is not None and Path(self.path).name == want["name"]:
            return self.pipe
        self.pipe = self.path = None
        mx.clear_cache()
        for p in image_dits(self.root, "flux2_klein"):
            if Path(p).name == want["name"] and \
                    os.path.getsize(p) == want["size"]:
                break
        else:
            pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")
        if fx.file_identity(p)["sha256"] != want["sha256"]:
            pytest.fail(f"{p} is not the file the oracle read")
        self.pipe = api.load(p, log=lambda *_: None)
        if exact_process():
            # The exact gate's process: the file's float tensors in
            # float32, before a table wraps the leaves.
            _EXACT[p] = fg.exact_floats(self.pipe.dit, p)
        self.path = p
        return self.pipe


@pytest.fixture(scope="module")
def pipes(gguf_dir):
    return _Pipes(gguf_dir)


def _adapter(pipe, aid: str):
    from gmlx.image import lora

    ad = lora.load_adapter(pipe, _file(aid))
    assert len(ad.dit) == ADAPTERS[aid][1] and ad.text == ()
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


def _shipped(dit, arrays):
    return fg.measures(dit, arrays, own_time=True)


def test_the_manifests_read_the_step_and_follow_the_rule():
    """The committed manifests alone: each fixture takes the check 3 step
    of its file's checkpoint on the same DiT, names a public file under
    Apache-2.0, holds every row tolerance at twice the gap beside it, and
    the exact gate's tolerances follow their floors."""
    for aid, (source, pairs, spelling) in ADAPTERS.items():
        src, m = committed(source), _manifest(aid)
        ins = m["inputs"]
        assert ins["source"] == source, aid
        for k in ("step", "sigma", "refs", "seed", "size", "text_rows",
                  "reference_commit"):
            assert ins[k] == src["inputs"][k], (aid, k)
        assert m["files"]["dit"] == src["files"]["dit"], aid
        assert m["check"] == 7 and ins["lora_pairs"] == pairs, aid
        row = ins["adapter"]
        assert row["id"] == aid and row["license"] == "apache-2.0", aid
        assert row["spelling"] == spelling and row["pairs"] == pairs, aid
        assert len(row["revision"]) == 40, aid
        for k, g in m["results"]["rows_bf16_vs_fp32"].items():
            assert m["tolerances"][k]["rel_l2"] == 2 * g["rel_l2"], (aid, k)
        fg.exact_tols(m)
        assert m["results"]["key_table"]["same_layer_and_rows"], aid
    # The rank-stabilized header: alpha 128 over the root of rank 64.
    px = _manifest("pixel_art")["inputs"]["adapter"]
    assert px["header_config"]["transformer.use_rslora"] is True
    assert px["scales"] == [16.0] and px["peft"]
    assert _manifest("mikkoani")["inputs"]["adapter"]["scales"] == [0.5]
    assert _manifest("sketch")["inputs"]["adapter"]["scales"] == [1.0]


# The product gate.

def _product(pipe, arrays, m, tid, label: str) -> dict:
    got = _under(pipe, tid, _shipped, arrays)
    ratio = {k: v / fg.gate(k, m) for k, v in got.items()}
    top = max(ratio, key=ratio.get)
    print(f"{label}: {len(ratio)} measures, worst {got[top]:.2e} at {top}, "
          f"{ratio[top]:.2f} of its gate; velocity {got['velocity']:.2e} of "
          f"gate {m['tolerances']['velocity']['rel_l2']:.2e}")
    return got


@pytest.mark.integration
@pytest.mark.parametrize("aid", list(ADAPTERS))
def test_check7_community_step_rows_at_512(pipes, aid):
    from gmlx.image import api

    m = _manifest(aid)
    arrays, _ = fixture(_name(aid))
    pipe = pipes(m)
    ad = _adapter(pipe, aid)
    if aid == "pixel_art":
        assert {t for t, _, _ in ad.dit} >= set(MODULATIONS)
    tid = api.define_patch_table(pipe, _entries(ad, m["inputs"]["strength"]))
    got = _product(pipe, arrays, m, tid, f"flux2 check7 512 {aid}, "
                   f"{len(ad.dit)} pairs")
    out = fg.missed(got, m)
    assert not out, out


@pytest.mark.integration
@pytest.mark.parametrize("wrong", ["base", "strength 0.8"])
@pytest.mark.parametrize("aid", list(ADAPTERS))
def test_check7_community_rows_see_the_file_and_its_strength(pipes, aid,
                                                            wrong):
    """The base model misses the gate. The file at 0.8 of its strength
    moves the step by a fifth of the file's effect, and misses the gate
    where that fifth is over the velocity's gate."""
    from gmlx.image import api

    m = _manifest(aid)
    arrays, _ = fixture(_name(aid))
    pipe = pipes(m)
    tid, part = None, 1.0
    if wrong != "base":
        part = 0.2
        tid = api.define_patch_table(pipe, _entries(
            _adapter(pipe, aid), 0.8 * m["inputs"]["strength"]))
    out = fg.missed(_under(pipe, tid, _shipped, arrays), m)
    worst = max(out, key=lambda k: out[k][0] / out[k][1]) if out else None
    effect = part * m["results"]["adapter_effect"]["velocity"]
    gate = m["tolerances"]["velocity"]["rel_l2"]
    print(f"flux2 check7 {aid} {wrong}: {len(out)} measures miss"
          + (f", worst {worst} at {out[worst][0] / out[worst][1]:.1f} of "
             "its gate" if worst else "")
          + f"; its move of the velocity {effect:.2e}, gate {gate:.2e}")
    if effect > 2 * gate:
        assert out


@pytest.mark.integration
@pytest.mark.parametrize("aid", list(ADAPTERS))
def test_check7_community_velocity_of_the_request_path(pipes, aid):
    """api.denoise_step with the kernel routes at their default, after a
    step of the base model on the same conditioning, whose modulations
    and caption rows the pipeline keeps per table."""
    from gmlx.image import api

    m = _manifest(aid)
    arrays, _ = fixture(_name(aid))
    pipe = pipes(m)
    tid = api.define_patch_table(pipe, _entries(_adapter(pipe, aid),
                                                m["inputs"]["strength"]))
    refs = tuple(mx.array(arrays[f"refs.{i}"])[0]
                 for i in range(fg.n_refs(arrays)))
    cond = api.Conditioning(mx.array(arrays["features"]), f"check7 {aid}",
                            ref_latents=refs)

    def step(table):
        return np.array(api.denoise_step(
            pipe, mx.array(arrays["latents"]), float(arrays["sigma"]), cond,
            patch_table=table))

    base = step(None)
    v = step(tid)
    got = fx.compare(v, arrays["velocity"])
    tol = m["tolerances"]["velocity"]
    moved = fg.rel(base, arrays["velocity"])
    print(f"flux2 check7 {aid} request velocity: rel L2 {got['rel_l2']:.3e} "
          f"of {tol['rel_l2']:.3e}, cosine {got['cosine']:.6f} of "
          f"{tol['cosine_min']:.6f}; the base step is {moved:.2e} away")
    assert got["rel_l2"] <= tol["rel_l2"]
    assert got["cosine"] >= tol["cosine_min"]
    assert moved > tol["rel_l2"]


# The exact gate.

def _block(path: str) -> tuple | None:
    parts = path.split(".")
    if parts[0] in ("double_blocks", "single_blocks"):
        return parts[0], int(parts[1])
    return None


def _thirds_turned(entries: list) -> list:
    """Each pair on a third of a fused qkv on the next third's rows."""
    out = []
    for p, band, patches in entries:
        if band is not None:
            axis, start, n = band
            band = (axis, (start + n) % (3 * n), n)
        out.append((p, band, patches))
    return out


def wrong_tables(entries: list, strength: float) -> dict:
    """Wrong tables of a file's ``entries`` at ``strength``: the strength
    a tenth low, the pairs of the middle single block and of the first
    double block left out, the pairs on a third of a fused qkv on the
    next third's rows, and the pairs outside the blocks left out."""
    singles = sorted({b[1] for p, _, _ in entries
                      if (b := _block(p)) and b[0] == "single_blocks"})
    doubles = sorted({b[1] for p, _, _ in entries
                      if (b := _block(p)) and b[0] == "double_blocks"})
    mid = singles[len(singles) // 2]

    def scaled(k):
        return [(p, band, [(key, s * k) for key, s in patches])
                for p, band, patches in entries]

    out = {"strength a tenth low": scaled(0.9),
           f"single block {mid} left out": [
               e for e in entries if _block(e[0]) != ("single_blocks", mid)],
           f"double block {doubles[0]} left out": [
               e for e in entries
               if _block(e[0]) != ("double_blocks", doubles[0])]}
    if any(band is not None for _, band, _ in entries):
        out["qkv thirds turned"] = _thirds_turned(entries)
    if any(_block(p) is None for p, _, _ in entries):
        out["pairs outside the blocks left out"] = [
            e for e in entries if _block(e[0]) is not None]
    return out


@pytest.mark.integration
@pytest.mark.parametrize("aid", list(ADAPTERS))
def test_check7_community_exact_gate_at_512(pipes, request, aid):
    """Every measure within its exact gate under the file's table, the
    free-row and velocity gates from the floors measured now, and each
    wrong table of ``wrong_tables`` over the gate."""
    from gmlx.image import api

    m = _manifest(aid)
    arrays, _ = fixture(_name(aid))
    _file(aid)
    if not exact_process():
        return rerun_exact(request, "flux2 check7 exact")
    pipe = pipes(m)
    floats = _EXACT.get(pipes.path, 0)
    assert floats > 0
    ad = _adapter(pipe, aid)
    strength = m["inputs"]["strength"]
    entries = _entries(ad, strength)
    tid = api.define_patch_table(pipe, entries)
    label = f"flux2 check7 exact {aid}"
    tol = fg.check_floor(m, _under(pipe, tid, fg.exact_free_floor, arrays),
                         label)
    got = _under(pipe, tid, fg.exact_measures, arrays)
    print(fg.exact_line(f"{label}, {len(ad.dit)} pairs, {floats} float "
                        "tensors in fp32", got, tol))
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
