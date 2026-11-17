"""Check 7 of the image acceptance gate for MiniMax-H3-Image on its two
published LoRA files: the delta half against float64 references, and
check 3's step at 512x512 with the files applied live under a patch
table, against ComfyUI's MiniMaxH3ImageModel with them merged into its
weights, under the three gates of check 3.

The files are character-swap (rank 16) and pDMD-2steps (rank 128, under
``transformer.``), both in diffusers' module names, which ComfyUI's key
map does not read for MiniMax-H3. Character-swap's repository also ships
the original it was converted from (208 pairs in the release file's
names under ``diffusion_model.``), which stock ComfyUI places in full.
scripts/image_oracle/check7_h3_image.py merges each file in float32 on
stock's key map where that places every pair, and else on it with
diffusers' names added as diffusers' converter defines them; gmlx reads
the conversion where the oracle reads the original. The rows:

- swap: the original character-swap at 1.0 on stock ComfyUI, gmlx on its
  conversion, at sigma 0.75 on check3_h3_image_512x512_s3's inputs;
- stacked: the same with pDMD at 0.5 on the extended key map;
- pdmd2 s0 and s1: pDMD at 1.0 at both steps of the schedule its card
  gives, [1, 12/13, 0], where ComfyUI's step reads the PDD heads of the
  grid intervals it spans, as gmlx's does.

Every row is in fp32, with the oracle's own bf16 against fp32 gap of
each and check 3's rule.

The delta half reads the files with the family's key map on the CPU. Its
references are written here: from the keys of the file gmlx reads, with
diffusers' converter read backward (lora_fixtures.py: q, k and v are rows
0, 1 and 2 of the fused qkv in thirds, ff.net.0.proj is fc1 with its two
halves swapped), and from the original's keys in the release file's
names. Character-swap at 0.5 and at 1.0, the stacked pair at the
oracle's strengths and the original alone are held to DELTA_TOL on every
module.

The product gate holds each row of a step to twice the oracle's gap. The
exact gate runs the step in the oracle's fp32 arithmetic in a child
process: the teacher-forced rows within EXACT_TOL, the free rows and the
velocity within the gates the manifest records from gmlx's floors under
the table. The site gate holds each op of every block and each linear
leaf, the adapted leaves with their delta, to the stock op in float32 on
the same input (h3_sites.py).

On the stacked row each wrong table must miss a gate by SEEN at least:
no LoRA, pDMD at 1.0, fc1's halves left as diffusers stores them, the q
and k bands swapped, and the pairs of the refiner, the first block or the
last block left out. The exact gate sees each; the product gate sees all
but those of PRODUCT_BLIND, which bf16 rounding hides. The files' delta
on one leaf reads under its site gate, so the site test runs the last
block's pairs at LOUD times their strengths: a leaf there that drops its
delta fails the site gate, and no other leaf does.

The files stay outside the repository. Each manifest names them by name,
size and sha256, and the tests find them and the release GGUF under
KQUANT_TEST_GGUF_DIR, and the fixtures under GMLX_IMAGE_FIXTURES. Each
test skips until what it reads is there.
"""
from __future__ import annotations

import functools
import os
import re
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import h3_image_gates as hg
import h3_sites as hs
import qwen_gates as qg
from block_gates import exact_process, rerun_exact
from image_fixtures import fixture_view
from lora_fixtures import (H3_DIFFUSERS_GROUPS, H3_DIFFUSERS_LEAVES,
                           H3_DIFFUSERS_QKV, H3_DIFFUSERS_TOP,
                           reference_delta, rel_l2)
from oracle_fixture import ORACLE, committed, fixture, fx

from gmlx.image.families.h3_image import family as hi

SWAP = "check7_h3_image_swap_512x512_s3"
STACKED = "check7_h3_image_lora_512x512_s3"
PDMD2 = ("check7_h3_image_pdmd2_512x512_s0",
         "check7_h3_image_pdmd2_512x512_s1")
ROWS = (SWAP, STACKED, *PDMD2)
STEP = "check3_h3_image_512x512_s3"
HEADER = "h3_image_pruned_ud_q4_k_xl.header.json"
# pDMD-2steps' schedule, from its model card.
PDMD2_SIGMAS = (1.0, 12 / 13, 0.0)
ORIGINAL = "h3_character_swap_v1.5.safetensors"
CONVERTED = "h3_image_character_swap_diffusers.safetensors"
PDMD = "pytorch_lora_weights.safetensors"
# Each file the oracle merges: rank, prefix, pairs, and the key map that
# placed them.
ORACLE_FILES = {ORIGINAL: (16, "diffusion_model.", 208, "stock"),
                PDMD: (128, "transformer.", 312, "extended")}
# The pairs gmlx reads from each file: a pair on the fused qkv in
# diffusers' form is three.
PAIRS = {ORIGINAL: 208, CONVERTED: 312, PDMD: 312}
# What each row merges, (file, strength).
MERGED = {SWAP: ((ORIGINAL, 1.0),),
          STACKED: ((ORIGINAL, 1.0), (PDMD, 0.5)),
          PDMD2[0]: ((PDMD, 1.0),), PDMD2[1]: ((PDMD, 1.0),)}
DELTA_TOL = 1e-6
# A wrong table is seen when it moves a measure this many times its gate.
SEEN = 1.5
# Wrong tables the product gate cannot see, measured on the release file:
# the refiner's pairs left out read 0.91 of the gate at
# token_refiner.blocks.1, and the last block's 0.75 at blocks.49. The
# exact gate sees them at 92 and 871 times its gate.
PRODUCT_BLIND = {"refiner pairs left out", "last block pairs left out"}
# The strength factor of the last block's pairs in the site drop test.
# At the files' strengths a dropped delta reads 0.2 of a leaf's site gate
# at least, under it.
LOUD = 16.0


def _manifest(name: str) -> dict:
    if not (ORACLE / f"{name}.manifest.json").is_file():
        pytest.skip(f"{name}.manifest.json is not committed: run "
                    "scripts/image_oracle/check7_h3_image.py on the release "
                    "GGUF and the LoRA files and commit the manifest")
    return committed(name)


def _schedule(name: str) -> tuple:
    return PDMD2_SIGMAS if name in PDMD2 else hi.PDD_SIGMAS


@pytest.mark.parametrize("name", ROWS)
def test_the_manifests_follow_the_rule(name):
    """Check 3's rule on every row, check 3's step with its files and
    inputs at the row's sigma and schedule, and each file merged in
    float32 with every pair placed: the original on stock's key map, pDMD
    on it with diffusers' names added, and gmlx reading the original's
    conversion in its place."""
    m = _manifest(name)
    hg.assert_manifest_rule(m, _schedule(name))
    hg.exact_tols(m)
    inp, step = m["inputs"], committed(STEP)
    for k in ("dit", "oracle_dit"):
        assert m["files"][k] == step["files"][k], k
    for k, v in step["inputs"].items():
        if name not in PDMD2 or k not in ("sigma", "sample_sigmas"):
            assert inp[k] == v, k
    if name in PDMD2:
        assert inp["sigma"] == float(np.float32(
            PDMD2_SIGMAS[PDMD2.index(name)]))
    assert inp["lora_merge"] == "float32"
    rows = inp["adapters"]
    assert [(r["file"], r["strength"]) for r in rows] == list(MERGED[name])
    for i, r in enumerate(rows):
        rank, prefix, pairs, keys = ORACLE_FILES[r["file"]]
        assert m["files"][f"lora_{i}"]["name"] == r["file"]
        assert (r["ranks"], r["prefix"], r["pairs"], r["key_map"]) == (
            [rank], prefix, pairs, keys)
        assert r["stock_key_map_pairs"] == (pairs if keys == "stock" else 0)
        gmlx = CONVERTED if r["file"] == ORIGINAL else None
        assert r.get("gmlx_file") == gmlx
        if gmlx:
            assert m["files"][f"gmlx_lora_{i}"]["name"] == gmlx
    # Every row reads the stacked row's files, which the step half loads.
    same = {f["name"]: f for f in _manifest(STACKED)["files"].values()}
    for f in m["files"].values():
        assert f == same[f["name"]], f["name"]


@functools.lru_cache(maxsize=None)
def _find(root: str, name: str, size: int, sha256: str) -> str | None:
    for p in sorted(Path(root).rglob(name)):
        if p.is_file() and os.path.getsize(p) == size:
            if fx.file_identity(str(p))["sha256"] != sha256:
                pytest.fail(f"{p} is not the file the oracle read")
            return str(p)
    return None


def _file(root: Path, name: str, key: str) -> str:
    want = _manifest(name)["files"][key]
    p = _find(str(root), want["name"], want["size"], want["sha256"])
    if p is None:
        pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")
    return p


def _gmlx_files(m: dict) -> list:
    """(manifest file key, strength) of each file gmlx reads for a row."""
    return [(f"gmlx_lora_{i}" if "gmlx_file" in r else f"lora_{i}",
             r["strength"]) for i, r in enumerate(m["inputs"]["adapters"])]


# The delta half.

def _check_warnings(warnings) -> None:
    """The one warning a file carries: its keys spell diffusers' names,
    which ComfyUI drops."""
    assert len(warnings) == 1 and "ComfyUI does not read" in warnings[0], \
        warnings


def _dims() -> dict:
    """Module path to (out, in) of each linear weight of the release
    file, from its header."""
    view = fixture_view(HEADER)
    return {n.removesuffix(".weight"): (t.shape[1], t.shape[0])
            for n, t in view.tensors.items()
            if n.endswith(".weight") and len(t.shape) == 2}


_TOP = {d: g for g, d in H3_DIFFUSERS_TOP.items()}
_GROUPS = {d: g for g, d in H3_DIFFUSERS_GROUPS.items()}
_LEAVES = {d: g for g, d in H3_DIFFUSERS_LEAVES.items()}
_BLOCK = re.compile(r"(transformer_blocks|token_refiner\.refiner_blocks)"
                    r"\.(\d+)\.(.+)")


def _own_target(stem: str, dims: dict) -> tuple:
    """(module path, band, halves swapped) of a stem: in the release
    file's names under ``diffusion_model.`` as they stand, and in
    diffusers' names, with or without ``transformer.``, read backward from
    diffusers' converter."""
    name = stem.removeprefix("diffusion_model.")
    if name != stem and name in dims:
        return name, None, False
    stem = stem.removeprefix("transformer.")
    if stem in _TOP:
        return _TOP[stem], None, False
    group, idx, leaf = _BLOCK.fullmatch(stem).groups()
    block = f"{_GROUPS[group]}.{idx}"
    if leaf in H3_DIFFUSERS_QKV:
        module = f"{block}.attn.qkv_proj"
        third = dims[module][0] // 3
        return module, (0, H3_DIFFUSERS_QKV.index(leaf) * third, third), False
    module = f"{block}.{_LEAVES[leaf]}"
    return module, None, leaf == "ff.net.0.proj"


def _own_rows(path: str, dims: dict) -> dict:
    """module path -> [(down, up, alpha, band)] in float32, from the
    file's keys with ``_own_target``; the up rows of fc1 swapped back to
    the release file's gate and value order."""
    parts: dict = {}
    for key, arr in mx.load(path).items():
        stem, _, kind = key.rpartition(".lora_")
        assert kind in ("A.weight", "B.weight"), key
        parts.setdefault(stem, {})[kind[0]] = np.array(arr.astype(mx.float32))
    rows: dict = {}
    for stem, p in parts.items():
        module, band, swapped = _own_target(stem, dims)
        up = p["B"]
        if swapped:
            value, gate = np.split(up, 2)
            up = np.concatenate([gate, value])
        rows.setdefault(module, []).append((p["A"], up, None, band))
    return rows


@pytest.mark.integration
def test_check7_delta_half(gguf_dir):
    """gmlx's reading of character-swap's conversion at 0.5 and at 1.0,
    of the stacked pair at the oracle's strengths and of the original
    alone, against float64 on every module: from the keys of the file it
    reads, and for the conversion from the original's keys too."""
    from gmlx.image.lora import plan, reader

    paths = {CONVERTED: _file(gguf_dir, STACKED, "gmlx_lora_0"),
             ORIGINAL: _file(gguf_dir, STACKED, "lora_0"),
             PDMD: _file(gguf_dir, STACKED, "lora_1")}
    dims = _dims()
    keymap = hi.lora_map(hi.synth_config(fixture_view(HEADER)))
    res, own = {}, {}
    for k, path in paths.items():
        res[k] = reader.resolve(reader.read_lora(path), keymap)
        assert len(res[k].dit) == PAIRS[k] and res[k].text == ()
        if k == ORIGINAL:
            assert res[k].warnings == []
        else:
            _check_warnings(res[k].warnings)
        own[k] = _own_rows(path, dims)
    swap = [(CONVERTED, 1.0)]
    # (label, what gmlx reads, the references it is held to)
    cases = [("character-swap 0.5", [(CONVERTED, 0.5)],
              ([(CONVERTED, 0.5)], [(ORIGINAL, 0.5)])),
             ("character-swap 1.0", swap, (swap, [(ORIGINAL, 1.0)])),
             ("stacked", swap + [(PDMD, 0.5)],
              (swap + [(PDMD, 0.5)], [(ORIGINAL, 1.0), (PDMD, 0.5)])),
             ("original 1.0", [(ORIGINAL, 1.0)], ([(ORIGINAL, 1.0)],))]
    worst, modules = 0.0, 0
    for label, stack, refs in cases:
        got: dict = {}
        for k, strength in stack:
            for t in res[k].dit:
                got.setdefault(t.module, []).append(plan.Part(
                    t.pair.down, t.pair.up, t.pair.alpha, t.band, strength))
        wants = []
        for ref in refs:
            want: dict = {}
            for k, strength in ref:
                for module, rows in own[k].items():
                    want.setdefault(module, []).extend(
                        (*r, strength) for r in rows)
            assert set(got) == set(want), label
            wants.append(want)
        modules = max(modules, len(got))
        for module, parts in got.items():
            lm = plan.block_diagonal(module, parts, dims[module])
            delta = np.array(lm.b, np.float64) @ np.array(lm.a, np.float64)
            for want in wants:
                err = rel_l2(delta,
                             reference_delta(want[module], dims[module]))
                worst = max(worst, err)
                assert err <= DELTA_TOL, (label, module, err)
    print(f"h3 image check7 delta: {modules} modules, worst relative L2 "
          f"{worst:.3e}")


# The step half.

@pytest.fixture(scope="module")
def real(gguf_dir):
    """The pipeline with each file gmlx reads loaded, and each row's
    table, defined before any test casts the DiT."""
    from gmlx.image import api, lora

    m = _manifest(STACKED)
    pipe = api.load(_file(gguf_dir, STACKED, "dit"), log=lambda *_: None)
    loaded = {}
    for key, _ in _gmlx_files(m):
        path = _file(gguf_dir, STACKED, key)
        ad = lora.load_adapter(pipe, path)
        name = Path(path).name
        assert len(ad.dit) == PAIRS[name] and ad.text == ()
        _check_warnings(ad.warnings)
        loaded[name] = ad
    tables = {}
    for row in ROWS:
        rm = _manifest(row)
        tables[row] = api.define_patch_table(pipe, lora.table_entries(
            [(loaded[rm["files"][key]["name"]], strength)
             for key, strength in _gmlx_files(rm)]))
    return pipe, loaded, tables


def _under(pipe, tid: str | None, fn, *args):
    """``fn(pipe.dit, *args)`` under patch table ``tid``."""
    from gmlx.image.lora import install

    if tid is None:
        return fn(pipe.dit, *args)
    prev = install.activate(pipe, tid)
    try:
        return fn(pipe.dit, *args)
    finally:
        install.restore(pipe, prev)


def _swapped(pipe, key: str, a: int, b: int, c: int) -> str:
    """A pair whose up rows a:b and b:c trade places."""
    from gmlx.image.lora import install

    p = pipe.adapters.get(key)
    up = mx.concatenate([p.up[:a], p.up[b:c], p.up[a:b], p.up[c:]])
    return install.register_pair(pipe, p.down, up)


def wrong_tables(pipe, loaded: dict) -> dict:
    """Name to patch table id of each wrong table of the stacked row."""
    from gmlx.image import api, lora

    cfg = pipe.dit.cfg.block
    inner, ffn = cfg.inner, cfg.ffn
    swap, pdmd = loaded[CONVERTED], loaded[PDMD]
    entries = lora.table_entries([(swap, 1.0), (pdmd, 0.5)])

    def table(rows):
        return api.define_patch_table(pipe, rows)

    def fc1_swapped(path, band, patches):
        if not path.endswith("mlp.fc1"):
            return path, band, patches
        return path, band, [(_swapped(pipe, key, 0, ffn, 2 * ffn), s, a)
                            for key, s, a in patches]

    def qk_swapped(path, band, patches):
        if not path.endswith("attn.qkv_proj"):
            return path, band, patches
        start = {0: inner, inner: 0}.get(band[1], band[1])
        return path, (band[0], start, band[2]), patches

    last = f"blocks.{len(pipe.dit.blocks) - 1}."
    return {
        "no LoRA": None,
        "pDMD at 1.0": table(lora.table_entries([(swap, 1.0), (pdmd, 1.0)])),
        "fc1 halves as diffusers stores them": table(
            [fc1_swapped(*e) for e in entries]),
        "q and k bands swapped": table([qk_swapped(*e) for e in entries]),
        "refiner pairs left out": table(
            [e for e in entries if not e[0].startswith("token_refiner.")]),
        "first block pairs left out": table(
            [e for e in entries if not e[0].startswith("blocks.0.")]),
        "last block pairs left out": table(
            [e for e in entries if not e[0].startswith(last)]),
    }


# The exact gate. Each test reruns in a child process, which loads its
# own DiT. They come before the tests that hold the module's DiT, so a
# run of the whole file starts each child with none beside it.

@pytest.mark.integration
@pytest.mark.parametrize("name", ROWS)
def test_check7_exact_gate(request, name):
    """Every teacher-forced row within EXACT_TOL of the oracle in its
    arithmetic, and the free rows and the velocity within the fixture's
    gates from the floors measured now. On the stacked row each wrong
    table misses a gate by SEEN at least."""
    fixture(_manifest(name)["name"])
    record = hg.record_dir() is not None
    if not exact_process():
        return rerun_exact(request, f"h3 image check7 exact {name}")
    arrays, m = fixture(name)
    pipe, loaded, tables = request.getfixturevalue("real")
    # The wrong tables hold their own pairs beside the files'.
    pipe.adapters.max_bytes = max(pipe.adapters.max_bytes, 8 << 30)
    label = f"h3 image check7 exact {name}"
    seen = {}
    with hg.exact_floats(pipe.dit) as n:
        floor = _under(pipe, tables[name], hg.exact_free_floor, arrays, m)
        if record:
            m = hg.record_floors(m, floor,
                                 (ORACLE / f"{name}.manifest.json",))
        tol = hg.check_floor(m, floor, label)
        got = _under(pipe, tables[name], hg.exact_measures, arrays, m)
        print(hg.exact_line(f"{label}, {n} float tensors cast to fp32",
                            got, tol))
        if name == STACKED:
            for fault, wrong in wrong_tables(pipe, loaded).items():
                bad = _under(pipe, wrong, hg.exact_measures, arrays, m)
                line, seen[fault] = hg.fault_line(f"{label}, {fault}", bad,
                                                  tol)
                print(line)
    over = hg.exact_over(got, tol)
    assert not over, over
    weak = {k: v for k, v in seen.items() if not v >= SEEN}
    assert not weak, weak


# The product gate.

@pytest.mark.integration
@pytest.mark.parametrize("name", ROWS)
def test_check7_rows(real, name):
    pipe, _, tables = real
    arrays, m = fixture(_manifest(name)["name"])
    assert pipe.dit.act_dtype == mx.bfloat16
    assert not hg.layout_misses(pipe.dit, arrays, m)
    got = _under(pipe, tables[name], hg.measures, arrays, m)
    assert not hg.unmeasured(got, m)
    print(hg.worst_line(f"h3 image check7 {name}", got, m))
    out = hg.missed(got, m["tolerances"])
    assert not out, out


@pytest.mark.integration
def test_each_wrong_table_fails_the_check7_gate(real):
    """On the stacked row each wrong table outside PRODUCT_BLIND misses a
    row's gate by SEEN at least, and the right table passes after them."""
    pipe, loaded, tables = real
    arrays, m = fixture(_manifest(STACKED)["name"])
    pipe.adapters.max_bytes = max(pipe.adapters.max_bytes, 8 << 30)
    tols = m["tolerances"]
    seen = {}
    for fault, wrong in wrong_tables(pipe, loaded).items():
        got = _under(pipe, wrong, hg.measures, arrays, m)
        ratio = {k: v / hg.gate(tols[k]) for k, v in got.items()
                 if not k.startswith("cosine ")}
        top = max(ratio, key=ratio.get)
        seen[fault] = (ratio[top], top)
        print(f"h3 image check7 wrong table {fault}: {ratio[top]:.2f} of "
              f"its gate at {top}")
    weak = {k: v for k, v in seen.items()
            if k not in PRODUCT_BLIND and not v[0] >= SEEN}
    assert not weak, weak
    out = hg.missed(_under(pipe, tables[STACKED], hg.measures, arrays, m),
                    tols)
    assert not out, out


# The site gate, on the stacked row.

def table_sites(pipe, arrays: dict, tid: str) -> tuple:
    """The sites of one free-running production step under the table
    ``tid``, read against the exact ops with the table's delta, the kernel
    sites, the leaves on a float32 input and the velocity."""
    from gmlx.image.lora import install

    built = install.build_table(pipe, tid, pipe.dit.act_dtype, "block")
    sites, routed, skipped = {}, {}, set()
    prev = install.activate(pipe, tid)
    try:
        with hs.step_sites(pipe.dit, sites, routed=routed, skipped=skipped,
                           built=built):
            vel = hg.free_rows(pipe.dit, arrays)[hg.VELOCITY]
    finally:
        install.restore(pipe, prev)
    return sites, routed, skipped, vel


def _adapted(pipe, tid: str) -> set:
    return {p for p, _, _ in pipe.patch_tables[tid]["entries"]}


@pytest.mark.integration
def test_check7_site_gate(real):
    """Every op of every block and every linear leaf a step calls under
    the table, each adapted leaf with its delta, is within its site gate
    of the stock op in float32 on the same input, and that run's velocity
    passes the product gate."""
    pipe, _, tables = real
    tid = tables[STACKED]
    arrays, m = fixture(_manifest(STACKED)["name"])
    _, plan = hg.plan_of(pipe.dit, arrays)
    sites, routed, skipped, vel = table_sites(pipe, arrays, tid)
    missing = hs.missing_ops(pipe.dit, sites, plan.layout, routed, skipped,
                             hi.ATTENTION_CHUNK)
    assert not missing, missing
    adapted = _adapted(pipe, tid)
    # Each adapted leaf is held to two more roundings than its base, or
    # to the low-rank op's own count.
    held = {k[0] for k, (_, g) in sites.items() if len(k) == 2
            and g >= 2 * 3 * qg.SITE_UNIT}
    assert adapted - skipped <= held, sorted(adapted - skipped - held)[:4]
    lora_on = sum(v == "quantized_matmul_lora" for v in routed.values())
    print(f"h3 image check7 sites: {len(adapted)} adapted leaves, {lora_on} "
          f"calls on the lora op, leaves on a float32 input "
          f"{sorted(skipped)}")
    worst = qg.worst_by_op(sites)
    print(f"h3 image check7 sites: {len(sites)} readings, the largest over "
          "its gate by op: "
          + ", ".join(f"{k} {v:.2f}" for k, v in sorted(worst.items())))
    over = qg.over_gate(sites)
    assert not over, over
    rel = hg.rel(vel, arrays[hg.VELOCITY])
    assert rel <= m["tolerances"][hg.VELOCITY]["rel_l2"], rel


def _loud(pipe, loaded: dict, last: str) -> str:
    """The stacked row's table with the pairs of the blocks under ``last``
    at LOUD times their strengths."""
    from gmlx.image import api, lora

    entries = lora.table_entries([(loaded[CONVERTED], 1.0),
                                  (loaded[PDMD], 0.5)])
    return api.define_patch_table(pipe, [
        (p, b, [(k, s * LOUD, a) for k, s, a in patches])
        if p.startswith(last) else (p, b, patches)
        for p, b, patches in entries])


@pytest.mark.integration
def test_check7_a_leaf_that_drops_its_delta_fails_the_site_gate(
        real, monkeypatch):
    """With the last block's pairs at LOUD times their strengths every
    site is within its gate, and when those leaves run their base product
    alone their sites read the table's delta as error, and no other leaf
    moves past its gate."""
    from gmlx.image.lora import install

    pipe, loaded, _ = real
    arrays, _ = fixture(_manifest(STACKED)["name"])
    last = f"blocks.{len(pipe.dit.blocks) - 1}."
    tid = _loud(pipe, loaded, last)
    sites, _, _, _ = table_sites(pipe, arrays, tid)
    over = qg.over_gate(sites)
    assert not over, over
    dropped = {p for p in _adapted(pipe, tid) if p.startswith(last)}
    assert len(dropped) == 4, sorted(dropped)
    call = install.ImageLoRALinear.__call__

    def base_alone(self, x):
        return self.base(x) if self._path in dropped else call(self, x)

    monkeypatch.setattr(install.ImageLoRALinear, "__call__", base_alone)
    sites, _, _, _ = table_sites(pipe, arrays, tid)
    over = qg.over_gate(sites)
    leaves = {k[0] for k in over if len(k) == 2}
    low = min(sites[k][0] / sites[k][1] for k in sites
              if len(k) == 2 and k[0] in dropped)
    print(f"h3 image check7 sites, the last block's pairs at {LOUD:g} times "
          f"their strengths and their deltas dropped: {len(over)} readings "
          f"over the gate, {len(leaves)} of {len(dropped)} dropped leaves, "
          f"the least at {low:.1f} of its gate")
    assert leaves == dropped, sorted(dropped ^ leaves)
