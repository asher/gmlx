"""Check 7 of the image acceptance gate for Qwen-Image 2.1, step half: one
denoise step with a LoRA applied live through a patch table, against the
diffusers oracle with the delta merged into dense weights.

The oracle's adapter targets the reference layout, where the gate and the
projection of the image MLP are separate weights, so its merge places no
band. The test writes the same pairs in each dialect's key spelling,
loads the files into a pipeline, defines a patch table and reads check
3's rows under it, which qwen_gates.py describes.

The tiny rows run on the CPU from the fixtures committed under
tests/fixtures/image/oracle/, which
scripts/image_oracle/check7_qwen_image21_lora.py --self-test writes. Both
sides form the pairs from qwen_image21_tiny.py. The model runs in the
oracle's fp32 arithmetic, from a file with the fused weight and from one
with the pair apart. Every row is held to check 3's tiny bounds, and each
wrong table of ``wrong_tables`` must miss them: the gate and projection
bands swapped, the alpha left out, a group or one entry left out, a
strength a tenth off, and the table of another adapter.

The real rows need the Qwen-Image 2.1 GGUF under KQUANT_TEST_GGUF_DIR
and the fixtures the same script writes from it. They skip until both
are there. They hold the step to the three gates: the exact gate in a
child process, the product gate on the manifest's tolerances, and the
site gate of every linear leaf. The exact gate is the one that holds the
table: it makes the wrong tables of ``wrong_tables`` and those of
``structural_tables`` (one half of an image MLP left out or the two
swapped in the first and the last block, a sign, a block's delta on its
neighbour) and each must miss, and for the stack each adapter alone. The
product gates pass a half left out or swapped in the first or the last
block, and the site gate reads the ops under whatever table it is given.

The delta half of check 7 runs in test_image_qwen_lora_map.py.
"""
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import qwen_gates as qg
import qwen_tiny as qt
from block_gates import (EXACT_TOL, QWEN_EXACT_FREE_TOL, assert_all_dense,
                         exact_process, exact_products, free_running,
                         rerun_exact)
from image_fixtures import image_dits
from lora_fixtures import (DIALECTS, dialect_keys, header, reference_delta,
                           write_adapter, write_keys)
from oracle_fixture import ORACLE, committed, committed_fixture, fixture, fx

from gmlx.image import api, lora, registry
from gmlx.image.families.qwen_image21 import model as qm
from gmlx.image.lora import install, patch_table, plan, reader

TINY = {v: f"selftest_qwen_image21_tiny_lora_{v}"
        for v in qt.rule.LORA_VARIANTS}
# The fixtures of the files that no dialect writer spells: one with its
# LoRA config in the header, and one with a pair on the fused weight.
HEADER, SHARED = (f"selftest_qwen_image21_tiny_lora_{v}"
                  for v in qt.rule.LORA_FILES)
LAYOUTS = ("fused", "unfused")
# Variant and the dialect of each of its adapters. A dialect that stores
# an alpha writes the adapters that have one.
CASES = (("alpha", ("kohya",)), ("alpha", ("onetrainer",)),
         ("plain", ("diffusers",)), ("plain", ("peft",)),
         ("stacked", ("kohya", "diffusers")),
         ("stacked", ("onetrainer", "peft")))
IDS = [f"{v}-{'+'.join(d)}" for v, d in CASES]

# Check 3's bounds for the tiny model in fp32 on the CPU. With the
# oracle's timestep embedding the right table measures 5.0e-7 at most,
# against 5.3e-7 for the base model. With gmlx's own sinusoid it measures
# 3.2e-6 on the modulation and 3.7e-6 on a velocity through denoise_step.
# The weakest wrong table, one attention query pair of the stacked
# adapters left out, moves a step velocity to 324 times its bound, and
# the weakest of the named kinds, a strength a tenth low, to 913 times.
TOL = 4e-6
TEMB_TOL = 1e-5
OWN_TOL = 4e-5
# A wrong table is seen when it moves a measure this many times its gate.
SEEN = 2.0
# A step under a table against the same gmlx model with the float64 delta
# in its float32 weights: both sides run one sinusoid and one arithmetic.
MERGED_TOL = 4e-6
B = "transformer_blocks."
# The kinds of wrong table, by the start of their names.
KINDS = ("gate_and_proj", "alpha_left_out", "block_", "layers_outside",
         "only_", "strength_", "table_of", "the_empty_table")


def tiny_pipe(root, layout: str, act=mx.float32):
    """A pipeline over the tiny model, loaded from a GGUF with the fused
    weight or with the pair apart. ``act=None`` keeps the bf16
    activations of a load."""
    path = qt.write(Path(root) / f"{layout}.gguf",
                    qt.weights(fused=layout == "fused"))
    fam = qt.tiny_family()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(registry, "families", lambda: (fam,))
        pipe = api.load(path, log=lambda *_: None)
    assert pipe.dit.cfg.fused_mlp == (layout == "fused")
    if act is not None:
        pipe.dit.act_dtype = act
    return pipe


@pytest.fixture(scope="module")
def pipes(tmp_path_factory):
    root = tmp_path_factory.mktemp("qwen-lora")
    return {layout: tiny_pipe(root, layout) for layout in LAYOUTS}


def adapters_of(m: dict) -> list:
    """(pairs, alpha, strength) of each adapter the oracle merged for a
    fixture, the pairs as target to (down, up)."""
    out = []
    for row in m["inputs"]["adapters"]:
        if m["inputs"].get("pairs") == "qwen_image21_tiny.lora_pairs":
            pairs = qt.rule.lora_pairs(row["name"])
            assert qt.rule.lora_sha256(row["name"]) == row["pairs_sha256"]
        else:
            arrays, _ = fixture(m["inputs"]["pairs_fixture"])
            pre = row.get("prefix", "")
            pairs = {t: (arrays[f"{pre}down.{t}"], arrays[f"{pre}up.{t}"])
                     for t in row["targets"]}
        assert list(pairs) == row["targets"]
        out.append((pairs, row["alpha"], row["strength"]))
    return out


def load_rows(pipe, tmp, m: dict, dialects, bare: bool = False) -> list:
    """The fixture's adapters, each written in its dialect and loaded, as
    (LoadedAdapter, strength) rows. ``bare`` writes no alpha."""
    rows = []
    for i, ((pairs, alpha, strength), dialect) in enumerate(
            zip(adapters_of(m), dialects, strict=True)):
        alpha = None if bare else alpha
        assert (alpha is not None) == DIALECTS[dialect][4], dialect
        name = f"{m['name']}-{i}-{dialect}{'-bare' if bare else ''}"
        ad = lora.load_adapter(pipe, write_adapter(
            Path(tmp) / f"{name}.safetensors", dialect,
            {t: (d, u, alpha) for t, (d, u) in pairs.items()}))
        assert len(ad.dit) == len(pairs) and ad.text == ()
        assert ad.warnings == ()
        rows.append((ad, strength))
    return rows


def step_velocities(pipe, arrays: dict, table: str, key: str) -> dict:
    """The two velocities of a fixture through ``denoise_step`` under
    ``table``, the path a request takes, against the oracle."""
    cond = qg.conditioning(arrays, key)
    return {f"velocity_{tag}": fx.compare(np.array(api.denoise_step(
        pipe, mx.array(arrays[f"latents_{tag}"]),
        float(arrays[f"sigma_{tag}"]), cond, patch_table=table)),
        arrays[f"velocity_{tag}"]) for tag in "ab"}


def ratios(pipe, arrays: dict, table: str, key: str) -> dict:
    """Every measure of the tiny step under ``table`` over its bound:
    check 3's rows, branch rows and free run with the oracle's timestep
    embedding, the timestep rows of gmlx's own sinusoid as ``own``, and
    the two velocities through ``denoise_step`` as ``step``."""
    with mx.stream(mx.cpu):
        with qg.under_table(pipe, table):
            exact = qg.step(pipe.dit, arrays)
            times = qg.timestep_rows(pipe.dit, arrays)
        step = step_velocities(pipe, arrays, table, key)
    out = {k: g["rel_l2"] / TOL for k, g in exact.items()}
    out.update({f"own {k}": g["rel_l2"] / TEMB_TOL for k, g in times.items()})
    out.update({f"step {k}": g["rel_l2"] / OWN_TOL for k, g in step.items()})
    return out


def top(got: dict, only: str = "") -> tuple[str, float]:
    """The measure of ``got`` furthest over its bound, among the names
    that start with ``only``."""
    names = [k for k in got if k.startswith(only)]
    k = max(names, key=lambda r: got[r])
    return k, got[k]


def wrong_tables(pipe, tmp, m: dict, rows: list, each: bool = True,
                 every_block: bool = False) -> dict:
    """Wrong tables made from the (LoadedAdapter, strength) rows of a
    right one, as name to entries: the gate and the projection of every
    image MLP swapped, the adapters without their alpha, the entries of
    one block left out (the first, the middle and the last block, or
    with ``every_block`` each block in turn), with ``each`` every one
    entry left out, every strength a tenth off each way, and the table
    of one adapter of a stack, of an adapter the oracle did not merge,
    and of none."""
    cfg = pipe.dit.cfg
    ffn = cfg.ffn_dim
    entries = lora.table_entries(rows)
    out: dict = {}
    if cfg.fused_mlp:
        swap = {(0, 0, ffn): (0, ffn, ffn), (0, ffn, ffn): (0, 0, ffn)}
        out["gate_and_proj_bands_swapped"] = [
            (p, swap.get(b, b), ps) for p, b, ps in entries]
    else:
        names = {"gate_layer": "proj", "proj": "gate_layer"}
        out["gate_and_proj_leaves_swapped"] = [
            (".".join([*p.split(".")[:-1], names.get(p.split(".")[-1],
                                                     p.split(".")[-1])]),
             b, ps) for p, b, ps in entries]
    if any(alpha is not None for _, alpha, _ in adapters_of(m)):
        bare = load_rows(pipe, tmp, m, ("diffusers",) * len(rows), bare=True)
        out["alpha_left_out"] = lora.table_entries(
            [(ad, s) for (ad, _), (_, s) in zip(bare, rows)])
    blocks = sorted({int(p.split(".")[1]) for p, _, _ in entries
                     if p.startswith(B)})
    some = sorted({blocks[0], blocks[len(blocks) // 2], blocks[-1]})
    groups = {f"block_{i}": f"{B}{i}."
              for i in (blocks if every_block else some)}
    for name, prefix in groups.items():
        out[f"{name}_left_out"] = [e for e in entries
                                   if not e[0].startswith(prefix)]
    inside = [e for e in entries if e[0].startswith(B)]
    if len(inside) < len(entries):
        out["layers_outside_the_blocks_left_out"] = inside
    if each:
        for i, (p, b, _) in enumerate(entries):
            band = "" if b is None else f"_band_{b[1] // ffn}"
            out[f"only_{p}{band}_left_out"] = entries[:i] + entries[i + 1:]
    for f in (0.9, 1.1):
        out[f"strength_x{f}"] = [(p, b, [(k, s * f) for k, s in ps])
                                 for p, b, ps in entries]
    if len(rows) > 1:
        for i, row in enumerate(rows):
            out[f"table_of_adapter_{i}_alone"] = lora.table_entries([row])
    elif m["inputs"].get("pairs") == "qwen_image21_tiny.lora_pairs":
        two = lora.load_adapter(pipe, write_adapter(
            Path(tmp) / "another.safetensors", "diffusers",
            {t: (d, u, None)
             for t, (d, u) in qt.rule.lora_pairs("two").items()}))
        out["table_of_another_adapter"] = lora.table_entries(
            [(two, rows[0][1])])
    out["the_empty_table"] = []
    return out


def _block(path: str) -> int | None:
    return int(path.split(".")[1]) if path.startswith(B) else None


def _half(cfg, path: str, band) -> str | None:
    """``gate`` or ``proj`` for an entry on one half of a block's image
    MLP input product: a band of the fused weight, or one of the pair of
    leaves."""
    leaf = path.rsplit(".", 1)[-1]
    if not path.startswith(B) or ".img_mlp." not in path:
        return None
    if cfg.fused_mlp:
        if leaf != "gate_up" or band is None:
            return None
        return "gate" if band[1] == 0 else "proj"
    return {"gate_layer": "gate", "proj": "proj"}.get(leaf)


def structural_tables(pipe, entries: list) -> dict:
    """Wrong tables that keep every pair and every strength of the right
    one and place one of them wrong, as name to entries. In the first
    and in the last block the adapter is in: one half of the image MLP
    left out, the two halves swapped, and the sign of the block's
    strengths. The sign of every strength. The entries of the first, the
    middle and the last such block moved to the nearest block the
    adapter is not in, when there is one."""
    cfg = pipe.dit.cfg
    blocks = sorted({_block(p) for p, _, _ in entries} - {None})
    ends = sorted({blocks[0], blocks[-1]})
    out: dict = {}

    def half(e) -> tuple:
        return _block(e[0]), _half(cfg, e[0], e[1])

    for i in ends:
        halves = {half(e)[1]: (e[0], e[1]) for e in entries
                  if half(e)[0] == i and half(e)[1]}
        for h in halves:
            out[f"{h}_half_left_out_in_block_{i}"] = [
                e for e in entries if half(e) != (i, h)]
        if len(halves) == 2:
            other = {"gate": halves["proj"], "proj": halves["gate"]}
            out[f"halves_swapped_in_block_{i}"] = [
                (*other[half(e)[1]], e[2]) if half(e)[0] == i and half(e)[1]
                else e for e in entries]
        out[f"sign_in_block_{i}"] = [
            (p, b, [(k, -s if _block(p) == i else s) for k, s in ps])
            for p, b, ps in entries]
    out["sign_of_every_strength"] = [(p, b, [(k, -s) for k, s in ps])
                                     for p, b, ps in entries]
    free = [i for i in range(len(pipe.dit.transformer_blocks))
            if i not in blocks]
    if free:
        for i in sorted({blocks[0], blocks[len(blocks) // 2], blocks[-1]}):
            to = min(free, key=lambda j: (abs(j - i), j))
            out[f"block_{i}_delta_on_block_{to}"] = [
                (p.replace(f"{B}{i}.", f"{B}{to}.", 1)
                 if _block(p) == i else p, b, ps) for p, b, ps in entries]
    return out


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("variant,dialects", CASES, ids=IDS)
def test_a_tiny_step_under_a_patch_table_matches_the_merged_oracle(
        pipes, tmp_path, variant, dialects, layout):
    arrays, m = committed_fixture(TINY[variant])
    case, stack = qt.rule.LORA_VARIANTS[variant]
    assert m["inputs"]["case"] == case and m["check"] == 7
    assert [(r["name"], r["alpha"], r["strength"])
            for r in m["inputs"]["adapters"]] == list(stack)
    pipe = pipes[layout]
    rows = load_rows(pipe, tmp_path, m, dialects)
    entries = lora.table_entries(rows)
    # Adapter one has both halves of block 0's image MLP: the two bands
    # of the fused weight, or the two leaves.
    ffn, mlp = pipe.dit.cfg.ffn_dim, B + "0.img_mlp."
    assert sorted((p, b) for p, b, _ in entries
                  if p.startswith(mlp) and p != mlp + "out") == (
        [(mlp + "gate_up", (0, 0, ffn)), (mlp + "gate_up", (0, ffn, ffn))]
        if layout == "fused"
        else [(mlp + "gate_layer", None), (mlp + "proj", None)])
    tid = api.define_patch_table(pipe, entries)
    got = ratios(pipe, arrays, tid, m["name"])
    name, worst = top(got)
    exact = max(v for k, v in got.items()
                if not k.startswith(("own ", "step ")))
    print(f"qwen check7 tiny {variant} {'+'.join(dialects)} {layout}: "
          f"{len(got)} measures, worst {worst:.2f} of its bound at {name}; "
          f"with the oracle's embedding {exact * TOL:.2e} of {TOL:.0e}, "
          f"own sinusoid {top(got, 'own ')[1] * TEMB_TOL:.2e} of "
          f"{TEMB_TOL:.0e}, step velocities "
          f"{top(got, 'step ')[1] * OWN_TOL:.2e} of {OWN_TOL:.0e}")
    assert worst <= 1, (name, worst)
    # The base model is far outside the gate on the velocities alone.
    base = ratios(pipe, arrays, api.EMPTY_TABLE, m["name"])
    for k in ("free velocity_a", "free velocity_b", "step velocity_a",
              "step velocity_b"):
        assert base[k] >= 1000, (k, base[k])


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("variant", list(TINY))
def test_each_wrong_table_fails_the_tiny_gate(tmp_path, variant, layout):
    """Every wrong table misses a bound by SEEN at least, on all measures
    and on the two velocities of ``denoise_step`` alone."""
    arrays, m = committed_fixture(TINY[variant])
    pipe = tiny_pipe(tmp_path, layout)
    dialects = next(d for v, d in CASES if v == variant)
    rows = load_rows(pipe, tmp_path, m, dialects)
    right = api.define_patch_table(pipe, lora.table_entries(rows))
    assert top(ratios(pipe, arrays, right, m["name"]))[1] <= 1
    wrong = wrong_tables(pipe, tmp_path, m, rows)
    kinds: dict = {k: [] for k in KINDS}
    assert (variant == "plain") == ("alpha_left_out" not in wrong)
    n_each = sum(k.startswith("only_") for k in wrong)
    assert n_each == len(pipe.patch_tables[right]["entries"]) >= 18
    for name, entries in wrong.items():
        tid = api.define_patch_table(pipe, entries)
        assert tid != right, name
        got = ratios(pipe, arrays, tid, m["name"])
        whole, step = top(got)[1], top(got, "step ")[1]
        assert whole >= SEEN and step >= SEEN, (name, whole, step)
        kinds[next(k for k in KINDS if name.startswith(k))].append(
            (step, whole, name))
    assert [k for k, v in kinds.items() if not v] == (
        ["alpha_left_out"] if variant == "plain" else [])
    for kind, seen in kinds.items():
        if seen:
            step, whole, name = min(seen)
            print(f"qwen check7 tiny wrong tables {variant} {layout}, "
                  f"{kind}: {len(seen)} tables, the weakest is {name} at "
                  f"{step:.0f} bounds on a step velocity and {whole:.0f} "
                  "on its most moved measure")
    # The right table still passes after all of them ran.
    assert top(ratios(pipe, arrays, right, m["name"]))[1] <= 1


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("variant", list(TINY))
def test_each_structural_table_fails_the_tiny_gate(tmp_path, variant,
                                                   layout):
    """Each table of ``structural_tables`` is another table than the
    right one and misses a bound by SEEN at least."""
    arrays, m = committed_fixture(TINY[variant])
    pipe = tiny_pipe(tmp_path, layout)
    rows = load_rows(pipe, tmp_path, m, next(d for v, d in CASES
                                             if v == variant))
    entries = lora.table_entries(rows)
    right = api.define_patch_table(pipe, entries)
    wrong = structural_tables(pipe, entries)
    blocks = sorted({_block(p) for p, _, _ in entries} - {None})
    # An end block has a table per half the adapter is in, and the swap
    # where it is in both.
    need = {"sign_of_every_strength"}
    both = 0
    for i in (blocks[0], blocks[-1]):
        halves = {_half(pipe.dit.cfg, p, b) for p, b, _ in entries
                  if _block(p) == i} - {None}
        need |= {f"{h}_half_left_out_in_block_{i}" for h in halves}
        need |= {f"sign_in_block_{i}"}
        if len(halves) == 2:
            need.add(f"halves_swapped_in_block_{i}")
            both += 1
    assert both, "no end block has both halves"
    moved = {k for k in wrong if "_delta_on_block_" in k}
    assert need == set(wrong) - moved, sorted(need ^ (set(wrong) - moved))
    assert bool(moved) == (len(blocks) < len(pipe.dit.transformer_blocks))
    seen = {}
    for name, table in wrong.items():
        tid = api.define_patch_table(pipe, table)
        assert tid != right, name
        seen[name] = top(ratios(pipe, arrays, tid, m["name"]))[1]
        assert seen[name] >= SEEN, (name, seen[name])
    low = min(seen, key=seen.get)
    print(f"qwen check7 tiny structural tables {variant} {layout}: "
          f"{len(seen)} tables, the weakest is {low} at {seen[low]:.0f} "
          "bounds on its most moved measure")


@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_state_of_one_table_is_not_read_under_another(tmp_path, layout):
    """The prefix K and V and the modulation depend on the adapter. One
    conditioning key under the adapter's table and under the empty one
    gives the merged oracle's velocity and the base oracle's, in any
    order of calls."""
    arrays, m = committed_fixture(TINY["plain"])
    plain, _ = committed_fixture("selftest_qwen_image21_tiny_ref1")
    for k in ("features", "latents_a", "latents_b", "sigma_a", "sigma_b"):
        assert np.array_equal(arrays[k], plain[k])
    pipe = tiny_pipe(tmp_path, layout)
    fresh = tiny_pipe(tmp_path, layout)
    tid = api.define_patch_table(pipe, lora.table_entries(
        load_rows(pipe, tmp_path, m, ("diffusers",))))
    cond = qg.conditioning(arrays, "one key")

    def velocity(p, table, tag):
        with mx.stream(mx.cpu):
            return np.array(api.denoise_step(
                p, mx.array(arrays[f"latents_{tag}"]),
                float(arrays[f"sigma_{tag}"]), cond, patch_table=table))

    seen = {}
    for table, tag in ((tid, "a"), (api.EMPTY_TABLE, "a"), (tid, "b"),
                       (api.EMPTY_TABLE, "b"), (tid, "a")):
        v = velocity(pipe, table, tag)
        want = (arrays if table == tid else plain)[f"velocity_{tag}"]
        assert fx.rel_l2(v, want) <= OWN_TOL, (table == tid, tag)
        assert np.array_equal(seen.setdefault((table, tag), v), v)
    # Under the empty table the wrapped leaves give what plain ones give.
    for tag in "ab":
        assert np.array_equal(seen[(api.EMPTY_TABLE, tag)],
                              velocity(fresh, api.EMPTY_TABLE, tag))
    assert fresh.lora is None and pipe.lora is not None
    assert {k[4] for k in pipe.captions._items} == {api.EMPTY_TABLE, tid}
    assert {k[1] for k in pipe.modulations._items} == {api.EMPTY_TABLE, tid}
    assert len(pipe.captions) == 2 and len(pipe.modulations) == 4


def merged_step(layout: str, stack: list, arrays: dict) -> np.ndarray:
    """The second velocity of a fixture from the gmlx model with the
    float64 delta of (pairs, alpha, strength) rows in its weights, placed
    through the family's key map."""
    w = qt.weights(fused=layout == "fused")
    km = registry.get_family("qwen_image21").lora_map(qt.config_of(w))
    parts: dict = {}
    for pairs, alpha, strength in stack:
        for target, (down, up) in pairs.items():
            module, band = km.targets[target]
            parts.setdefault(module, []).append(
                (down, up, alpha, band, strength))
    for module, rows in parts.items():
        name = f"{module}.weight"
        w[name] = (w[name] + reference_delta(rows, w[name].shape)).astype(
            np.float32)
    model = qt.model_of(w)
    cond = qg.conditioning(arrays, "")
    with mx.stream(mx.cpu):
        return np.array(qm.forward_target(
            model, mx.array(arrays["latents_b"]), float(arrays["sigma_b"]),
            cond, qm.prefill_prefix(model, cond)), np.float64)


def _step(pipe, arrays: dict, table: str, key: str = "k") -> np.ndarray:
    with mx.stream(mx.cpu):
        return np.array(api.denoise_step(
            pipe, mx.array(arrays["latents_b"]), float(arrays["sigma_b"]),
            qg.conditioning(arrays, key), patch_table=table), np.float64)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_strength_change_is_a_new_table_with_no_reload(tmp_path,
                                                         monkeypatch, layout):
    arrays, m = committed_fixture(TINY["alpha"])
    (pairs, alpha, strength), = adapters_of(m)
    pipe = tiny_pipe(tmp_path, layout)
    adapter = {t: (d, u, alpha) for t, (d, u) in pairs.items()}
    path = write_adapter(tmp_path / "a.safetensors", "onetrainer", adapter)
    ad = lora.load_adapter(pipe, path)
    t1 = api.define_patch_table(pipe, lora.table_entries([(ad, strength)]))
    full = _step(pipe, arrays, t1)
    assert fx.rel_l2(full, arrays["velocity_b"]) <= OWN_TOL
    held = dict(pipe.adapters._items)
    wrappers = {p: mod for p, mod in pipe.dit.named_modules()
                if isinstance(mod, lora.ImageLoRALinear)}
    weights = {p: mod.base.weight for p, mod in wrappers.items()}
    assert len(wrappers) == len(install.linear_leaves(pipe.dit)) == (
        20 if layout == "fused" else 22)

    # From here the file is gone, no reader runs, and while the new table
    # is defined and its factors are built every leaf holds zeros of its
    # shape in place of its weight.
    os.remove(path)

    def boom(*a, **k):
        raise AssertionError("a strength change read a file")

    monkeypatch.setattr(mx, "load", boom)
    monkeypatch.setattr(reader, "read_lora", boom)
    monkeypatch.setattr(patch_table, "read_lora", boom)
    real = {p: mod.base for p, mod in wrappers.items()}
    for p, mod in wrappers.items():
        mod.base = SimpleNamespace(weight=mx.zeros(weights[p].shape))
    t2 = api.define_patch_table(pipe, lora.table_entries([(ad, 0.5)]))
    built = pipe.lora._table((t2, mx.float32, install.form()))
    for module, _, _ in pipe.patch_tables[t2]["entries"]:
        assert built.get(module) is not None
    for p, mod in wrappers.items():
        mod.base = real[p]

    assert t2 not in (t1, api.EMPTY_TABLE)
    builds = []
    monkeypatch.setattr(install, "build_table", lambda *a: builds.append(a))
    half = _step(pipe, arrays, t2)
    assert builds == []
    want = merged_step(layout, [(pairs, alpha, 0.5)], arrays)
    assert fx.rel_l2(half, want) <= MERGED_TOL, fx.rel_l2(half, want)
    assert fx.rel_l2(half, full) > 1e-2
    # The same pairs serve both tables, and no weight was replaced.
    assert dict(pipe.adapters._items) == held
    assert all(mod.base.weight is weights[p] for p, mod in wrappers.items())
    assert np.array_equal(_step(pipe, arrays, t1), full)


@pytest.mark.parametrize("strength", [0.5, 1.0, -0.4])
def test_a_tiny_step_at_each_strength_equals_the_step_on_merged_weights(
        tmp_path, strength):
    """The strengths of the delta half, through the step, on the fused
    weight: against the gmlx model with the float64 delta in its weights,
    whose own step the oracle holds."""
    arrays, m = committed_fixture(TINY["alpha"])
    (pairs, alpha, _), = adapters_of(m)
    pipe = tiny_pipe(tmp_path, "fused")
    ad = lora.load_adapter(pipe, write_adapter(
        tmp_path / "a.safetensors", "kohya",
        {t: (d, u, alpha) for t, (d, u) in pairs.items()}))
    tid = api.define_patch_table(pipe, lora.table_entries([(ad, strength)]))
    got = _step(pipe, arrays, tid)
    want = merged_step("fused", [(pairs, alpha, strength)], arrays)
    err = fx.rel_l2(got, want)
    print(f"qwen check7 tiny strength {strength}: relative L2 {err:.2e} of "
          f"{MERGED_TOL:.0e}")
    assert err <= MERGED_TOL
    assert fx.rel_l2(_step(pipe, arrays, api.EMPTY_TABLE), want) > 1e-2


def test_a_qwen_forward_reads_nothing_from_the_text_row_channel(
        tmp_path, monkeypatch):
    from gmlx import lora_rows
    from gmlx.load.modules import LoRAKQuantLinear

    def boom(*a, **k):
        raise AssertionError("an image forward read gmlx.lora_rows")

    assert not issubclass(lora.ImageLoRALinear, LoRAKQuantLinear)
    assert not hasattr(install, "lora_rows")
    arrays, m = committed_fixture(TINY["alpha"])
    pipe = tiny_pipe(tmp_path, "fused")
    rows = load_rows(pipe, tmp_path, m, ("kohya",))
    for name in ("row_scales", "row_factor", "dense_row_factor",
                 "dense_rows", "flat_rows", "flat_row_factor"):
        monkeypatch.setattr(lora_rows, name, boom)
    # A serve process in rows mode with no publish must not matter either.
    monkeypatch.setattr(lora_rows, "_mode", "rows")
    tid = api.define_patch_table(pipe, lora.table_entries(rows))
    # The prefill and the target forward both run under the table.
    with mx.stream(mx.cpu):
        got = step_velocities(pipe, arrays, tid, "rows")
    for tag in "ab":
        assert got[f"velocity_{tag}"]["rel_l2"] <= OWN_TOL, got


@pytest.mark.parametrize("how,kind", (("block", "_Block"),
                                      ("slice", "_Slice"),
                                      ("split", "_Split")))
def test_each_placement_gives_the_tiny_step_of_the_merged_oracle(
        tmp_path, monkeypatch, how, kind):
    monkeypatch.setattr(install, "_PLACEMENT", how)
    arrays, m = committed_fixture(TINY["alpha"])
    pipe = tiny_pipe(tmp_path, "fused")
    rows = load_rows(pipe, tmp_path, m, ("kohya",))
    tid = api.define_patch_table(pipe, lora.table_entries(rows))
    got = ratios(pipe, arrays, tid, m["name"])
    name, worst = top(got)
    print(f"qwen check7 tiny placement {how}: worst {worst:.2f} of its "
          f"bound at {name}")
    assert worst <= 1, (name, worst)
    built = pipe.lora._cur[(tid, mx.float32, install.form())]
    ffn = pipe.dit.cfg.ffn_dim
    # Block 0 has both bands of the fused weight and block 1 its gate
    # band alone. A whole weight is a block under every placement.
    both, gate = (built.get(f"{B}{i}.img_mlp.gate_up") for i in (0, 1))
    assert type(both).__name__ == kind and type(gate).__name__ == kind
    assert type(built.get(B + "0.img_mlp.out")).__name__ == "_Block"
    if how != "block":
        assert [s[:2] for s in both.spans] == [(0, ffn), (ffn, 2 * ffn)]
        assert [s[:2] for s in gate.spans] == (
            [(0, ffn)] if how == "slice" else [(0, ffn), (ffn, 2 * ffn)])


def test_a_gate_up_key_gives_the_tiny_step_of_the_merged_oracle(pipes,
                                                               tmp_path):
    """A file from the fused layout names img_mlp.gate_up, one pair over
    both halves. Block 0's gate_layer and proj pairs of the oracle's
    adapter are written as that one pair: the downs stacked, the ups on
    the diagonal with each half's alpha over rank folded in, and no alpha
    of its own. The step is then the oracle's. A file that holds the pair
    apart has no such module, and the key is refused with the name of
    that layout."""
    arrays, m = committed_fixture(TINY["alpha"])
    (pairs, alpha, strength), = adapters_of(m)
    mlp = B + "0.img_mlp."
    (dg, ug), (dp, up) = pairs[mlp + "gate_layer"], pairs[mlp + "proj"]
    rg, rp, ffn = dg.shape[0], dp.shape[0], ug.shape[0]
    down = np.concatenate([dg, dp])
    both = np.zeros((2 * ffn, rg + rp), np.float32)
    both[:ffn, :rg] = ug * np.float32(alpha / rg)
    both[ffn:, rg:] = up * np.float32(alpha / rp)
    keys = dialect_keys("kohya", {
        t: (d, u, alpha) for t, (d, u) in pairs.items()
        if t not in (mlp + "gate_layer", mlp + "proj")})
    stem = "lora_unet_" + (mlp + "gate_up").replace(".", "_")
    keys[stem + ".lora_down.weight"] = down
    keys[stem + ".lora_up.weight"] = both
    path = write_keys(tmp_path / "gate_up.safetensors", keys)

    pipe = pipes["fused"]
    ad = lora.load_adapter(pipe, path)
    assert (mlp + "gate_up", None) in {(p, b) for p, b, _ in ad.dit}
    assert len(ad.dit) == len(pairs) - 1 and ad.warnings == ()
    tid = api.define_patch_table(pipe, lora.table_entries([(ad, strength)]))
    got = ratios(pipe, arrays, tid, "gate_up key")
    name, worst = top(got)
    print(f"qwen check7 tiny gate_up key: worst {worst:.2f} of its bound "
          f"at {name}")
    assert worst <= 1, (name, worst)
    # Without the pair on the fused weight the step misses the gate.
    del keys[stem + ".lora_down.weight"], keys[stem + ".lora_up.weight"]
    less = lora.load_adapter(pipe, write_keys(
        tmp_path / "no_gate_up.safetensors", keys))
    got = ratios(pipe, arrays, api.define_patch_table(
        pipe, lora.table_entries([(less, strength)])), "no gate_up key")
    assert top(got)[1] >= SEEN and top(got, "step ")[1] >= SEEN
    with pytest.raises(lora.AdapterError,
                       match="1 of 17 targets name the fused "
                             "img_mlp.gate_up weight, and this DiT file "
                             "stores the gate and the projection apart"):
        lora.load_adapter(pipes["unfused"], path)


def test_a_dense_gate_up_pair_gives_the_tiny_step_of_the_merged_oracle(
        pipes, tmp_path):
    """One pair over the whole fused weight with one down: its up is
    dense, the gate rows over the projection rows. The oracle merged it
    as two pairs with that down, one on each layer of the reference
    model. Written as those two pairs it is the same step on both
    layouts."""
    arrays, m = committed_fixture(SHARED)
    (pairs, alpha, strength), = adapters_of(m)
    assert (m["inputs"]["case"], alpha, strength) == ("text", 2.0, 0.7)
    dense, swapped = {}, {}
    for t, (d, u) in pairs.items():
        if t.endswith("img_mlp.gate_layer"):
            dp, up = pairs[t[:-len("gate_layer")] + "proj"]
            assert np.array_equal(d, dp)
            name = t[:-len("gate_layer")] + "gate_up"
            dense[name] = (d, np.concatenate([u, up]), alpha)
            swapped[name] = (d, np.concatenate([up, u]), alpha)
        elif not t.endswith("img_mlp.proj"):
            dense[t] = swapped[t] = (d, u, alpha)
    pipe = pipes["fused"]
    ad = lora.load_adapter(pipe, write_adapter(
        tmp_path / "dense.safetensors", "kohya", dense))
    assert sorted((p, b) for p, b, _ in ad.dit) == [
        (B + "0.img_mlp.gate_up", None), (B + "1.attn.to_q", None),
        (B + "1.img_mlp.gate_up", None)]
    tid = api.define_patch_table(pipe, lora.table_entries([(ad, strength)]))
    got = ratios(pipe, arrays, tid, "dense")
    name, worst = top(got)
    print(f"qwen check7 tiny dense gate_up pair: worst {worst:.2f} of its "
          f"bound at {name}")
    assert worst <= 1, (name, worst)
    for layout in LAYOUTS:
        rows = load_rows(pipes[layout], tmp_path, m, ("kohya",))
        got = ratios(pipes[layout], arrays, api.define_patch_table(
            pipes[layout], lora.table_entries(rows)), f"apart {layout}")
        assert top(got)[1] <= 1, (layout, top(got))
    # The halves of the dense up in the other order miss the gate.
    bad = lora.load_adapter(pipe, write_adapter(
        tmp_path / "swapped.safetensors", "kohya", swapped))
    got = ratios(pipe, arrays, api.define_patch_table(
        pipe, lora.table_entries([(bad, strength)])), "swapped")
    assert top(got)[1] >= SEEN and top(got, "step ")[1] >= SEEN


@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_header_config_scales_the_tiny_step_as_the_reference_reads_it(
        tmp_path, layout):
    """The header row of check 7: a file as the reference's
    save_lora_weights writes it, with no alpha tensor and its LoRA config
    in the safetensors header. The oracle read the file with the
    reference loader and merged every pair at the alpha over rank of the
    config that loader hands on."""
    arrays, m = committed_fixture(HEADER)
    row, = m["inputs"]["adapters"]
    assert row["alpha"] is None
    assert row["header"] == row["loader_config"] == qt.rule.LORA_HEADER
    (pairs, _, strength), = adapters_of(m)
    keys = dialect_keys("diffusers",
                        {t: (d, u, None) for t, (d, u) in pairs.items()})
    pipe = tiny_pipe(tmp_path, layout)

    def load(name: str, fields: dict | None):
        return lora.load_adapter(pipe, write_keys(
            tmp_path / f"{name}.safetensors", keys,
            None if fields is None else header(fields)))

    ad = load("header", row["header"])
    targets = patch_table.key_map(pipe).targets
    held = {(module, band): pipe.adapters.get(key)
            for module, band, key in ad.dit}
    assert len(held) == len(row["scales"]) == 18
    for target, (rank, alpha) in row["scales"].items():
        pair = held[targets[target]]
        assert (int(pair.down.shape[0]), pair.alpha) == (rank, alpha)
    assert len(ad.warnings) == 1
    assert "ComfyUI does not read the header" in ad.warnings[0]
    tid = api.define_patch_table(pipe, lora.table_entries([(ad, strength)]))
    got = ratios(pipe, arrays, tid, m["name"])
    name, worst = top(got)
    print(f"qwen check7 tiny header {layout}: {len(got)} measures, worst "
          f"{worst:.2f} of its bound at {name}")
    assert worst <= 1, (name, worst)
    # The same keys with no header run at scale 1, and a header with
    # another alpha or without its patterns at other scales. Each misses.
    wrong = {"no_header": None,
             "another_alpha": {**row["header"], "lora_alpha": 2.0},
             "no_alpha_pattern": {**row["header"], "alpha_pattern": {}}}
    for label, fields in wrong.items():
        other = load(label, fields)
        assert (other.warnings == ()) == (fields is None)
        got = ratios(pipe, arrays, api.define_patch_table(
            pipe, lora.table_entries([(other, strength)])), label)
        whole, step = top(got)[1], top(got, "step ")[1]
        print(f"qwen check7 tiny header {layout}, {label}: {step:.0f} "
              f"bounds on a step velocity and {whole:.0f} on its most "
              "moved measure")
        assert whole >= SEEN and step >= SEEN, (label, whole, step)


def test_an_alpha_scales_a_pair_in_every_spelling_of_its_keys(pipes,
                                                             tmp_path):
    """The rule: a pair is scaled by its alpha tensor over its rank in
    every spelling of its keys, as ComfyUI reads a file. The reference
    loader differs in one spelling. The oracle ran it on a pair of rank 2
    with an alpha of 3: it folds 1.5 into a pair spelled lora_down and
    lora_up, and it drops the alpha of a pair spelled lora_A and lora_B,
    which it then runs at scale 1. A file with those keys and alpha
    tensors gives another image in the reference pipeline than in gmlx
    and in ComfyUI."""
    arrays, m = committed_fixture(TINY["alpha"])
    seen = m["inputs"]["lora_keys"]["alpha"]
    assert (seen["rank"], seen["alpha"]) == (2, 3.0)
    assert seen["lora_down, lora_up"]["factor"] == 1.5
    assert seen["lora_A, lora_B"]["factor"] == 1.0
    assert not seen["lora_down, lora_up"]["alpha_returned"]
    assert not seen["lora_A, lora_B"]["alpha_returned"]
    for down_s, up_s in ((".lora_down.weight", ".lora_up.weight"),
                         (".lora_A.weight", ".lora_B.weight")):
        pair, = reader.parse_pairs({
            seen["stem"] + down_s: mx.zeros((2, 3)),
            seen["stem"] + up_s: mx.zeros((4, 2)),
            seen["stem"] + ".alpha": mx.array(3.0)}, "f").pairs.values()
        assert plan.pair_scale(pair.alpha, pair.rank) == 1.5
    # The adapter of the alpha fixture spelled lora_A and lora_B with its
    # alpha tensors: the step is the oracle's at alpha over rank. At scale
    # 1, which is the reference loader's reading of those keys, it misses.
    (pairs, alpha, strength), = adapters_of(m)
    keys = {}
    for t, (d, u) in pairs.items():
        stem = "diffusion_model." + t
        keys[stem + ".lora_A.weight"], keys[stem + ".lora_B.weight"] = d, u
        keys[stem + ".alpha"] = np.asarray(alpha, np.float32)
    path = write_keys(tmp_path / "ab_alpha.safetensors", keys)
    bare = write_keys(tmp_path / "ab.safetensors", {
        k: v for k, v in keys.items() if not k.endswith(".alpha")})
    for layout in LAYOUTS:
        pipe = pipes[layout]
        ad = lora.load_adapter(pipe, path)
        assert len(ad.dit) == len(pairs) and ad.warnings == ()
        got = ratios(pipe, arrays, api.define_patch_table(
            pipe, lora.table_entries([(ad, strength)])), "ab alpha")
        assert top(got)[1] <= 1, (layout, top(got))
        got = ratios(pipe, arrays, api.define_patch_table(
            pipe, lora.table_entries(
                [(lora.load_adapter(pipe, bare), strength)])), "ab")
        assert top(got)[1] >= SEEN and top(got, "step ")[1] >= SEEN


def test_a_leaf_kept_in_float32_adds_its_delta_in_the_dtype_of_its_product(
        tmp_path, monkeypatch):
    """A family may keep the timestep embedder in float32 under bfloat16
    activations, as Z-Image does. Its first linear then gives a float32
    product from a bfloat16 input, and the factors of a pair on it are in
    the input's dtype. The leaf casts the delta to the dtype of the
    product and adds it there."""
    emb = "time_text_embed.timestep_embedder."
    fam = replace(qt.tiny_family(), fp32_keep=("norm", "timestep_embedder."))
    pairs = {t: p for t, p in qt.rule.lora_pairs("one").items()
             if t.startswith(emb)}
    alpha, strength = 3.0, 0.8
    w = qt.weights()
    merged = dict(w)
    for t, (down, up) in pairs.items():
        merged[t + ".weight"] = (w[t + ".weight"] + reference_delta(
            [(down, up, alpha, None, strength)],
            w[t + ".weight"].shape)).astype(np.float32)

    def load(name: str, weights: dict):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(registry, "families", lambda: (fam,))
            return api.load(qt.write(tmp_path / name, weights),
                            log=lambda *_: None)

    pipe, want = load("base.gguf", w), load("merged.gguf", merged)
    for p in (pipe, want):
        t = p.dit.time_text_embed.timestep_embedder
        assert p.dit.act_dtype == mx.bfloat16
        assert t.linear_1.weight.dtype == t.linear_2.weight.dtype \
            == mx.float32
        assert p.dit.img_in.weight.dtype == mx.bfloat16
    ad = lora.load_adapter(pipe, write_adapter(
        tmp_path / "t.safetensors", "kohya",
        {t: (d, u, alpha) for t, (d, u) in pairs.items()}))
    tid = api.define_patch_table(pipe, lora.table_entries([(ad, strength)]))

    mixed, real = [], install._add

    def spy(y, z, b_t):
        out = real(y, z, b_t)
        if y.dtype != b_t.dtype:
            mixed.append((y.dtype, z.dtype, b_t.dtype, out.dtype))
        return out

    monkeypatch.setattr(install, "_add", spy)
    branch = (mx.float32, mx.bfloat16, mx.bfloat16, mx.float32)
    # The leaf on its own: against the float64 product with the merged
    # weight. The products with A and with B each round to bfloat16.
    rng = np.random.default_rng(11)
    x = mx.array(rng.standard_normal((3, qt.rule.T_FREQ)).astype(
        np.float32)).astype(mx.bfloat16)
    with mx.stream(mx.cpu), qg.under_table(pipe, tid):
        leaf = pipe.dit.time_text_embed.timestep_embedder.linear_1
        got, base = leaf(x), leaf.base(x)
        mx.eval(got, base)
    assert mixed == [branch] and got.dtype == mx.float32
    ref = np.array(x.astype(mx.float32), np.float64) @ merged[
        emb + "linear_1.weight"].astype(np.float64).T
    gate = 2 * 2 * qg.SITE_UNIT
    err, lost = fx.rel_l2(np.array(got), ref), fx.rel_l2(np.array(base), ref)
    # The timestep embedding through a step: against the model with the
    # delta in its float32 weights.
    arrays, _ = committed_fixture("selftest_qwen_image21_tiny_text")
    cond = qg.conditioning(arrays, "k")
    lat, sigma = mx.array(arrays["latents_b"]), float(arrays["sigma_b"])
    mixed.clear()
    with mx.stream(mx.cpu):
        v = api.denoise_step(pipe, lat, sigma, cond, patch_table=tid)
        mx.eval(v)
        assert mixed and set(mixed) == {branch}
        t = pipe.dit.timesteps(sigma, 1)
        with qg.under_table(pipe, tid):
            temb = np.array(pipe.dit.temb(t))
        temb_want = np.array(want.dit.temb(t))
        temb_base = np.array(pipe.dit.temb(t))
        v_want = api.denoise_step(want, lat, sigma, cond)
    t_err, t_lost = fx.rel_l2(temb, temb_want), fx.rel_l2(temb_base,
                                                          temb_want)
    v_err = fx.rel_l2(np.array(v), np.array(v_want))
    print(f"qwen check7 tiny float32 leaf under bf16: leaf {err:.2e} of "
          f"{gate:.2e} ({lost:.2e} without its delta), timestep embedding "
          f"{t_err:.2e} ({t_lost:.2e} without), step velocity {v_err:.2e}")
    assert err <= gate and lost >= 10 * gate
    assert t_err <= gate and t_lost >= 10 * gate


def test_load_adapter_refuses_a_pair_of_another_width(tmp_path, pipes):
    """A name of this family with the shape of a relative, such as the
    older Qwen-Image transformer's blocks, is refused by its shape."""
    rng = np.random.default_rng(8)
    pipe = pipes["fused"]
    wide = {B + "0.attn.to_q": (
        rng.standard_normal((2, 48)).astype(np.float32),
        rng.standard_normal((48, 2)).astype(np.float32), None)}
    with pytest.raises(lora.AdapterError,
                       match="does not fit transformer_blocks.0.attn.to_q "
                             "of this Qwen-Image 2.1 DiT"):
        lora.load_adapter(pipe, write_adapter(
            tmp_path / "wide.safetensors", "diffusers", wide))
    # A projection pair with the rows of the whole fused weight.
    ffn, dim = pipe.dit.cfg.ffn_dim, pipe.dit.cfg.dim
    tall = {B + "0.img_mlp.proj": (
        rng.standard_normal((2, dim)).astype(np.float32),
        rng.standard_normal((2 * ffn, 2)).astype(np.float32), None)}
    with pytest.raises(lora.AdapterError, match="does not fit"):
        lora.load_adapter(pipe, write_adapter(
            tmp_path / "tall.safetensors", "diffusers", tall))


def test_the_key_map_resolves_what_the_reference_loader_resolves():
    """The oracle ran diffusers' own Qwen LoRA loader on one pair for
    every linear layer of the reference class in every key spelling. The
    key map places every key that the loader places, on the same layer,
    and also the keys the loader gets wrong at this revision."""
    m = committed(TINY["alpha"])
    keys = m["inputs"]["lora_keys"]
    layers = keys["linear_modules"]
    assert sorted(layers) == sorted(
        k[:-len(".weight")] for k, v in qt.unfuse(qt.weights()).items()
        if v.ndim == 2)
    spell = {k: tuple(v) for k, v in keys["spellings"].items()}
    assert {k: spell[k] for k in DIALECTS} == DIALECTS
    assert spell["dotted"][0] == "diffusion_model."
    kept = {}
    for layout in LAYOUTS:
        cfg = qt.config_of(qt.weights(fused=layout == "fused"))
        km = registry.get_family("qwen_image21").lora_map(cfg)
        under = reader._underscore_index(km.targets)
        for label, rows in keys["resolved"].items():
            assert len(rows) == len(layers) == 22
            prefix, underscored = spell[label][:2]
            ok = 0
            for layer, (stem, gave) in zip(layers, rows.items(),
                                           strict=True):
                assert stem == prefix + (layer.replace(".", "_")
                                         if underscored else layer)
                hit = reader._dit_target(stem, km.targets, under)
                assert hit is not None, (label, stem)
                assert hit[0] == layer and hit[1] == underscored
                ok += gave == layer
                assert gave == layer or gave not in layers, (stem, gave)
            kept[label] = ok
    # At this revision the loader places every key of its own spelling
    # and of the dotted one, and the kohya keys but those of gate_layer,
    # of the text projection and of the modulation. It strips neither
    # prefix of the other two spellings.
    assert kept == {"kohya": 17, "diffusers": 22, "peft": 0, "onetrainer": 0,
                    "dotted": 22}
    wrong = {s: g for s, g in keys["resolved"]["kohya"].items()
             if g not in layers}
    assert sorted(wrong) == sorted(
        "lora_unet_" + n for n in (
            "txt_in_in_layer", "txt_in_out_layer", "modulation_1",
            "transformer_blocks_0_img_mlp_gate_layer",
            "transformer_blocks_1_img_mlp_gate_layer"))


def test_tiny_leaf_sites_under_a_table_are_within_their_gates(tmp_path,
                                                             monkeypatch):
    """The site gate of the linear leaves and the product gate of the
    velocity, on the tiny model in bf16 as a load gives it."""
    arrays, m = committed_fixture(TINY["alpha"])
    pipe = tiny_pipe(tmp_path, "fused", act=None)
    assert pipe.dit.act_dtype == mx.bfloat16
    rows = load_rows(pipe, tmp_path, m, ("kohya",))
    tid = api.define_patch_table(pipe, lora.table_entries(rows))
    adapted = {p for p, _, _ in pipe.patch_tables[tid]["entries"]}

    def run(key):
        sites: dict = {}
        with mx.stream(mx.cpu), qg.leaf_sites(pipe, tid, sites):
            vel = step_velocities(pipe, arrays, tid, key)
        return sites, vel

    sites, vel = run("sites")
    assert {p for p, _ in sites} == set(install.linear_leaves(pipe.dit))
    assert {p for (p, _), (_, gate) in sites.items()
            if gate == 6 * qg.SITE_UNIT} == adapted
    # One prefill and two target steps: a block leaf runs three times.
    assert max(n for p, n in sites if p == B + "0.attn.to_q") == 3
    over = {k: v for k, v in sites.items() if not v[0] <= v[1]}
    where = max(sites, key=lambda k: sites[k][0] / sites[k][1])
    print(f"qwen check7 tiny sites: {len(sites)} leaf calls, worst "
          f"{sites[where][0]:.2e} of {sites[where][1]:.2e} at {where}")
    assert not over, over
    assert not qg.product_misses(m, {}, vel), vel
    # A leaf that drops its delta is outside the gate of its site.
    monkeypatch.setattr(install, "_add", lambda y, z, b_t: y)
    bad, _ = run("dropped")
    missed = {p for (p, _), (rel, gate) in bad.items() if not rel <= gate}
    assert missed == adapted, sorted(adapted - missed)


# Qwen-Image 2.1 Q4_K_M, when the file and its fixtures are there.

REAL = "check7_qwen_image21_lora_512"
REAL_VARIANTS = {"alpha": ("kohya", "onetrainer"),
                 "plain": ("diffusers", "peft")}
PLACEMENT = "GMLX_IMAGE_LORA_PLACEMENT"


def _real_manifest(name: str) -> dict:
    if not (ORACLE / f"{name}.manifest.json").is_file():
        pytest.skip(f"{name}.manifest.json is not committed: run "
                    "scripts/image_oracle/check7_qwen_image21_lora.py on "
                    "the Qwen-Image 2.1 GGUF and commit the manifests it "
                    "writes")
    return committed(name)


def test_the_real_manifests_name_the_adapters_the_rows_load():
    """The committed manifests alone: the dialects of each variant, the
    two adapters of the stack and where they meet, one pair fixture, one
    source step and one DiT file for all, and tolerances that follow the
    rule from the gaps beside them."""
    top = committed(REAL)
    ins = top["inputs"]
    assert {v: tuple(r["dialects"])
            for v, r in ins["variants"].items()} == REAL_VARIANTS
    stack = ins["stacked"]["adapters"]
    assert [(r["prefix"], r["dialect"], r["alpha"]) for r in stack] == [
        ("", "kohya", 4.0), ("two.", "diffusers", None)]
    one, two = (set(r["targets"]) for r in stack)
    assert one == set(ins["targets"]) and two & one and two - one

    def blocks(targets):
        return {int(t.split(".")[1]) for t in targets}

    # The second adapter shares blocks with the first, has blocks of its
    # own, and leaves some of the first's alone.
    assert blocks(two) & blocks(one) and blocks(two) - blocks(one)
    assert blocks(one) - blocks(two)
    assert set(top["tensors"]) == {
        f"{r['prefix']}{half}.{t}" for r in stack for t in r["targets"]
        for half in ("down", "up")}
    names = [r["fixture"] for r in ins["variants"].values()]
    for name in (*names, ins["stacked"]["fixture"]):
        m = committed(name)
        assert m["inputs"]["pairs_fixture"] == REAL, name
        assert m["inputs"]["source"] == ins["source"], name
        assert m["files"]["dit"] == top["files"]["dit"], name
        for tag in "ab":
            gap = m["results"][f"velocity_{tag}_bf16_vs_fp32"]
            tol = m["tolerances"]
            assert tol[f"velocity_{tag}_rel_l2"] == 2 * gap["rel_l2"]
            assert tol[f"velocity_{tag}_cosine_min"] == pytest.approx(
                1 - 4 * (1 - gap["cosine"]), abs=1e-12)
    assert [r["prefix"] for r in committed(
        ins["stacked"]["fixture"])["inputs"]["adapters"]] == ["", "two."]


@pytest.fixture(scope="module")
def real(gguf_dir):
    """A pipeline over the DiT GGUF the oracle read, or a skip."""
    m = _real_manifest(REAL)
    want = m["files"]["dit"]
    for p in image_dits(gguf_dir, "qwen_image21"):
        if Path(p).name == want["name"] and os.path.getsize(p) == want["size"]:
            break
    else:
        pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")
    if fx.file_identity(p)["sha256"] != want["sha256"]:
        pytest.fail(f"{p} is not the file the oracle read")
    return api.load(p, log=lambda *_: None)


def _variant(variant: str) -> tuple[dict, dict]:
    """The arrays and the manifest of a variant's fixture, or a skip."""
    row = _real_manifest(REAL)["inputs"]["variants"][variant]
    assert tuple(row["dialects"]) == REAL_VARIANTS[variant]
    arrays, m = fixture(_real_manifest(row["fixture"])["name"])
    assert m["inputs"]["pairs_fixture"] == REAL
    return arrays, m


def _real_table(pipe, tmp, variant: str, dialect: str | None = None):
    """The arrays and manifest of a variant, the rows of its adapter in
    ``dialect`` and the id of their table."""
    arrays, m = _variant(variant)
    rows = load_rows(pipe, tmp, m, (dialect or REAL_VARIANTS[variant][0],))
    return arrays, m, rows, api.define_patch_table(
        pipe, lora.table_entries(rows))


@pytest.mark.integration
@pytest.mark.parametrize("dialect", sorted(DIALECTS))
def test_check7_one_adapter_step_at_512(real, tmp_path, dialect):
    """The product gate on the velocities of both calls, through
    ``denoise_step``, with the adapter written in each dialect."""
    variant = next(v for v, d in REAL_VARIANTS.items() if dialect in d)
    arrays, m, rows, tid = _real_table(real, tmp_path, variant, dialect)
    got = step_velocities(real, arrays, tid, f"{variant} {dialect}")
    print(f"qwen check7 step {dialect} ({variant}): "
          + ", ".join(f"{k} {g['rel_l2']:.4e} of "
                      f"{m['tolerances'][k + '_rel_l2']:.4e}"
                      for k, g in got.items()))
    assert not qg.product_misses(m, {}, got), got
    # The base model, and the adapter without its alpha, fail the gate.
    base = step_velocities(real, arrays, api.EMPTY_TABLE, "base")
    assert qg.product_misses(m, {}, base), base
    if rows and m["inputs"]["adapters"][0]["alpha"] is not None:
        bare = load_rows(real, tmp_path, m, ("diffusers",), bare=True)
        wrong = step_velocities(real, arrays, api.define_patch_table(
            real, lora.table_entries([(bare[0][0], rows[0][1])])), "bare")
        assert qg.product_misses(m, {}, wrong), wrong


@pytest.mark.integration
@pytest.mark.parametrize("variant", list(REAL_VARIANTS))
def test_check7_product_gate_block_by_block_at_512(real, tmp_path, variant):
    """The product gate on every row, in both of the oracle's modes."""
    arrays, m, _, tid = _real_table(real, tmp_path, variant)
    with qg.under_table(real, tid):
        rows = qg.check3_rows(real.dit, arrays)
        free = {**qg.free_run(real.dit, arrays, own_temb=True),
                **qg.timestep_rows(real.dit, arrays)}
    print(f"qwen check7 block by block {variant}: "
          + qg.product_reading(m, rows, free))
    misses = qg.product_misses(m, rows, free)
    assert not misses, misses
    # A negative control: the base model. Every row of a block is fed the
    # merged oracle's input, so a row can only miss in a block the
    # adapter is in. The row does not see the adapter in every block: a
    # block's output is its input plus two gated branches, and in the
    # first adapted blocks what the adapter adds to it is under twice the
    # oracle's own bf16 gap. The exact gate leaves the adapter out of
    # each block in turn.
    adapted = {int(t.split(".")[1])
               for t in m["inputs"]["adapters"][0]["targets"]}
    with qg.under_table(real, api.EMPTY_TABLE):
        base = qg.check3_rows(real.dit, arrays)
    tol = m["tolerances"]["rows_rel_l2"]
    share = {r: g["rel_l2"] / tol[r] for r, g in base.items() if r in tol}
    top: dict = {}
    for r, v in share.items():
        if "transformer_blocks." in r:
            i = int(r.rsplit(".", 1)[1])
            top[i] = max(top.get(i, 0.0), v)
    seen = {i for i, v in top.items() if v > 1}
    clean = max(v for i, v in top.items() if i not in adapted)
    print(f"qwen check7 block by block {variant}, base model: the largest "
          "row of each adapted block over its gate: "
          + ", ".join(f"{i} {top[i]:.2f}" for i in sorted(adapted))
          + f"; a block without the adapter reads at most {clean:.2f}")
    assert not [r for r, v in share.items()
                if v > 1 and "transformer_blocks." not in r]
    # What the control holds: the base model misses a block row, and
    # only in a block the adapter is in. It does not hold which of those
    # blocks miss, which the exact gate does.
    assert seen and seen <= adapted, (sorted(seen), sorted(adapted))


@pytest.mark.integration
def test_check7_two_stacked_adapters_at_512(real, tmp_path):
    """Two adapters in one table, the second on fewer layers and partly
    in blocks of its own: the product gate on the velocities through
    ``denoise_step`` and on every row, and the table of either adapter
    alone fails it."""
    row = _real_manifest(REAL)["inputs"]["stacked"]
    arrays, m = fixture(_real_manifest(row["fixture"])["name"])
    assert m["inputs"]["pairs_fixture"] == REAL
    assert [r["prefix"] for r in m["inputs"]["adapters"]] == ["", "two."]
    rows = load_rows(real, tmp_path, m,
                     [r["dialect"] for r in row["adapters"]])
    entries = lora.table_entries(rows)
    # A layer holds both adapters, either one, or neither.
    held = {}
    for path, band, patches in entries:
        held.setdefault(len(patches), set()).add((path, band))
    assert set(held) == {1, 2}, {k: len(v) for k, v in held.items()}
    one, two = (set(r["targets"]) for r in m["inputs"]["adapters"])
    assert two - one and two & one and one - two
    if real.dit.cfg.fused_mlp:
        # The gate half of a fused weight holds two adapters where the
        # projection half holds one.
        ffn = real.dit.cfg.ffn_dim
        block = min(int(t.split(".")[1]) for t in two & one)
        path = f"{B}{block}.img_mlp.gate_up"
        assert (path, (0, 0, ffn)) in held[2]
        assert (path, (0, ffn, ffn)) in held[1]
    tid = api.define_patch_table(real, entries)
    got = step_velocities(real, arrays, tid, "stacked")
    print("qwen check7 stacked: "
          + ", ".join(f"{k} {g['rel_l2']:.4e} of "
                      f"{m['tolerances'][k + '_rel_l2']:.4e}"
                      for k, g in got.items()))
    assert not qg.product_misses(m, {}, got), got
    with qg.under_table(real, tid):
        blocks = qg.check3_rows(real.dit, arrays)
        free = {**qg.free_run(real.dit, arrays, own_temb=True),
                **qg.timestep_rows(real.dit, arrays)}
    print("qwen check7 stacked, block by block: "
          + qg.product_reading(m, blocks, free))
    misses = qg.product_misses(m, blocks, free)
    assert not misses, misses
    for i, alone in enumerate(rows):
        wrong = step_velocities(real, arrays, api.define_patch_table(
            real, lora.table_entries([alone])), f"alone {i}")
        assert qg.product_misses(m, {}, wrong), (i, wrong)


def _stacked_table(pipe, tmp):
    """The arrays and manifest of the stack's fixture, the rows of its
    two adapters and the id of their table."""
    row = _real_manifest(REAL)["inputs"]["stacked"]
    arrays, m = fixture(_real_manifest(row["fixture"])["name"])
    assert m["inputs"]["pairs_fixture"] == REAL
    rows = load_rows(pipe, tmp, m, [r["dialect"] for r in row["adapters"]])
    return arrays, m, rows, api.define_patch_table(
        pipe, lora.table_entries(rows))


def _exact_gate(pipe, tmp, variant: str, label: str) -> None:
    """The right table matches the merged oracle model on every measure
    in the oracle's arithmetic, and each wrong table fails: those of
    ``wrong_tables``, and for one adapter those of ``structural_tables``.

    The wrong tables pin the gates from above. The weakest, one half of
    the fused weight without its delta in the last block, moves its most
    moved measure by 2.4e-3, and the product gates pass it."""
    stacked = variant == "stacked"
    arrays, m, rows, tid = (_stacked_table(pipe, tmp) if stacked
                            else _real_table(pipe, tmp, variant))
    with exact_products(pipe.dit) as used:
        with qg.under_table(pipe, tid):
            got = qg.step(pipe.dit, arrays)
            # The embedder's two K-quant products, which a step with the
            # oracle's timestep embedding does not run.
            own = qg.timestep_rows(pipe.dit, arrays)
        assert_all_dense(pipe.dit, used)
        assert not qg.missed(own, EXACT_TOL), qg.missed(own, EXACT_TOL)
        assert (used.direct > 0) == (install.placement() == "split"), (
            install.placement(), used.direct)
        name = max(got, key=lambda r: got[r]["rel_l2"])
        forced = max((r for r in got if not free_running(r)),
                     key=lambda r: got[r]["rel_l2"])
        print(f"qwen check7 exact {label}: {len(got)} measures within "
              f"{got[name]['rel_l2']:.2e} of the oracle, at {name}, the "
              f"teacher-forced rows within {got[forced]['rel_l2']:.2e}, "
              f"at {forced}; gates {EXACT_TOL:.0e} and "
              f"{QWEN_EXACT_FREE_TOL:.0e} free-running")
        assert not qg.exact_missed(got), qg.exact_missed(got)
        entries = lora.table_entries(rows)
        blocks = sorted({_block(p) for p, _, _ in entries} - {None})
        if stacked:
            # The first, the middle and the last block left out, and each
            # adapter of the stack alone.
            wrong = wrong_tables(pipe, tmp, m, rows, each=False)
            assert {"table_of_adapter_0_alone",
                    "table_of_adapter_1_alone"} <= set(wrong)
        else:
            # The adapter left out of each block in turn: the product
            # gate on a block's row does not see it in every block.
            wrong = wrong_tables(pipe, tmp, m, rows, each=False,
                                 every_block=True)
            assert {k for k in wrong if k.startswith("block_")} == {
                f"block_{i}_left_out" for i in blocks}
            more = structural_tables(pipe, entries)
            first, last = blocks[0], blocks[-1]
            halves = ("gate", "proj")
            need = {f"{h}_half_left_out_in_block_{i}" for h in halves
                    for i in (first, last)}
            need |= {f"halves_swapped_in_block_{last}",
                     "sign_of_every_strength", f"sign_in_block_{first}",
                     f"sign_in_block_{last}"}
            assert need <= set(more), sorted(need - set(more))
            moved = [k for k in more if "_delta_on_block_" in k]
            assert {int(k.split("_")[1]) for k in moved} >= {first, last}
            assert not set(more) & set(wrong)
            wrong.update(more)
        weakest = None
        for fault, table in wrong.items():
            with qg.under_table(pipe, api.define_patch_table(pipe, table)):
                missed = qg.exact_missed(qg.step(pipe.dit, arrays))
            assert missed, (label, fault)
            high = max(missed, key=missed.get)
            over = max(v / qg.exact_gate(r) for r, v in missed.items())
            if weakest is None or over < weakest[0]:
                weakest = (over, fault)
            print(f"qwen check7 exact {label}, {fault}: {len(missed)} "
                  f"measures miss, up to {missed[high]:.2e} at {high}, "
                  f"{over:.0f} of its gate at most")
        print(f"qwen check7 exact {label}: {len(wrong)} wrong tables, the "
              f"weakest is {weakest[1]} at {weakest[0]:.0f} of a gate")


@pytest.mark.integration
@pytest.mark.parametrize("variant", [*REAL_VARIANTS, "stacked"])
def test_check7_exact_gate_at_512(real, tmp_path, request, variant):
    if variant != "stacked":
        _variant(variant)
    if not exact_process():
        return rerun_exact(request, "qwen check7 exact")
    _exact_gate(real, tmp_path, variant, variant)


@pytest.mark.integration
@pytest.mark.parametrize("variant", list(REAL_VARIANTS))
def test_check7_exact_gate_under_split_at_512(real, tmp_path, request,
                                              variant):
    """The split placement runs the fused weight as one K-quant product
    per band past its module's call, and the exact gate holds that path
    as it holds the default placement."""
    _variant(variant)
    if not exact_process() or os.environ.get(PLACEMENT) != "split":
        return rerun_exact(request, "qwen check7 exact",
                           **{PLACEMENT: "split"})
    assert install.placement() == "split"
    _exact_gate(real, tmp_path, variant, f"{variant} under split")


@pytest.mark.integration
@pytest.mark.parametrize("variant", list(REAL_VARIANTS))
def test_check7_site_gate_at_512(real, tmp_path, variant):
    """Every call of a linear leaf on the path a request takes, the
    leaves under the adapter included, is within its site gate of the
    exact product on the same input.

    The site gate holds the ops under a table, not the table. Its
    reference is the exact product with the delta of the table under
    test, so a wrong table reads as a right one: with a band left out,
    the bands swapped or every sign turned it reads no call over its
    gate. It holds that a leaf adds the delta the table gives it within
    the rounding of its ops: a leaf that added none would miss its site
    at most calls and not at all, since at the gain of the fixture the
    delta is under the gate of its leaf at part of them. The exact gate
    holds the table."""
    arrays, m, _, tid = _real_table(real, tmp_path, variant)
    sites: dict = {}
    shares: dict = {}
    routed: dict = {}
    fused: dict = {}
    with qg.leaf_sites(real, tid, sites, shares, routed, fused):
        vel = step_velocities(real, arrays, tid, f"{variant} sites")
    adapted = {p for p, _, _ in real.patch_tables[tid]["entries"]}
    assert routed.keys() <= shares.keys()
    worst = max((sites[k][0] / sites[k][1] for k in routed), default=0.0)
    print(f"qwen check7 sites {variant}: {len(routed)} of {len(shares)} "
          f"calls of an adapted leaf on an mlx-kquant op, n "
          f"{qg.bg.KERNEL_ROUNDINGS['lora']}, worst {worst:.2f} of its gate")
    # The delta of the adapter at each leaf it is in, as a share of the
    # leaf's output: what the site would read were the delta dropped.
    assert {p for p, _ in shares} == adapted
    gate = 2 * qg.LEAF_ROUNDINGS[True] * qg.SITE_UNIT
    low, mid, high = (float(np.quantile(list(shares.values()), q))
                      for q in (0.0, 0.5, 1.0))
    under = sorted(k for k, v in shares.items() if not v > gate)
    kinds: dict = {}
    for (path, _), v in shares.items():
        kinds.setdefault(path.split(".", 2)[2], []).append(v)
    print(f"qwen check7 sites {variant}: the delta is {low:.2e} to "
          f"{high:.2e} of a leaf output (median {mid:.2e}) over "
          f"{len(shares)} calls, against a site gate of {gate:.2e}; "
          f"{len(under)} calls are not above the gate, in blocks "
          f"{sorted({int(p.split('.')[1]) for p, _ in under})}; by leaf, "
          "smallest and median: "
          + ", ".join(f"{k} {min(v):.2e} {float(np.median(v)):.2e}"
                      for k, v in sorted(kinds.items())))
    # Most calls, not all, would miss their site without the delta.
    assert mid > gate and len(under) < len(shares) / 2, (mid, len(under))
    assert {p for p, _ in sites} == set(install.linear_leaves(real.dit))
    assert {key[0] for key, (_, gate) in sites.items()
            if gate == 6 * qg.SITE_UNIT and key not in fused} == adapted
    # The residual route takes only leaves with no delta. Where it ran, it
    # took every attention output and MLP down leaf the table leaves alone.
    on_resid = {p for p, _ in fused}
    assert not on_resid & adapted
    if on_resid:
        assert on_resid == {
            p for p in install.linear_leaves(real.dit)
            if p.endswith((".attn.to_out.0", ".img_mlp.out"))} - adapted
    print(f"qwen check7 sites {variant}: {len(fused)} leaf calls on the "
          "residual op")
    where = max(sites, key=lambda k: sites[k][0] / sites[k][1])
    print(f"qwen check7 sites {variant}: {len(sites)} leaf calls, worst "
          f"{sites[where][0]:.2e} of {sites[where][1]:.2e} at {where}")
    over = {k: v for k, v in sites.items() if not v[0] <= v[1]}
    assert not over, over
    assert not qg.product_misses(m, {}, vel), vel
