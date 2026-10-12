"""Check 7 of the image acceptance gate for Z-Image, step half: one
denoise step with a LoRA applied live, against the diffusers oracle with
the delta merged into dequantized dense weights.

The oracle's adapter targets the diffusers layout, where q, k and v are
separate weights, so its merge places no band. The test writes the
oracle's pairs in each dialect's key spelling, loads the file into a
pipeline over the Z-Image-Turbo GGUF, and runs check 3's step at 512 under
the patch table. A variant of the fixture fixes the alpha and strength of
the dialects that carry it. The tolerance is the manifest's: twice the
oracle's own bf16 against fp32 gap with the adapter merged, check 3's rule.
scripts/image_oracle/check7_z_image_lora.py writes the fixtures.

The step is gated as check 3 gates the base model, with the three gates
of block_gates.py. The exact gate and the product gate read the merged
oracle model's rows.

The exact gate holds the right table to the oracle on every row and every
branch row at EXACT_TOL. Each wrong table must fail it: the pairs of a
block group left out, the strength at 0.9 and 1.1 times its value, and q
swapped with k. A left out group moves the rows of that group alone, by
the adapter's effect as the manifest records it.

The product gate holds every block row, the free-running blocks and the
velocity to the manifest's tolerances. The block rows and the free-running
blocks run on the stock ops, and the velocity on the path a request takes,
kernel routes included. The velocity sums 34 blocks, and a
fault in one block or a strength a tenth off moves it less than its
tolerance. The block rows see a wrong table where the oracle's effect
puts it two tolerances out, and the mutation rows record which ones that
is.

The site gate holds each op of the step under the table to the exact op
on its own input, with the table's delta in the leaf's exact product. A
wrong production table is visible to it only where the part of a leaf's
delta it changes is over the gate of a site that reads it. The test
measures that for each wrong table and asserts the table fails or passes
as the measure says.

The delta half of check 7 runs on the CPU in test_image_lora_map.py. The
rows here need the GGUF under KQUANT_TEST_GGUF_DIR and the arrays under
GMLX_IMAGE_FIXTURES.
"""
from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from lora_fixtures import DIALECTS, write_adapter
from oracle_fixture import committed, fixture, fx
from block_gates import (EXACT_TOL, adapter_sites, assert_all_dense,
                         check3_rows, exact_missed, exact_process,
                         exact_products, free_run, gate, rerun_exact,
                         rerun_sites, right_sites, site_process, site_reach,
                         stock_ops_only, table_sites, table_velocity, thin)

NAME = "check7_z_image_turbo_lora_512"


def turbo_pipe(gguf_index, name: str):
    """A pipeline over the DiT GGUF that the fixture ``name`` was made
    from, or a skip without it."""
    if not (Path(__file__).resolve().parents[1] / "fixtures" / "image"
            / "oracle" / f"{name}.manifest.json").is_file():
        pytest.skip(f"no committed manifest for {name}")
    want = committed(name)["files"]["dit"]
    for p in gguf_index.get("lumina2", []):
        if Path(p).name == want["name"] and os.path.getsize(p) == want["size"]:
            break
    else:
        pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")
    if fx.file_identity(p)["sha256"] != want["sha256"]:
        pytest.fail(f"{p} is not the file the oracle read")
    from gmlx.image import api
    from gmlx.image.loader import load_dit
    from gmlx.image.registry import get_family

    model, _ = load_dit(p, log=lambda *_: None)
    return api.ImagePipeline(dit=model, family=get_family("z_image"),
                             variant="turbo")


@pytest.fixture(scope="module")
def pipe(gguf_index):
    return turbo_pipe(gguf_index, NAME)


def step(pipe, arrays, table=None) -> np.ndarray:
    from gmlx.image import api

    v = api.denoise_step(pipe, mx.array(arrays["latents"]),
                         float(arrays["sigma"]),
                         api.Conditioning(mx.array(arrays["cap_feats"])),
                         patch_table=table)
    return np.array(v)


# The block fixture read last. A fixture is about 1.7 GiB, and the rows of
# one variant or adapter run next to each other.
_BLOCKS: dict = {}


def need_blocks(name: str) -> None:
    """Skip without the manifest or the arrays of block fixture ``name``."""
    if not (Path(__file__).resolve().parents[1] / "fixtures" / "image"
            / "oracle" / f"{name}.manifest.json").is_file():
        pytest.skip(f"no committed manifest for {name}")
    if not os.environ.get(fx.ENV):
        pytest.skip(f"set {fx.ENV} to the image oracle fixture directory")
    m = committed(name)
    if not (fx.fixture_root() / m["npz"]).is_file():
        pytest.skip(f"{m['npz']} is not under {fx.ENV}; run "
                    f"{m['script']} to write it")


def blocks_fixture(name: str) -> tuple[dict, dict]:
    """The block fixture ``name``, or a skip without it."""
    need_blocks(name)
    if name not in _BLOCKS:
        _BLOCKS.clear()
        _BLOCKS[name] = fixture(name)
    return _BLOCKS[name]


def block_rows(pipe, arrays: dict, m: dict, entries, free: bool = True,
               branches: bool = False):
    """Every teacher-forced row of the step under the table of
    ``entries``, the branch rows with ``branches`` and the free-running
    block rows with ``free``, each against the merged oracle model, on
    the stock ops."""
    from gmlx.image import api
    from gmlx.image.lora import install

    tid = api.define_patch_table(pipe, entries)
    prev = install.activate(pipe, tid)
    try:
        seen: dict | None = {} if branches else None
        with stock_ops_only():
            rows = check3_rows(pipe.dit, arrays, m["inputs"]["rows"], seen)
            blocks = free_run(pipe.dit, arrays,
                              m["inputs"]["free_rows"])[0] if free else None
    finally:
        install.restore(pipe, prev)
    if seen is not None:
        assert sorted(seen) == sorted(m["inputs"]["branches"])
    return rows, seen, blocks


def block_misses(m: dict, rows: dict) -> dict:
    """Row name to (relative L2, tolerance) for each row that misses the
    manifest's tolerance."""
    tol = m["tolerances"]["rows_rel_l2"]
    return {r: (g["rel_l2"], tol[r]) for r, g in rows.items()
            if not g["rel_l2"] <= tol[r]}


def block_gate(pipe, name: str, entries, label: str,
               velocity: bool = True) -> None:
    """The product gate of the step under ``entries`` against the block
    fixture ``name``: teacher-forced rows and free-running blocks on the
    stock ops and, with ``velocity``, the velocity through
    ``denoise_step`` on the path a request takes."""
    from gmlx.image import api

    arrays, m = blocks_fixture(name)
    rows, _, free = block_rows(pipe, arrays, m, entries)
    vel = fx.compare(step(pipe, arrays, api.define_patch_table(
        pipe, entries)), arrays["velocity"])
    tol = m["tolerances"]
    print(f"check7 blocks {label}: velocity {vel['rel_l2']:.4e} of "
          f"{tol['velocity_rel_l2']:.4e}, cosine {vel['cosine']:.6f} of "
          f"{tol['velocity_cosine_min']:.6f}; above 0.8 of its gate: "
          f"{thin(m, rows, vel, free) or 'none'}")
    misses = [x for x in gate(m, rows, vel, free)
              if velocity or not x.startswith("velocity ")]
    assert not misses, misses


# A fault is sure to fail the block gate when it moves a row by this many
# times the row's tolerance: the right table is within the tolerance, so
# the wrong one is at least one tolerance outside it.
SEEN = 2.0


def mutations(pipe, entries) -> dict:
    """Wrong tables made from the entries of a right one, as name to
    (entries, kind, argument): the pairs of a block group left out, with
    the group's row prefix, the strength a tenth off each way, with the
    factor, and q swapped with k."""
    cfg = pipe.dit.cfg
    inner = cfg.n_heads * cfg.head_dim
    main = sorted({int(p.split(".")[1]) for p, _, _ in entries
                   if p.startswith("layers.")})
    groups = ["noise_refiner", "context_refiner"]
    groups += [f"layers.{i}" for i in sorted({main[0], main[len(main) // 2],
                                              main[-1]})]
    out = {}
    for group in groups:
        kept = [e for e in entries if not e[0].startswith(group + ".")]
        if len(kept) < len(entries):
            out[f"{group} left out"] = (kept, "group", group)
    for f in (0.9, 1.1):
        out[f"strength x{f}"] = ([(p, b, [(k, s * f) for k, s in ps])
                                  for p, b, ps in entries], "strength", f)
    swap = {(0, 0, inner): (0, inner, inner), (0, inner, inner): (0, 0, inner)}
    out["q and k swapped"] = ([(p, swap.get(b, b), ps)
                               for p, b, ps in entries], "swap", None)
    return out


def _in_group(row: str, group: str) -> bool:
    return row == group or row.startswith(group + ".")


def visible(m: dict, kind: str, arg) -> float:
    """How far a mutation is sure to move its most moved block row, in
    tolerances of that row, from the adapter's effect on each block as
    the oracle measured it. A left out group moves a row of the group by
    the whole effect and a strength factor f by |1 - f| of it. Swapped q
    and k move a block row by a part of the effect that the manifest does
    not record, so the product gate is never sure of that one."""
    effect = m["results"]["adapter_effect_teacher_forced"]
    tol = m["tolerances"]["rows_rel_l2"]
    over = {r: e / tol[r] for r, e in effect.items() if r in tol}
    if kind == "group":
        return max(v for r, v in over.items() if _in_group(r, arg))
    if kind == "strength":
        return abs(1 - arg) * max(over.values())
    return 0.0


def mutation_misses(pipe, name: str, entries, label: str) -> dict:
    """Mutation name to (the rows it misses under the product gate of
    fixture ``name``, whether the oracle's effect says the gate must see
    it). The right table misses none. A block fed the oracle's input
    shows a fault in its own pairs only, so the rows a left out group
    misses are rows of that group. A mutation the effect says is visible
    must miss a row."""
    arrays, m = blocks_fixture(name)
    rows, _, _ = block_rows(pipe, arrays, m, entries, free=False)
    assert not block_misses(m, rows)
    out = {}
    for mutation, (wrong, kind, arg) in mutations(pipe, entries).items():
        rows, _, _ = block_rows(pipe, arrays, m, wrong, free=False)
        missed = block_misses(m, rows)
        reach = visible(m, kind, arg)
        if kind == "group":
            assert all(_in_group(r, arg) for r in missed), (mutation,
                                                            sorted(missed))
        top = max(missed.items(), key=lambda kv: kv[1][0] / kv[1][1],
                  default=None)
        print(f"check7 mutation {label}, {mutation}: the oracle's effect "
              f"puts it at {reach:.1f} tolerances; {len(missed)} rows miss"
              + ("" if top is None else f", worst {top[0]} at "
                 f"{top[1][0] / top[1][1]:.1f} of its tolerance"))
        if reach >= SEEN:
            assert missed, (label, mutation, reach)
        out[mutation] = (missed, reach >= SEEN)
    return out


def exact_rows(pipe, arrays: dict, m: dict, entries) -> dict:
    """Every teacher-forced row and branch row under the table of
    ``entries`` in the oracle's arithmetic. Under the split placement a
    module with output bands runs one product per band past its own
    call, and under any other placement no product does."""
    from gmlx.image.lora import install

    with exact_products(pipe.dit) as used:
        rows, branches, _ = block_rows(pipe, arrays, m, entries, free=False,
                                       branches=True)
    assert_all_dense(pipe.dit, used)
    assert (used.direct > 0) == (install.placement() == "split"), (
        install.placement(), used.direct)
    return {**rows, **branches}


def exact_gate(pipe, name: str, entries, label: str) -> None:
    """The right table matches the merged oracle model on every row, and
    each wrong table of ``mutations`` fails where its fault is."""
    arrays, m = blocks_fixture(name)
    effect = m["results"]["adapter_effect_teacher_forced"]
    right = exact_rows(pipe, arrays, m, entries)
    top = max(right, key=lambda r: right[r]["rel_l2"])
    print(f"check7 exact {label}: the right table is within "
          f"{right[top]['rel_l2']:.2e} of the oracle on {len(right)} rows, "
          f"at {top}, gate {EXACT_TOL:.0e}")
    assert not exact_missed(right), exact_missed(right)
    wrong = mutations(pipe, entries)
    assert {"strength x0.9", "strength x1.1", "q and k swapped"} <= set(wrong)
    assert sum(k.endswith(" left out") for k in wrong) >= 3, sorted(wrong)
    for mutation, (table, kind, arg) in wrong.items():
        missed = exact_missed(exact_rows(pipe, arrays, m, table))
        assert missed, (label, mutation)
        top = max(missed, key=missed.get)
        print(f"check7 exact {label}, {mutation}: {len(missed)} rows miss, "
              f"most {top} at {missed[top]:.2e}")
        if kind == "group":
            # The step under the table without a group is the base model
            # in that group, so its rows miss by the adapter's effect and
            # no other row moves.
            assert all(_in_group(r, arg) for r in missed), (mutation,
                                                            sorted(missed))
            assert missed[top] == pytest.approx(effect[top], rel=0.02), (
                mutation, top, missed[top], effect[top])
        elif kind == "swap":
            assert any(r.endswith(".attention") for r in missed), mutation


def pair_sites(got, where: str, missed: dict, tag: str) -> set:
    """The glu sites under ``where`` of a step whose table left that
    group's deltas out, each held to what it must read. The site's exact
    side has the two deltas of its pair and the production pair has
    none, so the site reads their change c within the rounding r of the
    ops that ran, both from ``Sites.pair_effect``. With c - r over the
    gate the site must miss, and with c + r under it the site must pass.
    On the glu kernel the gate is that of the kernel's count, so a pair
    on the kernel without its deltas fails here from a smaller delta
    than on the stock ops."""
    out = set()
    for site, groups in got.pair_effect().items():
        if not site.startswith(where):
            continue
        out.add(site)
        low = max(c - r for c, r in groups.values())
        high = max(c + r for c, r in groups.values())
        read = max(got.readings()[site].values())
        print(f"{tag}: {site} on {got.routed.get(site, 'the stock ops')} "
              f"reads {read:.2f} gates, between {low:.2f} and {high:.2f}")
        assert low - 1e-3 <= read <= high + 1e-3, (tag, site, low, read, high)
        assert low <= 1 or site in missed, (tag, site, low, missed)
        assert high >= 1 or site not in missed, (tag, site, high, missed)
    return out


def site_gate(pipe, name: str, entries, label: str) -> None:
    """The site gate of the step under the table of ``entries`` on the
    inputs of block fixture ``name``, and each wrong production table of
    ``mutations`` but the q and k swap against the exact ops under the
    right table. A wrong table is visible only where the part of a leaf's
    delta it changes is over the gate of a site that reads it, which
    ``site_reach`` takes from the deltas measured here: all of the delta
    for a group left out and a tenth of it for a strength a tenth off.
    Over the gate the table must fail, at the sites of its changed leaves
    alone, and under it the table must pass. A feed-forward pair that the
    table leaves without its deltas can run on the glu kernel, where no
    leaf site reads it: its glu site reads both deltas, and
    ``pair_sites`` holds that site to what the reader measured there.
    The reach and the misses are both read over the gates of the run, so
    this holds the reader to the deltas it measured and not a gate to its
    size. A leaf gate that is too wide fails here only where it hides
    every group left out, and one that is too narrow where the right
    table misses it."""
    arrays, _ = blocks_fixture(name)
    sites, v = table_sites(pipe, arrays, entries)
    right_sites(sites, v, f"check7 sites {label}")
    effect = sites.effect()
    assert len(effect) == len({e[0] for e in entries})
    seen = {}
    for mutation, (table, kind, arg) in mutations(pipe, entries).items():
        if kind == "swap":
            continue
        part, where = (1.0, arg + ".") if kind == "group" else \
            (abs(1 - arg), "")
        got, _ = table_sites(pipe, arrays, entries, table)
        missed, reach = got.missed(), site_reach(effect, got, part, where)
        print(f"check7 sites {label}, {mutation}: the change is "
              f"{reach:.2f} gates at the site that reads it; "
              f"{len(missed)} sites miss"
              + (f", most {max(missed, key=missed.get)} at "
                 f"{max(missed.values()):.2f} gates" if missed else ""))
        stray = sorted(missed.keys() - adapter_sites(effect, where))
        assert not stray, (label, mutation, stray)
        pairs = pair_sites(got, where, missed, f"check7 sites {label}, "
                           f"{mutation}") if kind == "group" else set()
        assert bool(missed.keys() - pairs) == (reach > 1), (
            label, mutation, reach, missed)
        seen[mutation] = bool(missed)
    assert {"strength x0.9", "strength x1.1"} <= seen.keys()
    # A group left out must fail on every fixture, in one group at least.
    assert any(v for k, v in seen.items() if k.endswith(" left out")), seen


def _loaded(pipe, tmp_path, dialect: str):
    """The fixture's pairs written in ``dialect`` and loaded, with the
    variant that carries the dialect's arithmetic."""
    from gmlx.image import lora

    arrays, m = fixture(NAME)
    variants = m["inputs"]["variants"]
    variant = next(v for v, d in variants.items() if dialect in d["dialects"])
    alpha = variants[variant]["alpha"]
    assert (alpha is not None) == DIALECTS[dialect][4]
    adapter = {t: (arrays[f"down.{t}"], arrays[f"up.{t}"], alpha)
               for t in m["inputs"]["targets"]}
    ad = lora.load_adapter(pipe, write_adapter(
        tmp_path / f"{dialect}.safetensors", dialect, adapter))
    assert len(ad.dit) == len(adapter) and ad.text == ()
    return arrays, m, variant, adapter, ad


@pytest.mark.integration
@pytest.mark.parametrize("dialect", sorted(DIALECTS))
def test_check7_one_adapter_step_at_512(pipe, tmp_path, dialect):
    from gmlx.image import api, lora

    arrays, m, variant, adapter, ad = _loaded(pipe, tmp_path, dialect)
    alpha = m["inputs"]["variants"][variant]["alpha"]
    strength = m["inputs"]["variants"][variant]["strength"]
    tid = api.define_patch_table(pipe, lora.table_entries([(ad, strength)]))
    ref = arrays[f"velocity.{variant}"]
    tol = m["tolerances"][variant]
    got = fx.compare(step(pipe, arrays, tid), ref)
    print(f"check7 step {dialect} ({variant}): rel L2 {got['rel_l2']:.4e} "
          f"of {tol['velocity_rel_l2']:.4e}, cosine {got['cosine']:.6f} "
          f"of {tol['velocity_cosine_min']:.6f}")
    assert got["rel_l2"] <= tol["velocity_rel_l2"], got
    assert got["cosine"] >= tol["velocity_cosine_min"], got
    # The base model, and the adapter without its alpha, fail the gate, so
    # it sees the adapter and its alpha.
    base = fx.compare(step(pipe, arrays), ref)
    assert base["rel_l2"] > tol["velocity_rel_l2"], base
    if alpha is not None:
        bare = {t: (d, u, None) for t, (d, u, _) in adapter.items()}
        other = lora.load_adapter(pipe, write_adapter(
            tmp_path / "bare.safetensors", "diffusers", bare))
        wrong = fx.compare(step(pipe, arrays, api.define_patch_table(
            pipe, lora.table_entries([(other, strength)]))), ref)
        assert wrong["rel_l2"] > tol["velocity_rel_l2"], wrong
        print(f"check7 step {dialect}: base {base['rel_l2']:.4e}, without "
              f"alpha {wrong['rel_l2']:.4e}")


@pytest.mark.integration
@pytest.mark.parametrize("dialect", sorted(DIALECTS))
def test_check7_one_adapter_step_block_by_block_at_512(pipe, tmp_path,
                                                       dialect):
    from gmlx.image import lora

    _, m, variant, _, ad = _loaded(pipe, tmp_path, dialect)
    strength = m["inputs"]["variants"][variant]["strength"]
    block_gate(pipe, f"{NAME}_{variant}_blocks",
               lora.table_entries([(ad, strength)]), f"{dialect} ({variant})")


VARIANTS = sorted(committed(NAME)["inputs"]["variants"])


def _variant_entries(pipe, tmp_path, variant: str) -> list:
    from gmlx.image import lora

    row = committed(NAME)["inputs"]["variants"][variant]
    _, _, _, _, ad = _loaded(pipe, tmp_path, row["dialects"][0])
    return lora.table_entries([(ad, row["strength"])])


@pytest.mark.integration
@pytest.mark.parametrize("variant", VARIANTS)
def test_check7_exact_block_gate_at_512(pipe, tmp_path, request, variant):
    """The right table matches the merged oracle model on every row in the
    oracle's arithmetic, and all eight wrong tables fail."""
    name = f"{NAME}_{variant}_blocks"
    need_blocks(name)
    if not exact_process():
        return rerun_exact(request, "check7 exact")
    entries = _variant_entries(pipe, tmp_path, variant)
    assert len(mutations(pipe, entries)) == 8
    exact_gate(pipe, name, entries, variant)


PLACEMENT = "GMLX_IMAGE_LORA_PLACEMENT"


@pytest.mark.integration
@pytest.mark.parametrize("variant", VARIANTS)
def test_check7_exact_block_gate_under_split_at_512(pipe, tmp_path, request,
                                                    variant):
    """GMLX_IMAGE_LORA_PLACEMENT=split, which a CLI or API process reads
    from its environment, runs the base of a module with output bands as
    one K-quant product per band past the module's call. The exact gate
    holds that path as it holds the default placement: the right table
    matches the merged oracle model and all eight wrong tables fail."""
    name = f"{NAME}_{variant}_blocks"
    need_blocks(name)
    if not exact_process() or os.environ.get(PLACEMENT) != "split":
        return rerun_exact(request, "check7 exact", **{PLACEMENT: "split"})
    from gmlx.image.lora import install

    assert install.placement() == "split"
    entries = _variant_entries(pipe, tmp_path, variant)
    assert len(mutations(pipe, entries)) == 8
    exact_gate(pipe, name, entries, f"{variant} under split")


@pytest.mark.integration
@pytest.mark.parametrize("variant", VARIANTS)
def test_check7_site_gate_at_512(pipe, tmp_path, request, variant):
    """Every op of every block under the table is within its gate of the
    exact op on its own input, and each wrong table fails or passes as
    its measured change says."""
    name = f"{NAME}_{variant}_blocks"
    arrays, _ = blocks_fixture(name)
    entries = _variant_entries(pipe, tmp_path, variant)
    if not site_process():
        return rerun_sites(request, "check7 sites",
                           table_velocity(pipe, arrays, entries))
    site_gate(pipe, name, entries, variant)


@pytest.mark.integration
@pytest.mark.parametrize("variant", VARIANTS)
def test_check7_the_product_gate_fails_the_wrong_tables_it_can_see(
        pipe, tmp_path, variant):
    """Which wrong tables the product gate fails. ``mutation_misses``
    asserts each one that the oracle's effect puts two tolerances out on
    a block row, which for the synthetic adapter is its last main layer
    left out."""
    missed = mutation_misses(pipe, f"{NAME}_{variant}_blocks",
                             _variant_entries(pipe, tmp_path, variant),
                             variant)
    assert len(missed) == 8
    seen = sorted(k for k, (rows, _) in missed.items() if rows)
    blind = sorted(k for k, (rows, _) in missed.items() if not rows)
    print(f"check7 product mutations {variant}: the gate fails {seen}; it "
          f"passes {blind}")
    last = max((k for k in missed if k.startswith("layers.")),
               key=lambda k: int(k.split()[0].split(".")[1]))
    assert missed[last][1] and missed[last][0], last
