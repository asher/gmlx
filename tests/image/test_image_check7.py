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

The step is gated block by block, as check 3 gates the base model: each
block of the DiT is fed the merged oracle model's fp32 input to it and
compared with that model's output, and so are its attention and
feed-forward modules. Two gates read those rows.

The exact gate runs fp32 activations with exact float32 products and every
K-quant product on its dequantized weight, which is the arithmetic the
oracle's fp32 run does. The right table then matches the oracle on every
row to a few parts in a million, and the gate is EXACT_TOL. Each wrong
table must fail it: the pairs of a block group left out, the strength at
0.9 and 1.1 times its value, and q swapped with k. A left out group moves
the rows of that group alone, by the adapter's effect as the manifest
records it. MLX reads MLX_ENABLE_TF32 once per process, so with TF32 on
the test reruns itself in a child process with it off.

The product gate runs the path a request takes, bf16 activations on the
K-quant kernels, at the manifest's tolerances: every block row, the free
running blocks and the velocity at the rule, and the branch rows at
BRANCH_ALLOWANCE times it. The velocity sums 34 blocks, and a fault in one
block or a strength a tenth off moves it less than its tolerance. The
block rows see a wrong table where the oracle's effect puts it two
tolerances out, and the mutation rows record which ones that is.

The delta half of check 7 runs on the CPU in test_image_lora_map.py. The
rows here need the GGUF under KQUANT_TEST_GGUF_DIR and the arrays under
GMLX_IMAGE_FIXTURES.
"""
from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from lora_fixtures import DIALECTS, write_adapter
from oracle_fixture import committed, fixture, fx
from test_image_check3 import check3_rows, free_run, gate

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


# A branch row of a K-quant layer carries the kernels' own precision, which
# fp32 activations do not lower: the base model stands at 0.81 of the
# rule's tolerance on its worst branch row in check 3. The rule's tolerance
# is one sample of the oracle's bf16 rounding on that row and moves by up
# to a factor of 2 between fixtures, so a right table reaches 1.31 of it
# (measured over the five fixtures). The product gate holds the branch
# rows at this multiple of the rule. The exact gate has no such term.
BRANCH_ALLOWANCE = 1.5

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
    """The block fixture ``name`` with the product gate's branch
    tolerances, or a skip without it."""
    need_blocks(name)
    if name not in _BLOCKS:
        _BLOCKS.clear()
        arrays, m = fixture(name)
        tol = m["tolerances"]
        tol["branch_rel_l2"] = {b: t * BRANCH_ALLOWANCE
                                for b, t in tol["branch_rel_l2"].items()}
        _BLOCKS[name] = arrays, m
    return _BLOCKS[name]


def block_rows(pipe, arrays: dict, m: dict, entries, free: bool = True):
    """Every teacher-forced row and branch row of the step under the
    table of ``entries``, and the free-running block rows, each against
    the merged oracle model."""
    from gmlx.image import api
    from gmlx.image.lora import install

    tid = api.define_patch_table(pipe, entries)
    prev = install.activate(pipe, tid)
    try:
        branches: dict = {}
        rows = check3_rows(pipe.dit, arrays, m["inputs"]["rows"], branches)
        blocks = free_run(pipe.dit, arrays, m["inputs"]["free_rows"])[0] \
            if free else None
    finally:
        install.restore(pipe, prev)
    assert sorted(branches) == sorted(m["inputs"]["branches"])
    return rows, branches, blocks


def block_misses(m: dict, rows: dict, branches: dict) -> dict:
    """Row or branch row name to (relative L2, tolerance) for each one
    that misses the manifest's tolerance."""
    tol = m["tolerances"]
    out = {r: (g["rel_l2"], tol["rows_rel_l2"][r]) for r, g in rows.items()
           if not g["rel_l2"] <= tol["rows_rel_l2"][r]}
    out.update({b: (g["rel_l2"], tol["branch_rel_l2"][b])
                for b, g in branches.items()
                if not g["rel_l2"] <= tol["branch_rel_l2"][b]})
    return out


def block_gate(pipe, name: str, entries, label: str,
               velocity: bool = True) -> None:
    """Gate the step under ``entries`` block by block against the block
    fixture ``name``: teacher-forced rows, branch rows, free-running
    blocks and, with ``velocity``, the velocity through
    ``denoise_step``."""
    from gmlx.image import api

    arrays, m = blocks_fixture(name)
    rows, branches, free = block_rows(pipe, arrays, m, entries)
    vel = fx.compare(step(pipe, arrays, api.define_patch_table(
        pipe, entries)), arrays["velocity"])
    tol = m["tolerances"]
    worst = {kind: max(((g["rel_l2"] / tol[key][n], n)
                        for n, g in got.items()))
             for kind, key, got in (("row", "rows_rel_l2", rows),
                                    ("branch", "branch_rel_l2", branches),
                                    ("free block", "free_rel_l2", free))}
    print(f"check7 blocks {label}: "
          + "; ".join(f"worst {k} {n} at {r:.2f} of its tolerance"
                      for k, (r, n) in worst.items())
          + f"; velocity {vel['rel_l2']:.4e} of "
          f"{tol['velocity_rel_l2']:.4e}, cosine {vel['cosine']:.6f} of "
          f"{tol['velocity_cosine_min']:.6f}")
    misses = [x for x in gate(m, rows, vel, free, branches)
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
    """How far a mutation moves its most moved row, in tolerances of that
    row, from the adapter's effect on each block as the oracle measured
    it. A left out group moves a row of the group by the whole effect, a
    strength factor f by |1 - f| of it, and swapped q and k move an
    attention row by about half of it, since two of its four weights
    take the wrong pairs."""
    effect = m["results"]["adapter_effect_teacher_forced"]
    tol = {**m["tolerances"]["rows_rel_l2"],
           **m["tolerances"]["branch_rel_l2"]}
    over = {r: e / tol[r] for r, e in effect.items()}
    if kind == "group":
        return max(v for r, v in over.items() if _in_group(r, arg))
    if kind == "strength":
        return abs(1 - arg) * max(over.values())
    return 0.5 * max(v for r, v in over.items() if r.endswith(".attention"))


def mutation_misses(pipe, name: str, entries, label: str) -> dict:
    """Mutation name to (the rows it misses under the block gate of
    fixture ``name``, whether the oracle's effect says the gate must see
    it). The right table misses none. A block fed the oracle's input
    shows a fault in its own pairs only, so the rows a left out group
    misses are rows of that group. A mutation the effect says is visible
    must miss a row."""
    arrays, m = blocks_fixture(name)
    rows, branches, _ = block_rows(pipe, arrays, m, entries, free=False)
    assert not block_misses(m, rows, branches)
    out = {}
    for mutation, (wrong, kind, arg) in mutations(pipe, entries).items():
        rows, branches, _ = block_rows(pipe, arrays, m, wrong, free=False)
        missed = block_misses(m, rows, branches)
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


# The exact gate. The right table measures 6.7e-6 at most over the five
# fixtures, and the weakest wrong table moves its most moved row by 6.4e-4.
EXACT_TOL = 5e-5


def tf32() -> bool:
    return os.environ.get("MLX_ENABLE_TF32", "1") != "0"


def rerun_without_tf32(request) -> None:
    """Rerun this test in a child process with TF32 off, where it gates."""
    env = dict(os.environ, MLX_ENABLE_TF32="0")
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-s", "-p", "no:cacheprovider",
         request.node.nodeid], cwd=request.config.rootpath, env=env,
        capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if "check7 exact" in line:
            print(line[line.index("check7 exact"):])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    assert " passed" in r.stdout, r.stdout[-2000:]


@contextlib.contextmanager
def exact_products(dit):
    """Run ``dit`` in the oracle's fp32 arithmetic: fp32 activations, and
    every K-quant product as a float32 product on the dequantized weight.
    The dequantized weight is a node of the lazy graph, so one eval holds
    the weights of one block. The adapter path is the product path's."""
    import mlx_kquant as kq
    from mlx_kquant.nn import KQuantLinear

    def dense(self, x, lora=None):
        w = kq.dequantize(self["weight"], self["scales"], self.kquant_type,
                          mx.float32)
        y = x @ w.T
        return y + self["bias"] if "bias" in self else y

    act, call = dit.act_dtype, KQuantLinear.__call__
    dit.act_dtype, KQuantLinear.__call__ = mx.float32, dense
    try:
        yield
    finally:
        dit.act_dtype, KQuantLinear.__call__ = act, call


def exact_rows(pipe, arrays: dict, m: dict, entries) -> dict:
    """Relative L2 of every teacher-forced row and branch row under the
    table of ``entries`` in the oracle's arithmetic."""
    with exact_products(pipe.dit):
        rows, branches, _ = block_rows(pipe, arrays, m, entries, free=False)
    return {r: g["rel_l2"] for r, g in {**rows, **branches}.items()}


def exact_gate(pipe, name: str, entries, label: str) -> None:
    """The right table matches the merged oracle model on every row, and
    each wrong table of ``mutations`` fails where its fault is."""
    assert not tf32(), "the exact gate needs MLX_ENABLE_TF32=0"
    arrays, m = blocks_fixture(name)
    effect = m["results"]["adapter_effect_teacher_forced"]
    right = exact_rows(pipe, arrays, m, entries)
    top = max(right, key=right.get)
    print(f"check7 exact {label}: the right table is within "
          f"{right[top]:.2e} of the oracle on {len(right)} rows, at {top}, "
          f"gate {EXACT_TOL:.0e}")
    assert right[top] <= EXACT_TOL, (top, right[top])
    wrong = mutations(pipe, entries)
    assert {"strength x0.9", "strength x1.1", "q and k swapped"} <= set(wrong)
    assert sum(k.endswith(" left out") for k in wrong) >= 3, sorted(wrong)
    for mutation, (table, kind, arg) in wrong.items():
        got = exact_rows(pipe, arrays, m, table)
        missed = {r: d for r, d in got.items() if not d <= EXACT_TOL}
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
    if tf32():
        return rerun_without_tf32(request)
    entries = _variant_entries(pipe, tmp_path, variant)
    assert len(mutations(pipe, entries)) == 8
    exact_gate(pipe, name, entries, variant)


@pytest.mark.integration
@pytest.mark.parametrize("variant", VARIANTS)
def test_check7_the_product_gate_fails_the_wrong_tables_it_can_see(
        pipe, tmp_path, variant):
    """Which wrong tables the product gate fails. ``mutation_misses``
    asserts each one that the oracle's effect puts two tolerances out. The
    synthetic adapter is on every block, strongly enough that this covers
    the five left out groups."""
    missed = mutation_misses(pipe, f"{NAME}_{variant}_blocks",
                             _variant_entries(pipe, tmp_path, variant),
                             variant)
    assert len(missed) == 8
    seen = sorted(k for k, (rows, _) in missed.items() if rows)
    blind = sorted(k for k, (rows, _) in missed.items() if not rows)
    print(f"check7 product mutations {variant}: the gate fails {seen}; it "
          f"passes {blind}")
    for mutation, (rows, sure) in missed.items():
        if mutation.endswith(" left out"):
            assert sure and rows, mutation
