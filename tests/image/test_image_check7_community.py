"""Check 7 of the image acceptance gate for Z-Image on community LoRA
files: the delta half against a float64 reference, and one denoise step
with the adapter applied live against the diffusers oracle with the delta
merged into dequantized dense weights.

The files are public adapters that stay outside the repository. The
committed manifest names each by file name, size, sha256, repository,
revision and license, and the tests look for the file under
GMLX_TEST_LORA_DIR and skip without it. The reader runs with its defaults,
so a key the Z-Image map does not cover fails the test. The one warning a
file may carry is the reader's note on underscore keys that spell
diffusers names, which the kohya spelled file does for its attention
pairs.

The delta half builds its reference from the file's own keys with a
mapping written here: q, k and v of a block are rows 0, 1 and 2 of the
fused qkv weight in thirds. The step half is check 3's step at 512 under
the patch table, gated at twice the oracle's own bf16 against fp32 gap
with the adapter merged. scripts/image_oracle/check7_z_image_community.py
writes the fixture.
"""
from __future__ import annotations

import functools
import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from lora_fixtures import reference_delta, rel_l2
from oracle_fixture import ORACLE, committed, fixture, fx
from test_image_check7 import step, turbo_pipe

NAME = "check7_z_image_turbo_community_512"
ENV = "GMLX_TEST_LORA_DIR"
DELTA_TOL = 1e-6
SUFFIXES = {".lora_down.weight": "down", ".lora_up.weight": "up",
            ".lora_A.weight": "down", ".lora_B.weight": "up",
            ".alpha": "alpha"}
QKV = ("to_q", "to_k", "to_v")


def _ids() -> list:
    p = ORACLE / f"{NAME}.manifest.json"
    if not p.is_file():
        return []
    return sorted(json.loads(p.read_text())["inputs"]["adapters"])


ADAPTERS = _ids()


@functools.lru_cache(maxsize=None)
def _file(aid: str) -> str:
    """The adapter file under GMLX_TEST_LORA_DIR, checked by sha256."""
    root = os.environ.get(ENV)
    if not root:
        pytest.skip(f"set {ENV} to a directory of the community LoRA files")
    want = committed(NAME)["inputs"]["adapters"][aid]["file"]
    for p in sorted(Path(root).expanduser().rglob(want["name"])):
        if p.stat().st_size == want["size"]:
            if fx.file_identity(p)["sha256"] != want["sha256"]:
                pytest.fail(f"{p} is not the file the oracle read")
            return str(p)
    pytest.skip(f"{want['name']} is not under {ENV}")


@pytest.fixture(scope="module")
def pipe(gguf_index):
    return turbo_pipe(gguf_index, NAME)


def _own_targets(dit) -> dict:
    """Key stem without its prefix, dotted and with underscores, to
    (module path, band): this file's own reading of the Z-Image layout."""
    cfg = dit.cfg
    width = cfg.n_heads * cfg.head_dim
    out = {}
    for group, count in (("layers", len(dit.layers)),
                         ("noise_refiner", len(dit.noise_refiner)),
                         ("context_refiner", len(dit.context_refiner))):
        for i in range(count):
            block = f"{group}.{i}"
            rows = {f"{block}.attention.{q}": (f"{block}.attention.qkv",
                                               (0, j * width, width))
                    for j, q in enumerate(QKV)}
            rows[f"{block}.attention.to_out.0"] = (f"{block}.attention.out",
                                                   None)
            for leaf in ("feed_forward.w1", "feed_forward.w2",
                         "feed_forward.w3", "adaLN_modulation.0"):
                rows[f"{block}.{leaf}"] = (f"{block}.{leaf}", None)
            for name, target in rows.items():
                out[name] = out[name.replace(".", "_")] = target
    return out


def _own_rows(path: str, dit) -> dict:
    """module path -> [(down, up, alpha, band)] in float32, read from the
    file's keys with ``_own_targets``."""
    targets = _own_targets(dit)
    parts: dict = {}
    for key, arr in mx.load(path).items():
        suffix = next(s for s in SUFFIXES if key.endswith(s))
        stem = key[:-len(suffix)]
        for prefix in ("lora_unet_", "diffusion_model."):
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
        value = np.array(arr.astype(mx.float32))
        parts.setdefault(targets[stem], {})[SUFFIXES[suffix]] = value
    rows: dict = {}
    for (module, band), p in parts.items():
        alpha = float(p["alpha"]) if "alpha" in p else None
        rows.setdefault(module, []).append((p["down"], p["up"], alpha, band))
    return rows


def _check_warnings(warnings, row: dict) -> None:
    """Only the note on underscore keys with diffusers names, and only on
    a kohya spelled file."""
    assert all("spell diffusers names" in w for w in warnings), warnings
    assert bool(warnings) == (row["spelling"] == ["kohya"]), warnings


def _resolved(pipe, aid: str):
    from gmlx.image.lora import patch_table, reader

    return reader.resolve(reader.read_lora(_file(aid)),
                          patch_table.key_map(pipe))


@pytest.mark.integration
@pytest.mark.parametrize("aid", ADAPTERS)
def test_check7_community_delta_half(pipe, aid):
    """Strengths 0.5 and 1.0 and a stacked pair, against float64, on
    every module the file targets."""
    from gmlx.image.lora import install, plan

    row = committed(NAME)["inputs"]["adapters"][aid]
    other = ADAPTERS[(ADAPTERS.index(aid) + 1) % len(ADAPTERS)]
    res = {a: _resolved(pipe, a) for a in {aid, other}}
    own = {a: _own_rows(_file(a), pipe.dit) for a in {aid, other}}
    assert len(res[aid].dit) == row["pairs"] and res[aid].text == ()
    _check_warnings(res[aid].warnings, row)
    leaves = install.linear_leaves(pipe.dit)
    cases = [([(aid, 0.5)], "0.5"), ([(aid, 1.0)], "1.0"),
             ([(aid, 0.75), (other, -0.4)], "stacked")]
    worst, modules = 0.0, 0
    for stack, label in cases:
        got: dict = {}
        want: dict = {}
        for a, strength in stack:
            for t in res[a].dit:
                got.setdefault(t.module, []).append(plan.Part(
                    t.pair.down, t.pair.up, t.pair.alpha, t.band, strength))
            for module, rows in own[a].items():
                want.setdefault(module, []).extend(
                    (*r, strength) for r in rows)
        assert set(got) == set(want)
        modules = max(modules, len(got))
        for module, parts in got.items():
            dims = install.linear_dims(leaves[module])
            lm = plan.block_diagonal(module, parts, dims)
            delta = np.array(lm.b, np.float64) @ np.array(lm.a, np.float64)
            err = rel_l2(delta, reference_delta(want[module], dims))
            worst = max(worst, err)
            assert err <= DELTA_TOL, (aid, label, module, err)
    print(f"check7 community delta {aid}: {modules} modules, worst "
          f"relative L2 {worst:.3e}")


def _step_case(pipe, aid: str) -> dict:
    """The step under the adapter and without it, against the oracle."""
    from gmlx.image import api, lora

    arrays, m = fixture(NAME)
    row = m["inputs"]["adapters"][aid]
    ad = lora.load_adapter(pipe, _file(aid))
    assert len(ad.dit) == row["pairs"] and ad.text == ()
    _check_warnings(ad.warnings, row)
    tid = api.define_patch_table(pipe, lora.table_entries(
        [(ad, row["strength"])]))
    ref = arrays[f"velocity.{aid}"]
    v, v0 = step(pipe, arrays, tid), step(pipe, arrays)
    return {"tol": m["tolerances"][aid], "got": fx.compare(v, ref),
            "base": fx.compare(v0, ref),
            "own_base": fx.compare(v0, arrays["velocity_base"]),
            # The change the adapter makes, here and in the oracle.
            "move": fx.compare(v - v0, ref - arrays["velocity_base"]),
            "effect": m["results"][aid]["adapter_vs_base_fp32"]["rel_l2"]}


@pytest.mark.integration
@pytest.mark.parametrize("aid", ADAPTERS)
def test_check7_community_step_at_512(pipe, aid):
    """The relative L2 half of the gate, and that the gate tells the
    adapter from no adapter."""
    c = _step_case(pipe, aid)
    tol, got, base = c["tol"], c["got"], c["base"]
    print(f"check7 community step {aid}: rel L2 {got['rel_l2']:.4e} of "
          f"{tol['velocity_rel_l2']:.4e}, cosine {got['cosine']:.6f} of "
          f"{tol['velocity_cosine_min']:.6f}; no adapter against the "
          f"adapter's velocity {base['rel_l2']:.4e}, oracle effect "
          f"{c['effect']:.4e}; no adapter against the oracle's base "
          f"{c['own_base']['rel_l2']:.4e} cosine "
          f"{c['own_base']['cosine']:.6f}; change rel L2 "
          f"{c['move']['rel_l2']:.4e} cosine {c['move']['cosine']:.6f}")
    assert got["rel_l2"] <= tol["velocity_rel_l2"], got
    assert got["rel_l2"] < base["rel_l2"], (got, base)
    # Where the oracle's own change is clear of the tolerance, the step
    # without the adapter fails the gate.
    if c["effect"] > 2 * tol["velocity_rel_l2"]:
        assert base["rel_l2"] > tol["velocity_rel_l2"], base


# An adapter whose merged model the oracle runs in bfloat16 closer to its
# own float32 than gmlx runs the model with no adapter: the floor made
# from that gap is then above the cosine of the gmlx step without any
# adapter, and the adapter step misses it by the same margin.
COSINE_MISS = {
    "bdsqlsz_detailed_eyes": "floor 0.997793 from an oracle gap of 0.0470, "
    "and gmlx without an adapter is at 0.997350 against the oracle's base",
}


@pytest.mark.integration
@pytest.mark.parametrize("aid", [
    pytest.param(a, marks=pytest.mark.xfail(strict=True,
                                            reason=COSINE_MISS[a]))
    if a in COSINE_MISS else a for a in ADAPTERS])
def test_check7_community_step_cosine_at_512(pipe, aid):
    """The cosine half of the gate."""
    c = _step_case(pipe, aid)
    assert c["got"]["cosine"] >= c["tol"]["velocity_cosine_min"], c["got"]
