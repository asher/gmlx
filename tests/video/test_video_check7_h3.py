"""Check 7 of the video acceptance gate for MiniMax-H3: check 3's step
of the gmlx DiT under the live 8-step LoRA, row by row, against
ComfyUI's MiniMaxH3Model with the LoRA merged into its weights.

scripts/video_oracle/check7_h3.py writes the fixture: check 3's step and
rows at 512x512x22, the LoRA merged in float32, each row's tolerance
twice the oracle's own bf16 against fp32 gap. The gates are check 3's,
with the LoRA live through its patch table: the product gate on the model
as it ships, and the exact gate in the oracle's fp32 arithmetic.

Each wrong table must miss a gate by SEEN at least: no LoRA, the fc1
halves swapped, the q and k bands of qkv swapped, the alpha left out, and
the pairs of the refiner, the first block or the last block left out. The
product gate cannot see the refiner's pairs, which move the refiner's
rows by about one percent of their gate; the exact gate sees them.

Needs the MiniMax-H3 GGUF and the 8-step LoRA under KQUANT_TEST_GGUF_DIR
and the fixture under GMLX_VIDEO_FIXTURES. It skips until all are there.
"""
from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import pytest
from block_gates import EXACT_TOL, exact_process, rerun_exact
from video_fixtures import video_dits
from video_oracle_fixture import ORACLE, committed, fixture, fixture_present, fx

import h3_gates as hg

from gmlx.image.api import define_patch_table
from gmlx.image.lora import install as lora_install
from gmlx.image.lora.patch_table import load_adapter
from gmlx.video import pipeline

NAME = "check7_h3_lora_512x512x22"
STEP = "check3_h3_512x512x22"
LORA_FILE = "minimax_h3_fl2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors"
# A wrong table is seen when it moves a measure this many times its gate.
# The last block's pairs, left out, move its row by about twice its gate.
SEEN = 1.5
# Wrong tables the product gate cannot see, which bf16 rounding hides.
PRODUCT_BLIND = {"refiner_pairs_left_out"}


def _manifest() -> dict:
    if not (ORACLE / f"{NAME}.manifest.json").is_file():
        pytest.skip(f"{NAME}.manifest.json is not committed: run "
                    "scripts/video_oracle/check7_h3.py on the MiniMax-H3 "
                    "GGUF and the 8-step LoRA and commit the manifest")
    return committed(NAME)


def test_the_manifest_follows_the_rule():
    """Check 3's rule on every row, the LoRA merged in float32 with all
    its pairs placed, and check 3's step: its file, inputs and times."""
    m = _manifest()
    hg.assert_manifest_rule(m)
    hg.exact_tols(m)
    inp, step = m["inputs"], committed(STEP)
    assert inp["lora"] == m["files"]["lora"]["name"] == LORA_FILE
    assert inp["lora_pairs"] == 208 and inp["lora_strength"] == 1.0
    assert inp["lora_merge"] == "float32"
    assert m["files"]["dit"] == step["files"]["dit"]
    for k in ("shape", "blocks", "sigma", "seed", "features", "text_rows",
              "t_video", "t_audio", "shifts", "segment_kinds"):
        assert inp[k] == step["inputs"][k], k


def _find(root: Path, want: dict) -> str:
    for p in sorted(root.rglob(want["name"])):
        if p.is_file() and os.path.getsize(p) == want["size"]:
            if fx.file_identity(str(p))["sha256"] != want["sha256"]:
                pytest.fail(f"{p} is not the file the oracle read")
            return str(p)
    pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")


@pytest.fixture(scope="module")
def real(gguf_dir):
    fixture_present(_manifest()["name"])
    arrays, m = fixture(NAME)
    want = m["files"]["dit"]
    for p in video_dits(gguf_dir, "h3"):
        if Path(p).name == want["name"] and os.path.getsize(p) == want["size"]:
            break
    else:
        pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")
    if fx.file_identity(p)["sha256"] != want["sha256"]:
        pytest.fail(f"{p} is not the file the oracle read")
    pipe = pipeline.load(p, log=lambda *_: None)
    lora = _find(gguf_dir, m["files"]["lora"])
    tid = pipeline.adapter_table(pipe, [(lora, 1.0)])
    return pipe, lora, tid, arrays, m


def _under(pipe, tid: str, fn, *args):
    """``fn(pipe.dit, *args)`` under patch table ``tid``."""
    prev = lora_install.activate(pipe, tid)
    try:
        return fn(pipe.dit, *args)
    finally:
        lora_install.restore(pipe, prev)


def _measures(pipe, tid: str, arrays: dict, m: dict) -> dict:
    return _under(pipe, tid, hg.measures, arrays, m)


@pytest.mark.integration
def test_check7_rows(real):
    pipe, _, tid, arrays, m = real
    assert pipe.dit.act_dtype == mx.bfloat16
    assert not hg.layout_misses(pipe.dit, arrays, m)
    got = _measures(pipe, tid, arrays, m)
    assert not hg.unmeasured(got, m)
    print(hg.worst_line(f"h3 check7 {NAME}", got, m))
    out = hg.missed(got, m["tolerances"])
    assert not out, out


@pytest.mark.integration
def test_check7_exact_gate(real, request):
    """Every teacher-forced row within EXACT_TOL of the oracle in its
    arithmetic, and the free rows and the velocities within the fixture's
    gates from the floors measured now."""
    if not exact_process():
        return rerun_exact(request, "h3 check7 exact")
    pipe, _, tid, arrays, m = real
    label = f"h3 check7 exact {NAME}"
    with hg.exact_floats(pipe.dit) as n:
        floor = _under(pipe, tid, hg.exact_free_floor, arrays, m)
        if os.environ.get(hg.RECORD) == "1":
            m = hg.record_floors(m, floor, (
                ORACLE / f"{NAME}.manifest.json",
                fx.fixture_root() / f"{NAME}.manifest.json"))
        tol = hg.check_floor(m, floor, label)
        got = _under(pipe, tid, hg.exact_measures, arrays, m)
    print(hg.exact_line(f"{label}, {n} float tensors cast to fp32", got,
                        tol))
    over = hg.exact_over(got, tol)
    assert not over, over


def _swapped(pipe, key: str, a: int, b: int, c: int) -> str:
    """A pair whose up rows a:b and b:c trade places."""
    p = pipe.adapters.get(key)
    up = mx.concatenate([p.up[:a], p.up[b:c], p.up[a:b], p.up[c:]])
    return lora_install.register_pair(pipe, p.down, up, p.alpha)


def wrong_tables(pipe, lora: str) -> dict:
    """Name to patch table id of each wrong table."""
    rows = load_adapter(pipe, lora).dit
    cfg = pipe.dit.cfg

    def table(rows):
        return define_patch_table(pipe, [(mod, band, [(key, 1.0)])
                                         for mod, band, key in rows])

    def each(fn):
        return [(mod, band, fn(mod, key)) for mod, band, key in rows]

    def no_alpha(_mod, key):
        p = pipe.adapters.get(key)
        return lora_install.register_pair(pipe, p.down, p.up, None)

    last = f"blocks.{len(pipe.dit.blocks) - 1}."
    return {
        "no_lora": pipe.default_table,
        "fc1_halves_swapped": table(each(
            lambda mod, key: _swapped(pipe, key, 0, cfg.ffn,
                                      2 * cfg.ffn)
            if mod.endswith("mlp.fc1") else key)),
        "q_and_k_bands_swapped": table(each(
            lambda mod, key: _swapped(pipe, key, 0, cfg.inner, 2 * cfg.inner)
            if mod.endswith("qkv_proj") else key)),
        "alpha_left_out": table(each(no_alpha)),
        "refiner_pairs_left_out": table(
            [r for r in rows if not r[0].startswith("token_refiner.")]),
        "first_block_pairs_left_out": table(
            [r for r in rows if not r[0].startswith("blocks.0.")]),
        "last_block_pairs_left_out": table(
            [r for r in rows if not r[0].startswith(last)]),
    }


@pytest.mark.integration
def test_each_wrong_table_fails_the_check7_gate(real):
    """The right table passes after the wrong ones."""
    pipe, lora, tid, arrays, m = real
    # The wrong tables hold their own pairs beside the LoRA's.
    pipe.adapters.max_bytes = max(pipe.adapters.max_bytes, 8 << 30)
    tables = {k: v for k, v in wrong_tables(pipe, lora).items()
              if k not in PRODUCT_BLIND}
    tols = m["tolerances"]
    seen = {}
    for name, wrong in tables.items():
        got = _measures(pipe, wrong, arrays, m)
        ratio = {k: hg.over(v, k, tols) for k, v in got.items()}
        top = max(ratio, key=ratio.get)
        seen[name] = (ratio[top], top)
        print(f"h3 check7 wrong table {name}: {ratio[top]:.2f} of its "
              f"gate at {top}")
    weak = {k: v for k, v in seen.items() if not v[0] >= SEEN}
    assert not weak, weak
    out = hg.missed(_measures(pipe, tid, arrays, m), tols)
    assert not out, out


@pytest.mark.integration
def test_each_wrong_table_fails_the_check7_exact_gate(real, request):
    """In the oracle's arithmetic each wrong table moves a teacher-forced
    row by SEEN times EXACT_TOL at least, and the right table passes after
    them."""
    if not exact_process():
        return rerun_exact(request, "h3 check7 exact")
    pipe, lora, tid, arrays, m = real
    pipe.adapters.max_bytes = max(pipe.adapters.max_bytes, 8 << 30)
    tables = {**wrong_tables(pipe, lora), "right table": tid}
    seen = {}
    with hg.exact_floats(pipe.dit):
        for name, t in tables.items():
            got = _under(pipe, t, hg.exact_forced, arrays, m)
            top = max(got, key=got.get)
            seen[name] = (got[top] / EXACT_TOL, top)
            print(f"h3 check7 exact {name}: {seen[name][0]:.2f} of "
                  f"EXACT_TOL at {top}")
    right = seen.pop("right table")
    assert right[0] <= 1, right
    weak = {k: v for k, v in seen.items() if not v[0] >= SEEN}
    assert not weak, weak
