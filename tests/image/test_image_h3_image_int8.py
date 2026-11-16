"""MiniMax-H3-Image's int8 setting, as H3 video's: the four block linears
it converts and the bytes they hold, read from the release header; the
load's refusals and its gate; and the int8 terms of a step. CPU-only, no
weights. The product gate of each setting on the release file is in
test_image_check3_h3_image.py."""
from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import mlx_kquant as kq
import pytest
from image_fixtures import Z_HEADER, fixture_view

from gmlx.image import fingerprint as ifp
from gmlx.image import int8 as neutral
from gmlx.image import loader, memfit
from gmlx.image.families.h3_image import family as hi

HEADER = "h3_image_pruned_ud_q4_k_xl.header.json"
GiB = 1 << 30
# The int8 weight and float32 scale a row of the 200 converted matrices,
# less their K-quant bytes in the release file.
DELTA = 19_279_769_600 - 12_942_698_496
# fc1 at float32, its int8 codes and its scales: the largest conversion.
TRANSIENT = 5 * 28672 * 5376 + 4 * 28672


def _kernels(monkeypatch, nax: bool) -> None:
    """An mlx-kquant with the int8 ops, on a GPU with NAX units or
    without, so a test does not depend on the machine's kq."""
    for name in ("sdpa_qk8_inputs", "sdpa_qk8", "int8_quantize_rows",
                 "int8_matmul"):
        monkeypatch.setattr(kq, name, getattr(kq, name, lambda *a, **k: None),
                            raising=False)
    monkeypatch.setattr(kq, "nax_available", lambda: nax)


def test_int8_linears_convert_the_four_linears_of_each_block():
    """H3 video's block linears less its VSA gate, which this DiT lacks:
    qkv, out, fc1 and fc2 of each of the 50 blocks, and the bytes the int8
    DiT holds over the file and while one matrix converts, exact from the
    header."""
    view = fixture_view(HEADER)
    fam = hi.FAMILY
    assert fam.int8_attention
    got = neutral.int8_matrices(view, fam.strip_prefixes, fam.int8_linears)
    assert Counter((n, k) for n, k, _ in got) == {
        (21504, 5376): 50, (5376, 7168): 50, (28672, 5376): 50,
        (5376, 14336): 50}
    assert sum(b for *_, b in got) == 12_942_698_496
    assert neutral.int8_delta(view, fam.strip_prefixes,
                              fam.int8_linears) == DELTA
    assert neutral.int8_transient(view, fam.strip_prefixes,
                                  fam.int8_linears) == TRANSIENT
    held = {v: memfit.int8_held(view, fam, v) for v in neutral.INT8}
    assert held == {"off": 0, "attention": 0, "linears": DELTA,
                    "all": DELTA}


@pytest.fixture
def gated(monkeypatch):
    """``load_dit`` on a header, stopped at its load gate, which records
    the bytes it counts and its hint."""
    import gmlx.load.preflight as pre

    class Gated(Exception):
        pass

    seen = []

    def gate(weight_bytes, model_id, *, what, hint):
        seen.append((weight_bytes, hint))
        raise Gated

    monkeypatch.setattr(pre, "find_split_shards", lambda path: [path])
    monkeypatch.setattr(loader, "_scan_shard", lambda s: SimpleNamespace(
        truncated=False, kv={}, skipped={}))
    monkeypatch.setattr(loader, "gate_load", gate)

    def go(int8: str, header: str = HEADER, variant: str = "pruned"):
        view = fixture_view(header)
        monkeypatch.setattr(loader, "view_from_scans", lambda scans: view)
        ident = ifp.identify(view, variant=variant)
        n = len(seen)
        with pytest.raises(Gated):
            loader.load_dit("/d.gguf", identified=ident, int8=int8,
                            log=lambda *a: None)
        assert len(seen) == n + 1
        return seen[-1]
    go.seen = seen
    return go


def test_the_load_gate_counts_the_int8_dit_and_its_largest_conversion(
        gated, monkeypatch):
    _kernels(monkeypatch, nax=True)
    view = fixture_view(HEADER)
    dense = memfit.file_bytes(view) + memfit.cast_bytes(view, "bfloat16")
    plain = (dense, "Try a smaller DiT quant")
    assert gated("off") == plain
    assert gated("attention") == plain
    wide = (dense + DELTA + TRANSIENT,
            "Run with int8 off or attention, or try a smaller DiT quant")
    assert gated("linears") == wide
    assert gated("all") == wide


@pytest.mark.parametrize("int8,part", [("attention", "attention"),
                                       ("linears", "linears"),
                                       ("all", "attention")])
def test_a_gpu_without_nax_is_refused_before_the_gate(gated, monkeypatch,
                                                      int8, part):
    _kernels(monkeypatch, nax=False)
    with pytest.raises(ifp.ImageModelError) as e:
        gated(int8)
    assert str(e.value) == (
        f"MiniMax-H3-Image pruned runs with int8 {part}, whose kernel needs "
        "a GPU with NAX units (M5 or newer), and this Mac's GPU has none")
    assert gated.seen == []


@pytest.mark.parametrize("int8,why", [
    ("attention", "gmlx runs Z-Image with dense attention only"),
    ("all", "gmlx runs Z-Image with dense attention only"),
    ("linears", "gmlx runs Z-Image with its file's linears only")])
def test_a_family_without_int8_refuses_each_part(gated, monkeypatch, int8,
                                                 why):
    _kernels(monkeypatch, nax=True)
    with pytest.raises(ifp.ImageModelError) as e:
        gated(int8, Z_HEADER, "turbo")
    assert str(e.value) == why
    assert gated.seen == []
    gated("off", Z_HEADER, "turbo")


def test_an_int8_step_holds_its_int8_rows_beside_a_dense_one():
    """Each row of a step holds the int8 q and k of 56 heads of 128 and a
    float32 scale a head with int8 attention, and the int8 copy of fc2's
    14336 wide input with a float32 scale per 1024 columns with int8
    linears, as H3 video prices them; no other term moves."""
    row = memfit.PEAKS["h3_image"]
    kw = dict(family="h3_image", height=1024, width=1024,
              budget=100 * GiB, overcommit=False)
    dense = memfit.plan_memory(13 * GiB, 18 * GiB, GiB, **kw)
    rows = dense.rows * memfit.image_rows(row, 1024, 1024, None)
    q_k, fc2 = 2 * 56 * 128 + 2 * 56 * 4, 14336 + 14 * 4
    for attention, linears, more in (("int8", "file", q_k),
                                     ("dense", "int8", fc2),
                                     ("int8", "int8", q_k + fc2)):
        got = memfit.plan_memory(13 * GiB, 18 * GiB, GiB, attention=attention,
                                 linears=linears, **kw)
        assert got.terms["step_peak"] - dense.terms["step_peak"] == \
            rows * more
        assert {k: v for k, v in got.terms.items() if k != "step_peak"} == \
            {k: v for k, v in dense.terms.items() if k != "step_peak"}
