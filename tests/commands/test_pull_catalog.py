"""The pull catalog: its rules on fixture entries shaped like the real
ones, and the integrity of the seeded entries themselves."""

from __future__ import annotations

import dataclasses
import os
import re
import shlex

import pytest

from gmlx.load import arch_table, catalog
from gmlx.load import remote
from gmlx.load.discovery import quant_tag
from gmlx.load.preflight import strip_shard_suffix
from helpers import banned_words

GiB = 1 << 30
K = 1024
BANNED = ("kv", "drafter", "mtp", "moe", "experts", "mmproj")


def G(x: float) -> int:
    return int(x * GiB)


@pytest.fixture(autouse=True)
def _plain_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith(("GMLX_", "KV_")) or k in ("MAX_KV_SIZE", "MLX_VLM_GGUF_SPECULATIVE"):
            monkeypatch.delenv(k, raising=False)


def _r(quant, gb, tier="", *, text=None, table=0.0, sf=None, head=False):
    return catalog.Rung(quant=quant, bytes=G(gb), path=f"Model-{quant}.gguf", tier=tier,
                        text_bytes=G(text) if text is not None else 0,
                        table_bytes=G(table), streams_from_gib=sf, head=head)


def _m(name, rungs, preferred, *, mmproj=0.0, drafter=0.0, kind="dflash2", moe=False,
       native=False, cost=(("full", 0, 4096.0, 48),), heads=32, ctx=262144,
       cross_base=False, blurb="Good at long documents and code.", speed="Quick."):
    return catalog.Model(
        name=name, title=name.replace("-", " ").title(), repo=f"org/{name}", revision="abc1234",
        rungs=tuple(sorted(rungs, key=lambda r: r.bytes)), preferred=preferred, params="27B",
        blurb=blurb, speed=speed, license="apache-2.0", moe=moe, native_mtp=native,
        mmproj="mmproj-F16.gguf" if mmproj else None, mmproj_bytes=G(mmproj),
        drafter=(catalog.Drafter(kind, "org/drafter", "def5678", "drafter-Q8_0.gguf",
                                 G(drafter), cross_base) if drafter else None),
        kv_cost=catalog.KvCost(tuple(cost)) if cost else None, heads=heads, trained_ctx=ctx)


# A 27B-shaped dense entry with a native head (the head block is 1 GiB).
DENSE = _m("dense", [
    _r("UD-IQ3_XXS", 10.18, text=9.18, head=True), _r("UD-IQ3_S", 11.21, text=10.21, head=True),
    _r("UD-Q3_K_XL", 12.24, text=11.24, head=True),
    _r("Q4_0", 14.95, "fastest", text=13.95, head=True),
    _r("UD-Q4_K_M", 15.33, "balanced", text=14.33, head=True),
    _r("UD-Q6_K", 20.47, "high", text=19.47, head=True),
    _r("Q8_0", 27.05, "xhigh", text=26.05, head=True)],
    "high", mmproj=0.86, drafter=1.92, native=True)
# A 12B-shaped dense entry with a companion drafter and no head.
MID = _m("mid", [_r("Q4_K_M", 6.63, "balanced"), _r("Q6_K", 9.11, "high"),
                 _r("Q8_0", 12.5, "xhigh")],
         "high", mmproj=0.16, drafter=0.43, kind="gemma4-assistant", heads=16, ctx=131072)
# An E4B-shaped dense entry: only _XL files, so no fastest tier.
SMALL = _m("small", [
    _r("UD-IQ2_M", 3.29), _r("UD-IQ3_XXS", 3.45), _r("UD-Q3_K_XL", 3.93),
    _r("UD-Q4_K_XL", 4.43, "balanced"), _r("UD-Q6_K_XL", 6.76, "high"),
    _r("UD-Q8_K_XL", 7.63, "xhigh")],
    "high", mmproj=0.52, drafter=0.37, kind="dflash", cost=(("full", 0, 2048.0, 40),),
    heads=16, ctx=131072)
# A Flash-Next-shaped MoE: every file carries a 26.82 GiB table.
MOE = _m("flashy", [
    _r("UD-IQ3_XXS", 76.33, text=76.32, table=26.82, sf=32),
    _r("UD-Q3_K_XL", 83.81, text=83.80, table=26.82, sf=32),
    _r("UD-IQ4_XS", 87.25, "fastest", text=87.24, table=26.82, sf=32),
    _r("UD-Q4_K_XL", 103.69, "balanced", text=103.68, table=26.82, sf=32),
    _r("UD-Q6_K_XL", 157.55, "high", text=157.54, table=26.82, sf=48),
    _r("Q8_0", 175.30, "xhigh", text=175.29, table=26.82, sf=64)],
    "balanced", mmproj=0.84, drafter=2.6, kind="qwen4exp", moe=True,
    cost=(("full", 0, 8192.0, 12), ("rows", 1, 2000000.0, 36)), ctx=262144)
# A V4.1-shaped MoE whose tiers stream only from 96 GiB and whose compact
# quants stream from 48.
BIG = _m("big", [
    _r("UD-IQ2_XXS", 279.84, text=279.83, table=125.89, sf=48),
    _r("UD-IQ3_XXS", 367.63, text=367.62, table=150.21, sf=48),
    _r("UD-Q4_K_XL", 466.32, "balanced", text=466.31, table=188.83, sf=96),
    _r("UD-Q8_K_XL", 473.46, "xhigh", text=473.45, table=188.83, sf=96)],
    "balanced", mmproj=0.9, moe=True, ctx=1048576)
# A Bonsai-9B-shaped entry: every file under 4 bits, one with a head.
NOTIER = _m("notier", [
    _r("PTQ1_0", 1.86), _r("PQ2_0", 2.25), _r("Q2_0", 2.38),
    _r("PQ2_0-MTP-Q8_0", 2.49, text=2.24, head=True)],
    "PQ2_0-MTP-Q8_0", mmproj=0.58, drafter=1.47, cost=(("full", 0, 2048.0, 32),),
    heads=16, ctx=262144)
# A 7B-shaped dense entry with no add-ons.
PLAIN = _m("plain", [_r("Q4_K_M", 5.21, "balanced"), _r("Q6_K", 6.89, "high"),
                     _r("Q8_0", 8.92, "xhigh")], "high", ctx=524288)
# One tier that fits only without its drafter on a 6.8 GiB budget.
TIGHT = _m("tight", [_r("Q4_K_M", 6.0, "balanced")], "balanced", mmproj=0.5, drafter=1.0)

ALL = (DENSE, MID, SMALL, MOE, BIG, NOTIER, PLAIN, TIGHT)


def box(cls: int, budget_gib: float | None = None) -> catalog.Box:
    b = catalog.box_for_class(cls)
    if budget_gib is None:
        return b
    return catalog.Box(b.ram, b.ws, G(budget_gib), b.ceiling, b.cap)


def _fits(m, r, b, addon, speculative, t):
    from gmlx.serve.mem_preflight import prompt_kv_bytes
    s = min(2048, t)
    return (catalog.resident_bytes(m, r, speculative) + addon
            + prompt_kv_bytes(catalog.kv_costs(m.kv_cost), t)
            + min(b.cap, m.heads * s * (t + s) * 2) <= b.ceiling)


def _banned(text: str, *, skip_commands: bool = True) -> list:
    return banned_words(text, skip_commands=skip_commands)


# -- state, bytes, fit ------------------------------------------------------

def test_rung_state_is_the_file_alone_through_weight_verdict():
    from gmlx.stream import plan as sp
    r = catalog.find_rung(MOE, "balanced")
    b128 = box(128)
    assert sp.weight_verdict(r.text_bytes, r.table_bytes, b128.budget, b128.ram) == sp.VERDICT_TABLES
    assert catalog.rung_state(MOE, r, b128) == catalog.RESIDENT
    assert catalog.rung_state(MOE, r, box(64)) == catalog.STREAMS       # 32 <= 64, over the bar
    assert catalog.rung_state(MOE, r, box(24)) == catalog.OVER          # 32 > 24
    assert catalog.needs_class(MOE, r) == 32
    # A dense file over the bar never streams, whatever its share of RAM.
    q8 = catalog.find_rung(DENSE, "xhigh")
    assert catalog.rung_state(DENSE, q8, box(32)) == catalog.OVER
    assert catalog.needs_class(DENSE, q8) == 48
    assert catalog.rung_label(DENSE, q8, box(32), 0, True) == ("needs a 48 GB Mac", "")
    assert catalog.needs_class_model(BIG) == 48
    # Add-ons never change the state: a tier that fits alone is resident.
    high = catalog.find_rung(MID, "high")
    b16 = box(16)
    assert catalog.rung_state(MID, high, b16) == catalog.RESIDENT
    assert not catalog.fits_with(MID, high, b16, catalog.addon_bytes(MID, True, True), True)


def test_resident_bytes_counts_the_head_block_only_when_speculating():
    for r in DENSE.rungs:
        off, on = catalog.resident_bytes(DENSE, r, False), catalog.resident_bytes(DENSE, r, True)
        assert on - off == r.bytes - r.text_bytes == G(1.0)
    for r in PLAIN.rungs:
        assert catalog.resident_bytes(PLAIN, r, False) == catalog.resident_bytes(PLAIN, r, True)
    zero = catalog.Rung("Q4_0", G(10), "m.gguf", "fastest", head=True)
    assert catalog.resident_bytes(DENSE, zero, True) == catalog.resident_bytes(DENSE, zero, False) == G(10)
    table = catalog.find_rung(MOE, "balanced")
    assert catalog.resident_bytes(MOE, table, False) == table.text_bytes - table.table_bytes
    head = catalog.find_rung(NOTIER, "PQ2_0-MTP-Q8_0")
    assert catalog.has_head(NOTIER, head) and not catalog.has_head(NOTIER, NOTIER.rungs[0])
    assert catalog.speculates(NOTIER, NOTIER.rungs[0], False) is False
    assert catalog.speculates(NOTIER, NOTIER.rungs[0], True) is True
    assert catalog.speculates(DENSE, DENSE.rungs[0], False) is True
    assert catalog.speculates(DENSE, DENSE.rungs[0], True, streaming=True) is False


# -- the default plan ---------------------------------------------------------

def _walk(m, b):
    """The plan the rules describe, written out: the first (set, rung)
    that is resident and fits, else the lowest streaming tier, else the
    largest streaming compact."""
    compacts = sorted(catalog.compact_candidates(m), key=lambda r: r.bytes, reverse=True)
    for mm, dr in catalog._addon_sets(m):
        addon = catalog.addon_bytes(m, mm, dr)
        for r in catalog._ladder(m) + compacts:
            if (catalog.rung_state(m, r, b) == catalog.RESIDENT
                    and catalog.fits_with(m, r, b, addon, catalog.speculates(m, r, dr))):
                return (r.quant, mm, dr, False)
    pool = catalog.tiers(m) if catalog.has_tiers(m) else list(m.rungs)
    for r in sorted(pool, key=lambda r: r.bytes) + compacts:
        if catalog.rung_state(m, r, b) == catalog.STREAMS:
            return (r.quant, False, False, True)
    return None


@pytest.mark.parametrize("cls", [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512])
@pytest.mark.parametrize("m", ALL, ids=lambda m: m.name)
def test_default_plan_is_the_walk(m, cls):
    b = box(cls)
    dp = catalog.default_plan(m, b)
    want = _walk(m, b)
    got = None if dp is None else (dp.rung.quant, dp.mmproj, dp.drafter, dp.streaming)
    assert got == want
    if dp is not None and not dp.streaming:
        assert catalog.fits_with(m, dp.rung, b, catalog.plan_addon_bytes(dp), catalog.plan_speculates(dp))


def test_default_plan_drops_quality_before_speed_before_images():
    # 27B on 32 GiB: the preferred tier is over; balanced fits with both.
    dp = catalog.default_plan(DENSE, box(32))
    assert (dp.rung.tier, dp.mmproj, dp.drafter, dp.streaming) == ("balanced", True, True, False)
    # 27B on 24 GiB: no tier fits with both; the largest compact that does.
    dp = catalog.default_plan(DENSE, box(24))
    assert (dp.rung.quant, dp.mmproj, dp.drafter) == ("UD-IQ3_S", True, True)
    # 12B on 16 GiB: high fits with the tower but not the drafter; balanced
    # with both comes first.
    dp = catalog.default_plan(MID, box(16))
    assert (dp.rung.tier, dp.mmproj, dp.drafter) == ("balanced", True, True)
    # E4B on 8 GiB: every tier is resident by file alone, none fits a set;
    # the compact rung with both.
    b8 = box(8)
    assert catalog.rung_state(SMALL, catalog.find_rung(SMALL, "balanced"), b8) == catalog.RESIDENT
    dp = catalog.default_plan(SMALL, b8)
    assert (dp.rung.quant, dp.mmproj, dp.drafter) == ("UD-IQ3_XXS", True, True)
    # The tight entry: its one tier fits only without the drafter.
    dp = catalog.default_plan(TIGHT, box(16, budget_gib=6.8))
    assert (dp.rung.tier, dp.mmproj, dp.drafter) == ("balanced", True, False)
    # Nothing fits on 64 GiB for the MoE: the lowest streaming tier, both off.
    dp = catalog.default_plan(MOE, box(64))
    assert (dp.rung.tier, dp.mmproj, dp.drafter, dp.streaming) == ("fastest", False, False, True)
    # Only compact quants stream on 64 GiB for the big MoE.
    dp = catalog.default_plan(BIG, box(64))
    assert (dp.rung.quant, dp.streaming) == ("UD-IQ3_XXS", True)
    # Nothing at all on 16 GiB.
    assert catalog.default_plan(MOE, box(16)) is None
    # A native head is never an add-on: the 27B speculates with the drafter off.
    dp = catalog.default_plan(DENSE, box(128))
    assert dp.rung.tier == "high"
    assert catalog.speculates(DENSE, dp.rung, False) is True
    # A no-tier entry walks its quants from the preferred one down.
    dp = catalog.default_plan(NOTIER, box(8))
    assert (dp.rung.quant, dp.mmproj, dp.drafter) == ("PQ2_0-MTP-Q8_0", True, True)


def test_compact_rung_rules_and_shown_rungs():
    # The default plan's rung when that is a compact quant.
    assert catalog.compact_rung(DENSE, box(24)).quant == "UD-IQ3_S"
    # None while a tier is resident and the default is a tier.
    assert catalog.compact_rung(DENSE, box(32)) is None
    # No tier resident, a compact resident by file alone but not with its
    # head block under any set: the largest resident compact shows.
    b = box(24, budget_gib=10.0)
    assert catalog.default_plan(DENSE, b) is None
    assert catalog.compact_rung(DENSE, b).quant == "UD-IQ3_XXS"
    # No tier usable: the largest compact that streams shows and is the default.
    b64 = box(64)
    assert catalog.compact_rung(BIG, b64).quant == "UD-IQ3_XXS"
    assert catalog.default_plan(BIG, b64).rung.quant == "UD-IQ3_XXS"
    # A tier streams: no compact shows.
    assert catalog.compact_rung(MOE, b64) is None
    # No-tier entries show no compact; their quants show as they are.
    assert catalog.compact_rung(NOTIER, box(8)) is None
    assert [r.quant for r in catalog.shown_rungs(NOTIER, box(8))] == [
        r.quant for r in NOTIER.rungs]
    # shown_rungs: the usable tiers plus the compact, ascending, never an over tier.
    for m in ALL:
        for cls in (8, 24, 32, 64, 128):
            bb = box(cls)
            shown = catalog.shown_rungs(m, bb)
            assert list(shown) == sorted(shown, key=lambda r: r.bytes)
            for r in shown:
                assert catalog.usable(catalog.rung_state(m, r, bb))
            if catalog.has_tiers(m):
                tiers_ = [r for r in catalog.tiers(m) if catalog.usable(catalog.rung_state(m, r, bb))]
                c = catalog.compact_rung(m, bb)
                assert set(shown) == set(tiers_) | ({c} if c else set())
    b24 = box(24)
    assert [r.quant for r in catalog.shown_rungs(DENSE, b24)] == ["UD-IQ3_S", "Q4_0", "UD-Q4_K_M"]


# -- the context figure ---------------------------------------------------------

@pytest.mark.parametrize("cls", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("m", [DENSE, MID, SMALL, MOE, NOTIER, PLAIN], ids=lambda m: m.name)
def test_context_estimate_is_the_largest_fitting_multiple_of_1k(m, cls):
    b = box(cls)
    for r in m.rungs:
        for mm, dr in catalog._addon_sets(m):
            addon = catalog.addon_bytes(m, mm, dr)
            spec = catalog.speculates(m, r, dr)
            est = catalog.context_estimate(m, r, b, addon, spec)
            assert est is not None and est % K == 0 and est <= m.trained_ctx
            if est == 0:
                assert not _fits(m, r, b, addon, spec, K)
            else:
                assert _fits(m, r, b, addon, spec, est)
                if est < m.trained_ctx:
                    assert not _fits(m, r, b, addon, spec, est + K)


def test_context_estimate_details():
    # No cost: no figure, no text.
    blind = catalog.Model(**{**DENSE.__dict__, "kv_cost": None})
    r = catalog.find_rung(DENSE, "balanced")
    assert catalog.context_estimate(blind, r, box(32), 0, True) is None
    assert catalog.context_text(blind, r, box(32), 0, True) == ""
    # The estimate never gates: states, default and groups match the blind twin.
    for cls in (8, 24, 32, 128):
        b = box(cls)
        assert [catalog.rung_state(DENSE, x, b) for x in DENSE.rungs] == [
            catalog.rung_state(blind, x, b) for x in blind.rungs]
        a, c = catalog.default_plan(DENSE, b), catalog.default_plan(blind, b)
        assert (a and a.rung.quant) == (c and c.rung.quant)
    # Qwen3.5-4B-shaped on 8 GiB: positive where a flat cap would give 0.
    four = _m("four", [_r("Q4_K_M", 2.55, "balanced")], "balanced", mmproj=0.63,
              cost=(("full", 0, 147456.0, 1),), heads=32, ctx=262144)
    b8 = box(8)
    est = catalog.context_estimate(four, four.rungs[0], b8, G(0.63), False)
    assert est and est > 0
    from gmlx.serve.mem_preflight import prompt_kv_bytes
    flat = G(2.55) + G(0.63) + b8.cap + prompt_kv_bytes(catalog.kv_costs(four.kv_cost), K)
    assert flat > b8.ceiling
    # 12B with both add-ons on 16 GiB has room.
    mid = catalog.default_plan(MID, box(16))
    assert catalog.context_estimate(MID, mid.rung, box(16), catalog.plan_addon_bytes(mid), True) > 0
    # The speculation flag reaches only resident_bytes: a native head counts
    # its block with the drafter toggle on and off alike...
    hi = catalog.find_rung(DENSE, "high")
    b128 = box(128)
    on = catalog.context_estimate(DENSE, hi, b128, G(0.86) + G(1.92), True)
    off = catalog.context_estimate(DENSE, hi, b128, G(0.86) + G(1.92), catalog.speculates(DENSE, hi, False))
    assert on == off
    # ...and an entry with a drafter and no head prices the same add-on
    # bytes the same whatever the toggle.
    q6 = catalog.find_rung(MID, "high")
    assert (catalog.context_estimate(MID, q6, box(32), G(0.59), True)
            == catalog.context_estimate(MID, q6, box(32), G(0.59), False))
    # Capped at the trained context.
    assert catalog.context_estimate(PLAIN, PLAIN.rungs[0], box(512), 0, False) == PLAIN.trained_ctx


def test_context_text_and_rung_label_words():
    b = box(32)
    r = catalog.find_rung(DENSE, "balanced")
    addon = catalog.addon_bytes(DENSE, True, True)
    est = catalog.context_estimate(DENSE, r, b, addon, True)
    assert catalog.context_text(DENSE, r, b, addon, True) == f"Room for about {est // K}K tokens of context."
    assert catalog.rung_label(DENSE, r, b, addon, True) == (f"room for ~{est // K}K", "")
    # Under 1K.
    cramped = catalog.Box(b.ram, b.ws, b.budget, G(15.4), b.cap)
    assert catalog.context_text(DENSE, r, cramped, addon, True) == "Almost no room for context."
    assert catalog.rung_label(DENSE, r, cramped, addon, True) == ("almost no room", "")
    # The compact rung carries its note; a streaming rung its words.
    c = catalog.compact_rung(DENSE, box(24))
    label, note = catalog.rung_label(DENSE, c, box(24), addon, True)
    assert label.startswith("compact, room for ~") and note == "smaller, lower quality"
    s = catalog.find_rung(MOE, "fastest")
    assert catalog.rung_label(MOE, s, box(64), 0, False) == ("from SSD, slow", "")


# -- groups ---------------------------------------------------------------------

@pytest.fixture
def small_catalog(monkeypatch):
    models = {m.name: m for m in ALL}
    monkeypatch.setattr(catalog, "MODELS", models)
    monkeypatch.setattr(catalog, "RECOMMENDED", {
        8: ("notier", "small"), 16: ("mid", "plain"), 32: ("flashy", "dense"),
        64: ("flashy",), 128: ("flashy", "dense")})
    return models


def test_recommended_for_walks_down_the_classes(small_catalog):
    # 32 GiB: the MoE's default streams there, so it is skipped; the 27B leads.
    assert [m.name for m in catalog.recommended_for(box(32))] == ["dense"]
    # 48 GiB falls into the 32 class.
    assert [m.name for m in catalog.recommended_for(box(48))] == ["dense"]
    # 64 GiB: the only name streams, so the class below is used.
    assert [m.name for m in catalog.recommended_for(box(64))] == ["dense"]
    # 128 GiB: both run in memory, in list order.
    assert [m.name for m in catalog.recommended_for(box(128))] == ["flashy", "dense"]
    # 8 GiB keeps its order.
    assert [m.name for m in catalog.recommended_for(box(8))] == ["notier", "small"]
    # A box below every class falls to the first class.
    assert catalog.memory_class(4 * GiB) == 8
    assert catalog.memory_class(36 * GiB) == 32 and catalog.memory_class(128 * GiB) == 128


def test_groups_for_never_repeats_an_entry(small_catalog):
    for cls in (8, 32, 64, 128):
        rec, fits, streams, too_big = catalog.groups_for(box(cls))
        names = [m.name for g in (rec, fits, streams, too_big) for m in g]
        assert sorted(names) == sorted(small_catalog)
        for m in fits:
            assert any(catalog.rung_state(m, r, box(cls)) == catalog.RESIDENT for r in m.rungs)
        for m in streams:
            states = {catalog.rung_state(m, r, box(cls)) for r in m.rungs}
            assert catalog.STREAMS in states and catalog.RESIDENT not in states
        for m in too_big:
            assert all(catalog.rung_state(m, r, box(cls)) == catalog.OVER for r in m.rungs)
    rec, fits, streams, too_big = catalog.groups_for(box(32))
    assert [m.name for m in too_big] == ["big"] and [m.name for m in streams] == ["flashy"]


# -- plans -------------------------------------------------------------------------

def _parse(cmd: str) -> dict:
    words = shlex.split(cmd)
    assert words[:2] == ["gmlx", "pull"]
    kw: dict = {"quant": None, "mmproj": None, "drafter": None}
    it = iter(words[3:])
    for w in it:
        if w == "--quant":
            kw["quant"] = next(it)
        elif w in ("--vision", "--no-vision"):
            kw["mmproj"] = w == "--vision"
        elif w in ("--drafter", "--no-drafter"):
            kw["drafter"] = w == "--drafter"
        else:
            raise AssertionError(w)
    return words[2], kw


def test_plan_for_and_plan_command_round_trip():
    b32, b64, b128 = box(32), box(64), box(128)
    cases = [
        (DENSE, b32, {}), (DENSE, b32, {"quant": "high", "mmproj": False}),
        (DENSE, b32, {"quant": "Q6_K"}), (DENSE, b32, {"quant": "UD-Q6_K"}),
        (DENSE, b32, {"drafter": False}), (DENSE, b32, {"mmproj": False, "drafter": False}),
        (DENSE, b128, {"quant": "xhigh"}), (DENSE, b128, {}),
        (DENSE, b32, {"quant": "UD-IQ3_S"}),
        (MOE, b64, {}), (MOE, b64, {"mmproj": True}), (MOE, b64, {"quant": "balanced"}),
        (MOE, b128, {}), (MOE, b128, {"quant": "fastest", "drafter": False}),
        (NOTIER, box(8), {}), (NOTIER, box(8), {"quant": "PTQ1_0"}),
        (TIGHT, box(16, budget_gib=6.8), {}), (TIGHT, box(16, budget_gib=6.8), {"drafter": True}),
    ]
    for m, b, kw in cases:
        plan = catalog.plan_for(m, b, **kw)
        cmd = catalog.plan_command(plan, b)
        name, parsed = _parse(cmd)
        assert name == m.name
        assert catalog.plan_for(m, b, **parsed) == plan, cmd
    # The words the command carries.
    assert catalog.plan_command(catalog.plan_for(DENSE, b32), b32) == "gmlx pull dense"
    assert catalog.plan_command(catalog.plan_for(DENSE, b32, quant="high", mmproj=False), b32) == (
        "gmlx pull dense --quant high --no-vision")
    assert catalog.plan_for(DENSE, b32, quant="Q6_K") == catalog.plan_for(DENSE, b32, quant="UD-Q6_K")
    assert catalog.plan_command(catalog.plan_for(DENSE, b32, quant="UD-IQ3_S"), b32) == (
        "gmlx pull dense --quant UD-IQ3_S")
    assert catalog.plan_command(catalog.plan_for(DENSE, b128), b128) == "gmlx pull dense"
    assert catalog.plan_command(catalog.plan_for(DENSE, b128, quant="xhigh"), b128) == (
        "gmlx pull dense --quant xhigh")
    # The tight entry's default has the drafter off: --drafter when on, nothing when off.
    tb = box(16, budget_gib=6.8)
    assert catalog.plan_command(catalog.plan_for(TIGHT, tb), tb) == "gmlx pull tight"
    assert catalog.plan_command(catalog.plan_for(TIGHT, tb, drafter=True), tb) == "gmlx pull tight --drafter"
    assert catalog.plan_command(catalog.plan_for(TIGHT, tb, drafter=False), tb) == "gmlx pull tight"
    # A streaming plan: both off by default, --vision alone when images are on.
    plan = catalog.plan_for(MOE, b64)
    assert (plan.streaming, plan.mmproj, plan.drafter) == (True, False, False)
    assert catalog.plan_command(plan, b64) == "gmlx pull flashy"
    plan = catalog.plan_for(MOE, b64, mmproj=True, drafter=True)
    assert (plan.mmproj, plan.drafter) == (True, False)
    assert catalog.plan_command(plan, b64) == "gmlx pull flashy --vision"
    # A MoE tier over the bar named with --quant is a streaming plan; a dense one is not.
    plan = catalog.plan_for(MOE, b64, quant="balanced")
    assert plan.streaming and catalog.rung_state(MOE, plan.rung, b64) == catalog.STREAMS
    plan = catalog.plan_for(DENSE, b32, quant="xhigh")
    assert not plan.streaming and catalog.rung_state(DENSE, plan.rung, b32) == catalog.OVER
    assert plan.mmproj and plan.drafter


def test_plan_for_errors():
    with pytest.raises(catalog.NoResidentRung) as e:
        catalog.plan_for(MOE, box(16))
    assert "--quant UD-IQ3_XXS" in str(e.value) and "flashy" in str(e.value)
    with pytest.raises(ValueError) as e:
        catalog.plan_for(DENSE, box(32), quant="Q5_K_M")
    msg = str(e.value)
    assert "fastest" in msg and "xhigh" in msg and "UD-IQ3_S" in msg
    with pytest.raises(ValueError, match="plain has no vision tower"):
        catalog.plan_for(PLAIN, box(32), mmproj=True)
    with pytest.raises(ValueError, match="comes without faster replies; drop --drafter"):
        catalog.plan_for(PLAIN, box(32), drafter=True)
    with pytest.raises(ValueError, match="comes without faster replies; drop --no-drafter"):
        catalog.plan_for(PLAIN, box(32), drafter=False)
    # A named quant still pulls on a box with no default.
    plan = catalog.plan_for(MOE, box(16), quant="fastest")
    assert plan.rung.tier == "fastest" and plan.streaming and not plan.drafter


def test_plan_refs_pairs_bytes_and_local_state(tmp_path):
    b = box(32)
    plan = catalog.plan_for(DENSE, b)
    refs = catalog.plan_refs(plan)
    assert refs == ["hf:org/dense/Model-UD-Q4_K_M.gguf@abc1234",
                    "hf:org/dense/mmproj-F16.gguf@abc1234",
                    "hf:org/drafter/drafter-Q8_0.gguf@def5678"]
    for ref in refs:
        parsed = remote.parse_ref(ref)
        assert parsed.kind == "hf" and not parsed.is_dir
    assert remote.parse_ref(refs[0]).revision == "abc1234"
    assert catalog.plan_bytes(plan) == G(15.33) + G(0.86) + G(1.92)
    pairs = catalog.plan_pairs(plan, str(tmp_path), True)
    model_path = str(tmp_path / "org__dense" / "Model-UD-Q4_K_M.gguf")
    assert pairs == {model_path: {
        "alias": "dense", "mmproj": str(tmp_path / "org__dense" / "mmproj-F16.gguf"),
        "draft_gguf": str(tmp_path / "org__drafter" / "drafter-Q8_0.gguf")}}
    flat = catalog.plan_pairs(catalog.plan_for(DENSE, b, mmproj=False, drafter=False), str(tmp_path), False)
    assert flat == {str(tmp_path / "Model-UD-Q4_K_M.gguf"): {"alias": "dense"}}
    # A file in a folder of its repo keeps that folder, as the download does.
    assert catalog.dest_path("org/repo", "MTP/mtp-Q8_0.gguf", "/lib", True) == \
        "/lib/org__repo/MTP/mtp-Q8_0.gguf"
    assert catalog.dest_path("org/repo", "UD-IQ2_XXS/m-00001-of-00002.gguf", "/lib", False) == \
        "/lib/UD-IQ2_XXS/m-00001-of-00002.gguf"
    # Nothing on disk yet.
    on, remaining, per_ref = catalog.local_state(plan, str(tmp_path), True)
    assert (on, remaining) == (0, catalog.plan_bytes(plan))
    assert all(v == (0, n, False) for v, n in zip(per_ref.values(), (G(15.33), G(0.86), G(1.92))))
    # A full tower and a partial model.
    (tmp_path / "org__dense").mkdir()
    (tmp_path / "org__dense" / "mmproj-F16.gguf").write_bytes(b"x" * 1000)
    (tmp_path / "org__dense" / "Model-UD-Q4_K_M.gguf.part").write_bytes(b"y" * 500)
    on, remaining, per_ref = catalog.local_state(plan, str(tmp_path), True)
    assert on == 1500 and remaining == catalog.plan_bytes(plan) - 1500
    assert per_ref[refs[1]] == (1000, G(0.86), True)
    assert per_ref[refs[0]] == (500, G(15.33), False)


# -- text ----------------------------------------------------------------------------

def test_recommendation_and_entry_text_are_plain(small_catalog):
    b = box(32)
    groups = catalog.groups_for(b)
    text = catalog.recommendation_text(groups, b)
    assert text.isascii() and not _banned(text)
    lines = text.splitlines()
    assert lines[0] == "This Mac has 32 GB of memory."
    titles = catalog.group_titles(groups)
    assert titles[1] == "Also runs in memory"
    at = [lines.index(t) for t in titles]
    assert at == sorted(at)
    # One table per group: a header row, then one row per model, each model once.
    header = ["NAME", "DOWNLOAD", "IMAGES", "GOOD", "FOR"]
    assert lines[at[0] + 1].split() == header and lines[at[1] + 1].split() == header
    assert lines[at[3] + 1].split() == ["NAME", "NEEDS", "IMAGES", "GOOD", "FOR"]
    rows = [ln for ln in lines if ln.startswith("  ") and not ln.startswith("  NAME")]
    assert sorted(ln.split()[0] for ln in rows) == sorted(small_catalog)
    images_at = None                       # each table's IMAGES column, from its header
    for ln in lines:
        if ln.startswith("  NAME"):
            images_at = ln.index("IMAGES")
        elif ln in rows:
            m = small_catalog[ln.split()[0]]
            assert ln.rstrip().endswith(m.use), ln
            assert ln.ljust(80)[images_at:images_at + 3] == ("yes" if m.mmproj else "   "), ln
    assert any(ln.split()[1:4] == ["48", "GB", "Mac"] for ln in rows)
    # Within a group the largest download comes first.
    fits = [ln for ln in lines[at[1] + 2:at[2]] if ln.strip()]
    sizes = [float(ln.split()[1]) for ln in fits]
    assert sizes == sorted(sizes, reverse=True)
    assert "Room for about" not in text and text.rstrip().endswith(catalog.LIST_FOOTER)
    assert lines[-1] == catalog.OTHER_MODELS_LINE and len(lines[-1]) <= 80
    assert "extras:" not in text and "UD-Q4_K_M" not in text
    # --verbose keeps every line and adds the rungs and extras under each model.
    vtext = catalog.recommendation_text(groups, b, verbose=True)
    assert all(ln in vtext.splitlines() for ln in lines) and not _banned(vtext)
    vlines = vtext.splitlines()
    dense_at = next(i for i, ln in enumerate(vlines) if ln.startswith("  dense "))
    under = vlines[dense_at + 1:dense_at + 4]
    assert all(ln.startswith("      ") for ln in under)
    assert any(ln.strip().startswith("balanced  UD-Q4_K_M") and ln.rstrip().endswith("default")
               for ln in under)
    assert any(ln.strip().startswith("extras: image input") and "faster replies" in ln
               for ln in vlines[dense_at:dense_at + 8])
    assert len(vlines) > len(lines)
    assert "huggingface.co link" in lines[-1] and "hf:<org>/<repo>/<file.gguf>" in lines[-1]
    for m in small_catalog.values():
        entry = catalog.entry_text(m, b)
        assert entry.isascii() and not _banned(entry), entry
    entry = catalog.entry_text(DENSE, b)
    assert "needs a 48 GB Mac" in entry and "room for ~" in entry and "default" in entry
    assert "image input" in entry and "faster replies" in entry and "gmlx pull dense" in entry
    # An entry nothing of which runs here still lists its tiers.
    entry = catalog.entry_text(BIG, box(8))
    assert "Needs a 48 GB Mac." in entry and "balanced" in entry


# -- the seeded catalog ---------------------------------------------------------------

def _seed_models():
    return list(catalog.MODELS.values())


def test_every_seed_has_its_facts():
    for m in _seed_models():
        assert m.rungs, f"{m.name}: no rungs recorded; run scripts/refresh_pull_catalog.py"
        assert m.revision != "main" and len(m.revision) >= 7, m.name
        assert m.license, m.name
        assert m.arch, m.name


@pytest.mark.parametrize("m", _seed_models(), ids=lambda m: m.name)
def test_seed_rungs_are_well_formed(m):
    sizes = [r.bytes for r in m.rungs]
    assert sizes == sorted(sizes) and all(sizes)
    tier_names = [r.tier for r in m.rungs if r.tier]
    assert len(tier_names) == len(set(tier_names)) and set(tier_names) <= set(TIERS_SET)
    tiers_ = catalog.tiers(m)
    assert [r.bytes for r in tiers_] == sorted(r.bytes for r in tiers_)
    pref = catalog.preferred_rung(m)
    assert pref is not None, (m.name, m.preferred)
    if catalog.has_tiers(m):
        assert m.preferred in TIERS_SET
    else:
        assert m.preferred == pref.quant
    for r in m.rungs:
        base = strip_shard_suffix(os.path.basename(r.path))
        assert base.endswith(".gguf") and f"-{r.quant}.gguf" in base, (m.name, r.path)
        if r.tier:
            assert quant_tag(base) == catalog._norm_quant(r.quant), (m.name, r.quant)
        ref = remote.parse_ref(catalog.rung_ref(m, r))
        assert ref.kind == "hf" and not ref.is_dir and ref.revision == m.revision
        assert 0 < r.text_bytes <= r.bytes, (m.name, r.quant)
        assert r.table_bytes < r.bytes
        if not m.moe:
            assert r.streams_from_gib is None and r.table_bytes == 0
        if catalog.has_head(m, r):
            assert 0 < r.text_bytes < r.bytes, (m.name, r.quant)
        if not r.tier and catalog.has_tiers(m):
            assert re.match(r"^(UD-)?(IQ|Q)[123]_", r.quant), (m.name, r.quant)
    if m.drafter is not None:
        d = m.drafter
        assert d.repo and d.revision != "main" and d.file.endswith(".gguf") and d.bytes > 0
        assert d.kind == "dflash2" or arch_table.drafter_serves(d.kind, m.arch) is not False, d
    if m.mmproj:
        assert m.mmproj.endswith(".gguf") and m.mmproj_bytes > 0
    for f in m.ignored_files:
        assert f.endswith(".gguf")
        assert f not in {r.path for r in m.rungs} | {m.mmproj} | {m.drafter.file if m.drafter else None}
    if m.kv_cost is not None:
        assert m.heads > 0 and m.trained_ctx > 0
        assert catalog.kv_costs(m.kv_cost)
    if m.license == "other":
        assert m.license_note, m.name


TIERS_SET = set(catalog.TIERS)


@pytest.mark.parametrize("m", _seed_models(), ids=lambda m: m.name)
def test_seed_words_fit_the_detail_pane(m):
    for s in (m.blurb, m.speed):
        assert s.endswith(".") and ". " not in s, s
        assert s.isascii()
    assert len(f"{m.blurb} {catalog.license_text(m)}") <= 76, m.name
    assert len(m.speed) <= 76, m.name
    assert not _banned(m.blurb + "\n" + m.speed), m.name
    assert m.title and m.params, m.name
    # The params column reads as a count, never as the header's expert label.
    assert re.fullmatch(r"(E)?\d+(\.\d+)?B(-A\d+(\.\d+)?B)?", m.params), m.params


def test_params_label_takes_the_seed_then_a_plain_header_then_the_count():
    seed = catalog.Seed("x", "X", "org/x", "high", "B.", "Q.")
    label = catalog._params_label
    assert label(seed, {"params": "27B", "n_params": 27_300_000_000}) == "27B"
    assert label(seed, {"params": "26B-A4B", "n_params": 1}) == "26B-A4B"
    assert label(seed, {"params": "288x10B", "n_params": 321_400_000_000}) == "321B"
    assert label(seed, {"params": "288x10B"}) == "288x10B"
    assert label(seed, {"params": "", "n_params": 0}) == ""
    assert label(dataclasses.replace(seed, params="E4B"), {"params": "7.5B"}) == "E4B"


def test_seed_names_recommended_and_image_recipes():
    from gmlx.image.catalog import RECIPES
    names = set(catalog.MODELS)
    aliases = {a for m in catalog.MODELS.values() for a in m.aliases}
    assert not names & aliases
    taken = set(RECIPES) | {a for r in RECIPES.values() for a in r.aliases}
    assert not (names | aliases) & taken
    for cls, picks in catalog.RECOMMENDED.items():
        assert cls in catalog.MEMORY_CLASSES
        assert len(picks) == len(set(picks))
        for name in picks:
            assert name in names, name
    assert catalog.closest_name("qwen3.8-27") == "qwen3.8-27b"
    assert catalog.closest_name("zzz") is None
    assert catalog.model("bolt-e4b").name == "nemotron-3.5-bolt-e4b"
    assert catalog.model("QWEN3.8-27B").name == "qwen3.8-27b"
    assert len(catalog.MODELS) == len(catalog._SEED)


def test_seed_figures_on_real_boxes():
    # The smallest Mac has a recommended entry in memory with a context figure.
    b8 = box(8)
    rec = catalog.recommended_for(b8)
    assert rec and rec[0].name == catalog.RECOMMENDED[8][0]
    dp = catalog.default_plan(rec[0], b8)
    assert dp and not dp.streaming
    assert catalog.context_estimate(rec[0], dp.rung, b8, catalog.plan_addon_bytes(dp),
                                    catalog.plan_speculates(dp)) >= K
    # Flash-Next on a stock 128 GiB Mac: the balanced tier is in memory
    # through its table, and registers with no stream key.
    fn = catalog.model("qwen3.8-flash-next")
    r = catalog.find_rung(fn, "balanced")
    assert r.table_bytes > 0 and r.text_bytes > box(128).budget
    assert catalog.rung_state(fn, r, box(128)) == catalog.RESIDENT
    assert catalog.rung_state(fn, r, box(64)) == catalog.STREAMS
    # GLM-5.3-Flash on a 128 GiB Mac on macOS 26: a compact rung in memory.
    glm = catalog.model("glm-5.3-flash")
    ws = int(96.8 * GiB / 0.9)
    from gmlx.gen.prefill_decay import cap_bytes_for
    from gmlx.stream.plan import ceiling_for
    mac = catalog.Box(128 * GiB, ws, int(96.8 * GiB), int(ceiling_for(128 * GiB, ws)), int(cap_bytes_for(ws)))
    dp = catalog.default_plan(glm, mac)
    assert dp and not dp.streaming and dp.rung.quant == "UD-IQ2_XXS" and dp.mmproj
    assert catalog.context_text(glm, dp.rung, mac, catalog.plan_addon_bytes(dp),
                                catalog.plan_speculates(dp)).startswith("Room for about")
    # Every recommended name runs in memory somewhere at or under its class.
    for cls, picks in catalog.RECOMMENDED.items():
        for name in picks:
            m = catalog.model(name)
            assert any(catalog.rung_state(m, r, box(cls)) == catalog.RESIDENT for r in m.rungs), (cls, name)
