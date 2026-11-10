#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""A declared model type prices the fused prefill transient with no cache.

Five places price the prefill score transient. Three read the header (the
boot table, the run and chat fit check, the pull picker) and two run with
a model (the request preflight and admission). A model type that declares
fused attention gets one price from all five, so a context the boot table
gives is not refused by the request it was promised to. A model type with
no declaration keeps the dense price everywhere.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

import mlx.core as mx

import gmlx.gen.prefill_decay as pd
from gmlx.gen.prefill_decay import (
    TransientDeclaration,
    declared_transient_bytes,
    fused_mask_bytes,
    transient_declaration,
)
from tests.lfm3_header import write_lfm3_header

GiB = 1 << 30
DECL = TransientDeclaration(128)
HEADS = 16
CAP = 2e9

gpu = pytest.mark.skipif(
    os.environ.get("KQUANT_FORCE_CPU") == "1" or not mx.metal.is_available(),
    reason="the fused profile arms on the GPU only")


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    for name in ("GMLX_PREFILL_SCORE_PROFILE", "GMLX_PREFILL_SCORE_CAP_GB",
                 "GMLX_PREFILL_MIN_STEP", "PREFILL_STEP_SIZE",
                 "GMLX_ADMIT_RESERVE_GB", "GMLX_GOV_RESERVE_GB",
                 "GMLX_GOV_MARGIN", "GMLX_OVERCOMMIT", "KV_BITS",
                 "KV_QUANT_SCHEME", "MAX_KV_SIZE"):
        monkeypatch.delenv(name, raising=False)


def _dense(ctx: int, step: int = 2048) -> float:
    return HEADS * step * (ctx + step) * 2.0


def _stub(cfg: dict):
    """A model as the runtime readers see it: a config and nothing else."""
    return SimpleNamespace(config=SimpleNamespace(**cfg))


LFM3_CFG = dict(model_type="lfm3", num_attention_heads=HEADS, head_dim=128)


# -- the function ----------------------------------------------------------

def test_a_batch_chunk_holds_its_mask():
    for rows in (1, 2, 32):
        got = declared_transient_bytes(DECL, HEADS, True, 16384, rows, 1e12,
                                       2048)
        assert got == (rows + 1) * 2048 * (16384 + 2048)
    assert fused_mask_bytes(0) == 0 and fused_mask_bytes(3) == 4


def test_a_plain_cache_holds_only_the_cut_chunk():
    """One prompt on a plain cache masks with no array. What is left is a
    chunk cut under the fused width, which runs unfused."""
    got = declared_transient_bytes(DECL, HEADS, True, 32768, 0, CAP, 2048)
    assert got == HEADS * 8 * (32768 + 8) * 2.0
    assert got < _dense(32768) / 100


def test_the_price_follows_the_chunk_step():
    """A one-row batch at 32k with a step of 8,192 holds a 671 MB mask."""
    got = declared_transient_bytes(DECL, HEADS, True, 32768, 1, 1e12, 8192)
    assert got == 2 * 8192 * (32768 + 8192)
    assert got == pytest.approx(671e6, rel=0.01)
    assert got > 4 * declared_transient_bytes(DECL, HEADS, True, 32768, 1,
                                              1e12, 2048)


def test_the_cap_halves_the_step_of_a_wide_batch():
    """33 mask bytes per element pass a 2 GB cap near 27.5k tokens, and
    the decay then halves the step."""
    under = declared_transient_bytes(DECL, HEADS, True, 27000, 32, CAP, 2048)
    over = declared_transient_bytes(DECL, HEADS, True, 28000, 32, CAP, 2048)
    assert under == 33 * 2048 * (27000 + 2048) <= CAP
    assert over == 33 * 1024 * (28000 + 1024)


@pytest.mark.parametrize("rows", [4, 32])
def test_the_held_price_never_falls(rows):
    """The runtime price drops by half where the cap halves the step. The
    held price stays at the cap from there, so a search over the context
    is sound. It is never under the runtime price, and it is the runtime
    price until the mask first passes the cap."""
    def price(ctx, hold):
        return declared_transient_bytes(DECL, HEADS, True, ctx, rows, CAP,
                                        2048, hold_at_cap=hold)

    edge = int(CAP // ((rows + 1) * 2048)) - 2048
    assert price(edge + 1, False) < price(edge, False) <= CAP
    last = 0.0
    for ctx in range(1024, 1 << 21, 997):
        held, runtime = price(ctx, True), price(ctx, False)
        assert held >= last and held >= runtime
        assert held == (runtime if ctx <= edge else max(runtime, CAP))
        last = held
    assert last > CAP


@pytest.mark.parametrize("kw", [
    dict(decl=None),
    dict(kv_fp16=False),
    dict(heads=0),
    dict(decl=TransientDeclaration(512)),
])
def test_no_price_where_the_declaration_does_not_apply(kw):
    args = dict(decl=DECL, heads=HEADS, kv_fp16=True, ctx=16384, rows=1,
                cap=CAP, base=2048)
    args.update(kw)
    assert declared_transient_bytes(**args) is None


def test_no_price_with_the_profile_switch_off(monkeypatch):
    monkeypatch.setenv("GMLX_PREFILL_SCORE_PROFILE", "0")
    assert declared_transient_bytes(DECL, HEADS, True, 16384, 1, CAP,
                                    2048) is None


# -- the declaration -------------------------------------------------------

def test_lfm3_declares_from_a_config_alone():
    """A header reader has no model. The first lookup imports the model
    package, which registers the declaration."""
    assert transient_declaration(dict(LFM3_CFG)) == DECL
    assert transient_declaration(_stub(LFM3_CFG)) == DECL
    assert transient_declaration({"text_config": dict(LFM3_CFG)}) == DECL


def test_other_model_types_declare_nothing():
    for mt in ("llama", "flint", "beam", "qwen3_5", "faketype"):
        assert transient_declaration({"model_type": mt, "head_dim": 128}) \
            is None
    assert transient_declaration({"head_dim": 128}) is None
    assert sorted(pd._TRANSIENT_DECLARATIONS) == ["lfm3"]


def test_a_built_model_declares_through_its_args():
    from gmlx.models.lfm3.model import CONV, FULL, Model, ModelArgs

    args = ModelArgs(
        model_type="lfm3", hidden_size=64, num_hidden_layers=2,
        intermediate_size=96, num_attention_heads=4, num_key_value_heads=2,
        head_dim=128, vocab_size=32, layer_types=[CONV, FULL])
    assert transient_declaration(Model(args)) == DECL


# -- the runtime KV scheme -------------------------------------------------

def _policy(single: str, batched: str):
    return SimpleNamespace(single=SimpleNamespace(verdict=single),
                           batched=SimpleNamespace(verdict=batched))


@pytest.mark.parametrize("single, batched, fp16", [
    ("off", "off", True),
    ("dropped", "dropped", True),
    ("full", "full", False),
    ("partial", "partial", False),
    # Batching drops the scheme: a lone row still runs quantized.
    ("full", "dropped", False),
    ("dropped", "partial", False),
])
def test_kv_is_fp16_only_when_neither_stack_takes_a_scheme(
        single, batched, fp16):
    model = _stub(LFM3_CFG)
    model._gmlx_kv_policy = _policy(single, batched)
    assert pd.runtime_kv_fp16(model) is fp16


def test_a_model_with_no_policy_runs_fp16():
    assert pd.runtime_kv_fp16(_stub(LFM3_CFG)) is True


# -- the runtime no-cache branch -------------------------------------------

def test_no_cache_price_of_a_declared_model_is_the_fused_price(monkeypatch):
    monkeypatch.setattr(pd, "cpu_default_device", lambda: False)
    monkeypatch.setattr(pd, "_body_cap_bytes", lambda: CAP)
    model = _stub(LFM3_CFG)
    assert pd.score_transient_bytes(model, None, 30720) == \
        2 * 2048 * (30720 + 2048)
    assert pd.score_transient_bytes(model, None, 30720, prefill_rows=2) == \
        3 * 2048 * (30720 + 2048)
    # The request's own step, not the default.
    assert pd.score_transient_bytes(
        model, None, 24576, prefill_rows=1, base_step=8192) == \
        2 * 8192 * (24576 + 8192)


def test_no_cache_price_of_a_quantized_declared_model_is_dense(monkeypatch):
    monkeypatch.setattr(pd, "cpu_default_device", lambda: False)
    monkeypatch.setattr(pd, "_body_cap_bytes", lambda: 1e12)
    model = _stub(LFM3_CFG)
    model._gmlx_kv_policy = _policy("full", "dropped")
    assert pd.score_transient_bytes(model, None, 30720) == _dense(30720)


def test_no_cache_price_with_the_switch_off_is_dense(monkeypatch):
    """An idle engine and a busy one agree: neither arms a profile."""
    monkeypatch.setattr(pd, "cpu_default_device", lambda: False)
    monkeypatch.setattr(pd, "_body_cap_bytes", lambda: 1e12)
    monkeypatch.setenv("GMLX_PREFILL_SCORE_PROFILE", "0")
    assert pd.score_transient_bytes(_stub(LFM3_CFG), None, 30720) == \
        _dense(30720)


def test_no_cache_price_on_the_cpu_device_is_dense(monkeypatch):
    """The profile arms on the GPU only, so the CPU device keeps
    max(dense, fused) for every model type."""
    monkeypatch.setattr(pd, "cpu_default_device", lambda: True)
    monkeypatch.setattr(pd, "_body_cap_bytes", lambda: 1e12)
    assert pd.score_transient_bytes(_stub(LFM3_CFG), None, 30720) == \
        _dense(30720)


def test_no_cache_price_of_an_undeclared_model_does_not_move(monkeypatch):
    """A model type with a fused profile and no declaration keeps
    max(dense, fused). One with neither keeps dense."""
    monkeypatch.setattr(pd, "cpu_default_device", lambda: False)
    monkeypatch.setattr(pd, "_body_cap_bytes", lambda: 1e12)
    fused = pd.ScoreTransientProfile(heads=HEADS, bytes_per_elem=2,
                                     fused_min_step=9)
    monkeypatch.setitem(pd._SCORE_PROFILES, "fused_undeclared",
                        lambda model, cache: fused)
    for mt in ("fused_undeclared", "faketype"):
        model = _stub(dict(LFM3_CFG, model_type=mt))
        for rows in (None, 2):
            assert pd.score_transient_bytes(
                model, None, 30720, prefill_rows=rows) == _dense(30720)


# -- each reader passes its own inputs -------------------------------------

@pytest.fixture()
def calls(monkeypatch):
    """Record every call of the declared price and answer 1 byte."""
    seen = []

    def spy(decl, heads, kv_fp16, ctx, rows, cap, base, *,
            hold_at_cap=False):
        seen.append(SimpleNamespace(decl=decl, heads=heads, kv_fp16=kv_fp16,
                                    ctx=ctx, rows=rows, cap=cap, base=base,
                                    hold=hold_at_cap))
        return 1.0

    monkeypatch.setattr(pd, "declared_transient_bytes", spy)
    return seen


@pytest.fixture()
def shipped(tmp_path) -> str:
    return write_lfm3_header(tmp_path / "m.gguf", shipped=True,
                             context=1 << 22)


def _class_box(monkeypatch, cls: int) -> float:
    """Price as a stock Mac of ``cls`` GiB does. Returns its working set."""
    import gmlx.device as device
    import gmlx.serve.capacity as cap
    from gmlx.stream.plan import ceiling_for

    ram = cls * GiB
    ws = cap.working_set_for_ram(ram)
    monkeypatch.setattr(cap, "working_set_bytes", lambda: ws)
    monkeypatch.setattr(cap, "ceiling_bytes",
                        lambda w, m=None: ceiling_for(ram, w))
    monkeypatch.setattr(device, "metal_info", lambda: {})
    monkeypatch.setattr(pd, "_body_cap_bytes", lambda: pd.cap_bytes_for(ws))
    return ws


def test_boot_passes_the_width_it_prices(monkeypatch, shipped, calls):
    import gmlx.serve.capacity as cap

    _class_box(monkeypatch, 16)
    assert cap.derive_table(shipped, weight_bytes=2e9) is not None
    assert {c.rows for c in calls} == set(cap._WIDTHS)
    assert all(c.decl == DECL and c.kv_fp16 and c.heads == HEADS
               and c.cap is None and c.base == 2048 and c.hold
               for c in calls)


def test_no_context_under_a_wide_boot_answer_is_refused(monkeypatch, shipped):
    """Width 32 on a 32 GiB Mac holds enough KV for the mask to pass the
    cap. The weights put the budget line between the price at the cap and
    the halved price just past it, where an unheld search can answer above
    contexts that do not fit."""
    import gmlx.serve.capacity as cap
    import gmlx.serve.mem_preflight as mp
    import gmlx.serve.memory as sm

    ws = _class_box(monkeypatch, 32)
    two_gb = pd.cap_bytes_for(ws)
    edge = int(two_gb // (33 * 2048)) - 2048
    _cfg, costs, _heads, _fp16 = cap.boot_costs(shipped)
    room = cap.ceiling_bytes(ws) - sm.admit_reserve_bytes(ws)
    weights = room - 32 * mp.prompt_kv_bytes(costs, edge) - 0.75 * two_gb
    assert weights > 0

    def fits(ctx: int) -> bool:
        tr = declared_transient_bytes(DECL, HEADS, True, ctx, 32, None, 2048)
        return weights + 32 * mp.prompt_kv_bytes(costs, ctx) + tr <= room

    # The trap: a band under the edge does not fit, one past it does.
    assert not fits(edge) and fits(edge + 64)
    top = cap.derive_table(shipped, weight_bytes=weights)["max_ctx"][32]
    assert 0 < top < edge
    assert all(fits(ctx) for ctx in range(top % 16, top + 1, 16))
    assert fits(top)
    cap.clear_table()


def test_boot_prices_a_quantized_window_dense(monkeypatch, shipped):
    import gmlx.serve.capacity as cap

    _class_box(monkeypatch, 16)
    plain = cap.derive_table(shipped, weight_bytes=2e9)
    packed = cap.derive_table(shipped, weight_bytes=2e9,
                              env={"KV_BITS": "8"})
    assert cap.boot_costs(shipped, env={"KV_BITS": "8"})[3] is False
    # Packed KV is smaller, yet the dense transient costs more context.
    assert packed["max_ctx"][1] < plain["max_ctx"][1]


def test_run_and_chat_price_one_prompt_on_a_plain_cache(
        monkeypatch, shipped, calls):
    import gmlx.commands.tool_preflight as tp

    monkeypatch.setattr(tp, "working_set_bytes", lambda: 16e9)
    assert tp.estimate(shipped, 32768, kv_fp16=True) is not None
    assert calls and all(c.rows == 0 and c.kv_fp16 and c.decl == DECL
                         and c.base == tp._SCORE_CHUNK and not c.hold
                         for c in calls)
    for kw in ({"kv_fp16": False}, {}):
        calls.clear()
        tp.estimate(shipped, 32768, **kw)
        assert calls and not any(c.kv_fp16 for c in calls)


@pytest.mark.parametrize("flags, fp16", [
    ({}, True),
    ({"kv_quant_scheme": "uniform"}, True),
    ({"kv_quant_scheme": "auto"}, True),
    ({"kv_bits": 8}, False),
    ({"kv_bits": 8, "kv_quant_scheme": "uniform"}, False),
    ({"kv_quant_scheme": "kvarn"}, False),
    ({"kv_bits": 4, "kv_quant_scheme": "kvarn"}, False),
])
def test_the_kv_flags_say_whether_the_cache_is_plain(flags, fp16):
    """kvarn packs at its own default width with no --kv-bits."""
    from gmlx.commands.tool_preflight import args_kv_fp16

    assert args_kv_fp16(SimpleNamespace(**flags)) is fp16


@pytest.mark.parametrize("argv", [
    ["run", "--prompt", "hi"],
    ["run", "--bench-depths", "1024"],
    ["chat"],
])
@pytest.mark.parametrize("flags, fp16", [
    ([], True),
    (["--kv-bits", "8"], False),
    (["--kv-quant-scheme", "kvarn"], False),
])
def test_the_fit_check_of_each_tool_reads_the_kv_flags(
        monkeypatch, shipped, argv, flags, fp16):
    import gmlx.commands.tool_preflight as tp
    from gmlx.commands.cli import umbrella_main

    seen = []

    def spy(path, ctx_tokens=None, streaming=False, **kw):
        seen.append(kw)
        raise SystemExit(7)

    monkeypatch.setattr(tp, "check_or_exit", spy)
    try:
        code = umbrella_main([argv[0], shipped, *argv[1:], *flags])
    except SystemExit as exc:
        code = exc.code
    assert code == 7
    assert seen == [{"kv_fp16": fp16}]


def test_kv_bits_makes_the_fit_check_dense(monkeypatch, shipped):
    import gmlx.commands.tool_preflight as tp

    monkeypatch.setattr(tp, "working_set_bytes", lambda: 16e9)
    fused = tp.estimate(shipped, 32768, kv_fp16=True)
    dense = tp.estimate(shipped, 32768)
    assert dense["need"] - fused["need"] == pytest.approx(
        _dense(32768) - HEADS * 8 * (32768 + 8) * 2.0)


def _catalog_model(head_dim: int):
    from gmlx.load import catalog

    kv = catalog.KvCost((("full", 0, 2048.0, 6), ("window", 256, 2048.0, 5)))
    rung = catalog.Rung("Q4_K_M", int(1.1e9), "m.gguf", "balanced")
    return catalog.Model(
        name="m", title="M", repo="org/m", rungs=(rung,),
        preferred="balanced", params="2B", blurb="", speed="", license="mit",
        kv_cost=kv, heads=HEADS, trained_ctx=1 << 20,
        prefill_fused_head_dim=head_dim), rung


def test_the_picker_prices_one_served_row_with_the_box_cap(calls):
    from gmlx.load import catalog

    model, rung = _catalog_model(128)
    box = catalog.box_for_class(8)
    assert catalog.context_estimate(model, rung, box, 0, False)
    assert calls and all(
        c.rows == 1 and c.kv_fp16 and c.cap == box.cap and c.decl == DECL
        and c.heads == HEADS and c.base == 2048 for c in calls)


def test_the_picker_keeps_its_cap_on_the_dense_price_only():
    from gmlx.load import catalog

    box = catalog.box_for_class(8)
    fused_model, rung = _catalog_model(128)
    dense_model, _ = _catalog_model(0)
    fused = catalog.context_estimate(fused_model, rung, box, 0, False)
    dense = catalog.context_estimate(dense_model, rung, box, 0, False)
    assert fused > dense > 0
    # Dense, as before the fact existed: capped at box.cap.
    top = dense + catalog.K
    need = (catalog.resident_bytes(dense_model, rung, False)
            + (6 * top + 5 * 256) * 2048.0 + box.cap)
    assert need > box.ceiling


def test_the_request_preflight_prices_one_row(monkeypatch, shipped, calls):
    import gmlx.serve.capacity as cap
    import gmlx.serve.mem_preflight as mp

    monkeypatch.setattr(pd, "cpu_default_device", lambda: False)
    cfg, costs, _heads, _fp16 = cap.boot_costs(shipped)
    need = mp._need_bytes(_stub(cfg), costs, 30000)
    assert [c.rows for c in calls] == [1]
    assert calls[0].ctx == 30000 and calls[0].kv_fp16
    # Its own term: the prompt's KV.
    assert need == mp.prompt_kv_bytes(costs, 30000) + 1.0


def test_admission_prices_the_rows_that_join(monkeypatch, calls):
    """An idle engine has no cache, so admission prices the candidates'
    rows through the declaration."""
    import gmlx.serve.memory as sm

    monkeypatch.setattr(pd, "cpu_default_device", lambda: False)
    monkeypatch.setattr(pd, "headroom_bytes", lambda: 10e9)
    monkeypatch.setenv("GMLX_ADMIT_RESERVE_GB", "2")
    gen = _idle_gen(_stub(LFM3_CFG), [(None, 2048.0)])
    pending = [(uid, [0] * 3000, 200, {}, None, None) for uid in (1, 2, 3)]
    out = sm.project_admission(gen, pending)
    assert out is not None
    assert [c.rows for c in calls] == [3]
    assert calls[0].ctx == 3200 and calls[0].base == 2048


def _idle_gen(model, costs):
    """A generator with no batch yet, priced from the boot table's costs."""
    model._kq_boot_kv_costs = list(costs)
    return SimpleNamespace(
        model=model, _generation_batch=None, _prompt_batch=None,
        _unprocessed_sequences=[], prefill_step_size=2048,
        completion_batch_size=32, prefill_batch_size=8)


# -- the price agrees with the runtime on real caches (GPU) ----------------

def _tiny_lfm3():
    from gmlx.models.lfm3.model import CONV, FULL, SLIDING, Model, ModelArgs

    args = ModelArgs(
        model_type="lfm3", hidden_size=64, num_hidden_layers=5,
        intermediate_size=96, num_attention_heads=HEADS,
        num_key_value_heads=4, head_dim=128, vocab_size=32,
        layer_types=[CONV, CONV, FULL, CONV, SLIDING], sliding_window=256)
    model = Model(args)
    model.config = SimpleNamespace(model_type="lfm3",
                                   num_attention_heads=HEADS, head_dim=128)
    return model


def _caches(model, rows: int):
    from gmlx.cache.kv_cache import (ArraysCache, BatchKVCache,
                                     BatchRotatingKVCache)
    from gmlx.models.lfm3.model import CONV

    if rows == 0:
        return model.make_cache()
    out = []
    for layer in model.layers:
        if layer.layer_type == CONV:
            out.append(ArraysCache(size=1, left_padding=[0] * rows))
        elif layer.use_sliding:
            out.append(BatchRotatingKVCache(256, [0] * rows))
        else:
            out.append(BatchKVCache([0] * rows))
    return out


@gpu
@pytest.mark.parametrize("rows", [0, 1, 2])
@pytest.mark.parametrize("ctx", [4096, 16384, 32768])
@pytest.mark.parametrize("step", [2048, 8192])
def test_the_declared_price_is_the_runtime_price(rows, ctx, step):
    model = _tiny_lfm3()
    decl = transient_declaration(model)
    runtime = pd.score_transient_bytes(
        model, _caches(model, rows), ctx, prefill_rows=rows or None,
        base_step=step)
    declared = declared_transient_bytes(decl, HEADS, True, ctx, rows, None,
                                        step)
    assert declared == pytest.approx(runtime, rel=0.01)
    assert declared < _dense(ctx, step) / 4


@gpu
@pytest.mark.parametrize("rows", [0, 1, 2])
def test_the_declaration_agrees_with_the_profile(rows):
    model = _tiny_lfm3()
    profile = pd.resolve_score_profile(model, _caches(model, rows))
    decl = transient_declaration(model)
    assert profile is not None
    assert profile.fused_min_step == pd.fused_min_rows(decl.head_dim) == 9
    assert profile.heads == HEADS and profile.bytes_per_elem == 2
    assert profile.fused_mask_bytes == fused_mask_bytes(rows)


@gpu
@pytest.mark.parametrize("cls", [8, 16])
def test_boot_table_contexts_pass_preflight_and_idle_admission(
        monkeypatch, shipped, cls):
    """What the boot table gives at widths 1 and 2, the request preflight
    and an idle engine's admission both take. Weights are Omni Q4_K_M and
    its 16-bit projector."""
    import gmlx.device as device
    import gmlx.serve.capacity as cap
    import gmlx.serve.mem_preflight as mp
    import gmlx.serve.memory as sm

    ws = _class_box(monkeypatch, cls)
    weights = 1.10e9 + 0.95e9
    table = cap.derive_table(shipped, weight_bytes=weights)
    cfg, costs, _heads, _fp16 = cap.boot_costs(shipped)
    model = _stub(cfg)
    reserve = sm.admit_reserve_bytes(ws)
    # A lone request with the batch drained, as available_drained_bytes
    # gives it; an idle engine's headroom.
    drained = ws - weights - reserve
    idle = ws - weights
    monkeypatch.setattr(pd, "headroom_bytes", lambda: idle)
    monkeypatch.setattr(device, "metal_value", lambda name: ws)
    for width in (1, 2):
        ctx = table["max_ctx"][width]
        assert ctx > 32768
        assert mp._need_bytes(model, costs, ctx) <= drained
        gen = _idle_gen(model, costs)
        pending = [(uid, [0] * ctx, 0, {}, None, None)
                   for uid in range(width)]
        projected, head, _parts = sm.project_admission(gen, pending)
        assert projected <= head
    cap.clear_table()
