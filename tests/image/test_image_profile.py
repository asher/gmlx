"""StepProfiler on a tiny Z-Image model: every leaf module and seam call is
booked once, the records add up to the step's wall time, seams carry their
caller's path, and removal restores the classes and the seams."""
from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn

from image_fixtures import tiny_z_model

from gmlx.image import ops
from gmlx.image import profile as prof


def _step(m):
    rng = np.random.default_rng(0)
    lat = mx.array(rng.standard_normal((1, 4, 12, 10)).astype(np.float32))
    cap = mx.array(rng.standard_normal((1, 5, 16)).astype(np.float32))
    return lambda: m(lat, 0.6, cap)


def test_records_cover_the_step_and_every_seam():
    m, _ = tiny_z_model()
    fn = _step(m)
    want = np.array(fn())
    with prof.StepProfiler(m) as p:
        records, wall = p.step(fn)
        got = p.records
    assert got is records
    booked = sum(r.seconds for r in records)
    assert booked == pytest.approx(wall, rel=1e-6, abs=1e-6)
    seams = {r.name for r in records if r.kind == "seam"}
    assert seams == set(ops.SEAMS) | {"layer_norm"}
    paths = {(r.name, r.path) for r in records if r.kind == "seam"}
    assert ("sdpa", "layers.1.attention") in paths
    assert ("swiglu", "context_refiner.0.feed_forward") in paths
    assert ("gated_residual", "noise_refiner.0") in paths
    assert ("layer_norm", "final_layer") in paths
    mods = [r for r in records if r.kind == "module"]
    qkv = [r for r in mods if r.path == "layers.0.attention.qkv"]
    assert len(qkv) == 1 and qkv[0].name == "Linear"
    assert qkv[0].codec == "float32" and qkv[0].out_shape[-1] == 3 * 64
    # One glue record before each op, plus the tail.
    n_ops = sum(1 for r in records if r.kind != "glue")
    assert sum(1 for r in records if r.kind == "glue") == n_ops + 1
    # The instrumented step computes the same thing.
    assert np.allclose(np.array(fn()), want)


def test_remove_restores_classes_and_seams():
    m, _ = tiny_z_model()
    before = {p: type(x) for p, x in m.named_modules()}
    seams = {n: getattr(ops, n) for n in ops.SEAMS}
    with prof.StepProfiler(m):
        assert type(m.layers[0].attention.qkv).__name__ == "TimedLinear"
        assert ops.sdpa is not seams["sdpa"]
    assert {p: type(x) for p, x in m.named_modules()} == before
    assert {n: getattr(ops, n) for n in ops.SEAMS} == seams
    assert "_prof_owner" not in m.layers[0].attention.qkv.__dict__
    assert type(m.layers[0].attention.qkv) is nn.Linear


def test_summary_ratio_and_groups():
    m, _ = tiny_z_model()
    fn = _step(m)
    plain = [prof.plain_step(fn) for _ in range(3)]
    with prof.StepProfiler(m) as p:
        records, wall = p.step(fn)
    s = prof.summarize(records, [wall], plain)
    assert s["step"]["booked_s"] == pytest.approx(wall, rel=1e-6, abs=1e-6)
    assert s["step"]["instrumented_over_plain"] > 0
    assert s["step"]["corrected_ops_over_plain"] == pytest.approx(
        s["step"]["ops_s"] / s["step"]["plain_median_s"])
    shares = sum(g["share"] for g in s["by_class"])
    assert shares == pytest.approx(1.0)
    keys = {g["key"] for g in s["by_pattern"]}
    assert "module:layers.N.attention.qkv:Linear" in keys
    assert "seam:layers.N.attention:sdpa" in keys


def _rec(kind, name, seconds):
    return prof.OpRecord(kind, name, name, "", [], [], [], "", seconds)


def test_summary_takes_one_round_trip_off_each_op_record():
    records = [_rec("glue", "a", 0.02), _rec("module", "a", 0.5),
               _rec("glue", "b", 0.01), _rec("seam", "b", 0.2),
               _rec("glue", "c", 0.0), _rec("module", "c", 0.05),
               _rec("glue", "tail", 0.02)]
    s = prof.summarize(records, [0.8], [0.6, 0.62, 0.61], sync_s=0.1)
    st = s["step"]
    assert st["ops_s"] == pytest.approx(0.75)
    assert st["ops_corrected_s"] == pytest.approx(0.4 + 0.1 + 0.0)
    assert st["corrected_ops_over_plain"] == pytest.approx(0.5 / 0.61)
    assert st["instrumented_over_plain"] == pytest.approx(0.8 / 0.61)
    by = {g["key"]: g for g in s["by_class"]}
    assert by["module:a:"]["corrected_seconds"] == pytest.approx(0.4)
    assert by["module:c:"]["corrected_seconds"] == 0.0
    assert by["glue"]["corrected_seconds"] == pytest.approx(0.05)
    assert sum(g["corrected_share"] for g in s["by_class"]) ==         pytest.approx(1.0)
    assert by["module:a:"]["corrected_share"] == pytest.approx(0.4 / 0.55)
    assert by["module:a:"]["share"] == pytest.approx(0.5 / 0.8)


def test_sync_round_trip_reports_a_small_positive_cost():
    r = prof.sync_round_trip(iters=20, warmup=2)
    assert 0 < r["p10_s"] <= r["median_s"] <= r["p90_s"] < 0.1
    assert r["iters"] == 20


def test_pattern_folds_block_indices():
    assert prof.pattern("layers.12.attention.qkv") == "layers.N.attention.qkv"
    assert prof.pattern("t_embedder.mlp.0") == "t_embedder.mlp.N"
    assert prof._parent("final_layer.adaLN_modulation.1") == "final_layer"
    assert prof._parent("layers.3.attention_norm1") == "layers.3"


def test_sdpa_peak_reports_the_score_size():
    r = prof.sdpa_peak(64, 2, 32, dtype=mx.float32)
    assert r["score_bytes"] == 2 * 64 * 64 * 4
    assert r["output_bytes"] == 2 * 64 * 32 * 4
    assert r["peak_over_inputs_bytes"] >= 0
