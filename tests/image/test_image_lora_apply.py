"""Image adapters applied live on the tiny Z-Image DiT: a step under a
patch table against the same step on merged dense weights, the cache keys,
a strength change, and the refusals. CPU."""
from __future__ import annotations

import os
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from gmlx.image import api, lora
from gmlx.image.families.z_image.lora_map import lora_map
from gmlx.image.lora import install, patch_table, reader
from gmlx.image.registry import get_family
from image_fixtures import TINY_Z_CONFIG, tiny_z_model
from lora_fixtures import (DIALECTS, make_pair, reference_delta, rel_l2,
                           tiny_z_adapter, write_adapter)

# The live path against merged fp32 weights, relative L2 of the velocity.
STEP_TOL = 2e-5
# Placements on a K-quant base at bfloat16, relative L2 between them.
KQ_TOL = 1e-2
QKV = "layers.1.attention.qkv"


def _pipe() -> api.ImagePipeline:
    m, _ = tiny_z_model()
    return api.ImagePipeline(dit=m, family=get_family("z_image"),
                             variant="turbo")


def _inputs(key: str = "k1"):
    rng = np.random.default_rng(0)
    lat = mx.array(rng.standard_normal((1, 4, 6, 8)).astype(np.float32))
    cond = api.Conditioning(
        mx.array(rng.standard_normal((5, 16)).astype(np.float32)), key)
    return lat, cond


def _merged(stack) -> api.ImagePipeline:
    """The tiny pipeline with (adapter, strength) rows merged into its
    weights in float64."""
    m, params = tiny_z_model()
    km = lora_map(TINY_Z_CONFIG)
    rows: dict = {}
    for adapter, strength in stack:
        for target, (down, up, alpha) in adapter.items():
            module, band = km.targets[target]
            rows.setdefault(module, []).append(
                (down, up, alpha, band, strength))
    upd = []
    for module, parts in rows.items():
        w = params[f"{module}.weight"]
        upd.append((f"{module}.weight", mx.array(
            (w + reference_delta(parts, w.shape)).astype(np.float32))))
    m.load_weights(upd, strict=False)
    return api.ImagePipeline(dit=m, family=get_family("z_image"),
                             variant="turbo")


def _step(pipe, table=None, sigma=0.75, key="k1") -> np.ndarray:
    lat, cond = _inputs(key)
    # The CPU stream keeps fp32 matmuls exact. On the GPU they run at TF32,
    # whose 1e-3 error is above the gap to the merged weights.
    with mx.stream(mx.cpu):
        return np.array(api.denoise_step(pipe, lat, sigma, cond,
                                         patch_table=table), np.float64)


def _sample(pipe, cond, table=None, **kw) -> np.ndarray:
    with mx.stream(mx.cpu):
        return np.array(api.sample(pipe, cond, patch_table=table, **kw),
                        np.float64)


def _table(pipe, rows) -> str:
    return api.define_patch_table(pipe, lora.table_entries(rows))


@pytest.mark.parametrize("dialect", sorted(DIALECTS))
def test_a_step_under_an_adapter_equals_the_step_on_merged_weights(
        tmp_path, dialect):
    alpha = 6.0 if DIALECTS[dialect][4] else None
    one = tiny_z_adapter(1, alpha, wide=True)
    two = tiny_z_adapter(2, alpha)
    pipe = _pipe()
    base = _step(pipe)
    a1 = lora.load_adapter(pipe, write_adapter(tmp_path / "one.safetensors",
                                               dialect, one))
    a2 = lora.load_adapter(pipe, write_adapter(tmp_path / "two.safetensors",
                                               dialect, two))
    assert len(a1.dit) == 13 and a1.text == () and a1.name == \
        "one.safetensors"
    worst = 0.0
    for stack in ([(a1, one, 0.5)], [(a1, one, 1.0)],
                  [(a1, one, 0.75), (a2, two, -0.4)]):
        tid = _table(pipe, [(a, s) for a, _, s in stack])
        got = _step(pipe, tid)
        want = _step(_merged([(ad, s) for _, ad, s in stack]))
        err = rel_l2(got, want)
        worst = max(worst, err)
        assert err <= STEP_TOL, (dialect, stack[0][2], err)
        # The adapter moves the velocity far more than the tolerance.
        assert rel_l2(base, want) > 1e-2
    assert np.array_equal(_step(pipe), base)
    print(f"tiny adapter step {dialect}: worst relative L2 {worst:.3e}")


def test_no_adapter_keeps_the_plain_leaves_and_a_table_wraps_them(tmp_path):
    pipe = _pipe()
    base = _step(pipe)
    assert pipe.lora is None
    assert api.define_patch_table(pipe, []) == api.EMPTY_TABLE
    assert pipe.lora is None and not any(
        isinstance(m, lora.ImageLoRALinear)
        for _, m in pipe.dit.named_modules())
    paths = set(install.linear_leaves(pipe.dit))
    ad = lora.load_adapter(pipe, write_adapter(
        tmp_path / "a.safetensors", "kohya", tiny_z_adapter(1, 4.0)))
    assert pipe.lora is None
    tid = _table(pipe, [(ad, 1.0)])
    found = dict(pipe.dit.named_modules())
    wrapped = {p for p, m in found.items()
               if isinstance(m, lora.ImageLoRALinear)}
    assert wrapped == paths and pipe.lora.active is None
    assert all(type(found[p].base) is nn.Linear for p in wrapped)
    assert set(install.modules(pipe.dit)) == {
        p for p in found if not p.endswith(".base")}
    assert set(install.linear_leaves(pipe.dit)) == paths
    # The base model through the wrappers is the base model, bit for bit.
    assert np.array_equal(_step(pipe), base)
    assert np.array_equal(_step(pipe, api.EMPTY_TABLE), base)
    assert not np.array_equal(_step(pipe, tid), base)
    assert pipe.lora.active is None
    # A table defined later needs no second wrap.
    state = pipe.lora
    _table(pipe, [(ad, 0.5)])
    assert pipe.lora is state


def test_an_image_forward_reads_nothing_from_the_text_row_channel(
        tmp_path, monkeypatch):
    from gmlx import lora_rows
    from gmlx.load.modules import LoRAKQuantLinear

    def boom(*a, **k):
        raise AssertionError("an image forward read gmlx.lora_rows")

    assert not issubclass(lora.ImageLoRALinear, LoRAKQuantLinear)
    assert not hasattr(install, "lora_rows")
    adapter = tiny_z_adapter(1, 4.0, wide=True)
    pipe = _pipe()
    ad = lora.load_adapter(pipe, write_adapter(
        tmp_path / "a.safetensors", "kohya", adapter))
    for name in ("row_scales", "row_factor", "dense_row_factor",
                 "dense_rows", "flat_rows", "flat_row_factor"):
        monkeypatch.setattr(lora_rows, name, boom)
    # A serve process in rows mode with no publish must not matter either.
    monkeypatch.setattr(lora_rows, "_mode", "rows")
    tid = _table(pipe, [(ad, 0.8)])
    got = _step(pipe, tid)
    assert rel_l2(got, _step(_merged([(adapter, 0.8)]))) <= STEP_TOL


def test_a_strength_change_is_a_new_table_with_no_reload(tmp_path,
                                                         monkeypatch):
    adapter = tiny_z_adapter(1, 4.0, wide=True)
    pipe = _pipe()
    path = write_adapter(tmp_path / "a.safetensors", "onetrainer", adapter)
    ad = lora.load_adapter(pipe, path)
    t1 = _table(pipe, [(ad, 1.0)])
    full = _step(pipe, t1)
    pairs = dict(pipe.adapters._items)
    wrappers = {p: m for p, m in pipe.dit.named_modules()
                if isinstance(m, lora.ImageLoRALinear)}
    weights = {p: m.base.weight for p, m in wrappers.items()}

    # From here the file is gone, no reader runs, and while the new table
    # is defined and its factors are built every leaf holds zeros of its
    # shape in place of its weight.
    os.remove(path)

    def boom(*a, **k):
        raise AssertionError("a strength change read a file")

    monkeypatch.setattr(mx, "load", boom)
    monkeypatch.setattr(reader, "read_lora", boom)
    monkeypatch.setattr(patch_table, "read_lora", boom)
    real = {p: m.base for p, m in wrappers.items()}
    for p, m in wrappers.items():
        m.base = SimpleNamespace(weight=mx.zeros(weights[p].shape))
    t2 = _table(pipe, [(ad, 0.5)])
    built = pipe.lora._table((t2, mx.float32, install.placement()))
    for module, _, _ in pipe.patch_tables[t2]["entries"]:
        assert built.get(module) is not None
    for p, m in wrappers.items():
        m.base = real[p]

    assert t2 not in (t1, api.EMPTY_TABLE)
    builds = []
    monkeypatch.setattr(install, "build_table", lambda *a: builds.append(a))
    half = _step(pipe, t2)
    assert builds == []
    assert rel_l2(half, _step(_merged([(adapter, 0.5)]))) <= STEP_TOL
    assert rel_l2(half, full) > 1e-2
    # The same pairs serve both tables, and no weight was replaced.
    assert dict(pipe.adapters._items) == pairs
    assert all(m.base.weight is weights[p] for p, m in wrappers.items())
    assert np.array_equal(_step(pipe, t1), full)


def test_the_table_id_is_in_every_weight_dependent_cache_key(tmp_path):
    # The adapter patches leaves under the modulation and caption memos.
    adapter = {k: v for k, v in tiny_z_adapter(1, 4.0, wide=True).items()
               if k in ("t_embedder.mlp.2", "cap_embedder.1",
                        "layers.0.adaLN_modulation.0")}
    pipe = _pipe()
    ad = lora.load_adapter(pipe, write_adapter(
        tmp_path / "a.safetensors", "diffusers",
        {k: (d, u, None) for k, (d, u, _) in adapter.items()}))
    base = _step(pipe)
    assert len(pipe.modulations) == 1 and len(pipe.captions) == 1
    t1 = _table(pipe, [(ad, 1.0)])
    t2 = _table(pipe, [(ad, -1.0)])
    plain = {k: (d, u, None) for k, (d, u, _) in adapter.items()}
    for tid, strength in ((t1, 1.0), (t2, -1.0)):
        got = _step(pipe, tid)
        assert rel_l2(got, _step(_merged([(plain, strength)]))) <= STEP_TOL
        assert rel_l2(got, base) > 1e-2
    assert np.array_equal(_step(pipe), base)
    tables = {api.EMPTY_TABLE, t1, t2}
    assert {k[1] for k in pipe.modulations._items} == tables
    assert {k[1] for k in pipe.captions._items} == tables
    assert len(pipe.modulations) == 3 and len(pipe.captions) == 3
    assert {k[0] for k in pipe.tables._items} == {t1, t2}
    assert all(k[1] == mx.float32 for k in pipe.tables._items)
    assert pipe.tables.bytes > 0
    plan = api.plan_request(pipe, height=64, width=64, budget=float(1 << 40))
    assert plan.terms["adapters"] == pipe.adapters.bytes + pipe.tables.bytes


def test_built_tables_are_cached_per_table_id_and_dtype(tmp_path,
                                                        monkeypatch):
    pipe = _pipe()
    ad = lora.load_adapter(pipe, write_adapter(
        tmp_path / "a.safetensors", "kohya", tiny_z_adapter(1, 4.0)))
    tid = _table(pipe, [(ad, 1.0)])
    builds = []
    real = install.build_table

    def counting(pipe, tid, dtype, how):
        builds.append((tid, dtype))
        return real(pipe, tid, dtype, how)

    monkeypatch.setattr(install, "build_table", counting)
    _step(pipe, tid)
    _step(pipe, tid, sigma=0.5)
    assert builds == [(tid, mx.float32)]
    leaf = dict(pipe.dit.named_modules())[QKV]
    x = mx.ones((1, 3, TINY_Z_CONFIG["dim"]), mx.bfloat16)
    pipe.lora.active = tid
    try:
        y = leaf(x)
        y2 = leaf(x)
    finally:
        pipe.lora.active = None
    assert np.array_equal(np.array(y.astype(mx.float32)),
                          np.array(y2.astype(mx.float32)))
    assert builds == [(tid, mx.float32), (tid, mx.bfloat16)]
    f = pipe.lora._cur[(tid, mx.bfloat16, install.placement())].get(QKV)
    assert all(a.dtype == mx.bfloat16 for a in f.arrays())
    # q, k and v at ranks 4, 2 and 8 run as one pair of rank 14.
    assert [a.shape for a in f.arrays()] == [(64, 14), (14, 192)]


def test_pairs_are_cached_by_content_hash(tmp_path, monkeypatch):
    adapter = tiny_z_adapter(1, 4.0)
    pipe = _pipe()
    k = lora.load_adapter(pipe, write_adapter(
        tmp_path / "k.safetensors", "kohya", adapter))
    held = len(pipe.adapters)
    assert held == len(adapter) and pipe.adapters.bytes == k.nbytes
    # The same pairs in another dialect are the same content.
    o = lora.load_adapter(pipe, write_adapter(
        tmp_path / "o.safetensors", "onetrainer", adapter))
    assert len(pipe.adapters) == held and o.sha256 == k.sha256
    assert sorted(o.dit) == sorted(k.dit) and o is not k
    assert _table(pipe, [(o, 0.7)]) == _table(pipe, [(k, 0.7)])
    # An unchanged file is not read again.
    reads = []
    real = patch_table.read_lora
    monkeypatch.setattr(patch_table, "read_lora",
                        lambda *a, **kw: reads.append(a) or real(*a, **kw))
    assert lora.load_adapter(pipe, str(tmp_path / "k.safetensors")) is k
    assert reads == []
    # Another alpha or another byte is another pair.
    d, u, _ = adapter["layers.0.feed_forward.w1"]
    h = lora.pair_hash(mx.array(d), mx.array(u), 4.0)
    assert h in pipe.adapters._items and len(h) == 64
    assert lora.pair_hash(mx.array(d), mx.array(u), None) != h
    assert lora.pair_hash(mx.array(d), mx.array(u), 4.5) != h
    d2 = d.copy()
    d2[0, 0] = np.nextafter(d2[0, 0], np.float32(1))
    assert lora.pair_hash(mx.array(d2), mx.array(u), 4.0) != h
    assert lora.pair_hash(mx.array(d).astype(mx.bfloat16), mx.array(u),
                          4.0) != h
    assert lora.register_pair(pipe, d, u, 4.0) == h
    assert len(pipe.adapters) == held
    # A rewritten file is read again.
    os.utime(tmp_path / "k.safetensors", ns=(1, 1))
    assert lora.load_adapter(pipe, str(tmp_path / "k.safetensors")) is not k
    assert len(reads) == 1 and len(pipe.adapters) == held


def test_a_table_that_names_an_unknown_or_misfit_pair_is_refused():
    pipe = _pipe()
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="does not hold"):
        api.define_patch_table(pipe, [(QKV, None, [("a" * 64, 1.0)])])
    d, u, _ = make_pair(rng, 64, 64, 2)
    band = lora.register_pair(pipe, d, u)
    with pytest.raises(lora.AdapterError, match="does not fit"):
        api.define_patch_table(pipe, [(QKV, None, [(band, 1.0)])])
    with pytest.raises(lora.AdapterError,
                       match="64x64 .* the target is 170x64"):
        api.define_patch_table(pipe, [("layers.0.feed_forward.w1", None,
                                       [(band, 1.0)])])
    assert list(pipe.patch_tables) == [api.EMPTY_TABLE] and pipe.lora is None
    tid = api.define_patch_table(pipe, [(QKV, (0, 64, 64), [(band, 1.0)])])
    assert tid in pipe.patch_tables
    for d, u in ((d[0], u), (d, u[:, :1]), (d.astype(np.int32), u)):
        with pytest.raises(lora.AdapterError):
            lora.register_pair(pipe, d, u)


def test_a_zero_strength_entry_runs_the_base_leaf(tmp_path):
    pipe = _pipe()
    base = _step(pipe)
    ad = lora.load_adapter(pipe, write_adapter(
        tmp_path / "a.safetensors", "kohya", tiny_z_adapter(1, 4.0)))
    tid = _table(pipe, [(ad, 0.0)])
    assert tid != api.EMPTY_TABLE
    assert np.array_equal(_step(pipe, tid), base)
    assert pipe.tables.bytes == 0


def test_load_adapter_refuses_what_does_not_belong(tmp_path):
    pipe = _pipe()
    rng = np.random.default_rng(0)
    flux = {"transformer_blocks.0.attn.to_q": make_pair(rng, 8, 8, 2)}
    with pytest.raises(lora.AdapterFamilyError, match="not a Z-Image"):
        lora.load_adapter(pipe, write_adapter(
            tmp_path / "flux.safetensors", "diffusers", flux))
    # A Z-Image name at another width: an adapter for another checkpoint.
    wide = {"layers.0.attention.to_q": make_pair(rng, 3840, 3840, 2)}
    with pytest.raises(lora.AdapterError,
                       match="does not fit layers.0.attention.qkv of this "
                             "Z-Image DiT: the pair is 3840x3840"):
        lora.load_adapter(pipe, write_adapter(
            tmp_path / "wide.safetensors", "diffusers", wide))
    # One key too many, on a layer this DiT does not have.
    extra = dict(tiny_z_adapter(1))
    extra["layers.7.attention.to_q"] = make_pair(rng, 64, 64, 2)
    p = write_adapter(tmp_path / "extra.safetensors", "diffusers", extra)
    with pytest.raises(lora.UnmatchedKeysError,
                       match="1 of 9 targets name no Z-Image module: "
                             "transformer.layers.7.attention.to_q"):
        lora.load_adapter(pipe, p)
    ad = lora.load_adapter(pipe, p, allow_unmatched=True)
    assert len(ad.dit) == 8 and "skipped" in ad.warnings[0]
    with pytest.raises(lora.AdapterError, match="No such file"):
        lora.load_adapter(pipe, str(tmp_path / "missing.safetensors"))
    assert len(pipe.adapters) == 8 and pipe.lora is None
    no_map = api.ImagePipeline(
        dit=pipe.dit, variant="turbo",
        family=SimpleNamespace(title="Other", lora_map=None))
    with pytest.raises(lora.AdapterError, match="Other has no LoRA key map"):
        lora.load_adapter(no_map, p)


def test_sample_runs_under_a_patch_table(tmp_path):
    adapter = tiny_z_adapter(1, 4.0, wide=True)
    pipe = _pipe()
    ad = lora.load_adapter(pipe, write_adapter(
        tmp_path / "a.safetensors", "peft",
        {k: (d, u, None) for k, (d, u, _) in adapter.items()}))
    tid = _table(pipe, [(ad, 0.6)])
    _, cond = _inputs()
    kw = dict(height=48, width=64, seed=3, steps=3, cfg=1.0)
    got = _sample(pipe, cond, tid, **kw)
    plain = {k: (d, u, None) for k, (d, u, _) in adapter.items()}
    want = _sample(_merged([(plain, 0.6)]), cond, **kw)
    assert rel_l2(got, want) <= 10 * STEP_TOL
    base = _sample(pipe, cond, **kw)
    assert rel_l2(base, want) > 1e-2


@pytest.mark.parametrize("how,kind", (("block", "_Block"),
                                      ("slice", "_Slice"),
                                      ("split", "_Split")))
def test_each_placement_gives_the_step_on_merged_weights(tmp_path,
                                                         monkeypatch, how,
                                                         kind):
    assert install.PLACEMENTS == ("block", "slice", "split")
    monkeypatch.setattr(install, "_PLACEMENT", how)
    adapter = tiny_z_adapter(1, 4.0, wide=True)
    pipe = _pipe()
    ad = lora.load_adapter(pipe, write_adapter(
        tmp_path / "a.safetensors", "kohya", adapter))
    tid = _table(pipe, [(ad, 0.8)])
    got = _step(pipe, tid)
    assert rel_l2(got, _step(_merged([(adapter, 0.8)]))) <= STEP_TOL
    built = pipe.lora._cur[(tid, mx.float32, how)]
    # q, k and v bands take the placement, and a whole weight is a block.
    assert type(built.get(QKV)).__name__ == kind
    assert type(built.get("noise_refiner.0.attention.qkv")).__name__ == kind
    assert type(built.get("layers.0.feed_forward.w1")).__name__ == "_Block"
    if how != "block":
        spans = [s[:2] for s in built.get(
            "noise_refiner.0.attention.qkv").spans]
        assert spans == ([(128, 192)] if how == "slice"
                         else [(0, 128), (128, 192)])


def test_the_placements_agree_on_a_kquant_base():
    kq = pytest.importorskip("mlx_kquant")
    from mlx_kquant.nn import KQuantLinear

    from gmlx.image.lora import plan

    rng = np.random.default_rng(0)
    inp, band = 256, 32
    w = mx.array((rng.standard_normal((3 * band, inp)) / 16)
                 .astype(np.float32))
    base = KQuantLinear(inp, 3 * band, False, "q4_k")
    base.weight, base.scales = kq.quantize(w, "q4_k")
    parts = []
    for i, rank in ((0, 4), (2, 2)):
        d, u, _ = make_pair(rng, band, inp, rank)
        parts.append(plan.Part(mx.array(d), mx.array(u), None,
                               (0, i * band, band), 0.7))
    x = mx.array(rng.standard_normal((2, 5, inp)).astype(np.float32)) \
        .astype(mx.bfloat16)
    out = {}
    for how in install.PLACEMENTS:
        f = install._factors("m", parts, (3 * band, inp), mx.bfloat16, how,
                             base)
        y = f(base, x)
        assert y.dtype == mx.bfloat16 and y.shape == (2, 5, 3 * band)
        out[how] = np.array(y.astype(mx.float32), np.float64)
    plain = np.array(base(x).astype(mx.float32), np.float64)
    assert rel_l2(out["block"], plain) > 0.05
    # Band 1 has no pair, so every placement leaves it as the base gives it.
    for how in install.PLACEMENTS:
        assert rel_l2(out[how][..., band:2 * band],
                      plain[..., band:2 * band]) <= KQ_TOL, how
    assert rel_l2(out["slice"], out["block"]) <= KQ_TOL
    assert rel_l2(out["split"], out["block"]) <= KQ_TOL
