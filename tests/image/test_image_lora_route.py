"""The ``lora`` route of the image ops: when the product of a K-quant leaf
and the low-rank term of its adapter run as one mlx-kquant op, what the
op is handed, and its result against a float64 reference. The decisions
run on a recording op with the device probe stood in, so they hold on any
mlx-kquant and on any default device: the op is the build's on a GPU with
tensor ops and a stand-in on the stock ops elsewhere. Every call the
route does not take is held bit for bit to the path with no route."""
from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import mlx_kquant as kq
from mlx_kquant.nn import KQuantLinear

from gmlx.image import api, lora, ops
from gmlx.image import profile as prof
from gmlx.image.lora import install, plan
from gmlx.image.ops import kq_routes
from gmlx.image.registry import get_family
from image_fixtures import (glu_rounded_once, kq_leaf, kq_z_model,
                            lora_rounded_once, qk_rounded_once)
from lora_fixtures import make_pair, reference_delta, rel_l2

QKV = "layers.0.attention.qkv"
OTHER = "layers.1.attention.qkv"
W1 = "layers.0.feed_forward.w1"
CAP_W1 = "context_refiner.0.feed_forward.w1"
DIM = 256
BANDS = [(0, i * DIM, DIM) for i in range(3)]
# Relative L2 of a leaf's output under an adapter to the float64 sum on
# the dequantized weight, in bfloat16: the op and the stock ops both read
# 2.3e-3 to 2.6e-3. Each wrong adapter of the list below is 0.05 or more
# away.
TOL = 4e-3
WRONG = 10 * TOL
DEVICE_PROBE = kq_routes._gpu


def _gpu() -> bool:
    return mx.default_device() == mx.gpu


class _Spy:
    """Counts the calls of ``fn`` and keeps the arguments of each."""

    def __init__(self, fn):
        self.fn, self.calls, self.seen = fn, 0, []

    def __call__(self, *args, **kwargs):
        self.calls += 1
        self.seen.append((args, kwargs))
        return self.fn(*args, **kwargs)


@pytest.fixture
def op(monkeypatch):
    """The lora route alone in force on a recording op, with tensor ops
    and a GPU default device reported present. The routes, the op table
    and the placement are put back after the test."""
    monkeypatch.setattr(ops, "qk_norm_rope", ops.qk_norm_rope)
    monkeypatch.setattr(kq_routes, "_active", kq_routes._active)
    monkeypatch.setattr(kq_routes, "_OPS", dict(kq_routes._OPS))
    monkeypatch.setattr(install, "_PLACEMENT", "block")
    real = _gpu() and kq.nax_available() \
        and kq_routes._kq_op("quantized_matmul_lora")
    spy = _Spy(real or lora_rounded_once)
    kq_routes._OPS.update(quantized_matmul_lora=spy, nax=True)
    monkeypatch.setattr(kq_routes, "_gpu", lambda: True)
    # The leaves of these tests are 256 wide with 768 outputs at most,
    # under a floor of the route's table, which has its own test.
    monkeypatch.setattr(kq_routes, "_LORA_MIN_N", 0)
    assert kq_routes.install(("lora",)) == {"lora"}
    return spy


def _pipe(codec: str = "q5_k", bands: int = 3) -> api.ImagePipeline:
    """A small Z-Image DiT in bfloat16 whose w1 and w3 are q4_k leaves
    and whose fused qkv of the two main layers is a ``codec`` leaf. With
    two ``bands`` that leaf has the shape of a fused gate and up weight,
    for a test that calls the leaf and runs no step."""
    m = kq_z_model(mx.bfloat16)
    for i, blk in enumerate(m.layers):
        blk.attention.qkv = kq_leaf(codec, bands * DIM, DIM, 30 + i)
    return api.ImagePipeline(dit=m, family=get_family("z_image"),
                             variant="turbo")


def _x(rows: int = 1024, dtype=mx.bfloat16, seed: int = 0) -> mx.array:
    rng = np.random.default_rng(seed)
    x = mx.array(rng.standard_normal((1, rows, DIM)).astype(np.float32))
    x = x.astype(dtype)
    mx.eval(x)
    return x


def _pairs(seed: int, ranks=(4, 4, 4), alpha=None, out: int = DIM) -> list:
    rng = np.random.default_rng(seed)
    return [make_pair(rng, out, DIM, r, alpha) for r in ranks]


def _entries(pipe, path: str, rows: list) -> list:
    """Table entries of ``rows``, each (band, pair, strength), on
    ``path``: one entry per band, with the pairs of the band in the order
    of the rows."""
    by_band: dict = {}
    for band, (d, u, a), s in rows:
        by_band.setdefault(band, []).append(
            (lora.register_pair(pipe, d, u, a), s))
    return [(path, band, patches) for band, patches in by_band.items()]


def _qkv_rows(pairs, strength=1.0, bands=BANDS) -> list:
    return [(b, p, strength) for b, p in zip(bands, pairs)]


def _leaf(pipe, path: str):
    obj = pipe.dit
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def _run(pipe, entries, path: str, x: mx.array) -> mx.array:
    """The output of the leaf at ``path`` on ``x`` under the table of
    ``entries``."""
    tid = api.define_patch_table(pipe, entries)
    prev = install.activate(pipe, tid)
    try:
        y = _leaf(pipe, path)(x)
        mx.eval(y)
    finally:
        install.restore(pipe, prev)
    return y


def _today(pipe, entries, path: str, x: mx.array) -> mx.array:
    """The same on the path with no route."""
    with kq_routes.only(()):
        return _run(pipe, entries, path, x)


def _ref(pipe, path: str, x: mx.array, rows: list) -> np.ndarray:
    """float64 ``x @ (W + delta)^T`` on the dequantized weight, with the
    delta of ``rows``, each (band, pair, strength)."""
    base = _leaf(pipe, path)
    base = getattr(base, "base", base)
    w = np.array(kq.dequantize(base.weight, base.scales, base.kquant_type,
                               dtype=mx.float32), np.float64)
    delta = reference_delta([(d, u, a, band, s)
                             for band, (d, u, a), s in rows], w.shape) \
        if rows else 0.0
    return np.array(x.astype(mx.float32), np.float64) @ (w + delta).T


def _f64(a: mx.array) -> np.ndarray:
    return np.array(a.astype(mx.float32), np.float64)


def _same(a: mx.array, b: mx.array) -> bool:
    return a.dtype == b.dtype and bool(mx.array_equal(a, b).item())


# What the op is handed.

def test_a_kquant_leaf_under_a_table_takes_one_op(op):
    pipe = _pipe()
    pairs = _pairs(1, ranks=(4, 2, 6), alpha=3.0)
    rows = _qkv_rows(pairs, 0.7)
    x = _x()
    y = _run(pipe, _entries(pipe, QKV, rows), QKV, x)
    assert op.calls == 1 and y.shape == (1, 1024, 3 * DIM)
    assert y.dtype == mx.bfloat16
    base = _leaf(pipe, QKV).base
    args, kwargs = op.seen[0]
    assert len(args) == 5 and sorted(kwargs) == ["bands", "strength"]
    assert args[0] is x and args[1] is base["weight"]
    assert args[2] == "q5_k"
    a_t, b_t = args[3:]
    assert a_t.shape == (DIM, 12) and b_t.shape == (12, 3 * DIM)
    assert a_t.dtype == b_t.dtype == mx.bfloat16
    # One band per projection, each with its own block of ranks.
    assert kwargs["bands"] == [(0, 256, 0, 4), (256, 512, 4, 2),
                               (512, 768, 6, 6)]
    s = kwargs["strength"]
    assert s.dtype == mx.float32 and s.shape == (12,)
    assert np.array_equal(np.array(s), np.full(12, 0.7, np.float32))
    # alpha / rank is in B and the strength is not: B is the up factor of
    # each band times its 3 / rank.
    want = np.zeros((12, 3 * DIM), np.float32)
    for (c0, c1, z0, r), (_, up, _) in zip(kwargs["bands"], pairs):
        want[z0:z0 + r, c0:c1] = up.T * np.float32(3.0 / r)
    assert _same(b_t, mx.array(want).astype(mx.bfloat16))
    assert rel_l2(_f64(y), _ref(pipe, QKV, x, rows)) <= TOL


def test_a_whole_weight_pair_takes_the_op_with_no_band_table(op):
    pipe = _pipe()
    pair = _pairs(2, ranks=(8,), out=64)[0]
    rows = [(None, pair, 1.0)]
    x = _x()
    y = _run(pipe, _entries(pipe, W1, rows), W1, x)
    args, kwargs = op.seen[0]
    assert op.calls == 1 and args[2] == "q4_k"
    assert kwargs["bands"] is None and kwargs["strength"].shape == (8,)
    assert args[3].shape == (DIM, 8) and args[4].shape == (8, 64)
    assert rel_l2(_f64(y), _ref(pipe, W1, x, rows)) <= TOL


def test_stacked_adapters_join_their_ranks_in_each_band(op):
    """Two adapters on q, k and v: each band holds the ranks of both, in
    the order the table lists its pairs, each at its own strength."""
    pipe = _pipe()
    one, two = _pairs(3, ranks=(4, 4, 4)), _pairs(4, ranks=(2, 2, 2), alpha=1.0)
    rows = _qkv_rows(one, 1.0) + _qkv_rows(two, -0.5)
    x = _x()
    y = _run(pipe, _entries(pipe, QKV, rows), QKV, x)
    kwargs = op.seen[0][1]
    assert op.calls == 1
    assert kwargs["bands"] == [(0, 256, 0, 6), (256, 512, 6, 6),
                               (512, 768, 12, 6)]
    got = np.array(kwargs["strength"]).reshape(3, 6)
    for band in got:
        assert band.tolist() in ([1, 1, 1, 1, -0.5, -0.5],
                                 [-0.5, -0.5, 1, 1, 1, 1])
    assert rel_l2(_f64(y), _ref(pipe, QKV, x, rows)) <= TOL


def test_bands_with_a_gap_leave_the_gap_as_the_base_gives_it(op):
    pipe = _pipe()
    pairs = _pairs(5, ranks=(4, 4))
    rows = _qkv_rows(pairs, 1.0, [BANDS[0], BANDS[2]])
    x = _x()
    y = _run(pipe, _entries(pipe, QKV, rows), QKV, x)
    assert op.seen[0][1]["bands"] == [(0, 256, 0, 4), (512, 768, 4, 4)]
    ref = _ref(pipe, QKV, x, rows)
    assert rel_l2(_f64(y), ref) <= TOL
    plain = _ref(pipe, QKV, x, [])
    assert rel_l2(_f64(y)[..., DIM:2 * DIM], plain[..., DIM:2 * DIM]) <= TOL


@pytest.mark.parametrize("case", ["overlap", "input_band"])
def test_a_table_no_band_table_holds_goes_in_whole(op, case):
    """A pair over the whole weight beside band pairs, and a pair on a
    band of the input: every output row takes every rank of the
    block-diagonal pair, so the op gets no band table."""
    pipe = _pipe()
    rng = np.random.default_rng(6)
    if case == "overlap":
        rows = [(None, make_pair(rng, 3 * DIM, DIM, 2), 0.8),
                (BANDS[0], make_pair(rng, DIM, DIM, 4), 0.8)]
    else:
        rows = [((1, 64, 128), make_pair(rng, 3 * DIM, 128, 4), 0.8)]
    x = _x()
    entries = _entries(pipe, QKV, rows)
    y = _run(pipe, entries, QKV, x)
    assert op.calls == 1 and op.seen[0][1]["bands"] is None
    assert plan.band_table(
        [plan.Part(mx.array(d), mx.array(u), a, band, s)
         for band, (d, u, a), s in rows], (3 * DIM, DIM)) is None
    assert rel_l2(_f64(y), _f64(_today(pipe, entries, QKV, x))) <= 2 * TOL


def test_the_band_table_joins_the_ranks_of_parts_in_band_order():
    """A frozen table lists a module's parts in band order, so the parts
    reach the band table in it; parts out of it are refused."""
    def part(band, rank):
        return plan.Part(mx.zeros((rank, DIM)), mx.zeros((DIM, rank)), None,
                         band, 1.0)

    dims = (3 * DIM, DIM)
    parts = [part(BANDS[0], 4), part(BANDS[1], 1), part(BANDS[2], 2),
             part(BANDS[2], 3)]
    table = plan.band_table(parts, dims)
    assert table == [(0, 256, 0, 4), (256, 512, 4, 1), (512, 768, 5, 5)]
    with pytest.raises(ValueError, match="band order"):
        plan.band_table([parts[2], parts[0], parts[3], parts[1]], dims)
    # The entries of a frozen table come out in that order whatever order
    # they went in.
    _, frozen = api._freeze_table(tuple(
        (QKV, b, ()) for b in (BANDS[2], BANDS[0], BANDS[1])))
    assert [e[1] for e in frozen["entries"]] == [BANDS[0], BANDS[1],
                                                 BANDS[2]]
    assert plan.band_table([part(None, 4), part(None, 2)], dims) is None
    # A band over the whole output beside another is an overlap.
    assert plan.band_table([part((0, 0, 3 * DIM), 4), part(BANDS[1], 2)],
                           dims) is None
    assert plan.band_table([part((0, 128, 256), 4), part(BANDS[1], 2)],
                           dims) is None


def test_a_strength_change_builds_no_new_pair(op):
    """Two tables of the same pairs at other strengths hand the op the
    same A and B arrays, and each its own strength array."""
    pipe = _pipe()
    pairs = _pairs(7)
    x = _x()
    seen = []
    for s in (1.0, 0.6, -0.4):
        rows = _qkv_rows(pairs, s)
        y = _run(pipe, _entries(pipe, QKV, rows), QKV, x)
        assert rel_l2(_f64(y), _ref(pipe, QKV, x, rows)) <= TOL, s
        seen.append(y)
    assert op.calls == 3
    first = op.seen[0][0]
    for (args, kwargs), s in zip(op.seen, (1.0, 0.6, -0.4)):
        assert args[3] is first[3] and args[4] is first[4]
        assert np.array_equal(np.array(kwargs["strength"]),
                              np.full(12, s, np.float32))
    held = [k for k in pipe.tables._items if k[0] == install.KERNEL]
    assert len(held) == 1 and held[0][1] == QKV
    assert not _same(seen[0], seen[1]) and not _same(seen[1], seen[2])
    # Another pair on a band is another table of weights.
    other = _pairs(8)
    _run(pipe, _entries(pipe, QKV, _qkv_rows(other)), QKV, x)
    assert op.seen[3][0][3] is not first[3]
    assert len([k for k in pipe.tables._items
                if k[0] == install.KERNEL]) == 2


def test_a_kernel_pair_stays_while_a_table_built_over_it_is_held(op):
    """The kernel pair two tables share is counted once, under its own
    key, and in neither table. Bounds lowered to nothing shed the table
    that last ran only with its pairs, and the kernel pair only with the
    last table built over it."""
    pipe = _pipe()
    pairs = _pairs(7)
    x = _x()
    tids = []
    for s in (1.0, 0.6):
        entries = _entries(pipe, QKV, _qkv_rows(pairs, s))
        _run(pipe, entries, QKV, x)
        tids.append(api.define_patch_table(pipe, entries))
    (key,) = [k for k in pipe.tables._items if k[0] == install.KERNEL]
    a_t, b_t, _ = pipe.tables.peek(key)
    pair = int(a_t.nbytes) + int(b_t.nbytes)
    built = [pipe.tables.peek(k) for k in pipe.tables._items if k[0] in tids]
    assert len(built) == 2
    assert pipe.tables.bytes == sum(t.nbytes for t in built) + pair
    pipe.tables.max_bytes = 0
    install.shed_to_bounds(pipe)
    (cur,) = pipe.lora._cur.values()
    assert cur is built[1]
    assert [pipe.tables.peek(k) for k in pipe.tables._items
            if k[0] in tids] == [cur]
    assert pipe.tables.peek(key) is not None
    assert pipe.tables.bytes == cur.nbytes + pair
    pipe.adapters.max_bytes = 0
    install.shed_to_bounds(pipe)
    assert pipe.adapters.bytes == pipe.tables.bytes == 0
    assert not pipe.lora._cur


def test_a_put_that_drops_the_table_that_ran_sheds_its_kernel_pair(op):
    """The table that ran last is held past a tables bound of nothing,
    with its kernel pair. A put that needs the room of its pairs drops
    it, and the kernel pair it kept goes in the same call."""
    pipe = _pipe()
    _run(pipe, _entries(pipe, QKV, _qkv_rows(_pairs(7))), QKV, _x())
    pipe.tables.max_bytes = 0
    install.shed_to_bounds(pipe)
    assert pipe.lora._cur and pipe.tables.bytes > 0
    pipe.adapters.max_bytes = pipe.adapters.bytes
    lora.register_pair(pipe, *_pairs(8)[0])
    assert not pipe.lora._cur
    assert pipe.tables.bytes == 0


def _two_over_one_dit() -> tuple:
    p1 = _pipe()
    return p1, api.ImagePipeline(dit=p1.dit, family=p1.family,
                                 variant="turbo")


def test_each_pipeline_over_one_dit_runs_and_counts_its_own_table(op):
    """Two pipelines over one DiT define the same table, under one id.
    Each runs the table built over its own pairs and counts it in its own
    tables cache. The pipeline whose table did not run last holds nothing
    for the patch state, and its forward that stops before the first
    leaf records nothing in either cache."""
    p1, p2 = _two_over_one_dit()
    pairs, x = _pairs(7), _x()
    y1 = _run(p1, _entries(p1, QKV, _qkv_rows(pairs)), QKV, x)
    (t1,) = p1.lora._cur.values()
    y2 = _run(p2, _entries(p2, QKV, _qkv_rows(pairs)), QKV, x)
    ((key, t2),) = p2.lora._cur.items()
    assert p2.lora is p1.lora and t2 is not t1 and _same(y1, y2)
    assert p1.tables.peek(key) is t1 and p2.tables.peek(key) is t2
    stepped = p2.tables.used(key)
    install.restore(p1, install.activate(p1, key[0]))
    assert p1.tables.peek(key) is t1 and p2.lora._cur == {key: t2}
    assert p2.tables.used(key) == stepped
    p1.tables.max_bytes = 0
    install.shed_to_bounds(p1)
    assert p1.tables.bytes == 0


def test_drop_built_keeps_the_tables_another_pipeline_over_the_dit_ran(op):
    """drop_built of one pipeline over a DiT drops its own tables and
    keeps those the patch state holds for the other, whose tables cache
    stays over its bound by them alone."""
    p1, p2 = _two_over_one_dit()
    pairs, x = _pairs(7), _x()
    _run(p1, _entries(p1, QKV, _qkv_rows(pairs)), QKV, x)
    _run(p2, _entries(p2, QKV, _qkv_rows(pairs, 0.6)), QKV, x)
    cur = dict(p2.lora._cur)
    p2.tables.max_bytes = 0
    install.shed_to_bounds(p2)
    held = p2.tables.bytes
    install.drop_built(p1)
    assert not [k for k in p1.tables.keys(lambda k: k[0] != install.KERNEL)]
    assert p2.lora._cur == cur and p2.tables.bytes == held > 0
    _check_caches(p2)


def _w1_patch(pipe, rng, hold=()) -> str:
    """A ComfyUI patch on the K-quant w1, kept as register_patch keeps
    one."""
    out, inp = install.linear_dims(_leaf(pipe, W1))
    d, u, _ = make_pair(rng, out, inp, 4)
    return lora.register_patch(pipe, mx.array(u), mx.array(d), hold=hold)


def _w1_step(pipe, keys) -> None:
    """A forward of w1 under the table of ``keys``, or of the base model
    without keys."""
    _w1_run(pipe, api.define_patch_table(
        pipe, [(W1, None, [(k, 1.0) for k in keys])] if keys else []))


def _w1_run(pipe, tid) -> None:
    """A forward of w1 under the table ``tid``, defined before."""
    inp = install.linear_dims(_leaf(pipe, W1))[1]
    x = mx.array(np.random.default_rng(0).standard_normal(
        (1, 64, inp)).astype(np.float32)).astype(mx.bfloat16)
    prev = install.activate(pipe, tid)
    try:
        mx.eval(_leaf(pipe, W1)(x))
    finally:
        install.restore(pipe, prev)


def test_patches_kept_with_no_hold_shed_the_stale_tables_before_them(
        op, monkeypatch):
    """A library caller keeps a table's two patches with no hold on a full
    pair cache whose pairs two stepped tables hold. The tables go in the
    order they stepped, each before the pairs kept after its step, so the
    first patch is still held for the define. Each put drops one table, in
    the order the check gives."""
    order = _OrderCheck(monkeypatch)
    rng = np.random.default_rng(11)
    pipe = _pipe()
    for _ in range(2):
        _w1_step(pipe, [_w1_patch(pipe, rng)])
    pipe.adapters.max_bytes = pipe.adapters.bytes
    keys = [_w1_patch(pipe, rng) for _ in range(2)]
    _w1_step(pipe, keys)
    assert pipe.adapters.bytes == pipe.adapters.max_bytes
    assert order.drops == 2


def test_register_patch_keeps_the_pairs_it_holds_for_the_table(op):
    """A table's first patch is kept, another table steps after it, and
    the second patch needs room on a full pair cache. With the first
    patch held, the room sheds the stepped table and its pair, though
    the first patch was used before that step."""
    rng = np.random.default_rng(12)
    pipe = _pipe()
    old = [_w1_patch(pipe, rng)]
    _w1_step(pipe, old)
    pipe.adapters.max_bytes = 2 * pipe.adapters.bytes
    first = _w1_patch(pipe, rng)
    _w1_step(pipe, old)
    keys = [first, _w1_patch(pipe, rng, hold=[first])]
    _w1_step(pipe, keys)
    assert pipe.adapters.peek(old[0]) is None


def test_a_forward_of_the_base_model_is_no_step_of_the_last_table(op):
    """The table that ran last stays held through a forward of the base
    model, which does not step it. A patch kept between the two was used
    after the table's step, so the room for another drops the table and
    its pair first."""
    rng = np.random.default_rng(13)
    pipe = _pipe()
    old = [_w1_patch(pipe, rng)]
    _w1_step(pipe, old)
    pipe.adapters.max_bytes = 2 * pipe.adapters.bytes
    kept = _w1_patch(pipe, rng)
    _w1_step(pipe, [])
    assert pipe.lora._cur
    _w1_patch(pipe, rng)
    assert pipe.adapters.peek(kept) is not None
    assert pipe.adapters.peek(old[0]) is None


def test_a_define_is_a_use_of_the_pairs_it_names(op):
    """A library caller keeps patch p, steps a table over another patch,
    then defines a table over p and keeps one more patch on a full pair
    cache. The define is p's last use, after that step, so the stepped
    table and its patch go, p stays, and the new table steps."""
    rng = np.random.default_rng(31)
    pipe = _pipe()
    p = _w1_patch(pipe, rng)
    _w1_step(pipe, [_w1_patch(pipe, rng)])
    tid = api.define_patch_table(pipe, [(W1, None, [(p, 1.0)])])
    pipe.adapters.max_bytes = pipe.adapters.bytes
    _w1_patch(pipe, rng)
    assert pipe.adapters.peek(p) is not None
    _w1_run(pipe, tid)


def test_a_build_is_a_use_of_the_pairs_it_reads(op):
    """A table is defined over patch q, patch r is kept, and the table
    steps, which builds it over q. The table then leaves the tables cache
    for another's build, and a patch that needs one pair of room sheds r,
    kept before the build, so the table steps again over q."""
    rng = np.random.default_rng(32)
    pipe = _pipe()
    q = _w1_patch(pipe, rng)
    tid = api.define_patch_table(pipe, [(W1, None, [(q, 1.0)])])
    r = _w1_patch(pipe, rng)
    _w1_run(pipe, tid)
    pipe.tables.max_bytes = pipe.tables.bytes
    _w1_step(pipe, [_w1_patch(pipe, rng)])
    assert not any(k[0] == tid for k in pipe.tables.keys(lambda k: True))
    pipe.adapters.max_bytes = pipe.adapters.bytes
    _w1_patch(pipe, rng)
    assert pipe.adapters.peek(q) is not None and pipe.adapters.peek(r) is None
    _w1_run(pipe, tid)


def test_a_run_of_steps_makes_its_last_step_the_tables_use(op):
    """A table steps, a patch is kept, and the table steps again from the
    patch state, which reads no cache at the step's start. The end of that
    step is the table's use, so the room for another patch sheds the patch
    kept between the steps and keeps the table."""
    rng = np.random.default_rng(25)
    pipe = _pipe()
    t = [_w1_patch(pipe, rng)]
    _w1_step(pipe, t)
    between = _w1_patch(pipe, rng)
    _w1_step(pipe, t)
    pipe.adapters.max_bytes = pipe.adapters.bytes
    _w1_patch(pipe, rng)
    assert pipe.adapters.peek(between) is None
    assert pipe.adapters.peek(t[0]) is not None and pipe.lora._cur


@pytest.mark.parametrize("strength,alpha", [(1.0, None), (1.0, 3.0),
                                            (0.7, None), (-0.5, 2.0)])
def test_one_table_serves_the_op_and_a_call_of_few_rows(op, strength, alpha):
    """A leaf that runs at 1024 rows and at 19, as a block of Qwen-Image
    2.1 does on its image rows and its text rows: the op takes the first,
    and the second is the path with no route, bit for bit. The stock
    factors are the pair's A, and the pair's B at strength 1.0, so the
    table holds no second copy of them. Any other strength holds one B."""
    pipe = _pipe()
    rows = _qkv_rows(_pairs(18, alpha=alpha), strength)
    entries = _entries(pipe, QKV, rows)
    tid = api.define_patch_table(pipe, entries)
    big, small = _x(), _x(19, seed=1)
    prev = install.activate(pipe, tid)
    try:
        leaf = _leaf(pipe, QKV)
        y_big, y_small = leaf(big), leaf(small)
        mx.eval(y_big, y_small)
        f = pipe.lora.factors(QKV, mx.bfloat16)
    finally:
        install.restore(pipe, prev)
    assert op.calls == 1 and op.seen[0][0][0] is big
    assert rel_l2(_f64(y_big), _ref(pipe, QKV, big, rows)) <= TOL
    assert _same(y_small, _today(pipe, entries, QKV, small))
    stock = f.stock()
    assert type(f) is install._Kernel and type(stock) is install._Block
    assert stock.a_t is f.a_t
    assert (stock.b_t is f.b_t) == (strength == 1.0)
    extra = 0 if strength == 1.0 else int(f.b_t.nbytes)
    assert sum(int(a.nbytes) for a in f.arrays()) \
        == int(f.strength.nbytes) + extra


def test_a_leaf_with_no_entry_runs_its_base(op):
    pipe = _pipe()
    x = _x()
    entries = _entries(pipe, QKV, _qkv_rows(_pairs(9)))
    tid = api.define_patch_table(pipe, entries)
    other = _leaf(pipe, OTHER)
    plain = other.base(x)
    prev = install.activate(pipe, tid)
    try:
        y = other(x)
        mx.eval(y, plain)
    finally:
        install.restore(pipe, prev)
    assert op.calls == 0 and _same(y, plain)


# The calls the route does not take run the path of today, bit for bit.

class _Scaled(KQuantLinear):
    def __call__(self, x, lora=None):
        return 2 * super().__call__(x, lora)


@pytest.mark.parametrize("case", [
    "route_off", "op_absent", "float_leaf", "bias", "subclass", "float32_x",
    "float16_x", "rows_1023", "codec", "outputs", "ranks", "cpu_device",
    "no_tensor_ops", "slice", "split"])
def test_a_call_the_route_does_not_take_is_the_path_of_today(
        op, monkeypatch, case):
    pipe = _pipe("q8_0" if case == "codec" else "q5_k")
    x = _x()
    path = QKV
    if case == "route_off":
        assert kq_routes.install(("qk", "glu")) <= {"qk", "glu"}
    elif case == "op_absent":
        kq_routes._OPS["quantized_matmul_lora"] = None
        assert "lora" not in kq_routes.install(kq_routes.ROUTES)
    elif case == "float_leaf":
        path = "layers.0.attention.out"
    elif case == "bias":
        pipe.dit.layers[0].attention.qkv = kq_leaf("q5_k", 3 * DIM, DIM, 30,
                                                   bias=True)
    elif case == "subclass":
        pipe.dit.layers[0].attention.qkv.__class__ = _Scaled
    elif case == "float32_x":
        x = x.astype(mx.float32)
    elif case == "float16_x":
        x = x.astype(mx.float16)
    elif case == "rows_1023":
        x = x[:, :1023]
    elif case == "codec":
        pass
    elif case == "outputs":
        monkeypatch.setattr(kq_routes, "_LORA_MIN_N", 3 * DIM + 1)
    elif case == "ranks":
        # 4 ranks on a column.
        monkeypatch.setattr(kq_routes, "_LORA_MAX_RANKS", 3)
    elif case == "cpu_device":
        monkeypatch.setattr(kq_routes, "_gpu", DEVICE_PROBE)
    elif case == "no_tensor_ops":
        kq_routes._OPS["nax"] = False
    else:
        monkeypatch.setattr(install, "_PLACEMENT", case)
    rng = np.random.default_rng(10)
    if path == QKV:
        rows = _qkv_rows(_pairs(10, alpha=2.0), 0.7)
    else:
        rows = [(None, make_pair(rng, DIM, DIM, 4, 2.0), 0.7)]
    entries = _entries(pipe, path, rows)
    if case == "cpu_device":
        with mx.stream(mx.cpu):
            assert not kq_routes._gpu()
            got = _run(pipe, entries, path, x)
            want = _today(pipe, entries, path, x)
    else:
        got = _run(pipe, entries, path, x)
        want = _today(pipe, entries, path, x)
    assert op.calls == 0
    assert _same(got, want)
    ref = _ref(pipe, path, x, rows) if case not in ("float_leaf", "bias",
                                                    "subclass") else None
    if ref is not None:
        assert rel_l2(_f64(got), ref) <= TOL


def test_the_route_reads_the_device_at_each_call(op, monkeypatch):
    """One built table serves a call on the CPU, on the stock ops, and
    the next on the GPU, on the op."""
    monkeypatch.setattr(kq_routes, "_gpu", DEVICE_PROBE)
    pipe = _pipe()
    x = _x()
    entries = _entries(pipe, QKV, _qkv_rows(_pairs(11)))
    with mx.stream(mx.cpu):
        _run(pipe, entries, QKV, x)
    assert op.calls == 0
    _run(pipe, entries, QKV, x)
    assert op.calls == (1 if _gpu() else 0)


def test_an_absent_op_builds_the_tables_of_today(op):
    """On an mlx-kquant without the op the route is not in force, and a
    table is built and kept as with no route: the block placement, with
    the strength in B."""
    kq_routes._OPS["quantized_matmul_lora"] = None
    assert "lora" not in kq_routes.install(kq_routes.ROUTES)
    pipe = _pipe()
    x = _x()
    entries = _entries(pipe, QKV, _qkv_rows(_pairs(12), 0.5))
    got = _run(pipe, entries, QKV, x)
    keys = list(pipe.tables._items)
    assert [k[2] for k in keys] == ["block"]
    f = pipe.tables.get(keys[0]).get(QKV)
    assert type(f) is install._Block
    assert _same(got, _today(pipe, entries, QKV, x))
    assert list(pipe.tables._items) == keys


# The table of the route.

def _bands(n: int, width: int, rank: int) -> list:
    return [(i * width, (i + 1) * width, i * rank, rank) for i in range(n)]


TABLE = {
    # case: (codec, out, in, x dtype, ranks, bands, taken)
    "z_qkv_rank_32": ("q5_k", 11520, 3840, mx.bfloat16, 96,
                      _bands(3, 3840, 32), True),
    "z_qkv_rank_16": ("q5_k", 11520, 3840, mx.bfloat16, 48,
                      _bands(3, 3840, 16), True),
    "z_qkv_q6_k_rank_128": ("q6_k", 11520, 3840, mx.bfloat16, 384,
                            _bands(3, 3840, 128), True),
    "z_qkv_two_of_three_bands": ("q5_k", 11520, 3840, mx.bfloat16, 128,
                                 [(0, 3840, 0, 64), (7680, 11520, 64, 64)],
                                 True),
    "qwen_gate_up_rank_16": ("q4_k", 24576, 4096, mx.bfloat16, 32,
                             _bands(2, 12288, 16), True),
    "qwen_gate_up_rank_1": ("q4_k", 24576, 4096, mx.bfloat16, 2,
                            _bands(2, 12288, 1), True),
    "z_w1_whole_rank_64": ("q4_k", 10240, 3840, mx.bfloat16, 64, None, True),
    "z_w1_whole_rank_16": ("q4_k", 10240, 3840, mx.bfloat16, 16, None,
                           True),
    "ranks_256_on_a_column": ("q4_k", 10240, 3840, mx.bfloat16, 256, None,
                              True),
    "ranks_257_on_a_column": ("q4_k", 10240, 3840, mx.bfloat16, 257, None,
                              False),
    "float16_x": ("q5_k", 11520, 3840, mx.float16, 96, _bands(3, 3840, 32),
                  False),
    "q8_0": ("q8_0", 11520, 3840, mx.bfloat16, 96, _bands(3, 3840, 32),
             False),
    "q3_k": ("q3_k", 11520, 3840, mx.bfloat16, 96, _bands(3, 3840, 32),
             False),
    "z_out_square": ("q5_k", 3840, 3840, mx.bfloat16, 128, None, False),
    "z_w2_inner_dim_10240": ("q4_k", 3840, 10240, mx.bfloat16, 128, None,
                             False),
    "inner_dim_4096": ("q4_k", 8192, 4096, mx.bfloat16, 64, None, True),
    "inner_dim_4352": ("q4_k", 8704, 4352, mx.bfloat16, 64, None, False),
    "outputs_8160": ("q4_k", 8160, 3840, mx.bfloat16, 64, None, False),
    "band_starts_off_64": ("q5_k", 11520, 3840, mx.bfloat16, 64,
                           [(0, 3872, 0, 32), (3872, 11520, 32, 32)], False),
    "band_ends_off_32": ("q5_k", 11520, 3840, mx.bfloat16, 64,
                         [(0, 3856, 0, 32), (7680, 11520, 32, 32)], False),
    "bands_16": ("q4_k", 16384, 4096, mx.bfloat16, 512, _bands(16, 1024, 32),
                 True),
    "bands_17": ("q4_k", 17408, 4096, mx.bfloat16, 544, _bands(17, 1024, 32),
                 False),
    "no_band": ("q5_k", 11520, 3840, mx.bfloat16, 96, [], False),
}


@pytest.mark.parametrize("case", list(TABLE))
def test_the_table_of_calls_the_route_takes(case):
    """The leaves of the two families and the bounds of the table, on
    leaves of the real dims with no weights set: the table reads the
    codec, the dims, the ranks and the bands, and no weight."""
    codec, out, inp, dtype, ranks, bands, taken = TABLE[case]
    leaf = KQuantLinear(inp, out, False, codec)
    assert kq_routes.lora_static(leaf, dtype, ranks, bands) == taken
    if taken:
        assert not kq_routes.lora_static(
            KQuantLinear(inp, out, True, codec), dtype, ranks, bands)
        assert not kq_routes.lora_static(
            nn.Linear(inp, out, bias=False), dtype, ranks, bands)


def test_the_route_takes_a_call_from_1024_rows(op):
    leaf = KQuantLinear(DIM, 3 * DIM, False, "q5_k")
    for rows, taken in ((1023, False), (1024, True), (4128, True)):
        x = mx.zeros((1, rows, DIM), mx.bfloat16)
        assert (kq_routes.lora_leaf(leaf, x) is leaf) == taken, rows
    assert kq_routes.lora_leaf(leaf, mx.zeros((2, 512, DIM),
                                              mx.bfloat16)) is leaf
    assert kq_routes.lora_leaf(leaf, mx.zeros((DIM,), mx.bfloat16)) is None
    assert kq_routes.lora_leaf(leaf, mx.zeros((1024, 0), mx.bfloat16)) is None


# The op against a float64 reference, and the wrong adapters.

def _wrong(pipe, x: mx.array, one: list, two: list, other: list) -> dict:
    """The float64 outputs of the wrong adapters of a stack of ``one`` at
    0.8 and ``two`` at -0.5 on the bands of a fused weight: a band
    dropped, two bands swapped, the sign of a strength, an alpha ignored,
    the second adapter dropped, and the pairs of another block."""
    bands = BANDS[:len(one)]

    def ref(a, b, sa=0.8, sb=-0.5, ba=bands, bb=bands):
        return _ref(pipe, QKV, x, _qkv_rows(a, sa, ba) + _qkv_rows(b, sb, bb))

    bare = [(d, u, None) for d, u, _ in one]
    return {
        "a band dropped": ref(one[:-1], two),
        "bands swapped": ref(one, two, ba=[bands[1], bands[0], *bands[2:]]),
        "strength sign": ref(one, two, sb=0.5),
        "alpha ignored": ref(bare, two),
        "second adapter dropped": ref(one, []),
        "wrong block": ref(other, two),
    }


@pytest.mark.parametrize("codec", ["q4_k", "q5_k", "q6_k"])
@pytest.mark.parametrize("rows", [1024, 4608])
@pytest.mark.parametrize("bands", [3, 2], ids=["qkv", "gate_up"])
def test_the_op_against_the_float64_reference(op, codec, rows, bands):
    """The leaf's output under a stack of two adapters on the bands of a
    fused weight, the three of q, k and v or the two of a gate and up
    weight, is the float64 sum on the dequantized weight within the gate,
    no further from it than the stock ops are, and every wrong adapter is
    far outside the gate. On a GPU with tensor ops the op is the build's,
    and 4608 rows are on its kernel."""
    pipe = _pipe(codec, bands)
    one = _pairs(13, ranks=(4,) * bands, alpha=2.0)
    two = _pairs(14, ranks=(2,) * bands)
    other = _pairs(15, ranks=(4,) * bands, alpha=2.0)
    right = _qkv_rows(one, 0.8) + _qkv_rows(two, -0.5)
    x = _x(rows)
    entries = _entries(pipe, QKV, right)
    got = _f64(_run(pipe, entries, QKV, x))
    stock = _f64(_today(pipe, entries, QKV, x))
    assert op.calls == 1 and np.isfinite(got).all()
    ref = _ref(pipe, QKV, x, right)
    err, err_stock = rel_l2(got, ref), rel_l2(stock, ref)
    print(f"lora {codec} rows {rows} bands {bands}: op {err:.3e} stock "
          f"{err_stock:.3e}")
    assert err <= TOL, err
    assert err <= 1.1 * err_stock, (err, err_stock)
    for name, wrong in _wrong(pipe, x, one, two, other).items():
        off = rel_l2(got, wrong)
        assert off > WRONG, (name, off)
    # A wrong table handed to the route gives the wrong sum: the route
    # adds nothing that hides it.
    swapped = _qkv_rows(one, 0.8, [BANDS[1], BANDS[0], BANDS[2]][:bands]) \
        + _qkv_rows(two, -0.5)
    bad = _f64(_run(pipe, _entries(pipe, QKV, swapped), QKV, x))
    assert rel_l2(bad, ref) > WRONG
    assert rel_l2(bad, _ref(pipe, QKV, x, swapped)) <= TOL


@pytest.mark.parametrize("inp, out, bands", [(3840, 11520, 3),
                                              (3072, 24576, 2)],
                         ids=["z_qkv", "qwen_gate_up"])
def test_the_op_at_a_table_width_against_the_float64_reference(
        op, inp, out, bands):
    """The same row at the widths of the route's table: the fused qkv
    weight of Z-Image, three bands of 3840, and the fused gate and up
    weight of Qwen-Image 2.1, two bands of 12288. A strength a tenth off
    at these widths alone moves the output many gates."""
    pipe = _pipe()
    pipe.dit.layers[0].attention.qkv = kq_leaf("q4_k", out, inp, 40)
    width = out // bands
    on = [(0, i * width, width) for i in range(bands)]
    rng = np.random.default_rng(16)
    one = [make_pair(rng, width, inp, 4, 2.0) for _ in range(bands)]
    two = [make_pair(rng, width, inp, 2) for _ in range(bands)]
    rows = _qkv_rows(one, 0.8, on) + _qkv_rows(two, -0.5, on)
    rng = np.random.default_rng(17)
    x = mx.array(rng.standard_normal((1, 1024, inp)).astype(np.float32)
                 ).astype(mx.bfloat16)
    entries = _entries(pipe, QKV, rows)
    got = _f64(_run(pipe, entries, QKV, x))
    stock = _f64(_today(pipe, entries, QKV, x))
    assert op.calls == 1 and op.seen[0][1]["bands"] == [
        (c, c + width, 6 * i, 6) for i, c in enumerate(range(0, out, width))]
    ref = _ref(pipe, QKV, x, rows)
    err, err_stock = rel_l2(got, ref), rel_l2(stock, ref)
    print(f"lora {inp}x{out} bands {bands}: op {err:.3e} stock "
          f"{err_stock:.3e}")
    assert err <= TOL and err <= 1.1 * err_stock, (err, err_stock)
    # The delta at a tenth less strength is many gates away.
    low = _qkv_rows(one, 0.72, on) + _qkv_rows(two, -0.45, on)
    assert rel_l2(got, _ref(pipe, QKV, x, low)) > WRONG / 4


# Kept results, and the profiler.

def test_kept_results_follow_the_lora_route(op, monkeypatch):
    """The caption stream of a step under a table that adapts a K-quant
    leaf of the context refiner is kept by the routes in force: with the
    lora route in force it is the stream the op made, and with it out of
    force the stream of the stock ops, each bit for bit the step of a
    fresh pipeline. A caption stream has fewer rows than the route
    takes, so the floor on rows is dropped here: the test holds the key,
    for a table that takes such a call."""
    monkeypatch.setattr(kq_routes, "_LORA_MIN_ROWS", 1)
    kq_routes._OPS.update(qk_norm_rope=qk_rounded_once,
                          quantized_matmul_glu=glu_rounded_once,
                          quantized_matmul_lora=lora_rounded_once)
    assert kq_routes.install(kq_routes.ROUTES) == set(kq_routes.ROUTES)
    rng = np.random.default_rng(0)
    lat = rng.standard_normal((1, 4, 64, 64)).astype(np.float32)
    feats = rng.standard_normal((5, 16)).astype(np.float32)
    pair = _pairs(16, ranks=(8,), out=64)[0]

    def made():
        pipe = _pipe()
        tid = api.define_patch_table(
            pipe, _entries(pipe, CAP_W1, [(None, pair, 1.0)]))
        return pipe, tid

    def step(made) -> np.ndarray:
        pipe, tid = made
        cond = api.Conditioning(mx.array(feats), key="kept")
        return np.array(api.denoise_step(pipe, lat, 0.6, cond,
                                         patch_table=tid))

    kept = made()
    on = step(kept)
    with kq_routes.only(("qk", "glu")):
        off, fresh_off = step(kept), step(made())
    again, fresh_on = step(kept), step(made())
    assert not np.array_equal(on, off)
    assert np.array_equal(off, fresh_off)
    assert np.array_equal(again, on) and np.array_equal(on, fresh_on)
    # Each set of routes keeps its own caption stream and modulation.
    assert len(kept[0].captions) == len(kept[0].modulations) == 2
    routes = {k[-1] for k in kept[0].captions._items}
    assert routes == {frozenset(kq_routes.ROUTES), frozenset(("qk", "glu"))}


def test_a_timed_leaf_keeps_the_route_and_books_one_record(op):
    """Under an installed StepProfiler a K-quant leaf is a timed subclass
    of its own class, and the route takes it as it takes the plain one.
    The op books one seam record under the leaf's path, in place of the
    leaf's module record, and the records add up to the wall time."""
    pipe = _pipe()
    x = _x()
    entries = _entries(pipe, QKV, _qkv_rows(_pairs(17)))
    want = _run(pipe, entries, QKV, x)
    assert op.calls == 1
    with prof.StepProfiler(pipe.dit) as p:
        assert type(_leaf(pipe, QKV).base).__name__ == "TimedKQuantLinear"
        out = []
        records, wall = p.step(
            lambda: out.append(_run(pipe, entries, QKV, x)) or out[0])
        ran = [(r.kind, r.name, r.path) for r in records if r.kind != "glue"]
        assert ran == [("seam", "lora_fused", QKV + ".base")]
        fused = next(r for r in records if r.kind == "seam")
        assert fused.in_shapes[0] == [1, 1024, DIM]
        assert fused.out_shape == [1, 1024, 3 * DIM]
        assert sum(r.seconds for r in records) == pytest.approx(
            wall, rel=1e-6, abs=1e-6)
    assert op.calls == 2 and _same(out[0], want)
    # A call the route does not take books the leaf, as with no route.
    with kq_routes.only(()), prof.StepProfiler(pipe.dit) as p:
        records, _ = p.step(lambda: _run(pipe, entries, QKV, x))
    assert [(r.kind, r.name) for r in records if r.kind != "glue"] == [
        ("module", "KQuantLinear")]
    assert type(_leaf(pipe, QKV).base) is KQuantLinear
    assert ops.lora_fused is kq_routes.lora_fused


# The two LoRA caches under a random sequence of their callers'
# operations, in the kernel form and on stock leaves.

SLOTS = [*((QKV, b) for b in BANDS), (QKV, None), (OTHER, None), (W1, None),
         ("layers.0.attention.out", None), ("layers.1.feed_forward.w2", None)]
OWN_ALPHA = 4.0
OPS_PER_SEED = 300


def _slot_dims(pipe, slot) -> tuple:
    path, band = slot
    out, inp = install.linear_dims(_leaf(pipe, path))
    return (band[2] if band else out), inp


def _new_pair(pipe, rng, slot) -> tuple:
    out, inp = _slot_dims(pipe, slot)
    d, u, _ = make_pair(rng, out, inp, int(rng.choice([2, 4, 8])))
    dtype = mx.bfloat16 if rng.random() < 0.5 else mx.float32
    d, u = mx.array(d).astype(dtype), mx.array(u).astype(dtype)
    mx.eval(d, u)
    return d, u


def _reach(obj, out: dict) -> None:
    """The arrays ``obj`` keeps alive, by identity: a pair, a built table's
    parts and factors, a kernel pair. The leaves of the DiT and the
    closures over them are left out."""
    if isinstance(obj, mx.array):
        out[id(obj)] = obj
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            _reach(o, out)
    elif isinstance(obj, dict):
        for o in obj.values():
            _reach(o, out)
    elif isinstance(obj, (install.Pair, plan.Part)):
        _reach([obj.down, obj.up], out)
    elif type(obj).__module__ == install.__name__:
        for k, v in vars(obj).items():
            if k not in ("_leaves", "_pairs", "_plain", "fns"):
                _reach(v, out)


def _check_caches(pipe) -> None:
    """The distinct arrays the two caches and the patch state keep alive
    add up to the bytes the caches count; a cache over its bound holds
    only what is pinned; every held table's pairs are in the pair cache;
    each cache's clock holds a use for each of its entries and no other."""
    ad, tb = pipe.adapters, pipe.tables
    for cache in (ad, tb):
        assert set(cache._used) == set(cache.keys(lambda k: True))
    live: dict = {}
    for k in ad.keys(lambda k: True):
        _reach(ad.peek(k), live)
    held = []
    for k in tb.keys(lambda k: True):
        v = tb.peek(k)
        if k[0] == install.KERNEL:
            _reach(v, live)
        else:
            held.append(v)
    cur = list(pipe.lora._cur.values()) if pipe.lora is not None else []
    for t in held + cur:
        _reach(t, live)
    assert sum(int(a.nbytes) for a in live.values()) == ad.bytes + tb.bytes
    if ad.bytes > ad.max_bytes:
        assert set(ad.keys(lambda k: True)) <= ad.kept
    if tb.bytes > tb.max_bytes:
        mine = {id(t) for t in cur}
        kernels = set().union(*(t.kernels for t in cur))
        free = [(k[0][:8], tb.nbytes(k)) for k in tb.keys(lambda k: True)
                if k not in kernels and id(tb.peek(k)) not in mine]
        assert not free, (tb.bytes, tb.max_bytes, free)
    pair_arrays: dict = {}
    for k in ad.keys(lambda k: True):
        _reach(ad.peek(k), pair_arrays)
    for t in held + cur:
        assert all(ad.peek(k) is not None for k in t.keys)
        parts: dict = {}
        _reach(t._parts, parts)
        assert parts.keys() <= pair_arrays.keys()


class _Sequence:
    """A pipeline, the entry's own pairs, and the pairs and tables the
    operations made, with one method per operation."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.pipe = _pipe()
        self.own = [(slot, *_new_pair(self.pipe, self.rng, slot))
                    for slot in (SLOTS[1], (W1, None), SLOTS[-2])]
        self.own_keys: list = []
        self.load_own()
        own = self.pipe.adapters.bytes
        self.bounds = {"adapters": own + (96 << 10), "tables": 96 << 10}
        self.pipe.adapters.max_bytes = self.bounds["adapters"]
        self.pipe.tables.max_bytes = self.bounds["tables"]
        self.pool: list = []        # (slot, down, up, alpha, twin)
        self.tids: list = []
        self.refused = self.missing = 0

    def load_own(self) -> None:
        # The entry reads its file into a cache fitted to hold it, then
        # sets the keep set.
        need = sum(int(d.nbytes) + int(u.nbytes) for _s, d, u in self.own)
        self.pipe.adapters.max_bytes = max(self.pipe.adapters.max_bytes, need)
        self.own_keys = []
        for _slot, d, u in self.own:
            self.own_keys.append(install.register_pair(
                self.pipe, d, u, OWN_ALPHA, hold=self.own_keys))
        self.pipe.adapters.keep(self.own_keys)

    def _held(self) -> tuple:
        pipe = self.pipe
        cur = pipe.lora._cur.values() if pipe.lora is not None else ()
        return (set(pipe.adapters.keys(lambda k: True)),
                set(pipe.tables.keys(lambda k: True)), {id(t) for t in cur})

    def _put(self, row, hold=()) -> str | None:
        slot, d, u, alpha, twin = row
        pipe = self.pipe
        nbytes = int(d.nbytes) + int(u.nbytes)
        before = self._held()
        try:
            return install.register_pair(pipe, d, u, alpha, twin=twin,
                                         hold=hold)
        except install.PairRoomError:
            # Only a pair that cannot fit beside the pairs that stay, and
            # its refusal sheds nothing.
            stay = set(pipe.adapters.kept) | set(hold)
            assert sum(pipe.adapters.nbytes(k) for k in stay) + nbytes \
                > pipe.adapters.max_bytes
            assert self._held() == before
            self.refused += 1
            return None

    def _fresh(self) -> tuple:
        slot = SLOTS[self.rng.integers(len(SLOTS))]
        alpha = None if self.rng.random() < 0.5 else 2.0
        return (slot, *_new_pair(self.pipe, self.rng, slot), alpha, None)

    def _own_row(self, copy: bool) -> tuple:
        i = int(self.rng.integers(len(self.own)))
        slot, d, u = self.own[i]
        if copy:
            # An images API read of a copy of the entry's file with no
            # header: the same content at full bytes under alpha None.
            d, u = (mx.array(np.array(a.astype(mx.float32))).astype(a.dtype)
                    for a in (d, u))
            mx.eval(d, u)
            return slot, d, u, None, None
        return slot, d, u, None, self.own_keys[i]

    def register(self) -> None:
        row = self._fresh()
        if self._put(row) is not None:
            self.pool.append(row)

    def twin(self) -> None:
        row = self._own_row(copy=False)
        if self._put(row) is not None:
            self.pool.append(row)

    def copy(self) -> None:
        row = self._own_row(copy=True)
        if self._put(row) is not None:
            self.pool.append(row)

    def define(self) -> None:
        # A step route's define: every pair of the body registered, the
        # earlier ones held for the later, then the table.
        rows = []
        for _ in range(int(self.rng.integers(1, 4))):
            pick = self.rng.random()
            if pick < 0.4 or not self.pool:
                rows.append(self._fresh())
            elif pick < 0.8:
                rows.append(self.pool[self.rng.integers(len(self.pool))])
            else:
                rows.append(self._own_row(copy=self.rng.random() < 0.3))
        if self.rng.random() < 0.3:
            i = int(self.rng.integers(len(self.own)))
            rows.append((self.own[i][0], *self.own[i][1:], OWN_ALPHA, None))
        keys: list = []
        entries: dict = {}
        for row in rows:
            key = self._put(row, keys)
            if key is None:
                return
            keys.append(key)
            patches = entries.setdefault(row[0], [])
            if key not in [k for k, _s in patches]:
                patches.append((key, float(self.rng.choice([1.0, 0.5,
                                                            -0.7]))))
        self.pool += [r for r in rows if not any(r is q for q in self.pool)]
        self.tids.append(api.define_patch_table(
            self.pipe, [(p, b, patches) for (p, b), patches in
                        entries.items()]))

    def step(self) -> None:
        pipe = self.pipe
        self.tids = [t for t in self.tids if t in pipe.patch_tables]
        if not self.tids:
            return
        tid = self.tids[self.rng.integers(len(self.tids))]
        dtype = mx.bfloat16 if self.rng.random() < 0.7 else mx.float32
        rows = int(self.rng.choice([2, 64]))
        paths = sorted({p for p, _b, _s in pipe.patch_tables[tid]["entries"]})
        prev = install.activate(pipe, tid)
        try:
            for path in paths:
                inp = install.linear_dims(_leaf(pipe, path))[1]
                x = mx.array(self.rng.standard_normal(
                    (1, rows, inp)).astype(np.float32)).astype(dtype)
                mx.eval(_leaf(pipe, path)(x))
        except lora.AdapterError:
            # The step route's 409: a pair of the table is no longer held.
            assert any(pipe.adapters.peek(k) is None
                       for _p, _b, patches in pipe.patch_tables[tid]["entries"]
                       for k, _s in patches)
            self.missing += 1
        finally:
            install.restore(pipe, prev)

    def shrink(self) -> None:
        which = "adapters" if self.rng.random() < 0.5 else "tables"
        frac = float(self.rng.choice([0.0, 0.125, 0.25, 0.5, 1.0, 1.0]))
        cache = getattr(self.pipe, which)
        cache.max_bytes = int(self.bounds[which] * frac)
        install.shed_to_bounds(self.pipe)

    def drop_dit(self) -> None:
        from types import SimpleNamespace

        from gmlx.serve import image_service, video_service

        pipe, dit, vae = self.pipe, self.pipe.dit, self.pipe.vae
        served = SimpleNamespace(pipe=pipe, adapters=[])
        if self.rng.random() < 0.5:
            # A video entry's encode: the DiT and every pair go, and its
            # ensure_dit reads the entry's file again.
            video_service.drop_dit(served)
            pipe.dit = dit
            install.rewrap(pipe)
            self.load_own()
        else:
            # A staged image entry's encode: the DiT goes, the pairs stay.
            image_service._drop_dit(served)
            pipe.dit, pipe.vae = dit, vae
            install.rewrap(pipe)

    def release(self) -> None:
        from types import SimpleNamespace

        from gmlx.serve import image_service

        # The pool lets the entry go, and loads it again.
        image_service._release([SimpleNamespace(pipe=self.pipe)])
        self.load_own()

    OPS = (("register", 4), ("twin", 1), ("copy", 1), ("define", 5),
           ("step", 6), ("shrink", 2), ("drop_dit", 1), ("release", 1))


# The pair cache's order of use, modelled apart from make_room: at each
# turn the oldest of the unpinned pairs, by last use, and of the held
# tables that can go, by last step, goes until the bytes fit. A pair stays
# pinned while any held table names it.

def _pair_room_inputs(pipe, hold) -> tuple:
    """What the pair branch of make_room reads, without a use: each pair
    with its bytes and last use, the keep set with ``hold``, and each held
    table with its keys, last step and whether a forward runs it."""
    ad, tb = pipe.adapters, pipe.tables
    pairs = [(k, ad.nbytes(k), ad.used(k)) for k in ad.keys(lambda k: True)]
    steps = {id(tb.peek(k)): tb.used(k)
             for k in tb.keys(lambda k: k[0] != install.KERNEL)}
    running = install._running(pipe)
    tables = [(id(t), t.keys, steps.get(id(t), math.inf), id(t) in running)
              for t in install._tables(pipe)]
    return pairs, set(ad.kept) | set(hold), tables, ad.bytes, ad.max_bytes


def _one_order(inputs, nbytes: int) -> tuple:
    """The pairs shed, the tables dropped and whether ``nbytes`` fit, in
    one order of use."""
    pairs, base, tables, total, bound = inputs
    if total + nbytes <= bound:
        return set(), set(), True
    base = set(base).union(*(keys for _i, keys, _s, run in tables if run))
    if nbytes and total - sum(b for k, b, _u in pairs if k not in base) \
            + nbytes > bound:
        return set(), set(), False
    left = {k: (b, u) for k, b, u in pairs}
    live = {i: (keys, step) for i, keys, step, run in tables if not run}
    shed, dropped = set(), set()
    while total + nbytes > bound:
        pinned = base.union(*(keys for keys, _s in live.values()))
        turns = [(u, 0, k) for k, (_b, u) in left.items() if k not in pinned]
        turns += [(step, 1, i) for i, (keys, step) in live.items()
                  if not keys <= base]
        if not turns:
            break
        _use, kind, x = min(turns, key=lambda t: t[:2])
        if kind == 0:
            total -= left.pop(x)[0]
            shed.add(x)
        else:
            del live[x]
            dropped.add(x)
    return shed, dropped, total + nbytes <= bound


class _OrderCheck:
    """make_room with each call for the pair cache checked against
    ``_one_order``. ``puts`` counts the puts, of more than 0 bytes, that
    shed or dropped, ``drops`` the puts that dropped a table, and
    ``bounds`` the sheds to a bound, of 0 bytes, that shed or dropped."""

    def __init__(self, monkeypatch):
        self.make_room = install.make_room
        self.puts = self.drops = self.bounds = 0
        monkeypatch.setattr(install, "make_room", self)

    def __call__(self, pipe, cache, nbytes, hold=()):
        if cache is not pipe.adapters:
            return self.make_room(pipe, cache, nbytes, hold)
        hold = list(hold)
        inputs = _pair_room_inputs(pipe, hold)
        want = _one_order(inputs, nbytes)
        fit = self.make_room(pipe, cache, nbytes, hold)
        got = ({k for k, _b, _u in inputs[0]} - set(cache.keys(lambda k: True)),
               {i for i, *_ in inputs[2]}
               - {id(t) for t in install._tables(pipe)}, fit)
        if nbytes:
            self.puts += bool(got[0] or got[1])
            self.drops += bool(got[1])
        else:
            self.bounds += bool(got[0] or got[1])
        assert got == want, (nbytes, sorted(k[:8] for k in got[0]),
                             sorted(k[:8] for k in want[0]), len(got[1]),
                             len(want[1]), fit, want[2])
        return fit


@pytest.mark.parametrize("seed", range(20))
def test_a_random_sequence_counts_every_lora_array_once(op, monkeypatch, seed):
    """Every array the pair cache, the tables cache and the patch state
    keep alive is counted once, each cache is over its bound only by what
    is pinned, every held table's pairs are held, and the pair cache sheds
    in one order of use, after each of a few hundred operations of the
    step routes, the images API and the entry in a seeded random order."""
    order = _OrderCheck(monkeypatch)
    seq = _Sequence(seed)
    names = [n for n, _w in _Sequence.OPS]
    weights = np.array([w for _n, w in _Sequence.OPS], np.float64)
    ran = dict.fromkeys(names, 0)
    done: list = []
    for _ in range(OPS_PER_SEED):
        name = names[seq.rng.choice(len(names), p=weights / weights.sum())]
        getattr(seq, name)()
        ran[name] += 1
        done.append(name)
        try:
            _check_caches(seq.pipe)
        except AssertionError as e:
            raise AssertionError(f"after {done[-6:]}, op {len(done)}") from e
    assert (all(ran.values()) and seq.refused and seq.missing and order.puts
            and order.bounds), (ran, seq.refused, seq.missing, order.puts,
                                order.bounds)
