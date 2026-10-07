"""The ``lora`` route of the image ops: when the product of a K-quant leaf
and the low-rank term of its adapter run as one mlx-kquant op, what the
op is handed, and its result against a float64 reference. The decisions
run on a recording op with the device probe stood in, so they hold on any
mlx-kquant and on any default device: the op is the build's on a GPU with
tensor ops and a stand-in on the stock ops elsewhere. Every call the
route does not take is held bit for bit to the path with no route."""
from __future__ import annotations

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
    order, table = plan.band_table(
        [plan.Part(mx.array(d), mx.array(u), a, band, s)
         for band, (d, u, a), s in rows], (3 * DIM, DIM))
    assert table is None and order == list(range(len(rows)))
    assert rel_l2(_f64(y), _f64(_today(pipe, entries, QKV, x))) <= 2 * TOL


def test_the_band_table_orders_parts_by_band_and_joins_ranks():
    def part(band, rank):
        return plan.Part(mx.zeros((rank, DIM)), mx.zeros((DIM, rank)), None,
                         band, 1.0)

    dims = (3 * DIM, DIM)
    parts = [part(BANDS[2], 2), part(BANDS[0], 4), part(BANDS[2], 3),
             part(BANDS[1], 1)]
    order, table = plan.band_table(parts, dims)
    assert order == [1, 3, 0, 2]
    assert table == [(0, 256, 0, 4), (256, 512, 4, 1), (512, 768, 5, 5)]
    assert plan.band_table([part(None, 4), part(None, 2)], dims) == (
        [0, 1], None)
    # A band over the whole output beside another is an overlap.
    assert plan.band_table([part((0, 0, 3 * DIM), 4), part(BANDS[1], 2)],
                           dims)[1] is None
    assert plan.band_table([part((0, 128, 256), 4), part(BANDS[1], 2)],
                           dims)[1] is None


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
        assert kq_routes.install(kq_routes.ROUTES) <= {"qk", "glu"}
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
