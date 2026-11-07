"""Per-row sampler, seed and logprob settings: keyed rows, byte-identical
one-config batches, flattened verify draws, and the request-to-row path."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm")

from gmlx._vendor.mlx_vlm.server import generation as gen_mod  # noqa: E402
from gmlx._vendor.mlx_vlm.server.generation import (  # noqa: E402
    GenerationArguments,
    _position_keys,
)

import gmlx.spec.speculative as spec  # noqa: E402
from gmlx.serve import row_sampling as rsm  # noqa: E402
from gmlx.serve.patches.sampling import _FastPositionedSampler  # noqa: E402
from gmlx.serve.row_sampling import RowSampler, RowSettings  # noqa: E402


def _sampler(**kw):
    kw.setdefault("temperature", 1.0)
    return _FastPositionedSampler(**kw)


def _logprobs(rows=2, vocab=64, seed=0):
    x = mx.random.uniform(shape=(rows, vocab), key=mx.random.key(seed)) * 4
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


def _row(sampler=None, seed=None, key=None, **kw):
    if key is None:
        key = rsm.GREEDY_KEY if sampler is None else ("t1",)
    rs = RowSettings(sampler=sampler, seed=seed, key=key, **kw)
    return rs


def _settings(**rows):
    out = {}
    for uid, rs in rows.items():
        u = int(uid[1:])
        out[u] = rs
        rsm.register_seed(rs.sampler, u, rs.seed)
    return out


# The sampler-side keyed path.

def test_unseeded_rows_keep_stock_keys():
    s = _sampler()
    s._kq_row_seeds["other"] = 999   # registry active, rows unseeded
    s._kq_rows = ["a", "b"]
    keys = s._row_keys([0, 0], [5, 6])
    assert mx.array_equal(keys, _position_keys(s.seed, [0, 0], [5, 6]))


def test_seeded_row_gets_its_own_key():
    s = _sampler()
    s._kq_row_seeds["a"] = 123
    s._kq_rows = ["a", "b"]
    keys = s._row_keys([0, 0], [5, 5])
    stock = _position_keys(s.seed, [0, 0], [5, 5])
    assert not mx.array_equal(keys[0], stock[0])
    assert mx.array_equal(keys[1], stock[1])
    assert mx.array_equal(
        keys[0], mx.random.key(gen_mod._position_seed(123, 0, 5)))


def test_seeded_target_draw_replays():
    lp = _logprobs(rows=4)
    s = _sampler()
    s._kq_row_seeds["u"] = 42
    s._kq_rows = ["u"]
    a = spec._seeded_target_draw(s, lp, base_pos=10)
    b = spec._seeded_target_draw(s, lp, base_pos=10)
    assert mx.array_equal(a, b)
    assert s._kq_rows == ["u"]      # context restored


# RowSampler.

def test_a_one_config_batch_draws_the_stock_keys_byte_for_byte():
    lp = _logprobs(rows=3)
    a, b, c = _sampler(top_p=0.9), _sampler(top_p=0.9), _sampler(top_p=0.9)
    rows = _settings(u1=_row(a), u2=_row(b), u3=_row(c))
    rs = RowSampler(rows, _row())
    with rs.bind([1, 2, 3]):
        got = rs.sample_target(lp, row_ids=[0, 0, 0], positions=[4, 4, 4])
    ref = _sampler(top_p=0.9).sample_target(
        lp, row_ids=[0, 0, 0], positions=[4, 4, 4])
    assert mx.array_equal(got, ref)


def test_a_seeded_row_draws_its_own_stream():
    lp = _logprobs(rows=1)
    rows = _settings(u7=_row(_sampler(), seed=42))
    rs = RowSampler(rows, _row())

    def run():
        with rs.bind([7]):
            return [int(rs.sample_target(lp, row_ids=[0], positions=[p]).item())
                    for p in range(1, 12)]

    solo = _sampler()
    solo._kq_row_seeds[7] = 42
    solo._kq_rows = [7]
    ref = [int(solo.sample_target(lp, row_ids=[0], positions=[p]).item())
           for p in range(1, 12)]
    assert run() == run() == ref


def test_an_unseeded_row_beside_a_seeded_one_draws_the_stock_keys():
    lp = _logprobs(rows=2)
    rows = _settings(u7=_row(_sampler(), seed=42), u8=_row(_sampler()))
    rs = RowSampler(rows, _row())
    with rs.bind([7, 8]):
        out = rs.sample_target(lp, row_ids=[0, 0], positions=[3, 3])
    ref = _sampler().sample_target(lp, row_ids=[0, 0], positions=[3, 3])
    assert int(out[1].item()) == int(ref[1].item())
    # The seeded row's instance keeps only its own seed after the draw.
    assert rows[8].sampler._kq_row_seeds == {}
    assert rows[7].sampler._kq_row_seeds == {7: 42}


def test_a_greedy_row_in_a_sampled_batch_takes_its_argmax():
    lp = _logprobs(rows=2, seed=3)
    rows = _settings(u1=_row(), u2=_row(_sampler(temperature=5.0)))
    rs = RowSampler(rows, _row())
    for pos in range(8):
        with rs.bind([1, 2]):
            out = rs.sample_target(lp, row_ids=[0, 0], positions=[pos, pos])
        assert int(out[0].item()) == int(mx.argmax(lp[0]).item())


def test_a_flattened_draw_maps_rows_to_uids():
    lp = _logprobs(rows=6, seed=4)
    rows = _settings(u1=_row(_sampler(temperature=5.0)), u2=_row())
    rs = RowSampler(rows, _row())
    with rs.bind([1, 2]):
        out = rs(lp)
    want = mx.argmax(lp[3:], axis=-1)
    assert mx.array_equal(out[3:], want)


def test_a_draw_of_the_wrong_row_count_raises():
    rs = RowSampler({}, _row())
    with rs.bind([1, 2]), pytest.raises(RuntimeError, match="bound to 2 rows, draw has 3"):
        rs(_logprobs(rows=3))


def test_effective_names_the_cheapest_sampler():
    s = _sampler()
    rows = _settings(u1=_row(), u2=_row(s), u3=_row(_sampler(), seed=5))
    rs = RowSampler(rows, _row())
    with rs.bind([1]):
        assert rs.effective() is None
    with rs.bind([2]):
        assert rs.effective() is s
    with rs.bind([1, 2]):
        assert rs.effective() is rs
    with rs.bind([3]):
        assert rs.effective() is rs


def test_unknown_and_none_uids_take_the_default_row():
    default = _row(logprobs=True, top_k=3)
    rs = RowSampler({}, default)
    assert rs.row(None) is default and rs.row(99) is default


def test_prune_keeps_live_uids(monkeypatch):
    monkeypatch.setattr(rsm, "MAX_ROWS", 3)
    store = {u: _row() for u in range(6)}
    gen = SimpleNamespace(
        _unprocessed_sequences=[(0, [1], 1, {}, None, None)],
        _prompt_batch=SimpleNamespace(uids=[1]),
        _generation_batch=SimpleNamespace(
            uids=[2], _all_uids=[2, 3],
            _pending_injections=[SimpleNamespace(_all_uids=[4])]))
    rsm.prune(store, gen)
    assert sorted(store) == [0, 1, 2, 3, 4]


# Request to row.

class _RG:
    top_logprobs_k = 5
    _make_sampler = gen_mod.ResponseGenerator._make_sampler
    _row_settings = gen_mod.ResponseGenerator._row_settings


@pytest.fixture
def fast_sampler(monkeypatch):
    from gmlx.serve.patches import sampling

    def make(self, args):
        if args.temperature == 0:
            return None
        return _FastPositionedSampler(
            temperature=args.temperature, top_p=args.top_p,
            seed=getattr(args, "seed", None))

    monkeypatch.setattr(_RG, "_make_sampler", make)
    return sampling


def test_a_seeded_request_builds_an_unseeded_sampler(fast_sampler):
    rs = _RG()._row_settings(GenerationArguments(temperature=0.7, seed=42))
    from gmlx._vendor.mlx_vlm.server.generation import DEFAULT_SEED
    assert rs.seed == 42 and rs.sampler.seed == DEFAULT_SEED
    assert rs.key == (0.7, 1.0, 0, 0.0, 0.0, False, 1.0)


def test_a_greedy_request_drops_its_seed(fast_sampler):
    rs = _RG()._row_settings(GenerationArguments(temperature=0.0, seed=42))
    assert rs.sampler is None and rs.seed is None and rs.key == rsm.GREEDY_KEY


def test_top_k_is_capped_and_needs_logprobs(fast_sampler):
    rg = _RG()
    on = rg._row_settings(GenerationArguments(logprobs=True, top_logprobs=50))
    assert on.logprobs and on.top_k == 5
    # A route that clears logprobs after building the arguments.
    args = GenerationArguments(logprobs=True, top_logprobs=3)
    args.logprobs = False
    off = rg._row_settings(args)
    assert not off.logprobs and off.top_k == 0


def test_a_none_cap_reads_as_zero(fast_sampler):
    rg = _RG()
    rg.top_logprobs_k = None
    rs = rg._row_settings(GenerationArguments(logprobs=True, top_logprobs=3))
    assert rs.top_k == 0


def test_a_sampler_without_row_keys_keeps_the_seed_in_its_key(monkeypatch):
    def make(self, args):
        return SimpleNamespace(seed=getattr(args, "seed", None))

    monkeypatch.setattr(_RG, "_make_sampler", make)
    rs = _RG()._row_settings(GenerationArguments(temperature=0.7, seed=9))
    assert rs.sampler.seed == 9 and rs.seed is None
    assert rs.key[-1] == ("seed", 9)
