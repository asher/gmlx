"""Per-request seed: keyed rows, byte-identical unseeded rows, plumbing."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

pytest.importorskip("mlx_vlm")

from mlx_vlm.generate import ar  # noqa: E402
from mlx_vlm.server import generation as gen_mod  # noqa: E402
from mlx_vlm.server.generation import _position_keys  # noqa: E402

import gmlx.seed_rows as sr  # noqa: E402
import gmlx.speculative as spec  # noqa: E402
from gmlx.server_patches.sampling import _FastPositionedSampler  # noqa: E402


def _sampler(**kw):
    kw.setdefault("temperature", 1.0)
    return _FastPositionedSampler(**kw)


def _logprobs(rows=2, vocab=64, seed=0):
    x = mx.random.uniform(shape=(rows, vocab), key=mx.random.key(seed))
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


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


def test_rows_context_mismatch_falls_back():
    s = _sampler()
    s._kq_row_seeds["a"] = 123
    s._kq_rows = ["a"]              # 1 row context, 2-row draw
    keys = s._row_keys([0, 0], [5, 6])
    assert mx.array_equal(keys, _position_keys(s.seed, [0, 0], [5, 6]))


def test_solo_replay_is_exact():
    lp = _logprobs(rows=1)

    def run():
        s = _sampler(top_p=0.9)
        s._kq_row_seeds[7] = 42
        s._kq_rows = [7]
        return [int(s.sample_target(lp, row_ids=[0], positions=[p]).item())
                for p in range(1, 12)]

    assert run() == run()


def test_mixed_batch_unseeded_row_matches_solo_stock():
    lp = _logprobs(rows=2)
    mixed = _sampler()
    mixed._kq_row_seeds[7] = 42
    mixed._kq_rows = [7, 8]
    out = mixed.sample_target(lp, row_ids=[0, 0], positions=[3, 3])
    stock = _sampler()
    ref = stock.sample_target(lp, row_ids=[0, 0], positions=[3, 3])
    assert int(out[1].item()) == int(ref[1].item())


def test_registry_caps():
    s = _sampler()
    for i in range(sr._MAX_SEEDS + 10):
        sr.register_row_seed(s, i, i)
    assert len(s._kq_row_seeds) == sr._MAX_SEEDS
    assert 0 not in s._kq_row_seeds
    assert sr._MAX_SEEDS + 9 in s._kq_row_seeds


def test_seeded_target_draw_replays():
    lp = _logprobs(rows=4)
    s = _sampler()
    s._kq_row_seeds["u"] = 42
    s._kq_rows = ["u"]
    a = spec._seeded_target_draw(s, lp, base_pos=10)
    b = spec._seeded_target_draw(s, lp, base_pos=10)
    assert mx.array_equal(a, b)
    assert s._kq_rows == ["u"]      # context restored


def test_seeded_target_draw_unseeded_uses_process_stream():
    lp = _logprobs(rows=4)
    calls = []
    orig_call = _FastPositionedSampler.__call__

    class Probe(_FastPositionedSampler):
        def __call__(self, logprobs):
            calls.append(1)
            return orig_call(self, logprobs)

    p = Probe(temperature=1.0)
    p._kq_rows = ["u"]              # no seed registered
    spec._seeded_target_draw(p, lp, base_pos=10)
    assert calls == [1]


@pytest.fixture
def installed(monkeypatch):
    saved = (gen_mod.ResponseGenerator._make_thinking_budget_criteria,
             ar.BatchGenerator.insert, ar.GenerationBatch._step,
             ar.PromptProcessingBatch.generate,
             ar.SpeculativeGenerationBatch.next)
    monkeypatch.setattr(
        gen_mod.ResponseGenerator, "_make_thinking_budget_criteria",
        lambda self, args, input_ids: None)
    monkeypatch.setattr(
        ar.BatchGenerator, "insert",
        lambda self, prompts, **kw: list(range(100, 100 + len(prompts))))
    monkeypatch.setattr(ar.GenerationBatch, "_step",
                        lambda self: self.sampler._kq_rows)
    monkeypatch.setattr(ar.PromptProcessingBatch, "generate",
                        lambda self, sampler, *a, **k: sampler._kq_rows)
    monkeypatch.setattr(ar.SpeculativeGenerationBatch, "next",
                        lambda self: self.sampler._kq_rows)
    sr._PENDING.clear()
    sr.install_per_request_seed()
    yield
    (gen_mod.ResponseGenerator._make_thinking_budget_criteria,
     ar.BatchGenerator.insert, ar.GenerationBatch._step,
     ar.PromptProcessingBatch.generate,
     ar.SpeculativeGenerationBatch.next) = saved


def test_insert_registers_the_request_seed(installed):
    s = _sampler()
    rg = SimpleNamespace()
    gen_mod.ResponseGenerator._make_thinking_budget_criteria(
        rg, SimpleNamespace(seed=42, temperature=1.0), None)
    bg = SimpleNamespace(sampler=s)
    uids = ar.BatchGenerator.insert(bg, [[1, 2, 3]])
    assert s._kq_row_seeds == {uids[0]: 42}


def test_greedy_request_seed_is_ignored(installed):
    s = _sampler()
    rg = SimpleNamespace()
    gen_mod.ResponseGenerator._make_thinking_budget_criteria(
        rg, SimpleNamespace(seed=42, temperature=0), None)
    bg = SimpleNamespace(sampler=s)
    ar.BatchGenerator.insert(bg, [[1, 2, 3]])
    assert s._kq_row_seeds == {}


def test_unseeded_insert_registers_nothing(installed):
    s = _sampler()
    rg = SimpleNamespace()
    gen_mod.ResponseGenerator._make_thinking_budget_criteria(
        rg, SimpleNamespace(seed=None, temperature=1.0), None)
    bg = SimpleNamespace(sampler=s)
    ar.BatchGenerator.insert(bg, [[1, 2, 3]])
    assert s._kq_row_seeds == {}


def test_step_publishes_uids_only_when_seeds_exist(installed):
    s = _sampler()
    gb = SimpleNamespace(sampler=s, uids=[1, 2])
    assert ar.GenerationBatch._step(gb) is None   # no seeds: no context
    s._kq_row_seeds[1] = 9
    assert ar.GenerationBatch._step(gb) == [1, 2]
    assert s._kq_rows is None                     # cleared after the step


def test_prompt_generate_and_spec_next_publish_uids(installed):
    s = _sampler()
    s._kq_row_seeds[1] = 9
    pb = SimpleNamespace(uids=[1])
    assert ar.PromptProcessingBatch.generate(pb, s) == [1]
    sb = SimpleNamespace(sampler=s, _all_uids=[1, 3])
    assert ar.SpeculativeGenerationBatch.next(sb) == [1, 3]
    assert s._kq_rows is None
