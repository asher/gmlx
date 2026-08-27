"""Serve memory trace: record builder and install gating."""

import json

import mlx.core as mx

from mlx_vlm.generate import ar

import gmlx.serve.memtrace as smt


class FakeKV:
    def __init__(self, rows=1, length=256, offset=100):
        self.keys = mx.zeros((rows, 4, length, 8), dtype=mx.float16)
        self.values = mx.zeros((rows, 4, length, 8), dtype=mx.float16)
        self.offset = offset


class FakePool:
    def __init__(self, length=64):
        self.pools = [mx.zeros((1, 2, length, 8)), mx.zeros((1, 2, length, 8))]
        self.offset = length


class FakeList:
    def __init__(self, caches):
        self.caches = caches


class FakeBatch:
    def __init__(self, prompt_cache, uids=(1, 2)):
        self.prompt_cache = prompt_cache
        self.uids = list(uids)
        self._num_tokens = [3] * len(self.uids)

    def __len__(self):
        return len(self.uids)


class FakeGen:
    def __init__(self, prompt_cache):
        self._generation_batch = FakeBatch(prompt_cache)
        self._prompt_batch = None
        self._unprocessed_sequences = [(7, [1, 2, 3], 64, {}, None, None)]
        self._prompt_time_counter = 0.25


def _kv_bytes(c):
    return c.keys.nbytes + c.values.nbytes


def test_cache_report_bytes_kinds_and_flattening():
    kv, pool = FakeKV(), FakePool()
    total, kinds, sig, shapes = smt._cache_report([FakeList([kv, pool])])
    assert total == _kv_bytes(kv) + sum(a.nbytes for a in pool.pools)
    assert kinds["FakeKV"]["n"] == 1
    assert kinds["FakeKV"]["off"] == [100, 100]
    assert kinds["FakeKV"]["alen"] == [256, 256]
    assert kinds["FakePool"]["bytes"] == sum(a.nbytes for a in pool.pools)
    assert len(shapes) == 2 and shapes[0]["kind"] == "FakeKV"


def test_record_marks_shape_changes_once():
    gen = FakeGen([FakeKV(length=256)])
    first = smt._record(gen, 0.001)
    assert "gen" in first["shapes"]
    assert first["rows"] == 2 and first["pend"] == [7]
    second = smt._record(gen, 0.001)
    assert "shapes" not in second
    # block growth: allocation length changes, shapes dump again
    gen._generation_batch.prompt_cache = [FakeKV(length=512)]
    third = smt._record(gen, 0.001)
    assert "gen" in third["shapes"]
    assert third["own"]["gen"]["kinds"]["FakeKV"]["alen"] == [512, 512]
    assert third["tick"] == 3


def test_record_prompt_batch_and_spec_attrs():
    gen = FakeGen([FakeKV()])
    pb = FakeBatch([FakeKV(offset=10)], uids=(9,))
    pb._processed_prompt_columns = 128
    pb._total_prompt_tokens = 4300
    gen._prompt_batch = pb
    gen._generation_batch.hidden = mx.zeros((1, 16, 8))
    gen._generation_batch.shared_kv_states = [mx.zeros((2, 4))]
    rec = smt._record(gen, 0.002)
    assert rec["pb"] == {"uids": [9], "done": 128, "total": 4300}
    assert rec["own"]["pb"]["bytes"] == _kv_bytes(pb.prompt_cache[0])
    assert rec["own"]["spec"]["hidden"] == 16 * 8 * 4
    assert rec["own"]["spec"]["shared_kv_states"] == 2 * 4 * 4


def test_install_off_without_env(monkeypatch):
    monkeypatch.delenv("GMLX_SERVE_MEMSTATS", raising=False)
    orig = ar.BatchGenerator._next
    assert smt.install_serve_memtrace() is False
    assert ar.BatchGenerator._next is orig


def test_install_traces_ticks_to_jsonl(monkeypatch, tmp_path):
    out = tmp_path / "trace.jsonl"
    monkeypatch.setenv("GMLX_SERVE_MEMSTATS", str(out))

    def _fake_next(self, **kw):
        return [], []

    monkeypatch.setattr(ar.BatchGenerator, "_next", _fake_next)
    try:
        assert smt.install_serve_memtrace() is True
        assert smt.install_serve_memtrace() is True  # idempotent
        gen = FakeGen([FakeKV()])
        ar.BatchGenerator._next(gen)
        ar.BatchGenerator._next(gen)
        lines = [json.loads(x) for x in out.read_text().splitlines()]
    finally:
        smt._writer.close()
        smt._writer = None
    assert "meta" in lines[0]
    assert [x["tick"] for x in lines[1:]] == [1, 2]
    assert lines[1]["own"]["gen"]["bytes"] > 0
    assert "shapes" in lines[1] and "shapes" not in lines[2]
