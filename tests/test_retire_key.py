"""Next-turn retirement keys: memo hops, LCP prediction, store decisions."""

import mlx.core as mx
import pytest
from mlx_vlm.models.cache import KVCache, RotatingKVCache

from gmlx import cache_snapshot as cs
from gmlx import retire_key as rk
from gmlx import speculative as sp


@pytest.fixture(autouse=True)
def _clean_memos():
    rk._reset_for_tests()
    yield
    rk._reset_for_tests()


def _fake_ctx(next_ids, media=False, decoded="hello"):
    class Tok:
        eos_token = "<eos>"

        def decode(self, ids, skip_special_tokens=False):
            return decoded

    class Proc:
        tokenizer = Tok()

    return {
        "messages": [{"role": "user", "content": "hi"}],
        "kw": {},
        "processor": Proc(),
        "config": {},
        "render": lambda p, c, msgs, **kw: "RENDER",
        "preprocess": lambda text: {"input_ids": [list(next_ids)]},
        "media": media,
    }


def _kv(tokens, heads=2, dim=4):
    c = KVCache()
    c.update_and_fetch(mx.zeros((1, heads, tokens, dim)),
                       mx.zeros((1, heads, tokens, dim)))
    return c


class _Manager:
    def __init__(self):
        self.exact = []

    def store_exact_cache(self, ids, snap, extra_hash=0):
        self.exact.append((list(ids), snap))
        return True

    def release(self, blocks):
        pass


# -- memo hops --

def test_memo_hop_and_lookup():
    rk.register_render("T", {"messages": [], "kw": {}})
    pre = object()
    rk.register_ids("T", [1, 2, 3], pre)
    ctx = rk.lookup_render_ctx([1, 2, 3])
    assert ctx is not None and ctx["preprocess"] is pre
    assert rk.lookup_render_ctx([1, 2]) is None


def test_ids_hop_requires_registered_text():
    rk.register_ids("never-rendered", [7, 8], object())
    assert rk.lookup_render_ctx([7, 8]) is None


def test_memo_caps():
    for i in range(rk._TEXT_MEMO_CAP + 4):
        rk.register_render(f"t{i}", {})
    assert len(rk._TEXT_MEMO) == rk._TEXT_MEMO_CAP
    rk.register_render("k", {})
    for i in range(rk._IDS_MEMO_CAP + 4):
        rk.register_ids("k", [i], object())
    assert len(rk._IDS_MEMO) == rk._IDS_MEMO_CAP


# -- LCP prediction --

def test_lcp_faithful_render_covers_sequence():
    seq = [1, 2, 3, 4]
    ctx = _fake_ctx(next_ids=[1, 2, 3, 4, 9])
    assert rk.next_turn_lcp(ctx, seq, [4]) == 4


def test_lcp_divergence_point():
    seq = [1, 2, 3, 4]
    ctx = _fake_ctx(next_ids=[1, 2, 9, 9])
    assert rk.next_turn_lcp(ctx, seq, [4]) == 2


def test_lcp_media_and_missing_ctx_return_none():
    assert rk.next_turn_lcp(None, [1], [1]) is None
    assert rk.next_turn_lcp(_fake_ctx([1], media=True), [1], [1]) is None


def test_lcp_prediction_failure_returns_none():
    ctx = _fake_ctx([1, 2])
    ctx["render"] = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError())
    assert rk.next_turn_lcp(ctx, [1, 2], [2]) is None


def test_build_assistant_message_thinking_split():
    ctx = _fake_ctx([1])
    msg = rk.build_assistant_message(
        ctx, "<think>\nplan\n</think>\n\nanswer")
    assert msg["role"] == "assistant"
    assert msg["content"].strip() == "answer"
    assert "plan" in (msg.get("reasoning_content") or "")


def test_predict_render_gets_generation_prompt_off():
    seen = {}

    def render(p, c, msgs, **kw):
        seen.update(kw)
        seen["last"] = msgs[-1]
        return "X"

    ctx = _fake_ctx([1, 2, 3])
    ctx["render"] = render
    rk.predict_next_ids(ctx, {"role": "assistant", "content": "a"})
    assert seen["add_generation_prompt"] is False
    assert seen["last"]["role"] == "assistant"


# -- retirement_store max_len mechanics --

def test_block_store_truncates_ids(monkeypatch):
    seen = {}

    def harvest(manager, cache, row, ids, extra_hash=0):
        seen["ids"] = list(ids)
        return ["b"]

    import mlx_vlm.apc as apc
    monkeypatch.setattr(apc, "harvest_blocks_from_batch_cache", harvest)
    ok = cs.retirement_store(_Manager(), "block", [1, 2, 3, 4, 5], [_kv(5)],
                             max_len=3)
    assert ok and seen["ids"] == [1, 2, 3]


def test_exact_store_truncates_all_kv_snapshot():
    m = _Manager()
    ok = cs.retirement_store(m, "exact", [1, 2, 3, 4, 5, 6],
                             [_kv(6), _kv(6)], max_len=4)
    assert ok
    ids, snap = m.exact[0]
    assert ids == [1, 2, 3, 4]
    assert all(int(c.offset) == 4 and c.keys.shape[2] == 4 for c in snap)


def test_exact_store_untruncated_without_max_len():
    m = _Manager()
    ok = cs.retirement_store(m, "exact", [1, 2, 3, 4, 5, 6], [_kv(6)])
    assert ok and m.exact[0][0] == [1, 2, 3, 4, 5, 6]


def test_exact_store_skips_rotating_truncation():
    m = _Manager()
    rot = RotatingKVCache(max_size=4)
    rot.update_and_fetch(mx.zeros((1, 2, 6, 4)), mx.zeros((1, 2, 6, 4)))
    ok = cs.retirement_store(m, "exact", [1, 2, 3, 4, 5, 6],
                             [_kv(6), rot], max_len=4)
    assert not ok and not m.exact


def test_ckpt_store_skipped_on_truncation(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("ckpt_store must not run on a truncated key")

    monkeypatch.setattr(cs, "ckpt_store", boom)
    ok = cs.retirement_store(_Manager(), "ckpt", [1, 2, 3, 4], [_kv(4)],
                             max_len=3)
    assert not ok


# -- _retire_b1 decision wiring --

def _retire_setup(monkeypatch, lcp, mode="exact"):
    calls = {}

    def fake_store(manager, m, seq, cache, row=0, extra_hash=0,
                   max_len=None):
        calls["store"] = (list(seq), max_len, m)
        return True

    monkeypatch.setattr(cs, "retirement_store", fake_store)
    monkeypatch.setattr(rk, "next_turn_lcp",
                        lambda ctx, seq, gen: lcp)

    class Model:
        _kq_apc_manager = _Manager()

    full = [1, 2, 3]
    gen = [4, 5]
    cache = [_kv(5)]
    retire = {"full_ids": full, "mode": mode, "extra_hash": 0,
              "render_ctx": {"messages": []}}
    return Model(), cache, gen, retire, calls


def test_retire_b1_full_match_stores_whole(monkeypatch):
    model, cache, gen, retire, calls = _retire_setup(monkeypatch, lcp=5)
    sp._retire_b1(model, cache, gen, retire)
    seq, max_len, _ = calls["store"]
    assert seq == [1, 2, 3, 4, 5] and max_len is None


def test_retire_b1_divergence_passes_max_len(monkeypatch):
    model, cache, gen, retire, calls = _retire_setup(monkeypatch, lcp=2)
    sp._retire_b1(model, cache, gen, retire)
    assert calls["store"][1] == 2


def test_retire_b1_truncation_skips_drafter_sidecar(monkeypatch):
    model, cache, gen, retire, calls = _retire_setup(monkeypatch, lcp=2)

    def boom(*a, **kw):
        raise AssertionError("sidecar must not store under a truncated key")

    monkeypatch.setattr(cs, "drafter_sidecar_store", boom)

    class Drafter:
        _kq_head_covered = True
        _kq_head_request = None

    sp._retire_b1(model, cache, gen, retire, drafter=Drafter())
    assert calls["store"][1] == 2


def test_retire_b1_kill_switch(monkeypatch):
    model, cache, gen, retire, calls = _retire_setup(monkeypatch, lcp=2)
    monkeypatch.setenv("GMLX_APC_RETIRE_LCP", "0")

    def boom(*a, **kw):
        raise AssertionError("lcp must not run with the kill switch set")

    monkeypatch.setattr(rk, "next_turn_lcp", boom)
    sp._retire_b1(model, cache, gen, retire)
    assert calls["store"][1] is None


def test_retire_b1_no_ctx_keeps_today(monkeypatch):
    model, cache, gen, retire, calls = _retire_setup(monkeypatch, lcp=None)
    retire["render_ctx"] = None
    sp._retire_b1(model, cache, gen, retire)
    assert calls["store"][1] is None
