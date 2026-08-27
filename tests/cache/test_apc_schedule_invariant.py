"""Schedule invariant: stores and lookups intersect for real queries.

CPU-only, runs in CI; fails on main. This is the tripwire for the defect
class the apc-overhaul branch exists for (bug 1): every store position
came from the real boundary builder, every store runs through the real
mid-prefill cursor, and the assertions replay the two queries a serve
deployment actually issues -- an identical resend and a turn-2 re-render
whose gen-prompt/think tail is replaced (bug 3's divergence shape). The
invariant in words: after a request completes, some stored record sits
strictly below every query we will actually issue, at the depth the
builder promises.
"""

from types import SimpleNamespace

import pytest
from mlx_vlm.apc import APCManager

import gmlx.spec.engine as se
import gmlx.cache.retire_key as retire_key
from gmlx.cache.snapshot import (
    _ckpt_records,
    ckpt_full_store_redundant,
    ckpt_lookup,
    ckpt_store,
    retirement_store,
)

from test_ckpt_tier import make_hybrid_cache, make_swa_cache

GDN_TAGS = ("kv", "arr", "arr", "kv", "arr")
SWA_TAGS = ("kv", "rot:32:0", "rot:32:0", "kv", "rot:32:0")
TAIL = 7                       # gen-prompt/think tail the re-render strips


def _run_request(man, tags, ids, p_stable, monkeypatch):
    """Drive one request through the real store scheduler: arm, walk the
    boundary schedule via the production cursor, post-prefill p=N with
    the production drop gate, then retirement with the re-render LCP."""
    make = make_hybrid_cache if "arr" in tags else make_swa_cache
    monkeypatch.setattr(retire_key, "lookup_render_ctx",
                        lambda i: {"stub": True})
    monkeypatch.setattr(retire_key, "prompt_stable_lcp",
                        lambda ctx, i: p_stable)
    meta = {"full_input_ids": list(ids), "prefix_len": 0, "extra_hash": 0}
    batch = SimpleNamespace(
        prefill_step_size=2048, _kq_ckpt_armed=True, _apc_manager=man,
        _apc_meta=[meta], prompt_cache=None,
        model=SimpleNamespace(_kq_apc_ckpt_layout=tags),
        _row_real_tokens_processed=lambda i: 0)
    se._ckpt_arm_schedule(batch, meta, len(ids), 0, 16)
    for pos, _kind in list(meta["ckpt_boundaries"]):
        batch.prompt_cache = make(pos, seed=pos % 977)
        batch._row_real_tokens_processed = lambda i, b=pos: b
        se._ckpt_mid_prefill_store(batch)
    assert meta.get("checkpoint_done", not meta["ckpt_boundaries"]) or \
        meta.get("ckpt_boundaries") == []
    if not ckpt_full_store_redundant(meta):
        if ckpt_store(man, ids, make(len(ids), seed=11)):
            meta.setdefault("ckpt_stored_boundaries", []).append(len(ids))
    seq = list(ids) + [9000 + i for i in range(6)]
    retirement_store(man, "ckpt", seq, make(len(seq), seed=13),
                     max_len=p_stable, decode_snaps=None)
    return meta


def _expected(man, p_stable, query_len):
    """The floor the builder promises: the deepest live record strictly
    below the query that a re-rendered turn can still prefix-match."""
    live = [r.p for r in _ckpt_records(man).values()
            if r.p <= p_stable and r.p < query_len]
    return max(live, default=0)


@pytest.mark.parametrize("tags,n,resend,turn2", [
    # GDN: replay floor (GMLX_APC_CKPT_REPLAY_MIN=1024) and the 2048
    # chunk grid bound what short prompts can keep; both misses below
    # are the documented cost of not churning >100 MB records.
    (GDN_TAGS, 300, 0, 0),
    (GDN_TAGS, 1500, 1499, 0),
    (GDN_TAGS, 5000, 4999, "grid"),
    # SWA: window clones are cheap, so replay has no byte floor and the
    # exact off-grid turn boundary keeps the full stable prefix.
    (SWA_TAGS, 300, 299, "exact"),
    (SWA_TAGS, 1500, 1499, "exact"),
    (SWA_TAGS, 5000, 4999, "exact"),
], ids=lambda v: str(v) if not isinstance(v, tuple) else (
    "gdn" if "arr" in v else "swa"))
def test_store_lookup_intersection(tags, n, resend, turn2, monkeypatch):
    man = APCManager(num_blocks=1024, block_size=16)
    stable = list(range(1000, 1000 + n - TAIL))
    ids = stable + list(range(70, 70 + TAIL))
    p_stable = len(stable)
    meta = _run_request(man, tags, ids, p_stable, monkeypatch)

    # (a) identical resend: the client sends the same prompt again.
    warm, got = ckpt_lookup(man, ids, extra_hash=0)
    assert got == resend, f"resend adopted {got}, promised {resend}"
    if resend:
        assert warm is not None

    # (b) turn 2: same conversation re-rendered -- the tail is replaced
    # by the assistant message and a new user message follows.
    turn2_ids = stable + list(range(500, 500 + TAIL + 30))
    floor = {"grid": (p_stable // 2048) * 2048,
             "exact": p_stable}.get(turn2, turn2)
    warm, got = ckpt_lookup(man, turn2_ids, extra_hash=0)
    assert got == _expected(man, p_stable, len(turn2_ids)) == floor, (
        f"turn-2 adopted {got}, builder promises {floor}")
    if floor:
        assert warm is not None

    # (c) the p=N record only exists when no render-stable boundary
    # landed (the drop gate) -- adjacent N-1/N records are what
    # strip-on-extend fought over.
    stored = meta.get("ckpt_stored_boundaries") or []
    assert (len(ids) in stored) == (not ckpt_full_store_redundant(meta))
