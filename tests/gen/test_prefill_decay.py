"""Depth-decay prefill chunk scaling: tier math + batch resolution."""

from types import SimpleNamespace

import pytest

import gmlx.gen.prefill_decay as pd

HEADS = 32
GB = 1e9


def _cap(monkeypatch, gb):
    monkeypatch.setenv("GMLX_PREFILL_SCORE_CAP_GB", str(gb))


def _transient(step, depth, heads=HEADS):
    return heads * step * (depth + step) * 2


def test_shallow_keeps_base(monkeypatch):
    _cap(monkeypatch, 1.0)
    # 32 * 2048 * 2048 * 2 = 268 MB fits 1 GB
    assert pd.decayed_step(2048, 0, HEADS) == 2048


def test_decay_tiers_with_depth(monkeypatch):
    _cap(monkeypatch, 1.0)
    seen = []
    for depth in (0, 8_192, 32_768, 100_000, 200_000):
        step = pd.decayed_step(2048, depth, HEADS)
        seen.append(step)
        # chosen tier fits the cap unless already at the floor
        if step > 256:
            assert _transient(step, depth) <= 1.0 * GB
        # the next tier up would NOT fit (largest-fitting property)
        if step < 2048:
            assert _transient(step * 2, depth) > 1.0 * GB
    # monotone non-increasing in depth
    assert seen == sorted(seen, reverse=True)
    assert seen[0] == 2048 and seen[-1] == 256


def test_profile_rows_price_the_live_rows_not_the_chunk(monkeypatch):
    _cap(monkeypatch, 6.0)
    prof = pd.ScoreTransientProfile(heads=1, bytes_per_elem=4, rows=2048)
    # 2048 x (384000 + 8192) x 4 = 3.2 GB fits where the chunk-wide price
    # (12.8 GB) would have halved the step twice
    assert pd.decayed_step(8192, 384_000, HEADS, profile=prof) == 8192
    assert pd.score_transient_bytes.__name__ == "score_transient_bytes"
    # once the live rows alone overflow, the step halves below them
    assert pd.decayed_step(8192, 3_000_000, HEADS, profile=prof) == 256
    assert pd.decayed_step(8192, 3_000_000, HEADS, profile=prof._replace(
        rows=0)) == 256


def test_floor_holds_at_extreme_depth(monkeypatch):
    _cap(monkeypatch, 0.5)
    assert pd.decayed_step(2048, 10_000_000, HEADS) == 256


def test_explicit_small_base_passes_through(monkeypatch):
    _cap(monkeypatch, 0.5)
    # an explicit PREFILL_STEP_SIZE at/below the floor stays authoritative
    assert pd.decayed_step(128, 10_000_000, HEADS) == 128
    assert pd.decayed_step(256, 10_000_000, HEADS) == 256


def test_min_step_env(monkeypatch):
    _cap(monkeypatch, 0.5)
    monkeypatch.setenv("GMLX_PREFILL_MIN_STEP", "512")
    assert pd.decayed_step(2048, 10_000_000, HEADS) == 512


def test_idempotent(monkeypatch):
    _cap(monkeypatch, 1.0)
    for depth in (0, 50_000, 200_000):
        once = pd.decayed_step(2048, depth, HEADS)
        assert pd.decayed_step(once, depth, HEADS) == once


def test_none_and_zero_base_pass_through(monkeypatch):
    _cap(monkeypatch, 1.0)
    assert pd.decayed_step(0, 1000, HEADS) == 0
    assert pd.decayed_step(None, 1000, HEADS) is None


def test_kv_depth():
    caches = [SimpleNamespace(offset=100), SimpleNamespace(offset=4096),
              SimpleNamespace(), SimpleNamespace(offset=None)]
    assert pd.kv_depth(caches) == 4096
    assert pd.kv_depth([]) == 0
    assert pd.kv_depth(None) == 0


def test_score_heads():
    m = SimpleNamespace(config=SimpleNamespace(num_attention_heads=24))
    assert pd.score_heads(m) == 24
    nested = SimpleNamespace(config=SimpleNamespace(
        num_attention_heads=None,
        text_config=SimpleNamespace(num_attention_heads=32)))
    assert pd.score_heads(nested) == 32
    assert pd.score_heads(SimpleNamespace(config=SimpleNamespace())) == 32
    # config objects work directly too (drafter path)
    assert pd.score_heads(SimpleNamespace(num_attention_heads=16)) == 16


def _batch(base, depth, heads=HEADS):
    return SimpleNamespace(
        prefill_step_size=base,
        prompt_cache=[SimpleNamespace(offset=depth)],
        model=SimpleNamespace(
            config=SimpleNamespace(num_attention_heads=heads)),
    )


def test_decayed_for_batch(monkeypatch):
    _cap(monkeypatch, 1.0)
    assert pd.decayed_for_batch(_batch(2048, 0)) == 2048
    assert pd.decayed_for_batch(_batch(2048, 100_000)) == 256
    assert pd.decayed_for_batch(_batch(None, 100_000)) is None


def test_admission_prices_the_step_the_server_runs(monkeypatch):
    _cap(monkeypatch, 100.0)
    monkeypatch.delenv("PREFILL_STEP_SIZE", raising=False)
    m = SimpleNamespace(config=SimpleNamespace(
        model_type="dense_x", num_attention_heads=HEADS))
    cache = [SimpleNamespace(offset=0)]
    assert pd.score_transient_bytes(m, cache, 1000) == _transient(2048, 1000)
    # a streamed serve adopted the wider chunk
    m._gmlx_adopted_prefill_step = 8192
    assert pd.score_transient_bytes(m, cache, 1000) == _transient(8192, 1000)
    # the generator's own step, when the caller has it
    assert pd.score_transient_bytes(
        m, cache, 1000, base_step=4096) == _transient(4096, 1000)
    del m._gmlx_adopted_prefill_step
    monkeypatch.setenv("PREFILL_STEP_SIZE", "4096")
    assert pd.score_transient_bytes(m, cache, 1000) == _transient(4096, 1000)


def test_kill_switch(monkeypatch):
    _cap(monkeypatch, 1.0)
    monkeypatch.setenv("GMLX_PREFILL_DECAY", "0")
    assert pd.decayed_for_batch(_batch(2048, 100_000)) == 2048
    assert pd.install_prefill_decay() is False


def _headroom(monkeypatch, val):
    monkeypatch.setattr(pd, "_headroom_bytes", lambda: val)


def test_seed_follows_body_cap_when_headroom_unknown(monkeypatch):
    _cap(monkeypatch, 1.0)
    _headroom(monkeypatch, None)
    for depth in (0, 32_768, 200_000):
        assert (pd.decayed_seed_step(2048, depth, HEADS)
                == pd.decayed_step(2048, depth, HEADS))


def test_seed_pins_base_when_headroom_large(monkeypatch):
    _cap(monkeypatch, 1.0)
    _headroom(monkeypatch, 90 * GB)  # seed cap = 45 GB
    depth = 200_000
    # 32 * 2048 * ~202k * 2 = 26.5 GB fits half the 90 GB headroom
    assert pd.decayed_seed_step(2048, depth, HEADS) == 2048
    # the body still decays under its own 1 GB cap
    assert pd.decayed_step(2048, depth, HEADS) == 256


def test_seed_decays_under_tight_headroom(monkeypatch):
    _cap(monkeypatch, 1.0)
    _headroom(monkeypatch, 30 * GB)  # seed cap = 15 GB
    # 1024 tier (13.2 GB) fits, 2048 (26.5 GB) does not
    assert pd.decayed_seed_step(2048, 200_000, HEADS) == 1024
    # negative headroom floors at the body cap
    _headroom(monkeypatch, -5 * GB)
    assert pd.decayed_seed_step(2048, 200_000, HEADS) == 256


def test_seed_cap_env_wins_over_headroom(monkeypatch):
    _cap(monkeypatch, 1.0)
    _headroom(monkeypatch, 2 * GB)
    monkeypatch.setenv("GMLX_MTP_SEED_SCORE_CAP_GB", "30")
    # 32 * 2048 * ~202k * 2 = 26.5 GB fits the explicit 30 GB cap
    assert pd.decayed_seed_step(2048, 200_000, HEADS) == 2048


def test_seed_cap_env_invalid_falls_back(monkeypatch):
    _cap(monkeypatch, 1.0)
    _headroom(monkeypatch, None)
    monkeypatch.setenv("GMLX_MTP_SEED_SCORE_CAP_GB", "banana")
    assert (pd.decayed_seed_step(2048, 200_000, HEADS)
            == pd.decayed_step(2048, 200_000, HEADS))


def test_seed_step_honors_kill_switch(monkeypatch):
    _cap(monkeypatch, 1.0)
    monkeypatch.setenv("GMLX_PREFILL_DECAY", "0")
    assert pd.decayed_seed_step(2048, 10_000_000, HEADS) == 2048


def _fresh_registry(monkeypatch):
    monkeypatch.setattr(pd, "_UNTRACKED_WEIGHTS", {})
    monkeypatch.setattr(pd, "_UNTRACKED_OWNERS", {})
    monkeypatch.setattr(pd, "_WEIGHTS_OWNER", None)


def test_note_untracked_weights_unkeyed_dropped(monkeypatch):
    # unkeyed bytes could never be forgotten at eviction; the old anonymous
    # accumulator was the eviction leak in miniature, so key=None is dropped
    _fresh_registry(monkeypatch)
    pd.note_untracked_weights(10 * GB)
    pd.note_untracked_weights(5 * GB)
    assert pd.untracked_weight_bytes() == 0


def test_note_untracked_weights_same_key_first_wins(monkeypatch):
    # a drafter reloading the target's GGUF maps the same pages: the
    # re-registration must not double the count (it serialized admissions)
    monkeypatch.setattr(pd, "_UNTRACKED_WEIGHTS", {})
    key = ("/models/a.gguf",)
    pd.note_untracked_weights(87 * GB, key=key)
    pd.note_untracked_weights(87 * GB, key=key)
    assert pd.untracked_weight_bytes() == 87 * GB
    pd.note_untracked_weights(2 * GB, key=key)  # subset re-walk ignored
    assert pd.untracked_weight_bytes() == 87 * GB


def test_note_untracked_weights_first_wins_over_phantom(monkeypatch):
    # when active memory tracks the wire bytes, the bracketed first walk
    # registers ~0; a re-walk with no read of its own sees no delta and
    # would register the full bytes again - it must not override the first
    monkeypatch.setattr(pd, "_UNTRACKED_WEIGHTS", {})
    key = ("/models/a.gguf",)
    pd.note_untracked_weights(0, key=key)
    pd.note_untracked_weights(28 * GB, key=key)
    assert pd.untracked_weight_bytes() == 0


def test_note_untracked_weights_distinct_keys_sum(monkeypatch):
    _fresh_registry(monkeypatch)
    pd.note_untracked_weights(10 * GB, key=("/models/a.gguf",))
    pd.note_untracked_weights(4 * GB, key=("/models/b.gguf",))
    pd.note_untracked_weights(1 * GB)  # unkeyed: dropped
    assert pd.untracked_weight_bytes() == 14 * GB


def test_forget_drops_bytes_at_zero_owners(monkeypatch):
    # the eviction leak: a torn-down model's registration must leave the
    # estimate, or every model ever loaded stays subtracted from headroom
    # and the admission gate defers every request to the ceiling
    _fresh_registry(monkeypatch)
    pd.set_untracked_weights_owner("entry-a")
    pd.note_untracked_weights(87 * GB, key=("/models/a.gguf",))
    pd.set_untracked_weights_owner(None)
    pd.forget_untracked_weights("entry-a")
    assert pd.untracked_weight_bytes() == 0


def test_shared_key_survives_until_last_owner(monkeypatch):
    # two resident entries sharing shards (target + paired drafter reload,
    # the designed first-wins case): evicting either alone must keep the
    # pages counted; a build-window diff would under-attribute here and
    # flip the defect optimistic
    _fresh_registry(monkeypatch)
    key = ("/models/a.gguf",)
    pd.set_untracked_weights_owner("entry-a")
    pd.note_untracked_weights(87 * GB, key=key)
    pd.set_untracked_weights_owner("entry-b")
    pd.note_untracked_weights(87 * GB, key=key)  # re-registration: first wins
    pd.set_untracked_weights_owner(None)
    pd.forget_untracked_weights("entry-a")
    assert pd.untracked_weight_bytes() == 87 * GB
    pd.forget_untracked_weights("entry-b")
    assert pd.untracked_weight_bytes() == 0


def test_forget_leaves_other_entries_and_ownerless_keys(monkeypatch):
    _fresh_registry(monkeypatch)
    pd.note_untracked_weights(5 * GB, key=("/m/cli.gguf",))  # no owner window
    pd.set_untracked_weights_owner("entry-a")
    pd.note_untracked_weights(10 * GB, key=("/m/a.gguf",))
    pd.set_untracked_weights_owner("entry-b")
    pd.note_untracked_weights(4 * GB, key=("/m/b.gguf",))
    pd.set_untracked_weights_owner(None)
    pd.forget_untracked_weights("entry-a")
    assert pd.untracked_weight_bytes() == 9 * GB
    pd.forget_untracked_weights("entry-b")
    assert pd.untracked_weight_bytes() == 5 * GB  # ownerless key persists


def test_deduct_shrinks_registration_floored_at_zero(monkeypatch):
    # streamed-out expert bytes are page cache, not wired weights: the
    # streaming install deducts them or headroom sits permanently negative
    _fresh_registry(monkeypatch)
    key = ("/models/moe.gguf",)
    pd.note_untracked_weights(162 * GB, key=key)
    pd.deduct_untracked_weights(150 * GB, key)
    assert pd.untracked_weight_bytes() == 12 * GB
    pd.deduct_untracked_weights(50 * GB, key)
    assert pd.untracked_weight_bytes() == 0
    pd.deduct_untracked_weights(1 * GB, ("/models/other.gguf",))  # unknown
    pd.deduct_untracked_weights(1 * GB, None)                     # keyless
    assert pd.untracked_weight_bytes() == 0


def _fresh_streamed(monkeypatch):
    _fresh_registry(monkeypatch)
    monkeypatch.setattr(pd, "_STREAMED_TRACKED", {})
    monkeypatch.setattr(pd, "_STREAMED_CAP", {})
    monkeypatch.setattr(pd, "_BUILD_CREDIT", {})


def test_streamed_credit_raises_headroom(monkeypatch):
    # allocator-tracked streamed expert arrays sit inside active memory but
    # their pages are reclaimable file cache: without the add-back, an
    # over-RAM streaming model reads ~-150 GB headroom and every request
    # sheds from red (the 153 GB DSv4 verify)
    _fresh_streamed(monkeypatch)
    monkeypatch.setattr(
        pd.mx, "device_info",
        lambda: {"max_recommended_working_set_size": 120 * GB})
    monkeypatch.setattr(pd.mx, "get_active_memory", lambda: 237 * GB)
    assert pd.headroom_bytes() == -117 * GB
    pd.note_streamed_tracked_bytes(156 * GB, key=("/models/moe.gguf",))
    assert pd.headroom_bytes() == 39 * GB


def test_streamed_credit_keyless_and_first_wins(monkeypatch):
    _fresh_streamed(monkeypatch)
    pd.note_streamed_tracked_bytes(10 * GB)          # keyless: dropped
    assert pd.streamed_tracked_bytes() == 0
    key = ("/models/moe.gguf",)
    pd.note_streamed_tracked_bytes(156 * GB, key=key)
    pd.note_streamed_tracked_bytes(156 * GB, key=key)  # reload: first wins
    assert pd.streamed_tracked_bytes() == 156 * GB


def test_streamed_credit_sources_sum_under_union_cap(monkeypatch):
    # compose streams table and experts: the two sites' credits sum, each
    # idempotent per source, and the union never exceeds the tracked cap
    _fresh_streamed(monkeypatch)
    key = ("/models/moe.gguf",)
    pd.note_streamed_tracked_bytes(54 * GB, key=key, source="table",
                                   cap=170 * GB)
    pd.note_streamed_tracked_bytes(109 * GB, key=key, source="experts",
                                   cap=170 * GB)
    assert pd.streamed_tracked_bytes() == 163 * GB
    pd.note_streamed_tracked_bytes(109 * GB, key=key, source="experts",
                                   cap=170 * GB)  # reload: no double-credit
    assert pd.streamed_tracked_bytes() == 163 * GB


def _active(monkeypatch, ws_gb=120):
    box = {"active": 10 * GB}
    monkeypatch.setattr(
        pd.mx, "device_info",
        lambda: {"max_recommended_working_set_size": ws_gb * GB})
    monkeypatch.setattr(pd.mx, "get_active_memory", lambda: box["active"])
    return box


def test_build_credit_follows_active_growth_up_to_the_cap(monkeypatch):
    # Tracked regime: the walk grows active memory by the file before the
    # install credits it. The build credit covers the growth, capped at the
    # streamed bytes, so the every-token weights still count as live.
    _fresh_streamed(monkeypatch)
    box = _active(monkeypatch)
    pd.note_build_credit("owner", 330 * GB)
    assert pd.headroom_bytes() == 110 * GB
    box["active"] = 349 * GB
    assert pd.build_credit_bytes() == 330 * GB
    assert pd.headroom_bytes() == 120 * GB - 349 * GB + 330 * GB
    box["active"] = 30 * GB                 # untracked regime: views not in active
    assert pd.build_credit_bytes() == 20 * GB
    pd.forget_untracked_weights("owner")
    assert pd.headroom_bytes() == 90 * GB


def test_build_credit_ends_at_the_install_credit_or_deduction(monkeypatch):
    # The install credits tracked streamed bytes (or deducts untracked
    # ones) at the first request, after the build. Any of these ends the
    # build credit for the key's owners, a table's included (a table-only
    # install credits nothing else), so the two never overlap. Another
    # model's credit does not end it.
    _fresh_streamed(monkeypatch)
    box = _active(monkeypatch)
    key = ("/models/moe.gguf",)
    pd.set_untracked_weights_owner("owner")
    try:
        pd.note_build_credit("owner", 330 * GB)
        pd.note_untracked_weights(0.0, key=key)     # the warm pass, tracked regime
    finally:
        pd.set_untracked_weights_owner(None)
    box["active"] = 349 * GB
    pd.note_streamed_tracked_bytes(2 * GB, key=("/models/other.gguf",), source="experts")
    assert pd.build_credit_bytes() == 330 * GB
    pd.note_streamed_tracked_bytes(54 * GB, key=key, source="table", cap=339 * GB)
    assert pd.build_credit_bytes() == 0
    pd.note_streamed_tracked_bytes(276 * GB, key=key, source="experts", cap=339 * GB)
    assert pd.headroom_bytes() == 120 * GB - 349 * GB + 332 * GB
    # Untracked regime: the warm pass registers the file, the install
    # deducts the streamed share.
    _fresh_streamed(monkeypatch)
    box = _active(monkeypatch)
    pd.set_untracked_weights_owner("owner")
    try:
        pd.note_build_credit("owner", 330 * GB)
        pd.note_untracked_weights(339 * GB, key=key)
    finally:
        pd.set_untracked_weights_owner(None)
    assert pd.headroom_bytes() == 120 * GB - 10 * GB - 339 * GB + 330 * GB
    pd.deduct_untracked_weights(330 * GB, key)
    assert pd.build_credit_bytes() == 0
    assert pd.headroom_bytes() == 120 * GB - 10 * GB - 9 * GB


def test_streamed_credit_union_cap_clamps(monkeypatch):
    _fresh_streamed(monkeypatch)
    key = ("/models/moe.gguf",)
    pd.note_streamed_tracked_bytes(90 * GB, key=key, source="table",
                                   cap=100 * GB)
    pd.note_streamed_tracked_bytes(90 * GB, key=key, source="experts",
                                   cap=100 * GB)
    assert pd.streamed_tracked_bytes() == 100 * GB


def test_release_takes_back_credit_and_cap(monkeypatch):
    # Tracked regime: the stacks are credited, then unmapped, and the
    # tracker stops counting them; a credit left in place reads them free
    # twice.
    _fresh_streamed(monkeypatch)
    key = ("/models/moe.gguf",)
    pd.note_streamed_tracked_bytes(54 * GB, key=key, source="table",
                                   cap=170 * GB)
    pd.note_streamed_tracked_bytes(109 * GB, key=key, source="experts",
                                   cap=170 * GB)
    pd.release_streamed_tracked_bytes(100 * GB, key, "experts")
    assert pd.streamed_tracked_bytes() == 63 * GB
    assert pd._STREAMED_CAP[key] == 70 * GB
    pd.release_streamed_tracked_bytes(50 * GB, key, "experts")
    # The experts credit clamps at zero, and the cap falls by all 50 GB:
    # the tracker stopped counting each freed byte.
    assert pd.streamed_tracked_bytes() == 20 * GB
    pd.release_streamed_tracked_bytes(5 * GB, ("/models/other.gguf",))
    pd.release_streamed_tracked_bytes(5 * GB, None)
    assert pd.streamed_tracked_bytes() == 20 * GB


# The accounting invariant. The weights are N (every-token), T (a
# streamable table) and E (the routed experts); each part loads either
# allocator-tracked (in get_active_memory) or zero-copy (in the untracked
# registry). The arena and the KV are tracked allocations. At every stage
# the free figure must be the real one, ws minus the resident weights,
# the arena and the KV, and the install must end the serve build credit.
_N, _E, _ARENA, _KV, _WS = 20 * GB, 100 * GB, 8 * GB, 4 * GB, 120 * GB
_TABLES = {"small-table": 27 * GB, "large-table": 130 * GB}   # T < E, T >= E
_REGIMES = {                    # the parts that load allocator-tracked
    "zero-copy": set(),
    "tracked": {"N", "T", "E"},
    "views-untracked": {"N"},   # casts and repacks tracked, views not
    "experts-tracked": {"E"},
}
_INSTALLS = {                   # (model parts, streamed parts)
    "compose": ({"N", "T", "E"}, {"T", "E"}),
    "table-only": ({"N", "T", "E"}, {"T"}),
    "experts-only": ({"N", "E"}, {"E"}),
}
_CASES = [(r, i, t, serve, unmap)
          for r in _REGIMES for i in _INSTALLS for t in _TABLES
          for serve in (False, True) for unmap in (False, True)
          if (not unmap or "E" in _INSTALLS[i][1])
          and (i != "experts-only" or t == "small-table")]


@pytest.mark.parametrize("regime,install,table,serve,unmap", _CASES)
def test_free_is_the_real_free_at_every_stage(monkeypatch, regime, install,
                                              table, serve, unmap):
    _fresh_streamed(monkeypatch)
    box = _active(monkeypatch, ws_gb=_WS // GB)
    size = {"N": _N, "T": _TABLES[table], "E": _E}
    parts, streamed = _INSTALLS[install]
    tracked_parts = _REGIMES[regime] & parts
    total = sum(size[p] for p in parts)
    tracked = sum(size[p] for p in tracked_parts)
    key = ("/models/moe.gguf",)

    def real(resident, extra=0.0):
        return _WS - resident - extra

    # The build: serve opens a window and credits the bytes the header
    # predicts the install will stream (capacity.predicted_stream_bytes)
    # until the install accounts them.
    resident = total - sum(size[p] for p in streamed)
    box["active"] = 0.0
    if serve:
        pd.set_untracked_weights_owner("entry")
        pd.note_build_credit("entry", sum(size[p] for p in streamed))
    box["active"] = float(tracked)
    pd.note_untracked_weights(float(total - tracked), key=key)
    pd.set_untracked_weights_owner(None)
    if serve:
        assert pd.headroom_bytes() == real(resident)

    # The install, in install_expert_streaming's order: the table share,
    # the walk over the experts, the experts share, the end of the credit.
    untracked0 = pd.untracked_weight_bytes_for(key)
    split = max(0.0, total - untracked0)
    table_credit = 0.0
    if "T" in streamed:
        table_credit = pd.credit_streamed(key, size["T"], split, "table")
        if "E" not in streamed:
            pd.end_build_credit(key)
        if serve or "E" not in streamed:
            # Across the pin, the prefetcher and the wrap walk.
            left = _N if "E" in streamed else _N + _E
            assert pd.headroom_bytes() == real(left)
    if "E" in streamed:
        pd.credit_streamed(key, _E, split, "experts", prior=table_credit)
    pd.end_build_credit(key)
    box["active"] = float(tracked + _ARENA + _KV)
    assert pd.build_credit_bytes() == 0
    assert pd.headroom_bytes() == real(resident, _ARENA + _KV)

    # The feeders take the stacks: their tracked bytes leave active memory.
    if unmap:
        freed = _E if "E" in tracked_parts else 0
        box["active"] -= freed
        if freed:
            pd.release_streamed_tracked_bytes(freed, key, "experts")
        assert pd.headroom_bytes() == real(resident, _ARENA + _KV)


def test_a_keyless_release_ends_the_open_build_credit(monkeypatch):
    # No weights key: nothing was credited, so the release ends the open
    # owner's build credit, which would keep crediting the freed stacks.
    _fresh_streamed(monkeypatch)
    box = _active(monkeypatch)
    pd.set_untracked_weights_owner("owner")
    try:
        pd.note_build_credit("owner", 330 * GB)
        box["active"] = 349 * GB
        assert pd.build_credit_bytes() == 330 * GB
        pd.release_streamed_tracked_bytes(0, None)     # nothing freed
        assert pd.build_credit_bytes() == 330 * GB
        pd.release_streamed_tracked_bytes(300 * GB, None)
        assert pd.build_credit_bytes() == 0
    finally:
        pd.set_untracked_weights_owner(None)


def test_forget_drops_streamed_credit_with_owner(monkeypatch):
    # an evicted streaming model must stop crediting headroom, or the
    # phantom credit overstates free working set for its successors
    _fresh_streamed(monkeypatch)
    key = ("/models/moe.gguf",)
    pd.set_untracked_weights_owner("entry-a")
    pd.note_untracked_weights(0.0, key=key)  # warm pass: tracked load
    pd.note_streamed_tracked_bytes(156 * GB, key=key)
    pd.set_untracked_weights_owner(None)
    assert pd.streamed_tracked_bytes() == 156 * GB
    pd.forget_untracked_weights("entry-a")
    assert pd.streamed_tracked_bytes() == 0


# --- arch score profiles -------------------------------------------------

FLASH = pd.ScoreTransientProfile(heads=1, bytes_per_elem=4, depth_divisor=4)


def _register(monkeypatch, model_type, provider):
    monkeypatch.setitem(pd._SCORE_PROFILES, model_type, provider)


def _pbatch(base, depth, model_type="fake_arch", heads=64):
    return SimpleNamespace(
        prefill_step_size=base,
        prompt_cache=[SimpleNamespace(offset=depth)],
        model=SimpleNamespace(config=SimpleNamespace(
            model_type=model_type, num_attention_heads=heads)),
    )


def test_profile_holds_full_step_at_depth(monkeypatch):
    _cap(monkeypatch, 5.0)
    # flash-native transient [1, step, depth/4] fp32: full step to ~2.4M
    assert pd.decayed_step(2048, 200_000, 64, profile=FLASH) == 2048
    assert pd.decayed_step(2048, 1_000_000, 64, profile=FLASH) == 2048
    assert pd.decayed_step(2048, 3_000_000, 64, profile=FLASH) == 1024
    # dense model at the same cap floors out
    assert pd.decayed_step(2048, 200_000, 64) == 256


def test_dense_equivalent_profile_matches_default(monkeypatch):
    _cap(monkeypatch, 1.0)
    dense = pd.ScoreTransientProfile(heads=HEADS, bytes_per_elem=2,
                                     depth_divisor=1)
    for depth in (0, 8_192, 32_768, 100_000, 200_000):
        assert (pd.decayed_step(2048, depth, HEADS, profile=dense)
                == pd.decayed_step(2048, depth, HEADS))


def test_profile_floor_and_passthrough(monkeypatch):
    _cap(monkeypatch, 0.5)
    heavy = pd.ScoreTransientProfile(heads=64, bytes_per_elem=4,
                                     depth_divisor=1)
    assert pd.decayed_step(2048, 10_000_000, 64, profile=heavy) == 256
    # explicit base at/below the floor still passes through untouched
    assert pd.decayed_step(128, 10_000_000, 64, profile=heavy) == 128


def test_profile_idempotent(monkeypatch):
    _cap(monkeypatch, 5.0)
    for depth in (0, 200_000, 3_000_000):
        once = pd.decayed_step(2048, depth, 64, profile=FLASH)
        assert pd.decayed_step(once, depth, 64, profile=FLASH) == once


def test_profile_cap_env_still_authoritative(monkeypatch):
    _cap(monkeypatch, 0.1)  # the env floor
    # a tiny explicit cap forces decay even under a flash profile
    assert pd.decayed_step(2048, 300_000, 64, profile=FLASH) == 256


def test_resolve_score_profile_registry(monkeypatch):
    _register(monkeypatch, "fake_arch", lambda m, c: FLASH)
    m = SimpleNamespace(config=SimpleNamespace(model_type="fake_arch"))
    assert pd.resolve_score_profile(m, None) == FLASH
    other = SimpleNamespace(config=SimpleNamespace(model_type="other"))
    assert pd.resolve_score_profile(other, None) is None
    # dict-shaped configs resolve too (CLI/MTP wrapper paths)
    md = SimpleNamespace(config={"model_type": "fake_arch"})
    assert pd.resolve_score_profile(md, None) == FLASH
    assert pd.resolve_score_profile(SimpleNamespace(), None) is None


def test_resolve_score_profile_provider_failures(monkeypatch):
    _register(monkeypatch, "none_arch", lambda m, c: None)
    _register(monkeypatch, "raise_arch",
              lambda m, c: (_ for _ in ()).throw(RuntimeError("boom")))
    none_m = SimpleNamespace(config=SimpleNamespace(model_type="none_arch"))
    raise_m = SimpleNamespace(config=SimpleNamespace(model_type="raise_arch"))
    assert pd.resolve_score_profile(none_m, None) is None
    assert pd.resolve_score_profile(raise_m, None) is None


def test_resolve_score_profile_kill_switch(monkeypatch):
    _register(monkeypatch, "fake_arch", lambda m, c: FLASH)
    monkeypatch.setenv("GMLX_PREFILL_SCORE_PROFILE", "0")
    m = SimpleNamespace(config=SimpleNamespace(model_type="fake_arch"))
    assert pd.resolve_score_profile(m, None) is None


def test_decayed_for_batch_with_profile(monkeypatch):
    _cap(monkeypatch, 1.0)
    _register(monkeypatch, "fake_arch", lambda m, c: FLASH)
    # profile holds the full step where the dense model would floor out
    assert pd.decayed_for_batch(_pbatch(2048, 100_000)) == 2048
    unregistered = _pbatch(2048, 100_000, model_type="plain_arch")
    assert pd.decayed_for_batch(unregistered) == 256


def test_profile_base_step_swap(monkeypatch):
    _cap(monkeypatch, 5.0)
    monkeypatch.delenv("PREFILL_STEP_SIZE", raising=False)
    with_base = FLASH._replace(base_step=4096)
    _register(monkeypatch, "fake_arch", lambda m, c: with_base)
    # stock 2048 base swaps to the arch default at any depth it fits
    assert pd.decayed_for_batch(_pbatch(2048, 0)) == 4096
    assert pd.decayed_for_batch(_pbatch(2048, 200_000)) == 4096
    # an explicit PREFILL_STEP_SIZE keeps the swap off
    monkeypatch.setenv("PREFILL_STEP_SIZE", "2048")
    assert pd.decayed_for_batch(_pbatch(2048, 0)) == 2048
    monkeypatch.delenv("PREFILL_STEP_SIZE")
    # a non-stock batch base is explicit config: no swap
    assert pd.decayed_for_batch(_pbatch(1024, 0)) == 1024
    # the kill switch bypasses swap and decay alike
    monkeypatch.setenv("GMLX_PREFILL_DECAY", "0")
    assert pd.decayed_for_batch(_pbatch(2048, 200_000)) == 2048


def test_needs_processing_answers_as_the_decayed_step_would(monkeypatch):
    """For a model that projects every column, stock needs_processing reads
    the prefill step only to test it for None. Run with the decayed step in
    place of the batch's, it answers as stock does: decay at depth, the arch
    base swap, a pinned media step and a None step, each with no
    embeddings, a one-token prompt, a longer one and a pending checkpoint
    column."""
    from gmlx._vendor.mlx_vlm.generate.ar import PromptProcessingBatch

    stock = PromptProcessingBatch.needs_processing
    _cap(monkeypatch, 1.0)
    monkeypatch.delenv("PREFILL_STEP_SIZE", raising=False)
    _register(monkeypatch, "fake_arch",
              lambda m, c: FLASH._replace(base_step=4096))

    def with_decayed_step(b):
        base = b.prefill_step_size
        b.prefill_step_size = pd.decayed_for_batch(b)
        try:
            return stock(b)
        finally:
            b.prefill_step_size = base

    def pinned():
        b = _batch(2048, 0)
        b._gmlx_pinned_step = 300
        return b

    makers = (lambda: _batch(2048, 100_000), lambda: _pbatch(2048, 0),
              pinned, lambda: _batch(None, 100_000))
    moved = 0
    for make in makers:
        for embeds in (None, (1, 1, 8), (1, 9, 8)):
            for col in (None, 5):
                b = make()
                b.draft_model = None
                b._inputs_embeds = (None if embeds is None
                                    else SimpleNamespace(shape=embeds))
                b._next_apc_checkpoint_column = lambda col=col: col
                moved += pd.decayed_for_batch(b) != b.prefill_step_size
                assert with_decayed_step(b) == stock(b), (make, embeds, col)
    assert moved == 18  # every case but the None step moves it


def _installed(monkeypatch):
    from gmlx._vendor.mlx_vlm.generate.ar import PromptProcessingBatch as P

    monkeypatch.setattr(P, "prompt_step", P.prompt_step)
    monkeypatch.setattr(P, "needs_processing", P.needs_processing)
    monkeypatch.setattr(P, pd._FLAG, False, raising=False)
    return P


def test_install_wraps_prompt_step_and_needs_processing(monkeypatch):
    P = _installed(monkeypatch)
    step, needs = P.prompt_step, P.needs_processing
    assert pd.install_prefill_decay()
    assert P.prompt_step is not step
    assert P.needs_processing is not needs


def test_a_batch_that_keeps_the_last_logits_folds_at_the_decayed_step(
        monkeypatch):
    """A batch whose model keeps only the last logits hands generate() the
    columns left once they fit a step. The wrapped needs_processing reads
    the decayed step, so at depth 1000 columns take chunks first, where at
    the base step all 1000 would fold into generate(). A speculative batch
    and a model that projects every column keep the last column alone, at
    any step."""
    _cap(monkeypatch, 1.0)
    monkeypatch.setattr(pd, "_LIVE_DECODE_ROWS", 0)
    P = _installed(monkeypatch)
    assert pd.install_prefill_decay()

    def batch(keeps, draft=None, depth=100_000, cols=1000):
        b = _batch(2048, depth)
        b.model.language_model = SimpleNamespace(
            supports_logits_to_keep=keeps)
        b.draft_model = draft
        b._inputs_embeds = SimpleNamespace(shape=(1, cols, 8))
        b._next_apc_checkpoint_column = lambda: None
        return b

    assert pd.decayed_for_batch(batch(True)) == 256
    assert P.needs_processing(batch(True)) is True
    assert P.needs_processing(batch(True, cols=256)) is False
    assert P.needs_processing(batch(True, depth=0)) is False
    assert P.needs_processing(batch(True, depth=0, cols=2049)) is True
    assert P.needs_processing(batch(False, cols=2)) is True
    assert P.needs_processing(batch(True, draft=object(), cols=2)) is True
    assert P.needs_processing(batch(False, cols=1)) is False


def test_a_streamed_tail_merges_into_the_folded_last_forward(monkeypatch):
    """A streamed MoE prefill widens the step so a short tail joins the
    chunk before it. A batch that folds merges over the whole prompt, so
    2100 columns at a step of 2048 run as one forward, where a batch that
    keeps the last column alone merges over 2099 and still leaves it."""
    import gmlx.stream.expert_streaming as es

    _cap(monkeypatch, 1.0)
    monkeypatch.setattr(pd, "_LIVE_DECODE_ROWS", 0)
    monkeypatch.setattr(es, "moe_streaming_active", lambda model: True)
    P = _installed(monkeypatch)
    assert pd.install_prefill_decay()

    def batch(keeps, draft=None):
        b = _batch(2048, 0)
        b.model.language_model = SimpleNamespace(
            supports_logits_to_keep=keeps)
        b.draft_model = draft
        b._inputs_embeds = SimpleNamespace(shape=(1, 2100, 8))
        b._next_apc_checkpoint_column = lambda: None
        return b

    assert pd.decayed_for_batch(batch(True)) == 2100
    assert P.needs_processing(batch(True)) is False
    assert pd.decayed_for_batch(batch(False)) == 2099
    assert P.needs_processing(batch(False)) is True
    assert pd.decayed_for_batch(batch(True, draft=object())) == 2099


def test_a_folding_streamed_prefill_reaches_generate_as_its_step_decays(
        monkeypatch):
    """A folding batch with no checkpoint pause runs the stock loop to
    generate() on a streamed model whose step decays with depth. Inside a
    chunk the step is sized already: sized again from the decayed width,
    the tail merge would take the whole rest as one chunk the stock step
    never runs, and the loop would spin."""
    from prefill_loop import Model, drain, install, make_batch

    install(monkeypatch, {"p": None, "sys": None}, [])
    _cap(monkeypatch, 5.4)
    for n in (28677, 28818, 20483, 37153):
        chunks = drain(make_batch(Model(folds=True), n, apc=False))
        assert sum(chunks) == n
        if n == 28677:
            assert chunks == [8192, 8192, 4096, 4096, 4096, 5]


def test_decayed_seed_step_with_profile(monkeypatch):
    _cap(monkeypatch, 1.0)
    monkeypatch.setenv("GMLX_MTP_SEED_SCORE_CAP_GB", "1")
    assert pd.decayed_seed_step(2048, 200_000, 64) == 256
    assert pd.decayed_seed_step(2048, 200_000, 64, profile=FLASH) == 2048


# --- provider builder ------------------------------------------------------

class _Pool(SimpleNamespace):
    pass


class _BatchPool(SimpleNamespace):
    pass


def _built(**kw):
    kw.setdefault("profile", FLASH)
    kw.setdefault("require_cache", _Pool)
    kw.setdefault("disarm_cache", _BatchPool)
    return pd.build_score_profile(**kw)


def _layer(*caches):
    return SimpleNamespace(caches=list(caches))


def test_walk_caches_flattens_cachelist():
    a, b, c = (SimpleNamespace(offset=i) for i in range(3))
    assert list(pd._walk_caches([_layer(a, b), c])) == [a, b, c]
    assert list(pd._walk_caches(None)) == []


def test_build_kernels_gate():
    armed = {"on": False}
    prov = _built(kernels_armed=lambda: armed["on"])
    cache = [_layer(_Pool(offset=100))]
    assert prov(None, cache) is None
    armed["on"] = True
    assert prov(None, cache) == FLASH


def test_build_require_cache_missing_returns_none():
    prov = _built()
    assert prov(None, [SimpleNamespace(offset=100)]) is None
    assert prov(None, None) is None
    assert prov(None, [_layer(_Pool(offset=100))]) == FLASH


def test_build_disarm_cache():
    prov = _built()
    assert prov(None, [_layer(_Pool(offset=1), _BatchPool(offset=1))]) is None


def test_build_non_int_offset_disarms():
    prov = _built()
    assert prov(None, [_layer(_Pool(offset=[3, 5]))]) is None


def test_build_quantized_required_disarms_by_default():
    qpool = _Pool(offset=100, is_quantized=True)
    assert _built()(None, [_layer(qpool)]) is None
    assert _built(allow_quantized_pools=True)(None, [_layer(qpool)]) == FLASH


def test_build_bits_on_other_cache_disarms():
    # a quantized non-required cache (local KV with bits) always disarms,
    # even when quantized required pools are allowed
    prov = _built(allow_quantized_pools=True)
    cache = [_layer(SimpleNamespace(offset=1, bits=8), _Pool(offset=100))]
    assert prov(None, cache) is None


def test_build_callable_profile_resolved_per_call():
    live = {"base": None}
    prov = _built(profile=lambda: FLASH._replace(base_step=live["base"]))
    cache = [_layer(_Pool(offset=100))]
    assert prov(None, cache).base_step is None
    live["base"] = 4096
    assert prov(None, cache).base_step == 4096


# --- seed profile threading + logging ---------------------------------------

def test_seed_profile_resolves_against_drafter_cache(monkeypatch):
    from gmlx.spec.mtp_drafter import QwenMTPDrafter
    prov = pd.build_score_profile(profile=FLASH, require_cache=_Pool)
    _register(monkeypatch, "fake_arch", prov)
    drafter = SimpleNamespace(config=SimpleNamespace(model_type="fake_arch"),
                              _cache=[SimpleNamespace(offset=7)])
    # a plain-KV drafter cache has no required pool: dense seed decay stays
    assert QwenMTPDrafter._seed_profile(drafter) is None
    drafter._cache = [_Pool(offset=7)]
    assert QwenMTPDrafter._seed_profile(drafter) == FLASH


def test_seed_decay_log(monkeypatch, capsys):
    monkeypatch.setenv("GMLX_MTP_SEED_SCORE_CAP_GB", "1")
    monkeypatch.setenv("GMLX_PREFILL_DECAY_LOG", "1")
    assert pd.decayed_seed_step(2048, 200_000, 64) == 256
    out = capsys.readouterr().out
    assert "[prefill-decay] seed depth 200000: step 2048 -> 256" in out


def test_seed_decay_log_silent_when_unchanged(monkeypatch, capsys):
    monkeypatch.setenv("GMLX_MTP_SEED_SCORE_CAP_GB", "1000")
    monkeypatch.setenv("GMLX_PREFILL_DECAY_LOG", "1")
    assert pd.decayed_seed_step(2048, 200_000, 64) == 2048
    assert capsys.readouterr().out == ""


# --- headroom-aware body cap -------------------------------------------------

GIB = float(1 << 30)


def _hcap(monkeypatch, *, gate=True, noted=8 * GIB, room=55 * GB,
          legacy=6.0 * GB):
    monkeypatch.delenv("GMLX_PREFILL_SCORE_CAP_GB", raising=False)
    monkeypatch.setenv("GMLX_PREFILL_HEADROOM_CAP", "1" if gate else "0")
    monkeypatch.setattr(pd, "_NOTED_CACHE_LIMIT", noted)
    monkeypatch.setattr(pd, "_headroom_bytes", lambda: room)
    monkeypatch.setattr(pd, "_WS_CAP_BYTES", legacy)


def test_headroom_cap_default_off(monkeypatch):
    _hcap(monkeypatch, gate=True)
    monkeypatch.delenv("GMLX_PREFILL_HEADROOM_CAP")
    assert pd._body_cap_bytes() == 6.0 * GB


def test_headroom_cap_requires_noted_limit(monkeypatch):
    _hcap(monkeypatch, noted=None)
    assert pd._body_cap_bytes() == 6.0 * GB


def test_headroom_cap_raises_at_depth(monkeypatch):
    # the 122B shape: 32 heads, 55 GB free, 8 GiB cache limit. Raised cap
    # = min(27.5 GB, 8.59 GB) = 8.59 GB: 1024 at d110k (legacy 512) and 512
    # at d200k (legacy 256); held transients (7.3 / 6.6 GB) stay under the
    # recycle threshold.
    _hcap(monkeypatch, gate=False)
    assert pd.decayed_step(2048, 110_000, 32) == 512
    assert pd.decayed_step(2048, 200_000, 32) == 256
    _hcap(monkeypatch, gate=True)
    assert pd._body_cap_bytes() == 8 * GIB
    assert pd.decayed_step(2048, 110_000, 32) == 1024
    assert pd.decayed_step(2048, 200_000, 32) == 512


def test_headroom_cap_env_cap_still_wins(monkeypatch):
    _hcap(monkeypatch)
    monkeypatch.setenv("GMLX_PREFILL_SCORE_CAP_GB", "1.0")
    assert pd._body_cap_bytes() == 1.0 * GB


def test_headroom_cap_probe_failure_falls_back(monkeypatch):
    _hcap(monkeypatch, room=None)
    assert pd._body_cap_bytes() == 6.0 * GB


def test_headroom_cap_never_below_legacy(monkeypatch):
    # tight headroom on a limit-configured box: legacy floor holds
    _hcap(monkeypatch, room=4 * GB)
    assert pd._body_cap_bytes() == 6.0 * GB


def test_note_cache_limit_none_clears(monkeypatch):
    monkeypatch.setattr(pd, "_NOTED_CACHE_LIMIT", None)
    pd.note_cache_limit(8 * GIB)
    assert pd._NOTED_CACHE_LIMIT == 8 * GIB
    pd.note_cache_limit(None)
    assert pd._NOTED_CACHE_LIMIT is None


def test_seed_cap_unchanged_by_gate(monkeypatch):
    # seed = max(body cap, 0.5*headroom); with ample headroom the seed cap is
    # 0.5h either way -- the certified seed formula is untouched by the gate
    _hcap(monkeypatch, gate=False, room=90 * GB)
    monkeypatch.delenv("GMLX_MTP_SEED_SCORE_CAP_GB", raising=False)
    off = pd.decayed_seed_step(2048, 10_000_000, 64)
    _hcap(monkeypatch, gate=True, room=90 * GB)
    assert pd.decayed_seed_step(2048, 10_000_000, 64) == off


def test_headroom_cap_no_noted_limit_matches_legacy_ladder(monkeypatch):
    _hcap(monkeypatch, gate=True, noted=None)
    for depth in (0, 8_192, 46_000, 92_000, 110_000, 200_000, 400_000):
        with_gate = pd.decayed_step(2048, depth, 32)
        _hcap(monkeypatch, gate=False, noted=None)
        assert with_gate == pd.decayed_step(2048, depth, 32)
        _hcap(monkeypatch, gate=True, noted=None)


def _moe(e, k, nested=False):
    c = SimpleNamespace(model_type="moe_x", num_attention_heads=HEADS,
                        num_experts=e, num_experts_per_tok=k)
    if nested:
        return SimpleNamespace(config=SimpleNamespace(text_config=c))
    return SimpleNamespace(args=c, model_type="moe_x")


@pytest.fixture
def resident(monkeypatch):
    import gmlx.load.dtypes as dtypes
    import gmlx.stream.expert_streaming as es

    monkeypatch.setattr(es, "experts_streamed", lambda m: False)
    monkeypatch.setattr(dtypes, "gpu_arch_gen", lambda: 17)
    monkeypatch.delenv("PREFILL_STEP_SIZE", raising=False)
    monkeypatch.delenv("GMLX_MOE_WIDE_STEP", raising=False)
    _headroom(monkeypatch, 40 * GB)


def test_moe_wide_step_waits_for_a_new_enough_gpu(monkeypatch, resident):
    import gmlx.load.dtypes as dtypes

    # one generation under the line keeps the stock chunk by default
    monkeypatch.setattr(dtypes, "gpu_arch_gen", lambda: 16)
    assert pd._run_base(_moe(256, 8), None, None) == 2048
    # an explicit GMLX_MOE_WIDE_STEP applies on any chip
    monkeypatch.setenv("GMLX_MOE_WIDE_STEP", "4096")
    assert pd._run_base(_moe(256, 8), None, None) == 4096
    monkeypatch.delenv("GMLX_MOE_WIDE_STEP")
    # an unknown generation counts as new
    monkeypatch.setattr(dtypes, "gpu_arch_gen", lambda: 0)
    assert pd._run_base(_moe(256, 8), None, None) == 4096
    # a value that is not a number counts as unset
    monkeypatch.setattr(dtypes, "gpu_arch_gen", lambda: 16)
    monkeypatch.setenv("GMLX_MOE_WIDE_STEP", "yes")
    assert pd._run_base(_moe(256, 8), None, None) == 2048


def test_moe_with_few_rows_per_expert_takes_the_wide_step(resident):
    # 256 experts at top 6 or 8 and 128 at top 8: 48, 64 and 128 rows per
    # expert of a 2048 chunk
    for e, k in ((256, 6), (256, 8), (128, 8)):
        assert pd._run_base(_moe(e, k), None, None) == 4096
    assert pd._run_base(_moe(256, 8, nested=True), None, None) == 4096
    # 32 experts at top 4 (256 rows per expert) and a dense model keep 2048
    assert pd._run_base(_moe(32, 4), None, None) == 2048
    dense = SimpleNamespace(config=SimpleNamespace(num_attention_heads=HEADS))
    assert pd._run_base(dense, None, None) == 2048


def test_moe_wide_step_runs_the_serve_chunk(monkeypatch, resident):
    _cap(monkeypatch, 100.0)
    batch = SimpleNamespace(prefill_step_size=2048, model=_moe(256, 8),
                            prompt_cache=[SimpleNamespace(offset=0)],
                            uids=[0])
    assert pd.decayed_for_batch(batch) == 4096
    # 2 prompts give 128 rows per expert, 3 give 192
    batch.uids = [0, 1]
    assert pd.decayed_for_batch(batch) == 4096
    batch.uids = [0, 1, 2]
    assert pd.decayed_for_batch(batch) == 2048
    assert pd._run_base(_moe(256, 8), None, None, rows=3) == 2048


def test_moe_wide_step_logs_its_base_once(monkeypatch, resident, capsys):
    _cap(monkeypatch, 100.0)
    monkeypatch.setenv("GMLX_PREFILL_DECAY_LOG", "1")
    monkeypatch.setattr(pd, "_PROFILE_LOGGED", set())
    batch = SimpleNamespace(prefill_step_size=2048, model=_moe(256, 8),
                            prompt_cache=[SimpleNamespace(offset=0)])
    pd.decayed_for_batch(batch)
    pd.decayed_for_batch(batch)
    out = capsys.readouterr().out
    assert out.count("[prefill-decay] base step 2048 -> 4096") == 1


def test_moe_wide_step_yields(monkeypatch, resident):
    import gmlx.stream.expert_streaming as es

    m = _moe(256, 8)
    _headroom(monkeypatch, 21 * GB)  # 21e9 bytes, under the 20 GiB floor
    assert pd._run_base(m, None, None) == 2048
    _headroom(monkeypatch, 40 * GB)
    monkeypatch.setenv("GMLX_PREFILL_DECAY", "0")
    assert pd._run_base(m, None, None) == 2048
    monkeypatch.delenv("GMLX_PREFILL_DECAY")
    monkeypatch.setenv("PREFILL_STEP_SIZE", "2048")
    assert pd._run_base(m, None, None) == 2048
    monkeypatch.delenv("PREFILL_STEP_SIZE")
    monkeypatch.setenv("GMLX_MOE_WIDE_STEP", "0")
    assert pd._run_base(m, None, None) == 2048
    monkeypatch.delenv("GMLX_MOE_WIDE_STEP")
    monkeypatch.setattr(es, "experts_streamed", lambda m: True)
    assert pd._run_base(m, None, None) == 2048
    monkeypatch.setattr(es, "experts_streamed", lambda m: False)
    # an arch profile's own step wins
    prof = pd.ScoreTransientProfile(base_step=8192)
    assert pd._run_base(m, prof, None) == 8192


def test_resident_ckpt_grid_follows_the_moe_wide_step(monkeypatch, resident):
    from gmlx.spec import ckpt

    batch = SimpleNamespace(prefill_step_size=2048, model=_moe(256, 8),
                            prompt_cache=[], uids=[0])
    assert ckpt._ckpt_run_step(batch) == 4096
    assert ckpt._ckpt_grids(batch, 0, 16, None, ())[0] == [(0, 4096)]
    batch.uids = [0, 1, 2]
    assert ckpt._ckpt_run_step(batch) == 2048
    batch.uids = [0]
    # recurrent state keeps the batch's grid
    from gmlx.cache.kv_cache import ArraysCache, KVCache

    batch.prompt_cache = [KVCache(), ArraysCache(size=2)]
    assert ckpt._ckpt_run_step(batch) == 2048
    batch.prompt_cache = [KVCache()]
    assert ckpt._ckpt_run_step(batch) == 4096
    monkeypatch.setenv("PREFILL_STEP_SIZE", "2048")
    assert ckpt._ckpt_run_step(batch) == 2048
    monkeypatch.delenv("PREFILL_STEP_SIZE")
    # an arch profile's base step keeps the batch's grid
    monkeypatch.setattr(pd, "resolve_score_profile", lambda m, c:
                        pd.ScoreTransientProfile(base_step=8192))
    assert ckpt._ckpt_run_step(batch) == 2048
