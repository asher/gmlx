"""Checkpoint-cursor interval schedule (owned mid-prefill store hook).

The hook piggybacks the stock column machinery: after storing at a
boundary it advances meta["checkpoint_len"] to the next interval point,
latching checkpoint_done only after the terminal (the natural-chunk grid
point at or below the stock guard column). The stock exact store is
suppressed by the advance, not the latch -- asserted here against the
real unbound upstream method.
"""

from types import SimpleNamespace

import mlx.core as mx
from mlx_vlm.apc import APCManager

import gmlx.spec.engine as se
from gmlx.cache.snapshot import ckpt_lookup
from gmlx.spec.speculative import _sidecar_boundary

from test_ckpt_tier import make_hybrid_cache


def _batch(step=2048):
    return SimpleNamespace(prefill_step_size=step)


# -- cursor init: grid snapping, restored skip, terminal clamp --

def _positions(bounds):
    assert all(kind == "boundary" for _, kind in bounds)
    return [p for p, _ in bounds]


def test_cursor_grid_and_terminal():
    bounds, terminal, interval = se._ckpt_cursor_init(
        _batch(), guard=27000, restored=0, block_size=16)
    assert (terminal, interval) == (26624, 4096)
    assert _positions(bounds) == [4096, 8192, 12288, 16384, 20480,
                                  24576, 26624]


def test_cursor_skips_restored_prefix():
    bounds, terminal, interval = se._ckpt_cursor_init(
        _batch(), guard=27000, restored=8192, block_size=16)
    assert _positions(bounds)[0] == 12288
    # Restored past the terminal: nothing left to checkpoint.
    assert se._ckpt_cursor_init(
        _batch(), guard=27000, restored=26624, block_size=16) == ([], 0, 0)


def test_cursor_interval_snaps_up_to_chunk_grid(monkeypatch):
    monkeypatch.setenv("GMLX_APC_CKPT_INTERVAL", "1000")
    bounds, terminal, interval = se._ckpt_cursor_init(
        _batch(), guard=27000, restored=0, block_size=16)
    assert interval == 2048                       # never below one chunk
    assert _positions(bounds)[0] == 2048


def test_cursor_zero_interval_is_terminal_only(monkeypatch):
    monkeypatch.setenv("GMLX_APC_CKPT_INTERVAL", "0")
    bounds, terminal, interval = se._ckpt_cursor_init(
        _batch(), guard=27000, restored=0, block_size=16)
    assert (terminal, interval) == (26624, 0)
    assert _positions(bounds) == [26624]


def test_cursor_no_step_uses_block_grid():
    bounds, terminal, interval = se._ckpt_cursor_init(
        _batch(step=None), guard=100, restored=0, block_size=16)
    assert (terminal, interval) == (96, 4096)
    assert _positions(bounds) == [96]


# -- Stage 5: replay boundary at N-1 --

def _arm_meta(n, tags, restored=0, guard=None, step=2048):
    batch = SimpleNamespace(
        prefill_step_size=step,
        model=SimpleNamespace(_kq_apc_ckpt_layout=tuple(tags)))
    meta = {"full_input_ids": list(range(n))}
    se._ckpt_arm_schedule(batch, meta, guard if guard is not None else n,
                          restored, block_size=16)
    return meta


def test_replay_boundary_short_prompt_arr():
    # Sub-interval prompt: the interval/terminal schedule is empty (bug
    # 1's shape) and the replay boundary is the only store point.
    meta = _arm_meta(1500, ("kv", "arr"))
    assert meta["ckpt_boundaries"] == [(1499, "replay")]
    assert meta["checkpoint_len"] == 1499
    assert meta["ckpt_stored_boundaries"] == []


def test_replay_boundary_arr_floor(monkeypatch):
    # Below GMLX_APC_CKPT_REPLAY_MIN a GDN clone costs the same >100 MB
    # as a deep one; the boundary is not armed.
    assert _arm_meta(500, ("kv", "arr"))["ckpt_boundaries"] == []
    monkeypatch.setenv("GMLX_APC_CKPT_REPLAY_MIN", "100")
    assert _arm_meta(500, ("kv", "arr"))["ckpt_boundaries"] == [
        (499, "replay")]


def test_replay_boundary_rot_window_floor():
    # No byte floor for window clones, but N-1 must be at or past the
    # window (the store's grid gate declines below it).
    assert _arm_meta(300, ("rot:512:0", "kv"))["ckpt_boundaries"] == []
    assert _arm_meta(600, ("rot:512:0", "kv"))["ckpt_boundaries"] == [
        (599, "replay")]


def test_replay_boundary_appends_after_terminal():
    meta = _arm_meta(5000, ("kv", "arr"))
    assert meta["ckpt_boundaries"] == [(4096, "boundary"), (4999, "replay")]


def test_replay_boundary_grid_collision_keeps_boundary_kind():
    # N-1 landing exactly on the terminal column: one boundary, plain
    # kind. A grid-aligned boundary record adopts freely -- identical
    # resend included -- while a replay flip would gate turn-2 and
    # branch adoption out on recurrent layouts yet still satisfy the
    # p=N drop.
    meta = _arm_meta(4097, ("kv", "arr"))
    assert meta["ckpt_boundaries"] == [(4096, "boundary")]


def test_replay_boundary_kill_switch(monkeypatch):
    monkeypatch.setenv("GMLX_APC_CKPT_REPLAY", "0")
    assert _arm_meta(1500, ("kv", "arr"))["ckpt_boundaries"] == []


def test_replay_boundary_skipped_when_restored_past():
    # Identical-resend warm turn: the adopted prefix already sits at
    # N-1; re-storing it would churn the LRU for nothing.
    assert _arm_meta(1500, ("kv", "arr"),
                     restored=1499)["ckpt_boundaries"] == []


# -- Stage 6: render-stable turn boundaries --

def _stub_p_stable(monkeypatch, p_stable):
    import gmlx.cache.retire_key as retire_key
    monkeypatch.setattr(retire_key, "lookup_render_ctx",
                        lambda ids: {"stub": True})
    monkeypatch.setattr(retire_key, "prompt_stable_lcp",
                        lambda ctx, ids: p_stable)


def test_turn_boundary_grid_all_layouts(monkeypatch):
    _stub_p_stable(monkeypatch, 2100)
    meta = _arm_meta(5000, ("kv", "arr"))
    # The grid point below p_stable joins the schedule; p_stable itself
    # is never armed on arr layouts (off-grid chunking drifts GDN state).
    assert meta["ckpt_boundaries"] == [
        (2048, "boundary"), (4096, "boundary"), (4999, "replay")]
    assert meta["ckpt_p_stable_bounds"] == [2048]
    assert meta["ckpt_p_stable"] == 2100


def test_turn_boundary_rot_exact(monkeypatch):
    # Rot-only layouts also pause exactly at p_stable: the attention
    # split is exact and turn 2 adopts the full stable prefix.
    _stub_p_stable(monkeypatch, 2100)
    meta = _arm_meta(5000, ("rot:512:0", "kv"))
    assert meta["ckpt_boundaries"] == [
        (2048, "boundary"), (2100, "boundary"), (4096, "boundary"),
        (4999, "replay")]
    assert meta["ckpt_p_stable_bounds"] == [2048, 2100]


def test_turn_boundary_rot_below_window_grid_only(monkeypatch):
    _stub_p_stable(monkeypatch, 2100)
    meta = _arm_meta(5000, ("rot:4096:0", "kv"))
    assert meta["ckpt_boundaries"] == [
        (2048, "boundary"), (4096, "boundary"), (4999, "replay")]
    assert meta["ckpt_p_stable_bounds"] == [2048]


def test_turn_boundary_grid_collides_with_terminal(monkeypatch):
    _stub_p_stable(monkeypatch, 4990)
    meta = _arm_meta(5000, ("kv", "arr"))
    assert meta["ckpt_boundaries"] == [(4096, "boundary"), (4999, "replay")]
    assert meta["ckpt_p_stable_bounds"] == [4096]


def test_turn_boundary_kill_switch(monkeypatch):
    _stub_p_stable(monkeypatch, 2100)
    monkeypatch.setenv("GMLX_APC_CKPT_TURN", "0")
    meta = _arm_meta(5000, ("kv", "arr"))
    assert meta["ckpt_p_stable_bounds"] == []
    assert meta["ckpt_boundaries"] == [(4096, "boundary"), (4999, "replay")]


def test_turn_boundary_no_render_ctx(monkeypatch):
    import gmlx.cache.retire_key as retire_key
    monkeypatch.setattr(retire_key, "lookup_render_ctx", lambda ids: None)
    meta = _arm_meta(5000, ("kv", "arr"))
    assert meta["ckpt_p_stable_bounds"] == []
    assert meta["ckpt_boundaries"] == [(4096, "boundary"), (4999, "replay")]


def test_turn_boundary_small_prompt_skips_prediction(monkeypatch):
    # Below the cheapest armable boundary the render+tokenize prediction
    # is pure cost; the schedule must not even look up the render ctx.
    import gmlx.cache.retire_key as retire_key

    def _boom(ids):
        raise AssertionError("render ctx consulted below the arm floor")

    monkeypatch.setattr(retire_key, "lookup_render_ctx", _boom)
    # Floors: turn needs one chunk (arr) or the window (rot-only); the
    # system anchor floors at GMLX_APC_CKPT_SYS_MIN (256), raised to the
    # replay byte floor (1024) on arr layouts.
    assert _arm_meta(1000, ("kv", "arr"))["ckpt_p_stable_bounds"] == []
    assert _arm_meta(250, ("rot:512:0", "kv"))["ckpt_p_stable_bounds"] == []
    # At the floor the prediction runs again.
    _stub_p_stable(monkeypatch, 550)
    meta = _arm_meta(600, ("rot:512:0", "kv"))
    assert meta["ckpt_p_stable_bounds"] == [550]


# -- advance + latch across a schedule --

def _armed_batch(man, ids, first, terminal, interval):
    bounds = []
    if interval:
        b = first
        while b < terminal:
            bounds.append((b, "boundary"))
            b += interval
    bounds.append((terminal, "boundary"))
    meta = {
        "full_input_ids": ids,
        "prefix_len": 0,
        "extra_hash": 7,
        "ckpt_boundaries": bounds,
        "checkpoint_len": bounds[0][0],
        "ckpt_terminal": terminal,
        "ckpt_interval": interval,
        "ckpt_last_stored": 0,
    }
    return SimpleNamespace(
        _kq_ckpt_armed=True, _apc_manager=man, _apc_meta=[meta],
        prompt_cache=None, _row_real_tokens_processed=lambda idx: 0), meta


def test_cursor_advances_and_latches():
    man = APCManager(num_blocks=96, block_size=16)
    ids = list(range(500, 500 + 160))
    batch, meta = _armed_batch(man, ids, first=32, terminal=128, interval=32)
    for boundary in (32, 64, 96, 128):
        assert meta["checkpoint_len"] == boundary
        assert not meta.get("checkpoint_done")
        batch.prompt_cache = make_hybrid_cache(boundary, seed=boundary)
        batch._row_real_tokens_processed = (
            lambda idx, b=boundary: b)
        se._ckpt_mid_prefill_store(batch)
        assert meta["ckpt_last_stored"] == boundary
    assert meta.get("checkpoint_done") is True
    # Strip-on-extend keeps the newest two boundaries hittable plus the
    # first one, promoted to the chain's anchor; the interior boundary
    # is stripped by design.
    for probe, hit in ((40, 32), (72, 32), (104, 96), (136, 128)):
        warm, got = ckpt_lookup(man, ids[:probe], extra_hash=7)
        assert got == hit


def test_cursor_off_boundary_chunk_is_a_noop():
    man = APCManager(num_blocks=64, block_size=16)
    ids = list(range(500, 500 + 120))
    batch, meta = _armed_batch(man, ids, first=32, terminal=96, interval=32)
    batch._row_real_tokens_processed = lambda idx: 24
    se._ckpt_mid_prefill_store(batch)
    assert meta["checkpoint_len"] == 32 and meta["ckpt_last_stored"] == 0


def test_cursor_advances_past_failed_store():
    """A declined store never wedges the cursor, and ckpt_last_stored only
    records boundaries that actually landed."""
    man = APCManager(num_blocks=64, block_size=16)
    ids = list(range(500, 500 + 120))
    batch, meta = _armed_batch(man, ids, first=32, terminal=64, interval=32)
    # Cache offset 48 != boundary 32: the store's offset guard declines.
    batch.prompt_cache = make_hybrid_cache(48)
    batch._row_real_tokens_processed = lambda idx: 32
    se._ckpt_mid_prefill_store(batch)
    assert meta["checkpoint_len"] == 64 and meta["ckpt_last_stored"] == 0
    batch.prompt_cache = make_hybrid_cache(64)
    batch._row_real_tokens_processed = lambda idx: 64
    se._ckpt_mid_prefill_store(batch)
    assert meta.get("checkpoint_done") and meta["ckpt_last_stored"] == 64


# -- suppression by advance: the stock exact store never fires --

def test_stock_store_suppressed_by_advance():
    """The plan's explicit ordering test: the owned hook runs immediately
    before the stock _store_apc_exact_checkpoints, and the advance (not the
    latch) is what keeps the stock multi-GB clone from firing on an armed
    row -- at every boundary of the schedule, not just the last."""
    from mlx_vlm.generate import ar

    man = APCManager(num_blocks=64, block_size=16)
    calls = []
    orig = man.store_exact_cache
    man.store_exact_cache = (
        lambda *a, **k: (calls.append(a), orig(*a, **k))[1])
    ids = list(range(500, 500 + 120))
    batch, meta = _armed_batch(man, ids, first=32, terminal=96, interval=32)
    batch._apc_mode = "exact"
    batch._apc_prompt_cache_for_store = lambda idx: batch.prompt_cache
    stock_store = ar.PromptProcessingBatch._store_apc_exact_checkpoints
    for boundary in (32, 64, 96):
        batch.prompt_cache = make_hybrid_cache(boundary, seed=boundary)
        batch._row_real_tokens_processed = (
            lambda idx, b=boundary: b)
        se._ckpt_mid_prefill_store(batch)   # the two adjacent lines from
        stock_store(batch)                  # _mtp_prompt_step, same order
    assert calls == []
    assert meta.get("checkpoint_done") is True


# -- retire ctx: sidecar keys on the last stored boundary --

def test_sidecar_boundary_reads_last_stored():
    meta = {"ckpt_last_stored": 24576, "checkpoint_len": 28672}
    ctx = {"mode": "ckpt", "apc_meta": meta, "checkpoint_len": 4096}
    assert _sidecar_boundary(ctx) == 24576
    meta["ckpt_last_stored"] = 0
    assert _sidecar_boundary(ctx) == 0      # nothing stored -> no boundary key
    assert _sidecar_boundary(
        {"mode": "exact", "checkpoint_len": 4096}) == 4096


# -- eager publication: a later request's lookup sees mid-prefill stores --

def test_second_request_hits_first_requests_checkpoint():
    """The 1-cold + N-restores conversion: request A's mid-prefill boundary
    store publishes to the manager records immediately, so request B (same
    prefix, prefilled after) restores from it at its own prefill init."""
    se._bind_l1_view()
    man = APCManager(num_blocks=256, block_size=16)
    p_a, boundary = 120, 96
    shared = list(range(900, 900 + boundary))
    ids_a = shared + list(range(50, 50 + p_a - boundary))

    model = SimpleNamespace(
        _kq_apc_manager=man, _kq_apc_mode="exact",
        config=SimpleNamespace(),
        make_cache=lambda: make_hybrid_cache(0))
    batch_a = SimpleNamespace(
        model=model, _input_ids=mx.array([ids_a]),
        _inputs_embeds=mx.zeros((1, p_a, 4)),
        prompt_cache=make_hybrid_cache(p_a), _prompt_kwargs={},
        prefill_step_size=32)
    se._mtp_prefill_init(batch_a)
    assert getattr(batch_a, "_kq_ckpt_armed", False)
    meta = batch_a._apc_meta[0]
    cl = int(meta["checkpoint_len"])
    assert cl == boundary                   # guard-trimmed terminal grid
    batch_a.prompt_cache = make_hybrid_cache(cl, seed=1)
    batch_a._row_real_tokens_processed = lambda idx: cl
    se._ckpt_mid_prefill_store(batch_a)
    assert meta["ckpt_last_stored"] == cl

    # Request B shares the first `boundary` tokens, diverges after.
    ids_b = shared + list(range(700, 700 + 40))
    batch_b = SimpleNamespace(
        model=model, _input_ids=mx.array([ids_b]),
        _inputs_embeds=mx.zeros((1, len(ids_b), 4)),
        prompt_cache=make_hybrid_cache(len(ids_b)), _prompt_kwargs={},
        prefill_step_size=32, _prompt_length_aware_keys=())
    se._mtp_prefill_init(batch_b)
    assert batch_b._mtp_l1_prefix_len == boundary


# -- formation: ckpt-tier models serialize prompt admission --

def test_ckpt_formation_serializes_prefill(monkeypatch):
    """Simultaneous arrivals must not coalesce into a multi-row prompt
    batch on a ckpt-tier model (the owned APC declines B>1 prefill); B=1
    formation converts a burst into 1 cold + N-1 restores. Pure-KV models
    keep batched formation."""
    import importlib

    from test_ckpt_tier import KVCache

    ar = importlib.import_module("mlx_vlm.generate.ar")

    class _ProbeBG:
        def __init__(self, model, processor, **kwargs):
            self.prefill_batch_size = kwargs.get("prefill_batch_size", 8)

    monkeypatch.setattr(ar, "BatchGenerator", _ProbeBG)
    se._install_apc_manager_stash()
    se._bind_l1_view()
    man = APCManager(num_blocks=64, block_size=16)

    hybrid = SimpleNamespace(make_cache=lambda: make_hybrid_cache(0))
    bg = ar.BatchGenerator(hybrid, None, draft_model=object(),
                           apc_manager=man, prefill_batch_size=8)
    assert bg.prefill_batch_size == 1

    def kv_cache():
        c = KVCache()
        return [c, KVCache()]

    kv_model = SimpleNamespace(make_cache=kv_cache)
    bg2 = ar.BatchGenerator(kv_model, None, draft_model=object(),
                            apc_manager=man, prefill_batch_size=8)
    assert bg2.prefill_batch_size == 8

    bg3 = ar.BatchGenerator(SimpleNamespace(), None, draft_model=object(),
                            apc_manager=None, prefill_batch_size=8)
    assert bg3.prefill_batch_size == 8


# -- system-prefix anchor boundary --

def _stub_sys(monkeypatch, lcp):
    import gmlx.cache.retire_key as retire_key
    monkeypatch.setattr(retire_key, "lookup_render_ctx",
                        lambda ids: {"stub": True})
    monkeypatch.setattr(retire_key, "prompt_stable_lcp",
                        lambda ctx, ids: None)
    monkeypatch.setattr(retire_key, "system_prefix_lcp",
                        lambda ctx, ids: lcp)


def test_sys_boundary_block_grid_attention(monkeypatch):
    # Attention layouts snap the system-prefix stop to the block grid;
    # the record kind is anchor, exempt from strip-on-extend.
    _stub_sys(monkeypatch, 2900)
    meta = _arm_meta(5000, ("rot:512:0", "kv"))
    assert meta["ckpt_boundaries"] == [
        (2896, "anchor"), (4096, "boundary"), (4999, "replay")]
    assert meta["ckpt_sys_bound"] == 2896
    assert meta["checkpoint_len"] == 2896


def test_sys_boundary_chunk_grid_on_arr(monkeypatch):
    # arr layouts keep the chunk grid (off-grid chunking drifts GDN
    # state), so the stop lands a full grid point below the offset.
    _stub_sys(monkeypatch, 2900)
    meta = _arm_meta(5000, ("kv", "arr"))
    assert meta["ckpt_boundaries"] == [
        (2048, "anchor"), (4096, "boundary"), (4999, "replay")]
    assert meta["ckpt_sys_bound"] == 2048


def test_sys_boundary_floors(monkeypatch):
    # Below GMLX_APC_CKPT_SYS_MIN no anchor lands; arr layouts raise the
    # floor to the replay byte floor (state clones are size-independent).
    _stub_sys(monkeypatch, 200)
    meta = _arm_meta(5000, ("rot:512:0", "kv"))
    assert "ckpt_sys_bound" not in meta
    assert all(k != "anchor" for _, k in meta["ckpt_boundaries"])
    _stub_sys(monkeypatch, 1500)          # grid point 0 on the 2048 grid
    meta = _arm_meta(5000, ("kv", "arr"))
    assert "ckpt_sys_bound" not in meta


def test_sys_boundary_upgrades_boundary_never_replay(monkeypatch):
    # Collision with a plain boundary upgrades it to anchor; a replay
    # arriving at the same position afterwards never downgrades it.
    _stub_sys(monkeypatch, 4100)
    meta = _arm_meta(5000, ("rot:512:0", "kv"))
    assert meta["ckpt_boundaries"] == [(4096, "anchor"), (4999, "replay")]
    meta = _arm_meta(4097, ("rot:512:0", "kv"))
    assert meta["ckpt_boundaries"] == [(4096, "anchor")]


def test_sys_boundary_kill_switch(monkeypatch):
    _stub_sys(monkeypatch, 2900)
    monkeypatch.setenv("GMLX_APC_CKPT_SYS", "0")
    meta = _arm_meta(5000, ("rot:512:0", "kv"))
    assert all(k != "anchor" for _, k in meta["ckpt_boundaries"])
    assert "ckpt_sys_bound" not in meta


def test_sys_boundary_skipped_when_restored_past(monkeypatch):
    # A warm sibling restored at or past the anchor has nothing to
    # re-store; the hit already refreshed the record's LRU position.
    _stub_sys(monkeypatch, 2900)
    meta = _arm_meta(5000, ("rot:512:0", "kv"), restored=3000)
    assert all(k != "anchor" for _, k in meta["ckpt_boundaries"])
    assert "ckpt_sys_bound" not in meta
