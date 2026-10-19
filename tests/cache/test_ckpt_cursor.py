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
from gmlx._vendor.mlx_vlm.apc import APCManager

import gmlx.spec.ckpt as ckpt
import gmlx.spec.engine as se
import gmlx.spec.mtp_prefill as mtp_prefill
from gmlx.cache.snapshot import ckpt_lookup
from gmlx.spec.speculative import _sidecar_boundary
from helpers import bind_stock_checkpoint_methods

from test_ckpt_tier import make_hybrid_cache


def _batch(step=2048):
    return SimpleNamespace(prefill_step_size=step)


# -- cursor init: grid snapping, restored skip, terminal clamp --

def _positions(bounds):
    assert all(kind == "boundary" for _, kind in bounds)
    return [p for p, _ in bounds]


def test_cursor_grid_and_terminal():
    bounds, terminal, interval = ckpt._ckpt_cursor_init(
        _batch(), guard=27000, restored=0, block_size=16)
    assert (terminal, interval) == (26624, 4096)
    assert _positions(bounds) == [4096, 8192, 12288, 16384, 20480,
                                  24576, 26624]


def test_cursor_skips_restored_prefix():
    bounds, terminal, interval = ckpt._ckpt_cursor_init(
        _batch(), guard=27000, restored=8192, block_size=16)
    assert _positions(bounds)[0] == 12288
    # Restored past the terminal: nothing left to checkpoint.
    assert ckpt._ckpt_cursor_init(
        _batch(), guard=27000, restored=26624, block_size=16) == ([], 0, 0)


def test_cursor_interval_snaps_up_to_chunk_grid(monkeypatch):
    monkeypatch.setenv("GMLX_APC_CKPT_INTERVAL", "1000")
    bounds, terminal, interval = ckpt._ckpt_cursor_init(
        _batch(), guard=27000, restored=0, block_size=16)
    assert interval == 2048                       # never below one chunk
    assert _positions(bounds)[0] == 2048


def test_cursor_zero_interval_is_terminal_only(monkeypatch):
    monkeypatch.setenv("GMLX_APC_CKPT_INTERVAL", "0")
    bounds, terminal, interval = ckpt._ckpt_cursor_init(
        _batch(), guard=27000, restored=0, block_size=16)
    assert (terminal, interval) == (26624, 0)
    assert _positions(bounds) == [26624]


def test_cursor_no_step_uses_block_grid():
    bounds, terminal, interval = ckpt._ckpt_cursor_init(
        _batch(step=None), guard=100, restored=0, block_size=16)
    assert (terminal, interval) == (96, 4096)
    assert _positions(bounds) == [96]


def test_streamed_grid_starts_at_the_restored_prefix(monkeypatch):
    """A streamed model reads every expert once per chunk, so a boundary
    that cuts its chunk adds a full pass. Its grid starts where this
    prefill's chunks start, the restored prefix, not at token 0."""
    def batch(streamed, tags=("rot:2048:0", "kv")):
        return SimpleNamespace(prefill_step_size=8192, model=SimpleNamespace(
            _kq_streaming=streamed, _kq_apc_ckpt_layout=tags))

    # A warm turn at 7299 + 1735 tokens: one chunk, no cut at 8192.
    assert ckpt._ckpt_cursor_init(
        batch(True), guard=9033, restored=7299, block_size=16) == ([], 0, 0)
    bounds, terminal, _ = ckpt._ckpt_cursor_init(
        batch(False), guard=9033, restored=7299, block_size=16)
    assert _positions(bounds) == [8192]
    # A long warm turn stores at its own chunk edges.
    bounds, terminal, _ = ckpt._ckpt_cursor_init(
        batch(True), guard=30000, restored=7299, block_size=16)
    assert _positions(bounds) == [15491, 23683]
    # A cold prompt keeps the absolute grid, and so do recurrent layouts.
    bounds, _, _ = ckpt._ckpt_cursor_init(
        batch(True), guard=30000, restored=0, block_size=16)
    assert _positions(bounds) == [8192, 16384, 24576]
    bounds, _, _ = ckpt._ckpt_cursor_init(
        batch(True, ("kv", "arr")), guard=30000, restored=7299,
        block_size=16)
    assert _positions(bounds) == [8192, 16384, 24576]


def test_streamed_origin_snaps_to_a_block_below_the_rotating_window():
    """Below a rotating window the store takes only block-aligned
    positions, so an off-block restore there starts the grid at its
    block. Past the window the grid starts at the restore itself."""
    batch = SimpleNamespace(prefill_step_size=32, model=SimpleNamespace(
        _kq_streaming=True, _kq_apc_ckpt_layout=("rot:64:0", "kv")))
    bounds, _, _ = ckpt._ckpt_cursor_init(
        batch, guard=56, restored=5, block_size=16)
    assert _positions(bounds) == [32]
    bounds, _, _ = ckpt._ckpt_cursor_init(
        batch, guard=200, restored=100, block_size=16)
    assert _positions(bounds) == [196]


def test_grid_follows_the_profile_step_the_loop_runs(monkeypatch):
    """On a streamed model an arch profile's step replaces a default
    batch step in the prefill loop, and a cut off it costs an expert
    pass, so the grid is cut at that step too. A resident model keeps
    the batch's grid."""
    import gmlx.gen.prefill_decay as pd

    monkeypatch.delenv("PREFILL_STEP_SIZE", raising=False)
    monkeypatch.setitem(pd._SCORE_PROFILES, "grid_probe", lambda m, c: (
        pd.ScoreTransientProfile(heads=4, bytes_per_elem=4, depth_divisor=4,
                                 base_step=8192)))

    def batch(streamed):
        return SimpleNamespace(
            prefill_step_size=2048, prompt_cache=[], model=SimpleNamespace(
                model_type="grid_probe", _kq_streaming=streamed,
                _kq_apc_ckpt_layout=("arr", "kv")))

    bounds, terminal, interval = ckpt._ckpt_cursor_init(
        batch(True), guard=20000, restored=0, block_size=16)
    assert (interval, terminal) == (8192, 16384)
    assert _positions(bounds) == [8192, 16384]
    _, terminal, interval = ckpt._ckpt_cursor_init(
        batch(False), guard=20000, restored=0, block_size=16)
    assert (interval, terminal) == (4096, 18432)
    monkeypatch.setenv("PREFILL_STEP_SIZE", "2048")
    _, _, interval = ckpt._ckpt_cursor_init(
        batch(True), guard=20000, restored=0, block_size=16)
    assert interval == 4096


# -- Stage 5: replay boundary at N-1 --

def _arm_meta(n, tags, restored=0, guard=None, step=2048):
    batch = SimpleNamespace(
        prefill_step_size=step,
        model=SimpleNamespace(_kq_apc_ckpt_layout=tuple(tags)))
    meta = {"full_input_ids": list(range(n))}
    ckpt._ckpt_arm_schedule(batch, meta, guard if guard is not None else n,
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


def test_turn_boundary_streamed_grid_starts_at_the_restore(monkeypatch):
    _stub_p_stable(monkeypatch, 9020)
    batch = SimpleNamespace(prefill_step_size=8192, model=SimpleNamespace(
        _kq_streaming=True, _kq_apc_ckpt_layout=("rot:2048:0", "kv")))
    meta = {"full_input_ids": list(range(9034))}
    ckpt._ckpt_arm_schedule(batch, meta, 9033, 7299, block_size=16)
    # No grid point inside the one chunk; the exact pause and N-1 stay.
    assert meta["ckpt_boundaries"] == [(9020, "boundary"), (9033, "replay")]
    assert meta["ckpt_p_stable_bounds"] == [9020]


def _streamed_meta(n, restored, tags=("rot:2048:0", "kv"), streamed=True,
                   **model):
    batch = SimpleNamespace(prefill_step_size=8192, model=SimpleNamespace(
        _kq_streaming=streamed, _kq_apc_ckpt_layout=tuple(tags), **model))
    meta = {"full_input_ids": list(range(n))}
    ckpt._ckpt_arm_schedule(batch, meta, n - 16, restored, block_size=16)
    return meta


def _at(bounds):
    return [p for p, _ in bounds]


def _no_decay(monkeypatch, sys_anchor=False):
    # The score cap that keeps the step whole at every depth used here.
    monkeypatch.setenv("GMLX_PREFILL_SCORE_CAP_GB", "1000")
    monkeypatch.delenv("PREFILL_STEP_SIZE", raising=False)
    if not sys_anchor:
        monkeypatch.setattr(ckpt, "_ckpt_sys_boundary", lambda *a: None)


def test_streamed_schedule_keeps_a_merged_tail_whole(monkeypatch):
    """A streamed prefill widens its last chunk by up to an eighth so a
    short tail joins it. A grid point in that tail would cut it into a
    second full expert pass, while the turn pause after it stores the
    turn anyway."""
    _no_decay(monkeypatch)
    # A cold 9000-token prompt: one chunk, not 8192 and then 798.
    _stub_p_stable(monkeypatch, 8990)
    meta = _streamed_meta(9000, 0)
    assert _at(meta["ckpt_boundaries"]) == [8990, 8999]
    assert meta["ckpt_p_stable_bounds"] == [8990]
    assert _at(_streamed_meta(9000, 0, streamed=False)[
        "ckpt_boundaries"]) == [8192, 8990, 8999]
    # A warm turn of 8192 + 100 tokens: one pass, as for 8000 tokens.
    _stub_p_stable(monkeypatch, 7299 + 8292 - 14)
    meta = _streamed_meta(7299 + 8292, 7299)
    assert _at(meta["ckpt_boundaries"]) == [15577, 15590]
    # A grid point well before the end stays.
    _stub_p_stable(monkeypatch, 19990)
    assert _at(_streamed_meta(20000, 0)["ckpt_boundaries"]) == [
        8192, 16384, 19990, 19999]
    # A turn pause below the grid point starts the chunk: 8000 and 1499,
    # not 8000, 192 and 1307.
    _stub_p_stable(monkeypatch, 8000)
    assert _at(_streamed_meta(9500, 0)["ckpt_boundaries"]) == [8000, 9499]


def test_streamed_schedule_keeps_a_grid_point_a_next_turn_needs(
        monkeypatch):
    """A grid point stays when it is the only record a next turn can
    adopt: on a media prompt, whose turn record is not known, or where
    the window keeps the exact turn pause out. A text prompt with no turn
    record drops it: a resend that extends it adopts the N-1 record."""
    _no_decay(monkeypatch)
    _stub_p_stable(monkeypatch, None)
    meta = _streamed_meta(9000, 0)
    assert _at(meta["ckpt_boundaries"]) == [8999]
    assert meta["ckpt_p_stable_bounds"] == []
    ids = list(range(9000))
    ids[100:356] = [99999] * 256
    batch = SimpleNamespace(prefill_step_size=8192, model=SimpleNamespace(
        _kq_streaming=True, _kq_apc_ckpt_layout=("rot:2048:0", "kv"),
        config=SimpleNamespace(image_token_id=99999)))
    meta = {"full_input_ids": ids}
    ckpt._ckpt_arm_schedule(batch, meta, 8984, 0, block_size=16)
    assert _at(meta["ckpt_boundaries"]) == [8192, 8999]
    _stub_p_stable(monkeypatch, 8286)
    meta = _streamed_meta(8300, 0, tags=("rot:8290:0", "kv"))
    assert _at(meta["ckpt_boundaries"]) == [8192, 8299]
    assert meta["ckpt_p_stable_bounds"] == [8192]


def test_streamed_schedule_drops_a_cut_only_where_the_loop_merges(
        monkeypatch):
    """A grid point goes only where the loop would merge the tail past
    it. A batch that runs its last token in the last chunk merges over
    the whole prompt, so a tail one token over an eighth stays cut. A
    step that decays at depth never merges."""
    _no_decay(monkeypatch)
    _stub_p_stable(monkeypatch, 9203)
    folding = {"language_model": SimpleNamespace(
        supports_logits_to_keep=True)}
    assert _at(_streamed_meta(9217, 0, **folding)[
        "ckpt_boundaries"]) == [8192, 9203, 9216]
    assert _at(_streamed_meta(9216, 0)["ckpt_boundaries"]) == [9203, 9215]
    # 16 heads under a 5.4 GB cap decay 8192 to 4096 past about 12.4k.
    monkeypatch.setenv("GMLX_PREFILL_SCORE_CAP_GB", "5.4")
    _stub_p_stable(monkeypatch, 16000 + 8492 - 14)
    meta = _streamed_meta(16000 + 8492, 16000,
                          config=SimpleNamespace(num_attention_heads=16))
    assert 24192 in _at(meta["ckpt_boundaries"])


def test_streamed_cold_grid_starts_at_the_system_prefix(monkeypatch):
    """The anchor at the end of the system prefix cuts a cold prompt, so
    the chunks after it start there and so does the grid: two passes for
    10000 tokens, not three. A resident model keeps the grid from 0."""
    import gmlx.cache.retire_key as retire_key

    _no_decay(monkeypatch, sys_anchor=True)
    _stub_p_stable(monkeypatch, 9986)
    monkeypatch.setattr(retire_key, "system_prefix_lcp", lambda ctx, ids: 3000)
    meta = _streamed_meta(10000, 0)
    assert meta["ckpt_boundaries"] == [
        (2992, "anchor"), (9986, "boundary"), (9999, "replay")]
    assert _at(_streamed_meta(10000, 0, streamed=False)[
        "ckpt_boundaries"]) == [2992, 8192, 9986, 9999]
    _stub_p_stable(monkeypatch, 19986)
    assert _at(_streamed_meta(20000, 0)["ckpt_boundaries"]) == [
        2992, 11184, 19986, 19999]
    # No grid point below the anchor, where no chunk edge is.
    monkeypatch.setattr(retire_key, "system_prefix_lcp", lambda ctx, ids: 9010)
    _stub_p_stable(monkeypatch, 9000)
    assert _at(_streamed_meta(10000, 0)["ckpt_boundaries"]) == [
        9000, 9008, 9999]


def test_an_mtp_batch_arms_its_grid_from_the_base_step(monkeypatch):
    """Serve installs the MTP prompt step before the load installs prefill
    decay, so the MTP body that arms the schedule runs while the decay
    wrapper holds this chunk's merged width in prefill_step_size. The
    grid follows the batch's step all the same."""
    import gmlx.gen.prefill_decay as pd
    from gmlx._vendor.mlx_vlm.generate.ar import PromptProcessingBatch as P

    _no_decay(monkeypatch)
    _stub_p_stable(monkeypatch, 8986)
    seen = {}

    def mtp_body(batch):
        seen["step"] = batch.prefill_step_size
        meta = {"full_input_ids": list(range(9000))}
        ckpt._ckpt_arm_schedule(batch, meta, 8984, 0, block_size=16)
        seen["meta"] = meta

    monkeypatch.setattr(P, "prompt_step", mtp_body)
    monkeypatch.setattr(P, "needs_processing", P.needs_processing)
    monkeypatch.setattr(P, pd._FLAG, False, raising=False)
    assert pd.install_prefill_decay()
    model = SimpleNamespace(_kq_streaming=True,
                            _kq_apc_ckpt_layout=("arr", "kv"))
    batch = SimpleNamespace(
        prefill_step_size=8192, model=model, prompt_cache=[],
        draft_model=object(), _gmlx_pinned_step=None,
        _inputs_embeds=SimpleNamespace(shape=(1, 9000, 8)),
        _next_apc_checkpoint_column=lambda: None)
    P.prompt_step(batch)
    assert seen["step"] == 8999
    assert seen["meta"]["ckpt_boundaries"] == [
        (8192, "boundary"), (8999, "replay")]
    assert seen["meta"]["ckpt_p_stable_bounds"] == [8192]
    assert batch.prefill_step_size == 8192
    assert not hasattr(batch, "_gmlx_base_step")


def test_short_cold_prompt_keeps_the_anchor_out_of_the_turn_set(
        monkeypatch):
    """The anchor is its own record. Less than one step past it, it is
    neither the turn's grid point nor the terminal."""
    import gmlx.cache.retire_key as retire_key

    _no_decay(monkeypatch, sys_anchor=True)
    _stub_p_stable(monkeypatch, 4986)
    monkeypatch.setattr(retire_key, "system_prefix_lcp", lambda ctx, ids: 3000)
    meta = _streamed_meta(5000, 0)
    assert meta["ckpt_boundaries"] == [
        (2992, "anchor"), (4986, "boundary"), (4999, "replay")]
    assert meta["ckpt_p_stable_bounds"] == [4986]
    assert (meta["ckpt_terminal"], meta["ckpt_interval"]) == (0, 0)


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
    return bind_stock_checkpoint_methods(SimpleNamespace(
        _kq_ckpt_armed=True, _apc_manager=man, _apc_meta=[meta],
        prompt_cache=None, _row_real_tokens_processed=lambda idx: 0)), meta


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
        ckpt._ckpt_mid_prefill_store(batch)
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
    ckpt._ckpt_mid_prefill_store(batch)
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
    ckpt._ckpt_mid_prefill_store(batch)
    assert meta["checkpoint_len"] == 64 and meta["ckpt_last_stored"] == 0
    batch.prompt_cache = make_hybrid_cache(64)
    batch._row_real_tokens_processed = lambda idx: 64
    ckpt._ckpt_mid_prefill_store(batch)
    assert meta.get("checkpoint_done") and meta["ckpt_last_stored"] == 64


# -- suppression by advance: the stock exact store never fires --

def test_stock_store_suppressed_by_advance():
    """The plan's explicit ordering test: the owned hook runs immediately
    before the stock _store_apc_exact_checkpoints, and the advance (not the
    latch) is what keeps the stock multi-GB clone from firing on an armed
    row -- at every boundary of the schedule, not just the last."""
    from gmlx._vendor.mlx_vlm.generate import ar

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
        ckpt._ckpt_mid_prefill_store(batch)   # the two adjacent lines from
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
    mtp_prefill._mtp_prefill_init(batch_a)
    assert getattr(batch_a, "_kq_ckpt_armed", False)
    meta = batch_a._apc_meta[0]
    cl = int(meta["checkpoint_len"])
    assert cl == boundary                   # guard-trimmed terminal grid
    batch_a.prompt_cache = make_hybrid_cache(cl, seed=1)
    batch_a._row_real_tokens_processed = lambda idx: cl
    ckpt._ckpt_mid_prefill_store(batch_a)
    assert meta["ckpt_last_stored"] == cl

    # Request B shares the first `boundary` tokens, diverges after.
    ids_b = shared + list(range(700, 700 + 40))
    batch_b = SimpleNamespace(
        model=model, _input_ids=mx.array([ids_b]),
        _inputs_embeds=mx.zeros((1, len(ids_b), 4)),
        prompt_cache=make_hybrid_cache(len(ids_b)), _prompt_kwargs={},
        prefill_step_size=32, _prompt_length_aware_keys=())
    mtp_prefill._mtp_prefill_init(batch_b)
    assert batch_b._mtp_l1_prefix_len == boundary


# -- formation: ckpt-tier models serialize prompt admission --

def test_ckpt_formation_serializes_prefill(monkeypatch):
    """Simultaneous arrivals must not coalesce into a multi-row prompt
    batch on a ckpt-tier model (the owned APC declines B>1 prefill); B=1
    formation converts a burst into 1 cold + N-1 restores. Pure-KV models
    keep batched formation."""
    import importlib

    from test_ckpt_tier import KVCache

    ar = importlib.import_module("gmlx._vendor.mlx_vlm.generate.ar")

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
