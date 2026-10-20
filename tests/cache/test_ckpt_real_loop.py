"""The streamed checkpoint schedule on the real prefill loops.

Each prompt runs through the stock prompt_step under the decay wrapper
(plain and folding) or the owned MTP body, with the schedule the branch
arms and with main's: one grid from 0 at the batch's step, unpriced. The
schedule prices its candidates with a model of these loops, so the model
must give the chunks the loops run, and the pick must never cost more
expert passes or partial calls than main's schedule.
"""

from types import SimpleNamespace

import pytest

import gmlx.gen.prefill_decay as pd
import gmlx.spec.ckpt as ckpt
import prefill_loop
from prefill_loop import Model, cost, install, run

ROT = ("rot:2048:0", "kv")
KVARN = ("rot:2048:0", "kvarn:4:4:64")
# Score caps (GB) that keep the step flat, decay it with depth on a dense
# model, and refit it under a fused profile.
CONFIGS = {"flat": ("1000", "dense_probe"), "dense": ("5.4", "dense_probe"),
           "fused": ("2", "fused_probe")}
LOOPS = (("plain", False, False), ("fold", True, False), ("mtp", False, True))
_KIND = {k: (folds, mtp) for k, folds, mtp in LOOPS}
_FUSED = pd.ScoreTransientProfile(heads=32, bytes_per_elem=2,
                                  fused_min_step=1024, fused_mask_bytes=0)


@pytest.fixture
def loop(monkeypatch):
    state, stored = {"p": None, "sys": None}, []
    install(monkeypatch, state, stored)
    monkeypatch.delenv("PREFILL_STEP_SIZE", raising=False)
    monkeypatch.setitem(pd._SCORE_PROFILES, "fused_probe",
                        lambda m, c: _FUSED)

    def model(cfg, folds=False, tags=ROT):
        cap, mt = CONFIGS[cfg]
        monkeypatch.setenv("GMLX_PREFILL_SCORE_CAP_GB", cap)
        return Model(tags, folds=folds, mt=mt)

    def go(m, n, restored, p=None, sys=None, mtp=False, arm=None):
        state.update(p=p, sys=sys)
        stored.clear()
        return run(monkeypatch, m, n, restored, mtp=mtp, arm=arm)

    return SimpleNamespace(model=model, run=go, stored=stored)


def main_arm(batch, meta, guard, restored):
    """Main's schedule: its one grid, from 0 at the batch's step."""
    real = ckpt._ckpt_grids

    def one(*a):
        return real(*a)[0][-1:], False

    ckpt._ckpt_grids = one
    try:
        ckpt._ckpt_arm_schedule(batch, meta, guard, restored, 16)
    finally:
        ckpt._ckpt_grids = real


def _bounds(out):
    return out["meta"]["ckpt_boundaries"]


@pytest.mark.parametrize("cfg,loop_kind,restored,n,p,sys,want", [
    ("dense", "fold", 2999, 19283, 19269, 6000,
     [6000, 8192, 16384, 19269, 19282]),
    ("flat", "mtp", 7299, 24783, 24769, None, [24769, 24782]),
    ("flat", "plain", 7299, 15565, 15551, None, [8192, 15551, 15564]),
    ("flat", "mtp", 0, 8492, None, None, [8491]),
    ("fused", "fold", 30000, 39217, 39203, None, [32768, 39203, 39216]),
    ("dense", "mtp", 0, 40000, 39986, 3000,
     [2992, 8192, 16384, 24576, 32768, 39986, 39999]),
])
def test_main_arm_is_mains_schedule(loop, cfg, loop_kind, restored, n, p,
                                    sys, want):
    """The schedules main (f8264fd2) arms on these inputs, so the
    comparisons below are against main and not against the branch's
    own rules."""
    folds, mtp = _KIND[loop_kind]
    out = loop.run(loop.model(cfg, folds), n, restored, p, sys, mtp,
                   arm=main_arm)
    assert [b for b, _ in _bounds(out)] == want


def test_loop_model_is_exact_and_no_schedule_costs_more_than_mains(loop):
    """On every loop and decay setting, with and without a turn record or
    a system prefix, the chunks the schedule priced are the chunks the
    loop runs, and the schedule costs no more than main's."""
    fewer = runs = 0
    for cfg in CONFIGS:
        for _, folds, mtp in LOOPS:
            m = loop.model(cfg, folds)
            for r in (0, 2999, 7299, 30000):
                for extra in (500, 8300, 9278, 17484, 9217):
                    n = r + extra
                    for p in (None, n - 14):
                        for sys in (None, 3000):
                            head = loop.run(m, n, r, p, sys, mtp)
                            assert head["pred"] == head["chunks"], (
                                cfg, folds, mtp, r, n, p, sys)
                            main = loop.run(m, n, r, p, sys, mtp,
                                            arm=main_arm)
                            a, b = cost(head["chunks"]), cost(main["chunks"])
                            assert a <= b, (cfg, folds, mtp, r, n, p, sys)
                            fewer += a[0] < b[0]
                            runs += 1
    assert runs == 720
    assert fewer > 150


@pytest.mark.parametrize("cfg,loop_kind,restored,n,p,sys,chunks", [
    # The restored prefix's grid, where the anchor's crosses the decay
    # depth.
    ("dense", "fold", 2999, 19283, 19269, 6000, [3001, 5191, 8078, 13, 1]),
    # The MTP body's first chunk is sized before the restore.
    ("flat", "mtp", 7299, 16591, 16577, None, [9278, 13, 1]),
    # Two points cost a pass only together.
    ("flat", "mtp", 7299, 24783, 24769, None, [8742, 8728, 13, 1]),
    ("dense", "plain", 7299, 23782, 23768, None,
     [8241, 4096, 4096, 36, 13, 1]),
    # A tie that would leave a 60-token partial call.
    ("flat", "plain", 7299, 15565, 15551, None, [8252, 13, 1]),
    # A text prompt with no turn record keeps no grid point.
    ("flat", "mtp", 0, 8492, None, None, [8491, 1]),
    ("flat", "plain", 0, 9000, None, None, [8999, 1]),
    # Main's grid, where the restored prefix's splits the tail at depth.
    ("fused", "fold", 30000, 39217, 39203, None, [2768, 6435, 13, 1]),
])
def test_streamed_schedule_takes_the_cheapest_grid(loop, cfg, loop_kind,
                                                   restored, n, p, sys,
                                                   chunks):
    folds, mtp = _KIND[loop_kind]
    out = loop.run(loop.model(cfg, folds), n, restored, p, sys, mtp)
    assert out["chunks"] == chunks


def test_pick_keeps_the_record_the_next_turn_restores(loop):
    """Where a wide window keeps the exact turn pause out, the next turn
    restores the deepest record at or below its stable prefix. The pick
    keeps that record where main's schedule has it, so a pass saved now
    is not paid back on the next turn."""
    for tags in (("rot:32768:0", "kv"), ("rot:16384:0", "kv")):
        for _, folds, mtp in LOOPS:
            m = loop.model("dense", folds, tags)
            for r in (0, 4912, 13104, 21296):
                for extra in range(1500, 24000, 1622):
                    n = r + extra
                    p = n - 14
                    head = loop.run(m, n, r, p, mtp=mtp)
                    main = loop.run(m, n, r, p, mtp=mtp, arm=main_arm)

                    def deep(out):
                        return max([b for b, _ in _bounds(out) if b <= p],
                                   default=0)

                    assert deep(head) == deep(main), (tags, mtp, r, n)
    out = loop.run(loop.model("dense", tags=("rot:32768:0", "kv")), 8845,
                   4912, 8831)
    assert _bounds(out) == [(8192, "boundary")]
    assert out["chunks"] == [3280, 652, 1]


def test_a_cap_that_follows_headroom_keeps_mains_schedule(loop,
                                                          monkeypatch):
    """With GMLX_PREFILL_HEADROOM_CAP=1 the score cap falls as the
    prefill's KV grows, so the loop's chunks are not known at arm time
    and main's schedule stays."""
    m = loop.model("dense", folds=True)
    monkeypatch.delenv("GMLX_PREFILL_SCORE_CAP_GB")
    monkeypatch.setenv("GMLX_PREFILL_HEADROOM_CAP", "1")
    monkeypatch.setattr(pd, "_cap_bytes", lambda: 5.5e9)
    monkeypatch.setattr(pd, "_NOTED_CACHE_LIMIT", 40e9)
    live = {}
    make = prefill_loop.make_batch

    def made(*a, **k):
        live["b"] = make(*a, **k)
        return live["b"]

    monkeypatch.setattr(prefill_loop, "make_batch", made)
    monkeypatch.setattr(pd, "_headroom_bytes", lambda: 60e9 - 200e3 * max(
        c.offset for c in live["b"].prompt_cache))
    head = loop.run(m, 81666, 20000)
    main = loop.run(m, 81666, 20000, arm=main_arm)
    assert _bounds(head) == _bounds(main)
    assert cost(head["chunks"]) == cost(main["chunks"]) == (8, 0)
    # With no cache limit noted the cap stays fixed, so the schedule is
    # priced again.
    monkeypatch.setattr(pd, "_NOTED_CACHE_LIMIT", None)
    head = loop.run(m, 24783, 7299)
    main = loop.run(m, 24783, 7299, arm=main_arm)
    assert cost(head["chunks"]) < cost(main["chunks"])


def test_kvarn_disk_skeletons_go_where_mains_do(loop, monkeypatch):
    """A kvarn layout writes disk skeletons for main's terminal record and
    the turn records past it. Where the pick drops one, the next kept
    record takes it, the replay included, which a longer next prompt
    adopts. With no kept record past it, the deepest kept record takes
    it. No other record gets one."""
    m = loop.model("flat", tags=KVARN)
    loop.run(m, 8492, 0, arm=main_arm)
    assert loop.stored == [(8192, "boundary", True),
                           (8491, "replay", False)]
    loop.run(m, 8492, 0)
    assert loop.stored == [(8491, "replay", True)]
    loop.run(m, 9000, 0, sys=3000)
    assert loop.stored == [(2992, "anchor", False), (8999, "replay", True)]
    loop.run(m, 15565, 7299, 15551)
    assert loop.stored == [(15551, "boundary", True),
                           (15564, "replay", False)]
    fold = loop.model("flat", folds=True, tags=KVARN)
    loop.run(fold, 9217, 0, 8917, sys=1000)
    assert loop.stored == [(992, "anchor", False), (8917, "boundary", True),
                           (9216, "replay", False)]
    loop.run(fold, 8492, 0, 8192, sys=1000)
    assert loop.stored == [(992, "anchor", False), (8192, "boundary", True),
                           (8491, "replay", False)]
    loop.run(m, 8257, 0, 7957, mtp=True)
    assert loop.stored == [(7957, "boundary", True),
                           (8256, "replay", False)]
    monkeypatch.setenv("GMLX_APC_CKPT_REPLAY", "0")
    loop.run(m, 8257, 0, 7957, sys=1000)
    assert loop.stored == [(992, "anchor", False), (7957, "boundary", True)]


def test_kvarn_restart_restores_as_deep_as_mains_with_no_more_skeletons(
        loop):
    """After a restart only disk skeletons survive, so the next turn
    restores the deepest one at or below its stable prefix. The pick
    writes no more heavy skeletons than main, and that restore is never
    shallower, also when the next turn re-renders more than the prompt's
    last 16 tokens."""

    def skel():
        return [b for b, _, s in loop.stored if s]

    runs = 0
    for cfg in CONFIGS:
        for _, folds, mtp in LOOPS:
            m = loop.model(cfg, folds, KVARN)
            for r in (0, 7299):
                for extra in (500, 8300, 9217, 17484):
                    n = r + extra
                    for tail in (14, 300):
                        for sys in (None, 1000):
                            p = n - tail
                            loop.run(m, n, r, p, sys, mtp)
                            head = skel()
                            loop.run(m, n, r, p, sys, mtp, arm=main_arm)
                            main = skel()
                            assert len(head) <= len(main), (
                                cfg, folds, mtp, r, n, tail, sys)
                            assert max([b for b in head if b <= p],
                                       default=0) >= max(
                                [b for b in main if b <= p], default=0), (
                                cfg, folds, mtp, r, n, tail, sys)
                            runs += 1
    assert runs == 288


def test_turn_records_off_keep_mains_schedule(loop, monkeypatch):
    """With GMLX_APC_CKPT_TURN=0 the pick cannot see which record the next
    turn restores, and dropping the terminal there can cost the next turn
    more than this one saves. Main's schedule stays."""
    monkeypatch.setenv("GMLX_APC_CKPT_TURN", "0")
    for cfg in CONFIGS:
        for _, folds, mtp in LOOPS:
            m = loop.model(cfg, folds)
            for r, n in ((0, 24627), (7299, 15599), (7299, 24783)):
                head = loop.run(m, n, r, mtp=mtp)
                main = loop.run(m, n, r, mtp=mtp, arm=main_arm)
                assert _bounds(head) == _bounds(main), (cfg, folds, mtp, n)
                assert head["chunks"] == main["chunks"]
    out = loop.run(loop.model("flat", folds=True), 24627, 0)
    assert cost(out["chunks"]) == (3, 1)


def test_the_whole_prompt_store_is_skipped_where_mains_is(loop):
    """After prefill the whole prompt is stored unless a record the next
    turn restores landed. Where a wide window keeps the pick's grid from
    a turn point, its deepest record at or below the stable prefix
    counts, so a kvarn layout writes no heavy whole-prompt record that
    main skips."""
    from gmlx.cache.snapshot import ckpt_full_store_redundant

    for folds in (False, True):
        m = loop.model("flat", folds, ("rot:32768:0", "kvarn:4:4:64"))
        main = loop.run(m, 17408, 0, 15908, 12000, arm=main_arm)
        assert main["meta"]["ckpt_p_stable_bounds"] == [8192]
        assert ckpt_full_store_redundant(main["meta"])
        head = loop.run(m, 17408, 0, 15908, 12000)
        assert loop.stored == [(12000, "anchor", True)]
        assert head["meta"]["ckpt_p_stable_bounds"] == [12000]
        assert ckpt_full_store_redundant(head["meta"])
        # Where main's has no turn record either, both store it.
        main = loop.run(m, 17408, 0, 7000, 3000, arm=main_arm)
        assert not ckpt_full_store_redundant(main["meta"])
        head = loop.run(m, 17408, 0, 7000, 3000)
        assert not ckpt_full_store_redundant(head["meta"])


def test_no_turn_record_and_no_replay_keep_mains_schedule(loop,
                                                          monkeypatch):
    """A prompt with no render-stable prefix gets no turn record, and one
    shorter than a rotating window gets no replay record. Then nothing
    shows which record a longer next prompt restores, so main's schedule
    stays."""
    for tags in (("rot:32768:0", "kv"), ("rot:131072:0", "kv"),
                 ("rot:32768:0", "kvarn:4:4:64")):
        for cfg in CONFIGS:
            for _, folds, mtp in LOOPS:
                m = loop.model(cfg, folds, tags)
                for r, n in ((7299, 17484), (13104, 30000), (21808, 26657)):
                    head = loop.run(m, n, r, mtp=mtp)
                    main = loop.run(m, n, r, mtp=mtp, arm=main_arm)
                    assert "replay" not in dict(_bounds(main)).values()
                    assert _bounds(head) == _bounds(main), (
                        tags, cfg, folds, mtp, n)
                    assert head["chunks"] == main["chunks"]
    out = loop.run(loop.model("dense", True, ("rot:32768:0", "kv")),
                   26657, 21808)
    assert _bounds(out) == [(24576, "boundary")]
    monkeypatch.setenv("GMLX_APC_CKPT_REPLAY", "0")
    loop.run(loop.model("flat", tags=KVARN), 9000, 0, sys=3000)
    assert loop.stored == [(2992, "anchor", False), (8192, "boundary", True)]


def test_a_pricing_failure_arms_mains_schedule(loop, monkeypatch):
    """If pricing the candidates fails, main's schedule is armed whole:
    its records, skeletons and turn records."""
    m = loop.model("flat", tags=KVARN)
    main = loop.run(m, 15565, 7299, 15551, arm=main_arm)
    want = list(loop.stored)

    def fail(*a, **k):
        raise RuntimeError("pricing")

    monkeypatch.setattr(ckpt, "_ckpt_drop_tail_cuts", fail)
    head = loop.run(m, 15565, 7299, 15551)
    assert head["meta"] == main["meta"]
    assert head["chunks"] == main["chunks"]
    assert loop.stored == want
