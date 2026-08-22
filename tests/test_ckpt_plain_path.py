"""Checkpoint tier on the stock (non-speculative) serving path.

CPU-only unit tests: arming + lookup/trim at prompt-batch init, the
cursor riding the wrapped stock checkpoint store, and B=1 retirement at
generation-batch row exit.
"""

from types import SimpleNamespace

import mlx.core as mx
from mlx_vlm.apc import APCManager

from gmlx import retire_key, spec_engine
from gmlx.cache_snapshot import ckpt_lookup, ckpt_store
from test_ckpt_tier import LAYOUT, make_hybrid_cache

TAGS = tuple("arr" if k == "arr" else "kv" for k in LAYOUT)


def _fake_model():
    return SimpleNamespace(_kq_apc_ckpt=True, _kq_apc_ckpt_layout=TAGS,
                           config=SimpleNamespace())


def _plain_batch(man, ids, model=None):
    n = len(ids)
    meta = {"full_input_ids": list(ids), "prefix_len": 0, "extra_hash": 0,
            "apc_blocks": [], "checkpoint_len": max(0, n - 4)}
    return SimpleNamespace(
        model=model or _fake_model(),
        uids=[0],
        _apc_manager=man,
        _apc_mode="exact",
        _apc_meta=[meta],
        _right_pad_per_row=None,
        _inputs_embeds=mx.zeros((1, n, 4)),
        _input_ids=mx.array([ids]),
        _prompt_kwargs={},
        _prompt_length_aware_keys=[],
        _processed_prompt_columns=0,
        prefill_step_size=16,
        prompt_cache=make_hybrid_cache(2, seed=99),
        _apc_harvest_enabled=True,
    )


def test_plain_init_arms_cold_batch():
    spec_engine._bind_l1_view()
    man = APCManager(num_blocks=64, block_size=16)
    ids = list(range(100, 148))
    b = _plain_batch(man, ids)
    spec_engine._plain_ckpt_init(b)
    meta = b._apc_meta[0]
    assert b._kq_ckpt_armed and b._apc_harvest_enabled is False
    assert meta["ckpt_terminal"] > 0 and meta["checkpoint_len"] > 0
    stash = b.prompt_cache[0]._kq_apc_retire
    assert stash["mode"] == "ckpt" and stash["manager"] is man
    assert stash["snap_ok"] and stash["gen"] == []
    # Ring parameters: chunk-grid anchor, no rotating layers -> align 1.
    assert stash["snap_grid"] == 16 and stash["snap_align"] == 1
    # Cold: no trim happened.
    assert b._processed_prompt_columns == 0
    assert b._input_ids.shape[1] == len(ids)


def test_plain_init_restores_and_trims():
    spec_engine._bind_l1_view()
    man = APCManager(num_blocks=64, block_size=16)
    ids = list(range(300, 396))
    warm_src = make_hybrid_cache(32, seed=5)
    assert ckpt_store(man, ids[:32], warm_src)
    b = _plain_batch(man, ids)
    spec_engine._plain_ckpt_init(b)
    assert b._processed_prompt_columns == 32
    assert b._input_ids.shape[1] == 64
    assert b._inputs_embeds.shape[1] == 64
    for w, o in zip(b.prompt_cache, warm_src):
        if hasattr(o, "keys"):
            assert int(w.offset) == 32
    assert b._kq_ckpt_armed
    # Cursor starts above the restored prefix.
    assert b._apc_meta[0]["checkpoint_len"] > 32


def test_plain_init_leaves_batched_rows_stock():
    spec_engine._bind_l1_view()
    man = APCManager(num_blocks=64, block_size=16)
    ids = list(range(48))
    b = _plain_batch(man, ids)
    b._right_pad_per_row = [0]
    spec_engine._plain_ckpt_init(b)
    assert not getattr(b, "_kq_ckpt_armed", False)
    assert b._apc_harvest_enabled is True


def test_wrapped_stock_store_runs_cursor_and_suppresses_stock():
    from mlx_vlm.generate.ar import PromptProcessingBatch

    spec_engine._install_ckpt_checkpoint_store()
    man = APCManager(num_blocks=64, block_size=16)
    ids = list(range(400, 448))
    meta = {"full_input_ids": ids, "prefix_len": 0, "extra_hash": 0,
            "checkpoint_len": 32, "ckpt_terminal": 32, "ckpt_interval": 0,
            "ckpt_last_stored": 0}

    class Fake:
        model = _fake_model()
        _apc_manager = man
        _apc_mode = "exact"
        _apc_meta = [meta]
        _kq_ckpt_armed = True
        prompt_cache = make_hybrid_cache(32, seed=8)

        def _row_real_tokens_processed(self, i):
            return 32

        def _apc_prompt_cache_for_store(self, i):
            raise AssertionError("stock store must be suppressed")

    PromptProcessingBatch._store_apc_exact_checkpoints(Fake())
    assert meta["ckpt_last_stored"] == 32 and meta["checkpoint_done"]
    _, got = ckpt_lookup(man, ids[:32] + [1], extra_hash=0)
    assert got == 32
    assert man.stats.exact_stores == 0


def test_gen_batch_filter_retires_lone_row(monkeypatch):
    from mlx_vlm.generate.ar import GenerationBatch

    spec_engine._install_plain_ckpt_decode()
    man = APCManager(num_blocks=64, block_size=16)
    full = list(range(500, 532))
    gen = list(range(900, 916))
    cache = make_hybrid_cache(48, seed=11)      # offset == len(seq)
    stash = {"full_ids": full, "extra_hash": 0, "mode": "ckpt",
             "manager": man, "render_ctx": None, "snap_ok": True,
             "gen": gen}
    c0 = cache[0]
    c0._kq_apc_retire = stash
    gb = GenerationBatch.empty(model=None, sampler=None, stop_criteria=None)
    gb.uids = [7]
    gb.prompt_cache = cache
    gb.filter([])
    assert c0._kq_apc_retire is None
    warm, got = ckpt_lookup(man, full + gen + [1], extra_hash=0)
    assert got == 48
    assert not gb.uids and not gb.prompt_cache


def test_gen_batch_retire_uses_decode_snap_on_divergence(monkeypatch):
    from mlx_vlm.generate.ar import GenerationBatch

    spec_engine._install_plain_ckpt_decode()
    man = APCManager(num_blocks=64, block_size=16)
    full = list(range(600, 632))
    gen = list(range(950, 966))
    cache = make_hybrid_cache(48, seed=12)
    snap_states = [c for c in make_hybrid_cache(40, seed=13)
                   if not hasattr(c, "keys")]
    stash = {"full_ids": full, "extra_hash": 0, "mode": "ckpt",
             "manager": man, "render_ctx": {"stub": True}, "snap_ok": True,
             "gen": gen, "snaps": [(40, snap_states)]}
    cache[0]._kq_apc_retire = stash
    monkeypatch.setattr(retire_key, "next_turn_lcp",
                        lambda ctx, seq, g, **kw: 44)
    gb = GenerationBatch.empty(model=None, sampler=None, stop_criteria=None)
    gb.uids = [8]
    gb.prompt_cache = cache
    gb.filter([])
    warm, got = ckpt_lookup(man, full + gen[:8] + [1], extra_hash=0)
    assert got == 40
    for w, s in zip([w for w in warm if not hasattr(w, "keys")],
                    snap_states):
        for a, b in zip(w.cache, s.cache):
            assert mx.array_equal(a, b).item()


def _batched_hybrid_cache(lens, total):
    """LAYOUT-shaped B=len(lens) batch caches: BatchKVCache rows merged
    from different lengths (left-padded), row-dim ArraysCache states."""
    from mlx_vlm.models.cache import ArraysCache, BatchKVCache, KVCache

    H, D = 2, 8
    caches = []
    for i, kind in enumerate(LAYOUT):
        mx.random.seed(7000 + i)
        if kind == "kv":
            rows = []
            for p in lens:
                c = KVCache()
                c.state = (mx.random.normal((1, H, p, D)),
                           mx.random.normal((1, H, p, D)))
                rows.append(c)
            caches.append(BatchKVCache.merge(rows))
        else:
            c = ArraysCache(size=2)
            c.cache = [mx.random.normal((len(lens), 3, D)),
                       mx.random.normal((len(lens), H, D, D))]
            caches.append(c)
    assert max(lens) == total
    return caches


def test_gen_batch_filter_retires_leaving_row_batched():
    from mlx_vlm.generate.ar import GenerationBatch

    spec_engine._install_plain_ckpt_decode()
    man = APCManager(num_blocks=64, block_size=16)
    full = list(range(700, 724))
    gen = list(range(970, 986))                 # row 1: 24 + 16 = 40
    stash = {"full_ids": full, "extra_hash": 0, "mode": "ckpt",
             "manager": man, "render_ctx": None, "snap_ok": True,
             "gen": gen}
    gb = GenerationBatch.empty(model=None, sampler=None, stop_criteria=None)
    gb.uids = [7, 8]
    gb.max_tokens = [64, 64]
    gb._num_tokens = [0, 0]
    gb.prompt_cache = _batched_hybrid_cache([48, 40], 48)
    gb._kq_apc_retire_rows = {8: stash}
    gb.filter([0])
    assert gb.uids == [7]
    assert gb._kq_apc_retire_rows == {}
    warm, got = ckpt_lookup(man, full + gen + [1], extra_hash=0)
    assert got == 40


def test_gen_batch_extend_carries_stash():
    from mlx_vlm.generate.ar import GenerationBatch

    spec_engine._install_plain_ckpt_decode()
    stash = {"full_ids": [1, 2], "mode": "ckpt", "gen": []}
    other = GenerationBatch.empty(model=None, sampler=None,
                                  stop_criteria=None)
    other.uids = [9]
    other.prompt_cache = make_hybrid_cache(8, seed=21)
    other.prompt_cache[0]._kq_apc_retire = stash
    gb = GenerationBatch.empty(model=None, sampler=None, stop_criteria=None)
    gb.extend(other)
    assert gb._kq_apc_retire_rows == {9: stash}
    assert other.prompt_cache[0]._kq_apc_retire is None


def test_exact_anchor_init_arms_retire_stash():
    spec_engine._bind_l1_view()
    man = APCManager(num_blocks=64, block_size=16)
    ids = list(range(800, 848))
    model = SimpleNamespace(_kq_apc_ckpt=False, config=SimpleNamespace())
    b = _plain_batch(man, ids, model=model)
    spec_engine._plain_anchor_init(b)
    stash = b.prompt_cache[0]._kq_apc_retire
    assert stash["mode"] == "exact" and stash["manager"] is man
    assert stash["full_ids"] == ids and stash["gen"] == []


def test_gen_batch_filter_retires_exact_row():
    from mlx_vlm.generate.ar import GenerationBatch
    from mlx_vlm.models.cache import KVCache

    spec_engine._install_plain_ckpt_decode()
    man = APCManager(num_blocks=64, block_size=16)
    full = list(range(850, 882))
    gen = list(range(990, 1006))                # 32 + 16 = 48
    caches = []
    for i in range(2):
        mx.random.seed(8100 + i)
        c = KVCache()
        c.state = (mx.random.normal((1, 2, 48, 8)),
                   mx.random.normal((1, 2, 48, 8)))
        caches.append(c)
    stash = {"full_ids": full, "extra_hash": 0, "mode": "exact",
             "manager": man, "render_ctx": None, "gen": gen}
    caches[0]._kq_apc_retire = stash
    gb = GenerationBatch.empty(model=None, sampler=None, stop_criteria=None)
    gb.uids = [11]
    gb.prompt_cache = caches
    gb.filter([])
    warm, got = man.lookup_exact_cache(full + gen + [1], extra_hash=0)
    assert got == 48 and warm is not None


def test_plain_step_tick_disables_on_failure():
    # Runs per token: a deterministic failure must strike once, not
    # emit a traceback per step.
    stash = {"mode": "ckpt", "gen": 7}        # broken accounting slot
    gb = SimpleNamespace(
        uids=[3], prompt_cache=[SimpleNamespace(_kq_apc_retire=stash)])
    spec_engine._plain_step_tick(gb, [[5]])
    assert stash["snap_ok"] is False and "gen" not in stash
    spec_engine._plain_step_tick(gb, [[6]])   # gated off: silent no-op
    assert "gen" not in stash


def test_render_memo_preprocess_failure_is_safe():
    # A dead generator behind the memo's weakref must degrade to "no
    # prediction", never break retirement.
    ctx = {"render": lambda *a, **k: "text",
           "preprocess": None,
           "processor": SimpleNamespace(
               tokenizer=SimpleNamespace(
                   decode=lambda t, **k: "x", eos_token=None)),
           "config": None, "messages": [], "kw": {}}

    def dead(text):
        raise RuntimeError("retire render: generator unloaded")

    ctx["preprocess"] = dead
    assert retire_key.next_turn_lcp(ctx, [1, 2, 3], [3]) is None
