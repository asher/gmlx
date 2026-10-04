"""Checkpoint tier on the stock (non-speculative) serving path.

CPU-only unit tests: arming + lookup/trim at prompt-batch init, the
cursor riding the wrapped stock checkpoint store, and B=1 retirement at
generation-batch row exit.
"""

import logging
from types import SimpleNamespace

import mlx.core as mx
from mlx_vlm.apc import APCManager

import gmlx.cache.retire_key as retire_key
import gmlx.spec.ckpt as ckpt
import gmlx.spec.engine as engine
from gmlx.cache.snapshot import ckpt_lookup, ckpt_store
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
    engine._bind_l1_view()
    man = APCManager(num_blocks=64, block_size=16)
    ids = list(range(100, 148))
    b = _plain_batch(man, ids)
    ckpt._plain_ckpt_init(b)
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
    engine._bind_l1_view()
    man = APCManager(num_blocks=64, block_size=16)
    ids = list(range(300, 396))
    warm_src = make_hybrid_cache(32, seed=5)
    assert ckpt_store(man, ids[:32], warm_src)
    b = _plain_batch(man, ids)
    ckpt._plain_ckpt_init(b)
    assert b._processed_prompt_columns == 32
    assert b._input_ids.shape[1] == 64
    assert b._inputs_embeds.shape[1] == 64
    for w, o in zip(b.prompt_cache, warm_src):
        if hasattr(o, "keys"):
            assert int(w.offset) == 32
    assert b._kq_ckpt_armed
    # Cursor starts above the restored prefix.
    assert b._apc_meta[0]["checkpoint_len"] > 32


def test_plain_init_trims_mrope_position_ids_on_their_sequence_axis():
    """An image turn carries (3, B, L) position ids. The restore trims
    them on the last axis, as the stock trims do."""
    engine._bind_l1_view()
    man = APCManager(num_blocks=64, block_size=16)
    ids = list(range(500, 596))
    assert ckpt_store(man, ids[:32], make_hybrid_cache(32, seed=7))
    pos = mx.stack([mx.arange(96) + 1000 * k for k in (1, 2, 3)])[:, None]
    mask = mx.ones((1, 96), dtype=mx.int32)
    b = _plain_batch(man, ids)
    b._prompt_kwargs = {"position_ids": pos, "attention_mask": mask}
    b._prompt_length_aware_keys = ["position_ids", "attention_mask"]
    ckpt._plain_ckpt_init(b)
    assert b._processed_prompt_columns == 32
    assert b._prompt_kwargs["position_ids"].tolist() == pos[:, :, 32:].tolist()
    assert b._prompt_kwargs["attention_mask"].shape == (1, 64)


def _progress_batch(b, prompt_tokens):
    """The fields the stock ``prompt_progress`` reads, as the stock init
    and the prefill timer set them, and the row ids emptied as the stock
    ``generate`` leaves them before the server reads the progress."""
    b._prompt_uids = list(b.uids)
    b._prompt_tokens_per_row = [prompt_tokens] * len(b.uids)
    b._cached_tokens_per_row = [0] * len(b.uids)
    b._prompt_time_s = 0.5
    b.uids = []
    return b


def _prefill_line(caplog, progress):
    from mlx_vlm.server.generation import ResponseGenerator

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="mlx_vlm.server"):
        ResponseGenerator._log_prefill_completed(
            progress.uid, {"request_id": "r"}, progress)
    return [m for m in caplog.messages if "Prefill completed" in m][0]


def test_prefill_line_counts_the_checkpoint_restore(caplog):
    """A checkpoint restore keeps stock's per-row cached count at 0 (the
    prompt is trimmed in place), and the usage and the "Prefill
    completed" line report the restored tokens."""
    from mlx_vlm.generate.ar import PromptProcessingBatch

    from gmlx.spec.mtp_prefill import install_full_prompt_mtp_prefill

    install_full_prompt_mtp_prefill()
    engine._bind_l1_view()
    man = APCManager(num_blocks=64, block_size=16)
    ids = list(range(700, 796))
    assert ckpt_store(man, ids[:32], make_hybrid_cache(32, seed=11))
    b = _plain_batch(man, ids)
    ckpt._plain_ckpt_init(b)
    assert b._processed_prompt_columns == 32
    (progress,) = PromptProcessingBatch.prompt_progress(_progress_batch(b, 96))
    assert progress.cached_tokens == 32
    assert "prompt_tokens=96 cached_tokens=32 " in _prefill_line(caplog, progress)


def test_prefill_line_counts_the_mtp_prefix_restore(caplog):
    """The owned MTP ladder records its L0 or L1 restore the same way. A
    batch of two rows keeps the stock counts."""
    from mlx_vlm.generate.ar import PromptProcessingBatch

    from gmlx.spec.mtp_prefill import install_full_prompt_mtp_prefill

    install_full_prompt_mtp_prefill()
    b = _progress_batch(SimpleNamespace(uids=[4], _mtp_apc_prefix_len=40), 64)
    (progress,) = PromptProcessingBatch.prompt_progress(b)
    assert "prompt_tokens=64 cached_tokens=40 " in _prefill_line(caplog, progress)
    two = _progress_batch(SimpleNamespace(
        uids=[4, 5], _mtp_apc_prefix_len=40,
        _kq_apc_restored=(40, "ckpt")), 64)
    two._cached_tokens_per_row = [16, 0]
    rows = PromptProcessingBatch.prompt_progress(two)
    assert [r.cached_tokens for r in rows] == [16, 0]


def test_plain_init_leaves_batched_rows_stock():
    engine._bind_l1_view()
    man = APCManager(num_blocks=64, block_size=16)
    ids = list(range(48))
    b = _plain_batch(man, ids)
    b._right_pad_per_row = [0]
    ckpt._plain_ckpt_init(b)
    assert not getattr(b, "_kq_ckpt_armed", False)
    assert b._apc_harvest_enabled is True


def test_plain_init_skips_a_quantized_live_cache(caplog):
    """--kv-bits builds affine-quantized layers, which the tier cannot
    store. The batch is not armed, nothing stores, and the reason is
    logged once per model, not the zero-stores tripwire."""
    import logging

    from mlx_vlm.models.cache import QuantizedKVCache

    engine._bind_l1_view()
    man = APCManager(num_blocks=64, block_size=16)
    model = _fake_model()
    with caplog.at_level(logging.WARNING, logger="gmlx.spec.ckpt"):
        for _ in range(8):
            b = _plain_batch(man, list(range(48)), model=model)
            b.prompt_cache = [QuantizedKVCache(group_size=64, bits=8)
                              if hasattr(c, "keys") else c
                              for c in b.prompt_cache]
            ckpt._plain_ckpt_init(b)
            assert not getattr(b, "_kq_ckpt_armed", False)
            assert b._apc_harvest_enabled is False
    lines = [r.getMessage() for r in caplog.records]
    assert len([m for m in lines if "ckpt tier off" in m]) == 1
    assert "QuantizedKVCache" in lines[0] and "kvarn" in lines[0]
    assert not any("tripwire" in m for m in lines)


def test_wrapped_stock_store_runs_cursor_and_suppresses_stock():
    from mlx_vlm.generate.ar import PromptProcessingBatch

    ckpt._install_ckpt_checkpoint_store()
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

        _apc_uses_checkpoints = PromptProcessingBatch._apc_uses_checkpoints
        _apc_pending_checkpoint = PromptProcessingBatch._apc_pending_checkpoint

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

    ckpt._install_plain_ckpt_decode()
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

    ckpt._install_plain_ckpt_decode()
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

    ckpt._install_plain_ckpt_decode()
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

    ckpt._install_plain_ckpt_decode()
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
    engine._bind_l1_view()
    man = APCManager(num_blocks=64, block_size=16)
    ids = list(range(800, 848))
    model = SimpleNamespace(_kq_apc_ckpt=False, config=SimpleNamespace())
    b = _plain_batch(man, ids, model=model)
    ckpt._plain_anchor_init(b)
    stash = b.prompt_cache[0]._kq_apc_retire
    assert stash["mode"] == "exact" and stash["manager"] is man
    assert stash["full_ids"] == ids and stash["gen"] == []


def test_gen_batch_filter_retires_exact_row():
    from mlx_vlm.generate.ar import GenerationBatch
    from mlx_vlm.models.cache import KVCache

    ckpt._install_plain_ckpt_decode()
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
    ckpt._plain_step_tick(gb, [[5]])
    assert stash["snap_ok"] is False and "gen" not in stash
    ckpt._plain_step_tick(gb, [[6]])   # gated off: silent no-op
    assert "gen" not in stash


def test_plain_step_tick_reads_step_tuple_rows():
    # _step returns (tokens, lps, top_idx, top_lp); slot 0 is the flat
    # per-row token list. Rows past the first must accrue, and a None
    # emission skips that tick without disabling the row (the v1 tick
    # indexed the tuple itself, killing every row past the first).
    s1 = {"mode": "exact", "gen": []}
    s2 = {"mode": "exact", "gen": []}
    gb = SimpleNamespace(uids=[1, 2], prompt_cache=[SimpleNamespace()],
                         _kq_apc_retire_rows={1: s1, 2: s2})
    ckpt._plain_step_tick(gb, ([5, 9], None, None, None))
    assert s1["gen"] == [5] and s2["gen"] == [9]
    ckpt._plain_step_tick(gb, ([6, None], None, None, None))
    assert s1["gen"] == [5, 6] and s2["gen"] == [9]
    assert s2.get("snap_ok") is not False
    ckpt._plain_step_tick(gb, (None, None, None, None))
    assert s1["gen"] == [5, 6] and s2["gen"] == [9]


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
