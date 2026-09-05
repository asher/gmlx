"""Boot capacity table: frontier math, ceilings, gates, consumers."""

import pytest

import gmlx.serve.capacity as cap

GB = 1e9

CFG = {
    "num_hidden_layers": 10,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "head_dim": 64,
    "max_position_embeddings": 65536,
}
BPT = 2 * 8 * 64 * 2 * 10  # 20480 B/token across layers


@pytest.fixture
def rig(monkeypatch, tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"x")
    import gmlx.commands.tool_preflight as tp

    box = {"ws": 20.0 * GB, "max_buffer": 0.0, "resource_limit": 499000}

    def set_rig(weights_gb: float, ws_gb: float = 20.0,
                max_buffer_gb: float = 0.0, cfg=CFG):
        box["ws"] = ws_gb * GB
        box["max_buffer"] = max_buffer_gb * GB
        monkeypatch.setattr(tp, "_shards", lambda p: [str(f)])
        monkeypatch.setattr(cap.os.path, "getsize",
                            lambda p: int(weights_gb * GB))
        monkeypatch.setattr(tp, "_synth_config", lambda p: dict(cfg))
        monkeypatch.setattr(cap, "working_set_bytes", lambda: box["ws"])
        import mlx.core as mx
        monkeypatch.setattr(mx, "device_info", lambda: {
            "max_recommended_working_set_size": box["ws"],
            "max_buffer_length": box["max_buffer"],
            "resource_limit": box["resource_limit"]})
        import gmlx.serve.memory as sm
        monkeypatch.setattr(sm, "admit_reserve_bytes",
                            lambda ws, gen=None: 1.0 * GB)
        return str(f)

    monkeypatch.delenv("GMLX_OVERCOMMIT", raising=False)
    yield set_rig
    cap.clear_table()


def test_table_math_and_monotone_frontier(rig):
    path = rig(weights_gb=10.0, ws_gb=20.0)
    t = cap.derive_table(path)
    # budget 19, minus weights 10 and reserve 1 leaves 8 GB for
    # w*kv(d) + transient(d)
    assert t["budget_bytes"] == int(20.0 * GB * 0.95)
    ctx = t["max_ctx"]
    assert all(ctx[a] >= ctx[b] for a, b in zip((1, 2, 4, 8, 16),
                                                (2, 4, 8, 16, 32)))
    d1 = ctx[1]
    kv = d1 * BPT
    tr = 8 * 2048 * (d1 + 2048) * 2.0
    assert 10.0 * GB + kv + tr + 1.0 * GB <= t["budget_bytes"]
    assert t["max_width_at_depth"][4096] >= 1
    assert t["trained_ctx"] == 65536


def test_max_buffer_ceiling_binds(rig):
    # Generous bytes but a tiny max single buffer. With heads present
    # the priced score transient (also one buffer) binds first; without
    # them one layer's cache tensor (bpt_layer=2048 B/token) caps
    # context at buffer/2048, halved again at width 2.
    headless = {k: v for k, v in CFG.items()
                if k != "num_attention_heads"}
    headless["num_key_value_heads"] = 8
    path = rig(weights_gb=1.0, ws_gb=100.0, max_buffer_gb=0.1,
               cfg=headless)
    t = cap.derive_table(path)
    assert t["max_ctx"][1] == int(0.1 * GB / 2048)
    assert t["max_ctx"][2] == int(0.1 * GB / 4096)

    path = rig(weights_gb=1.0, ws_gb=100.0, max_buffer_gb=0.1)
    t = cap.derive_table(path)
    # transient bound: 8 heads x d x 2d x 2 <= buffer for d < step
    assert t["max_ctx"][1] == 1767


def test_kv_env_prices_the_table(rig, monkeypatch):
    # The table must price kv-quantized layers like request admission
    # does, or the saved bytes never widen the ceilings.
    monkeypatch.delenv("KV_BITS", raising=False)
    path = rig(weights_gb=10.0, ws_gb=20.0)
    base = cap.derive_table(path)
    kv8 = cap.derive_table(path, env={"KV_BITS": "8"})
    assert kv8["max_ctx"][1] > base["max_ctx"][1]
    # admission prices batched mode: MTP runs fp16 KV when batched
    mtp = cap.derive_table(
        path, env={"KV_BITS": "8", "MLX_VLM_GGUF_SPECULATIVE": "1"})
    assert mtp["max_ctx"] == base["max_ctx"]
    # upstream drops kv quantization for qat-marked ids
    qat = cap.derive_table(path + "-qat", env={"KV_BITS": "8"})
    assert qat["max_ctx"] == base["max_ctx"]
    # malformed widths keep fp16 pricing
    bad = cap.derive_table(path, env={"KV_BITS": "5"})
    assert bad["max_ctx"] == base["max_ctx"]


def test_boot_refusal_with_numbers(rig):
    path = rig(weights_gb=19.0, ws_gb=20.0)  # width 1 fits nothing
    with pytest.raises(RuntimeError, match="cannot fit at width 1"):
        cap.install_boot_table(path, None, "m")
    with pytest.raises(RuntimeError, match="19.0 GB"):
        cap.install_boot_table(path, None, "m")


def test_overcommit_disables_refusal_and_ceilings(rig, monkeypatch):
    path = rig(weights_gb=19.0, ws_gb=20.0)
    monkeypatch.setenv("GMLX_OVERCOMMIT", "1")
    t = cap.install_boot_table(path, None, "m")
    assert t is not None and t["overcommit"] is True
    assert cap.frontier_width() is None       # derived ceilings off


def test_frontier_width_and_decode_bound(rig, monkeypatch):
    path = rig(weights_gb=10.0, ws_gb=20.0)
    t = cap.install_boot_table(path, None, "m")
    widths = [w for w in (1, 2, 4, 8, 16, 32)
              if t["max_ctx"][w] >= 4096]
    assert cap.frontier_width() == max(widths)

    from gmlx.serve.decode_batch import decode_batch
    monkeypatch.delenv("GMLX_DECODE_BATCH", raising=False)
    monkeypatch.setattr(cap, "frontier_width", lambda min_ctx=4096: 2)
    assert decode_batch() == 2
    monkeypatch.setenv("GMLX_DECODE_BATCH", "6")
    assert decode_batch() == 6                # explicit env wins


def test_preload_gate(rig, monkeypatch):
    rig(weights_gb=10.0, ws_gb=20.0)
    import gmlx.gen.prefill_decay as pd
    monkeypatch.setattr(pd, "headroom_bytes", lambda: 5.0 * GB)
    monkeypatch.setattr(cap, "working_budget_bytes", lambda: 20.0 * GB)
    monkeypatch.setattr(cap, "_kernel_gate", lambda w, m: None)

    cap.preload_gate(4.0 * GB, "small")       # fits: no raise
    with pytest.raises(cap.LoadDeferred, match="deferred"):
        cap.preload_gate(10.0 * GB, "busybox")           # typed: the 503 seam
    with pytest.raises(RuntimeError, match="does not fit"):
        cap.preload_gate(25.0 * GB, "huge")
    monkeypatch.setenv("GMLX_OVERCOMMIT", "1")
    cap.preload_gate(25.0 * GB, "huge")       # override skips


def test_preload_gate_judges_the_serve_ceiling(rig, monkeypatch):
    # headroom_bytes() is measured against the full working set; the gate
    # takes the margin + reserve off it (the ceiling everything else uses).
    # 2026-08-25: judged raw, 86.7 GB was admitted next to a pinned 31.5 GB
    # resident on a 112 GB wire limit and Metal OOM'd.
    rig(weights_gb=10.0, ws_gb=20.0)
    import gmlx.gen.prefill_decay as pd
    monkeypatch.setattr(pd, "headroom_bytes", lambda: 5.0 * GB)
    monkeypatch.setattr(cap, "working_budget_bytes", lambda: 18.0 * GB)
    monkeypatch.setattr(cap, "_kernel_gate", lambda w, m: None)
    cap.preload_gate(3.0 * GB, "fits")                   # 5 - (20 - 18) = 3
    with pytest.raises(cap.LoadDeferred, match="free working set 3.0 GB"):
        cap.preload_gate(3.5 * GB, "edge")


def test_preload_gate_kernel_floor(rig, monkeypatch):
    # The kernel's view: other processes' pages are invisible to MLX
    # accounting, so a load must also leave the governor's reclaimable
    # floor standing. No armed floor (no governor) = no kernel check.
    rig(weights_gb=10.0, ws_gb=20.0)
    import gmlx.serve.governor as gov
    import gmlx.serve.kernel_vm as kv
    import gmlx.gen.prefill_decay as pd
    monkeypatch.setattr(pd, "headroom_bytes", lambda: 50.0 * GB)
    monkeypatch.setattr(cap, "working_budget_bytes", lambda: 20.0 * GB)
    monkeypatch.setattr(kv, "reclaimable_bytes", lambda: 6.0 * GB)
    monkeypatch.setattr(gov, "armed_kernel_floor_bytes", lambda: 0.0)
    cap.preload_gate(4.0 * GB, "unarmed")               # no governor: MLX only
    monkeypatch.setattr(gov, "armed_kernel_floor_bytes", lambda: 3.0 * GB)
    cap.preload_gate(3.0 * GB, "fits")                   # 6 - 3 floor = 3
    with pytest.raises(cap.LoadDeferred, match="reclaimable memory 6.0 GB"):
        cap.preload_gate(4.0 * GB, "squeezed")
    monkeypatch.setattr(kv, "reclaimable_bytes", lambda: None)
    cap.preload_gate(4.0 * GB, "no-probe")               # probe unavailable admits


def test_kernel_gate_credits_page_cache_resident_bytes(rig, monkeypatch):
    # 2026-08-26 ladder stall: a fully-cached 90 GB model was refused
    # because raw reclaimable (91 GB, mostly the model's OWN cached
    # bytes) sat under weights + floor (98 GB). Wiring cached pages
    # converts them, it does not allocate: the gate must judge only the
    # not-yet-resident bytes.
    rig(weights_gb=90.0, ws_gb=200.0)
    import gmlx.serve.governor as gov
    import gmlx.serve.kernel_vm as kv
    import gmlx.gen.prefill_decay as pd
    monkeypatch.setattr(pd, "headroom_bytes", lambda: 500.0 * GB)
    monkeypatch.setattr(cap, "working_budget_bytes", lambda: 200.0 * GB)
    monkeypatch.setattr(gov, "armed_kernel_floor_bytes", lambda: 8.0 * GB)
    monkeypatch.setattr(kv, "reclaimable_bytes", lambda: 91.0 * GB)
    monkeypatch.setattr(cap, "_resident_shard_bytes", lambda m: 87.0 * GB)
    cap.preload_gate(90.0 * GB, "cached")           # need 3 GB <= 91 - 8
    monkeypatch.setattr(cap, "_resident_shard_bytes", lambda m: 92.0 * GB)
    cap.preload_gate(90.0 * GB, "fully-cached")     # need <= 0: admit outright
    monkeypatch.setattr(cap, "_resident_shard_bytes", lambda m: 0.0)
    with pytest.raises(cap.LoadDeferred, match="90.0 GB not yet page-cached"):
        cap.preload_gate(90.0 * GB, "cold")
    monkeypatch.setattr(cap, "_resident_shard_bytes", lambda m: 5.0 * GB)
    with pytest.raises(cap.LoadDeferred, match="85.0 GB not yet page-cached"):
        cap.preload_gate(90.0 * GB, "barely-cached")


def test_memfit_delegates_to_capacity(monkeypatch):
    from gmlx.load.memfit import classify_fit
    assert classify_fit(65, 100) == "fits"
    assert classify_fit(66, 100) == "tight"
    assert classify_fit(85, 100) == "tight"
    assert classify_fit(86, 100) == "over"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_kernel_gate_waits_for_memory_still_being_freed(rig, monkeypatch):
    """A reload right after an unload reads the kernel before the torn-down
    model's pages are back: the gate re-samples while reclaimable rises
    and admits once it clears, and only defers once it stops rising."""
    rig(weights_gb=10.0, ws_gb=200.0)
    import gmlx.serve.governor as gov
    import gmlx.serve.kernel_vm as kv
    import gmlx.gen.prefill_decay as pd
    import time as _time
    monkeypatch.setattr(pd, "headroom_bytes", lambda: 500.0 * GB)
    monkeypatch.setattr(cap, "working_budget_bytes", lambda: 200.0 * GB)
    monkeypatch.setattr(gov, "armed_kernel_floor_bytes", lambda: 8.0 * GB)
    slept = []
    monkeypatch.setattr(_time, "sleep", lambda s: slept.append(s))
    # rising: 20 -> 60 -> 100 -> 120 GB; 104 + 8 needs 112
    readings = iter([60.0 * GB, 100.0 * GB, 120.0 * GB, 120.0 * GB])
    monkeypatch.setattr(kv, "reclaimable_bytes",
                        lambda: next(readings, 120.0 * GB))
    cap.preload_gate(104.0 * GB, "reload")             # first read 60: waits, admits
    assert len(slept) == 2                              # 100, then 120 cleared it
    # flat: 20 -> 20 -> 20: two non-rising samples end the wait, deferred
    slept.clear()
    monkeypatch.setattr(kv, "reclaimable_bytes", lambda: 20.0 * GB)
    with pytest.raises(cap.LoadDeferred, match="reclaimable memory 20.0 GB"):
        cap.preload_gate(104.0 * GB, "other-process")
    assert len(slept) == 2
    # rising but never enough: the deadline ends it
    slept.clear()
    clock = [0.0]
    monkeypatch.setattr(_time, "monotonic", lambda: clock[0])

    def creeping():
        clock[0] += 1.0
        return 20.0 * GB + clock[0] * GB
    monkeypatch.setattr(kv, "reclaimable_bytes", creeping)
    with pytest.raises(cap.LoadDeferred):
        cap.preload_gate(104.0 * GB, "creeping")
    assert 0 < len(slept) <= 4


HYBRID = {
    "num_hidden_layers": 32,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "full_attention_interval": 4,
    "linear_num_value_heads": 32,
    "linear_num_key_heads": 16,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "max_position_embeddings": 262144,
}
DENSE_TWIN = {k: v for k, v in HYBRID.items() if not k.startswith(
    ("full_attention", "linear_"))}


def test_hybrid_table_prices_state_not_growth(rig, monkeypatch):
    monkeypatch.delenv("KV_BITS", raising=False)
    dense = cap.derive_table(rig(weights_gb=10.0, ws_gb=20.0,
                                 cfg=DENSE_TWIN))
    hybrid = cap.derive_table(rig(weights_gb=10.0, ws_gb=20.0, cfg=HYBRID))
    # 8 of 32 layers grow; the 24 recurrent layers hold 51 MB in total.
    # The score transient (heads x 2048 x depth) bounds both tables, so
    # the 4x KV saving lands as about 2x context at width 1.
    assert hybrid["max_ctx"][1] > 1.8 * dense["max_ctx"][1]
    assert hybrid["max_width_at_depth"][65536] > dense["max_width_at_depth"][65536]
    kv8 = cap.derive_table(rig(weights_gb=10.0, ws_gb=20.0, cfg=HYBRID),
                           env={"KV_BITS": "8"})
    assert kv8["max_ctx"][1] > hybrid["max_ctx"][1]


def test_nested_text_config_prices_like_flat(rig):
    flat = cap.derive_table(rig(weights_gb=10.0, ws_gb=20.0))
    nested = cap.derive_table(rig(weights_gb=10.0, ws_gb=20.0, cfg={
        "text_config": dict(CFG), "model_type": "gemma4"}))
    assert nested["max_ctx"] == flat["max_ctx"]
