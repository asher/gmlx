"""Boot capacity table: frontier math, ceilings, gates, consumers."""

import pytest

import gmlx.capacity as cap

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
    import gmlx.tool_preflight as tp

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
        import gmlx.server_memory as sm
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

    from gmlx.decode_batch import decode_batch
    monkeypatch.delenv("GMLX_DECODE_BATCH", raising=False)
    monkeypatch.setattr(cap, "frontier_width", lambda min_ctx=4096: 2)
    assert decode_batch() == 2
    monkeypatch.setenv("GMLX_DECODE_BATCH", "6")
    assert decode_batch() == 6                # explicit env wins


def test_preload_gate(rig, monkeypatch):
    rig(weights_gb=10.0, ws_gb=20.0)
    import gmlx.prefill_decay as pd
    monkeypatch.setattr(pd, "headroom_bytes", lambda: 5.0 * GB)

    cap.preload_gate(4.0 * GB, "small")       # fits: no raise
    with pytest.raises(cap.LoadDeferred, match="deferred"):
        cap.preload_gate(10.0 * GB, "busybox")           # typed: the 503 seam
    with pytest.raises(RuntimeError, match="does not fit"):
        cap.preload_gate(25.0 * GB, "huge")
    monkeypatch.setenv("GMLX_OVERCOMMIT", "1")
    cap.preload_gate(25.0 * GB, "huge")       # override skips


def test_memfit_delegates_to_capacity(monkeypatch):
    from gmlx.memfit import classify_fit
    assert classify_fit(65, 100) == "fits"
    assert classify_fit(66, 100) == "tight"
    assert classify_fit(85, 100) == "tight"
    assert classify_fit(86, 100) == "over"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
