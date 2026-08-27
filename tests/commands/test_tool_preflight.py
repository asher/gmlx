"""Tool preflight: fit math, refusal gating, and the run-loop guard."""

import pytest

import gmlx.tool_preflight as tp

GB = 1e9

CFG = {
    "num_hidden_layers": 10,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "head_dim": 64,
    "max_position_embeddings": 65536,
}
# per token: 2 * 8 * 64 * 2 bytes * 10 layers = 20480 B/token


@pytest.fixture
def rig(monkeypatch, tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"x")

    def set_rig(weights_gb: float, ws_gb: float, cfg=CFG):
        monkeypatch.setattr(tp, "_shards", lambda p: [str(f)])
        monkeypatch.setattr(tp.os.path, "getsize",
                            lambda p: int(weights_gb * GB))
        monkeypatch.setattr(tp, "_synth_config", lambda p: dict(cfg))
        monkeypatch.setattr(tp, "working_set_bytes", lambda: ws_gb * GB)
        return str(f)

    return set_rig


def test_estimate_math_and_fit(rig):
    path = rig(weights_gb=10.0, ws_gb=20.0)
    est = tp.estimate(path, ctx_tokens=8192)
    kv = 8192 * 20480
    transient = 8 * 2048 * (8192 + 2048) * 2.0
    assert est["need"] == pytest.approx(10.0 * GB + kv + transient)
    assert est["fits"] is True
    assert est["largest_fit_ctx"] > 8192


def test_estimate_refuses_over_working_set(rig):
    path = rig(weights_gb=30.0, ws_gb=20.0)
    est = tp.estimate(path, ctx_tokens=4096)
    assert est["fits"] is False
    assert est["largest_fit_ctx"] == 0
    assert "weights alone exceed" in tp.refusal_text(est)


def test_largest_fit_ctx_is_the_boundary(rig):
    path = rig(weights_gb=10.0, ws_gb=20.0)
    est = tp.estimate(path, ctx_tokens=1024)
    z = est["largest_fit_ctx"]
    at = tp.estimate(path, ctx_tokens=z)
    over = tp.estimate(path, ctx_tokens=z + 1)
    assert at["fits"] and not over["fits"]
    assert str(z) in tp.refusal_text(over)


def test_default_ctx_prices_a_floor_session(rig):
    path = rig(weights_gb=1.0, ws_gb=20.0)
    assert tp.estimate(path)["ctx"] == 4096
    small = dict(CFG, max_position_embeddings=2048)
    path = rig(weights_gb=1.0, ws_gb=20.0, cfg=small)
    assert tp.estimate(path)["ctx"] == 2048


def test_check_or_exit_refuses_with_numbers(rig, capsys):
    path = rig(weights_gb=30.0, ws_gb=20.0)
    with pytest.raises(SystemExit) as ei:
        tp.check_or_exit(path, ctx_tokens=4096)
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert "cannot fit" in err and "30.0 GB" in err and "20.0 GB" in err


def test_check_or_exit_permits_fit_skip_and_kill_switch(rig, monkeypatch):
    path = rig(weights_gb=10.0, ws_gb=20.0)
    assert tp.check_or_exit(path)["fits"] is True
    assert tp.check_or_exit(path, streaming=True) is None
    monkeypatch.setenv("GMLX_TOOL_PREFLIGHT", "0")
    assert tp.check_or_exit(path) is None


def test_estimator_failure_permits(rig, monkeypatch):
    path = rig(weights_gb=30.0, ws_gb=20.0)
    monkeypatch.setattr(tp, "_synth_config",
                        lambda p: (_ for _ in ()).throw(ValueError("nope")))
    assert tp.estimate(path) is None
    assert tp.check_or_exit(path) is None


def test_guard_run_translates_allocator_errors(capsys):
    def boom():
        raise RuntimeError("[metal::malloc] Attempting to allocate "
                           "103918075904 bytes which is greater than ...")

    with pytest.raises(SystemExit) as ei:
        tp.guard_run(boom, est={"need": 30 * GB, "ctx": 4096})
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert "out of GPU memory mid-run" in err and "~30.0 GB" in err


def test_guard_run_passes_through_other_errors():
    with pytest.raises(RuntimeError, match="unrelated"):
        tp.guard_run(lambda: (_ for _ in ()).throw(
            RuntimeError("unrelated failure")))
    assert tp.guard_run(lambda: 41 + 1) == 42
