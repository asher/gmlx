"""MLX buffer-cache policy: precedence, auto engagement, shard sizing."""

import pytest

from gmlx import server_memory as sm

GIB = sm.GIB
WS = 120 * GIB


def _resolve(monkeypatch, env=None, cfg=None, weights=0, ws=WS):
    if env is None:
        monkeypatch.delenv("GMLX_CACHE_LIMIT_GB", raising=False)
    else:
        monkeypatch.setenv("GMLX_CACHE_LIMIT_GB", env)
    monkeypatch.setattr(sm, "model_weight_bytes", lambda p: weights)
    return sm.resolve_cache_limit(cfg, ["m.gguf"], ws)


def test_env_wins_over_config_and_auto(monkeypatch):
    limit, src = _resolve(monkeypatch, env="8", cfg=4.0, weights=100 * GIB)
    assert limit == 8 * GIB and src == "env"


def test_env_zero_disables_cache(monkeypatch):
    # MLX semantics preserved: 0 = no buffer cache at all.
    limit, src = _resolve(monkeypatch, env="0")
    assert limit == 0 and src == "env"


@pytest.mark.parametrize("word", ["off", "none", "unlimited", "-1"])
def test_env_unlimited_suppresses_auto(monkeypatch, word):
    limit, src = _resolve(monkeypatch, env=word, weights=100 * GIB)
    assert limit is None and src == "env"


def test_config_key_applies(monkeypatch):
    limit, src = _resolve(monkeypatch, cfg=6.5)
    assert limit == int(6.5 * GIB) and src == "config"


def test_config_negative_forces_unlimited(monkeypatch):
    limit, src = _resolve(monkeypatch, cfg=-1.0, weights=100 * GIB)
    assert limit is None and src == "config"


def test_auto_engages_under_pressure(monkeypatch):
    # 85 GiB weights on a 120 GiB ws (the 122B case): quarter-slack ~8.75 GiB.
    limit, src = _resolve(monkeypatch, weights=85 * GIB)
    assert limit == int(0.25 * 35 * GIB)
    assert src.startswith("auto")


def test_auto_silent_with_slack(monkeypatch):
    # 25 GiB weights (gemma-31b): no pressure, unlimited, receipts untouched.
    limit, src = _resolve(monkeypatch, weights=25 * GIB)
    assert limit is None and src == "unlimited"


def test_auto_clamps():
    # Barely over the pressure line -> huge slack quarter clamps to the ceiling.
    assert sm.auto_cache_limit_bytes(200 * GIB, 121 * GIB) == 12 * GIB
    # Nearly no slack -> floor.
    assert sm.auto_cache_limit_bytes(100 * GIB, 99 * GIB) == 4 * GIB
    assert sm.auto_cache_limit_bytes(0, 0) is None


def test_weight_bytes_sums_shards(monkeypatch, tmp_path):
    a = tmp_path / "m-00001-of-00002.gguf"
    b = tmp_path / "m-00002-of-00002.gguf"
    a.write_bytes(b"x" * 100)
    b.write_bytes(b"y" * 50)
    assert sm.model_weight_bytes(str(a)) == 150


def test_weight_bytes_hf_ref_is_zero():
    assert sm.model_weight_bytes("hf:org/repo/file.gguf") == 0
    assert sm.model_weight_bytes("/nonexistent/m.gguf") == 0
