import pytest


def pytest_runtest_setup(item):
    if item.get_closest_marker("needs_kvarn_ops") is None:
        return
    from gmlx.cache import kvarn_sdpa

    reason = kvarn_sdpa._probe()
    if reason:
        pytest.skip(f"kvarn ops unavailable: {reason}")


@pytest.fixture
def kvarn_ops_ok(monkeypatch):
    """kvarn eligibility without the Metal kernels: the resolver reads the
    ops probe, and the env knobs start clear."""
    from gmlx.cache import kvarn_sdpa

    monkeypatch.setattr(kvarn_sdpa, "_probe_result", (None,))
    for k in ("GMLX_KVARN", "GMLX_KVARN_BITS", "KV_BITS", "KV_TAIL_TOKENS",
              "KV_QUANT_SCHEME"):
        monkeypatch.delenv(k, raising=False)
