"""The decode progress line reports the rate since the previous progress
line, so a speculative stream, which delivers a round's tokens in one burst,
reads at its real speed instead of swinging between a burst and a verify."""

import logging

import pytest

gen = pytest.importorskip("gmlx._vendor.mlx_vlm.server.generation")
_log = gen.ResponseGenerator._log_decode_progress


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(gen.time, "perf_counter", lambda: now[0])
    monkeypatch.setenv("MLX_VLM_LOG_PROGRESS_INTERVAL", "10")
    return now


def _emit(info, n=1):
    return _log("u", info, token=7, text="x", finish_reason=None,
                token_count=n)


def _progress_rates(caplog):
    return [r.getMessage().rsplit("rate=", 1)[1] for r in caplog.records
            if r.getMessage().startswith("Decode progress")]


def test_burst_rounds_report_the_window_rate(clock, caplog):
    caplog.set_level(logging.INFO, logger="gmlx._vendor.mlx_vlm.server")
    info = {"request_id": "r1"}
    _emit(info)                       # first token opens decode
    for _ in range(5):                # rounds of 4 tokens, 0.5 s apart
        clock[0] += 0.5
        for _ in range(4):
            _emit(info)               # tokens of one round, no time between
    # Lines at tokens 10 and 20: 9 tokens over 1.5 s, then 10 over 1.0 s.
    assert _progress_rates(caplog) == ["6.0 tok/s", "10.0 tok/s"]


def test_completion_line_keeps_the_overall_rate(clock, caplog):
    caplog.set_level(logging.INFO, logger="gmlx._vendor.mlx_vlm.server")
    info = {"request_id": "r2"}
    _emit(info)
    clock[0] += 2.0
    _log("u", info, token=1, text="", finish_reason="stop", token_count=4)
    done = [r.getMessage() for r in caplog.records
            if r.getMessage().startswith("Decode completed")]
    assert done and "rate=2.0 tok/s" in done[0]


def test_the_rate_lives_in_the_copy():
    """No installer rebinds the method, so the copy's own code runs."""
    import gmlx.serve.patches.observability as obs

    method = gen.ResponseGenerator.__dict__["_log_decode_progress"]
    assert method.__func__.__module__ == gen.__name__
    assert not hasattr(obs, "install_decode_progress_rate")
