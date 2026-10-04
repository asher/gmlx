"""The decode progress line reports the rate since the previous progress
line, so a speculative stream, which delivers a round's tokens in one burst,
reads at its real speed instead of swinging between a burst and a verify."""

import logging

import pytest

from gmlx.serve.patches import observability as obs


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(obs.time, "perf_counter", lambda: now[0])
    monkeypatch.setenv("MLX_VLM_LOG_PROGRESS_INTERVAL", "10")
    return now


def _emit(info, n=1):
    return obs._log_decode_progress("u", info, token=7, text="x",
                                    finish_reason=None, token_count=n)


def _progress_rates(caplog):
    return [r.getMessage().rsplit("rate=", 1)[1] for r in caplog.records
            if r.getMessage().startswith("Decode progress")]


def test_burst_rounds_report_the_window_rate(clock, caplog):
    caplog.set_level(logging.INFO, logger="mlx_vlm.server")
    info = {"request_id": "r1"}
    _emit(info)                       # first token opens decode
    for _ in range(5):                # rounds of 4 tokens, 0.5 s apart
        clock[0] += 0.5
        for _ in range(4):
            _emit(info)               # tokens of one round, no time between
    # Lines at tokens 10 and 20: 9 tokens over 1.5 s, then 10 over 1.0 s.
    assert _progress_rates(caplog) == ["6.0 tok/s", "10.0 tok/s"]


def test_completion_line_keeps_the_overall_rate(clock, caplog):
    caplog.set_level(logging.INFO, logger="mlx_vlm.server")
    info = {"request_id": "r2"}
    _emit(info)
    clock[0] += 2.0
    obs._log_decode_progress("u", info, token=1, text="", finish_reason="stop",
                             token_count=4)
    done = [r.getMessage() for r in caplog.records
            if r.getMessage().startswith("Decode completed")]
    assert done and "rate=2.0 tok/s" in done[0]


def test_install_replaces_the_stock_hook_once():
    from mlx_vlm.server.generation import ResponseGenerator

    saved = ResponseGenerator.__dict__["_log_decode_progress"]
    flag = ResponseGenerator.__dict__.get(obs._PROGRESS_RATE_FLAG)
    try:
        if flag:
            delattr(ResponseGenerator, obs._PROGRESS_RATE_FLAG)
        obs.install_decode_progress_rate()
        owned = ResponseGenerator.__dict__["_log_decode_progress"]
        assert owned.__func__ is obs._log_decode_progress
        wrapper = staticmethod(lambda *a, **k: None)
        ResponseGenerator._log_decode_progress = wrapper
        obs.install_decode_progress_rate()  # a later wrapper survives
        assert ResponseGenerator.__dict__["_log_decode_progress"] is wrapper
    finally:
        ResponseGenerator._log_decode_progress = saved
        if flag:
            setattr(ResponseGenerator, obs._PROGRESS_RATE_FLAG, flag)
        elif obs._PROGRESS_RATE_FLAG in ResponseGenerator.__dict__:
            delattr(ResponseGenerator, obs._PROGRESS_RATE_FLAG)
