"""bench/serve-bench.py reads accept counts from GMLX_ROUND_LOG by the
column names in its "# idx" header, in the old layout and the current one."""

import importlib.util
import sys
from pathlib import Path

import pytest

import gmlx.spec.speculative as spec

_PATH = Path(__file__).resolve().parents[1] / "bench" / "serve-bench.py"


@pytest.fixture(scope="module")
def bench():
    mod_spec = importlib.util.spec_from_file_location("serve_bench", _PATH)
    mod = importlib.util.module_from_spec(mod_spec)
    # dataclasses look the module up by name while the script runs
    sys.modules["serve_bench"] = mod
    try:
        mod_spec.loader.exec_module(mod)
    finally:
        sys.modules.pop("serve_bench", None)
    return mod


def test_parses_the_old_layout(bench, tmp_path):
    log = tmp_path / "old.tsv"
    log.write_text("# session kv=10 maxtok=8\n"
                   "0\t1.0\t0.1\t0.1\t1.2\t3\t4\n"
                   "1\t1.0\t0.1\t0.1\t1.2\t1\t4\n")
    assert bench._parse_round_log(log) == (6, 4)


def test_parses_a_log_the_spec_loop_writes(bench, tmp_path, monkeypatch):
    log = tmp_path / "rounds.tsv"
    monkeypatch.setenv("GMLX_ROUND_LOG", str(log))
    monkeypatch.setattr(spec, "_ROUND_PROFILE", False)
    try:
        spec._round_log_session(10, 8)
        spec._round_log(1.0, 0.1, 0.1, 3, 4, gap_ms=0.5)
        spec._round_log(1.0, 0.1, 0.1, 7, 8, gap_ms=0.5, copied=7)
        spec._round_log_note("context copy rounds=1 copied=7 accepted=7")
    finally:
        spec._round_log_close()
        spec._round_log_state.update(fh=None, path=None)
    assert bench._parse_round_log(log) == (10, 10)


def test_parses_the_profiled_layout(bench, tmp_path):
    log = tmp_path / "prof.tsv"
    log.write_text("# idx\tdraft_ms\tverify_ms\twalk_ms\temit_ms\tbook_ms\t"
                   "total_ms\taccepted\tbs\tgap_ms\tcopied\n"
                   "0\t1\t2\t3\t4\t5\t15\t2\t4\t0.250\t0\n")
    assert bench._parse_round_log(log) == (3, 2)


def test_counts_the_batch_rounds(bench, tmp_path, monkeypatch):
    """A batch round logs the mean accept over its rows."""
    log = tmp_path / "batch.tsv"
    monkeypatch.setenv("GMLX_ROUND_LOG", str(log))
    monkeypatch.setattr(spec, "_ROUND_PROFILE", False)
    try:
        spec._round_log_session(10, 64)
        spec._round_log(1.0, 0.1, 0.1, round(5 / 2, 2), 8, gap_ms=0.5)
        spec._round_log(1.0, 0.1, 0.1, 3, 8, gap_ms=0.5)
    finally:
        spec._round_log_close()
        spec._round_log_state.update(fh=None, path=None)
    assert bench._parse_round_log(log) == (14, 5.5)


def test_weights_a_round_by_its_speculating_rows(bench, tmp_path,
                                                 monkeypatch):
    """A round counts its drafts once per speculating row, and a forced or
    plain round (no rows) counts none."""
    log = tmp_path / "rows.tsv"
    monkeypatch.setenv("GMLX_ROUND_LOG", str(log))
    monkeypatch.setattr(spec, "_ROUND_PROFILE", False)
    try:
        spec._round_log_session(10, 64)
        spec._round_log(1.0, 0.1, 0.1, 2.5, 4, gap_ms=0.5, rows=2)
        spec._round_log(1.0, 0.1, 0.1, 3, 4, gap_ms=0.5)
        spec._round_log(1.0, 0.1, 0.1, 5, 6, gap_ms=0.5, rows=0)
    finally:
        spec._round_log_close()
        spec._round_log_state.update(fh=None, path=None)
    assert bench._parse_round_log(log) == (9, 8)
