"""scripts/stress_tests.py: a round that runs no test is a failure."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location("_stress_tests", _ROOT / "scripts" / "stress_tests.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


stress = _load()


def test_a_round_that_runs_no_test_fails(tmp_path):
    # A missing path, like a missing plugin, ends pytest nonzero with no
    # failing test in its report.
    out = tmp_path / "out"
    rc = stress.main([str(tmp_path / "no_such_test.py"), "--count", "1", "--workers", "1",
                      "--stall-ms", "1", "--out", str(out)])
    assert rc == 1
    assert (out / "round-1.log").exists()
    assert (out / "flaky.json").read_text() == '{"flaky": []}'
