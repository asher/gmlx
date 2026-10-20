"""scripts/ci_tests.py: test selection, the flaky rerun, the flake-issue
counters and the growth report. gh and pytest are replaced by fakes."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from _pytest.junitxml import mangle_test_address

_ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location("_ci_tests", _ROOT / "scripts" / "ci_tests.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ct = _load()


@pytest.mark.parametrize("nodeid", [
    "tests/ci/test_x.py::test_a",
    "tests/ci/test_x.py::TestK::test_b[1-x.y]",
    "tests/test_x.py::test_c[a::b]",
])
def test_junit_keys_match_pytest(nodeid):
    names = mangle_test_address(nodeid)
    assert ct.junit_key(nodeid) == (".".join(names[:-1]), names[-1])


def test_the_docs_selection_is_the_docs_tests():
    picked = ct.selection("docs", skip_numerics=False, nightly=False)
    assert "tests/test_docs_cli.py" in picked and "tests/test_mkdocs_hooks.py" in picked
    assert all(p.startswith("tests/test_") for p in picked)


def test_the_311_leg_skips_the_numerics_folders():
    picked = ct.selection("full", skip_numerics=True, nightly=False)
    assert picked[:2] == ["-m", "not nightly"]
    assert picked[2:] == [f"--ignore=tests/{d}" for d in ct.NUMERICS]
    assert "--ignore=tests/load" in picked and "--ignore=tests/stream" in picked


def test_each_numerics_folder_exists():
    """A renamed folder would drop out of the skip list and the gate set."""
    missing = [d for d in ct.NUMERICS if not (_ROOT / "tests" / d).is_dir()]
    assert not missing, missing


def test_a_daily_leg_runs_the_nightly_tests():
    assert ct.selection("full", skip_numerics=False, nightly=True) == []


def _run(tmp_path, monkeypatch, codes, failed):
    calls = []
    lastfailed = tmp_path / "lastfailed"
    monkeypatch.setattr(ct, "LASTFAILED", lastfailed)

    def fake(args):
        calls.append(args)
        if len(calls) == 1 and failed:
            lastfailed.write_text(json.dumps({n: True for n in failed}))
        return codes[len(calls) - 1]

    monkeypatch.setattr(ct, "_pytest", fake)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    a = SimpleNamespace(select="full", skip_numerics=False, nightly=False,
                        junit=str(tmp_path / "junit.xml"), flaky_out=str(tmp_path / "flaky.json"))
    code = ct.run(a)
    return code, calls, json.loads((tmp_path / "flaky.json").read_text())["flaky"]


def test_a_test_that_passes_alone_is_flaky(tmp_path, monkeypatch, capsys):
    code, calls, flaky = _run(tmp_path, monkeypatch, [1, 0], ["tests/t.py::test_a"])
    assert code == 0 and flaky == ["tests/t.py::test_a"]
    assert "--lf" in calls[1] and "no:xdist" in calls[1] and "-n" not in calls[1]
    assert "::warning title=Flaky test::tests/t.py::test_a" in capsys.readouterr().out


def test_a_test_that_fails_alone_fails_the_step(tmp_path, monkeypatch):
    code, _, flaky = _run(tmp_path, monkeypatch, [1, 1], ["tests/t.py::test_a"])
    assert code == 1 and flaky == []


@pytest.mark.parametrize("first, failed", [(2, ["tests/t.py::test_a"]), (1, [])])
def test_no_rerun_without_failed_tests(tmp_path, monkeypatch, first, failed):
    code, calls, _ = _run(tmp_path, monkeypatch, [first], failed)
    assert code == first and len(calls) == 1


def test_counters_close_after_twenty_green_runs():
    state = {"green": 0, "absent": 0}
    for _ in range(ct.GREEN_TO_CLOSE - 1):
        state, reason = ct.tally(state, "passed", daily=False)
        assert reason is None
    assert ct.tally(state, "passed", daily=False)[1] is not None


def test_a_failure_resets_the_green_count():
    assert ct.tally({"green": 7, "absent": 0}, "failed", daily=False) == ({"green": 0, "absent": 0}, None)


def test_a_missing_test_closes_after_five_daily_runs():
    state = {"green": 3, "absent": ct.ABSENT_TO_CLOSE - 1}
    assert ct.tally(state, None, daily=False) == (state, None)
    assert "removed" in ct.tally(state, None, daily=True)[1]


def _issues(tmp_path, monkeypatch, listed, flaky, *, junit=None, count=False):
    calls = []

    def gh(*args):
        calls.append(args)
        return json.dumps(listed) if args[:2] == ("issue", "list") else ""

    monkeypatch.setattr(ct, "_gh", gh)
    (tmp_path / "flaky.json").write_text(json.dumps({"flaky": flaky}))
    junit_path = tmp_path / "junit.xml"
    junit_path.write_text(junit or "<testsuites/>")
    ct.issues(SimpleNamespace(flaky=str(tmp_path / "flaky.json"), junit=str(junit_path),
                              count=count, daily=False, run_url="RUN"))
    return calls


def test_a_new_flaky_test_opens_an_issue(tmp_path, monkeypatch):
    calls = _issues(tmp_path, monkeypatch, [], ["tests/t.py::test_a"])
    create = next(c for c in calls if c[:2] == ("issue", "create"))
    assert "flaky: tests/t.py::test_a" in create
    assert ct.parse_issue(create[-1]) == ("tests/t.py::test_a", {"green": 0, "absent": 0})


def test_a_known_flaky_test_reopens_its_issue(tmp_path, monkeypatch):
    body = ct.issue_body("tests/t.py::test_a", {"green": 5, "absent": 0}, "")
    calls = _issues(tmp_path, monkeypatch, [{"number": 4, "state": "CLOSED", "body": body}],
                    ["tests/t.py::test_a"])
    assert ("issue", "reopen", "4") in calls
    assert not any(c[:2] == ("issue", "create") for c in calls)


def test_a_green_run_closes_an_issue_at_the_count(tmp_path, monkeypatch):
    body = ct.issue_body("tests/t.py::test_a", {"green": ct.GREEN_TO_CLOSE - 1, "absent": 0}, "")
    junit = '<testsuites><testsuite><testcase classname="tests.t" name="test_a" time="1"/></testsuite></testsuites>'
    calls = _issues(tmp_path, monkeypatch, [{"number": 4, "state": "OPEN", "body": body}], [],
                    junit=junit, count=True)
    assert any(c[:3] == ("issue", "close", "4") for c in calls)


def _cases(n, t=1.0, prefix="t"):
    return {("tests.x", f"{prefix}{i}"): ("passed", t) for i in range(n)}


def test_growth_names_a_large_addition():
    base = _cases(10)
    _, notice = ct.growth_report({**base, **_cases(ct.NOTICE_CASES, prefix="n")}, base)
    assert notice and f"+{ct.NOTICE_CASES}" in notice


def test_growth_names_slow_new_tests():
    base = _cases(10)
    _, notice = ct.growth_report({**base, **_cases(1, t=ct.NOTICE_SECONDS, prefix="n")}, base)
    assert notice


def test_growth_is_quiet_for_a_small_change():
    base = _cases(10)
    text, notice = ct.growth_report({**base, **_cases(3, prefix="n")}, base)
    assert notice is None and "| tests | 10 | 13 | +3 |" in text
