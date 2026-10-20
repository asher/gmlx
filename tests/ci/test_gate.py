"""scripts/gate.py on scratch clones of a bare origin: setup, recording,
publishing by a plain push, pruning, and the gate/gpu decision order."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location("_gate", _ROOT / "scripts" / "gate.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


gate = _load()
NOW = datetime(2026, 10, 19, tzinfo=timezone.utc)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo,
                          check=True, capture_output=True, text=True).stdout.strip()


def _commit(repo: Path, path: str, text: str) -> str:
    f = repo / path
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text)
    _git(repo, "add", path)
    _git(repo, "commit", "-q", "-m", path)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def clones(tmp_path, monkeypatch):
    """A bare origin with one commit on main, and two clones set up for the gate."""
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    seed = tmp_path / "seed"
    _git(tmp_path, "clone", "-q", str(origin), str(seed))
    _commit(seed, "gmlx/a.py", "1")
    _git(seed, "push", "-q", "origin", "HEAD:main")
    out = []
    for name in ("a", "b"):
        repo = tmp_path / name
        _git(tmp_path, "clone", "-q", str(origin), str(repo))
        monkeypatch.setattr(gate, "ROOT", repo)
        gate.setup(None)
        out.append(repo)
    monkeypatch.setattr(gate, "ROOT", out[0])
    return out


def _result(passed=True, when=NOW, commit="c" * 40):
    return {"pass": passed, "commit": commit, "chip": "M3 Max", "when": when.isoformat(),
            "duration_s": 600, "tiers": ["gpu"], "log": []}


def _gate_status(repo, head, base=""):
    return gate.decide(head, base, cwd=repo, now=NOW)


def test_setup_is_idempotent(clones, monkeypatch, capsys):
    monkeypatch.setattr(gate, "ROOT", clones[0])
    gate.setup(None)
    assert "added" not in capsys.readouterr().out
    assert _git(clones[0], "config", "--get-all", "remote.origin.push").splitlines() == ["HEAD", gate.PUSH]


def test_each_run_records_a_result_under_a_name_of_its_own(clones, monkeypatch):
    """The ref names the tree and the run, never the machine."""
    monkeypatch.setattr(gate, "run_tier", lambda tier, workers, log: (0, 90.0))
    monkeypatch.setattr(gate, "_changed_since_main", lambda: [])
    for _ in range(2):
        assert gate.run(SimpleNamespace(workers=1, full=False, stress=None)) == 0
    tree = _git(clones[0], "rev-parse", "HEAD^{tree}")
    found = gate.results(clones[0])
    runs = [run for (t, run) in found if t == tree]
    assert len(runs) == 2 and all(len(run) == 12 and int(run, 16) >= 0 for run in runs)
    assert all(r["pass"] is True and r["ref"].startswith(f"{gate.MINE}/{tree}/") for r in found.values())


def test_run_refuses_a_dirty_tree(clones):
    (clones[0] / "gmlx" / "a.py").write_text("dirty")
    with pytest.raises(SystemExit, match="uncommitted"):
        gate.run(SimpleNamespace(workers=1, full=False, stress=None))


def test_a_plain_push_publishes_the_result_for_other_clones(clones):
    a, b = clones
    _git(a, "checkout", "-q", "-b", "feat")
    tree = _git(a, "rev-parse", "HEAD^{tree}")
    run = gate.record(_result(), tree, cwd=a).rsplit("/", 1)[1]
    _git(a, "push", "-q")
    assert _git(a, "ls-remote", "origin", "refs/gmlx-gate/*").endswith(f"refs/gmlx-gate/{tree}/{run}")
    _git(b, "fetch", "-q")
    assert gate.results(b)[(tree, run)]["ref"] == f"{gate.MIRROR}/{tree}/{run}"


def test_a_pruned_result_does_not_come_back(clones, monkeypatch):
    a, b = clones
    tree = _git(a, "rev-parse", "HEAD^{tree}")
    gate.record(_result(when=NOW - timedelta(days=40)), tree, cwd=a)
    _git(a, "push", "-q", "origin", f"{gate.MINE}/*:{gate.MIRROR}/*")
    _git(b, "fetch", "-q")
    assert gate.results(b)
    monkeypatch.setattr(gate, "ROOT", a)
    gate.prune(SimpleNamespace(local=False))
    _git(b, "push", "-q")
    assert _git(a, "ls-remote", "origin", "refs/gmlx-gate/*") == ""


def test_an_unpushed_result_survives_the_pruning_fetch(clones):
    a, _ = clones
    tree = _git(a, "rev-parse", "HEAD^{tree}")
    ref = gate.record(_result(), tree, cwd=a)
    _git(a, "fetch", "--prune", "-q", "origin", gate.FETCH)
    assert [r["ref"] for r in gate.results(a).values()] == [ref]


def test_run_drops_its_own_expired_results_before_recording(clones, monkeypatch):
    a, _ = clones
    gate.record(_result(when=datetime.now(timezone.utc) - timedelta(days=40)), "0" * 40, cwd=a)
    monkeypatch.setattr(gate, "run_tier", lambda tier, workers, log: (0, 1.0))
    monkeypatch.setattr(gate, "_changed_since_main", lambda: [])
    gate.run(SimpleNamespace(workers=1, full=False, stress=None))
    assert all(tree != "0" * 40 for tree, _ in gate.results(a))


def _gated_branch(repo):
    """A feature commit on a branch with a passing result for its tree."""
    _git(repo, "checkout", "-q", "-b", "feat")
    gated = _commit(repo, "gmlx/b.py", "1")
    gate.record(_result(commit=gated), _git(repo, "rev-parse", "HEAD^{tree}"), cwd=repo)
    return gated


def test_a_result_for_the_tree_passes_it(clones):
    a, _ = clones
    head = _gated_branch(a)
    assert _gate_status(a, head) == {"context": "gate/gpu", "state": "success", "description": "passed in 10m"}


def test_a_docs_commit_over_a_gated_tree_inherits_its_result(clones):
    a, _ = clones
    gated = _gated_branch(a)
    head = _commit(a, "docs/a.md", "typo")
    status = _gate_status(a, head)
    assert status["state"] == "success" and status["inherits"] == gated
    assert status["description"] == f"inherited from {gated[:10]}, docs-only delta"


@pytest.mark.parametrize("path", ["gmlx/b.py", "gmlx/comfyui_pack/README.md"])
def test_a_code_commit_over_a_gated_tree_is_pending(clones, path):
    a, _ = clones
    _gated_branch(a)
    head = _commit(a, path, "changed")
    assert _gate_status(a, head)["state"] == "pending"


def test_a_failure_for_the_tree_beats_an_inherited_pass(clones):
    a, _ = clones
    _gated_branch(a)
    head = _commit(a, "docs/a.md", "typo")
    gate.record(_result(passed=False, commit=head), _git(a, "rev-parse", "HEAD^{tree}"), cwd=a)
    assert _gate_status(a, head) == {"context": "gate/gpu", "state": "failure", "description": "failed"}


def test_a_docs_only_branch_needs_no_run(clones):
    a, _ = clones
    base = _git(a, "rev-parse", "HEAD")
    _git(a, "checkout", "-q", "-b", "docs")
    head = _commit(a, "docs/a.md", "new page")
    assert _gate_status(a, head, base)["description"] == "no code change"


def test_an_expired_result_is_ignored(clones):
    a, _ = clones
    _git(a, "checkout", "-q", "-b", "feat")
    head = _commit(a, "gmlx/b.py", "1")
    gate.record(_result(when=NOW - gate.WINDOW - timedelta(days=1)), _git(a, "rev-parse", "HEAD^{tree}"),
                cwd=a)
    assert _gate_status(a, head)["state"] == "pending"


@pytest.mark.parametrize("order,state", [((False, True), "success"), ((True, False), "failure")])
def test_the_newest_result_for_the_tree_decides(clones, order, state):
    """A pass after a flaky failure clears it, and a failure after a pass
    is not hidden by it, whichever Mac ran each."""
    a, _ = clones
    _git(a, "checkout", "-q", "-b", "feat")
    head = _commit(a, "gmlx/b.py", "1")
    tree = _git(a, "rev-parse", "HEAD^{tree}")
    for age, passed in zip((2, 1), order):
        gate.record(_result(passed=passed, when=NOW - timedelta(hours=age), commit=head), tree, cwd=a)
    assert _gate_status(a, head)["state"] == state


def test_a_result_names_no_folder_of_this_mac(tmp_path):
    log = tmp_path / "gpu.log"
    home = Path.home()
    log.write_text(f"FAILED tests/test_a.py::test_x - OSError: {home}/m.gguf\n...\n...\n1 failed in {home}/src\n")
    assert gate._summary_lines(log) == [
        "FAILED tests/test_a.py::test_x - OSError: ~/m.gguf", "...", "...", "1 failed in ~/src"]


def _write(root: Path, files: dict[str, str]) -> None:
    for path, text in files.items():
        f = root / path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)


def test_the_gate_set_is_the_numerics_folders_and_the_mlx_importers(tmp_path, monkeypatch):
    _write(tmp_path, {
        "tests/models/test_m.py": "",
        "tests/serve/test_capacity.py": "def test_a():\n    import mlx.core as mx\n",
        "tests/serve/test_kq.py": "from mlx_kquant import ops\n",
        # split, so the scan does not take this file for one that runs a child
        "tests/serve/test_child.py": "import sys\nCODE = '''\nimport mlx.core\n'''\nRUN = [sys." "executable]\n",
        # source text a test reads, never runs
        "tests/serve/test_source_text.py": "SOURCE = '''\nimport mlx.core\n'''\n",
        "tests/serve/test_routes.py": "import gmlx.serve\n",
        "tests/serve/mlx_helper.py": "import mlx.core\n",
        "tests/test_extra.py": "",
    })
    monkeypatch.setattr(gate, "GPU_EXTRA", ("tests/test_extra.py", "tests/test_gone.py"))
    assert gate.gate_set(tmp_path) == ["tests/models", "tests/serve/test_capacity.py", "tests/serve/test_child.py",
                                       "tests/serve/test_kq.py", "tests/test_extra.py"]


def _junit(cases: dict[str, bool]) -> str:
    rows = "".join(f'<testcase classname="{c}" name="{n}">{"<skipped/>" if s else ""}</testcase>'
                   for (c, n), s in ((k.rsplit("::", 1), s) for k, s in cases.items()))
    return f"<testsuites><testsuite>{rows}</testsuite></testsuites>"


def test_the_scope_check_lists_a_device_dependent_file_outside_the_set(tmp_path):
    _write(tmp_path, {"tests/models/test_m.py": "", "tests/serve/test_gpu.py": "",
                      "tests/serve/test_same.py": ""})
    (tmp_path / "gpu.xml").write_text(_junit({
        "tests.models.test_m::test_a": False, "tests.serve.test_gpu.TestK::test_b": False,
        "tests.serve.test_same::test_c": True}))
    (tmp_path / "cpu.xml").write_text(_junit({
        "tests.models.test_m::test_a": True, "tests.serve.test_gpu.TestK::test_b": True,
        "tests.serve.test_same::test_c": True}))
    assert gate.scope_misses(tmp_path / "gpu.xml", tmp_path / "cpu.xml", tmp_path) == ["tests/serve/test_gpu.py"]


def test_a_full_run_has_both_devices_and_the_scope_check():
    assert gate._tiers([], full=False, stress=None) == ["gpu"]
    assert gate._tiers([], full=True, stress=None) == ["gpu-full", "cpu-full", "scope"]
    assert gate._tiers(["gmlx/spinner.py"], full=True, stress=None)[-1] == "stress"


def test_a_full_run_writes_the_test_map(clones, monkeypatch):
    from coverage import CoverageData

    def fake_tier(tier, workers, log):
        if tier == "gpu-full":
            db = gate._state_dir() / "coverage" / ".coverage"
            db.parent.mkdir(parents=True, exist_ok=True)
            data = CoverageData(basename=str(db))
            data.set_context("tests/test_a.py::test_x|run")
            data.add_lines({str(clones[0] / "gmlx" / "a.py"): [1]})
            data.write()
        return 0, 1.0

    monkeypatch.setattr(gate, "run_tier", fake_tier)
    monkeypatch.setattr(gate, "_changed_since_main", lambda: [])
    monkeypatch.setattr(gate.affected_tests, "ROOT", clones[0])
    assert gate.run(SimpleNamespace(workers=1, full=True, stress=None)) == 0
    test_map, _ = gate.affected_tests.load_map(clones[0])
    assert test_map["tests"] == ["tests/test_a.py"]
    assert test_map["files"] == {"gmlx/a.py": [0]}
