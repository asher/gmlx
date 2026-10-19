"""scripts/affected_tests.py on a scratch tree: the import scan, conftest and
folder fallbacks, and the coverage test map with its exceptions."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from coverage import CoverageData

_ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location("_affected_tests", _ROOT / "scripts" / "affected_tests.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


at = _load()

FILES = {
    "gmlx/__init__.py": "",
    "gmlx/load/__init__.py": "",
    "gmlx/load/config_synth.py": "def synth():\n    return 1\n",
    "gmlx/serve/__init__.py": "",
    "gmlx/serve/capacity.py": "from gmlx.load import config_synth\n",
    "gmlx/serve/registry.py": "TABLE = {}\n",
    "gmlx/orphan/__init__.py": "",
    "gmlx/orphan/lonely.py": "",
    "tests/conftest.py": "",
    "tests/load/test_synth.py": "from gmlx.load import config_synth\n",
    "tests/serve/test_capacity.py": "from gmlx.serve import capacity\n",
    "tests/serve/test_registry.py": "from gmlx.serve.registry import TABLE\n",
    "tests/serve/serve_helper.py": "from gmlx.serve import capacity\nfrom gmlx.load import config_synth\n",
    "tests/serve/test_uses_helper.py": "import serve_helper\n",
    "tests/cache/conftest.py": "from gmlx.load.config_synth import synth\n",
    "tests/cache/test_cache.py": "",
    "tests/orphan/test_nothing.py": "",
    "tests/test_child.py": ("import subprocess, sys\n"
                            "CODE = '''\nfrom gmlx.serve import capacity\n'''\n"
                            "subprocess.run([sys.executable, '-c', CODE])\n"),
}


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=root,
                          check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def tree(tmp_path):
    for path, text in FILES.items():
        f = tmp_path / path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "seed")
    return tmp_path


def _coverage(root: Path, runs: dict[str, list[str]]) -> Path:
    """A coverage database where each context ran the given files."""
    db = root / ".coverage"
    data = CoverageData(basename=str(db))
    for context, files in runs.items():
        data.set_context(context)
        data.add_lines({str(root / f): [1] for f in files})
    data.write()
    return db


def test_a_module_selects_its_importers_and_a_conftest_folder(tree):
    tests, unmatched = at.select(["gmlx/load/config_synth.py"], tree)
    # capacity's tests reach it only through another gmlx module, which the
    # scan does not follow.
    assert tests == ["tests/cache", "tests/load/test_synth.py", "tests/serve/test_uses_helper.py"]
    assert unmatched == []


def test_a_child_interpreters_code_counts_as_an_import(tree):
    tests, _ = at.select(["gmlx/serve/capacity.py"], tree)
    assert tests == ["tests/serve/test_capacity.py", "tests/serve/test_uses_helper.py", "tests/test_child.py"]


def test_a_module_nothing_imports_falls_back_to_its_folder(tree):
    assert at.select(["gmlx/orphan/lonely.py"], tree) == (["tests/orphan"], [])


def test_a_changed_conftest_selects_its_folder(tree):
    assert at.select(["tests/cache/conftest.py", "tests/cache/test_cache.py"], tree) == (["tests/cache"], [])


def test_the_map_follows_gmlx_to_gmlx_calls(tree):
    db = _coverage(tree, {"tests/serve/test_capacity.py::test_a|run": ["gmlx/load/config_synth.py"]})
    test_map = at.build_map(db, _git(tree, "rev-parse", "HEAD"), tree)
    test_map["since"] = set()
    tests, _ = at.select(["gmlx/load/config_synth.py"], tree, test_map)
    # A test the map does not name and that runs no child is left out.
    assert tests == ["tests/serve/test_capacity.py"]


def test_the_map_adds_test_files_written_after_it(tree):
    db = _coverage(tree, {"tests/serve/test_capacity.py::test_a|run": ["gmlx/load/config_synth.py"]})
    (tree / "tests" / "load" / "test_new.py").write_text("from gmlx.load import config_synth\n")
    at.write_map(db, _git(tree, "rev-parse", "HEAD"), tree)
    test_map, notes = at.load_map(tree)
    assert test_map["since"] == {"tests/load/test_new.py"}
    assert notes == [f"test map from {_git(tree, 'rev-parse', 'HEAD')[:10]}, 0 days old"]
    tests, _ = at.select(["gmlx/load/config_synth.py"], tree, test_map)
    assert tests == ["tests/load/test_new.py", "tests/serve/test_capacity.py"]


def test_a_file_that_ran_only_at_import_counts_as_unknown(tree):
    db = _coverage(tree, {"": ["gmlx/serve/registry.py"],
                          "tests/serve/test_capacity.py::test_a|run": ["gmlx/load/config_synth.py"]})
    test_map = at.build_map(db, _git(tree, "rev-parse", "HEAD"), tree)
    assert test_map["files"]["gmlx/serve/registry.py"] == []
    test_map["since"] = set()
    assert at.select(["gmlx/serve/registry.py"], tree, test_map)[0] == ["tests/serve/test_registry.py"]


def test_without_a_map_the_scan_is_the_rule(tree):
    assert at.load_map(tree) == (None, ["no test map; selecting by imports only "
                                        "(scripts/gate.py run --full writes one)"])


def test_an_old_map_asks_for_a_refresh(tree):
    db = _coverage(tree, {"tests/load/test_synth.py::test_a|run": ["gmlx/load/config_synth.py"]})
    path = at.write_map(db, _git(tree, "rev-parse", "HEAD"), tree)
    data = json.loads(path.read_text())
    data["when"] = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    path.write_text(json.dumps(data))
    _, notes = at.load_map(tree)
    assert notes[-1] == "the test map is over four weeks old; refresh it with scripts/gate.py run --full"


def test_a_docs_change_selects_the_docs_tests(tree):
    (tree / "tests" / "test_docs_cli.py").write_text("")
    assert at.select(["docs/index.md"], tree) == (["tests/test_docs_cli.py"], [])


def test_another_file_selects_the_tests_that_name_it(tree):
    (tree / "tests" / "test_entry.py").write_text("CRATE = 'crates/gmlx-entry'\n")
    assert at.select(["crates/gmlx-entry/src/session.rs"], tree) == (["tests/test_entry.py"], [])
