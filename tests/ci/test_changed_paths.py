"""scripts/changed_paths.py: the docs rule, the pack rule, and the diff from
the merge-base on a scratch repository."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location("_changed_paths", _ROOT / "scripts" / "changed_paths.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


cp = _load()


@pytest.mark.parametrize("path, docs", [
    ("docs/user/run.md", True),
    ("overrides/main.html", True),
    ("mkdocs.yml", True),
    ("CHANGELOG.md", True),
    ("gmlx/comfyui_pack/README.md", False),
    ("tests/e2e/README.md", False),
    ("scripts/mkdocs_hooks.py", False),
])
def test_the_docs_rule(path, docs):
    assert cp.is_docs(path) is docs


def test_a_cli_change_does_not_start_the_pack_job():
    assert cp.classify(["gmlx/commands/cli.py", "gmlx/__init__.py"]) == {"code": True, "pack": False}


def test_a_pack_change_starts_the_pack_job():
    assert cp.classify(["gmlx/comfyui_pack/nodes.py"])["pack"] is True


def test_an_unknown_change_counts_as_everything():
    assert cp.classify(None) == {"code": True, "pack": True}


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


def test_a_docs_branch_behind_main_is_docs_only(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "main")
    _commit(tmp_path, "gmlx/a.py", "1")
    _git(tmp_path, "checkout", "-q", "-b", "docs")
    head = _commit(tmp_path, "docs/a.md", "1")
    _git(tmp_path, "checkout", "-q", "main")
    base = _commit(tmp_path, "gmlx/a.py", "2")
    paths = cp.changed(base, head, cwd=str(tmp_path))
    assert paths == ["docs/a.md"]
    assert cp.classify(paths)["code"] is False


@pytest.mark.parametrize("base", ["0" * 40, "f" * 40])
def test_a_base_that_is_no_commit_reports_no_paths(tmp_path, base):
    _git(tmp_path, "init", "-q", "-b", "main")
    head = _commit(tmp_path, "docs/a.md", "1")
    assert cp.changed(base, head, cwd=str(tmp_path)) is None


def test_a_rename_out_of_code_counts_as_code(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "main")
    base = _commit(tmp_path, "gmlx/notes.txt", "x" * 200)
    (tmp_path / "docs").mkdir()
    _git(tmp_path, "mv", "gmlx/notes.txt", "docs/notes.txt")
    _git(tmp_path, "commit", "-q", "-m", "mv")
    paths = cp.changed(base, "HEAD", cwd=str(tmp_path))
    assert cp.classify(paths)["code"] is True
