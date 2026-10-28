"""The video oracle harness in scripts/video_oracle: each script's declared
environment, the runner's refusals, the child environment, the fixtures
command's help and the oracle's commit with its dirty mark. CPU only. No
oracle environment is needed."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_DIR = Path(__file__).resolve().parents[2] / "scripts" / "video_oracle"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"_video_oracle_test_{name}", _DIR / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


runner = _load("run")


def test_each_oracle_script_declares_its_environment(tmp_path):
    assert runner.needs(_DIR / "check3_h3.py") == (
        "torch", "gguf", "einops", "comfy_kitchen")
    assert runner.needs(_DIR / "check4_h3.py") == ("torch", "diffusers")
    bare = tmp_path / "x.py"
    bare.write_text("NEEDS = ('torch',)\n")
    with pytest.raises(SystemExit, match="declares no ORACLE_NEEDS"):
        runner.needs(bare)


def test_the_child_gets_the_video_fixture_directory(tmp_path):
    env = runner.child_env(tmp_path, base={"PYTHONPATH": "/src/gmlx"})
    assert env["GMLX_VIDEO_FIXTURES"] == str(tmp_path)
    assert "GMLX_IMAGE_FIXTURES" not in env and "PYTHONPATH" not in env


def test_the_runner_refuses_its_own_interpreter():
    why = runner.preflight(sys.executable, ("numpy",))
    assert why and "this interpreter" in why


def _probe(monkeypatch, needed=("torch", "gguf"), **found):
    report = {"prefix": "/elsewhere", "torch": True, "gguf": True,
              "gmlx": False, "mlx": False, **found}

    def fake_run(cmd, **kw):
        assert cmd[1:3] == ["-I", "-c"]
        assert set(json.loads(cmd[4])) == set(needed) | {"gmlx", "mlx"}
        return subprocess.CompletedProcess(cmd, 0, json.dumps(report), "")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    return runner.preflight(sys.executable, needed)


def test_the_runner_refuses_an_environment_without_the_script_needs(
        monkeypatch):
    assert "mlx" in _probe(monkeypatch, mlx=True)
    assert "gmlx" in _probe(monkeypatch, gmlx=True)
    assert _probe(monkeypatch, gguf=False).endswith("cannot import gguf")
    assert _probe(monkeypatch) is None
    assert "does not exist" in runner.preflight("/no/such/python", ())


def test_the_fixtures_command_describes_video_fixtures():
    r = subprocess.run([sys.executable, str(_DIR / "fixtures.py"), "--help"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "video acceptance checks" in r.stdout
    assert "image acceptance" not in r.stdout


def test_the_oracle_commit_is_marked_dirty_by_a_tracked_change(tmp_path):
    oracle = _load("oracle")

    def git(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True,
                       capture_output=True)

    git("init", "-q")
    (tmp_path / "f").write_text("a")
    git("add", "f")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x")
    head = oracle.git_head(tmp_path)
    assert len(head) == 40
    (tmp_path / "new").write_text("b")
    assert oracle.git_head(tmp_path) == head
    (tmp_path / "f").write_text("c")
    assert oracle.git_head(tmp_path) == head + "-dirty"
