"""The image oracle harness in scripts/image_oracle: the fixture round trip,
the checks against a manifest, the fixture directory from the environment,
the child environment of the runner and its refusals. CPU only. No oracle
environment is needed."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_DIR = Path(__file__).resolve().parents[2] / "scripts" / "image_oracle"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"_image_oracle_{name}", _DIR / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fx = _load("fixtures")
runner = _load("run")


def _writer(name="check3_tiny"):
    w = fx.FixtureWriter(name, check=3, family="z_image",
                         script="scripts/image_oracle/x.py", argv=["x.py"])
    w.inputs.update(prompt="a red bicycle", noise_seed=7, sigma=0.5)
    w.add("latents", np.arange(24, dtype=np.float32).reshape(1, 4, 2, 3))
    w.add("block.out", np.linspace(-1, 1, 8).astype(np.float16))
    w.add("ids", np.array([3, 1, 2], np.int64))
    return w


def test_round_trip_writes_the_arrays_the_manifest_and_the_index(tmp_path):
    w = _writer()
    w.results["gap"] = 1e-3
    w.tolerances["final"] = 2e-3
    path = w.write(tmp_path)
    arrays, m = fx.load("check3_tiny", root=tmp_path)
    assert path == tmp_path / "check3_tiny.manifest.json"
    assert json.loads(path.read_text()) == m
    assert sorted(arrays) == ["block.out", "ids", "latents"]
    assert np.array_equal(arrays["latents"], w.arrays["latents"])
    assert arrays["block.out"].dtype == np.float16
    assert m["tensors"]["latents"] == {
        "shape": [1, 4, 2, 3], "dtype": "float32",
        "sha256": fx.array_sha256(w.arrays["latents"])}
    assert m["inputs"]["noise_seed"] == 7
    assert m["tolerances"] == {"final": 2e-3}
    assert m["oracle"]["python"] and "packages" in m["oracle"]

    _writer("check3_other").write(tmp_path)
    index = json.loads((tmp_path / "manifest.json").read_text())
    assert sorted(index["fixtures"]) == ["check3_other", "check3_tiny"]
    assert index["fixtures"]["check3_tiny"]["tensors"] == 3


def test_load_refuses_an_array_that_changed_after_the_manifest(tmp_path):
    w = _writer()
    w.write(tmp_path)
    changed = dict(w.arrays)
    changed["latents"] = changed["latents"].copy()
    changed["latents"][0, 0, 0, 0] += 1
    np.savez(tmp_path / "check3_tiny.npz", **changed)
    with pytest.raises(fx.FixtureError, match="latents differs"):
        fx.load("check3_tiny", root=tmp_path)


def test_load_checks_against_the_manifest_it_is_given(tmp_path):
    w = _writer()
    w.write(tmp_path)
    committed = json.loads((tmp_path / "check3_tiny.manifest.json")
                           .read_text())
    fx.load("check3_tiny", root=tmp_path, manifest=committed)

    stale = json.loads(json.dumps(committed))
    stale["tensors"]["ids"]["sha256"] = "0" * 64
    with pytest.raises(fx.FixtureError, match="ids differs"):
        fx.load("check3_tiny", root=tmp_path, manifest=stale)

    short = json.loads(json.dumps(committed))
    del short["tensors"]["block.out"]
    with pytest.raises(fx.FixtureError, match="block.out not in the manifest"):
        fx.load("check3_tiny", root=tmp_path, manifest=short)

    (tmp_path / "check3_tiny.npz").unlink()
    with pytest.raises(fx.FixtureError, match="missing"):
        fx.load("check3_tiny", root=tmp_path, manifest=committed)


def test_writer_refuses_arrays_a_fixture_cannot_hold():
    w = fx.FixtureWriter("t", check=3, family="z_image")
    with pytest.raises(fx.FixtureError, match="dtype"):
        w.add("x", np.array([1 + 2j]))
    with pytest.raises(fx.FixtureError, match="dtype"):
        w.add("y", np.array([object()]))
    w.add("z", np.zeros(2, np.float32))
    with pytest.raises(fx.FixtureError, match="already"):
        w.add("z", np.zeros(2, np.float32))
    with pytest.raises(fx.FixtureError, match="bad fixture name"):
        fx.FixtureWriter("../t", check=3, family="z_image")


def test_the_fixture_directory_comes_from_the_environment(monkeypatch,
                                                          tmp_path):
    monkeypatch.delenv(fx.ENV, raising=False)
    with pytest.raises(fx.FixtureError, match=fx.ENV):
        fx.fixture_root()
    monkeypatch.setenv(fx.ENV, str(tmp_path / "missing"))
    with pytest.raises(fx.FixtureError, match="not a directory"):
        fx.fixture_root()
    assert fx.fixture_root(create=True) == tmp_path / "missing"


def test_array_sha256_is_the_little_endian_bytes():
    a = np.arange(5, dtype=np.float32)
    assert fx.array_sha256(a) == fx.array_sha256(a.astype(">f4"))
    assert fx.array_sha256(a) != fx.array_sha256(a.astype(np.float64))
    assert fx.array_sha256(a[::2]) == fx.array_sha256(a[::2].copy())


def test_metrics():
    ref = np.array([3.0, 4.0])
    assert fx.rel_l2(ref * 1.01, ref) == pytest.approx(0.01)
    assert fx.cosine(ref * 2, ref) == pytest.approx(1.0)
    assert fx.max_abs(ref + [0.5, -1.0], ref) == 1.0
    assert fx.psnr(ref, ref, 1.0) == float("inf")
    assert fx.psnr(np.array([0.1]), np.array([0.0]), 1.0) == pytest.approx(20)


def test_the_child_runs_offline_and_off_the_caller_tree(tmp_path):
    base = {"PYTHONPATH": "/src/gmlx", "VIRTUAL_ENV": "/venv", "HOME": "/h",
            "HF_HUB_OFFLINE": "0"}
    env = runner.child_env(tmp_path, threads=3, base=base)
    assert "PYTHONPATH" not in env and "VIRTUAL_ENV" not in env
    assert env["HOME"] == "/h"
    assert env["HF_HUB_OFFLINE"] == "1" and env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env[runner.FIXTURES_ENV] == str(tmp_path)
    assert env["OMP_NUM_THREADS"] == "3"
    assert "OMP_NUM_THREADS" not in runner.child_env(tmp_path, base={})


def test_the_runner_refuses_its_own_interpreter():
    why = runner.preflight(sys.executable)
    assert why and "this interpreter" in why


def _probe(monkeypatch, **found):
    report = {"prefix": "/elsewhere", "torch": True, "diffusers": True,
              "gguf": True, "gmlx": False, "mlx": False, **found}

    def fake_run(cmd, **kw):
        assert cmd[1:3] == ["-I", "-c"]
        return subprocess.CompletedProcess(cmd, 0, json.dumps(report), "")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    return runner.preflight(sys.executable)


def test_the_runner_refuses_an_environment_that_holds_the_checked_code(
        monkeypatch):
    assert "mlx" in _probe(monkeypatch, mlx=True)
    assert "gmlx" in _probe(monkeypatch, gmlx=True)
    assert "diffusers" in _probe(monkeypatch, diffusers=False)
    assert _probe(monkeypatch) is None
    assert "does not exist" in runner.preflight("/no/such/python")


def test_the_runner_resolves_a_bare_script_name_in_its_directory():
    assert runner.resolve_script("fixtures.py") == _DIR / "fixtures.py"
    with pytest.raises(SystemExit):
        runner.resolve_script("no_such_oracle.py")
