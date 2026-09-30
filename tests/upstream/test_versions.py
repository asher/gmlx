"""The mlx version gate at CLI entry, and the copied packages' versions."""
from __future__ import annotations

import importlib.metadata as md
import json
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

import gmlx.upstream.versions as uv

_MANIFEST = Path(uv._COPY_DIR) / "manifest.json"


def _fake_mlx(monkeypatch, value):
    real = md.version

    def fake(pkg):
        if pkg != "mlx":
            return real(pkg)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(md, "version", fake)


def test_mlx_below_the_floor_raises(monkeypatch):
    _fake_mlx(monkeypatch, "0.31.2")
    with pytest.raises(RuntimeError, match=r"mlx 0\.31\.2 .*floor 0\.32\.2"):
        uv.check_mlx_version(quiet=True)


def test_mlx_at_or_above_the_floor_passes(monkeypatch):
    for found in (uv.MLX_FLOOR, "0.33.0", "1.0.0.dev1"):
        _fake_mlx(monkeypatch, found)
        assert uv.check_mlx_version(quiet=True) == []


def test_missing_mlx_metadata_warns_and_does_not_raise(monkeypatch):
    _fake_mlx(monkeypatch, md.PackageNotFoundError("mlx"))
    warnings = uv.check_mlx_version(quiet=True)
    assert any("source install" in w for w in warnings)


def test_the_floor_is_the_one_the_copied_mlx_vlm_declares():
    manifest = json.loads(_MANIFEST.read_text())
    floors = []
    for package in manifest["packages"].values():
        for text in package["requires"]:
            req = Requirement(text)
            if req.name != "mlx" or (req.marker is not None and not
                                     req.marker.evaluate({"extra": ""})):
                continue
            floors += [s.version for s in req.specifier
                       if s.operator in (">=", "==", "~=")]
    assert max(floors, key=Version) == uv.MLX_FLOOR


def test_the_copied_versions_are_the_manifest_versions():
    manifest = json.loads(_MANIFEST.read_text())
    want = {name.replace("_", "-"): p["version"]
            for name, p in manifest["packages"].items()}
    assert uv.copied_versions() == want


def test_doctor_names_the_copied_versions():
    from gmlx.commands import doctor

    detail = doctor.check_runtime()["detail"]
    for dist, version in uv.copied_versions().items():
        assert f"{dist} {version} (copied)" in detail


def test_doctor_fails_an_mlx_below_the_floor_the_cli_enforces(monkeypatch):
    from gmlx.commands import doctor

    _fake_mlx(monkeypatch, "0.32.1")
    check = doctor.check_runtime()
    assert check["status"] == "FAIL", check
    assert f"mlx 0.32.1 is below the supported floor {uv.MLX_FLOOR}" in \
        check["detail"]
    _fake_mlx(monkeypatch, uv.MLX_FLOOR)
    assert doctor.check_runtime()["status"] in ("PASS", "WARN")
    _fake_mlx(monkeypatch, md.PackageNotFoundError("mlx"))
    check = doctor.check_runtime()
    assert check["status"] == "WARN" and "source install" in check["detail"]

