"""Oracle fixtures for the video checks, read against the manifests
committed under tests/fixtures/video/oracle/.

``fixture`` reads arrays from GMLX_VIDEO_FIXTURES and skips while it is
unset or the arrays are missing. ``committed_fixture`` reads a small
fixture whose arrays are committed beside its manifest."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import host_box
import pytest

ROOT = Path(__file__).resolve().parents[2]
ORACLE = ROOT / "tests" / "fixtures" / "video" / "oracle"


def _fixtures_module():
    path = ROOT / "scripts" / "video_oracle" / "fixtures.py"
    spec = importlib.util.spec_from_file_location("_video_oracle_fixtures_t",
                                                  path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fx = _fixtures_module()


def committed(name: str) -> dict:
    return json.loads((ORACLE / f"{name}.manifest.json").read_text())


def fixture_present(name: str) -> dict:
    """The committed manifest of a fixture whose arrays are under
    GMLX_VIDEO_FIXTURES. Skips while they are not, before anything
    costly reads them."""
    if not os.environ.get(fx.ENV):
        pytest.skip(f"set {fx.ENV} to the video oracle fixture directory")
    m = committed(name)
    if not (fx.fixture_root() / m["npz"]).is_file():
        pytest.skip(f"{m['npz']} is not under {fx.ENV}; run "
                    f"{m['script']} to write it")
    return m


def fixture(name: str) -> tuple[dict, dict]:
    """Arrays of a fixture under GMLX_VIDEO_FIXTURES, checked against its
    committed manifest."""
    m = fixture_present(name)
    host_box.real_files()
    return fx.load(name, manifest=m)


def committed_fixture(name: str) -> tuple[dict, dict]:
    """Arrays committed beside the manifest, checked against it."""
    return fx.load(name, root=ORACLE, manifest=committed(name))
