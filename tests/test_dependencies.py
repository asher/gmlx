"""gmlx declares what its copy of mlx-vlm and mlx-lm imports, and never the
stock packages. Each requirement of the copied releases, as the manifest
recorded them, has a decision: declared with at least the copy's floor,
brought by another declared requirement, moved to an extra, or dropped."""
from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

_ROOT = Path(__file__).resolve().parents[1]

# A requirement of the copied releases that gmlx does not declare in core:
# the requirement that brings it, the extra that holds it, or why it goes.
_NOT_CORE = {
    "miniaudio": "brought by mlx-audio",
    "tqdm": "brought by mlx-audio",
    "opencv-python": "extra video",
    "websockets": "dropped: only the stock /v1/realtime route used it",
}


def _canonical(name: str) -> str:
    return name.lower().replace("_", "-")


def _project() -> dict:
    with open(_ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]


def _core() -> dict[str, Requirement]:
    return {_canonical(r.name): r
            for r in map(Requirement, _project()["dependencies"])}


def _copied_requirements() -> list[Requirement]:
    manifest = json.loads(
        (_ROOT / "gmlx" / "_vendor" / "manifest.json").read_text())
    out = []
    for package in manifest["packages"].values():
        for text in package["requires"]:
            r = Requirement(text)
            if r.marker is None or r.marker.evaluate({"extra": ""}):
                out.append(r)
    return out


def test_no_requirement_or_extra_names_a_stock_package():
    project = _project()
    names = [Requirement(r).name for r in project["dependencies"]]
    names += [Requirement(r).name
              for reqs in project["optional-dependencies"].values()
              for r in reqs]
    assert not {_canonical(n) for n in names} & {"mlx-vlm", "mlx-lm"}


def test_every_copied_requirement_has_a_decision():
    core, extras = _core(), _project()["optional-dependencies"]
    missing = []
    for req in _copied_requirements():
        name = _canonical(req.name)
        if name in _NOT_CORE:
            how = _NOT_CORE[name]
            if how.startswith("brought by "):
                assert how.removeprefix("brought by ") in core, how
            elif how.startswith("extra "):
                extra = extras[how.removeprefix("extra ")]
                assert name in {_canonical(Requirement(r).name)
                                for r in extra}, how
            assert name not in core, f"{name} is core and in _NOT_CORE"
            continue
        if name == "mlx":
            # mlx-kquant pins mlx; check_mlx_version holds the copy's floor.
            assert name in core
            continue
        if name not in core:
            missing.append(str(req))
            continue
        floors = [s.version for s in req.specifier if s.operator == ">="]
        ours = [s.version for s in core[name].specifier if s.operator == ">="]
        for floor in floors:
            assert ours and Version(max(ours, key=Version)) >= Version(floor), (
                f"gmlx declares {core[name]}, below the copy's {req}")
    assert not missing, missing


def test_load_video_names_the_video_extra(monkeypatch):
    from gmlx._vendor.mlx_vlm.utils import load_video

    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(ImportError, match=r"gmlx\[video\]"):
        load_video("clip.mp4")
