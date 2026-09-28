#!/usr/bin/env python3
"""The CI workflow pre-installs some dependencies by exact version to get a
wheel instead of an sdist build. A pin below the pyproject floor is worse than
no pin: the editable install upgrades it, so the job tests a version nobody
chose. This anchors the workflow pins against pyproject, and the mlx-kquant
pin against the mlx that the pinned mlx-vlm needs.
"""
from __future__ import annotations

import re
import shlex
import tomllib
from importlib import metadata
from pathlib import Path

import pytest
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW = _ROOT / ".github" / "workflows" / "test.yml"
_INSTALL = re.compile(r"pip install\s+(.+)")


def _canonical(name: str) -> str:
    return name.lower().replace("_", "-")


def _declared() -> dict[str, Requirement]:
    with open(_ROOT / "pyproject.toml", "rb") as f:
        deps = tomllib.load(f)["project"]["dependencies"]
    out = {}
    for spec in deps:
        r = Requirement(spec)
        out[_canonical(r.name)] = r
    return out


def _ci_pins() -> dict[str, Requirement]:
    """Every version-constrained requirement a `pip install` line in the
    workflow names, quoted or not. Flags, paths, and unconstrained names
    are skipped."""
    out = {}
    for line in _WORKFLOW.read_text().splitlines():
        m = _INSTALL.search(line)
        if m is None:
            continue
        for tok in shlex.split(m.group(1)):
            if tok.startswith("-"):
                continue
            try:
                r = Requirement(tok)
            except InvalidRequirement:
                continue
            if str(r.specifier):
                out[_canonical(r.name)] = r
    return out


def test_workflow_pins_found():
    """A rename of the install step must not turn this file into a no-op."""
    pins = _ci_pins()
    assert pins, f"no version-pinned pip install lines found in {_WORKFLOW}"
    assert "mlx-kquant" in pins, pins


@pytest.mark.parametrize("name", sorted(_ci_pins()))
def test_ci_pin_satisfies_pyproject(name):
    """Each exact CI pin must satisfy the pyproject constraint for that
    package. A wildcard pin checks its lower bound."""
    pin = _ci_pins()[name]
    declared = _declared().get(name)
    if declared is None or not str(declared.specifier):
        pytest.skip(f"{name} is unconstrained in pyproject")
    versions = [s.version for s in pin.specifier
                if s.operator in ("==", "===")]
    if not versions:
        pytest.skip(f"{name} pre-install is not an exact pin: {pin.specifier}")
    for v in versions:
        probe = v[:-2] + ".0" if v.endswith(".*") else v
        assert declared.specifier.contains(probe, prereleases=True), (
            f"CI pins {name} {pin.specifier}, which does not satisfy the "
            f"pyproject constraint {declared.specifier}. The editable "
            "install would silently upgrade it; bump the workflow pin.")


def _exact(req: Requirement) -> str | None:
    versions = [s.version for s in req.specifier if s.operator == "=="]
    return versions[0] if len(versions) == 1 else None


def _local_requirements(name: str, version: str | None):
    """The installed distribution's requirements, or None when it is
    missing or is not `version`."""
    try:
        if version is None or metadata.version(name) != version:
            return None
        reqs = metadata.requires(name) or []
    except metadata.PackageNotFoundError:
        return None
    return [Requirement(r) for r in reqs]


def _mlx_specifier(reqs) -> SpecifierSet:
    """The mlx constraint a plain install (no extras) applies."""
    spec = SpecifierSet()
    for r in reqs:
        if _canonical(r.name) != "mlx":
            continue
        if r.marker is None or r.marker.evaluate({"extra": ""}):
            spec &= r.specifier
    return spec


def test_kquant_pin_admits_the_mlx_vlm_mlx_floor():
    """An mlx-kquant release pins mlx exactly. The CI pin must name one whose
    mlx satisfies the floor the pinned mlx-vlm sets, or pip cannot install
    the two together. Read from the local install; skipped when the local
    mlx-kquant or mlx-vlm is not the pinned version."""
    kq = _local_requirements("mlx-kquant", _exact(_ci_pins()["mlx-kquant"]))
    vlm = _local_requirements("mlx-vlm", _exact(_declared()["mlx-vlm"]))
    if kq is None or vlm is None:
        pytest.skip("the local mlx-kquant or mlx-vlm is not the pinned version")
    floors = [s.version for s in _mlx_specifier(vlm)
              if s.operator in (">=", "==", "~=")]
    assert floors, f"mlx-vlm sets no mlx floor: {_mlx_specifier(vlm)}"
    floor = max(floors, key=Version)
    kq_mlx = _mlx_specifier(kq)
    assert kq_mlx.contains(floor, prereleases=True), (
        f"the pinned mlx-kquant requires mlx{kq_mlx}, which excludes mlx "
        f"{floor}, the floor of the pinned mlx-vlm. Bump the mlx-kquant pin "
        "to a release built against that mlx.")
