#!/usr/bin/env python3
"""The CI workflow pre-installs some dependencies by exact version to get a
wheel instead of an sdist build. A pin below the pyproject floor is worse than
no pin: the editable install upgrades it, so the job tests a version nobody
chose. This anchors the workflow pins against pyproject, and the mlx and
mlx-kquant pins against the mlx that the copied mlx-vlm needs. It also
checks that the test job installs the all extra and that the drift step
fails when its report fails.
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


def _copy_mlx_floor() -> str:
    """The highest mlx floor the copied releases declare, as the manifest
    recorded their requirements."""
    import json

    manifest = json.loads(
        (_ROOT / "gmlx" / "_vendor" / "manifest.json").read_text())
    reqs = [Requirement(r) for p in manifest["packages"].values()
            for r in p["requires"]]
    floors = [s.version for s in _mlx_specifier(reqs)
              if s.operator in (">=", "==", "~=")]
    assert floors, "the copied releases declare no mlx floor"
    return max(floors, key=Version)


def test_kquant_pin_admits_the_copy_mlx_floor():
    """An mlx-kquant release pins mlx exactly. The CI pin must name one whose
    mlx satisfies the floor the copied mlx-vlm declares, or gmlx refuses to
    run on it. Read from the local install; skipped when the local
    mlx-kquant is not the pinned version."""
    kq = _local_requirements("mlx-kquant", _exact(_ci_pins()["mlx-kquant"]))
    if kq is None:
        pytest.skip("the local mlx-kquant is not the pinned version")
    floor = _copy_mlx_floor()
    kq_mlx = _mlx_specifier(kq)
    assert kq_mlx.contains(floor, prereleases=True), (
        f"the pinned mlx-kquant requires mlx{kq_mlx}, which excludes mlx "
        f"{floor}, the floor of the copied mlx-vlm. Bump the mlx-kquant pin "
        "to a release built against that mlx.")


def test_ci_mlx_pin_meets_the_copy_floor():
    pin = _exact(_ci_pins()["mlx"])
    assert pin is not None, _ci_pins()["mlx"]
    assert Version(pin) >= Version(_copy_mlx_floor()), pin


def _run(job: str, step: str) -> str:
    """The run text of the step named ``step`` in workflow job ``job``."""
    import yaml

    steps = yaml.safe_load(_WORKFLOW.read_text())["jobs"][job]["steps"]
    return next(s["run"] for s in steps if s.get("name") == step)


def test_the_test_job_installs_the_all_extra():
    """The chat, voice and assistant tests skip themselves when their extra
    is absent, so the test job installs ``all`` for them to run."""
    toks = [tok for m in _INSTALL.finditer(_run("test", "Install"))
            for tok in shlex.split(m.group(1))]
    assert ".[all]" in toks, toks


def test_the_drift_step_fails_when_the_report_fails():
    """The report is piped through tee. Without pipefail set before the
    pipe, the step ends with the status of tee."""
    run = _run("drift", "Drift report")
    assert "set -o pipefail" in run, run
    assert run.index("set -o pipefail") < run.index("| tee"), run
