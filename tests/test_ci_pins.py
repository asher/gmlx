#!/usr/bin/env python3
"""The CI workflow pre-installs some dependencies by exact version to get a
wheel instead of an sdist build. A pin below the pyproject floor is worse than
no pin: the editable install upgrades it, so the job tests a version nobody
chose. This anchors the workflow pins against pyproject.
"""
from __future__ import annotations

import re
import shlex
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import InvalidRequirement, Requirement

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
