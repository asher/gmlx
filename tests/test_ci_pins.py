#!/usr/bin/env python3
"""The CI workflow pre-installs some dependencies by exact version to get a
wheel instead of an sdist build. A pin below the pyproject floor is worse than
no pin: the editable install upgrades it, so the job tests a version nobody
chose. This anchors the workflow pins against pyproject.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW = _ROOT / ".github" / "workflows" / "test.yml"
# `pip install "name==1.2.3"` inside the workflow's install step.
_PIN = re.compile(r'pip install\s+"([A-Za-z0-9._-]+==[^"]+)"')


def _declared() -> dict[str, Requirement]:
    with open(_ROOT / "pyproject.toml", "rb") as f:
        deps = tomllib.load(f)["project"]["dependencies"]
    out = {}
    for spec in deps:
        r = Requirement(spec)
        out[r.name.lower().replace("_", "-")] = r
    return out


def _ci_pins() -> dict[str, str]:
    text = _WORKFLOW.read_text()
    out = {}
    for spec in _PIN.findall(text):
        r = Requirement(spec)
        out[r.name.lower().replace("_", "-")] = str(r.specifier).lstrip("=")
    return out


def test_workflow_pins_found():
    """A rename of the install step must not turn this file into a no-op."""
    pins = _ci_pins()
    assert pins, f"no `pip install \"name==ver\"` pins found in {_WORKFLOW}"
    assert "mlx-kquant" in pins, pins


@pytest.mark.parametrize("name", sorted(_ci_pins()))
def test_ci_pin_satisfies_pyproject(name):
    """Each CI pin must satisfy the pyproject constraint for that package."""
    pinned = _ci_pins()[name]
    declared = _declared().get(name)
    if declared is None or not str(declared.specifier):
        pytest.skip(f"{name} is unconstrained in pyproject")
    assert declared.specifier.contains(pinned, prereleases=True), (
        f"CI pins {name}=={pinned}, which does not satisfy the pyproject "
        f"constraint {declared.specifier}. The editable install would "
        f"silently upgrade it; bump the workflow pin to match.")
