#!/usr/bin/env python3
"""EXTRA_PACKAGES mirrors pyproject's [project.optional-dependencies] by hand
(see gmlx/extras.py docstring); this anchors the two against drift."""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

from gmlx import extras

_SELF_REF = re.compile(r"^gmlx\[([^\]]+)\]$")


def _pyproject_extras() -> dict[str, set[str]]:
    """The declared optional-dependency sets, with self-referential entries
    like ``gmlx[stt,tts]`` flattened into the component extras' packages -
    recursively, since ``all`` nests ``talk`` which nests ``stt,tts``."""
    root = Path(extras.__file__).resolve().parent.parent
    with open(root / "pyproject.toml", "rb") as f:
        declared = tomllib.load(f)["project"]["optional-dependencies"]

    def flat(name: str, stack: tuple[str, ...] = ()) -> set[str]:
        assert name not in stack, f"extra self-reference cycle: {stack + (name,)}"
        out: set[str] = set()
        for p in declared[name]:
            m = _SELF_REF.match(p)
            if m:
                for sub in m.group(1).split(","):
                    out |= flat(sub.strip(), stack + (name,))
            else:
                out.add(p)
        return out

    return {name: flat(name) for name in declared}


def test_extra_packages_match_pyproject():
    declared = _pyproject_extras()
    assert set(extras.EXTRA_PACKAGES) == set(declared)
    for name, pkgs in declared.items():
        assert set(extras.EXTRA_PACKAGES[name]) == pkgs, name

def test_every_extra_has_probe_modules():
    assert set(extras._PROBE_MODULES) == set(extras.EXTRA_PACKAGES)


def test_extra_installed_requires_every_probe_module(monkeypatch):
    """A rebuilt venv can keep sounddevice but lose sherpa-onnx; probing a
    single module would then report `talk` installed while wake-word mode
    silently degrades to an open mic."""
    import importlib.util

    # Fall back to the REAL find_spec for every other name: in a dev venv
    # with [talk] fully installed this doubles as the guard that the probe-
    # module names themselves resolve (a typo'd probe entry would land in
    # the missing list and fail). In an env without the extra (CI installs
    # only [vlm,chat,assistant]) the genuinely-absent modules are expected
    # in the list too, so assert relative to the pre-monkeypatch baseline.
    baseline = set(extras.missing_extra_modules("talk"))
    real = importlib.util.find_spec

    def no_sherpa(name, *a, **k):
        return None if name == "sherpa_onnx" else real(name, *a, **k)

    monkeypatch.setattr(importlib.util, "find_spec", no_sherpa)
    assert set(extras.missing_extra_modules("talk")) == baseline | {"sherpa_onnx"}
    assert extras.extra_installed("talk") is False


def test_missing_extra_modules_empty_when_all_import(monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a, **k: object())
    assert extras.missing_extra_modules("talk") == []
    assert extras.extra_installed("talk") is True
