#!/usr/bin/env python3
"""EXTRA_PACKAGES mirrors pyproject's [project.optional-dependencies] by hand
(see gmlx/commands/extras.py docstring); this anchors the two against drift."""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import gmlx.commands.extras as extras

_SELF_REF = re.compile(r"^gmlx\[([^\]]+)\]$")


def _pyproject_extras() -> dict[str, set[str]]:
    """The declared optional-dependency sets, with self-referential entries
    like ``gmlx[stt,tts]`` flattened into the component extras' packages -
    recursively, since ``all`` nests ``talk`` which nests ``stt,tts``."""
    root = Path(extras.__file__).resolve().parents[2]
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


# -- install routing ----------------------------------------------------------
# gmlx can live in a plain venv or in an environment owned by `uv tool` or
# pipx. uv's has no pip at all, so the historical `python -m pip install` both
# failed and misreported; these pin the detection and the commands.


def _as_uv_tool(monkeypatch, tmp_path, receipt: str):
    (tmp_path / "uv-receipt.toml").write_text(receipt)
    monkeypatch.setattr(extras.sys, "prefix", str(tmp_path))


def test_route_defaults_to_pip(monkeypatch, tmp_path):
    monkeypatch.setattr(extras.sys, "prefix", str(tmp_path))
    assert extras.install_route() == extras.ROUTE_PIP
    assert extras.install_command("stt")[:3] == [extras.sys.executable,
                                                 "-m", "pip"]
    assert extras.install_hint("stt") == "pip install 'gmlx[stt]'"


def test_route_detects_pipx(monkeypatch, tmp_path):
    (tmp_path / "pipx_metadata.json").write_text("{}")
    monkeypatch.setattr(extras.sys, "prefix", str(tmp_path))
    assert extras.install_route() == extras.ROUTE_PIPX
    cmd = extras.install_command("stt")
    assert cmd[:3] == ["pipx", "inject", "gmlx"]
    assert set(extras.extra_packages("stt")) <= set(cmd)


def test_uv_route_merges_recorded_extras_and_pins_python(monkeypatch, tmp_path):
    """`uv tool install --force` re-resolves from the spec it is handed, so a
    bare `gmlx[tts]` would drop the already-installed `chat` extra and any
    --with packages, and re-pick uv's default interpreter."""
    _as_uv_tool(monkeypatch, tmp_path, """
[tool]
requirements = [
    { name = "gmlx", extras = ["chat"] },
    { name = "rich" },
]
python = "3.13"
""")
    assert extras.install_route() == extras.ROUTE_UV
    cmd = extras.install_command("tts")
    assert cmd[:4] == ["uv", "tool", "install", "--force"]
    assert cmd[4] == "gmlx[chat,tts]"          # merged, sorted, not clobbered
    assert "--with" in cmd and "rich" in cmd   # sibling requirement preserved
    assert cmd[-2:] == ["--python", "3.13"]    # interpreter carried over


def test_uv_route_survives_a_missing_or_broken_receipt(monkeypatch, tmp_path):
    _as_uv_tool(monkeypatch, tmp_path, "this is not toml {{{")
    cmd = extras.install_command("tts")
    assert cmd[:5] == ["uv", "tool", "install", "--force", "gmlx[tts]"]


def test_install_extra_refuses_when_the_installer_is_absent(monkeypatch,
                                                            tmp_path, capsys):
    """No `uv` on PATH: name the command instead of spawning something that
    cannot work (and never fall through to pip, which uv envs do not have)."""
    _as_uv_tool(monkeypatch, tmp_path, '[tool]\nrequirements = []\n')
    monkeypatch.setattr(extras.shutil, "which", lambda _: None)
    called = []
    assert extras.install_extra("tts", runner=lambda c: called.append(c)) is False
    assert called == []
    assert "uv tool install --force" in capsys.readouterr().err


def test_uv_route_preserves_version_pins(monkeypatch, tmp_path):
    """uv records a pin separately from the name
    (``{name = "rich", specifier = "==13.7.1"}``); rebuilding from the name
    alone would drop the constraint on every extra install."""
    _as_uv_tool(monkeypatch, tmp_path, """
[tool]
requirements = [
    { name = "gmlx", extras = ["chat"], specifier = ">=0.1.2" },
    { name = "rich", specifier = "==13.7.1" },
]
""")
    cmd = extras.install_command("tts")
    assert cmd[4] == "gmlx[chat,tts]>=0.1.2"
    assert cmd[cmd.index("--with") + 1] == "rich==13.7.1"


def test_uv_route_refuses_non_registry_sources(monkeypatch, tmp_path, capsys):
    """A tool installed from a checkout or a VCS cannot be rebuilt from its
    receipt, and a reconstructed `gmlx[...]` spec would silently repoint the
    install at PyPI. Refuse rather than move it."""
    _as_uv_tool(monkeypatch, tmp_path, """
[tool]
requirements = [{ name = "gmlx", directory = "/home/dev/gmlx" }]
""")
    assert extras.install_command("tts") == []
    hint = extras.install_hint("tts")
    assert "reinstall gmlx" in hint and "tts" in hint
    assert "uv tool install" not in hint          # never a wrong command

    called = []
    assert extras.install_extra("tts", runner=lambda c: called.append(c)) is False
    assert called == []
    assert "local or VCS source" in capsys.readouterr().err
