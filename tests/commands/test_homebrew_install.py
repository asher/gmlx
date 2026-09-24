"""A Homebrew install: keg detection, the install route, and the interpreter
path recorded in launchd agents and runfiles."""
from __future__ import annotations

import sys

import gmlx.commands.extras as extras
import gmlx.serve.procname as procname


def _keg(root, version="0.4.16", link=True):
    """A fake ``<prefix>/Cellar/gmlx/<version>`` keg with a venv in libexec,
    and ``<prefix>/opt/gmlx`` linked to it. Returns the venv interpreter."""
    keg = root / "Cellar" / "gmlx" / version
    (keg / "libexec" / "bin").mkdir(parents=True)
    (keg / "INSTALL_RECEIPT.json").write_text("{}")
    exe = keg / "libexec" / "bin" / "python"
    exe.write_text("")
    if link:
        opt = root / "opt"
        opt.mkdir(exist_ok=True)
        (opt / "gmlx").unlink(missing_ok=True)
        (opt / "gmlx").symlink_to(keg)
    return exe


def _run_from(monkeypatch, exe):
    monkeypatch.setattr(sys, "prefix", str(exe.parent.parent))
    monkeypatch.setattr(sys, "executable", str(exe))


def test_keg_uses_the_opt_path(monkeypatch, tmp_path):
    exe = _keg(tmp_path)
    _run_from(monkeypatch, exe)
    assert procname.homebrew_keg() == exe.parent.parent.parent.resolve()
    want = tmp_path.resolve() / "opt" / "gmlx" / "libexec" / "bin" / "python"
    assert procname.stable_executable() == str(want)


def test_keg_run_through_opt_gives_the_same_path(monkeypatch, tmp_path):
    _keg(tmp_path)
    via_opt = tmp_path / "opt" / "gmlx" / "libexec" / "bin" / "python"
    _run_from(monkeypatch, via_opt)
    want = tmp_path.resolve() / "opt" / "gmlx" / "libexec" / "bin" / "python"
    assert procname.stable_executable() == str(want)


def test_old_keg_after_upgrade_still_uses_opt(monkeypatch, tmp_path):
    """A server started before `brew upgrade` records the path that the next
    launch will use, not the version that cleanup deletes."""
    old = _keg(tmp_path, "0.4.16")
    _keg(tmp_path, "0.4.17")
    _run_from(monkeypatch, old)
    assert "/opt/gmlx/" in procname.stable_executable()


def test_keg_without_opt_link_keeps_the_real_path(monkeypatch, tmp_path):
    exe = _keg(tmp_path, link=False)
    _run_from(monkeypatch, exe)
    assert procname.stable_executable() == str(exe)


def test_opt_link_to_another_formula_is_ignored(monkeypatch, tmp_path):
    exe = _keg(tmp_path, link=False)
    other = tmp_path / "Cellar" / "other" / "1.0"
    other.mkdir(parents=True)
    (tmp_path / "opt").mkdir()
    (tmp_path / "opt" / "gmlx").symlink_to(other)
    _run_from(monkeypatch, exe)
    assert procname.stable_executable() == str(exe)


def test_plain_venv_is_not_a_keg(monkeypatch, tmp_path):
    exe = tmp_path / "venv" / "bin" / "python"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    _run_from(monkeypatch, exe)
    assert procname.homebrew_keg() is None
    assert procname.stable_executable() == str(exe)
    assert extras.install_route() == extras.ROUTE_PIP


def test_keg_route_never_runs_pip(monkeypatch, tmp_path, capsys):
    """The keg ships every extra and pip must not change it: a missing extra
    means a damaged install, repaired by reinstalling the formula."""
    _run_from(monkeypatch, _keg(tmp_path))
    assert extras.install_route() == extras.ROUTE_BREW
    assert extras.install_command("tts") == []
    assert extras.install_hint("tts") == "brew reinstall asher/gmlx/gmlx"
    assert extras.repair_hint("mlx-kquant") == "brew reinstall asher/gmlx/gmlx"

    called = []
    assert extras.install_extra("tts", runner=lambda c: called.append(c)) is False
    assert called == []
    err = capsys.readouterr().err
    assert "Homebrew install is missing packages" in err
    assert "brew reinstall asher/gmlx/gmlx" in err


def test_tool_receipts_win_over_a_keg(monkeypatch, tmp_path):
    exe = _keg(tmp_path)
    (exe.parent.parent / "uv-receipt.toml").write_text("[tool]\n")
    _run_from(monkeypatch, exe)
    assert extras.install_route() == extras.ROUTE_UV


def test_repair_hint_per_route(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    assert extras.repair_hint("mlx-kquant") == "pip install mlx-kquant"
    (tmp_path / "pipx_metadata.json").write_text("{}")
    assert extras.repair_hint("mlx-kquant") == "pipx reinstall gmlx"
    (tmp_path / "uv-receipt.toml").write_text("[tool]\n")
    assert extras.repair_hint("mlx-kquant") == "uv tool upgrade --reinstall gmlx"
