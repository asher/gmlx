"""``gmlx comfyui``: detection, the version check, the copy and its link,
the record, uninstall, status, the refresh at server start and the doctor
row, all on fake ComfyUI folders under a temporary home. CPU only."""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import stat
from pathlib import Path

import pytest
import yaml

import gmlx.commands.comfyui as cu
import gmlx.commands.doctor as doctor
from gmlx import DOCS_URL


@pytest.fixture(autouse=True)
def home(monkeypatch, tmp_path):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    return h


def _support(home):
    return home / "Library" / "Application Support"


def make_comfy(code, *, version="0.38.0", venv=".venv", git=False, exe="python"):
    """A ComfyUI code folder: main.py, its version file, custom_nodes and a
    venv."""
    (code / "custom_nodes").mkdir(parents=True, exist_ok=True)
    (code / "main.py").write_text("")
    if version:
        (code / "comfyui_version.py").write_text(
            f'# auto-generated\n__version__ = "{version}"\n')
    if venv:
        _venv(code / venv, exe)
    if git:
        (code / ".git").mkdir()
    return code


def _venv(folder, exe="python"):
    py = folder / "bin" / exe
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_text("")
    py.chmod(0o755)
    (folder / "pyvenv.cfg").write_text("")
    return py


def _installations(home, records, name="comfyui-desktop-2"):
    path = _support(home) / name / "installations.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records))
    return path


def _legacy(home, version="0.3.76"):
    """The old desktop app: its base folder and extra_models_config.yaml."""
    base = home / "Documents" / "ComfyUI"
    (base / "custom_nodes").mkdir(parents=True)
    _venv(base / ".venv")
    app = home / "App" / "ComfyUI"
    make_comfy(app, version=version, venv=None)
    cfg = _support(home) / "ComfyUI" / "extra_models_config.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(yaml.safe_dump({
        "comfyui_desktop": {"base_path": str(base),
                            "custom_nodes": str(base / "custom_nodes") + "/"},
        "desktop_extensions": {"custom_nodes": str(app / "custom_nodes")},
    }))
    return base, app


def _by_nodes(found):
    return {os.path.realpath(t.custom_nodes): t for t in found}


# Detection.

def test_detects_the_clone_at_home(home):
    code = make_comfy(home / "ComfyUI")
    [t] = cu.detect()
    assert (t.kind, t.custom_nodes, t.code) == ("clone", code / "custom_nodes", code)
    assert t.python == code / ".venv" / "bin" / "python"


def test_detects_comfy_cli_workspaces(home):
    """comfy-cli's macOS default, its Linux default, and the default and
    recent workspaces of its config."""
    mac = make_comfy(home / "Documents" / "comfy" / "ComfyUI")
    linux = make_comfy(home / "comfy" / "ComfyUI")
    other = make_comfy(home / "work" / "ComfyUI")
    recent = make_comfy(home / "recent" / "ComfyUI")
    ini = _support(home) / "comfy-cli" / "config.ini"
    ini.parent.mkdir(parents=True)
    ini.write_text(f"[DEFAULT]\ndefault_workspace = {home / 'work'}\n"
                   f"recent_workspace = {recent}\n")
    found = _by_nodes(cu.detect())
    assert {t.kind for t in found.values()} == {"comfy-cli"}
    assert set(found) == {os.path.realpath(code / "custom_nodes")
                          for code in (mac, linux, other, recent)}


def test_a_comfy_cli_config_with_only_a_recent_workspace(home):
    recent = make_comfy(home / "projects" / "ComfyUI")
    ini = _support(home) / "comfy-cli" / "config.ini"
    ini.parent.mkdir(parents=True)
    ini.write_text(f"[DEFAULT]\nrecent_workspace = {recent}\n")
    [t] = cu.detect()
    assert (t.kind, t.code) == ("comfy-cli", recent)


def test_the_help_names_comfy_clis_macos_workspace(capsys):
    assert "comfy-cli's ~/Documents/comfy/ComfyUI or the workspace its config names" in \
        cu.no_comfyui_help()
    with pytest.raises(SystemExit):
        cu.cmd_comfyui(["install", "--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "comfy-cli's ~/Documents/comfy/ComfyUI or the workspace its config names" in out


def test_detects_the_old_desktop_app_through_its_yaml(home):
    base, app = _legacy(home)
    [t] = cu.detect()
    assert t.kind == "desktop-legacy"
    assert t.custom_nodes == base / "custom_nodes"
    assert t.code == app
    assert t.python == base / ".venv" / "bin" / "python"


def test_detects_each_comfy_desktop_record_per_its_source(home):
    """The record table: standalone, adopted, comfybuilder with its policy,
    git, the old app's record and cloud."""
    root = home / "ComfyUI-Installs"
    stable = make_comfy(root / "stable" / "ComfyUI", venv=None)
    _venv(stable / ".venv", "python3")
    build = make_comfy(root / "build" / "ComfyUI", venv=None)
    _venv(root / "build" / "venv", "python3")
    policy = build / "governance" / "policy.signed.json"
    policy.parent.mkdir()
    policy.write_text("{}")
    plain = make_comfy(root / "plain" / "ComfyUI", venv=None)
    _venv(root / "plain" / "venv", "python3")
    gitc = make_comfy(root / "gitinst" / "ComfyUI", venv=None)
    gvenv = home / "venvs" / "gitc"
    _venv(gvenv, "python3")
    make_comfy(root / "adopted" / "ComfyUI", venv=None)
    # A cloud or remote record runs no local ComfyUI, even when its folder
    # holds one.
    make_comfy(root / "cloud" / "ComfyUI", venv=None)
    make_comfy(root / "remote" / "ComfyUI", venv=None)
    base, _app = _legacy(home)
    _installations(home, [
        {"id": "a", "name": "Stable", "sourceId": "standalone",
         "installPath": str(root / "stable"), "status": "installed"},
        {"id": "b", "name": "Governed", "sourceId": "comfybuilder",
         "installPath": str(root / "build")},
        {"id": "c", "name": "Plain build", "sourceId": "comfybuilder",
         "installPath": str(root / "plain")},
        {"id": "d", "name": "Git", "sourceId": "git", "installPath": str(root / "gitinst"),
         "venvPath": str(gvenv)},
        {"id": "e", "name": "ComfyUI Legacy Desktop", "sourceId": "desktop",
         "installPath": str(base)},
        {"id": "f", "name": "Migrated", "sourceId": "standalone", "adopted": True,
         "adoptedBaseDir": str(base),
         "adoptedPythonPath": str(base / ".venv" / "bin" / "python"),
         "installPath": str(root / "adopted")},
        {"id": "g", "name": "Cloud", "sourceId": "cloud", "installPath": str(root / "cloud")},
        {"id": "h", "name": "Remote", "sourceId": "remote",
         "installPath": str(root / "remote")},
        {"id": "i", "name": "Half", "sourceId": "standalone",
         "installPath": str(root / "half"), "status": "installing"},
    ])
    found = _by_nodes(cu.detect())
    want = {
        stable / "custom_nodes": ("desktop", stable / ".venv" / "bin" / "python3"),
        build / "custom_nodes": ("desktop-build", root / "build" / "venv" / "bin" / "python3"),
        plain / "custom_nodes": ("desktop-build", root / "plain" / "venv" / "bin" / "python3"),
        gitc / "custom_nodes": ("desktop-git", gvenv / "bin" / "python3"),
        # The adopted install runs with --base-directory at the old app's
        # folder, so its custom_nodes are there, not under installPath.
        base / "custom_nodes": ("desktop-adopted", base / ".venv" / "bin" / "python"),
    }
    assert set(found) == {os.path.realpath(p) for p in want}
    for nodes, (kind, py) in want.items():
        t = found[os.path.realpath(nodes)]
        assert (t.kind, t.python) == (kind, py), nodes
    assert found[os.path.realpath(build / "custom_nodes")].policy == policy
    assert found[os.path.realpath(plain / "custom_nodes")].policy is None
    _v, notes = cu.check_version(found[os.path.realpath(build / "custom_nodes")],
                                 cu.pack_meta())
    assert any("signed policy" in n and "policy.signed.json" in n for n in notes)


def test_desktop_record_of_the_old_app_alone_reads_its_yaml(home):
    base, app = _legacy(home)
    _installations(home, [{"sourceId": "desktop", "installPath": str(base),
                           "name": "ComfyUI Legacy Desktop"}])
    [t] = cu.detect()
    assert (t.kind, t.custom_nodes, t.code) == ("desktop-legacy", base / "custom_nodes", app)


def test_desktop_data_under_the_product_name_is_read_too(home):
    code = make_comfy(home / "ComfyUI-Installs" / "x" / "ComfyUI", venv=None)
    _installations(home, [{"sourceId": "standalone",
                           "installPath": str(code.parent)}], name="Comfy Desktop")
    [t] = cu.detect()
    assert t.custom_nodes == code / "custom_nodes"


def test_a_broken_installations_file_finds_nothing(home):
    path = _installations(home, [])
    path.write_text("{not json")
    assert cu.detect() == []


def test_comfyui_dir_takes_a_code_folder_a_base_folder_or_an_install(home, tmp_path):
    code = make_comfy(tmp_path / "x" / "ComfyUI")
    assert cu.target_for_dir(code).code == code
    assert cu.target_for_dir(code.parent).code == code
    base = tmp_path / "base"
    (base / "custom_nodes").mkdir(parents=True)
    t = cu.target_for_dir(base)
    assert (t.custom_nodes, t.code, t.kind) == (base / "custom_nodes", None, "folder")
    with pytest.raises(cu.ComfyError) as e:
        cu.target_for_dir(tmp_path / "nothing")
    assert e.value.code == 1 and "is not a ComfyUI folder" in str(e.value)


# The version check.

def test_refuses_a_comfyui_older_than_the_floor_and_writes_nothing(home):
    code = make_comfy(home / "ComfyUI", version="0.36.9", git=True)
    t = cu.target_for_dir(code)
    with pytest.raises(cu.ComfyError) as e:
        cu.install([t])
    msg = str(e.value)
    assert e.value.code == 1
    assert "0.36.9" in msg and cu.pack_meta()["COMFYUI_FLOOR"] in msg
    assert f"git -C {code} pull" in msg
    assert f"{code / '.venv' / 'bin' / 'python'} -m pip install -r {code}/requirements.txt" in msg
    assert msg.endswith(f"The ComfyUI guide:\n  {DOCS_URL}comfyui.html")
    assert not cu.data_root().exists()
    assert not os.path.lexists(code / "custom_nodes" / cu.LINK_NAME)
    assert not cu.settings_path().exists()


def test_a_venv_that_uv_made_updates_with_uv_pip(home):
    """A venv from uv has no pip, so the update step runs uv pip."""
    code = make_comfy(home / "ComfyUI", version="0.36.9", git=True)
    (code / ".venv" / "pyvenv.cfg").write_text("home = /x\nimplementation = CPython\n"
                                               "uv = 0.9.2\nversion_info = 3.12.11\n")
    with pytest.raises(cu.ComfyError) as e:
        cu.install([cu.target_for_dir(code)])
    py = code / ".venv" / "bin" / "python"
    assert f"uv pip install --python {py} -r {code}/requirements.txt" in str(e.value)
    assert "-m pip" not in str(e.value)


@pytest.mark.parametrize("kind, says", [
    ("desktop", "Update ComfyUI in Comfy Desktop"),
    ("desktop-legacy", "Install Comfy Desktop from https://comfy.org/download"),
    ("comfy-cli", "update comfy"),
])
def test_the_upgrade_step_names_the_kind(tmp_path, kind, says):
    code = make_comfy(tmp_path / "ComfyUI", version="0.30.0")
    t = cu.Target(code / "custom_nodes", code, kind)
    with pytest.raises(cu.ComfyError) as e:
        cu.check_version(t, cu.pack_meta())
    assert says in str(e.value)


def test_notes_an_untested_or_unknown_comfyui(tmp_path):
    meta = cu.pack_meta()
    newer = make_comfy(tmp_path / "a", version="9.0.0")
    v, notes = cu.check_version(cu.target_for_dir(newer), meta)
    assert v == "9.0.0" and any("newer than" in n for n in notes)
    unknown = make_comfy(tmp_path / "b", version=None)
    v, notes = cu.check_version(cu.target_for_dir(unknown), meta)
    assert v is None and any("unknown" in n for n in notes)
    floor = make_comfy(tmp_path / "c", version=meta["COMFYUI_FLOOR"])
    assert cu.check_version(cu.target_for_dir(floor), meta) == (meta["COMFYUI_FLOOR"], [])


# Install: the copy, the link and the record.

def _install_cli(*args):
    return cu.cmd_comfyui(["install", *args])


def test_install_copies_links_and_records(home, capsys):
    code = make_comfy(home / "ComfyUI")
    assert _install_cli("--model", "z-image-turbo") == 0
    out = capsys.readouterr().out
    assert "installed the gmlx nodes in ~/ComfyUI/custom_nodes/comfyui-gmlx" in out
    assert "Restart ComfyUI" in out
    copy = cu.copy_dir()
    assert copy.is_symlink() and not os.path.isabs(os.readlink(copy))
    stamp = cu.read_stamp(copy)
    assert stamp == cu.want_stamp()
    assert stamp["comfyui_tested"] == cu.pack_meta()["COMFYUI_TESTED"]
    link = code / "custom_nodes" / "comfyui-gmlx"
    assert link.is_symlink() and os.readlink(link) == str(copy)
    assert (link / "__init__.py").is_file()
    assert (link / "example_workflows" / "z-image-turbo-t2i.json").is_file()
    assert not list(link.rglob("__pycache__"))
    doc = json.loads(cu.settings_path().read_text())
    assert stat.S_IMODE(cu.settings_path().stat().st_mode) == 0o600
    assert doc["installs"] == [{"custom_nodes": str(code / "custom_nodes"),
                                "comfyui": str(code), "kind": "clone",
                                "python": str(code / ".venv" / "bin" / "python"),
                                "label": ""}]
    assert doc["comfyui_dir"] == str(code)
    assert doc["default_image_model"] == "z-image-turbo"
    assert doc["gmlx_install"] in ("homebrew", "uv", "pip")
    assert doc["gmlx_command"] == cu.gmlx_command()
    assert doc["gmlx_version"] == cu.gmlx_version()
    assert doc["pack_dir"] == str(copy) and doc["pack_source"] is False


def test_install_keeps_the_users_own_settings(home):
    make_comfy(home / "ComfyUI")
    cu.settings_path().parent.mkdir(parents=True, exist_ok=True)
    cu.settings_path().write_text(json.dumps({"base_url": "http://h:1", "api_key": "k"}))
    assert _install_cli() == 0
    doc = json.loads(cu.settings_path().read_text())
    assert doc["base_url"] == "http://h:1" and doc["api_key"] == "k"


@pytest.mark.parametrize("text", [
    '{"base_url": "http://studio:8080", "api_key": "sk-secret",}\n',
    '["http://studio:8080"]\n',
])
def test_a_settings_file_that_is_no_json_object_is_never_written_over(home, capsys, text):
    """A hand-edited file that does not parse may hold the user's own
    settings: every action refuses and names it, and none writes over it or
    deletes it."""
    code = make_comfy(home / "ComfyUI")
    path = cu.settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    for verb in (["install"], ["uninstall"], ["status"], ["status", "--json"]):
        assert cu.cmd_comfyui(verb) == 1, verb
        err = capsys.readouterr().err
        assert f"{path} is not a JSON object" in err and "Fix it" in err, verb
        assert path.read_text() == text, verb
    assert not os.path.lexists(code / "custom_nodes" / cu.LINK_NAME)
    assert not cu.data_root().exists()
    with pytest.raises(cu.ComfyError):
        cu.install([cu.target_for_dir(code)])
    assert not cu.data_root().exists() and path.read_text() == text
    assert "is not a JSON object" in cu.refresh_for_serve()
    row = doctor.check_comfyui()
    assert row["status"] == "WARN" and "is not a JSON object" in row["detail"]
    assert cu.remembered_dir() is None
    assert path.read_text() == text


_BAD_ENTRIES = [5, None, "x", {"custom_nodes": 5}, {"custom_nodes": ""},
                {"custom_nodes": ["/x"]}, {"custom_nodes": "/x/custom_nodes", "comfyui": 7},
                {"custom_nodes": "/y", "python": {}}]


@pytest.mark.parametrize("installs", [5, "~/ComfyUI", {"custom_nodes": "/x"}, _BAD_ENTRIES])
def test_malformed_install_records_are_skipped(home, capsys, installs):
    """A hand edit can leave anything in installs. Status, install,
    uninstall and doctor skip what is not a record and keep working."""
    code = make_comfy(home / "ComfyUI")
    path = cu.settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"installs": installs, "base_url": "http://h:1"}))
    assert cu.recorded_targets(cu.load_record()) == []
    assert cu.cmd_comfyui(["status"]) == 1
    assert doctor.check_comfyui() is None
    assert cu.cmd_comfyui(["install"]) == 0
    doc = cu.load_record()
    assert [r["custom_nodes"] for r in doc["installs"]] == [str(code / "custom_nodes")]
    assert doctor.check_comfyui()["status"] == "PASS"
    path.write_text(json.dumps({**doc, "installs": [*doc["installs"], *_BAD_ENTRIES]}))
    assert cu.cmd_comfyui(["status"]) == 0
    assert cu.cmd_comfyui(["uninstall"]) == 0
    assert not os.path.lexists(code / "custom_nodes" / cu.LINK_NAME)
    assert json.loads(path.read_text()) == {"base_url": "http://h:1"}


def test_a_second_install_changes_nothing(home, capsys):
    code = make_comfy(home / "ComfyUI")
    assert _install_cli() == 0
    link = code / "custom_nodes" / "comfyui-gmlx"
    before = (os.lstat(link).st_ino, os.lstat(cu.copy_dir()).st_ino,
              os.readlink(cu.copy_dir()))
    capsys.readouterr()
    [d], sync, dropped = cu.install([cu.target_for_dir(code)])
    assert dropped == []
    assert sync.state == "current" and d.changed is False
    assert (os.lstat(link).st_ino, os.lstat(cu.copy_dir()).st_ino,
            os.readlink(cu.copy_dir())) == before


def test_a_real_pack_folder_moves_into_disabled(home, capsys):
    code = make_comfy(home / "ComfyUI")
    old = code / "custom_nodes" / "ComfyUI-gmlx"
    (old / ".git").mkdir(parents=True)
    (old / "mine.txt").write_text("keep")
    assert _install_cli() == 0
    assert "moved to ~/ComfyUI/custom_nodes/.disabled/comfyui-gmlx-" in capsys.readouterr().out
    [aside] = (code / "custom_nodes" / ".disabled").iterdir()
    assert aside.name.startswith("comfyui-gmlx-") and (aside / "mine.txt").read_text() == "keep"
    entries = [e.name for e in (code / "custom_nodes").iterdir()]
    assert sorted(entries) == [".disabled", "comfyui-gmlx"]
    # A second folder the same day gets its own name.
    os.unlink(code / "custom_nodes" / "comfyui-gmlx")
    (code / "custom_nodes" / "comfyui-gmlx").mkdir()
    assert _install_cli() == 0
    assert len(list((code / "custom_nodes" / ".disabled").iterdir())) == 2


def test_an_old_link_to_a_clone_of_the_pack_is_replaced(home, tmp_path):
    code = make_comfy(home / "ComfyUI")
    clone = tmp_path / "ComfyUI-gmlx"
    clone.mkdir()
    (code / "custom_nodes" / "ComfyUI-gmlx").symlink_to(clone)
    assert _install_cli() == 0
    names = [e.name for e in (code / "custom_nodes").iterdir()]
    assert names == ["comfyui-gmlx"] and clone.is_dir()
    assert os.path.realpath(code / "custom_nodes" / "comfyui-gmlx") == \
        os.path.realpath(cu.copy_dir())


def test_link_source_links_the_working_tree(home):
    code = make_comfy(home / "ComfyUI")
    assert _install_cli("--link-source") == 0
    link = code / "custom_nodes" / "comfyui-gmlx"
    assert os.path.realpath(link) == os.path.realpath(cu.PACK_SOURCE)
    assert not cu.copy_dir().exists()
    assert json.loads(cu.settings_path().read_text())["pack_source"] is True


def test_no_comfyui_prints_the_two_ways_to_get_it(home, capsys):
    assert _install_cli() == 1
    err = capsys.readouterr().err
    assert "https://comfy.org/download" in err
    assert "git clone https://github.com/comfyanonymous/ComfyUI ~/ComfyUI" in err
    assert "uv venv --python 3.12 .venv" in err and "comfyui.html" in err


def test_several_comfyuis_ask_for_one(home, capsys):
    make_comfy(home / "ComfyUI")
    make_comfy(home / "comfy" / "ComfyUI")
    assert _install_cli() == 2
    err = capsys.readouterr().err
    assert "--comfyui-dir" in err and "~/ComfyUI " in err and "~/comfy/ComfyUI " in err
    assert not cu.data_root().exists()


def test_each_folder_links_to_the_one_copy(home):
    a = make_comfy(home / "ComfyUI")
    b = make_comfy(home / "comfy" / "ComfyUI")
    assert _install_cli("--comfyui-dir", str(a)) == 0
    assert _install_cli("--comfyui-dir", str(b)) == 0
    for code in (a, b):
        assert os.readlink(code / "custom_nodes" / "comfyui-gmlx") == str(cu.copy_dir())
    doc = cu.load_record()
    assert len(doc["installs"]) == 2
    # Without --comfyui-dir, install refreshes the recorded folders.
    assert len(cu.resolve_targets(None, doc)) == 2


def test_install_into_comfy_desktop_records_no_folder_to_launch(home):
    code = make_comfy(home / "ComfyUI-Installs" / "s" / "ComfyUI", venv=None)
    _installations(home, [{"sourceId": "standalone", "installPath": str(code.parent),
                           "name": "Stable"}])
    assert _install_cli() == 0
    assert (code / "custom_nodes" / "comfyui-gmlx").is_symlink()
    doc = cu.load_record()
    assert "comfyui_dir" not in doc and doc["installs"][0]["kind"] == "desktop"


# The refresh after a gmlx upgrade.

def _versions():
    return sorted(p.name for p in (cu.data_root() / "versions").iterdir())


def test_a_changed_stamp_refreshes_the_copy_atomically(home, monkeypatch):
    code = make_comfy(home / "ComfyUI")
    assert _install_cli() == 0
    old = os.readlink(cu.copy_dir())
    link_ino = os.lstat(code / "custom_nodes" / "comfyui-gmlx").st_ino
    monkeypatch.setattr(cu, "gmlx_version", lambda: "99.0.0")
    swaps = []
    real_replace = os.replace

    def replace(src, dst):
        swaps.append((os.path.basename(src), str(dst), os.path.islink(src)))
        return real_replace(src, dst)

    monkeypatch.setattr(cu.os, "replace", replace)
    line = cu.refresh_for_serve()
    assert line and "99.0.0" in line and "restart ComfyUI" in line
    new = os.readlink(cu.copy_dir())
    assert new != old and new.startswith(os.path.join("versions", "99.0.0-"))
    # The copy turned to the new version in one rename of a link, and the
    # link in custom_nodes did not change.
    assert swaps == [(f".{cu.LINK_NAME}.{os.getpid()}.tmp", str(cu.copy_dir()), True)]
    assert os.lstat(code / "custom_nodes" / "comfyui-gmlx").st_ino == link_ino
    assert cu.read_stamp(cu.copy_dir())["gmlx_version"] == "99.0.0"
    assert _versions() == [os.path.basename(new)]
    assert cu.refresh_for_serve() is None


def test_an_older_gmlx_replacing_a_newer_copy_says_so(home, monkeypatch):
    make_comfy(home / "ComfyUI")
    monkeypatch.setattr(cu, "gmlx_version", lambda: "99.0.0")
    assert _install_cli() == 0
    monkeypatch.setattr(cu, "gmlx_version", lambda: "1.0.0")
    line = cu.refresh_for_serve()
    assert "older gmlx 1.0.0" in line


def test_an_unchanged_stamp_touches_nothing(home):
    make_comfy(home / "ComfyUI")
    assert _install_cli() == 0
    before = {p: os.lstat(p).st_mtime_ns for p in cu.data_root().rglob("*")}
    record = cu.settings_path().read_text()
    assert cu.refresh_for_serve() is None
    assert {p: os.lstat(p).st_mtime_ns for p in cu.data_root().rglob("*")} == before
    assert cu.settings_path().read_text() == record


def test_a_missing_recorded_folder_still_gets_the_refresh(home, monkeypatch, tmp_path):
    """The links point at the copy's fixed path, so the folders that are
    there load the new nodes, and a folder on a drive that comes back does
    too."""
    a = make_comfy(home / "ComfyUI")
    b = make_comfy(tmp_path / "volume" / "ComfyUI")
    assert _install_cli("--comfyui-dir", str(a)) == 0
    assert _install_cli("--comfyui-dir", str(b)) == 0
    b.rename(tmp_path / "volume" / "gone")
    monkeypatch.setattr(cu, "gmlx_version", lambda: "99.0.0")
    assert "99.0.0" in cu.refresh_for_serve()
    assert cu.read_stamp(a / "custom_nodes" / cu.LINK_NAME)["gmlx_version"] == "99.0.0"
    (tmp_path / "volume" / "gone").rename(b)
    assert cu.read_stamp(b / "custom_nodes" / cu.LINK_NAME)["gmlx_version"] == "99.0.0"


def test_install_drops_a_recorded_folder_that_is_gone(home, tmp_path, capsys):
    a = make_comfy(home / "ComfyUI")
    b = make_comfy(tmp_path / "volume" / "ComfyUI")
    assert _install_cli("--comfyui-dir", str(a)) == 0
    assert _install_cli("--comfyui-dir", str(b)) == 0
    shutil.rmtree(b)
    capsys.readouterr()
    assert cu.cmd_comfyui(["status"]) == 1
    assert f"{b / 'custom_nodes'}: folder missing; run gmlx comfyui install" in \
        capsys.readouterr().out
    assert _install_cli() == 0
    out = capsys.readouterr().out
    assert out.count("from the record") == 1
    assert f"removed {b / 'custom_nodes'} from the record: the folder is gone" in out
    assert [r["custom_nodes"] for r in cu.load_record()["installs"]] == [str(a / "custom_nodes")]
    assert cu.cmd_comfyui(["status"]) == 0


@pytest.mark.parametrize("others, rc", [(0, 1), (2, 2)])
def test_a_gone_only_folder_leaves_the_record_when_install_finds_no_other(
        home, tmp_path, capsys, others, rc):
    """The only recorded ComfyUI is gone, and install finds none, or several,
    to use instead. It exits as before, but the record is clean, so status
    and doctor stop naming the gone folder."""
    b = make_comfy(tmp_path / "volume" / "ComfyUI")
    assert _install_cli("--comfyui-dir", str(b)) == 0
    shutil.rmtree(b)
    for code in [home / "ComfyUI", home / "comfy" / "ComfyUI"][:others]:
        make_comfy(code)
    capsys.readouterr()
    assert _install_cli() == rc
    out, err = capsys.readouterr()
    assert f"removed {b / 'custom_nodes'} from the record: the folder is gone" in out
    assert ("no ComfyUI found" if rc == 1 else "found 2 ComfyUI installs") in err
    assert cu.load_record()["installs"] == []
    assert doctor.check_comfyui() is None
    assert "folder missing" not in json.dumps(cu.status())


def test_install_names_each_comfyui_it_finds_without_the_nodes(home, capsys):
    """A ComfyUI added after the first install gets a note with the command
    that adds the nodes to it, since a bare install refreshes the recorded
    folders only."""
    make_comfy(home / "ComfyUI")
    assert _install_cli() == 0
    code = make_comfy(home / "ComfyUI-Installs" / "s" / "ComfyUI", venv=None)
    _installations(home, [{"sourceId": "standalone", "installPath": str(code.parent),
                           "name": "Stable"}])
    capsys.readouterr()
    assert _install_cli() == 0
    out = capsys.readouterr().out
    assert not (code / "custom_nodes" / "comfyui-gmlx").exists()
    cmd = "gmlx comfyui install --comfyui-dir ~/ComfyUI-Installs/s/ComfyUI"
    assert ("note: ~/ComfyUI-Installs/s/ComfyUI (Comfy Desktop install 'Stable') has no "
            f"gmlx nodes. To add them, run {cmd}") in out
    assert cu.cmd_comfyui(cmd.split()[2:]) == 0
    assert (code / "custom_nodes" / "comfyui-gmlx").is_symlink()
    capsys.readouterr()
    assert _install_cli() == 0
    assert "note:" not in capsys.readouterr().out


def test_a_folder_name_that_needs_quotes_is_quoted_in_the_note(home, tmp_path, capsys):
    make_comfy(home / "ComfyUI")
    assert _install_cli("--comfyui-dir", str(home / "ComfyUI")) == 0
    code = make_comfy(home / "My Installs" / "ComfyUI", venv=None)
    _installations(home, [{"sourceId": "standalone", "installPath": str(code.parent),
                           "name": "x"}])
    capsys.readouterr()
    assert _install_cli() == 0
    assert f"--comfyui-dir '{code}'" in capsys.readouterr().out


def test_two_installs_at_once_both_stay_in_the_record(home, tmp_path):
    """Two processes install into two folders while a third holds the lock.
    Each reads the record once it has the lock, so neither loses the other's
    entry."""
    import subprocess
    import sys
    import time

    a, b = make_comfy(tmp_path / "a"), make_comfy(tmp_path / "b")
    child = ("import pathlib, sys\n"
             "import gmlx.commands.comfyui as cu\n"
             "pathlib.Path(sys.argv[2]).write_text('')\n"
             "sys.exit(cu.cmd_comfyui(['install', '--comfyui-dir', sys.argv[1]]))\n")
    tree = str(Path(cu.__file__).resolve().parents[2])
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([tree, os.environ.get("PYTHONPATH", "")]))
    lock = cu.data_root() / ".lock"
    lock.parent.mkdir(parents=True)
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        procs = [subprocess.Popen([sys.executable, "-c", child, str(d), str(tmp_path / f"{d.name}.ready")],
                                  env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                 for d in (a, b)]
        deadline = time.monotonic() + 60
        while not all((tmp_path / f"{d.name}.ready").exists() for d in (a, b)):
            assert time.monotonic() < deadline and all(p.poll() is None for p in procs)
            time.sleep(0.05)
        # Long enough for each to reach the lock.
        time.sleep(1.0)
    finally:
        os.close(fd)
    for p in procs:
        out, err = p.communicate(timeout=120)
        assert p.returncode == 0, err
    got = sorted(r["custom_nodes"] for r in cu.load_record()["installs"])
    assert got == [str(a / "custom_nodes"), str(b / "custom_nodes")]


def _install_while_waiting_for_the_lock(monkeypatch, folder):
    """Make the next taker of the lock find that an install into
    ``folder`` ran while it waited."""
    import contextlib

    real = cu.locked

    @contextlib.contextmanager
    def waited():
        monkeypatch.setattr(cu, "locked", real)
        assert cu.cmd_comfyui(["install", "--comfyui-dir", str(folder)]) == 0
        with real():
            yield

    monkeypatch.setattr(cu, "locked", waited)


def _recorded() -> list:
    return sorted(r["custom_nodes"] for r in cu.load_record()["installs"])


def test_an_uninstall_keeps_an_install_that_ran_while_it_waited(home, tmp_path,
                                                                monkeypatch):
    a, b = make_comfy(tmp_path / "a"), make_comfy(tmp_path / "b")
    assert _install_cli("--comfyui-dir", str(a)) == 0
    _install_while_waiting_for_the_lock(monkeypatch, b)
    assert cu.cmd_comfyui(["uninstall", "--comfyui-dir", str(a)]) == 0
    assert _recorded() == [str(b / "custom_nodes")]
    assert (b / "custom_nodes" / cu.LINK_NAME).exists()
    assert not (a / "custom_nodes" / cu.LINK_NAME).exists()


def test_a_refresh_keeps_an_install_that_ran_while_it_waited(home, tmp_path,
                                                            monkeypatch):
    a, b = make_comfy(tmp_path / "a"), make_comfy(tmp_path / "b")
    assert _install_cli("--comfyui-dir", str(a)) == 0
    # A recorded gmlx that is gone makes the refresh write the record.
    doc = cu.load_record()
    doc["gmlx_command"] = str(tmp_path / "gone" / "gmlx")
    cu.save_record(doc)
    new = tmp_path / "bin" / "gmlx"
    monkeypatch.setattr(cu, "gmlx_command", lambda: str(new))
    _install_while_waiting_for_the_lock(monkeypatch, b)
    cu.refresh_for_serve()
    assert _recorded() == [str(a / "custom_nodes"), str(b / "custom_nodes")]


def test_no_record_means_no_refresh(home):
    make_comfy(home / "ComfyUI")
    assert cu.refresh_for_serve() is None
    assert not cu.data_root().exists() and not cu.settings_path().exists()


def test_a_source_link_is_never_refreshed(home, monkeypatch):
    make_comfy(home / "ComfyUI")
    assert _install_cli("--link-source") == 0
    monkeypatch.setattr(cu, "gmlx_version", lambda: "99.0.0")
    assert cu.refresh_for_serve() is None
    assert not cu.copy_dir().exists()


def test_the_refresh_never_stops_the_server(home, monkeypatch):
    make_comfy(home / "ComfyUI")
    assert _install_cli() == 0
    monkeypatch.setattr(cu, "gmlx_version", lambda: "99.0.0")

    def broken(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(cu.shutil, "copytree", broken)
    assert "could not refresh" in cu.refresh_for_serve()


def _lock_is_held() -> bool:
    """Whether some open file holds the copy's lock: a new open file of it
    cannot take the lock then."""
    fd = os.open(cu.data_root() / ".lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(fd)
    return False


def test_a_refresh_holds_the_lock_while_it_copies(home, monkeypatch):
    """Two gmlx processes that refresh at once would delete each other's
    folders, so the copy and the swap happen under one lock."""
    make_comfy(home / "ComfyUI")
    assert _install_cli() == 0
    assert not _lock_is_held()
    monkeypatch.setattr(cu, "gmlx_version", lambda: "99.0.0")
    held = []
    real_copytree, real_replace = cu.shutil.copytree, os.replace

    def copytree(*a, **k):
        held.append(("copy", _lock_is_held()))
        return real_copytree(*a, **k)

    def replace(src, dst):
        held.append(("swap", _lock_is_held()))
        return real_replace(src, dst)

    monkeypatch.setattr(cu.shutil, "copytree", copytree)
    monkeypatch.setattr(cu.os, "replace", replace)
    assert "99.0.0" in cu.refresh_for_serve()
    # copytree calls itself for each subfolder.
    assert {step for step, _ in held} == {"copy", "swap"} and all(h for _, h in held)
    assert not _lock_is_held()


def test_the_version_the_copy_points_at_is_never_deleted(home, monkeypatch):
    """When another process turns the copy to its own version between the
    swap and the cleanup, that version stays."""
    make_comfy(home / "ComfyUI")
    assert _install_cli() == 0
    theirs = os.readlink(cu.copy_dir())
    monkeypatch.setattr(cu, "gmlx_version", lambda: "99.0.0")
    real_replace = os.replace

    def replace(src, dst):
        real_replace(src, dst)
        if str(dst) == str(cu.copy_dir()):
            tmp = cu.data_root() / ".other"
            os.symlink(theirs, tmp)
            real_replace(tmp, dst)

    monkeypatch.setattr(cu.os, "replace", replace)
    cu.sync_copy()
    assert os.readlink(cu.copy_dir()) == theirs
    assert (cu.copy_dir() / "__init__.py").is_file()


def test_the_lock_nests_and_is_taken_again_after_an_uninstall_removed_it(home,
                                                                        monkeypatch):
    make_comfy(home / "ComfyUI")
    assert _install_cli() == 0
    with cu.locked():
        with cu.locked():
            assert _lock_is_held()
        assert _lock_is_held()
    assert not _lock_is_held()
    # An uninstall deletes the lock file while this process waits for it:
    # the lock it then holds must be on the file that is there.
    real_flock, waited = fcntl.flock, []

    def flock(fd, op):
        real_flock(fd, op)
        if not waited:
            waited.append(fd)
            os.unlink(cu.data_root() / ".lock")

    monkeypatch.setattr(cu.fcntl, "flock", flock)
    with cu.locked():
        assert waited and _lock_is_held()
    monkeypatch.setattr(cu.fcntl, "flock", real_flock)
    assert cu.cmd_comfyui(["uninstall"]) == 0
    assert not cu.data_root().exists()


def test_the_refresh_records_the_running_gmlx_when_the_recorded_one_is_gone(
        home, monkeypatch, tmp_path):
    make_comfy(home / "ComfyUI")
    assert _install_cli() == 0
    doc = cu.load_record()
    cu.save_record({**doc, "gmlx_command": str(tmp_path / "gone" / "gmlx")})
    now = _fake_exe(tmp_path / "bin" / "gmlx")
    monkeypatch.setattr(cu, "gmlx_command", lambda: str(now))
    assert cu.refresh_for_serve() is None
    assert cu.load_record()["gmlx_command"] == str(now)
    # With no command to record, the record stays as it is.
    cu.save_record({**doc, "gmlx_command": None})
    monkeypatch.setattr(cu, "gmlx_command", lambda: None)
    assert cu.refresh_for_serve() is None
    assert cu.load_record()["gmlx_command"] is None


def test_serve_start_refreshes_the_copy(home, monkeypatch, tmp_path, capsys):
    import gmlx.serve.server as srv
    from gmlx.config import ModelCfg, ServerCfg

    make_comfy(home / "ComfyUI")
    assert _install_cli() == 0
    monkeypatch.setattr(cu, "gmlx_version", lambda: "99.0.0")
    g = tmp_path / "m.gguf"
    g.write_text("x")
    seen = []
    monkeypatch.setattr(srv, "_import_serving", lambda: None)
    monkeypatch.setattr(srv, "_resolve_cfg", lambda a: (
        ServerCfg(models={"m": ModelCfg(id="m", path=str(g))}), None))
    monkeypatch.setattr(srv, "_serve", lambda cfg, a, fn: seen.append(
        cu.read_stamp(cu.copy_dir())["gmlx_version"]) or 0)
    capsys.readouterr()
    assert srv._cmd_serve([str(g), "-f"]) == 0
    assert seen == ["99.0.0"]
    assert "[server] updated the ComfyUI node pack to gmlx 99.0.0" in capsys.readouterr().out


# Uninstall and status.

def test_uninstall_removes_links_copy_and_record_and_keeps_folders(home, capsys):
    code = make_comfy(home / "ComfyUI")
    real = code / "custom_nodes" / "ComfyUI-gmlx"
    real.mkdir()
    assert _install_cli() == 0
    cu.settings_path().write_text(json.dumps({**cu.load_record(), "base_url": "http://h:1"}))
    assert cu.cmd_comfyui(["uninstall"]) == 0
    out = capsys.readouterr().out
    assert "removed ~/ComfyUI/custom_nodes/comfyui-gmlx" in out
    assert not os.path.lexists(code / "custom_nodes" / "comfyui-gmlx")
    assert not cu.data_root().exists()
    assert len(list((code / "custom_nodes" / ".disabled").iterdir())) == 1
    assert json.loads(cu.settings_path().read_text()) == {"base_url": "http://h:1"}


def test_uninstall_one_folder_keeps_the_other(home):
    a = make_comfy(home / "ComfyUI")
    b = make_comfy(home / "comfy" / "ComfyUI")
    for code in (a, b):
        assert _install_cli("--comfyui-dir", str(code)) == 0
    assert cu.cmd_comfyui(["uninstall", "--comfyui-dir", str(a)]) == 0
    assert not os.path.lexists(a / "custom_nodes" / "comfyui-gmlx")
    assert (b / "custom_nodes" / "comfyui-gmlx" / "__init__.py").is_file()
    assert [r["comfyui"] for r in cu.load_record()["installs"]] == [str(b)]
    assert cu.cmd_comfyui(["uninstall"]) == 0
    assert not cu.settings_path().exists() and not cu.data_root().exists()


def test_uninstall_leaves_a_folder_that_is_not_its_link(home, tmp_path):
    code = make_comfy(home / "ComfyUI")
    other = tmp_path / "elsewhere"
    other.mkdir()
    (code / "custom_nodes" / "comfyui-gmlx").symlink_to(other)
    assert cu.cmd_comfyui(["uninstall"]) == 0
    assert os.path.islink(code / "custom_nodes" / "comfyui-gmlx")


def test_uninstall_from_another_gmlx_removes_a_source_link(home, monkeypatch, tmp_path):
    """A --link-source install from one gmlx tree, removed by a gmlx whose
    own pack is elsewhere: the recorded pack folder makes the link gmlx's."""
    code = make_comfy(home / "ComfyUI")
    assert _install_cli("--link-source") == 0
    monkeypatch.setattr(cu, "PACK_SOURCE", tmp_path / "other-gmlx" / "comfyui_pack")
    assert cu.cmd_comfyui(["uninstall"]) == 0
    assert not os.path.lexists(code / "custom_nodes" / cu.LINK_NAME)
    assert not cu.settings_path().exists()


def test_status_before_and_after_install(home, capsys):
    make_comfy(home / "ComfyUI", version="9.0.0")
    assert cu.cmd_comfyui(["status"]) == 1
    out = capsys.readouterr().out
    assert "not installed" in out and "~/ComfyUI" in out
    assert _install_cli() == 0
    capsys.readouterr()
    assert cu.cmd_comfyui(["status"]) == 0
    out = capsys.readouterr().out
    assert "~/ComfyUI/custom_nodes: linked" in out and "newer than" in out
    assert "managed by gmlx and update with it" in out
    assert cu.cmd_comfyui(["status", "--json"]) == 0
    s = json.loads(capsys.readouterr().out)
    assert s["copy"] == "current" and s["problems"] == []
    assert s["folders"][0]["comfyui_version"] == "9.0.0"


def test_status_reports_a_stale_copy_and_a_lost_link(home, monkeypatch, capsys):
    code = make_comfy(home / "ComfyUI")
    assert _install_cli() == 0
    os.unlink(code / "custom_nodes" / "comfyui-gmlx")
    monkeypatch.setattr(cu, "gmlx_version", lambda: "99.0.0")
    capsys.readouterr()
    assert cu.cmd_comfyui(["status"]) == 1
    out = capsys.readouterr().out
    assert "warning: the pack copy is from gmlx" in out
    assert "~/ComfyUI/custom_nodes: missing; run gmlx comfyui install" in out


def test_the_verb_dispatches_and_refuses_an_unknown_action(capsys):
    assert cu.cmd_comfyui([]) == 0
    assert "install" in capsys.readouterr().out
    assert cu.cmd_comfyui(["frob"]) == 2


def test_doctor_row(home):
    assert doctor.check_comfyui() is None
    code = make_comfy(home / "ComfyUI")
    assert _install_cli() == 0
    row = doctor.check_comfyui()
    assert row["status"] == "PASS" and "~/ComfyUI/custom_nodes (ComfyUI 0.38.0)" in row["detail"]
    os.unlink(code / "custom_nodes" / "comfyui-gmlx")
    row = doctor.check_comfyui()
    assert row["status"] == "WARN" and "missing" in row["detail"]


def test_doctor_passes_a_comfyui_newer_than_tested_with_a_note(home):
    make_comfy(home / "ComfyUI", version="9.0.0")
    assert _install_cli() == 0
    row = doctor.check_comfyui()
    tested = cu.pack_meta()["COMFYUI_TESTED"]
    assert row["status"] == "PASS"
    assert (f"~/ComfyUI/custom_nodes (ComfyUI 9.0.0, newer than {tested}, the newest this "
            "gmlx was tested on)") in row["detail"]
    # Older than the floor is a problem.
    (home / "ComfyUI" / "comfyui_version.py").write_text('__version__ = "0.30.0"\n')
    row = doctor.check_comfyui()
    floor = cu.pack_meta()["COMFYUI_FLOOR"]
    assert row["status"] == "WARN"
    assert f"ComfyUI 0.30.0 is older than {floor}, which the nodes need" in row["detail"]


def test_doctor_reports_any_failure_to_read_the_pack_as_a_row(home, monkeypatch):
    make_comfy(home / "ComfyUI")
    assert _install_cli() == 0

    def broken(doc=None):
        raise TypeError("'int' object is not iterable")

    monkeypatch.setattr(cu, "status", broken)
    row = doctor.check_comfyui()
    assert row["status"] == "WARN" and "not iterable" in row["detail"]


def _fake_exe(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return path


def test_the_recorded_gmlx_is_the_running_one_never_a_path_lookup(monkeypatch, tmp_path):
    """Another gmlx on PATH is never recorded: the console script beside the
    interpreter, else the script this process started from, else none."""
    monkeypatch.setenv("PATH", str(_fake_exe(tmp_path / "other" / "gmlx").parent))
    venv = tmp_path / "venv"
    monkeypatch.setattr(cu.sys, "prefix", str(venv))
    monkeypatch.setattr(cu.sys, "executable", str(venv / "bin" / "python"))
    monkeypatch.setattr(cu.sys, "argv", [str(tmp_path / "src" / "gmlx" / "__main__.py")])
    monkeypatch.setattr(cu, "_scripts_dirs", lambda: [str(tmp_path / "no-scripts")])
    assert cu.gmlx_command() is None
    started = _fake_exe(tmp_path / "bin" / "gmlx")
    monkeypatch.setattr(cu.sys, "argv", [str(started), "comfyui", "install"])
    assert cu.gmlx_command() == os.path.realpath(started)
    script = _fake_exe(venv / "bin" / "gmlx")
    script.write_text(f"#!{venv / 'bin' / 'python'}\nimport gmlx\n")
    assert cu.gmlx_command() == str(script)


def test_the_recorded_gmlx_of_homebrew_is_its_link_that_survives_upgrades(monkeypatch,
                                                                         tmp_path):
    prefix = tmp_path / "homebrew"
    libexec = prefix / "Cellar" / "gmlx" / "0.5.0" / "libexec"
    script = _fake_exe(libexec / "bin" / "gmlx")
    script.write_text(f"#!{libexec / 'bin' / 'python'}\nimport gmlx\n")
    (prefix / "bin").mkdir()
    (prefix / "bin" / "gmlx").symlink_to(script)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(cu.sys, "prefix", str(libexec))
    monkeypatch.setattr(cu.sys, "executable", str(libexec / "bin" / "python"))
    assert cu.gmlx_command() == str(prefix / "bin" / "gmlx")
    # A brew link to another formula's gmlx is not this install's.
    (prefix / "bin" / "gmlx").unlink()
    (prefix / "bin" / "gmlx").symlink_to(_fake_exe(tmp_path / "elsewhere" / "gmlx"))
    assert cu.gmlx_command() == str(script)


def test_install_kind_is_read_from_where_gmlx_runs(monkeypatch, tmp_path):
    cellar = tmp_path / "Cellar" / "gmlx" / "0.5.0" / "libexec"
    cellar.mkdir(parents=True)
    monkeypatch.setattr(cu.sys, "prefix", str(cellar))
    assert cu.gmlx_install() == "homebrew"
    tool = tmp_path / "tools" / "gmlx"
    tool.mkdir(parents=True)
    (tool / "uv-receipt.toml").write_text("")
    monkeypatch.setattr(cu.sys, "prefix", str(tool))
    assert cu.gmlx_install() == "uv"
    monkeypatch.setattr(cu.sys, "prefix", str(tmp_path))
    assert cu.gmlx_install() == "pip"


def _user_scripts(monkeypatch, tmp_path, *, venv: bool):
    """A Python whose own scripts folder holds no gmlx, and the user's
    scripts folder, ~/.local/bin as on uv's Pythons. Returns the user's
    gmlx path and this interpreter."""
    base = tmp_path / "python"
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(cu.sys, "prefix", str(tmp_path / "venv") if venv else str(base))
    monkeypatch.setattr(cu.sys, "base_prefix", str(base))
    monkeypatch.setattr(cu.sys, "executable", str(base / "bin" / "python3.12"))
    monkeypatch.setattr(cu.sys, "argv", ["-m"])
    user = tmp_path / ".local" / "bin" / "gmlx"

    def get_path(name, scheme=None):
        assert name == "scripts"
        return str(user.parent) if scheme else str(tmp_path / "scripts")

    monkeypatch.setattr(cu.sysconfig, "get_path", get_path)
    return user, cu.sys.executable


def test_the_recorded_gmlx_may_come_from_the_users_scripts_folder(monkeypatch, tmp_path):
    """A pip install --user puts the console script in the user's scripts
    folder, not beside the interpreter."""
    user, python = _user_scripts(monkeypatch, tmp_path, venv=False)
    _fake_exe(user).write_text(f"#!{python}\nimport gmlx\n")
    assert cu.gmlx_command() == str(user)


def test_another_installs_gmlx_in_the_users_scripts_folder_is_not_recorded(monkeypatch,
                                                                           tmp_path):
    """uv tool and pipx put their gmlx in ~/.local/bin, which is also the
    user's scripts folder of uv's Pythons. Its shebang names their own
    interpreter, so it is not this install's."""
    user, _python = _user_scripts(monkeypatch, tmp_path, venv=False)
    tool = tmp_path / ".local" / "share" / "uv" / "tools" / "gmlx" / "bin" / "python"
    _fake_exe(user).write_text(f"#!{tool}\nimport gmlx\n")
    assert cu.gmlx_command() is None


def test_brews_gmlx_in_homebrew_pythons_scripts_folder_is_not_recorded(monkeypatch,
                                                                       tmp_path):
    """Homebrew's Python puts console scripts in Homebrew's bin folder,
    where brew also links its own gmlx, which runs on the gmlx formula's
    Python. `python3.13 -m gmlx` from a checkout records none of it,
    whether the interpreter runs from its opt folder or from that bin
    folder."""
    brew = tmp_path / "homebrew"
    formula = brew / "Cellar" / "gmlx" / "0.5.0" / "libexec" / "bin" / "python"
    _fake_exe(formula.parent / "gmlx").write_text(f"#!{formula}\nimport gmlx\n")
    (brew / "bin").mkdir(parents=True)
    (brew / "bin" / "gmlx").symlink_to(formula.parent / "gmlx")
    monkeypatch.setenv("PATH", str(brew / "bin"))
    monkeypatch.setattr(cu.sys, "prefix", str(brew / "opt" / "python"))
    monkeypatch.setattr(cu.sys, "base_prefix", str(brew / "opt" / "python"))
    monkeypatch.setattr(cu.sys, "argv", ["-m"])
    monkeypatch.setattr(cu.sysconfig, "get_path",
                        lambda name, scheme=None: str(
                            tmp_path / "user-bin" if scheme else brew / "bin"))
    for python in (brew / "opt" / "python" / "bin" / "python3.13",
                   brew / "bin" / "python3.13"):
        monkeypatch.setattr(cu.sys, "executable", str(python))
        assert cu.gmlx_command() is None, python
    # A pip install into that Python names it in its shebang, and is recorded.
    (brew / "bin" / "gmlx").unlink()
    _fake_exe(brew / "bin" / "gmlx").write_text(f"#!{python}\nimport gmlx\n")
    assert cu.gmlx_command() == str(brew / "bin" / "gmlx")


def _sh_launcher(python) -> str:
    """The script head pip writes when the interpreter path has spaces."""
    q = "'" * 3
    return f'#!/bin/sh\n{q}exec\' "{python}" "$0" "$@"\n\' {q}\nimport gmlx\n'


def test_pips_shell_shebang_for_a_path_with_spaces_names_the_python(monkeypatch,
                                                                    tmp_path):
    user, python = _user_scripts(monkeypatch, tmp_path, venv=False)
    _fake_exe(user).write_text(_sh_launcher(python))
    assert cu.gmlx_command() == str(user)
    user.write_text(_sh_launcher(tmp_path / "other app" / "python"))
    assert cu.gmlx_command() is None


def _uv_relocatable(name="python") -> str:
    """The script head uv writes in a venv made with --relocatable."""
    q = "'" * 3
    return (f"#!/bin/sh\n{q}exec' \"$(dirname -- \"$(realpath -- \"$0\")\")\"/'{name}' "
            f"\"$0\" \"$@\"\n' {q}\nimport gmlx\n")


def test_a_relocatable_venvs_gmlx_is_recorded(monkeypatch, tmp_path):
    """uv's relocatable launcher names python beside the script itself."""
    base = _fake_exe(tmp_path / "base" / "bin" / "python3.12")
    venv, other = tmp_path / "venv", tmp_path / "other"
    for v in (venv, other):
        (v / "bin").mkdir(parents=True)
        (v / "bin" / "python").symlink_to(base)
        (v / "bin" / "python3").symlink_to(base)
        _fake_exe(v / "bin" / "gmlx").write_text(_uv_relocatable())
    monkeypatch.setattr(cu.sys, "prefix", str(venv))
    monkeypatch.setattr(cu.sys, "argv", ["-m"])
    monkeypatch.setattr(cu, "_scripts_dirs", lambda: [])
    monkeypatch.setenv("PATH", str(other / "bin"))
    for exe in ("python", "python3"):
        monkeypatch.setattr(cu.sys, "executable", str(venv / "bin" / exe))
        assert cu.gmlx_command() == str(venv / "bin" / "gmlx"), exe
    # Another venv's relocatable gmlx names its own python.
    (venv / "bin" / "gmlx").unlink()
    assert cu.gmlx_command() is None


def test_homebrews_opt_and_cellar_spellings_are_one_python(monkeypatch, tmp_path):
    """Homebrew's Python runs as opt/python@3.13/bin/python3.13 while pip
    writes the Cellar path in a shebang, or the other way round."""
    cellar = tmp_path / "Cellar" / "python@3.13" / "3.13.11"
    real = _fake_exe(cellar / "bin" / "python3.13")
    (tmp_path / "opt").mkdir()
    (tmp_path / "opt" / "python@3.13").symlink_to(cellar)
    opt = tmp_path / "opt" / "python@3.13" / "bin" / "python3.13"
    script = tmp_path / "scripts" / "gmlx"
    monkeypatch.setattr(cu.sys, "executable", str(opt))
    _fake_exe(script).write_text(f"#!{real}\nimport gmlx\n")
    assert cu._runs_on_this_python(str(script))
    monkeypatch.setattr(cu.sys, "executable", str(real))
    script.write_text(f"#!{opt}\nimport gmlx\n")
    assert cu._runs_on_this_python(str(script))
    # A venv's python links to that Python from another folder.
    venv = tmp_path / "venv" / "bin" / "python"
    venv.parent.mkdir(parents=True)
    venv.symlink_to(real)
    script.write_text(f"#!{venv}\nimport gmlx\n")
    assert not cu._runs_on_this_python(str(script))


def test_a_shebang_may_name_this_python_by_another_name_in_its_folder(monkeypatch,
                                                                      tmp_path):
    user, python = _user_scripts(monkeypatch, tmp_path, venv=False)
    real = _fake_exe(Path(python))
    alias = real.parent / "python3"
    alias.symlink_to(real.name)
    _fake_exe(user).write_text(f"#!{alias}\nimport gmlx\n")
    assert cu.gmlx_command() == str(user)


def test_a_venv_has_no_users_scripts_folder(monkeypatch, tmp_path):
    user, python = _user_scripts(monkeypatch, tmp_path, venv=True)
    _fake_exe(user).write_text(f"#!{python}\nimport gmlx\n")
    assert cu.gmlx_command() is None
    assert cu._scripts_dirs() == [str(tmp_path / "scripts")]


def test_a_gmlx_on_path_is_recorded_only_when_it_runs_on_this_python(monkeypatch,
                                                                     tmp_path):
    base = _fake_exe(tmp_path / "base" / "python3")
    venv, other = tmp_path / "venv", tmp_path / "other-venv"
    for v in (venv, other):
        (v / "bin").mkdir(parents=True)
        (v / "bin" / "python").symlink_to(base)
    monkeypatch.setattr(cu.sys, "prefix", str(venv))
    monkeypatch.setattr(cu.sys, "executable", str(venv / "bin" / "python"))
    monkeypatch.setattr(cu.sys, "argv", ["-m"])
    monkeypatch.setattr(cu, "_scripts_dirs", lambda: [])
    on_path = tmp_path / "path" / "gmlx"
    _fake_exe(on_path).write_text(f"#!{venv / 'bin' / 'python'}\nimport gmlx\n")
    monkeypatch.setenv("PATH", str(on_path.parent))
    assert cu.gmlx_command() == str(on_path)
    # Another venv's python links to the same base Python, and is still
    # another install.
    on_path.write_text(f"#!{other / 'bin' / 'python'}\nimport gmlx\n")
    assert cu.gmlx_command() is None


_BARE = r'''
import importlib.abc
import runpy
import sys

tree, comfy = sys.argv[1:3]
sys.path.insert(0, tree)
allowed = set(sys.stdlib_module_names) | {"gmlx", "__main__"}
tried = []


class NoThirdParty(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        top = name.partition(".")[0]
        if top in allowed or top.startswith("_sysconfigdata_"):
            return None
        tried.append(name)
        raise ImportError(f"{name} is not installed here", name=name)


sys.meta_path.insert(0, NoThirdParty())
sys.argv = ["gmlx", "comfyui", "install", "--comfyui-dir", comfy]
code = 0
try:
    runpy.run_module("gmlx", run_name="__main__", alter_sys=True)
except SystemExit as e:
    code = e.code
print("tried:", sorted(set(tried)), file=sys.stderr)
sys.exit(code)
'''


def test_python_m_gmlx_comfyui_install_runs_without_gmlxs_dependencies(home, tmp_path):
    """CI installs the pack with ``python -m gmlx comfyui install`` on a
    Python that has none of gmlx's dependencies. A child that refuses every
    package outside the standard library and gmlx runs that path."""
    import subprocess
    import sys

    code = make_comfy(tmp_path / "ComfyUI")
    tree = str(Path(cu.__file__).resolve().parents[2])
    r = subprocess.run([sys.executable, "-I", "-c", _BARE, tree, str(code)],
                       env=dict(os.environ), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "installed the gmlx nodes in" in r.stdout
    link = code / "custom_nodes" / cu.LINK_NAME
    assert os.path.realpath(link) == os.path.realpath(cu.copy_dir())
    assert [i["custom_nodes"] for i in cu.load_record()["installs"]] == [
        str(code / "custom_nodes")]
