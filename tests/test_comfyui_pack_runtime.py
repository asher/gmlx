"""The node pack's runtime pieces that need neither torch nor ComfyUI: the
recorded gmlx command and its PATH fallback, the message that names the two
paths to delete, the upgrade command per install kind, the default model,
and the stand-in nodes a failing full import registers. The pack loads the
way ComfyUI loads it, through the link `gmlx comfyui install` makes in a
fake ComfyUI folder. CPU only."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import gmlx
import gmlx.commands.comfyui as cu

_ROOT = Path(__file__).resolve().parents[1]
_PACK = _ROOT / "gmlx" / "comfyui_pack"


def _fake_exe(folder: Path, name: str = "gmlx") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    exe = folder / name
    exe.write_text("#!/bin/sh\nexit 0\n")
    exe.chmod(0o755)
    return exe


@pytest.fixture
def link(monkeypatch, tmp_path):
    """The pack installed into a fake ComfyUI; its link in custom_nodes."""
    home = tmp_path / "home"
    code = home / "ComfyUI"
    (code / "custom_nodes").mkdir(parents=True)
    (code / "main.py").write_text("")
    (code / "comfyui_version.py").write_text('__version__ = "0.38.0"\n')
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("GMLX_DEFAULT_IMAGE_MODEL", raising=False)
    cu.install([cu.target_for_dir(code)], model="qwen-image")
    return code / "custom_nodes" / cu.LINK_NAME


def _settings(link: Path):
    """The pack's settings module as loaded through the link."""
    spec = importlib.util.spec_from_file_location("_pack_settings", link / "settings.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _record(**changes):
    doc = cu.load_record()
    doc.update(changes)
    cu.save_record(doc)


def test_the_pack_reads_every_key_install_records(link):
    text = (_PACK / "settings.py").read_text()
    for key in ("gmlx_command", "gmlx_install", "gmlx_python", "default_image_model",
                "pack_dir", "pack_source"):
        assert f'"{key}"' in text and key in cu._RECORD_KEYS, key


def test_the_recorded_gmlx_comes_first_then_path(link, monkeypatch, tmp_path):
    s = _settings(link)
    recorded = _fake_exe(tmp_path / "cellar" / "bin")
    on_path = _fake_exe(tmp_path / "path")
    _record(gmlx_command=str(recorded))
    monkeypatch.setenv("PATH", str(on_path.parent))
    assert s.gmlx_command() == str(recorded)
    recorded.unlink()
    assert s.gmlx_command() == str(on_path)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert s.gmlx_command() is None


def test_no_gmlx_names_the_link_and_the_copy_folder_to_delete(link, monkeypatch, tmp_path):
    s = _settings(link)
    assert s.PACK_DIR == str(link)
    _record(gmlx_command=str(tmp_path / "gone" / "gmlx"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    msg = s.not_installed()
    assert f"{tmp_path / 'gone' / 'gmlx'} does not exist" in msg
    assert f"deleting {link} and {cu.data_root()}" in msg
    assert s.INSTALL_URL in msg
    # Without the record, the copy folder comes from where the link points.
    cu.settings_path().unlink()
    assert s.leftovers() == [str(link), os.path.realpath(cu.data_root())]
    # Deleting the two paths removes the pack.
    link.unlink()
    shutil.rmtree(cu.data_root())
    assert not os.listdir(link.parent)


def test_upgrade_command_per_install_kind(link):
    s = _settings(link)
    assert s.upgrade_command({"gmlx_install": "homebrew"}) == "brew upgrade gmlx"
    assert s.upgrade_command({"gmlx_install": "uv"}) == "uv tool upgrade gmlx"
    assert s.upgrade_command({"gmlx_install": "pip", "gmlx_python": "/a b/python"}) == \
        "'/a b/python' -m pip install -U gmlx pillow opencv-python"
    assert s.upgrade_command({}) is None
    assert s.INSTALL_URL in s.upgrade_step({})
    rec = cu.load_record()
    assert s.upgrade_command() == s.upgrade_command(
        {"gmlx_install": rec["gmlx_install"], "gmlx_python": rec["gmlx_python"]})


def test_the_install_url_is_the_installation_page(link):
    assert _settings(link).INSTALL_URL == f"{gmlx.DOCS_URL}installation.html"
    assert (_ROOT / "docs" / "installation.md").is_file()


def test_the_default_model_is_the_recorded_one(link, monkeypatch):
    s = _settings(link)
    assert s.default_image_model() == "qwen-image"
    monkeypatch.setenv("GMLX_DEFAULT_IMAGE_MODEL", "flux")
    assert s.default_image_model() == "flux"
    monkeypatch.delenv("GMLX_DEFAULT_IMAGE_MODEL")
    cu.settings_path().unlink()
    assert s.default_image_model() == "z-image-turbo"


_LOAD = r"""
import importlib.util, json, sys
link, root = sys.argv[1:3]
if root:
    sys.path.insert(0, root)
spec = importlib.util.spec_from_file_location("comfyui-gmlx", link + "/__init__.py",
                                              submodule_search_locations=[link])
m = importlib.util.module_from_spec(spec)
sys.modules["comfyui-gmlx"] = m
try:
    spec.loader.exec_module(m)
except ImportError as e:
    print(json.dumps({"import_error": str(e)}))
    raise SystemExit(0)
nodes = m.NODE_CLASS_MAPPINGS
out = {"names": sorted(nodes), "display": m.NODE_DISPLAY_NAME_MAPPINGS,
       "inputs": nodes["GmlxModelLoader"].INPUT_TYPES(),
       "status": nodes["GmlxServerStatus"]().run(model_name="x")["result"][0],
       "torch": "torch" in sys.modules}
try:
    nodes["GmlxModelLoader"]().run(model_name="x")
except RuntimeError as e:
    out["error"] = str(e)
print(json.dumps(out))
"""


def _load_in_a_comfyui(link: Path, root: Path | None, path: str) -> dict:
    env = {"GMLX_COMFY_SETTINGS": str(cu.settings_path()), "PATH": path,
           "HOME": os.environ["HOME"]}
    r = subprocess.run([sys.executable, "-I", "-c", _LOAD, str(link), str(root or "")],
                       capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def _fake_comfyui(tmp_path) -> Path:
    """A ComfyUI that changed every internal the pack uses."""
    root = tmp_path / "newcomfy"
    (root / "comfy").mkdir(parents=True)
    (root / "comfy" / "__init__.py").write_text("")
    (root / "comfyui_version.py").write_text('__version__ = "0.99.0"\n')
    return root


def _node_names() -> list:
    tree = ast.parse((_PACK / "nodes.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == \
                "NODE_CLASS_MAPPINGS":
            return sorted(ast.literal_eval(k) for k in node.value.keys)
    raise AssertionError("no NODE_CLASS_MAPPINGS in nodes.py")


@pytest.mark.parametrize("kind, says", [
    ("homebrew", "brew upgrade gmlx"),
    ("uv", "uv tool upgrade gmlx"),
    ("pip", "-m pip install -U gmlx pillow opencv-python"),
    (None, "installation.html"),
])
def test_a_failing_import_registers_stand_ins_that_name_the_fix(link, tmp_path, kind, says):
    gmlx_bin = _fake_exe(tmp_path / "bin")
    _record(gmlx_install=kind, gmlx_command=str(gmlx_bin))
    out = _load_in_a_comfyui(link, _fake_comfyui(tmp_path), str(tmp_path / "empty"))
    assert out["names"] == _node_names()
    assert all(v.endswith("(not loaded)") for v in out["display"].values())
    assert out["inputs"]["required"]["model_name"][1]["default"] == "qwen-image"
    assert out["torch"] is False
    err = out["error"]
    for want in ("comfy.model_patcher.ModelPatcher", "ComfyUI 0.99.0",
                 f"up to {cu.pack_meta()['COMFYUI_TESTED']}", says, "restart ComfyUI"):
        assert want in err, want
    assert "not installed" not in err
    assert out["status"] == err


def test_a_stand_in_with_gmlx_gone_names_the_two_paths(link, tmp_path):
    _record(gmlx_command=str(tmp_path / "gone" / "gmlx"))
    out = _load_in_a_comfyui(link, _fake_comfyui(tmp_path), str(tmp_path / "empty"))
    assert "gmlx is not installed" in out["error"]
    assert f"deleting {link} and {cu.data_root()}" in out["error"]


def test_outside_comfyui_the_import_fails_as_it_is(link, tmp_path):
    out = _load_in_a_comfyui(link, None, str(tmp_path / "empty"))
    assert "comfy" in out["import_error"]
