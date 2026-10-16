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
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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


def _discovery(link: Path):
    """The pack's discovery module, loaded through the link as part of a
    package whose __init__ does not run, with nothing kept from a probe."""
    name = "_pack_standalone"
    for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
        del sys.modules[key]
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(link)]
    sys.modules[name] = pkg
    return importlib.import_module(f"{name}.discovery")


def _catalog() -> list:
    from gmlx.image.catalog import RECIPES
    return list(RECIPES)


def _dead_port() -> int:
    """A loopback port nothing listens on."""
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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


def test_a_source_link_names_only_the_link_to_delete(link):
    """A --link-source install points into a gmlx checkout, whose folders
    are never named for deletion."""
    code = link.parent.parent
    cu.install([cu.target_for_dir(code)], link_source=True)
    s = _settings(link)
    assert os.path.realpath(link) == os.path.realpath(_PACK)
    assert s.leftovers() == [str(link)]
    assert str(_PACK.parent) not in s.not_installed()


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
import logging
records = []
handler = logging.Handler()
handler.emit = records.append
logging.getLogger("gmlx").addHandler(handler)
try:
    spec.loader.exec_module(m)
except ImportError as e:
    print(json.dumps({"import_error": str(e)}))
    raise SystemExit(0)
nodes = m.NODE_CLASS_MAPPINGS
out = {"names": sorted(nodes), "display": m.NODE_DISPLAY_NAME_MAPPINGS,
       "inputs": nodes["GmlxModelLoader"].INPUT_TYPES(),
       "validate": nodes["GmlxModelLoader"].VALIDATE_INPUTS("not-listed"),
       "status": nodes["GmlxServerStatus"]().run(model_name="x")["result"][0],
       "torch": "torch" in sys.modules,
       "logged": [[r.levelname, r.exc_info[0].__name__ if r.exc_info else None]
                  for r in records]}
try:
    nodes["GmlxModelLoader"]().run(model_name="x")
except RuntimeError as e:
    out["error"] = str(e)
print(json.dumps(out))
"""


def _load_in_a_comfyui(link: Path, root: Path | None, path: str) -> dict:
    # A server that answers nowhere, so no server on this Mac feeds the list.
    env = {"GMLX_COMFY_SETTINGS": str(cu.settings_path()), "PATH": path,
           "HOME": os.environ["HOME"], "GMLX_BASE_URL": f"http://127.0.0.1:{_dead_port()}"}
    r = subprocess.run([sys.executable, "-I", "-c", _LOAD, str(link), str(root or "")],
                       capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def _fake_comfyui(tmp_path, version: str = "0.99.0", patcher: str | None = None) -> Path:
    """A ComfyUI that changed every internal the pack uses, or whose
    model_patcher module runs ``patcher`` when imported."""
    root = tmp_path / "newcomfy"
    (root / "comfy").mkdir(parents=True)
    (root / "comfy" / "__init__.py").write_text("")
    if patcher is not None:
        (root / "comfy" / "model_patcher.py").write_text(patcher)
    (root / "comfyui_version.py").write_text(f'__version__ = "{version}"\n')
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
    # gmlx answers with nothing, so the list holds only the recorded
    # default, and any saved name still reaches run.
    names, opts = out["inputs"]["required"]["model_name"]
    assert names == ["qwen-image"] and opts["default"] == "qwen-image"
    assert out["validate"] is True
    assert out["torch"] is False
    err = out["error"]
    for want in ("comfy.model_patcher.ModelPatcher", "ComfyUI 0.99.0",
                 f"up to {cu.pack_meta()['COMFYUI_TESTED']}", says,
                 "then run `gmlx comfyui install` and restart ComfyUI"):
        assert want in err, want
    assert "not installed" not in err
    assert out["status"] == err
    # The log keeps the traceback of the import that failed.
    assert out["logged"] == [["ERROR", "ImportError"]]


def test_any_error_in_the_pack_import_registers_stand_ins(link, tmp_path):
    _record(gmlx_install="homebrew", gmlx_command=str(_fake_exe(tmp_path / "bin")))
    root = _fake_comfyui(tmp_path, patcher="raise RuntimeError('a ComfyUI module broke')\n")
    out = _load_in_a_comfyui(link, root, str(tmp_path / "empty"))
    assert out["names"] == _node_names()
    assert "RuntimeError: a ComfyUI module broke" in out["error"]
    assert out["logged"] == [["ERROR", "RuntimeError"]]


def test_a_comfyui_older_than_the_floor_is_the_one_to_update(link, tmp_path):
    _record(gmlx_install="homebrew", gmlx_command=str(_fake_exe(tmp_path / "bin")))
    out = _load_in_a_comfyui(link, _fake_comfyui(tmp_path, version="0.30.0"),
                             str(tmp_path / "empty"))
    err = out["error"]
    assert f"They need ComfyUI {cu.pack_meta()['COMFYUI_FLOOR']} or newer" in err
    assert "Update ComfyUI, then restart it" in err and f"{gmlx.DOCS_URL}comfyui.html" in err
    assert "brew upgrade gmlx" not in err


def test_the_guide_link_ends_the_message_when_gmlx_is_gone_too(link, tmp_path):
    _record(gmlx_command=str(tmp_path / "gone" / "gmlx"))
    out = _load_in_a_comfyui(link, _fake_comfyui(tmp_path, version="0.30.0"),
                             str(tmp_path / "empty"))
    err = out["error"]
    assert "Update ComfyUI, then restart it. gmlx is not installed: " in err
    assert err.endswith(f". The ComfyUI guide: {gmlx.DOCS_URL}comfyui.html")


def test_a_stand_in_with_gmlx_gone_names_the_two_paths(link, tmp_path):
    _record(gmlx_command=str(tmp_path / "gone" / "gmlx"))
    out = _load_in_a_comfyui(link, _fake_comfyui(tmp_path), str(tmp_path / "empty"))
    assert "gmlx is not installed" in out["error"]
    assert f"deleting {link} and {cu.data_root()}" in out["error"]


def test_outside_comfyui_the_import_fails_as_it_is(link, tmp_path):
    out = _load_in_a_comfyui(link, None, str(tmp_path / "empty"))
    assert "comfy" in out["import_error"]


# The loaders' widgets: the model list and the server address.

def _gmlx_script(folder: Path, body: str) -> Path:
    """A gmlx command whose shell ``body`` sees the arguments as "$1 $2"."""
    folder.mkdir(parents=True, exist_ok=True)
    exe = folder / "gmlx"
    exe.write_text("#!/bin/sh\n" + body + "\n")
    exe.chmod(0o755)
    return exe


def _answers(listing: dict | None = None, status: dict | None = None) -> str:
    """Shell that prints ``listing`` for `list` and ``status`` for `status`,
    with builtins only, as the tests run with an empty PATH."""
    lines = ['case "$1" in']
    if listing is not None:
        lines.append(f"  list) echo '{json.dumps(listing)}' ;;")
    if status is not None:
        lines.append(f"  status) echo '{json.dumps(status)}' ;;")
    lines.append("esac")
    return "\n".join(lines)


def _listing(*ids, default=None) -> dict:
    return {"config": "/x/gmlx.yaml", "aliases": {}, "default_model": None,
            "default_image_model": default,
            "models": [{"id": i, "kind": "image", "default": i == default} for i in ids]}


@pytest.fixture
def widgets(link, monkeypatch, tmp_path):
    """The discovery module of the installed pack, with no server answering."""
    d = _discovery(link)
    monkeypatch.setenv("GMLX_BASE_URL", f"http://127.0.0.1:{_dead_port()}")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.delenv("GMLX_API_KEY", raising=False)
    monkeypatch.delenv("GMLX_DEFAULT_IMAGE_MODEL", raising=False)
    _record(gmlx_command=str(tmp_path / "gone" / "gmlx"))
    yield d
    d.forget()


def _probed(d) -> dict:
    """The loader inputs after a probe from now has answered."""
    d.forget()
    d.refresh().join(30)
    return d.loader_inputs()["required"]


def test_pull_names_are_the_catalog_recipes(widgets):
    from gmlx.image.catalog import RECIPES
    assert widgets.PULL_NAMES == {r.name: tuple(r.aliases) for r in RECIPES.values()}
    assert widgets.PULL_KINDS == {r.name: (r.family, r.variant) for r in RECIPES.values()}
    assert list(widgets.PULL_NAMES) == _catalog()
    assert widgets.settings.FALLBACK_MODEL in widgets.PULL_NAMES


def test_a_kept_list_is_probed_again_after_keep_seconds(widgets, tmp_path, monkeypatch):
    """A slow probe keeps its answer no longer: R after a pull shows the new
    list within seconds."""
    monkeypatch.setattr(widgets, "KEEP", 0.2)
    exe = _gmlx_script(tmp_path / "bin", "/bin/sleep 1.5; " + _answers(_listing("a-model")))
    _record(gmlx_command=str(exe))
    assert _probed(widgets)["model_name"][0] == ["a-model"]
    _gmlx_script(tmp_path / "bin", _answers(_listing("b-model")))
    time.sleep(0.3)
    widgets.loader_inputs()            # old answer; starts a probe
    widgets.refresh().join(30)
    assert widgets.loader_inputs()["required"]["model_name"][0] == ["b-model"]


def test_the_model_list_is_the_configured_ids_default_first(widgets, tmp_path, monkeypatch):
    exe = _gmlx_script(tmp_path / "bin", _answers(
        _listing("krea-2-turbo", "my-qwen", "z-image-turbo", default="my-qwen")))
    _record(gmlx_command=str(exe), default_image_model="z-image-turbo")
    names, opts = _probed(widgets)["model_name"]
    assert names == ["my-qwen", "krea-2-turbo", "z-image-turbo"]
    assert opts["default"] == "my-qwen"
    assert widgets.widgets()["source"] == "config"
    # launch's --model comes first.
    monkeypatch.setenv("GMLX_DEFAULT_IMAGE_MODEL", "z-image-turbo")
    assert _probed(widgets)["model_name"][0][0] == "z-image-turbo"


class _Models(BaseHTTPRequestHandler):
    ids: list = []
    default = None

    def do_GET(self):
        ok = (self.path == "/internal/image-steps/v1/models"
              and self.headers.get("X-Gmlx-Steps-Contract") == "1.0")
        body = json.dumps({"models": [{"model": i} for i in self.ids],
                           "default_model": self.default}).encode()
        self.send_response(200 if ok else 400)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_a_server_that_answers_gives_the_list(widgets, tmp_path, monkeypatch):
    _Models.ids, _Models.default = ["qwen-image-2.1", "z-image-turbo", "krea-2-turbo"], None
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Models)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_port}"
        monkeypatch.setenv("GMLX_BASE_URL", url)
        exe = _gmlx_script(tmp_path / "bin", _answers(_listing("other")))
        _record(gmlx_command=str(exe))
        req = _probed(widgets)
        # The recorded default (qwen-image) is not there; z-image-turbo is.
        assert widgets.widgets() == {
            "url": url, "models": ["z-image-turbo", "qwen-image-2.1", "krea-2-turbo"],
            "source": "server"}
        assert req["server_url"][1]["default"] == url
        # The server's default image model comes first.
        _Models.default = "krea-2-turbo"
        assert _probed(widgets)["model_name"][0] == [
            "krea-2-turbo", "qwen-image-2.1", "z-image-turbo"]
    finally:
        srv.shutdown()
        srv.server_close()


def test_without_gmlx_or_models_the_list_is_only_the_default(widgets, tmp_path, monkeypatch):
    """No unpulled model is listed: the default name alone, which a run
    refuses with its `gmlx pull`."""
    _record(gmlx_command=str(tmp_path / "gone" / "gmlx"))
    _probed(widgets)
    assert widgets.widgets()["models"] == ["qwen-image"]
    assert widgets.widgets()["source"] == "default"
    exe = _gmlx_script(tmp_path / "bin", _answers(_listing()))
    _record(gmlx_command=str(exe))
    _probed(widgets)
    assert widgets.widgets()["models"] == ["qwen-image"]
    monkeypatch.setenv("GMLX_DEFAULT_IMAGE_MODEL", "krea-2-turbo")
    assert _probed(widgets)["model_name"][0] == ["krea-2-turbo"]
    assert "`gmlx pull qwen-image`" in widgets.not_on_server(
        "qwen-image", "http://127.0.0.1:8080", [])


def test_a_hung_gmlx_never_stalls_or_fails_the_widgets(widgets, tmp_path, monkeypatch):
    """The probe runs on a background thread, so the widgets answer at once
    with the default model's name alone while gmlx hangs."""
    exe = _gmlx_script(tmp_path / "bin", "exec /bin/sleep 30")
    _record(gmlx_command=str(exe))
    monkeypatch.delenv("GMLX_BASE_URL")
    t0 = time.monotonic()
    req = widgets.loader_inputs()["required"]
    assert time.monotonic() - t0 < 10
    assert req["model_name"][0] == ["qwen-image"]
    assert req["server_url"][1]["default"] == widgets.DEFAULT_URL
    # One probe at a time: the calls while it hangs start no other.
    probe = widgets.refresh()
    t0 = time.monotonic()
    for _ in range(20):
        widgets.loader_inputs()
        assert widgets.refresh() is probe
    assert time.monotonic() - t0 < 10
    assert probe.is_alive()


def test_a_probe_that_raises_still_gives_widgets(widgets, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("probe")
    monkeypatch.setattr(widgets, "resolve_base_url", boom)
    req = _probed(widgets)
    assert req["model_name"][0][0] == "qwen-image"
    assert req["server_url"][1]["default"] == os.environ["GMLX_BASE_URL"]


def test_the_address_default_is_the_resolved_one(widgets, tmp_path, monkeypatch):
    assert _probed(widgets)["server_url"][1]["default"] == os.environ["GMLX_BASE_URL"]
    monkeypatch.delenv("GMLX_BASE_URL")
    # No server runs: gmlx status names the host and port of gmlx's config.
    exe = _gmlx_script(tmp_path / "bin", _answers(
        status={"running": False, "host": "0.0.0.0", "port": 9132, "answering": False}))
    _record(gmlx_command=str(exe))
    assert _probed(widgets)["server_url"][1]["default"] == "http://127.0.0.1:9132"
    assert widgets.resolve_base_url("") == "http://127.0.0.1:9132"
    _gmlx_script(tmp_path / "bin", _answers(
        status={"running": True, "url": "http://127.0.0.1:9133"}))
    assert widgets.resolve_base_url("") == "http://127.0.0.1:9133"
    _gmlx_script(tmp_path / "bin", _answers(status={"servers": [
        {"running": True, "url": "http://127.0.0.1:9134"},
        {"running": False, "url": "http://127.0.0.1:9135"}]}))
    assert widgets.resolve_base_url("") == "http://127.0.0.1:9134"


def test_the_server_state_is_the_runfile_of_the_port(widgets, tmp_path):
    assert widgets.server_state("http://127.0.0.1:9132") == (None, None)
    exe = _gmlx_script(tmp_path / "bin", _answers(status={
        "running": True, "port": 9132, "log": "/x/9132.log"}))
    _record(gmlx_command=str(exe))
    assert widgets.server_state("http://127.0.0.1:9132") == (True, "/x/9132.log")
    # No status row for the port: unknown, so the wait goes on.
    assert widgets.server_state("http://127.0.0.1:9133") == (None, None)
    _gmlx_script(tmp_path / "bin", _answers(status={"servers": [
        {"running": True, "port": 9134}, {"running": False, "port": 9135, "log": "/x/l"}]}))
    assert widgets.server_state("http://127.0.0.1:9134") == (True, None)
    assert widgets.server_state("http://127.0.0.1:9135") == (False, "/x/l")


def _rows(*ids) -> list:
    kinds = {"z-image-turbo": ("z_image", "turbo"), "qwen-image": ("qwen_image21", "base"),
             "qwen-image-2.1": ("qwen_image21", "base")}
    return [{"model": i, "family": kinds[i][0], "variant": kinds[i][1]} for i in ids]


def test_an_unconfigured_model_is_refused_with_its_pull(widgets):
    msg = widgets.not_on_server("qwen-image-2.1", "http://127.0.0.1:8080", _rows("z-image-turbo"))
    assert "`gmlx pull qwen-image`" in msg and "z-image-turbo" in msg
    assert "http://127.0.0.1:8080" in msg
    assert widgets.pull_name("Qwen-Image-2.1") == "qwen-image"
    msg = widgets.not_on_server("flux", "http://h:1", [])
    assert "`gmlx pull z-image-turbo`" in msg and "(it has: none)" in msg
    assert widgets.validate("anything") is True
    assert widgets.validate(" ") == "model_name is empty"


@pytest.mark.parametrize("asked, has", [("qwen-image-2.1", "qwen-image"),
                                        ("qwen-image", "qwen-image-2.1")])
def test_a_model_the_server_has_under_another_id_is_named(widgets, asked, has):
    """A workflow names the alias, and `gmlx pull` registered the model under
    the recipe name, or the other way round: running the pull again would
    not add the name, so the refusal names the id to pick."""
    msg = widgets.not_on_server(asked, "http://127.0.0.1:8080", _rows("z-image-turbo", has))
    assert f"It has that model as {has!r}: pick {has!r}" in msg
    assert "or run `gmlx pull qwen-image` and pick the id it adds" in msg


# gmlx Sigmas on a schedule that follows the image size.

def _schedules():
    """The pack's schedules module, which imports numpy alone."""
    spec = importlib.util.spec_from_file_location("_pack_schedules", _PACK / "schedules.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _status(family: str, variant: str) -> dict:
    from gmlx.image.registry import get_family
    from gmlx.serve import image_steps as steps

    return {"sampling": steps.sampling(get_family(family), variant),
            "latent": dict(steps.COMFY[family]["latent"])}


SIZES = [(512, 512), (1024, 1024), (1024, 576), (576, 1024), (1360, 768), (2048, 2048),
         (256, 1024), (1056, 1056)]


def test_the_server_names_a_size_rule_for_each_family_whose_schedule_follows_the_size():
    """A served family's schedule follows the image size when it needs a token
    count, and only then does its status carry a rule, unless the variant is
    distilled to fixed sigmas."""
    from gmlx.image.registry import get_family
    from gmlx.serve import image_steps as steps

    for name in steps.COMFY:
        fam = get_family(name)
        for variant, row in fam.sampling.items():
            needs = False
            if fam.schedule is not None and not row.sigmas:
                fam.schedule(4, tokens=4096, variant=variant)
                try:
                    fam.schedule(4, tokens=None, variant=variant)
                except ValueError:
                    needs = True
            got = steps.sampling(fam, variant)["schedule"]
            assert (got is not None) == needs, (name, variant, got)
    assert set(steps.SIZE_SCHEDULES) == {"flux2_klein", "qwen_image21"}


@pytest.mark.parametrize("family, variant", [("flux2_klein", "distilled"),
                                             ("flux2_klein", "base"),
                                             ("qwen_image21", "base")])
def test_gmlx_sigmas_forms_the_schedule_the_family_samples_on(family, variant):
    """From the status alone the pack's schedule of an image is the family's
    own schedule of that image's tokens, bit for bit, and below a denoise of
    1 the tail of the longer schedule, as ComfyUI's are."""
    import numpy as np

    from gmlx.image.registry import get_family

    sch, fam, status = _schedules(), get_family(family), _status(family, variant)
    for w, h in SIZES:
        tokens = (h // 16) * (w // 16)
        for n in (2, 3, 4, 9, 28, 50):
            want = np.asarray(fam.schedule(n, tokens=tokens, variant=variant), np.float32)
            got = sch.size_sigmas(status, "m", n, 1.0, w, h)
            assert got.dtype == np.float32 and got.shape == want.shape, (w, h, n)
            assert np.array_equal(got.view(np.uint32), want.view(np.uint32)), (w, h, n)
        want = np.asarray(fam.schedule(10, tokens=tokens, variant=variant), np.float32)[-5:]
        got = sch.size_sigmas(status, "m", 4, 0.4, w, h)
        assert np.array_equal(got.view(np.uint32), want.view(np.uint32)), (w, h)


def test_gmlx_sigmas_without_a_size_is_refused_on_a_size_schedule():
    """FLUX.2 klein's refusal names Flux2Scheduler, the stock node of its
    schedule. A variant whose schedule does not follow the size takes none."""
    sch = _schedules()
    for family, variant, stock in (("flux2_klein", "distilled", True),
                                   ("qwen_image21", "base", False)):
        for w, h in ((0, 0), (1024, 0), (0, 1024)):
            with pytest.raises(sch.SizeNeeded) as e:
                sch.size_sigmas(_status(family, variant), "m", 4, 1.0, w, h)
            assert ("Flux2Scheduler" in str(e.value)) == stock, family
            assert "width and height" in str(e.value)
    for family, variant in (("z_image", "turbo"), ("qwen_image21", "turbo"),
                            ("krea2", "turbo"), ("flux3_klein", "distilled"),
                            ("flux3_klein", "base")):
        assert sch.size_sigmas(_status(family, variant), "m", 4, 1.0, 0, 0) is None
    with pytest.raises(ValueError, match="which these nodes do not know"):
        sch.size_sigmas({"sampling": {"schedule": {"rule": "flux9"}}}, "m", 4, 1.0, 64, 64)


def test_the_stub_server_carries_the_status_size_rules():
    """The stub the pack suites run against gives each model the rule the
    server's status gives it."""
    spec = importlib.util.spec_from_file_location(
        "_pack_stub", _ROOT / "tests" / "comfyui_pack" / "stub_server.py")
    stub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stub)
    for mid, (family, variant, sampling) in stub.MODELS.items():
        assert sampling["schedule"] == _status(family, variant)["sampling"]["schedule"], mid
