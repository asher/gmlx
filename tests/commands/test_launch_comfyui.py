"""``gmlx launch comfyui`` and the image toggles of ``gmlx launch open-webui``."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import gmlx.commands.comfyui as comfyui
import gmlx.commands.launch as launch

_MODELS = [
    {"id": "qwen3-4b", "default": False},
    {"id": "chat", "default": True},
    {"id": "z-image-turbo", "image": True, "default": True, "family": "z_image"},
    {"id": "qwen-image", "image": True, "default": False, "edit": True},
]


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path_factory):
    monkeypatch.setenv("HOME", str(tmp_path_factory.mktemp("home")))
    monkeypatch.delenv("GMLX_API_KEY", raising=False)
    monkeypatch.setattr(launch, "_server_ready", lambda base, api_key=None: True)
    monkeypatch.setattr(launch, "_auth_required", lambda base: False)
    monkeypatch.setattr(launch, "_warn_if_stale_server", lambda host, port: None)
    monkeypatch.setattr(launch, "probe_models",
                        lambda base, api_key=None, client=None, **k: _MODELS)


def _comfy(root, *, venv=".venv", version="0.38.0", git=False):
    d = root / "ComfyUI"
    (d / "custom_nodes").mkdir(parents=True)
    (d / "main.py").write_text("")
    if version:
        (d / "comfyui_version.py").write_text(f'__version__ = "{version}"\n')
    if git:
        (d / ".git").mkdir()
    if venv:
        py = d / venv / "bin" / "python"
        py.parent.mkdir(parents=True)
        py.write_text("")
        py.chmod(0o755)
    return d


def _run(argv):
    calls = {}

    def fake_exec(binary, args, env):
        calls.update(binary=binary, argv=args, env=env)
        return 0

    rc = launch.cmd_launch(argv, exec_fn=fake_exec)
    return rc, calls


def _no_server(monkeypatch):
    started = []
    monkeypatch.setattr(launch, "_ensure_server", lambda a: started.append(a))
    return started


def test_comfyui_runs_on_its_own_interpreter_with_the_pack_settings(tmp_path):
    d = _comfy(tmp_path)
    rc, calls = _run(["comfyui", "--comfyui-dir", str(d)])
    assert rc == 0
    assert calls["binary"] == str(d / ".venv" / "bin" / "python")
    assert calls["argv"] == [calls["binary"], str(d / "main.py"), "--listen",
                             "127.0.0.1", "--port", "8188"]
    env = calls["env"]
    assert env["GMLX_BASE_URL"] == "http://127.0.0.1:8080"
    assert env["GMLX_DEFAULT_IMAGE_MODEL"] == "z-image-turbo"
    assert "GMLX_API_KEY" not in env


def test_comfyui_passes_the_api_key_and_the_image_model_asked_for(tmp_path):
    d = _comfy(tmp_path, venv="venv")
    rc, calls = _run(["comfyui", "--comfyui-dir", str(d), "--api-key", "k",
                      "--model", "qwen-image"])
    assert rc == 0
    assert calls["env"]["GMLX_API_KEY"] == "k"
    assert calls["env"]["GMLX_DEFAULT_IMAGE_MODEL"] == "qwen-image"


def test_comfyui_refuses_a_chat_model(tmp_path, capsys):
    d = _comfy(tmp_path)
    rc, calls = _run(["comfyui", "--comfyui-dir", str(d), "--model", "chat"])
    assert rc == launch.EXIT_FAILURE and calls == {}
    assert "not an image model" in capsys.readouterr().err


def test_the_launch_puts_the_pack_in_once_and_says_so(tmp_path, capsys):
    d = _comfy(tmp_path)
    rc, _calls = _run(["comfyui", "--comfyui-dir", str(d)])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.count("[launch] installed the gmlx nodes in") == 1
    link = d / "custom_nodes" / "comfyui-gmlx"
    assert os.path.realpath(link) == os.path.realpath(comfyui.copy_dir())
    assert (link / "__init__.py").is_file()
    rc, _calls = _run(["comfyui", "--comfyui-dir", str(d)])
    assert rc == 0 and "installed the gmlx nodes" not in capsys.readouterr().out


def test_the_launch_remembers_the_folder(tmp_path):
    d = _comfy(tmp_path)
    assert _run(["comfyui", "--comfyui-dir", str(d)])[0] == 0
    rc, calls = _run(["comfyui"])
    assert rc == 0 and calls["argv"][1] == str(d / "main.py")


def test_a_real_pack_folder_moves_aside(tmp_path, capsys):
    d = _comfy(tmp_path)
    (d / "custom_nodes" / "ComfyUI-gmlx").mkdir()
    rc, _calls = _run(["comfyui", "--comfyui-dir", str(d)])
    assert rc == 0
    assert "moved to" in capsys.readouterr().out
    assert [p.name.startswith("comfyui-gmlx-")
            for p in (d / "custom_nodes" / ".disabled").iterdir()] == [True]


def test_no_comfyui_prints_how_to_get_it_and_exits_1(monkeypatch, capsys):
    started = _no_server(monkeypatch)
    rc, calls = _run(["comfyui"])
    assert rc == launch.EXIT_FAILURE == 1 and calls == {} and started == []
    err = capsys.readouterr().err
    assert "no ComfyUI found in ~/ComfyUI," in err
    assert "https://comfy.org/download" in err
    assert "git clone https://github.com/comfyanonymous/ComfyUI ~/ComfyUI" in err
    assert "comfyui.html" in err


def test_a_comfyui_too_old_names_the_floor_and_its_upgrade(tmp_path, monkeypatch, capsys):
    d = _comfy(tmp_path, version="0.30.1", git=True)
    started = _no_server(monkeypatch)
    rc, calls = _run(["comfyui", "--comfyui-dir", str(d)])
    assert rc == launch.EXIT_FAILURE and calls == {} and started == []
    err = capsys.readouterr().err
    assert "ComfyUI 0.30.1" in err and comfyui.pack_meta()["COMFYUI_FLOOR"] in err
    assert f"git -C {d} pull" in err
    assert not os.path.lexists(d / "custom_nodes" / "comfyui-gmlx")


def test_a_folder_that_is_not_comfyui_exits_1(tmp_path, monkeypatch, capsys):
    started = _no_server(monkeypatch)
    rc, _calls = _run(["comfyui", "--comfyui-dir", str(tmp_path / "nothing")])
    assert rc == launch.EXIT_FAILURE and started == []
    assert "is not a ComfyUI folder" in capsys.readouterr().err


def test_a_comfyui_without_its_environment_says_how_to_make_one(tmp_path, monkeypatch,
                                                                capsys):
    d = _comfy(tmp_path, venv=None)
    started = _no_server(monkeypatch)
    rc, _calls = _run(["comfyui", "--comfyui-dir", str(d)])
    assert rc == launch.EXIT_UNAVAILABLE and started == []
    err = capsys.readouterr().err
    assert "no Python environment of its own" in err
    assert "uv venv --python 3.12 .venv" in err


def test_comfy_desktop_gets_the_pack_and_the_step_to_start_it(monkeypatch, capsys):
    home = Path(os.environ["HOME"])
    code = home / "ComfyUI-Installs" / "Stable" / "ComfyUI"
    (code / "custom_nodes").mkdir(parents=True)
    (code / "main.py").write_text("")
    (code / "comfyui_version.py").write_text('__version__ = "0.38.0"\n')
    data = home / "Library" / "Application Support" / "comfyui-desktop-2"
    data.mkdir(parents=True)
    (data / "installations.json").write_text(json.dumps([
        {"sourceId": "standalone", "installPath": str(code.parent), "name": "Stable"}]))
    started = _no_server(monkeypatch)
    rc, calls = _run(["comfyui"])
    assert rc == launch.EXIT_UNAVAILABLE and calls == {} and started == []
    err = capsys.readouterr().err
    assert err.startswith("[launch] installed the gmlx nodes in "
                          "~/ComfyUI-Installs/Stable/ComfyUI/custom_nodes/comfyui-gmlx")
    assert "[launch] [launch]" not in err
    assert "Comfy Desktop install 'Stable' runs its own ComfyUI" in err
    assert (code / "custom_nodes" / "comfyui-gmlx").is_symlink()


def test_several_comfyuis_ask_for_one(monkeypatch, capsys):
    home = Path(os.environ["HOME"])
    _comfy(home / "comfy")
    _comfy(home / "work")
    ini = home / "Library" / "Application Support" / "comfy-cli" / "config.ini"
    ini.parent.mkdir(parents=True)
    ini.write_text(f"[DEFAULT]\ndefault_workspace = {home / 'work'}\n")
    started = _no_server(monkeypatch)
    rc, _calls = _run(["comfyui"])
    assert rc == launch.EXIT_USAGE and started == []
    err = capsys.readouterr().err
    assert "Run gmlx launch comfyui with --comfyui-dir" in err
    assert "~/comfy/ComfyUI" in err and "~/work/ComfyUI" in err


@pytest.mark.parametrize("flag, msg", [
    (["--container"], "runs on the Mac"),
    (["--mount", "/x"], "runs on the Mac"),
    (["--config-path", "/x"], "--config-path does not apply to comfyui"),
])
def test_comfyui_runs_on_the_mac_only(tmp_path, flag, msg, capsys):
    d = _comfy(tmp_path)
    with pytest.raises(SystemExit) as e:
        launch.cmd_launch(["comfyui", "--comfyui-dir", str(d)] + flag)
    assert e.value.code == 2
    assert msg in capsys.readouterr().err


def test_comfyui_dir_goes_with_comfyui_only(capsys):
    with pytest.raises(SystemExit):
        launch.cmd_launch(["opencode", "--comfyui-dir", "/x"])
    assert "--comfyui-dir applies only to comfyui" in capsys.readouterr().err


def test_comfyui_takes_its_own_flags_after_the_separator(tmp_path):
    d = _comfy(tmp_path)
    _rc, calls = _run(["comfyui", "--comfyui-dir", str(d), "--",
                       "--use-pytorch-cross-attention"])
    assert calls["argv"][-3:] == ["--port", "8188",
                                  "--use-pytorch-cross-attention"]


def test_comfyui_moves_off_the_server_port(tmp_path):
    d = _comfy(tmp_path)
    _rc, calls = _run(["comfyui", "--comfyui-dir", str(d), "--port", "8188"])
    assert calls["argv"][-1] == "8189"


def test_an_image_model_is_never_a_chat_default():
    models = [{"id": "z", "image": True, "default": True}, {"id": "c"}]
    assert launch._pick_default(models, None) is None
    assert launch.chat_models(models) == [{"id": "c"}]


def test_open_webui_turns_image_generation_on_for_the_image_default():
    env = launch.build_open_webui_env("http://h:1/v1", data_dir="/d",
                                      image_model="z-image-turbo")
    assert env["ENABLE_IMAGE_GENERATION"] == "true"
    assert env["IMAGE_GENERATION_ENGINE"] == "openai"
    assert env["IMAGES_OPENAI_API_BASE_URL"] == "http://h:1/v1"
    assert env["IMAGE_GENERATION_MODEL"] == "z-image-turbo"
    off = launch.build_open_webui_env("http://h:1/v1", data_dir="/d")
    assert not any(k.startswith(("IMAGE", "ENABLE_IMAGE")) for k in off)
    assert "ENABLE_IMAGE_EDIT" not in env


def test_open_webui_turns_image_edits_on_for_a_model_that_edits():
    env = launch.build_open_webui_env("http://h:1/v1", data_dir="/d",
                                      image_model="qwen-image",
                                      image_edit=True)
    assert env["ENABLE_IMAGE_EDIT"] == "true"
    assert env["IMAGE_EDIT_ENGINE"] == "openai"
    assert env["IMAGES_EDIT_OPENAI_API_BASE_URL"] == "http://h:1/v1"
    assert env["IMAGES_EDIT_OPENAI_API_KEY"] == env["IMAGES_OPENAI_API_KEY"]
    assert env["IMAGE_EDIT_MODEL"] == "qwen-image"


@pytest.mark.parametrize("model,edits", [("z-image-turbo", False),
                                         ("qwen-image", True)])
def test_open_webui_launch_edits_only_with_a_model_that_edits(
        monkeypatch, tmp_path, model, edits):
    monkeypatch.setattr(launch.shutil, "which", lambda name: f"/usr/bin/{name}")
    models = [dict(m, default=m["id"] == model) if m.get("image") else m
              for m in _MODELS]
    monkeypatch.setattr(launch, "probe_models",
                        lambda base, api_key=None, client=None, **k: models)
    rc, calls = _run(["open-webui", "--config-path", str(tmp_path)])
    assert rc == 0
    assert calls["env"].get("IMAGE_EDIT_MODEL") == (model if edits else None)


def test_open_webui_launch_reads_the_image_default_from_the_server(
        monkeypatch, tmp_path):
    monkeypatch.setattr(launch.shutil, "which", lambda name: f"/usr/bin/{name}")
    rc, calls = _run(["open-webui", "--config-path", str(tmp_path)])
    assert rc == 0
    assert calls["env"]["IMAGE_GENERATION_MODEL"] == "z-image-turbo"
    assert calls["env"]["DEFAULT_MODELS"] == "chat"


@pytest.mark.parametrize("served, found, size", [
    (("/srv.yaml", {"image_models": {"z-image-turbo": {
        "defaults": {"size": "512x512"}}}}), None, "512x512"),
    (("/srv.yaml", {"image_models": {"z-image-turbo": {}}}), "768x768",
     "1024x1024"),
    (None, "768x768", "768x768"),
    (None, None, "1024x1024"),
    (("/srv.yaml", {"image_models": {"z-image-turbo": {
        "defaults": {"size": "big"}}}}), None, "1024x1024"),
])
def test_open_webui_takes_the_image_size_of_the_default_entry(
        monkeypatch, tmp_path, served, found, size):
    """The size comes from the config of the server the launch uses, else
    from the first default config, else it is 1024x1024."""
    from types import SimpleNamespace

    monkeypatch.setattr(launch.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(launch, "_launch_served", lambda a: served)
    cfg = None if found is None else SimpleNamespace(image_models={
        "z-image-turbo": SimpleNamespace(defaults={"size": found})})
    monkeypatch.setattr(launch, "_discover_config", lambda: (cfg, None))
    rc, calls = _run(["open-webui", "--config-path", str(tmp_path)])
    assert rc == 0
    assert calls["env"]["IMAGE_SIZE"] == size
