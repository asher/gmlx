"""``gmlx launch comfyui`` and the image toggles of ``gmlx launch open-webui``."""
from __future__ import annotations

import os

import pytest

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


def _comfy(tmp_path, *, pack=True, venv=".venv"):
    d = tmp_path / "ComfyUI"
    (d / "custom_nodes").mkdir(parents=True)
    (d / "main.py").write_text("")
    if venv:
        py = d / venv / "bin" / "python"
        py.parent.mkdir(parents=True)
        py.write_text("")
        py.chmod(0o755)
    if pack:
        pack_dir = tmp_path / "pack"
        pack_dir.mkdir()
        (d / "custom_nodes" / "ComfyUI-gmlx").symlink_to(pack_dir)
    return d


def _run(argv):
    calls = {}

    def fake_exec(binary, args, env):
        calls.update(binary=binary, argv=args, env=env)
        return 0

    rc = launch.cmd_launch(argv, exec_fn=fake_exec)
    return rc, calls


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


def test_a_missing_pack_prints_its_install_command_and_returns_2(tmp_path,
                                                                 monkeypatch,
                                                                 capsys):
    d = _comfy(tmp_path, pack=False)
    started = []
    monkeypatch.setattr(launch, "_ensure_server", lambda a: started.append(a))
    rc, calls = _run(["comfyui", "--comfyui-dir", str(d)])
    assert rc == 2 and calls == {} and started == []
    err = capsys.readouterr().err
    assert "git clone https://github.com/asher/ComfyUI-gmlx" in err
    assert str(d / "custom_nodes" / "ComfyUI-gmlx") in err


@pytest.mark.parametrize("venv, main, msg", [
    (None, True, "no Python environment of its own"),
    (".venv", False, "is not a ComfyUI folder"),
])
def test_comfyui_needs_its_folder_and_interpreter(tmp_path, capsys, venv, main,
                                                  msg):
    d = _comfy(tmp_path, venv=venv)
    if not main:
        os.remove(d / "main.py")
    rc, _calls = _run(["comfyui", "--comfyui-dir", str(d)])
    assert rc == launch.EXIT_UNAVAILABLE
    assert msg in capsys.readouterr().err


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
