"""`gmlx launch --container`: the launch order, the dry run, the attaching
shell and the per-client container settings. The server probe is faked, the
``container`` command is the fake from conftest, and the supervisor is a
recording stand-in, so no VM runs."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest

import gmlx.commands.launch as launch
import gmlx.commands.launch_container as lc
import gmlx.serve.lifecycle as lifecycle
from gmlx.container import runtime, session

MODELS = [{"id": "qwen3.6-27b", "default": True, "context_length": 65536}]


@pytest.fixture
def env(fake_container, tmp_path, monkeypatch):
    home = tmp_path / "home"
    proj = home / "src" / "proj"
    proj.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(proj)
    entry = tmp_path / "gmlx-entry"
    entry.write_bytes(b"\x7fELF-fake")
    monkeypatch.setattr(runtime, "entry_path", lambda: entry)

    def get_json(url, timeout=5.0, headers=None):
        return {"data": MODELS} if url.endswith("/models") else {}
    monkeypatch.setattr(launch, "_http_get_json", get_json)
    monkeypatch.setattr(launch, "_server_ready", lambda base, api_key=None: True)
    monkeypatch.setattr(launch, "_auth_required", lambda base: False)
    monkeypatch.setattr(launch, "_warn_if_stale_server", lambda host, port: None)
    monkeypatch.setattr(lifecycle, "auto_target", lambda h, p: ("127.0.0.1", 8080))
    monkeypatch.setattr(lifecycle, "read_run", lambda h, p: None)
    monkeypatch.setattr(session, "stdin_is_tty", lambda: False)
    runs = []

    def supervise(spec, **kw):
        runs.append({"spec": spec, **kw})
        return 0
    monkeypatch.setattr(session, "supervise", supervise)
    fake_container.runs = runs
    fake_container.home = home
    fake_container.proj = proj
    return fake_container


def _user_config(home, text):
    cfg = home / ".config" / "gmlx" / "gmlx.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(text)


def _run(argv, exec_fn=None):
    return launch.cmd_launch(argv, exec_fn=exec_fn or (lambda *a: pytest.fail("exec")))


# Deciding container mode

def test_container_flags_conflict_with_no_container(env):
    with pytest.raises(SystemExit):
        _run(["pi", "--mount", "~/src", "--no-container"])


def test_container_only_flags_imply_container(env):
    assert _run(["pi", "--rebuild"]) == 0
    assert env.runs


def test_no_mount_cwd_implies_container(env):
    assert _run(["pi", "--no-mount-cwd"]) == 0
    assert env.runs


def test_no_mount_cwd_conflicts_with_no_container(env, capsys):
    with pytest.raises(SystemExit):
        _run(["pi", "--no-mount-cwd", "--no-container"])
    assert "--no-mount-cwd applies only in container mode" in capsys.readouterr().err


def test_a_broken_launch_block_leaves_host_mode_running(env, capsys, monkeypatch):
    which = launch.shutil.which
    monkeypatch.setattr(launch.shutil, "which",
                        lambda name: "/usr/bin/pi" if name == "pi" else which(name))
    _user_config(env.home, "launch:\n  container:\n    bogus: 1\n")
    calls = []
    assert _run(["pi"], exec_fn=lambda *a: calls.append(a) or 0) == 0
    assert calls and not env.runs
    assert "ignoring the launch settings, so pi runs on the Mac" in capsys.readouterr().err
    assert _run(["pi", "--container"]) == 1                 # asked for, so it stops
    assert _run(["pi", "--rebuild"]) == 1


def test_config_enables_container_mode_only_from_the_user_file(env, capsys, monkeypatch):
    which = launch.shutil.which
    monkeypatch.setattr(launch.shutil, "which",
                        lambda name: "/usr/bin/pi" if name == "pi" else which(name))
    (env.proj / "gmlx.yaml").write_text("launch:\n  container:\n    enabled: true\n"
                                        "    mounts: [~/.ssh]\n")
    calls = []
    assert _run(["pi"], exec_fn=lambda *a: calls.append(a) or 0) == 0
    assert calls and not env.runs                      # host mode
    assert "ignoring the launch block" in capsys.readouterr().err
    _user_config(env.home, "launch:\n  container:\n    enabled: true\n")
    assert _run(["pi"]) == 0 and env.runs


# The dry run

def test_dry_run_path_through_the_order(env, capsys, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the dry run must not do this")
    monkeypatch.setattr(session, "cleanup_stale", boom)
    monkeypatch.setattr(runtime, "acquire_runtime", boom)
    monkeypatch.setattr(session, "lock_volumes", boom)
    monkeypatch.setattr(launch, "_keep_model", boom)
    session.write_record("claude-code", {"name": "leftover"})
    rc = _run(["claude-code", "--container", "--config-only", "--api-key", "sekrit",
               "--model", "qwen3.6-27b", "--", "--continue"])
    out = capsys.readouterr()
    assert rc == 0 and not env.runs
    assert "sekrit" not in out.out + out.err
    assert "container dry run" in out.out
    assert "container run --rm --init" in out.out
    assert "-e ANTHROPIC_AUTH_TOKEN" in out.out and "-e IS_SANDBOX=1" in out.out
    assert out.out.rstrip().endswith("-- claude --continue")
    assert not env.calls("build") and not env.calls("image", "pull")
    assert not env.calls("volume", "create") and not env.calls("system", "start")
    assert session.read_record("claude-code") is None      # step 5 ran
    assert session.try_session_lock("claude-code") is not None   # released at exit


def test_dry_run_with_the_service_stopped(env, capsys):
    env.update(running=False)
    assert _run(["pi", "--container", "--config-only"]) == 0
    out = capsys.readouterr().out
    assert "service stopped" in out and "unknown until the service runs" in out
    assert not env.calls("system", "start")


def test_dry_run_reports_rebuild(env, capsys):
    assert _run(["pi", "--container", "--config-only", "--rebuild"]) == 0
    assert "would be rebuilt" in capsys.readouterr().out
    assert not env.calls("build")


def test_dry_run_prints_the_replaced_command(env, capsys):
    _user_config(env.home, "launch:\n  container:\n    clients:\n      elia:\n"
                           "        command: [/usr/local/bin/start.sh, elia]\n")
    assert _run(["elia", "--container", "--config-only"]) == 0
    out = capsys.readouterr().out
    assert "replaces the client's own command, elia -m gmlx/qwen3.6-27b" in out
    assert out.rstrip().endswith("-- /usr/local/bin/start.sh elia")


def test_missing_entry_fails_step_3(env, capsys, monkeypatch):
    monkeypatch.setattr(runtime, "entry_path", lambda: env.home / "nope")
    assert _run(["pi", "--container"]) == 1
    assert "scripts/build_guest_entry.py" in capsys.readouterr().err


def test_config_path_is_refused(env, capsys):
    assert _run(["pi", "--container", "--config-path", "/tmp/x"]) == 1
    assert "--config-path does not apply" in capsys.readouterr().err


# A real run, up to the supervisor

def test_run_hands_the_supervisor_the_guest_command(env):
    assert _run(["pi", "--container", "--", "--continue"]) == 0
    spec = env.runs[0]["spec"]
    assert spec.command == ["pi", "--continue"]
    assert spec.image_ref.startswith("gmlx.invalid/launch-pi@sha256:")
    assert spec.api_port == 8080 and env.runs[0]["api_targets"] == [("127.0.0.1", 8080)]
    assert spec.workdir == os.path.realpath(env.proj)
    models = json.loads((spec.plan.home / ".pi" / "agent" / "models.json").read_text())
    assert models["providers"]["gmlx"]["baseUrl"] == "http://127.0.0.1:8080/v1"
    assert not (env.home / ".pi").exists()                # the real home is untouched
    assert env.runs[0]["record"]["shares"][0]["host"] == os.path.realpath(env.proj)
    assert env.calls("build")


def test_claude_code_guest_env(env):
    assert _run(["claude-code", "--container", "--network", "none"]) == 0
    spec = env.runs[0]["spec"]
    assert spec.env_values["IS_SANDBOX"] == "1"
    assert spec.env_values["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert "ANTHROPIC_AUTH_TOKEN" in spec.env_names
    assert spec.child_env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8080"


def test_clipboard_images_reaches_the_session_and_the_record(env):
    _user_config(env.home, "launch:\n  container:\n    clipboard: images\n"
                           "    clients:\n      pi:\n        clipboard: off\n")
    assert _run(["pi", "--container"]) == 0
    assert env.runs[0]["spec"].plan.clipboard == "off"     # the client value wins
    assert env.runs[0]["record"]["clipboard"] is False
    assert _run(["omp", "--container"]) == 0
    spec = env.runs[1]["spec"]
    assert spec.plan.clipboard == "images" and env.runs[1]["record"]["clipboard"] is True
    assert spec.env_values["WAYLAND_DISPLAY"] == "wayland-0"


def test_no_display_variable_without_clipboard_images(env):
    assert _run(["omp", "--container"]) == 0
    assert "WAYLAND_DISPLAY" not in env.runs[0]["spec"].env_values


def test_configured_env_passes_by_name(env):
    _user_config(env.home, "launch:\n  container:\n    env: [GH_TOKEN, MODE=fast]\n")
    assert _run(["pi", "--container"]) == 0
    spec = env.runs[0]["spec"]
    assert {"GH_TOKEN", "MODE"} <= set(spec.env_names)
    assert spec.child_env["MODE"] == "fast" and "GH_TOKEN" not in spec.child_env


def test_open_webui_listens_on_loopback_with_host_and_port(env):
    assert _run(["open-webui", "--container"]) == 0
    spec = env.runs[0]["spec"]
    assert spec.web_port == 3000
    assert spec.command[:6] == ["open-webui", "serve", "--host", "127.0.0.1", "--port", "3000"]
    assert spec.env_values["HOST"] == "127.0.0.1" and spec.env_values["PORT"] == "3000"
    assert "PORT" not in spec.env_names
    assert not spec.plan.cwd_shared                      # no share by default
    assert spec.child_env["DATA_DIR"].startswith(str(spec.plan.home))
    assert os.path.isdir(spec.child_env["DATA_DIR"])      # the official image needs it


def test_open_webui_command_image_gets_a_secret_key_file(env):
    env.update(registry={"ghcr.io/open-webui/open-webui:main": {
        "digest": "sha256:" + "5" * 64, "entrypoint": ["bash", "start.sh"],
        "workdir": "/app/backend"}})
    _user_config(env.home, "launch:\n  container:\n    clients:\n      open-webui:\n"
                           "        image: ghcr.io/open-webui/open-webui:main\n"
                           "        command: image\n")
    assert _run(["open-webui", "--container"]) == 0
    spec = env.runs[0]["spec"]
    assert spec.command == ["bash", "start.sh"] and spec.workdir == "/app/backend"
    assert spec.env_values["WEBUI_SECRET_KEY_FILE"] == str(spec.plan.home / ".webui_secret_key")
    assert spec.env_values["PORT"] == "3000"
    assert env.calls("run")[0][-2:] == ["--check", "bash"]


def test_dsh_web_profile_gets_no_open_and_a_port(env):
    assert _run(["dsh", "--container"]) == 0
    spec = env.runs[0]["spec"]
    assert spec.web_port == 3080
    assert spec.command[-3:] == ["--no-open", "--port", "3080"]
    assert spec.env_values["HOST"] == "127.0.0.1"


def test_dsh_stdio_profiles_are_refused(env, capsys):
    assert _run(["dsh", "--container", "--config-only", "--dsh-profile", "acp"]) == 1
    assert "--no-container" in capsys.readouterr().err


def test_shell_runs_the_shell_with_the_passthrough(env):
    assert _run(["pi", "--container", "--shell", "--", "-c", "npm test"]) == 0
    spec = env.runs[0]["spec"]
    assert spec.shell and spec.command == ["-c", "npm test"]


def test_guest_url_rules(monkeypatch):
    import gmlx.container.relay as relay
    monkeypatch.setattr(relay, "resolve_targets", lambda h, p: [("10.0.0.2", p)])
    assert lc.guest_url("http://0.0.0.0:8080/v1") == (
        "http://127.0.0.1:8080/v1", 8080, [("127.0.0.1", 8080)])
    assert lc.guest_url("http://[::1]:9000/api/v1")[0:2] == ("http://127.0.0.1:9000/api/v1", 9000)
    assert lc.guest_url("http://box.local:8000/v1")[2] == [("10.0.0.2", 8000)]
    assert lc.guest_url("https://api.example.com/v1") == ("https://api.example.com/v1", None, [])


def test_network_none_refuses_an_https_server(env, capsys):
    assert _run(["pi", "--container", "--network", "none", "--base-url",
                 "https://api.example.com/v1"]) == 1
    assert "network: none cannot reach" in capsys.readouterr().err


# The session lock and --shell attach

@pytest.fixture
def running_session(env):
    lock = session.try_session_lock("pi")
    proj = os.path.realpath(env.proj)
    session.write_record("pi", {"name": "gmlx-pi-abc123", "workdir": proj, "clipboard": False,
                                "shares": [{"host": proj, "guest": proj, "readonly": False}]})
    env.update(containers=[{"name": "gmlx-pi-abc123",
                            "labels": {"gmlx.launch": "1", "gmlx.launch.client": "pi"}}])
    yield env
    lock.release()


def test_second_session_is_refused_with_the_shell_hint(running_session, capsys):
    assert _run(["pi", "--container"]) == 1
    err = capsys.readouterr().err
    assert "gmlx-pi-abc123" in err and "gmlx launch pi --shell" in err


def test_shell_attaches_to_the_running_session(running_session, capsys):
    calls = []
    (running_session.proj / "sub").mkdir()
    os.chdir(running_session.proj / "sub")
    rc = _run(["pi", "--shell", "--", "-c", "ls"], exec_fn=lambda *a: calls.append(a) or 0)
    assert rc == 0
    argv = calls[0][1]
    proj = os.path.realpath(running_session.proj)
    assert argv[1:] == ["exec", "-i", "--cwd", f"{proj}/sub", "gmlx-pi-abc123",
                        "/opt/gmlx/gmlx-entry", "--shell", "--", "-c", "ls"]
    assert "attaching to gmlx-pi-abc123" in capsys.readouterr().out


def test_shell_attach_passes_clipboard_when_the_session_has_it(running_session):
    record = session.read_record("pi")
    session.write_record("pi", {**record, "clipboard": True})
    calls = []
    assert _run(["pi", "--shell"], exec_fn=lambda *a: calls.append(a) or 0) == 0
    argv = calls[0][1]
    assert argv[argv.index("/opt/gmlx/gmlx-entry") + 1:] == ["--clipboard", "--shell", "--"]


def test_shell_attach_from_an_unshared_folder(running_session, capsys):
    calls = []
    os.chdir(running_session.home)
    assert _run(["pi", "--shell"], exec_fn=lambda *a: calls.append(a) or 0) == 0
    assert "--cwd" not in calls[0][1]
    assert "not shared with this session" in capsys.readouterr().out


def test_shell_attach_refuses_new_session_flags(running_session, capsys):
    assert _run(["pi", "--shell", "--model", "x"]) == 1
    assert "--model applies only to a new session" in capsys.readouterr().err


def test_shell_attach_while_the_session_starts(running_session, capsys):
    session.remove_record("pi")
    assert _run(["pi", "--shell"]) == 1
    assert "still starting" in capsys.readouterr().err


# The private home during the handler

def test_guest_home_restores_after_an_exception(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", "/real")
    monkeypatch.setenv("DSH_HOME", "/real/dsh")
    monkeypatch.delenv("HERMES_CONFIG", raising=False)
    with pytest.raises(RuntimeError):
        with lc.guest_home(tmp_path):
            assert os.environ["HOME"] == str(tmp_path) and "DSH_HOME" not in os.environ
            raise RuntimeError
    assert os.environ["HOME"] == "/real" and os.environ["DSH_HOME"] == "/real/dsh"
    assert "HERMES_CONFIG" not in os.environ


def test_handlers_read_only_the_host_variables_guest_home_covers():
    """A handler that reads another host variable must be reviewed for
    container mode, which runs it with HOME pointed at the private home."""
    tree = ast.parse(Path(launch.__file__).read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner = ast.unparse(node.func.value)
            if owner in ("os.environ", "os") and node.func.attr in ("get", "getenv"):
                if node.args and isinstance(node.args[0], ast.Constant):
                    names.add(node.args[0].value)
        if isinstance(node, ast.Subscript) and ast.unparse(node.value) == "os.environ":
            if isinstance(node.slice, ast.Constant):
                names.add(node.slice.value)
    assert names == {"HERMES_CONFIG", "AUDIO_TTS_VOICE", "DSH_HOME"}


def test_dsh_reads_its_token_url_instead_of_the_terminal(env):
    assert _run(["dsh", "--container"]) == 0
    spec = env.runs[0]["spec"]
    assert spec.url_pattern and not spec.interactive and not spec.tty
