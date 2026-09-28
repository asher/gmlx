"""`gmlx launch --container`: the launch order, the dry run, the attaching
shell and the per-client container settings. The server probe is faked, the
``container`` command is the fake from conftest, and the supervisor is a
recording stand-in, so no VM runs."""

from __future__ import annotations

import ast
import json
import os
import signal
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
    _user_config(env.home, "launch:\n  container:\n    network: offline\n")
    calls = []
    assert _run(["pi"], exec_fn=lambda *a: calls.append(a) or 0) == 0
    assert calls and not env.runs
    assert "ignoring the launch settings, so pi runs on the Mac" in capsys.readouterr().err
    assert _run(["pi", "--container"]) == 1                 # asked for, so it stops
    assert _run(["pi", "--rebuild"]) == 1


@pytest.mark.parametrize("block", [
    "launch:\n  container:\n    enabled: true\n    bogus: 1\n",
    "launch:\n  container:\n    clients:\n      pi:\n        enabled: true\n"
    "        memory: lots\n",
    "launch:\n  container:\n    enabled: true\n    network: offline\n",
])
def test_a_broken_block_that_enables_container_mode_never_runs_on_the_mac(env, capsys,
                                                                           monkeypatch,
                                                                           block):
    which = launch.shutil.which
    monkeypatch.setattr(launch.shutil, "which",
                        lambda name: "/usr/bin/pi" if name == "pi" else which(name))
    _user_config(env.home, block)
    calls = []
    assert _run(["pi"], exec_fn=lambda *a: calls.append(a) or 0) == 1
    err = capsys.readouterr().err
    assert not calls and not env.runs
    assert "turns container mode on for pi" in err and "--no-container" in err
    assert ".config/gmlx/gmlx.yaml" in err
    assert _run(["pi", "--no-container"], exec_fn=lambda *a: calls.append(a) or 0) == 0
    assert calls                                            # asked for the Mac


@pytest.mark.parametrize("block", [
    "launch:\n  container:\n    enabled: \"yes\"\n",
    "launch:\n  container:\n    enabled: 1\n",
    "launch:\n  container:\n    enabled: false\n    bogus: 1\n",
    "launch:\n  container:\n    clients:\n      pi: true\n",
    "launch:\n  container: true\n",
    "launch:\n  container: [enabled]\n",
    "launch:\n  container:\n    clients: [pi]\n",
    "launch:\n  containers:\n    enabled: true\n",
    "launch:\n  container:\n    enable: true\n",
    "launch: true\n",
    "- launch\n",
    "launch:\n  container:\n    enabled: true\n    clients:\n      pi:\n"
    "        enabled: false\n        bogus: 1\n",
    "launch:\n  container:\n    clients:\n      pi:\n        enable: true\n",
    "launch:\n  container:\n    clients:\n      pie:\n        enabled: true\n",
    "lauch:\n  container:\n    enabled: true\n",
])
def test_every_unclear_enabled_shape_refuses(env, capsys, monkeypatch, block):
    which = launch.shutil.which
    monkeypatch.setattr(launch.shutil, "which",
                        lambda name: "/usr/bin/pi" if name == "pi" else which(name))
    _user_config(env.home, block)
    calls = []
    assert _run(["pi"], exec_fn=lambda *a: calls.append(a) or 0) == 1
    assert not calls and not env.runs
    assert "--no-container" in capsys.readouterr().err


def test_a_misspelled_launch_key_with_a_container_block_refuses(env, capsys, monkeypatch):
    """The server's strict top-level check does not run when the server is
    already up, so launch checks the key itself."""
    which = launch.shutil.which
    monkeypatch.setattr(launch.shutil, "which",
                        lambda name: "/usr/bin/pi" if name == "pi" else which(name))
    _user_config(env.home, "lauch:\n  container:\n    clients:\n      pi:\n"
                           "        enabled: true\n")
    calls = []
    assert _run(["pi"], exec_fn=lambda *a: calls.append(a) or 0) == 1
    err = capsys.readouterr().err
    assert not calls and not env.runs
    assert "unknown top-level key 'lauch'" in err and "Did you mean launch?" in err
    assert "may turn container mode on for pi" in err
    assert _run(["pi", "--no-container"], exec_fn=lambda *a: calls.append(a) or 0) == 0
    assert calls


@pytest.mark.parametrize("block", [
    "launch:\n  container:\n    enabled: false\n    memory: lots\n",
    "launch:\n  container:\n    memory: lots\n",
    "launch:\n  container:\n    clients:\n      pi:\n        enabled: false\n"
    "        memory: lots\n",
])
def test_a_broken_block_that_clearly_leaves_the_client_off_runs_on_the_mac(
        env, capsys, monkeypatch, block):
    which = launch.shutil.which
    monkeypatch.setattr(launch.shutil, "which",
                        lambda name: "/usr/bin/pi" if name == "pi" else which(name))
    _user_config(env.home, block)
    calls = []
    assert _run(["pi"], exec_fn=lambda *a: calls.append(a) or 0) == 0
    assert calls and "ignoring the launch settings" in capsys.readouterr().err


def test_a_broken_block_that_enables_another_client_runs_this_one_on_the_mac(env, capsys,
                                                                             monkeypatch):
    which = launch.shutil.which
    monkeypatch.setattr(launch.shutil, "which",
                        lambda name: "/usr/bin/pi" if name == "pi" else which(name))
    _user_config(env.home, "launch:\n  container:\n    clients:\n      omp:\n"
                           "        enabled: true\n        bogus: 1\n")
    calls = []
    assert _run(["pi"], exec_fn=lambda *a: calls.append(a) or 0) == 0
    assert calls and "ignoring the launch settings" in capsys.readouterr().err


def test_unreadable_yaml_never_runs_on_the_mac(env, capsys):
    _user_config(env.home, "launch: [unclosed\n")
    assert _run(["pi"]) == 1
    assert "may turn container mode on for pi" in capsys.readouterr().err


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


def test_dry_run_reports_the_base_of_a_build_image(env, capsys):
    ctx = env.home / "ctx"
    ctx.mkdir()
    (ctx / "Containerfile").write_text("FROM gmlx.invalid/launch-claude-code:base\n")
    _user_config(env.home, "launch:\n  container:\n    clients:\n      pi:\n"
                           f"        build: {ctx}\n")
    assert _run(["pi", "--container", "--config-only"]) == 0
    out = capsys.readouterr().out
    assert "[launch] base gmlx.invalid/launch-claude-code:base: would be built first" in out
    assert not env.calls("build")


def test_shell_config_only_on_a_running_session_names_the_flag(env, capsys):
    lock = session.try_session_lock("pi")
    try:
        assert _run(["pi", "--shell", "--config-only"]) == 1
    finally:
        lock.release()
    assert "--config-only applies only to a new session" in capsys.readouterr().err


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


@pytest.mark.parametrize("command, word", [
    ("", "pi"),                                                    # the handler's binary
    ("        command: [/usr/local/bin/start.sh, pi]\n", "/usr/local/bin/start.sh"),
])
def test_the_check_word_follows_the_command_form(env, command, word):
    env.update(registry={"docker.io/me/pi:1": {"digest": "sha256:" + "6" * 64}})
    _user_config(env.home, "launch:\n  container:\n    clients:\n      pi:\n"
                           "        image: docker.io/me/pi:1\n" + command)
    assert _run(["pi", "--container"]) == 0
    assert env.calls("run")[0][-2:] == ["--check", word]


def test_command_image_with_nothing_to_run_warns_under_shell(env, capsys):
    env.update(registry={"docker.io/me/bare:1": {"digest": "sha256:" + "7" * 64}})
    _user_config(env.home, "launch:\n  container:\n    clients:\n      pi:\n"
                           "        image: docker.io/me/bare:1\n        command: image\n")
    assert _run(["pi", "--container"]) == 1
    assert "sets no ENTRYPOINT or CMD" in capsys.readouterr().err
    assert _run(["pi", "--container", "--shell"]) == 0
    assert "warning:" in capsys.readouterr().out and env.runs[-1]["spec"].shell


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


def test_guest_url_refuses_a_host_that_does_not_resolve(monkeypatch):
    import socket as _socket

    import gmlx.container.relay as relay

    def fail(h, p):
        raise _socket.gaierror(8, "nodename nor servname provided")
    monkeypatch.setattr(relay, "resolve_targets", fail)
    with pytest.raises(lc.SettingsError, match="cannot resolve the server host box.invalid"):
        lc.guest_url("http://box.invalid:8000/v1")
    monkeypatch.setattr(relay, "resolve_targets", lambda h, p: [])
    with pytest.raises(lc.SettingsError, match="no IPv4 or IPv6 address"):
        lc.guest_url("http://box.invalid:8000/v1")


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
    monkeypatch.setenv("HERMES_HOME", "/real/hermes")
    with pytest.raises(RuntimeError):
        with lc.guest_home(tmp_path):
            assert os.environ["HOME"] == str(tmp_path)
            assert "DSH_HOME" not in os.environ and "HERMES_HOME" not in os.environ
            raise RuntimeError
    assert os.environ["HOME"] == "/real" and os.environ["DSH_HOME"] == "/real/dsh"
    assert os.environ["HERMES_HOME"] == "/real/hermes"


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
    assert names == {"HERMES_HOME", "AUDIO_TTS_VOICE", "DSH_HOME"}


def test_dsh_reads_its_token_url_instead_of_the_terminal(env):
    assert _run(["dsh", "--container"]) == 0
    spec = env.runs[0]["spec"]
    assert spec.url_pattern and not spec.interactive and not spec.tty


# The guest owns its private home, so no handler follows a link it plants

def _written_files(home: Path) -> list[Path]:
    return sorted(p for p in home.rglob("*") if p.is_file() and not p.is_symlink()
                  and not p.name.startswith(".gmlx-entry"))


@pytest.mark.parametrize("client", sorted(set(launch._HARNESSES) - {"claude-code", "open-webui"}))
def test_a_planted_link_never_reaches_a_mac_file(env, client, tmp_path, capsys):
    from gmlx.container import settings
    assert _run([client, "--container"]) == 0
    home = settings.private_home_path(client)
    written = _written_files(home)
    assert written, f"{client} wrote nothing into its private home"
    secret = tmp_path / "mac-secret"
    outside = tmp_path / "mac-folder"
    outside.mkdir()
    for target in written:
        rel = target.relative_to(home)
        # A link at the file itself, pointing at a Mac file.
        secret.write_text("mac: secret\n")
        target.unlink()
        target.symlink_to(secret)
        assert _run([client, "--container"]) == 1, rel
        assert secret.read_text() == "mac: secret\n" and target.is_symlink()
        assert "symbolic link" in capsys.readouterr().err
        target.unlink()
        # A link at the folder above it, pointing at a Mac folder.
        parent = target.parent
        if parent == home:
            continue
        kept = parent.with_name(parent.name + ".kept")
        parent.rename(kept)
        parent.symlink_to(outside, target_is_directory=True)
        assert _run([client, "--container"]) == 1, rel
        assert list(outside.iterdir()) == [], rel
        assert "symbolic link" in capsys.readouterr().err
        parent.unlink()
        kept.rename(parent)


def test_open_webui_data_dir_is_never_created_through_a_link(env, tmp_path, capsys):
    from gmlx.container import settings
    assert _run(["open-webui", "--container"]) == 0
    data = settings.private_home_path("open-webui") / ".open-webui"
    assert data.is_dir() and not data.is_symlink()
    data.rmdir()
    outside = tmp_path / "mac-folder"
    outside.mkdir()
    data.symlink_to(outside / "sub", target_is_directory=True)
    assert _run(["open-webui", "--container"]) == 1
    assert not (outside / "sub").exists()
    assert "symbolic link" in capsys.readouterr().err


def test_a_named_pipe_in_the_private_home_never_blocks_a_read(env, capsys):
    from gmlx.container import settings
    assert _run(["hermes", "--container"]) == 0
    cfg = settings.private_home_path("hermes") / ".hermes" / "config.yaml"
    cfg.unlink()
    os.mkfifo(cfg)
    assert _run(["hermes", "--container"]) == 1
    assert "not a regular file" in capsys.readouterr().err


def test_handlers_touch_files_only_through_the_confined_helpers():
    """The handlers run on the Mac against a private home the guest can
    change, so every file access must go through gmlx.container.confine."""
    tree = ast.parse(Path(launch.__file__).read_text())
    # Host mode only: the Mac's own dsh install and the Mac's hermes backup.
    allowed = {"_dsh_version", "_hermes_backup"}
    bad = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef) or fn.name in allowed:
            continue
        if not (fn.name.startswith("_launch_") or fn.name.startswith("_dsh")
                or fn.name in {"_load_json", "_load_yaml"}):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                if (name in {"open", "os.makedirs", "shutil.copy", "shutil.copyfile"}
                        or name.endswith((".read_text", ".write_text", ".read_bytes",
                                          ".write_bytes", ".mkdir", ".open",
                                          ".touch", ".rename", ".replace"))):
                    if not name.startswith("confine."):
                        bad.append(f"{fn.name}: {name}")
    assert bad == []


# Refusals come before any build or download

@pytest.mark.parametrize("argv, message", [
    (["pi", "--container", "--base-url", "http://nosuch.invalid:8080/v1"],
     "cannot resolve the server host"),
    (["pi", "--container", "--network", "none", "--base-url", "https://example.com/v1"],
     "network: none cannot reach"),
    (["pi", "--container", "--base-url", "http://127.0.0.1:99999/v1"],
     "not a number from 0 to 65535"),
])
def test_refusals_come_before_the_image(env, capsys, argv, message):
    assert _run(argv) == 1
    assert message in capsys.readouterr().err
    assert not env.calls("build") and not env.calls("image", "pull")


def test_a_bad_seed_and_a_deleted_cwd_stop_before_the_image(env, capsys,
                                                                        monkeypatch):
    _user_config(env.home, "launch:\n  container:\n    clients:\n      pi:\n"
                           "        seed: [/etc]\n")
    assert _run(["pi", "--container"]) == 1
    assert "not inside your home" in capsys.readouterr().err
    assert not env.calls("build")
    _user_config(env.home, "launch:\n  container:\n    enabled: false\n")
    gone = env.home / "gone"
    gone.mkdir()
    monkeypatch.chdir(gone)
    gone.rmdir()
    assert _run(["pi", "--container"]) == 1
    assert "the current folder no longer exists" in capsys.readouterr().err
    assert not env.calls("build")


def test_no_server_and_no_config_stops_before_the_image(env, monkeypatch, capsys):
    monkeypatch.setattr(launch, "_server_ready", lambda base, api_key=None: False)
    monkeypatch.setattr(launch, "_discover_config", lambda: (None, None))
    guided = []
    monkeypatch.setattr(launch, "_guide_to_init", lambda *a: guided.append(a))
    assert _run(["pi", "--container"]) == 2
    assert guided and not env.calls("build") and not env.calls("image", "pull")


def test_ports_follow_the_server_the_check_found(env, monkeypatch):
    """Step 6 guesses the server port before the server check, which may
    find the server elsewhere, so the web port is worked out again."""
    monkeypatch.setattr(lifecycle, "auto_target", lambda h, p: ("127.0.0.1", 3000))
    real = launch._ensure_server

    def moved(a):
        a.host, a.port = "127.0.0.1", 8080
        a.base_url = "http://127.0.0.1:8080/v1"
        return real(a)
    monkeypatch.setattr(launch, "_ensure_server", moved)
    assert _run(["open-webui", "--container"]) == 0
    assert env.runs[0]["spec"].web_port == 3000


def test_https_server_allows_forwarding_port_443(env):
    _user_config(env.home, "launch:\n  container:\n    forward: [443]\n")
    assert _run(["pi", "--container", "--base-url", "https://example.com/v1"]) == 0
    assert env.runs[0]["spec"].plan.forward == [443]


def test_shell_on_a_web_app_opens_no_browser(env):
    assert _run(["open-webui", "--shell"]) == 0
    assert env.runs[0]["opener"] is None


def test_command_image_checks_the_first_word_with_the_passthrough(env):
    env.update(registry={"docker.io/me/bare:1": {"digest": "sha256:" + "7" * 64}})
    _user_config(env.home, "launch:\n  container:\n    clients:\n      pi:\n"
                           "        image: docker.io/me/bare:1\n        command: image\n")
    assert _run(["pi", "--container", "--", "node", "x.js"]) == 0
    assert env.calls("run")[0][-2:] == ["--check", "node"]
    assert env.runs[0]["spec"].command == ["node", "x.js"]


def test_attach_defaults_match_the_parser():
    """--shell refuses every flag whose value differs from these defaults
    when it attaches, so they must follow the parser."""
    from tests.commands.test_launch import _parse_launch_args
    a = _parse_launch_args(["pi"])
    for dest, default in lc._ATTACH_DEFAULTS.items():
        assert getattr(a, dest) == default, dest
    rest = set(vars(a)) - set(lc._ATTACH_DEFAULTS) - {
        "harness", "container", "shell", "passthrough"}
    assert rest == set(), rest


# First-run steps, the builder notice and the dry run's image

def test_steps_are_numbered_only_when_they_run(env, capsys):
    assert _run(["pi", "--container"]) == 0            # a first build on a running service
    out = capsys.readouterr().out
    assert "[launch] step 1: building" in out
    assert env.runs[-1]["summary"][0] == "[launch] step 2: start pi"
    assert _run(["pi", "--container"]) == 0            # the image exists now
    assert "step " not in capsys.readouterr().out
    assert not any(line.startswith("[launch] step") for line in env.runs[-1]["summary"]), env.runs[-1]["summary"]


def test_a_restarted_service_with_a_ready_image_gets_two_steps(env, capsys, monkeypatch):
    assert _run(["pi", "--container"]) == 0
    capsys.readouterr()
    env.update(running=False)
    monkeypatch.setattr(session, "stdin_is_tty", lambda: True)
    assert _run(["pi", "--container"]) == 0
    out = capsys.readouterr().out
    assert "[launch] step 1: start the container service" in out and "building" not in out
    assert env.runs[-1]["summary"][0] == "[launch] step 2: start pi"


def test_step_7_reports_an_idle_builder(env, capsys, monkeypatch):
    from gmlx.container import images
    # Other test runs can start the fake `container build` at the same time.
    monkeypatch.setattr(images, "_other_builds", lambda: False)
    env.update(builder=True)
    assert _run(["pi", "--container"]) == 0
    assert "container builder stop" in capsys.readouterr().out


def test_dry_run_names_the_build_tag_and_the_image_command(env, capsys):
    ctx = env.home / "ctx"
    ctx.mkdir()
    (ctx / "Containerfile").write_text("FROM debian:bookworm-slim\n")
    _user_config(env.home, "launch:\n  container:\n    clients:\n      pi:\n"
                           f"        build: {ctx}\n")
    assert _run(["pi", "--container", "--config-only"]) == 0
    assert "[launch] image gmlx.invalid/launch-pi-build:" in capsys.readouterr().out
    env.update(registry={"docker.io/me/tool:1": {
        "digest": "sha256:" + "8" * 64, "entrypoint": ["/bin/tool"], "cmd": ["serve"],
        "workdir": "/srv"}})
    _user_config(env.home, "launch:\n  container:\n    clients:\n      pi:\n"
                           "        image: docker.io/me/tool:1\n        command: image\n")
    assert _run(["pi", "--container", "--config-only"]) == 0
    out = capsys.readouterr().out
    assert "would be pulled" in out
    assert "<ENTRYPOINT and CMD of the image>" in out
    assert _run(["pi", "--container", "--config-only", "--", "check"]) == 0
    assert capsys.readouterr().out.rstrip().endswith("-- '<ENTRYPOINT of the image>' check")
    assert _run(["pi", "--container"]) == 0            # pulls it
    capsys.readouterr()
    assert _run(["pi", "--container", "--config-only", "--", "check"]) == 0
    out = capsys.readouterr().out
    assert "--workdir /srv" in out and out.rstrip().endswith("-- /bin/tool check")


def test_a_refusal_comes_before_the_service_start(env, capsys, monkeypatch):
    env.update(running=False)
    monkeypatch.setattr(session, "stdin_is_tty", lambda: True)
    monkeypatch.chdir(env.home)                        # the home folder is refused
    assert _run(["pi", "--container"]) == 1
    assert "will not share the current folder" in capsys.readouterr().err
    assert not env.calls("system", "start")


def test_an_explicit_port_with_no_server_and_no_config_stops_early(env, monkeypatch):
    env.update(running=False)
    monkeypatch.setattr(session, "stdin_is_tty", lambda: True)
    monkeypatch.setattr(launch, "_server_ready", lambda base, api_key=None: False)
    monkeypatch.setattr(launch, "_discover_config", lambda: (None, None))
    monkeypatch.setattr(launch, "_guide_to_init", lambda *a: None)
    assert _run(["pi", "--container", "--port", "9999"]) == 2
    assert not env.calls("system", "start") and not env.calls("build")


def test_attach_from_a_deleted_folder_is_a_clean_error(running_session, capsys, monkeypatch):
    gone = running_session.home / "gone"
    gone.mkdir()
    monkeypatch.chdir(gone)
    gone.rmdir()
    assert _run(["pi", "--shell"]) == 1
    assert "the current folder no longer exists" in capsys.readouterr().err


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP])
def test_a_signal_during_the_build_runs_its_clean_up(env, capsys, monkeypatch, signum):
    from gmlx.container import images
    cleaned = []
    got = []

    def build(*a, **k):
        try:
            os.kill(os.getpid(), signum)
            for _ in range(1000):              # the handler runs between bytecodes
                pass
        finally:
            cleaned.append(True)
    monkeypatch.setattr(images, "ensure_image", build)

    def record(sig, _frame):
        got.append(sig)
    # A recording handler stands in for the default one, which would end
    # pytest when launch fails to take the signal over.
    before = signal.signal(signum, record)
    try:
        rc = _run(["pi", "--container"])
        after = signal.getsignal(signum)
    finally:
        signal.signal(signum, before)
    assert rc == 128 + signum and got == []
    assert cleaned and "stopped by signal" in capsys.readouterr().err
    assert after is record                     # restored


def test_the_signal_exception_is_not_an_exception():
    """An ``except Exception`` in step 8 must never swallow a SIGTERM."""
    assert not issubclass(lc._Signalled, Exception)


def test_step_7_leaves_an_owed_builder_for_a_build_about_to_run(env, monkeypatch):
    from gmlx.container import images
    seen = []
    monkeypatch.setattr(images, "builder_notice",
                        lambda say=None, settle=True: seen.append(settle))
    assert _run(["pi", "--container"]) == 0             # builds the image
    assert _run(["pi", "--container"]) == 0             # the image is ready
    assert seen == [False, True]


def test_step_7_stops_an_owed_builder_before_a_pull(env, monkeypatch):
    """Only a build uses the builder, and nothing stops it after a pull."""
    from gmlx.container import images
    seen = []
    monkeypatch.setattr(images, "builder_notice",
                        lambda say=None, settle=True: seen.append(settle))
    env.update(registry={"docker.io/me/pi:1": {"digest": "sha256:" + "6" * 64}})
    _user_config(env.home, "launch:\n  container:\n    clients:\n      pi:\n"
                           "        image: docker.io/me/pi:1\n")
    assert _run(["pi", "--container"]) == 0
    assert env.calls("image", "pull") and seen == [True]
