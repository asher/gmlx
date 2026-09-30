"""`gmlx launch --container`: the launch order, the dry run, the attaching
shell and the per-client container settings. The server probe is faked, the
``container`` command is the fake from conftest, and the supervisor is a
recording stand-in, so no VM runs."""

from __future__ import annotations

import ast
import json
import os
import shlex
import shutil
import signal
import socket
import tempfile
import urllib.parse
from pathlib import Path

import pytest

import gmlx.commands.launch as launch
import gmlx.commands.launch_container as lc
import gmlx.serve.lifecycle as lifecycle
from gmlx.config import LAUNCH_CLIENTS
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
    # A short TMPDIR, so the fake server's session sockets fit the 104-byte
    # limit of a socket path.
    tmp = tempfile.mkdtemp(prefix="gl-", dir="/tmp")
    monkeypatch.setenv("TMPDIR", tmp)
    server = _SessionServer(tmp)
    monkeypatch.setattr(launch, "_http_post_json", server.post)
    monkeypatch.setattr(launch, "_http_delete", server.delete)
    runs = []

    def supervise(spec, **kw):
        runs.append({"spec": spec, **kw})
        return 0
    monkeypatch.setattr(session, "supervise", supervise)
    fake_container.runs = runs
    fake_container.server = server
    fake_container.home = home
    fake_container.proj = proj
    yield fake_container
    server.close()
    shutil.rmtree(tmp, ignore_errors=True)


class _SessionServer:
    """The session endpoint of a gmlx server, as the contract describes it.
    ``status`` set to a number answers every session request with it."""

    def __init__(self, tmp):
        self.tmp = tmp
        self.status: int | None = None
        self.body: bytes | None = None
        self.tools = {"home": ["web", "files"], "quiet": []}
        self.posts: list[tuple[str, dict, str | None]] = []
        self.deletes: list[tuple[str, str | None]] = []
        self.count = 0
        self.sockets: list[socket.socket] = []

    def _error(self, url, code, body=None):
        import io
        import urllib.error
        body = self.body if self.body is not None else body
        fp = io.BytesIO(body) if body is not None else None
        return urllib.error.HTTPError(url, code, "refused", None, fp)  # type: ignore[arg-type]

    def _socket(self, url) -> str:
        """A private socket in a session folder, as the server makes one."""
        port = urllib.parse.urlsplit(url).port
        folder = Path(self.tmp) / f"gmlx-sessions-127-0-0-1-{port}"
        folder.mkdir(mode=0o700, exist_ok=True)
        path = str(folder / f"{self.count:012x}.sock")
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind(path)
        os.chmod(path, 0o600)
        self.sockets.append(s)
        return path

    def close(self):
        for s in self.sockets:
            s.close()

    def post(self, url, body, *, api_key=None, timeout=3.0):
        if not url.endswith("/launch/sessions"):
            return {}
        self.posts.append((url, dict(body), api_key))
        if self.status is not None:
            raise self._error(url, self.status)
        if not {"client", "assistants"} <= set(body) <= {"client", "assistants", "web_ports"}:
            raise self._error(url, 400, b'{"error": {"type": "invalid_request_error", '
                                        b'"message": "the body must hold client"}}')
        self.count += 1
        listed = body["assistants"]
        return {"id": f"s{self.count}", "socket": self._socket(url),
                "assistants": {a: {"tools": self.tools[a]} for a in listed if a in self.tools},
                "unknown": [a for a in listed if a not in self.tools]}

    def delete(self, url, *, api_key=None, timeout=3.0):
        self.deletes.append((url, api_key))
        return 204


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


def test_reseed_implies_container_and_reaches_the_seed_step(env, monkeypatch):
    from gmlx.container import settings
    seen = []
    monkeypatch.setattr(settings, "seed_home",
                        lambda home, seeds, reseed=False, writable=(): seen.append(reseed) or [])
    assert _run(["pi", "--reseed"]) == 0
    assert env.runs and seen == [True]
    with pytest.raises(SystemExit):
        _run(["pi", "--reseed", "--no-container"])


def test_the_image_step_gets_the_read_write_shares(env, monkeypatch):
    from gmlx.container import images
    seen = []
    real = images.resolve_image

    def spy(*a, **k):
        seen.append(k.get("writable"))
        return real(*a, **k)
    monkeypatch.setattr(images, "resolve_image", spy)
    (env.home / "notes").mkdir()
    assert _run(["pi", "--mount", str(env.home / "notes") + ":ro"]) == 0
    proj = os.path.realpath(env.proj)
    assert seen and proj in seen[0]
    assert os.path.realpath(env.home / "notes") not in seen[0]      # read-only


def test_no_mount_cwd_implies_container(env):
    assert _run(["pi", "--no-mount-cwd"]) == 0
    assert env.runs


def test_no_mount_cwd_conflicts_with_no_container(env, capsys):
    with pytest.raises(SystemExit):
        _run(["pi", "--no-mount-cwd", "--no-container"])
    assert "--no-mount-cwd applies only in container mode" in capsys.readouterr().err


@pytest.mark.parametrize("flags", [["--container", "--no-container"],
                                   ["--no-container", "--container"]])
def test_container_and_no_container_together_are_refused(env, capsys, flags):
    with pytest.raises(SystemExit):
        _run(["pi", *flags])
    assert "--container and --no-container cannot go together" in capsys.readouterr().err
    assert not env.runs


def test_no_mount_cwd_says_where_the_client_starts(env):
    assert _run(["pi", "--no-mount-cwd"]) == 0
    assert ("[launch] the current folder is not shared, so pi starts in its private home"
            in env.runs[0]["summary"])


def test_reseed_without_a_seed_says_so(env, capsys):
    assert _run(["pi", "--reseed"]) == 0
    assert ("[launch] --reseed has nothing to copy, because no seed is configured for pi."
            in capsys.readouterr().out.splitlines())


def test_a_shell_prints_no_client_summary(env, capsys):
    """Under --shell the client does not start, so the lines that describe
    it stay out."""
    assert _run(["pi", "--shell"]) == 0
    assert "[launch] opening a shell instead of pi" in env.runs[0]["summary"]
    out = capsys.readouterr().out
    assert "[launch] pi ->" not in out and "merged" not in out


def test_a_data_folder_that_is_a_file_is_a_clean_error(env, capsys, monkeypatch, tmp_path):
    data = tmp_path / "data"
    (data / "gmlx").mkdir(parents=True)
    (data / "gmlx" / "launch").write_text("not a folder")
    monkeypatch.setenv("XDG_DATA_HOME", str(data))
    assert _run(["pi", "--container"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("[launch] ") and "Move or remove it." in err
    assert "Traceback" not in err


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
    "container:\n  enabled: true\n",
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
    assert "unknown top-level key 'lauch'" in err
    # The file is named once, and the question ends its own sentence.
    assert "Did you mean launch? That file may turn container mode on for pi" in err
    assert err.count("gmlx.yaml") == 1
    assert _run(["pi", "--no-container"], exec_fn=lambda *a: calls.append(a) or 0) == 0
    assert calls


def test_a_top_level_container_block_refuses(env, capsys, monkeypatch):
    """Written one level too high, the block never runs the client on the
    Mac without a word."""
    which = launch.shutil.which
    monkeypatch.setattr(launch.shutil, "which",
                        lambda name: "/usr/bin/pi" if name == "pi" else which(name))
    _user_config(env.home, "container:\n  enabled: true\n")
    calls = []
    assert _run(["pi"], exec_fn=lambda *a: calls.append(a) or 0) == 1
    err = capsys.readouterr().err
    assert not calls and not env.runs
    assert "Did you mean launch: container:? That file may turn container mode on" in err


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
    lines = capsys.readouterr().err.splitlines()
    # The sentence starts its own line after the parser's location lines.
    assert lines[-2].lstrip().startswith("in ") and lines[-2].rstrip()[-1].isdigit()
    assert lines[-1].startswith("That file may turn container mode on for pi")


def test_config_enables_container_mode_only_from_the_user_file(env, capsys, monkeypatch):
    which = launch.shutil.which
    monkeypatch.setattr(launch.shutil, "which",
                        lambda name: "/usr/bin/pi" if name == "pi" else which(name))
    (env.proj / "gmlx.yaml").write_text("launch:\n  container:\n    enabled: true\n"
                                        "    mounts: [~/.ssh]\n")
    calls = []
    assert _run(["pi"], exec_fn=lambda *a: calls.append(a) or 0) == 0
    assert calls and not env.runs                      # host mode
    assert "gmlx no longer reads ./gmlx.yaml" in capsys.readouterr().err
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


def test_dry_run_opens_with_its_header(env, capsys):
    assert _run(["pi", "--container", "--config-only"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == ("[launch] container dry run: no image is built or pulled, and no "
                        "container is started.")
    assert lines[1].startswith("[launch] container 1.4")


def test_an_attaching_dry_run_prints_no_prerequisites(env, capsys):
    lock = session.try_session_lock("pi")
    try:
        assert _run(["pi", "--shell", "--config-only"]) == 1
    finally:
        lock.release()
    out = capsys.readouterr()
    assert out.out == "" and "--config-only applies only to a new session" in out.err


@pytest.mark.parametrize("args, why", [
    (["--no-start"], "and --no-start keeps launch from starting one"),
    ([], "and no config was found to start one from")])
def test_a_dry_run_without_a_server_shows_the_plan(env, capsys, monkeypatch, args, why):
    monkeypatch.setattr(launch, "_server_ready", lambda base, api_key=None: False)
    if args:
        _user_config(env.home, "server:\n  port: 8080\n")
    assert _run(["pi", "--container", "--config-only", *args]) == 0
    out = capsys.readouterr()
    assert "gmlx init" not in out.err
    assert "[launch] sharing ~/src/proj (read-write, working folder)" in out.out
    assert "[launch] image gmlx.invalid/launch-pi:" in out.out
    assert out.out.splitlines()[-1] == (
        f"[launch] no server answers at http://127.0.0.1:8080/v1, {why}, so the dry run "
        "shows no client configuration and no command.")
    assert not env.server.posts and not env.calls("build")


def test_a_dry_run_with_an_unreachable_base_url_shows_the_plan(env, capsys, monkeypatch):
    monkeypatch.setattr(launch, "_server_ready", lambda base, api_key=None: False)

    def get_json(url, timeout=5.0, headers=None):
        raise launch.urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))
    monkeypatch.setattr(launch, "_http_get_json", get_json)
    assert _run(["pi", "--container", "--config-only",
                 "--base-url", "http://127.0.0.1:48699"]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == (
        "[launch] cannot reach the server at http://127.0.0.1:48699/v1 (Connection refused), "
        "so the dry run shows no client configuration and no command. Check the URL, or "
        "start that server.")


def test_an_unreachable_base_url_is_refused_before_the_image_steps(env, capsys, monkeypatch):
    monkeypatch.setattr(launch, "_server_ready", lambda base, api_key=None: False)

    def get_json(url, timeout=5.0, headers=None):
        raise launch.urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))
    monkeypatch.setattr(launch, "_http_get_json", get_json)
    assert _run(["pi", "--container", "--base-url", "https://example.invalid/v1"]) == 1
    assert capsys.readouterr().err == (
        "[launch] cannot reach the server at https://example.invalid/v1 (Connection "
        "refused). Check the URL, or start that server.\n")
    assert not env.calls("image") and not env.calls("build") and not env.runs


def test_a_server_without_session_sockets_is_refused_before_the_image_steps(env, capsys):
    env.server.status = 404
    env.update(running=False)
    assert _run(["pi", "--container"]) == 1
    assert "does not offer session sockets" in capsys.readouterr().err
    assert not env.calls("system", "start") and not env.calls("build")
    assert not env.calls("image")


@pytest.mark.parametrize("models, argv, message", [
    ([], ["pi"], "has no models yet. Download one with gmlx pull"),
    (MODELS, ["pi", "--model", "nosuch"],
     "--model nosuch is not a model the server offers. It offers qwen3.6-27b."),
    ([{"id": "m-a"}, {"id": "m-b"}], ["goose"], "goose needs a default model")])
def test_a_model_the_launch_cannot_use_is_refused_before_the_image_steps(
        env, capsys, monkeypatch, models, argv, message):
    def get_json(url, timeout=5.0, headers=None):
        return {"data": models} if url.endswith("/models") else {}
    monkeypatch.setattr(launch, "_http_get_json", get_json)
    env.update(running=False)
    assert _run([argv[0], "--container", *argv[1:]]) == 1
    assert message in capsys.readouterr().err
    assert not env.calls("system", "start") and not env.calls("build")
    assert not env.calls("image") and not env.runs


def test_a_multi_line_error_keeps_its_lines(env, capsys, monkeypatch):
    which = launch.shutil.which
    monkeypatch.setattr(launch.shutil, "which",
                        lambda name: "/usr/bin/pi" if name == "pi" else which(name))
    _user_config(env.home, "launch:\n  container: [unclosed\n")
    assert _run(["pi", "--no-container"], exec_fn=lambda *a: 0) == 0
    err = capsys.readouterr().err
    assert "\\x0a" not in err
    assert "ignoring the launch settings" in err and len(err.splitlines()) > 1


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
    # The window the server reports, so Claude Code on Linux does not warn
    # about a model outside its catalog.
    assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" in spec.env_names
    assert spec.child_env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "65536"


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


def test_open_webui_listens_on_loopback_with_host_and_port(env, capsys):
    assert _run(["open-webui", "--container"]) == 0
    out = capsys.readouterr().out
    # The session prints the one address and opens it.
    assert "http://localhost" not in out and "open the URL" not in out
    assert "[launch] Open WebUI keeps its chat history and database in " in out
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


def test_launch_warns_about_an_empty_pythonpath_entry(env, monkeypatch, capsys):
    monkeypatch.setenv("PYTHONPATH", ":/abs/lib")
    assert _run(["pi", "--container"]) == 0
    assert "PYTHONPATH has an empty or relative entry" in capsys.readouterr().out


def test_dsh_web_profile_gets_no_open_and_a_port(env):
    assert _run(["dsh", "--container"]) == 0
    spec = env.runs[0]["spec"]
    assert spec.web_port == 3080
    assert spec.command[-3:] == ["--no-open", "--port", "3080"]
    assert spec.env_values["HOST"] == "127.0.0.1"


def _dsh_manifest(profile, bundles):
    from gmlx.container import settings
    d = settings.private_home("dsh") / ".dsh" / "profiles" / profile
    d.mkdir(parents=True, exist_ok=True)
    (d / "package.json").write_text(json.dumps({"dsh": {"profile": {"bundles": bundles}}}))


def test_a_guest_manifest_never_makes_a_dsh_profile_a_web_session(env):
    """The profile manifests lie in the private home, which the guest
    writes, so the profile name alone decides whether the Mac binds a web
    port and opens a browser."""
    _dsh_manifest("mycli", [launch._DSH_WEB_BUNDLE])
    assert _run(["dsh", "--container", "--dsh-profile", "mycli"]) == 0
    assert env.runs[-1]["spec"].web_port is None
    assert env.runs[-1].get("opener") is None
    _dsh_manifest("gmlx", ["@deepseek-ai/dsh-cli"])
    assert _run(["dsh", "--container"]) == 0
    assert env.runs[-1]["spec"].web_port == 3080


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


# The session socket of the server

def _files_holding(root, text):
    return [f for f in Path(root).rglob("*") if f.is_file() and text in f.read_text("latin-1")]


@pytest.mark.parametrize("client", LAUNCH_CLIENTS)
def test_client_configs_get_the_placeholder_and_never_the_key(env, capsys, client):
    assert _run([client, "--container", "--api-key", "sekrit"]) == 0
    spec = env.runs[0]["spec"]
    out = capsys.readouterr()
    assert not _files_holding(spec.plan.home, "sekrit")
    for where in (json.dumps(spec.child_env), json.dumps(spec.env_values),
                  shlex.join(session.compose_run_argv(spec)), shlex.join(spec.command),
                  out.out, out.err):
        assert "sekrit" not in where
    if client != "omp":                     # omp has no key setting at all
        assert (_files_holding(spec.plan.home, lc.SESSION_KEY)
                or lc.SESSION_KEY in spec.child_env.values())
    # The probe runs with the server's key, and so will the session request.
    assert env.server.posts == [("http://127.0.0.1:8080/v1/launch/sessions",
                                 {"probe": True}, "sekrit")]
    server = env.runs[0]["server_session"]
    assert (server.base_url, server.api_key, server.client) == (
        "http://127.0.0.1:8080/v1", "sekrit", client)


@pytest.mark.parametrize("client", LAUNCH_CLIENTS)
def test_a_dry_run_prints_no_key(env, capsys, client):
    assert _run([client, "--container", "--config-only", "--api-key", "sekrit"]) == 0
    out = capsys.readouterr()
    assert "sekrit" not in out.out + out.err


def test_the_supervisor_gets_the_configured_assistants(env):
    _user_config(env.home, "launch:\n  container:\n    clients:\n      aichat:\n"
                           "        assistants: [home, nope]\n")
    assert _run(["aichat", "--container"]) == 0
    server = env.runs[0]["server_session"]
    assert server.assistants == ["home", "nope"] and server.id is None
    assert server.open().endswith("/gmlx-sessions-127-0-0-1-8080/000000000001.sock")
    assert env.server.posts[-1][1] == {"client": "aichat", "assistants": ["home", "nope"]}
    assert server.lines() == [
        "[launch] aichat can use assistant home, whose tools run on the Mac: web, files",
        "[launch] warning: the server has no assistant nope, which "
        "launch.container.clients.aichat.assistants lists"]


def test_other_clients_get_no_assistants(env):
    _user_config(env.home, "launch:\n  container:\n    clients:\n      aichat:\n"
                           "        assistants: [home]\n")
    assert _run(["pi", "--container"]) == 0
    assert env.runs[0]["server_session"].assistants == []


@pytest.mark.parametrize("status", [404, 405])
def test_a_server_without_session_sockets_is_refused(env, capsys, status):
    env.server.status = status
    assert _run(["pi", "--container"]) == 1
    err = capsys.readouterr().err
    assert "does not offer session sockets" in err and "gmlx restart" in err
    assert not env.runs


@pytest.mark.parametrize("body, shown", [
    (b'{"error": {"type": "server_error", "message": "cannot open a launch session '
     b'socket: 32 sessions are in use"}}',
     "(503): cannot open a launch session socket: 32 sessions are in use"),
    (b"<html>", "(503): refused"), (None, "(503): refused")])
def test_a_server_that_cannot_open_a_socket_shows_its_message(env, capsys, body, shown):
    env.server.status, env.server.body = 503, body
    assert _run(["pi", "--container"]) == 1
    err = capsys.readouterr().err
    assert "could not open a session socket " + shown in err
    assert "gmlx restart" not in err and not env.runs


def test_a_session_request_the_server_cannot_serve_shows_its_message(env):
    server = _session(env)
    env.server.status = 503
    env.server.body = b'{"error": {"message": "no folder gives a short enough path"}}'
    with pytest.raises(launch.LaunchError) as e:
        server.open()
    assert "(503): no folder gives a short enough path" in str(e.value)
    assert "gmlx restart" not in str(e.value)
    env.server.status = 404
    with pytest.raises(launch.LaunchError, match="gmlx restart"):
        server.open()


@pytest.mark.parametrize("status", [401, 403])
def test_a_refused_key_names_the_key(env, capsys, status):
    env.server.status = status
    assert _run(["pi", "--container", "--api-key", "wrong"]) == 1
    err = capsys.readouterr().err
    assert "refused the API key" in err and "wrong" not in err
    assert not env.runs


def test_a_server_that_cannot_be_reached_is_a_clean_error(env, capsys, monkeypatch):
    def down(url, body, **k):
        raise OSError(61, "Connection refused")
    monkeypatch.setattr(launch, "_http_post_json", down)
    assert _run(["pi", "--container"]) == 1
    assert "cannot reach the server at http://127.0.0.1:8080/v1" in capsys.readouterr().err


def test_a_remote_server_keeps_the_key_and_gets_no_session(env):
    # 192.0.2.0/24 is reserved for documentation, so no Mac holds it.
    assert _run(["claude-code", "--container", "--base-url", "http://192.0.2.5:8080/v1",
                 "--api-key", "sekrit"]) == 0
    run = env.runs[0]
    assert run["server_session"] is None and not env.server.posts
    assert run["api_targets"] == [("192.0.2.5", 8080)]
    assert run["spec"].child_env["ANTHROPIC_AUTH_TOKEN"] == "sekrit"
    assert ("[launch] http://192.0.2.5:8080/v1 is not a server on this Mac, so launch "
            "cannot limit it to a session socket, and claude-code gets the key you passed "
            "and every route the server offers.") in run["summary"]


def test_an_https_server_says_launch_cannot_limit_it(env):
    assert _run(["pi", "--container", "--base-url", "https://example.com/v1"]) == 0
    assert any("https://example.com/v1 is an https server" in line
               and "pi reaches every route" in line for line in env.runs[0]["summary"])


@pytest.mark.parametrize("url", [
    "http://localhost:8080/v1", "http://localhost.:8080/v1", "http://127.0.0.2:8080/v1",
    "http://127.1:8080/v1", "http://0x7f000001:8080/v1", "http://[::1]:8080/v1",
    "http://[::ffff:127.0.0.1]:8080/v1", "http://0.0.0.0:8080/v1"])
def test_every_spelling_of_a_local_server_gets_a_session(env, url):
    assert _run(["claude-code", "--container", "--base-url", url, "--api-key", "sekrit"]) == 0
    run = env.runs[0]
    assert run["server_session"] is not None
    assert "sekrit" not in json.dumps(run["spec"].child_env)
    assert not any("cannot limit" in line for line in run["summary"])


def test_a_server_on_the_macs_lan_address_gets_a_session(env, monkeypatch):
    """A managed server bound to the Mac's LAN address is a local server."""
    real = lc._own_address
    monkeypatch.setattr(lc, "_own_address", lambda a: a == "192.168.1.20" or real(a))
    monkeypatch.setattr(lifecycle, "auto_target", lambda h, p: ("192.168.1.20", 8080))
    assert _run(["claude-code", "--container", "--api-key", "sekrit"]) == 0
    run = env.runs[0]
    assert run["server_session"] is not None
    assert "sekrit" not in json.dumps(run["spec"].child_env)
    assert env.server.posts[0][0] == "http://192.168.1.20:8080/v1/launch/sessions"


def test_own_address_takes_loopback_and_bindable_addresses_only(monkeypatch):
    for addr in ("127.0.0.1", "127.9.9.9", "::1", "::ffff:127.0.0.1"):
        assert lc._own_address(addr), addr
    for addr in ("192.0.2.5", "2001:db8::5", "0.0.0.0", "::", "224.0.0.1", "not-an-ip"):
        assert not lc._own_address(addr), addr
    bound = []

    class FakeSocket:
        def __init__(self, family, kind):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def bind(self, addr):
            bound.append(addr)
            if addr[0] != "192.168.1.20":
                raise OSError(49, "Can't assign requested address")
    monkeypatch.setattr(lc.socket, "socket", FakeSocket)
    assert lc._own_address("192.168.1.20") and not lc._own_address("192.168.1.21")
    assert bound == [("192.168.1.20", 0), ("192.168.1.21", 0)]


@pytest.mark.parametrize("status, line", [
    (None, "the server at http://127.0.0.1:8080/v1 offers session sockets"),
    (404, "warning: the server at http://127.0.0.1:8080/v1 does not offer session sockets")])
def test_dry_run_reports_session_sockets_and_opens_none(env, capsys, status, line):
    env.server.status = status
    assert _run(["claude-code", "--container", "--config-only", "--api-key", "sekrit"]) == 0
    out = capsys.readouterr()
    assert line in out.out and "sekrit" not in out.out + out.err
    assert [body for _, body, _ in env.server.posts] == [{"probe": True}]
    assert not env.server.deletes and not env.runs


def _session(env, assistants=()):
    return lc.ServerSession("http://127.0.0.1:8080/v1", "sekrit", "aichat", list(assistants))


def test_a_renewed_session_ends_the_old_one(env):
    server = _session(env, ["home"])
    server._delete_later = server._delete        # type: ignore[method-assign]
    assert server.open().endswith("/000000000001.sock")
    assert server.renew().endswith("/000000000002.sock")
    assert env.server.posts[-1][1] == {"client": "aichat", "assistants": ["home"]}
    server.close()
    assert sorted(env.server.deletes) == [
        ("http://127.0.0.1:8080/v1/launch/sessions/s1", "sekrit"),
        ("http://127.0.0.1:8080/v1/launch/sessions/s2", "sekrit")]
    assert server.renew() is None               # a closed session gets no new socket
    assert ("http://127.0.0.1:8080/v1/launch/sessions/s3", "sekrit") in env.server.deletes


def _impostor(env, monkeypatch, path):
    """A server that answers the session request with ``path``."""
    real = env.server.post

    def post(url, body, **kw):
        reply = real(url, body, **kw)
        return {**reply, "socket": str(path)} if "client" in body else reply
    monkeypatch.setattr(launch, "_http_post_json", post)


def test_launch_relays_only_to_a_session_socket_of_the_server(env, monkeypatch, tmp_path):
    """Whatever answers at the server's address names the socket the relay
    hands the client, so a path that is not a private session socket of a
    server on that port is refused, at open and at renew."""
    agent = tmp_path / "agent"
    agent.mkdir()
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(str(Path(env.server.tmp) / "agent.sock"))
    try:
        other_port = Path(env.server.tmp) / "gmlx-sessions-127-0-0-1-9999"
        other_port.mkdir(mode=0o700)
        s2 = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s2.bind(str(other_port / "0123456789ab.sock"))
        os.chmod(other_port / "0123456789ab.sock", 0o600)
        for path, why in ((Path(env.server.tmp) / "agent.sock",
                           "its name is not that of a session socket"),
                          (other_port / "0123456789ab.sock",
                           "it is not in a session folder of a server on port 8080"),
                          ("/etc/passwd", "its name is not that of a session socket")):
            _impostor(env, monkeypatch, path)
            server = _session(env)
            with pytest.raises(launch.LaunchError) as e:
                server.open()
            assert f"named {path} as the session socket" in str(e.value)
            assert why in str(e.value)
            assert server.renew() is None
        s2.close()
    finally:
        s.close()


@pytest.mark.parametrize("answer", [
    {"ok": True},                                        # a 2xx from something else
    (400, b'{"detail": "bad"}'),                         # a 400 that is not gmlx's
    (400, None)])
def test_only_a_gmlx_refusal_of_the_probe_counts_as_sessions_offered(env, capsys,
                                                                     monkeypatch, answer):
    def post(url, body, **kw):
        if isinstance(answer, dict):
            return answer
        raise env.server._error(url, answer[0], answer[1])
    monkeypatch.setattr(launch, "_http_post_json", post)
    assert _run(["pi", "--container"]) == 1
    assert "does not offer session sockets" in capsys.readouterr().err
    assert not env.runs


def test_a_renewal_the_server_refuses_gives_no_path(env):
    server = _session(env)
    logged = []
    server.log = logged.append
    server.open()
    env.server.status = 404
    assert server.renew() is None
    # The reason reaches the log at once, and the supervisor prints it later.
    assert server.refused and "gmlx restart" in server.refused
    assert len(logged) == 1 and "gave no new session socket" in logged[0]
    env.server.status = None
    assert server.renew() is not None and server.refused is None
    server.close()
    assert ("http://127.0.0.1:8080/v1/launch/sessions/s1", "sekrit") in env.server.deletes


def test_close_before_open_sends_nothing(env):
    _session(env).close()
    assert not env.server.deletes


@pytest.mark.parametrize("reply", [
    {}, {"id": "s1"}, {"id": "", "socket": "/tmp/a.sock"}, {"id": "s1", "socket": "a.sock"},
    {"id": "s1", "socket": "/tmp/a.sock", "assistants": []},
    {"id": "s1", "socket": "/tmp/a.sock", "unknown": "home"}, ["not", "a", "mapping"],
    {"id": "s1", "socket": "/tmp/a.sock", "assistants": {"home": {"tools": "web"}}},
    {"id": "s1", "socket": "/tmp/a.sock", "assistants": {"home": ["web"]}}])
def test_a_malformed_session_reply_is_a_clean_error(env, monkeypatch, reply):
    monkeypatch.setattr(launch, "_http_post_json", lambda *a, **k: reply)
    with pytest.raises(launch.LaunchError, match="unexpected form"):
        _session(env).open()


def test_a_session_line_for_an_alias_with_no_tools(env):
    server = _session(env, ["quiet"])
    server.open()
    assert server.lines() == ["[launch] aichat can use assistant quiet, which has no tools"]


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
    assert capsys.readouterr().err == (
        "[launch] a pi session is already running in gmlx-pi-abc123, and one session of a "
        "client runs at a time. End that session to launch another, or open a shell in it "
        "with: gmlx launch pi --shell\n")


@pytest.mark.parametrize("record", [{"name": "gmlx-pi-abc123"},
                                    {"name": "gmlx-pi-abc123", "workdir": "/w",
                                     "shares": [{"guest": "/w"}]}])
def test_shell_with_a_damaged_record_is_a_clean_error(running_session, capsys, record):
    session.record_path("pi").write_text(json.dumps(record))
    assert _run(["pi", "--shell"], exec_fn=lambda *a: pytest.fail("exec")) == 1
    err = capsys.readouterr().err
    assert "is damaged, so --shell cannot attach" in err
    assert err.endswith("End that session and launch again. Stop it with: container stop "
                        "gmlx-pi-abc123\n")
    assert _run(["pi", "--container"]) == 1              # the refusal still names the way
    assert "gmlx launch pi --shell" in capsys.readouterr().err


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
    assert _run(["pi", "--shell", "--mount", "/tmp"]) == 1
    assert "--mount applies only to a new session" in capsys.readouterr().err


def test_shell_attach_ignores_the_server_flags(running_session, capsys):
    """The flags that chose the session's server and model are what the
    user typed to start it, so the shell takes them and says it ignores
    them."""
    calls = []
    assert _run(["pi", "--shell", "--port", "48611", "--no-start"],
                exec_fn=lambda *a: calls.append(a) or 0) == 0
    assert calls
    assert ("[launch] --port and --no-start apply only to a new session, so the shell "
            "ignores them.") in capsys.readouterr().out


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
    assert env.runs[0]["server_session"].web_ports == [3000]


@pytest.mark.parametrize("client", ["open-webui", "dsh"])
def test_a_browser_app_session_names_its_pages(env, client):
    """The server refuses the app's pages on its TCP port while the session
    is open, so the page cannot go around the session."""
    assert _run([client, "--container"]) == 0
    run = env.runs[0]
    port = run["spec"].web_port
    assert run["server_session"].web_ports == [port]
    run["server_session"].open()
    run["server_session"].renew()
    assert [body for _, body, _ in env.server.posts[-2:]] == [
        {"client": client, "assistants": [], "web_ports": [port]}] * 2


def test_a_terminal_client_session_names_no_pages(env):
    assert _run(["pi", "--container"]) == 0
    server_session = env.runs[0]["server_session"]
    assert server_session.web_ports == []
    server_session.open()
    assert env.server.posts[-1][1] == {"client": "pi", "assistants": []}


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
    defaults = {**lc._ATTACH_IGNORED, **lc._ATTACH_REFUSED}
    for dest, default in defaults.items():
        assert getattr(a, dest) == default, dest
    rest = set(vars(a)) - set(defaults) - {"harness", "container", "shell", "passthrough"}
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


# Terminal output

def test_a_guest_named_git_folder_prints_no_terminal_controls(env, capsys, monkeypatch):
    """The guest names a git folder in its private home whose name holds an
    OSC 52 clipboard write and CSI cursor moves. Launch prints a note that
    names the folder, with every control shown as an escape."""
    import subprocess
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    evil = "G\x1b]52;c;ZWNobyBwd25lZAo=\x07\x1b[2K\x1b[1A\x9b"
    from gmlx.container import settings
    gitdir = settings.private_home("pi") / evil
    repo = env.home / "tmprepo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "."], cwd=repo, check=True)
    os.rename(repo / ".git", gitdir)
    (env.proj / ".git").write_text(f"gitdir: {gitdir}\n")
    assert _run(["pi", "--container"]) == 0
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert "\\x1b]52;c;ZWNobyBwd25lZAo=\\x07\\x1b[2K\\x1b[1A\\x9b" in text
    assert not any(ch in text for ch in "\x1b\x07\x9b")


def test_a_share_of_another_clients_build_folder_is_refused(env, capsys):
    box = env.proj / "box"
    box.mkdir()
    (box / "Containerfile").write_text("FROM x\n")
    _user_config(env.home, f"launch:\n  container:\n    clients:\n      omp:\n"
                           f"        build: {box}\n")
    assert _run(["pi", "--container"]) == 1
    err = capsys.readouterr().err
    assert "the build: folder of omp" in err and not env.runs


def test_a_launch_records_its_read_write_shares(env):
    from gmlx.container import settings
    assert _run(["pi", "--container"]) == 0
    assert os.path.realpath(env.proj) in settings.shared_history()


def test_the_dry_run_records_no_shares(env):
    from gmlx.container import settings
    assert _run(["pi", "--container", "--config-only"]) == 0
    assert settings.shared_history() == []
