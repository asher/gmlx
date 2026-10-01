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
from gmlx.container import runtime, session, settings

MODELS = [{"id": "qwen3.6-27b", "default": True, "context_length": 65536}]


@pytest.fixture
def env(fake_container, tmp_path, monkeypatch):
    home = tmp_path / "home"
    proj = home / "src" / "proj"
    proj.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(proj)
    # The fake service runs, and a running service has its kernel.
    _install_kernel(home)
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
        if kw.get("on_start"):
            kw["on_start"]()              # container run has started
        return 0
    monkeypatch.setattr(session, "supervise", supervise)
    copies = []

    def run_copy(argv, env, *, name, copy_id):
        """Record a joined copy's argv without its random copy ID."""
        at = argv.index("--copy-id")
        assert argv[at + 1] == copy_id and argv[at - 1] == "--join" and name in argv
        copies.append((argv[0], argv[:at] + argv[at + 2:], env))
        return 0
    monkeypatch.setattr(session, "run_copy", run_copy)
    fake_container.runs = runs
    fake_container.copies = copies
    fake_container.server = server
    fake_container.home = home
    fake_container.proj = proj
    fake_container.project = settings.project_id(settings.canonical(str(proj)))
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
        if not {"client", "assistants"} <= set(body) <= {"client", "assistants", "web_ports",
                                                         "project", "replaces"}:
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


def _project(env, client):
    """The project a launch of ``client`` from the project folder keys."""
    if client in settings.NO_CWD_CLIENTS:
        return settings.PROJECT_DEFAULT
    return env.project


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


def test_a_dry_run_with_reseed_keeps_the_private_copies(env, capsys):
    (env.home / ".foorc").write_text("mac v1\n")
    _user_config(env.home, "launch:\n  container:\n    clients:\n      pi:\n"
                           "        seed: [\"~/.foorc\"]\n")
    assert _run(["pi", "--container"]) == 0
    copy = settings.private_home_path("pi", env.project) / ".foorc"
    copy.write_text("edited in the container\n")
    assert _run(["pi", "--container", "--config-only", "--reseed"]) == 0
    assert copy.read_text() == "edited in the container\n"
    assert ("[launch] the dry run copies no seed again. A launch with --reseed copies "
            "~/.foorc again, in place of the copies in the private home."
            in capsys.readouterr().out.splitlines())
    assert _run(["pi", "--container", "--reseed"]) == 0
    assert copy.read_text() == "mac v1\n"


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


def test_a_data_folder_reached_through_a_link_launches(env, monkeypatch, tmp_path):
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data-link").symlink_to(tmp_path / "data")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data-link"))
    assert _run(["pi", "--container"]) == 0
    home = env.runs[0]["spec"].plan.home
    assert str(home) == os.path.realpath(home)


def test_a_project_id_that_another_folder_keys_is_refused(env, capsys):
    settings.private_home("pi", env.project)
    settings.write_project_record("pi", env.project, "/Users/u/other/proj")
    assert _run(["pi", "--container"]) == 1
    assert "because it belongs to /Users/u/other/proj" in capsys.readouterr().err
    assert not env.runs


def test_a_container_program_in_the_share_is_refused(env, capsys, monkeypatch, tmp_path):
    """An activated venv in the project puts a folder the client can write
    first on PATH, and launch runs the container program after the client
    exits."""
    venv_bin = env.proj / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    shutil.copy(tmp_path / "fakebin" / "container", venv_bin / "container")
    monkeypatch.setenv("PATH", f"{venv_bin}:{os.environ['PATH']}")
    assert _run(["pi", "--container"]) == 1
    err = capsys.readouterr().err
    assert ("[launch] launch found the container program at ~/src/proj/.venv/bin/container, "
            "which lies in ~/src/proj, a folder this launch shares.") in err
    assert not env.runs


def test_a_broken_launch_block_leaves_host_mode_running(env, capsys, monkeypatch):
    which = launch.shutil.which
    monkeypatch.setattr(launch.shutil, "which",
                        lambda name: "/usr/bin/pi" if name == "pi" else which(name))
    _user_config(env.home, "launch:\n  container:\n    network: offline\n")
    calls = []
    assert _run(["pi"], exec_fn=lambda *a: calls.append(a) or 0) == 0
    assert calls and not env.runs
    assert "ignoring the launch settings, so pi runs on the Mac" in capsys.readouterr().err
    assert _run(["pi", "--container"]) == launch.EXIT_CONFIG                 # asked for, so it stops
    assert _run(["pi", "--rebuild"]) == launch.EXIT_CONFIG
    # Each message names the file that holds the bad value.
    err = capsys.readouterr().err
    assert err.count(str(env.home / ".config" / "gmlx" / "gmlx.yaml")) == 2, err
    assert "launch.container.network" in err


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
    assert _run(["pi"], exec_fn=lambda *a: calls.append(a) or 0) == launch.EXIT_CONFIG
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
    assert _run(["pi"], exec_fn=lambda *a: calls.append(a) or 0) == launch.EXIT_CONFIG
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
    assert _run(["pi"], exec_fn=lambda *a: calls.append(a) or 0) == launch.EXIT_CONFIG
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
    assert _run(["pi"], exec_fn=lambda *a: calls.append(a) or 0) == launch.EXIT_CONFIG
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


@pytest.mark.parametrize("text", ["launch: [unclosed\n", "server:\n  api_key: [k\n"])
def test_unreadable_yaml_never_runs_on_the_mac(env, capsys, text):
    """A file that does not parse may have no launch block, so the message
    does not name one."""
    _user_config(env.home, text)
    assert _run(["pi"]) == launch.EXIT_CONFIG
    lines = capsys.readouterr().err.splitlines()
    # The sentence starts its own line after the parser's location lines.
    assert lines[-2].lstrip().startswith("in ") and lines[-2].rstrip()[-1].isdigit()
    assert lines[-1] == ("gmlx launch cannot read that file, so it cannot tell whether "
                         "container mode is on for pi. Fix the file, or pass "
                         "--no-container to run pi on the Mac.")


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
    session.write_record("claude-code", env.project, {"name": "leftover"})
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
    assert session.read_record("claude-code", env.project) is None      # step 5 ran
    assert session.try_session_lock("claude-code", env.project) is not None   # released


def test_dry_run_opens_with_its_header(env, capsys):
    assert _run(["pi", "--container", "--config-only"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == ("[launch] container dry run: no image is built or pulled, and no "
                        "container is started.")
    assert lines[1].startswith("[launch] container 1.5.0")


def test_an_attaching_dry_run_prints_no_prerequisites(env, capsys):
    lock = session.try_session_lock("pi", env.project)
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
    assert _run(["pi", "--container", "--base-url", "https://example.invalid/v1"]) == launch.EXIT_UNAVAILABLE
    assert capsys.readouterr().err == (
        "[launch] cannot reach the server at https://example.invalid/v1 (Connection "
        "refused). Check the URL, or start that server.\n")
    assert not env.calls("image") and not env.calls("build") and not env.runs


def test_a_server_without_session_sockets_is_refused_before_the_image_steps(env, capsys):
    env.server.status = 404
    env.update(running=False)
    assert _run(["pi", "--container"]) == launch.EXIT_UNAVAILABLE
    assert "does not offer session sockets" in capsys.readouterr().err
    assert not env.calls("system", "start") and not env.calls("build")
    assert not env.calls("image")


@pytest.mark.parametrize("models, argv, message, code", [
    ([], ["pi"], "has no models yet. Download one with gmlx pull", launch.EXIT_UNAVAILABLE),
    (MODELS, ["pi", "--model", "nosuch"],
     "--model nosuch is not a model the server offers. It offers qwen3.6-27b.",
     launch.EXIT_FAILURE),
    ([{"id": "m-a"}, {"id": "m-b"}], ["goose"], "goose needs a default model",
     launch.EXIT_FAILURE)])
def test_a_model_the_launch_cannot_use_is_refused_before_the_image_steps(
        env, capsys, monkeypatch, models, argv, message, code):
    def get_json(url, timeout=5.0, headers=None):
        return {"data": models} if url.endswith("/models") else {}
    monkeypatch.setattr(launch, "_http_get_json", get_json)
    env.update(running=False)
    assert _run([argv[0], "--container", *argv[1:]]) == code
    assert message in capsys.readouterr().err
    assert not env.calls("system", "start") and not env.calls("build")
    assert not env.calls("image") and not env.runs


def test_a_refused_model_leaves_only_the_session_lock(env, capsys):
    from gmlx.container.state import data_path

    def files():
        return {p for root in (env.home, data_path()) for p in root.rglob("*") if p.is_file()}
    before = files()
    assert _run(["pi", "--container", "--model", "nosuch"]) == launch.EXIT_FAILURE
    assert "is not a model the server offers" in capsys.readouterr().err
    assert files() - before == {settings.project_dir_path("pi", env.project) / "session.lock"}


def _no_server(env, monkeypatch):
    """No server answers until launch starts one from the user config. The
    list returned holds the number of container calls at each start."""
    state = {"up": False}
    monkeypatch.setattr(launch, "_server_ready", lambda base, api_key=None: state["up"])
    config = env.home / ".config" / "gmlx" / "gmlx.yaml"
    _user_config(env.home, "server:\n  port: 8080\n")

    class Cfg:
        host, port, api_key = "127.0.0.1", 8080, None
    monkeypatch.setattr(launch, "_discover_config", lambda: (Cfg(), str(config)))
    starts = []

    def autostart(**kw):
        starts.append(len(env.log))
        state["up"] = True
        return 0, True, "qwen3.6-27b"
    monkeypatch.setattr(launch, "_autostart", autostart)
    return starts


def test_a_refused_model_with_no_server_leaves_only_the_session_lock(env, capsys,
                                                                     monkeypatch):
    from gmlx.container.state import data_path

    starts = _no_server(env, monkeypatch)

    def paths():
        return {p for root in (env.home, data_path()) for p in root.rglob("*")}
    before = paths()
    assert _run(["pi", "--container", "--model", "nosuch"]) == launch.EXIT_FAILURE
    assert "is not a model the server offers" in capsys.readouterr().err
    assert starts
    lock = settings.project_dir_path("pi", env.project) / "session.lock"
    assert {p for p in paths() - before if p.is_file()} == {lock}
    assert not settings.private_home_path("pi", env.project).exists()
    assert not env.calls("image") and not env.calls("build") and not env.runs


def test_a_launch_starts_the_server_before_the_image_steps(env, monkeypatch):
    starts = _no_server(env, monkeypatch)
    assert _run(["pi", "--container"]) == 0
    assert len(starts) == 1 and env.runs
    assert not [c for c in env.log[:starts[0]] if c[:1] in (["image"], ["build"])]


@pytest.mark.parametrize("argv, message", [
    (["--mount", "~/nosuch"], "does not exist"),
    (["--mount", "~/src/proj:/"], "cannot use / in the container")])
def test_a_refused_share_or_flag_stops_before_the_server_starts(env, capsys, monkeypatch,
                                                                argv, message):
    starts = _no_server(env, monkeypatch)
    assert _run(["pi", "--container", *argv]) != 0
    assert message in capsys.readouterr().err
    assert not starts


def test_a_refused_home_share_stops_before_the_server_starts(env, capsys, monkeypatch):
    starts = _no_server(env, monkeypatch)
    monkeypatch.chdir(env.home)
    assert _run(["pi", "--container"]) == 1
    assert "will not share the current folder" in capsys.readouterr().err
    assert not starts


def test_a_dry_run_with_no_server_still_starts_it(env, capsys, monkeypatch):
    starts = _no_server(env, monkeypatch)
    assert _run(["pi", "--container", "--config-only"]) == 0
    assert len(starts) == 1
    assert "left a background server running" in capsys.readouterr().err


def test_the_session_probe_uses_the_key_of_the_config_the_server_records(
        env, monkeypatch, tmp_path):
    served = tmp_path / "served.yaml"
    served.write_text("server:\n  api_key: from-served-config\n")
    _user_config(env.home, "server:\n  api_key: from-user-config\n")
    monkeypatch.setattr(launch, "_auth_required", lambda base: True)
    monkeypatch.setattr(lifecycle, "read_run", lambda h, p: {
        "pid": os.getpid(), "api_key_set": True, "config_abspath": str(served)})
    assert _run(["pi", "--container"]) == 0
    assert {key for _url, _body, key in env.server.posts} == {"from-served-config"}


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
    lock = session.try_session_lock("pi", env.project)
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
    assert _run(["pi", "--container"]) == launch.EXIT_UNAVAILABLE
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
    # The window the server reports, so Claude Code compacts before the
    # conversation outgrows the model.
    assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" in spec.env_names
    assert spec.child_env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "65536"


_REPLACED = ("[launch] Claude Code gets CLAUDE_CODE_MAX_CONTEXT_TOKENS=65536, the window of "
             "qwen3.6-27b, in place of your {}")


@pytest.mark.parametrize("mac, entry, gets, line", [
    ("50000", None, "65536", None),            # a shell export alone does not count
    (None, "NAME=50000", "50000", None),
    (None, "NAME=200000", "65536", _REPLACED.format("200000")),
    ("50000", "NAME", "50000", None),          # NAME alone takes the Mac's value
    ("200000", "NAME", "65536", _REPLACED.format("200000")),
    (None, "NAME", "65536", None),
    ("1", "NAME=abc", "65536", _REPLACED.format("abc"))])
def test_claude_code_in_a_container_keeps_the_smaller_context_window(
        env, capsys, monkeypatch, mac, entry, gets, line):
    name = "CLAUDE_CODE_MAX_CONTEXT_TOKENS"
    if mac is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, mac)
    if entry:
        _user_config(env.home, "launch:\n  container:\n    clients:\n      claude-code:\n"
                               f"        env: [\"{entry.replace('NAME', name)}\"]\n")
    assert _run(["claude-code", "--container"]) == 0
    spec = env.runs[0]["spec"]
    assert spec.child_env[name] == gets and name in spec.env_names
    out = capsys.readouterr().out
    if line:
        assert line in out.splitlines()
    else:
        assert "in place of your" not in out


@pytest.mark.parametrize("entry, line", [
    (None, "CLAUDE_CODE_MAX_CONTEXT_TOKENS=65536, the window of qwen3.6-27b"),
    ("NAME= 50000", "CLAUDE_CODE_MAX_CONTEXT_TOKENS=50000, your own value"),
    ("NAME=200000", None)])
def test_the_dry_run_shows_the_context_window_claude_code_gets(env, capsys, monkeypatch,
                                                               entry, line):
    name = "CLAUDE_CODE_MAX_CONTEXT_TOKENS"
    monkeypatch.delenv(name, raising=False)
    if entry:
        _user_config(env.home, "launch:\n  container:\n    clients:\n      claude-code:\n"
                               f"        env: [\"{entry.replace('NAME', name)}\"]\n")
    assert _run(["claude-code", "--container", "--config-only"]) == 0
    out = capsys.readouterr().out.splitlines()
    shown = [s for s in out if s.startswith(f"[launch] Claude Code gets {name}=")]
    if line:
        assert shown == [f"[launch] Claude Code gets {line}"]
    else:
        assert shown == [_REPLACED.format("200000")]
    assert f"-e {name} " in next(s for s in out if " run " in s)


@pytest.mark.parametrize("entry", ["NAME=200000", "NAME"])
def test_with_no_window_known_the_entry_reaches_claude_code(env, monkeypatch, entry):
    name = "CLAUDE_CODE_MAX_CONTEXT_TOKENS"
    monkeypatch.setenv(name, "200000")
    monkeypatch.setattr(launch, "_http_get_json", lambda url, timeout=5.0, headers=None: (
        {"data": [{"id": "qwen3.6-27b", "default": True}]} if url.endswith("/models") else {}))
    _user_config(env.home, "launch:\n  container:\n"
                           f"    env: [\"{entry.replace('NAME', name)}\"]\n")
    assert _run(["claude-code", "--container"]) == 0
    spec = env.runs[0]["spec"]
    assert name in spec.env_names
    # With NAME alone, the Mac's value reaches the container through the
    # environment of container run.
    assert spec.child_env.get(name, os.environ[name]) == "200000"


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


def test_an_ssh_agent_socket_reaches_container_run(env, monkeypatch):
    sock_dir = Path(tempfile.mkdtemp(prefix="ga-", dir="/tmp"))
    try:
        path = sock_dir / "agent.sock"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.bind(str(path))
            _user_config(env.home, f"launch:\n  container:\n    clients:\n      pi:\n"
                                   f"        ssh_agent: {path}\n")
            monkeypatch.setattr(settings, "_ssh_add_list", lambda sock: 0)
            assert _run(["pi", "--container"]) == 0
        spec = env.runs[0]["spec"]
        assert spec.plan.ssh_agent is True
        assert spec.child_env["SSH_AUTH_SOCK"] == os.path.realpath(path)
        assert "SSH_AUTH_SOCK" not in spec.env_names and "SSH_AUTH_SOCK" not in spec.env_values
    finally:
        shutil.rmtree(sock_dir, ignore_errors=True)


def test_ssh_agent_true_without_an_agent_passes_no_ssh(env, monkeypatch, capsys):
    _user_config(env.home, "launch:\n  container:\n    ssh_agent: true\n")
    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
    assert _run(["pi", "--container", "--config-only"]) == 0
    out = capsys.readouterr().out
    assert "SSH_AUTH_SOCK is not set, so the container gets no SSH agent" in out
    assert "container run" in out and "--ssh" not in out
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/env.sock")
    monkeypatch.setattr(settings, "_ssh_add_list", lambda sock: 0)
    assert _run(["pi", "--container", "--config-only"]) == 0
    assert "--ssh" in capsys.readouterr().out


def test_ssh_agent_true_forwards_the_real_path_of_ssh_auth_sock(env, monkeypatch):
    """Apple's relay opens the forwarded path again for each connection."""
    sock_dir = Path(tempfile.mkdtemp(prefix="ga-", dir="/tmp"))
    try:
        path = sock_dir / "agent.sock"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.bind(str(path))
            (sock_dir / "link.sock").symlink_to(path)
            monkeypatch.setenv("SSH_AUTH_SOCK", str(sock_dir / "link.sock"))
            _user_config(env.home, "launch:\n  container:\n    ssh_agent: true\n")
            monkeypatch.setattr(settings, "_ssh_add_list", lambda sock: 0)
            assert _run(["pi", "--container"]) == 0
        assert env.runs[0]["spec"].child_env["SSH_AUTH_SOCK"] == os.path.realpath(path)
    finally:
        shutil.rmtree(sock_dir, ignore_errors=True)


@pytest.mark.parametrize("dry", [False, True])
def test_an_agent_with_no_keys_gets_a_line(env, monkeypatch, capsys, dry):
    _user_config(env.home, "launch:\n  container:\n    ssh_agent: true\n")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/env.sock")
    monkeypatch.setattr(settings, "_ssh_add_list", lambda sock: 1)
    assert _run(["pi", "--container", *(["--config-only"] if dry else [])]) == 0
    assert "[launch] ssh_agent is on, but the SSH agent holds no keys" in capsys.readouterr().out


def test_a_merged_file_is_named_from_the_private_home(env, capsys):
    assert _run(["pi", "--container"]) == 0
    assert ("[launch] merged ~/.pi/agent/models.json and ~/.pi/agent/settings.json in "
            "the private home\n") in capsys.readouterr().out


def test_the_open_webui_login_hint_is_gone_once_its_data_folder_exists(env, capsys):
    assert _run(["open-webui", "--container"]) == 0
    assert "without a login" in capsys.readouterr().out
    assert _run(["open-webui", "--container"]) == 0
    assert "without a login" not in capsys.readouterr().out


def test_open_webui_listens_on_loopback_with_host_and_port(env, capsys):
    assert _run(["open-webui", "--container"]) == 0
    out = capsys.readouterr().out
    # The session prints the one address and opens it.
    assert "http://localhost" not in out and "open the URL" not in out
    assert ("[launch] Open WebUI keeps its chat history and database in ~/.open-webui "
            "in the private home\n") in out
    assert ("[launch] To use Open WebUI without a login, stop it before you create an "
            "account, add WEBUI_AUTH=false to launch.container.clients.open-webui.env in "
            "your gmlx config, and launch again.\n") in out
    assert "WEBUI_AUTH=false gmlx launch" not in out
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


def _dsh_manifest(profile, bundles, project=settings.PROJECT_DEFAULT):
    d = settings.private_home("dsh", project) / ".dsh" / "profiles" / profile
    d.mkdir(parents=True, exist_ok=True)
    (d / "package.json").write_text(json.dumps({"dsh": {"profile": {"bundles": bundles}}}))


def test_a_guest_manifest_never_makes_a_dsh_profile_a_web_session(env):
    """The profile manifests lie in the private home, which the guest
    writes, so the profile name alone decides whether the Mac binds a web
    port and opens a browser."""
    _dsh_manifest("mycli", [launch._DSH_WEB_BUNDLE], env.project)
    assert _run(["dsh", "--container", "--dsh-profile", "mycli"]) == 0
    assert env.runs[-1]["spec"].web_port is None
    assert env.runs[-1].get("opener") is None
    _dsh_manifest("gmlx", ["@deepseek-ai/dsh-cli"], env.project)
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


def test_a_localhost_domain_warns_unless_the_network_is_none(env, capsys, monkeypatch,
                                                             tmp_path):
    from gmlx.container import localhost_domains
    etc = tmp_path / "etc"
    (etc / "pf.anchors").mkdir(parents=True)
    (etc / "pf.anchors" / "com.apple.container").write_text(
        "rdr inet from any to 203.0.113.113 -> 127.0.0.1 # host.container.internal\n")
    monkeypatch.setattr(localhost_domains, "ETC", etc)
    assert _run(["pi", "--container"]) == 0
    assert ("[launch] warning: the Apple container localhost domain host.container.internal "
            "(203.0.113.113) sends this container to the loopback address of this Mac"
            ) in capsys.readouterr().out
    assert _run(["pi", "--container", "--network", "none"]) == 0
    assert "localhost domain" not in capsys.readouterr().out


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
    assert env.server.posts[-1][1] == {"client": "aichat", "assistants": ["home", "nope"],
                                       "project": server.project}
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
    assert _run(["pi", "--container"]) == launch.EXIT_UNAVAILABLE
    err = capsys.readouterr().err
    assert "does not offer session sockets" in err and "gmlx restart" in err
    assert "run the client on the Mac with --no-container." in err
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
    assert _run(["pi", "--container"]) == launch.EXIT_UNAVAILABLE
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


def test_a_server_on_every_address_gets_a_bracketed_url(env, monkeypatch, capsys):
    """A server bound to :: has an IPv6 host, which a URL holds in brackets."""
    monkeypatch.setattr(lifecycle, "auto_target", lambda h, p: ("::", 8080))
    seen = []
    monkeypatch.setattr(launch, "_server_ready",
                        lambda base, api_key=None: seen.append(base) or True)
    assert _run(["pi", "--container"]) == 0, capsys.readouterr().err
    assert seen and set(seen) == {"http://[::]:8080/v1"}
    assert env.runs


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
    # The renewal names the session it replaces.
    assert env.server.posts[-1][1] == {"client": "aichat", "assistants": ["home"],
                                       "replaces": "s1"}
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
    assert _run(["pi", "--container"]) == launch.EXIT_UNAVAILABLE
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
    lock = session.try_session_lock("pi", env.project)
    proj = os.path.realpath(env.proj)
    session.write_record("pi", env.project, {
        "name": "gmlx-pi-abc123", "workdir": proj, "clipboard": False,
        "shares": [{"host": proj, "guest": proj, "readonly": False}],
        "command": ["pi", "--provider", "gmlx"], "project": proj})
    env.update(containers=[{"name": "gmlx-pi-abc123", "labels": {
        "gmlx.launch": "1", "gmlx.launch.client": "pi", "gmlx.launch.project": env.project,
        "gmlx.launch.pid": str(os.getpid())}}])
    yield env
    lock.release()


_ENTRY = "/opt/gmlx/gmlx-entry"


def test_a_second_launch_joins_the_running_session(running_session, capsys):
    calls = running_session.copies
    rc = _run(["pi", "--container", "--", "--continue"])
    assert rc == 0 and not running_session.runs
    argv = calls[0][1]
    proj = os.path.realpath(running_session.proj)
    assert argv[1:] == ["exec", "-i", "--cwd", proj, "gmlx-pi-abc123", _ENTRY, "--join", "--",
                        "pi", "--provider", "gmlx", "--continue"]
    assert capsys.readouterr().out == "[launch] joining the running pi session for ~/src/proj\n"


def test_a_join_of_a_command_image_session_replaces_cmd(running_session):
    record = session.read_record("pi", running_session.project)
    session.write_record("pi", running_session.project,
                         {**record, "command": ["/entry", "serve"], "entrypoint": ["/entry"]})
    calls = running_session.copies
    assert _run(["pi", "--container"]) == 0
    assert calls[0][1][-3:] == ["--", "/entry", "serve"]
    assert _run(["pi", "--container", "--", "chat"]) == 0
    assert calls[1][1][-3:] == ["--", "/entry", "chat"]


def test_a_session_without_a_recorded_command_takes_only_a_shell(running_session, capsys):
    record = session.read_record("pi", running_session.project)
    session.write_record("pi", running_session.project, {**record, "command": None})
    assert _run(["pi", "--container"]) == 1
    assert "does not record the command it runs" in capsys.readouterr().err


def test_another_project_starts_its_own_session_beside_a_running_one(running_session, capsys):
    other = running_session.home / "src" / "other"
    other.mkdir()
    os.chdir(other)
    assert _run(["pi", "--container"]) == 0
    spec = running_session.runs[0]["spec"]
    project = settings.project_id(settings.canonical(str(other)))
    assert spec.session.project == project != running_session.project
    assert spec.plan.home == settings.private_home_path("pi", project)
    assert "shares files with this session" not in capsys.readouterr().out
    assert "--label" in (argv := session.compose_run_argv(spec))
    assert f"gmlx.launch.project={project}" in argv


def _subfolder(env, *parts) -> str:
    folder = env.proj.joinpath(*parts)
    folder.mkdir(parents=True)
    os.chdir(folder)
    return os.path.realpath(folder)


def test_a_launch_from_a_subfolder_joins_the_session_that_shares_it(running_session, capsys):
    sub = _subfolder(running_session, "sub", "deep")
    calls = running_session.copies
    assert _run(["pi", "--container"]) == 0
    assert not running_session.runs
    assert calls[0][1][1:6] == ["exec", "-i", "--cwd", sub, "gmlx-pi-abc123"]
    assert capsys.readouterr().out == "[launch] joining the running pi session for ~/src/proj\n"
    assert _run(["pi", "--shell"]) == 0
    assert calls[1][1][1:] == ["exec", "-i", "--cwd", sub, "gmlx-pi-abc123",
                               _ENTRY, "--join", "--shell", "--"]
    # The subfolder's own project folder held only the lock, so it is gone.
    assert not settings.project_dir_path("pi", settings.project_id(sub)).exists()


def test_a_joining_subfolder_keeps_its_own_home(running_session):
    sub = _subfolder(running_session, "sub")
    home = settings.private_home("pi", settings.project_id(sub))
    assert _run(["pi", "--container"], exec_fn=lambda *a: 0) == 0
    assert home.is_dir() and (home.parent / "session.lock").exists()


def test_a_sibling_folder_with_a_longer_name_starts_its_own_session(running_session):
    sibling = running_session.home / "src" / "proj2"
    sibling.mkdir()
    os.chdir(sibling)
    assert _run(["pi", "--container"]) == 0
    project = running_session.runs[0]["spec"].session.project
    assert project == settings.project_id(os.path.realpath(sibling))


def test_a_record_whose_container_is_gone_counts_as_no_session(running_session, capsys):
    running_session.update(containers=[])
    sub = _subfolder(running_session, "sub")
    assert _run(["pi", "--container"]) == 0
    assert running_session.runs[0]["spec"].session.project == settings.project_id(sub)
    assert "shares files with this session" not in capsys.readouterr().out


def test_a_subfolder_launch_skips_a_session_whose_launch_is_gone(running_session, capsys):
    """A killed launch leaves its container running with no relays, so a
    copy in it could reach no server. The launch reports the container."""
    running_session.update(containers=[{"name": "gmlx-pi-abc123", "labels": {
        "gmlx.launch": "1", "gmlx.launch.client": "pi",
        "gmlx.launch.project": running_session.project, "gmlx.launch.pid": "999999"}}])
    sub = _subfolder(running_session, "sub")
    assert _run(["pi", "--container"]) == 0
    assert not running_session.copies
    assert running_session.runs[0]["spec"].session.project == settings.project_id(sub)
    assert ("[launch] gmlx-pi-abc123 from an earlier pi launch is still running and holds 4G "
            "of memory. Stop it with: container stop gmlx-pi-abc123"
            in capsys.readouterr().out.splitlines())


def test_a_read_only_share_takes_no_launch_of_another_project(running_session, capsys):
    """A session that only reads ~/src cannot change another project's
    files, so that project gets its own session, home and history."""
    record = session.read_record("pi", running_session.project)
    src = os.path.realpath(running_session.home / "src")
    session.write_record("pi", running_session.project, {**record, "shares": [
        *record["shares"], {"host": src, "guest": src, "readonly": True}]})
    other = running_session.home / "src" / "other"
    other.mkdir()
    os.chdir(other)
    assert _run(["pi", "--container"]) == 0
    assert not running_session.copies
    project = settings.project_id(os.path.realpath(other))
    assert running_session.runs[0]["spec"].session.project == project
    assert "shares files with this session" in capsys.readouterr().out


def test_a_subfolder_of_a_project_shared_read_only_joins_its_session(running_session):
    record = session.read_record("pi", running_session.project)
    session.write_record("pi", running_session.project, {**record, "shares": [
        {**record["shares"][0], "readonly": True}]})
    sub = _subfolder(running_session, "sub")
    assert _run(["pi", "--container"]) == 0
    assert not running_session.runs
    assert running_session.copies[0][1][1:6] == ["exec", "-i", "--cwd", sub, "gmlx-pi-abc123"]


_OVERLAP = ("[launch] the running pi session for ~/src/proj shares files with this session. "
            "File locks do not reach from one virtual machine to another, so do not let two "
            "clients change the same file at once.")


def test_another_client_in_the_project_starts_its_own_session_with_a_warning(
        running_session, capsys):
    assert _run(["claude-code", "--container"]) == 0
    assert running_session.runs[0]["spec"].session.project == running_session.project
    assert _OVERLAP in capsys.readouterr().out.splitlines()
    _subfolder(running_session, "sub")
    assert _run(["claude-code", "--container"]) == 0     # a folder inside the share
    assert _OVERLAP in capsys.readouterr().out.splitlines()


def test_a_launch_from_the_parent_folder_starts_its_own_session_with_a_warning(
        running_session, capsys):
    src = running_session.home / "src"
    os.chdir(src)
    assert _run(["pi", "--container"]) == 0
    project = running_session.runs[0]["spec"].session.project
    assert project == settings.project_id(os.path.realpath(src))
    assert _OVERLAP in capsys.readouterr().out.splitlines()


def test_the_session_with_the_longest_share_takes_the_join(running_session, capsys):
    src = os.path.realpath(running_session.home / "src")
    lock = session.try_session_lock("pi", "src-0badc0de")
    session.write_record("pi", "src-0badc0de", {
        "name": "gmlx-pi-out999", "workdir": src, "clipboard": False,
        "shares": [{"host": src, "guest": src, "readonly": False}], "project": src})
    running_session.update(containers=[*running_session.load()["containers"], {
        "name": "gmlx-pi-out999", "labels": {"gmlx.launch": "1", "gmlx.launch.client": "pi",
                                             "gmlx.launch.project": "src-0badc0de",
                                             "gmlx.launch.pid": str(os.getpid())}}])
    calls = running_session.copies
    try:
        _subfolder(running_session, "sub")
        assert _run(["pi", "--shell"]) == 0
        (running_session.home / "src" / "other").mkdir()
        os.chdir(running_session.home / "src" / "other")
        assert _run(["pi", "--shell"]) == 0
    finally:
        lock.release()
    assert calls[0][1][5] == "gmlx-pi-abc123"
    assert calls[1][1][5] == "gmlx-pi-out999"
    assert "opening a shell in the running pi session for ~/src (gmlx-pi-out999)" in (
        capsys.readouterr().out)


@pytest.mark.parametrize("record", [{"name": "gmlx-pi-abc123"},
                                    {"name": "gmlx-pi-abc123", "workdir": "/w",
                                     "shares": [{"guest": "/w"}]}])
def test_shell_with_a_damaged_record_is_a_clean_error(running_session, capsys, record):
    session.record_path("pi", running_session.project).write_text(json.dumps(record))
    assert _run(["pi", "--shell"], exec_fn=lambda *a: pytest.fail("exec")) == 1
    err = capsys.readouterr().err
    assert "is damaged, so this launch cannot join" in err
    assert err.endswith("End that session and launch again. Stop it with: container stop "
                        "gmlx-pi-abc123\n")
    assert _run(["pi", "--container"]) == 1
    assert "is damaged" in capsys.readouterr().err


def test_shell_attaches_to_the_running_session(running_session, capsys):
    calls = running_session.copies
    rc = _run(["pi", "--shell", "--", "-c", "ls"])
    assert rc == 0
    argv = calls[0][1]
    proj = os.path.realpath(running_session.proj)
    assert argv[1:] == ["exec", "-i", "--cwd", proj, "gmlx-pi-abc123",
                        _ENTRY, "--join", "--shell", "--", "-c", "ls"]
    assert ("opening a shell in the running pi session for ~/src/proj (gmlx-pi-abc123)"
            in capsys.readouterr().out)


def test_shell_attach_passes_clipboard_when_the_session_has_it(running_session):
    record = session.read_record("pi", running_session.project)
    session.write_record("pi", running_session.project, {**record, "clipboard": True})
    calls = running_session.copies
    assert _run(["pi", "--shell"]) == 0
    argv = calls[0][1]
    assert argv[argv.index(_ENTRY) + 1:] == ["--clipboard", "--join", "--shell", "--"]


def test_shell_attach_to_a_session_that_shares_no_folder(env, capsys):
    lock = session.try_session_lock("pi", "default")
    session.write_record("pi", "default", {"name": "gmlx-pi-def456", "workdir": "/h",
                                           "clipboard": False, "shares": []})
    env.update(containers=[{"name": "gmlx-pi-def456", "labels": {
        "gmlx.launch": "1", "gmlx.launch.client": "pi", "gmlx.launch.project": "default"}}])
    calls = env.copies
    try:
        assert _run(["pi", "--shell", "--no-mount-cwd"]) == 0
    finally:
        lock.release()
    assert "--cwd" not in calls[0][1]
    assert ("the current folder is not shared with this session, so the shell opens in its "
            "working folder /h.") in capsys.readouterr().out


def test_a_join_from_a_folder_the_session_does_not_share_names_its_shares(env, capsys):
    """A default-project session takes every launch of its client, so a
    copy from another folder says where it runs."""
    a = env.home / "src" / "a"
    a.mkdir()
    a = os.path.realpath(a)
    lock = session.try_session_lock("pi", "default")
    session.write_record("pi", "default", {
        "name": "gmlx-pi-def456", "workdir": a, "clipboard": False, "command": ["pi"],
        "shares": [{"host": a, "guest": a, "readonly": False}]})
    env.update(containers=[{"name": "gmlx-pi-def456", "labels": {
        "gmlx.launch": "1", "gmlx.launch.client": "pi", "gmlx.launch.project": "default",
        "gmlx.launch.pid": str(os.getpid())}}])
    try:
        assert _run(["pi", "--container", "--no-mount-cwd"]) == 0
    finally:
        lock.release()
    assert "--cwd" not in env.copies[0][1]
    assert capsys.readouterr().out == (
        "[launch] joining the running pi session\n"
        "[launch] the current folder is not shared with this session, which shares ~/src/a, "
        f"so pi starts in its working folder {a}.\n")


def test_a_web_app_join_from_a_folder_it_does_not_share_says_so(env, capsys, monkeypatch):
    import webbrowser
    monkeypatch.setattr(webbrowser, "open", lambda url: None)
    data = env.home / "data"
    data.mkdir()
    data = os.path.realpath(data)
    lock = _web_session(env, "open-webui", web_port=3000,
                        shares=[{"host": data, "guest": "/data", "readonly": True}])
    try:
        assert _run(["open-webui", "--container", "--mount-cwd"]) == 0
    finally:
        lock.release()
    assert capsys.readouterr().out == (
        "[launch] --mount-cwd applies only to a new session, so this launch ignores it.\n"
        "[launch] open-webui is already running at http://127.0.0.1:3000/\n"
        "[launch] the current folder is not shared with this session, which shares ~/data "
        "(read-only).\n")


def test_shell_attach_refuses_new_session_flags(running_session, capsys):
    assert _run(["pi", "--shell", "--mount", "/tmp"]) == 1
    assert "--mount applies only to a new session" in capsys.readouterr().err
    assert _run(["pi", "--image", "x"]) == 1
    assert ("a pi session is already running for ~/src/proj, so this launch joins it, and "
            "--image applies only to a new session.") in capsys.readouterr().err


def test_shell_attach_ignores_the_server_flags(running_session, capsys):
    """The flags that chose the session's server and model are what the
    user typed to start it, so the shell takes them and says it ignores
    them."""
    calls = running_session.copies
    assert _run(["pi", "--shell", "--port", "48611", "--no-start"]) == 0
    assert calls
    assert ("[launch] --port and --no-start apply only to a new session, so the shell "
            "ignores them.") in capsys.readouterr().out
    assert _run(["pi", "--container", "--model", "m"]) == 0
    assert ("[launch] --model applies only to a new session, so this copy ignores it."
            in capsys.readouterr().out)


def test_shell_attach_while_the_session_starts(running_session, capsys):
    session.remove_record("pi", running_session.project)
    assert _run(["pi", "--shell"]) == launch.EXIT_TEMPFAIL
    assert "still starting" in capsys.readouterr().err


def test_a_launch_while_the_session_ends_says_so(running_session, capsys):
    record = session.read_record("pi", running_session.project)
    session.write_record("pi", running_session.project,
                         {**record, "ending": True, "pid": os.getpid()})
    line = ("[launch] the pi session for ~/src/proj is ending. Launch again once it has "
            "stopped.\n")
    assert _run(["pi", "--container"]) == launch.EXIT_TEMPFAIL
    assert capsys.readouterr().err == line
    sub = _subfolder(running_session, "sub")
    assert _run(["pi", "--container"]) == launch.EXIT_TEMPFAIL
    assert capsys.readouterr().err == line
    assert not running_session.runs and not running_session.copies
    assert not settings.project_dir_path("pi", settings.project_id(sub)).exists()


def _starting(env, pid):
    """The record of a launch in the project folder that has not reached
    its container yet."""
    proj = os.path.realpath(env.proj)
    session.write_record("pi", env.project, {
        "name": "", "workdir": proj, "starting": True, "pid": pid,
        "shares": [{"host": proj, "guest": proj, "readonly": False}], "project": proj})
    env.update(containers=[])


def test_a_subfolder_launch_waits_for_a_session_that_starts(running_session, capsys):
    _starting(running_session, os.getpid())
    sub = _subfolder(running_session, "sub")
    assert _run(["pi", "--container"]) == launch.EXIT_TEMPFAIL
    assert capsys.readouterr().err == ("[launch] the pi session for ~/src/proj is still "
                                       "starting. Try again in a moment.\n")
    assert not running_session.runs
    assert not settings.project_dir_path("pi", settings.project_id(sub)).exists()
    _starting(running_session, 999999)                   # that launch was killed
    assert _run(["pi", "--container"]) == 0
    assert running_session.runs[0]["spec"].session.project == settings.project_id(sub)


def test_a_launch_whose_own_project_starts_looks_for_an_enclosing_session(running_session):
    """Two launches from one subfolder at once: the one that finds the lock
    held joins the session that holds the folder, as the other will."""
    sub = _subfolder(running_session, "sub")
    lock = session.try_session_lock("pi", settings.project_id(sub))
    try:
        assert _run(["pi", "--container"]) == 0
    finally:
        lock.release()
    assert running_session.copies[0][1][5] == "gmlx-pi-abc123"


def test_a_new_session_is_visible_while_it_starts(env, monkeypatch):
    seen = []
    monkeypatch.setattr(session, "cleanup_stale", lambda client, project, **kw: seen.append(
        session.read_record(client, project)))
    assert _run(["pi", "--container"]) == 0
    proj = os.path.realpath(env.proj)
    assert seen == [{"name": "", "workdir": proj, "starting": True, "pid": os.getpid(),
                     "shares": [{"host": proj, "guest": proj, "readonly": False}],
                     "project": proj, "web": False, "web_port": None}]
    assert session.read_record("pi", env.project) is None
    assert _run(["pi", "--container", "--network", "none", "--base-url",
                 "https://api.example.com/v1"]) == 1
    assert session.read_record("pi", env.project) is None
    assert _run(["pi", "--container", "--config-only"]) == 0
    assert len(seen) == 1 and session.read_record("pi", env.project) is None


def _web_session(env, client, key="default", **record):
    lock = session.try_session_lock(client, key)
    session.write_record(client, key, {
        "name": f"gmlx-{client}-abc123", "workdir": "/w", "clipboard": False, "shares": [],
        "web": True, **record})
    env.update(containers=[{"name": f"gmlx-{client}-abc123", "labels": {
        "gmlx.launch": "1", "gmlx.launch.client": client, "gmlx.launch.project": key,
        "gmlx.launch.pid": str(os.getpid())}}])
    return lock


def _dsh_session(env, **record):
    """A dsh web session that shares the project folder."""
    proj = os.path.realpath(env.proj)
    return _web_session(env, "dsh", env.project, web_port=3080, profile="gmlx", workdir=proj,
                        shares=[{"host": proj, "guest": proj, "readonly": False}],
                        project=proj, **record)


def test_a_second_launch_of_a_web_app_opens_the_running_one(env, capsys, monkeypatch):
    import webbrowser
    opened = []
    monkeypatch.setattr(webbrowser, "open", opened.append)
    lock = _web_session(env, "open-webui", web_port=3000)
    try:
        assert _run(["open-webui", "--container"]) == 0
    finally:
        lock.release()
    assert opened == ["http://127.0.0.1:3000/"] and not env.runs
    assert capsys.readouterr().out == ("[launch] open-webui is already running at "
                                       "http://127.0.0.1:3000/\n")


def test_a_second_dsh_launch_opens_the_recorded_token_url(env, capsys, monkeypatch):
    import webbrowser
    opened = []
    monkeypatch.setattr(webbrowser, "open", opened.append)
    lock = _dsh_session(env)
    try:
        assert _run(["dsh", "--container"]) == 0
        assert opened == [] and "has not printed its address yet" in capsys.readouterr().out
        record = session.read_record("dsh", env.project)
        session.write_record("dsh", env.project,
                             {**record, "url": "http://evil.example/?token=t"})
        assert _run(["dsh", "--container"]) == 0 and opened == []
        session.write_record("dsh", env.project,
                             {**record, "url": "http://127.0.0.1:3080/?token=t"})
        assert _run(["dsh", "--container"]) == 0
    finally:
        lock.release()
    assert opened == ["http://127.0.0.1:3080/?token=t"]


def test_dsh_keeps_a_home_and_volumes_per_project(env):
    _user_config(env.home, "launch:\n  container:\n    clients:\n      dsh:\n"
                           "        volumes: [\"tools:/tools\"]\n")
    assert _run(["dsh", "--container"]) == 0
    spec = env.runs[0]["spec"]
    assert spec.session.project == env.project and spec.web_port == 3080
    assert spec.plan.home == settings.private_home_path("dsh", env.project)
    assert [m.source for m in spec.plan.volumes] == [
        settings.project_volume_name("tools", env.project)]
    assert env.runs[0]["record"]["project"] == os.path.realpath(env.proj)


def test_dsh_runs_one_web_session_at_a_time(env, capsys):
    other = env.home / "src" / "other"
    other.mkdir()
    lock = _dsh_session(env)
    os.chdir(other)
    try:
        assert _run(["dsh", "--container"]) == launch.EXIT_TEMPFAIL
        assert capsys.readouterr().err == (
            "[launch] the dsh session for ~/src/proj is running, and dsh runs one web session "
            "at a time, because its web app has one port on the Mac. To open it, launch dsh "
            "from ~/src/proj. To start one here, end it first.\n")
        assert _run(["dsh", "--container", "--config-only"]) == 0
        assert _run(["dsh", "--container", "--dsh-profile", "headless"]) == 0
        assert len(env.runs) == 1 and env.runs[0]["spec"].web_port is None
        env.update(containers=[{"name": "gmlx-dsh-abc123", "labels": {
            "gmlx.launch": "1", "gmlx.launch.client": "dsh",
            "gmlx.launch.project": env.project, "gmlx.launch.pid": "999999"}}])
        assert _run(["dsh", "--container"]) == 0         # its launch is gone
    finally:
        lock.release()
    assert env.runs[1]["spec"].session.project == settings.project_id(os.path.realpath(other))


@pytest.mark.parametrize("client, where", [
    ("open-webui", "at http://127.0.0.1:3000/"), ("dsh", "at the address it prints")])
def test_a_second_launch_of_a_web_app_that_runs_a_shell_says_so(env, capsys, monkeypatch,
                                                                 client, where):
    import webbrowser
    opened = []
    monkeypatch.setattr(webbrowser, "open", opened.append)
    key = env.project if client == "dsh" else "default"
    proj = os.path.realpath(env.proj)
    shares = [{"host": proj, "guest": proj, "readonly": False}] if client == "dsh" else []
    lock = _web_session(env, client, key, web_port=3000 if client == "open-webui" else 3080,
                        shell=True, shares=shares, url="http://127.0.0.1:3080/?token=t")
    try:
        assert _run([client, "--container"]) == 0
    finally:
        lock.release()
    assert opened == [] and not env.runs and not env.copies
    assert capsys.readouterr().out == (
        f"[launch] the running {client} session runs a shell, so {client} answers only after "
        f"you start it in that shell, {where}. To open another shell in the session, run: "
        f"gmlx launch {client} --shell\n")


def test_a_dsh_launch_with_another_profile_is_refused(env, capsys):
    lock = _web_session(env, "dsh", web_port=3080, profile="gmlx")
    try:
        assert _run(["dsh", "--container", "--no-mount-cwd", "--dsh-profile", "headless"]) == 1
    finally:
        lock.release()
    assert ("a dsh session with the gmlx profile is already running, and a project runs one "
            "session at a time. End it to start the headless profile.") in capsys.readouterr().err


def test_a_shell_on_a_running_web_app_opens_a_shell(env):
    lock = _web_session(env, "open-webui", web_port=3000)
    calls = env.copies
    try:
        assert _run(["open-webui", "--shell"]) == 0
    finally:
        lock.release()
    assert calls[0][1][-4:] == [_ENTRY, "--join", "--shell", "--"]


def test_the_record_names_the_command_the_project_and_the_web_port(env):
    assert _run(["pi", "--container", "--", "--continue"]) == 0
    record = env.runs[0]["record"]
    assert record["command"] == env.runs[0]["spec"].command[:-1]
    assert env.runs[0]["spec"].command[-1] == "--continue"
    assert record["project"] == settings.canonical(str(env.proj))
    assert record["web"] is False and record["web_port"] is None and record["shell"] is False
    assert _run(["open-webui", "--container"]) == 0
    record = env.runs[1]["record"]
    assert record["web"] is True and record["web_port"] == 3000 and record["project"] is None
    assert _run(["open-webui", "--container", "--shell"]) == 0
    assert env.runs[2]["record"]["shell"] is True


def test_a_shell_session_records_the_clients_command(env):
    assert _run(["pi", "--container", "--shell"]) == 0
    record = env.runs[0]["record"]
    assert record["command"] and record["command"][0] == "pi"


# The private home of each project

def test_an_explicit_share_of_the_current_folder_keys_its_folder(env):
    """A home and volumes that one project's client writes never reach a
    session of another project through --mount ."""
    _user_config(env.home, "launch:\n  container:\n    mount_cwd: false\n")
    assert _run(["claude-code", "--container", "--mount", "."]) == 0
    assert env.runs[0]["spec"].session.project == env.project
    other = env.home / "src" / "other"
    other.mkdir()
    os.chdir(other)
    assert _run(["claude-code", "--container", "--mount", "."]) == 0
    assert env.runs[1]["spec"].session.project == settings.project_id(os.path.realpath(other))
    # A share that holds the current folder keys the folder of that share.
    assert _run(["claude-code", "--container", "--mount", "~/src:ro"]) == 0
    src = os.path.realpath(env.home / "src")
    assert env.runs[2]["spec"].session.project == settings.project_id(src)
    assert env.runs[2]["record"]["project"] == src
    # A launch that shares only other folders keys the default project.
    assert _run(["claude-code", "--container", "--mount", str(env.proj)]) == 0
    assert env.runs[3]["spec"].session.project == settings.PROJECT_DEFAULT
    assert env.runs[3]["record"]["project"] is None


def test_a_new_home_says_its_history_starts_empty_once(env, capsys):
    assert _run(["pi", "--container"]) == 0
    line = ("[launch] pi keeps its own history for this project in the container, starting "
            "empty. Its history on the Mac stays on the Mac.")
    assert line in capsys.readouterr().out
    assert _run(["pi", "--container"]) == 0
    assert "keeps its own history" not in capsys.readouterr().out
    assert _run(["elia", "--container"]) == 0
    assert ("[launch] elia keeps its own history in the container, starting empty."
            in capsys.readouterr().out)


def test_a_claude_code_home_starts_ready(env):
    (env.home / ".claude.json").write_text(json.dumps({"theme": "dark", "userID": "u"}))
    assert _run(["claude-code", "--container"]) == 0
    doc = json.loads((settings.private_home_path("claude-code", env.project)
                      / ".claude.json").read_text())
    assert doc == {"hasCompletedOnboarding": True, "theme": "dark"}


def test_client_volumes_get_a_name_per_project(env):
    _user_config(env.home, "launch:\n  container:\n    volumes: [\"shared:/shared\"]\n"
                           "    clients:\n      pi:\n        volumes: [\"pg:/pg\"]\n"
                           "      elia:\n        volumes: [\"pg:/pg\"]\n")
    assert _run(["pi", "--container"]) == 0
    names = {m.target: m.source for m in env.runs[0]["spec"].plan.volumes}
    assert names == {"/shared": "shared", "/pg": settings.project_volume_name("pg", env.project)}
    assert _run(["elia", "--container"]) == 0            # the default project keeps the name
    assert {m.source for m in env.runs[1]["spec"].plan.volumes} == {"shared", "pg"}


def test_the_memory_of_every_launch_container_is_named(env, capsys, monkeypatch):
    monkeypatch.setattr(session, "mac_memory_bytes", lambda: 64 << 30)
    env.update(containers=[{"name": "gmlx-omp-1", "memory": 8 << 30, "labels": {
        "gmlx.launch": "1", "gmlx.launch.client": "omp", "gmlx.launch.project": "default",
        "gmlx.launch.pid": str(os.getpid())}}])
    assert _run(["pi", "--container"]) == 0
    # The two virtual machines hold 128M each on top of 8G and the 4G default.
    assert ("[launch] with 1 other launch container running, launch containers will hold "
            "12.2G of the Mac's 64G of memory") in capsys.readouterr().out


# Removing a private home

def test_remove_home_without_a_terminal_prints_the_command(env, capsys):
    settings.private_home("pi", env.project)
    assert _run(["pi", "--remove-home"]) == 1
    err = capsys.readouterr().err
    folder = settings.project_dir_path("pi", env.project)
    assert f"Remove the home yourself with: rm -rf {folder}" in err
    assert folder.is_dir()


def test_remove_home_asks_and_removes_only_this_projects_home(env, capsys, monkeypatch):
    monkeypatch.setattr(session, "stdin_is_terminal", lambda: True)
    home = settings.private_home("pi", env.project)
    (home / "notes").write_text("n")
    outside = env.home / "keep"
    outside.mkdir()
    (home / "link").symlink_to(outside, target_is_directory=True)
    other = settings.private_home("pi", "other-12345678")
    answers = iter(["n", "y"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    assert _run(["pi", "--remove-home"]) == 1
    assert "nothing was removed" in capsys.readouterr().out and home.is_dir()
    assert _run(["pi", "--remove-home"]) == 0
    assert not settings.project_dir_path("pi", env.project).exists()
    assert other.is_dir() and outside.is_dir()
    assert _run(["pi", "--remove-home"]) == 0
    assert "has no private home for ~/src/proj" in capsys.readouterr().out


def test_remove_home_names_the_default_project(env, capsys):
    assert _run(["elia", "--remove-home"]) == 0
    assert capsys.readouterr().out == ("[launch] elia has no private home for the default "
                                       "project, so nothing was removed.\n")


def test_remove_home_finds_the_project_of_an_explicit_share(env, capsys):
    settings.private_home("pi", env.project)
    assert _run(["pi", "--remove-home", "--no-mount-cwd", "--mount", "."]) == 1
    folder = settings.project_dir_path("pi", env.project)
    assert (f"Remove the home yourself with: rm -rf {folder}"
            in capsys.readouterr().err)


def test_remove_home_takes_ctrl_d_as_no(env, capsys, monkeypatch):
    monkeypatch.setattr(session, "stdin_is_terminal", lambda: True)
    home = settings.private_home("pi", env.project)

    def eof(prompt):
        raise EOFError
    monkeypatch.setattr("builtins.input", eof)
    assert _run(["pi", "--remove-home"]) == 1
    assert capsys.readouterr().out == "\n[launch] nothing was removed.\n"
    assert home.is_dir()


def test_remove_home_refuses_while_the_session_runs(running_session, capsys):
    settings.private_home("pi", running_session.project)
    assert _run(["pi", "--remove-home"]) == launch.EXIT_TEMPFAIL
    assert "the pi session for ~/src/proj is running" in capsys.readouterr().err


def test_remove_home_starts_nothing(env, capsys):
    assert _run(["pi", "--remove-home", "--shell"]) == 1
    assert "cannot go with --shell" in capsys.readouterr().err


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
    assert _run([client, "--container"]) == 0
    home = settings.private_home_path(client, _project(env, client))
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
    cfg = settings.private_home_path("hermes", env.project) / ".hermes" / "config.yaml"
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
    assert _run(["pi", "--container"]) == launch.EXIT_CONFIG
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
    first = run["server_session"].id
    run["server_session"].renew()
    body = {"client": client, "assistants": [], "web_ports": [port],
            "project": run["server_session"].project}
    assert [body for _, body, _ in env.server.posts[-2:]] == [
        body, {**body, "replaces": first}]


def test_a_terminal_client_session_names_no_pages(env):
    assert _run(["pi", "--container"]) == 0
    server_session = env.runs[0]["server_session"]
    assert server_session.web_ports == []
    server_session.open()
    # The server keeps the prompt cache of each client and project apart.
    project = settings.project_id(settings.canonical(os.getcwd()))
    assert env.server.posts[-1][1] == {"client": "pi", "assistants": [], "project": project}


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
    defaults = {**lc._JOIN_IGNORED, **lc._JOIN_REFUSED}
    for dest, default in defaults.items():
        assert getattr(a, dest) == default, dest
    rest = set(vars(a)) - set(defaults) - {"harness", "container", "shell", "passthrough",
                                           "remove_home", "mount_cwd", "dsh_profile"}
    assert rest == set(), rest


# First-run steps, the builder notice and the dry run's image

def test_steps_are_numbered_only_when_they_run(env, capsys):
    assert _run(["pi", "--container"]) == 0            # a first build on a running service
    out = capsys.readouterr().out
    assert "[launch] step 1 of 2: building the pi image" in out
    # The last step line comes before the lines the client's setup prints.
    assert out.index("[launch] step 2 of 2: starting pi\n") < out.index("[launch] pi -> ")
    assert _run(["pi", "--container"]) == 0            # the image exists now
    assert "step " not in capsys.readouterr().out
    assert not any(line.startswith("[launch] step") for line in env.runs[-1]["summary"]), env.runs[-1]["summary"]


def test_the_last_step_line_prints_before_the_session_summary(env, capsys, monkeypatch):
    """supervise prints the summary when the session starts, so the step
    line that names the client must already be out by then."""
    printed = []

    def supervise(spec, *, say=print, summary=(), **kw):
        printed.append(capsys.readouterr().out)
        for line in summary:
            say(line)
        return 0
    monkeypatch.setattr(session, "supervise", supervise)
    assert _run(["pi", "--container"]) == 0
    before, after = printed[0], capsys.readouterr().out
    assert "[launch] step 2 of 2: starting pi\n" in before
    assert after.startswith("[launch] ") and "step " not in after, after


def _kernels(home):
    return home / "Library" / "Application Support" / "com.apple.container" / "kernels"


def _install_kernel(home):
    """The kernel that the first start of the container service installs."""
    kernels = _kernels(home)
    kernels.mkdir(parents=True)
    (kernels / "vmlinux-6.18").write_bytes(b"kernel")
    (kernels / "default.kernel-arm64").symlink_to(kernels / "vmlinux-6.18")


def _remove_kernel(home):
    shutil.rmtree(_kernels(home))


def test_the_first_service_start_names_the_kernel_download(env, capsys, monkeypatch):
    _remove_kernel(env.home)
    env.update(running=False)
    monkeypatch.setattr(session, "stdin_is_tty", lambda: True)
    assert _run(["pi", "--container"]) == 0
    steps = [line for line in capsys.readouterr().out.splitlines() if " step " in line]
    assert steps == [
        "[launch] step 1 of 3: starting the container service. Its first start asks to "
        "install a Linux kernel, which downloads about 700 MB once.",
        "[launch] step 2 of 3: building the pi image, which takes a few minutes. Later "
        "launches reuse it.",
        "[launch] step 3 of 3: starting pi"]
    assert env.calls("system", "start") == [["system", "start"]]


def test_a_first_service_start_with_the_image_ready_keeps_three_steps(env, capsys,
                                                                       monkeypatch):
    assert _run(["pi", "--container"]) == 0
    capsys.readouterr()
    _remove_kernel(env.home)
    env.update(running=False)
    monkeypatch.setattr(session, "stdin_is_tty", lambda: True)
    assert _run(["pi", "--container"]) == 0
    out = capsys.readouterr().out
    assert "[launch] step 2 of 3: found gmlx.invalid/launch-pi:" in out
    assert "[launch] step 3 of 3: starting pi\n" in out


def test_the_summary_names_network_none(env):
    assert _run(["pi", "--container", "--network", "none"]) == 0
    assert ("[launch] with network none, the client reaches only the gmlx server and the "
            "forwarded ports, and a download such as npm install fails") in env.runs[-1]["summary"]
    assert _run(["pi", "--container"]) == 0
    assert not any("network none" in line for line in env.runs[-1]["summary"])


@pytest.mark.parametrize("tty", [True, False])
def test_a_restarted_service_starts_without_the_first_run_text(env, capsys, monkeypatch, tty):
    assert _run(["pi", "--container"]) == 0
    capsys.readouterr()
    env.update(running=False)
    monkeypatch.setattr(session, "stdin_is_tty", lambda: tty)
    assert _run(["pi", "--container"]) == 0
    out = capsys.readouterr().out
    assert "[launch] starting the container service\n" in out
    assert "700 MB" not in out and "step " not in out
    assert env.calls("system", "start") == [["system", "start", "--disable-kernel-install"]]
    assert not any(line.startswith("[launch] step") for line in env.runs[-1]["summary"])


def test_a_first_service_start_without_a_terminal_names_the_command(env, capsys):
    _remove_kernel(env.home)
    env.update(running=False)
    assert _run(["pi", "--container"]) == launch.EXIT_UNAVAILABLE
    assert capsys.readouterr().err == (
        "[launch] the container service is not running, and its first start asks whether to "
        "install a Linux kernel. Run it once in a terminal with: container system start\n")
    assert not env.calls("system", "start")


def test_an_old_container_names_both_upgrade_routes(env, capsys):
    env.update(version="1.4.1")
    assert _run(["pi", "--container"]) == launch.EXIT_UNAVAILABLE
    assert capsys.readouterr().err == (
        "[launch] container mode needs Apple container 1.5.0 or newer, and this Mac has "
        "1.4.1. Upgrade with: brew upgrade container, or install the newer release from "
        "https://github.com/apple/container/releases.\n")


_OPEN_BIND = ("[launch] warning: the server at http://0.0.0.0:8080/v1 listens on more than "
              "the loopback address and needs no key. The container can reach every route "
              "of the server at the Mac's address on the container network. Set "
              "server.api_key in the server's config and restart the server.")


@pytest.mark.parametrize("host, keyed, warned", [
    ("0.0.0.0", False, True), ("0.0.0.0", True, False), ("127.0.0.1", False, False)])
def test_a_keyless_server_beyond_loopback_is_named(env, monkeypatch, host, keyed, warned):
    monkeypatch.setattr(lifecycle, "auto_target", lambda h, p: (host, 8080))
    monkeypatch.setattr(launch, "_auth_required", lambda base: keyed)
    assert _run(["pi", "--container"]) == 0
    summary = env.runs[-1]["summary"]
    assert (_OPEN_BIND in summary) is warned
    assert env.runs[-1]["server_session"] is not None


@pytest.mark.parametrize("base, targets, open_", [
    ("http://0.0.0.0:8080/v1", [("127.0.0.1", 8080)], True),
    ("http://[::]:8080/v1", [("::1", 8080)], True),
    ("http://192.168.1.5:8080/v1", [("192.168.1.5", 8080)], True),
    ("http://mac.local:8080/v1", [("192.168.1.5", 8080)], True),
    ("http://127.0.0.1:8080/v1", [("127.0.0.1", 8080)], False),
    ("http://127.1:8080/v1", [("127.0.0.1", 8080)], False),
    ("http://localhost:8080/v1", [("127.0.0.1", 8080), ("::1", 8080)], False),
    ("http://[::ffff:127.0.0.1]:8080/v1", [("::ffff:127.0.0.1", 8080)], False)])
def test_open_bind(base, targets, open_):
    assert lc.open_bind(base, targets) is open_


_NO_KERNEL = ("[launch] Apple container has no Linux kernel, so no container can start. "
              "Install it with: container system kernel set --recommended\n")


@pytest.mark.parametrize("dry", [False, True])
def test_a_running_service_without_a_kernel_is_refused(env, capsys, dry):
    _remove_kernel(env.home)
    assert _run(["pi", "--container", *(["--config-only"] if dry else [])]) == \
        launch.EXIT_UNAVAILABLE
    assert capsys.readouterr().err == _NO_KERNEL
    assert not env.calls("build") and not env.runs


def test_a_declined_kernel_is_refused_before_the_build(env, capsys, monkeypatch):
    _remove_kernel(env.home)
    env.update(running=False, kernel_answer="n")
    monkeypatch.setattr(session, "stdin_is_tty", lambda: True)
    assert _run(["pi", "--container"]) == launch.EXIT_UNAVAILABLE
    assert capsys.readouterr().err == _NO_KERNEL
    assert not env.calls("build")


def test_a_failed_first_start_names_the_kernel_command(env, capsys, monkeypatch):
    _remove_kernel(env.home)
    env.update(running=False, start_rc=1)
    monkeypatch.setattr(session, "stdin_is_tty", lambda: True)
    assert _run(["pi", "--container"]) == launch.EXIT_UNAVAILABLE
    assert capsys.readouterr().err == (
        "[launch] `container system start` failed (exit 1). " + _NO_KERNEL.removeprefix(
            "[launch] "))


def test_a_warm_launch_repeats_no_container_query(env):
    _user_config(env.home, "launch:\n  container:\n    clients:\n      pi:\n"
                           "        volumes: [cache:/root/.cache]\n")
    assert _run(["pi", "--container"]) == 0
    env.update(log=[])
    assert _run(["pi", "--container"]) == 0
    log = [" ".join(a[:2]) for a in env.log]
    assert log == ["--version", "system status", "ls --all", "image inspect",
                   "builder status", "volume list", "image inspect"], log
    inspected = [a[2] for a in env.calls("image", "inspect")]
    assert inspected[0].startswith("gmlx.invalid/launch-pi@sha256:")
    assert inspected[1] == "gmlx.invalid/launch-pi:base"


def test_notes_that_matter_once_print_once(env, capsys, monkeypatch):
    docs = env.home / "Documents" / "proj"
    docs.mkdir(parents=True)
    monkeypatch.chdir(docs)
    monkeypatch.setattr(lc.settings, "MEMORY_WARN_FRACTION", 0.0)
    assert _run(["pi", "--container", "--config-only"]) == 0     # a dry run records nothing
    out = capsys.readouterr().out
    assert "macOS guards" in out and "the container gets 4G" in out
    assert _run(["pi", "--container"]) == 0
    out = capsys.readouterr().out
    assert "macOS guards" in out and "the container gets 4G" in out
    assert _run(["pi", "--container"]) == 0
    out = capsys.readouterr().out
    assert "macOS guards" not in out and "the container gets" not in out
    _user_config(env.home, "launch:\n  container:\n    memory: 6G\n")
    assert _run(["pi", "--container"]) == 0
    assert "the container gets 6G" in capsys.readouterr().out


def test_a_launch_that_stops_before_container_run_uses_up_no_once_line(env, capsys,
                                                                       monkeypatch):
    """The first launch that reaches the macOS privacy question must still
    show the guarded-folder notice, and the history line of a new home."""
    docs = env.home / "Documents" / "proj"
    docs.mkdir(parents=True)
    monkeypatch.chdir(docs)
    env.update(fail_build="no network")
    assert _run(["pi", "--container"]) != 0
    out = capsys.readouterr().out
    assert "macOS guards" in out and "keeps its own history for this project" in out
    env.update(fail_build=False)
    assert _run(["pi", "--container"]) == 0
    out = capsys.readouterr().out
    assert "macOS guards" in out and "keeps its own history for this project" in out
    assert _run(["pi", "--container"]) == 0
    out = capsys.readouterr().out
    assert "macOS guards" not in out and "keeps its own history" not in out


def test_the_image_age_note_prints_once_a_day(env, monkeypatch):
    from gmlx.container import images, notices
    assert _run(["pi", "--container"]) == 0
    pins = json.loads(images._pins_path().read_text())
    for note in pins.values():
        note["at"] -= 40 * 86400
    images._pins_path().write_text(json.dumps(pins))
    assert _run(["pi", "--container"]) == 0
    assert any("launch built this image 40 days ago" in line
               for line in env.runs[-1]["summary"])
    assert _run(["pi", "--container"]) == 0
    assert not any("days ago." in line for line in env.runs[-1]["summary"])
    later = notices.time.time() + notices.DAY
    monkeypatch.setattr(notices.time, "time", lambda: later)
    assert _run(["pi", "--container"]) == 0
    assert any("launch built this image" in line for line in env.runs[-1]["summary"])


def test_a_volume_size_warning_prints_once_for_each_size(env, capsys):
    env.update(volumes=[{"name": "cache", "labels": {"gmlx.launch": "1"}, "size": "8G",
                         "bytes": 8 << 30}])
    # A volume listed for every client keeps its name in each project.
    _user_config(env.home, "launch:\n  container:\n    volumes: [cache:/root/.cache]\n")
    assert _run(["pi", "--container"]) == 0
    assert "the volume cache has 8G, not the configured 32G" in capsys.readouterr().out
    assert _run(["pi", "--container"]) == 0
    assert "the volume cache has" not in capsys.readouterr().out
    _user_config(env.home, "launch:\n  container:\n    volumes: [cache:/root/.cache:16G]\n")
    assert _run(["pi", "--container"]) == 0
    assert "the volume cache has 8G, not the configured 16G" in capsys.readouterr().out


def test_a_launch_deletes_the_image_the_config_no_longer_names(env):
    env.update(registry={"me/box:1": {"digest": "sha256:" + "1" * 64},
                         "me/box:2": {"digest": "sha256:" + "2" * 64}})
    _user_config(env.home, "launch:\n  container:\n    clients:\n      pi:\n"
                           "        image: me/box:1\n        command: [bash]\n")
    assert _run(["pi", "--container"]) == 0
    assert "docker.io/me/box:1" in env.load()["images"]
    _user_config(env.home, "launch:\n  container:\n    clients:\n      pi:\n"
                           "        image: me/box:2\n        command: [bash]\n")
    assert _run(["pi", "--container"]) == 0
    store = env.load()["images"]
    assert "docker.io/me/box:1" not in store and "docker.io/me/box:2" in store


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
    assert _run(["pi", "--container", "--port", "9999"]) == launch.EXIT_CONFIG
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


@pytest.mark.parametrize("signum", [signal.SIGHUP, signal.SIGTERM])
def test_step_8_leaves_a_signal_ignored_on_entry(signum):
    """nohup leaves SIGHUP ignored, so the image step goes on through it."""
    saved = signal.signal(signum, signal.SIG_IGN)
    try:
        with lc._signals_raise():
            os.kill(os.getpid(), signum)
            assert signal.getsignal(signum) == signal.SIG_IGN
        assert signal.getsignal(signum) == signal.SIG_IGN
    finally:
        signal.signal(signum, saved)


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
    gitdir = settings.private_home("pi", env.project) / evil
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
    assert "the omp build: folder" in err and not env.runs


def test_a_launch_records_its_read_write_shares(env):
    from gmlx.container import settings
    assert _run(["pi", "--container"]) == 0
    assert os.path.realpath(env.proj) in settings.shared_history()


def test_the_dry_run_records_no_shares(env):
    from gmlx.container import settings
    assert _run(["pi", "--container", "--config-only"]) == 0
    assert settings.shared_history() == []


# gmlx's own state while a handler runs with HOME in the private home

def _planted_state_links(env, monkeypatch, client, tmp_path):
    """Leave gmlx's cache and data folders at their defaults in HOME, as on
    a Mac that sets neither, and plant guest links to ``.cache`` and
    ``.local`` in the private home. Returns the folders the links name."""
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    home = settings.private_home_path(client, _project(env, client))
    home.mkdir(parents=True)
    targets = []
    for name in (".cache", ".local"):
        target = tmp_path / f"planted{name}"
        target.mkdir()
        (home / name).symlink_to(target)
        targets.append(target)
    return targets


def test_the_claude_code_handler_reads_the_server_config_on_the_mac(
        env, monkeypatch, tmp_path, capsys):
    planted = _planted_state_links(env, monkeypatch, "claude-code", tmp_path)
    served = tmp_path / "served.yaml"
    served.write_text("server:\n  cache: {enabled: false}\n")
    mac_run = env.home / ".cache" / "gmlx" / "run.json"
    mac_run.parent.mkdir(parents=True)
    mac_run.write_text(json.dumps({"pid": os.getpid(), "config_abspath": str(served)}))

    def read_run(host, port):
        # The runfile is found where gmlx's cache folder resolves now.
        path = lifecycle.runtime_dir() / "run.json"
        return json.loads(path.read_text()) if path.is_file() else None
    monkeypatch.setattr(lifecycle, "read_run", read_run)
    assert _run(["claude-code", "--container"]) == 0
    assert "[launch] the server's prompt cache is off" in capsys.readouterr().out
    assert [list(p.iterdir()) for p in planted] == [[], []]


def test_the_aichat_note_is_recorded_on_the_mac(env, monkeypatch, tmp_path, capsys):
    planted = _planted_state_links(env, monkeypatch, "aichat", tmp_path)
    assert _run(["aichat", "--container"]) == 0
    assert _run(["aichat", "--container"]) == 0
    assert capsys.readouterr().out.count("llm-functions") == 1
    assert (env.home / ".local/share/gmlx/launch/notices.json").is_file()
    assert [list(p.iterdir()) for p in planted] == [[], []]


def test_the_no_models_refusal_reads_no_runfile_in_the_private_home(env, monkeypatch):
    homes = []
    monkeypatch.setattr(lifecycle, "read_run",
                        lambda h, p: homes.append(os.environ["HOME"]) or None)
    probes = []

    def get(url, timeout=5.0, headers=None):
        # The server loses its models after launch's own probe, so the
        # handler's probe gives the refusal.
        if not url.endswith("/models"):
            return {}
        probes.append(url)
        return {"data": MODELS if len(probes) == 1 else []}
    monkeypatch.setattr(launch, "_http_get_json", get)
    assert _run(["pi", "--container"]) == launch.EXIT_UNAVAILABLE
    assert len(probes) == 2
    assert homes and set(homes) == {str(env.home)}
