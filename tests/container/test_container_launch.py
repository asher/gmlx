"""`gmlx launch --container`: the launch order, the dry run, the attaching
shell and the per-client container settings. The server probe is faked, the
``container`` command is the fake from conftest, and the supervisor is a
recording stand-in, so no VM runs."""

from __future__ import annotations

import ast
import json
import os
import re
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import urllib.parse
from pathlib import Path
from types import SimpleNamespace

import pytest

import gmlx.commands.launch as launch
import gmlx.commands.launch_container as lc
import gmlx.serve.lifecycle as lifecycle
from gmlx.config import AGENT_RUN_SCRIPT, LAUNCH_CLIENTS, LaunchClientCfg
from gmlx.container import cli, runtime, session, settings, web_ports
from gmlx.serve import procname

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
    # The Mac ports of the web apps that other programs use. No test binds
    # a port of the range, which a program on this Mac may use.
    busy: set[int] = set()
    monkeypatch.setattr(web_ports, "_free", lambda port: port not in busy)
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
    fake_container.busy_ports = busy
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


def test_a_launch_builds_the_share_tables_once(env, monkeypatch):
    """The current folder, an explicit share and a seed of one launch use
    one build of the tables, and each folder gets one check. The next
    launch builds them again."""
    (env.home / ".foorc").write_text("mac v1\n")
    (env.home / "notes").mkdir()
    _user_config(env.home, "launch:\n  container:\n    clients:\n      pi:\n"
                           "        seed: [\"~/.foorc\"]\n")
    builds, checks = [], []
    written, check = settings._sensitive_written, settings._table_refusal

    def counted_written(home, *rest):
        builds.append(home)
        return written(home, *rest)

    def counted_check(path, home, tables):
        checks.append(path)
        return check(path, home, tables)
    monkeypatch.setattr(settings, "_sensitive_written", counted_written)
    monkeypatch.setattr(settings, "_table_refusal", counted_check)
    assert _run(["pi", "--container", "--mount", str(env.home / "notes")]) == 0
    assert len(builds) == 1
    assert checks and len(checks) == len(set(checks))
    builds.clear()
    assert _run(["pi", "--container", "--config-only"]) == 0
    assert len(builds) == 1


def test_a_launch_asks_xcode_select_once(env, monkeypatch, tmp_path):
    """For /usr/bin/git, launch runs the git of the developer folder. One
    launch asks xcode-select for that folder once, for the share checks
    and every git run, and the next launch asks again."""
    ran = tmp_path / "xcode-select-runs"
    dev = tmp_path / "Developer"
    (dev / "usr" / "bin").mkdir(parents=True)
    git = dev / "usr" / "bin" / "git"
    git.write_text('#!/bin/sh\n[ "$1" = --version ] && exit 0\nexit 1\n')
    git.chmod(0o755)
    xcode = tmp_path / "xcode-select"
    xcode.write_text(f"#!/bin/sh\necho run >> {ran}\necho {dev}\n")
    xcode.chmod(0o755)
    monkeypatch.setattr(settings, "XCODE_SELECT", str(xcode))
    found = settings._system_program
    monkeypatch.setattr(settings, "_system_program",
                        lambda name: "/usr/bin/git" if name == "git" else found(name))
    assert _run(["pi", "--container"]) == 0
    assert ran.read_text().splitlines() == ["run"]
    assert _run(["pi", "--container", "--config-only"]) == 0
    assert ran.read_text().splitlines() == ["run", "run"]


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
    assert ("[launch] launch found the container command at ~/src/proj/.venv/bin/container, "
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



@pytest.mark.parametrize("dry", [False, True], ids=["step-6", "step-9"])
def test_a_server_launch_starts_gets_no_path_entry_a_client_can_write(env, monkeypatch, dry):
    """The server runs programs by name, such as ffmpeg for an audio request
    of the client, and the menu bar starts the server again with its own
    PATH. A launch starts the server in step 6, and a dry run in step 9."""
    state = {"up": False}
    monkeypatch.setattr(launch, "_server_ready", lambda base, api_key=None: state["up"])
    config = env.home / ".config" / "gmlx" / "gmlx.yaml"
    _user_config(env.home, "server:\n  port: 8080\n")

    class Cfg:
        host, port, api_key, menubar = "127.0.0.1", 8080, None, True
    monkeypatch.setattr(launch, "_discover_config", lambda: (Cfg(), str(config)))
    monkeypatch.setattr(launch, "_preload_descr", lambda cfg: (None, None))
    seen = {}

    class Proc:
        pid, returncode = 4242, None

        def poll(self):
            return None

    def spawn(serve_args, **kw):
        seen["server"] = procname.child_env()["PATH"]
        state["up"] = True
        return Proc(), env.home / "server.log"

    def bar(**kw):
        seen["menubar"] = procname.child_env()["PATH"]
        return 0
    monkeypatch.setattr(lifecycle, "start_background_nowait", spawn)
    monkeypatch.setattr(lifecycle, "start_menubar", bar)
    monkeypatch.setattr(lifecycle, "gui_session_available", lambda: True)
    venv = env.proj / ".venv" / "bin"
    venv.mkdir(parents=True)
    link = env.home / "link-bin"
    link.symlink_to(venv)
    tools = env.home / "tools"
    tools.mkdir()
    given = os.environ["PATH"]
    path = os.pathsep.join([str(venv), str(link), "", "bin", str(tools), given])
    monkeypatch.setenv("PATH", path)
    assert _run(["pi", "--container", *(["--config-only"] if dry else [])]) == 0
    want = os.pathsep.join([str(tools), given])
    assert seen == ({"server": want} if dry else {"server": want, "menubar": want})
    assert os.environ["PATH"] == path and procname.child_env()["PATH"] == path


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
    assert capsys.readouterr().err.endswith(
        "and --config-only applies only to a new session. End the session, then launch "
        "again with --config-only.\n")


def test_dry_run_prints_the_replaced_command(env, capsys):
    _user_config(env.home, "launch:\n  container:\n    clients:\n      elia:\n"
                           "        command: [/usr/local/bin/start.sh, elia]\n")
    assert _run(["elia", "--container", "--config-only"]) == 0
    out = capsys.readouterr().out
    assert "replaces the client's own command, elia -m gmlx/qwen3.6-27b" in out
    assert out.rstrip().endswith("-- /usr/local/bin/start.sh elia")


def test_missing_entry_fails_step_3(env, capsys, monkeypatch):
    """The guest entry has a name of its own, apart from Apple's container
    command, and a path in your home folder starts with ~."""
    monkeypatch.setattr(runtime, "entry_path", lambda: Path(os.path.realpath(env.home)) / "nope")
    assert _run(["pi", "--container"]) == launch.EXIT_UNAVAILABLE
    assert capsys.readouterr().err == (
        "[launch] the guest entry ~/nope is not built. In a git checkout, build it with: "
        "python scripts/build_guest_entry.py\n")
    assert _run(["pi", "--container", "--config-only"]) == 0
    assert ("[launch] the guest entry ~/nope is not built. Build it with: python "
            "scripts/build_guest_entry.py\n") in capsys.readouterr().out


def test_an_old_container_in_your_home_folder_is_named_with_a_tilde(env, capsys,
                                                                    monkeypatch):
    env.update(version="1.4.1")
    folder = Path(os.path.realpath(env.home)) / "bin"
    folder.mkdir()
    shutil.copy2(shutil.which("container") or "", folder / "container")
    _package_scripts(folder)
    monkeypatch.setenv("PATH", f"{folder}:/usr/bin:/bin")
    assert _run(["pi", "--container"]) == launch.EXIT_UNAVAILABLE
    assert capsys.readouterr().err == (
        "[launch] container mode needs Apple container 1.5.0 or newer, and "
        "~/bin/container, the first container command on PATH, is version 1.4.1. Stop "
        "the container service with: container system stop. Then upgrade Apple container "
        "with: ~/bin/update-container.sh\n")


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
    assert spec.web_port == 3100
    assert spec.command[:6] == ["open-webui", "serve", "--host", "127.0.0.1", "--port", "3100"]
    assert spec.env_values["HOST"] == "127.0.0.1" and spec.env_values["PORT"] == "3100"
    assert "PORT" not in spec.env_names
    assert not spec.plan.cwd_shared                      # no share by default
    assert spec.child_env["DATA_DIR"].startswith(str(spec.plan.home))
    assert os.path.isdir(spec.child_env["DATA_DIR"])      # the official image needs it


@pytest.mark.parametrize("entry, gets", [(None, "http://[::1]:3100"),
                                         ("CORS_ALLOW_ORIGIN=https://webui.example",
                                          "https://webui.example")])
def test_open_webui_in_a_container_takes_calls_only_from_its_own_pages(
        env, monkeypatch, entry, gets):
    """Open WebUI lets every page read its answers with the sign-in cookie
    and make a Function, which runs Python, unless CORS_ALLOW_ORIGIN names
    its own address. An env entry of the config wins, and an exported value
    does not reach the guest."""
    monkeypatch.setenv("CORS_ALLOW_ORIGIN", "*")
    if entry:
        _user_config(env.home, "launch:\n  container:\n    clients:\n      open-webui:\n"
                               f"        env: [\"{entry}\"]\n")
    assert _run(["open-webui", "--container"]) == 0
    spec = env.runs[0]["spec"]
    assert spec.web_port == 3100
    assert spec.child_env["CORS_ALLOW_ORIGIN"] == gets
    assert "CORS_ALLOW_ORIGIN" in spec.env_names


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
    assert spec.env_values["PORT"] == "3100"
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
    assert spec.web_port == 3100
    assert spec.command[-3:] == ["--no-open", "--port", "3100"]
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
    assert env.runs[-1]["spec"].web_port == 3100


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
     b'socket: no session socket folder gives a path shorter than 104 bytes. Set '
     b'TMPDIR to a shorter path."}}',
     "(503): cannot open a launch session socket: no session socket folder gives a "
     "path shorter than 104 bytes. Set TMPDIR to a shorter path. Its log may say more"),
    (b"<html>", "(503): refused"), (None, "(503): refused")])
def test_a_server_that_cannot_open_a_socket_shows_its_message(env, capsys, body, shown):
    env.server.status, env.server.body = 503, body
    assert _run(["pi", "--container"]) == 1
    err = capsys.readouterr().err
    assert "could not open a session socket " + shown in err
    assert "gmlx restart" not in err and not env.runs


def test_a_server_that_holds_its_most_sessions_exits_to_try_again(env, capsys):
    """Only the session limit clears by itself, so only it exits 75."""
    env.server.status = 503
    env.server.body = (b'{"error": {"type": "server_overloaded", "message": "32 launch '
                       b'sessions are open, and each one has an open connection. Wait '
                       b'for a request to end, or stop another launch, then launch '
                       b'again."}}')
    assert _run(["pi", "--container"]) == launch.EXIT_TEMPFAIL
    err = capsys.readouterr().err
    assert ("could not open a session (503): 32 launch sessions are open, and each one "
            "has an open connection. Wait for a request to end, or stop another launch, "
            "then launch again.\n") in err
    assert "Its log may say more" not in err and not env.runs


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


def _claude_code_session(env, record: dict, state: str | None) -> None:
    """A claude-code session over the project folder of ``env``. ``state``
    is that of its container, or None when container ls does not list it."""
    proj = os.path.realpath(env.proj)
    project = settings.project_id(proj)
    session.write_record("claude-code", project, {
        "name": "gmlx-claude-code-abc123", "workdir": proj, "clipboard": False,
        "shares": [{"host": proj, "guest": proj, "readonly": False}], "project": proj,
        **session.launch_owner(), **record})
    env.update(containers=[] if state is None else [{
        "name": "gmlx-claude-code-abc123", "state": state, "labels": {
            "gmlx.launch": "1", "gmlx.launch.client": "claude-code",
            "gmlx.launch.project": project, "gmlx.launch.pid": str(os.getpid())}}])


def _overlap(capsys) -> list[str]:
    return [line for line in capsys.readouterr().out.splitlines() if "shares files" in line]


@pytest.mark.parametrize(("record", "state", "named"), [
    ({"name": "", "starting": True}, None, "the starting"),     # image step or service start
    ({}, "stopped", "the starting"),                             # the virtual machine boots
    ({}, None, "the starting"),
    ({}, "running", "the running"),
    ({"ending": True}, "running", "the running"),                # the container stops now
    ({"ending": True}, "stopped", None),
    ({"pid": 999999, "pid_start": 1}, None, None),               # a killed launch
    ({"pid": 999999, "pid_start": 1}, "running", "the running"),  # its leftover container
])
def test_the_overlap_warning_names_another_session_that_starts_or_runs(env, capsys, record,
                                                                       state, named):
    _claude_code_session(env, record, state)
    assert _run(["pi", "--container"]) == 0
    assert env.runs[0]["spec"].session.project == env.project
    lines = _overlap(capsys)
    if named is None:
        assert not lines
    else:
        assert lines == [f"[launch] {named} claude-code session for ~/src/proj shares files "
                         "with this session. File locks do not reach from one virtual machine "
                         "to another, so do not let two clients change the same file at once."]


def test_the_overlap_warning_counts_a_starting_session_while_container_ls_fails(
        env, capsys, monkeypatch):
    """Only the marks tell the state then. A live launch with no mark can
    run or boot its session, so the line names no state."""
    from gmlx.container import cli

    def down():
        raise cli.ContainerError("`container ls --all --format` failed (exit 1): XPC error.")
    monkeypatch.setattr(cli, "list_launch_containers", down)
    for record, named in (({"name": "", "starting": True}, "the starting claude-code"),
                          ({}, "the claude-code"), ({"ending": True}, None)):
        _claude_code_session(env, record, None)
        assert _run(["pi", "--container", "--config-only"]) == 0
        lines = _overlap(capsys)
        assert lines == ([] if named is None else [
            f"[launch] {named} session for ~/src/proj shares files with this session. File "
            "locks do not reach from one virtual machine to another, so do not let two "
            "clients change the same file at once."])


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
    monkeypatch.setattr(session, "open_in_browser", lambda url: True)
    data = env.home / "data"
    data.mkdir()
    data = os.path.realpath(data)
    lock = _web_session(env, "open-webui", web_port=3100,
                        shares=[{"host": data, "guest": "/data", "readonly": True}])
    try:
        assert _run(["open-webui", "--container", "--mount-cwd"]) == 0
    finally:
        lock.release()
    assert capsys.readouterr().out == (
        "[launch] --mount-cwd applies only to a new session, so this launch ignores it.\n"
        "[launch] open-webui is already running at http://[::1]:3100/\n"
        "[launch] the current folder is not shared with this session, which shares ~/data "
        "(read-only).\n")


def test_shell_attach_refuses_new_session_flags(running_session, capsys):
    assert _run(["pi", "--shell", "--mount", "/tmp"]) == 1
    assert "--mount applies only to a new session" in capsys.readouterr().err
    assert _run(["pi", "--image", "x"]) == 1
    assert capsys.readouterr().err == (
        "[launch] a pi session is already running for ~/src/proj, and --image applies "
        "only to a new session. To join the session, leave out --image. To use --image, "
        "end the session, then launch again.\n")


def test_the_mount_join_step_names_each_share_as_mount_gives_it():
    """The refused --mount names the session's shares in the form that
    joins it: the container path when it is not the folder, and :ro for a
    read-only share."""
    home = settings._host_home()
    record = {"shares": [{"host": f"{home}/src/proj", "guest": f"{home}/src/proj"},
                         {"host": f"{home}/data", "guest": "/data", "readonly": True},
                         {"host": "/opt/x", "guest": "/opt/x/", "readonly": True}]}
    assert lc._share_specs(record) == ["~/src/proj", "~/data:/data:ro", "/opt/x:ro"]
    assert lc._share_specs({"shares": []}) == []


def test_the_mount_that_keyed_a_session_joins_it_again(running_session, capsys, monkeypatch):
    """With mount_cwd false, --mount . keys the project folder, so the
    command that started the session joins it when you type it again."""
    import webbrowser
    monkeypatch.setattr(webbrowser, "open", lambda url: None)
    _user_config(running_session.home, "launch:\n  container:\n    mount_cwd: false\n")
    proj = os.path.realpath(running_session.proj)
    assert _run(["pi", "--container", "--mount", "."]) == 0
    assert _run(["pi", "--shell", "--mount", f"{proj}:rw"]) == 0
    assert not running_session.runs
    assert [c[1][5] for c in running_session.copies] == ["gmlx-pi-abc123"] * 2
    assert "applies only to a new session" not in capsys.readouterr().out
    # Another mode, another container path or another folder needs a new
    # session. A share through a link names the session's folder, but a new
    # session refuses it, so a join does too.
    link = running_session.home / "proj-link"
    link.symlink_to(proj)
    for mounts in ([".:ro"], [".:/work"], [".", "~/data"], [str(link)]):
        assert _run(["pi", "--container", *(w for m in mounts for w in ("--mount", m))]) == 1
        assert capsys.readouterr().err == (
            "[launch] a pi session is already running for ~/src/proj, and --mount applies "
            "only to a new session. To join the session, leave out --mount, or give --mount "
            "only shares that the session has, as it has them: ~/src/proj. To change the "
            "shares, end the session, then launch again.\n")
    assert len(running_session.copies) == 2
    # While another launch starts the session or it ends, the answer waits
    # for that, as it does for a launch that names no --mount.
    record = session.read_record("pi", running_session.project)
    session.remove_record("pi", running_session.project)
    for mounts in (["--mount", "."], ["--mount-cwd"]):
        assert _run(["pi", "--container", *mounts]) == launch.EXIT_TEMPFAIL
        assert capsys.readouterr().err == _STILL_STARTING
    session.write_record("pi", running_session.project,
                         {**record, "ending": True, "pid": os.getpid()})
    assert _run(["pi", "--container", "--mount", ".:ro"]) == launch.EXIT_TEMPFAIL
    assert "is ending. Launch again once it has stopped." in capsys.readouterr().err
    session.write_record("pi", running_session.project, record)
    lock = _dsh_session(running_session)
    try:
        assert _run(["dsh", "--container", "--mount", "."]) == 0
    finally:
        lock.release()
    assert "dsh is already running" in capsys.readouterr().out and not running_session.runs


def test_a_launch_that_shares_no_folder_joins_the_session_of_its_folder(running_session,
                                                                        capsys):
    """With mount_cwd false, a launch without --mount keys the default
    project. The session that --mount . started in the folder takes it, as
    a launch from a folder inside a session's project folder joins it."""
    _user_config(running_session.home, "launch:\n  container:\n    mount_cwd: false\n")
    proj = os.path.realpath(running_session.proj)
    assert _run(["pi", "--container"]) == 0
    assert _run(["pi", "--shell"]) == 0
    sub = _subfolder(running_session, "sub")
    assert _run(["pi", "--container"]) == 0
    assert not running_session.runs
    assert [c[1][4:6] for c in running_session.copies] == [
        [proj, "gmlx-pi-abc123"], [proj, "gmlx-pi-abc123"], [sub, "gmlx-pi-abc123"]]
    assert capsys.readouterr().out.splitlines() == [
        "[launch] joining the running pi session for ~/src/proj",
        "[launch] opening a shell in the running pi session for ~/src/proj (gmlx-pi-abc123)",
        "[launch] joining the running pi session for ~/src/proj"]
    assert not settings.project_dir_path("pi", settings.PROJECT_DEFAULT).exists()
    # --no-mount-cwd and a share of another folder ask for the default project.
    data = running_session.home / "data"
    data.mkdir()
    assert _run(["pi", "--container", "--no-mount-cwd"]) == 0
    assert _run(["pi", "--container", "--mount", os.path.realpath(data)]) == 0
    assert [r["spec"].session.project for r in running_session.runs] == [
        settings.PROJECT_DEFAULT] * 2


def test_a_dsh_session_keyed_by_a_mount_opens_from_its_folder(env, capsys, monkeypatch):
    """With mount_cwd false, a plain launch in the folder of a dsh session
    that --mount . keyed opens that session. A launch from another folder
    starts a session of its own project, at another address."""
    monkeypatch.setattr(session, "open_in_browser", lambda url: None)
    _user_config(env.home, "launch:\n  container:\n    mount_cwd: false\n")
    other = env.home / "src" / "other"
    other.mkdir()
    lock = _dsh_session(env)
    try:
        os.chdir(env.proj)
        assert _run(["dsh", "--container"]) == 0
        assert "dsh is already running" in capsys.readouterr().out and not env.runs
        os.chdir(other)
        assert _run(["dsh", "--container", "--mount", "."]) == 0
    finally:
        lock.release()
    assert len(env.runs) == 1
    assert env.runs[0]["spec"].web_port in range(3100, 3200)
    assert env.runs[0]["spec"].web_port != 3101


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


_STILL_STARTING = ("[launch] the pi session for ~/src/proj is still starting. Try again in a "
                   "moment.\n")


def test_a_subfolder_launch_waits_while_the_session_boots(running_session, capsys):
    """The supervisor writes the full record before container run, and the
    container does not run while its virtual machine boots."""
    record = session.read_record("pi", running_session.project)
    session.write_record("pi", running_session.project, {**record, "pid": os.getpid()})
    running_session.update(containers=[{**running_session.load()["containers"][0],
                                        "state": "stopped"}])
    sub = _subfolder(running_session, "sub")
    assert _run(["pi", "--container"]) == launch.EXIT_TEMPFAIL
    assert capsys.readouterr().err == _STILL_STARTING
    running_session.update(containers=[])
    assert _run(["pi", "--container"]) == launch.EXIT_TEMPFAIL
    assert not running_session.runs and not running_session.copies
    session.write_record("pi", running_session.project, {**record, "pid": 999999})
    assert _run(["pi", "--container"]) == 0              # that launch was killed
    assert running_session.runs[0]["spec"].session.project == settings.project_id(sub)


def test_a_subfolder_launch_waits_while_the_session_starts_the_service(running_session,
                                                                      capsys, monkeypatch):
    """container ls fails while the first launch starts the service."""
    from gmlx.container import cli
    _starting(running_session, os.getpid())

    def down():
        raise cli.ContainerError("`container ls` failed: the service is not running")
    monkeypatch.setattr(cli, "list_launch_containers", down)
    _subfolder(running_session, "sub")
    for dry in ([], ["--config-only"]):
        assert _run(["pi", "--container", *dry]) == launch.EXIT_TEMPFAIL
        assert capsys.readouterr().err == _STILL_STARTING
    assert not running_session.runs


def test_a_subfolder_launch_names_the_container_query_that_fails(running_session, capsys,
                                                                 monkeypatch):
    """The full record of a live launch has no mark, so while container ls
    fails, the session can run, boot or end."""
    from gmlx.container import cli
    record = session.read_record("pi", running_session.project)
    session.write_record("pi", running_session.project, {**record, **session.launch_owner()})
    errors = [cli.ContainerError("`container ls --all --format` failed (exit 1): XPC error.")]

    def hang(*_a, **_k):
        raise subprocess.TimeoutExpired("container", 10)

    def down():
        if errors:
            raise errors[0]
        # The query runs out of time, as against a stuck service.
        with monkeypatch.context() as m:
            m.setattr(cli.subprocess, "run", hang)
            cli._run(["ls", "--all", "--format", "json"], capture=True, timeout=10)
    monkeypatch.setattr(cli, "list_launch_containers", down)
    _subfolder(running_session, "sub")
    for dry in ([], ["--config-only"]):
        assert _run(["pi", "--container", *dry]) == 1
        assert capsys.readouterr().err == (
            "[launch] the pi session for ~/src/proj shares this folder, and launch cannot "
            "tell whether it runs, because `container ls --all --format` failed (exit 1): "
            "XPC error. Try again once `container ls` works.\n")
    errors.pop(0)
    # The restart step comes after the retry, and says that it ends the session.
    assert _run(["pi", "--container"]) == launch.EXIT_UNAVAILABLE
    assert capsys.readouterr().err == (
        "[launch] the pi session for ~/src/proj shares this folder, and launch cannot "
        "tell whether it runs, because `container ls --all --format` gave no answer in "
        "10 s, so the container service may be stuck. Try again once `container ls` "
        "works. A restart of the service also stops that session. Restart it with: "
        "container system stop && container system start\n")
    assert not running_session.runs and not running_session.copies


def test_a_record_whose_pid_now_names_another_process_holds_nothing(running_session,
                                                                     capsys):
    """A launch that was killed leaves its record, and the system can give
    its process ID to another process, such as launchd's 1 here."""
    me = session.launch_owner()
    _starting(running_session, os.getpid())
    record = {**session.read_record("pi", running_session.project), **me}
    session.write_record("pi", running_session.project, record)
    sub = _subfolder(running_session, "sub")
    assert _run(["pi", "--container"]) == launch.EXIT_TEMPFAIL
    assert capsys.readouterr().err == _STILL_STARTING
    for name in ("", "gmlx-pi-abc123"):
        session.write_record("pi", running_session.project, {
            **record, "name": name, "starting": not name, "pid": 1})
        running_session.update(containers=[{"name": "gmlx-pi-abc123", "labels": {
            "gmlx.launch": "1", "gmlx.launch.client": "pi",
            "gmlx.launch.project": running_session.project, "gmlx.launch.pid": "1"}}])
        assert _run(["pi", "--container"]) == 0
    assert [r["spec"].session.project for r in running_session.runs] == [
        settings.project_id(sub)] * 2
    assert not running_session.copies


def test_the_session_record_names_its_launch(env):
    assert _run(["pi", "--container"]) == 0
    assert {k: env.runs[0]["record"][k] for k in ("pid", "pid_start")} == session.launch_owner()


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
    assert seen == [{"name": "", "workdir": proj, "starting": True, **session.launch_owner(),
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


def _dsh_session(env, web_port=3101, **record):
    """A dsh web session that shares the project folder."""
    proj = os.path.realpath(env.proj)
    return _web_session(env, "dsh", env.project, web_port=web_port, profile="gmlx", workdir=proj,
                        shares=[{"host": proj, "guest": proj, "readonly": False}],
                        project=proj, **record)


def test_a_second_launch_of_a_web_app_opens_the_running_one(env, capsys, monkeypatch):
    opened = []
    monkeypatch.setattr(session, "open_in_browser", opened.append)
    lock = _web_session(env, "open-webui", web_port=3100)
    try:
        assert _run(["open-webui", "--container"]) == 0
    finally:
        lock.release()
    assert opened == ["http://[::1]:3100/"] and not env.runs
    assert capsys.readouterr().out == ("[launch] open-webui is already running at "
                                       "http://[::1]:3100/\n")


@pytest.mark.parametrize("client", ["open-webui", "dsh"])
def test_launch_never_opens_the_browser_through_webbrowser(env, monkeypatch, client):
    """Python's webbrowser runs osascript from PATH, which a guest can
    reach through a shared folder, so both the new session and a second
    launch open the address with session.open_in_browser."""
    import webbrowser
    used = []
    monkeypatch.setattr(webbrowser, "open", used.append)
    opened = []
    monkeypatch.setattr(session, "open_in_browser", opened.append)
    assert _run([client, "--container"]) == 0
    assert env.runs[0]["opener"] is session.open_in_browser
    port = env.runs[0]["spec"].web_port
    url = f"http://[::1]:{port}/" + ("?token=t" if client == "dsh" else "")
    key = env.runs[0]["spec"].session.project
    lock = _web_session(env, client, key, web_port=port, url=url,
                        profile="gmlx" if client == "dsh" else None)
    try:
        assert _run([client, "--container"]) == 0
    finally:
        lock.release()
    assert opened == [url] and used == []


def test_a_second_dsh_launch_opens_the_recorded_token_url(env, capsys, monkeypatch):
    opened = []
    monkeypatch.setattr(session, "open_in_browser", opened.append)
    lock = _dsh_session(env)
    try:
        assert _run(["dsh", "--container"]) == 0
        assert opened == [] and "has not printed its address yet" in capsys.readouterr().out
        record = session.read_record("dsh", env.project)
        session.write_record("dsh", env.project,
                             {**record, "url": "http://evil.example/?token=t"})
        assert _run(["dsh", "--container"]) == 0 and opened == []
        # The Mac serves the app at [::1] only, so the guest's own address
        # would open nothing.
        session.write_record("dsh", env.project,
                             {**record, "url": "http://127.0.0.1:3101/?token=t"})
        assert _run(["dsh", "--container"]) == 0 and opened == []
        session.write_record("dsh", env.project,
                             {**record, "url": "http://[::1]:3101/?token=t"})
        assert _run(["dsh", "--container"]) == 0
    finally:
        lock.release()
    assert opened == ["http://[::1]:3101/?token=t"]


def test_dsh_keeps_a_home_and_volumes_per_project(env):
    _user_config(env.home, "launch:\n  container:\n    clients:\n      dsh:\n"
                           "        volumes: [\"tools:/tools\"]\n")
    assert _run(["dsh", "--container"]) == 0
    spec = env.runs[0]["spec"]
    assert spec.session.project == env.project and spec.web_port == 3100
    assert spec.plan.home == settings.private_home_path("dsh", env.project)
    assert [m.source for m in spec.plan.volumes] == [
        settings.project_volume_name("tools", env.project)]
    assert env.runs[0]["record"]["project"] == os.path.realpath(env.proj)


def test_dsh_sessions_of_two_projects_run_at_once_on_ports_of_their_own(env, capsys):
    """Each project's dsh web app has a Mac port of its own, so the pages of
    one project's guest leave nothing at the address of another project,
    and a project keeps its address from one launch to the next."""
    assert _run(["dsh", "--container"]) == 0
    first = env.runs[0]["spec"].web_port
    other = env.home / "src" / "other"
    other.mkdir()
    lock = _dsh_session(env, web_port=first)
    os.chdir(other)
    try:
        assert _run(["dsh", "--container"]) == 0
    finally:
        lock.release()
    second = env.runs[1]["spec"]
    assert second.session.project == settings.project_id(os.path.realpath(other))
    assert (first, second.web_port) == (3100, 3101)
    assert second.command[-3:] == ["--no-open", "--port", "3101"]
    assert env.runs[1]["server_session"].web_ports == [3101]
    session.remove_record("dsh", env.project)            # the first session ended
    env.update(containers=[])
    os.chdir(env.proj)
    assert _run(["dsh", "--container"]) == 0
    assert env.runs[2]["spec"].web_port == 3100
    assert web_ports.recorded("dsh", env.project) == 3100
    assert "is not free" not in capsys.readouterr().out


@pytest.mark.parametrize("client", ["open-webui", "dsh"])
def test_container_browser_apps_never_use_the_host_mode_ports(env, monkeypatch, client):
    """Host mode serves Open WebUI on 3000 or 3001 and dsh on 3080 or 3081,
    and the server refuses the pages of a session's ports for 15 minutes
    after it ends. A container session uses neither, whatever port the
    gmlx server has."""
    for server_port in (8080, 3000, 3080, 3100):
        monkeypatch.setattr(lifecycle, "auto_target",
                            lambda h, p, port=server_port: ("127.0.0.1", port))
        assert _run([client, "--container"]) == 0
        run = env.runs[-1]
        port = run["spec"].web_port
        assert 3100 <= port <= 3199 and port != server_port
        assert run["server_session"].web_ports == [port]
        assert run["record"]["web_port"] == port
    assert launch.web_port_for("open-webui", 8080) == 3000
    assert launch.web_port_for("open-webui", 3000) == 3001
    assert launch.web_port_for("dsh", 3080) == 3081


def test_the_web_port_skips_the_forwarded_ports(env, monkeypatch):
    """forward gives the container's ports their Mac ports of the same
    number, so the web app never takes one, also when the server check
    moves the web port."""
    _user_config(env.home, "launch:\n  container:\n    clients:\n      dsh:\n"
                           "        forward: [3100, 3101]\n")
    assert _run(["dsh", "--container"]) == 0
    assert env.runs[-1]["spec"].web_port == 3102
    assert env.runs[-1]["spec"].plan.forward == [3100, 3101]
    real = launch._ensure_server

    def moved(a):
        a.host, a.port = "127.0.0.1", 3103
        a.base_url = "http://127.0.0.1:3103/v1"
        return real(a)
    monkeypatch.setattr(launch, "_ensure_server", moved)
    other = env.home / "src" / "other"
    other.mkdir()
    os.chdir(other)
    assert _run(["dsh", "--container"]) == 0                # step 6 took 3103
    assert env.runs[-1]["spec"].web_port == 3104
    assert env.runs[-1]["spec"].plan.forward == [3100, 3101]


def test_a_busy_recorded_port_moves_the_app_with_one_line(env, capsys):
    assert _run(["dsh", "--container"]) == 0
    assert env.runs[0]["spec"].web_port == 3100
    capsys.readouterr()
    env.busy_ports.add(3100)
    assert _run(["dsh", "--container"]) == 0
    assert env.runs[1]["spec"].web_port == 3101
    assert web_ports.recorded("dsh", env.project) == 3101
    assert capsys.readouterr().out.count(
        "[launch] port 3100 of the dsh web app of this project is not free, so the app moves "
        "to port 3101. The browser keeps sign-ins and saved data by address, so the app can "
        "ask you to sign in again.\n") == 1
    env.busy_ports.clear()
    assert _run(["dsh", "--container"]) == 0
    assert env.runs[2]["spec"].web_port == 3101           # the new port stays


@pytest.mark.parametrize("mount_cwd", [True, False])
def test_a_launch_refuses_when_no_port_of_the_range_is_free(env, capsys, monkeypatch,
                                                              mount_cwd):
    """The refusal names the command that removes the home of the project
    that keeps the port. With mount_cwd: false a plain --remove-home in the
    project's folder would key the default project, so the command names
    --mount ., which keys the folder either way."""
    if not mount_cwd:
        _user_config(env.home, "launch:\n  container:\n    mount_cwd: false\n")
    env.busy_ports.update(range(3101, 3200))
    assert _run(["dsh", "--container", *([] if mount_cwd else ["--mount", "."])]) == 0
    assert env.runs[0]["spec"].web_port == 3100
    other = env.home / "src" / "other"
    other.mkdir()
    os.chdir(other)
    capsys.readouterr()
    assert _run(["dsh", "--container", "--mount", "."]) == launch.EXIT_TEMPFAIL
    assert capsys.readouterr().err == (
        "[launch] no Mac port from 3100 to 3199 is free for the dsh web app, because other "
        "projects keep them or other programs use them. To free the port of a project you "
        "no longer need, remove its private home. For the project used longest ago, run "
        "gmlx launch dsh --remove-home --mount . in ~/src/proj.\n")
    assert len(env.runs) == 1
    other_project = settings.project_id(os.path.realpath(other))
    assert web_ports.recorded("dsh", other_project) is None
    os.chdir(env.home / "src" / "proj")
    monkeypatch.setattr(session, "stdin_is_terminal", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert _run(["dsh", "--remove-home", "--mount", "."]) == 0
    assert web_ports.recorded("dsh", env.project) is None
    assert not settings.private_home_path("dsh", env.project).exists()


def test_the_refusal_names_a_folder_where_no_mount_cwd_keys_the_default_project(
        env, capsys, monkeypatch):
    """A mounts: entry that holds the current folder makes --no-mount-cwd
    key the folder of that entry, so the step for the default project of
    dsh names /, which no share holds."""
    _user_config(env.home, "launch:\n  container:\n    clients:\n      dsh:\n"
                           "        mounts: [~/src/proj]\n")
    other = env.home / "src" / "other"
    other.mkdir()
    os.chdir(other)
    assert _run(["dsh", "--container", "--no-mount-cwd"]) == 0
    assert web_ports.recorded("dsh", settings.PROJECT_DEFAULT) == 3100
    env.busy_ports.update(range(3101, 3200))
    sub = env.proj / "sub"
    sub.mkdir()
    os.chdir(sub)
    capsys.readouterr()
    assert _run(["dsh", "--container"]) == launch.EXIT_TEMPFAIL
    assert capsys.readouterr().err.endswith(
        "run gmlx launch dsh --remove-home --no-mount-cwd in /.\n")
    monkeypatch.setattr(session, "stdin_is_terminal", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert _run(["dsh", "--remove-home", "--no-mount-cwd"]) == 0      # in the share
    assert "dsh has no private home for ~/src/proj" in capsys.readouterr().out
    assert web_ports.recorded("dsh", settings.PROJECT_DEFAULT) == 3100
    os.chdir("/")
    assert _run(["dsh", "--remove-home", "--no-mount-cwd"]) == 0
    assert web_ports.recorded("dsh", settings.PROJECT_DEFAULT) is None
    assert not settings.private_home_path("dsh").exists()


def test_the_dry_run_shows_the_port_and_records_nothing(env, capsys):
    assert _run(["dsh", "--container", "--config-only"]) == 0
    assert "--no-open --port 3100" in capsys.readouterr().out
    assert web_ports.recorded("dsh", env.project) is None
    assert _run(["open-webui", "--container"]) == 0            # takes 3100
    env.busy_ports.add(3101)
    assert _run(["dsh", "--container", "--config-only"]) == 0
    assert "--no-open --port 3102" in capsys.readouterr().out
    assert web_ports.recorded("dsh", env.project) is None
    assert _run(["dsh", "--container"]) == 0
    assert web_ports.recorded("dsh", env.project) == 3102
    env.busy_ports.add(3102)
    assert _run(["dsh", "--container", "--config-only"]) == 0
    out = capsys.readouterr().out
    assert "--no-open --port 3103" in out and "so the app would move to port 3103" in out
    assert web_ports.recorded("dsh", env.project) == 3102


def test_a_port_whose_first_launch_stopped_is_free_again(env, capsys):
    """A launch takes the port before it makes the private home. When it
    stops first, the port goes to the next project once that launch is
    gone."""
    _user_config(env.home, "launch:\n  container:\n    clients:\n      dsh:\n"
                           "        forward: [8080]\n")
    assert _run(["dsh", "--container"]) == 1                   # forward lists the server
    assert web_ports.recorded("dsh", env.project) == 3100
    assert not settings.private_home_path("dsh", env.project).exists()
    _user_config(env.home, "")
    other = env.home / "src" / "other"
    other.mkdir()
    os.chdir(other)
    _forget_web_launch("dsh", env.project)
    assert _run(["dsh", "--container"]) == 0
    assert env.runs[0]["spec"].web_port == 3100
    assert web_ports.recorded("dsh", env.project) is None


def _forget_web_launch(client: str, project: str) -> None:
    """The launch that took the web port of a project has exited."""
    path = settings.data_path() / "web-ports.json"
    record = json.loads(path.read_text())
    record["projects"][client][project]["pid"] = 999999
    path.write_text(json.dumps(record))


# A page that stays open would be the same origin as the next app on the
# port, and it can store data again after a clear, so the tabs close first.
_SITE_DATA = ("Its pages possibly left a service worker and stored data there, and a page "
              "that is still open keeps running. Close each browser tab and window of that "
              "address, and each window that the app's pages opened, or quit the browser. "
              "Then clear the site data of that address in your browser.\n")


def _reused_advice(port: int) -> str:
    return (f"They possibly left a service worker and stored data at http://[::1]:{port}, and "
            "a page that is still open keeps running and can store data again. So close each "
            "browser tab and window of that address, and each window that its pages opened, "
            "or quit the browser. Then clear the site data of that address before you open "
            "the app.")


def _reuse_line(client: str, port: int) -> str:
    return (f"[launch] the {client} web app of this project takes port {port}, which the pages "
            f"of another project or app used. {_reused_advice(port)} This launch does not "
            "open the browser, so you can do that first.\n")


def test_a_port_another_project_used_goes_to_a_new_project_only_last(env, capsys):
    """A port keeps what the pages of the project that used it left: a
    service worker that sees the next page and its sign-in token. So a new
    project takes a port that no project used. It gets a used one only when
    no other port is free, and then launch says to clear the site data and
    does not open the browser."""
    assert _run(["dsh", "--container"]) == 0                    # project A on 3100
    env.busy_ports.add(3100)                                    # another program, for a time
    assert _run(["dsh", "--container"]) == 0                    # A moves to 3101
    env.busy_ports.clear()
    other = env.home / "src" / "other"
    other.mkdir()
    os.chdir(other)
    capsys.readouterr()
    assert _run(["dsh", "--container"]) == 0                    # project B
    assert env.runs[2]["spec"].web_port == 3102
    assert "site data" not in capsys.readouterr().out
    # A's folder removed by hand, and A's launch has exited.
    _forget_web_launch("dsh", env.project)
    shutil.rmtree(settings.project_dir_path("dsh", env.project))
    third = env.home / "src" / "third"
    third.mkdir()
    os.chdir(third)
    assert _run(["dsh", "--container"]) == 0                    # project C
    assert env.runs[3]["spec"].web_port == 3103
    assert env.runs[3]["opener"] is not None
    assert "site data" not in capsys.readouterr().out
    env.busy_ports.update(range(3104, 3200))
    fourth = env.home / "src" / "fourth"
    fourth.mkdir()
    os.chdir(fourth)
    assert _run(["dsh", "--container"]) == 0                    # project D
    assert env.runs[4]["spec"].web_port == 3100
    assert env.runs[4]["opener"] is None
    assert capsys.readouterr().out.count(_reuse_line("dsh", 3100)) == 1
    assert _run(["dsh", "--container"]) == 0                    # D's own port from now on
    assert env.runs[5]["spec"].web_port == 3100
    assert env.runs[5]["opener"] is not None
    assert "site data" not in capsys.readouterr().out


@pytest.mark.parametrize("url, status", [
    ("http://[::1]:3100/?token=t",
     "[launch] dsh is already running at http://[::1]:3100/?token=t\n"),
    (None, "[launch] dsh is already running, and its web app has not printed its address "
           "yet. The launch that started it shows the address once it is ready.\n")])
def test_a_second_launch_on_a_used_port_does_not_open_the_browser(env, capsys, monkeypatch,
                                                                   url, status):
    """The launch that took a port that another project's pages used did
    not open the browser, so you can clear the site data first. A second
    launch of that session does not open it either."""
    opened = []
    monkeypatch.setattr(session, "open_in_browser", opened.append)
    assert _run(["open-webui", "--container"]) == 0             # 3100, served
    shutil.rmtree(settings.project_dir_path("open-webui", settings.PROJECT_DEFAULT))
    _forget_web_launch("open-webui", settings.PROJECT_DEFAULT)
    env.busy_ports.update(range(3101, 3200))
    assert _run(["dsh", "--container"]) == 0
    record = env.runs[-1]["record"]
    assert record["web_port"] == 3100 and record["reused"] is True
    assert env.runs[-1]["opener"] is None
    capsys.readouterr()
    lock = _web_session(env, "dsh", env.project, **{
        **{k: v for k, v in record.items() if k != "name"}, "url": url})
    try:
        assert _run(["dsh", "--container"]) == 0
    finally:
        lock.release()
    assert opened == [] and len(env.runs) == 2 and not env.copies
    assert capsys.readouterr().out == status + (
        "[launch] the pages of another project or app used port 3100 before this session. "
        f"{_reused_advice(3100)} This launch does not open the browser, so you can do that "
        "first.\n")


def test_a_project_folder_removed_by_hand_keeps_its_port_last(env, capsys):
    """The session marks its port as served when it starts, so the port of
    a project that ran once stays last after its folder is deleted."""
    assert _run(["dsh", "--container"]) == 0                    # project A on 3100
    _forget_web_launch("dsh", env.project)
    shutil.rmtree(settings.project_dir_path("dsh", env.project))
    other = env.home / "src" / "other"
    other.mkdir()
    os.chdir(other)
    capsys.readouterr()
    assert _run(["dsh", "--container"]) == 0
    assert env.runs[1]["spec"].web_port == 3101
    assert "site data" not in capsys.readouterr().out


def test_the_dry_run_says_a_used_port_would_go_to_this_project(env, capsys):
    assert _run(["open-webui", "--container"]) == 0             # 3100, served
    shutil.rmtree(settings.project_dir_path("open-webui", settings.PROJECT_DEFAULT))
    _forget_web_launch("open-webui", settings.PROJECT_DEFAULT)
    env.busy_ports.update(range(3101, 3200))
    capsys.readouterr()
    assert _run(["dsh", "--container", "--config-only"]) == 0
    out = capsys.readouterr().out
    assert ("[launch] the dsh web app of this project would take port 3100, which the pages "
            f"of another project or app used. {_reused_advice(3100)}\n") in out
    assert "does not open the browser" not in out


# dsh opens only at the address with its login token, which it prints with
# the address it has in the container.
_TOKEN_STEP = (". dsh then prints its address with a login token. Open that address with "
               "[::1] in place of 127.0.0.1.")


@pytest.mark.parametrize("client, port, command, how", [
    ("open-webui", 3100, None, ", where it must listen on 127.0.0.1:$PORT"),
    ("dsh", 3101, ["dsh", "--profile", "gmlx", "--no-open", "--port", "3101"],
     " with: dsh --profile gmlx --no-open --port 3101. dsh then prints its address with a "
     "login token. Open that address with [::1] in place of 127.0.0.1.")])
def test_a_second_launch_of_a_web_app_that_runs_a_shell_says_so(env, capsys, monkeypatch,
                                                                 client, port, command, how):
    """A bare dsh in the shell listens on its own default port, 3080, which
    the session does not serve. So the line names the recorded command. dsh
    opens only at the address with its login token, which it prints with
    the address it has in the container."""
    opened = []
    monkeypatch.setattr(session, "open_in_browser", opened.append)
    key = env.project if client == "dsh" else "default"
    proj = os.path.realpath(env.proj)
    shares = [{"host": proj, "guest": proj, "readonly": False}] if client == "dsh" else []
    lock = _web_session(env, client, key, web_port=port, shell=True, shares=shares,
                        url="http://[::1]:3101/?token=t", command=command)
    try:
        assert _run([client, "--container"]) == 0
    finally:
        lock.release()
    assert opened == [] and not env.runs and not env.copies
    assert capsys.readouterr().out == (
        f"[launch] the running {client} session runs a shell. To open another shell in the "
        f"session, run: gmlx launch {client} --shell\n"
        f"[launch] {client} answers at http://[::1]:{port}/ once you start it in that "
        f"shell{how}\n")


def test_a_shell_session_of_an_image_command_names_the_folder_it_needs(env, capsys):
    """The image's command runs in the image's working folder, and a shell
    starts in the private home or the shared folder. So the start command
    that launch names changes to the image's folder first."""
    env.update(registry={"ghcr.io/open-webui/open-webui:main": {
        "digest": "sha256:" + "5" * 64, "cmd": ["bash", "start.sh"],
        "workdir": "/app/backend"}})
    _user_config(env.home, "launch:\n  container:\n    clients:\n      open-webui:\n"
                           "        image: ghcr.io/open-webui/open-webui:main\n"
                           "        command: image\n")
    assert _run(["open-webui", "--container", "--shell"]) == 0
    record = env.runs[-1]["record"]
    assert env.runs[-1]["spec"].workdir != "/app/backend"
    assert record["command"] == ["bash", "start.sh"]
    assert record["command_workdir"] == "/app/backend"
    capsys.readouterr()
    lock = _web_session(env, "open-webui", **{k: v for k, v in record.items() if k != "name"})
    try:
        assert _run(["open-webui", "--container"]) == 0
    finally:
        lock.release()
    assert capsys.readouterr().out.endswith(
        "[launch] open-webui answers at http://[::1]:3100/ once you start it in that "
        "shell with: cd /app/backend && bash start.sh\n")


def test_a_dsh_shell_line_leaves_out_the_template_once_the_profile_exists(env, capsys):
    """The first start of dsh in the shell makes the profile from the web
    template, and dsh refuses --from-default-profile once the profile
    exists. So the first launch says to leave it out after that start, and a
    second launch names the command without it once the private home holds
    the profile's manifest."""
    assert _run(["dsh", "--container", "--shell"]) == 0
    run = env.runs[-1]
    record = run["record"]
    at = record["command"].index("--from-default-profile")
    assert record["command"][at + 1] == "web"
    assert ("[launch] the first start of dsh in the shell makes its gmlx profile from the "
            "web template. To start dsh again after that, leave out --from-default-profile "
            "web.") in run["summary"]
    rest = {k: v for k, v in record.items() if k != "name"}

    def second_launch() -> str:
        capsys.readouterr()
        lock = _web_session(env, "dsh", env.project, **rest)
        try:
            assert _run(["dsh", "--container"]) == 0
        finally:
            lock.release()
        return capsys.readouterr().out

    # dsh has not started in the shell yet, so the profile is not there.
    assert second_launch().endswith(f" with: {shlex.join(record['command'])}{_TOKEN_STEP}\n")
    profile = settings.private_home_path("dsh", env.project) / ".dsh" / "profiles" / "gmlx"
    profile.mkdir(parents=True)
    (profile / "package.json").write_text('{"name": "gmlx"}')
    without = [*record["command"][:at], *record["command"][at + 2:]]
    assert second_launch().endswith(
        "[launch] dsh answers at http://[::1]:3100/ once you start it in that shell "
        f"with: {shlex.join(without)}{_TOKEN_STEP}\n")
    # A link that the guest puts in the private home is not followed.
    elsewhere = env.home / "elsewhere"
    shutil.move(profile.parent.parent, elsewhere)
    os.symlink(elsewhere, profile.parent.parent)
    assert second_launch().endswith(f" with: {shlex.join(record['command'])}{_TOKEN_STEP}\n")


def test_a_dsh_launch_with_another_profile_is_refused(env, capsys):
    lock = _web_session(env, "dsh", web_port=3101, profile="gmlx")
    try:
        assert _run(["dsh", "--container", "--no-mount-cwd", "--dsh-profile", "headless"]) == 1
    finally:
        lock.release()
    assert ("a dsh session with the gmlx profile is already running, and a project runs one "
            "session at a time. End it to start the headless profile.") in capsys.readouterr().err


def test_a_shell_on_a_running_web_app_opens_a_shell(env):
    lock = _web_session(env, "open-webui", web_port=3100)
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
    assert record["web"] is True and record["web_port"] == 3100 and record["project"] is None
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


def test_remove_home_releases_the_web_port_of_the_project(env, capsys, monkeypatch):
    """Launch names each address the project's pages used, so you can clear
    what they left. Another project takes those ports only last."""
    monkeypatch.setattr(session, "stdin_is_terminal", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert _run(["dsh", "--container"]) == 0
    env.busy_ports.add(3100)
    assert _run(["dsh", "--container"]) == 0                    # moves to 3101
    env.busy_ports.clear()
    assert web_ports.recorded("dsh", env.project) == 3101
    capsys.readouterr()
    assert _run(["dsh", "--remove-home"]) == 0
    assert web_ports.recorded("dsh", env.project) is None
    assert capsys.readouterr().out.endswith(
        "[launch] the web app of this project used http://[::1]:3100 and "
        "http://[::1]:3101. Its pages possibly left a service worker and stored data there, "
        "and a page that is still open keeps running. Close each browser tab and window of "
        "these addresses, and each window that the app's pages opened, or quit the browser. "
        "Then clear the site data of these addresses in your browser.\n")
    other = env.home / "src" / "other"
    other.mkdir()
    os.chdir(other)
    assert _run(["dsh", "--container"]) == 0
    assert env.runs[-1]["spec"].web_port == 3102
    assert _run(["pi", "--container"]) == 0
    os.chdir(env.proj)
    capsys.readouterr()
    assert _run(["pi", "--remove-home"]) == 0
    assert "site data" not in capsys.readouterr().out


def test_remove_home_without_a_home_still_names_the_addresses(env, capsys):
    """A project folder removed by hand leaves the pages' data in the
    browser, so --remove-home still names the address and frees the entry."""
    assert _run(["dsh", "--container"]) == 0
    _forget_web_launch("dsh", env.project)
    shutil.rmtree(settings.project_dir_path("dsh", env.project))
    capsys.readouterr()
    assert _run(["dsh", "--remove-home"]) == 0
    assert capsys.readouterr().out == (
        "[launch] dsh has no private home for ~/src/proj, so nothing was removed.\n"
        "[launch] the web app of this project used http://[::1]:3100. " + _SITE_DATA)
    assert web_ports.recorded("dsh", env.project) is None
    assert not settings.project_dir_path("dsh", env.project).exists()


def test_remove_home_without_a_home_keeps_the_port_of_a_running_first_launch(env, capsys):
    """A first launch takes its port before it makes the home."""
    web_ports.choose("dsh", env.project)                        # this process runs
    capsys.readouterr()
    assert _run(["dsh", "--remove-home"]) == 0
    assert capsys.readouterr().out == ("[launch] dsh has no private home for ~/src/proj, so "
                                       "nothing was removed.\n")
    assert web_ports.recorded("dsh", env.project) == 3100


def _next_project_port(env) -> tuple[int, object]:
    """The web port and the opener of a dsh launch in a new project."""
    other = env.home / "src" / "other"
    other.mkdir()
    os.chdir(other)
    assert _run(["dsh", "--container"]) == 0
    return env.runs[-1]["spec"].web_port, env.runs[-1]["opener"]


def test_remove_home_after_a_first_launch_that_stopped_names_no_address(env, capsys):
    """A first launch that stopped before its session started served no
    pages, so --remove-home names no address and the port is not kept last."""
    _user_config(env.home, "launch:\n  container:\n    clients:\n      dsh:\n"
                           "        forward: [8080]\n")
    assert _run(["dsh", "--container"]) == 1                   # forward lists the server
    _user_config(env.home, "")
    _forget_web_launch("dsh", env.project)
    capsys.readouterr()
    assert _run(["dsh", "--remove-home"]) == 0
    assert capsys.readouterr().out == ("[launch] dsh has no private home for ~/src/proj, so "
                                       "nothing was removed.\n")
    assert web_ports.recorded("dsh", env.project) is None
    port, opener = _next_project_port(env)
    assert port == 3100 and opener is not None


def test_remove_home_of_a_home_whose_session_never_started_names_no_address(
        env, capsys, monkeypatch):
    """A launch makes the private home before its session starts."""
    monkeypatch.setattr(session, "stdin_is_terminal", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    fake = session.supervise
    monkeypatch.setattr(session, "supervise", lambda spec, **kw: 1)    # container run fails
    assert _run(["dsh", "--container"]) == 1
    monkeypatch.setattr(session, "supervise", fake)
    assert settings.private_home_path("dsh", env.project).is_dir()
    assert not session.started_path("dsh", env.project).exists()
    _forget_web_launch("dsh", env.project)
    capsys.readouterr()
    assert _run(["dsh", "--remove-home"]) == 0
    out = capsys.readouterr().out
    assert "[launch] removed " in out and "site data" not in out
    assert web_ports.recorded("dsh", env.project) is None
    port, opener = _next_project_port(env)
    assert port == 3100 and opener is not None


def test_remove_home_names_the_port_of_a_started_project_the_list_lost(env, capsys,
                                                                       monkeypatch):
    """The start mark lies in the folder that --remove-home removes, so
    launch reads it first."""
    monkeypatch.setattr(session, "stdin_is_terminal", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert _run(["dsh", "--container"]) == 0                    # 3100, started
    path = settings.data_path() / "web-ports.json"
    record = json.loads(path.read_text())
    record["served"] = {}
    record["projects"]["dsh"][env.project]["pid"] = 999999
    path.write_text(json.dumps(record))
    capsys.readouterr()
    assert _run(["dsh", "--remove-home"]) == 0
    assert capsys.readouterr().out.endswith(
        "[launch] the web app of this project used http://[::1]:3100. " + _SITE_DATA)
    port, opener = _next_project_port(env)
    assert port == 3101


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
    # Open WebUI reads CORS_ALLOW_ORIGIN from the Mac in host mode only.
    assert names == {"HERMES_HOME", "AUDIO_TTS_VOICE", "DSH_HOME", "CORS_ALLOW_ORIGIN"}


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
    real = launch._ensure_server

    def moved(a):
        a.host, a.port = "127.0.0.1", 3100
        a.base_url = "http://127.0.0.1:3100/v1"
        return real(a)
    monkeypatch.setattr(launch, "_ensure_server", moved)
    assert _run(["open-webui", "--container"]) == 0
    assert env.runs[0]["spec"].web_port == 3101
    assert env.runs[0]["server_session"].web_ports == [3101]
    assert web_ports.recorded("open-webui", settings.PROJECT_DEFAULT) == 3101


def test_a_port_a_first_launch_left_before_its_start_is_not_used(env, monkeypatch):
    """A launch makes the private home before its session starts. A port
    that a first launch recorded and then left served no pages, so the next
    project takes it with no line."""
    real = launch._ensure_server

    def moved(a):
        a.host, a.port = "127.0.0.1", 3100
        a.base_url = "http://127.0.0.1:3100/v1"
        return real(a)
    monkeypatch.setattr(launch, "_ensure_server", moved)
    assert _run(["open-webui", "--container"]) == 0
    assert env.runs[0]["spec"].web_port == 3101
    monkeypatch.setattr(launch, "_ensure_server", real)
    assert _run(["dsh", "--container"]) == 0
    assert env.runs[1]["spec"].web_port == 3100
    assert env.runs[1]["opener"] is not None


def test_a_move_for_the_server_found_says_so_only_for_a_port_from_before(env, monkeypatch,
                                                                         capsys):
    """Step 6 records the port of a first launch before the server check.
    When the check moves that port, the app had no address before, so no
    line says that it moves."""
    real = launch._ensure_server

    def moved(a):
        a.host, a.port = "127.0.0.1", 3100
        a.base_url = "http://127.0.0.1:3100/v1"
        return real(a)
    monkeypatch.setattr(launch, "_ensure_server", moved)
    assert _run(["open-webui", "--container"]) == 0             # first launch
    assert env.runs[0]["spec"].web_port == 3101
    assert "is not free" not in capsys.readouterr().out
    monkeypatch.setattr(launch, "_ensure_server", real)
    assert _run(["dsh", "--container"]) == 0                    # 3100, server on 8080
    assert env.runs[1]["spec"].web_port == 3100
    monkeypatch.setattr(launch, "_ensure_server", moved)
    capsys.readouterr()
    assert _run(["dsh", "--container"]) == 0
    assert env.runs[2]["spec"].web_port == 3102
    assert capsys.readouterr().out.count(
        "[launch] port 3100 of the dsh web app of this project is not free, so the app moves "
        "to port 3102.") == 1


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
    when it attaches, so they must follow the parser. --remove-home, --stop
    and --list never join, and --detach is decided before the join."""
    from tests.commands.test_launch import _parse_launch_args
    a = _parse_launch_args(["pi"])
    defaults = {**lc._JOIN_IGNORED, **lc._JOIN_REFUSED}
    for dest, default in defaults.items():
        assert getattr(a, dest) == default, dest
    rest = set(vars(a)) - set(defaults) - {"harness", "container", "shell", "passthrough",
                                           "argv_given", "remove_home", "detach", "stop",
                                           "list", "mount_cwd", "dsh_profile"}
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


@pytest.mark.parametrize("version, have", [("1.4.1", "is version 1.4.1"),
                                           ("dev", "gives no version number")])
def test_an_old_container_names_the_program_and_both_upgrade_routes(env, capsys, version,
                                                                     have):
    """An older container in /usr/local/bin can come before Homebrew's on PATH."""
    env.update(version=version)
    assert _run(["pi", "--container"]) == launch.EXIT_UNAVAILABLE
    assert capsys.readouterr().err == (
        f"[launch] container mode needs Apple container 1.5.0 or newer, and "
        f"{shutil.which('container')}, the first container command on PATH, {have}. "
        "Upgrade with: brew upgrade container, or install the newer release from "
        "https://github.com/apple/container/releases.\n")


@pytest.mark.parametrize("version, line", [
    ("1.4.1", "container 1.4.1 at {path} is older than the 1.5.0 this mode needs"),
    ("dev", "{path}, the first container command on PATH, gives no version number, and "
            "this mode needs 1.5.0 or newer")], ids=["old", "no-version"])
def test_a_dry_run_names_an_old_container_program(env, capsys, version, line):
    """The dry run names each container program that a launch refuses."""
    env.update(version=version)
    assert _run(["pi", "--container", "--config-only"]) == 0
    assert (f"[launch] {line.format(path=shutil.which('container'))}. Upgrade with: brew "
            "upgrade container, or install the newer release from "
            "https://github.com/apple/container/releases.\n") in capsys.readouterr().out


def _later_container(monkeypatch, folder: Path, version: str) -> Path:
    """A container program at the end of PATH that gives ``version``. It
    writes the file ``ran`` beside its folder when it runs."""
    folder.mkdir(parents=True)
    program = folder / "container"
    program.write_text(f"#!/bin/sh\ntouch {folder.parent / 'ran'}\n"
                       f"echo 'container CLI version {version} (build: release)'\n")
    program.chmod(0o755)
    monkeypatch.setenv("PATH", f"{os.environ['PATH']}:{folder}")
    return program


def _package_scripts(folder: Path) -> None:
    """The scripts that Apple's installer package puts beside the program."""
    for name in ("update-container.sh", "uninstall-container.sh"):
        (folder / name).write_text("#!/bin/sh\n")


def _homebrew_install(tmp_path, monkeypatch) -> Path:
    """A copy of the fake in a Homebrew layout, first on PATH: a link in
    bin to the program in the Cellar. Returns the bin folder."""
    fake = Path(shutil.which("container") or "")
    cellar = tmp_path / "brew" / "Cellar" / "container" / "1.4.1" / "bin"
    cellar.mkdir(parents=True)
    shutil.copy2(fake, cellar / "container")
    (tmp_path / "brew" / "bin").mkdir()
    (tmp_path / "brew" / "bin" / "container").symlink_to(cellar / "container")
    monkeypatch.setenv("PATH", f"{tmp_path / 'brew' / 'bin'}:/usr/bin:/bin")
    return tmp_path / "brew" / "bin"


@pytest.mark.parametrize("install", ["package", "homebrew", "other"])
def test_a_newer_container_later_on_path_gets_its_own_step(env, capsys, monkeypatch,
                                                           tmp_path, install):
    """When an older program comes first on PATH, an upgrade of the newer one
    changes nothing."""
    env.update(version="1.4.1")
    first = Path(shutil.which("container") or "").parent
    if install == "package":
        _package_scripts(first)
    elif install == "homebrew":
        first = _homebrew_install(tmp_path, monkeypatch)
    later = _later_container(monkeypatch, tmp_path / "later" / "bin", "1.5.0")
    other = {"package": f", or remove the older install with: {first}/uninstall-container.sh -k",
             "homebrew": ", or upgrade the first one with: brew upgrade container",
             "other": "."}[install]
    assert _run(["pi", "--container"]) == launch.EXIT_UNAVAILABLE
    assert capsys.readouterr().err == (
        f"[launch] container mode needs Apple container 1.5.0 or newer, and "
        f"{first}/container, the first container command on PATH, is version 1.4.1. "
        f"{later} comes later on PATH and is version 1.5.0. Stop the container service "
        f"with: container system stop. Then put {later.parent} before {first} on PATH"
        f"{other}\n")
    assert _run(["pi", "--container", "--config-only"]) == 0
    assert (f"[launch] container 1.4.1 at {first}/container is older than the 1.5.0 this "
            f"mode needs. {later} comes later on PATH") in capsys.readouterr().out


@pytest.mark.parametrize("install", ["package", "homebrew"])
def test_a_single_old_container_gets_the_step_of_its_install(env, capsys, monkeypatch,
                                                             tmp_path, install):
    env.update(version="1.4.1")
    first = Path(shutil.which("container") or "").parent
    if install == "package":
        _package_scripts(first)
        step = (f"Stop the container service with: container system stop. Then upgrade "
                f"Apple container with: {first}/update-container.sh")
    else:
        first = _homebrew_install(tmp_path, monkeypatch)
        step = "Upgrade it with: brew upgrade container"
    # A later program that is also too old changes nothing.
    _later_container(monkeypatch, tmp_path / "later" / "bin", "1.4.0")
    assert _run(["pi", "--container"]) == launch.EXIT_UNAVAILABLE
    assert capsys.readouterr().err == (
        f"[launch] container mode needs Apple container 1.5.0 or newer, and "
        f"{first}/container, the first container command on PATH, is version 1.4.1. "
        f"{step}\n")
    assert (tmp_path / "later" / "ran").exists()


def test_a_later_container_a_client_could_replace_does_not_run(env, capsys, monkeypatch,
                                                               tmp_path):
    env.update(version="1.4.1")
    folder = tmp_path / "later" / "bin"
    _later_container(monkeypatch, folder, "1.5.0")
    settings.record_shares(SimpleNamespace(mounts=[settings.Mount(str(folder),
                                                                  str(folder))]))
    assert _run(["pi", "--container"]) == launch.EXIT_UNAVAILABLE
    assert capsys.readouterr().err.endswith(
        "is version 1.4.1. Upgrade with: brew upgrade container, or install the newer "
        "release from https://github.com/apple/container/releases.\n")
    assert not (tmp_path / "later" / "ran").exists()


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


def test_a_server_bound_beyond_loopback_is_named_through_127_0_0_1(env, monkeypatch):
    """A server bound to 0.0.0.0 also answers at 127.0.0.1, so --port finds
    its bind in the runfile. A runfile of another port or of a server that
    is gone does not count."""
    lifecycle.write_run("0.0.0.0", 8081, {"host": "0.0.0.0", "port": 8081,
                                          "managed_by": "launchd"})
    lifecycle.write_run("0.0.0.0", 8080, {"host": "0.0.0.0", "port": 8080, "pid": 999999})
    assert _run(["pi", "--container", "--port", "8080"]) == 0
    assert not any("needs no key" in line for line in env.runs[-1]["summary"])
    lifecycle.write_run("0.0.0.0", 8080, {"host": "0.0.0.0", "port": 8080,
                                          "managed_by": "launchd"})
    assert _run(["pi", "--container", "--port", "8080"]) == 0
    assert _OPEN_BIND.replace("0.0.0.0", "127.0.0.1") in env.runs[-1]["summary"]
    monkeypatch.setattr(launch, "_auth_required", lambda base: True)
    assert _run(["pi", "--container", "--port", "8080"]) == 0
    assert not any("needs no key" in line for line in env.runs[-1]["summary"])


# The env fixture replaces auto_target. This is the function that launch uses.
_auto_target = lifecycle.auto_target


@pytest.mark.parametrize("argv, ports", [
    (["pi", "--container"], [8080]),
    (["pi", "--container", "--port", "8080"], [8080]),
    (["pi", "--container"], [8080, 8081])])
def test_a_launch_runs_no_ps_that_a_share_on_path_holds(env, monkeypatch, argv, ports):
    """A client can write a ps into a read-write share whose folder is on
    PATH. Launch checks the processes of the servers' runfiles, for the open
    bind and to find the one live server, with the system's ps only."""
    bin_ = env.proj / ".venv" / "bin"
    bin_.mkdir(parents=True)
    marker = env.home / "planted-ps-ran"
    (bin_ / "ps").write_text(f"#!/bin/sh\necho \"$@\" >> {marker}\nexec /bin/ps \"$@\"\n")
    (bin_ / "ps").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(lifecycle, "auto_target", _auto_target)
    for port in ports:
        lifecycle.write_run("127.0.0.1", port, {"host": "127.0.0.1", "port": port,
                                                "pid": os.getpid(), "managed_by": "detach"})
    assert _run(argv) == 0
    assert not marker.exists()


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


@pytest.mark.parametrize("dry", [False, True])
def test_the_kernel_check_reads_the_running_service_s_folder(env, capsys, dry):
    """`container system start --app-root ROOT` keeps the kernel in ROOT, and
    `container system status` names ROOT."""
    args = ["pi", "--container", *(["--config-only"] if dry else [])]
    root = env.home / "ext-disk" / "container"
    (root / "kernels").mkdir(parents=True)
    (root / "kernels" / "default.kernel-arm64").write_bytes(b"kernel")
    _remove_kernel(env.home)
    env.update(app_root=str(root))
    assert _run(args) == 0
    shutil.rmtree(root / "kernels")
    _install_kernel(env.home)
    capsys.readouterr()
    assert _run(args) == launch.EXIT_UNAVAILABLE
    assert capsys.readouterr().err == _NO_KERNEL


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


def test_a_first_start_that_never_answers_names_no_kernel_command(env, capsys,
                                                                   monkeypatch):
    """`container system kernel set` needs a service that answers."""
    _remove_kernel(env.home)
    env.update(running=False, start_down=True)
    monkeypatch.setattr(session, "stdin_is_tty", lambda: True)
    assert _run(["pi", "--container"]) == launch.EXIT_UNAVAILABLE
    assert capsys.readouterr().err == (
        "[launch] `container system start` failed (exit 1). The container service does "
        "not answer. Read its log with: container system logs\n")
    assert not env.calls("build")


@pytest.mark.parametrize("elsewhere", [False, True], ids=["alone", "a-build-elsewhere"])
def test_a_warm_launch_repeats_no_container_query(env, monkeypatch, elsewhere):
    """The test runs of another worktree call their fake container as
    `container builder ...`, which the Mac's process list shows while they
    run. The container calls of a launch test do not change with them."""
    import subprocess

    from gmlx.container import images
    real = subprocess.run
    other = {"runs": elsewhere}

    def run(argv, *a, **kw):
        if list(argv) == ["/bin/ps", "-Ao", "command="]:
            return subprocess.CompletedProcess(argv, 0, stdout=(
                "/tmp/other/fakebin/container builder status\n" if other["runs"] else ""))
        return real(argv, *a, **kw)
    monkeypatch.setattr(images.subprocess, "run", run)
    _user_config(env.home, "launch:\n  container:\n    clients:\n      pi:\n"
                           "        volumes: [cache:/root/.cache]\n")
    assert _run(["pi", "--container"]) == 0
    other["runs"] = False
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


def _deliver(signum):
    """Send ``signum`` to this process. Its handler runs between bytecodes."""
    os.kill(os.getpid(), signum)
    for _ in range(1000):
        pass


def _first_of(error):
    """The exception that the first signal raised, which the later ones replaced."""
    while error.__context__ is not None:
        error = error.__context__
    return error


def test_step_8_ignores_the_second_signal_and_raises_on_the_third():
    """A closed window sends a second SIGHUP while the clean-up runs. A third
    signal, and each one after it, stops a clean-up that waits for a
    container service that does not answer."""
    steps = []
    with pytest.raises(lc._Signalled) as last, lc._signals_raise():
        try:
            _deliver(signal.SIGHUP)
        finally:
            _deliver(signal.SIGHUP)
            steps.append("cleaned up")
            try:
                _deliver(signal.SIGTERM)
                steps.append("third ignored")
            finally:
                _deliver(signal.SIGTERM)
                steps.append("fourth ignored")
    assert steps == ["cleaned up"]
    assert last.value.signum == signal.SIGTERM
    # Only a signal after the ignored one ends the clean-ups still to start.
    assert last.value.ends_cleanup and not _first_of(last.value).ends_cleanup


def test_step_8_counts_ctrl_c_with_the_other_signals():
    """Ctrl-C two times during a build must not stop the clean-up that
    records the builder's owed stop."""
    steps = []
    with pytest.raises(KeyboardInterrupt) as last, lc._signals_raise():
        try:
            _deliver(signal.SIGINT)
        finally:
            _deliver(signal.SIGINT)
            steps.append("cleaned up")
            _deliver(signal.SIGINT)
            steps.append("third ignored")
    assert steps == ["cleaned up"]
    assert last.value.ends_cleanup and not _first_of(last.value).ends_cleanup
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


@pytest.mark.parametrize("signum", [signal.SIGHUP, signal.SIGTERM, signal.SIGINT])
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


def test_an_unlisted_profile_reads_no_runfile_in_the_private_home(env, monkeypatch,
                                                                   tmp_path, capsys):
    """The window of an unlisted id@profile comes from the served config,
    which launch reads on the Mac before the handler runs. A reader of
    gmlx state that follows HOME would find the guest's named pipe."""
    home = settings.private_home_path("claude-code", _project(env, "claude-code"))
    pipe = home / ".cache" / "gmlx" / "run.json"
    pipe.parent.mkdir(parents=True)
    os.mkfifo(pipe)
    served = tmp_path / "served.yaml"
    served.write_text("profiles:\n  fast: {load: {max_kv_size: 4096}}\n")
    mac_run = env.home / ".cache" / "gmlx" / "run.json"
    mac_run.parent.mkdir(parents=True)
    mac_run.write_text(json.dumps({"pid": os.getpid(), "config_abspath": str(served)}))
    opened = []

    def read_run(host, port):
        path = Path(os.environ["HOME"]) / ".cache" / "gmlx" / "run.json"
        if path.is_fifo():
            opened.append(path)          # a real read would wait here
            return None
        return json.loads(path.read_text()) if path.is_file() else None
    monkeypatch.setattr(lifecycle, "read_run", read_run)
    assert _run(["claude-code", "--container", "--model", "qwen3.6-27b@fast"]) == 0
    assert opened == []
    assert "because its profile can change it" in capsys.readouterr().out


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


@pytest.mark.parametrize("client", ["pi", "claude-code"])
def test_a_served_config_that_nests_too_deeply_reads_as_none(env, monkeypatch, capsys, client):
    """A client can write the server's config when a session shares its
    folder read-write. Only the handlers of Claude Code and of an agent
    with api: anthropic read it."""
    served = env.proj / "gmlx.yaml"
    served.write_text("[" * 5000 + "]" * 5000 + "\n")
    monkeypatch.setattr(lifecycle, "read_run", lambda h, p: {
        "pid": os.getpid(), "host": h, "port": p, "config_abspath": str(served)})
    real, reads = launch._served_config, []

    def served_config(h, p, **kw):
        # The key lookup reads the file at the server's start for every
        # client. The read that this test counts is the one for the window.
        if not kw.get("at_start"):
            reads.append((h, p))
        return real(h, p, **kw)
    monkeypatch.setattr(launch, "_served_config", served_config)
    assert _run([client, "--container"]) == 0
    assert len(reads) == (client == "claude-code")
    assert real("127.0.0.1", 8080) is None
    assert launch._runfile_key("127.0.0.1", 8080) is None


# Custom agents (launch.agents)

_BOT = ("launch:\n  container:\n    open_browser: false\n  agents:\n    bot:\n"
        "      image: docker.io/me/bot:1\n      command: [bot, --serve]\n")
_BOT_IMG = {"digest": "sha256:" + "b" * 64}


def _agent(env, text=_BOT, **registry):
    env.update(registry={"docker.io/me/bot:1": _BOT_IMG, **registry})
    _user_config(env.home, text)


def test_an_agent_runs_its_image_and_command_with_the_api_variables(env):
    _agent(env)
    assert _run(["bot", "--", "--x"]) == 0
    spec = env.runs[0]["spec"]
    assert spec.command == ["bot", "--serve", "--x"]
    assert spec.image_ref.startswith("docker.io/me/bot@sha256:")
    assert spec.session.client == "agent-bot" and spec.session.name.startswith("gmlx-agent-bot-")
    assert spec.plan.home == settings.private_home_path("agent-bot", env.project)
    assert spec.workdir == os.path.realpath(env.proj)        # mount_cwd is on by default
    names = set(spec.env_names)
    assert {"GMLX_BASE_URL", "GMLX_API_KEY", "GMLX_MODEL", "OPENAI_BASE_URL",
            "OPENAI_API_BASE", "OPENAI_API_KEY"} <= names
    assert not names & {"ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"}
    assert spec.child_env["GMLX_BASE_URL"] == "http://127.0.0.1:8080/v1"
    assert spec.child_env["OPENAI_API_BASE"] == "http://127.0.0.1:8080/v1"
    assert spec.child_env["GMLX_API_KEY"] == spec.child_env["OPENAI_API_KEY"] == (
        "gmlx-container-session")
    assert spec.child_env["GMLX_MODEL"] == "qwen3.6-27b"     # the server's default
    assert not any(k.startswith(("GMLX_", "OPENAI_")) for k in spec.env_values)
    env.runs[0]["server_session"].open()
    assert env.server.posts[-1][1]["client"] == "agent-bot"
    assert env.calls("run")[0][-2:] == ["--check", "bot"]
    assert not (spec.plan.home / ".config").exists()          # no config file written


def test_an_agent_dry_run_shows_the_command_and_no_replaced_line(env, capsys):
    _agent(env)
    assert _run(["bot", "--config-only", "--", "--x"]) == 0
    out = capsys.readouterr().out
    assert "replaces the client's own command" not in out
    assert out.rstrip().endswith("-- bot --serve --x")
    assert "[launch] bot -> http://127.0.0.1:8080/v1  (1 model(s), default qwen3.6-27b)" in out
    bare = out.replace("gmlx-agent-bot-", "").replace("/agent-bot/", "")
    assert "agent-bot" not in bare.replace("gmlx.launch.client=agent-bot", "")


@pytest.mark.parametrize("api, names, absent", [
    ("anthropic", {"ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_MODEL",
                   "ANTHROPIC_SMALL_FAST_MODEL", "CLAUDE_CODE_MAX_CONTEXT_TOKENS"},
     {"ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY"}),
    ("none", set(), {"OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY"}),
])
def test_the_api_setting_chooses_the_variables(env, api, names, absent):
    _agent(env, _BOT + f"      api: {api}\n")
    assert _run(["bot"]) == 0
    spec = env.runs[0]["spec"]
    assert names | {"GMLX_BASE_URL", "GMLX_API_KEY", "GMLX_MODEL"} <= set(spec.env_names)
    assert not absent & set(spec.env_names)
    if api == "anthropic":
        assert spec.child_env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8080"
        assert spec.child_env["ANTHROPIC_API_KEY"] == "gmlx-container-session"
        assert spec.child_env["ANTHROPIC_MODEL"] == spec.child_env["GMLX_MODEL"] == "qwen3.6-27b"
        assert spec.child_env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "65536"


def test_the_model_comes_from_the_flag_the_setting_or_the_server(env, capsys, monkeypatch):
    models = [{"id": "qwen3.6-27b", "default": True}, {"id": "small", "default": False}]
    monkeypatch.setattr(launch, "_http_get_json",
                        lambda url, timeout=5.0, headers=None: {"data": models}
                        if url.endswith("/models") else {})
    _agent(env, _BOT + "      model: small\n      api: anthropic\n")
    assert _run(["bot"]) == 0
    spec = env.runs[0]["spec"]
    assert spec.child_env["GMLX_MODEL"] == spec.child_env["ANTHROPIC_MODEL"] == "small"
    assert "[launch] small stays loaded while idle." in capsys.readouterr().out
    assert _run(["bot", "--model", "qwen3.6-27b", "--no-keep"]) == 0
    assert env.runs[1]["spec"].child_env["GMLX_MODEL"] == "qwen3.6-27b"
    _agent(env, _BOT + "      model: nope\n")
    assert _run(["bot"]) == 1
    assert ("launch.agents.bot.model nope is not a model the server offers"
            in capsys.readouterr().err)
    assert len(env.runs) == 2                                # refused in step 6
    models[0]["default"] = False
    _agent(env, _BOT + "      api: anthropic\n")
    assert _run(["bot"]) == 0
    spec = env.runs[2]["spec"]
    assert not {"GMLX_MODEL", "ANTHROPIC_MODEL", "ANTHROPIC_SMALL_FAST_MODEL"} & set(
        spec.env_names)
    assert ("[launch] the server marks no default model, so bot gets no GMLX_MODEL. Pass "
            "--model, or set launch.agents.bot.model.") in capsys.readouterr().out.splitlines()


def test_a_global_env_entry_that_launch_sets_is_named_once_with_its_block(env, capsys):
    _user_config(env.home, "launch:\n  container:\n    open_browser: false\n"
                           "    env: [IS_SANDBOX=1, UV_CACHE_DIR=/c]\n")
    line = ("[launch] the entry IS_SANDBOX in launch.container.env has no effect for "
            "claude-code, because launch sets IS_SANDBOX in its container.")
    assert _run(["claude-code", "--container"]) == 0
    out = capsys.readouterr().out
    assert line in out and "UV_CACHE_DIR" not in out and "Remove" not in out
    assert _run(["claude-code", "--container"]) == 0
    assert line not in capsys.readouterr().out                 # once
    assert env.runs[0]["spec"].env_values["IS_SANDBOX"] == "1"


def test_an_anthropic_agent_names_itself_in_the_context_window_line(env, capsys):
    _agent(env, _BOT + "      api: anthropic\n"
                       "      env: [CLAUDE_CODE_MAX_CONTEXT_TOKENS=200000]\n")
    assert _run(["bot"]) == 0
    out = capsys.readouterr().out
    assert ("[launch] bot gets CLAUDE_CODE_MAX_CONTEXT_TOKENS=65536, the window of "
            "qwen3.6-27b, in place of your 200000") in out
    assert "Claude Code" not in out
    assert env.runs[0]["spec"].child_env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "65536"


def test_an_anthropic_agent_gets_the_window_of_a_profile_that_keeps_it(env, monkeypatch,
                                                                         tmp_path, capsys):
    """A profile of the server's config that sets no load or cache key keeps
    the window of its base model, as for Claude Code."""
    served = tmp_path / "served.yaml"
    served.write_text("profiles:\n  mine: {sampling: {temperature: 0.3}}\n")
    monkeypatch.setattr(lifecycle, "read_run", lambda h, p: {
        "pid": os.getpid(), "host": h, "port": p, "config_abspath": str(served)})
    _agent(env, _BOT + "      api: anthropic\n")
    assert _run(["bot", "--model", "qwen3.6-27b@mine"]) == 0
    assert env.runs[0]["spec"].child_env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "65536"
    assert "cannot tell the context window" not in capsys.readouterr().out


def test_env_entries_win_over_the_handler_only_with_a_value(env, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    _agent(env, _BOT + "      env: [OPENAI_API_KEY=mine, OPENAI_BASE_URL]\n")
    assert _run(["bot"]) == 0
    spec = env.runs[0]["spec"]
    assert spec.child_env["OPENAI_API_KEY"] == "mine"                  # on purpose
    assert spec.child_env["OPENAI_BASE_URL"] == "http://127.0.0.1:8080/v1"   # not by accident
    assert spec.env_names.count("OPENAI_BASE_URL") == 1


def test_a_second_launch_joins_the_agent_and_notes_no_configured_model(running_agent, capsys):
    env = running_agent
    assert _run(["bot", "--", "--again"]) == 0
    assert env.copies and env.copies[-1][1][-3:] == ["bot", "--serve", "--again"]
    out = capsys.readouterr().out
    assert "[launch] joining the running bot session" in out
    assert "applies only to a new session" not in out       # model: is not a flag
    assert _run(["bot", "--model", "m"]) == 0
    assert "[launch] --model applies only to a new session" in capsys.readouterr().out


@pytest.fixture
def running_agent(env):
    _agent(env, _BOT + "      model: qwen3.6-27b\n")
    lock = session.try_session_lock("agent-bot", env.project)
    proj = os.path.realpath(env.proj)
    session.write_record("agent-bot", env.project, {
        "name": "gmlx-agent-bot-abc123", "workdir": proj, "clipboard": False,
        "shares": [{"host": proj, "guest": proj, "readonly": False}],
        "command": ["bot", "--serve"], "project": proj})
    env.update(containers=[{"name": "gmlx-agent-bot-abc123", "labels": {
        "gmlx.launch": "1", "gmlx.launch.client": "agent-bot",
        "gmlx.launch.project": env.project, "gmlx.launch.pid": str(os.getpid())}}])
    yield env
    lock.release()


def test_a_planted_config_in_the_private_home_is_never_read(env, monkeypatch):
    _agent(env)
    home = settings.private_home("agent-bot", env.project)
    planted = home / ".config" / "gmlx" / "gmlx.yaml"
    planted.parent.mkdir(parents=True)
    planted.write_text("launch:\n  agents:\n    bot:\n      image: docker.io/me/bot:1\n"
                       "      command: [evil]\n")
    import gmlx.config as cfgmod
    real = cfgmod.default_config_paths

    def spy(*a, **kw):
        assert os.environ["HOME"] != str(home), "the config was looked up in the private home"
        return real(*a, **kw)
    monkeypatch.setattr(cfgmod, "default_config_paths", spy)
    assert _run(["bot"]) == 0
    assert env.runs[0]["spec"].command == ["bot", "--serve"]


def test_build_folder_refusals_cover_agents_in_both_directions(env, capsys, tmp_path):
    ctx = env.home / "box"
    ctx.mkdir()
    (ctx / "Containerfile").write_text("FROM debian\n")
    _agent(env, "launch:\n  agents:\n    bot:\n      build: ~/box\n      command: [bot]\n")
    # A client launch may not share the agent's build folder read-write.
    assert _run(["pi", "--container", "--mount", str(ctx)]) == 1
    err = capsys.readouterr().err
    assert "could change the bot build: folder ~/box" in err
    # An agent launch may not share a client's build folder read-write.
    (env.home / "pibox").mkdir()
    (env.home / "pibox" / "Containerfile").write_text("FROM debian\n")
    _agent(env, "launch:\n  container:\n    clients:\n      pi:\n        build: ~/pibox\n"
                "  agents:\n    bot:\n      image: docker.io/me/bot:1\n      command: [bot]\n")
    assert _run(["bot", "--mount", str(env.home / "pibox")]) == 1
    assert "could change the pi build: folder ~/pibox" in capsys.readouterr().err
    assert not env.runs


def test_agent_messages_use_the_agent_config_path(env, capsys):
    _agent(env, "launch:\n  agents:\n    bot:\n      build: box\n      command: [bot]\n")
    assert _run(["bot"]) == 1
    assert "launch.agents.bot.build is 'box'" in capsys.readouterr().err
    _agent(env, "launch:\n  agents:\n    bot:\n      image: docker.io/me/bare:1\n"
                "      command: image\n", **{"docker.io/me/bare:1": {"digest": "sha256:" + "7" * 64}})
    assert _run(["bot"]) == 1
    assert "Set launch.agents.bot.command to the command to run" in capsys.readouterr().err


def test_an_agent_takes_none_of_the_client_special_cases(env):
    from gmlx.commands import launch_container as LC
    plan = settings.resolve_plan("agent-bot", LaunchClientCfg(), cwd=str(env.proj),
                                 mount_cwd=None, cli_mounts=[], network="none",
                                 api_port=8080, web_port=None, build_folders={},
                                 project=env.project, project_volumes=[])
    assert LC._client_env("agent-bot", plan, None, ["bot"], None) == {}
    assert LC._client_env("agent-bot", plan, None, ["bot"], 8501) == {"HOST": "127.0.0.1",
                                                                      "PORT": "8501"}
    home = settings.private_home("agent-bot", env.project)
    settings.ready_home("agent-bot", home)
    assert list(home.iterdir()) == []
    assert launch.web_port_for("agent-bot", 8080) is None


def test_printed_lines_name_the_agent_without_its_key(env, capsys):
    """The key appears only where it is part of a name by design: paths,
    container names, image references, volume names, session folders and
    the log file."""
    _agent(env, _BOT + "      volumes: [data:/data]\n")
    allowed = re.compile(r"gmlx-agent-bot-|launch-agent-bot|/agent-bot/|last-agent-bot-|"
                         r"agent-bot-[0-9a-f]{6}-[0-9a-f]{6}|gmlx\.launch\.client=agent-bot")
    for argv, rc in ([["bot", "--config-only"], 0], [["bot", "--shell"], 0], [["bot"], 0],
                     [["bot", "--remove-home"], 1], [["bot", "--mount", "/nonexistent"], 1]):
        assert _run(argv) == rc, argv
        out = capsys.readouterr()
        summary = env.runs[-1]["summary"] if env.runs else []
        for line in (out.out + out.err).splitlines() + summary:
            if "agent-bot" in line:
                assert allowed.search(line), line


# Runtime agents (runtime: python)

_RT = ("launch:\n  container:\n    open_browser: false\n  agents:\n    ally:\n"
       "      runtime: python\n      command: [python, -m, ally]\n")


def _runtime(env, extra="", **registry):
    _agent(env, _RT + extra, **registry)


def _deps_volume(env):
    return settings.project_volume_name("gmlx-agent-ally-uv", env.project)


# A runtime agent's command runs under the script that syncs with uv and
# then runs the command in place of itself, with the agent's name as $0.
_SYNCED = ["sh", "-c", AGENT_RUN_SCRIPT, "ally"]


def test_a_runtime_agent_runs_under_the_sync_script_with_the_uv_variables_and_its_volume(env):
    _runtime(env)
    assert _run(["ally", "--", "--x"]) == 0
    spec = env.runs[0]["spec"]
    proj = os.path.realpath(env.proj)
    assert spec.command == [*_SYNCED, "python", "-m", "ally", "--x"]
    assert spec.image_ref.startswith("gmlx.invalid/launch-runtime-python@sha256:")
    uv = {k: v for k, v in spec.env_values.items() if k.startswith("UV_")}
    assert uv == {"UV_PROJECT": proj, "UV_PROJECT_ENVIRONMENT": "/opt/agent/venv",
                  "UV_CACHE_DIR": "/opt/agent/cache", "UV_PYTHON_INSTALL_DIR": "/opt/agent/python"}
    assert not any(n.startswith("UV_") for n in spec.env_names)      # by value, not from the Mac
    name = _deps_volume(env)
    assert [(m.source, m.target, m.size) for m in spec.plan.volumes] == [(name, "/opt/agent", "32G")]
    assert env.calls("volume", "create")[0][-1] == name
    assert not any("--check" in c for c in env.calls("run"))         # uv is the runtime's binary
    summary = env.runs[0]["summary"]
    assert any(re.match(r"\[launch\] image gmlx\.invalid/launch-runtime-python:[0-9a-f]+ with uv "
                        r"0\.12\.22", line) for line in summary)
    assert any(line.startswith(f"[launch] volume {name} at /opt/agent (32G limit, ")
               for line in summary)
    assert spec.workdir == proj
    env.runs[0]["server_session"].open()
    assert env.server.posts[-1][1]["client"] == "agent-ally"


def test_a_runtime_agent_dry_run_shows_the_sync_script_and_the_volume(env, capsys):
    _runtime(env)
    assert _run(["ally", "--config-only"]) == 0
    out = capsys.readouterr().out
    assert out.rstrip().endswith(f"-- {shlex.join(_SYNCED)} python -m ally")
    assert "replaces the client's own command" not in out
    assert f"[launch] volume {_deps_volume(env)}: would be created" in out
    assert f"-e UV_PROJECT={os.path.realpath(env.proj)}" in out
    assert "-e UV_PROJECT_ENVIRONMENT=/opt/agent/venv" in out and "-e UV_LOCKED" not in out
    assert f"type=volume,source={_deps_volume(env)},target=/opt/agent" in out


def test_a_source_is_shared_read_only_and_locks_the_environment(env):
    lib = env.home / "src" / "lib"
    lib.mkdir()
    _runtime(env, "      source: ~/src/lib\n")
    assert _run(["ally"]) == 0
    spec = env.runs[0]["spec"]
    real = os.path.realpath(lib)
    assert spec.env_values["UV_PROJECT"] == real and spec.env_values["UV_LOCKED"] == "1"
    assert {(m.source, m.readonly) for m in spec.plan.shares} == {
        (os.path.realpath(env.proj), False), (real, True)}
    assert spec.workdir == os.path.realpath(env.proj)
    assert "[launch] sharing ~/src/lib (read-only, source folder)" in env.runs[0]["summary"]
    # A source inside the working folder adds no share, and uv may update
    # the lock there.
    (env.proj / "sub").mkdir()
    _runtime(env, "      source: ~/src/proj/sub\n")
    assert _run(["ally"]) == 0
    spec = env.runs[1]["spec"]
    assert spec.env_values["UV_PROJECT"] == os.path.realpath(env.proj / "sub")
    assert "UV_LOCKED" not in spec.env_values and len(spec.plan.shares) == 1
    # A source inside a share mounted elsewhere is named by its guest path.
    (env.home / "data" / "proj").mkdir(parents=True)
    _runtime(env, "      source: ~/data/proj\n      mounts: [~/data:/data]\n")
    assert _run(["ally"]) == 0
    assert env.runs[2]["spec"].env_values["UV_PROJECT"] == "/data/proj"


def test_a_source_link_and_an_unshared_project_are_refused_in_step_6(env, capsys):
    lib = env.home / "src" / "lib"
    lib.mkdir()
    (env.home / "src" / "link").symlink_to(lib, target_is_directory=True)
    _runtime(env, "      source: ~/src/link\n")
    assert _run(["ally"]) == 1
    assert "the share ~/src/link is a symbolic link to ~/src/lib" in capsys.readouterr().err
    _runtime(env)
    assert _run(["ally", "--no-mount-cwd"]) == 1
    assert ("the current folder is not shared, so uv has no project to install. Launch with "
            "--mount-cwd, or set launch.agents.ally.source to the project folder."
            ) in capsys.readouterr().err
    assert not env.runs and not env.calls("volume", "create") and not env.calls("build")


def test_the_dependency_volume_is_named_per_project_or_configured(env, capsys):
    _runtime(env)
    assert _run(["ally", "--mount", f"{env.home}/src:/opt/agent/venv"]) == 1
    assert ("~/src cannot use /opt/agent/venv, inside /opt/agent, where uv keeps the agent's "
            "environment") in capsys.readouterr().err
    _runtime(env, "      volumes: [deps:/opt/agent:64G]\n")
    assert _run(["ally"]) == 0
    name = settings.project_volume_name("deps", env.project)
    assert [(m.source, m.target, m.size) for m in env.runs[0]["spec"].plan.volumes] == [
        (name, "/opt/agent", "64G")]
    # The default project keeps the plain name.
    (env.home / "src" / "lib").mkdir()
    _runtime(env, "      mount_cwd: false\n      source: ~/src/lib\n")
    assert _run(["ally"]) == 0
    spec = env.runs[1]["spec"]
    assert [m.source for m in spec.plan.volumes] == ["gmlx-agent-ally-uv"]
    assert spec.plan.project == settings.PROJECT_DEFAULT and spec.workdir == str(spec.plan.home)


def test_network_none_sets_uv_offline(env):
    _runtime(env, "      network: none\n")
    assert _run(["ally"]) == 0
    spec = env.runs[0]["spec"]
    assert spec.env_values["UV_OFFLINE"] == "1" and spec.plan.network == "none"


def test_an_own_image_keeps_the_sync_script_and_checks_uv(env):
    img = {"digest": "sha256:" + "c" * 64}
    _runtime(env, **{"ghcr.io/me/uv:1": img})
    assert _run(["ally", "--image", "ghcr.io/me/uv:1"]) == 0
    spec = env.runs[0]["spec"]
    assert spec.command == [*_SYNCED, "python", "-m", "ally"]
    assert spec.image_ref.startswith("ghcr.io/me/uv@sha256:")
    assert [c[-2:] for c in env.calls("run")] == [["--check", "uv"], ["--check", "sh"]]
    assert spec.env_values["UV_PROJECT"] == os.path.realpath(env.proj)
    _runtime(env, "      image: ghcr.io/me/uv:1\n", **{"ghcr.io/me/uv:1": img})
    assert _run(["ally"]) == 0
    assert env.runs[1]["spec"].command[:4] == _SYNCED
    assert len(env.calls("run")) == 2              # a passed check is remembered per image


@pytest.mark.parametrize("word", ["uv", "sh"])
def test_an_own_image_without_uv_or_sh_is_refused_with_the_runtime_hint(env, capsys, word):
    _runtime(env, "      image: ghcr.io/me/lacks:1\n", **{"ghcr.io/me/lacks:1": {
        "digest": "sha256:" + "d" * 64}})
    env.update(checks={word: [127, f"[launch] {word} is not on the image's PATH (/usr/bin). "
                                   "Install it in the image, or set "
                                   "launch.container.clients.<client>.command."]})
    assert _run(["ally"]) == 1
    assert (f"[launch] {word} is not on the image's PATH (/usr/bin). Install it in the image, "
            "or remove launch.agents.ally.runtime, so the command runs as written, without uv."
            ) in capsys.readouterr().err
    assert not env.runs


def test_the_checked_word_follows_the_image_source(env):
    """uv for a runtime agent, the command's first word for an image with a
    command list, and the ENTRYPOINT for command: image."""
    _agent(env, "launch:\n  agents:\n    bot:\n      image: docker.io/me/bot:1\n"
                "      command: image\n",
           **{"docker.io/me/bot:1": {**_BOT_IMG, "entrypoint": ["/srv/bot"], "cmd": ["--serve"]}})
    assert _run(["bot"]) == 0
    assert env.calls("run")[0][-2:] == ["--check", "/srv/bot"]
    assert env.runs[0]["spec"].command == ["/srv/bot", "--serve"]


def test_a_join_of_a_runtime_agent_runs_the_sync_script_from_the_record(env, capsys):
    _runtime(env)
    lock = session.try_session_lock("agent-ally", env.project)
    proj = os.path.realpath(env.proj)
    session.write_record("agent-ally", env.project, {
        "name": "gmlx-agent-ally-abc123", "workdir": proj, "clipboard": False,
        "shares": [{"host": proj, "guest": proj, "readonly": False}],
        "command": [*_SYNCED, "python", "-m", "ally"], "project": proj})
    env.update(containers=[{"name": "gmlx-agent-ally-abc123", "labels": {
        "gmlx.launch": "1", "gmlx.launch.client": "agent-ally",
        "gmlx.launch.project": env.project, "gmlx.launch.pid": str(os.getpid())}}])
    try:
        assert _run(["ally", "--", "--x"]) == 0
        assert _run(["ally", "--shell"]) == 0
    finally:
        lock.release()
    joined, shell = env.copies[-2][1], env.copies[-1][1]
    assert joined[-8:] == [*_SYNCED, "python", "-m", "ally", "--x"]
    assert shell[-2:] == ["--shell", "--"]
    assert "[launch] joining the running ally session" in capsys.readouterr().out


def test_a_source_agent_without_the_current_folder_uses_one_volume(env, monkeypatch):
    """The docs' advice for an agent with a source: --no-mount-cwd runs it in
    the default project, with one dependency volume from any folder."""
    lib = env.home / "src" / "lib"
    lib.mkdir()
    _runtime(env, "      source: ~/src/lib\n")
    assert _run(["ally"]) == 0
    first = [m.source for m in env.runs[0]["spec"].plan.volumes]
    assert first == [_deps_volume(env)] and first != ["gmlx-agent-ally-uv"]
    for folder in (env.proj, env.home / "src"):
        monkeypatch.chdir(folder)
        assert _run(["ally", "--no-mount-cwd"]) == 0
        spec = env.runs[-1]["spec"]
        assert [m.source for m in spec.plan.volumes] == ["gmlx-agent-ally-uv"]
        assert spec.env_values["UV_PROJECT"] == os.path.realpath(lib)


# AGENT_RUN_SCRIPT runs under the image's /bin/sh, which is dash on Debian.
# Each test runs it under every shell this machine has of the two.
_SHELLS = [sh for sh in ("/bin/sh", "/bin/dash") if os.path.exists(sh)]

# A stand-in for uv. It logs its arguments and UV_LOCKED, prints FAKE_PY for
# `python find`, and exits with FAKE_UV_RC.
_FAKE_UV = """#!/bin/sh
echo "uv $* locked=${UV_LOCKED:-}" >> "$UV_LOG"
case "$1 $2" in
"python find") echo "$FAKE_PY" ;;
esac
exit "${FAKE_UV_RC:-0}"
"""


def _program(path, body: str):
    """An executable shell program at ``path`` that runs ``body``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)
    return path


@pytest.fixture(params=_SHELLS)
def agent_run(request, tmp_path):
    """Run AGENT_RUN_SCRIPT as the container would: in the working folder
    ``work``, with the project folder ``proj`` as UV_PROJECT, the venv
    ``venv`` and the stand-in uv first on PATH. Return the process, its
    output and the uv log."""
    shell = request.param
    _program(tmp_path / "bin" / "uv", _FAKE_UV.split("\n", 1)[1])
    venv = tmp_path / "venv"
    _program(venv / "bin" / "python", 'echo "venv $*"')
    (tmp_path / "work").mkdir()
    (tmp_path / "proj").mkdir()
    log = tmp_path / "uv.log"

    def run(*command, path=None, **extra):
        log.write_text("")
        env = {"PATH": path or f"{tmp_path / 'bin'}:/usr/bin:/bin", "UV_LOG": str(log),
               "UV_PROJECT": str(tmp_path / "proj"), "UV_PROJECT_ENVIRONMENT": str(venv),
               **extra}
        proc = subprocess.Popen([shell, "-c", AGENT_RUN_SCRIPT, "ally", *command],
                                cwd=tmp_path / "work", env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        out, err = proc.communicate(timeout=30)
        return proc, out, err, log.read_text().splitlines()

    return run


def test_the_agent_script_syncs_then_runs_the_command_in_its_place(agent_run, tmp_path):
    venv = tmp_path / "venv"
    proc, out, err, uv = agent_run("sh", "-c", 'echo "$$ $VIRTUAL_ENV ${PATH%%:*} $1"', "x",
                                   "a b")
    assert uv == ["uv sync --inexact locked="]
    assert out == f"{proc.pid} {venv} {venv}/bin a b\n"         # the same process
    assert proc.returncode == 0 and err == ""
    proc, _, _, uv = agent_run("sh", "-c", "exit 7")
    assert proc.returncode == 7 and uv == ["uv sync --inexact locked="]   # at every launch


@pytest.mark.parametrize("value,syncs", [
    ("1", False), ("true", False), ("YES", False), ("on", False), ("t", False), ("y", False),
    ("0", True), ("false", True), ("", True)])
def test_the_agent_script_skips_the_sync_for_a_true_uv_no_sync(agent_run, value, syncs):
    proc, out, _, uv = agent_run("python", "-V", UV_NO_SYNC=value)
    assert out == "venv -V\n" and proc.returncode == 0
    assert uv == (["uv sync --inexact locked="] if syncs else [])


def test_the_agent_script_runs_python_files_as_uv_run_does(agent_run, tmp_path):
    work = tmp_path / "work"
    (work / "plain.py").write_text("print(1)\n")
    (work / "AGENT.PY").write_text("print(1)\n")
    (work / "old.pyc").write_bytes(b"\0")
    assert agent_run("plain.py", "--x")[1] == "venv plain.py --x\n"
    assert agent_run("AGENT.PY")[1] == "venv AGENT.PY\n"
    assert agent_run("old.pyc")[1] == "venv old.pyc\n"
    # A .py name that is no file runs from PATH, as a script that a package
    # installs into the environment's bin folder does.
    _program(tmp_path / "venv" / "bin" / "tool.py", 'echo "tool $*"')
    assert agent_run("tool.py", "a")[1] == "tool a\n"


def test_the_agent_script_looks_up_a_relative_script_in_the_project_folder(agent_run,
                                                                          tmp_path):
    proj, work = tmp_path / "proj", tmp_path / "work"
    (proj / "agent.py").write_text("print(1)\n")
    assert agent_run("agent.py")[1] == f"venv {proj}/agent.py\n"
    (work / "agent.py").write_text("print(1)\n")
    assert agent_run("agent.py")[1] == "venv agent.py\n"            # the working folder first
    _program(proj / "bin" / "start", 'echo "start $*"')
    assert agent_run("bin/start", "a")[1] == "start a\n"
    assert agent_run(str(proj / "bin" / "start"))[1] == "start \n"
    proc, out, err, _ = agent_run("bin/none")             # the shell names the path
    assert proc.returncode != 0 and out == "" and "bin/none" in err


def _script_env(tmp_path):
    """A stand-in for the environment that uv makes for a script block."""
    senv = tmp_path / "senv"
    (senv / "bin").mkdir(parents=True)
    (senv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    return senv, _program(senv / "bin" / "python3", 'echo "$VIRTUAL_ENV ${PATH%%:*} $*"')


def test_the_agent_script_runs_a_script_block_in_its_own_environment(agent_run, tmp_path):
    senv, py = _script_env(tmp_path)
    work = tmp_path / "work"
    (work / "agent.py").write_text("# /// script\n# dependencies = []\n# ///\nprint(1)\n")
    proc, out, err, uv = agent_run("agent.py", "--x", FAKE_PY=str(py), UV_LOCKED="1")
    assert uv == ["uv sync --script agent.py locked=", "uv python find --script agent.py locked="]
    assert out == f"{senv} {senv}/bin agent.py --x\n" and proc.returncode == 0
    assert err == ("[launch] agent.py has no lockfile, so uv installs the dependencies that "
                   "its script block names. Run uv lock --script agent.py to pin them.\n")
    (work / "agent.py.lock").write_text("")
    _, _, err, uv = agent_run("agent.py", FAKE_PY=str(py), UV_LOCKED="1")
    assert err == "" and uv[0] == "uv sync --script agent.py locked=1"
    _, _, _, uv = agent_run("agent.py", FAKE_PY=str(py), UV_NO_SYNC="1")
    assert uv == ["uv python find --script agent.py locked="]


@pytest.mark.parametrize("text,script", [
    ("# /// script\r\n# dependencies = []\r\n# ///\r\n", True),       # CRLF
    ("x = 1\n# /// script\n# ///\n", True),
    ("x = '# /// script'\n# /// script\n# ///\n", False),             # uv reads the first
    ("# /// scripts\n# ///\n", False),
    ("# /// script \n# ///\n", False),
    ("print(1)\n", False)])
def test_the_agent_script_finds_a_script_block_by_uvs_rule(agent_run, tmp_path, text, script):
    _, py = _script_env(tmp_path)
    (tmp_path / "work" / "agent.py").write_text(text, newline="")
    _, out, _, uv = agent_run("agent.py", FAKE_PY=str(py))
    assert (uv[0] == "uv sync --script agent.py locked=") is script
    assert out.startswith("venv agent.py") is not script


def test_the_agent_script_names_a_command_that_the_container_lacks(agent_run):
    proc, out, err, _ = agent_run("research-bot")
    assert proc.returncode == 127 and out == ""
    assert err == ("[launch] ally cannot start, because the container has no command "
                   "research-bot. Check launch.agents.ally.command, and that the project "
                   "installs it.\n")
    _, _, err, _ = agent_run("no\\cbot")                              # printed as given
    assert "has no command no\\cbot. Check" in err


def test_the_agent_script_stops_when_uv_or_grep_fails(agent_run, tmp_path):
    proc, out, _, uv = agent_run("sh", "-c", "echo ran", FAKE_UV_RC="2")
    assert proc.returncode == 2 and out == "" and uv == ["uv sync --inexact locked="]
    (tmp_path / "work" / "agent.py").write_text("print(1)\n")
    proc, out, err, uv = agent_run("agent.py", path=str(tmp_path / "bin"))
    assert proc.returncode == 127 and out == "" and "grep" in err and uv == []


def test_a_client_plan_gets_no_uv_variables(env):
    _user_config(env.home, "launch:\n  container:\n    open_browser: false\n")
    assert _run(["pi", "--container"]) == 0
    spec = env.runs[0]["spec"]
    assert spec.plan.source_guest is None and lc._agent_env(spec.plan) == {}
    assert not any(k.startswith("UV_") for k in spec.env_values)


# --remove-home on a runtime agent

def _ally_state(env, *, home=True, volume=True):
    """A private home and a dependency volume of ally's project, as a
    launch leaves them. The volume's disk image is a small file."""
    name = _deps_volume(env)
    if home:
        settings.private_home("agent-ally", env.project)
    if volume:
        disk = env.home / "disk.img"
        disk.write_bytes(b"x" * 4096)
        env.update(volumes=[{"name": name, "labels": {"gmlx.launch": "1"}, "size": "32G",
                             "bytes": 32 << 30, "source": str(disk)}])
    return name


def _terminal(monkeypatch, *answers):
    monkeypatch.setattr(session, "stdin_is_terminal", lambda: True)
    asked: list[str] = []
    replies = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt: asked.append(prompt) or next(replies))
    return asked


def test_remove_home_of_a_runtime_agent_offers_the_volume_in_one_question(env, capsys,
                                                                           monkeypatch):
    _runtime(env)
    name = _ally_state(env)
    asked = _terminal(monkeypatch, "n", "y")
    home = settings.private_home_path("agent-ally", env.project)
    assert _run(["ally", "--remove-home"]) == 1
    assert asked == [f"[launch] remove the private home of ally for ~/src/proj, 0M at "
                     f"{settings._tilde(str(home))}, with its settings and history, and its "
                     f"dependency volume {name}, under 1M on the Mac? [y/N] "]
    assert "nothing was removed" in capsys.readouterr().out
    assert home.is_dir() and env.load()["volumes"]
    assert _run(["ally", "--remove-home"]) == 0
    out = capsys.readouterr().out
    assert not settings.project_dir_path("agent-ally", env.project).exists()
    assert env.load()["volumes"] == [] and f"[launch] deleted the volume {name}" in out
    assert env.calls("volume", "delete") == [["volume", "delete", name]]


def test_remove_home_offers_the_volume_alone_and_nothing_when_neither_exists(env, capsys,
                                                                              monkeypatch):
    _runtime(env)
    name = _ally_state(env, home=False)
    asked = _terminal(monkeypatch, "n", "y")
    folder = settings.project_dir_path("agent-ally", env.project)
    # The session lock makes the project's folder, and neither answer keeps it.
    assert _run(["ally", "--remove-home"]) == 1
    assert not folder.exists() and env.load()["volumes"]
    assert _run(["ally", "--remove-home"]) == 0
    assert asked == [f"[launch] ally has no private home for ~/src/proj. Delete its dependency "
                     f"volume {name}, under 1M on the Mac? [y/N] "] * 2
    assert env.load()["volumes"] == []
    assert not (settings.data_path() / "agent-ally").exists()
    assert "agent-ally" not in settings.launch_targets_on_disk()
    assert _run(["ally", "--remove-home"]) == 0
    assert capsys.readouterr().out.endswith(
        "[launch] ally has no private home for ~/src/proj, so nothing was removed.\n")
    assert len(asked) == 2


def test_remove_home_never_offers_a_configured_volume(env, monkeypatch):
    _runtime(env, "      volumes: [gmlx-agent-ally-uv:/opt/agent]\n")
    name = _ally_state(env)
    asked = _terminal(monkeypatch, "y")
    assert _run(["ally", "--remove-home"]) == 0
    assert len(asked) == 1 and "volume" not in asked[0]
    assert [v["name"] for v in env.load()["volumes"]] == [name]
    assert not env.calls("volume", "delete")


def test_remove_home_without_a_terminal_names_both_commands(env, capsys):
    _runtime(env)
    name = _ally_state(env)
    folder = settings.project_dir_path("agent-ally", env.project)
    assert _run(["ally", "--remove-home"]) == 1
    assert (f"Remove the home yourself with: rm -rf {folder}\n  and delete the volume with: "
            f"container volume delete {name}") in capsys.readouterr().err
    assert folder.is_dir() and env.load()["volumes"]
    shutil.rmtree(folder)
    assert _run(["ally", "--remove-home"]) == 1
    assert (f"Delete the volume yourself with: container volume delete {name}"
            in capsys.readouterr().err)
    assert env.load()["volumes"]


def test_remove_home_refuses_a_volume_in_use_and_removes_nothing(env, capsys, monkeypatch):
    _terminal(monkeypatch)
    _runtime(env)
    name = _ally_state(env)
    held = session.lock_volumes([settings.Mount(name, "/opt/agent", kind="volume")])
    try:
        assert _run(["ally", "--remove-home"]) == launch.EXIT_TEMPFAIL
        assert f"the volume {name} is in use by another launch session" in capsys.readouterr().err
    finally:
        held[0].release()
    env.update(containers=[{"name": "gmlx-agent-ally-zzz", "state": "running",
                            "volumes": [name], "labels": {"gmlx.launch": "1"}}])
    assert _run(["ally", "--remove-home"]) == launch.EXIT_TEMPFAIL
    assert (f"the running container gmlx-agent-ally-zzz uses the volume {name}"
            in capsys.readouterr().err)
    assert settings.private_home_path("agent-ally", env.project).is_dir()
    assert env.load()["volumes"] and not env.calls("volume", "delete")


def test_a_failed_volume_delete_prints_the_command_after_the_home_is_removed(env, capsys,
                                                                             monkeypatch):
    _terminal(monkeypatch, "y")
    _runtime(env)
    name = _ally_state(env)
    env.update(refuse_volume_delete=[name])
    assert _run(["ally", "--remove-home"]) == 1
    out = capsys.readouterr().out
    assert "[launch] removed " in out and f"[launch] the volume {name} was not deleted: " in out
    assert f"Delete it with: container volume delete {name}" in out
    assert not settings.project_dir_path("agent-ally", env.project).exists()


@pytest.mark.parametrize("service", ["stopped", "not installed"])
def test_remove_home_with_the_service_stopped_removes_the_home_and_says_so(env, capsys,
                                                                            monkeypatch,
                                                                            service):
    asked = _terminal(monkeypatch, "y")
    _runtime(env)
    name = _ally_state(env)
    if service == "stopped":
        env.update(running=False)
    else:
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
    assert _run(["ally", "--remove-home"]) == 0
    out = capsys.readouterr().out
    assert len(asked) == 1 and "volume" not in asked[0]
    assert (f"[launch] the container service is stopped, so the dependency volume {name} was "
            "not looked for. Start it with: container system start. Then run --remove-home "
            f"again, or delete the volume with: container volume delete {name}"
            if service == "stopped" else
            f"[launch] Apple container is not installed, so the dependency volume {name} was "
            "not looked for. Install it with: brew install container. Then run --remove-home "
            "again.") in out
    assert env.load()["volumes"] and not settings.project_dir_path("agent-ally",
                                                                   env.project).exists()


# Agents with a browser interface (web_port)

def test_an_agent_with_web_port_is_a_web_app(env, monkeypatch, capsys):
    """The app listens on web_port in the guest, and the Mac serves it at
    [::1] on a port of the project's own, as it serves a client's app."""
    import webbrowser
    used = []
    monkeypatch.setattr(webbrowser, "open", used.append)
    _agent(env, "launch:\n  agents:\n    bot:\n      image: docker.io/me/bot:1\n"
                "      command: [bot, --serve]\n      web_port: 8501\n      env: [PORT=1]\n")
    assert _run(["bot"]) == 0
    out = "".join(capsys.readouterr())
    assert ("[launch] the entry PORT in launch.agents.bot.env has no effect, because launch "
            "sets PORT in the container for bot. Remove the entry.") in out
    run = env.runs[0]
    spec = run["spec"]
    assert (spec.web_port, spec.web_guest_port) == (3100, 8501)
    assert run["record"]["web"] is True and run["record"]["web_port"] == 3100
    assert web_ports.recorded("agent-bot", env.project) == 3100
    assert spec.env_values["HOST"] == "127.0.0.1" and spec.env_values["PORT"] == "8501"
    assert "PORT" not in spec.env_names                 # the env entry PORT=1 loses
    assert run["opener"] is session.open_in_browser     # open_browser is on by default
    run["server_session"].open()
    assert env.server.posts[-1][1]["web_ports"] == [3100]
    argv = session.compose_run_argv(spec)
    assert "--publish-socket" in argv and argv[argv.index("--unix") + 1].endswith("=8501")
    _agent(env, _BOT + "      web_port: 8501\n")      # open_browser: false
    assert _run(["bot"]) == 0
    assert env.runs[1]["opener"] is None and env.runs[1]["spec"].web_port == 3100
    assert _run(["bot", "--config-only"]) == 0
    assert used == []


def test_an_agent_web_port_equal_to_the_server_port_is_refused_at_both_sites(env, capsys,
                                                                             monkeypatch):
    """The guest reaches the gmlx server on the server's port, so the app
    cannot listen there."""
    _agent(env, _BOT + "      web_port: 8080\n")
    assert _run(["bot"]) == 1
    assert ("[launch] launch.agents.bot.web_port is 8080, the gmlx server's port. Choose "
            "another port for bot's web app.") in capsys.readouterr().err
    assert not env.runs and not env.calls("image", "pull")
    # The server check finds the server on another port than step 6 assumed.
    ports = iter([8080, 8081, 8081, 8081, 8081, 8081])
    monkeypatch.setattr(lifecycle, "auto_target", lambda h, p: ("127.0.0.1", next(ports)))
    _agent(env, _BOT + "      web_port: 8081\n")
    assert _run(["bot"]) == 1
    assert "launch.agents.bot.web_port is 8081, the gmlx server's port" in capsys.readouterr().err
    assert not env.runs
    # A forward equal to the web port is refused as it is for a client.
    _agent(env, _BOT + "      web_port: 8501\n      forward: [8501]\n")
    assert _run(["bot"]) == 1
    assert "forward lists 8501, the web app's own port" in capsys.readouterr().err


def test_targets_with_one_web_port_get_mac_ports_of_their_own(env):
    """Two agents with the same web_port, and a client, run at once, since
    each project of each target gets its own Mac port."""
    lock = _web_session(env, "open-webui", web_port=3100)
    web_ports.choose("open-webui", settings.PROJECT_DEFAULT)
    try:
        _agent(env, _BOT + "      web_port: 3000\n    ann:\n      image: docker.io/me/bot:1\n"
                    "      command: [ann]\n      web_port: 3000\n")
        assert _run(["bot"]) == 0
        assert _run(["ann"]) == 0
    finally:
        lock.release()
    assert [(r["spec"].web_port, r["spec"].web_guest_port) for r in env.runs] == [
        (3101, 3000), (3102, 3000)]


def test_agent_sessions_of_two_projects_run_at_once_and_a_second_launch_opens_its_own(
        env, capsys, monkeypatch):
    opened = []
    monkeypatch.setattr(session, "open_in_browser", opened.append)
    # open_browser stays on, so the second launch opens the running app.
    _agent(env, "launch:\n  agents:\n    bot:\n      image: docker.io/me/bot:1\n"
                "      command: [bot, --serve]\n      web_port: 8501\n")
    other = env.home / "src" / "other"
    other.mkdir()
    real = os.path.realpath(other)
    key = settings.project_id(real)
    web_ports.choose("agent-bot", key)
    lock = _web_session(env, "agent-bot", key, web_port=3100, project=real)
    try:
        assert _run(["bot"]) == 0
        assert env.runs[0]["spec"].web_port == 3101
        monkeypatch.chdir(other)
        assert _run(["bot"]) == 0
        assert opened == ["http://[::1]:3100/"] and len(env.runs) == 1
        assert "[launch] bot is already running at http://[::1]:3100/" in capsys.readouterr().out
    finally:
        lock.release()


# Sessions in the background: --detach, --stop and --list

@pytest.mark.parametrize("argv", [
    ["pi", "--detach", "--stop"], ["--list", "--remove-home"], ["bot", "--detach", "--shell"],
    ["dsh", "--stop", "--config-only"], ["pi", "--stop", "--", "--x"], ["bot", "--list", "--", "x"],
])
def test_session_flags_that_cannot_go_together_are_refused(env, argv):
    _agent(env)
    with pytest.raises(SystemExit):
        _run(argv)
    assert not env.runs


@pytest.mark.parametrize("argv, what", [
    (["pi"], "pi"), (["dsh", "--dsh-profile", "tui"], "the dsh profile tui")])
def test_detach_refuses_a_client_that_needs_a_terminal(env, capsys, argv, what):
    assert _run([*argv, "--detach"]) == 1
    assert capsys.readouterr().err == (
        "[launch] --detach runs Open WebUI, a dsh web profile or a custom agent in the "
        f"background, and {what} needs a terminal. Launch it without --detach.\n")
    assert not env.runs


# A stand-in for the launch that --detach starts. It records how it was
# started, prints a line, and reports on the pipe what FAKE_RC, FAKE_STARTED
# and FAKE_URL ask for. Then it waits until the test kills it.
_FAKE_LAUNCH = """\
import json, os, signal, sys
fd = int(os.environ.pop("GMLX_LAUNCH_DETACH_FD"))
null = os.stat("/dev/null")
with open(os.environ["FAKE_LOG"], "w") as f:
    json.dump({"argv": sys.argv[1:], "pid": os.getpid(), "leader": os.getsid(0) == os.getpid(),
               "stdin_null": os.path.samestat(os.fstat(0), null),
               "out": os.fstat(1).st_ino, "err": os.fstat(2).st_ino}, f)
print("[launch] the fake launch starts", flush=True)
if os.environ.get("FAKE_RC"):
    sys.exit(int(os.environ["FAKE_RC"]))
if os.environ.get("FAKE_STARTED", "1") == "1":
    os.write(fd, b'{"event": "started"}\\n')
if os.environ.get("FAKE_URL"):
    os.write(fd, json.dumps({"event": "answers", "url": os.environ["FAKE_URL"]}).encode()
             + b"\\n")
signal.pause()
"""


@pytest.fixture
def background(env, tmp_path, monkeypatch):
    script = tmp_path / "fake_launch.py"
    script.write_text(_FAKE_LAUNCH)
    log = tmp_path / "fake-launch.json"
    monkeypatch.setattr(procname, "gmlx_argv", lambda exe: [sys.executable, str(script)])
    monkeypatch.setenv("FAKE_LOG", str(log))
    procs: list[subprocess.Popen] = []
    real = lc._follow

    def follow(proc, *a, **kw):
        procs.append(proc)
        return real(proc, *a, **kw)
    monkeypatch.setattr(lc, "_follow", follow)
    yield SimpleNamespace(log=log, procs=procs, info=lambda: json.loads(log.read_text()))
    for proc in procs:
        proc.kill()
        proc.wait()


def _shown_output(client, project):
    return settings._tilde(str(session.output_path(client, project)))


def test_detach_starts_the_launch_again_in_the_background_and_returns_once_the_app_answers(
        env, background, capsys, monkeypatch):
    monkeypatch.setenv("FAKE_URL", "http://[::1]:3100/")
    assert _run(["open-webui", "--detach", "--model", "qwen3.6-27b"]) == 0
    info = background.info()
    assert info["argv"] == ["launch", "open-webui", "--model", "qwen3.6-27b", "--container"]
    assert info["leader"] and info["stdin_null"]
    project = _project(env, "open-webui")
    path = session.output_path("open-webui", project)
    assert info["out"] == info["err"] == os.stat(path).st_ino
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert capsys.readouterr().out == (
        "[launch] the fake launch starts\n"
        "[launch] open-webui runs in the background at http://[::1]:3100/.\n"
        f"[launch] its output goes to {_shown_output('open-webui', project)}. gmlx launch "
        "--list shows the running sessions, and gmlx launch open-webui --stop in this folder "
        "ends this one.\n")
    assert not env.runs
    lock = session.try_session_lock("open-webui", project)       # this launch let go
    assert lock is not None
    lock.release()


def test_detach_of_an_agent_keeps_its_arguments_and_returns_once_its_container_runs(
        env, background, capsys, monkeypatch):
    _agent(env)
    runs = []
    monkeypatch.setattr(lc, "_session_runs", lambda client, project: runs.append(
        (client, project)) or True)
    assert _run(["bot", "--detach", "--", "--wait", "--detach"]) == 0
    assert background.info()["argv"] == ["launch", "bot", "--container", "--", "--wait",
                                         "--detach"]
    assert runs == [("agent-bot", env.project)]
    out = capsys.readouterr().out
    assert "[launch] bot runs in the background for ~/src/proj.\n" in out
    assert "gmlx launch bot --stop in this folder ends this one." in out


def test_a_background_launch_that_fails_passes_on_its_exit_code(env, background, capsys,
                                                                 monkeypatch):
    monkeypatch.setenv("FAKE_RC", "69")
    assert _run(["open-webui", "--detach"]) == 69
    assert capsys.readouterr().out == "[launch] the fake launch starts\n"


@pytest.mark.parametrize("agent", [False, True])
def test_detach_stops_waiting_after_its_limit_and_the_session_goes_on(
        env, background, capsys, monkeypatch, agent):
    monkeypatch.setattr(lc, "DETACH_ANSWER_WAIT", 0.0)
    monkeypatch.setattr(lc, "DETACH_RUN_WAIT", 0.0)
    monkeypatch.setattr(lc, "_session_runs", lambda client, project: False)
    if agent:
        _agent(env)
    assert _run(["bot" if agent else "open-webui", "--detach"]) == 0
    assert background.procs[0].poll() is None                   # it goes on
    out = capsys.readouterr().out
    assert ("[launch] bot runs in the background for ~/src/proj, and its container is still "
            "starting.\n" if agent else "[launch] open-webui runs in the background, and its "
            "web app has not answered yet.\n") in out


def test_ctrl_c_ends_only_the_wait_of_detach(env, background, capsys, monkeypatch):
    def interrupted(*a):
        raise KeyboardInterrupt
    monkeypatch.setattr(lc, "select", SimpleNamespace(select=interrupted))
    monkeypatch.setenv("FAKE_STARTED", "0")
    assert _run(["open-webui", "--detach"]) == 130
    assert background.procs[0].poll() is None
    shown = _shown_output("open-webui", _project(env, "open-webui"))
    assert capsys.readouterr().out.endswith(
        f"[launch] open-webui goes on starting in the background, and its output goes to "
        f"{shown}. gmlx launch --list shows it, and gmlx launch open-webui --stop in this "
        "folder ends it.\n")


def test_detach_refuses_an_output_file_that_is_a_link(env, background, capsys, tmp_path):
    path = session.output_path("open-webui", _project(env, "open-webui"))
    path.parent.mkdir(parents=True, exist_ok=True)
    target = tmp_path / "elsewhere"
    target.write_text("keep")
    path.symlink_to(target)
    assert _run(["open-webui", "--detach"]) == 1
    assert "cannot write the output file" in capsys.readouterr().err
    assert target.read_text() == "keep" and not background.log.exists()


def test_the_output_file_takes_every_write_at_its_end(tmp_path):
    """A launch that still writes to the file after the next detached
    launch empties it leaves no gap of zero bytes."""
    path = tmp_path / "output.log"
    first = session.open_output(path)
    try:
        os.write(first, b"a" * 100)
        second = session.open_output(path)
        try:
            os.write(first, b"old\n")
            os.write(second, b"new\n")
        finally:
            os.close(second)
    finally:
        os.close(first)
    assert path.read_bytes() == b"old\nnew\n"


@pytest.mark.parametrize("client", ["open-webui", "bot"])
def test_a_launch_that_detach_started_reports_its_start_and_the_address(
        env, monkeypatch, client):
    if client == "bot":
        _agent(env)
    read_end, write_end = os.pipe()
    monkeypatch.setenv(lc.DETACH_FD_ENV, str(write_end))
    assert _run([client, "--container"]) == 0
    assert lc.DETACH_FD_ENV not in os.environ                    # the client never sees it
    run = env.runs[0]
    key, project = run["spec"].session.client, run["spec"].session.project
    assert run["record"]["detached"] is True
    assert run["record"]["output"] == str(session.output_path(key, project))
    if client == "open-webui":
        run["on_answer"]("http://[::1]:3100/")
    with os.fdopen(read_end, "rb") as events:
        assert [json.loads(line) for line in events] == [{"event": "started"}] + (
            [{"event": "answers", "url": "http://[::1]:3100/"}] if client == "open-webui"
            else [])


def test_detach_events_go_only_to_a_pipe_and_a_closed_reader_is_no_error(
        env, monkeypatch, tmp_path):
    with open(tmp_path / "f", "w") as f:
        monkeypatch.setenv(lc.DETACH_FD_ENV, str(f.fileno()))
        assert lc._DetachEvents.from_env() is None
    monkeypatch.setenv(lc.DETACH_FD_ENV, "x")
    assert lc._DetachEvents.from_env() is None
    read_end, write_end = os.pipe()
    monkeypatch.setenv(lc.DETACH_FD_ENV, str(write_end))
    events = lc._DetachEvents.from_env()
    assert events is not None and not os.get_inheritable(write_end)
    os.close(read_end)
    events.send("started")
    events.answered("http://[::1]:3100/")


def test_a_second_detach_of_a_web_app_names_the_running_one(env, background, capsys,
                                                            monkeypatch):
    opened = []
    monkeypatch.setattr(session, "open_in_browser", opened.append)
    lock = _web_session(env, "open-webui", web_port=3100)
    try:
        assert _run(["open-webui", "--detach"]) == 0
    finally:
        lock.release()
    assert capsys.readouterr().out == ("[launch] open-webui is already running at "
                                       "http://[::1]:3100/\n")
    assert not background.log.exists()


def test_a_second_detach_of_an_agent_is_refused(env, background, capsys):
    _agent(env)
    proj = os.path.realpath(env.proj)
    lock = _web_session(env, "agent-bot", env.project, web=False, workdir=proj, project=proj,
                        shares=[{"host": proj, "guest": proj, "readonly": False}])
    try:
        assert _run(["bot", "--detach"]) == 1
    finally:
        lock.release()
    assert capsys.readouterr().err == (
        "[launch] bot already runs for ~/src/proj, and --detach starts only a new session. "
        "gmlx launch --list shows the sessions, and gmlx launch bot --stop in this folder "
        "ends this one.\n")
    assert not background.log.exists() and not env.copies


def test_a_second_detach_of_a_session_that_starts_is_busy(env, background, capsys):
    _agent(env)
    lock = session.try_session_lock("agent-bot", env.project)
    proj = os.path.realpath(env.proj)
    session.write_record("agent-bot", env.project, {
        "name": "", "workdir": proj, "starting": True, **session.launch_owner(),
        "shares": [{"host": proj, "guest": proj, "readonly": False}], "project": proj,
        "web": False, "web_port": None})
    try:
        assert _run(["bot", "--detach"]) == launch.EXIT_TEMPFAIL
    finally:
        lock.release()
    assert "is still starting" in capsys.readouterr().err
    assert not background.log.exists()


@pytest.mark.parametrize("flags, stop", [
    (["--no-mount-cwd"], "gmlx launch bot --stop --no-mount-cwd"),
    (["--mount-cwd", "--mount", "~/my data:ro"],
     "gmlx launch bot --stop --mount-cwd --mount '~/my data:ro'"),
])
def test_the_stop_step_after_detach_repeats_the_flags_that_chose_the_project(
        env, background, capsys, monkeypatch, flags, stop):
    _agent(env)
    (env.home / "my data").mkdir()
    monkeypatch.setattr(lc, "_session_runs", lambda client, project: True)
    assert _run(["bot", "--detach", *flags]) == 0
    assert f"{stop} in this folder ends this one." in capsys.readouterr().out


def _launch_stand_in(ignore_term=False):
    """A process that stands for the launch that runs a session. A thread
    reaps it, as the shell reaps a launch."""
    code = ("import signal, sys\n"
            + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_term else "")
            + "print('ready', flush=True)\nsys.stdin.read()\n")
    proc = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None and proc.stdout.readline() == "ready\n"
    threading.Thread(target=proc.wait, daemon=True).start()
    return proc


def _pi_session(env, owner, **record):
    proj = os.path.realpath(env.proj)
    session.write_record("pi", env.project, {
        "name": "gmlx-pi-abc123", "workdir": proj, "clipboard": False,
        "shares": [{"host": proj, "guest": proj, "readonly": False}],
        "command": ["pi"], "project": proj, "pid": owner.pid,
        "pid_start": session._process_start(owner.pid), **record})
    env.update(containers=[{"name": "gmlx-pi-abc123", "labels": {
        "gmlx.launch": "1", "gmlx.launch.client": "pi", "gmlx.launch.project": env.project,
        "gmlx.launch.pid": str(owner.pid)}}])


@pytest.fixture
def owner():
    procs = []

    def start(ignore_term=False):
        procs.append(_launch_stand_in(ignore_term))
        return procs[-1]
    yield start
    for proc in procs:
        proc.kill()
        proc.wait()


@pytest.mark.parametrize("where", ["here", "subfolder"])
def test_stop_ends_the_launch_that_runs_the_session(env, capsys, owner, where):
    """From a subfolder, --stop ends the session a launch from there joins."""
    proc = owner()
    _pi_session(env, proc)
    if where == "subfolder":
        _subfolder(env, "sub")
    assert _run(["pi", "--stop"]) == 0
    assert proc.wait(5) == -signal.SIGTERM
    assert capsys.readouterr().out == "[launch] stopped the pi session for ~/src/proj.\n"
    assert not env.calls("stop") and not env.calls("delete")


def test_stop_never_signals_a_process_that_took_the_launchs_id(env, capsys, owner):
    proc = owner()
    _pi_session(env, proc)
    record = session.read_record("pi", env.project)
    session.write_record("pi", env.project, {**record, "pid_start": record["pid_start"] - 1})
    env.update(delete_removes=True)
    assert _run(["pi", "--stop"]) == 0
    assert proc.poll() is None
    assert env.calls("stop")[0][-1] == env.calls("delete")[0][-1] == "gmlx-pi-abc123"
    assert capsys.readouterr().out == ("[launch] stopped the container gmlx-pi-abc123 of the "
                                       "pi session for ~/src/proj.\n")


def test_stop_stops_the_container_of_a_session_with_no_launch_to_signal(running_session, capsys):
    """A record from an older launch names no start time, and a running
    container of the project may have no record that launch can read."""
    running = running_session.load()["containers"]
    running_session.update(delete_removes=True)
    assert _run(["pi", "--stop"]) == 0
    assert [c[-1] for c in running_session.calls("delete")] == ["gmlx-pi-abc123"]
    session.record_path("pi", running_session.project).unlink()
    running_session.update(containers=running)
    assert _run(["pi", "--stop"]) == 0
    assert [c[-1] for c in running_session.calls("delete")] == ["gmlx-pi-abc123"] * 2
    assert capsys.readouterr().out == (
        "[launch] stopped the container gmlx-pi-abc123 of the pi session for ~/src/proj.\n"
        "[launch] stopped the container gmlx-pi-abc123 of the pi session for ~/src/proj.\n")


def test_stop_gives_up_on_a_launch_that_does_not_end(env, capsys, owner, monkeypatch):
    monkeypatch.setattr(lc, "STOP_WAIT", 0.0)
    proc = owner(ignore_term=True)
    _pi_session(env, proc)
    assert _run(["pi", "--stop"]) == launch.EXIT_TEMPFAIL
    assert proc.poll() is None
    assert capsys.readouterr().err == (
        "[launch] the pi session for ~/src/proj has not ended after 0 s. Stop its container "
        "with: container stop gmlx-pi-abc123\n")


def test_stop_names_the_launch_of_a_session_with_no_container_yet(env, capsys, owner,
                                                                   monkeypatch):
    monkeypatch.setattr(lc, "STOP_WAIT", 0.0)
    proc = owner(ignore_term=True)
    _pi_session(env, proc, name="", starting=True)
    assert _run(["pi", "--stop"]) == launch.EXIT_TEMPFAIL
    assert capsys.readouterr().err == (
        "[launch] the pi session for ~/src/proj has not ended after 0 s. It has no container "
        f"yet. End its launch with: kill -KILL {proc.pid}\n")


def test_stop_says_so_when_the_container_does_not_stop(running_session, capsys):
    assert _run(["pi", "--stop"]) == launch.EXIT_TEMPFAIL
    assert capsys.readouterr().err == (
        "[launch] the container gmlx-pi-abc123 of the pi session for ~/src/proj did not stop. "
        "Stop it with: container stop gmlx-pi-abc123\n")


def test_stop_with_no_session_names_the_folders_where_one_runs(env, capsys):
    assert _run(["pi", "--stop"]) == 0
    assert capsys.readouterr().out == "[launch] no pi session runs for ~/src/proj.\n"
    other = env.home / "src" / "other"
    other.mkdir()
    real = os.path.realpath(other)
    key = settings.project_id(real)
    lock = _web_session(env, "pi", key, web=False, workdir=real, project=real,
                        shares=[{"host": real, "guest": real, "readonly": False}])
    try:
        assert _run(["pi", "--stop"]) == 0
    finally:
        lock.release()
    assert capsys.readouterr().out == (
        "[launch] no pi session runs for ~/src/proj.\n"
        "[launch] to end the pi session for ~/src/other, run gmlx launch pi --stop --mount . "
        "in ~/src/other\n")
    assert not env.calls("stop")


def test_stop_reaches_a_session_of_the_default_project_with_the_step_it_names(env, capsys,
                                                                              owner):
    """A session that --no-mount-cwd started keys the default project, which
    a bare --stop in the project folder does not reach."""
    proc = owner()
    _pi_session(env, proc)
    record = session.read_record("pi", env.project)
    session.remove_record("pi", env.project)
    session.write_record("pi", settings.PROJECT_DEFAULT, {**record, "project": None,
                                                          "workdir": "/root", "shares": []})
    env.update(containers=[{"name": "gmlx-pi-abc123", "labels": {
        "gmlx.launch": "1", "gmlx.launch.client": "pi",
        "gmlx.launch.project": settings.PROJECT_DEFAULT, "gmlx.launch.pid": str(proc.pid)}}])
    assert _run(["pi", "--stop"]) == 0
    assert capsys.readouterr().out == (
        "[launch] no pi session runs for ~/src/proj.\n"
        "[launch] to end the pi session in the default project, run gmlx launch pi --stop "
        "--no-mount-cwd in /\n")
    os.chdir("/")
    assert _run(["pi", "--stop", "--no-mount-cwd"]) == 0
    assert proc.wait(5) == -signal.SIGTERM


def test_list_with_nothing_running(env, capsys):
    assert _run(["--list"]) == 0
    assert capsys.readouterr().out == "[launch] no launch session runs.\n"
    _agent(env)
    assert _run(["bot", "--list"]) == 0
    assert capsys.readouterr().out == "[launch] no bot session runs.\n"


def _dead_pid() -> int:
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


def test_list_shows_each_session_its_output_file_and_leftovers(env, capsys):
    output = str(session.output_path("open-webui", settings.PROJECT_DEFAULT))
    lock = _web_session(env, "open-webui", web_port=3100, detached=True, output=output,
                        **session.launch_owner())
    proj = os.path.realpath(env.proj)
    settings.write_project_record("pi", env.project, proj)
    env.update(containers=[*env.load()["containers"], {"name": "gmlx-pi-old", "labels": {
        "gmlx.launch": "1", "gmlx.launch.client": "pi", "gmlx.launch.project": env.project,
        "gmlx.launch.pid": str(_dead_pid())}}])
    try:
        assert _run(["--list"]) == 0
        everything = capsys.readouterr().out
        assert _run(["open-webui", "--list"]) == 0
        one = capsys.readouterr().out
    finally:
        lock.release()
    started = lc._started_cell(session.launch_owner()["pid_start"])
    assert everything.splitlines() == [
        "TARGET      PROJECT            STATE     LAUNCH    ADDRESS             STARTED",
        f"open-webui  (default project)  running   detached  http://[::1]:3100/  {started}",
        "pi          ~/src/proj         leftover  -         -                   -",
        f"[launch] the open-webui session writes its output to {settings._tilde(output)}",
        "[launch] to end the open-webui session, run gmlx launch open-webui --stop",
        "[launch] gmlx-pi-old is left over from a launch that is gone. Stop it with: container "
        "stop gmlx-pi-old"]
    assert "gmlx-pi-old" not in one and "open-webui  (default project)" in one


def test_the_session_of_an_agent_that_left_the_config_names_its_container(env, capsys):
    _agent(env)
    proj = os.path.realpath(env.proj)
    lock = _web_session(env, "agent-bot", env.project, web=False, workdir=proj, project=proj,
                        shares=[{"host": proj, "guest": proj, "readonly": False}],
                        **session.launch_owner())
    _user_config(env.home, "launch:\n  container:\n    enabled: false\n")
    step = ("[launch] bot is not in launch.agents, so gmlx launch cannot stop its session for "
            "~/src/proj. Stop it with: container stop gmlx-agent-bot-abc123")
    try:
        assert _run(["--list"]) == 0
        assert step in capsys.readouterr().out.splitlines()
        assert _run(["bot", "--list"]) == 0
        assert step in capsys.readouterr().out.splitlines()
        with pytest.raises(SystemExit):
            _run(["bot", "--stop"])
    finally:
        lock.release()
    assert ("Launch keeps the data of an agent bot from before, and gmlx launch bot --list "
            "names the container that runs each of its sessions.") in capsys.readouterr().err


def test_list_and_status_leave_out_the_login_token_of_dsh(env, capsys, monkeypatch):
    monkeypatch.setattr(lc.sys, "platform", "darwin")
    lock = _web_session(env, "dsh", web_port=3101, url="http://[::1]:3101/?token=secret",
                        **session.launch_owner())
    try:
        assert _run(["--list"]) == 0
        out = capsys.readouterr().out
        lines = lc.status_lines()
    finally:
        lock.release()
    assert "secret" not in out and "  http://[::1]:3101/  " in out
    assert lines[0] == "launch session dsh in the default project: running, http://[::1]:3101/"


def test_status_names_the_launch_sessions_and_asks_no_container_without_a_record(
        env, monkeypatch):
    monkeypatch.setattr(lc.sys, "platform", "darwin")
    assert lc.status_lines() == []
    assert not env.calls("ls")
    lock = _web_session(env, "open-webui", web_port=3100, detached=True)
    try:
        assert lc.status_lines() == [
            "launch session open-webui: running, detached, http://[::1]:3100/",
            "  1 launch session - `gmlx launch --list` lists them with the command that ends "
            "each one"]
    finally:
        lock.release()


def test_list_and_status_wait_briefly_for_the_service_and_say_when_it_does_not_answer(
        env, capsys, monkeypatch, owner):
    monkeypatch.setattr(lc.sys, "platform", "darwin")
    _pi_session(env, owner())
    waited = []

    def stuck():
        waited.append(cli._query_timeout)
        raise cli.Stuck("`container ls --all --format` gave no answer in 5 s, so the container "
                        "service may be stuck")
    monkeypatch.setattr(cli, "list_launch_containers", stuck)
    why = ("`container ls --all --format` gave no answer in 5 s, so the container service may "
           f"be stuck. {cli.RESTART_HINT}")
    assert _run(["--list"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[1].split()[:4] == ["pi", "~/src/proj", "unknown", "foreground"]
    assert ("[launch] the container list is not available, so a session can show as unknown, "
            f"and no leftover container is listed: {why}") in out
    lines = lc.status_lines()
    assert lines[:2] == ["launch session pi for ~/src/proj: unknown",
                         f"launch sessions: the container list is not available: {why}"]
    assert waited == [lc.LIST_QUERY_TIMEOUT] * 2


@pytest.fixture
def planted(env, tmp_path, monkeypatch):
    """A container program that a guest wrote in a folder an earlier session
    shared read-write, first on PATH. It leaves a mark when it runs."""
    tools = env.home / "tools"
    (tools / "bin").mkdir(parents=True)
    ran = tmp_path / "planted-ran"
    program = tools / "bin" / "container"
    program.write_text(f"#!/bin/sh\ntouch '{ran}'\nexit 1\n")
    program.chmod(0o755)
    history = settings.shared_history_path()
    history.parent.mkdir(parents=True, exist_ok=True)
    history.write_text(json.dumps({"shared": [os.path.realpath(tools)]}))
    monkeypatch.setenv("PATH", f"{tools / 'bin'}:{os.environ['PATH']}")
    return ran


_PLANTED = ("launch found the container command at ~/tools/bin/container, which lies in "
            "~/tools, a folder an earlier session shared read-write.")


@pytest.mark.parametrize("argv", [["pi", "--stop"], ["--list"], ["pi", "--list"],
                                  ["ally", "--remove-home"]])
def test_a_command_that_starts_no_session_refuses_a_planted_container_program(
        env, planted, capsys, monkeypatch, argv):
    if argv[0] == "ally":
        _runtime(env)
        _ally_state(env)
        _terminal(monkeypatch, "y")
    assert _run(argv) == 1
    assert _PLANTED in capsys.readouterr().err
    assert not planted.exists()
    if argv[0] == "ally":
        assert settings.private_home_path("agent-ally", env.project).is_dir()


def test_status_names_a_planted_container_program_and_runs_it_never(env, planted,
                                                                     monkeypatch):
    monkeypatch.setattr(lc.sys, "platform", "darwin")
    assert lc.status_lines() == []
    lock = _web_session(env, "open-webui", web_port=3100)
    try:
        lines = lc.status_lines()
    finally:
        lock.release()
    assert len(lines) == 1 and lines[0].startswith(f"launch sessions not listed: {_PLANTED}")
    assert not planted.exists()
