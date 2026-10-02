#!/usr/bin/env python3
"""Process layer behind the background `gmlx serve` / `stop` / `restart` / `status`
/ `logs` / `service` (and the menu-bar companion it raises). CPU-only - every external
touchpoint (the child process, the
readiness probe, signals, launchctl) is faked, so no server starts and no real signal
is sent. The point is the *contracts*: PID-identity before signalling, group kill with
SIGTERM (never SIGHUP), child-death fail-fast, and an absolute-interpreter relaunch."""
from __future__ import annotations

import os
import subprocess
import plistlib
import signal
import sys

import pytest

import gmlx.serve.lifecycle as lc  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_state(monkeypatch, tmp_path):
    # Keep runfiles, logs, and the LaunchAgents plist out of the real home/cache.
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


class _FakeProc:
    def __init__(self, pid=4242, poll_value=None):
        self.pid = pid
        self._poll = poll_value
        self.returncode = poll_value

    def poll(self):
        return self._poll


# runfile + target resolution
def test_status_garbage_runfile_notes_answering_process(monkeypatch, capsys):
    # An unparseable runfile must not claim "no managed server" while the port
    # answers /health - the note tells the user something IS running there.
    p = lc.run_path("127.0.0.1", 8080)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("NOT-JSON{{{")
    monkeypatch.setattr(lc, "_health_ok", lambda host, port, timeout=1.5: True)
    rc = lc.status("127.0.0.1", 8080)
    out = capsys.readouterr().out
    assert rc == 3
    assert "no managed server" in out
    assert "IS answering" in out


def test_status_garbage_runfile_dead_port_stays_quiet(monkeypatch, capsys):
    p = lc.run_path("127.0.0.1", 8080)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("NOT-JSON{{{")
    monkeypatch.setattr(lc, "_health_ok", lambda host, port, timeout=1.5: False)
    rc = lc.status("127.0.0.1", 8080)
    out = capsys.readouterr().out
    assert rc == 3
    assert "no managed server" in out
    assert "answering" not in out


def test_runfile_round_trip_and_list():
    lc.write_run("127.0.0.1", 8080, {"pid": 1, "host": "127.0.0.1", "port": 8080})
    assert lc.read_run("127.0.0.1", 8080)["pid"] == 1
    assert lc.read_run("127.0.0.1", 9999) is None
    assert len(lc.list_runs()) == 1


def test_auto_target_single_then_default():
    assert lc.auto_target(None, None) == ("127.0.0.1", 8080)     # nothing -> default
    lc.write_run("0.0.0.0", 9001, {"host": "0.0.0.0", "port": 9001})
    assert lc.auto_target(None, None) == ("0.0.0.0", 9001)       # the single one
    lc.write_run("0.0.0.0", 9002, {"host": "0.0.0.0", "port": 9002})
    assert lc.auto_target(None, None) == ("127.0.0.1", 8080)     # ambiguous -> default
    assert lc.auto_target("0.0.0.0", 9002) == ("0.0.0.0", 9002)  # explicit honoured


def test_auto_target_config_beats_hardcoded_default():
    # With no runfile, a default-location config's port wins over 8080, so a
    # bare status/stop/ps/launch all talk about the server the user configured.
    from pathlib import Path
    cfg_dir = Path(os.environ["HOME"]) / ".config" / "gmlx"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "gmlx.yaml").write_text("server:\n  port: 9123\nmodels: {}\n")
    assert lc.auto_target(None, None) == ("127.0.0.1", 9123)
    assert lc.auto_target(None, 7000) == ("127.0.0.1", 7000)     # explicit still wins
    # A corrupt config must not break the lifecycle verbs - fall back quietly.
    (cfg_dir / "gmlx.yaml").write_text("server: [broken\n")
    assert lc.auto_target(None, None) == ("127.0.0.1", 8080)


def test_bare_status_lists_all_when_multiple(monkeypatch, capsys):
    import gmlx.serve.server as srv
    lc.write_run("127.0.0.1", 9001, {"pid": 11, "host": "127.0.0.1", "port": 9001,
                                     "managed_by": "detach"})
    lc.write_run("127.0.0.1", 9002, {"pid": 22, "host": "127.0.0.1", "port": 9002,
                                     "managed_by": "detach"})
    monkeypatch.setattr(lc, "identity_ok", lambda run: True)
    monkeypatch.setattr(lc, "_health_ok", lambda h, p: True)
    assert srv._cmd_status([]) == 0
    out = capsys.readouterr().out
    assert ":9001" in out and ":9002" in out                     # both reported


def test_bare_stop_refuses_when_multiple(monkeypatch, capsys):
    import gmlx.serve.server as srv
    lc.write_run("127.0.0.1", 9001, {"pid": 11, "host": "127.0.0.1", "port": 9001})
    lc.write_run("127.0.0.1", 9002, {"pid": 22, "host": "127.0.0.1", "port": 9002})
    monkeypatch.setattr(lc, "identity_ok", lambda run: True)     # both live
    monkeypatch.setattr(lc, "pid_alive", lambda pid: True)
    assert srv._cmd_stop([]) == 2
    err = capsys.readouterr().err
    assert "9001" in err and "9002" in err and "--port" in err


# identity (B1): os.kill(pid,0) is not enough - the cmdline must look like ours
def test_identity_ok_dead_pid(monkeypatch):
    monkeypatch.setattr(lc, "pid_alive", lambda pid: False)
    assert lc.identity_ok({"pid": 999, "port": 8080}) is False


def test_identity_ok_reused_pid_is_not_ours(monkeypatch):
    monkeypatch.setattr(lc, "pid_alive", lambda pid: True)
    monkeypatch.setattr(lc, "_proc_cmdline", lambda pid: "/usr/bin/vim notes.txt")
    assert lc.identity_ok({"pid": 999, "port": 8080}) is False


def test_identity_ok_our_server(monkeypatch):
    monkeypatch.setattr(lc, "pid_alive", lambda pid: True)
    monkeypatch.setattr(
        lc, "_proc_cmdline",
        lambda pid: f"{sys.executable} -m gmlx serve --host 127.0.0.1 --port 8080")
    assert lc.identity_ok({"pid": 999, "port": 8080}) is True


@pytest.mark.skipif(not os.path.exists("/bin/ps"), reason="the system has no /bin/ps")
def test_identity_runs_the_system_ps_not_one_on_path(monkeypatch, tmp_path):
    """A folder on PATH can lie in a share that a container client writes.
    The identity checks, which launch and the spawn guard run, use /bin/ps."""
    marker = tmp_path / "planted-ps-ran"
    planted = tmp_path / "share" / "bin" / "ps"
    planted.parent.mkdir(parents=True)
    planted.write_text(f"#!/bin/sh\necho \"$@\" >> {marker}\nexec /bin/ps \"$@\"\n")
    planted.chmod(0o755)
    monkeypatch.setenv("PATH", f"{planted.parent}{os.pathsep}{os.environ.get('PATH', '')}")
    assert lc._proc_cmdline(os.getpid())                  # this process's command
    lc.identity_ok({"pid": os.getpid(), "port": 8080})
    lc.stale_reason({"pid": os.getpid(), "port": 8080})
    lc.write_menubar_run(os.getpid())
    lc.menubar_alive()
    assert not marker.exists()


# child invocation (B4): absolute interpreter so launchd's bare PATH still resolves it
def test_child_argv_is_absolute_interpreter(monkeypatch):
    monkeypatch.setattr(lc.procname, "named_python", lambda: None)
    argv = lc.child_argv(["--config", "/abs/c.yaml"])
    assert argv[0] == os.path.abspath(sys.executable)
    assert os.path.isabs(argv[0])
    assert argv[1:5] == ["-P", "-m", "gmlx", "serve"]
    assert argv[-2:] == ["--config", "/abs/c.yaml"]


# macOS: the daemon runs through the gmlx-named stub so ps / Activity Monitor
# don't show "Python"; the spawn env points the stub back at this venv.
def test_child_argv_prefers_named_stub(monkeypatch):
    monkeypatch.setattr(lc.procname, "named_python", lambda: "/tmp/proc/gmlx")
    argv = lc.child_argv(["--config", "/abs/c.yaml"])
    assert argv[0] == "/tmp/proc/gmlx"
    assert argv[1:5] == ["-P", "-m", "gmlx", "serve"]


def test_child_env_carries_venv_interpreter():
    env = lc.procname.child_env()
    assert env["PYTHONEXECUTABLE"] == os.path.abspath(sys.executable)


def test_child_env_drops_the_pythonpath_entries_that_name_the_current_folder(
        monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", ":/abs/x:rel:")
    assert lc.procname.child_env()["PYTHONPATH"] == "/abs/x"
    monkeypatch.setenv("PYTHONPATH", ":")
    assert "PYTHONPATH" not in lc.procname.child_env()
    # A real child: -P alone still imports a package from the current folder
    # through an empty entry, and the scrubbed environment does not.
    (tmp_path / "planted").mkdir()
    (tmp_path / "planted" / "__init__.py").write_text("")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", ":/nonexistent")
    probe = [sys.executable, "-P", "-c", "import planted"]
    assert subprocess.run(probe, env=dict(os.environ)).returncode == 0
    assert subprocess.run(probe, env=lc.procname.child_env(),
                          stderr=subprocess.DEVNULL).returncode != 0


def test_a_child_path_block_gives_its_path_to_the_server_and_the_menu_bar(monkeypatch):
    """Container launch starts the server and the menu bar in such a block,
    so they get a PATH with no folder that a client can write. Neither
    this process nor another thread gets that PATH."""
    import threading

    monkeypatch.setenv("PATH", "/shared/.venv/bin:/usr/bin:/bin")
    spawned = []

    def fake_popen(argv, **kw):
        spawned.append(kw["env"]["PATH"])
        return _FakeProc(pid=7777)
    monkeypatch.setattr(lc.procname, "named_python", lambda: None)
    monkeypatch.setattr(lc.procname, "menubar_bundle", lambda: None)
    monkeypatch.setattr(lc, "menubar_alive", lambda: False)
    monkeypatch.setattr(lc.subprocess, "Popen", fake_popen)
    other = []
    with lc.procname.child_path("/usr/bin:/bin"):
        assert lc.start_background_nowait(["--config", "/abs/c.yaml"], host="127.0.0.1",
                                          port=8080) is not None
        assert lc.start_menubar(auto=True) == 0
        t = threading.Thread(target=lambda: other.append(lc.procname.child_env()["PATH"]))
        t.start()
        t.join()
        assert os.environ["PATH"] == "/shared/.venv/bin:/usr/bin:/bin"
    assert spawned == ["/usr/bin:/bin", "/usr/bin:/bin"]
    assert other == ["/shared/.venv/bin:/usr/bin:/bin"]
    assert lc.procname.child_env()["PATH"] == "/shared/.venv/bin:/usr/bin:/bin"


@pytest.mark.parametrize("value, holds", [
    (None, False), ("", False), ("/a:/b", False), (":/a", True), ("/a:", True),
    ("rel", True), ("/a::/b", True)])
def test_pythonpath_holds_cwd(monkeypatch, value, holds):
    if value is None:
        monkeypatch.delenv("PYTHONPATH", raising=False)
    else:
        monkeypatch.setenv("PYTHONPATH", value)
    assert lc.procname.pythonpath_holds_cwd() is holds


# The stub copy must survive codesign's in-place rewrite: TCC keys the mic /
# notification grants to the ad-hoc CDHash, so re-copying (-> re-signing) on
# every launch would silently revoke them on every restart.
def test_copy_stub_keeps_signed_copy(monkeypatch, tmp_path):
    src = tmp_path / "python-stub"
    src.write_bytes(b"stub v1")
    monkeypatch.setattr(lc.procname, "_stub_path", lambda: str(src))

    dest = tmp_path / "proc" / "gmlx"
    assert lc.procname._copy_stub(dest) is True
    assert dest.read_bytes() == b"stub v1"

    dest.write_bytes(b"stub v1 + adhoc signature")   # what codesign -f does
    assert lc.procname._copy_stub(dest) is False     # skip: source unchanged
    assert dest.read_bytes() == b"stub v1 + adhoc signature"

    src.write_bytes(b"stub v2!")                     # interpreter upgraded
    assert lc.procname._copy_stub(dest) is True
    assert dest.read_bytes() == b"stub v2!"


# python-build-standalone interpreters (uv-managed pythons) link libpython as
# @executable_path/../lib/libpythonX.Y.dylib - the stub copy needs a sibling
# lib symlink or it aborts in dyld before main().
def test_copy_stub_links_relative_runtime_lib(monkeypatch, tmp_path):
    py = tmp_path / "cpython" / "bin" / "python3.12"
    py.parent.mkdir(parents=True)
    py.write_bytes(b"stub v1")
    lib = tmp_path / "cpython" / "lib"
    lib.mkdir()
    monkeypatch.setattr(lc.procname, "_stub_path", lambda: str(py))

    dest = tmp_path / "cache" / "proc" / "gmlx"
    assert lc.procname._copy_stub(dest) is True
    link = tmp_path / "cache" / "lib"
    assert link.is_symlink() and os.readlink(str(link)) == str(lib)

    # The link is re-ensured even when the copy itself is skipped (a stub
    # copied by an older gmlx predates the symlink entirely).
    link.unlink()
    assert lc.procname._copy_stub(dest) is False
    assert link.is_symlink() and os.readlink(str(link)) == str(lib)


# Interpreter switch (uv -> uv upgrade, or uv -> framework build): the link
# follows the new source, or goes away when the new build has no sibling lib/.
def test_copy_stub_refreshes_stale_runtime_lib_link(monkeypatch, tmp_path):
    old = tmp_path / "old-cpython" / "lib"
    old.mkdir(parents=True)
    dest = tmp_path / "cache" / "proc" / "gmlx"
    link = tmp_path / "cache" / "lib"
    link.parent.mkdir(parents=True)
    link.symlink_to(old)

    py = tmp_path / "cpython" / "bin" / "python3.12"
    py.parent.mkdir(parents=True)
    py.write_bytes(b"stub v2")
    (tmp_path / "cpython" / "lib").mkdir()
    monkeypatch.setattr(lc.procname, "_stub_path", lambda: str(py))
    assert lc.procname._copy_stub(dest) is True
    assert os.readlink(str(link)) == str(tmp_path / "cpython" / "lib")

    fw = tmp_path / "Python.app" / "Contents" / "MacOS" / "Python"
    fw.parent.mkdir(parents=True)
    fw.write_bytes(b"framework stub")
    monkeypatch.setattr(lc.procname, "_stub_path", lambda: str(fw))
    assert lc.procname._copy_stub(dest) is True
    assert not link.is_symlink() and not link.exists()


# A real directory named lib (user data) is never clobbered or replaced.
def test_copy_stub_leaves_real_lib_dir_alone(monkeypatch, tmp_path):
    py = tmp_path / "cpython" / "bin" / "python3.12"
    py.parent.mkdir(parents=True)
    py.write_bytes(b"stub v1")
    (tmp_path / "cpython" / "lib").mkdir()
    monkeypatch.setattr(lc.procname, "_stub_path", lambda: str(py))

    real = tmp_path / "cache" / "lib"
    real.mkdir(parents=True)
    (real / "keep.txt").write_text("mine")
    assert lc.procname._copy_stub(tmp_path / "cache" / "proc" / "gmlx") is True
    assert not real.is_symlink()
    assert (real / "keep.txt").read_text() == "mine"


# launchd boot shim: the plist execs the venv python; the entry point re-execs
# through a freshly-refreshed stub. Refresh-then-exec is the contract - a plist
# pointing at a copied stub crash-loops in dyld after an interpreter swap.
def test_launchd_reexec_refreshes_then_execs(monkeypatch):
    calls = {}
    monkeypatch.delenv("GMLX_LAUNCHD_REEXEC", raising=False)

    def fake_execve(path, argv, env):
        calls["exec"] = (path, argv, env)
        raise SystemExit(99)          # execve never returns; simulate

    monkeypatch.setattr(os, "execve", fake_execve)
    with pytest.raises(SystemExit):
        lc.procname.launchd_reexec(lambda: "/tmp/stub",
                                   ["serve", "--foreground", "--launchd"])
    path, argv, env = calls["exec"]
    assert path == "/tmp/stub"
    assert argv == ["/tmp/stub", "-P", "-m", "gmlx", "serve", "--foreground",
                    "--launchd"]
    assert env["GMLX_LAUNCHD_REEXEC"] == "1"       # exec'd process skips
    assert env["PYTHONEXECUTABLE"] == os.path.abspath(sys.executable)


def test_launchd_reexec_guard_and_degrade(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not exec")

    monkeypatch.setattr(os, "execve", boom)
    # Second pass (post-exec): the guard is consumed and nothing happens.
    monkeypatch.setenv("GMLX_LAUNCHD_REEXEC", "1")
    lc.procname.launchd_reexec(lambda: "/tmp/stub", ["serve"])
    assert "GMLX_LAUNCHD_REEXEC" not in os.environ  # popped: no grandchild leak
    # Refresh failure degrades to running under the venv interpreter.
    lc.procname.launchd_reexec(lambda: None, ["serve"])

    def broken_refresh():
        raise OSError("no cache dir")

    lc.procname.launchd_reexec(broken_refresh, ["serve"])


def test_launchd_reexec_exec_failure_returns(monkeypatch):
    monkeypatch.delenv("GMLX_LAUNCHD_REEXEC", raising=False)

    def fail_execve(path, argv, env):
        raise OSError("exec format error")

    monkeypatch.setattr(os, "execve", fail_execve)
    lc.procname.launchd_reexec(lambda: "/tmp/stub", ["serve"])   # no raise


# The bundle exe's stamp must live outside Contents/: codesign seals the
# bundle's subcomponents and errors out on a foreign file next to the binary.
def test_copy_stub_external_stamp(monkeypatch, tmp_path):
    src = tmp_path / "python-stub"
    src.write_bytes(b"stub v1")
    monkeypatch.setattr(lc.procname, "_stub_path", lambda: str(src))

    dest = tmp_path / "gmlx.app" / "Contents" / "MacOS" / "gmlx"
    stamp = tmp_path / "gmlx.app.src"
    assert lc.procname._copy_stub(dest, stamp=stamp) is True
    assert stamp.exists()
    assert list(dest.parent.iterdir()) == [dest]     # nothing else in MacOS/

    dest.write_bytes(b"stub v1 + adhoc signature")
    assert lc.procname._copy_stub(dest, stamp=stamp) is False


# launchd depends on the bundle (the LaunchAgent references it by
# absolute path), so it lives in Application Support - cache cleaners delete
# ~/.cache. A pre-relocation bundle in the cache is retired.
def test_menubar_bundle_relocates_to_app_support(monkeypatch, tmp_path):
    if sys.platform != "darwin":
        pytest.skip("bundle is macOS-only")
    src = tmp_path / "python-stub"
    src.write_bytes(b"stub v1")
    monkeypatch.setattr(lc.procname, "_stub_path", lambda: str(src))
    old = lc.procname._proc_dir() / "gmlx.app" / "Contents" / "MacOS"
    old.mkdir(parents=True)
    (old / "gmlx").write_bytes(b"old copy")
    (lc.procname._proc_dir() / "gmlx.app.src").write_text("old")

    exe = lc.procname.menubar_bundle()
    assert exe is not None
    assert str(lc.procname._app_dir()) in exe
    with open(exe, "rb") as f:
        assert f.read() == b"stub v1"
    assert os.path.exists(os.path.join(os.path.dirname(exe),
                                       "..", "Info.plist"))
    assert not (lc.procname._proc_dir() / "gmlx.app").exists()
    assert not (lc.procname._proc_dir() / "gmlx.app.src").exists()


def test_the_bundle_is_signed_with_the_system_codesign(monkeypatch, tmp_path):
    """A folder on PATH can lie in a share that a container client writes.
    Launch signs the menu bar bundle when it starts a server, so the copies
    are signed with /usr/bin/codesign, not with the first one on PATH."""
    if sys.platform != "darwin":
        pytest.skip("bundle is macOS-only")
    src = tmp_path / "python-stub"
    src.write_bytes(b"stub v1")
    monkeypatch.setattr(lc.procname, "_stub_path", lambda: str(src))
    signed = []
    monkeypatch.setattr(lc.procname.subprocess, "run",
                        lambda argv, **kw: signed.append(list(argv)))
    assert lc.procname.menubar_bundle() is not None
    assert lc.procname.agent_trampoline() is not None
    assert len(signed) == 2
    assert all(argv[0] == "/usr/bin/codesign" for argv in signed)


# start_background: happy path bakes host/port + writes a `running` runfile
def test_start_background_happy_path(monkeypatch):
    captured = {}

    def fake_popen(argv, **kw):
        captured["argv"] = argv
        captured["kw"] = kw
        return _FakeProc(pid=4242)

    monkeypatch.setattr(lc.procname, "named_python", lambda: None)
    monkeypatch.setattr(lc.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(lc, "_ready", lambda h, p, k=None, expect_pid=None: True)

    rc = lc.start_background(["/abs/m.gguf", "--config", "/abs/c.yaml"],
                             host="127.0.0.1", port=8080,
                             config_abspath="/abs/c.yaml")
    assert rc == 0
    argv = captured["argv"]
    assert argv[0] == os.path.abspath(sys.executable)
    # the child serves in the foreground (it is itself the detached process)
    assert argv[-5:] == ["--host", "127.0.0.1", "--port", "8080", "--foreground"]
    assert captured["kw"]["start_new_session"] is True
    run = lc.read_run("127.0.0.1", 8080)
    assert run["status"] == "running"
    assert run["pid"] == 4242 and run["pgid"] == 4242
    assert run["config_abspath"] == "/abs/c.yaml"
    assert run["managed_by"] == "detach"
    assert run["api_key_set"] is False


# start_background: a child that dies before readiness fails fast and clears the runfile
def test_start_background_child_death_fail_fast(monkeypatch):
    monkeypatch.setattr(lc.subprocess, "Popen",
                        lambda argv, **kw: _FakeProc(pid=4242, poll_value=1))
    # never reached, but prove readiness isn't what returns 0 here
    monkeypatch.setattr(lc, "_ready", lambda h, p, k=None, expect_pid=None: True)
    rc = lc.start_background(["--config", "/abs/c.yaml"], host="127.0.0.1", port=8080)
    assert rc == 1
    assert lc.read_run("127.0.0.1", 8080) is None       # stale runfile cleared


def test_start_background_port_in_use_names_the_port(monkeypatch, capsys):
    monkeypatch.setattr(lc.subprocess, "Popen",
                        lambda argv, **kw: _FakeProc(pid=4242, poll_value=1))
    monkeypatch.setattr(lc, "_ready", lambda h, p, k=None, expect_pid=None: True)
    monkeypatch.setattr(lc, "_log_tail", lambda log, n:
                        "ERROR: [Errno 48] error while attempting to bind on "
                        "address ('127.0.0.1', 9005): address already in use\n")
    rc = lc.start_background(["--config", "/abs/c.yaml"], host="127.0.0.1", port=9005)
    assert rc == 1
    err = capsys.readouterr().err
    assert "port 9005 on 127.0.0.1 is already in use" in err
    assert "gmlx serve --port <N>" in err
    assert "Errno 48" not in err                        # headline replaces raw tail
    assert lc.read_run("127.0.0.1", 9005) is None


@pytest.mark.parametrize("tail, want", [
    ("error: --config: no such file: gmlx.yaml\n",
     "error: server exited (code 2) before it was ready\n"
     "error: --config: no such file: gmlx.yaml\n"),
    ("bind on address ('127.0.0.1', 8080): address already in use\n",
     "error: port 8080 on 127.0.0.1 is already in use - another process is "
     "listening there\n"),
])
def test_a_start_writes_why_it_failed_to_the_stream_it_is_given(
        monkeypatch, capsys, tail, want):
    """The menu bar starts a server from a worker thread, and redirecting
    standard error there would take the other threads' lines too."""
    import io

    monkeypatch.setattr(lc.subprocess, "Popen",
                        lambda argv, **kw: _FakeProc(pid=4242, poll_value=2))
    monkeypatch.setattr(lc, "_log_tail", lambda log, n: tail)
    err = io.StringIO()
    assert lc.launch_detached(["/py", "-m", "gmlx", "serve"], host="127.0.0.1",
                              port=8080, err=err) == 1
    assert err.getvalue().startswith(want)
    assert capsys.readouterr().err == ""


def test_start_background_refuses_when_already_up(monkeypatch):
    lc.write_run("127.0.0.1", 8080, {"pid": 7, "pgid": 7, "host": "127.0.0.1",
                                     "port": 8080, "managed_by": "detach"})
    monkeypatch.setattr(lc, "identity_ok", lambda run: True)
    called = {"popen": False}
    monkeypatch.setattr(lc.subprocess, "Popen",
                        lambda *a, **k: called.__setitem__("popen", True))
    rc = lc.start_background(["--config", "/abs/c.yaml"], host="127.0.0.1", port=8080)
    assert rc == 1
    assert called["popen"] is False                     # never spawned a second one


def test_spawn_refuses_live_but_unhealthy_server(monkeypatch):
    # The guard refuses on identity alone - a live, ours, correct-port server that
    # is still preloading (unhealthy) holds the bind. (Health no longer gates: a
    # second serve must not double-spawn during the first's model load.)
    lc.write_run("127.0.0.1", 8080, {"pid": 7, "pgid": 7, "host": "127.0.0.1",
                                     "port": 8080, "managed_by": "detach"})
    monkeypatch.setattr(lc, "identity_ok", lambda run: True)
    monkeypatch.setattr(lc, "_health_ok", lambda h, p, timeout=1.5: False)
    monkeypatch.setattr(lc.subprocess, "Popen",
                        lambda *a, **k: pytest.fail("must not spawn over a live server"))
    assert lc._spawn_detached(["--config", "/abs/c.yaml"],
                              host="127.0.0.1", port=8080) is None


def test_spawn_records_the_config_as_an_absolute_path(monkeypatch, tmp_path):
    # A relative path in the runfile would name another file when a later
    # command reads it from another folder.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(lc.procname, "named_python", lambda: None)
    monkeypatch.setattr(lc.subprocess, "Popen", lambda argv, **kw: _FakeProc(pid=4243))
    lc._spawn_detached(["--config", "gmlx.yaml"], host="127.0.0.1", port=8080,
                       config_abspath="gmlx.yaml")
    assert lc.read_run("127.0.0.1", 8080)["config_abspath"] == str(tmp_path / "gmlx.yaml")


def test_spawn_records_the_file_a_config_link_names(monkeypatch, tmp_path):
    """Launch reads the recorded config without following a link, so a
    dotfiles link must not hide the server's key from it."""
    import gmlx.commands.launch as launch
    work, dots = tmp_path / "work", tmp_path / "dots"
    work.mkdir()
    dots.mkdir()
    (dots / "gmlx.yaml").write_text("server:\n  api_key: k1\n")
    (work / "gmlx.yaml").symlink_to(dots / "gmlx.yaml")
    monkeypatch.setattr(lc.procname, "named_python", lambda: None)
    seen = {}
    monkeypatch.setattr(lc.subprocess, "Popen",
                        lambda argv, **kw: seen.update(kw) or _FakeProc(pid=os.getpid()))
    lc._spawn_detached(["--config", str(work / "gmlx.yaml")], host="127.0.0.1",
                       port=8080, config_abspath=str(work / "gmlx.yaml"))
    assert lc.read_run("127.0.0.1", 8080)["config_abspath"] == str(dots / "gmlx.yaml")
    assert seen["cwd"] == str(work)                 # the folder the user named
    assert launch._runfile_key("127.0.0.1", 8080) == "k1"


def test_a_retargeted_config_link_still_reloads_the_server(monkeypatch, tmp_path):
    """The server reads its config again through the link it started with,
    so gmlx init, sync-models and pull reach it through the file the link
    names now, and Edit config opens that file."""
    import gmlx.commands.launch as launch
    import gmlx.commands.menubar as mb
    dots = tmp_path / "dots"
    dots.mkdir()
    (dots / "a.yaml").write_text("server:\n  api_key: k1\n")
    (dots / "b.yaml").write_text("server:\n  api_key: k2\n")
    link = tmp_path / "gmlx.yaml"
    link.symlink_to(dots / "a.yaml")
    monkeypatch.setattr(lc.procname, "named_python", lambda: None)
    monkeypatch.setattr(lc.subprocess, "Popen", lambda argv, **kw: _FakeProc(pid=os.getpid()))
    lc._spawn_detached(["serve", "--config", str(link)], host="127.0.0.1", port=8080,
                       config_abspath=str(dots / "a.yaml"))
    run = lc.read_run("127.0.0.1", 8080)
    assert run["config_given"] == str(link)
    link.unlink()
    link.symlink_to(dots / "b.yaml")
    monkeypatch.setattr(lc, "identity_ok", lambda run: True)
    sent = []
    monkeypatch.setattr(lc.os, "kill", lambda pid, sig: sent.append(pid))
    assert lc.reload_config(str(link)) == [("127.0.0.1", 8080, os.getpid())]
    assert lc.reload_config(str(dots / "b.yaml")) == [("127.0.0.1", 8080, os.getpid())]
    assert lc.reload_config(str(dots / "a.yaml")) == []
    assert mb.build_menu_model({"url": "http://127.0.0.1:8080", "reachable": False,
                                "auth_required": False, "resident": [], "error": None},
                               run)["config_path"] == str(
        dots / "b.yaml")
    assert "gmlx pull --config " + str(dots / "b.yaml") in launch.no_models_message(
        "http://127.0.0.1:8080")
    # The server runs with the key of the file it started with.
    assert launch._runfile_key("127.0.0.1", 8080) == "k1"


def test_a_login_start_after_a_retarget_records_the_file_the_server_reads(
        monkeypatch, tmp_path):
    """The menu bar's login start replays the recorded argv, which names the
    config link, and passes the real path of the earlier start. The runfile
    names the file that the link leads to now, which the server reads, and
    the server runs in the link's folder, as at the first start."""
    import gmlx.commands.launch as launch
    import gmlx.commands.menubar as mb
    work, dots = tmp_path / "work", tmp_path / "dots"
    work.mkdir()
    dots.mkdir()
    (dots / "a.yaml").write_text("server:\n  api_key: key-A\n")
    (dots / "b.yaml").write_text("server:\n  api_key: key-B\n")
    link = work / "gmlx.yaml"
    link.symlink_to(dots / "a.yaml")
    monkeypatch.setattr(lc.procname, "named_python", lambda: None)
    folders = []
    monkeypatch.setattr(lc.subprocess, "Popen", lambda argv, **kw: folders.append(
        kw["cwd"]) or _FakeProc(pid=os.getpid()))
    monkeypatch.setattr(lc, "_ready", lambda *a, **k: True)
    monkeypatch.setattr(lc, "_served_model_count", lambda *a, **k: 1)
    monkeypatch.setattr(lc, "_warn_missing_models", lambda *a, **k: None)
    argv = ["/py", "-m", "gmlx", "serve", "--config", str(link), "--host", "127.0.0.1",
            "--port", "8080", "--foreground"]
    assert lc.launch_detached(argv, host="127.0.0.1", port=8080, config_abspath=str(link),
                              api_key_set=True, cwd=str(tmp_path)) == 0
    run = lc.read_run("127.0.0.1", 8080)
    record = {"argv": run["argv"], "host": "127.0.0.1", "port": 8080,
              "config_abspath": run["config_abspath"], "api_key_set": True,
              "cwd": run["cwd"]}
    assert record["config_abspath"] == str(dots / "a.yaml")
    link.unlink()
    link.symlink_to(dots / "b.yaml")
    lc._remove_run("127.0.0.1", 8080)
    assert mb.start_from_record(record, None, "S") == 0
    run = lc.read_run("127.0.0.1", 8080)
    assert run["config_abspath"] == str(dots / "b.yaml")
    assert folders == [str(work), str(work)]
    assert launch._runfile_key("127.0.0.1", 8080) == "key-B"
    assert mb._key_from_config(run) == "key-B"


def test_a_login_start_after_a_retarget_compares_the_models_of_the_file_it_reads(
        monkeypatch, tmp_path, capsys):
    """The notes after a ready start compare the served models with the file
    the server reads, not with the file of the earlier start that the login
    start record names."""
    import gmlx.commands.menubar as mb
    work, dots = tmp_path / "work", tmp_path / "dots"
    work.mkdir()
    dots.mkdir()
    (dots / "a.yaml").write_text("models:\n  m1: {path: /x/m1.gguf}\n"
                                 "  m2: {path: /x/m2.gguf}\n")
    (dots / "b.yaml").write_text("models:\n  m9: {path: /x/m9.gguf}\n")
    link = work / "gmlx.yaml"
    link.symlink_to(dots / "b.yaml")
    served = ["m9"]
    monkeypatch.setattr(lc.procname, "named_python", lambda: None)
    monkeypatch.setattr(lc.subprocess, "Popen", lambda argv, **kw: _FakeProc(pid=os.getpid()))
    monkeypatch.setattr(lc, "_ready", lambda *a, **k: True)
    monkeypatch.setattr(lc, "_served_model_count", lambda *a, **k: len(served))
    monkeypatch.setattr(lc, "get_json", lambda url, **k: {"data": [{"id": i} for i in served]})
    record = {"argv": ["/py", "-m", "gmlx", "serve", "--config", str(link), "--host",
                       "127.0.0.1", "--port", "8080", "--foreground"],
              "host": "127.0.0.1", "port": 8080, "config_abspath": str(dots / "a.yaml"),
              "api_key_set": False, "cwd": str(work)}
    assert mb.start_from_record(record, None, "S") == 0
    out, err = capsys.readouterr()
    assert "configured model" not in out + err
    # The file of the earlier start is gone, and the file the server reads
    # lists no model, so gmlx pull adds one to that file.
    (dots / "a.yaml").unlink()
    (dots / "b.yaml").write_text("models: {}\n")
    served.clear()
    lc._remove_run("127.0.0.1", 8080)
    assert mb.start_from_record(record, None, "S") == 0
    out, err = capsys.readouterr()
    assert f"add a model: gmlx pull <hf:ref> --config {dots / 'b.yaml'}" in out
    assert "is gone" not in out + err


def test_a_background_start_keeps_the_config_file_the_server_records(
        monkeypatch, tmp_path, capsys):
    """A login start that an older gmlx recorded with no config reads the
    default config. The server records that file before it answers, and the
    start marks that runfile as running, so the record stays and the notes
    name that config."""
    import gmlx.commands.menubar as mb
    conf = tmp_path / "home" / ".config" / "gmlx" / "gmlx.yaml"
    conf.parent.mkdir(parents=True)
    conf.write_text("models: {}\n")
    monkeypatch.setattr(lc.procname, "named_python", lambda: None)
    monkeypatch.setattr(lc.subprocess, "Popen", lambda argv, **kw: _FakeProc(pid=os.getpid()))

    def ready(host, port, *a, **k):
        lc.stamp_run(host, port, config_given=str(conf),
                     config_real=os.path.realpath(conf), bare=True)
        return True
    monkeypatch.setattr(lc, "_ready", ready)
    monkeypatch.setattr(lc, "_served_model_count", lambda *a, **k: 0)
    record = {"argv": ["/py", "-m", "gmlx", "serve", "--host", "127.0.0.1", "--port",
                       "8080", "--foreground"],
              "host": "127.0.0.1", "port": 8080, "config_abspath": None,
              "api_key_set": False, "cwd": str(tmp_path)}
    assert mb.start_from_record(record, None, "S") == 0
    run = lc.read_run("127.0.0.1", 8080)
    assert run["status"] == "running"
    assert run["config_abspath"] == os.path.realpath(conf)
    out = capsys.readouterr().out
    assert "add a model: gmlx pull <hf:ref>\n" in out


def test_after_a_reload_through_a_retargeted_link_launch_reads_its_profiles(
        monkeypatch, tmp_path):
    """A reload reads the file that the config link leads to now, and the
    server records that file. Launch then reads the profiles the server
    runs, while the key stays the one of the file the server started with."""
    import gmlx.commands.launch as launch
    import gmlx.serve.server as srv
    from gmlx.config import ConfigError
    dots = tmp_path / "dots"
    dots.mkdir()
    (dots / "a.yaml").write_text("server:\n  api_key: key-A\n"
                                 "profiles:\n  fast: {sampling: {temperature: 0.2}}\n")
    (dots / "b.yaml").write_text("server:\n  api_key: key-B\n"
                                 "profiles:\n  fast: {load: {max_kv_size: 4096}}\n")
    link = tmp_path / "gmlx.yaml"
    link.symlink_to(dots / "a.yaml")
    monkeypatch.setattr(lc.procname, "named_python", lambda: None)
    monkeypatch.setattr(lc.subprocess, "Popen", lambda argv, **kw: _FakeProc(pid=os.getpid()))
    lc._spawn_detached(["serve", "--config", str(link)], host="127.0.0.1", port=8080,
                       config_abspath=str(link))
    assert launch._profile_keeps_window(launch._served_config("127.0.0.1", 8080), "m1@fast")
    link.unlink()
    link.symlink_to(dots / "b.yaml")
    # Until the server reads its config again, it runs the profiles of a.yaml.
    assert launch._served_config("127.0.0.1", 8080)[0] == str(dots / "a.yaml")
    reload = srv._recording_reload(lambda: {"models": 0}, "127.0.0.1", 8080, str(link))
    assert reload() == {"models": 0}
    served = launch._served_config("127.0.0.1", 8080)
    assert served is not None and served[0] == str(dots / "b.yaml")
    assert not launch._profile_keeps_window(served, "m1@fast")
    assert launch._runfile_key("127.0.0.1", 8080) == "key-A"

    def broken():
        raise ConfigError("broken")

    link.unlink()
    link.symlink_to(dots / "a.yaml")
    with pytest.raises(ConfigError):
        srv._recording_reload(broken, "127.0.0.1", 8080, str(link))()
    assert launch._served_config("127.0.0.1", 8080)[0] == str(dots / "b.yaml")
    # A reload of a server with another --config on this port records nothing.
    srv._recording_reload(lambda: {}, "127.0.0.1", 8080, str(dots / "a.yaml"))()
    assert launch._served_config("127.0.0.1", 8080)[0] == str(dots / "b.yaml")


def test_a_server_runs_in_its_config_folder_never_the_launch_folder(monkeypatch, tmp_path):
    """The launch folder may be a share a container client writes, so a
    relative path in the config must not resolve there."""
    share, conf, home = tmp_path / "proj", tmp_path / "conf", tmp_path / "home"
    for d in (share, conf, home):
        d.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(share)
    monkeypatch.setattr(lc.procname, "named_python", lambda: None)
    seen = []
    monkeypatch.setattr(lc.subprocess, "Popen",
                        lambda argv, **kw: seen.append(kw["cwd"]) or _FakeProc(pid=4245))
    config = str(conf / "gmlx.yaml")
    for port, kw, want in ((18081, {"config_abspath": config}, conf),
                           (18082, {"config_abspath": config, "cwd": str(share)}, conf),
                           (18083, {"cwd": str(share)}, share),    # no config
                           (18084, {}, home)):
        lc._spawn_detached(["serve"], host="127.0.0.1", port=port, **kw)
        assert seen[-1] == str(want)
        assert lc.read_run("127.0.0.1", port)["cwd"] == str(want)


def test_restart_replays_the_recorded_folder(monkeypatch):
    lc.write_run("127.0.0.1", 8080, {"pid": 555, "pgid": 555, "host": "127.0.0.1",
                                     "port": 8080, "managed_by": "detach",
                                     "argv": ["serve", "--models-dir", "/abs/models"],
                                     "cwd": "/abs/models"})
    monkeypatch.setattr(lc, "menubar_alive", lambda: False)
    monkeypatch.setattr(lc, "stop", lambda h, p, **kw: 0)
    got = {}
    monkeypatch.setattr(lc, "launch_detached", lambda *a, **kw: got.update(kw) or 0)
    assert lc.restart("127.0.0.1", 8080) == 0
    assert got["cwd"] == "/abs/models"


def _old_run(argv, **kw):
    lc.write_run("127.0.0.1", 8080, {"pid": 555, "pgid": 555, "host": "127.0.0.1",
                                     "port": 8080, "managed_by": "detach",
                                     "argv": ["/py", "-m", "gmlx", "serve", *argv,
                                              "--foreground"], **kw})


def _restart_spies(monkeypatch):
    calls = {"stop": 0, "start": []}
    monkeypatch.setattr(lc, "menubar_alive", lambda: False)
    monkeypatch.setattr(lc, "stop", lambda h, p, **kw: calls.__setitem__(
        "stop", calls["stop"] + 1) or 0)
    monkeypatch.setattr(lc, "launch_detached",
                        lambda argv, **kw: calls["start"].append((argv, kw)) or 0)
    return calls


def test_restart_resolves_an_old_relative_config_in_the_server_folder(
        monkeypatch, tmp_path):
    (tmp_path / "gmlx.yaml").write_text("models: {}\n")
    _old_run(["--config", "gmlx.yaml"], config_abspath="gmlx.yaml")
    monkeypatch.setattr(lc, "identity_ok", lambda run: True)
    monkeypatch.setattr(lc, "process_cwd", lambda pid: str(tmp_path))
    calls = _restart_spies(monkeypatch)
    assert lc.restart("127.0.0.1", 8080) == 0
    want = str(tmp_path / "gmlx.yaml")
    (argv, kw), = calls["start"]
    assert calls["stop"] == 1
    assert argv == ["/py", "-m", "gmlx", "serve", "--config", want, "--foreground"]
    assert kw["config_abspath"] == want


@pytest.mark.parametrize("alive, folder, want", [
    (True, "/no/such/folder",
     "error: /no/such/folder/gmlx.yaml, the config this server started with, is gone, "
     "so the server keeps running. Put the file back, run gmlx stop, then run gmlx "
     "serve --config /no/such/folder/gmlx.yaml --port 8080. If you moved the file to "
     "~/.config/gmlx/gmlx.yaml, run gmlx stop, then run gmlx serve --port 8080.\n"),
    (True, None,
     "error: this server started with --config gmlx.yaml from a folder gmlx cannot "
     "find, so it keeps running. Run gmlx stop, then run gmlx serve --config "
     "<folder>/gmlx.yaml --port 8080, where <folder> is the folder that holds "
     "gmlx.yaml.\n"),
    (False, None,
     "error: this server is not running, and an older gmlx recorded its config as "
     "gmlx.yaml without its folder. Start it with gmlx serve --config "
     "<folder>/gmlx.yaml --port 8080, where <folder> is the folder that holds "
     "gmlx.yaml.\n"),
])
def test_restart_keeps_a_server_whose_old_config_it_cannot_find(
        monkeypatch, capsys, alive, folder, want):
    _old_run(["--config", "gmlx.yaml"], config_abspath="gmlx.yaml")
    monkeypatch.setattr(lc, "identity_ok", lambda run: alive)
    monkeypatch.setattr(lc, "process_cwd", lambda pid: folder)
    calls = _restart_spies(monkeypatch)
    assert lc.restart("127.0.0.1", 8080) == 1
    assert calls == {"stop": 0, "start": []}
    assert capsys.readouterr().err == want


def test_a_gone_config_leads_with_the_move_when_the_user_config_exists(
        monkeypatch, capsys, tmp_path):
    """The menu bar's notification shows only the first 240 characters."""
    moved = tmp_path / "home" / ".config" / "gmlx" / "gmlx.yaml"
    moved.parent.mkdir(parents=True)
    moved.write_text("models: {}\n")
    _old_run(["--config", "gmlx.yaml"], config_abspath="gmlx.yaml")
    monkeypatch.setattr(lc, "identity_ok", lambda run: True)
    monkeypatch.setattr(lc, "process_cwd", lambda pid: "/no/such/folder")
    _restart_spies(monkeypatch)
    assert lc.restart("127.0.0.1", 8080) == 1
    assert capsys.readouterr().err == (
        "error: /no/such/folder/gmlx.yaml, the config this server started with, is "
        "gone, so the server keeps running. If you moved the file to "
        "~/.config/gmlx/gmlx.yaml, run gmlx stop, then run gmlx serve --port 8080. "
        "Otherwise, put the file back, run gmlx stop, then run gmlx serve --config "
        "/no/such/folder/gmlx.yaml --port 8080.\n")


def test_an_old_relative_config_resolves_in_the_server_folder_for_pull(
        monkeypatch, tmp_path):
    monkeypatch.setattr(lc, "identity_ok", lambda run: True)
    monkeypatch.setattr(lc, "process_cwd", lambda pid: str(tmp_path))
    run = {"pid": 555, "config_abspath": "gmlx.yaml"}
    want = str(tmp_path / "gmlx.yaml")
    assert lc.run_config_path(run) == want
    assert lc.pull_config_flag(lc.run_config_path(run)) == f" --config {want}"
    assert lc.run_config_path({**run, "cwd": "/srv"}) == "/srv/gmlx.yaml"
    assert lc.run_config_path({"config_abspath": "/abs/c.yaml"}) == "/abs/c.yaml"
    assert lc.run_config_path({}) is None
    monkeypatch.setattr(lc, "process_cwd", lambda pid: None)
    assert lc.run_config_path(run) == "gmlx.yaml"


def test_zero_models_hint_points_at_the_log_when_configured_models_were_skipped(
        tmp_path):
    conf = tmp_path / "gmlx.yaml"
    conf.write_text("models:\n  a: {path: /no/a.gguf}\n  b: {path: /no/b.gguf}\n")
    assert lc._zero_models_hint(str(conf)) == (
        "0 of 2 configured models loaded - see `gmlx logs` for what was skipped")
    conf.write_text("models: {}\n")
    assert lc._zero_models_hint(str(conf)) == (
        f"add a model: gmlx pull <hf:ref> --config {conf}")


def test_restart_keeps_a_server_whose_config_does_not_load(monkeypatch, capsys,
                                                           tmp_path):
    conf = tmp_path / "gmlx.yaml"
    conf.write_text("container:\n  enabled: true\n")
    _old_run(["--config", str(conf)], config_abspath=str(conf))
    calls = _restart_spies(monkeypatch)
    assert lc.restart("127.0.0.1", 8080) == 1
    assert calls == {"stop": 0, "start": []}
    assert capsys.readouterr().err == (
        f"error: {conf} does not load, so the server keeps running. Fix the file, "
        "then run gmlx restart.\nconfig (top level): unknown key container. Did you "
        "mean launch: container:?\n")


def test_restart_keeps_an_old_server_that_had_no_config(monkeypatch, capsys, tmp_path):
    _old_run([])
    calls = _restart_spies(monkeypatch)
    assert lc.restart("127.0.0.1", 8080) == 1
    assert calls == {"stop": 0, "start": []}
    assert "gmlx serve now needs one, so it keeps running" in capsys.readouterr().err

    conf = tmp_path / "home" / ".config" / "gmlx" / "gmlx.yaml"
    conf.parent.mkdir(parents=True)
    conf.write_text("container:\n  enabled: true\n")
    assert lc.restart("127.0.0.1", 8080) == 1
    assert calls == {"stop": 0, "start": []}
    assert capsys.readouterr().err.startswith(
        f"error: {conf} does not load, so the server keeps running.")

    conf.write_text("models: {}\n")
    assert lc.restart("127.0.0.1", 8080) == 0
    assert calls["stop"] == 1
    (argv, kw), = calls["start"]
    assert argv == ["/py", "-m", "gmlx", "serve", "--config", str(conf), "--foreground"]
    assert kw["config_abspath"] == str(conf)


@pytest.mark.parametrize("flag", [None, "--mmproj", "--draft-gguf", "--adapter"])
def test_restart_keeps_a_server_whose_model_file_is_gone(monkeypatch, capsys, tmp_path,
                                                         flag):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    args = ["/gone/m.gguf"] if flag is None else [str(model), flag, "/gone/x.gguf"]
    _old_run(args)
    calls = _restart_spies(monkeypatch)
    assert lc.restart("127.0.0.1", 8080) == 1
    assert calls == {"stop": 0, "start": []}
    what = "model" if flag is None else f"{flag} file"
    gone = "/gone/m.gguf" if flag is None else "/gone/x.gguf"
    assert capsys.readouterr().err == (
        f"error: {gone}, the {what} this server started with, is gone, so the server "
        "keeps running. Put the file back, then run gmlx restart.\n")


def test_restart_sends_its_errors_to_the_err_stream(monkeypatch, capsys, tmp_path):
    import io
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    _old_run([str(model)])
    calls = _restart_spies(monkeypatch)
    stops = []
    monkeypatch.setattr(lc, "stop", lambda h, p, **kw: stops.append(kw["err"]) or 0)
    err = io.StringIO()
    assert lc.restart("127.0.0.1", 8080, err=err) == 0
    assert stops == [err] and calls["start"][0][1]["err"] is err
    _old_run(["/gone/m.gguf"])
    assert lc.restart("127.0.0.1", 8080, err=err) == 1
    assert "so the server keeps running" in err.getvalue()
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("args", [["--models-dir", "/abs/models"], ["m.gguf"]])
def test_restart_adds_no_config_to_a_start_that_names_its_models(monkeypatch, tmp_path,
                                                                 args):
    if args == ["m.gguf"]:
        (tmp_path / "m.gguf").write_bytes(b"GGUF")
        args = [str(tmp_path / "m.gguf")]
    conf = tmp_path / "home" / ".config" / "gmlx" / "gmlx.yaml"
    conf.parent.mkdir(parents=True)
    conf.write_text("container:\n  enabled: true\n")
    _old_run(args)
    calls = _restart_spies(monkeypatch)
    assert lc.restart("127.0.0.1", 8080) == 0
    (argv, kw), = calls["start"]
    assert argv == ["/py", "-m", "gmlx", "serve", *args, "--foreground"]
    assert kw["config_abspath"] is None


def test_process_cwd_reads_the_folder_of_a_process(tmp_path):
    proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"],
                            cwd=tmp_path, stdin=subprocess.PIPE)
    try:
        assert lc.process_cwd(proc.pid) == os.path.realpath(tmp_path)
    finally:
        proc.communicate(b"")
    assert lc.process_cwd(None) is None


def test_spawn_detached_serializes_and_refuses_second(monkeypatch):
    # Two sequential spawns on the same bind: the first writes the runfile inside
    # the lock; the second reads it and refuses (the serialized check->write that a
    # concurrent race would otherwise interleave). identity_ok is True only once a
    # runfile exists (read_run is None on the first call, so the guard is skipped).
    monkeypatch.setattr(lc, "identity_ok", lambda run: True)
    spawns = []
    monkeypatch.setattr(lc.subprocess, "Popen",
                        lambda *a, **k: spawns.append(1) or _FakeProc(pid=4242))
    first = lc._spawn_detached(["--config", "/abs/c.yaml"], host="127.0.0.1", port=8080)
    assert first is not None and len(spawns) == 1
    assert lc.read_run("127.0.0.1", 8080)["pid"] == 4242
    second = lc._spawn_detached(["--config", "/abs/c.yaml"], host="127.0.0.1", port=8080)
    assert second is None and len(spawns) == 1          # second refused, no 2nd spawn


# menu-bar companion: GUI gate, dedup'd detached spawn, identity check
def test_gui_session_available(monkeypatch):
    monkeypatch.setattr(lc.sys, "platform", "darwin")
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    assert lc.gui_session_available() is True
    monkeypatch.setenv("SSH_CONNECTION", "1.2.3.4 5 6.7.8.9 22")
    assert lc.gui_session_available() is False           # no Aqua session over SSH


def test_gui_session_unavailable_off_macos(monkeypatch):
    monkeypatch.setattr(lc.sys, "platform", "linux")
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    assert lc.gui_session_available() is False


def test_start_menubar_spawns_foreground_child(monkeypatch):
    captured = {}

    def fake_popen(argv, **kw):
        captured["argv"] = argv
        captured["kw"] = kw
        return _FakeProc(pid=7777)

    monkeypatch.setattr(lc, "menubar_alive", lambda: False)
    monkeypatch.setattr(lc.procname, "menubar_bundle", lambda: None)
    monkeypatch.setattr(lc.subprocess, "Popen", fake_popen)
    rc = lc.start_menubar(extra=["--interval", "9"])
    assert rc == 0
    argv = captured["argv"]
    assert argv[0] == os.path.abspath(sys.executable)
    assert argv[1:7] == ["-P", "-m", "gmlx", "launch", "menubar", "--foreground"]
    # No --host/--port pinned: the one bar tracks the primary, not the spawning server.
    assert "--host" not in argv and "--port" not in argv
    assert argv[-2:] == ["--interval", "9"]
    assert captured["kw"]["start_new_session"] is True
    assert captured["kw"]["cwd"] == os.path.expanduser("~")   # never a project share
    import json
    rec = json.loads(lc.menubar_run_path().read_text())
    assert rec["pid"] == 7777                            # single pidfile recorded


def test_start_menubar_noop_when_already_running(monkeypatch):
    monkeypatch.setattr(lc, "menubar_alive", lambda: True)
    called = {"popen": False}
    monkeypatch.setattr(lc.subprocess, "Popen",
                        lambda *a, **k: called.__setitem__("popen", True))
    assert lc.start_menubar() == 0
    assert called["popen"] is False                      # one menu bar per machine


def test_start_menubar_is_global_singleton(monkeypatch):
    """A second start (a second `serve` on any port) must not spawn a second bar."""
    spawns = {"n": 0}

    def fake_popen(argv, **kw):
        spawns["n"] += 1
        return _FakeProc(pid=5555)

    monkeypatch.setattr(lc.procname, "menubar_bundle", lambda: None)
    monkeypatch.setattr(lc.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(lc, "pid_alive", lambda pid: True)
    monkeypatch.setattr(
        lc, "_proc_cmdline",
        lambda pid: f"{sys.executable} -m gmlx launch menubar --foreground")
    assert lc.start_menubar() == 0                        # first serve spawns it
    assert lc.start_menubar() == 0                        # second serve is a no-op
    assert spawns["n"] == 1


def test_menubar_alive_checks_cmdline(monkeypatch):
    lc.write_menubar_run(4242)
    monkeypatch.setattr(lc, "pid_alive", lambda pid: True)
    monkeypatch.setattr(
        lc, "_proc_cmdline",
        lambda pid: f"{sys.executable} -m gmlx launch menubar --foreground")
    assert lc.menubar_alive() is True
    monkeypatch.setattr(lc, "_proc_cmdline", lambda pid: "/usr/bin/vim notes.txt")
    assert lc.menubar_alive() is False                    # a recycled PID isn't ours


def test_menubar_alive_ignore_pid_discounts_own_record(monkeypatch):
    """The detached-start parent records the child's pid before the child's
    already-running check runs; without ignore_pid the child saw itself as a
    running bar and quit at once (no bar from `serve` or `launch menubar`)."""
    lc.write_menubar_run(4242)
    monkeypatch.setattr(lc, "pid_alive", lambda pid: True)
    monkeypatch.setattr(
        lc, "_proc_cmdline",
        lambda pid: f"{sys.executable} -m gmlx launch menubar --foreground")
    assert lc.menubar_alive() is True
    assert lc.menubar_alive(ignore_pid=4242) is False     # own record
    assert lc.menubar_alive(ignore_pid=9999) is True      # someone else's


# stop (B2): kills the process GROUP with SIGTERM - never SIGHUP (the reload signal)
def test_stop_uses_killpg_sigterm(monkeypatch):
    lc.write_run("127.0.0.1", 8080, {"pid": 555, "pgid": 555, "host": "127.0.0.1",
                                     "port": 8080, "managed_by": "detach"})
    monkeypatch.setattr(lc, "identity_ok", lambda run: True)
    monkeypatch.setattr(lc, "_wait_gone", lambda pid, timeout: True)
    sent = []
    monkeypatch.setattr(lc.os, "killpg", lambda pgid, sig: sent.append((pgid, sig)))
    rc = lc.stop("127.0.0.1", 8080)
    assert rc == 0
    assert sent == [(555, signal.SIGTERM)]
    assert all(sig != signal.SIGHUP for _, sig in sent)
    assert lc.read_run("127.0.0.1", 8080) is None       # runfile removed


def test_stop_stale_runfile_does_not_signal(monkeypatch):
    lc.write_run("127.0.0.1", 8080, {"pid": 555, "pgid": 555, "host": "127.0.0.1",
                                     "port": 8080, "managed_by": "detach"})
    monkeypatch.setattr(lc, "identity_ok", lambda run: False)
    sent = []
    monkeypatch.setattr(lc.os, "killpg", lambda pgid, sig: sent.append(sig))
    rc = lc.stop("127.0.0.1", 8080)
    assert rc == 0
    assert sent == []                                   # never signalled a stranger
    assert lc.read_run("127.0.0.1", 8080) is None


def test_stop_launchd_redirects(monkeypatch, capsys):
    lc.write_run("127.0.0.1", 8080, {"pid": None, "host": "127.0.0.1", "port": 8080,
                                     "managed_by": "launchd"})
    sent = []
    monkeypatch.setattr(lc.os, "killpg", lambda pgid, sig: sent.append(sig))
    rc = lc.stop("127.0.0.1", 8080)
    assert rc == 1 and sent == []
    assert "service uninstall" in capsys.readouterr().err


# reload_config: SIGHUP only --config servers running THIS config, identity-checked.
# SIGHUP (not SIGTERM) is the reload signal - the inverse of stop()'s contract above.
def test_reload_config_signals_matching_server(monkeypatch):
    cfg = "/abs/c.yaml"
    lc.write_run("127.0.0.1", 8080, {"pid": 555, "host": "127.0.0.1", "port": 8080,
                                     "config_abspath": cfg})
    monkeypatch.setattr(lc, "identity_ok", lambda run: True)
    sent = []
    monkeypatch.setattr(lc.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    result = lc.reload_config(cfg)
    assert sent == [(555, signal.SIGHUP)]
    assert result == [("127.0.0.1", 8080, 555)]


def test_reload_config_discriminates_among_servers(monkeypatch):
    cfg = "/abs/c.yaml"
    lc.write_run("127.0.0.1", 8080, {"pid": 11, "host": "127.0.0.1", "port": 8080,
                                     "config_abspath": cfg})
    lc.write_run("127.0.0.1", 9000, {"pid": 22, "host": "127.0.0.1", "port": 9000,
                                     "config_abspath": "/abs/other.yaml"})
    monkeypatch.setattr(lc, "identity_ok", lambda run: True)
    sent = []
    monkeypatch.setattr(lc.os, "kill", lambda pid, sig: sent.append(pid))
    assert lc.reload_config(cfg) == [("127.0.0.1", 8080, 11)]
    assert sent == [11]                                # only the one running this config


def test_reload_config_skips_single_model_server(monkeypatch):
    # No config_abspath => started without --config => NO SIGHUP handler installed, so
    # the default disposition would KILL it. The recorded-path gate must never fire.
    lc.write_run("127.0.0.1", 8080, {"pid": 555, "host": "127.0.0.1", "port": 8080})
    monkeypatch.setattr(lc, "identity_ok", lambda run: True)
    sent = []
    monkeypatch.setattr(lc.os, "kill", lambda pid, sig: sent.append(sig))
    assert lc.reload_config("/abs/c.yaml") == []
    assert sent == []


def test_reload_config_skips_stale_pid(monkeypatch):
    lc.write_run("127.0.0.1", 8080, {"pid": 555, "host": "127.0.0.1", "port": 8080,
                                     "config_abspath": "/abs/c.yaml"})
    monkeypatch.setattr(lc, "identity_ok", lambda run: False)   # recycled / dead pid
    sent = []
    monkeypatch.setattr(lc.os, "kill", lambda pid, sig: sent.append(sig))
    assert lc.reload_config("/abs/c.yaml") == []
    assert sent == []                                  # never signalled a stranger


def test_reload_config_swallows_dead_process(monkeypatch):
    # The pid passed identity_ok but vanished before os.kill - best-effort, no raise.
    lc.write_run("127.0.0.1", 8080, {"pid": 555, "host": "127.0.0.1", "port": 8080,
                                     "config_abspath": "/abs/c.yaml"})
    monkeypatch.setattr(lc, "identity_ok", lambda run: True)

    def _gone(pid, sig):
        raise ProcessLookupError

    monkeypatch.setattr(lc.os, "kill", _gone)
    assert lc.reload_config("/abs/c.yaml") == []


# status: process-layer only, needs no key
def test_status_not_running_when_no_runfile(capsys):
    assert lc.status("127.0.0.1", 8080) == 3
    assert "no managed server" in capsys.readouterr().out


def test_status_json_running(monkeypatch, capsys):
    lc.write_run("127.0.0.1", 8080, {"pid": 321, "pgid": 321, "host": "127.0.0.1",
                                     "port": 8080, "url": "http://127.0.0.1:8080",
                                     "managed_by": "detach", "started_at": 0,
                                     "api_key_set": False})
    monkeypatch.setattr(lc, "identity_ok", lambda run: True)
    monkeypatch.setattr(lc, "_health_ok", lambda h, p, timeout=1.5: True)
    import json
    rc = lc.status("127.0.0.1", 8080, as_json=True)
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["running"] is True and out["pid"] == 321


# logs --clear truncates, never unlinks (an unlink under a held fd loses the log)
def test_tail_log_clear_truncates_keeps_file():
    lp = lc.log_path("127.0.0.1", 8080)
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text("line1\nline2\n")
    assert lc.tail_log("127.0.0.1", 8080, clear=True) == 0
    assert lp.exists() and lp.read_text() == ""         # emptied, not removed


# plist rendering (B4): absolute args, KeepAlive dict, no key baked into ProgramArguments
def test_render_plist_round_trip():
    args = lc.child_argv(["--config", "/abs/c.yaml", "--host", "127.0.0.1",
                          "--port", "8080"])
    raw = lc.render_plist("com.gmlx.serve.server.x", args, "/abs/server.log",
                          env={"PATH": "/venv/bin:/usr/bin"}, keepalive=True)
    pl = plistlib.loads(raw)
    assert pl["Label"] == "com.gmlx.serve.server.x"
    assert pl["ProgramArguments"] == args
    assert os.path.isabs(pl["ProgramArguments"][0])
    assert pl["KeepAlive"] == {"SuccessfulExit": False}
    assert pl["StandardOutPath"] == "/abs/server.log"
    assert pl["EnvironmentVariables"]["PATH"].startswith("/venv/bin")
    assert all("api" not in a.lower() for a in pl["ProgramArguments"])


def test_render_plist_no_keepalive():
    pl = plistlib.loads(lc.render_plist("L", ["/bin/x"], "/l", keepalive=False))
    assert pl["KeepAlive"] is False
    assert "WorkingDirectory" not in pl
    pl = plistlib.loads(lc.render_plist("L", ["/bin/x"], "/l", cwd="/abs/conf"))
    assert pl["WorkingDirectory"] == "/abs/conf"


# service install drives launchctl bootstrap with the gui domain (mac-faked)
def test_service_install_launchctl_argv(monkeypatch):
    monkeypatch.setattr(lc.sys, "platform", "darwin")
    runs = []

    class _R:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(argv, **kw):
        runs.append(argv)
        return _R()

    monkeypatch.setattr(lc.subprocess, "run", fake_run)
    rc = lc.service_install(["--config", "/abs/c.yaml"], host="127.0.0.1", port=8080,
                            config_abspath="/abs/c.yaml")
    assert rc == 0
    bootstrap = [a for a in runs if "bootstrap" in a]
    assert bootstrap and bootstrap[0][:3] == ["/bin/launchctl", "bootstrap",
                                              f"gui/{os.getuid()}"]
    assert lc._plist_path("127.0.0.1", 8080).exists()
    run = lc.read_run("127.0.0.1", 8080)
    assert run["managed_by"] == "launchd"
    assert "--foreground" in run["argv"]          # launchd runs serve in the foreground
    assert run["config_given"] == "/abs/c.yaml"


# The plist execs the bundle trampoline (Login Items attribute the agent to
# gmlx.app, not "Python"), never a copied stub: no gmlx code runs before
# launchd's exec, so a pinned stub path crash-loops in dyld after an
# interpreter swap. `serve --launchd` re-execs through a fresh stub instead.
def test_service_install_plist_uses_trampoline_and_shim(monkeypatch):
    monkeypatch.setattr(lc.sys, "platform", "darwin")
    monkeypatch.setattr(lc.procname, "agent_trampoline",
                        lambda: "/app/gmlx.app/Contents/MacOS/gmlx-agent")

    class _R:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(lc.subprocess, "run", lambda *a, **k: _R())
    rc = lc.service_install(["--config", "/abs/c.yaml"], host="127.0.0.1",
                            port=8080, config_abspath="/abs/c.yaml")
    assert rc == 0
    pl = plistlib.loads(lc._plist_path("127.0.0.1", 8080).read_bytes())
    assert pl["ProgramArguments"][0].endswith("gmlx-agent")
    assert pl["ProgramArguments"][1] == "serve"
    assert "--launchd" in pl["ProgramArguments"]
    assert "--foreground" in pl["ProgramArguments"]
    assert "PYTHONEXECUTABLE" not in pl["EnvironmentVariables"]


# bootout unwinds asynchronously: a bootstrap landing mid-unwind fails once,
# then succeeds. The legacy load -w fallback must be verified, not trusted -
# it returns 0 without loading on current macOS.
def test_load_agent_retries_bootstrap_after_bootout_race(monkeypatch):
    monkeypatch.setattr(lc.time, "sleep", lambda s: None)
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)

        class _R:
            stdout = ""
            stderr = "Bootstrap failed: 5: Input/output error"
            returncode = 0
        if argv[1] == "bootstrap":
            _R.returncode = 0 if len([a for a in calls
                                      if a[1] == "bootstrap"]) >= 3 else 5
        return _R()

    monkeypatch.setattr(lc.subprocess, "run", fake_run)
    assert lc._load_agent("com.test", lc._menubar_agent_plist_path()) is None
    assert len([a for a in calls if a[1] == "bootstrap"]) == 3


def test_load_agent_verifies_legacy_load_fallback(monkeypatch):
    monkeypatch.setattr(lc.time, "sleep", lambda s: None)

    def fake_run(argv, **kw):
        class _R:
            stdout = ""
            stderr = "Bootstrap failed: 5: Input/output error"
            # load -w lies with 0; print says not loaded; bootstrap fails
            returncode = 0 if argv[1] == "load" else 5
        return _R()

    monkeypatch.setattr(lc.subprocess, "run", fake_run)
    err = lc._load_agent("com.test", lc._menubar_agent_plist_path())
    assert err and "Bootstrap failed" in err


def test_agent_entry_falls_back_to_venv_python(monkeypatch):
    monkeypatch.setattr(lc.procname, "agent_trampoline", lambda: None)
    assert lc._agent_entry() == [os.path.abspath(sys.executable),
                                 "-P", "-m", "gmlx"]


# The trampoline: a signed sh script inside the bundle. It must exec the
# BUNDLE BINARY, not the venv python - TCC pins the process identity at the
# first non-platform exec, so a python in the middle makes every permission
# prompt say "python3.12". The venv python is only the stale-copy fallback.
def test_agent_trampoline_execs_bundle_binary_first(monkeypatch, tmp_path):
    if sys.platform != "darwin":
        pytest.skip("bundle is macOS-only")
    src = tmp_path / "python-stub"
    src.write_bytes(b"stub v1")
    monkeypatch.setattr(lc.procname, "_stub_path", lambda: str(src))
    signed = []
    monkeypatch.setattr(
        lc.procname.subprocess, "run",
        lambda argv, **kw: signed.append(list(argv)) or None)

    tramp = lc.procname.agent_trampoline()
    assert tramp is not None and tramp.endswith("gmlx-agent")
    assert os.path.dirname(tramp).endswith("Contents/MacOS")
    exe = os.path.join(os.path.dirname(tramp), "gmlx")
    with open(tramp) as f:
        body = f.read()
    assert body.startswith("#!/bin/sh\n")
    assert f'BIN="{exe}"' in body
    assert 'exec "$BIN" -P -m gmlx "$@"' in body       # TCC pins here
    assert body.index('exec "$BIN"') < body.index('exec "$PY"')
    assert f'PY="{os.path.abspath(sys.executable)}"' in body
    assert 'export PYTHONEXECUTABLE="$PY"' in body      # before the probe
    assert body.index("PYTHONEXECUTABLE") < body.index('if "$BIN" -c ""')
    assert 'export GMLX_LAUNCHD_REEXEC=1' in body      # happy path: no re-exec
    assert 'exec "$PY" -P -m gmlx "$@"' in body        # stale-copy fallback
    assert any(tramp in argv for argv in signed)        # script got signed

    signed.clear()
    assert lc.procname.agent_trampoline() == tramp      # unchanged: no re-sign
    assert not any(tramp in argv for argv in signed)


def _happy_launchctl(monkeypatch, runs=None):
    class _R:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(argv, **kw):
        if runs is not None:
            runs.append(argv)
        return _R()

    monkeypatch.setattr(lc.subprocess, "run", fake_run)
    monkeypatch.setattr(lc.procname, "agent_trampoline",
                        lambda: "/app/gmlx.app/Contents/MacOS/gmlx-agent")


_AUTOSTART_ARGV = ["/old/stub", "-m", "gmlx", "serve",
                   "--config", "/abs/c.yaml", "--foreground"]


def _fake_start_background(monkeypatch, calls):
    def fake(serve_args, *, host, port, config_abspath=None, **kw):
        calls.append(list(serve_args))
        lc.write_run(host, port, {
            "pid": 4242, "managed_by": "detach", "host": host, "port": port,
            "argv": list(_AUTOSTART_ARGV), "config_abspath": config_abspath,
            "api_key_set": True})
        return 0
    monkeypatch.setattr(lc, "start_background", fake)


def test_service_install_menubar_starts_server_and_records_autostart(monkeypatch):
    import gmlx.commands.menubar as mb
    monkeypatch.setattr(lc, "_require_macos", lambda what: 0)
    _happy_launchctl(monkeypatch)
    started = []
    _fake_start_background(monkeypatch, started)
    monkeypatch.setattr(lc, "stop_menubar", lambda: False)

    rc = lc.service_install_menubar(["--config", "/abs/c.yaml"],
                                    host="127.0.0.1", port=8080,
                                    config_abspath="/abs/c.yaml")
    assert rc == 0
    assert started == [["--config", "/abs/c.yaml"]]   # server brought up now
    pl = plistlib.loads(lc._menubar_agent_plist_path().read_bytes())
    assert pl["Label"] == lc.MENUBAR_AGENT_LABEL
    assert pl["ProgramArguments"] == [
        "/app/gmlx.app/Contents/MacOS/gmlx-agent",
        "launch", "menubar", "--foreground", "--launchd"]
    auto = mb.load_menubar_settings()["autostart"]
    assert auto["argv"] == _AUTOSTART_ARGV            # replayed at login
    assert auto["host"] == "127.0.0.1" and auto["port"] == 8080
    assert auto["config_abspath"] == "/abs/c.yaml"
    assert auto["api_key_set"] is True


def test_service_install_menubar_no_autostart(monkeypatch):
    import gmlx.commands.menubar as mb
    monkeypatch.setattr(lc, "_require_macos", lambda what: 0)
    _happy_launchctl(monkeypatch)
    _fake_start_background(monkeypatch, [])
    monkeypatch.setattr(lc, "stop_menubar", lambda: False)
    rc = lc.service_install_menubar([], host="127.0.0.1", port=8080,
                                    autostart=False)
    assert rc == 0
    assert mb.load_menubar_settings()["autostart"] is None
    assert lc._menubar_agent_plist_path().exists()    # bar still installs


def test_service_install_menubar_keeps_running_server(monkeypatch):
    # A healthy server already on the bind is adopted, not restarted.
    import gmlx.commands.menubar as mb
    monkeypatch.setattr(lc, "_require_macos", lambda what: 0)
    _happy_launchctl(monkeypatch)
    monkeypatch.setattr(lc, "stop_menubar", lambda: False)
    lc.write_run("127.0.0.1", 8080, {
        "pid": 1, "managed_by": "detach", "host": "127.0.0.1", "port": 8080,
        "argv": list(_AUTOSTART_ARGV), "config_abspath": "/abs/c.yaml"})
    monkeypatch.setattr(lc, "identity_ok", lambda run: True)

    def boom(*a, **k):
        raise AssertionError("must not restart a healthy server")

    monkeypatch.setattr(lc, "start_background", boom)
    assert lc.service_install_menubar([], host="127.0.0.1", port=8080) == 0
    assert mb.load_menubar_settings()["autostart"]["argv"] == _AUTOSTART_ARGV


def test_service_install_menubar_refuses_an_old_relative_config(monkeypatch, capsys):
    # The runfile argv becomes the login record, and a login start runs in /.
    import gmlx.commands.menubar as mb
    monkeypatch.setattr(lc, "_require_macos", lambda what: 0)
    lc.write_run("127.0.0.1", 8081, {
        "pid": 1, "managed_by": "detach", "host": "127.0.0.1", "port": 8081,
        "argv": ["/py", "-m", "gmlx", "serve", "--config", "gmlx.yaml", "--foreground"],
        "config_abspath": "gmlx.yaml"})
    monkeypatch.setattr(lc, "identity_ok", lambda run: True)
    assert lc.service_install_menubar([], host="127.0.0.1", port=8081) == 2
    assert capsys.readouterr().err == (
        "error: the server at http://127.0.0.1:8081 started with --config gmlx.yaml, a "
        "relative path that a login start cannot find. Stop it with gmlx stop --port "
        "8081, then run gmlx service install again.\n")
    assert mb.load_menubar_settings()["autostart"] is None
    assert not lc._menubar_agent_plist_path().exists()


def test_service_install_menubar_refuses_over_headless_agent(monkeypatch, capsys):
    monkeypatch.setattr(lc, "_require_macos", lambda what: 0)
    pp = lc._plist_path("127.0.0.1", 8080)
    pp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_bytes(b"headless")
    rc = lc.service_install_menubar([], host="127.0.0.1", port=8080)
    assert rc == 2
    assert "--headless" in capsys.readouterr().err


def test_service_uninstall_removes_menubar_agent_and_autostart(monkeypatch, capsys):
    import gmlx.commands.menubar as mb
    monkeypatch.setattr(lc, "_require_macos", lambda what: 0)
    _happy_launchctl(monkeypatch)
    mp = lc._menubar_agent_plist_path()
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_bytes(b"agent")
    mb.save_menubar_settings({"hotkey": "on", "autostart": {
        "argv": list(_AUTOSTART_ARGV), "host": "127.0.0.1", "port": 8080,
        "config_abspath": None, "api_key_set": False}})
    assert lc.service_uninstall("127.0.0.1", 8080) == 0
    assert not mp.exists()
    got = mb.load_menubar_settings()
    assert got["autostart"] is None
    assert got["hotkey"] == "on"                      # other prefs untouched
    assert lc.MENUBAR_AGENT_LABEL in capsys.readouterr().out


def test_service_is_macos_only(monkeypatch, capsys):
    monkeypatch.setattr(lc.sys, "platform", "linux")
    assert lc.service_install([], host="127.0.0.1", port=8080) == 2
    assert lc.service_install_menubar([], host="127.0.0.1", port=8080) == 2
    assert lc.service_uninstall("127.0.0.1", 8080) == 2
    assert lc.service_status("127.0.0.1", 8080) == 2
    assert "macOS" in capsys.readouterr().err


def test_ready_accepts_empty_model_catalog(monkeypatch):
    # A config with no models yet boots a healthy server whose /v1/models data
    # is []. Readiness must accept that (it used to stall the full timeout and
    # print "may still be loading" about a healthy server) while still
    # requiring the OpenAI list shape to reject a foreign/half-bound server.
    import io
    import json as _json

    monkeypatch.setattr(lc, "_health_ok", lambda h, p: True)

    def fake_urlopen(payload):
        class _R(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def opener(req, timeout=None):
            return _R(_json.dumps(payload).encode())

        return opener

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen",
                        fake_urlopen({"object": "list", "data": []}))
    assert lc._ready("127.0.0.1", 18080) is True
    monkeypatch.setattr(urllib.request, "urlopen",
                        fake_urlopen({"detail": "not an openai server"}))
    assert lc._ready("127.0.0.1", 18080) is False


def test_ready_pins_expected_pid(monkeypatch):
    # A foreign gmlx already holding the port answers /health and /v1/models
    # with ITS pid; without the pin, serve declares the (doomed) child up and
    # writes its pid into the runfile. Readiness with expect_pid must only pass
    # when /health names that exact process.
    import io
    import json as _json
    import urllib.request

    def route_urlopen(health_body):
        class _R(io.BytesIO):
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def opener(req, timeout=None):
            url = req if isinstance(req, str) else req.full_url
            if url.endswith("/health"):
                return _R(_json.dumps(health_body).encode())
            return _R(_json.dumps({"object": "list", "data": []}).encode())

        return opener

    monkeypatch.setattr(urllib.request, "urlopen",
                        route_urlopen({"status": "healthy", "pid": 111}))
    assert lc._ready("127.0.0.1", 18080, expect_pid=111) is True
    assert lc._ready("127.0.0.1", 18080, expect_pid=222) is False
    assert lc._ready("127.0.0.1", 18080) is True                 # no pin: old behavior
    # A health body with no pid cannot prove identity - stay not-ready.
    monkeypatch.setattr(urllib.request, "urlopen",
                        route_urlopen({"status": "healthy"}))
    assert lc._ready("127.0.0.1", 18080, expect_pid=111) is False


def test_status_stale_pid_foreign_responder_not_running(monkeypatch, capsys):
    # Dead runfile pid + a live HTTP responder on the port (a foreign server)
    # must read as NOT running, naming the foreign responder - never "healthy".
    lc.write_run("127.0.0.1", 18080, {"pid": 999999, "host": "127.0.0.1",
                                      "port": 18080, "managed_by": "detach"})
    monkeypatch.setattr(lc, "identity_ok", lambda run: False)
    monkeypatch.setattr(lc, "_health_ok", lambda h, p: True)
    assert lc.status("127.0.0.1", 18080) == 3
    out = capsys.readouterr().out
    assert "not running" in out and "different process is answering" in out


def test_service_uninstall_leaves_detach_runfile(monkeypatch, capsys):
    # `service uninstall` must not delete a `serve --background` runfile - that
    # orphans the still-running server from stop/status/logs.
    class _R:
        returncode = 1
        stdout = ""
        stderr = ""

    monkeypatch.setattr(lc, "_require_macos", lambda what: 0)
    monkeypatch.setattr(lc.subprocess, "run", lambda *a, **k: _R())
    lc.write_run("127.0.0.1", 8080,
                 {"pid": 4242, "managed_by": "detach", "port": 8080})
    monkeypatch.setattr(lc, "identity_ok", lambda run: True)
    assert lc.service_uninstall("127.0.0.1", 8080) == 0
    assert lc.read_run("127.0.0.1", 8080) is not None   # runfile intact
    assert "gmlx stop" in capsys.readouterr().out


def test_service_install_refuses_over_detach_server(monkeypatch, capsys):
    monkeypatch.setattr(lc, "_require_macos", lambda what: 0)
    lc.write_run("127.0.0.1", 8080,
                 {"pid": 4242, "managed_by": "detach", "port": 8080})
    monkeypatch.setattr(lc, "identity_ok", lambda run: True)
    rc = lc.service_install(["serve"], host="127.0.0.1", port=8080)
    assert rc == 2
    assert "stop" in capsys.readouterr().err


def test_hand_edited_runfile_pids_degrade_to_stale(tmp_path, monkeypatch):
    # A non-int pid must read as dead/stale, never crash os.kill.
    monkeypatch.setattr(lc, "runtime_dir", lambda: tmp_path)
    (tmp_path / "run-a-1.json").write_text('{"pid": "abc", "host": "a", "port": 1}')
    (tmp_path / "run-a-2.json").write_text('{"pid": "123", "host": "a", "port": 2}')
    (tmp_path / "run-a-3.json").write_text("[1, 2]")          # not even a dict
    runs = lc.list_runs()
    assert {r["port"]: r["pid"] for r in runs} == {1: None, 2: 123}
    assert lc.pid_alive("abc") is False
    assert lc.read_run("a", 1)["pid"] is None


def test_stop_menubar_without_runfile_is_clean(tmp_path, monkeypatch):
    monkeypatch.setattr(lc, "runtime_dir", lambda: tmp_path)
    assert lc.stop_menubar() is False


# run-*.lock cleanup
def test_remove_run_keeps_lock_file(tmp_path, monkeypatch):
    """Unlinking the lock under a live flock holder reopens the double-spawn
    window: the next `serve` creates a fresh inode and locks that instead."""
    monkeypatch.setattr(lc, "runtime_dir", lambda: tmp_path)
    lc.write_run("127.0.0.1", 8080, {"pid": 1, "host": "127.0.0.1", "port": 8080})
    lc._run_lock("127.0.0.1", 8080).write_text("")      # what _run_locked leaves
    lc._remove_run("127.0.0.1", 8080)
    assert not lc.run_path("127.0.0.1", 8080).exists()
    assert lc._run_lock("127.0.0.1", 8080).exists()


def test_remove_run_if_pid_spares_a_newer_runfile(tmp_path, monkeypatch):
    """A `serve` that won the guard while stop() was killing the old process
    owns the runfile now; stop() must not delete it."""
    monkeypatch.setattr(lc, "runtime_dir", lambda: tmp_path)
    lc.write_run("127.0.0.1", 8080, {"pid": 222, "host": "127.0.0.1", "port": 8080})
    lc._remove_run_if_pid("127.0.0.1", 8080, 111)       # we killed 111, not 222
    assert lc.read_run("127.0.0.1", 8080)["pid"] == 222
    lc._remove_run_if_pid("127.0.0.1", 8080, 222)
    assert not lc.run_path("127.0.0.1", 8080).exists()


def test_stop_does_not_delete_a_server_started_during_the_kill(
        tmp_path, monkeypatch, capsys):
    """stop() reads pid 111, kills it, and while it waits a `serve` writes its
    own runfile for pid 222. The surviving server must stay visible."""
    monkeypatch.setattr(lc, "runtime_dir", lambda: tmp_path)
    monkeypatch.setattr(lc, "identity_ok", lambda run: True)
    monkeypatch.setattr(lc, "_maybe_stop_auto_menubar", lambda: None)
    monkeypatch.setattr(lc.os, "killpg", lambda *a: None)
    lc.write_run("127.0.0.1", 8080, {"pid": 111, "pgid": 111,
                                     "host": "127.0.0.1", "port": 8080})

    def _wait_gone(pid, timeout):
        lc.write_run("127.0.0.1", 8080, {"pid": 222, "pgid": 222,
                                         "host": "127.0.0.1", "port": 8080})
        return True

    monkeypatch.setattr(lc, "_wait_gone", _wait_gone)
    assert lc.stop("127.0.0.1", 8080) == 0
    assert lc.read_run("127.0.0.1", 8080)["pid"] == 222


def test_spawn_guard_announces_contention_once(tmp_path, monkeypatch):
    """stop() can hold the guard through its kill wait (~20s); a serve/stop
    arriving meanwhile must say why it stalled instead of blocking silently -
    and stay quiet when the lock is free."""
    import fcntl

    monkeypatch.setattr(lc, "runtime_dir", lambda: tmp_path)
    calls = []
    with open(lc._run_lock("127.0.0.1", 8080), "w") as holder:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX)

        def on_wait():
            calls.append("waited")
            fcntl.flock(holder.fileno(), fcntl.LOCK_UN)   # let the waiter in

        with lc._spawn_guard_lock("127.0.0.1", 8080, on_wait=on_wait):
            pass
    assert calls == ["waited"]

    with lc._spawn_guard_lock("127.0.0.1", 8080,
                              on_wait=lambda: calls.append("free")):
        pass
    assert calls == ["waited"]                # uncontended: no message


# auto-raised menu bar follows the last server down
def test_menubar_run_auto_flag_round_trip():
    lc.write_menubar_run(123)
    assert lc.menubar_is_auto() is False                # manual by default
    lc.write_menubar_run(123, auto=True)
    assert lc.menubar_is_auto() is True


def test_start_menubar_auto_flag_reaches_child_argv(monkeypatch):
    monkeypatch.setattr(lc, "menubar_alive", lambda: False)
    monkeypatch.setattr(lc.procname, "menubar_bundle", lambda: None)
    spawned = []

    def fake_popen(argv, **kw):
        spawned.append(argv)
        return _FakeProc(pid=777)

    monkeypatch.setattr(lc.subprocess, "Popen", fake_popen)
    assert lc.start_menubar(auto=True) == 0
    assert "--auto-raised" in spawned[0]
    assert lc.menubar_is_auto() is True
    lc.remove_menubar_run()
    assert lc.start_menubar() == 0                      # manual: no flag recorded
    assert "--auto-raised" not in spawned[1]
    assert lc.menubar_is_auto() is False


def _stoppable_run(monkeypatch, port=8080):
    lc.write_run("127.0.0.1", port, {"pid": 555, "pgid": 555, "host": "127.0.0.1",
                                     "port": port, "managed_by": "detach"})
    monkeypatch.setattr(lc, "identity_ok", lambda run: True)
    monkeypatch.setattr(lc, "_wait_gone", lambda pid, timeout: True)
    monkeypatch.setattr(lc.os, "killpg", lambda pgid, sig: None)


def test_stop_last_server_stops_auto_menubar(monkeypatch, capsys):
    _stoppable_run(monkeypatch)
    monkeypatch.setattr(lc, "menubar_alive", lambda: True)
    stopped = []
    monkeypatch.setattr(lc, "stop_menubar", lambda: stopped.append(1) or True)
    lc.write_menubar_run(777, auto=True)
    assert lc.stop("127.0.0.1", 8080) == 0
    assert stopped == [1]
    assert "auto-raised menu bar" in capsys.readouterr().out


def test_stop_keeps_manual_menubar(monkeypatch):
    _stoppable_run(monkeypatch)
    monkeypatch.setattr(lc, "menubar_alive", lambda: True)
    stopped = []
    monkeypatch.setattr(lc, "stop_menubar", lambda: stopped.append(1) or True)
    lc.write_menubar_run(777)                           # manual launch
    assert lc.stop("127.0.0.1", 8080) == 0
    assert stopped == []                                # its owner asked for it


def test_stop_keeps_auto_menubar_while_servers_remain(monkeypatch):
    _stoppable_run(monkeypatch)
    lc.write_run("127.0.0.1", 8090, {"pid": 556, "host": "127.0.0.1", "port": 8090})
    monkeypatch.setattr(lc, "menubar_alive", lambda: True)
    stopped = []
    monkeypatch.setattr(lc, "stop_menubar", lambda: stopped.append(1) or True)
    lc.write_menubar_run(777, auto=True)
    assert lc.stop("127.0.0.1", 8080) == 0
    assert stopped == []                                # 8090 still wants the bar


def _restartable_run(monkeypatch):
    _stoppable_run(monkeypatch)
    run = lc.read_run("127.0.0.1", 8080)
    lc.write_run("127.0.0.1", 8080, {**run, "argv": ["serve", "--models-dir", "/m"]})
    monkeypatch.setattr(lc, "menubar_alive", lambda **kw: True)
    signalled = []
    monkeypatch.setattr(lc.os, "kill", lambda pid, sig: signalled.append(pid))
    return signalled


@pytest.mark.parametrize("started", [0, 1])
def test_restart_from_the_auto_menubar_keeps_the_bar(monkeypatch, started):
    """The real stop runs: a bar that runs the restart must not stop itself."""
    signalled = _restartable_run(monkeypatch)
    lc.write_menubar_run(os.getpid(), auto=True)
    monkeypatch.setattr(lc, "launch_detached", lambda *a, **kw: started)
    assert lc.restart("127.0.0.1", 8080) == started
    assert signalled == []
    assert lc.menubar_run_path().exists()
    assert lc.menubar_is_auto()


def test_restart_keeps_an_auto_menubar_up_across_the_stop(monkeypatch):
    signalled = _restartable_run(monkeypatch)
    lc.write_menubar_run(777, auto=True)
    monkeypatch.setattr(lc, "launch_detached", lambda *a, **kw: 0)
    assert lc.restart("127.0.0.1", 8080) == 0
    assert signalled == []
    assert lc.menubar_run_path().exists()


def test_a_failed_restart_stops_an_auto_menubar_it_does_not_run_in(monkeypatch):
    signalled = _restartable_run(monkeypatch)
    lc.write_menubar_run(777, auto=True)
    monkeypatch.setattr(lc, "launch_detached", lambda *a, **kw: 1)
    assert lc.restart("127.0.0.1", 8080) == 1
    assert signalled == [777]
    assert not lc.menubar_run_path().exists()


# stale runfiles: identified with a reason, never ambiguous, cleared by stop
def _live_by_pid(monkeypatch, live_pids):
    monkeypatch.setattr(lc, "pid_alive", lambda pid: pid in live_pids)
    monkeypatch.setattr(lc, "identity_ok",
                        lambda run: run.get("pid") in live_pids)


def test_stale_reason_names_the_cause(monkeypatch):
    monkeypatch.setattr(lc, "pid_alive", lambda pid: pid == 500)
    monkeypatch.setattr(lc, "_proc_cmdline", lambda pid: "/usr/bin/vim notes.txt")
    assert lc.stale_reason({"pid": 400, "port": 1}) == "pid 400 exited"
    assert lc.stale_reason({"pid": 500, "port": 1}) == \
        "pid 500 is now another process (vim)"
    assert lc.stale_reason({"pid": None, "port": 1}) == "no pid recorded"
    assert lc.stale_reason(None) == "unreadable runfile"
    assert lc.stale_reason({"pid": 400, "managed_by": "launchd"}) is None
    monkeypatch.setattr(lc, "_proc_cmdline", lambda pid: "gmlx serve --port 1")
    assert lc.stale_reason({"pid": 500, "port": 1}) is None


def test_classify_and_prune_keep_the_live_server(monkeypatch):
    import time
    lc.write_run("127.0.0.1", 9001, {"pid": 11, "host": "127.0.0.1", "port": 9001,
                                     "started_at": time.time() - 3 * 86400})
    lc.write_run("127.0.0.1", 9002, {"pid": 22, "host": "127.0.0.1", "port": 9002,
                                     "started_at": time.time() - 90})
    _live_by_pid(monkeypatch, {22})
    live, stale = lc.classify_runs()
    assert [r["port"] for r in live] == [9002]
    assert [(r["port"], r["stale_reason"], r["age"]) for r in stale] == \
        [(9001, "pid 11 exited", "started 3d ago")]
    assert lc.auto_target(None, None) == ("127.0.0.1", 9002)      # live wins
    removed = lc.prune_stale_runs()
    assert [r["port"] for r in removed] == [9001]
    assert [r["port"] for r in lc.list_runs()] == [9002]
    assert lc.prune_stale_runs() == []


def test_prune_respects_a_runfile_rewritten_by_a_new_serve(monkeypatch):
    lc.write_run("127.0.0.1", 9001, {"pid": 11, "host": "127.0.0.1", "port": 9001})
    _live_by_pid(monkeypatch, set())
    # a serve wins the spawn guard and writes a live runfile between the
    # classification and the removal
    real_read = lc.read_run

    def rewrite_then_read(host, port):
        lc.write_run(host, port, {"pid": 33, "host": host, "port": port})
        _live_by_pid(monkeypatch, {33})
        return real_read(host, port)

    monkeypatch.setattr(lc, "read_run", rewrite_then_read)
    assert lc.prune_stale_runs() == []
    assert real_read("127.0.0.1", 9001)["pid"] == 33


def test_bare_stop_clears_stale_then_stops_the_live_one(monkeypatch, capsys):
    import gmlx.serve.server as srv
    lc.write_run("127.0.0.1", 9001, {"pid": 11, "host": "127.0.0.1", "port": 9001})
    lc.write_run("127.0.0.1", 9002, {"pid": 22, "host": "127.0.0.1", "port": 9002})
    lc.write_run("127.0.0.1", 9003, {"pid": 33, "host": "127.0.0.1", "port": 9003})
    _live_by_pid(monkeypatch, {22})
    stopped = []
    monkeypatch.setattr(lc, "stop", lambda h, p, timeout: stopped.append(p) or 0)
    assert srv._cmd_stop([]) == 0
    err = capsys.readouterr().err
    assert "cleared 2 stale runfiles" in err and "9001" in err and "9003" in err
    assert "pid 11 exited" in err
    assert stopped == [9002]                                     # not ambiguous
    assert [r["port"] for r in lc.list_runs()] == [9002]


def test_stop_stale_signals_nothing(monkeypatch, capsys):
    import gmlx.serve.server as srv
    lc.write_run("127.0.0.1", 9001, {"pid": 11, "host": "127.0.0.1", "port": 9001})
    lc.write_run("127.0.0.1", 9002, {"pid": 22, "host": "127.0.0.1", "port": 9002})
    _live_by_pid(monkeypatch, {22})
    monkeypatch.setattr(lc, "stop", lambda *a, **k: pytest.fail("must not signal"))
    assert srv._cmd_stop(["--stale"]) == 0
    assert "cleared 1 stale runfile:" in capsys.readouterr().out
    assert [r["port"] for r in lc.list_runs()] == [9002]
    assert srv._cmd_stop(["--stale"]) == 0
    assert "no stale runfiles" in capsys.readouterr().out


def test_bare_status_lists_stale_with_reason(monkeypatch, capsys):
    import json
    import gmlx.serve.server as srv
    lc.write_run("127.0.0.1", 9001, {"pid": 11, "host": "127.0.0.1", "port": 9001,
                                     "managed_by": "detach"})
    lc.write_run("127.0.0.1", 9002, {"pid": 22, "host": "127.0.0.1", "port": 9002,
                                     "managed_by": "detach", "started_at": 0})
    _live_by_pid(monkeypatch, {11})
    monkeypatch.setattr(lc, "_health_ok", lambda h, p, timeout=1.5: True)
    assert srv._cmd_status([]) == 0                             # the live one
    out = capsys.readouterr().out
    assert "http://127.0.0.1:9001: up" in out
    assert "stale runfile 127.0.0.1:9002: pid 22 exited" in out
    assert "gmlx stop --stale" in out
    assert [r["port"] for r in lc.list_runs()] == [9001, 9002]  # status clears nothing
    # explicit target on the stale one: the reason, not just "not running"
    assert srv._cmd_status(["--port", "9002"]) == 3
    assert "not running (stale runfile: pid 22 exited" in capsys.readouterr().out
    # json: live servers plus the stale list, only when several runfiles
    lc.write_run("127.0.0.1", 9003, {"pid": 33, "host": "127.0.0.1", "port": 9003,
                                     "managed_by": "detach"})
    _live_by_pid(monkeypatch, {11, 33})
    assert srv._cmd_status(["--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert [s["port"] for s in doc["servers"]] == [9001, 9003]
    assert [s["port"] for s in doc["stale"]] == [9002]


# source stamp: the runfile records the tree the server booted with, so a
# later `status`/`launch` (running the new code) can flag a stale server
def _stamp_tree(tmp_path):
    pkg = tmp_path / "pkg"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "sub" / "m.py").write_text("")
    os.utime(pkg / "__init__.py", (1000.0, 1000.0))
    os.utime(pkg / "sub" / "m.py", (1001.0, 1001.0))
    return pkg


def test_source_stamp_counts_and_newest_mtime(monkeypatch, tmp_path):
    pkg = _stamp_tree(tmp_path)
    monkeypatch.setattr(lc, "_source_root", lambda: pkg)
    assert lc.source_stamp() == {"files": 2, "newest_mtime": 1001.0}


def test_source_changed_detects_edit_add_delete(monkeypatch, tmp_path):
    pkg = _stamp_tree(tmp_path)
    monkeypatch.setattr(lc, "_source_root", lambda: pkg)
    run = {"source_stamp": lc.source_stamp()}
    assert lc.source_changed(run) is False

    os.utime(pkg / "sub" / "m.py", (3000.0, 3000.0))     # in-place edit
    assert lc.source_changed(run) is True
    os.utime(pkg / "sub" / "m.py", (1001.0, 1001.0))
    assert lc.source_changed(run) is False

    extra = pkg / "extra.py"                             # added file, older mtime
    extra.write_text("")
    os.utime(extra, (900.0, 900.0))
    assert lc.source_changed(run) is True
    extra.unlink()
    assert lc.source_changed(run) is False

    (pkg / "sub" / "m.py").unlink()                      # deletion
    assert lc.source_changed(run) is True


def test_source_changed_unknown_without_stamp():
    # Pre-stamp runfiles (older servers) must read unknown, never stale.
    assert lc.source_changed(None) is None
    assert lc.source_changed({}) is None
    assert lc.source_changed({"source_stamp": "bogus"}) is None


def test_stamp_run_refreshes_and_noops_without_runfile(monkeypatch, tmp_path):
    pkg = _stamp_tree(tmp_path)
    monkeypatch.setattr(lc, "_source_root", lambda: pkg)
    lc.stamp_run("127.0.0.1", 9001)                      # unmanaged: no runfile
    assert lc.read_run("127.0.0.1", 9001) is None
    old = {"files": 0, "newest_mtime": 0}
    lc.write_run("127.0.0.1", 9001, {"pid": 11, "source_stamp": old})
    lc.stamp_run("127.0.0.1", 9001)                      # another server's runfile
    assert lc.read_run("127.0.0.1", 9001)["source_stamp"] == old
    lc.write_run("127.0.0.1", 9001, {
        "pid": None, "managed_by": "launchd", "source_stamp": old})
    lc.stamp_run("127.0.0.1", 9001)                      # not under --launchd
    assert lc.read_run("127.0.0.1", 9001)["source_stamp"] == old
    lc.stamp_run("127.0.0.1", 9001, launchd=True)        # launchd respawn refresh
    run = lc.read_run("127.0.0.1", 9001)
    assert run["source_stamp"] == lc.source_stamp()
    assert lc.source_changed(run) is False
    lc.write_run("127.0.0.1", 9001, {"pid": os.getpid(), "source_stamp": old})
    lc.stamp_run("127.0.0.1", 9001)                      # a background start's child
    assert lc.read_run("127.0.0.1", 9001)["source_stamp"] == lc.source_stamp()


def test_a_launchd_respawn_records_the_config_file_it_read(tmp_path):
    """launchd starts the agent again at a login with no gmlx command around
    it. After a retarget, the server reads the file that its --config link
    leads to now and keeps that file's key, so the runfile names that file."""
    import gmlx.commands.launch as launch
    dots = tmp_path / "dots"
    dots.mkdir()
    (dots / "a.yaml").write_text("server:\n  api_key: key-A\n")
    (dots / "b.yaml").write_text("server:\n  api_key: key-B\n")
    link = tmp_path / "gmlx.yaml"
    link.symlink_to(dots / "a.yaml")
    lc.write_run("127.0.0.1", 8080, {
        "pid": None, "host": "127.0.0.1", "port": 8080, "managed_by": "launchd",
        "config_abspath": str(dots / "a.yaml"), "config_given": str(link),
        "config_reloaded": str(dots / "a.yaml"), "api_key_set": True})
    link.unlink()
    link.symlink_to(dots / "b.yaml")
    lc.stamp_run("127.0.0.1", 8080, config_given=str(link),
                 config_real=os.path.realpath(link), launchd=True)
    run = lc.read_run("127.0.0.1", 8080)
    assert run["config_abspath"] == str(dots / "b.yaml")
    # A reload of the earlier start no longer describes this server.
    assert "config_reloaded" not in run
    assert launch._runfile_key("127.0.0.1", 8080) == "key-B"
    assert launch._served_config("127.0.0.1", 8080)[0] == str(dots / "b.yaml")
    # A server with another --config on this port leaves the record as it is.
    lc.stamp_run("127.0.0.1", 8080, config_given=str(tmp_path / "other.yaml"),
                 config_real=str(dots / "a.yaml"), launchd=True)
    assert lc.read_run("127.0.0.1", 8080)["config_abspath"] == str(dots / "b.yaml")


def test_a_bare_start_records_the_default_config_in_its_runfile(tmp_path):
    """An older gmlx installed a headless agent with no --config when no
    config existed. After gmlx init, launchd starts it again and it reads the
    default config, so its runfile names that file, and launch reads the key
    and the profiles from it."""
    import gmlx.commands.launch as launch
    conf = tmp_path / "gmlx.yaml"
    conf.write_text("server:\n  api_key: key-A\n"
                    "profiles:\n  mine:\n    load: {max_kv_size: 4096}\n")
    real = os.path.realpath(conf)
    bare = ["/app/gmlx-agent", "serve", "--host", "127.0.0.1", "--port", "8080",
            "--foreground", "--launchd"]
    lc.write_run("127.0.0.1", 8080, {
        "pid": None, "host": "127.0.0.1", "port": 8080, "managed_by": "launchd",
        "config_abspath": None, "argv": bare})
    # A server with a --config on this port leaves the record as it is.
    lc.stamp_run("127.0.0.1", 8080, config_given=str(conf), config_real=real,
                 launchd=True)
    assert lc.read_run("127.0.0.1", 8080)["config_abspath"] is None
    lc.stamp_run("127.0.0.1", 8080, config_given=str(conf), config_real=real, bare=True,
                 launchd=True)
    run = lc.read_run("127.0.0.1", 8080)
    assert (run["config_given"], run["config_abspath"]) == (str(conf), real)
    assert lc.reload_config_path(run) == real
    assert launch._runfile_key("127.0.0.1", 8080) == "key-A"
    served = launch._served_config("127.0.0.1", 8080)
    assert served is not None and served[0] == real
    assert not launch._profile_keeps_window(served, "m1@mine")
    # A start that names a model folder reads no config.
    folder = [*bare[:2], "--models-dir", str(tmp_path), *bare[2:]]
    lc.write_run("127.0.0.1", 8080, {
        "pid": None, "host": "127.0.0.1", "port": 8080, "managed_by": "launchd",
        "config_abspath": None, "argv": folder})
    lc.stamp_run("127.0.0.1", 8080, config_given=str(conf), config_real=real, bare=True,
                 launchd=True)
    run = lc.read_run("127.0.0.1", 8080)
    assert run["config_abspath"] is None and "config_given" not in run


def test_a_login_agent_of_an_older_gmlx_records_its_config_link(tmp_path):
    """An older gmlx recorded the --config of a login agent only in its argv,
    and recorded the link as config_abspath. The first start of the agent
    with this gmlx records the link and the file it leads to, so launch reads
    the key and a reload is recorded."""
    import gmlx.commands.launch as launch
    dots = tmp_path / "dots"
    dots.mkdir()
    (dots / "a.yaml").write_text("server:\n  api_key: key-A\n")
    link = tmp_path / "gmlx.yaml"
    link.symlink_to(dots / "a.yaml")
    real = os.path.realpath(link)
    argv = ["/app/gmlx-agent", "serve", "--config", str(link), "--host", "127.0.0.1",
            "--port", "8080", "--foreground", "--launchd"]
    old = {"pid": None, "host": "127.0.0.1", "port": 8080, "managed_by": "launchd",
           "config_abspath": str(link), "argv": argv, "api_key_set": True}
    lc.write_run("127.0.0.1", 8080, old)
    assert launch._runfile_key("127.0.0.1", 8080) is None
    # A server with another --config on this port leaves the record as it is.
    lc.stamp_run("127.0.0.1", 8080, config_given=str(tmp_path / "other.yaml"),
                 config_real=real, launchd=True)
    assert "config_given" not in lc.read_run("127.0.0.1", 8080)
    lc.stamp_run("127.0.0.1", 8080, config_given=str(link), config_real=real,
                 launchd=True)
    run = lc.read_run("127.0.0.1", 8080)
    assert (run["config_given"], run["config_abspath"]) == (str(link), real)
    assert launch._runfile_key("127.0.0.1", 8080) == "key-A"
    lc.note_config_reload("127.0.0.1", 8080, config_given=str(link), config_real=real,
                          launchd=True)
    assert lc.read_run("127.0.0.1", 8080)["config_reloaded"] == real


def test_a_second_server_on_the_bind_leaves_the_record_of_the_first(tmp_path):
    """A second server on the bind of a running one writes its boot record
    before its bind fails. The running server keeps the key of the file it
    read at its start, so its record stays as it is."""
    import gmlx.commands.launch as launch
    dots = tmp_path / "dots"
    dots.mkdir()
    (dots / "a.yaml").write_text("server:\n  api_key: key-A\n")
    (dots / "b.yaml").write_text("server:\n  api_key: key-B\n")
    link = tmp_path / "gmlx.yaml"
    link.symlink_to(dots / "b.yaml")
    a_real, b_real = os.path.realpath(dots / "a.yaml"), os.path.realpath(link)
    argv = ["/py", "-m", "gmlx", "serve", "--config", str(link), "--host",
            "127.0.0.1", "--port", "8080", "--foreground"]
    first = {"host": "127.0.0.1", "port": 8080, "config_given": str(link),
             "config_abspath": a_real, "config_reloaded": a_real,
             "api_key_set": True, "source_stamp": {"files": 0, "newest_mtime": 0}}
    bare = ["/app/gmlx-agent", "serve", "--host", "127.0.0.1", "--port", "8080",
            "--foreground", "--launchd"]
    for run in ({**first, "pid": os.getppid(), "managed_by": "detach", "argv": argv},
                {**first, "pid": None, "managed_by": "launchd", "argv": argv},
                {"pid": None, "host": "127.0.0.1", "port": 8080,
                 "managed_by": "launchd", "config_abspath": None, "argv": bare}):
        lc.write_run("127.0.0.1", 8080, run)
        before = lc.read_run("127.0.0.1", 8080)
        lc.stamp_run("127.0.0.1", 8080, config_given=str(link), config_real=b_real)
        lc.stamp_run("127.0.0.1", 8080, config_given=str(link), config_real=b_real,
                     bare=True)
        lc.note_config_reload("127.0.0.1", 8080, config_given=str(link),
                              config_real=b_real)
        assert lc.read_run("127.0.0.1", 8080) == before
    lc.write_run("127.0.0.1", 8080, {**first, "pid": None, "managed_by": "launchd",
                                     "argv": argv})
    assert launch._runfile_key("127.0.0.1", 8080) == "key-A"


def test_status_notes_stale_source(monkeypatch, capsys):
    lc.write_run("127.0.0.1", 9001, {
        "pid": 11, "host": "127.0.0.1", "port": 9001, "managed_by": "detach",
        "source_stamp": {"files": 1, "newest_mtime": 1.0}})
    monkeypatch.setattr(lc, "identity_ok", lambda run: True)
    monkeypatch.setattr(lc, "_health_ok", lambda h, p, timeout=1.5: True)
    monkeypatch.setattr(lc, "_served_model_count", lambda h, p, key=None: 1)
    monkeypatch.setattr(lc, "source_stamp",
                        lambda: {"files": 2, "newest_mtime": 2.0})
    assert lc.status("127.0.0.1", 9001) == 0
    out = capsys.readouterr().out
    assert "source changed on disk" in out and "gmlx restart" in out

    monkeypatch.setattr(lc, "source_stamp",
                        lambda: {"files": 1, "newest_mtime": 1.0})
    assert lc.status("127.0.0.1", 9001) == 0
    assert "source changed" not in capsys.readouterr().out


@pytest.mark.parametrize("config, hint", [
    ("default", "add a model: gmlx pull <hf:ref>"),
    ("my gmlx.yaml", "add a model: gmlx pull <hf:ref> --config '{tmp}/my gmlx.yaml'"),
    ("gone.yaml", "the config it started with, {tmp}/gone.yaml, is gone, so run gmlx "
                  "restart for the steps"),
    (None, "add a GGUF to a --models-dir folder, then run gmlx restart")])
def test_status_with_no_models_says_how_to_add_one(monkeypatch, capsys, tmp_path,
                                                   config, hint):
    """gmlx pull names the server's config when it would not find it, and a
    config that is gone gets the restart step instead."""
    import gmlx.config as cfgmod

    default = tmp_path / "gmlx.yaml"
    default.write_text("server: {}\n")
    (tmp_path / "my gmlx.yaml").write_text("server: {}\n")
    monkeypatch.setattr(cfgmod, "default_config_paths",
                        lambda note_local=True: [default])
    if config == "default":
        config = str(default)
    elif config:
        config = str(tmp_path / config)
    hint = hint.format(tmp=tmp_path)
    lc.write_run("127.0.0.1", 9001, {
        "pid": 11, "host": "127.0.0.1", "port": 9001, "managed_by": "detach",
        "config_abspath": config})
    monkeypatch.setattr(lc, "identity_ok", lambda run: True)
    monkeypatch.setattr(lc, "_health_ok", lambda h, p, timeout=1.5: True)
    monkeypatch.setattr(lc, "_served_model_count", lambda h, p, key=None: 0)
    assert lc.status("127.0.0.1", 9001) == 0
    assert f"  0 models served: requests will 404 - {hint}\n" in capsys.readouterr().out


def test_a_gmlx_package_in_the_current_folder_never_runs(tmp_path):
    """A container client can write gmlx/__init__.py into a shared project.
    A server or menu bar spawned from that folder must still run the
    installed gmlx."""
    proj = tmp_path / "proj"
    (proj / "gmlx").mkdir(parents=True)
    marker = tmp_path / "planted-code-ran"
    (proj / "gmlx" / "__init__.py").write_text(
        f"open({str(marker)!r}, 'w').write('x')\n")
    (proj / "gmlx" / "__main__.py").write_text("")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = {**os.environ, "PYTHONPATH": root}
    argv = [*lc.procname.gmlx_argv(sys.executable), "--version"]
    r = subprocess.run(argv, cwd=proj, env=env, capture_output=True, timeout=120)
    assert r.returncode == 0, r.stderr.decode()[-500:]
    assert not marker.exists()
    for built in (lc.child_argv([]), lc._agent_entry()):
        assert built[1:4] == ["-P", "-m", "gmlx"] or built[0].endswith("-agent")


def test_a_replayed_argv_from_an_older_gmlx_gets_safe_path(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    seen = []
    monkeypatch.setattr(lc.subprocess, "Popen",
                        lambda argv, **kw: seen.append(argv) or _FakeProc(pid=4244))
    lc._spawn_detached(["/stub", "-m", "gmlx", "serve", "--foreground"],
                       host="127.0.0.1", port=8080)
    assert seen == [["/stub", "-P", "-m", "gmlx", "serve", "--foreground"]]
    assert lc.procname.with_safe_path(["/x", "-P", "-m", "gmlx"]) == ["/x", "-P", "-m", "gmlx"]
