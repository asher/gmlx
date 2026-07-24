#!/usr/bin/env python3
"""Process layer behind the background `gmlx serve` / `stop` / `restart` / `status`
/ `logs` / `service` (and the menu-bar companion it raises). CPU-only - every external
touchpoint (the child process, the
readiness probe, signals, launchctl) is faked, so no server starts and no real signal
is sent. The point is the *contracts*: PID-identity before signalling, group kill with
SIGTERM (never SIGHUP), child-death fail-fast, and an absolute-interpreter relaunch."""
from __future__ import annotations

import os
import plistlib
import signal
import sys

import pytest

from gmlx import lifecycle as lc  # noqa: E402


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
    from gmlx import server as srv
    lc.write_run("127.0.0.1", 9001, {"pid": 11, "host": "127.0.0.1", "port": 9001,
                                     "managed_by": "detach"})
    lc.write_run("127.0.0.1", 9002, {"pid": 22, "host": "127.0.0.1", "port": 9002,
                                     "managed_by": "detach"})
    monkeypatch.setattr(lc, "identity_ok", lambda run: True)
    monkeypatch.setattr(lc, "_health_ok", lambda h, p: True)
    assert srv._cmd_status([]) == 0
    out = capsys.readouterr().out
    assert ":9001" in out and ":9002" in out                     # both reported


def test_bare_stop_refuses_when_multiple(capsys):
    from gmlx import server as srv
    lc.write_run("127.0.0.1", 9001, {"pid": 11, "host": "127.0.0.1", "port": 9001})
    lc.write_run("127.0.0.1", 9002, {"pid": 22, "host": "127.0.0.1", "port": 9002})
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


# child invocation (B4): absolute interpreter so launchd's bare PATH still resolves it
def test_child_argv_is_absolute_interpreter(monkeypatch):
    monkeypatch.setattr(lc.procname, "named_python", lambda: None)
    argv = lc.child_argv(["--config", "/abs/c.yaml"])
    assert argv[0] == os.path.abspath(sys.executable)
    assert os.path.isabs(argv[0])
    assert argv[1:4] == ["-m", "gmlx", "serve"]
    assert argv[-2:] == ["--config", "/abs/c.yaml"]


# macOS: the daemon runs through the gmlx-named stub so ps / Activity Monitor
# don't show "Python"; the spawn env points the stub back at this venv.
def test_child_argv_prefers_named_stub(monkeypatch):
    monkeypatch.setattr(lc.procname, "named_python", lambda: "/tmp/proc/gmlx")
    argv = lc.child_argv(["--config", "/abs/c.yaml"])
    assert argv[0] == "/tmp/proc/gmlx"
    assert argv[1:4] == ["-m", "gmlx", "serve"]


def test_child_env_carries_venv_interpreter():
    env = lc.procname.child_env()
    assert env["PYTHONEXECUTABLE"] == os.path.abspath(sys.executable)


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
    assert argv == ["/tmp/stub", "-m", "gmlx", "serve", "--foreground",
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


# The bundle is launchd-load-bearing (the LaunchAgent references it by
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
    assert argv[1:6] == ["-m", "gmlx", "launch", "menubar", "--foreground"]
    # No --host/--port pinned: the one bar tracks the primary, not the spawning server.
    assert "--host" not in argv and "--port" not in argv
    assert argv[-2:] == ["--interval", "9"]
    assert captured["kw"]["start_new_session"] is True
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
    raw = lc.render_plist("com.gmlx.server.x", args, "/abs/server.log",
                          env={"PATH": "/venv/bin:/usr/bin"}, keepalive=True)
    pl = plistlib.loads(raw)
    assert pl["Label"] == "com.gmlx.server.x"
    assert pl["ProgramArguments"] == args
    assert os.path.isabs(pl["ProgramArguments"][0])
    assert pl["KeepAlive"] == {"SuccessfulExit": False}
    assert pl["StandardOutPath"] == "/abs/server.log"
    assert pl["EnvironmentVariables"]["PATH"].startswith("/venv/bin")
    assert all("api" not in a.lower() for a in pl["ProgramArguments"])


def test_render_plist_no_keepalive():
    pl = plistlib.loads(lc.render_plist("L", ["/bin/x"], "/l", keepalive=False))
    assert pl["KeepAlive"] is False


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
    assert bootstrap and bootstrap[0][:3] == ["launchctl", "bootstrap",
                                              f"gui/{os.getuid()}"]
    assert lc._plist_path("127.0.0.1", 8080).exists()
    run = lc.read_run("127.0.0.1", 8080)
    assert run["managed_by"] == "launchd"
    assert "--foreground" in run["argv"]          # launchd runs serve in the foreground


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
                                 "-m", "gmlx"]


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
    assert 'exec "$BIN" -m gmlx "$@"' in body       # TCC pins here
    assert body.index('exec "$BIN"') < body.index('exec "$PY"')
    assert f'PY="{os.path.abspath(sys.executable)}"' in body
    assert 'export PYTHONEXECUTABLE="$PY"' in body      # before the probe
    assert body.index("PYTHONEXECUTABLE") < body.index('if "$BIN" -c ""')
    assert 'export GMLX_LAUNCHD_REEXEC=1' in body      # happy path: no re-exec
    assert 'exec "$PY" -m gmlx "$@"' in body        # stale-copy fallback
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
    from gmlx import menubar as mb
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
    from gmlx import menubar as mb
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
    from gmlx import menubar as mb
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


def test_service_install_menubar_refuses_over_headless_agent(monkeypatch, capsys):
    monkeypatch.setattr(lc, "_require_macos", lambda what: 0)
    pp = lc._plist_path("127.0.0.1", 8080)
    pp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_bytes(b"headless")
    rc = lc.service_install_menubar([], host="127.0.0.1", port=8080)
    assert rc == 2
    assert "--headless" in capsys.readouterr().err


def test_service_uninstall_removes_menubar_agent_and_autostart(monkeypatch, capsys):
    from gmlx import menubar as mb
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


def test_restart_reraises_auto_menubar(monkeypatch):
    lc.write_run("127.0.0.1", 8080, {"pid": 555, "pgid": 555, "host": "127.0.0.1",
                                     "port": 8080, "managed_by": "detach",
                                     "argv": ["serve", "--port", "8080"]})
    monkeypatch.setattr(lc, "menubar_alive", lambda: True)
    lc.write_menubar_run(777, auto=True)
    monkeypatch.setattr(lc, "stop", lambda h, p, timeout=15.0: 0)
    monkeypatch.setattr(lc, "launch_detached", lambda *a, **kw: 0)
    monkeypatch.setattr(lc, "gui_session_available", lambda: True)
    raised = []
    monkeypatch.setattr(lc, "start_menubar",
                        lambda **kw: raised.append(kw.get("auto")) or 0)
    assert lc.restart("127.0.0.1", 8080) == 0
    assert raised == [True]                             # bar comes back auto-raised
