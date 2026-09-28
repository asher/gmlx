"""gmlx/container/session.py and runtime.py: the run command, the session
and volume locks, cleanup, the runtime folder and the supervisor."""

from __future__ import annotations

import http.server
import json
import os
import signal
import socket
import stat
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from gmlx.container import runtime, session
from gmlx.container.settings import ContainerPlan, Mount, SettingsError
from gmlx.container.state import FileLock

D1 = "sha256:" + "1" * 64


def _plan(tmp_path, **kw):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    values = dict(client="pi", home=home, mounts=[
        Mount("/Users/u/src/proj", "/Users/u/src/proj", note="working folder"),
        Mount(str(home), str(home), kind="home"),
        Mount("/Users/u/src/proj/ro", "/Users/u/src/proj/ro", readonly=True),
        Mount("nm", "/Users/u/src/proj/node_modules", kind="volume", size="8G")],
        workdir="/Users/u/src/proj", cwd_shared=True, forward=[5432], network="default",
        cpus=4, memory="4G", ssh_agent=False, env=[], open_browser=True, clipboard="off",
        seed=[])
    values.update(kw)
    return ContainerPlan(**values)


def _spec(tmp_path, **kw):
    sess = session.Session("pi", "abc123", tmp_path / "sess")
    values = dict(session=sess, plan=_plan(tmp_path), image_ref="gmlx.invalid/launch-pi@" + D1,
                  runtime_dir=tmp_path / "rt", command=["pi", "--continue"],
                  workdir="/Users/u/src/proj",
                  env_values={"HOME": "/h", "LANG": "C.UTF-8"},
                  env_names=["OPENAI_API_KEY"], child_env={"OPENAI_API_KEY": "sekrit"},
                  api_port=8080, web_port=None, labels={"gmlx.launch.runtime": "abc"})
    values.update(kw)
    return session.RunSpec(**values)


# The run command

def test_golden_run_argv(tmp_path):
    argv = session.compose_run_argv(_spec(tmp_path))
    s = str(tmp_path / "sess")
    home = str(tmp_path / "home")
    assert argv[:8] == ["container", "run", "--rm", "--init", "--progress", "none",
                        "--name", "gmlx-pi-abc123"]
    labels = [argv[i + 1] for i, a in enumerate(argv) if a == "--label"]
    assert labels == ["gmlx.launch=1", "gmlx.launch.client=pi",
                      f"gmlx.launch.pid={os.getpid()}", "gmlx.launch.runtime=abc"]
    assert "-t" not in argv and "--network" not in argv and "--ssh" not in argv
    envs = [argv[i + 1] for i, a in enumerate(argv) if a == "-e"]
    assert envs == ["HOME=/h", "LANG=C.UTF-8", "OPENAI_API_KEY"]
    assert "sekrit" not in " ".join(argv)
    mounts = [argv[i + 1] for i, a in enumerate(argv) if a == "--mount"]
    assert mounts[0] == f"type=bind,source={tmp_path / 'rt'},target=/opt/gmlx,readonly"
    assert set(mounts[1:]) == {
        "type=bind,source=/Users/u/src/proj,target=/Users/u/src/proj",
        "type=bind,source=/Users/u/src/proj/ro,target=/Users/u/src/proj/ro,readonly",
        "type=volume,source=nm,target=/Users/u/src/proj/node_modules",
        f"type=bind,source={home},target={home}"}
    assert argv[argv.index("--entrypoint") + 1] == "/opt/gmlx/gmlx-entry"
    assert argv[argv.index("--workdir") + 1] == "/Users/u/src/proj"
    tail = argv[argv.index("gmlx.invalid/launch-pi@" + D1):]
    assert tail == ["gmlx.invalid/launch-pi@" + D1,
                    "--tcp", "8080=/var/host-services/gmlx-api.sock",
                    "--tcp", "5432=/var/host-services/gmlx-fwd-5432.sock",
                    "--", "pi", "--continue"]
    vs = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
    assert vs == [f"{s}/api.sock:/var/host-services/gmlx-api.sock",
                  f"{s}/fwd-5432.sock:/var/host-services/gmlx-fwd-5432.sock"]


def test_mounts_are_ordered_by_guest_depth(tmp_path):
    argv = session.compose_run_argv(_spec(tmp_path))
    targets = [a.split("target=")[1].split(",")[0]
               for i, a in enumerate(argv) if i and argv[i - 1] == "--mount"]
    depths = [t.rstrip("/").count("/") for t in targets]
    assert depths == sorted(depths)


def test_web_app_network_none_ssh_tty_and_shell(tmp_path):
    spec = _spec(tmp_path, plan=_plan(tmp_path, network="none", ssh_agent=True, forward=[]),
                 web_port=3000, tty=True, shell=True, command=["-c", "npm test"])
    argv = session.compose_run_argv(spec)
    assert argv[argv.index("--network") + 1] == "none" and "--ssh" in argv and "-t" in argv
    assert f"{tmp_path / 'sess'}/web.sock:/var/host-services/gmlx-web.sock" in argv
    assert argv[-6:] == ["--unix", "/var/host-services/gmlx-web.sock=3000", "--shell", "--",
                         "-c", "npm test"]


def test_clipboard_socket_and_flag_only_under_images(tmp_path):
    argv = session.compose_run_argv(_spec(tmp_path))
    assert "--clipboard" not in argv and not any("clip.sock" in a for a in argv)
    argv = session.compose_run_argv(_spec(tmp_path, plan=_plan(tmp_path, clipboard="images")))
    assert f"{tmp_path / 'sess'}/clip.sock:/var/host-services/gmlx-clip.sock" in argv
    assert argv[-4:] == ["--clipboard", "--", "pi", "--continue"]


def test_forwarded_ports_work_under_network_none(tmp_path):
    spec = _spec(tmp_path, plan=_plan(tmp_path, network="none"))
    argv = session.compose_run_argv(spec)
    assert "5432=/var/host-services/gmlx-fwd-5432.sock" in argv


# Session folders, locks and the record

def test_new_session_falls_back_to_tmpdir_for_long_paths(fake_container, tmp_path, monkeypatch,
                                                         request):
    import shutil
    import tempfile
    short_root = tempfile.mkdtemp(dir="/tmp")
    request.addfinalizer(lambda: shutil.rmtree(short_root, ignore_errors=True))
    monkeypatch.setenv("XDG_CACHE_HOME", short_root)
    short = session.new_session("pi", [5432])
    assert short.dir.parent == session.cache_dir() and short.dir.name.startswith("pi-")
    assert oct(short.dir.stat().st_mode & 0o777) == "0o700"
    deep = tmp_path / ("d" * 90)
    monkeypatch.setenv("XDG_CACHE_HOME", str(deep))
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    long = session.new_session("pi", [65535])
    assert long.dir.parent == tmp_path and long.dir.name.startswith("gmlx-launch-pi-")


def test_session_lock_refuses_a_second_session(fake_container):
    first = session.try_session_lock("pi")
    assert first is not None
    assert session.try_session_lock("pi") is None
    assert session.try_session_lock("omp") is not None
    first.release()
    assert session.try_session_lock("pi") is not None


_HOLDER = textwrap.dedent("""
    import subprocess, sys, time
    from gmlx.container.state import FileLock
    lock = FileLock(sys.argv[1])
    child = subprocess.Popen(["sleep", "30"])
    print(child.pid, flush=True)
    time.sleep(60)
""")


def test_lock_is_free_after_kill_9_while_the_child_runs(tmp_path):
    path = tmp_path / "session.lock"
    holder = subprocess.Popen([sys.executable, "-c", _HOLDER, str(path)],
                              stdout=subprocess.PIPE, text=True)
    child_pid = int(holder.stdout.readline())
    try:
        with pytest.raises(Exception):
            FileLock(path, blocking=False)
        holder.send_signal(signal.SIGKILL)
        holder.wait()
        os.kill(child_pid, 0)                          # the child still runs
        FileLock(path, blocking=False).release()       # yet the lock is free
    finally:
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_record_round_trip(fake_container):
    record = {"name": "gmlx-pi-1", "workdir": "/w", "clipboard": False,
              "shares": [{"host": "/h", "guest": "/g", "readonly": True}]}
    session.write_record("pi", record)
    assert session.read_record("pi") == record
    assert stat.S_IMODE(session.record_path("pi").stat().st_mode) == 0o600
    assert [p.name for p in session.client_dir("pi").iterdir()
            if p.name.endswith(".tmp")] == []
    session.remove_record("pi")
    assert session.read_record("pi") is None


@pytest.mark.parametrize("record", [
    {"name": "gmlx-pi-1"},
    {"name": "gmlx-pi-1", "workdir": "/w", "shares": [{"host": "/h"}]},
    {"name": "gmlx-pi-1", "workdir": "/w", "shares": "/h"},
    ["gmlx-pi-1"],
])
def test_a_damaged_record_is_a_clean_error(fake_container, record):
    session.record_path("pi").write_text(json.dumps(record))
    with pytest.raises(SettingsError, match="session record .* is damaged"):
        session.read_record("pi")


def test_a_deeply_nested_record_is_a_clean_error(fake_container):
    session.record_path("pi").write_text("[" * 100_000 + "]" * 100_000)
    with pytest.raises(SettingsError, match="is damaged"):
        session.read_record("pi")


def test_the_record_is_never_written_through_a_planted_temporary_link(fake_container,
                                                                      tmp_path, monkeypatch):
    target = tmp_path / "elsewhere"
    target.write_text("keep")
    monkeypatch.setattr(session.secrets, "token_hex", lambda n: "fixed")
    tmp = session.client_dir("pi") / f".session.json.{os.getpid()}.fixed.tmp"
    tmp.symlink_to(target)
    with pytest.raises(SettingsError, match="cannot write the session record"):
        session.write_record("pi", {"name": "x"})
    assert target.read_text() == "keep"


# Cleanup

def test_cleanup_stale_touches_only_the_launching_client(fake_container, tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    fake_container.update(containers=[
        {"name": "gmlx-pi-aaaaaa", "labels": {"gmlx.launch": "1", "gmlx.launch.client": "pi"}},
        {"name": "gmlx-omp-bbbbbb", "labels": {"gmlx.launch": "1", "gmlx.launch.client": "omp"}}])
    ours = session.cache_dir() / "pi-aaaaaa"
    ours.mkdir()
    tmp_ours = tmp_path / "gmlx-launch-pi-cccccc"
    tmp_ours.mkdir()
    theirs = session.cache_dir() / "omp-dddddd"
    theirs.mkdir()
    lookalike = session.cache_dir() / "pi-extra-eeeeee"
    lookalike.mkdir()
    said = []
    session.cleanup_stale("pi", keep_runtime=None, say=said.append)
    assert said == ["[launch] removed the leftover container gmlx-pi-aaaaaa of an earlier "
                    "session"]
    assert ["stop", "--time", "5", "gmlx-pi-aaaaaa"] in fake_container.log
    assert ["delete", "--force", "gmlx-pi-aaaaaa"] in fake_container.log
    assert not any("gmlx-omp-bbbbbb" in a for a in fake_container.log)
    assert not ours.exists() and not tmp_ours.exists()
    assert theirs.exists() and lookalike.exists()


def test_orphan_notices_list_dead_launches_of_other_clients():
    from gmlx.container.cli import Container
    dead = Container("gmlx-omp-1", "running", {"gmlx.launch": "1", "gmlx.launch.client": "omp",
                                                "gmlx.launch.pid": "999999"}, "", "",
                     memory_bytes=4 << 30)
    live = Container("gmlx-goose-1", "running", {"gmlx.launch": "1",
                                                  "gmlx.launch.client": "goose",
                                                  "gmlx.launch.pid": str(os.getpid())}, "", "")
    mine = Container("gmlx-pi-1", "running", {"gmlx.launch": "1", "gmlx.launch.client": "pi",
                                              "gmlx.launch.pid": "999999"}, "", "")
    lines = session.orphan_notices("pi", [dead, live, mine])
    assert lines == ["[launch] gmlx-omp-1 from an earlier omp launch is still running and "
                     "holds 4G of memory. Stop it with: container stop gmlx-omp-1"]


# The runtime folder

def _entry(tmp_path, data=b"\x7fELF-fake"):
    path = tmp_path / "gmlx-entry"
    path.write_bytes(data)
    return path


def test_a_colon_in_the_cache_path_moves_the_session_to_tmpdir(fake_container, tmp_path,
                                                                 monkeypatch, request):
    import shutil
    import tempfile
    short_root = tempfile.mkdtemp(dir="/tmp")
    request.addfinalizer(lambda: shutil.rmtree(short_root, ignore_errors=True))
    monkeypatch.setenv("XDG_CACHE_HOME", f"{short_root}/a:b")
    monkeypatch.setenv("TMPDIR", short_root)
    sess = session.new_session("pi", [])
    assert sess.dir.parent == Path(short_root)
    monkeypatch.setenv("TMPDIR", f"{short_root}/c:d")
    with pytest.raises(SettingsError, match="contains ':'"):
        session.new_session("pi", [])


def test_a_runtime_folder_container_cannot_mount_is_refused(fake_container, tmp_path,
                                                            monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "a,b"))
    with pytest.raises(SettingsError, match="the runtime folder"):
        runtime.acquire_runtime(_entry(tmp_path))


def test_runtime_copy_and_cleanup(fake_container, tmp_path):
    folder, lock = runtime.acquire_runtime(_entry(tmp_path))
    assert folder.name == runtime.entry_digest(_entry(tmp_path))
    assert oct((folder / "gmlx-entry").stat().st_mode & 0o777) == "0o755"
    for tool in ("xclip", "xsel", "wl-paste"):
        assert os.readlink(folder / "bin" / tool) == "../gmlx-entry"
        assert (folder / "bin" / tool).resolve() == (folder / "gmlx-entry").resolve()
    old, old_lock = runtime.acquire_runtime(_entry(tmp_path / "..", b"older"))
    old_lock.release()
    assert runtime.cleanup_runtime(keep=folder.name) == [old]
    assert folder.exists()
    lock.release()


_SHARED = textwrap.dedent("""
    import sys, time
    from gmlx.container.state import FileLock
    lock = FileLock(sys.argv[1], shared=True)
    print("held", flush=True)
    time.sleep(60)
""")


def test_cleanup_leaves_a_folder_another_process_holds(fake_container, tmp_path):
    old, lock = runtime.acquire_runtime(_entry(tmp_path, b"old"))
    lock.release()
    holder = subprocess.Popen([sys.executable, "-c", _SHARED, str(old / ".lock")],
                              stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        assert runtime.cleanup_runtime(keep=None) == []
        assert old.exists()
    finally:
        holder.kill()
        holder.wait()
    assert runtime.cleanup_runtime(keep=None) == [old]


def test_runtime_retries_when_the_folder_goes_between_find_and_lock(fake_container, tmp_path,
                                                                   monkeypatch):
    import shutil
    real = runtime.FileLock
    calls = {"n": 0}

    def racing(path, **kw):
        calls["n"] += 1
        lock = real(path, **kw)
        if calls["n"] == 1:
            # A cleanup removes the folder after the lock opened, and another
            # launch installs it again: complete, but with a new lock file.
            folder = Path(path).parent
            inode = os.stat(path).st_ino
            shutil.rmtree(folder)
            runtime._install(folder, _entry(tmp_path))
            assert os.stat(path).st_ino != inode
        return lock
    monkeypatch.setattr(runtime, "FileLock", racing)
    folder, lock = runtime.acquire_runtime(_entry(tmp_path))
    assert calls["n"] == 2 and lock.still_current()
    lock.release()


def test_runtime_copies_again_when_a_link_is_missing(fake_container, tmp_path):
    folder, lock = runtime.acquire_runtime(_entry(tmp_path))
    lock.release()
    (folder / "bin" / "xsel").unlink()
    again, lock = runtime.acquire_runtime(_entry(tmp_path))
    assert again == folder and os.readlink(folder / "bin" / "xsel") == "../gmlx-entry"
    lock.release()


# Volumes

def _vol(name="pg", size="8G"):
    return Mount(name, "/var/lib/postgresql", kind="volume", size=size)


def test_volume_lock_refuses_a_second_session(fake_container):
    held = session.lock_volumes([_vol()])
    with pytest.raises(SettingsError, match="in use by another launch session"):
        session.lock_volumes([_vol()])
    for lock in held:
        lock.release()
    assert not any(os.get_inheritable(lock.fd) for lock in session.lock_volumes([_vol()]))


def test_volume_mounted_elsewhere_is_refused():
    from gmlx.container.cli import Container
    other = Container("by-hand", "running", {}, "", "", volumes=["pg"])
    with pytest.raises(SettingsError, match="by-hand"):
        session.check_volumes_free([_vol()], [other])
    session.check_volumes_free([_vol("other")], [other])


def test_ensure_volumes_creates_and_reports_sizes(fake_container):
    fake_container.update(volumes=[{"name": "old", "size": None},
                                   {"name": "small", "size": "4G", "bytes": 4 << 30}])
    said = []
    session.ensure_volumes([_vol("pg"), _vol("old"), _vol("small")], said.append)
    assert ["volume", "create", "--label", "gmlx.launch=1", "-s", "8G", "pg"] in fake_container.log
    assert any("512 GB default" in line for line in said)
    assert any("has 4G, not the configured 8G" in line for line in said)


def test_volume_lines_use_allocated_blocks(fake_container, tmp_path, monkeypatch):
    img = tmp_path / "volume.img"
    with open(img, "wb") as f:
        f.truncate(8 << 30)                           # sparse: large apparent size
        f.write(b"x" * 4096)
    fake_container.update(volumes=[{"name": "pg", "size": "8G", "source": str(img)}])
    lines = session.volume_lines([_vol()])
    assert lines[0].startswith("[launch] volume pg at /var/lib/postgresql (8G limit, ")
    assert "8G used" not in lines[0]
    import shutil
    monkeypatch.setattr(shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(1, 1, 1 << 20))
    assert "warning: the Mac disk" in session.volume_lines([_vol()])[-1]


# open_when_ready

def test_open_when_ready_waits_for_an_http_response():
    """The relay accepts at once, so the first connections only get an end
    of file. The browser opens at the first HTTP answer."""
    answered = threading.Event()
    requests = []

    class Late(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(1)
            if not answered.is_set():
                self.close_connection = True       # an end of file: not ready
                return
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass
    server = http.server.HTTPServer(("127.0.0.1", 0), Late)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    opened = []
    stop = threading.Event()
    t = threading.Thread(target=session.open_when_ready,
                         args=(port, opened.append, stop), kwargs={"timeout": 20})
    t.start()
    deadline = time.monotonic() + 10
    while not requests:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    time.sleep(0.2)
    assert opened == []
    answered.set()
    t.join(10)
    server.shutdown()
    server.server_close()
    assert opened == [f"http://127.0.0.1:{port}/"]


def test_open_when_ready_gives_up_with_the_address():
    said = []
    session.open_when_ready(1, lambda url: None, threading.Event(), said.append, timeout=0.2)
    assert "http://127.0.0.1:1/" in said[0] and "127.0.0.1:$PORT" in said[0]


# The supervisor

def test_supervise_passes_values_only_in_the_child_env(fake_container, tmp_path):
    fake_container.update(run_rc=7)
    sess = session.new_session("pi", [])
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]))
    said = []
    rc = session.supervise(spec, api_targets=[("127.0.0.1", 9)], record={"name": sess.name},
                           say=said.append, summary=["[launch] summary"])
    assert rc == 7
    run = fake_container.load()["runs"][0]
    assert run["env"]["OPENAI_API_KEY"] == "sekrit"
    assert "sekrit" not in " ".join(run["argv"])
    assert said == ["[launch] summary"]
    assert not sess.dir.exists() and session.read_record("pi") is None


@pytest.mark.parametrize("stdin_terminal, same_group", [(True, True), (False, False)])
def test_supervise_keeps_a_terminal_reader_in_the_foreground(
        fake_container, tmp_path, monkeypatch, stdin_terminal, same_group):
    # stdin a terminal and stdout a pipe: no -t, but the child reads the
    # terminal, so a background group would stop it with SIGTTIN.
    monkeypatch.setattr(session, "stdin_is_terminal", lambda: stdin_terminal)
    sess = session.new_session("pi", [])
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]), tty=False)
    session.supervise(spec, api_targets=[("127.0.0.1", 9)], record={}, say=lambda line: None)
    run = fake_container.load()["runs"][0]
    assert (run["pgid"] == os.getpgrp()) is same_group
    assert "-t" not in run["argv"]


def test_supervise_keeps_lock_descriptors_out_of_the_child(fake_container, tmp_path):
    lock = session.try_session_lock("pi")
    assert lock is not None
    sess = session.new_session("pi", [])
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]))
    try:
        session.supervise(spec, api_targets=[("127.0.0.1", 9)], record={}, say=lambda line: None)
    finally:
        lock.release()
    files = fake_container.load()["runs"][0]["open_files"]
    assert files and not [f for f in files if f.endswith("session.lock")]


def test_supervise_gives_only_the_api_relay_the_request_head_deadline(
        fake_container, tmp_path, monkeypatch):
    made = []
    real = session.Relay

    def spy(*a, **k):
        made.append((k["name"], k.get("idle_until_head", False)))
        return real(*a, **k)
    monkeypatch.setattr(session, "Relay", spy)
    sess = session.new_session("pi", [])
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[6379]))
    session.supervise(spec, api_targets=[("127.0.0.1", 9)], record={}, say=lambda line: None)
    assert sorted(made) == [("gmlx api", True), ("port 6379", False)]


def test_supervise_stops_and_deletes_a_container_still_listed(fake_container, tmp_path):
    sess = session.new_session("pi", [])
    fake_container.update(containers=[{"name": sess.name}])
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]))
    session.supervise(spec, api_targets=[("127.0.0.1", 9)], record={}, say=lambda line: None)
    log = fake_container.load()["log"]
    assert ["stop", "--time", "5", sess.name] in log
    assert ["delete", "--force", sess.name] in log
    assert not sess.dir.exists()


def test_cleanup_errors_leave_the_exit_code_and_remove_the_folder(
        fake_container, tmp_path, monkeypatch):
    sess = session.new_session("pi", [])
    fake_container.update(containers=[{"name": sess.name}], run_rc=3)

    def stuck(name, *, timeout=10):
        raise session.cli.ContainerError("`container stop` gave no answer in 65 s.")
    monkeypatch.setattr(session.cli, "stop", stuck)
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]))
    rc = session.supervise(spec, api_targets=[("127.0.0.1", 9)], record={}, say=lambda line: None)
    assert rc == 3
    assert not sess.dir.exists()
    assert "cleanup: `container stop` gave no answer" in (
        session.cache_dir() / "last-pi.log").read_text()


def test_signal_thread_errors_go_to_the_log(monkeypatch):
    logged, done = [], threading.Event()

    def stuck(name, *, timeout=10):
        raise session.cli.ContainerError("stuck")
    monkeypatch.setattr(session.cli, "stop", stuck)
    monkeypatch.setattr(session.cli, "containers", lambda: [_listed("gmlx-pi-1")])
    sig = session._Signals("gmlx-pi-1", tty=False,
                           log=lambda line: (logged.append(line), done.set()))
    sig._on_term(signal.SIGTERM, None)
    assert done.wait(5)
    assert logged == ["signal: stuck"]


@pytest.mark.parametrize("family, host", [(socket.AF_INET, "127.0.0.1"),
                                          (socket.AF_INET, "0.0.0.0"),
                                          (socket.AF_INET6, "::")])
def test_supervise_refuses_a_busy_web_port(fake_container, tmp_path, family, host):
    # A bind to 127.0.0.1 succeeds while another program listens on the
    # wildcard address, so the port is probed first.
    busy = socket.socket(family)
    busy.bind((host, 0))
    busy.listen()
    port = busy.getsockname()[1]
    sess = session.new_session("dsh", [])
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]), web_port=port)
    try:
        with pytest.raises(SettingsError, match=rf"cannot listen on 127\.0\.0\.1:{port} for "
                           r"the web app: another program answers on .*Stop that program"):
            session.supervise(spec, api_targets=[("127.0.0.1", 9)], record={})
    finally:
        busy.close()
    assert fake_container.load().get("runs") is None
    assert not sess.dir.exists()


def test_token_url_opens_only_the_session_web_port(monkeypatch):
    import io
    out = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", type("S", (), {"buffer": out})())
    opened = []
    lines = (b"booting\n"
             b"dsh web: http://evil.example/?token=x\n"
             b"dsh web: http://127.0.0.1:3080/?token=abc\n"
             b"dsh web: http://127.0.0.1:3080/?token=again\n")
    session._tee_for_url(io.BytesIO(lines), r"dsh web: (\S+)", 3080, opened.append)
    assert opened == ["http://127.0.0.1:3080/?token=abc"]
    assert out.getvalue() == lines


def test_web_app_with_a_token_line_reads_no_terminal(tmp_path):
    spec = _spec(tmp_path, web_port=3080, interactive=False, url_pattern=r"x (\S+)")
    argv = session.compose_run_argv(spec)
    assert "-i" not in argv and "-t" not in argv


def _listed(name):
    from gmlx.container import cli
    return cli.Container(name=name, state="running", labels={}, image="", image_digest="")


def test_signals_stop_then_kill_then_end_the_cli(monkeypatch):
    from gmlx.container import cli
    calls = []
    monkeypatch.setattr(cli, "containers", lambda: [_listed("gmlx-pi-1")])
    monkeypatch.setattr(cli, "stop", lambda name, timeout=10: calls.append(("stop", name)))
    monkeypatch.setattr(cli, "kill", lambda name, signal=None: calls.append(("kill", signal)))
    monkeypatch.setattr(session._Signals, "_bg", staticmethod(lambda fn, *a, **k: fn(*a, **k)))
    child = subprocess.Popen(["sleep", "30"])
    sig = session._Signals("gmlx-pi-1", tty=False)
    sig.child = child
    sig._on_int(signal.SIGINT, None)
    sig._on_term(signal.SIGTERM, None)
    sig._on_term(signal.SIGHUP, None)
    assert calls == [("kill", "SIGINT"), ("stop", "gmlx-pi-1"), ("kill", None)]
    sig._on_term(signal.SIGTERM, None)
    assert child.wait(5) == -signal.SIGKILL
    tty = session._Signals("gmlx-pi-1", tty=True)
    tty._on_int(signal.SIGINT, None)                 # the terminal delivers it
    assert len(calls) == 3


def test_a_sigterm_before_the_container_exists_stops_it_once_listed(monkeypatch):
    from gmlx.container import cli
    listings = iter([[], [], [_listed("gmlx-pi-1")]])
    calls, done = [], threading.Event()
    monkeypatch.setattr(cli, "containers", lambda: next(listings))
    monkeypatch.setattr(cli, "stop", lambda name, timeout=10: (calls.append(name), done.set()))
    sig = session._Signals("gmlx-pi-1", tty=False)
    sig._on_term(signal.SIGTERM, None)
    assert done.wait(5) and calls == ["gmlx-pi-1"]


def test_a_pending_stop_gives_up_when_the_child_exits(monkeypatch):
    from gmlx.container import cli
    calls, logged = [], []
    monkeypatch.setattr(cli, "containers", lambda: [])
    monkeypatch.setattr(cli, "stop", lambda name, timeout=10: calls.append(name))
    monkeypatch.setattr(session._Signals, "_bg", staticmethod(lambda fn, *a, **k: fn(*a, **k)))
    sig = session._Signals("gmlx-pi-1", tty=False, log=logged.append)
    sig.done.set()
    sig._on_term(signal.SIGTERM, None)
    assert calls == []


def test_a_second_signal_before_the_container_exists_kills_the_cli(monkeypatch):
    from gmlx.container import cli
    monkeypatch.setattr(cli, "containers", lambda: [])
    monkeypatch.setattr(session._Signals, "_bg", staticmethod(lambda fn, *a, **k: None))
    child = subprocess.Popen(["sleep", "30"])
    sig = session._Signals("gmlx-pi-1", tty=False)
    sig.child = child
    sig._on_term(signal.SIGTERM, None)
    sig.count = 1
    sig._kill()
    assert child.wait(5) == -signal.SIGKILL


def test_signal_threads_log_any_error(monkeypatch):
    from gmlx.container import cli
    logged, done = [], threading.Event()

    def broken():
        raise OSError(24, "Too many open files")
    monkeypatch.setattr(cli, "containers", broken)
    sig = session._Signals("gmlx-pi-1", tty=False,
                           log=lambda line: (logged.append(line), done.set()))
    sig._on_term(signal.SIGTERM, None)
    assert done.wait(5) and "Too many open files" in logged[0]


def test_teardown_steps_run_even_when_each_fails(fake_container, tmp_path, monkeypatch):
    from gmlx.container import cli
    fake_container.update(run_rc=5)
    sess = session.new_session("pi", [])
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]))

    def emfile(*a, **k):
        raise OSError(24, "Too many open files")
    monkeypatch.setattr(session, "remove_record", emfile)
    monkeypatch.setattr(cli, "containers", emfile)
    monkeypatch.setattr(session.RelayLoop, "stop", emfile)
    rc = session.supervise(spec, api_targets=[("127.0.0.1", 9)], record={},
                           say=lambda line: None)
    assert rc == 5 and not sess.dir.exists()
    log = (session.cache_dir() / "last-pi.log").read_text()
    assert "cannot remove the session record" in log and "cannot stop the relay loop" in log


def test_signal_handlers_stay_until_teardown_ends(fake_container, tmp_path, monkeypatch):
    before = signal.getsignal(signal.SIGINT)
    seen = []

    def remove(name, *, stop, log):
        seen.append(signal.getsignal(signal.SIGINT))
    monkeypatch.setattr(session, "_remove_container", remove)
    sess = session.new_session("pi", [])
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]))
    session.supervise(spec, api_targets=[("127.0.0.1", 9)], record={}, say=lambda line: None)
    assert seen and seen[0] is not before and getattr(seen[0], "__self__", None) is not None
    assert signal.getsignal(signal.SIGINT) is before


def test_a_web_app_without_an_opener_only_prints_the_address(fake_container, tmp_path,
                                                            monkeypatch):
    monkeypatch.setattr(session, "open_when_ready",
                        lambda *a, **k: pytest.fail("no browser under --shell"))
    sess = session.new_session("open-webui", [])
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]), web_port=0)
    said = []
    session.supervise(spec, api_targets=[("127.0.0.1", 9)], record={}, say=said.append,
                      opener=None)
    assert said == ["[launch] open http://127.0.0.1:0/ in a browser"]


def test_the_tee_keeps_copying_when_the_opener_fails(monkeypatch):
    import io
    out = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", type("S", (), {"buffer": out})())
    logged = []

    def opener(url):
        raise RuntimeError("no browser")
    lines = b"dsh web: http://127.0.0.1:3080/?token=abc\n" + b"x\n" * 1000
    session._tee_for_url(io.BytesIO(lines), r"dsh web: (\S+)", 3080, opener, logged.append)
    assert out.getvalue() == lines
    assert logged == ["cannot open the browser (RuntimeError: no browser)"]


class _Chunks:
    """A pipe that hands out the given chunks, and fails on readline."""

    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.asked = []

    def read1(self, n):
        self.asked.append(n)
        return self.chunks.pop(0) if self.chunks else b""

    def readline(self):
        raise AssertionError("the reader must not wait for a newline")


def test_the_dsh_reader_copies_bounded_chunks_without_a_newline(monkeypatch):
    import io
    out = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", type("S", (), {"buffer": out})())
    blob = b"A" * (session.TEE_CHUNK * 3)
    stream = _Chunks([blob[i:i + session.TEE_CHUNK]
                      for i in range(0, len(blob), session.TEE_CHUNK)])
    session._tee_for_url(stream, r"dsh web: (\S+)", 3080, lambda url: None)
    assert out.getvalue() == blob
    assert set(stream.asked) == {session.TEE_CHUNK}


def test_the_dsh_url_split_across_reads_opens_whole(monkeypatch):
    import io
    monkeypatch.setattr(sys, "stdout", type("S", (), {"buffer": io.BytesIO()})())
    opened = []
    stream = _Chunks([b"x" * 5000 + b"dsh web: http://127.0.0.1:3080/?tok",
                      b"en=abc\n"])
    session._tee_for_url(stream, r"dsh web: (\S+)", 3080, opened.append)
    assert opened == ["http://127.0.0.1:3080/?token=abc"]


def test_a_dsh_url_cut_off_by_the_end_of_output_never_opens(monkeypatch):
    import io
    monkeypatch.setattr(sys, "stdout", type("S", (), {"buffer": io.BytesIO()})())
    opened = []
    session._tee_for_url(_Chunks([b"dsh web: http://127.0.0.1:3080/?tok"]),
                         r"dsh web: (\S+)", 3080, opened.append)
    assert opened == []


def test_the_session_log_is_private_and_never_follows_a_link(tmp_path):
    target = tmp_path / "elsewhere"
    target.write_text("keep")
    link = tmp_path / "last-pi.log"
    link.symlink_to(target)
    log = session._SessionLog(link)
    log("hello")
    log.close()
    assert target.read_text() == "keep"
    link.unlink()
    old = os.umask(0o002)
    try:
        log = session._SessionLog(link)
    finally:
        os.umask(old)
    log("hello")
    log.close()
    assert stat.S_IMODE(link.stat().st_mode) == 0o600 and "hello" in link.read_text()
    log = session._SessionLog(link)
    log("cannot open \x1b]52;c;ZXZpbA==\x07")
    log.close()
    assert "\x1b" not in link.read_text() and "\\x1b]52" in link.read_text()


def test_the_container_delete_gets_its_own_timeout(monkeypatch):
    from gmlx.container import cli
    seen = []
    monkeypatch.setattr(session, "_safe_containers", lambda: [_listed("gmlx-pi-1")])
    monkeypatch.setattr(cli, "stop", lambda name, timeout: None)
    monkeypatch.setattr(cli, "delete", lambda name: seen.append(cli._query_timeout))
    with cli.query_timeout(session.TEARDOWN_QUERY_TIMEOUT):
        session._remove_container("gmlx-pi-1", stop=True, log=lambda line: None)
    assert seen == [session.TEARDOWN_DELETE_TIMEOUT] == [30.0]


def test_a_container_left_after_the_cleanup_is_named(monkeypatch, capsys):
    logged = []
    monkeypatch.setattr(session, "_safe_containers", lambda: [_listed("gmlx-pi-1")])
    session._report_leftover("gmlx-pi-1", log=logged.append)
    err = capsys.readouterr().err
    assert "gmlx-pi-1 is still there" in err and "container delete --force gmlx-pi-1" in err
    assert logged
    monkeypatch.setattr(session, "_safe_containers", lambda: [])
    session._report_leftover("gmlx-pi-1", log=logged.append)
    assert capsys.readouterr().err == ""


def test_say_escapes_terminal_controls(capsys):
    session._say("[launch] sharing /x/\x1b]52;c;ZXZpbA==\x07\x1b[2A\x9b\x7fdone\r\n")
    out = capsys.readouterr().out
    assert "\x1b" not in out and "\x07" not in out and "\x9b" not in out and "\x7f" not in out
    assert out == ("[launch] sharing /x/\\x1b]52;c;ZXZpbA==\\x07\\x1b[2A\\x9b\\x7fdone"
                   "\\x0d\\x0a\n")


def test_the_session_log_stops_at_its_limit(tmp_path):
    log = session._SessionLog(tmp_path / "last-pi.log", limit=200)
    for i in range(100):
        log(f"line {i}")
    log.close()
    log("after close")                                   # never raises
    text = (tmp_path / "last-pi.log").read_text()
    assert len(text) <= 200 + 60 and text.rstrip().endswith("the log reached its size limit")


def test_supervise_raises_the_open_file_limit_first():
    """Each relayed connection holds a descriptor of the supervisor, so it
    raises its limit before any listener opens. gmlx/rlimit.py has its own
    tests."""
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(session.supervise))
    lines: dict = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            lines.setdefault(ast.unparse(n.func), n.lineno)
    assert "raise_nofile_limit" in lines
    assert lines["raise_nofile_limit"] < lines["RelayLoop"]


def test_cleanup_removes_only_stale_install_folders(fake_container, tmp_path):
    root = runtime.runtime_root()
    root.mkdir(parents=True, exist_ok=True)
    old, new = root / ".tmp-deadbeef", root / ".tmp-0badcafe"
    old.mkdir()
    new.mkdir()
    stale = time.time() - runtime.TMP_STALE - 60
    os.utime(old, (stale, stale))
    assert runtime.cleanup_runtime(keep=None) == [old]
    assert new.exists() and not old.exists()


def test_a_shell_on_a_web_app_names_the_address_without_open(fake_container, tmp_path):
    sess = session.new_session("open-webui", [])
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]), web_port=0,
                 shell=True)
    said = []
    session.supervise(spec, api_targets=[("127.0.0.1", 9)], record={}, say=said.append,
                      opener=None)
    assert said == ["[launch] the web app answers at http://127.0.0.1:0/ once you start "
                    "it from the shell"]


def test_a_container_run_that_cannot_start_is_a_clean_error(fake_container, tmp_path,
                                                            monkeypatch):
    from gmlx.container import cli

    def emfile(*a, **k):
        raise OSError(24, "Too many open files")
    monkeypatch.setattr(session.subprocess, "Popen", emfile)
    sess = session.new_session("pi", [])
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]))
    with pytest.raises(cli.ContainerError, match="cannot start `container run` "
                                                 r"\(Too many open files\)"):
        session.supervise(spec, api_targets=[("127.0.0.1", 9)], record={},
                          say=lambda line: None)
    assert not sess.dir.exists()


def test_session_folder_and_record_errors_are_clean(fake_container, tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setattr(session, "cache_dir", lambda: blocker)
    monkeypatch.setenv("TMPDIR", str(blocker))
    with pytest.raises(SettingsError, match="cannot create the session folder"):
        session.new_session("pi", [])
    monkeypatch.setattr(session, "record_path", lambda client: blocker / "session.json")
    with pytest.raises(SettingsError, match="cannot write the session record"):
        session.write_record("pi", {})


def test_a_sigint_without_a_terminal_waits_for_the_container(monkeypatch):
    from gmlx.container import cli
    listings = iter([[], [], [_listed("gmlx-pi-1")]])
    calls, done = [], threading.Event()
    monkeypatch.setattr(cli, "containers", lambda: next(listings))
    monkeypatch.setattr(cli, "kill", lambda name, signal=None: (calls.append((name, signal)),
                                                                done.set()))
    sig = session._Signals("gmlx-pi-1", tty=False)
    sig._on_int(signal.SIGINT, None)
    assert done.wait(5) and calls == [("gmlx-pi-1", "SIGINT")]


def test_teardown_queries_use_a_short_timeout(fake_container, tmp_path, monkeypatch):
    from gmlx.container import cli
    seen = []

    def remove(name, *, stop, log):
        seen.append(cli._query_timeout)
    monkeypatch.setattr(session, "_remove_container", remove)
    sess = session.new_session("pi", [])
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]))
    session.supervise(spec, api_targets=[("127.0.0.1", 9)], record={}, say=lambda line: None)
    assert seen == [session.TEARDOWN_QUERY_TIMEOUT] == [5.0]
    assert cli._query_timeout is None


def test_the_session_names_a_container_the_cleanup_left(fake_container, tmp_path,
                                                        monkeypatch, capsys):
    sess = session.new_session("pi", [])
    monkeypatch.setattr(session, "_remove_container", lambda name, *, stop, log: None)
    monkeypatch.setattr(session, "_safe_containers", lambda: [_listed(sess.name)])
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]))
    session.supervise(spec, api_targets=[("127.0.0.1", 9)], record={}, say=lambda line: None)
    assert f"container delete --force {sess.name}" in capsys.readouterr().err


def test_a_third_signal_abandons_a_teardown_that_waits(fake_container, tmp_path, monkeypatch):
    fake_container.update(run_rc=3)
    before = signal.getsignal(signal.SIGTERM)

    def hung(name, *, stop, log):
        for _ in range(3):
            os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(30)                             # a service that gives no answer
    monkeypatch.setattr(session, "_remove_container", hung)
    sess = session.new_session("pi", [])
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]))
    start = time.monotonic()
    rc = session.supervise(spec, api_targets=[("127.0.0.1", 9)], record={},
                           say=lambda line: None)
    assert rc == 3 and time.monotonic() - start < 10
    assert not sess.dir.exists() and signal.getsignal(signal.SIGTERM) is before
    assert "abandoned after a third signal" in (
        session.cache_dir() / "last-pi.log").read_text()


def test_signals_while_the_session_folder_is_removed_raise_nothing(fake_container, tmp_path,
                                                                    monkeypatch):
    fake_container.update(run_rc=3)
    real = session.shutil.rmtree

    sess = session.new_session("pi", [])

    def noisy(path, **kw):
        if Path(path) == sess.dir:
            for _ in range(3):
                os.kill(os.getpid(), signal.SIGTERM)
        real(path, **kw)
    monkeypatch.setattr(session.shutil, "rmtree", noisy)
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]))
    rc = session.supervise(spec, api_targets=[("127.0.0.1", 9)], record={},
                           say=lambda line: None)
    assert rc == 3 and not sess.dir.exists()


def test_a_low_open_file_limit_is_logged(fake_container, tmp_path, monkeypatch):
    monkeypatch.setattr(session, "raise_nofile_limit", lambda: 2048)
    sess = session.new_session("pi", [])
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]))
    session.supervise(spec, api_targets=[("127.0.0.1", 9)], record={}, say=lambda line: None)
    text = (session.cache_dir() / "last-pi.log").read_text()
    assert "can open only 2048 files at a time" in text
