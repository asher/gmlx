"""gmlx/container/session.py and runtime.py: the run command, the session
and volume locks, cleanup, the runtime folder and the supervisor."""

from __future__ import annotations

import http.server
import os
import signal
import socket
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
    session.write_record("pi", {"name": "gmlx-pi-1"})
    assert session.read_record("pi") == {"name": "gmlx-pi-1"}
    session.remove_record("pi")
    assert session.read_record("pi") is None


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
    session.cleanup_stale("pi", keep_runtime=None)
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
        if calls["n"] == 1:
            shutil.rmtree(Path(path).parent)          # a cleanup wins the race
            Path(path).parent.mkdir()
            Path(path).touch()
        return real(path, **kw)
    monkeypatch.setattr(runtime, "FileLock", racing)
    folder, lock = runtime.acquire_runtime(_entry(tmp_path))
    assert calls["n"] >= 2 and (folder / "gmlx-entry").is_file()
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
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    opened = []
    stop = threading.Event()
    t = threading.Thread(target=session.open_when_ready,
                         args=(port, opened.append, stop), kwargs={"timeout": 20})
    t.start()
    conn, _ = listener.accept()                        # a relay accepts at once
    conn.close()                                       # and gives EOF: not ready
    time.sleep(0.2)
    assert opened == []
    listener.close()

    class Ok(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass
    server = http.server.HTTPServer(("127.0.0.1", port), Ok)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    t.join(10)
    server.shutdown()
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


def test_supervise_refuses_a_busy_web_port(fake_container, tmp_path):
    busy = socket.socket()
    busy.bind(("127.0.0.1", 0))
    busy.listen()
    sess = session.new_session("dsh", [])
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]),
                 web_port=busy.getsockname()[1])
    with pytest.raises(SettingsError, match="busy"):
        session.supervise(spec, api_targets=[("127.0.0.1", 9)], record={})
    busy.close()
    assert fake_container.load().get("runs") is None


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


def test_signals_stop_then_kill_then_end_the_cli(monkeypatch):
    from gmlx.container import cli
    calls = []
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
