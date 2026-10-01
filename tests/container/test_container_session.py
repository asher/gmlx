"""gmlx/container/session.py and runtime.py: the run command, the session
and volume locks, cleanup, the runtime folder and the supervisor."""

from __future__ import annotations

import errno
import http.server
import json
import os
import pty
import select
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

from gmlx.container import runtime, session, settings
from gmlx.container.settings import ContainerPlan, Mount, SettingsError
from gmlx.container.state import FileLock

D1 = "sha256:" + "1" * 64


@pytest.fixture(autouse=True)
def _no_recheck(request, monkeypatch):
    """The plans here name folders that do not exist. The tests of the
    check itself, named test_recheck_*, run the real one."""
    if not request.node.name.startswith("test_recheck_"):
        monkeypatch.setattr(session, "recheck_sources", lambda spec: None)


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

@pytest.mark.parametrize("ssh_agent, socket_path, env_sock, ssh", [
    (True, "/private/tmp/env.sock", "/tmp/env.sock", True),     # the plan resolves it
    (True, None, "/tmp/env.sock", False),                       # never unchecked
    (True, None, None, False),
    (True, None, "", False),
    (True, "/tmp/own.sock", None, True),
    (False, None, "/tmp/env.sock", False),
])
def test_ssh_only_with_an_agent_to_forward(tmp_path, monkeypatch, ssh_agent, socket_path,
                                           env_sock, ssh):
    """Apple's runtime sets SSH_AUTH_SOCK in the guest for every --ssh, so
    launch passes it only when the Mac has an agent to forward."""
    if env_sock is None:
        monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
    else:
        monkeypatch.setenv("SSH_AUTH_SOCK", env_sock)
    plan = _plan(tmp_path, ssh_agent=ssh_agent, ssh_socket=socket_path)
    assert ("--ssh" in session.compose_run_argv(_spec(tmp_path, plan=plan))) is ssh


def test_golden_run_argv(tmp_path):
    argv = session.compose_run_argv(_spec(tmp_path))
    s = str(tmp_path / "sess")
    home = str(tmp_path / "home")
    assert argv[:8] == ["container", "run", "--rm", "--init", "--progress", "none",
                        "--name", "gmlx-pi-abc123"]
    labels = [argv[i + 1] for i, a in enumerate(argv) if a == "--label"]
    assert labels == ["gmlx.launch=1", "gmlx.launch.client=pi", "gmlx.launch.project=default",
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


def test_web_app_network_none_ssh_tty_and_shell(tmp_path, monkeypatch):
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/env.sock")
    spec = _spec(tmp_path, plan=_plan(tmp_path, network="none", ssh_agent=True, forward=[],
                                      ssh_socket="/private/tmp/env.sock"),
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
    short = session.new_session("pi", "app-12345678", [5432])
    tag = session._project_tag("app-12345678")
    assert short.dir.parent == session.cache_dir()
    assert short.dir.name.startswith(f"pi-{tag}-") and short.project == "app-12345678"
    assert oct(short.dir.stat().st_mode & 0o777) == "0o700"
    deep = tmp_path / ("d" * 90)
    monkeypatch.setenv("XDG_CACHE_HOME", str(deep))
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    long = session.new_session("pi", "app-12345678", [65535])
    assert long.dir.parent == tmp_path and long.dir.name.startswith(f"gmlx-launch-pi-{tag}-")


def test_session_lock_refuses_a_second_session_of_one_project(fake_container):
    first = session.try_session_lock("pi", "app-12345678")
    assert first is not None
    assert session.try_session_lock("pi", "app-12345678") is None
    other = session.try_session_lock("pi", "web-87654321")     # another project
    assert other is not None
    assert session.try_session_lock("omp", "app-12345678") is not None
    first.release()
    other.release()
    assert session.try_session_lock("pi", "app-12345678") is not None


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
    session.write_record("pi", "app-12345678", record)
    assert session.read_record("pi", "app-12345678") == record
    assert session.read_record("pi", "default") is None
    path = session.record_path("pi", "app-12345678")
    assert path.parent.name == "app-12345678" and stat.S_IMODE(path.stat().st_mode) == 0o600
    assert [p.name for p in path.parent.iterdir() if p.name.endswith(".tmp")] == []
    session.remove_record("pi", "app-12345678")
    assert session.read_record("pi", "app-12345678") is None


def test_a_record_keeps_the_command_and_web_address(fake_container):
    record = {"name": "gmlx-dsh-1", "workdir": "/w", "clipboard": False, "shares": [],
              "command": ["dsh", "--no-open"], "entrypoint": None, "project": None,
              "web": True, "web_port": 3080, "profile": "gmlx",
              "url": "http://127.0.0.1:3080/?token=x"}
    session.write_record("dsh", "default", record)
    assert session.read_record("dsh", "default") == record


@pytest.mark.parametrize("record", [
    {"name": "gmlx-pi-1"},
    {"name": "gmlx-pi-1", "workdir": "/w", "shares": [{"host": "/h"}]},
    {"name": "gmlx-pi-1", "workdir": "/w", "shares": "/h"},
    {"name": "gmlx-pi-1", "workdir": "/w", "shares": [], "command": "pi"},
    {"name": "gmlx-pi-1", "workdir": "/w", "shares": [], "web_port": True},
    ["gmlx-pi-1"],
])
def test_a_damaged_record_is_a_clean_error(fake_container, record):
    session.write_record("pi", "default", {"name": "x"})       # creates the folder
    session.record_path("pi", "default").write_text(json.dumps(record))
    with pytest.raises(SettingsError, match="session file .* is damaged"):
        session.read_record("pi", "default")


def test_a_deeply_nested_record_is_a_clean_error(fake_container):
    session.write_record("pi", "default", {"name": "x"})
    session.record_path("pi", "default").write_text("[" * 100_000 + "]" * 100_000)
    with pytest.raises(SettingsError, match="is damaged"):
        session.read_record("pi", "default")


def test_the_record_is_never_written_through_a_planted_temporary_link(fake_container,
                                                                      tmp_path, monkeypatch):
    target = tmp_path / "elsewhere"
    target.write_text("keep")
    monkeypatch.setattr(session.secrets, "token_hex", lambda n: "fixed")
    folder = session.settings.project_dir("pi", "default")
    tmp = folder / f".session.json.{os.getpid()}.fixed.tmp"
    tmp.symlink_to(target)
    with pytest.raises(SettingsError, match="cannot write the session file"):
        session.write_record("pi", "default", {"name": "x"})
    assert target.read_text() == "keep"


def test_records_lists_each_project_with_a_readable_record(fake_container):
    good = {"name": "gmlx-pi-1", "workdir": "/w", "shares": []}
    session.write_record("pi", "a-11111111", good)
    session.write_record("pi", "b-22222222", {**good, "name": "gmlx-pi-2"})
    session.write_record("pi", "c-33333333", good)
    session.record_path("pi", "c-33333333").write_text("{")
    session.settings.project_dir("pi", "d-44444444")
    assert [(p, r["name"]) for p, r in session.records("pi")] == [
        ("a-11111111", "gmlx-pi-1"), ("b-22222222", "gmlx-pi-2")]
    assert session.records("omp") == []


def test_a_record_runs_only_while_its_labeled_container_runs():
    from gmlx.container.cli import Container
    record = {"name": "gmlx-pi-1"}
    running = Container("gmlx-pi-1", "running", _labels("pi", "a-11111111"), "", "")
    assert session.record_runs("pi", "a-11111111", record, [running])
    assert not session.record_runs("pi", "b-22222222", record, [running])
    assert not session.record_runs("omp", "a-11111111", record, [running])
    stopped = Container("gmlx-pi-1", "stopped", _labels("pi", "a-11111111"), "", "")
    assert not session.record_runs("pi", "a-11111111", record, [stopped])
    assert not session.record_runs("pi", "a-11111111", {"name": "gmlx-pi-2"}, [running])


# Cleanup

def _labels(client, project=None):
    labels = {"gmlx.launch": "1", "gmlx.launch.client": client}
    if project is not None:
        labels["gmlx.launch.project"] = project
    return labels


def test_cleanup_stale_names_a_container_its_delete_left(fake_container):
    fake_container.update(containers=[
        {"name": "gmlx-pi-aaaaaa", "labels": _labels("pi", "app-12345678")}])
    said = []
    session.cleanup_stale("pi", "app-12345678", keep_runtime=None, say=said.append)
    assert said == ["[launch] the leftover container gmlx-pi-aaaaaa of an earlier session is "
                    "still there. Remove it with: container delete --force gmlx-pi-aaaaaa"]


def test_cleanup_stale_touches_only_the_launching_project(fake_container, tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    fake_container.update(delete_removes=True, containers=[
        {"name": "gmlx-pi-aaaaaa", "labels": _labels("pi", "app-12345678")},
        {"name": "gmlx-pi-ffffff", "labels": _labels("pi", "web-87654321")},
        {"name": "gmlx-omp-bbbbbb", "labels": _labels("omp", "app-12345678")}])
    tag, other = session._project_tag("app-12345678"), session._project_tag("web-87654321")
    ours = session.cache_dir() / f"pi-{tag}-aaaaaa"
    ours.mkdir()
    tmp_ours = tmp_path / f"gmlx-launch-pi-{tag}-cccccc"
    tmp_ours.mkdir()
    sibling = session.cache_dir() / f"pi-{other}-ffffff"
    sibling.mkdir()
    theirs = session.cache_dir() / f"omp-{tag}-dddddd"
    theirs.mkdir()
    lookalike = session.cache_dir() / "pi-extra-eeeeee"
    lookalike.mkdir()
    said = []
    session.cleanup_stale("pi", "app-12345678", keep_runtime=None, say=said.append)
    assert said == ["[launch] removed the leftover container gmlx-pi-aaaaaa of an earlier "
                    "session"]
    assert ["stop", "--time", "5", "gmlx-pi-aaaaaa"] in fake_container.log
    assert ["delete", "--force", "gmlx-pi-aaaaaa"] in fake_container.log
    assert not any("gmlx-omp-bbbbbb" in a or "gmlx-pi-ffffff" in a for a in fake_container.log)
    assert not ours.exists() and not tmp_ours.exists()
    assert theirs.exists() and sibling.exists() and lookalike.exists()
    # The other project's lock was never taken.
    assert not (session.settings.project_dir_path("pi", "web-87654321") / "session.lock").exists()


def test_a_session_lock_on_a_removed_file_is_taken_again(monkeypatch):
    real, taken = session.FileLock, []

    def removed_after_the_open(path, **kw):
        lock = real(path, **kw)
        if not taken:                     # another launch removed the folder meanwhile
            os.unlink(path)
        taken.append(lock)
        return lock
    monkeypatch.setattr(session, "FileLock", removed_after_the_open)
    lock = session.try_session_lock("pi", "app-12345678")
    assert lock is taken[1] and lock.still_current() and taken[0].fd is None
    lock.release()


def test_a_session_lock_whose_folder_went_before_the_open_is_taken_again(monkeypatch):
    real, tries = session.FileLock, []

    def folder_gone(path, **kw):
        tries.append(path)
        if len(tries) == 1:
            # A joining launch dropped the empty folder after the mkdir.
            path.parent.rmdir()
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(path))
        return real(path, **kw)
    monkeypatch.setattr(session, "FileLock", folder_gone)
    lock = session.try_session_lock("pi", "app-12345678")
    assert lock is not None and lock.still_current() and len(tries) == 2
    lock.release()


def test_a_session_lock_stops_trying_when_the_folder_keeps_going(monkeypatch):
    tries = []

    def gone(path, **kw):
        tries.append(path)
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(path))
    monkeypatch.setattr(session, "FileLock", gone)
    with pytest.raises(FileNotFoundError):
        session.try_session_lock("pi", "app-12345678")
    assert len(tries) == session._LOCK_TRIES


def test_an_unused_project_folder_is_dropped_only_with_nothing_but_the_lock():
    lock = session.try_session_lock("pi", "app-12345678")
    assert lock is not None
    session.drop_unused_project("pi", "app-12345678", lock)
    assert not session.settings.project_dir_path("pi", "app-12345678").exists()
    lock.release()
    lock = session.try_session_lock("pi", "web-87654321")
    assert lock is not None
    session.settings.private_home("pi", "web-87654321")
    session.drop_unused_project("pi", "web-87654321", lock)
    assert (session.settings.project_dir_path("pi", "web-87654321") / "session.lock").exists()
    lock.release()


def test_orphan_notices_list_dead_launches_of_other_clients():
    from gmlx.container.cli import Container
    dead = Container("gmlx-omp-1", "running", {"gmlx.launch": "1", "gmlx.launch.client": "omp",
                                                "gmlx.launch.pid": "999999"}, "", "",
                     memory_bytes=4 << 30)
    live = Container("gmlx-goose-1", "running", {"gmlx.launch": "1",
                                                  "gmlx.launch.client": "goose",
                                                  "gmlx.launch.pid": str(os.getpid())}, "", "")
    mine = Container("gmlx-pi-1", "running", {**_labels("pi", "app-12345678"),
                                              "gmlx.launch.pid": "999999"}, "", "")
    sibling = Container("gmlx-pi-2", "running", {**_labels("pi", "web-87654321"),
                                                 "gmlx.launch.pid": "999999"}, "", "")
    lines = session.orphan_notices("pi", "app-12345678", [dead, live, mine, sibling])
    assert lines == ["[launch] gmlx-omp-1 from an earlier omp launch is still running and "
                     "holds 4G of memory. Stop it with: container stop gmlx-omp-1",
                     "[launch] gmlx-pi-2 from an earlier pi launch is still running. Stop it "
                     "with: container stop gmlx-pi-2"]


def test_the_memory_line_counts_every_running_launch_container(monkeypatch):
    from gmlx.container.cli import Container
    monkeypatch.setattr(session, "mac_memory_bytes", lambda: 64 << 30)
    other = Container("gmlx-omp-1", "running", _labels("omp", "default"), "", "",
                      memory_bytes=8 << 30)
    stopped = Container("gmlx-pi-1", "stopped", _labels("pi", "default"), "", "",
                        memory_bytes=8 << 30)
    assert session.memory_line([stopped], "4G") is None
    assert session.memory_line([other, stopped], "4G") == (
        "[launch] with 1 other launch container running, launch containers will hold 12G of "
        "the Mac's 64G of memory, which the model server cannot use.")


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
    sess = session.new_session("pi", "default", [])
    assert sess.dir.parent == Path(short_root)
    monkeypatch.setenv("TMPDIR", f"{short_root}/c:d")
    with pytest.raises(SettingsError, match="contains a colon"):
        session.new_session("pi", "default", [])


def test_a_runtime_folder_container_cannot_mount_is_refused(fake_container, tmp_path,
                                                            monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "a,b"))
    with pytest.raises(SettingsError, match="launch's program folder .* contains a comma"):
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
    with pytest.raises(settings.Busy, match="in use by another launch session"):
        session.lock_volumes([_vol()])
    for lock in held:
        lock.release()
    assert not any(os.get_inheritable(lock.fd) for lock in session.lock_volumes([_vol()]))


def test_volume_mounted_elsewhere_is_refused():
    from gmlx.container.cli import Container
    other = Container("by-hand", "running", {}, "", "", volumes=["pg"])
    with pytest.raises(settings.Busy, match="by-hand"):
        session.check_volumes_free([_vol()], [other])
    session.check_volumes_free([_vol("other")], [other])


def test_a_size_of_a_few_kibibytes_reads_as_under_1m():
    assert session.gb(0) == "0M"
    assert session.gb(300 << 10) == "under 1M"
    assert session.gb(1 << 19) == "under 1M"
    assert session.gb((1 << 19) + 4096) == "1M"
    assert session.gb(3 << 30) == "3G"


def test_ensure_volumes_creates_and_reports_sizes(fake_container):
    fake_container.update(volumes=[{"name": "old", "size": None},
                                   {"name": "small", "size": "4G", "bytes": 4 << 30}])
    said = []
    session.ensure_volumes([_vol("pg"), _vol("old"), _vol("small")], said.append)
    assert ["volume", "create", "--label", "gmlx.launch=1", "-s", "8G", "pg"] in fake_container.log
    assert said == [
        "[launch] warning: the volume old was created without a size, so it has Apple's "
        "512 GB default. Deleting it loses its data, and the next launch creates it with the "
        "configured size. Delete it with: container volume delete old",
        "[launch] warning: the volume small has 4G, not the configured 8G, because a size "
        "applies only when a volume is created. Deleting it loses its data, and the next "
        "launch creates it with the configured size. Delete it with: container volume "
        "delete small"]


def test_volume_lines_use_allocated_blocks(fake_container, tmp_path, monkeypatch):
    img = tmp_path / "volume.img"
    with open(img, "wb") as f:
        f.truncate(8 << 30)                           # sparse: large apparent size
        f.write(b"x" * 4096)
    fake_container.update(volumes=[{"name": "pg", "size": "8G", "bytes": 8 << 30,
                                    "source": str(img)}])
    lines = session.volume_lines([_vol()])
    assert lines[0].startswith("[launch] volume pg at /var/lib/postgresql (8G limit, ")
    assert "8G used" not in lines[0]
    import shutil
    monkeypatch.setattr(shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(1, 1, 1 << 20))
    assert "warning: the Mac disk" in session.volume_lines([_vol()])[-1]


def test_volume_lines_show_the_size_the_volume_has(fake_container):
    """A size applies only when a volume is created, so the summary shows the
    size the volume has, not the configured one the warning contrasts."""
    fake_container.update(volumes=[{"name": "pg", "size": "4G", "bytes": 4 << 30},
                                   {"name": "old", "size": None}])
    lines = session.volume_lines([_vol("pg"), _vol("old")])
    assert lines[0].startswith("[launch] volume pg at /var/lib/postgresql (4G limit, ")
    assert lines[1].startswith("[launch] volume old at /var/lib/postgresql (512G limit, ")


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
    assert "the browser was not opened" in said[0]
    said.clear()
    session.open_when_ready(1, lambda url: None, threading.Event(), said.append, timeout=0.2,
                            browser=False)
    assert "browser" not in said[0] and "127.0.0.1:$PORT" in said[0]


# The supervisor

def test_supervise_passes_values_only_in_the_child_env(fake_container, tmp_path):
    fake_container.update(run_rc=7)
    sess = session.new_session("pi", "default", [])
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]))
    said = []
    rc = session.supervise(spec, api_targets=[("127.0.0.1", 9)], record={"name": sess.name},
                           say=said.append, summary=["[launch] summary"])
    assert rc == 7
    run = fake_container.load()["runs"][0]
    assert run["env"]["OPENAI_API_KEY"] == "sekrit"
    assert "sekrit" not in " ".join(run["argv"])
    assert said == ["[launch] summary"]
    assert not sess.dir.exists() and session.read_record("pi", "default") is None


def test_supervise_runs_on_start_once_container_run_has_started(fake_container, tmp_path):
    sess = session.new_session("pi", "default", [])
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]))
    started = []
    session.supervise(spec, api_targets=[("127.0.0.1", 9)], record={}, say=lambda line: None,
                      on_start=lambda: started.append(True))
    assert started == [True] and fake_container.load()["runs"]


def test_a_start_record_that_fails_never_ends_the_session(fake_container, tmp_path):
    fake_container.update(run_rc=7)
    sess = session.new_session("pi", "default", [])
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]))

    def fail():
        raise RuntimeError("no record")
    assert session.supervise(spec, api_targets=[("127.0.0.1", 9)], record={},
                             say=lambda line: None, on_start=fail) == 7


def test_supervise_marks_the_record_as_ending_until_the_container_is_gone(
        fake_container, tmp_path, monkeypatch):
    """A launch that would join a session in its teardown hears that it
    ends, not that it starts."""
    sess = session.new_session("pi", "default", [])
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]))
    record = {"name": sess.name, "workdir": "/w", "clipboard": False, "shares": []}
    seen = []
    monkeypatch.setattr(session, "_remove_container",
                        lambda name, **kw: seen.append(session.read_record("pi", "default")))
    session.supervise(spec, api_targets=[("127.0.0.1", 9)], record=record,
                      say=lambda line: None)
    assert seen == [{**record, "ending": True, "pid": os.getpid()}]
    assert session.session_state("pi", "default", seen[0], []) == "ending"
    assert session.read_record("pi", "default") is None


@pytest.mark.parametrize("stdin_terminal, same_group", [(True, True), (False, False)])
def test_supervise_keeps_a_terminal_reader_in_the_foreground(
        fake_container, tmp_path, monkeypatch, stdin_terminal, same_group):
    # stdin a terminal and stdout a pipe: no -t, but the child reads the
    # terminal, so a background group would stop it with SIGTTIN.
    monkeypatch.setattr(session, "stdin_is_terminal", lambda: stdin_terminal)
    sess = session.new_session("pi", "default", [])
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]), tty=False)
    session.supervise(spec, api_targets=[("127.0.0.1", 9)], record={}, say=lambda line: None)
    run = fake_container.load()["runs"][0]
    assert (run["pgid"] == os.getpgrp()) is same_group
    assert "-t" not in run["argv"]


def test_supervise_keeps_lock_descriptors_out_of_the_child(fake_container, tmp_path):
    lock = session.try_session_lock("pi", "default")
    assert lock is not None
    sess = session.new_session("pi", "default", [])
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
        made.append((k["name"], k.get("idle_until_head", False),
                     k.get("max_connections", session.CONNECTIONS_MAX)))
        return real(*a, **k)
    monkeypatch.setattr(session, "Relay", spy)
    sess = session.new_session("pi", "default", [])
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[6379]))
    session.supervise(spec, api_targets=[("127.0.0.1", 9)], record={}, say=lambda line: None)
    # Without a session socket the API relay keeps its own cap, and a
    # forwarded port holds fewer, so the guest cannot take every client slot
    # of a Mac service.
    assert sorted(made) == [("gmlx api", True, session.CONNECTIONS_MAX),
                            ("port 6379", False, session.FORWARD_CONNECTIONS_MAX)]


def test_supervise_stops_and_deletes_a_container_still_listed(fake_container, tmp_path):
    sess = session.new_session("pi", "default", [])
    fake_container.update(containers=[{"name": sess.name}])
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]))
    session.supervise(spec, api_targets=[("127.0.0.1", 9)], record={}, say=lambda line: None)
    log = fake_container.load()["log"]
    assert ["stop", "--time", "5", sess.name] in log
    assert ["delete", "--force", sess.name] in log
    assert not sess.dir.exists()


def test_cleanup_errors_leave_the_exit_code_and_remove_the_folder(
        fake_container, tmp_path, monkeypatch):
    sess = session.new_session("pi", "default", [])
    fake_container.update(containers=[{"name": sess.name}], run_rc=3)

    def stuck(name, *, timeout=10):
        raise session.cli.ContainerError("`container stop` gave no answer in 65 s.")
    monkeypatch.setattr(session.cli, "stop", stuck)
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]))
    rc = session.supervise(spec, api_targets=[("127.0.0.1", 9)], record={}, say=lambda line: None)
    assert rc == 3
    assert not sess.dir.exists()
    assert "cleanup: `container stop` gave no answer" in (
        session.cache_dir() / "last-pi-default.log").read_text()


def test_signal_thread_errors_go_to_the_log(monkeypatch):
    logged, done = [], threading.Event()

    def stuck(name, *, timeout=10):
        raise session.cli.ContainerError("stuck")
    monkeypatch.setattr(session.cli, "stop", stuck)
    monkeypatch.setattr(session.cli, "containers", lambda **kw: [_listed("gmlx-pi-1")])
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
    sess = session.new_session("dsh", "default", [])
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]), web_port=port)
    try:
        with pytest.raises(settings.Busy, match=rf"cannot listen on 127\.0\.0\.1:{port} for "
                           r"the web app \(another program answers on .*Stop that program"):
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


def test_the_token_url_is_recorded_without_a_browser(monkeypatch):
    """A second launch opens the dsh web app from the session record, so the
    address is recorded even when this launch opens no browser."""
    import io
    monkeypatch.setattr(sys, "stdout", type("S", (), {"buffer": io.BytesIO()})())
    found = []
    lines = (b"dsh web: http://evil.example/?token=x\n"
             b"dsh web: http://127.0.0.1:3080/?token=abc\n")
    session._tee_for_url(io.BytesIO(lines), r"dsh web: (\S+)", 3080, None, found=found.append)
    assert found == ["http://127.0.0.1:3080/?token=abc"]


def test_supervise_records_the_token_url(fake_container, tmp_path, monkeypatch):
    sess = session.new_session("dsh", "default", [])
    record = {"name": sess.name, "workdir": "/w", "clipboard": False, "shares": [],
              "web": True, "web_port": 3080}
    seen = []
    real = session.write_record

    def spy(client, project, rec):
        seen.append(dict(rec))
        real(client, project, rec)
    monkeypatch.setattr(session, "write_record", spy)
    fake = tmp_path / "container"
    fake.write_text("#!/bin/sh\necho 'dsh web: http://127.0.0.1:3080/?token=t'\n")
    fake.chmod(0o755)
    monkeypatch.setattr(session.cli, "find", lambda: str(fake))
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]), web_port=None,
                 interactive=False, url_pattern=r"dsh web: (\S+)")
    spec.web_port = 3080
    monkeypatch.setattr(session, "_listen", lambda make, addr, what: type(
        "R", (), {"close": lambda self: None})())
    monkeypatch.setattr(sys, "stdout", type("S", (), {"buffer": __import__("io").BytesIO()})())
    session.supervise(spec, api_targets=None, record=record, say=lambda line: None)
    assert seen[-1]["url"] == "http://127.0.0.1:3080/?token=t"
    assert session.read_record("dsh", "default") is None      # removed at the end


def test_the_token_url_holds_no_terminal_controls(monkeypatch):
    import io

    from gmlx.commands import launch_container as lc
    monkeypatch.setattr(sys, "stdout", type("S", (), {"buffer": io.BytesIO()})())
    opened = []
    lines = (b"dsh web: http://127.0.0.1:3080/?t=\x1b]52;c;x\x07\n"
             b"dsh web: http://127.0.0.1:3080\x1b/?t=1\n"
             b"dsh web: http://127.0.0.1:3080/?token=ok\n")
    session._tee_for_url(io.BytesIO(lines), lc._DSH_URL_LINE, 3080, opened.append)
    # A line with a control in its URL opens nothing, not even the part
    # before the control.
    assert opened == ["http://127.0.0.1:3080/?token=ok"]


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
    monkeypatch.setattr(cli, "containers", lambda **kw: [_listed("gmlx-pi-1")])
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


@pytest.mark.parametrize("sig", [signal.SIGHUP, signal.SIGTERM, signal.SIGINT])
def test_the_session_leaves_a_signal_ignored_on_entry(sig):
    """nohup leaves SIGHUP ignored, so the session does not stop on it, and
    ``container run`` inherits the ignored signal."""
    saved = signal.signal(sig, signal.SIG_IGN)
    try:
        handlers = session._Signals("gmlx-pi-1", tty=False)
        handlers.install()
        assert signal.getsignal(sig) == signal.SIG_IGN
        child = subprocess.run([sys.executable, "-c", "import signal, sys; "
                                f"sys.exit(signal.getsignal({int(sig)}) == signal.SIG_IGN)"])
        handlers.restore()
        assert child.returncode == 1 and signal.getsignal(sig) == signal.SIG_IGN
    finally:
        signal.signal(sig, saved)


def test_a_sigterm_before_the_container_exists_stops_it_once_listed(monkeypatch):
    from gmlx.container import cli
    listings = iter([[], [], [_listed("gmlx-pi-1")]])
    calls, done = [], threading.Event()
    monkeypatch.setattr(cli, "containers", lambda **kw: next(listings))
    monkeypatch.setattr(cli, "stop", lambda name, timeout=10: (calls.append(name), done.set()))
    sig = session._Signals("gmlx-pi-1", tty=False)
    sig._on_term(signal.SIGTERM, None)
    assert done.wait(5) and calls == ["gmlx-pi-1"]


def test_a_pending_stop_gives_up_when_the_child_exits(monkeypatch):
    from gmlx.container import cli
    calls, logged = [], []
    monkeypatch.setattr(cli, "containers", lambda **kw: [])
    monkeypatch.setattr(cli, "stop", lambda name, timeout=10: calls.append(name))
    monkeypatch.setattr(session._Signals, "_bg", staticmethod(lambda fn, *a, **k: fn(*a, **k)))
    sig = session._Signals("gmlx-pi-1", tty=False, log=logged.append)
    sig.done.set()
    sig._on_term(signal.SIGTERM, None)
    assert calls == []


def test_a_second_signal_before_the_container_exists_kills_the_cli(monkeypatch):
    from gmlx.container import cli
    monkeypatch.setattr(cli, "containers", lambda **kw: [])
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

    def broken(**kw):
        raise OSError(24, "Too many open files")
    monkeypatch.setattr(cli, "containers", broken)
    sig = session._Signals("gmlx-pi-1", tty=False,
                           log=lambda line: (logged.append(line), done.set()))
    sig._on_term(signal.SIGTERM, None)
    assert done.wait(5) and "Too many open files" in logged[0]


def test_teardown_steps_run_even_when_each_fails(fake_container, tmp_path, monkeypatch):
    from gmlx.container import cli
    fake_container.update(run_rc=5)
    sess = session.new_session("pi", "default", [])
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]))

    def emfile(*a, **k):
        raise OSError(24, "Too many open files")
    monkeypatch.setattr(session, "remove_record", emfile)
    monkeypatch.setattr(cli, "containers", emfile)
    monkeypatch.setattr(session.RelayLoop, "stop", emfile)
    rc = session.supervise(spec, api_targets=[("127.0.0.1", 9)], record={},
                           say=lambda line: None)
    assert rc == 5 and not sess.dir.exists()
    log = (session.cache_dir() / "last-pi-default.log").read_text()
    assert "cannot remove the session record" in log and "cannot stop the relay loop" in log


def test_signal_handlers_stay_until_teardown_ends(fake_container, tmp_path, monkeypatch):
    before = signal.getsignal(signal.SIGINT)
    seen = []

    def remove(name, *, stop, log):
        seen.append(signal.getsignal(signal.SIGINT))
    monkeypatch.setattr(session, "_remove_container", remove)
    sess = session.new_session("pi", "default", [])
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]))
    session.supervise(spec, api_targets=[("127.0.0.1", 9)], record={}, say=lambda line: None)
    assert seen and seen[0] is not before and getattr(seen[0], "__self__", None) is not None
    assert signal.getsignal(signal.SIGINT) is before


def test_a_web_app_without_an_opener_prints_the_address_once_it_answers(
        fake_container, tmp_path, monkeypatch):
    """The app can take minutes to start, so an address printed at once
    would give a refused connection."""
    waits = []

    def wait(port, ready, stop, say, *, browser):
        waits.append((port, browser))
        ready(f"http://127.0.0.1:{port}/")
    monkeypatch.setattr(session, "open_when_ready", wait)
    sess = session.new_session("open-webui", "default", [])
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]), web_port=0)
    said = []
    session.supervise(spec, api_targets=[("127.0.0.1", 9)], record={}, say=said.append,
                      opener=None)
    assert waits == [(0, False)]
    assert said == ["[launch] the web app answers at http://127.0.0.1:0/"]


def test_a_web_app_with_an_opener_says_launch_opens_it(fake_container, tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(session, "open_when_ready",
                        lambda port, opener, stop, say, *, browser: opened.append(port))
    sess = session.new_session("open-webui", "default", [])
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]), web_port=0)
    said = []
    session.supervise(spec, api_targets=[("127.0.0.1", 9)], record={}, say=said.append,
                      opener=lambda url: None)
    assert said == ["[launch] opening http://127.0.0.1:0/ in your browser once the app "
                    "answers"]
    assert opened == [0]


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


def test_the_cleanup_calls_run_in_a_process_group_of_their_own(fake_container,
                                                               monkeypatch):
    """A signal can start the cleanup, and the shell can send a second one
    while it runs."""
    from gmlx.container import cli
    groups = []
    real = cli.subprocess.Popen

    def popen(argv, **kw):
        groups.append((argv[1], kw.get("process_group"), kw.get("stdin")))
        return real(argv, **kw)
    fake_container.update(containers=[{"name": "gmlx-pi-1"}])
    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    session._remove_container("gmlx-pi-1", stop=True, log=lambda line: None)
    session._report_leftover("gmlx-pi-1", log=lambda line: None)
    assert groups == [(call, 0, subprocess.DEVNULL) for call in ("ls", "stop", "delete", "ls")]


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


def test_printed_names_escape_format_and_separator_characters():
    """A right-to-left override would print a shared folder's name in
    another order than the one shared."""
    from gmlx.container import images
    from gmlx.container.text import printable
    name = "/x/evil\u202etxt.exe\u2028line\u00a0nbsp \u200bzw\U000e0041tag"
    out = printable(name)
    assert out == "/x/evil\\u202etxt.exe\\u2028line\\xa0nbsp \\u200bzw\\U000e0041tag"
    assert images._shown(name) == out
    assert printable("caf\u00e9 \u65e5\u672c \u2713") == "caf\u00e9 \u65e5\u672c \u2713"


def test_guest_lines_are_logged_once_a_minute_for_each_kind(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(session.time, "monotonic", lambda: clock[0])
    log = session._SessionLog(tmp_path / "last-pi.log")
    for i in range(500):
        log.guest(f"gmlx api: cannot relay a connection ([Errno {i}] refused)")
        log.guest("clipboard: request too long, connection closed")
    for i in range(3):
        log.guest_event(f"clipboard: sent an image of {i + 1} bytes")
    clock[0] += 61
    log.guest("gmlx api: cannot relay a connection ([Errno 61] refused)")
    log.guest("clipboard: request too long, connection closed")
    log.close()
    lines = (tmp_path / "last-pi.log").read_text().splitlines()
    relayed = [x for x in lines if "cannot relay" in x]
    assert len(relayed) == 2 and relayed[1].endswith(
        "(and 499 more like it since the last one logged)")
    assert len([x for x in lines if "sent an image" in x]) == 3
    assert len([x for x in lines if "request too long" in x]) == 2
    assert not any("were not logged" in x for x in lines)      # nothing left over


def test_the_log_keeps_room_for_launch_lines_after_a_guest_fills_it(tmp_path):
    log = session._SessionLog(tmp_path / "last-pi.log", limit=4000, reserve=1000, every=0)
    for i in range(200):
        log.guest(f"gmlx api: kind {chr(65 + i % 26)}{i}")
    log.guest_event("clipboard: sent an image of 5 bytes")
    log("cleanup: stopped gmlx-pi-1")
    log.close()
    text = (tmp_path / "last-pi.log").read_text()
    assert "size limit for lines the container causes" in text
    assert "sent an image" not in text
    assert text.rstrip().endswith("cleanup: stopped gmlx-pi-1")
    assert len(text) <= 4000


def test_the_log_counts_skipped_guest_lines_at_close(tmp_path):
    log = session._SessionLog(tmp_path / "last-pi.log")
    for _ in range(5):
        log.guest("clipboard: too many requests are waiting, answered busy")
    log.close()
    text = (tmp_path / "last-pi.log").read_text()
    assert text.count("answered busy") == 2
    assert "clipboard: too many requests are waiting, answered busy: 4 more like it " \
           "were not logged" in text


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
    sess = session.new_session("open-webui", "default", [])
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
    sess = session.new_session("pi", "default", [])
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
        session.new_session("pi", "default", [])
    monkeypatch.setattr(session, "record_path", lambda client, project: blocker / "session.json")
    with pytest.raises(SettingsError, match="cannot write the session file"):
        session.write_record("pi", "default", {})


def test_a_sigint_without_a_terminal_waits_for_the_container(monkeypatch):
    from gmlx.container import cli
    listings = iter([[], [], [_listed("gmlx-pi-1")]])
    calls, done = [], threading.Event()
    monkeypatch.setattr(cli, "containers", lambda **kw: next(listings))
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
    sess = session.new_session("pi", "default", [])
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]))
    session.supervise(spec, api_targets=[("127.0.0.1", 9)], record={}, say=lambda line: None)
    assert seen == [session.TEARDOWN_QUERY_TIMEOUT] == [5.0]
    assert cli._query_timeout is None


def test_the_session_names_a_container_the_cleanup_left(fake_container, tmp_path,
                                                        monkeypatch, capsys):
    sess = session.new_session("pi", "default", [])
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
    sess = session.new_session("pi", "default", [])
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]))
    start = time.monotonic()
    rc = session.supervise(spec, api_targets=[("127.0.0.1", 9)], record={},
                           say=lambda line: None)
    assert rc == 3 and time.monotonic() - start < 10
    assert not sess.dir.exists() and signal.getsignal(signal.SIGTERM) is before
    assert "abandoned after a third signal" in (
        session.cache_dir() / "last-pi-default.log").read_text()


def test_signals_while_the_session_folder_is_removed_raise_nothing(fake_container, tmp_path,
                                                                    monkeypatch):
    fake_container.update(run_rc=3)
    real = session.shutil.rmtree

    sess = session.new_session("pi", "default", [])

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
    sess = session.new_session("pi", "default", [])
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]))
    session.supervise(spec, api_targets=[("127.0.0.1", 9)], record={}, say=lambda line: None)
    text = (session.cache_dir() / "last-pi-default.log").read_text()
    assert "can open only 2048 files at a time" in text


def _real_layout(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    real = os.path.realpath
    plan = _plan(tmp_path, mounts=[Mount(real(proj), real(proj), note="working folder"),
                                   Mount(real(home), real(home), kind="home")])
    return proj, plan, None


def test_recheck_passes_the_folders_the_plan_checked(tmp_path, monkeypatch):
    proj, plan, _ = _real_layout(tmp_path)
    monkeypatch.setattr(session.runtime, "_complete", lambda folder: True)
    (tmp_path / "rt").mkdir()
    session.recheck_sources(_spec(tmp_path, plan=plan))


def test_recheck_refuses_a_share_swapped_for_a_link(tmp_path, monkeypatch):
    proj, plan, _ = _real_layout(tmp_path)
    monkeypatch.setattr(session.runtime, "_complete", lambda folder: True)
    (tmp_path / "rt").mkdir()
    (tmp_path / "elsewhere").mkdir()
    proj.rmdir()
    proj.symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    with pytest.raises(session.SettingsError, match="changed after launch checked it"):
        session.recheck_sources(_spec(tmp_path, plan=plan))


def test_recheck_refuses_a_changed_runtime_folder(tmp_path, monkeypatch):
    proj, plan, _ = _real_layout(tmp_path)
    monkeypatch.setattr(session.runtime, "_complete", lambda folder: False)
    (tmp_path / "rt").mkdir()
    with pytest.raises(session.SettingsError, match="holds launch's program for the container, changed"):
        session.recheck_sources(_spec(tmp_path, plan=plan))


def test_supervise_checks_the_sources_before_the_run(fake_container, tmp_path, monkeypatch):
    def refuse(spec):
        raise session.SettingsError("swapped")
    monkeypatch.setattr(session, "recheck_sources", refuse)
    sess = session.new_session("pi", "default", [])
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]))
    with pytest.raises(session.SettingsError, match="swapped"):
        session.supervise(spec, api_targets=[("127.0.0.1", 9)], record={},
                          say=lambda line: None)
    assert not fake_container.load().get("runs")


# The server session

class _ServerSession:
    def __init__(self, fail=None):
        self.fail, self.calls = fail, []

    def open(self):
        self.calls.append("open")
        if self.fail:
            raise self.fail
        return "/tmp/gmlx-s/1.sock"

    def renew(self):
        return "/tmp/gmlx-s/2.sock"

    def close(self):
        self.calls.append("close")

    def lines(self):
        return ["[launch] pi can use assistant home, whose tools run on the Mac: web"]


def test_supervise_relays_the_api_to_the_session_socket(fake_container, tmp_path, monkeypatch):
    made = []
    real = session.Relay

    def spy(loop, listen, connect, **k):
        made.append((k["name"], connect, k.get("renew"), k.get("max_connections"),
                     k.get("check_every")))
        return real(loop, listen, connect, **k)
    monkeypatch.setattr(session, "Relay", spy)
    sess = session.new_session("pi", "default", [])
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]))
    server, said = _ServerSession(), []
    session.supervise(spec, api_targets=[("127.0.0.1", 8080)], record={}, say=said.append,
                      summary=["[launch] summary"], server_session=server)
    # The relay holds as many connections as the session socket serves.
    assert made == [("gmlx api", ["/tmp/gmlx-s/1.sock"], server.renew,
                     session.SESSION_CONNECTIONS_MAX, session.TARGET_CHECK_GAP)]
    assert said == ["[launch] summary", *server.lines()]
    assert server.calls == ["open", "close"]


def test_supervise_prints_a_refused_renewal_after_the_client_exits(fake_container, tmp_path):
    """The client owns the terminal while it runs, so the reason waits for
    its exit. The session log got it at once."""
    sess = session.new_session("pi", "default", [])
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]))
    server, said = _ServerSession(), []
    server.refused = "cannot reach the server at http://127.0.0.1:8080/v1 (refused)."
    session.supervise(spec, api_targets=[("127.0.0.1", 8080)], record={}, say=said.append,
                      server_session=server)
    assert said[-2:] == ["[launch] the server stopped answering on the session socket "
                         "and gave no new one, so the client could not reach it after that.",
                         "[launch] cannot reach the server at http://127.0.0.1:8080/v1 "
                         "(refused)."]
    assert callable(server.log)                    # the session log's guest lines


def test_a_refused_server_session_starts_no_container(fake_container, tmp_path):
    sess = session.new_session("pi", "default", [])
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]))
    server = _ServerSession(fail=SettingsError("refused"))
    with pytest.raises(SettingsError, match="refused"):
        session.supervise(spec, api_targets=[("127.0.0.1", 8080)], record={},
                          say=lambda line: None, server_session=server)
    assert fake_container.load().get("runs", []) == []
    assert server.calls == ["open", "close"]
    assert not sess.dir.exists() and session.read_record("pi", "default") is None


# A joined copy whose launch ends

def _copy(fake_container, tmp_path, script: str, *, kill: bool = False,
          notify: bool = False) -> int:
    """Run ``script`` as the ``container exec`` of a joined copy, with the
    paths ``pid`` and ``heard`` as its arguments. On a hangup the fake
    ``container`` sends SIGHUP to the pid the script writes to ``pid`` when
    ``kill`` is set, and writes a line to the FIFO ``heard`` when ``notify``
    is set."""
    pid, heard = tmp_path / "pid", tmp_path / "heard"
    os.mkfifo(heard)
    fake_container.update(hangup_kill=str(pid) if kill else "",
                          hangup_notify=str(heard) if notify else "")
    return session.run_copy(["sh", "-c", script, "sh", str(pid), str(heard)], dict(os.environ),
                            name="gmlx-pi-abc123", copy_id="0f3a")


def test_a_closed_window_hangs_up_the_joined_copy(fake_container, tmp_path):
    """SIGHUP to launch reaches the copy as SIGHUP, through the guest entry,
    and launch returns the copy's exit code once the hangup is done."""
    saved = signal.getsignal(signal.SIGHUP)
    rc = _copy(fake_container, tmp_path, 'echo $$ > "$1"; kill -HUP $PPID; exec sleep 30',
               kill=True)
    assert rc == 128 + signal.SIGHUP
    assert fake_container.calls("exec") == [
        ["exec", "gmlx-pi-abc123", runtime.GUEST_ENTRY, "--hangup", "0f3a"]]
    assert signal.getsignal(signal.SIGHUP) is saved


def test_sigint_never_ends_a_joining_launch(fake_container, tmp_path):
    assert _copy(fake_container, tmp_path, "kill -INT $PPID; exit 4") == 4
    assert not fake_container.calls("exec")


@pytest.mark.parametrize("sig", [signal.SIGHUP, signal.SIGTERM, signal.SIGINT])
def test_a_signal_ignored_on_entry_stays_ignored(fake_container, tmp_path, sig):
    """A signal ignored at the start, as nohup leaves SIGHUP, stays ignored.
    The joining launch does nothing on it, and ``container exec`` inherits
    it."""
    saved = signal.signal(sig, signal.SIG_IGN)
    try:
        name = sig.name.removeprefix("SIG")
        assert _copy(fake_container, tmp_path, f"kill -{name} $$; kill -{name} $PPID; exit 3") == 3
        assert signal.getsignal(sig) == signal.SIG_IGN
    finally:
        signal.signal(sig, saved)
    assert not fake_container.calls("exec")


_COPY_ON_A_TERMINAL = textwrap.dedent("""
    import fcntl, os, sys, termios
    from gmlx.container import session
    def cooked():
        # The kernel adds PENDIN on the way back from raw mode, so only the
        # modes that raw mode changes are compared.
        mode = termios.tcgetattr(0)
        return (mode[1] & termios.OPOST, mode[3] & (termios.ICANON | termios.ECHO))
    fcntl.ioctl(0, termios.TIOCSCTTY, 0)
    before = cooked()
    copy = ["sh", "-c", 'stty raw -echo; kill -TERM $PPID; read x < "$1"; kill -TERM $PPID; '
            'exec sleep 30', "sh", sys.argv[1]]
    code = session.run_copy(copy, dict(os.environ), name="gmlx-pi-1", copy_id="0f3a")
    print("exit", code, "restored" if cooked() == before and all(before) else "raw", flush=True)
""")


def test_a_killed_container_exec_leaves_the_terminal_as_it_was(fake_container, tmp_path):
    """The copy sets raw mode, as ``container exec -t`` does, and a second
    SIGTERM kills it before it can reset the terminal."""
    heard = tmp_path / "heard"
    os.mkfifo(heard)
    fake_container.update(hangup_notify=str(heard))
    master, slave = pty.openpty()
    launch = subprocess.Popen([sys.executable, "-c", _COPY_ON_A_TERMINAL, str(heard)],
                              stdin=slave, stdout=slave, stderr=slave, start_new_session=True)
    os.close(slave)
    out, deadline = b"", time.monotonic() + 20
    try:
        while select.select([master], [], [], max(0, deadline - time.monotonic()))[0]:
            try:
                chunk = os.read(master, 4096)
            except OSError:                # the terminal has no process left
                break
            if not chunk:
                break
            out += chunk
        assert launch.wait(20) == 0
    finally:
        if launch.poll() is None:
            launch.kill()
        os.close(master)
    assert b"exit 137 restored" in out, out


_SUPERVISE_ON_A_TERMINAL = textwrap.dedent("""
    import fcntl, os, pickle, sys, termios
    from gmlx.container import session
    def cooked():
        mode = termios.tcgetattr(0)
        return (mode[1] & termios.OPOST, mode[3] & (termios.ICANON | termios.ECHO))
    fcntl.ioctl(0, termios.TIOCSCTTY, 0)
    before = cooked()
    with open(sys.argv[1], "rb") as f:
        spec = pickle.load(f)
    heard = sys.argv[2]
    # A reader end that stays open, so a write never waits for the run's read.
    spare = os.open(heard, os.O_RDONLY | os.O_NONBLOCK)
    def tell(*args, **kw):
        with open(heard, "w") as f:
            f.write("heard\\n")
    session.cli.stop = session.cli.kill = tell
    session._Signals._listed = lambda self: True
    session.recheck_sources = lambda spec: None
    session.compose_run_argv = lambda spec, binary: [
        "sh", "-c", 'stty raw -echo; kill -TERM $PPID; read x < "$1"; kill -TERM $PPID; '
        'read x < "$1"; kill -TERM $PPID; exec sleep 30', "sh", heard]
    code = session.supervise(spec, api_targets=[("127.0.0.1", 9)], record={},
                             say=lambda line: None)
    print("exit", code, "restored" if cooked() == before and all(before) else "raw", flush=True)
""")


def _on_a_terminal(argv: list[str], timeout: float = 20, *,
                   answers: list[tuple[bytes, bytes]] = (), go: Path | None = None) -> bytes:
    """Run ``argv`` on a new pseudo-terminal of its own, and return what it
    wrote there. Each item of ``answers`` is a line the program writes and
    the bytes the terminal then types, as a terminal answers a query. A line
    to the FIFO ``go`` then lets the program go on."""
    master, slave = pty.openpty()
    # Spare ends, so neither the test nor the program waits to open the FIFO.
    held = [os.open(go, os.O_RDONLY | os.O_NONBLOCK), os.open(go, os.O_WRONLY)] if go else []
    proc = subprocess.Popen(argv, stdin=slave, stdout=slave, stderr=slave,
                            start_new_session=True)
    os.close(slave)
    out, deadline = b"", time.monotonic() + timeout
    pending = list(answers)
    try:
        while select.select([master], [], [], max(0, deadline - time.monotonic()))[0]:
            try:
                chunk = os.read(master, 4096)
            except OSError:                # the terminal has no process left
                break
            if not chunk:
                break
            out += chunk
            if pending and pending[0][0] in out:
                out = out.replace(pending[0][0], b"", 1)
                os.write(master, pending.pop(0)[1])
                os.write(held[1], b"go\n")
        assert proc.wait(timeout) == 0, out
    finally:
        if proc.poll() is None:
            proc.kill()
        os.close(master)
        for fd in held:
            os.close(fd)
    return out


def test_a_third_signal_leaves_the_terminal_as_it_was(fake_container, tmp_path):
    """The run sets raw mode, as ``container run -t`` does, and the third
    SIGTERM kills it before it can reset the terminal."""
    import pickle

    sess = session.new_session("pi", "default", [])
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]), tty=True)
    (tmp_path / "spec").write_bytes(pickle.dumps(spec))
    os.mkfifo(tmp_path / "heard")
    out = _on_a_terminal([sys.executable, "-c", _SUPERVISE_ON_A_TERMINAL,
                          str(tmp_path / "spec"), str(tmp_path / "heard")])
    assert b"exit 137 restored" in out, out


# The terminal takes these, as the shell would, while a typed key does not
# wait: no line editing, no echo.
_WAITING_INPUT = textwrap.dedent("""
    import fcntl, os, select, sys, termios
    from gmlx.container import session
    fcntl.ioctl(0, termios.TIOCSCTTY, 0)
    mode = termios.tcgetattr(0)
    mode[3] &= ~(termios.ICANON | termios.ECHO)
    termios.tcsetattr(0, termios.TCSANOW, mode)
    go = sys.argv[1]
    def waiting():
        return "input waits" if select.select([0], [], [], 0)[0] else "no input waits"
""")

# The client asks the terminal a question as it quits, and the cleanup
# takes long enough for a second answer to arrive.
_SUPERVISE_ASKS = _WAITING_INPUT + textwrap.dedent("""
    import pickle
    with open(sys.argv[2], "rb") as f:
        spec = pickle.load(f)
    def remove(name, **kw):
        print("cleaning up", flush=True)
        with open(go) as f:
            f.readline()
    session._remove_container = remove
    session.recheck_sources = lambda spec: None
    session.compose_run_argv = lambda spec, binary: [
        "sh", "-c", 'echo asking; read x < "$1"', "sh", go]
    session.supervise(spec, api_targets=[("127.0.0.1", 9)], record={}, say=lambda line: None)
    print(waiting(), flush=True)
""")

_COPY_ASKS = _WAITING_INPUT + textwrap.dedent("""
    session.run_copy(["sh", "-c", 'echo asking; read x < "$1"', "sh", go], dict(os.environ),
                     name="gmlx-pi-1", copy_id="0f3a")
    print(waiting(), flush=True)
""")

_DA1_ANSWER = b"\x1b[?62;22c"


def test_a_session_on_a_terminal_ends_with_no_input_waiting(fake_container, tmp_path):
    import pickle

    sess = session.new_session("pi", "default", [])
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]), tty=True)
    (tmp_path / "spec").write_bytes(pickle.dumps(spec))
    os.mkfifo(tmp_path / "go")
    out = _on_a_terminal([sys.executable, "-c", _SUPERVISE_ASKS, str(tmp_path / "go"),
                          str(tmp_path / "spec")], go=tmp_path / "go",
                         answers=[(b"asking", _DA1_ANSWER), (b"cleaning up", b"\x1b[24;1R")])
    assert b"no input waits" in out, out


def test_a_joined_copy_on_a_terminal_ends_with_no_input_waiting(fake_container, tmp_path):
    os.mkfifo(tmp_path / "go")
    out = _on_a_terminal([sys.executable, "-c", _COPY_ASKS, str(tmp_path / "go")],
                         go=tmp_path / "go", answers=[(b"asking", _DA1_ANSWER)])
    assert b"no input waits" in out, out


def test_a_session_without_a_terminal_leaves_typed_input(fake_container, tmp_path,
                                                         monkeypatch):
    flushed = []
    monkeypatch.setattr(session.termios, "tcflush", lambda fd, queue: flushed.append(fd))
    sess = session.new_session("pi", "default", [])
    spec = _spec(tmp_path, session=sess, plan=_plan(tmp_path, forward=[]), tty=False)
    session.supervise(spec, api_targets=[("127.0.0.1", 9)], record={}, say=lambda line: None)
    assert flushed == []


def test_restoring_the_terminal_drops_the_input_that_waits(monkeypatch):
    seen = []
    monkeypatch.setattr(session.os, "tcgetpgrp", lambda fd: os.getpgrp())
    monkeypatch.setattr(session.termios, "tcsetattr", lambda fd, when, mode: seen.append(when))
    session._restore_terminal([])
    assert seen == [session.termios.TCSAFLUSH]


def test_a_second_sigterm_kills_the_container_exec(fake_container, tmp_path):
    """The copy stays after the first SIGTERM's hangup, as one that ignores
    SIGHUP does, so the second one kills ``container exec``."""
    rc = _copy(fake_container, tmp_path,
               'kill -TERM $PPID; read x < "$2"; kill -TERM $PPID; exec sleep 30', notify=True)
    assert rc == 128 + signal.SIGKILL
    assert len(fake_container.calls("exec")) == 1


class _Watch:
    """The folder of the fake ``container``'s watch mode, with the test's
    ends of its FIFOs open. The test holds a spare end of each, so the
    fake's opens never wait and a read waits for a line."""

    def __init__(self, folder: Path, hold: str):
        folder.mkdir()
        self.folder = folder
        os.mkfifo(folder / "events")
        os.mkfifo(folder / "release")
        (folder / hold).touch()
        self.events = os.open(folder / "events", os.O_RDONLY | os.O_NONBLOCK)
        self.fds = [self.events, os.open(folder / "events", os.O_WRONLY),
                    os.open(folder / "release", os.O_RDONLY | os.O_NONBLOCK)]
        self.release_fd = os.open(folder / "release", os.O_WRONLY)
        self.fds.append(self.release_fd)
        self.seen: list[str] = []
        self.rest = b""

    def wait_for(self, *starts: str, timeout: float = 20) -> None:
        """Read the events until a line starts with each of ``starts``. A
        call that a signal ended fails the test at once."""
        deadline = time.monotonic() + timeout
        while not all(any(line.startswith(s) for line in self.seen) for s in starts):
            assert not [line for line in self.seen if line.startswith("signalled ")], self.seen
            left = deadline - time.monotonic()
            assert left > 0, f"the fake container did not report {starts}: {self.seen}"
            if select.select([self.events], [], [], left)[0]:
                *done, self.rest = (self.rest + os.read(self.events, 4096)).split(b"\n")
                self.seen += [line.decode() for line in done]

    def release(self) -> None:
        os.write(self.release_fd, b"go\n")

    def close(self) -> None:
        self.release()
        for fd in self.fds:
            os.close(fd)


_LAUNCH_IN_A_WINDOW = textwrap.dedent("""
    import os, signal, sys
    from gmlx.container import session
    if sys.argv[1] == "copy":
        copy = [sys.executable, "-c",
                "import signal; print('ready', flush=True); signal.pause()"]
        code = session.run_copy(copy, dict(os.environ), name="gmlx-pi-1", copy_id="0f3a")
        print("exit", code, flush=True)
    else:
        signals = session._Signals("gmlx-pi-1", tty=False,
                                   log=lambda line: print(line, flush=True))
        signals.install()
        print("ready", flush=True)
        sys.stdin.read()
""")


@pytest.mark.parametrize("mode, hold", [("copy", "hangup"), ("signals", "ls"),
                                        ("signals", "stop")])
def test_the_second_sighup_of_a_closed_window_misses_the_container_calls(
        fake_container, tmp_path, monkeypatch, mode, hold):
    """A closed window sends SIGHUP to launch's process group, and the shell
    sends its jobs a second one as it exits. The ``container`` call that
    launch started for the first one runs in a group of its own, so the
    second one does not end it, nor a joining launch that waits for it. The
    fake holds the call ``hold`` until the second SIGHUP is sent."""
    fake_container.update(containers=[{"name": "gmlx-pi-1"}])
    watch = _Watch(tmp_path / "watch", hold)
    monkeypatch.setenv("FAKE_CONTAINER_WATCH", str(watch.folder))
    launch = subprocess.Popen([sys.executable, "-c", _LAUNCH_IN_A_WINDOW, mode],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                              start_new_session=True)
    try:
        assert launch.stdout.readline() == "ready\n"
        os.killpg(launch.pid, signal.SIGHUP)
        watch.wait_for(f"start {hold} ")
        os.killpg(launch.pid, signal.SIGHUP)
        watch.release()
        watch.wait_for(*(["end hangup 0"] if mode == "copy" else ["end stop 0", "end kill 0"]))
        launch.stdin.close()
        assert launch.wait(20) == 0
        assert launch.stdout.read() == ("exit 129\n" if mode == "copy" else "")
        assert [line for line in watch.seen if line.startswith("start ")
                and not line.endswith(" 1")] == []
    finally:
        if launch.poll() is None:
            launch.kill()
        watch.close()
