"""gmlx/container/web_ports.py: the Mac port of each project's browser app,
and the record that keeps it."""

from __future__ import annotations

import json
import os
import shutil
import socket
import threading

import pytest

from gmlx.container import session, settings, web_ports


@pytest.fixture
def free(monkeypatch):
    """No test binds a port of the range, which a program on this Mac may
    use. The set holds the ports that other programs use."""
    busy: set[int] = set()
    monkeypatch.setattr(web_ports, "_free", lambda port: port not in busy)
    return busy


def _record_path():
    return settings.data_path() / "web-ports.json"


def test_a_project_keeps_its_port_and_another_project_takes_the_next(free):
    assert web_ports.choose("dsh", "a-1") == (3100, None, False)
    assert web_ports.choose("dsh", "b-2") == (3101, None, False)
    assert web_ports.choose("open-webui", "default") == (3102, None, False)
    assert web_ports.choose("dsh", "a-1") == (3100, 3100, False)
    assert web_ports.recorded("dsh", "b-2") == 3101
    assert os.stat(_record_path()).st_mode & 0o777 == 0o600


def test_a_busy_or_avoided_port_moves_the_project(free):
    assert web_ports.choose("dsh", "a-1") == (3100, None, False)
    free.add(3100)
    choice = web_ports.choose("dsh", "a-1")
    assert choice == (3101, 3100, False) and choice.moved
    free.clear()
    assert web_ports.choose("dsh", "a-1", avoid={3101}) == (3100, 3101, False)
    assert web_ports.recorded("dsh", "a-1") == 3100


def test_without_record_nothing_is_written(free):
    assert web_ports.choose("dsh", "a-1", record=False) == (3100, None, False)
    assert not _record_path().exists()
    web_ports.choose("dsh", "a-1")
    before = _record_path().read_bytes()
    free.add(3100)
    assert web_ports.choose("dsh", "a-1", record=False) == (3101, 3100, False)
    assert _record_path().read_bytes() == before


def test_an_entry_stays_while_its_home_exists_or_its_launch_runs(free):
    web_ports.choose("dsh", "a-1")                      # this process runs
    settings.private_home("dsh", "b-2")
    web_ports.choose("dsh", "b-2")
    doc = json.loads(_record_path().read_text())
    doc["projects"]["dsh"]["a-1"]["pid"] = 999999       # that launch is gone, no home
    doc["projects"]["dsh"]["b-2"]["pid"] = 999999       # gone too, but it has a home
    _record_path().write_text(json.dumps(doc))
    assert web_ports.choose("dsh", "c-3") == (3100, None, False)
    assert web_ports.recorded("dsh", "a-1") is None
    assert web_ports.recorded("dsh", "b-2") == 3101


@pytest.mark.parametrize("owner", [
    {"pid": 1},                                          # a process of another user
    {"pid": os.getppid(), "pid_start": 1},              # a later process with the ID
], ids=["other-user", "reused-id"])
def test_a_process_that_is_not_the_launch_keeps_no_port(free, owner):
    """A first launch that was refused before it made the home leaves its
    entry. When the system gives its process ID to another process, that
    process is not the launch, so the port is free again, also for
    --remove-home of that project."""
    web_ports.choose("dsh", "a-1")
    doc = json.loads(_record_path().read_text())
    doc["projects"]["dsh"]["a-1"] = {"port": 3100, **owner}
    _record_path().write_text(json.dumps(doc))
    assert web_ports.release("dsh", "a-1", unless_running=True) == []
    assert web_ports.recorded("dsh", "a-1") is None
    _record_path().write_text(json.dumps(doc))
    assert web_ports.choose("dsh", "b-2") == (3100, None, False)
    assert web_ports.recorded("dsh", "a-1") is None


def test_the_running_launch_keeps_its_port_by_id_and_start_time(free):
    web_ports.choose("dsh", "a-1")
    web_ports.choose("dsh", "b-2")                       # the record is read and written again
    doc = json.loads(_record_path().read_text())
    assert doc["projects"]["dsh"]["a-1"] == {"port": 3100, **session.launch_owner()}
    assert web_ports.release("dsh", "a-1", unless_running=True) == []
    assert web_ports.recorded("dsh", "a-1") == 3100


def _forget_launch(client: str, project: str) -> None:
    """The launch that took the port of a project has exited."""
    doc = json.loads(_record_path().read_text())
    doc["projects"][client][project]["pid"] = 999999
    _record_path().write_text(json.dumps(doc))


def test_a_port_another_project_served_goes_last_and_says_so(free):
    """A port keeps the service worker and the stored data of the pages of
    the project that used it. So it goes to another project only when no
    other port is free, and the choice says so."""
    home = settings.private_home("dsh", "a-1")
    web_ports.choose("dsh", "a-1")
    web_ports.mark_served("dsh", "a-1", 3100)
    free.add(3100)
    assert web_ports.choose("dsh", "a-1").port == 3101   # another program held 3100
    web_ports.mark_served("dsh", "a-1", 3101)
    free.clear()
    assert web_ports.choose("dsh", "b-2") == (3102, None, False)
    _forget_launch("dsh", "a-1")
    shutil.rmtree(home.parent)                            # the folder removed by hand
    assert web_ports.choose("dsh", "c-3") == (3103, None, False)
    free.update(range(3104, 3200))
    assert web_ports.choose("dsh", "d-4") == (3100, None, True)
    assert web_ports.choose("dsh", "d-4") == (3100, 3100, True)
    web_ports.mark_served("dsh", "d-4", 3100)
    assert web_ports.choose("dsh", "d-4") == (3100, 3100, False)
    assert web_ports.choose("dsh", "e-5", record=False) == (3101, None, True)


def test_a_project_gets_back_a_port_it_served(free):
    for project in ("a-1", "b-2"):
        web_ports.choose("dsh", project)
        web_ports.mark_served("dsh", project, web_ports.recorded("dsh", project))
        _forget_launch("dsh", project)
    web_ports.choose("dsh", "c-3")                       # the entries without a home go
    assert web_ports.recorded("dsh", "b-2") is None
    assert web_ports.choose("dsh", "b-2") == (3101, None, False)


def test_a_started_project_brings_back_the_served_port_of_its_entry(free):
    """A launch makes the private home before the session starts, so only
    the mark of a started session counts the port as served."""
    settings.private_home("dsh", "a-1")
    web_ports.choose("dsh", "a-1")
    web_ports.choose("dsh", "b-2")
    web_ports.choose("dsh", "a-1")
    doc = json.loads(_record_path().read_text())
    assert doc["served"] == {}                           # no session started yet
    session.mark_started("dsh", "a-1")
    doc["served"] = "damaged"
    _record_path().write_text(json.dumps(doc))
    assert web_ports.choose("dsh", "b-2") == (3101, 3101, False)
    assert json.loads(_record_path().read_text())["served"] == {"3100": ["dsh", "a-1"]}
    assert web_ports.choose("dsh", "b-2", avoid={3101}) == (3102, 3101, False)


def test_a_started_project_never_takes_over_a_port_another_project_served(free):
    """A project that started a session on one port and then took a port
    that another project served, in a launch that stopped before its start,
    still meets the other project's data there."""
    web_ports.choose("dsh", "c-3")
    web_ports.mark_served("dsh", "c-3", 3100)
    web_ports.release("dsh", "c-3")
    settings.private_home("dsh", "a-1")
    session.mark_started("dsh", "a-1")
    free.update(range(3101, 3200))
    assert web_ports.choose("dsh", "a-1") == (3100, None, True)
    assert web_ports.choose("dsh", "a-1") == (3100, 3100, True)


def test_release_unless_running_keeps_the_entry_of_a_running_launch(free):
    web_ports.choose("dsh", "a-1")                       # this process runs
    web_ports.mark_served("dsh", "a-1", 3100)
    assert web_ports.release("dsh", "a-1", unless_running=True) == [3100]
    assert web_ports.recorded("dsh", "a-1") == 3100
    _forget_launch("dsh", "a-1")
    assert web_ports.release("dsh", "a-1", unless_running=True) == [3100]
    assert web_ports.recorded("dsh", "a-1") is None


def test_mark_served_ignores_a_port_outside_the_range(free):
    web_ports.mark_served("dsh", "a-1", 3000)
    assert not _record_path().exists()


def test_a_full_range_raises_busy_and_names_remove_home(free):
    free.update(range(3101, 3200))
    settings.private_home("dsh", "a-1")
    web_ports.choose("dsh", "a-1")
    with pytest.raises(settings.Busy, match="run gmlx launch dsh --remove-home"):
        web_ports.choose("dsh", "b-2")
    free.add(3100)
    web_ports.release("dsh", "a-1")
    with pytest.raises(settings.Busy, match="Stop a program that uses one of these ports"):
        web_ports.choose("dsh", "b-2")


def test_release_returns_the_served_ports_and_keeps_them_last(free):
    assert web_ports.release("dsh", "a-1") == []
    assert not _record_path().exists()
    settings.private_home("dsh", "a-1")
    web_ports.choose("dsh", "a-1")
    web_ports.mark_served("dsh", "a-1", 3100)
    free.add(3100)
    web_ports.choose("dsh", "a-1")                      # moves to 3101
    web_ports.mark_served("dsh", "a-1", 3101)
    free.clear()
    assert web_ports.release("dsh", "a-1") == [3100, 3101]
    assert web_ports.recorded("dsh", "a-1") is None
    assert web_ports.choose("dsh", "b-2") == (3102, None, False)


def test_release_counts_the_entry_port_only_for_a_project_that_started(free):
    """A launch that stopped before its session started served no pages on
    its port. A caller that removed the start mark says that the project
    started."""
    settings.private_home("dsh", "a-1")
    web_ports.choose("dsh", "a-1")
    assert web_ports.release("dsh", "a-1") == []
    assert web_ports.choose("dsh", "b-2") == (3100, None, False)
    settings.private_home("dsh", "c-3")
    assert web_ports.choose("dsh", "c-3").port == 3101
    assert web_ports.release("dsh", "c-3", started=True) == [3101]
    assert web_ports.choose("dsh", "d-4") == (3102, None, False)


@pytest.mark.parametrize("text", [
    "not json", "[]", '{"projects": []}', '{"projects": {"dsh": []}}',
    '{"projects": {"dsh": {"a-1": {"port": 3000}}}}',
    '{"projects": {"dsh": {"a-1": {"port": true}}}}',
    '{"served": []}', '{"served": {"3000": ["dsh", "a-1"]}}',
    '{"served": {"x": ["dsh", "a-1"]}}', '{"served": {"3100": "dsh"}}',
    '{"served": {"3100": ["dsh", 1]}}'])
def test_a_damaged_record_or_entry_keeps_no_port(free, text):
    _record_path().parent.mkdir(parents=True, exist_ok=True)
    _record_path().write_text(text)
    assert web_ports.recorded("dsh", "a-1") is None
    assert web_ports.choose("dsh", "b-2") == (3100, None, False)


def test_a_linked_record_is_neither_read_nor_followed(free, tmp_path):
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text(json.dumps({"projects": {"dsh": {"a-1": {"port": 3100,
                                                                  "pid": os.getpid()}}}}))
    _record_path().parent.mkdir(parents=True, exist_ok=True)
    _record_path().symlink_to(elsewhere)
    assert web_ports.recorded("dsh", "a-1") is None
    web_ports.choose("dsh", "b-2")
    assert not _record_path().is_symlink()
    assert json.loads(elsewhere.read_text())["projects"]["dsh"] == {
        "a-1": {"port": 3100, "pid": os.getpid()}}


def test_launches_that_start_at_once_take_ports_of_their_own(free):
    got: list[int] = []
    lock = threading.Lock()

    def take(n: int) -> None:
        port = web_ports.choose("dsh", f"p-{n}").port
        with lock:
            got.append(port)
    threads = [threading.Thread(target=take, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(got) == list(range(3100, 3108))


@pytest.mark.parametrize("family, host", [(socket.AF_INET, "127.0.0.1"),
                                          (socket.AF_INET6, "::1")])
def test_a_port_with_a_listener_is_not_free(family, host):
    """The session listens on ::1, as this check does, and a program that
    listens on 127.0.0.1 also holds the port."""
    with socket.socket(family) as listener:
        listener.bind((host, 0))
        listener.listen()
        port = listener.getsockname()[1]
        assert web_ports._free(port) is False
    assert web_ports._free(port) is True
