"""gmlx/container/web_ports.py: the Mac port of each project's browser app,
and the record that keeps it."""

from __future__ import annotations

import json
import os
import socket
import threading

import pytest

from gmlx.container import settings, web_ports


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
    assert web_ports.choose("dsh", "a-1") == (3100, None)
    assert web_ports.choose("dsh", "b-2") == (3101, None)
    assert web_ports.choose("open-webui", "default") == (3102, None)
    assert web_ports.choose("dsh", "a-1") == (3100, None)
    assert web_ports.recorded("dsh", "b-2") == 3101
    assert os.stat(_record_path()).st_mode & 0o777 == 0o600


def test_a_busy_or_avoided_port_moves_the_project(free):
    assert web_ports.choose("dsh", "a-1") == (3100, None)
    free.add(3100)
    assert web_ports.choose("dsh", "a-1") == (3101, 3100)
    free.clear()
    assert web_ports.choose("dsh", "a-1", avoid={3101}) == (3100, 3101)
    assert web_ports.recorded("dsh", "a-1") == 3100


def test_without_record_nothing_is_written(free):
    assert web_ports.choose("dsh", "a-1", record=False) == (3100, None)
    assert not _record_path().exists()
    web_ports.choose("dsh", "a-1")
    before = _record_path().read_bytes()
    free.add(3100)
    assert web_ports.choose("dsh", "a-1", record=False) == (3101, 3100)
    assert _record_path().read_bytes() == before


def test_an_entry_stays_while_its_home_exists_or_its_launch_runs(free):
    web_ports.choose("dsh", "a-1")                      # this process runs
    settings.private_home("dsh", "b-2")
    web_ports.choose("dsh", "b-2")
    doc = json.loads(_record_path().read_text())
    doc["dsh"]["a-1"]["pid"] = 999999                   # that launch is gone, no home
    doc["dsh"]["b-2"]["pid"] = 999999                   # gone too, but it has a home
    _record_path().write_text(json.dumps(doc))
    assert web_ports.choose("dsh", "c-3") == (3100, None)
    assert web_ports.recorded("dsh", "a-1") is None
    assert web_ports.recorded("dsh", "b-2") == 3101


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


def test_release_returns_the_port_and_frees_it(free):
    assert web_ports.release("dsh", "a-1") is None
    assert not _record_path().exists()
    settings.private_home("dsh", "a-1")
    web_ports.choose("dsh", "a-1")
    assert web_ports.release("dsh", "a-1") == 3100
    assert web_ports.recorded("dsh", "a-1") is None
    assert web_ports.choose("dsh", "b-2") == (3100, None)


@pytest.mark.parametrize("text", ["not json", "[]", '{"dsh": []}',
                                  '{"dsh": {"a-1": {"port": 3000}}}',
                                  '{"dsh": {"a-1": {"port": true}}}'])
def test_a_damaged_record_or_entry_keeps_no_port(free, text):
    _record_path().parent.mkdir(parents=True, exist_ok=True)
    _record_path().write_text(text)
    assert web_ports.recorded("dsh", "a-1") is None
    assert web_ports.choose("dsh", "b-2") == (3100, None)


def test_a_linked_record_is_neither_read_nor_followed(free, tmp_path):
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text(json.dumps({"dsh": {"a-1": {"port": 3100, "pid": os.getpid()}}}))
    _record_path().parent.mkdir(parents=True, exist_ok=True)
    _record_path().symlink_to(elsewhere)
    assert web_ports.recorded("dsh", "a-1") is None
    web_ports.choose("dsh", "b-2")
    assert not _record_path().is_symlink()
    assert json.loads(elsewhere.read_text())["dsh"] == {"a-1": {"port": 3100,
                                                               "pid": os.getpid()}}


def test_launches_that_start_at_once_take_ports_of_their_own(free):
    got: list[int] = []
    lock = threading.Lock()

    def take(n: int) -> None:
        port, _ = web_ports.choose("dsh", f"p-{n}")
        with lock:
            got.append(port)
    threads = [threading.Thread(target=take, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(got) == list(range(3100, 3108))


def test_a_port_with_a_listener_is_not_free():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        assert web_ports._free(port) is False
    assert web_ports._free(port) is True
