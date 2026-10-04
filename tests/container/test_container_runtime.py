"""gmlx/container/runtime.py: the runtime folder a launch reuses must hold
the entry its name promises."""

from __future__ import annotations

import os
import stat

from gmlx.container import runtime


def _entry(tmp_path, data=b"\x7fELF-fake"):
    path = tmp_path / "gmlx-entry"
    path.write_bytes(data)
    return path


def test_a_replaced_entry_is_copied_again(fake_container, tmp_path):
    """A session that shared the launch data folder writes another program
    into the runtime folder. The next launch must not run it."""
    source = _entry(tmp_path)
    folder, lock = runtime.acquire_runtime(source)
    lock.release()
    (folder / "gmlx-entry").write_bytes(b"#!/bin/sh\necho planted\n")
    again, lock = runtime.acquire_runtime(source)
    assert again == folder
    assert (folder / "gmlx-entry").read_bytes() == source.read_bytes()
    lock.release()


def test_an_entry_replaced_by_a_link_is_copied_again(fake_container, tmp_path):
    source = _entry(tmp_path)
    folder, lock = runtime.acquire_runtime(source)
    lock.release()
    planted = tmp_path / "planted"
    planted.write_bytes(source.read_bytes())
    (folder / "gmlx-entry").unlink()
    (folder / "gmlx-entry").symlink_to(planted)
    again, lock = runtime.acquire_runtime(source)
    assert not os.path.islink(again / "gmlx-entry")
    lock.release()


def test_the_runtime_folders_are_private(fake_container, tmp_path):
    old = os.umask(0o002)
    try:
        folder, lock = runtime.acquire_runtime(_entry(tmp_path))
    finally:
        os.umask(old)
    lock.release()
    for path in (folder.parent, folder):
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o700, path
