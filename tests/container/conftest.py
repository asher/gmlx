"""Fixtures for the container-mode tests. ``test_container_entry.py`` needs
none of them, so the Linux CI job can run it with ``--noconftest``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_FAKE = Path(__file__).with_name("fake_container.py")


class FakeContainer:
    """The state file of the fake ``container`` command."""

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> dict:
        return json.loads(self.path.read_text())

    def save(self, state: dict) -> None:
        self.path.write_text(json.dumps(state))

    def update(self, **values) -> None:
        state = self.load()
        state.update(values)
        self.save(state)

    @property
    def log(self) -> list[list[str]]:
        return self.load().get("log", [])

    def calls(self, *prefix: str) -> list[list[str]]:
        return [a for a in self.log if a[:len(prefix)] == list(prefix)]


@pytest.fixture
def fake_container(tmp_path, monkeypatch) -> FakeContainer:
    """A fake ``container`` first on ``PATH``, with the launch data and cache
    folders under ``tmp_path``."""
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    script = bin_dir / "container"
    script.write_text(f"#!/bin/sh\nexec {sys.executable} {_FAKE} \"$@\"\n")
    script.chmod(0o755)
    state = tmp_path / "container-state.json"
    state.write_text("{}")
    monkeypatch.setenv("FAKE_CONTAINER_STATE", str(state))
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return FakeContainer(state)
