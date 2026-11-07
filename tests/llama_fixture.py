"""Read the llama.cpp fixtures that scripts/llama_fixtures.py writes.

A fixture names each file it read by name, size and sha256. A test finds
the file by name and size, and checks the sha256 of a file of at most
``HASH_LIMIT`` bytes on each run. A larger file is not hashed: the hash
would add time to every run, and the test's own checks catch a wrong file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"
HASH_LIMIT = 512 * 2**20


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def find_file(fixture: dict, name: str, candidates) -> str:
    """The path among ``candidates`` with the name and size the fixture
    recorded for ``name``. Skips when none matches, and fails when a file
    within the hash limit has another sha256."""
    want = fixture["files"][name]
    for p in candidates:
        if Path(p).name == want["name"] and os.path.getsize(p) == want["size"]:
            break
    else:
        pytest.skip(f"{name} is not under KQUANT_TEST_GGUF_DIR")
    if want["size"] <= HASH_LIMIT:
        import hashlib

        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(1 << 24):
                h.update(chunk)
        if h.hexdigest() != want["sha256"]:
            pytest.fail(f"{p} is not the file the fixture recorded")
    return str(p)


def check_checked_in(fixture: dict, path: Path) -> None:
    """Fail when a file checked in beside the fixture is not the one the
    fixture recorded. Unlike ``find_file``, never skips."""
    import hashlib

    want = fixture["files"][path.name]
    data = path.read_bytes()
    if (len(data), hashlib.sha256(data).hexdigest()) != (want["size"], want["sha256"]):
        pytest.fail(f"{path} is not the file the fixture recorded; regenerate both")
