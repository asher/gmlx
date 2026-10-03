"""The suite's guard against writes to the user's Hugging Face cache
(hf_cache_guard.py), checked on a guarded folder under tmp_path."""
from __future__ import annotations

import os
import shutil

import pytest

import hf_cache_guard


@pytest.fixture
def guarded(monkeypatch, tmp_path):
    """A folder the guard treats as the user's cache, with one file in it.
    The writes refused in it are taken, so they fail no test."""
    root = tmp_path / "hf"
    (root / "hub").mkdir(parents=True)
    (root / "hub" / "kept").write_text("x")
    monkeypatch.setattr(hf_cache_guard, "ROOTS", (str(root) + os.sep,))
    start = len(hf_cache_guard.WRITES)
    yield root
    hf_cache_guard.take(start)


def test_the_guard_is_on_for_the_users_cache():
    home = os.path.expanduser(os.environ.get("HF_HOME")
                              or "~/.cache/huggingface")
    assert os.path.realpath(home) + os.sep in hf_cache_guard.ROOTS


@pytest.mark.parametrize("write", [
    lambda r: open(r / "hub" / "new", "w"),
    lambda r: open(r / "hub" / "kept", "a"),
    lambda r: os.open(r / "hub" / "new", os.O_WRONLY | os.O_CREAT),
    lambda r: os.makedirs(r / "hub" / "models--a--b" / "refs"),
    lambda r: os.symlink("kept", r / "hub" / "link"),
    lambda r: os.replace(r / "hub" / "kept", r / "hub" / "moved"),
    lambda r: os.remove(r / "hub" / "kept"),
    lambda r: shutil.rmtree(r / "hub"),
    lambda r: os.utime(r / "hub" / "kept"),
], ids=["open-w", "open-a", "os-open", "makedirs", "symlink", "replace",
        "remove", "rmtree", "utime"])
def test_a_write_in_the_cache_is_refused_and_recorded(guarded, write):
    start = len(hf_cache_guard.WRITES)
    with pytest.raises(PermissionError, match="make a cache under tmp_path"):
        write(guarded)
    writes = hf_cache_guard.take(start)
    assert writes and all(p.startswith(str(guarded)) for _, p in writes)
    assert "were refused" in hf_cache_guard.failure(writes)
    assert (guarded / "hub" / "kept").read_text() == "x"


def test_a_write_from_outside_into_the_cache_is_refused(guarded, tmp_path):
    (tmp_path / "out").write_text("y")
    with pytest.raises(PermissionError):
        shutil.copyfile(tmp_path / "out", guarded / "hub" / "copy")
    with pytest.raises(PermissionError):
        os.rename(tmp_path / "out", guarded / "hub" / "moved")
    assert sorted(os.listdir(guarded / "hub")) == ["kept"]


def test_reads_and_writes_beside_the_cache_pass(guarded, tmp_path):
    start = len(hf_cache_guard.WRITES)
    assert (guarded / "hub" / "kept").read_text() == "x"
    assert os.listdir(guarded / "hub") == ["kept"]
    beside = tmp_path / "hf-other"
    beside.mkdir()
    (beside / "f").write_text("z")
    assert hf_cache_guard.WRITES[start:] == []
    assert hf_cache_guard.failure([]) is None


@pytest.mark.parametrize("env,expect", [
    ({}, ["~/.cache/huggingface", "~/.cache/huggingface/hub"]),
    ({"HF_HOME": "/h"}, ["/h", "/h/hub"]),
    ({"HF_HOME": "/h", "HF_HUB_CACHE": "/c"}, ["/h", "/c"]),
    ({"XDG_CACHE_HOME": "/x"}, ["/x/huggingface", "/x/huggingface/hub"]),
    ({"HUGGINGFACE_HUB_CACHE": "/old"}, ["~/.cache/huggingface", "/old"]),
])
def test_the_guarded_folders_follow_the_environment(env, expect):
    want = set()
    for e in expect:
        e = os.path.expanduser(e)
        want |= {os.path.abspath(e) + os.sep, os.path.realpath(e) + os.sep}
    assert set(hf_cache_guard.roots(env)) == want
