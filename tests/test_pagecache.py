"""Exit-time page-cache release (gmlx.pagecache): the MS_INVALIDATE sweep
and its larger-than-RAM registration gate."""

import os
import sys

import pytest

from gmlx import pagecache

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="Darwin UBC semantics"
)

_MB = 1 << 20


@pytest.fixture
def cached_file(tmp_path):
    """A 16MB file whose pages are clean and resident in the UBC."""
    p = tmp_path / "blob.bin"
    fd = os.open(p, os.O_WRONLY | os.O_CREAT, 0o644)
    os.write(fd, os.urandom(16 * _MB))
    os.fsync(fd)
    os.close(fd)
    with open(p, "rb") as f:
        while f.read(1 << 22):
            pass
    return str(p)


def test_release_evicts_cached_pages(cached_file):
    before = pagecache.resident_file_bytes(cached_file)
    assert before > 8 * _MB  # freshly read: mostly resident
    released = pagecache.release_file_cache([cached_file])
    assert released == before
    assert pagecache.resident_file_bytes(cached_file) < _MB


def test_release_missing_file_is_quiet(tmp_path):
    assert pagecache.release_file_cache([str(tmp_path / "absent")]) == 0


def test_register_gates_on_ram(cached_file, monkeypatch):
    monkeypatch.setattr(pagecache, "_groups", [])
    monkeypatch.setattr(pagecache, "_hook_installed", True)  # no real atexit
    # File smaller than RAM: remnant is the next load's warm set - keep it.
    pagecache.register_streaming_release([cached_file])
    assert pagecache._groups == []
    # "RAM" smaller than the file: streaming model, sweep at exit.
    monkeypatch.setattr(pagecache, "_physical_ram_bytes", lambda: 4 * _MB)
    pagecache.register_streaming_release([cached_file])
    assert pagecache._groups == [[cached_file]]
    pagecache.register_streaming_release([cached_file])  # idempotent
    assert pagecache._groups == [[cached_file]]


def test_register_env_opt_out(cached_file, monkeypatch):
    monkeypatch.setattr(pagecache, "_groups", [])
    monkeypatch.setattr(pagecache, "_hook_installed", True)
    monkeypatch.setattr(pagecache, "_physical_ram_bytes", lambda: 4 * _MB)
    monkeypatch.setenv("GMLX_RELEASE_PAGECACHE", "0")
    pagecache.register_streaming_release([cached_file])
    assert pagecache._groups == []


def test_release_streaming_for_sweeps_evicted_group(cached_file, monkeypatch):
    """Server eviction sweeps the model's own shard group immediately and
    exactly once; unregistered paths are a quiet no-op."""
    monkeypatch.setattr(pagecache, "_groups", [])
    monkeypatch.setattr(pagecache, "_hook_installed", True)
    monkeypatch.setattr(pagecache, "_physical_ram_bytes", lambda: 4 * _MB)
    pagecache.register_streaming_release([cached_file])
    released = pagecache.release_streaming_for(cached_file)
    assert released > 8 * _MB
    assert pagecache._groups == []  # group consumed; exit sweep skips it
    assert pagecache.release_streaming_for(cached_file) == 0


def test_sweep_log_gated_on_session_verbosity(cached_file, monkeypatch, capsys):
    """The sweep itself always runs; its chatter follows the load session's
    verbosity, captured at registration time."""
    from gmlx import loadlog

    monkeypatch.setattr(pagecache, "_groups", [])
    monkeypatch.setattr(pagecache, "_hook_installed", True)
    monkeypatch.setattr(pagecache, "_physical_ram_bytes", lambda: 4 * _MB)
    monkeypatch.setattr(pagecache, "_log_release", False)
    monkeypatch.setattr(loadlog, "is_verbose", lambda: False)
    pagecache.register_streaming_release([cached_file])
    pagecache._exit_sweep()
    assert "[pagecache]" not in capsys.readouterr().out
    with open(cached_file, "rb") as f:  # re-prime the cache for round two
        while f.read(1 << 22):
            pass
    monkeypatch.setattr(loadlog, "is_verbose", lambda: True)
    pagecache.register_streaming_release([cached_file])
    pagecache._exit_sweep()
    assert "[pagecache] released" in capsys.readouterr().out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
