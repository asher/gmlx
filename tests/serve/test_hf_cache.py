#!/usr/bin/env python3
"""Offline-resolve window (``gmlx.hf_cache.offline_resolve``): a local path or
empty ref is a no-op; a repo id is ensured in the HF cache (cache-only probe, with
a one-time download only on a genuine miss) and then loaded with ``HF_HUB_OFFLINE``
forced on for the duration and restored after. Pure CPU - ``huggingface_hub`` is
stubbed, so no network and no real cache is touched."""
from __future__ import annotations

import sys
import types

import pytest

from gmlx import hf_cache


def _stub_hf(monkeypatch, *, cached: bool, offline_start: bool = False):
    """Install a fake huggingface_hub whose snapshot_download honours
    local_files_only: succeeds when ``cached``, else raises
    LocalEntryNotFoundError on the probe (a plain call 'downloads'). Returns
    ``(calls, constants, bars)`` for assertions; ``constants.HF_HUB_OFFLINE``
    starts at ``offline_start``."""
    calls = []
    bars = []

    class LocalEntryNotFoundError(Exception):
        pass

    def snapshot_download(repo, local_files_only=False, **kw):
        calls.append({"repo": repo, "local_files_only": local_files_only,
                      "offline": constants.HF_HUB_OFFLINE})
        if local_files_only and not cached:
            raise LocalEntryNotFoundError(repo)
        return f"/cache/{repo}"

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot_download
    constants = types.ModuleType("huggingface_hub.constants")
    constants.HF_HUB_OFFLINE = offline_start
    hub.constants = constants
    errors = types.ModuleType("huggingface_hub.errors")
    errors.LocalEntryNotFoundError = LocalEntryNotFoundError
    utils = types.ModuleType("huggingface_hub.utils")
    utils.LocalEntryNotFoundError = LocalEntryNotFoundError  # older-hub fallback
    utils.disable_progress_bars = lambda: bars.append("off")
    utils.enable_progress_bars = lambda: bars.append("on")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", utils)
    return calls, constants, bars


def test_local_dir_is_noop(monkeypatch, tmp_path):
    calls, constants, bars = _stub_hf(monkeypatch, cached=True)
    d = tmp_path / "my-tts"
    d.mkdir()
    with hf_cache.offline_resolve(str(d)):
        pass
    assert calls == []                       # never touched the Hub
    assert bars == []                        # progress bars untouched
    assert constants.HF_HUB_OFFLINE is False


def test_empty_ref_is_noop(monkeypatch):
    calls, _constants, _bars = _stub_hf(monkeypatch, cached=True)
    with hf_cache.offline_resolve(""):
        pass
    assert calls == []


def test_cached_repo_forces_offline_then_restores(monkeypatch):
    calls, constants, bars = _stub_hf(monkeypatch, cached=True, offline_start=False)
    assert constants.HF_HUB_OFFLINE is False
    with hf_cache.offline_resolve("org/model"):
        # inside the window the backend resolves offline + quiet
        assert constants.HF_HUB_OFFLINE is True
        assert bars == ["off"]
    # restored afterwards
    assert constants.HF_HUB_OFFLINE is False
    assert bars == ["off", "on"]
    # exactly one ensure-cache call, and it was the no-network probe (flag still
    # OFF at the time of the probe so a real miss could download)
    assert calls == [{"repo": "org/model", "local_files_only": True,
                      "offline": False}]


def test_cached_repo_restores_prior_offline_true(monkeypatch):
    # If the process was already offline, the prior value (True) is restored.
    _calls, constants, _bars = _stub_hf(monkeypatch, cached=True, offline_start=True)
    with hf_cache.offline_resolve("org/model"):
        assert constants.HF_HUB_OFFLINE is True
    assert constants.HF_HUB_OFFLINE is True


def test_uncached_repo_downloads_once(monkeypatch, capsys):
    calls, constants, _bars = _stub_hf(monkeypatch, cached=False)
    with hf_cache.offline_resolve("org/missing"):
        assert constants.HF_HUB_OFFLINE is True
    # first the offline probe (raises), then the real download - both BEFORE the
    # flag flips ON, so the download isn't starved offline.
    assert [(c["local_files_only"], c["offline"]) for c in calls] == [
        (True, False), (False, False)]
    assert "downloading once" in capsys.readouterr().out
    assert constants.HF_HUB_OFFLINE is False  # restored


def test_restores_offline_on_exception(monkeypatch):
    # An error inside the block still restores the prior offline flag.
    _calls, constants, bars = _stub_hf(monkeypatch, cached=True, offline_start=False)
    with pytest.raises(ValueError):
        with hf_cache.offline_resolve("org/model"):
            raise ValueError("boom")
    assert constants.HF_HUB_OFFLINE is False
    assert bars == ["off", "on"]             # progress bars re-enabled too


def test_network_fetch_waits_for_offline_window(monkeypatch):
    """A guarded load-path fetch must not run inside another thread's
    forced-offline window (it would see HF_HUB_OFFLINE=True and fail)."""
    import threading
    import time

    calls, constants, _bars = _stub_hf(monkeypatch, cached=True)
    entered = threading.Event()
    release = threading.Event()
    seen_offline = []

    def hold_window():
        with hf_cache.offline_resolve("org/model"):
            entered.set()
            release.wait(5)

    t = threading.Thread(target=hold_window)
    t.start()
    assert entered.wait(5)

    def fetch():
        with hf_cache.network_fetch_allowed():
            seen_offline.append(constants.HF_HUB_OFFLINE)

    f = threading.Thread(target=fetch)
    f.start()
    time.sleep(0.05)
    assert seen_offline == []          # blocked while the window is open
    assert constants.HF_HUB_OFFLINE is True
    release.set()
    f.join(5)
    t.join(5)
    assert seen_offline == [False]     # ran after the flag was restored
