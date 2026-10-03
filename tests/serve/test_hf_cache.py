#!/usr/bin/env python3
"""Offline-resolve window (``gmlx.serve.hf_cache.offline_resolve``): a local path or
empty ref is a no-op; a repo id is ensured in the HF cache (cache-only probe, with
a one-time download only on a genuine miss) and then loaded with ``HF_HUB_OFFLINE``
forced on for the duration and restored after.

The window tests stub ``huggingface_hub``. The fetch tests run the real
huggingface_hub cache code against a scratch cache in tmp_path, with a local
stand-in for the Hub's HTTP side (fake_hub.py), so no test reaches the network
or the real cache."""
from __future__ import annotations

import sys
import types

import pytest

import gmlx.serve.hf_cache as hf_cache


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
    # One cache lookup, the probe that makes no request. The flag was still
    # off at the probe, so a real miss could download.
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
    # First the offline probe, which raises, then the real download. Both run
    # before the flag is turned on, so the download is not starved offline.
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


# The pinned fetch, on the real huggingface_hub cache code. C1 is the commit
# the cache holds, C2 a newer head on the Hub.
C1 = "1" * 40
C2 = "2" * 40
REPO = "acme/encoder"
_FILES_C1 = {"config.json": '{"model_type": "bert"}',
             "README.md": "---\nbase_model: acme/base\n---\n# C1\n",
             "modules.json": "[]"}
_FILES_C2 = {**_FILES_C1, "README.md": "# C2\n",
             "1_Pooling/config.json": "{}"}


@pytest.fixture
def hub(monkeypatch, tmp_path):
    """A fake Hub holding REPO at C1 and at the newer head C2, and a scratch
    cache holding the C1 snapshot without its model card, as a download of
    the weights, configs and tokenizer files leaves it."""
    import huggingface_hub
    from fake_hub import FakeHub, use_hub

    fake = FakeHub()
    fake.commit(REPO, C1, _FILES_C1, head=False)
    fake.commit(REPO, C2, _FILES_C2)
    use_hub(monkeypatch, fake.endpoint, str(tmp_path / "hf"))
    huggingface_hub.snapshot_download(REPO, revision=C1,
                                      allow_patterns=["config.json"])
    store = tmp_path / "hf" / "models--acme--encoder"
    (store / "refs").mkdir(exist_ok=True)
    (store / "refs" / "main").write_text(C1)
    fake.requests.clear()
    try:
        yield fake
        assert fake.tokens == [], "the user's token reached the fake Hub"
    finally:
        fake.close()


def _card(snapshot):
    return ["README.md", "modules.json", "1_Pooling/config.json"]


def _window_snapshot(repo):
    import huggingface_hub

    return huggingface_hub.snapshot_download(repo, local_files_only=True)


def test_a_fetch_reads_the_cached_commit_and_moves_no_ref(hub, tmp_path):
    # Each file is requested at C1, the commit of the cached snapshot. The
    # Hub's head moves nothing in the cache, and the window resolves C1.
    from fake_hub import cache_state

    cache = str(tmp_path / "hf")
    with hf_cache.offline_resolve(REPO, wanted=_card):
        snapshot = _window_snapshot(REPO)
    assert snapshot.endswith(C1)
    assert hub.requests == [
        ("GET", f"/{REPO}/resolve/{C1}/README.md"),
        ("GET", f"/{REPO}/resolve/{C1}/modules.json"),
        ("GET", f"/{REPO}/resolve/{C1}/1_Pooling/config.json")]
    assert cache_state(cache, REPO) == {
        "main": C1, "snapshots": {C1: ["README.md", "config.json",
                                       "modules.json"]}}
    with open(f"{snapshot}/README.md") as f:
        assert f.read() == _FILES_C1["README.md"]
    # The file sits where huggingface_hub puts it, and the hub finds it.
    import huggingface_hub

    assert huggingface_hub.try_to_load_from_cache(
        REPO, "README.md", revision=C1) == f"{snapshot}/README.md"
    # Its blob has the mode of the blob huggingface_hub wrote, a new file's
    # mode under the umask.
    import os
    import stat

    mask = os.umask(0)
    os.umask(mask)

    def mode(name):
        return stat.S_IMODE(os.stat(os.path.realpath(f"{snapshot}/{name}")).st_mode)

    assert mode("README.md") == mode("config.json") == 0o666 & ~mask
    # The fetch ran once for this snapshot: the next load requests nothing.
    hub.requests.clear()
    with hf_cache.offline_resolve(REPO, wanted=_card):
        pass
    assert hub.requests == []


def test_a_settled_snapshot_requests_nothing(hub, tmp_path):
    import os

    with hf_cache.offline_resolve(REPO, wanted=lambda snapshot: []):
        pass
    with hf_cache.offline_resolve(REPO, wanted=lambda snapshot: ["config.json"]):
        pass
    assert hub.requests == []
    assert not os.path.exists(tmp_path / "hf" / "models--acme--encoder"
                              / hf_cache.FETCHED_DIR)


def test_offline_fetches_nothing_and_leaves_no_mark(hub, tmp_path, monkeypatch):
    import os

    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    with hf_cache.offline_resolve(REPO, wanted=_card):
        assert _window_snapshot(REPO).endswith(C1)
    assert hub.requests == []
    assert not os.path.exists(tmp_path / "hf" / "models--acme--encoder"
                              / hf_cache.FETCHED_DIR)


def _mark(tmp_path):
    return str(tmp_path / "hf" / "models--acme--encoder" / hf_cache.FETCHED_DIR
               / C1)


def test_a_server_error_ends_the_fetch_and_the_next_start_tries_again(
        hub, tmp_path, capsys):
    # A 503 ends the fetch after one request, with no retry. It is no answer
    # about the file, so no mark is left and the next start asks again.
    import os

    from fake_hub import cache_state

    hub.fail_status = 503
    with hf_cache.offline_resolve(REPO, wanted=_card):
        pass
    assert hub.requests == [("GET", f"/{REPO}/resolve/{C1}/README.md")]
    assert ("README.md not fetched (OSError: HTTP 503); the next start tries "
            "again") in capsys.readouterr().err
    assert not os.path.exists(_mark(tmp_path))
    hub.fail_status = None
    with hf_cache.offline_resolve(REPO, wanted=_card):
        pass
    assert len(hub.requests) == 4
    assert os.path.exists(_mark(tmp_path))
    assert cache_state(str(tmp_path / "hf"), REPO)["main"] == C1


@pytest.mark.parametrize("status", [401, 403])
def test_a_refused_request_is_an_answer_and_is_not_repeated(
        hub, tmp_path, capsys, status):
    # A gated or private repo answers 401 or 403. The fetch ends, and the
    # mark, named in the log, keeps the next start from asking again.
    hub.fail_status = status
    with hf_cache.offline_resolve(REPO, wanted=_card):
        pass
    assert len(hub.requests) == 1
    assert (f"README.md not fetched (OSError: HTTP {status}); delete "
            f"{_mark(tmp_path)} to try again") in capsys.readouterr().err
    with hf_cache.offline_resolve(REPO, wanted=_card):
        pass
    assert len(hub.requests) == 1


def test_a_failed_connection_leaves_no_mark_and_the_next_start_fetches(
        hub, tmp_path, monkeypatch, capsys):
    # A first start with no connection to the Hub, then a start online: the
    # online start fetches the card at C1, as if the first had not run.
    import os

    from fake_hub import cache_state, refused_port, use_hub

    use_hub(monkeypatch, refused_port(), str(tmp_path / "hf"))
    with hf_cache.offline_resolve(REPO, wanted=_card):
        pass
    err = capsys.readouterr().err
    assert "README.md not fetched (ConnectError" in err
    assert "the next start tries again" in err
    assert not os.path.exists(_mark(tmp_path))
    use_hub(monkeypatch, hub.endpoint, str(tmp_path / "hf"))
    with hf_cache.offline_resolve(REPO, wanted=_card):
        pass
    assert hub.requests == [
        ("GET", f"/{REPO}/resolve/{C1}/README.md"),
        ("GET", f"/{REPO}/resolve/{C1}/modules.json"),
        ("GET", f"/{REPO}/resolve/{C1}/1_Pooling/config.json")]
    assert cache_state(str(tmp_path / "hf"), REPO)["snapshots"][C1] == [
        "README.md", "config.json", "modules.json"]
    assert os.path.exists(_mark(tmp_path))


def test_one_deadline_bounds_the_whole_fetch(hub, tmp_path, monkeypatch,
                                             capsys):
    # A Hub that answers each request after 0.8 s. The first file arrives,
    # and the second request gets what is left of one FETCH_TIMEOUT, so the
    # call ends at about 1 s, not 3 x 0.8 s. A timeout is an answer: the
    # mark keeps the next start from asking again.
    import time

    monkeypatch.setattr(hf_cache, "FETCH_TIMEOUT", 1.0)
    hub.delay = 0.8
    started = time.monotonic()
    with hf_cache.offline_resolve(REPO, wanted=_card):
        pass
    assert time.monotonic() - started < 1.6
    assert len(hub.requests) == 2
    assert (f"modules.json not fetched (ReadTimeout: timed out); delete "
            f"{_mark(tmp_path)} to try again") in capsys.readouterr().err
    with hf_cache.offline_resolve(REPO, wanted=_card):
        pass
    assert len(hub.requests) == 2


def test_a_black_holed_hub_costs_one_short_try_with_no_lock_held(
        hub, tmp_path, monkeypatch):
    # The endpoint accepts and never answers. The fetch gives up after its
    # timeout, on its one connection. While it waits, it holds no lock: a
    # load-path fetch of another thread proceeds.
    import threading
    import time

    from fake_hub import BlackHole, use_hub

    hole = BlackHole()
    use_hub(monkeypatch, hole.endpoint, str(tmp_path / "hf"))
    monkeypatch.setattr(hf_cache, "FETCH_TIMEOUT", 1.0)
    done = threading.Event()

    def load():
        with hf_cache.offline_resolve(REPO, wanted=_card):
            done.set()

    started = time.monotonic()
    t = threading.Thread(target=load, daemon=True)
    t.start()
    try:
        deadline = time.monotonic() + 5
        while hole.accepted == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert hole.accepted == 1 and not done.is_set()
        entered = threading.Event()

        def other():
            with hf_cache.network_fetch_allowed():
                entered.set()

        threading.Thread(target=other, daemon=True).start()
        assert entered.wait(0.5), "a fetch waited on the offline lock"
        assert not done.is_set()
        assert done.wait(10)
        assert time.monotonic() - started < 5
        t.join(5)
        assert hole.accepted == 1
        # The next load does not try again.
        with hf_cache.offline_resolve(REPO, wanted=_card):
            pass
        assert hole.accepted == 1
    finally:
        hole.close()


def test_a_read_only_cache_requests_nothing_and_says_why(hub, tmp_path,
                                                         capsys):
    # A cache the server cannot write would keep neither the files nor the
    # mark, and every start would request them again. None does.
    import os
    import stat

    store = tmp_path / "hf" / "models--acme--encoder"
    folders = [d for d, _, _ in os.walk(store)]
    for d in folders:
        os.chmod(d, stat.S_IMODE(os.stat(d).st_mode) & ~0o222)
    try:
        if os.access(store, os.W_OK):
            pytest.skip("the folder stays writable to this user")
        for _ in range(2):
            with hf_cache.offline_resolve(REPO, wanted=_card):
                assert _window_snapshot(REPO).endswith(C1)
        assert hub.requests == []
        assert (f"README.md, modules.json, 1_Pooling/config.json not "
                f"fetched, since the HF cache folder {store} is read-only"
                in capsys.readouterr().err)
    finally:
        for d in folders:
            os.chmod(d, 0o755)


def test_a_mark_that_cannot_be_written_is_said_and_tried_again(
        hub, tmp_path, capsys):
    # A file where the mark folder belongs: the files are fetched, the mark
    # cannot be written, the log says so, and the next start asks again.
    (tmp_path / "hf" / "models--acme--encoder" / hf_cache.FETCHED_DIR
     ).write_text("")
    with hf_cache.offline_resolve(REPO, wanted=_card):
        pass
    assert len(hub.requests) == 3
    assert (f"{_mark(tmp_path)} could not be written, so the next start "
            "tries again") in capsys.readouterr().err
    with hf_cache.offline_resolve(REPO, wanted=_card):
        pass
    assert hub.requests[3:] == [
        ("GET", f"/{REPO}/resolve/{C1}/1_Pooling/config.json")]


def test_a_fetch_writes_nothing_outside_the_snapshot(hub, tmp_path):
    import os

    hub.commit(REPO, C1, {**_FILES_C1, "../escape": "x"}, head=False)
    before = sorted(os.listdir(tmp_path / "hf" / "models--acme--encoder"))
    with hf_cache.offline_resolve(
            REPO, wanted=lambda s: ["../escape", "../../x", "/etc/x"]):
        pass
    assert hub.requests == []
    assert sorted(os.listdir(tmp_path / "hf" / "models--acme--encoder")) == \
        sorted(before + [hf_cache.FETCHED_DIR])
