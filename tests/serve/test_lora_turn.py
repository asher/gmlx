"""The residency pool's LoRA turn: a worker job books the LoRA bytes it
allocates next under the pool's lock, sheds and evicts as a cold load
does, and waits only for chat models in use (residency). The pool here
holds image entries whose pipelines are stand-ins, so no model loads;
their caches are dicts of keys and bytes."""

import contextlib
import gc
import random
import threading
from types import SimpleNamespace

import pytest

from gmlx.serve import residency, residency_lora, residency_workers
from gmlx.serve.residency import (IMAGE_KIND, STORE, VIDEO_KIND,
                                  LoraRefused, _ResidencyPool)

GB = 10**9
# Seconds a pool thread may take before the test calls it stuck.
WAIT = 10


class Cache:
    """An entry's LoRA caches as the pool reads them: request keys and
    their bytes, and its own bytes. A shed lets the oldest keys go."""

    def __init__(self, own=0):
        self.items: dict = {}
        self.own = own

    def figures(self):
        return dict(self.items), self.own

    def shed(self, nbytes, hold=()):
        dropped = 0
        for k in list(self.items):
            if dropped >= nbytes:
                break
            if k not in hold:
                dropped += self.items.pop(k)
        return dropped


class Store:
    """The upload store as a turn and an upload see it: its bytes, one
    int the pool reads with no lock, and its shed."""

    def __init__(self):
        self.items: dict = {}
        self.n = 0
        self.lock = threading.Lock()

    def shed(self, nbytes, hold=()):
        dropped = 0
        with self.lock:
            for k in list(self.items):
                if dropped >= nbytes:
                    break
                if k not in hold:
                    dropped += self.items.pop(k)
            self.n -= dropped
        return dropped

    def put(self, pool, key, n):
        """An admitted body's put, then its done."""
        with self.lock:
            self.items[key] = n
            self.n += n
        pool.upload_done(n)

    def keep(self, key, n):
        """Bytes held before the test, admitted by no one."""
        with self.lock:
            self.items[key] = n
            self.n += n


def make_pool(monkeypatch, budget_gb, retry=1):
    from gmlx.serve import capacity as cap

    monkeypatch.setattr(cap, "preload_gate", lambda *a, **k: None)
    monkeypatch.delenv("GMLX_OVERCOMMIT", raising=False)
    monkeypatch.setattr(residency_workers, "_holds_image_worker", lambda: True)
    monkeypatch.setattr(residency_workers, "_holds_video_worker", lambda: True)
    # The pool collects after each pipeline release. The pipelines here are
    # stand-ins, so a collection frees nothing, and in a full test process
    # each one walks a large heap.
    monkeypatch.setattr(gc, "collect", lambda generation=2: 0)
    released = []
    # The thread each pipeline release ran on, by pipeline.
    pool_threads: dict = {}

    def release(box):
        pipe = box.pop()
        released.append(pipe)
        pool_threads[pipe.name] = threading.current_thread().name
        pool.caches.pop(pipe.name, None)

    pool = _ResidencyPool(
        SimpleNamespace(_last_entry=None), None, None, int(budget_gb * GB),
        (), image_fns=(lambda spec: spec.fp, lambda spec: spec.pipe,
                       release))
    pool._video_fns = pool._image_fns
    pool.released = released
    pool.release_threads = pool_threads
    pool._teardown_chat = lambda e: released.append(e.model_path)
    pool._worker_for = lambda *a, **k: contextlib.nullcontext()
    pool.head = None
    pool._headroom_fn = lambda: pool.head
    pool.caches = {}
    pool.store = Store()

    def unlocked(fn):
        # A cache's lock taken inside a hold of the pool's lock deadlocks
        # with a finalizer that releases into the pool under it.
        def run(*a):
            assert not pool._lock._is_owned(), "a LoRA cache read under _lock"
            return fn(*a)
        return run

    pool._lora_figures = unlocked(lambda e: pool.caches.setdefault(
        e.model_path, Cache()).figures())
    pool._lora_shed = unlocked(lambda e, n, hold, own:
                               pool.caches[e.model_path].shed(n, hold))
    pool._store_shed = unlocked(lambda n, hold=(): pool.store.shed(n, hold))
    pool._store_live = lambda: pool.store.n
    # The deadline reads the server's readiness, code outside the pool.
    pool._retry_fn = unlocked(lambda: retry)
    return pool


def spec(name, fp_gb):
    return SimpleNamespace(fp=int(fp_gb * GB), stream=None, path=name,
                           pipe=SimpleNamespace(name=name))


def image(pool, name, fp_gb, *, busy=False):
    e = pool.acquire(name, None, IMAGE_KIND, build_spec=spec(name, fp_gb))
    if not busy:
        pool.release(e)
    return e


def video(pool, name, fp_gb, *, busy=False):
    e = pool.acquire(name, None, VIDEO_KIND, build_spec=spec(name, fp_gb))
    if not busy:
        pool.release(e)
    return e


def chat(pool, name, fp_gb, *, busy=1, retained=0, pinned=False):
    """A chat entry put in the pool as a cold load would leave it."""
    e = residency._Entry(cache_key=(name, None, "chat"), model_path=name,
                         footprint=int(fp_gb * GB), busy=busy,
                         retained=retained, pinned=pinned)
    with pool._lock:
        pool._entries[e.cache_key] = e
        pool._touch(e)
        pool._publish()
    return e


def start(name, fn):
    out: dict = {}

    def run():
        try:
            out["value"] = fn()
        except BaseException as e:
            out["error"] = e
    t = threading.Thread(target=run, name=name, daemon=True)
    t.start()
    return t, out


def finish(t, out):
    t.join(WAIT)
    assert not t.is_alive(), f"{t.name} is stuck"
    if "error" in out:
        raise out["error"]
    return out.get("value")


def on_worker(fn, kind=IMAGE_KIND):
    """``fn`` on a thread named as the workers of ``kind`` are."""
    name = "video-worker-test" if kind == VIDEO_KIND else "image-worker-test"
    return finish(*start(name, fn))


class Waits(threading.Condition):
    """The pool's release condition, recording the timeout of each wait
    a turn takes on it and setting ``waiting`` at the first."""

    def __init__(self, lock):
        super().__init__(lock)
        self.timeouts: list = []
        self.waiting = threading.Event()

    def wait(self, timeout=None):
        self.timeouts.append(timeout)
        self.waiting.set()
        return super().wait(timeout)


def waits(pool) -> Waits:
    pool._idle = Waits(pool._lock)
    return pool._idle


def turn(pool, e, need, **kw):
    kw.setdefault("subject", "the test's LoRA")
    kind = e.kind if e is not STORE else IMAGE_KIND
    return on_worker(lambda: pool.lora_turn(e, int(need), **kw), kind)


def job(pool, e, key, need, **kw):
    """One job: its turn, its put, its settle."""
    def run():
        try:
            pool.lora_turn(e, int(need), subject="the test's LoRA", **kw)
            pool.caches[e.model_path].items[key] = int(need)
        finally:
            pool.lora_settle(e, "the test's LoRA")
    on_worker(run, e.kind)


def live_sums(pool):
    es = list(pool._entries.values())
    return (sum(e.footprint for e in es),
            sum(e.lora.held + e.lora.pending for e in es if e.lora))


class Seams:
    """Pauses a pool thread the first time it reaches each named seam,
    until the test lets it go."""

    def __init__(self, *names):
        self.at = {n: threading.Event() for n in names}
        self.go = {n: threading.Event() for n in names}
        self.hit: list = []

    def __call__(self, name):
        self.hit.append(name)
        if name in self.at and not self.at[name].is_set():
            self.at[name].set()
            assert self.go[name].wait(WAIT)

    def reach(self, name):
        assert self.at[name].wait(WAIT), f"no thread reached {name}"


def test_a_turn_runs_on_the_worker(monkeypatch):
    pool = make_pool(monkeypatch, 10)
    e = image(pool, "x", 2, busy=True)
    with pytest.raises(RuntimeError, match="image worker"):
        pool.lora_turn(e, GB, subject="s")
    turn(pool, e, GB)
    assert e.lora.pending == GB


def test_a_turn_evicts_past_the_bytes_still_tearing_down(monkeypatch):
    """Bytes of a teardown still running count in the turn's shortfall as
    they count in its fit, so the turn evicts the idle entry that makes
    the room, where it evicted nothing and refused."""
    pool = make_pool(monkeypatch, 10)
    e = image(pool, "x", 2, busy=True)
    image(pool, "y", 4)
    with pool._lock:
        pool._tearing = 2 * GB          # another teardown, not yet done
    turn(pool, e, 3 * GB)
    assert e.lora.pending == 3 * GB
    assert [p.name for p in pool.released] == ["y"]


def test_a_video_turn_runs_on_the_video_worker(monkeypatch):
    """A video entry's job allocates on the video worker, so its turn
    there is refused on the image worker."""
    pool = make_pool(monkeypatch, 10)
    v = video(pool, "v", 2, busy=True)
    with pytest.raises(RuntimeError, match="video worker"):
        on_worker(lambda: pool.lora_turn(v, GB, subject="s"), IMAGE_KIND)
    turn(pool, v, GB)
    assert v.lora.pending == GB


def test_a_video_turn_short_by_an_idle_image_entry_names_it(monkeypatch):
    """The room a video turn misses is held by an idle image entry: the
    turn never evicts it, which would release its pipeline on the image
    worker, and its 503 names it with the unload that makes room."""
    pool = make_pool(monkeypatch, 10)
    zimage = image(pool, "zimage", 4)
    v = video(pool, "v", 4, busy=True)
    w = waits(pool)
    with pytest.raises(LoraRefused) as ei:
        turn(pool, v, 3 * GB)
    assert (ei.value.status, ei.value.message, ei.value.transient) == (
        503, "the test's LoRA need 3.0 GB of memory, 2.0 GB is free, and "
        "'zimage' is idle on the image worker; unload it (POST /unload) to "
        "make room", False)
    assert pool.entry_resident(zimage) and pool.released == []
    assert w.timeouts == []


def test_a_store_turn_short_by_an_idle_video_entry_names_it(monkeypatch):
    pool = make_pool(monkeypatch, 10)
    clip = video(pool, "clip", 4)
    image(pool, "x", 4, busy=True)
    with pytest.raises(LoraRefused) as ei:
        turn(pool, STORE, 3 * GB, subject="the upload")
    assert ei.value.message == (
        "the upload need 3.0 GB of memory, 2.0 GB is free, and 'clip' is "
        "idle on the video worker; unload it (POST /unload) to make room")
    assert pool.entry_resident(clip) and pool.released == []


def test_a_video_turn_evicts_its_own_idle_entries_and_chat(monkeypatch):
    """The entries a video turn can evict are released inline on the
    video worker, or need no worker: an idle chat model and an idle video
    entry go, the idle image entry beside them stays."""
    pool = make_pool(monkeypatch, 12)
    zimage = image(pool, "zimage", 2)
    old = video(pool, "old", 3)
    qwen = chat(pool, "qwen3", 3, busy=0)
    v = video(pool, "v", 3, busy=True)
    turn(pool, v, 5 * GB)
    assert not pool.entry_resident(old) and not pool.entry_resident(qwen)
    assert pool.entry_resident(zimage)
    assert pool.release_threads == {"old": "video-worker-test"}


def test_a_video_turn_waits_on_an_image_release_before_it_evicts(
        monkeypatch):
    """An unload of an image model, as its job ends, releases its pipeline
    on the image worker once the build lock drops, so its 3 GB still
    count as the video turn takes the lock. The turn waits on that
    release, which makes its room, where it evicted the idle chat
    model."""
    pool = make_pool(monkeypatch, 10, retry=WAIT)
    zimage = image(pool, "zimage", 3)
    qwen = chat(pool, "qwen3", 3, busy=0)
    v = video(pool, "v", 3, busy=True)
    # The unload does not hold the image worker: its job held it at the
    # unload's reservation.
    monkeypatch.setattr(residency_workers, "_holds_image_worker", lambda: False)
    fp, load, release = pool._image_fns
    at, go, either = threading.Event(), threading.Event(), threading.Event()

    def queued(box):
        # Queued on the image worker behind its next job.
        at.set()
        assert go.wait(WAIT)
        release(box)
    pool._image_fns = (fp, load, queued)
    w = waits(pool)
    cond_wait = w.wait

    def wait(timeout=None):
        either.set()
        return cond_wait(timeout)
    w.wait = wait
    unload = start("unload", lambda: pool.evict("zimage"))
    assert at.wait(WAIT), "no release started"
    assert not pool.entry_resident(zimage)
    done = threading.Event()

    def run():
        try:
            pool.lora_turn(v, 3 * GB, subject="the test's LoRA")
            done.set()
        finally:
            either.set()
    job = start("video-worker-test", run)
    try:
        assert either.wait(WAIT)
        assert w.waiting.wait(WAIT) and not done.is_set(), \
            "the turn did not wait on the release"
    finally:
        go.set()
    finish(*job)
    finish(*unload)
    assert pool.entry_resident(qwen)
    assert v.lora.pending == 3 * GB
    assert pool.figures.tearing == 0


class Queued:
    """Unloads of image entries whose pipeline releases are queued on the
    image worker behind a job: each runs once its ``go`` is set."""

    def __init__(self, pool, monkeypatch, *names):
        # The unload does not hold the image worker: its job held it at
        # the unload's reservation.
        monkeypatch.setattr(residency_workers, "_holds_image_worker", lambda: False)
        fp, load, release = pool._image_fns
        self.at = {n: threading.Event() for n in names}
        self.go = {n: threading.Event() for n in names}

        def queued(box):
            name = box[-1].name
            self.at[name].set()
            assert self.go[name].wait(WAIT)
            release(box)
        pool._image_fns = (fp, load, queued)
        self.unloads = {}
        for n in names:
            self.unloads[n] = start(f"unload-{n}", lambda n=n: pool.evict(n))
            assert self.at[n].wait(WAIT), f"no release of {n} started"

    def run(self, *names):
        for n in names or list(self.unloads):
            self.go[n].set()
            if n in self.unloads:
                finish(*self.unloads.pop(n))


class BuildLock:
    """The pool's build lock, setting ``blocked`` when a turn finds it
    held by another thread once ``armed`` is set."""

    def __init__(self):
        self.lock = threading.Lock()
        self.armed = threading.Event()
        self.blocked = threading.Event()

    def acquire(self, blocking=True, timeout=-1):
        if self.armed.is_set() and self.lock.acquire(blocking=False):
            return True
        if self.armed.is_set():
            self.blocked.set()
        return self.lock.acquire(blocking, timeout)

    def release(self):
        self.lock.release()

    def __enter__(self):
        self.acquire()

    def __exit__(self, *exc):
        self.release()


def slow_first_pass(pool, monkeypatch, seconds):
    """The turn's clock runs ``seconds`` ahead at each shed, as a shed of
    large caches takes that long, with no sleep."""
    import time

    ahead = [0.0]
    monkeypatch.setattr(residency_lora, "time", SimpleNamespace(
        monotonic=lambda: time.monotonic() + ahead[0], sleep=time.sleep,
        time=time.time))
    shed = pool._lora_shed

    def slow(*a):
        ahead[0] += seconds
        return shed(*a)
    pool._lora_shed = slow


def test_a_turn_evicts_once_its_wait_on_a_stuck_release_elapses(
        monkeypatch):
    """A release queued behind a job that does not end keeps its bytes:
    the turn waits on it for half the deadline left after a slow shed,
    then evicts the idle chat model and books, with the release still
    counted."""
    pool = make_pool(monkeypatch, 10, retry=1)
    image(pool, "zimage", 3)
    qwen = chat(pool, "qwen3", 3, busy=0)
    v = video(pool, "v", 3, busy=True)
    q = Queued(pool, monkeypatch, "zimage")
    w = waits(pool)
    slow_first_pass(pool, monkeypatch, 0.6)
    try:
        turn(pool, v, 3 * GB)
        assert not pool.entry_resident(qwen)
        assert v.lora.pending == 3 * GB
        assert pool.figures.tearing == 3 * GB
        # Half the 0.4 s left once the 0.6 s shed is done.
        assert w.timeouts and 0 < w.timeouts[0] <= 0.2
    finally:
        q.run()
    assert pool.figures.tearing == 0


def test_a_turn_waits_for_the_build_lock_after_its_wait_aside(
        monkeypatch):
    """The wait on a release takes half the deadline: a build that holds
    the lock as it ends makes the turn wait for it and evict, not 503."""
    pool = make_pool(monkeypatch, 10, retry=2)
    image(pool, "zimage", 3)
    qwen = chat(pool, "qwen3", 3, busy=0)
    v = video(pool, "v", 3, busy=True)
    q = Queued(pool, monkeypatch, "zimage")
    lock = pool._build_lock = BuildLock()
    w = waits(pool)
    try:
        job = start("video-worker-test", lambda: pool.lora_turn(
            v, 3 * GB, subject="the test's LoRA"))
        assert w.waiting.wait(WAIT), "the turn did not wait on the release"
        with lock:
            lock.armed.set()
            assert lock.blocked.wait(WAIT)
        finish(*job)
        assert not pool.entry_resident(qwen)
        assert v.lora.pending == 3 * GB
    finally:
        q.run()


def test_a_turn_short_of_more_than_a_release_frees_does_not_wait(
        monkeypatch):
    """A release on the other worker that frees less than the turn is
    short of leaves it to evict anyway: the turn evicts at once."""
    pool = make_pool(monkeypatch, 10, retry=1)
    image(pool, "zimage", 1)
    qwen = chat(pool, "qwen3", 3, busy=0)
    v = video(pool, "v", 4, busy=True)
    q = Queued(pool, monkeypatch, "zimage")
    w = waits(pool)
    try:
        # 1 + 3 + 4 held of 10: 4 GB more is 2 GB short, 1 GB tearing.
        turn(pool, v, 4 * GB)
        assert not pool.entry_resident(qwen)
        assert v.lora.pending == 4 * GB
        assert w.timeouts == []
    finally:
        q.run()


def test_a_turn_whose_shed_covers_it_books_beside_a_queued_release(
        monkeypatch):
    """The turn sheds its own idle LoRA bytes before it looks at a release
    queued on the other worker: when the shed covers it, it books with no
    wait and evicts nothing."""
    pool = make_pool(monkeypatch, 12, retry=1)
    image(pool, "zimage", 3)
    qwen = chat(pool, "qwen3", 3, busy=0)
    v = video(pool, "v", 3, busy=True)
    job(pool, v, "old", 2 * GB)
    q = Queued(pool, monkeypatch, "zimage")
    w = waits(pool)
    try:
        # 3 + 3 + 3 + 2 held of 12: 2 GB more is 1 GB short, and the
        # idle 2 GB of "old" covers it.
        turn(pool, v, 2 * GB)
        assert w.timeouts == []
        assert pool.entry_resident(qwen)
        assert pool.caches["v"].items == {}
        assert v.lora.pending == 2 * GB
    finally:
        q.run()


def test_a_turn_stops_waiting_once_the_first_of_two_releases_covers_it(
        monkeypatch):
    """Two releases queued on the image worker: the first frees what the
    turn is short of, so it books then, with the second still queued."""
    pool = make_pool(monkeypatch, 12, retry=4 * WAIT)
    image(pool, "za", 2)
    image(pool, "zb", 3)
    qwen = chat(pool, "qwen3", 3, busy=0)
    v = video(pool, "v", 3, busy=True)
    q = Queued(pool, monkeypatch, "za", "zb")
    w = waits(pool)
    done = threading.Event()

    def run():
        pool.lora_turn(v, 2 * GB, subject="the test's LoRA")
        done.set()
    job_ = start("video-worker-test", run)
    try:
        # 2 + 3 + 3 + 3 held of 12: 2 GB more is 1 GB short.
        assert w.waiting.wait(WAIT), "the turn did not wait on the releases"
        q.run("za")
        assert done.wait(WAIT), "the turn waited past the release that " \
            "covers it"
        finish(*job_)
        assert pool.entry_resident(qwen)
        assert v.lora.pending == 2 * GB
        assert pool.figures.tearing == 3 * GB
    finally:
        q.run()


def test_a_release_queued_during_the_wait_does_not_hide_the_bytes_freed(
        monkeypatch):
    """A second release queued on the image worker while the turn waits
    adds to the bytes tearing down: the first release still frees what the
    turn is short of, so it books then."""
    pool = make_pool(monkeypatch, 12, retry=4 * WAIT)
    image(pool, "za", 2)
    image(pool, "zb", 3)
    qwen = chat(pool, "qwen3", 3, busy=0)
    v = video(pool, "v", 3, busy=True)
    q = Queued(pool, monkeypatch, "za")
    w = waits(pool)
    done = threading.Event()

    def run():
        pool.lora_turn(v, 2 * GB, subject="the test's LoRA")
        done.set()
    job_ = start("video-worker-test", run)
    try:
        # 2 + 3 + 3 + 3 held of 12: 2 GB more is 1 GB short.
        assert w.waiting.wait(WAIT), "the turn did not wait on the release"
        q.at["zb"], q.go["zb"] = threading.Event(), threading.Event()
        q.unloads["zb"] = start("unload-zb", lambda: pool.evict("zb"))
        assert q.at["zb"].wait(WAIT), "no release of zb started"
        q.run("za")
        assert done.wait(WAIT), "the turn waited past the release that " \
            "covers it"
        finish(*job_)
        assert pool.entry_resident(qwen)
        assert v.lora.pending == 2 * GB
        assert pool.figures.tearing == 3 * GB
    finally:
        q.run()


def test_a_video_turn_sheds_an_idle_upload_with_the_image_worker_busy(
        monkeypatch):
    """An upload is its body's bytes, made on no worker: a video turn
    sheds it in its own hold, while the image worker runs a job, and
    leaves the image entry's idle caches alone."""
    from gmlx.serve import image_service
    from gmlx.serve.subservice import SingleWorker

    pool = make_pool(monkeypatch, 10)
    x = image(pool, "x", 2, busy=True)
    v = video(pool, "v", 4, busy=True)
    job(pool, x, "x1", GB)
    pool.store.keep("u1", 2 * GB)
    busy = SingleWorker("image-worker-busy")
    monkeypatch.setattr(image_service, "WORKER", busy)
    go = threading.Event()
    blocked = busy.submit(go.wait, WAIT)
    try:
        # 9 GB held of 10: 2 GB more needs the upload gone.
        turn(pool, v, 3 * GB)
    finally:
        go.set()
        blocked.result()
    assert pool.store.items == {} and pool.caches["x"].items == {"x1": GB}
    assert v.lora.pending == 3 * GB


def test_the_ledger_holds_the_keys_the_caches_hold(monkeypatch):
    pool = make_pool(monkeypatch, 10)
    e = image(pool, "x", 2, busy=True)
    job(pool, e, "a", GB)
    job(pool, e, "b", GB // 2)
    assert e.lora.keys == {"a": GB, "b": GB // 2}
    assert e.lora.pending == 0
    assert pool._resident_bytes() == 2 * GB + GB + GB // 2
    pool.caches["x"].items.pop("a")
    pool.lora_settle(e, "s")
    assert e.lora.keys == {"b": GB // 2}
    assert pool.figures.lora == GB // 2


def test_own_bytes_past_the_term_count_and_own_keys_do_not(monkeypatch):
    pool = make_pool(monkeypatch, 10)
    pool.caches["x"] = Cache(own=GB)
    e = image(pool, "x", 2, busy=True)
    # A stand-in pipeline with no own term: all own bytes are excess.
    assert e.lora.excess == GB and e.lora.keys == {}
    pool.caches["x"].own = GB // 2
    pool.lora_settle(e, "s")
    assert e.lora.held == GB // 2


def test_an_unload_drops_the_ledger(monkeypatch):
    pool = make_pool(monkeypatch, 10)
    e = image(pool, "x", 2, busy=True)
    job(pool, e, "a", GB)
    pool.release(e)
    assert pool.evict("x")
    assert e.lora is None
    assert pool._resident_bytes() == 0
    assert pool.figures.lora == 0 and pool.figures.entries == ()


def test_a_settle_past_the_booking_logs_an_overrun(monkeypatch, caplog):
    pool = make_pool(monkeypatch, 10)
    got = []
    pool.overrun_hook = lambda subject, n: got.append(n)
    e = image(pool, "x", 2, busy=True)

    def run():
        pool.lora_turn(e, GB, subject="s")
        pool.caches["x"].items["a"] = 2 * GB
        pool.lora_settle(e, "s")
    with caplog.at_level("INFO", logger=residency._log.name):
        on_worker(run)
    assert got == [GB]
    assert [r.getMessage() for r in caplog.records
            if "past their turn's booking" in r.getMessage()] == [
        "s hold 1000.0 MB of LoRA memory past their turn's booking"]


def test_a_job_that_raises_after_its_turn_leaves_nothing_pending(
        monkeypatch):
    pool = make_pool(monkeypatch, 10)
    e = image(pool, "x", 2, busy=True)

    def run():
        try:
            pool.lora_turn(e, GB, subject="s")
            raise ValueError("the forward failed")
        finally:
            pool.lora_settle(e, "s")
    with pytest.raises(ValueError):
        on_worker(run)
    assert e.lora.pending == 0


def test_a_turn_sheds_its_own_entry_then_others_then_the_uploads(
        monkeypatch):
    pool = make_pool(monkeypatch, 10)
    y = image(pool, "y", 2)
    x = image(pool, "x", 2, busy=True)
    idle = image(pool, "idle", 1)
    job(pool, y, "y1", GB)
    job(pool, x, "x1", GB)
    job(pool, x, "mine", GB)
    pool.store.keep("u1", GB)
    # 9 GB held of 10: a 3 GB turn on x needs 2 GB, which x's own idle
    # key and y's give; the uploads and the idle model stay.
    turn(pool, x, 3 * GB, hold=("mine",))
    assert pool.caches["x"].items == {"mine": GB}
    assert pool.caches["y"].items == {}
    assert pool.store.items == {"u1": GB}
    assert pool.entry_resident(idle)
    assert y.lora.keys == {}


def test_a_turn_evicts_an_idle_model_after_the_sheds(monkeypatch):
    pool = make_pool(monkeypatch, 10)
    idle = image(pool, "idle", 3)
    x = image(pool, "x", 4, busy=True)
    job(pool, x, "x1", GB)
    pool.store.keep("u1", GB)
    # 9 GB held of 10: a 4 GB turn sheds the two idle 1 GB keys and then
    # evicts the idle model.
    turn(pool, x, 4 * GB)
    assert pool.caches["x"].items == {}
    assert pool.store.items == {}
    assert not pool.entry_resident(idle)
    assert len(pool.released) == 1
    assert x.lora.pending == 4 * GB
    assert pool._resident_bytes() + pool.store.n <= 10 * GB


def test_the_measure_follows_the_eviction(monkeypatch):
    pool = make_pool(monkeypatch, 20)
    idle = image(pool, "idle", 3)
    x = image(pool, "x", 4, busy=True)
    order = []
    # The machine holds what the pool holds; the budget is not the bound.
    machine = 9 * GB

    def head():
        order.append("measure")
        return machine - sum(e.footprint for e in pool._entries.values())

    pool._headroom_fn = head
    pool._seam = lambda name: order.append(name)
    with pytest.raises(LoraRefused):
        # 2 GB measured, 3 GB needed, and the budget fits: no eviction,
        # since the budget alone evicts.
        turn(pool, x, 3 * GB)
    assert pool.entry_resident(idle)
    pool._budget = 9 * GB
    order.clear()
    turn(pool, x, 4 * GB)
    assert not pool.entry_resident(idle)
    # The teardown comes before the measure the booking is checked on.
    last = len(order) - 1 - order[::-1].index("teardown")
    assert "measure" in order[last:]


def test_a_strict_turn_evicts_nothing_it_cannot_fit(monkeypatch):
    pool = make_pool(monkeypatch, 10)
    idle = image(pool, "idle", 2)
    x = image(pool, "x", 4, busy=True)
    chat(pool, "c", 3, busy=1, retained=1)
    # 9 GB held; 4 GB more needs 3 GB, and the idle model frees 2.
    with pytest.raises(LoraRefused) as ei:
        turn(pool, x, 4 * GB)
    assert ei.value.status == 503
    assert ei.value.retry_after == 1
    assert pool.entry_resident(idle)
    assert pool._resident_bytes() == 9 * GB


def test_a_turn_no_unload_can_fit_gets_413(monkeypatch):
    pool = make_pool(monkeypatch, 10)
    chat(pool, "p", 3, busy=0, pinned=True)
    x = image(pool, "x", 4, busy=True)
    with pytest.raises(LoraRefused) as ei:
        turn(pool, x, 3 * GB + 1, held=0)
    assert ei.value.status == 413
    assert ("need 3.1 GB of memory, over the 3.0 GB the server's memory "
            "budget leaves beside 'x' and the pinned models"
            in ei.value.message)
    # The route's check says the same, from the figures alone.
    with pytest.raises(LoraRefused) as er:
        pool.lora_hopeless(x, 3 * GB + 1, "the test's LoRA")
    assert er.value.message == ei.value.message
    pool.lora_hopeless(x, 3 * GB, "the test's LoRA")
    # The bytes it holds of its own keys count, and the text names them.
    pool.caches["x"].items["k"] = GB + 1
    with pytest.raises(LoraRefused) as eh:
        turn(pool, x, 2 * GB, held=GB + 1, hold=("k",))
    assert eh.value.status == 413 and eh.value.message == ei.value.message


def test_a_turn_that_allocates_nothing_passes_on_a_pool_past_its_budget(
        monkeypatch):
    pool = make_pool(monkeypatch, 10)
    x = image(pool, "x", 4, busy=True)
    chat(pool, "c", 6.5, busy=1)
    turn(pool, x, 0)
    assert pool.lora_ledger(x) == (0, {})


def test_a_turn_waits_for_a_chat_model_in_use_and_runs_on_its_release(
        monkeypatch):
    pool = make_pool(monkeypatch, 10, retry=5)
    x = image(pool, "x", 4, busy=True)
    c = chat(pool, "c", 4, busy=1)
    w = waits(pool)
    t, out = start("image-worker-test",
                   lambda: pool.lora_turn(x, 3 * GB, subject="s"))
    assert w.waiting.wait(WAIT), "the turn did not wait"
    assert t.is_alive()
    with pool._lock:
        c.busy = 0
    pool.release(SimpleNamespace(busy=0))
    finish(t, out)
    assert x.lora.pending == 3 * GB
    assert not pool.entry_resident(c)


def test_a_turn_waits_at_most_the_retry_and_names_the_models_in_use(
        monkeypatch):
    pool = make_pool(monkeypatch, 10, retry=1)
    x = image(pool, "x", 4, busy=True)
    chat(pool, "c", 4, busy=1)
    w = waits(pool)
    with pytest.raises(LoraRefused) as ei:
        turn(pool, x, 3 * GB)
    # Each wait is bounded by what is left of the one deadline.
    assert w.timeouts and all(0 < s <= 1 for s in w.timeouts)
    assert ei.value.status == 503 and ei.value.retry_after == 1
    assert ("need 3.0 GB of memory, and 2.0 GB is free beside the models "
            "in use (c); retry when one is idle" in ei.value.message)


def test_no_wait_on_a_worker_entry_in_use_or_a_preload(monkeypatch):
    pool = make_pool(monkeypatch, 10, retry=5)
    x = image(pool, "x", 4, busy=True)
    image(pool, "y", 4, busy=True)
    w = waits(pool)
    with pytest.raises(LoraRefused) as ei:
        turn(pool, x, 3 * GB)
    assert w.timeouts == []
    assert "models in use (y)" in ei.value.message
    pool2 = make_pool(monkeypatch, 10, retry=5)
    x2 = image(pool2, "x", 4, busy=True)
    chat(pool2, "pre", 4, busy=1, retained=1)
    w2 = waits(pool2)
    with pytest.raises(LoraRefused) as ei:
        turn(pool2, x2, 3 * GB)
    assert w2.timeouts == []
    assert "unload a model or retry" in ei.value.message


def test_a_measured_short_turn_refuses_at_once_with_no_chat_in_use(
        monkeypatch):
    pool = make_pool(monkeypatch, 10, retry=5)
    x = image(pool, "x", 2, busy=True)
    pool.head = 2 * GB
    with pytest.raises(LoraRefused) as ei:
        turn(pool, x, 3 * GB)
    assert ei.value.message == (
        "the test's LoRA need 3.0 GB of memory, and 2.0 GB is free in the "
        "working set now; close other apps or unload a model to make room")


def test_a_turn_counts_the_other_pending_bookings(monkeypatch):
    pool = make_pool(monkeypatch, 10)
    x = image(pool, "x", 2, busy=True)
    y = image(pool, "y", 1, busy=True)
    pool.head = 2 * GB
    turn(pool, x, int(1.5 * GB))
    with pytest.raises(LoraRefused) as ei:
        turn(pool, y, int(1.5 * GB))
    assert (ei.value.status, ei.value.message) == (
        503, "the test's LoRA need 1.5 GB of memory, and 0.5 GB is free in "
        "the working set now; retry when the models in use finish")


def test_overcommit_sheds_and_never_refuses(monkeypatch):
    """Under GMLX_OVERCOMMIT=1 a turn past the budget sheds the idle
    LoRA bytes and books, and evicts no model: the idle entry an eviction
    would take for the room stays resident."""
    pool = make_pool(monkeypatch, 6)
    monkeypatch.setenv("GMLX_OVERCOMMIT", "1")
    x = image(pool, "x", 2, busy=True)
    idle = image(pool, "idle", 2)
    job(pool, x, "old", GB)
    turn(pool, x, 3 * GB)
    assert pool.caches["x"].items == {}
    assert x.lora.pending == 3 * GB
    assert pool.entry_resident(idle)


def test_a_turn_during_a_cold_load_names_the_load(monkeypatch):
    pool = make_pool(monkeypatch, 10, retry=1)
    x = image(pool, "x", 4, busy=True)
    seams = Seams("roomed")
    pool._seam = seams
    load = start("loader", lambda: image(pool, "big", 5, busy=True))
    seams.reach("roomed")
    try:
        with pytest.raises(LoraRefused) as ei:
            turn(pool, x, 2 * GB)
    finally:
        seams.go["roomed"].set()
    finish(*load)
    assert ("and the server is loading big (5.0 GB) now; retry when it is "
            "loaded" in ei.value.message)
    assert ei.value.retry_after == 1


def test_a_turn_never_waits_on_the_encoder_its_own_job_holds(monkeypatch):
    # The images API job holds its shared encoder through both turns.
    pool = make_pool(monkeypatch, 10, retry=5)
    own = image(pool, "own", 3, busy=True)
    enc = chat(pool, "enc-qwen3", 4)
    w = waits(pool)
    turn(pool, own, 3 * GB, beside=(enc,))
    assert own.lora.pending == 3 * GB
    pool.lora_settle(own, "s")
    # Short beside another job's model: 503 at once, naming that one.
    other = image(pool, "other", 2, busy=True)
    with pytest.raises(LoraRefused) as ei:
        turn(pool, own, 2 * GB, beside=(enc,))
    assert ei.value.message == (
        "the test's LoRA need 2.0 GB of memory, and 1.0 GB is free beside "
        "the models in use (other); retry when one is idle")
    assert pool.entry_resident(other)
    # More than the budget leaves beside the model and its encoder: 413.
    with pytest.raises(LoraRefused) as ei:
        turn(pool, own, 4 * GB, beside=(enc,))
    assert (ei.value.status, ei.value.message) == (413, (
        "the test's LoRA need 4.0 GB of memory, over the 3.0 GB the "
        "server's memory budget leaves beside 'own', 'enc-qwen3' and the "
        "pinned models"))
    assert w.timeouts == [] and pool.entry_resident(enc)
    assert own.lora.pending == 0


@pytest.mark.parametrize("at", ["turn", "cold load"])
def test_a_chat_teardown_that_raises_still_releases_an_evicted_image(
        monkeypatch, at):
    pool = make_pool(monkeypatch, 10)
    own = image(pool, "own", 2, busy=True)
    side = image(pool, "side-image", 3)
    chat(pool, "old-chat", 3, busy=0)
    with pool._lock:
        side.seq = 0

    def boom(e):
        raise RuntimeError("chat teardown failed")
    pool._teardown_chat = boom
    with pytest.raises(RuntimeError, match="chat teardown failed"):
        if at == "turn":
            turn(pool, own, 7 * GB)
        else:
            image(pool, "new", 7)
    assert not pool.entry_resident(side)
    assert [p.name for p in pool.released] == ["side-image"]
    assert pool._tearing == 0 and pool.figures.tearing == 0
    assert pool._building == 0 and pool.figures.building == 0


def test_no_grant_takes_the_bytes_of_a_victim_still_torn_down(monkeypatch):
    pool = make_pool(monkeypatch, 10, retry=0)
    x = image(pool, "x", 4, busy=True)
    chat(pool, "y", 4, busy=0)
    seams = Seams("teardown")
    pool._seam = seams
    load = start("loader", lambda: image(pool, "z", 4))
    seams.reach("teardown")
    try:
        # The budget counts y until its teardown ends.
        assert pool.figures.tearing == 4 * GB
        assert not pool.upload_admit(GB, None)
        with pytest.raises(LoraRefused) as ei:
            turn(pool, x, 2 * GB)
    finally:
        seams.go["teardown"].set()
    finish(*load)
    assert "the server is loading z (4.0 GB) now" in ei.value.message
    assert x.lora.pending == 0 and pool._incoming == 0
    assert pool.figures.tearing == 0


def test_a_torn_image_counts_until_its_pipeline_is_released(monkeypatch):
    """A cold load evicts an idle image model, whose pipeline's release on
    the worker runs on past its teardown: the budget counts its bytes
    until the release ends, and no upload is admitted into them."""
    pool = make_pool(monkeypatch, 10)
    image(pool, "x", 3, busy=True)
    y = image(pool, "y", 4)
    fp, load, release = pool._image_fns
    at, go = threading.Event(), threading.Event()

    def slow(box):
        at.set()
        assert go.wait(WAIT)
        release(box)
    pool._image_fns = (fp, load, slow)
    loader = start("loader", lambda: image(pool, "z", 4))
    try:
        assert at.wait(WAIT), "no release started"
        assert not pool.entry_resident(y)
        assert pool.figures.tearing == 4 * GB
        # x, z's build and 2 GB fit only once y's pipeline is gone.
        assert not pool.upload_admit(2 * GB, None)
    finally:
        go.set()
    finish(*loader)
    assert pool.figures.tearing == 0 and pool._tearing == 0
    assert pool.upload_admit(2 * GB, None)


def test_the_budget_503_names_no_pinned_model_in_use(monkeypatch):
    """A pinned chat model streams while upload bodies in flight leave the
    budget short: its bytes stay once it is idle, so the 503 names no
    model and says to unload one or retry."""
    pool = make_pool(monkeypatch, 10)
    x = image(pool, "x", 4, busy=True)
    chat(pool, "qwen3", 4, pinned=True)
    assert pool.upload_admit(GB, None)
    w = waits(pool)
    with pytest.raises(LoraRefused) as ei:
        turn(pool, x, 2 * GB)
    assert (ei.value.status, ei.value.message) == (
        503, "the test's LoRA need 2.0 GB of memory, and 1.0 GB is free; "
        "unload a model or retry when the requests in flight finish")
    assert w.timeouts == []


@pytest.mark.parametrize("back", [True, False])
def test_a_turn_whose_build_lock_stays_held_past_its_deadline_says_so(
        monkeypatch, back):
    """A turn left short by the working set while an unload holds the
    build lock past the deadline: with the memory back by then, the 503
    says a load or unload held it up, and with it still short, it names
    the working set's room. Neither names a free figure over its need."""
    pool = make_pool(monkeypatch, 10, retry=0)
    x = image(pool, "x", 4, busy=True)
    heads = iter([GB // 2, 5 * GB if back else GB // 2])
    pool._headroom_fn = lambda: next(heads)
    with pool._build_lock:
        with pytest.raises(LoraRefused) as ei:
            turn(pool, x, GB)
    assert (ei.value.status, ei.value.message) == (
        503, "the test's LoRA need 1.0 GB of memory, and " + (
            "the server is loading or unloading another model now; retry "
            "when it is done" if back else
            "0.5 GB is free in the working set now; close other apps or "
            "unload a model to make room"))
    assert x.lora.pending == 0


def test_a_cold_load_after_a_turn_sees_its_booking(monkeypatch):
    pool = make_pool(monkeypatch, 10)
    x = image(pool, "x", 4, busy=True)
    turn(pool, x, 3 * GB)
    assert pool.figures.resident == 7 * GB
    idle = image(pool, "idle", 2)
    image(pool, "next", 2)
    # The 2 GB load fits beside 7 GB only by evicting the idle one.
    assert not pool.entry_resident(idle)


@pytest.mark.parametrize("case", ["turn", "repriced", "reaper"])
def test_no_lock_is_held_at_a_teardown_or_a_measure(monkeypatch, case):
    """No hold of the pool's lock spans a teardown or a measure: at a
    turn's eviction and measure and a clear, at a repriced build's
    eviction, and at the reaper's unload."""
    pool = make_pool(monkeypatch, 10)
    held = []
    pool._seam = lambda name: held.append(
        (name, pool._lock._is_owned())) if name in ("teardown",
                                                     "measure") else None
    idle = image(pool, "idle", 3)
    want = {"teardown"}
    if case == "turn":
        x = image(pool, "x", 4, busy=True)
        turn(pool, x, 4 * GB)
        image(pool, "z", 1)
        pool.clear()
        want.add("measure")
    elif case == "repriced":
        image(pool, "y", 4, busy=True)
        reprice(pool, "z", 6)
        z = pool.acquire("z", None, IMAGE_KIND, build_spec=spec("z", 3))
        assert z.footprint == 6 * GB
    else:
        pool._in_flight_fn = lambda: 0
        with pool._lock:
            idle.ttl, idle.last_access = 1, -1e9
        assert pool.reap_idle() == ["idle"]
    assert want <= {n for n, _ in held}
    assert not any(h for _, h in held), held
    assert not pool.entry_resident(idle)


def test_routes_read_the_figures_while_the_lock_is_held(monkeypatch):
    pool = make_pool(monkeypatch, 10)
    x = image(pool, "x", 4, busy=True)
    turn(pool, x, GB)
    out = {}
    with pool._lock:
        t, o = start("route", lambda: (pool.plan_beside(x),
                                       pool.lora_hopeless(x, GB, "s")))
        out["v"] = finish(t, o)
    assert out["v"] == (GB, None)


def test_a_store_turn_evicts_an_idle_model(monkeypatch):
    pool = make_pool(monkeypatch, 10)
    idle = image(pool, "idle", 4)
    image(pool, "x", 4)
    pool.store.keep("old", GB)
    n = 3 * GB
    assert not pool.upload_admit(n, None)
    on_worker(lambda: pool.lora_turn(STORE, n, subject="the upload"))
    assert pool.store.items == {}
    assert pool._incoming == n
    assert not pool.entry_resident(idle)


class StoreWorker:
    """The image worker of a store turn, a thread of its own, recording
    whether each cancel an upload tries takes its job off the queue."""

    def __init__(self, monkeypatch):
        from gmlx.serve import image_service
        from gmlx.serve.subservice import SingleWorker

        self.real = SingleWorker("image-worker-store")
        self.cancelled: list = []
        self.tried = threading.Event()
        monkeypatch.setattr(image_service, "WORKER", self)
        self.store_turn = image_service.store_turn

    def submit(self, fn, *a, **k):
        fut = self.real.submit(fn, *a, **k)
        cancel = fut.cancel

        def tried():
            got = cancel()
            self.cancelled.append(got)
            self.tried.set()
            return got

        fut.cancel = tried
        return fut

    def drain(self):
        self.real.submit(lambda: None).result(WAIT)
        self.real.get().shutdown(wait=True)


@pytest.mark.parametrize("short", ["neither", "measured", "budget"])
def test_a_store_turn_not_started_by_its_deadline_never_runs(monkeypatch,
                                                             short):
    """An upload whose store turn waits behind a step on the worker past
    the deadline: the queued turn is cancelled, and the upload gets the
    503 of what is short then, the budget or the working set, or with
    neither short, of the worker's job. Once the worker is free nothing is
    admitted for it."""
    pool = make_pool(monkeypatch, 10)
    image(pool, "x", 4, busy=True)
    n = 3 * GB
    if short == "measured":
        pool.head, n = GB // 2, GB
    elif short == "budget":
        pool.store.keep("old", 4 * GB)
    seams = Seams()
    pool._seam = seams
    w = StoreWorker(monkeypatch)
    go = threading.Event()
    w.submit(go.wait, WAIT)
    try:
        with pytest.raises(LoraRefused) as err:
            w.store_turn(pool, n, "the upload")
    finally:
        go.set()
        w.drain()
    assert (err.value.status, err.value.message, err.value.retry_after) == (
        503, {"neither": "the upload need 3.0 GB of memory, and the image "
                         "worker is busy with x; retry when it is idle",
              "measured": "the upload need 1.0 GB of memory, and 0.5 GB is "
                          "free in the working set now; retry when the "
                          "models in use finish",
              "budget": "the upload need 3.0 GB of memory, and 2.0 GB is "
                        "free beside the models in use (x); retry when one "
                        "is idle"}[short], 1)
    assert w.cancelled == [True]
    assert "turn" not in seams.hit and pool._incoming == 0


def test_a_store_turn_started_by_its_deadline_runs_to_its_end(monkeypatch):
    """An upload whose store turn started before the deadline and runs
    past it: the upload's cancel fails, it waits for that turn, and it is
    admitted, never answered 503 with its bytes admitted."""
    pool = make_pool(monkeypatch, 10)
    seams = Seams("turn")
    pool._seam = seams
    w = StoreWorker(monkeypatch)
    try:
        t, out = start("upload", lambda: w.store_turn(pool, 3 * GB,
                                                      "the upload"))
        seams.reach("turn")
        assert w.tried.wait(WAIT)
        seams.go["turn"].set()
        finish(t, out)
    finally:
        seams.go["turn"].set()
        w.drain()
    assert w.cancelled == [False]
    assert pool._incoming == 3 * GB


def test_an_upload_and_a_turn_never_both_take_the_last_room(monkeypatch):
    pool = make_pool(monkeypatch, 10)
    x = image(pool, "x", 4, busy=True)
    # The upload is admitted first, and the turn sees it.
    assert pool.upload_admit(4 * GB, None)
    with pytest.raises(LoraRefused):
        turn(pool, x, 4 * GB)
    assert x.lora.pending == 0
    # The turn books first, and the upload sees it.
    pool.store.put(pool, "u", 4 * GB)
    pool.store.shed(4 * GB)
    turn(pool, x, 4 * GB)
    assert not pool.upload_admit(4 * GB, None)
    assert pool._incoming == 0


def reprice(pool, name, fp_gb):
    """The build of ``name`` holds ``fp_gb`` once installed, more than its
    header priced, as a streamed install that streamed less does."""
    real = pool._build

    def build(*a, **k):
        e = real(*a, **k)
        if e.model_path == name:
            e.footprint = int(fp_gb * GB)
            e.priced_up = True
        return e
    pool._build = build


def test_a_repriced_build_evicts_once_for_its_new_price(monkeypatch):
    pool = make_pool(monkeypatch, 10)
    w = image(pool, "w", 1)
    image(pool, "y", 4, busy=True)
    reprice(pool, "z", 5)
    z = pool.acquire("z", None, IMAGE_KIND, build_spec=spec("z", 3))
    assert z.footprint == 5 * GB
    assert w.cache_key in pool._entries
    assert pool._resident_bytes() == 10 * GB and pool._building == 0


def test_a_build_that_fails_after_its_room_check_leaves_no_building(
        monkeypatch):
    from gmlx.serve.capacity import LoadDeferred

    pool = make_pool(monkeypatch, 10)
    image(pool, "y", 8, busy=True)
    real = pool._build
    reprice(pool, "z", 4)
    # The refit cannot fit 4 GB beside 8 GB in use, and defers.
    with pytest.raises(LoadDeferred):
        pool.acquire("z", None, IMAGE_KIND, build_spec=spec("z", 2))
    assert pool._building == 0 and pool._resident_bytes() == 8 * GB
    assert pool.figures.building == 0 and pool._loading is None

    # A build that fails between its refit and its insert.
    pool._build = real
    reprice(pool, "v", 1.5)
    released = len(pool.released)

    def fail(name):
        if name == "inserting":
            raise RuntimeError("the pipeline's caches cannot be read")
    pool._seam = fail
    with pytest.raises(RuntimeError, match="cannot be read"):
        pool.acquire("v", None, IMAGE_KIND, build_spec=spec("v", 1))
    assert pool._building == 0 and pool._resident_bytes() == 8 * GB
    assert len(pool.released) == released + 1


# Review 4's sizes: a 10 GB budget with 8 GB resident, a 2 GB cold load
# and a 1.5 GB turn.
def _race_pool(monkeypatch):
    pool = make_pool(monkeypatch, 10, retry=1)
    w = image(pool, "w", 2)
    y = image(pool, "y", 6, busy=True)
    return pool, w, y


def _turn(pool, y):
    return start("image-worker-test", lambda: pool.lora_turn(
        y, int(1.5 * GB), subject="the LoRA factors of this step"))


def test_a_turn_after_a_cold_loads_room_check_waits_on_the_build(
        monkeypatch):
    pool, w, y = _race_pool(monkeypatch)
    seams = Seams("roomed")
    pool._seam = seams
    load = start("load-z", lambda: pool.acquire(
        "z", None, IMAGE_KIND, build_spec=spec("z", 2)))
    seams.reach("roomed")
    t = _turn(pool, y)
    t[0].join(WAIT)
    assert not t[0].is_alive(), "the turn outwaited the paused build"
    seams.go["roomed"].set()
    finish(*load)
    with pytest.raises(LoraRefused) as e:
        finish(*t)
    assert e.value.status == 503
    assert "loading z (2.0 GB)" in e.value.message
    assert pool._resident_bytes() == 10 * GB


def test_a_turn_during_the_insert_sees_the_new_entry(monkeypatch):
    pool, w, y = _race_pool(monkeypatch)
    seams = Seams("insert")
    pool._seam = seams
    load = start("load-z", lambda: pool.acquire(
        "z", None, IMAGE_KIND, build_spec=spec("z", 2)))
    seams.reach("insert")
    t = _turn(pool, y)
    # The turn waits on _lock while the insert holds it.
    t[0].join(0.3)
    seams.go["insert"].set()
    finish(*load)
    finish(*t)
    # The turn found 10 GB held and evicted the idle model for its room.
    assert w.cache_key not in pool._entries
    assert y.lora.pending == int(1.5 * GB)
    assert pool._resident_bytes() == int(9.5 * GB)


def test_a_cold_load_after_a_turns_booking_sees_it(monkeypatch):
    pool, w, y = _race_pool(monkeypatch)
    seams = Seams("booked", "roomed")
    pool._seam = seams
    t = _turn(pool, y)
    seams.reach("booked")
    seams.go["roomed"].set()
    load = start("load-z", lambda: pool.acquire(
        "z", None, IMAGE_KIND, build_spec=spec("z", 2)))
    # The load's room check waits on _lock while the turn holds it.
    seams.at["roomed"].wait(0.3)
    seams.go["booked"].set()
    finish(*t)
    finish(*load)
    assert y.lora.pending == int(1.5 * GB)
    # The load evicted the idle model to fit beside the booking.
    assert w.cache_key not in pool._entries
    assert pool._resident_bytes() == int(9.5 * GB)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_a_random_sequence_keeps_the_four_invariants(monkeypatch, seed):
    rng = random.Random(seed)
    budget = 8 * GB
    pool = make_pool(monkeypatch, budget / GB, retry=0)
    outside = [0]

    def head():
        # The uploads are host memory the measure misses.
        held = (sum(e.footprint for e in pool._entries.values())
                + sum(sum(c.items.values()) + c.own
                      for c in pool.caches.values()))
        return budget + GB - held - outside[0]

    pool._headroom_fn = head

    def seam(name):
        if name in ("teardown", "measure"):
            assert not pool._lock._is_owned(), f"_lock held at {name}"
    pool._seam = seam
    names = ["a", "b", "c", "d"]
    counts = {"turns": 0, "refused": 0, "uploads": 0, "store": 0}

    def check(booked_owner=None):
        for e in pool._entries.values():
            if e.lora is None:
                continue
            keys, own = pool.caches[e.model_path].figures()
            if booked_owner is not e:
                assert e.lora.pending == 0, e.model_path
            assert e.lora.keys == keys, e.model_path
            assert e.lora.own == own
        f = pool.figures
        assert (f.footprints, f.lora) == live_sums(pool)
        assert f.entries == tuple(pool._entries.values())
        # Every teardown and pipeline release ran to its end.
        assert f.tearing == 0 and pool._tearing == 0

    def entry(name):
        return pool._entries.get((name, None, IMAGE_KIND))

    def run():
        for i in range(300):
            outside[0] = rng.choice([0, 0, GB, 4 * GB])
            op = rng.choice(["turn", "turn", "turn", "cancel", "upload",
                             "store", "load", "evict", "drop"])
            name = rng.choice(names)
            if op in ("turn", "cancel"):
                e = entry(name)
                if e is None:
                    continue
                with pool._lock:
                    e.busy += 1
                    pool._publish()
                need = rng.randrange(1, 6) * GB // 2
                key = f"{name}{i}"
                c = pool.caches[name]
                hold = tuple(rng.sample(sorted(c.items), min(1, len(c.items))))
                try:
                    pool.lora_turn(e, need, subject="s", hold=hold)
                    counts["turns"] += 1
                    resident = (pool._resident_bytes() + pool.store.n
                                + pool._incoming)
                    assert resident <= budget, (i, resident)
                    check(booked_owner=e)
                    if op == "turn":
                        c.items[key] = need
                except LoraRefused:
                    counts["refused"] += 1
                finally:
                    pool.lora_settle(e, "s")
                    pool.release(e)
                check()
            elif op == "upload":
                n = rng.randrange(1, 4) * GB // 2
                if pool.upload_admit(n, head()):
                    pool.store.put(pool, f"u{i}", n)
                    counts["uploads"] += 1
            elif op == "store":
                n = rng.randrange(1, 4) * GB // 2
                try:
                    pool.lora_turn(STORE, n, subject="u")
                    pool.store.put(pool, f"s{i}", n)
                    counts["store"] += 1
                except LoraRefused:
                    pass
            elif op == "load":
                if entry(name) is None:
                    image(pool, name, rng.randrange(2, 5))
            elif op == "evict":
                pool.evict(name)
            elif op == "drop":
                c = pool.caches.get(name)
                if c and c.items:
                    c.items.pop(rng.choice(sorted(c.items)))
                    e = entry(name)
                    if e is not None:
                        pool.lora_settle(e, "s")
            check()
            if pool.store.n and rng.random() < 0.2:
                pool.store.shed(GB)
    on_worker(run)
    assert counts["turns"] > 20 and counts["refused"] > 0, counts


# The caches a ledger is set from (image_service).
def _lru_pipe(default="dflt"):
    from gmlx.image.conditioning import ByteLRU

    return SimpleNamespace(adapters=ByteLRU(1 << 40), tables=ByteLRU(1 << 40),
                           default_table=default, lora=None,
                           patch_tables={})


def test_the_ledger_counts_request_keys_and_the_own_bytes_apart():
    from gmlx.image.lora import install
    from gmlx.serve.image_service import lora_figures

    pipe = _lru_pipe()
    pipe.adapters.keep({"own1"})
    pipe.adapters.put("own1", object(), 100, shed=False)
    pipe.adapters.put("req", object(), 50, shed=False)
    k_own = (install.KERNEL, "p", (("own1", None, 1.0),), "f32")
    k_req = (install.KERNEL, "p", (("req", None, 1.0),), "f32")
    k_mix = (install.KERNEL, "q", (("own1", None, 1.0), ("req", None, 1.0)),
             "f32")
    pipe.tables.put(("dflt", "f32", "kernel"), object(), 30, shed=False)
    pipe.tables.put(k_own, object(), 20, shed=False)
    pipe.tables.put(k_req, object(), 10, shed=False)
    pipe.tables.put(k_mix, object(), 5, shed=False)
    pipe.tables.put(("other", "f32", "kernel"), object(), 40, shed=False)
    keys, own = lora_figures(SimpleNamespace(pipeline=SimpleNamespace(
        pipe=pipe)))
    assert keys == {"req": 50, k_req: 10, k_mix: 5,
                    ("other", "f32", "kernel"): 40}
    assert own == 150


def test_a_shed_returns_the_bytes_it_let_go():
    from gmlx.serve.image_service import shed_lora

    pipe = _lru_pipe()
    pipe.adapters.put("a", object(), 50, shed=False)
    pipe.adapters.put("b", object(), 60, shed=False)
    e = SimpleNamespace(pipeline=SimpleNamespace(pipe=pipe))
    assert shed_lora(e, 10, hold=("a",), own=True) == 60
    assert pipe.adapters.sizes() == {"a": 50}
    assert shed_lora(e, 10) == 50


def test_a_pipeline_release_on_the_worker_runs_inline():
    from gmlx.serve import image_service

    seen = finish(*start("image-worker-test", lambda: image_service._on_worker(
        threading.current_thread)))
    assert seen.name == "image-worker-test"


def test_an_upload_fits_the_budget_and_the_measure_beside_the_store(
        monkeypatch):
    from gmlx.serve.image_steps import _Bytes

    pool = make_pool(monkeypatch, 10)
    image(pool, "x", 4, busy=True)
    store = _Bytes(None, "the step routes' adapter cache", "adapter_cache_mb")
    store.put("old", SimpleNamespace(nbytes=GB))
    pool._store_live = lambda: store.bytes

    def admit(n, head):
        return pool.upload_admit(int(n), head)

    # 6 GB the budget leaves, of which the store holds 1.
    assert not admit(5.5 * GB, None) and pool._incoming == 0
    assert admit(5 * GB, None)
    pool.upload_done(5 * GB)
    # A measure of 3 GB bounds the store, which it does not see.
    assert not admit(2.5 * GB, 3 * GB)
    assert admit(2 * GB, 3 * GB)
    pool.upload_done(2 * GB)
    # A booking the measure does not see yet counts too.
    x = pool._entries[("x", None, IMAGE_KIND)]
    turn(pool, x, GB)
    assert not admit(1.5 * GB, 3 * GB)
    assert admit(GB, 3 * GB)
    assert store.bytes + pool._incoming == 2 * GB


def test_the_cap_shed_at_a_put_skips_the_held_keys():
    """A put past the store's configured cap drops the oldest tensors on
    the putting thread: the keys a define or a step holds stay, and the
    next oldest go in their place."""
    from gmlx.serve.image_steps import _Bytes

    store = _Bytes(3 * GB, "the step routes' adapter cache",
                   "adapter_cache_mb")
    for k in "abc":
        store.put(k, SimpleNamespace(nbytes=GB))
    with store.holding(["a"]):
        store.put("d", SimpleNamespace(nbytes=GB))
        assert list(store._items) == ["a", "c", "d"]
    assert store.bytes == 3 * GB


def _recorded_holds(monkeypatch, pool):
    """The pool's own worker reservations, each recorded by kind in the
    order taken, and held while its block runs."""
    from gmlx.serve import image_service, video_service

    del pool._worker_for
    order: list = []
    held: set = set()

    def fake(kind):
        @contextlib.contextmanager
        def reserved():
            order.append(kind)
            held.add(kind)
            try:
                yield
            finally:
                held.discard(kind)
        return reserved

    monkeypatch.setattr(image_service, "reserved_worker", fake(IMAGE_KIND))
    monkeypatch.setattr(video_service, "reserved_worker", fake(VIDEO_KIND))
    monkeypatch.setattr(residency_workers, "_holds_image_worker",
                        lambda: IMAGE_KIND in held)
    monkeypatch.setattr(residency_workers, "_holds_video_worker",
                        lambda: VIDEO_KIND in held)
    return order


def test_a_cold_load_holds_only_the_workers_it_loads_or_releases_on(
        monkeypatch):
    """An image load that evicts an idle chat model holds the image
    worker it builds on, and not the video worker of the idle video
    entry it leaves resident."""
    pool = make_pool(monkeypatch, 9)
    old = chat(pool, "old", 3, busy=0)
    clip = video(pool, "clip", 3)
    order = _recorded_holds(monkeypatch, pool)
    image(pool, "zimage", 4)
    assert order == [IMAGE_KIND]
    assert not pool.entry_resident(old)
    assert pool.entry_resident(clip)


def test_a_reaper_tick_holds_only_the_workers_of_what_it_reaps(
        monkeypatch):
    """An idle image entry under its TTL, beside a running video part:
    the tick holds no worker, so it waits behind no job. Past the TTL it
    holds the image worker alone, and reaps the entry."""
    pool = make_pool(monkeypatch, 10)
    idle = image(pool, "idle", 3)
    clip = video(pool, "clip", 3, busy=True)
    pool._in_flight_fn = lambda: 0
    order = _recorded_holds(monkeypatch, pool)
    with pool._lock:
        idle.ttl = clip.ttl = 100
    assert pool.reap_idle() == []
    assert order == []
    with pool._lock:
        idle.last_access -= 1000
    assert pool.reap_idle() == ["idle"]
    assert order == [IMAGE_KIND]
    assert pool.entry_resident(clip)


def test_a_cold_load_that_evicts_on_both_workers_holds_image_then_video(
        monkeypatch):
    """The one order every thread takes the workers in: a load whose
    eviction releases an image and a video pipeline holds both, image
    first, and releases each on its own worker."""
    pool = make_pool(monkeypatch, 10)
    image(pool, "zimage", 3)
    video(pool, "clip", 3)
    order = _recorded_holds(monkeypatch, pool)
    chat_load = spec("big", 8)
    e = pool.acquire("big", None, VIDEO_KIND, build_spec=chat_load)
    pool.release(e)
    assert order == [IMAGE_KIND, VIDEO_KIND]
    assert sorted(p.name for p in pool.released) == ["clip", "zimage"]


def test_a_cold_load_counts_a_victims_lora_bytes_when_it_picks_workers(
        monkeypatch):
    """An idle image entry holds 3 GB of LoRA factors beside its 2 GB of
    weights. Its eviction makes the room of an image load, so the load
    holds the image worker alone, and not the video worker of the idle
    clip it leaves resident."""
    pool = make_pool(monkeypatch, 10)
    zimage = image(pool, "zimage", 2, busy=True)
    job(pool, zimage, "z1", 3 * GB)
    pool.release(zimage)
    clip = video(pool, "clip", 3)
    order = _recorded_holds(monkeypatch, pool)
    pool.release(pool.acquire("new", None, IMAGE_KIND,
                              build_spec=spec("new", 5)))
    assert order == [IMAGE_KIND]
    assert not pool.entry_resident(zimage) and pool.entry_resident(clip)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_two_workers_turn_at_once_and_never_release_on_each_other(
        monkeypatch, seed):
    """The image and the video worker run turns at once while another
    thread loads and unloads entries of both kinds. No turn releases a
    pipeline of the other worker's kind, none reserves a worker, and each
    turn's booking fits the budget."""
    budget = 10 * GB
    pool = make_pool(monkeypatch, budget / GB, retry=0)
    kinds: dict = {}
    reserved: list = []

    @contextlib.contextmanager
    def record(ks):
        reserved.append(threading.current_thread().name)
        yield

    # The pool's own reservations, recorded by the thread that takes them.
    del pool._worker_for
    monkeypatch.setattr(residency_workers, "_reserved_workers", record)
    stop = threading.Event()
    errors: list = []

    def entries(kind):
        with pool._lock:
            return [e for e in pool._entries.values() if e.kind == kind]

    def worker(kind):
        rng = random.Random(seed * 10 + len(kind))
        try:
            for i in range(150):
                es = entries(kind)
                if not es:
                    continue
                e = rng.choice(es)
                with pool._lock:
                    if not pool.entry_resident(e):
                        continue
                    e.busy += 1
                try:
                    need = rng.randrange(1, 5) * GB // 2
                    pool.lora_turn(e, need, subject="s")
                    with pool._lock:
                        resident = (pool._resident_bytes() + pool.store.n
                                    + pool._incoming)
                    assert resident <= budget, (kind, i, resident)
                    pool.caches.setdefault(e.model_path, Cache()).items[
                        f"{kind}{i}"] = need
                except LoraRefused:
                    pass
                finally:
                    pool.lora_settle(e, "s")
                    pool.release(e)
        except BaseException as err:  # noqa: BLE001 - reported below
            errors.append(err)

    def loads():
        rng = random.Random(seed)
        n = 0
        try:
            while not stop.is_set():
                kind = rng.choice([IMAGE_KIND, VIDEO_KIND])
                name = f"{kind[5]}{n}"
                n += 1
                kinds[name] = kind
                make = image if kind == IMAGE_KIND else video
                make(pool, name, rng.randrange(1, 4))
                if rng.random() < 0.3:
                    es = entries(rng.choice([IMAGE_KIND, VIDEO_KIND]))
                    if es:
                        try:
                            pool.evict(rng.choice(es).model_path)
                        except residency.ModelBusyError:
                            pass
        except BaseException as err:  # noqa: BLE001 - reported below
            errors.append(err)

    loader = threading.Thread(target=loads, daemon=True)
    loader.start()
    a = start("image-worker-test", lambda: worker(IMAGE_KIND))
    b = start("video-worker-test", lambda: worker(VIDEO_KIND))
    finish(*a)
    finish(*b)
    stop.set()
    loader.join(WAIT)
    assert not loader.is_alive(), "the loader is stuck"
    assert not errors, errors
    assert reserved and not [r for r in reserved if "worker" in r]
    for name, thread in pool.release_threads.items():
        want = "video-worker" if kinds[name] == VIDEO_KIND else "image-worker"
        assert "worker" not in thread or thread.startswith(want), (name,
                                                                   thread)
