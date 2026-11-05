"""The residency pool's LoRA turn: a worker job books the LoRA bytes it
allocates next under the pool's lock, sheds and evicts as a cold load
does, and waits only for chat models in use (residency). The pool here
holds image entries whose pipelines are stand-ins, so no model loads;
their caches are dicts of keys and bytes."""

import contextlib
import random
import threading
import time
from types import SimpleNamespace

import pytest

from gmlx.serve import residency
from gmlx.serve.residency import (IMAGE_KIND, STORE, LoraRefused,
                                  _ResidencyPool)

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
    """The upload store as a turn and an upload see it."""

    def __init__(self):
        self.items: dict = {}
        self.incoming = 0
        self.lock = threading.Lock()

    def live(self):
        with self.lock:
            return sum(self.items.values()) + self.incoming

    def shed(self, nbytes, hold=()):
        dropped = 0
        with self.lock:
            for k in list(self.items):
                if dropped >= nbytes:
                    break
                if k not in hold:
                    dropped += self.items.pop(k)
        return dropped

    def admit(self, pool, n, head):
        """Count ``n`` first, then read the pool; undo when over."""
        with self.lock:
            self.incoming += n
        room = pool.upload_room(head)
        if room is None or room >= self.live():
            return True
        with self.lock:
            self.incoming -= n
        return False

    def put(self, key, n):
        with self.lock:
            self.incoming -= n
            self.items[key] = n


def make_pool(monkeypatch, budget_gb, retry=1):
    from gmlx.serve import capacity as cap

    monkeypatch.setattr(cap, "preload_gate", lambda *a, **k: None)
    monkeypatch.delenv("GMLX_OVERCOMMIT", raising=False)
    monkeypatch.setattr(residency, "_holds_image_worker", lambda: True)
    released = []

    def release(box):
        pipe = box.pop()
        released.append(pipe)
        pool.caches.pop(pipe.name, None)

    pool = _ResidencyPool(
        SimpleNamespace(_last_entry=None), None, None, int(budget_gb * GB),
        (), image_fns=(lambda spec: spec.fp, lambda spec: spec.pipe,
                       release))
    pool.released = released
    pool._teardown_chat = lambda e: released.append(e.model_path)
    pool._worker_for = lambda *a, **k: contextlib.nullcontext()
    pool.head = None
    pool._headroom_fn = lambda: pool.head
    pool.caches = {}
    pool.store = Store()
    pool._lora_figures = lambda e: pool.caches.setdefault(
        e.model_path, Cache()).figures()
    pool._lora_shed = lambda e, n, hold, own: \
        pool.caches[e.model_path].shed(n, hold)
    pool._store_shed = lambda n, hold=(): pool.store.shed(n, hold)
    pool._store_live = pool.store.live
    pool._retry_fn = lambda: retry
    return pool


def spec(name, fp_gb):
    return SimpleNamespace(fp=int(fp_gb * GB), stream=None, path=name,
                           pipe=SimpleNamespace(name=name))


def image(pool, name, fp_gb, *, busy=False):
    e = pool.acquire(name, None, IMAGE_KIND, build_spec=spec(name, fp_gb))
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


def on_worker(fn):
    """``fn`` on a thread named as the image worker's are."""
    return finish(*start("image-worker-test", fn))


def turn(pool, e, need, **kw):
    kw.setdefault("subject", "the test's LoRA")
    return on_worker(lambda: pool.lora_turn(e, int(need), **kw))


def job(pool, e, key, need, **kw):
    """One job: its turn, its put, its settle."""
    def run():
        try:
            pool.lora_turn(e, int(need), subject="the test's LoRA", **kw)
            pool.caches[e.model_path].items[key] = int(need)
        finally:
            pool.lora_settle(e, "the test's LoRA")
    on_worker(run)


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


def test_a_settle_past_the_booking_logs_an_overrun(monkeypatch):
    pool = make_pool(monkeypatch, 10)
    got = []
    pool.overrun_hook = lambda subject, n: got.append(n)
    e = image(pool, "x", 2, busy=True)

    def run():
        pool.lora_turn(e, GB, subject="s")
        pool.caches["x"].items["a"] = 2 * GB
        pool.lora_settle(e, "s")
    on_worker(run)
    assert got == [GB]


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
    assert pool.figures.pending == 0


def test_a_turn_sheds_its_own_entry_then_others_then_the_uploads(
        monkeypatch):
    pool = make_pool(monkeypatch, 10)
    y = image(pool, "y", 2)
    x = image(pool, "x", 2, busy=True)
    idle = image(pool, "idle", 1)
    job(pool, y, "y1", GB)
    job(pool, x, "x1", GB)
    job(pool, x, "mine", GB)
    pool.store.items["u1"] = GB
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
    pool.store.items["u1"] = GB
    # 9 GB held of 10: a 4 GB turn sheds the two idle 1 GB keys and then
    # evicts the idle model.
    turn(pool, x, 4 * GB)
    assert pool.caches["x"].items == {}
    assert pool.store.items == {}
    assert not pool.entry_resident(idle)
    assert len(pool.released) == 1
    assert x.lora.pending == 4 * GB
    assert pool._resident_bytes() + pool.store.live() <= 10 * GB


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
    t, out = start("image-worker-test",
                   lambda: pool.lora_turn(x, 3 * GB, subject="s"))
    time.sleep(0.3)
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
    t0 = time.monotonic()
    with pytest.raises(LoraRefused) as ei:
        turn(pool, x, 3 * GB)
    took = time.monotonic() - t0
    assert 0.9 < took < 3
    assert ei.value.status == 503 and ei.value.retry_after == 1
    assert ("need 3.0 GB of memory, and 2.0 GB is free beside the models "
            "in use (c); retry when one is idle" in ei.value.message)


def test_no_wait_on_a_worker_entry_in_use_or_a_preload(monkeypatch):
    pool = make_pool(monkeypatch, 10, retry=5)
    x = image(pool, "x", 4, busy=True)
    image(pool, "y", 4, busy=True)
    t0 = time.monotonic()
    with pytest.raises(LoraRefused) as ei:
        turn(pool, x, 3 * GB)
    assert time.monotonic() - t0 < 1
    assert "models in use (y)" in ei.value.message
    pool2 = make_pool(monkeypatch, 10, retry=5)
    x2 = image(pool2, "x", 4, busy=True)
    chat(pool2, "pre", 4, busy=1, retained=1)
    t0 = time.monotonic()
    with pytest.raises(LoraRefused) as ei:
        turn(pool2, x2, 3 * GB)
    assert time.monotonic() - t0 < 1
    assert "unload a model or retry" in ei.value.message


def test_a_measured_short_turn_refuses_at_once_with_no_chat_in_use(
        monkeypatch):
    pool = make_pool(monkeypatch, 10, retry=5)
    x = image(pool, "x", 2, busy=True)
    pool.head = 2 * GB
    with pytest.raises(LoraRefused) as ei:
        turn(pool, x, 3 * GB)
    assert ("2.0 GB is free in the working set now; close other apps"
            in ei.value.message)


def test_a_turn_counts_the_other_pending_bookings(monkeypatch):
    pool = make_pool(monkeypatch, 10)
    x = image(pool, "x", 2, busy=True)
    y = image(pool, "y", 1, busy=True)
    pool.head = 2 * GB
    turn(pool, x, int(1.5 * GB))
    with pytest.raises(LoraRefused) as ei:
        turn(pool, y, int(1.5 * GB))
    assert ei.value.status == 503


def test_overcommit_sheds_and_never_refuses(monkeypatch):
    pool = make_pool(monkeypatch, 4)
    monkeypatch.setenv("GMLX_OVERCOMMIT", "1")
    x = image(pool, "x", 2, busy=True)
    job(pool, x, "old", GB)
    turn(pool, x, 3 * GB)
    assert pool.caches["x"].items == {}
    assert x.lora.pending == 3 * GB


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


def test_a_cold_load_after_a_turn_sees_its_booking(monkeypatch):
    pool = make_pool(monkeypatch, 10)
    x = image(pool, "x", 4, busy=True)
    turn(pool, x, 3 * GB)
    assert pool.figures.resident == 7 * GB
    idle = image(pool, "idle", 2)
    image(pool, "next", 2)
    # The 2 GB load fits beside 7 GB only by evicting the idle one.
    assert not pool.entry_resident(idle)


def test_no_lock_is_held_at_a_teardown_or_a_measure(monkeypatch):
    pool = make_pool(monkeypatch, 10)
    held = []
    pool._seam = lambda name: held.append(
        (name, pool._lock._is_owned())) if name in ("teardown",
                                                     "measure") else None
    idle = image(pool, "idle", 3)
    x = image(pool, "x", 4, busy=True)
    turn(pool, x, 4 * GB)
    image(pool, "z", 1)
    pool.clear()
    assert {n for n, _ in held} == {"teardown", "measure"}
    assert not any(h for _, h in held), held
    assert not pool.entry_resident(idle)


def test_routes_read_the_figures_while_the_lock_is_held(monkeypatch):
    pool = make_pool(monkeypatch, 10)
    x = image(pool, "x", 4, busy=True)
    turn(pool, x, GB)
    out = {}
    with pool._lock:
        t, o = start("route", lambda: (pool.plan_beside(x),
                                       pool.upload_room(None),
                                       pool.lora_hopeless(x, GB, "s")))
        out["v"] = finish(t, o)
    assert out["v"][:2] == (GB, 5 * GB)


def test_a_store_turn_evicts_an_idle_model(monkeypatch):
    pool = make_pool(monkeypatch, 10)
    idle = image(pool, "idle", 4)
    image(pool, "x", 4)
    pool.store.items["old"] = GB
    n = 3 * GB
    assert not pool.store.admit(pool, n, None)

    def run():
        pool.lora_turn(STORE, n, subject="the upload",
                       admit=lambda head: pool.store.admit(pool, n, head))
    on_worker(run)
    assert pool.store.items == {}
    assert pool.store.incoming == n
    assert not pool.entry_resident(idle)


def test_an_upload_and_a_turn_never_both_take_the_last_room(monkeypatch):
    pool = make_pool(monkeypatch, 10)
    x = image(pool, "x", 4, busy=True)
    # The upload counts itself, then a turn checks against it.
    assert pool.store.admit(pool, 4 * GB, None)
    with pytest.raises(LoraRefused):
        turn(pool, x, 4 * GB)
    assert x.lora.pending == 0
    # The turn books first, and the upload sees it.
    pool.store.put("u", 4 * GB)
    pool.store.items.clear()
    turn(pool, x, 4 * GB)
    assert not pool.store.admit(pool, 4 * GB, None)
    assert pool.store.incoming == 0


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


def test_an_upload_admitted_as_a_turn_books_takes_the_room_from_it(
        monkeypatch):
    pool = make_pool(monkeypatch, 10)
    x = image(pool, "x", 4, busy=True)
    got = []

    def seam(name):
        # The upload reads the figures from before the turn's booking.
        if name == "booked" and not got:
            got.append(pool.store.admit(pool, 4 * GB, None))

    pool._seam = seam
    with pytest.raises(LoraRefused) as ei:
        turn(pool, x, 4 * GB)
    assert got == [True] and ei.value.status == 503
    assert x.lora.pending == 0 and pool.figures.pending == 0
    assert pool._resident_bytes() + pool.store.live() <= 10 * GB


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
    assert pool.figures.building == 0 and pool.figures.loading is None

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
                    resident = pool._resident_bytes() + pool.store.live()
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
                if pool.store.admit(pool, n, head()):
                    pool.store.put(f"u{i}", n)
                    counts["uploads"] += 1
            elif op == "store":
                n = rng.randrange(1, 4) * GB // 2
                try:
                    pool.lora_turn(
                        STORE, n, subject="u",
                        admit=lambda h, n=n: pool.store.admit(pool, n, h))
                    pool.store.put(f"s{i}", n)
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
            if pool.store.live() and rng.random() < 0.2:
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

    def admit(n, head):
        return store.admit(int(n), lambda: pool.upload_room(head))

    # 6 GB the budget leaves, of which the store holds 1.
    assert not admit(5.5 * GB, None) and store.incoming == 0
    assert admit(5 * GB, None)
    store.done(5 * GB)
    # A measure of 3 GB bounds the store, which it does not see.
    assert not admit(2.5 * GB, 3 * GB)
    assert admit(2 * GB, 3 * GB)
    store.done(2 * GB)
    # A booking the measure does not see yet counts too.
    x = pool._entries[("x", None, IMAGE_KIND)]
    turn(pool, x, GB)
    assert not admit(1.5 * GB, 3 * GB)
    assert admit(GB, 3 * GB)
    assert store.live() == 2 * GB
