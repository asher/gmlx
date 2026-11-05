"""The residency pool's books of LoRA bytes: the grants, the claims, the
shrink queue and the cold build in flight (lora_books, residency). The
pool here holds image entries whose pipelines are stand-ins, so no model
loads; the LoRA bytes are booked through the pool's own calls."""

import asyncio
import concurrent.futures
import contextlib
import gc
import threading
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from gmlx.image.conditioning import ByteLRU
from gmlx.serve import residency
from gmlx.serve.capacity import LoadDeferred
from gmlx.serve.lora_books import STORE, LoraRefused
from gmlx.serve.residency import IMAGE_KIND, _BusyHold, _ResidencyPool

GB = 10**9
# Seconds a pool thread may take before the test calls it stuck.
WAIT = 10


def make_pool(monkeypatch, budget_gb):
    from gmlx.serve import capacity as cap

    monkeypatch.setattr(cap, "preload_gate", lambda *a, **k: None)
    monkeypatch.delenv("GMLX_OVERCOMMIT", raising=False)
    monkeypatch.setattr(residency, "_holds_image_worker", lambda: True)
    released = []
    pool = _ResidencyPool(
        SimpleNamespace(_last_entry=None), None, None, int(budget_gb * GB),
        (), image_fns=(lambda spec: spec.fp, lambda spec: spec.pipe,
                       lambda box: released.append(box.pop())))
    pool.released = released
    pool._worker_for = lambda *a, **k: contextlib.nullcontext()
    pool.head = None
    pool._headroom_fn = lambda: pool.head
    pool.caches = {}
    pool._lora_shed = lambda e, n, hold, own: pool.caches[id(e)].shed(n, hold)
    pool._store_shed = lambda n, hold=(): pool.caches[id(STORE)].shed(n, hold)
    return pool


def spec(name, fp_gb, pipe=None):
    return SimpleNamespace(fp=int(fp_gb * GB), stream=None, path=name,
                           pipe=pipe if pipe is not None else
                           SimpleNamespace(name=name))


def image(pool, name, fp_gb, *, busy=False, pipe=None):
    e = pool.acquire(name, None, IMAGE_KIND, build_spec=spec(name, fp_gb, pipe))
    if not busy:
        pool.release(e)
    return e


class Cache:
    """An owner's LoRA cache as the books see it: its keys and their bytes.
    A shed lets the oldest keys go and posts them through the pool's
    hook, as the pipe caches do."""

    def __init__(self, pool, owner):
        self.items: dict = {}
        self.gone = pool.books.release_hook(owner)
        pool.caches[id(owner)] = self

    def shed(self, nbytes, hold=()):
        out = []
        for k in list(self.items):
            if nbytes <= 0:
                break
            if k not in hold:
                nbytes -= self.items.pop(k)
                out.append(k)
        self.gone(out)

    def drop(self, *keys):
        for k in keys:
            self.items.pop(k)
        self.gone(keys)


def claim_and_check(pool, booking):
    pool.shed_claims(booking)
    pool.release_claims(booking)


def put(pool, booking, key, nbytes):
    """A job's put: the key enters from its booking, then the cache holds it."""
    booking.enter(key, nbytes)
    pool.caches[id(booking.owner)].items[key] = nbytes


def book(pool, owner, key, nbytes):
    """One job that reserves, puts ``key`` and settles."""
    b = pool.reserve_lora(owner, int(nbytes), subject="the test's LoRA")
    claim_and_check(pool, b)
    put(pool, b, key, int(nbytes))
    pool.settle_lora(b, {key: int(nbytes)})
    return b


def ledger(pool, owner):
    return pool.lora_held(owner)[2]


def resident(pool):
    return pool._resident_bytes()


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
    return out["value"]


def on_worker(fn):
    """``fn`` on a thread named as the image worker's are."""
    return finish(*start("image-worker-test", fn))


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


# T7
def test_a_reservation_runs_on_a_request_thread_and_a_turn_on_the_worker(
        monkeypatch):
    pool = make_pool(monkeypatch, 10)
    y = image(pool, "y", 4, busy=True)
    with pytest.raises(RuntimeError, match="never on the image worker"):
        on_worker(lambda: pool.reserve_lora(y, GB, subject="s"))

    async def from_the_loop():
        pool.reserve_lora(y, GB, subject="s")
    with pytest.raises(RuntimeError, match="never on the event loop"):
        asyncio.run(from_the_loop())
    with pytest.raises(RuntimeError, match="runs on the image worker"):
        pool.turn_lora(y, None, GB, subject="s")
    assert pool.lora_held(y) == (0, 0, {})

    b = pool.reserve_lora(y, GB, subject="s")
    assert on_worker(lambda: pool.turn_lora(y, b, 2 * GB, subject="s")) is b
    assert pool.lora_held(y)[1] == 2 * GB


# T10, the books' cases; the caches' own are in the install tests.
def test_the_ledger_holds_the_keys_the_caches_hold(monkeypatch):
    pool = make_pool(monkeypatch, 20)
    y = image(pool, "y", 4, busy=True)
    c = Cache(pool, y)

    def books_equal_the_cache():
        held, pending, led = pool.lora_held(y)
        assert led == c.items
        assert held == sum(c.items.values()) and pending == 0
        assert resident(pool) == 4 * GB + held <= 20 * GB

    book(pool, y, "p1", GB)
    book(pool, y, "p2", 2 * GB)
    books_equal_the_cache()
    # A shed lowers held by the shed keys' bytes.
    c.shed(GB)
    assert "p1" not in c.items
    books_equal_the_cache()

    # A key shed right after its put leaves nothing booked.
    b = pool.reserve_lora(y, GB, subject="s")
    put(pool, b, "p3", GB)
    c.drop("p3")
    pool.settle_lora(b, {})
    books_equal_the_cache()

    # A held key put again by a later job is booked once, before and
    # after its settle.
    b = pool.reserve_lora(y, GB, subject="s")
    put(pool, b, "p2", 2 * GB)
    put(pool, b, "p5", GB)
    assert ledger(pool, y) == {"p2": 2 * GB, "p5": GB}
    pool.settle_lora(b, {"p2": 2 * GB, "p5": GB})
    books_equal_the_cache()


def test_an_unload_drops_the_ledger_and_the_claims_on_it(monkeypatch):
    pool = make_pool(monkeypatch, 10)
    x = image(pool, "x", 4)
    y = image(pool, "y", 4, busy=True)
    Cache(pool, x)
    Cache(pool, y)
    book(pool, x, "x1", GB)
    b = pool.reserve_lora(y, 2 * GB, subject="s")
    assert b.claims_on() == [(x, GB)]
    assert pool.evict("x")
    with pool._lock:
        assert pool.books.rec(x) is None
        assert pool.books.claimed() == 0
    assert b.claims_on() == []
    assert resident(pool) == 6 * GB


# T12
def test_two_reservations_share_the_measured_room(monkeypatch):
    pool = make_pool(monkeypatch, 10)
    y = image(pool, "y", 5, busy=True)
    # 5 GB free by the pool's figures, 2 GB measured.
    pool.head = 2 * GB
    first = pool.reserve_lora(y, int(1.5 * GB), subject="s")
    with pytest.raises(LoraRefused) as e:
        pool.reserve_lora(y, int(1.5 * GB),
                          subject="the LoRA factors of this step")
    assert e.value.status == 503
    assert e.value.message == (
        "the LoRA factors of this step need 1.5 GB of memory, and 0.5 GB is "
        "free in the working set now; retry when the models in use finish")
    assert pool.lora_held(y)[1] == int(1.5 * GB)

    # The headroom fell after the reserve: the turn takes what the job
    # has yet to allocate, against the room now.
    pool.head = GB
    with pytest.raises(LoraRefused) as e:
        on_worker(lambda: pool.turn_lora(
            y, first, int(1.5 * GB), subject="the LoRA factors of this step"))
    assert e.value.status == 503
    assert "and 1.0 GB is free in the working set now" in e.value.message


def _measured_as_resident(pool, outside: int = 0) -> None:
    """The working set measured as the pool's residents fill it, with
    ``outside`` bytes held by other processes."""
    pool._headroom_fn = (lambda: pool._budget - pool._resident_bytes()
                         - outside)


def test_a_route_eviction_under_a_measured_headroom_credits_its_victim(
        monkeypatch):
    """A 10 GB pool, the owner at 3 GB, an idle model at 6 GB, the working
    set measured as the residents fill it: the idle model holds the room a
    2 GB request needs by both figures. The route evicts it and grants the
    request, its measured room crediting what the model it chose frees,
    as the budget does. The turn measures again before the forward."""
    pool = make_pool(monkeypatch, 10)
    own = image(pool, "own", 3)
    idle = image(pool, "idle", 6)
    _measured_as_resident(pool)
    b = pool.reserve_lora(own, 2 * GB, subject="the request's adapters")
    assert b.granted == 2 * GB
    assert idle not in pool._entries.values()
    assert resident(pool) == 5 * GB <= pool._budget
    assert on_worker(lambda: pool.turn_lora(
        own, b, 2 * GB, subject="the request's adapters")) is b
    b.cancel()


def test_a_measured_room_short_after_the_credit_names_no_busy_model(
        monkeypatch):
    """The same eviction with 3 GB of the working set held outside the
    pool: crediting the idle model leaves a 6 GB request short, so it is
    refused 503 and the idle model stays. No model is in use, so the text
    says what frees memory, not to wait for one."""
    pool = make_pool(monkeypatch, 10)
    own = image(pool, "own", 3)
    idle = image(pool, "idle", 2)
    _measured_as_resident(pool, 3 * GB)
    with pytest.raises(LoraRefused) as e:
        pool.reserve_lora(own, 6 * GB, subject="the request's adapters")
    assert e.value.status == 503
    assert e.value.message == (
        "the request's adapters need 6.0 GB of memory, and 2.0 GB is free in "
        "the working set now; close other apps or unload a model to make "
        "room")
    assert idle in pool._entries.values()
    assert pool.lora_held(own) == (0, 0, {})


def test_an_upload_short_beside_no_model_in_use_says_what_frees_memory(
        monkeypatch):
    """An upload the budget is short for, beside a pinned model and a
    request's pending booking, with no idle model to name and no model in
    use: the 503 says to unload a model or retry, never "models in use"."""
    pool = make_pool(monkeypatch, 10)
    pinned = image(pool, "pinned", 5)
    pinned.pinned = True
    other = pool.reserve_lora(pinned, 3 * GB, subject="s")
    with pytest.raises(LoraRefused) as e:
        pool.reserve_lora(STORE, 3 * GB,
                          subject="the LoRA matrices of this upload")
    assert e.value.status == 503
    assert e.value.message == (
        "the LoRA matrices of this upload need 3.0 GB of memory, and 2.0 GB "
        "is free; unload a model or retry when the requests in flight "
        "finish")
    other.cancel()


# T14 (b)
class _Collecting:
    """The keys a shrink reports. Reading them forces a collection, as a
    collection can run at any allocation of a shed."""

    def __init__(self, keys, ready):
        self.keys = list(keys)
        self.ready = ready

    def __len__(self):
        return len(self.keys)

    def __iter__(self):
        self.ready.set()
        gc.collect()
        return iter(self.keys)


@pytest.fixture
def no_auto_gc():
    """Only the test's own collection finalizes its cycles."""
    gc.disable()
    try:
        yield
    finally:
        gc.enable()


def test_a_collection_inside_a_shrink_finalizes_holds_while_a_reader_holds_the_lock(
        monkeypatch, no_auto_gc):
    pool = make_pool(monkeypatch, 10)
    y = image(pool, "y", 4, busy=True)
    c = Cache(pool, y)
    book(pool, y, "p1", GB)
    # A request hold and a booking, both garbage in one cycle: only a
    # collection finalizes them. The hold's release takes _lock.
    cycle = [_BusyHold(pool, y), pool.reserve_lora(y, GB, subject="s")]
    cycle.append(cycle)
    del cycle
    locked, ready = threading.Event(), threading.Event()

    def reader():
        with pool._lock:
            locked.set()
            assert ready.wait(WAIT)
            return resident(pool)

    def shrink():
        c.items.pop("p1")
        c.gone(_Collecting(["p1"], ready))

    r = start("reader", reader)
    assert locked.wait(WAIT)
    s = start("shrink", shrink)
    finish(*r)
    finish(*s)
    assert pool.lora_held(y) == (0, 0, {})
    assert y.busy == 0
    assert resident(pool) == 4 * GB


def test_the_insert_reads_the_caches_before_its_hold(monkeypatch, no_auto_gc):
    pool = make_pool(monkeypatch, 10)
    y = image(pool, "y", 4, busy=True)
    pool.acquire("y", None, IMAGE_KIND, build_spec=spec("y", 4))
    adapters = ByteLRU(1 << 40)
    adapters.put("own", object(), GB // 2, shed=False)
    adapters.keep({"own"})
    pipe = SimpleNamespace(pipe=SimpleNamespace(adapters=adapters,
                                                tables=ByteLRU(1 << 40)),
                           lora_own_term=GB)
    cycle = [_BusyHold(pool, y)]
    cycle.append(cycle)
    del cycle
    seams = Seams("insert")
    pool._seam = seams
    load = start("load-z", lambda: pool.acquire(
        "z", None, IMAGE_KIND, build_spec=spec("z", 2, pipe)))
    seams.reach("insert")
    locked = threading.Event()

    def shed_under_the_cache_lock():
        # A shed of the new entry's cache, inside whose lock a collection
        # finalizes a hold that releases through _lock.
        with adapters._lock:
            locked.set()
            gc.collect()
    s = start("shed", shed_under_the_cache_lock)
    assert locked.wait(WAIT)
    seams.go["insert"].set()
    z = finish(*load)
    finish(*s)
    assert y.busy == 1
    with pool._lock:
        rec = pool.books.rec(z)
        assert rec.own == {"own": GB // 2} and rec.own_term == GB


# T15
def test_two_reservations_over_one_entrys_idle_pairs_claim_them_once(
        monkeypatch):
    pool = make_pool(monkeypatch, 10)
    x = image(pool, "x", 4, busy=True)
    y = image(pool, "y", 5, busy=True)
    Cache(pool, x)
    Cache(pool, y)
    book(pool, x, "x1", GB)
    assert resident(pool) == 10 * GB
    seams = Seams("reserve")
    pool._seam = seams
    subject = "the LoRA factors of this step"
    one = start("request-1", lambda: pool.reserve_lora(y, GB, subject=subject))
    seams.reach("reserve")
    two = start("request-2", lambda: pool.reserve_lora(y, GB, subject=subject))
    seams.go["reserve"].set()
    for t, _ in (one, two):
        t.join(WAIT)
    outs = [one[1], two[1]]
    granted = [o["value"] for o in outs if "value" in o]
    refused = [o["error"] for o in outs if "error" in o]
    assert len(granted) == 1 and len(refused) == 1
    assert granted[0].claims_on() == [(x, GB)]
    assert refused[0].status == 503
    assert refused[0].message == (
        "the LoRA factors of this step need 1.0 GB of memory, and 0.0 GB is "
        "free beside the models in use (x, y); retry when one is idle")
    b = granted[0]
    claim_and_check(pool, b)
    put(pool, b, "y1", GB)
    pool.settle_lora(b, {"y1": GB})
    assert ledger(pool, x) == {} and ledger(pool, y) == {"y1": GB}
    assert resident(pool) == 10 * GB


def test_an_entry_evicted_while_claimed_frees_its_room_once(monkeypatch):
    pool = make_pool(monkeypatch, 10)
    x = image(pool, "x", 4)
    y = image(pool, "y", 4, busy=True)
    Cache(pool, x)
    Cache(pool, y)
    book(pool, x, "x1", GB)
    b = pool.reserve_lora(y, 2 * GB, subject="s")
    assert b.claims_on() == [(x, GB)]
    assert pool.evict("x")
    assert resident(pool) == 6 * GB
    # A reservation in the window after the teardown is granted the room
    # it freed, and no more.
    with pytest.raises(LoraRefused) as e:
        pool.reserve_lora(y, int(4.5 * GB), subject="s")
    assert e.value.status == 503
    third = pool.reserve_lora(y, 4 * GB, subject="s")
    assert third.claims_on() == []
    # The claimant's shed frees nothing for the gone entry; its re-check
    # passes and it runs.
    claim_and_check(pool, b)
    put(pool, b, "y1", 2 * GB)
    pool.settle_lora(b, {"y1": 2 * GB})
    put(pool, third, "y2", 4 * GB)
    pool.settle_lora(third, {"y2": 4 * GB})
    assert resident(pool) == 10 * GB


def test_a_claim_whose_bytes_left_another_way_ends_in_a_503(monkeypatch):
    pool = make_pool(monkeypatch, 10)
    x = image(pool, "x", 4, busy=True)
    y = image(pool, "y", 4, busy=True)
    cx = Cache(pool, x)
    Cache(pool, y)
    book(pool, x, "x1", GB)
    b = pool.reserve_lora(y, 2 * GB, subject="the LoRA factors of this step")
    # x's own job lets the claimed pair go; another booking takes the
    # room the claim counted on.
    cx.drop("x1")
    other = pool.reserve_lora(y, GB, subject="s")
    with pytest.raises(LoraRefused) as e:
        claim_and_check(pool, b)
    assert e.value.status == 503
    assert e.value.message == ("the LoRA factors of this step lost the "
                               "memory booked for them to other work; retry")
    put(pool, other, "y1", GB)
    pool.settle_lora(other, {"y1": GB})
    assert pool.lora_held(y) == (GB, 0, {"y1": GB})
    assert resident(pool) == 9 * GB


# T17
def test_a_cancelled_request_leaves_no_booking_behind(monkeypatch):
    pool = make_pool(monkeypatch, 10)
    x = image(pool, "x", 4)
    y = image(pool, "y", 4, busy=True)
    seams = Seams()
    pool._seam = seams
    real = concurrent.futures.ThreadPoolExecutor(1)
    futures = []

    class Executor:
        def submit(self, *a, **k):
            futures.append(real.submit(*a, **k))
            return futures[-1]

    async def cancelled():
        task = asyncio.ensure_future(residency.reserve_on_thread(
            pool, y, 3 * GB, subject="s", executor=Executor()))
        # The reservation would evict x, so its second attempt waits on
        # the build lock the test holds.
        while "reserve" not in seams.hit:
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    pool._build_lock.acquire()
    try:
        asyncio.run(cancelled())
    finally:
        pool._build_lock.release()
    real.shutdown(wait=True)
    # The thread booked when it returned; the future still holds the
    # booking, so only its done callback cancelled it.
    assert futures[0].result() is not None
    assert x.cache_key not in pool._entries
    assert pool.lora_held(y) == (0, 0, {})

    # A booking dropped unsettled is cancelled through its posted cancel.
    b = pool.reserve_lora(y, GB, subject="s")
    assert pool.lora_held(y)[1] == GB
    del b
    assert pool.lora_held(y) == (0, 0, {})


# T18
def test_under_overcommit_a_new_lora_sheds_the_idle_one(monkeypatch):
    pool = make_pool(monkeypatch, 10)
    monkeypatch.setenv("GMLX_OVERCOMMIT", "1")
    w = image(pool, "w", 1)
    y = image(pool, "y", 8, busy=True)
    Cache(pool, y)
    book(pool, y, "old", GB)
    assert resident(pool) == 10 * GB
    pool.head = 0
    b = pool.reserve_lora(y, 3 * GB, subject="s")
    assert b.claims_on() == [(y, GB)]
    claim_and_check(pool, b)
    put(pool, b, "new", 3 * GB)
    pool.settle_lora(b, {"new": 3 * GB})
    assert ledger(pool, y) == {"new": 3 * GB}
    assert w.cache_key in pool._entries


# T25
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


def test_a_reservation_during_a_cold_build_sees_the_build(monkeypatch):
    pool = make_pool(monkeypatch, 10)
    y = image(pool, "y", 4, busy=True)
    seams = Seams("roomed")
    pool._seam = seams
    load = start("load-z", lambda: pool.acquire(
        "z", None, IMAGE_KIND, build_spec=spec("z", 5)))
    seams.reach("roomed")
    with pytest.raises(LoraRefused) as e:
        pool.reserve_lora(y, 2 * GB, subject="the LoRA factors of this step")
    assert e.value.status == 503
    assert e.value.message == (
        "the LoRA factors of this step need 2.0 GB of memory, and 1.0 GB is "
        "free beside the models in use (y); retry when one is idle")
    seams.go["roomed"].set()
    finish(*load)
    assert resident(pool) == 9 * GB and pool._building == 0


def test_a_repriced_build_evicts_once_for_its_new_price(monkeypatch):
    pool = make_pool(monkeypatch, 10)
    w = image(pool, "w", 1)
    image(pool, "y", 4, busy=True)
    reprice(pool, "z", 5)
    z = pool.acquire("z", None, IMAGE_KIND, build_spec=spec("z", 3))
    assert z.footprint == 5 * GB
    assert w.cache_key in pool._entries
    assert resident(pool) == 10 * GB and pool._building == 0


def test_a_build_that_fails_after_its_room_check_leaves_no_building(
        monkeypatch):
    pool = make_pool(monkeypatch, 10)
    image(pool, "y", 8, busy=True)
    real = pool._build
    reprice(pool, "z", 4)
    # The refit cannot fit 4 GB beside 8 GB in use, and defers.
    with pytest.raises(LoadDeferred):
        pool.acquire("z", None, IMAGE_KIND, build_spec=spec("z", 2))
    assert pool._building == 0 and resident(pool) == 8 * GB

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
    assert pool._building == 0 and resident(pool) == 8 * GB
    assert len(pool.released) == released + 1


# T26: review 4's sizes, a 10 GB budget with 8 GB resident, a 2 GB cold
# load and a 1.5 GB turn booking.
def _race_pool(monkeypatch):
    pool = make_pool(monkeypatch, 10)
    w = image(pool, "w", 2)
    y = image(pool, "y", 6, busy=True)
    return pool, w, y


def _turn(pool, y):
    return start("image-worker-test", lambda: pool.turn_lora(
        y, None, int(1.5 * GB), subject="the LoRA factors of this step"))


def test_a_turn_after_a_cold_loads_room_check_sees_the_build(monkeypatch):
    pool, w, y = _race_pool(monkeypatch)
    seams = Seams("roomed")
    pool._seam = seams
    load = start("load-z", lambda: pool.acquire(
        "z", None, IMAGE_KIND, build_spec=spec("z", 2)))
    seams.reach("roomed")
    turn = _turn(pool, y)
    turn[0].join(WAIT)
    seams.go["roomed"].set()
    finish(*load)
    with pytest.raises(LoraRefused) as e:
        finish(*turn)
    assert e.value.status == 503
    assert resident(pool) == 10 * GB


def test_a_turn_during_the_insert_sees_the_new_entry(monkeypatch):
    pool, w, y = _race_pool(monkeypatch)
    seams = Seams("insert")
    pool._seam = seams
    load = start("load-z", lambda: pool.acquire(
        "z", None, IMAGE_KIND, build_spec=spec("z", 2)))
    seams.reach("insert")
    turn = _turn(pool, y)
    # The turn waits on _lock while the insert holds it.
    turn[0].join(0.3)
    seams.go["insert"].set()
    finish(*load)
    with pytest.raises(LoraRefused) as e:
        finish(*turn)
    assert e.value.status == 503
    assert resident(pool) == 10 * GB


def test_a_cold_load_after_a_turns_decision_sees_its_booking(monkeypatch):
    pool, w, y = _race_pool(monkeypatch)
    seams = Seams("turn", "roomed")
    pool._seam = seams
    turn = _turn(pool, y)
    seams.reach("turn")
    seams.go["roomed"].set()
    load = start("load-z", lambda: pool.acquire(
        "z", None, IMAGE_KIND, build_spec=spec("z", 2)))
    # The load's room check waits on _lock while the turn holds it.
    seams.at["roomed"].wait(0.3)
    seams.go["turn"].set()
    b = finish(*turn)
    finish(*load)
    assert b.pending == int(1.5 * GB)
    # The load evicted the idle model to fit beside the booking.
    assert w.cache_key not in pool._entries
    assert resident(pool) == int(9.5 * GB)


# The pipe caches and the upload store, hooked to the books.
class Overrun(AssertionError):
    pass


def _raise_overrun(subject, nbytes):
    raise Overrun(f"{nbytes} bytes past their booking: {subject}")


def lora_pipe(pool, name, *, own=(), own_term=0, busy=True):
    """An image entry whose pipeline has a served model's LoRA caches,
    hooked to the pool's books at its insert. ``own`` pairs are its own,
    loaded before the insert."""
    from gmlx.image.lora import install

    p = SimpleNamespace(adapters=ByteLRU(1 << 40), tables=ByteLRU(1 << 40),
                        lora=None, patch_tables={}, default_table=None)
    keys = [install.register_pair(p, *pr) for pr in own]
    p.adapters.keep(keys)
    served = SimpleNamespace(pipe=p, lora_own_term=own_term)
    return image(pool, name, 0, busy=busy, pipe=served), p


def pair(rng, nbytes):
    """A float32 pair of ``nbytes``: rank 1, in and out of nbytes / 8."""
    side = nbytes // 8
    return (mx.array(rng.standard_normal((1, side)).astype(np.float32)),
            mx.array(rng.standard_normal((side, 1)).astype(np.float32)))


def wire(rng, nbytes):
    from gmlx.serve.image_wire import WireTensor

    raw = rng.standard_normal(nbytes // 4).astype(np.float32).tobytes()
    return WireTensor("F32", (nbytes // 4,), memoryview(raw))


def upload_store(pool, max_bytes=1 << 40):
    from gmlx.serve.image_steps import _Bytes

    store = _Bytes(max_bytes, "the adapter store", "adapter_cache_mb")
    store.on_put = pool.books.put_hook(STORE)
    store.on_gone = pool.books.release_hook(STORE)
    pool._store_shed = store.shed
    return store


def store_keys(store):
    return {k: store.nbytes(k) for k in list(store._items)}


def upload(pool, store, key, t, subject="the LoRA matrices of this upload"):
    from gmlx.serve import image_service

    b = pool.reserve_lora(STORE, t.nbytes, subject=subject)
    image_service.lora_claims(pool, b)
    with b.active():
        store.put(key, t)
    pool.settle_lora(b, image_service.lora_figures((store,), b.keys))


def test_the_ledger_follows_the_pipe_caches(monkeypatch):
    """T10: each put enters, each shed and drop releases, and a held key
    put again books once (F1)."""
    from gmlx.image.lora import install
    from gmlx.serve import image_service

    pool = make_pool(monkeypatch, 0.01)
    pool.books.overrun_hook = _raise_overrun
    rng = np.random.default_rng(0)
    y, p = lora_pipe(pool, "y")
    KB = 100_000

    def held_equals_the_caches():
        held = {k: p.adapters.nbytes(k) for k in p.adapters.keys(bool)}
        held.update({k: p.tables.nbytes(k) for k in p.tables.keys(bool)})
        h, pending, led = pool.lora_held(y)
        assert led == held and h == sum(held.values()) and pending == 0
        assert resident(pool) <= pool._budget

    def job(nbytes, put):
        b = pool.reserve_lora(y, nbytes, subject="s")
        try:
            with b.active():
                return put()
        finally:
            image_service.settle_job(pool, b, p)
            held_equals_the_caches()

    k1, k2 = job(2 * KB, lambda: [install.register_pair(p, *pair(rng, KB))
                                  for _ in range(2)])
    # An LRU shed of the pair cache releases the pair it let go.
    p.adapters.max_bytes = int(2.5 * KB)
    k3 = job(KB, lambda: install.register_pair(p, *pair(rng, KB)))
    assert set(p.adapters.keys(bool)) == {k2, k3}
    # A pair shed by the next put of its own job leaves nothing booked.
    p.adapters.max_bytes = KB
    job(2 * KB, lambda: [install.register_pair(p, *pair(rng, KB))
                         for _ in range(2)])
    assert len(p.adapters) == 1
    # A pair over the cache is refused before it is put: its booking
    # returns whole at the settle.
    with pytest.raises(install.PairRoomError):
        job(2 * KB, lambda: install.register_pair(p, *pair(rng, 2 * KB)))

    # A built table put again at the same bytes after each forward books
    # once; grown, it books the growth.
    key = ("t", mx.bfloat16, "block")
    table = object()
    job(KB, lambda: install._keep_table(p, key, table, KB))
    job(0, lambda: install._keep_table(p, key, table, KB))
    job(KB, lambda: install._keep_table(p, key, table, 2 * KB))
    assert pool.lora_held(y)[2][key] == 2 * KB
    # A drop of the tables cache releases what it let go.
    p.tables.clear()
    held_equals_the_caches()


def test_an_own_put_is_not_booked_and_an_unbooked_put_is_an_overrun(
        monkeypatch):
    from gmlx.image.lora import install
    from gmlx.serve.lora_books import own_load

    pool = make_pool(monkeypatch, 0.01)
    pool.books.overrun_hook = _raise_overrun
    rng = np.random.default_rng(1)
    own = [pair(rng, 100_000)]
    y, p = lora_pipe(pool, "y", own=own, own_term=150_000)
    with pool._lock:
        rec = pool.books.rec(y)
        assert sum(rec.own.values()) == 100_000 and rec.own_term == 150_000
    # The own pairs loaded again, as a video entry's after an encode,
    # are its own.
    p.adapters.keep(())
    p.adapters.clear()
    with own_load():
        install.register_pair(p, *own[0])
    assert pool.lora_held(y) == (0, 0, {})
    with pytest.raises(Overrun, match="no booking"):
        install.register_pair(p, *pair(rng, 100_000))
    # A put past its booking's figure is an overrun at the settle.
    from gmlx.serve import image_service

    b = pool.reserve_lora(y, 50_000, subject="the LoRA factors of this step")
    with b.active():
        install.register_pair(p, *pair(rng, 100_000))
    with pytest.raises(Overrun, match="50000 bytes .* this step"):
        image_service.settle_job(pool, b, p)


def test_the_upload_store_books_its_puts_and_sheds(monkeypatch):
    """T10: a store eviction lowers the uploads."""
    from gmlx.serve import image_service

    pool = make_pool(monkeypatch, 0.01)
    pool.books.overrun_hook = _raise_overrun
    rng = np.random.default_rng(2)
    store = upload_store(pool, max_bytes=250_000)
    for i in range(3):
        upload(pool, store, f"u{i}", wire(rng, 100_000))
        assert ledger(pool, STORE) == store_keys(store)
    assert len(store_keys(store)) == 2
    store.shed(1)
    assert ledger(pool, STORE) == store_keys(store)
    store.clear()
    assert pool.lora_held(STORE) == (0, 0, {})
    # Another reservation's claim sheds the store between a job's read of
    # its figures and its settle: the key it put is not booked.
    b = pool.reserve_lora(STORE, 100_000, subject="s")
    with b.active():
        store.put("late", wire(rng, 100_000))
    figures = image_service.lora_figures((store,), b.keys)
    assert figures == {"late": 100_000}
    store.shed(store.bytes)
    pool.settle_lora(b, figures)
    assert pool.lora_held(STORE) == (0, 0, {})


def test_an_evicting_upload_on_the_event_loop_never_waits_on_the_lock(
        monkeypatch):
    """T14 (a)."""
    pool = make_pool(monkeypatch, 0.01)
    rng = np.random.default_rng(3)
    store = upload_store(pool, max_bytes=150_000)
    upload(pool, store, "old", wire(rng, 100_000))
    b = pool.reserve_lora(STORE, 100_000, subject="s")
    locked, done = threading.Event(), threading.Event()

    def holder():
        with pool._lock:
            locked.set()
            return done.wait(WAIT)
    h = start("holder", holder)
    assert locked.wait(WAIT)

    async def put_on_the_loop():
        with b.active():
            store.put("new", wire(rng, 100_000))
        done.set()
    asyncio.run(put_on_the_loop())
    assert finish(*h), "the put waited on the pool's lock"
    pool.settle_lora(b, {"new": 100_000})
    assert ledger(pool, STORE) == store_keys(store) == {"new": 100_000}


def test_rounds_of_bookings_and_sheds_against_evicting_puts(monkeypatch):
    """T14 (c): uploads from three threads, two of them evicting each
    other's past the store's bound and one booking past the pool's
    budget, end with the books equal to the keys held."""
    pool = make_pool(monkeypatch, 0.0005)
    store = upload_store(pool, max_bytes=300_000)
    refused = []

    def uploads(seed, prefix, nbytes):
        rng = np.random.default_rng(seed)
        for i in range(200):
            # Keys repeat, so a put also replaces a key it put before,
            # or puts again one a shed let go.
            try:
                upload(pool, store, f"{prefix}{i % 7}", wire(rng, nbytes))
            except LoraRefused:
                refused.append(prefix)

    threads = [start(f"uploads-{n}", lambda s=s, n=n, b=b: uploads(s, n, b))
               for s, n, b in ((4, "a", 100_000), (5, "b", 100_000),
                               (6, "c", 200_000))]
    for t in threads:
        finish(*t)
    assert ledger(pool, STORE) == store_keys(store)
    held, pending, _ = pool.lora_held(STORE)
    assert pending == 0 and held == store.bytes <= pool._budget
    with pool._lock:
        assert pool.books.claimed() == 0


def test_a_claim_sheds_the_default_table_another_entry_ran_last(
        monkeypatch, tmp_path):
    """T15, probe B of review 4: entry x ran its own default table last
    and its excess is booked; y's claim sheds it through _release_cur, and
    y runs."""
    from image_fixtures import tiny_z_model
    from lora_fixtures import tiny_z_adapter, write_adapter

    from gmlx.image import api, lora
    from gmlx.image.lora import install
    from gmlx.image.registry import get_family
    from gmlx.serve import image_service

    pool = make_pool(monkeypatch, 1)
    pool._lora_shed = None
    pool.books.overrun_hook = _raise_overrun
    m, _ = tiny_z_model()
    pipe = api.ImagePipeline(dit=m, family=get_family("z_image"),
                             variant="turbo")
    ad = lora.load_adapter(pipe, write_adapter(
        tmp_path / "own.safetensors", "kohya", tiny_z_adapter(7)))
    pipe.default_table = api.define_patch_table(
        pipe, lora.table_entries([(ad, 0.8)]))
    pipe.adapters.keep(image_service._pair_keys([(ad, 0.8)]))
    own_pairs = pipe.adapters.bytes
    served = SimpleNamespace(pipe=pipe, lora_own_term=own_pairs)
    x = image(pool, "x", 0, busy=True, pipe=served)
    y = image(pool, "y", 0, busy=True)
    figs = install.table_figures(pipe, pipe.default_table, mx.float32)
    excess = sum(f for f, _h in figs.values())
    assert excess > 0
    b = pool.reserve_lora(x, excess, subject="s")
    rng = np.random.default_rng(0)
    lat = mx.array(rng.standard_normal((1, 4, 6, 8)).astype(np.float32))
    cond = api.Conditioning(
        mx.array(rng.standard_normal((5, 16)).astype(np.float32)), "k")
    with b.active(), mx.stream(mx.cpu):
        mx.eval(api.denoise_step(pipe, lat, 0.75, cond))
    image_service.settle_job(pool, b, pipe)
    with pool._lock:
        assert pool.books.excess(x) == excess
    assert pool.lora_held(x) == (excess, 0, {})
    # y needs the room x's excess holds.
    pool._budget = excess + excess // 2
    c = pool.reserve_lora(y, excess, subject="s")
    assert c.claims_on() == [(x, excess - excess // 2)]
    image_service.lora_claims(pool, c)
    assert pool.lora_held(x) == (0, 0, {})
    assert resident(pool) == excess
