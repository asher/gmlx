"""The LoRA bytes of image and video models, booked with the residency
pool before they are allocated (docs/internals/image-steps.md).

The books hold, for each owner (a pool entry, or the step routes' upload
store), a ledger of the keys a booking paid for, the bytes of its open
bookings, and the claims open bookings hold on its idle bytes. Booked
bytes grow only under the pool's lock. They shrink without it: a cache
that lets keys go, a settle and a cancel post to a queue, which every
holder of the pool's lock drains before it reads the books. So a grant
that misses a concurrent change misses a shrink, and a post never waits,
under a cache's lock or in a finalizer. A cache posts the key it puts
from under its own lock too, so the queue holds each cache's puts and
drops in the order the cache made them."""

from __future__ import annotations

import contextlib
import contextvars
import logging
import math
import queue
import weakref

_log = logging.getLogger(__name__)


class _Store:
    """The owner of the step routes' uploads."""

    def __repr__(self) -> str:
        return "uploads"


STORE = _Store()

# The booking of the job or route that runs: the puts it makes enter
# their keys from it.
_CURRENT: contextvars.ContextVar = contextvars.ContextVar(
    "lora_booking", default=None)
# Set while an entry loads its own adapters: their pairs are the entry's
# own, within its footprint.
_OWN: contextvars.ContextVar = contextvars.ContextVar("lora_own",
                                                       default=False)


class LoraRefused(Exception):
    """A booking the pool cannot grant: ``status`` 413 when no unload
    makes room, 503 when one can."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def gb(n: float, up: bool) -> str:
    """``n`` bytes in decimal GB to one decimal, rounded up or down."""
    x = max(0.0, float(n)) / 1e9 * 10
    return f"{(math.ceil(x) if up else math.floor(x)) / 10:.1f}"


class _Open:
    """The books' record of one open booking. The books hold it, never the
    ``Booking``, so a booking nobody closed is still collected."""

    __slots__ = ("owner", "granted", "pending", "claims", "entered",
                 "closed")

    def __init__(self, owner, n: int):
        self.owner = owner
        self.granted = int(n)
        self.pending = int(n)
        self.claims: dict = {}          # id(source) -> [source, bytes]
        self.entered: dict = {}         # key -> bytes it entered at
        self.closed = False


class Booking:
    """Bytes the pool granted one job or route before it allocates them.
    The keys it puts enter the ledger from it; its settle sets each to the
    cache's figure and returns the rest. A booking never settled or
    cancelled is cancelled when it is collected."""

    def __init__(self, books, rec: _Open, subject: str):
        self.books = books
        self._rec = rec
        self.subject = subject
        # The dry run's figure of each key it priced.
        self.priced: dict = {}
        # The keys it entered, kept by the job's own thread so the settle's
        # figures are read before the pool's lock.
        self.keys: set = set()
        self.done = False

    @property
    def owner(self):
        return self._rec.owner

    @property
    def granted(self) -> int:
        return self._rec.granted

    @property
    def pending(self) -> int:
        return self._rec.pending

    @property
    def claimed(self) -> int:
        return sum(b for _s, b in self._rec.claims.values())

    def claims_on(self) -> list:
        """(source, bytes) of each claim it holds."""
        return [(s, b) for s, b in self._rec.claims.values()]

    def enter(self, key, nbytes: int) -> None:
        """Book ``key`` from this booking as its cache puts it, up to its
        priced figure when it has one. A key booked already is raised to
        the figure, never booked twice."""
        self.keys.add(key)
        self.books.post(("enter", self._rec, key,
                         int(self.priced.get(key, nbytes))))

    def cancel(self) -> None:
        """Give back what was not entered, and the claims. A shrink, as a
        claim is at most its booking."""
        if not self.done:
            self.done = True
            self.books.post(("cancel", self._rec))

    def __del__(self):
        try:
            if not self.done:
                self.done = True
                self.books.post(("cancel", self._rec))
        except Exception:  # noqa: S110 - GC-time cleanup must never raise
            pass

    @contextlib.contextmanager
    def active(self):
        """Make this the booking the puts of the block enter from."""
        token = _CURRENT.set(self)
        try:
            yield self
        finally:
            _CURRENT.reset(token)


def current():
    """The booking the puts of this job enter from, or None."""
    return _CURRENT.get()


@contextlib.contextmanager
def own_load():
    """The block loads an entry's own adapters: their pairs are its own."""
    token = _OWN.set(True)
    try:
        yield
    finally:
        _OWN.reset(token)


def loading_own() -> bool:
    return _OWN.get()


class _Owner:
    __slots__ = ("ref", "ledger", "held", "pending", "own", "own_term",
                 "excess")

    def __init__(self, owner, own_term: int = 0):
        self.ref = owner
        self.ledger: dict = {}          # key -> booked bytes
        self.held = 0                   # the ledger and the excess
        self.pending = 0                # open bookings not yet entered
        self.own: dict = {}             # own keys -> cache bytes
        self.own_term = int(own_term)   # the footprint's own LoRA term
        self.excess = 0                 # own bytes past own_term, booked

    def own_used(self) -> int:
        return sum(self.own.values())


def _ref(owner):
    """A weak reference to ``owner``, or to the store, which lives as long
    as the process."""
    if owner is STORE:
        return lambda: STORE
    return weakref.ref(owner)


class Books:
    """The ledger of every owner, the open claims and the shrink queue.
    Every method but ``post`` and ``release_hook`` runs under the pool's
    lock. The queue drains at every grant and every read of the books;
    posts follow cache puts, drops and clears and booking closes only."""

    def __init__(self):
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._owners: dict = {id(STORE): _Owner(STORE)}
        self.claims: dict = {}          # id(source) -> bytes
        self.open: set = set()          # _Open records
        # Tests set it to raise on bytes past a booking.
        self.overrun_hook = None

    # posts, from any thread
    def post(self, op) -> None:
        self._q.put(op)

    def release_hook(self, owner):
        """The ``on_gone`` of an owner's caches: posts the keys they let
        go. It holds the owner weakly, so a released pipeline keeps no
        entry alive."""
        ref = _ref(owner)

        def gone(keys) -> None:
            o = ref()
            if o is not None and keys:
                self.post(("release", o, tuple(keys)))
        return gone

    def put_hook(self, owner):
        """The ``on_put`` of an owner's caches. A key put while the owner
        loads its own adapters is its own; any other enters from the
        booking of the job that puts it. A put that grows the bytes with
        no booking is an overrun."""
        ref = _ref(owner)

        def put(key, nbytes, old) -> None:
            o = ref()
            if o is None:
                return
            if _OWN.get():
                self.post(("own", o, key, int(nbytes)))
                return
            b = _CURRENT.get()
            if b is not None and b.owner is o and not b.done:
                b.enter(key, nbytes)
            elif int(nbytes) > int(old):
                self.overrun(f"a LoRA put of {key!r} with no booking",
                             int(nbytes) - int(old))
        return put

    def overrun(self, subject: str, nbytes: int) -> None:
        """Bytes past their booking: a pricing bug. Logged; a test's hook
        raises."""
        _log.warning("LoRA bytes %.3f GB past their booking: %s",
                     nbytes / 1e9, subject)
        if self.overrun_hook is not None:
            self.overrun_hook(subject, nbytes)

    # under the pool's lock
    def attach(self, owner, own_term: int = 0, own: dict | None = None):
        rec = self.rec(owner)
        if rec is None:
            rec = _Owner(owner, own_term)
            self._owners[id(owner)] = rec
        rec.own_term = int(own_term)
        if own:
            rec.own.update(own)
        return rec

    def rec(self, owner):
        rec = self._owners.get(id(owner))
        return rec if rec is not None and rec.ref is owner else None

    def drop(self, owner) -> None:
        """An owner torn down: its ledger, held and pending go, and the
        claims on it."""
        self._owners.pop(id(owner), None)
        self.claims.pop(id(owner), None)
        for r in list(self.open):
            if r.owner is owner:
                r.closed = True
                r.pending = 0
                self.open.discard(r)
                self._unclaim(r)
            else:
                r.claims.pop(id(owner), None)

    def drain(self) -> None:
        while True:
            try:
                op = self._q.get_nowait()
            except queue.Empty:
                return
            getattr(self, "_op_" + op[0])(*op[1:])

    def _op_release(self, owner, keys) -> None:
        rec = self.rec(owner)
        if rec is None:
            return
        for k in keys:
            b = rec.ledger.pop(k, None)
            if b is not None:
                rec.held -= b
            elif rec.own.pop(k, None) is not None:
                # The excess only shrinks here; it grows at a settle.
                cut = max(0, rec.own_used() - rec.own_term)
                if cut < rec.excess:
                    rec.held -= rec.excess - cut
                    rec.excess = cut

    def _op_own(self, owner, key, nbytes) -> None:
        rec = self.rec(owner)
        if rec is not None:
            rec.own[key] = int(nbytes)

    def _op_enter(self, r: _Open, key, nbytes) -> None:
        rec = self.rec(r.owner)
        if rec is None or r not in self.open:
            return
        cur = rec.ledger.get(key, 0)
        move = max(0, min(int(nbytes) - cur, r.pending))
        r.pending -= move
        rec.pending -= move
        rec.ledger[key] = cur + move
        rec.held += move
        r.entered[key] = r.entered.get(key, 0) + move

    def _op_cancel(self, r: _Open) -> None:
        self.close(r)

    def close(self, r: _Open) -> None:
        """Return a booking's pending and drop its claims."""
        if r not in self.open:
            return
        self.open.discard(r)
        rec = self.rec(r.owner)
        if rec is not None:
            rec.pending -= r.pending
        r.pending = 0
        self._unclaim(r)
        r.closed = True

    def _unclaim(self, r: _Open) -> None:
        for sid, (_src, b) in r.claims.items():
            left = self.claims.get(sid, 0) - b
            if left > 0:
                self.claims[sid] = left
            else:
                self.claims.pop(sid, None)
        r.claims.clear()

    def unclaim(self, booking: Booking) -> None:
        self._unclaim(booking._rec)

    def open_booking(self, owner, n: int, subject: str,
                     claims=()) -> Booking:
        rec = self.rec(owner) or self.attach(owner)
        r = _Open(owner, n)
        rec.pending += int(n)
        self.open.add(r)
        self._claim(r, claims)
        return Booking(self, r, subject)

    def _claim(self, r: _Open, claims) -> None:
        for src, c in claims:
            if c <= 0:
                continue
            got = r.claims.setdefault(id(src), [src, 0])
            got[1] += int(c)
            self.claims[id(src)] = self.claims.get(id(src), 0) + int(c)

    def grow(self, booking: Booking, n: int, claims=()) -> None:
        """A turn booking: ``n`` more for an open booking."""
        r = booking._rec
        rec = self.rec(r.owner)
        r.granted += int(n)
        r.pending += int(n)
        if rec is not None:
            rec.pending += int(n)
        self._claim(r, claims)

    def is_open(self, booking: Booking) -> bool:
        return booking._rec in self.open

    def settle(self, booking: Booking, figures: dict) -> int:
        """Set each key the booking entered and the cache still holds to
        the cache's figure in ``figures`` (a key gone is left to its
        release), take the growth of the owner's excess from it, and
        return the rest. Returns the bytes past the booking, which the
        caller books under the pool's lock and logs."""
        r = booking._rec
        booking.done = True
        if r not in self.open:
            return 0
        rec = self.rec(r.owner)
        over = 0
        if rec is not None:
            grow = sum(int(figures[k]) - rec.ledger[k] for k in r.entered
                       if k in rec.ledger and k in figures)
            excess = max(0, rec.own_used() - rec.own_term)
            grow += max(0, excess - rec.excess)
            over = max(0, grow - r.pending)
            for k in r.entered:
                if k in rec.ledger and k in figures:
                    rec.held += int(figures[k]) - rec.ledger[k]
                    rec.ledger[k] = int(figures[k])
            if excess > rec.excess:
                rec.held += excess - rec.excess
                rec.excess = excess
        self.close(r)
        return over

    # totals, under the pool's lock after a drain
    def total(self) -> int:
        return sum(r.held + r.pending for r in self._owners.values())

    def pending_total(self, but: Booking | None = None) -> int:
        skip = None if but is None else but._rec
        return sum(r.pending for r in self.open if r is not skip)

    def claimed(self) -> int:
        return sum(self.claims.values())

    def held(self, owner) -> int:
        rec = self.rec(owner)
        return 0 if rec is None else rec.held

    def pending(self, owner) -> int:
        rec = self.rec(owner)
        return 0 if rec is None else rec.pending

    def idle(self, owner, but: int = 0) -> int:
        """What ``owner`` holds that no booking has claimed, less ``but``
        bytes of the claimant's own keys."""
        return max(0, self.held(owner) - self.claims.get(id(owner), 0)
                   - int(but))

    def ledger(self, owner) -> dict:
        rec = self.rec(owner)
        return {} if rec is None else dict(rec.ledger)

    def excess(self, owner) -> int:
        rec = self.rec(owner)
        return 0 if rec is None else rec.excess
