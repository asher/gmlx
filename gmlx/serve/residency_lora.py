"""The residency pool's LoRA turn (docs/internals/image-steps.md)."""

from __future__ import annotations

import os
import threading
import time
from typing import NamedTuple

from .residency_types import (STORE, LoraRefused, _WORKER_NAMES, _WORKER_ORDER,
                              _WORKER_WORDS, _log, _mine, _owner_kind, gb)


class _Aside(NamedTuple):
    """A turn's answer that it waits on the releases queued on the other
    worker: the bytes it is short of, and the running count of the bytes
    their releases freed as it saw it."""

    short: int
    freed: int


class _LoraTurn:
    """The LoRA turn of a worker's job and the upload store's admission."""

    # The LoRA turn (docs/internals/image-steps.md). A worker's job books
    # the LoRA bytes it allocates next at its turn, under the build lock a
    # cold load holds, with _lock inside it for each read or write of the
    # entries and ledgers. No hold of _lock spans a teardown, a pipeline
    # release, a collection, a measure or a cache lock.

    def _reconcile(self, entry, figs) -> int:
        """Under _lock: set the ledger of ``entry``, the turn's owner, to
        ``figs`` and end its booking. Returns the bytes it holds past what
        it booked."""
        led = entry.lora if entry is not STORE else None
        if led is None or figs is None or not self._in_pool(entry):
            return 0
        led.set(figs)
        over = led.held - (led.base + led.pending)
        led.pending = 0
        led.base = led.held
        return max(0, over)

    def _overrun(self, subject: str, over: int) -> None:
        if over <= 0:
            return
        _log.info("%s hold %.1f MB of LoRA memory past their turn's "
                  "booking", subject, over / 1e6)
        if self.overrun_hook is not None:
            self.overrun_hook(subject, over)

    def _uploads(self) -> int:
        """Under _lock: the uploads' bytes, held and admitted. The store's
        count is one int read with no lock: a put counts its bytes there
        before they leave ``_incoming``, so the sum never reads short."""
        live = (self._store_live or _default_store_live)()
        return int(live) + self._incoming

    def _retry(self) -> int:
        if self._retry_fn is not None:
            return int(self._retry_fn())
        from .patches.request_flow import deferred_retry_after

        return int(deferred_retry_after())

    def _lora_resident(self, live: int) -> int:
        """Under _lock: what the budget counts with ``live`` upload bytes,
        the victims whose teardown runs included."""
        return self._resident_bytes() + self._tearing + live

    def _measured_room(self, owner, head, live: int):
        """Under _lock: the measured room of a turn, or None unmeasured:
        the measure less what the streams keep, the bookings of other
        owners, the build in flight and the uploads, host memory the
        measure misses."""
        if head is None:
            return None
        other = sum(e.lora.pending for e in self._entries.values()
                    if e.lora is not None and e is not owner)
        return (head - self._reserved_room_bytes() - other - self._building
                - live)

    def _fits(self, owner, need: int, head, live: int) -> bool:
        """Under _lock: whether ``need`` more bytes fit the budget and the
        measured room."""
        if self._budget and self._lora_resident(live) + need > self._budget:
            return False
        room = self._measured_room(owner, head, live)
        return room is None or need <= room

    def _book(self, owner, need: int) -> None:
        """Under _lock: book ``need`` bytes on the owner's ledger, or count
        them as admitted for the upload store."""
        if owner is STORE:
            self._incoming += int(need)
            return
        led = owner.lora
        if led is not None:
            led.base = led.held
            led.pending = int(need)

    def upload_admit(self, n: int, head) -> bool:
        """Count the ``n`` bytes of an upload body in the pool's incoming
        figure when they fit the budget and the measured room ``head``
        (read with no lock) beside what the pool holds. Runs off the event
        loop and takes the pool lock for the count. It books, claims and
        sheds no model or LoRA memory."""
        from .capacity import overcommit

        over = overcommit()
        with self._lock:
            ok = over or self._fits(STORE, n, head, self._uploads())
            if ok:
                self._book(STORE, n)
                self._publish()
        return ok

    def upload_done(self, n: int) -> None:
        """An admitted body was put, or failed: its bytes leave the
        admitted count."""
        with self._lock:
            self._incoming = max(0, self._incoming - int(n))
            self._publish()

    def _worker_guard(self, owner) -> None:
        kind = _owner_kind(owner)
        if not threading.current_thread().name.startswith(
                _WORKER_NAMES[kind]):
            raise RuntimeError(f"a LoRA turn runs on the {_WORKER_WORDS[kind]}"
                               " worker, where its job allocates")

    def lora_turn(self, owner, need: int, *, subject: str, hold=(),
                  held: int = 0, store=(), beside=()) -> None:
        """The turn of a job on its owner's worker: room for ``need`` bytes
        it allocates next, booked on the ledger of ``owner``, a pool entry,
        or admitted into the upload store. ``hold`` names the owner's keys
        its job keeps, ``held`` their bytes held now, ``store`` the uploads
        it keeps and ``beside`` the other entries its job holds, such as a
        borrowed encoder, which count as its own. Sheds idle LoRA bytes,
        evicts idle models as a cold load does, measures and books, or
        waits for chat models in use within the deadline. A release
        queued on the other worker frees its bytes first: after it sheds
        and before it evicts, the turn waits on it until the bytes freed
        cover its shortfall, within one window of half the deadline left
        when it first waits, which it may wait in more than once; the
        rest keeps the wait for the build lock.
        Raises LoraRefused: 413 when no unload makes room, else 503 with
        Retry-After."""
        from .capacity import overcommit

        self._worker_guard(owner)
        need = max(0, int(need))
        beside = tuple(beside)
        over = overcommit()
        retry = self._retry()
        deadline = time.monotonic() + retry
        figs = self._read_lora(owner)
        head = None if over else self._measured_lora_room()
        with self._lock:
            late = self._reconcile(owner, figs)
            # A turn that allocates nothing needs no room.
            ok = owner is not STORE and (
                need == 0 or self._fits(owner, need, head, self._uploads()))
            if ok:
                self._book(owner, need)
                self._hit("booked")
            self._publish()
        self._overrun(subject, late)
        if ok:
            return
        err = None
        aside_end = None
        while True:
            now = time.monotonic()
            left = deadline - now
            if err is not None and left <= 0:
                raise err
            if not self._build_lock.acquire(timeout=max(0.0, left)):
                head = (None if over or err is not None
                        else self._measured_lora_room())
                with self._lock:
                    if err is not None and not self._building:
                        raise err
                    raise self._late_503(owner, need, subject, retry, head,
                                         beside)
            try:
                got = self._turn_room(
                    owner, need, subject, hold, held, store, retry, over,
                    beside, aside=left > 0 and (aside_end is None
                                                or now < aside_end))
            finally:
                self._build_lock.release()
            if got is None:
                return
            if isinstance(got, _Aside):
                # Off the build lock, which the release may wait on. One
                # window, half the deadline left as the turn first waits:
                # past it, the turn evicts as before, with the rest to
                # wait for the build lock.
                if aside_end is None:
                    t = time.monotonic()
                    aside_end = t + (deadline - t) / 2
                with self._lock:
                    while (self._aside(owner) > 0 and self._aside_freed(owner)
                           - got.freed < got.short):
                        left = aside_end - time.monotonic()
                        if left <= 0:
                            break
                        self._idle.wait(left)
                continue
            seen, err = got
            with self._lock:
                while self._releases == seen:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise err
                    self._idle.wait(left)

    def _turn_room(self, owner, need, subject, hold, held, store, retry,
                   over, beside, aside: bool = True):
        """Under the build lock: shed, evict, measure, then check and book
        in one hold. Returns None when booked, (the release count, the
        503) to wait on, or, with ``aside``, an _Aside when the releases
        queued on the other worker free the bytes the turn is still short
        of after it sheds: the turn waits on them before it picks a
        victim."""
        self._hit("turn")
        head = None if over else self._measured_lora_room()
        with self._lock:
            short = self._shortfall(owner, need, head, self._uploads())
        if short > 0:
            self._shed_for(owner, short, hold, store)
            if aside and not over and self._has_aside(owner):
                head = self._measured_lora_room()
                with self._lock:
                    short = self._shortfall(owner, need, head,
                                            self._uploads())
                    if 0 < short <= self._aside(owner):
                        return _Aside(short, self._aside_freed(owner))
        torn: list = []
        released: list = []
        try:
            with self._lock:
                if not over:
                    self._evict_for_room(need, torn=torn, strict=True,
                                         extra=self._uploads(),
                                         kind=_owner_kind(owner))
                self._publish()
        finally:
            try:
                self._teardown_all(torn, released)
            finally:
                self._release_images(released)
        head = None if over else self._measured_lora_room()
        with self._lock:
            live = self._uploads()
            if over or self._fits(owner, need, head, live):
                self._book(owner, need)
                self._publish()
                return None
            return self._refusal(owner, need, head, live, held, subject,
                                 retry, beside)

    def _shortfall(self, owner, need: int, head, live: int) -> int:
        """Under _lock: the bytes past the budget or the measured room."""
        short = 0
        if self._budget:
            short = self._lora_resident(live) + need - self._budget
        room = self._measured_room(owner, head, live)
        if room is not None:
            short = max(short, need - room)
        return max(0, short)

    def _shed_for(self, owner, nbytes: int, hold, store) -> None:
        """Shed ``nbytes`` of idle LoRA bytes, after _lock dropped: the
        owner's own request bytes (``hold`` kept), its worker's other
        entries' by their last use, then the idle uploads (``store``
        kept). The store's own turn sheds the uploads first. The other
        worker's caches are never shed here, where its job may read them;
        an upload is its body's bytes, no worker's arrays, so any turn
        sheds it."""
        shed = self._lora_shed or _default_lora_shed
        store_shed = self._store_shed or _default_store_shed
        left = int(nbytes)
        kind = _owner_kind(owner)
        with self._lock:
            others = sorted((e for e in self._entries.values()
                             if e.lora is not None and e is not owner
                             and _mine(e, kind)),
                            key=lambda e: e.seq)
        order = ([STORE] + others if owner is STORE
                 else [owner] + others + [STORE])
        for src in order:
            if left <= 0:
                return
            if src is STORE:
                left -= int(store_shed(left, tuple(store)) or 0)
                continue
            mine = src is owner
            left -= int(shed(src, left, tuple(hold) if mine else (), mine)
                        or 0)
            figs = self._read_lora(src)
            with self._lock:
                if figs is not None and src.lora is not None \
                        and self._in_pool(src):
                    src.lora.set(figs)
                    src.lora.base = src.lora.held
                self._publish()

    @staticmethod
    def _own(owner, beside) -> list:
        """The entries a turn's job holds as its own: its owner, past the
        upload store, and ``beside``."""
        return ([] if owner is STORE else [owner]) + list(beside)

    def _hopeless_413(self, budget, entries, owner, beside, n: int,
                      subject: str) -> None:
        """The 413 of ``n`` bytes no unload makes room for: over the budget
        less the footprints of the job's own entries and the pinned
        models."""
        if not budget:
            return
        own = self._own(owner, beside)
        ids = {id(e) for e in own}
        kept = (sum(e.footprint for e in own)
                + sum(e.footprint for e in entries
                      if e.pinned and id(e) not in ids))
        if n <= budget - kept:
            return
        names = ", ".join(repr(self._lora_name(e)) for e in own)
        where = f"{names} and the pinned models" if names else \
            "the pinned models"
        raise LoraRefused(
            413, f"{subject} need {gb(n, True)} GB of memory, over the "
            f"{gb(budget - kept, False)} GB the server's memory budget "
            f"leaves beside {where}")

    def _lora_503(self, subject: str, n: int, room: int, retry: int,
                  own=(), measured: bool = False,
                  held_up: str | None = None) -> "LoraRefused":
        """Under _lock: the 503 of a turn left short. It names the cold
        load in flight, else what held up a turn with room (``held_up``),
        else the room the budget leaves beside the models in use whose
        bytes go once idle, past the job's own entries ``own``, or,
        ``measured``, the working set's."""
        lead = f"{subject} need {gb(n, True)} GB of memory"
        if self._building:
            return LoraRefused(
                503, f"{lead}, and the server is loading {self._loading} "
                f"({gb(self._building, True)} GB) now; retry when it is "
                "loaded", retry, transient=True)
        free = f"{gb(room, False)} GB is free"
        transient = True
        if held_up is not None:
            tail = held_up
        elif measured:
            busy = self._lora_busy(own)
            transient = bool(busy)
            tail = (f"{free} in the working set now; retry when the models "
                    "in use finish" if busy else
                    f"{free} in the working set now; close other apps or "
                    "unload a model to make room")
        else:
            busy = self._lora_busy(own, pinned=False)
            transient = bool(busy)
            tail = (f"{free} beside the models in use ({', '.join(busy)}); "
                    "retry when one is idle" if busy else
                    f"{free}; unload a model or retry when the requests in "
                    "flight finish")
        return LoraRefused(503, f"{lead}, and {tail}", retry,
                           transient=transient)

    def _aside_503(self, subject: str, n: int, room: int, retry: int,
                   es, ids, kind: str, short: int):
        """Under _lock: the 503 of a turn of worker ``kind`` that the idle
        entries of the other worker would make ``short`` bytes of room
        for, naming them, or None when they would not. A turn never waits
        on the other worker, so it evicts none of them."""
        names, freed = [], 0
        for e in sorted((e for e in es if e.is_image and e.kind != kind
                         and id(e) not in ids and not e.pinned
                         and e.busy == 0), key=lambda e: e.seq):
            if freed >= short:
                break
            names.append(repr(self._lora_name(e)))
            freed += self._entry_bytes(e)
        if not names or freed < short:
            return None
        other = next(k for k in _WORKER_ORDER if k != kind)
        them = "it" if len(names) == 1 else "them"
        verb = "is" if len(names) == 1 else "are"
        return LoraRefused(
            503, f"{subject} need {gb(n, True)} GB of memory, "
            f"{gb(room, False)} GB is free, and {', '.join(names)} {verb} "
            f"idle on the {_WORKER_WORDS[other]} worker; unload {them} "
            "(POST /unload) to make room", retry)

    def _refusal(self, owner, need, head, live, held, subject, retry,
                 beside):
        """Under _lock, in the hold that found the turn short: raise the
        413 when no unload makes room, the 503 at once when no chat model
        in use holds the bytes, else return (the release count, the 503)
        for the turn to wait on. The entries the job holds are its own,
        never waited for."""
        budget = self._budget
        es = list(self._entries.values())
        self._hopeless_413(budget, es, owner, beside, need + held, subject)
        own = self._own(owner, beside)
        ids = {id(e) for e in own}
        kind = _owner_kind(owner)
        chat = [e for e in es if id(e) not in ids and not e.pinned
                and not e.is_image and e.busy > e.retained]
        resident = self._lora_resident(live)
        if budget and resident + need > budget:
            # The idle entries this turn can evict: chat and its worker's.
            idle = sum(self._entry_bytes(e) for e in es
                       if id(e) not in ids and not e.pinned and e.busy == 0
                       and _mine(e, kind))
            freed = idle + sum(self._entry_bytes(e) for e in chat)
            short = resident + need - budget
            if freed < short:
                aside = self._aside_503(subject, need,
                                        max(0, budget - resident), retry,
                                        es, ids, kind, short - freed)
                if aside is not None:
                    raise aside
            err = self._lora_503(subject, need, max(0, budget - resident),
                                 retry, own)
            if not chat or freed < short:
                raise err
            return self._releases, err
        # The measured room is short: the budget passed in this hold.
        room = self._measured_room(owner, head, live)
        err = self._lora_503(subject, need, room or 0, retry, own,
                             measured=True)
        if not chat:
            raise err
        return self._releases, err

    def _late_503(self, owner, need: int, subject: str, retry: int, head,
                  beside=(), worker: bool = False) -> "LoraRefused":
        """Under _lock: the 503 of a turn that never reached its room
        check by its deadline, from the pool now and the measure ``head``
        read before the hold: the cold load in flight, else the budget
        short, else the measured room short, else what held the turn up,
        its worker's job (``worker``) or the build lock."""
        own = self._own(owner, beside)
        live = self._uploads()
        free = self._budget - self._lora_resident(live)
        if self._budget and need > free:
            return self._lora_503(subject, need, max(0, free), retry, own)
        room = self._measured_room(owner, head, live)
        if room is not None and need > room:
            return self._lora_503(subject, need, max(0, room), retry, own,
                                  measured=True)
        if worker:
            kind = _owner_kind(owner)
            names = [self._lora_name(e) for e in self._entries.values()
                     if e.kind == kind and e.busy > e.retained]
            held_up = (f"the {_WORKER_WORDS[kind]} worker is busy"
                       + (f" with {', '.join(names)}" if names else "")
                       + "; retry when it is idle")
        else:
            held_up = ("the server is loading or unloading another model "
                       "now; retry when it is done")
        return self._lora_503(subject, need, 0, retry, own, held_up=held_up)

    def lora_late(self, owner, need: int, subject: str) -> None:
        """The 503 of a turn that its worker did not start within
        its deadline. The measure and the deadline are read before the
        pool's lock."""
        from .capacity import overcommit

        retry = self._retry()
        head = None if overcommit() else self._measured_lora_room()
        with self._lock:
            raise self._late_503(owner, int(need), subject, retry, head,
                                 worker=True)

    def lora_settle(self, entry, subject: str) -> None:
        """A job's settle, on the worker in its finally: the ledger set to
        what the caches hold once it ran, its booking ended. A ledger past
        its booking logs one line."""
        figs = self._read_lora(entry)
        with self._lock:
            over = self._reconcile(entry, figs)
            self._publish()
        self._overrun(subject, over)

    def lora_ledger(self, entry) -> tuple:
        """(pending, {key: bytes}) of the ledger of ``entry``."""
        with self._lock:
            led = entry.lora
            if led is None:
                return 0, {}
            return led.pending, dict(led.keys)

    def lora_hopeless(self, owner, n: int, subject: str,
                      beside=()) -> None:
        """A route's check of ``n`` bytes its job's turn would book, from
        ``figures`` with no lock: the 413 when no unload makes room beside
        the job's own entries, ``owner`` and ``beside``."""
        from .capacity import overcommit

        f = self.figures
        if overcommit():
            return
        self._hopeless_413(f.budget, f.entries, owner, tuple(beside),
                           int(n), subject)

    @staticmethod
    def _lora_name(e) -> str:
        return e.loaded_as or os.path.basename(str(e.model_path))

    def _measured_lora_room(self):
        """The free working set measured now, against the serve ceiling,
        or None where it cannot be measured. Read with no pool lock."""
        self._hit("measure")
        if self._headroom_fn is not None:
            return self._headroom_fn()
        from gmlx.gen.prefill_decay import headroom_bytes

        from .capacity import working_budget_bytes, working_set_bytes

        head = headroom_bytes()
        if head is None:
            return None
        budget, ws = working_budget_bytes(), working_set_bytes()
        if budget is not None and ws is not None:
            head -= max(0.0, ws - budget)
        return head

    def _lora_busy(self, own=(), pinned: bool = True) -> list:
        """Under _lock: the names of the models in use beside the entries
        ``own`` a job holds, the pinned ones only with ``pinned``."""
        ids = {id(e) for e in own}
        return [self._lora_name(e) for e in self._entries.values()
                if e.busy > e.retained and id(e) not in ids
                and (pinned or not e.pinned)]

    def _torn(self, kind, n: int) -> None:
        """Under _lock: ``n`` tearing bytes, of a pipeline of worker
        ``kind`` or of a chat model (None), are gone. A turn that waits on
        the other worker's release wakes."""
        self._tearing = max(0, self._tearing - n)
        if kind is not None:
            left = max(0, self._tearing_kind[kind] - n)
            self._freed_kind[kind] += self._tearing_kind[kind] - left
            self._tearing_kind[kind] = left
        self._publish()
        if n:
            self._idle.notify_all()

    def _has_aside(self, owner) -> bool:
        with self._lock:
            return self._aside(owner) > 0

    def _aside(self, owner) -> int:
        """Under _lock: the tearing bytes of the pipelines the other worker
        than the turn's of ``owner`` releases. With the build lock held,
        each is a release queued there by a teardown done after it dropped
        the lock, which frees its bytes once it runs."""
        kind = _owner_kind(owner)
        return sum(n for k, n in self._tearing_kind.items() if k != kind)

    def _aside_freed(self, owner) -> int:
        """Under _lock: the running count of the bytes the releases on the
        other worker than the turn's of ``owner`` freed."""
        kind = _owner_kind(owner)
        return sum(n for k, n in self._freed_kind.items() if k != kind)


def _default_lora_shed(entry, nbytes: int, hold, own: bool) -> int:
    from . import image_service

    return image_service.shed_lora(entry, nbytes, hold, own)


def _default_store_shed(nbytes: int, hold=()) -> int:
    from .image_steps import STATE

    return STATE.adapters.shed(nbytes, hold)


def _default_store_live() -> int:
    from .image_steps import STATE

    return STATE.adapters.bytes
