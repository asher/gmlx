"""Bound replay of a verify: its records go onto command buffers of ours.

``GMLX_SPEC_VERIFY_BOUND`` (on by default, ``0`` off; needs a kq build
with ``bound_call``). A B=1 verify that its caller submits at once
(``gmlx.spec.verify_submit.now``) does not build one mlx node a record and
hand the graph to ``mx.async_eval``: each record that has its program is
encoded as it is reached, onto a command buffer of a kq bound session, and
the buffers are committed ahead of fixed layers. The Metal calls are the
ones the replay inside an eval makes, on the same operands, so the bits are
the same; only the host path and the buffer boundaries differ.

What stays mlx. A record input that is still lazy (the embedding, the
table rows, a stage built from primitives between two records) is
submitted in one ``mx.async_eval`` after our open buffer is committed, and
the record is bound after it. A record still learning its program
or with an input laid out another way runs in mlx's eval, and the record
after it submits it as a lazy input. A record that cannot be bound for
another reason (it never replays direct, a parameter out of its range)
ends the bound part of that verify: our buffer is committed and the rest of
the verify is the lazy graph of today, evaluated by the caller.

Host reads. A bound output carries no mlx event (libmlx does not export
the constructor), so a host read of one, or an eval of views of it alone,
does not wait for our buffer; the round reads the verify through the head
(a kernel, which waits). ``wait()`` is there for any other reader.

Order. An output of a bound record is an array in flight, as after
``mx.async_eval``. mlx orders GPU work by fences between encoders, not by
commit order; a session encoder waits for the writers of its inputs and
enters its outputs in mlx's map when it ends, so an mlx eval made after
that waits for it. The open buffer is therefore committed ahead of every
mlx eval this module makes and at the end of the forward, and code that
runs between two bound records must not evaluate an array that depends on
the open buffer.

Failure. kq handles a command buffer of the session that ends in error as
mlx 0.32 handles its own: the completion handler keeps the error, worded as
mlx words it, and the session raises it on its caller's thread. The round
loop calls ``check()`` after its host read of the verify and ahead of any
use of the round's tokens (``speculative_round_check``), so tick_guard takes
the RuntimeError as it takes mlx's with the lever off. A session that
raised is closed, and the thread's next verify opens a new one.

Lifetime. A thread keeps one session, and kq keeps its pool of buffers
between kernels, as large as the largest verify it ran, until it is
closed. The thread-local holder closes it when it goes: its thread ended
(a server's generation worker, one a model load) or the thread-local was
replaced.
"""

from __future__ import annotations

import atexit
import os
import sys
import threading
import time

import mlx.core as mx

from gmlx.envflags import env_bool, env_int
from gmlx.spec.verify_submit import is_now
from gmlx.upstream import gdn_patches as _gdn

_MODE = env_int("GMLX_SPEC_VERIFY_BOUND", 1)
_STATS = env_bool("GMLX_SPEC_VERIFY_BOUND_STATS", False)
# Host seconds by piece in the stats (a harness sets it): the encodes, the
# mlx submissions, the commits, the prepare pass.
_TIMES = False
# GMLX_SPEC_VERIFY_BOUND_TRACE=PATH: a line per commit and mlx submission
# (perf_counter seconds, what, layer), to put names on command buffer stamps.
_TRACE_PATH = os.environ.get("GMLX_SPEC_VERIFY_BOUND_TRACE")
_trace_lines: list = []


def _trace(what: str, li: int) -> None:
    _trace_lines.append(f"{time.perf_counter():.6f} {what} {li}\n")
    if len(_trace_lines) >= 4096:
        _trace_flush()


def _trace_flush() -> None:
    if _trace_lines and _TRACE_PATH:
        with open(_TRACE_PATH, "a") as f:
            f.writelines(_trace_lines)
    _trace_lines.clear()


def _parse_marks(text: str) -> frozenset:
    return frozenset(int(p) for p in text.split(",") if p.strip().isdigit())


# Layers a buffer of ours ends ahead of. A layer runs about ten times longer
# on the GPU than its encode takes on the host, so short buffers first and
# doubling ones after keep committed work ahead of the GPU at every width.
_MARKS = _parse_marks(
    os.environ.get("GMLX_SPEC_VERIFY_BOUND_MARKS", "1,2,4,8,16,32"))

_local = threading.local()
_kq_mod: list = [None]
stats = {"verifies": 0, "whole": 0, "bound_calls": 0, "lazy_calls": 0,
         "commits": 0, "submits": 0, "copies": 0, "ended": {}, "lazy_in": {},
         "t_encode": 0.0, "t_submit": 0.0, "t_commit": 0.0,
         "t_prepare": 0.0, "qsa": {}, "lazy_rec": {}}


def _kq():
    if _kq_mod[0] is None:
        try:
            import mlx_kquant as kq

            _kq_mod[0] = kq if hasattr(kq, "bound_call") else False
        except ImportError:
            _kq_mod[0] = False
    return _kq_mod[0]


def mode() -> int:
    return _MODE


def on() -> bool:
    """The bound verify can run: the lever on and a kq with bound
    sessions."""
    return _MODE > 0 and bool(_kq())


_bound_kw: list = [None]


def record_kw() -> dict:
    """Keywords for kq.fuse_graph of a verify record while the lever is
    on: a record a session replays (kq lists more primitive types for it).
    Empty with the lever off or a kq without them."""
    if _MODE <= 0:
        return {}
    if _bound_kw[0] is None:
        try:
            import mlx_kquant as kq

            doc = (getattr(kq.fuse_graph, "__doc__", None) or "").split("\n")[0]
        except ImportError:
            doc = ""
        _bound_kw[0] = "bound" in doc
    return {"bound": True} if _bound_kw[0] else {}


def wanted() -> bool:
    return bool(_MODE > 0 and is_now())


class _Held:
    """A thread's kq bound session, closed when the thread-local lets it
    go."""

    __slots__ = ("kq", "sid")

    def __init__(self, kq, sid: int):
        self.kq, self.sid = kq, sid

    def close(self) -> None:
        kq, self.kq = self.kq, None
        if kq:
            kq.bound_session_close(self.sid)

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: S110 (kq may be gone at exit)
            pass


class Session:
    """One verify forward. ``call`` stands in for ``kq.fused_call``."""

    __slots__ = ("kq", "sid", "live", "open", "pending", "li", "bound",
                 "lazy", "sent")

    def __init__(self, kq, sid: int):
        self.kq, self.sid = kq, sid
        self.live = True        # records still go to our buffers
        self.open = False       # our buffer holds calls not yet committed
        self.pending: list = []  # lazy arrays to submit with the next mlx eval
        self.li = -1
        self.bound = 0
        self.lazy = 0
        self.sent = False       # a buffer of ours was committed

    def layer(self, li: int) -> None:
        """The forward reached layer ``li``."""
        self.li = li
        if self.open and li in _MARKS:
            self.flush()

    def flush(self) -> None:
        if self.open:
            t0 = time.perf_counter() if _TIMES else 0.0
            self.sent = True
            try:
                self.kq.bound_commit(self.sid)
            except RuntimeError:
                self._failed()
                raise
            if _TRACE_PATH:
                _trace("commit", self.li)
            self.open = False
            stats["commits"] += 1
            if _TIMES:
                stats["t_commit"] += time.perf_counter() - t0

    def submit(self, arrays) -> None:
        """One mlx submission of ``arrays`` and whatever waits, after our
        open buffer."""
        self.flush()
        todo = [a for a in arrays if isinstance(a, mx.array)] + self.pending
        self.pending = []
        if todo:
            t0 = time.perf_counter() if _TIMES else 0.0
            if _TRACE_PATH:
                _trace("submit", self.li)
            mx.async_eval(*todo)
            if _TRACE_PATH:
                _trace("submitted", self.li)
            stats["submits"] += 1
            if _TIMES:
                stats["t_submit"] += time.perf_counter() - t0

    def _failed(self) -> None:
        # kq raised the failure of a buffer of the session (or an mlx
        # submission between records raised): the session is closed.
        self.live = self.open = False
        self.pending = []
        _drop()

    def _end(self, why) -> None:
        self.flush()
        self.live = False
        key = (self.li, why)
        stats["ended"][key] = stats["ended"].get(key, 0) + 1

    def call(self, handle: int, ins: list):
        kq = self.kq
        if not self.live:
            self.lazy += 1
            return kq.fused_call(handle, ins)
        t0 = time.perf_counter() if _TIMES else 0.0
        try:
            r = kq.bound_call(self.sid, handle, ins)
            if _TIMES:
                stats["t_encode"] += time.perf_counter() - t0
            if type(r) is int and r == 3:
                if _TIMES:
                    key = (self.li,
                           kq.bound_session_info(self.sid)["last_lazy_input"])
                    stats["lazy_in"][key] = stats["lazy_in"].get(key, 0) + 1
                # A lazy input (the embedding, the table rows, a stage built
                # from primitives): one mlx submission, then the record.
                self.submit(ins)
                r = kq.bound_call(self.sid, handle, ins)
        except RuntimeError:
            if self.live:
                self._failed()
            raise
        if type(r) is int:
            self.lazy += 1
            if r == 5 or (r == 4 and kq.fused_direct_info(handle).get(
                    "state") in ("new", "recorded")):
                # A record still learning its program, or an input laid out
                # another way: this one runs in mlx's eval, and the record
                # that reads it submits it as a lazy input.
                key = (self.li, r)
                stats["lazy_rec"][key] = stats["lazy_rec"].get(key, 0) + 1
                return kq.fused_call(handle, ins)
            self._end(r)
            return kq.fused_call(handle, ins)
        self.open = True
        self.bound += 1
        return r

    def copy(self, arrays: list):
        """Copies of ``arrays`` on our buffer (kq ``bound_copy``), or None
        where kq cannot (a lazy array): the caller copies through mlx."""
        if not self.live or not arrays:
            return None
        try:
            r = self.kq.bound_copy(self.sid, arrays)
        except RuntimeError:
            self._failed()
            raise
        if type(r) is int:
            return None
        self.open = True
        stats["copies"] += len(r)
        return r

    def close(self) -> None:
        self.flush()
        # Arrays nothing bound took along stay lazy inputs of the caches,
        # as without the lever.
        self.pending = []
        stats["verifies"] += 1
        stats["bound_calls"] += self.bound
        stats["lazy_calls"] += self.lazy
        if self.live and self.bound:
            stats["whole"] += 1


def current():
    return getattr(_local, "session", None)


def begin(B: int):
    """A session for a verify forward, or None: the lever is off, the
    caller did not ask, the batch is not 1."""
    if not wanted() or B != 1 or mx.default_device() != mx.gpu:
        return None
    kq = _kq()
    if not kq or current() is not None:
        return None
    # The check a round left out (it raised first) is made here: no verify
    # starts on a session one of whose buffers failed.
    check()
    held = getattr(_local, "sid", None)
    if held is None or held.kq is not kq:
        held = _local.sid = _Held(kq, kq.bound_session())
        # Step counts the records read as they are (gdn_patches).
        _gdn.HOST_STEPS = True
    s = Session(kq, held.sid)
    _local.session = s
    if _TRACE_PATH:
        _trace("begin", -1)
    return s


def end(s: Session) -> None:
    try:
        s.close()
        if _TRACE_PATH:
            _trace("end", s.li)
    finally:
        _local.session = None
        held = getattr(_local, "sid", None)
        if s.sent and held is not None and held.sid == s.sid:
            _local.unchecked = held


def check() -> None:
    """After the round's host read of the verify, ahead of any use of its
    tokens: waits for the buffers the thread's session committed (they ran
    ahead of what the read waited for) and raises the error of one that
    failed, worded as mlx words its own. The session is then closed."""
    held = getattr(_local, "unchecked", None)
    if held is None:
        return
    _local.unchecked = None
    if not held.kq:
        return
    try:
        held.kq.bound_wait(held.sid)
    except RuntimeError:
        _drop()
        raise


def _drop() -> None:
    # Closing commits a buffer the session holds open, so no fence of it
    # in mlx's map is left without its update.
    held = getattr(_local, "sid", None)
    _local.sid = None
    _local.unchecked = None
    if held is not None:
        held.close()


def noting() -> bool:
    return _STATS or _TIMES


def note(layer, why: str) -> None:
    """With the stats on: what an attention layer of a verify took
    (the whole record, or why not), by layer ratio."""
    if _STATS or _TIMES:
        key = (getattr(layer.self_attn, "ratio", 0), why)
        q = stats["qsa"]
        q[key] = q.get(key, 0) + 1


def wait() -> None:
    """Block until the buffers this thread's session committed have run.
    For a host read of a bound output that no mlx kernel stands in front
    of: such an array carries no event, so reading its bytes (or an eval
    of views of it alone) does not wait for our buffer."""
    held = getattr(_local, "sid", None)
    if held is not None and held.kq:
        try:
            held.kq.bound_wait(held.sid)
        except RuntimeError:
            _drop()
            raise


def prepare(fn, *args) -> None:
    if not _TIMES:
        return fn(*args)
    t0 = time.perf_counter()
    fn(*args)
    stats["t_prepare"] += time.perf_counter() - t0


def call(handle: int, ins: list):
    """``kq.fused_call`` for a record of a verify layer: bound while a
    session is live."""
    s = getattr(_local, "session", None)
    if s is None:
        import mlx_kquant as kq

        return kq.fused_call(handle, ins)
    return s.call(handle, ins)


def _report() -> None:
    print("[q4] verify bound:", stats, file=sys.stderr, flush=True)


if _STATS and _MODE > 0:
    atexit.register(_report)
if _TRACE_PATH and _MODE > 0:
    atexit.register(_trace_flush)
