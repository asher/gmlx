"""Shared pieces of the routes that run a request as an engine job.

A job route holds its model, reads the request's adapter scales, and runs
its forwards on the model's engine thread (``engine_jobs``). The systemone
and scoring routes use these helpers. Scoring requests on one engine take
turns in arrival order; the waiters count toward the queue depth cap."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import threading
from typing import Callable, Optional

from fastapi import HTTPException

import gmlx.serve.bridge_vlm as serving


def request_lora(spec, body):
    """The request's adapter scales, or a 400 for a malformed field."""
    from gmlx.serve.lora_request import AdapterRequestError
    from gmlx.serve.lora_request import request_lora as parse

    try:
        return parse(spec, body)
    except AdapterRequestError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None


def salted(lora, rg):
    """``lora`` salted from the adapters ``rg``'s copy loaded; a missing
    identity is a server defect, a 500."""
    from gmlx.serve.bridge_vlm import loaded_adapters_of
    from gmlx.serve.lora_request import AdapterIdentityMissing
    from gmlx.serve.lora_request import salted as salt

    try:
        return salt(lora, loaded_adapters_of(rg))
    except AdapterIdentityMissing as e:
        raise HTTPException(status_code=500, detail=str(e)) from None


def resolve_engine(pick: Callable[[], str], profile, runtime, *,
                   decisions: bool = False):
    """Load the model ``pick()`` names and return ``(rg, hold, spec,
    model_str)``: the engine, its residency hold, the model spec and the
    model string. The caller releases ``hold``. ``decisions``: the route
    answers with the decision forward, so a decision model may load."""
    from gmlx.load.decision import decision_scope
    from gmlx.serve.residency import _http_from_resolver_error

    app_mod = importlib.import_module("gmlx._vendor.mlx_vlm.server.app")
    # The residency seam resolves the model string with the body profile
    # from this context variable.
    token = serving.set_request_profile(profile)
    try:
        try:
            model_str = pick()
        except (serving.ModelNotFound, serving.ModelFileMissing,
                serving.UnknownProfile, serving.NoModelSpecified) as e:
            raise _http_from_resolver_error(e) from e
        with decision_scope() if decisions else contextlib.nullcontext():
            app_mod.get_cached_model(model_str)
        spec = serving.get_active_spec()
    finally:
        serving.reset_request_profile(token)
    guard = runtime.response_generator
    rg = getattr(guard, "_rg", guard)
    hold = getattr(guard, "_hold", None)
    if rg is None:
        if hold is not None:
            hold.release()
        raise HTTPException(status_code=500, detail=(
            "the model engine is unavailable; restart the server"))
    return rg, hold, spec, model_str


_DISCONNECT_POLL_S = 0.5


class _Turn:
    """An engine's scoring turn: the lock, its waiters and its users."""

    __slots__ = ("lock", "waiters", "users")

    def __init__(self):
        self.lock = asyncio.Lock()
        self.waiters = 0
        self.users = 0


# id(engine) -> its turn, while a request holds it or waits for it.
_TURNS: dict[int, _Turn] = {}


def _key(rg) -> int:
    return id(getattr(rg, "_rg", rg))


def scoring_waiters(rg) -> int:
    """Scoring requests waiting for ``rg``'s engine, for the queue cap."""
    turn = _TURNS.get(_key(rg))
    return turn.waiters if turn is not None else 0


async def watch_disconnect(http_request, stop: threading.Event) -> None:
    """Set ``stop`` and return when the client disconnects. Returns when
    ``stop`` is set otherwise."""
    try:
        while not stop.is_set():
            if await http_request.is_disconnected():
                stop.set()
                return
            await asyncio.sleep(_DISCONNECT_POLL_S)
    except asyncio.CancelledError:
        pass


@contextlib.asynccontextmanager
async def scoring_turn(rg, watcher: Optional[asyncio.Task] = None):
    """Hold ``rg``'s scoring turn for the block. Requests wait on the event
    loop in the order they ask, so a waiter holds no thread. When
    ``watcher`` (``watch_disconnect``) ends first, the wait is dropped and
    ``JobCancelled`` raised; the next waiter keeps its place."""
    from gmlx.serve.engine_jobs import JobCancelled

    key = _key(rg)
    turn = _TURNS.get(key)
    if turn is None:
        turn = _TURNS[key] = _Turn()
    turn.users += 1
    try:
        turn.waiters += 1
        acquire = asyncio.ensure_future(turn.lock.acquire())
        granted = False
        try:
            done, _ = await asyncio.wait(
                {acquire} if watcher is None else {acquire, watcher},
                return_when=asyncio.FIRST_COMPLETED)
            granted = acquire in done
        finally:
            turn.waiters -= 1
            if not granted:
                if not acquire.cancel() and not acquire.cancelled() \
                        and acquire.exception() is None:
                    turn.lock.release()
        if not granted:
            raise JobCancelled("the client disconnected while waiting")
        try:
            yield
        finally:
            turn.lock.release()
    finally:
        turn.users -= 1
        if turn.users == 0 and _TURNS.get(key) is turn:
            del _TURNS[key]
