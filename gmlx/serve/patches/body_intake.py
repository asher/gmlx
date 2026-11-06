"""The request bodies the media gate leaves unread, read by their routes.

The media gate (``media_gate``) reads every other body whole before its
route runs, to a fixed bound. The routes here take bodies whose size is
the user's payload, such as the images API's input images. Each prices
its body before its first byte is read, from its Content-Length, against
the bytes free beside the models loaded, read from the pool's figures
with no lock. The price is an estimate and books nothing. A body the
machine cannot hold is refused with both figures, and none of it is
read. A body with no Content-Length is read until it passes the figure,
and refused there. The gate's own checks of the media a body names run
here, on the body read, as the gate runs them on every other route.
"""

from __future__ import annotations

from starlette.requests import Request

# The routes whose bodies the gate leaves to them: POST only. The images
# API is served with and without the /v1 prefix.
IMAGE_PATHS = frozenset(f"{p}/images/{r}" for p in ("/v1", "")
                        for r in ("generations", "edits", "variations"))


class BodyRefused(Exception):
    """A body a route does not read: ``status``, the message, and the
    request field it names as ``param`` when there is one."""

    def __init__(self, status: int, message: str, param: str | None = None):
        super().__init__(message)
        self.status = status
        self.param = param

    @property
    def headers(self) -> dict:
        """A 413 leaves the body unread past its figure: the answer ends
        the connection."""
        return {"Connection": "close"} if self.status == 413 else {}


def reads_own_body(request) -> bool:
    """Whether the route of ``request`` reads and prices its own body, so
    the gate leaves it unread."""
    return request.method == "POST" and request.url.path in IMAGE_PATHS


def _mib(n: int) -> str:
    """``n`` bytes in GiB, MiB or KiB, whichever it reaches."""
    for unit, shift in (("GiB", 30), ("MiB", 20)):
        if n >= 1 << shift:
            return f"{n / (1 << shift):.{2 if shift == 30 else 1}f} {unit}"
    return f"{n / 1024:.1f} KiB"


def free_bytes() -> int | None:
    """The bytes free beside what the pool holds and the uploads admitted,
    from its figures with no lock, or None with no budget to price by
    (no pool, no working budget, or GMLX_OVERCOMMIT set)."""
    from ..capacity import overcommit

    if overcommit():
        return None
    from ._common import _get_pool

    pool = _get_pool()
    if pool is None:
        return None
    f = pool.figures
    if not f.budget:
        return None
    return max(0, f.budget - f.resident - f.incoming)


def content_length(request) -> int | None:
    """The body's Content-Length, or None with none sent."""
    length = request.headers.get("content-length")
    if length is None:
        return None
    try:
        n = int(length)
    except ValueError:
        raise BodyRefused(400, "the Content-Length is not a number") from None
    if n < 0:
        raise BodyRefused(400, "the Content-Length is not a number")
    return n


def _over_free(n: int, free: int, sent: bool) -> BodyRefused:
    size = f"{_mib(n)}" if sent else f"over {_mib(n)}"
    return BodyRefused(413, f"the request body is {size}, and {_mib(free)} "
                       "of memory is free beside the models loaded; send a "
                       "smaller body, or unload a model to make room")


async def read_in_ram(request: Request, free: int | None = None) -> bytes:
    """The body of ``request``, priced against ``free`` bytes (the pool's
    free figure when None) before a byte of it is read. A body with no
    Content-Length is refused once what arrived passes the figure. The
    body read is kept on the request, so ``request.form()`` and
    ``request.json()`` parse it after."""
    if free is None:
        free = free_bytes()
    n = content_length(request)
    if n is not None and free is not None and n > free:
        raise _over_free(n, free, True)
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if free is not None and size > free:
            raise _over_free(free, free, False)
        chunks.append(chunk)
    raw = b"".join(chunks)
    request._body = raw
    return raw


async def check_media(request: Request, raw: bytes) -> None:
    """The media gate's checks of the body ``raw``, as the gate runs them
    on every route it reads: a media reference the server does not take,
    a form sent to a route that takes JSON, a body that is not JSON. The
    body was priced whole, so an inline image takes no byte limit of its
    own; the item count and the pixel limits hold."""
    from ._common import SESSION_BODY_KEY, SESSION_SCOPE_KEY
    from .media_gate import MediaRefused, body_priced, check_request

    session = request.scope.get(SESSION_SCOPE_KEY) is not None
    parsed = request.scope.pop(SESSION_BODY_KEY, None)
    try:
        with body_priced():
            await check_request(request.method, request.url.path,
                                request.headers, raw, inline_only=session,
                                parsed=parsed)
    except MediaRefused as e:
        raise BodyRefused(400, str(e), e.param) from None
