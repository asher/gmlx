"""Install-once patching helpers."""

from __future__ import annotations


_LINKS = ("__wrapped__", "_gmlx_orig")
_MAX_LAYERS = 256


def wrapped_with(fn, flag: str) -> bool:
    """Whether ``fn``, or any function it wraps, carries ``flag``.

    Several installers wrap the same upstream function, and some run again
    on every model load. An installer that reads its flag on the outermost
    function only misses it when another wrapper sits on top, and adds a
    second layer. Each wrapper in such a chain links to what it wraps
    through ``__wrapped__`` (``functools.wraps``) or ``_gmlx_orig``, and
    the walk follows both. A cycle, or a chain of attributes generated on
    access, ends the walk."""
    seen: set[int] = set()
    todo = [fn]
    while todo and len(seen) < _MAX_LAYERS:
        fn = todo.pop()
        if fn is None or id(fn) in seen:
            continue
        seen.add(id(fn))
        if getattr(fn, flag, False):
            return True
        todo.extend(getattr(fn, link, None) for link in _LINKS)
    return False


class ClassPatch:
    """One install-once class-method swap. ``install`` stashes the stock
    method on the first call and never reinstalls, so repeated loads in one
    process keep a single dispatch layer. ``stock`` is the pre-patch method
    (the replacement's fallback for unflagged instances); ``installed``
    makes "what's patched" introspectable and lets tests reset the seam.
    ``stock`` is captured once, ever: a reset-then-reinstall must not read
    the already-patched class attribute, or the replacement would become its
    own fallback and recurse."""

    def __init__(self) -> None:
        self.installed = False
        self.stock = None

    def install(self, cls, attr: str, replacement) -> None:
        if self.installed:
            return
        if self.stock is None:
            self.stock = getattr(cls, attr)
        setattr(cls, attr, replacement)
        self.installed = True
