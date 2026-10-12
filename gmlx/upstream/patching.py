"""Install-once patching helpers."""

from __future__ import annotations


_LINKS = ("__wrapped__", "_gmlx_orig")
_MAX_LAYERS = 256


def _walk(fn):
    """``fn`` and every function under it, each once. A cycle, or a chain
    of attributes generated on access, ends the walk."""
    seen: set[int] = set()
    todo = [fn]
    while todo and len(seen) < _MAX_LAYERS:
        fn = todo.pop()
        if fn is None or id(fn) in seen:
            continue
        seen.add(id(fn))
        yield fn
        todo.extend(getattr(fn, name, None) for name in _LINKS)


def root(fn):
    """The function at the bottom of the wrap chain over ``fn``: the one
    that wraps nothing. ``fn`` itself when it wraps nothing."""
    for f in _walk(fn):
        if all(getattr(f, name, None) is None for name in _LINKS):
            return f
    return fn


def wrapped_with(fn, flag: str) -> bool:
    """Whether ``fn``, or any function it wraps, carries ``flag``.

    Several installers wrap the same upstream function, and some run again
    on every model load. An installer that reads its flag on the outermost
    function only misses it when another wrapper sits on top, and adds a
    second layer. Each wrapper in such a chain links to what it wraps
    through ``__wrapped__`` (``functools.wraps``) or ``_gmlx_orig``
    (``link``), and the walk follows both."""
    return any(getattr(f, flag, False) for f in _walk(fn))


def link(wrapper, orig, flag: str | None = None):
    """Link ``wrapper`` to ``orig``, the function it wraps, so that
    ``wrapped_with`` and ``under`` see through it, and set ``flag`` on it
    when given. Returns ``wrapper``."""
    wrapper.__dict__["_gmlx_orig"] = orig
    if flag is not None:
        wrapper.__dict__[flag] = True
    return wrapper


def _inner(fn):
    """The function ``fn`` wraps directly. A wrapper that copies another's
    ``__dict__`` (``functools.wraps`` does) also carries that one's link,
    which points further down, so of two links the upper one is direct."""
    a = getattr(fn, "__wrapped__", None)
    b = getattr(fn, "_gmlx_orig", None)
    if a is None or b is None or a is b:
        return b if a is None else a
    return a if any(f is b for f in _walk(a)) else b


def under(fn, flag: str):
    """The function under the lowest layer of ``fn`` that carries
    ``flag``, or ``fn`` when no layer does. An installer that replaces its
    own layer, or reads past it, takes this as the original instead of
    unwrapping the outermost function one level, which drops or misreads
    a wrap another installer put on top."""
    below = fn
    for _ in range(_MAX_LAYERS):
        if fn is None:
            break
        nxt = _inner(fn)
        if getattr(fn, flag, False) and nxt is not None:
            below = nxt
        fn = nxt
    return below


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
