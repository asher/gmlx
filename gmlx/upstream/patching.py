"""Install-once class-method patching."""

from __future__ import annotations


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
