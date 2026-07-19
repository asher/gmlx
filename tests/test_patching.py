"""ClassPatch: install-once semantics and the sticky stock capture."""

from __future__ import annotations

from gmlx.patching import ClassPatch


def test_install_once_and_stock_capture():
    class V:
        def hello(self):
            return "stock"

    cp = ClassPatch()
    original = V.hello
    cp.install(V, "hello", lambda self: "patched")
    assert cp.installed and cp.stock is original
    assert V().hello() == "patched"
    cp.install(V, "hello", lambda self: "other")   # double install is a no-op
    assert V().hello() == "patched"


def test_reset_then_reinstall_keeps_original_stock():
    # Tests reset .installed and re-run an installer (the envflags kill-site
    # loop does exactly this); stock must stay the genuine original - a
    # re-capture would store the replacement as its own fallback and recurse.
    class V:
        def hello(self):
            return "stock"

    cp = ClassPatch()
    original = V.hello

    def patched(self):
        return "patched:" + cp.stock(self)

    cp.install(V, "hello", patched)
    cp.installed = False
    cp.install(V, "hello", patched)
    assert cp.stock is original
    assert V().hello() == "patched:stock"
