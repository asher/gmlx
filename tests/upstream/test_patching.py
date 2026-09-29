"""ClassPatch: install-once semantics and the sticky stock capture.
wrapped_with: an installer finds its own layer under another one."""

from __future__ import annotations

import pytest

from gmlx.upstream.patching import ClassPatch


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


def test_wrapped_with_reads_every_layer():
    from gmlx.upstream.patching import wrapped_with

    def stock():
        return None

    def inner():
        return None

    def outer():
        return None

    inner.__dict__["_a"] = True
    inner.__wrapped__ = stock  # type: ignore[attr-defined]
    outer.__wrapped__ = inner  # type: ignore[attr-defined]
    assert wrapped_with(outer, "_a") and not wrapped_with(outer, "_b")
    stock.__wrapped__ = outer  # type: ignore[attr-defined]
    assert not wrapped_with(outer, "_b")  # a cycle ends the walk


def _render_layers():
    """Frames named get_chat_template on one render: each wrapper layer is
    one, and the stock stand-in is the last."""
    import sys

    frame, n = sys._getframe(1), 0
    while frame is not None:
        n += frame.f_code.co_name == "get_chat_template"
        frame = frame.f_back
    return n


@pytest.mark.parametrize("reasoning_first", [True, False])
def test_loads_keep_one_get_chat_template_layer_per_installer(
        monkeypatch, reasoning_first):
    """The VLM loader installs the thinking default and the system content
    fix on every load, and the server installs both again. Each wrapped the
    other's layer, so every load added two."""
    pu = pytest.importorskip("mlx_vlm.prompt_utils")
    from gmlx.tui.reasoning import install_template_default_thinking
    from gmlx.upstream.system_content_fix import install_system_content_fix

    seen = []

    def get_chat_template(processor, messages, add_generation_prompt,
                          tokenize=False, **kwargs):
        seen.append(_render_layers())
        return ""

    monkeypatch.setattr(pu, "get_chat_template", get_chat_template)
    installers = [install_template_default_thinking,
                  install_system_content_fix]
    if not reasoning_first:
        installers.reverse()
    for _load in range(5):
        for install in installers:
            install()
        pu.get_chat_template(None, [], True)
    assert seen == [3] * 5
