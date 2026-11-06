"""gmlx/serve/patches/body_intake.py: the figure a route prices its own
body against, the bytes free beside what the pool holds."""
from __future__ import annotations

from types import SimpleNamespace

from gmlx.serve.patches import _common
from gmlx.serve.patches import body_intake as bi


def test_the_free_figure_is_the_budget_less_the_resident_and_incoming(
        monkeypatch):
    """The resident bytes and the upload bodies admitted both count, and
    GMLX_OVERCOMMIT or a pool with no budget leaves the body unpriced."""
    figures = SimpleNamespace(budget=100 << 20, resident=30 << 20,
                              incoming=20 << 20)
    monkeypatch.setattr(_common, "_get_pool",
                        lambda: SimpleNamespace(figures=figures))
    monkeypatch.delenv("GMLX_OVERCOMMIT", raising=False)
    assert bi.free_bytes() == 50 << 20
    figures.incoming = 90 << 20
    assert bi.free_bytes() == 0
    monkeypatch.setenv("GMLX_OVERCOMMIT", "1")
    assert bi.free_bytes() is None
    monkeypatch.delenv("GMLX_OVERCOMMIT")
    figures.budget = 0
    assert bi.free_bytes() is None
