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


def test_a_refusal_rounds_the_need_up_and_the_figure_down():
    """A body one byte past the figure never reads as the figure."""
    assert bi._mib((1 << 30) + 1, True) == "1.01 GiB"
    assert bi._mib((1 << 30) - 1) == "1023.9 MiB"
    assert str(bi._over_free((1 << 30) + 1, 1 << 30, True)).startswith(
        "the request body is 1.01 GiB, and 1.00 GiB of memory is free")
    assert str(bi._over_room((8 << 30) + 100_000, 8 << 30)).startswith(
        "the files of this job are 8.01 GiB, and the video job store has "
        "8.00 GiB of room left;")
    assert bi._mib((2 << 20) - 1) == "1.9 MiB"
    assert bi._mib((2 << 20) - 1, True) == "2.0 MiB"
    assert bi._mib(1023, True) == "1.0 KiB"
