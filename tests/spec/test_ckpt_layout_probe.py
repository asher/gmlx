"""_ckpt_layout_for failure handling: a broken probe reads as "no ckpt
signature", never as a valid empty layout, and it says so in the log."""

from __future__ import annotations

from types import SimpleNamespace

from gmlx.spec import engine


class _Broken:
    def make_cache(self):
        raise RuntimeError("cache probe broke")


def test_probe_failure_returns_none_and_warns(caplog):
    model = SimpleNamespace(language_model=_Broken())
    with caplog.at_level("WARNING", logger="gmlx.spec.engine"):
        assert engine._ckpt_layout_for(model) is None
    assert any("ckpt layout probe failed" in r.message for r in caplog.records)
    # The failure is stashed; the re-read stays None without re-probing.
    model.language_model = None
    assert engine._ckpt_layout_for(model) is None
