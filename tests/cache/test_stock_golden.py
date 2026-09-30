# vendor_upstream: skip
"""load_golden refuses a golden recorded from another script, another
version of the stock packages or another mlx, and names the recorder."""
from __future__ import annotations

import json

import pytest
import stock_golden


def _write(tmp_path, monkeypatch, **recorded):
    monkeypatch.setattr(stock_golden, "GOLDEN", tmp_path)
    golden = {"script_sha256": stock_golden._sha("script"),
              "recorded": {"packages": stock_golden._copied_versions(),
                           "mlx": stock_golden._mlx_version(), **recorded}}
    (tmp_path / "probe.json").write_text(json.dumps(golden))


def test_a_golden_of_this_script_copy_and_mlx_loads(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch)
    assert stock_golden.load_golden("probe", "script")["script_sha256"]


@pytest.mark.parametrize("recorded, names", [
    ({"mlx": "0.0.1"}, ("mlx 0.0.1", "this run has mlx")),
    ({"packages": {"mlx_vlm": "0.0.2", "mlx_lm": "0.0.3"}},
     ("'mlx_vlm': '0.0.2'", "the copy holds")),
])
def test_a_golden_of_another_version_names_it_and_the_recorder(
        tmp_path, monkeypatch, recorded, names):
    _write(tmp_path, monkeypatch, **recorded)
    with pytest.raises(AssertionError) as err:
        stock_golden.load_golden("probe", "script")
    for name in (*names, "record it again with tests/cache/stock_golden.py"):
        assert name in str(err.value)


def test_a_golden_of_another_script_is_refused(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch)
    with pytest.raises(AssertionError, match="another version of its script"):
        stock_golden.load_golden("probe", "another script")
