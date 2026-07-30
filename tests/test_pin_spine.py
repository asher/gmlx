"""Spine pinning: non-expert range scan + mlock lifecycle."""

from __future__ import annotations

import numpy as np

from gmlx import pin_spine


def _write_model(path):
    from gguf import GGUFWriter

    w = GGUFWriter(str(path), "llama")
    w.add_uint32("llama.block_count", 2)
    for li in range(2):
        for kind in ("gate", "up", "down"):
            w.add_tensor(f"blk.{li}.ffn_{kind}_exps.weight",
                         np.zeros((4, 8, 16), dtype=np.float16))
        w.add_tensor(f"blk.{li}.attn_q.weight",
                     np.zeros((16, 16), dtype=np.float16))
    w.add_tensor("blk.0.ffn_gate_shexp.weight",
                 np.zeros((8, 16), dtype=np.float16))
    w.add_tensor("token_embd.weight", np.zeros((32, 16), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def test_spine_ranges_excludes_expert_stacks(tmp_path):
    from gguf import GGUFReader

    p = tmp_path / "m.gguf"
    _write_model(p)
    ranges = pin_spine.spine_ranges(str(p))
    assert set(ranges) == {str(p)}
    rs = ranges[str(p)]
    assert rs and all(n > 0 for _, n in rs)
    # every non-expert tensor byte is covered; expert bytes may only leak
    # in via page-alignment slack at range edges
    page = pin_spine._PAGE
    for t in GGUFReader(str(p)).tensors:
        off, end = int(t.data_offset), int(t.data_offset) + int(t.n_bytes)
        covered = any(a <= off and end <= a + n for a, n in rs)
        if "_exps" in t.name:
            inner = any(a + page <= off and end <= a + n - page
                        for a, n in rs)
            assert not inner, t.name
        else:
            assert covered, t.name


def test_spine_pin_lifecycle(tmp_path):
    p = tmp_path / "m.gguf"
    _write_model(p)
    pin = pin_spine.SpinePin(pin_spine.spine_ranges(str(p)))
    assert pin.pinned_bytes > 0
    assert pin.pinned_bytes == pin.total_bytes  # tiny file: no refusal
    pin.close()
    pin.close()  # idempotent (teardown + GC)


def test_maybe_pin_spine_env_off(tmp_path, monkeypatch, capsys):
    p = tmp_path / "m.gguf"
    _write_model(p)
    monkeypatch.setenv("GMLX_PIN_SPINE", "0")
    assert pin_spine.maybe_pin_spine(str(p)) is None
    assert "spine pin off" in capsys.readouterr().out
