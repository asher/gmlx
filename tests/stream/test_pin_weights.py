"""Spine pinning: non-expert range scan + mlock lifecycle."""

from __future__ import annotations

import re
import subprocess
import sys

import numpy as np
import pytest

import gmlx.stream.pin_weights as pin_weights


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


def test_every_token_ranges_excludes_expert_stacks(tmp_path):
    from gguf import GGUFReader

    p = tmp_path / "m.gguf"
    _write_model(p)
    ranges = pin_weights.every_token_ranges(str(p))
    assert set(ranges) == {str(p)}
    rs = ranges[str(p)]
    assert rs and all(n > 0 for _, n in rs)
    # every non-expert tensor byte is covered; expert bytes may only leak
    # in via page-alignment slack at range edges
    page = pin_weights._PAGE
    for t in GGUFReader(str(p)).tensors:
        off, end = int(t.data_offset), int(t.data_offset) + int(t.n_bytes)
        covered = any(a <= off and end <= a + n for a, n in rs)
        if "_exps" in t.name:
            inner = any(a + page <= off and end <= a + n - page
                        for a, n in rs)
            assert not inner, t.name
        else:
            assert covered, t.name


def test_weight_pin_lifecycle(tmp_path):
    p = tmp_path / "m.gguf"
    _write_model(p)
    pin = pin_weights.WeightsPin(pin_weights.every_token_ranges(str(p)))
    assert pin.pinned_bytes > 0
    assert pin.pinned_bytes == pin.total_bytes  # tiny file: no refusal
    pin.close()
    pin.close()  # idempotent (teardown + GC)


def test_maybe_pin_weights_env_off(tmp_path, monkeypatch, capsys):
    p = tmp_path / "m.gguf"
    _write_model(p)
    monkeypatch.setenv("GMLX_PIN_WEIGHTS", "0")
    assert pin_weights.maybe_pin_weights(str(p)) is None
    assert "weight pin off" in capsys.readouterr().out


def _vm_stat():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", out).group(1))
    return {
        k: int(re.search(rf"{k}:\s+(\d+)\.", out).group(1)) * page
        for k in ("File-backed pages", "Pages wired down")
    }


@pytest.mark.skipif(sys.platform != "darwin", reason="vm_stat is macOS only")
def test_a_pinned_page_leaves_the_reclaimable_snapshot(tmp_path):
    """mlock must move a file-backed page out of the File-backed count.

    ``_available_ram_bytes`` offers free + purgeable + file-backed, and
    ``_decode_arena_bytes`` charges only the *unpinned* share of the
    non-expert weights against that offer, on the grounds that a live pin
    is already out of the snapshot. Measured on macOS 26 the move is one
    for one. If a future release ever left mlocked pages counted as
    file-backed, that offer would overstate by the whole pin and every
    streaming arena would size past the machine, so this is a canary
    rather than a test of our own code.
    """
    p = tmp_path / "big.bin"
    n = 32 << 20
    p.write_bytes(b"\0" * n)

    before = _vm_stat()
    pin = pin_weights.WeightsPin({str(p): [(0, n)]})
    try:
        if pin.pinned_bytes < n:
            pytest.skip("mlock refused; wire limit reached")
        after = _vm_stat()
    finally:
        pin.close()

    # Other activity moves these counters, so assert the direction and
    # allow generous slack rather than an exact delta.
    wired = after["Pages wired down"] - before["Pages wired down"]
    filed = after["File-backed pages"] - before["File-backed pages"]
    assert wired >= 0.5 * n, f"pin did not wire: {wired} of {n}"
    assert filed <= 0.5 * n, (
        "mlocked pages still count as file-backed; _available_ram_bytes now "
        "overstates the offer by the whole weight pin and _decode_arena_bytes "
        f"must charge it (file-backed moved {filed:+d} for a {n} pin)")


def test_reserved_bytes_from_another_install_block_the_pin(
        tmp_path, monkeypatch, capsys):
    # mlock has no backpressure and wired pages are invisible to jetsam, so
    # a second pin that ignores what a live install already holds wires the
    # two together past the machine. The RAM fraction has to see both.
    p = tmp_path / "m.gguf"
    _write_model(p)
    ranges = pin_weights.every_token_ranges(str(p))
    total = sum(n for rs in ranges.values() for _, n in rs)
    # Pretend the box has 2x the pin: alone it clears the 60% fraction,
    # together with a 0.9x install it does not.
    real, page = pin_weights.os.sysconf, pin_weights._PAGE
    monkeypatch.setattr(
        pin_weights.os, "sysconf",
        lambda k: 2 * total // page + 1 if k == "SC_PHYS_PAGES" else real(k))

    # Alone it fits: the whole pin is well under the fraction of "RAM".
    pin = pin_weights.maybe_pin_weights(str(p))
    assert pin is not None
    pin.close()

    # Charged against another install holding most of the box, it does not.
    assert pin_weights.maybe_pin_weights(
        str(p), reserved_bytes=int(0.9 * total)) is None
    out = capsys.readouterr().out
    assert "weight pin skipped" in out
    assert "another live streaming install already holds" in out
