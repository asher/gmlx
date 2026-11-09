"""The feeders' view of a layer's expert ranges: plain stacks pass through,
and a fused gate_up stack splits into its gate and up halves."""
from __future__ import annotations

from gmlx.stream.feeder_common import FeederRange, feeder_ranges, kind_bytes

E = 8


def test_plain_ranges_pass_through():
    raw = [("p", 100, 800, E, "gate"), ("p", 900, 800, E, "up"),
           ("p", 1700, 400, E, "down")]
    assert feeder_ranges(raw) == [
        FeederRange("p", 100, 100, 100, E, "gate", 100, None),
        FeederRange("p", 900, 100, 100, E, "up", 900, None),
        FeederRange("p", 1700, 50, 50, E, "down", 1700, None),
    ]
    assert kind_bytes(raw) == {"gate": 800, "up": 800, "down": 400}


def test_a_fused_range_splits_at_each_experts_midpoint():
    raw = [("p", 100, 1600, E, "gate_up"), ("p", 1700, 400, E, "down")]
    assert feeder_ranges(raw) == [
        FeederRange("p", 100, 200, 100, E, "gate", 100, 0),
        FeederRange("p", 200, 200, 100, E, "up", 100, 1),
        FeederRange("p", 1700, 50, 50, E, "down", 1700, None),
    ]
    assert kind_bytes(raw) == {"gate": 800, "up": 800, "down": 400}


def test_layers_the_feeders_cannot_cover():
    down = ("p", 1700, 400, E, "down")
    # No expert count: no per-expert offsets, but the size is still known.
    flat = [("p", 100, 1600, 0, "gate_up"), ("p", 1700, 400, 0, "down")]
    assert feeder_ranges(flat) is None
    assert kind_bytes(flat) == {"gate": 800, "up": 800, "down": 400}
    # Mixed expert counts.
    assert feeder_ranges([("p", 100, 1600, E, "gate_up"),
                          ("p", 1700, 400, E // 2, "down")]) is None
    # A fused stack beside a separate gate, and a layer missing a kind.
    both = [("p", 100, 1600, E, "gate_up"), ("p", 0, 800, E, "gate"), down]
    assert feeder_ranges(both) is None and kind_bytes(both) is None
    up_down = [("p", 0, 800, E, "up"), down]
    assert feeder_ranges(up_down) is None and kind_bytes(up_down) is None
