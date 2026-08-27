"""Modifier metadata and the pure flag matcher (no Quartz - runs on any
platform)."""
from __future__ import annotations

import pytest

from gmlx.hotkey import (
    PUSH_TO_TALK_MODIFIERS,
    combo_label,
    privacy_pane_url,
    space_flags_match,
)

# CGEventFlags fixtures: mask bits per Quartz, device bits per NX_DEVICE*.
_FN = 0x800000
_CMD, _LCMD, _RCMD = 0x100000, 0x0008, 0x0010
_ALT, _RALT = 0x80000, 0x0040
_CTRL = 0x40000


def test_every_modifier_has_label_and_matcher():
    for mod in PUSH_TO_TALK_MODIFIERS:
        assert combo_label(mod).endswith(" + Space")
        assert space_flags_match(0, mod) is False


def test_globe_matches_secondary_fn_flag():
    assert space_flags_match(_FN | 0x100, "globe")
    assert not space_flags_match(_CMD | _LCMD, "globe")


def test_right_command_needs_the_right_device_bit():
    assert space_flags_match(_CMD | _RCMD | 0x100, "right-command")
    # left Cmd+Space is Spotlight - must not fire
    assert not space_flags_match(_CMD | _LCMD | 0x100, "right-command")
    assert not space_flags_match(_RCMD, "right-command")   # bit without mask


def test_right_option_and_control():
    assert space_flags_match(_ALT | _RALT, "right-option")
    assert not space_flags_match(_ALT | 0x0020, "right-option")  # left opt
    assert space_flags_match(_CTRL | 0x1, "control")             # either side
    assert space_flags_match(_CTRL | 0x2000, "control")


def test_privacy_pane_url():
    assert privacy_pane_url().endswith("Privacy_Accessibility")


def test_unknown_modifier_rejected():
    from gmlx.hotkey import HotkeyTap
    with pytest.raises(ValueError):
        HotkeyTap("globe-space", lambda: None)      # old trigger name
    with pytest.raises(ValueError):
        HotkeyTap("command", lambda: None)


def test_held_combo_fires_once_and_swallows_repeats():
    """Key autorepeat between the matched down and its up must not re-fire
    (a held combo would otherwise chime-spam and thrash listen/dismiss),
    but the repeats and the key-up still get swallowed - no spaces leak
    into the focused app."""
    from gmlx.hotkey import HotkeyTap
    fired = []
    tap = HotkeyTap("globe", lambda: fired.append(1))
    assert tap._handle_space(True, _FN) is True      # matched down: fire
    assert tap._handle_space(True, _FN) is True      # autorepeat: no re-fire
    assert tap._handle_space(True, 0) is True        # repeat after fn release
    assert fired == [1]
    assert tap._handle_space(False, 0) is True       # the key-up, swallowed
    assert tap._handle_space(False, 0) is False      # unrelated later key-up
    assert tap._handle_space(True, 0) is False       # plain Space types again
    assert fired == [1]
    assert tap._handle_space(True, _FN) is True      # combo works again
    assert fired == [1, 1]
