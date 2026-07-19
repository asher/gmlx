"""Global <modifier>+Space tap-to-talk hotkey behind the menu bar app.

Space pressed while a configured modifier key is held (``globe`` by
default; ``talk.push_to_talk_modifier`` in the config picks another for
keyboards without a Globe key). An active session CGEventTap (Accessibility
permission) swallows the Space key-down and its key-up so no space character
reaches the focused app; the combo reaches event taps because the modifier's
flag rides on the Space key event. Holding Globe as a modifier suppresses
the system's own bare-press Globe action, so the user's "Press Globe key to"
setting (emoji, dictation, ...) keeps working untouched.

A bare Globe *press* is invisible to session event taps on current macOS -
the WindowServer routes it to the system shortcut handler without posting a
keycode-63 ``flagsChanged`` (verified on macOS 26; it is observable only at
raw HID as usage page 0xFF usage 0x03) - which is why the trigger is a
combo, not a double-press.

The tap and TCC permission calls live behind lazy imports (darwin-only) and
are only reached once the user enables the hotkey, never at app launch.
"""
from __future__ import annotations

import threading
from collections.abc import Callable

SPACE_KEYCODE = 49

# ``talk.push_to_talk_modifier`` values. Right-side variants deliberately:
# left Cmd+Space is Spotlight and left Option+Space is a common launcher
# bind, while the right-side keys are nearly always free.
PUSH_TO_TALK_MODIFIERS = ("globe", "right-command", "right-option", "control")

_MODIFIER_GLYPHS = {"globe": "\U0001f310",            # the Globe/fn key
                    "right-command": "Right \u2318",
                    "right-option": "Right \u2325",
                    "control": "\u2303"}

# CGEventFlags checks per modifier: (flag mask, device bit). The flag masks
# are kCGEventFlagMaskSecondaryFn/Command/Alternate/Control; the device bits
# are the NX_DEVICER*KEYMASK lows that distinguish right from left (verified
# live: left-cmd sets 0x8, right-cmd 0x10). Constants inlined so the check
# stays pure and testable without Quartz.
_MODIFIER_MASKS = {
    "globe": (0x800000, 0),
    "right-command": (0x100000, 0x0010),
    "right-option": (0x80000, 0x0040),
    "control": (0x40000, 0),
}


def combo_label(modifier: str) -> str:
    """Menu-facing name of the combo, e.g. ``"\U0001f310 + Space"``."""
    return f"{_MODIFIER_GLYPHS[modifier]} + Space"


def space_flags_match(flags: int, modifier: str) -> bool:
    """True when a Space key event's CGEventFlags carry the modifier."""
    mask, device_bit = _MODIFIER_MASKS[modifier]
    if not flags & mask:
        return False
    return not device_bit or bool(flags & device_bit)


# -- TCC permission (lazy ApplicationServices imports) -------------------------

def preflight() -> bool:
    """Silent permission check - never prompts, never registers the app."""
    from ApplicationServices import AXIsProcessTrusted
    return bool(AXIsProcessTrusted())


def request() -> bool:
    """Ask for the permission - the only call here that can show a TCC
    prompt (and it registers the app in the Privacy pane either way)."""
    import ApplicationServices as AS
    return bool(AS.AXIsProcessTrustedWithOptions(
        {AS.kAXTrustedCheckOptionPrompt: True}))


def privacy_pane_url() -> str:
    return ("x-apple.systempreferences:com.apple.preference.security"
            "?Privacy_Accessibility")


# -- the event tap -------------------------------------------------------------

class HotkeyTap:
    """An active session CGEventTap on its own CFRunLoop thread.

    ``on_fire`` is called on the tap thread - hosts must marshal to their
    own threads (the menu bar uses ``AppHelper.callAfter``). ``start()``
    returns False when tap creation fails, i.e. the permission is missing
    or was silently revoked (ad-hoc re-sign of the app stub)."""

    def __init__(self, modifier: str, on_fire: Callable[[], None]):
        if modifier not in PUSH_TO_TALK_MODIFIERS:
            raise ValueError(f"unknown push-to-talk modifier {modifier!r}")
        self.modifier = modifier
        self.on_fire = on_fire
        self._swallow_space_up = False
        self._tap = None
        self._runloop = None
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        if self._thread is not None:
            return True
        import Quartz as q
        mask = (q.CGEventMaskBit(q.kCGEventKeyDown)
                | q.CGEventMaskBit(q.kCGEventKeyUp))
        tap = q.CGEventTapCreate(q.kCGSessionEventTap, q.kCGHeadInsertEventTap,
                                 q.kCGEventTapOptionDefault, mask,
                                 self._callback, None)
        if tap is None:
            return False
        self._tap = tap
        source = q.CFMachPortCreateRunLoopSource(None, tap, 0)
        ready = threading.Event()

        def pump() -> None:
            self._runloop = q.CFRunLoopGetCurrent()
            q.CFRunLoopAddSource(self._runloop, source,
                                 q.kCFRunLoopCommonModes)
            q.CGEventTapEnable(tap, True)
            ready.set()
            q.CFRunLoopRun()

        self._thread = threading.Thread(target=pump, daemon=True,
                                        name="gmlx-hotkey")
        self._thread.start()
        ready.wait(2.0)
        return True

    def stop(self) -> None:
        if self._tap is None:
            return
        import Quartz as q
        q.CGEventTapEnable(self._tap, False)
        if self._runloop is not None:
            q.CFRunLoopStop(self._runloop)
        if self._thread is not None:
            self._thread.join(2.0)
        q.CFMachPortInvalidate(self._tap)
        self._tap = None
        self._runloop = None
        self._thread = None

    # tap-thread callback: must never raise (an exception here kills the tap)
    def _callback(self, proxy, etype, event, refcon):
        import Quartz as q
        try:
            if etype in (q.kCGEventTapDisabledByTimeout,
                         q.kCGEventTapDisabledByUserInput):
                q.CGEventTapEnable(self._tap, True)
                return event
            keycode = q.CGEventGetIntegerValueField(
                event, q.kCGKeyboardEventKeycode)
            # Active tap: returning None swallows the event.
            if (keycode == SPACE_KEYCODE
                    and self._handle_space(etype == q.kCGEventKeyDown,
                                           q.CGEventGetFlags(event))):
                return None
            return event
        except Exception:
            return event

    def _handle_space(self, is_down: bool, flags: int) -> bool:
        """Swallow/fire decision for a Space key event; True = swallow.
        Between the matched key-down and its key-up, autorepeat key-downs
        are swallowed without re-firing - holding the combo is one tap, not
        a chime-spamming toggle storm."""
        if is_down:
            if self._swallow_space_up:
                return True
            if space_flags_match(flags, self.modifier):
                self._swallow_space_up = True
                self._fire()
                return True
            return False
        if self._swallow_space_up:
            self._swallow_space_up = False
            return True
        return False

    def _fire(self) -> None:
        try:
            self.on_fire()
        except Exception:
            pass  # a failing callback must not kill the event tap
