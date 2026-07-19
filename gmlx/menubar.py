"""macOS menu-bar monitor for a backgrounded ``gmlx serve``.

A thin status-bar app (rumps) over the **existing** server endpoints - it never adds
control surface. It shows whether the server is up (and how busy), lists the resident
models with an Unload action each (size, default/pinned/kept markers, eviction
countdown), and offers Reload-config / Restart / Stop / Copy-URL / Open-logs. "Edit
config" opens a floating validate/save/reload editor on the managed server's YAML
(:mod:`.menubar_config`). A tracked server dying unexpectedly posts a notification
(see :class:`DownNotifier`).

Threading (the menu must never freeze): a daemon worker thread polls the server on an
interval and writes a shared snapshot; the rumps timer, on the main run-loop thread,
only reads that snapshot and relabels the menu. Every action (unload / reload / stop /
restart / open-logs) is dispatched to a short-lived worker thread too, so the
SIGTERM->wait->SIGKILL stop grace never blocks the UI.

Auth follows the rest of the CLI: ``/health`` is unauthenticated, so "up vs down" needs
no key; the resident-model list reads ``/v1/metrics`` with a Bearer key resolved from
``--api-key`` or, failing that, the ``server.api_key`` in the managed server's own
config (via the runfile's recorded path). A 401 means *up, key required* - never down.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from urllib.parse import urlparse

from .lifecycle import get_json as _get_json
from .lifecycle import human_gb
from .lifecycle import post_json as _post_json
from .lifecycle import server_root as _server_root


def _split_url(url: str) -> tuple:
    u = urlparse(url if "://" in url else "http://" + url)
    try:
        port = u.port or 8080   # .port raises on a non-numeric port string
    except ValueError:
        port = 8080
    return (u.hostname or "127.0.0.1", port)


def _key_from_config(run: dict | None) -> str | None:
    """The managed server's ``server.api_key`` read from its own recorded config - the
    zero-config path for monitoring a server you launched. Returns None unless the
    runfile says a key is set and the config still parses."""
    if not run or not run.get("api_key_set"):
        return None
    cfg_path = run.get("config_abspath")
    if not cfg_path:
        return None
    try:
        from .config import load_config
        return getattr(load_config(cfg_path), "api_key", None)
    except Exception:
        return None  # best-effort config read; unreadable -> no key


def resolve_api_key(args_key: str | None, run: dict | None) -> str | None:
    """An explicit ``--api-key`` wins; otherwise read the managed server's config."""
    return args_key or _key_from_config(run)


def talk_model_from_config(run: dict | None) -> str | None:
    """``talk.model`` from the managed server's recorded config - the id the
    voice loop would use ahead of the server's default model. None when unset,
    unreadable, or the server wasn't launched from a config."""
    cfg_path = (run or {}).get("config_abspath")
    if not cfg_path:
        return None
    try:
        from .config import load_config
        return getattr(load_config(cfg_path).talk, "model", None)
    except Exception:
        return None  # best-effort config read; unreadable -> unset


def ptt_modifier_from_config(run: dict | None) -> str:
    """``talk.push_to_talk_modifier`` from the managed server's recorded
    config; ``"globe"`` when unset, unreadable, or invalid."""
    from .hotkey import PUSH_TO_TALK_MODIFIERS
    cfg_path = (run or {}).get("config_abspath")
    if cfg_path:
        try:
            from .config import load_config
            mod = getattr(load_config(cfg_path).talk,
                          "push_to_talk_modifier", "globe")
            if mod in PUSH_TO_TALK_MODIFIERS:
                return mod
        except Exception:
            pass  # unreadable config -> default modifier
    return "globe"


def poll(url: str, api_key: str | None = None, timeout: float = 1.5) -> dict:
    """One non-blocking probe -> snapshot. ``/health`` decides reachable (no key); the
    resident list comes from ``/v1/metrics`` (Bearer). A 401 sets ``auth_required`` but
    leaves ``reachable`` true - *up, key required*, not down."""
    root = _server_root(url)
    snap = {"url": url, "reachable": False, "auth_required": False,
            "resident": [], "queue_depth": 0, "in_flight": 0,
            "talk_ready": False, "default_model": None, "error": None}
    try:
        _get_json(root + "/health", timeout=timeout)
        snap["reachable"] = True
    except (urllib.error.URLError, OSError, ValueError) as e:
        snap["error"] = str(e)
        return snap
    try:
        payload = _get_json(root + "/v1/metrics", api_key=api_key, timeout=timeout)
        # Shape-guarded: valid JSON from something else on the port must not
        # kill the poll thread.
        srv = payload.get("server") if isinstance(payload, dict) else None
        srv = srv if isinstance(srv, dict) else {}
        rm = srv.get("resident_models")
        snap["resident"] = ([e for e in rm if isinstance(e, dict)]
                            if isinstance(rm, list) else [])
        snap["queue_depth"] = int(srv.get("request_queue_depth") or 0)
        snap["in_flight"] = sum(int(e.get("busy") or 0)
                                for e in snap["resident"])
    except urllib.error.HTTPError as e:
        if e.code == 401:
            snap["auth_required"] = True
        else:
            snap["error"] = f"metrics HTTP {e.code}"
    except (urllib.error.URLError, OSError, ValueError, TypeError) as e:
        snap["error"] = str(e)
    try:
        # The stt/tts marker entries in /v1/models (the same ones `gmlx talk`
        # keys on) say whether a voice loop would work against this server.
        payload = _get_json(root + "/v1/models", api_key=api_key, timeout=timeout)
        entries = payload.get("data") if isinstance(payload, dict) else None
        entries = ([e for e in entries if isinstance(e, dict)]
                   if isinstance(entries, list) else [])
        snap["talk_ready"] = (any(e.get("stt") for e in entries)
                              and any(e.get("tts") for e in entries))
        snap["default_model"] = next(
            (e.get("id") for e in entries if e.get("default")), None)
    except (urllib.error.URLError, OSError, ValueError):
        pass
    return snap


def _existing_default_config() -> str | None:
    """Fallback path for the Edit-config item when no runfile records one (server
    never started, or cleanly stopped): the first default config location that
    exists. The config is exactly what you want to edit while the server is down."""
    try:
        from .config import default_config_paths
        for p in default_config_paths():
            if p.exists():
                return str(p)
    except Exception:
        pass  # probe only; no readable default config -> no path
    return None


def _port_of(snapshot: dict, run: dict | None) -> int | None:
    if run and run.get("port"):
        return int(run["port"])
    try:
        return _split_url(snapshot.get("url") or "")[1]
    except (ValueError, TypeError):
        return None


# Persisted menu-bar preferences (the tap-to-talk hotkey, and the launchd
# agent's server-autostart record). Deliberately separate from the server
# YAML: this is client UI state, and the menu bar must never rewrite the
# user's config file.

def menubar_settings_path():
    from . import lifecycle
    return lifecycle.runtime_dir() / "menubar-settings.json"


def _parse_autostart(raw) -> dict | None:
    """The validated server-autostart record (written by `gmlx service
    install`, replayed at login), or None. argv is the detach child argv the
    runfile recorded - the same thing `restart` replays."""
    try:
        argv = [str(x) for x in raw["argv"]]
        if not argv:
            return None
        return {"argv": argv,
                "host": str(raw.get("host") or "127.0.0.1"),
                "port": int(raw.get("port") or 8080),
                "config_abspath": (str(raw["config_abspath"])
                                   if raw.get("config_abspath") else None),
                "api_key_set": bool(raw.get("api_key_set"))}
    except (TypeError, KeyError, ValueError):
        return None


def load_menubar_settings() -> dict:
    out = {"hotkey": "off", "autostart": None, "volume": 1.0}
    try:
        with open(menubar_settings_path()) as f:
            data = json.load(f)
    except Exception:
        return out  # missing/corrupt settings file -> defaults
    if isinstance(data, dict):
        # "globe-space" is the pre-modifier-config on-value; unknown values
        # (including the retired "globe-double") fall to off.
        if data.get("hotkey") in ("on", "globe-space"):
            out["hotkey"] = "on"
        out["autostart"] = _parse_autostart(data.get("autostart"))
        try:
            vol = float(data.get("volume", 1.0))
            if vol == vol:                       # NaN never clamps sanely
                out["volume"] = min(1.0, max(0.0, vol))
        except (TypeError, ValueError):
            pass
    return out


def save_menubar_settings(settings: dict) -> None:
    path = menubar_settings_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(settings, f, indent=2)
        os.replace(tmp, path)
    except OSError:
        pass    # a preference that fails to persist must not take down the bar


def _autostart_server_once() -> None:
    """launchd-agent boot: start the recorded server unless this login
    already ran autostart. The boot-time stamp is the don't-fight-the-user
    rule - a menu bar respawned by KeepAlive mid-session skips this, so a
    server the user deliberately stopped stays stopped until the next
    login. Best-effort: any failure just leaves the bar showing "down" with
    its one-click Start."""
    from . import lifecycle, procname
    auto = load_menubar_settings().get("autostart")
    if not auto:
        return
    boot = lifecycle.boot_time()
    stamp = lifecycle.autostart_stamp_path()
    try:
        if boot and stamp.read_text() == boot:
            return                    # not the first bar of this login
    except OSError:
        pass
    try:
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(boot)
    except OSError:
        pass
    host, port = auto["host"], auto["port"]
    run = lifecycle.read_run(host, port)
    if run and (run.get("managed_by") == "launchd"
                or lifecycle.identity_ok(run)):
        return                        # already up (or launchd's problem)
    argv = list(auto["argv"])
    # The recorded argv[0] may predate an interpreter swap; refresh the stub
    # and point at it (falling back to whatever was recorded).
    exe = procname.named_python()
    if exe:
        argv[0] = exe
    try:
        lifecycle.launch_detached(argv, host=host, port=port,
                                  config_abspath=auto.get("config_abspath"),
                                  api_key_set=bool(auto.get("api_key_set")))
    except Exception:
        pass  # best-effort autostart replay; the menu stays usable without it


def build_menu_model(snapshot: dict, run: dict | None,
                     talk_model: str | None = None,
                     session: dict | None = None,
                     fallback_config: str | None = None,
                     hotkey: dict | None = None,
                     volume: float | None = None) -> dict:
    """Pure description of the menu from a :func:`poll` snapshot + the runfile dict (or
    None). No rumps, no I/O - unit-tested directly. The rumps app reads this and lays
    out menu items; it carries no presentation strings of its own. ``talk_model``
    is the config's ``talk.model`` override (see :func:`talk_model_from_config`);
    it beats the server's default-marked id in the talk item label, mirroring the
    voice loop's own resolution order. ``session`` is a live voice session's
    :meth:`_VoiceSession.snapshot` (or None): it replaces the start-talk item
    with the session controls and puts the voice-state glyph in the title.
    ``fallback_config`` (see :func:`_existing_default_config`) backs the
    Edit-config item when the runfile doesn't record a config path.
    ``hotkey`` is ``{"enabled", "available", "label", "error"}`` for the
    global tap-to-talk toggle (None or unavailable hides the item); it is a
    local setting, deliberately independent of server reachability.
    ``volume`` (0.0-1.0, or None to hide) is the persisted output gain; it
    surfaces as a slider among the session controls, so it only renders
    when ``session`` is present."""
    reachable = bool(snapshot.get("reachable"))
    auth_required = bool(snapshot.get("auth_required"))
    in_flight = int(snapshot.get("in_flight") or 0)
    queued = int(snapshot.get("queue_depth") or 0)
    managed_by = (run or {}).get("managed_by")
    pid = (run or {}).get("pid")
    port = _port_of(snapshot, run)
    where = f":{port}" if port else ""

    if not reachable:
        state, glyph = "down", "○"
        header = f"Server: down ({where})" if where else "Server: down"
    elif auth_required:
        state, glyph = "key-required", "◐"
        header = f"Server: up - key required ({where})".replace(" ()", "")
    else:
        state = "up"
        glyph = "◉" if (in_flight or queued) else "●"
        pid_s = f"pid {pid}, " if pid else ""
        inside = f"{pid_s}{where}".strip(", ")
        header = f"Server: up ({inside})" if inside else "Server: up"

    busy_parts = []
    if in_flight:
        busy_parts.append(f"{in_flight} generating")
    if queued:
        busy_parts.append(f"{queued} queued")
    busy_line = ", ".join(busy_parts) if (state == "up" and busy_parts) else None

    default_id = snapshot.get("default_model")
    models = []
    for e in snapshot.get("resident") or []:
        ids = e.get("ids") or ([e["id"]] if e.get("id") else [])
        name = ", ".join(ids) or os.path.basename(e.get("model_path") or "model")
        fb = e.get("footprint_bytes") or 0
        marks = f"  ({human_gb(fb)})" if fb else ""
        if default_id and default_id in ids:
            marks += "  [default]"
        if e.get("pinned"):
            marks += "  [pinned]"
        elif e.get("kept"):
            marks += "  [kept]"
        elif e.get("ttl_s"):
            # pinned/kept entries are reaper-exempt; only a plain entry evicts
            left = max(0, int(e["ttl_s"] - (e.get("idle_s") or 0)))
            marks += (f"  evicts in {round(left / 60)}m" if left >= 120
                      else f"  evicts in {left}s")
        models.append({"id": (ids[0] if ids else name),
                       "label": f"{name}{marks}",
                       "unloadable": bool(ids)})

    if state == "key-required":
        models_header = "Loaded models - pass --api-key to list"
    elif models:
        models_header = "Loaded models"
    elif reachable and snapshot.get("error"):
        # /health up but /v1/metrics failed: we don't know what's resident,
        # and claiming "none" would be a lie.
        models_header = "Loaded models unknown (metrics error)"
    elif reachable:
        models_header = "No models resident"
    else:
        models_header = None        # down: the header already says so - no models line

    has_run = run is not None
    relaunchable = has_run and (managed_by == "launchd"
                                or bool((run or {}).get("argv")))
    voice_model = talk_model or snapshot.get("default_model")
    hotkey_model = None
    if hotkey and hotkey.get("available"):
        hotkey_model = {"enabled": bool(hotkey.get("enabled")),
                        "label": hotkey.get("label") or "\U0001f310 + Space",
                        "error": hotkey.get("error")}
    title = f"gmlx {glyph}"
    talk_session = None
    if session is not None:
        vg = "🔇" if session.get("muted") else _VOICE_GLYPHS.get(
            session.get("state"), "🎤")
        title = f"{title} {vg}"
        talk_session = {"line": voice_session_line(session),
                        "busy": bool(session.get("busy")),
                        "muted": bool(session.get("muted")),
                        "error": session.get("error"),
                        "has_memory": bool(session.get("has_memory")),
                        "volume": volume}
    return {
        "title": title,
        "state": state,
        "header": header,
        "busy_line": busy_line,
        "url": snapshot.get("url"),
        "models_header": models_header,
        "models": models,
        "can_reload": reachable and not auth_required,
        # server up or down - fixing the config is a down-state activity too
        "config_path": (run or {}).get("config_abspath") or fallback_config,
        # voice chat needs the server's stt + tts markers (and a readable
        # /v1/models); a live session replaces the start item with controls
        "can_talk": reachable and not auth_required
                    and bool(snapshot.get("talk_ready")) and session is None,
        "talk_label": (f"Talk to {voice_model}…" if voice_model
                       else "Talk to model…"),
        "talk_session": talk_session,
        "hotkey": hotkey_model,
        "can_stop": reachable and has_run and managed_by == "detach",
        # up => "Restart"; down (with a runfile to relaunch from) => "Start"
        "can_restart": reachable and relaunchable,
        "can_start": (not reachable) and relaunchable,
        "restart_kind": managed_by if has_run else None,
        "managed_by": managed_by,
        "log": (run or {}).get("log"),
    }


def down_message(url: str | None, run: dict | None,
                 pid_dead: bool) -> str:
    """The up->down notification body. A detach server whose recorded pid is
    gone crashed (a clean ``gmlx stop`` removes the runfile and suppresses
    the notification entirely); a live-but-unresponsive one gets the softer
    wording. launchd-managed runs have no usable pid and KeepAlive is
    already restarting them, so they never read as crashed."""
    where = url or "the server"
    if run and run.get("managed_by") == "detach" and pid_dead:
        return (f'{where} crashed - "Open logs" in the menu has the tail; '
                '"Start server" relaunches it')
    return f"{where} stopped responding"


class DownNotifier:
    """Up->down transition detector behind the death notification. Watching a
    detached server die is the one job a monitor must not do silently - but the
    bar's own Stop/Restart items bring the server down on purpose, so those
    call :meth:`expect` first, opening a grace window that swallows the next
    observed down. A clean CLI ``gmlx stop`` is separately suppressed by the
    caller (it removes the runfile; a crash leaves it behind)."""

    def __init__(self, grace_s: float = 90.0, clock=time.monotonic):
        self._clock = clock
        self._grace = grace_s
        self._last: bool | None = None
        self._until = 0.0

    def expect(self) -> None:
        """The bar itself is about to take the server down."""
        self._until = self._clock() + self._grace

    def observe(self, reachable: bool) -> bool:
        """Feed one poll result; True when this is a down flip worth notifying."""
        prev, self._last = self._last, bool(reachable)
        return bool(prev) and not reachable and self._clock() >= self._until


# In-process voice session (menubar-native `gmlx talk`, no terminal window).
_VOICE_GLYPHS = {"idle": "🎤", "listening": "🎤", "capturing": "🎤",
                 "transcribing": "💭", "thinking": "💭", "speaking": "🔊",
                 "starting": "…"}


def voice_session_line(sess: dict) -> str:
    """One-line description of a live voice session for the menu header row."""
    if sess.get("error"):
        return f"Voice chat: {sess['error']}"
    if sess.get("muted"):
        return "Voice chat: muted"
    state = sess.get("state", "idle")
    if state in ("idle", "listening") and sess.get("wake"):
        return f'Voice chat: say "{sess["wake"]}"'
    return f"Voice chat: {state}"


class _VoiceSession:
    """State bag for a headless TalkLoop running inside the menubar process.
    The loop thread owns loop/cleanup; the run-loop thread reads ``snapshot``
    and ``transcript`` (plain attribute reads under the GIL - same idiom as
    the poll snapshot)."""

    def __init__(self):
        self.loop = None
        self.cleanup = None
        self.control: queue.Queue = queue.Queue()
        self.thread: threading.Thread | None = None
        self.lines: deque = deque(maxlen=400)
        self.wake: str | None = None
        self.error: str | None = None
        self.starting = True

    def alive(self) -> bool:
        return self.starting or (self.thread is not None
                                 and self.thread.is_alive())

    def snapshot(self) -> dict:
        m = getattr(self.loop, "m", None)
        state = getattr(m, "state", None) or ("starting" if self.starting
                                              else "idle")
        return {"state": state, "muted": bool(getattr(m, "muted", False)),
                "wake": self.wake, "error": self.error,
                "busy": state in ("transcribing", "thinking", "speaking"),
                "has_memory": getattr(getattr(self.loop, "brain", None),
                                      "memory", None) is not None}

    def transcript(self) -> str:
        return "\n".join(self.lines)


def _no_talk_flags():
    """A flags namespace with every talk flag unset, so ``_merged_settings``
    resolves purely from the YAML ``talk:`` block (the menu bar has no CLI)."""
    import types
    return types.SimpleNamespace(
        model=None, voice=None, speed=None, language=None, system=None,
        max_tokens=None, mode=None, wake_word=None, wake_threshold=None,
        vad_threshold=None, vad_silence_ms=None, min_speech_ms=None,
        input_device=None, output_device=None, no_chime=False, brain=None)


class LogMerger:
    """Incremental multi-file tailer behind the unified logs panel: each call
    returns the lines each source grew since the last call, prefixed with the
    source's label, in arrival order (a merged live tail). Pure - no AppKit -
    so the seeding/increment/truncation behavior is unit-testable."""

    SEED_BYTES = 16 * 1024        # first sight of a file: start this far back

    def __init__(self):
        self._state: dict = {}    # path -> [offset, partial-line carry]

    def read_new(self, sources) -> list:
        out = []
        for label, path in sources:
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            off, carry = self._state.get(path, (None, ""))
            if off is None:
                off = max(0, size - self.SEED_BYTES)
            if size < off:                    # truncated/rotated: start over
                off, carry = 0, ""
            if size > off:
                try:
                    with open(path, "rb") as f:
                        f.seek(off)
                        chunk = f.read(size - off)
                except OSError:
                    continue
                off += len(chunk)
                text = carry + chunk.decode("utf-8", errors="replace")
                lines = text.split("\n")
                carry = lines.pop()           # "" when the chunk ended on \n
                out.extend(f"{label:>7} | {ln}" for ln in lines if ln.strip())
            self._state[path] = [off, carry]
        return out


class _TranscriptPanel:
    """Floating AppKit text panel showing the voice session's transcript.
    pyobjc is already a rumps dependency, so this adds no install weight.
    All methods run on the main run-loop thread (called from rumps timers)."""

    def __init__(self):
        from AppKit import (NSBackingStoreBuffered, NSClosableWindowMask,
                            NSFont, NSMakeRect, NSPanel, NSResizableWindowMask,
                            NSScrollView, NSTextView, NSTitledWindowMask,
                            NSUtilityWindowMask)
        mask = (NSTitledWindowMask | NSClosableWindowMask
                | NSResizableWindowMask | NSUtilityWindowMask)
        self.panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, 460, 320), mask, NSBackingStoreBuffered, False)
        self.panel.setTitle_("Voice chat")
        self.panel.setFloatingPanel_(True)
        self.panel.setReleasedWhenClosed_(False)   # user close = hide, reusable
        # Utility panels default hidesOnDeactivate=YES, and an accessory app
        # is never active: the window server keeps the panel off screen even
        # though isVisible flips true (menu label toggles, nothing appears).
        self.panel.setHidesOnDeactivate_(False)
        scroll = NSScrollView.alloc().initWithFrame_(
            self.panel.contentView().bounds())
        scroll.setHasVerticalScroller_(True)
        scroll.setAutoresizingMask_(18)            # width + height sizable
        self.text = NSTextView.alloc().initWithFrame_(scroll.bounds())
        self.text.setEditable_(False)
        self.text.setFont_(NSFont.userFixedPitchFontOfSize_(12.0))
        self.text.setAutoresizingMask_(18)
        scroll.setDocumentView_(self.text)
        self.panel.contentView().addSubview_(scroll)
        self.panel.center()
        self._last = None

    def visible(self) -> bool:
        return bool(self.panel.isVisible())

    def show(self) -> None:
        # Regardless-variant: the plain orderFront_ is a no-op while the
        # app is inactive. No app activation - the transcript is a passive
        # display and should not steal focus from what the user is doing.
        self.panel.orderFrontRegardless()

    def hide(self) -> None:
        self.panel.orderOut_(None)

    def set_text(self, text: str) -> None:
        if text == self._last:
            return
        self._last = text
        self.text.setString_(text)
        self.text.scrollToEndOfDocument_(None)


class _LogsPanel:
    """Floating unified live tail of the server and menu-bar logs, fed a
    :class:`LogMerger` increment on each refresh tick while visible. Same
    AppKit pattern (and hidesOnDeactivate fix) as :class:`_TranscriptPanel`."""

    MAX_LINES = 2000

    def __init__(self):
        from AppKit import (NSBackingStoreBuffered, NSClosableWindowMask,
                            NSFont, NSMakeRect, NSPanel, NSResizableWindowMask,
                            NSScrollView, NSTextView, NSTitledWindowMask,
                            NSUtilityWindowMask)
        mask = (NSTitledWindowMask | NSClosableWindowMask
                | NSResizableWindowMask | NSUtilityWindowMask)
        self.panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, 720, 440), mask, NSBackingStoreBuffered, False)
        self.panel.setTitle_("gmlx logs")
        self.panel.setFloatingPanel_(True)
        self.panel.setReleasedWhenClosed_(False)
        self.panel.setHidesOnDeactivate_(False)
        scroll = NSScrollView.alloc().initWithFrame_(
            self.panel.contentView().bounds())
        scroll.setHasVerticalScroller_(True)
        scroll.setAutoresizingMask_(18)
        self.text = NSTextView.alloc().initWithFrame_(scroll.bounds())
        self.text.setEditable_(False)
        self.text.setFont_(NSFont.userFixedPitchFontOfSize_(11.0))
        self.text.setAutoresizingMask_(18)
        scroll.setDocumentView_(self.text)
        self.panel.contentView().addSubview_(scroll)
        self.panel.center()
        self._merger = LogMerger()
        self._lines: list = []

    def visible(self) -> bool:
        return bool(self.panel.isVisible())

    def show(self) -> None:
        self.panel.orderFrontRegardless()

    def hide(self) -> None:
        self.panel.orderOut_(None)

    def tick(self, sources) -> None:
        new = self._merger.read_new(sources)
        if not new:
            return
        self._lines.extend(new)
        del self._lines[:-self.MAX_LINES]
        self.text.setString_("\n".join(self._lines))
        self.text.scrollToEndOfDocument_(None)


def talk_command_file(cmd: str, directory: str) -> str:
    """Write (or refresh) the executable .command stub that runs the voice
    loop, and return its path."""
    path = os.path.join(directory, "gmlx-talk.command")
    with open(path, "w") as f:
        f.write(f"#!/bin/zsh\nexec {cmd}\n")
    os.chmod(path, 0o755)
    return path


def open_talk_terminal(cmd: str, *, run=subprocess.run,
                       directory: str | None = None) -> None:
    """Launch ``cmd`` in a terminal window by ``open``ing a .command stub.

    LaunchServices hands the file to a terminal with no AppleEvents - scripting
    Terminal.app directly both raises the automation-permission popup (worded
    as whatever app owns the menu bar's process tree asking to "control
    Terminal") and hijacks users of other terminals into Terminal.app. When
    iTerm2 is running it gets the file explicitly, since LaunchServices'
    default handler for .command is Terminal.app unless remapped."""
    path = talk_command_file(cmd, directory or tempfile.gettempdir())
    if run(["pgrep", "-xq", "iTerm2"], capture_output=True).returncode == 0:
        if run(["open", "-a", "iTerm", path],
               capture_output=True).returncode == 0:
            return
    run(["open", path], capture_output=True)


class _MenuBarApp:
    """rumps wrapper. Constructed only after a successful ``import rumps`` so the rest of
    the module stays importable (and unit-testable) on Linux / a stripped install."""

    def __init__(self, url: str | None, api_key: str | None, interval: float,
                 host: str | None, port: int | None, *, dynamic: bool = False):
        import rumps
        self._rumps = rumps
        self._api_key = api_key
        self.interval = max(1.0, float(interval))
        self._dynamic = dynamic
        self._lock = threading.Lock()
        # Dynamic mode: no fixed target - resolve the primary server now (and on every
        # poll), so one menu bar follows "the" server as servers come and go.
        if dynamic:
            from . import lifecycle
            host, port = lifecycle.auto_target(None, None)
            url = f"http://{host}:{port}"
        self.url = url
        self.host = host
        self.port = port
        self._snapshot = {"url": url, "reachable": False, "auth_required": False,
                          "resident": [], "queue_depth": 0, "in_flight": 0,
                          "error": None}
        self._stop_evt = threading.Event()
        self._notify = DownNotifier()
        # talk.model override for the talk item label, read once (like the key)
        self._talk_model = talk_model_from_config(self._runinfo(host, port))
        # hotkey combo, from the same config (re-read whenever it is enabled)
        self._ptt_modifier = ptt_modifier_from_config(self._runinfo(host, port))
        self._voice: _VoiceSession | None = None
        self._volume_slider = None               # lazy rumps.SliderMenuItem
        self._panel = None                       # lazy _TranscriptPanel
        self._cfg_panel = None                   # lazy menubar_config.ConfigPanel
        self._logs_panel = None                  # lazy _LogsPanel
        self._settings = load_menubar_settings()
        self._hotkey_tap = None                  # live hotkey.HotkeyTap
        self._hotkey_error: str | None = None
        self._hotkey_avail: bool | None = None
        self._hotkey_arming = False              # an async arm is in flight
        self.app = rumps.App("gmlx", title="gmlx ○", quit_button=None)
        self._timer = rumps.Timer(self._refresh, self.interval)
        # Fast repaint while a voice session is live: state glyph + transcript
        # move at speech cadence, not at the server-poll cadence.
        self._voice_timer = rumps.Timer(self._refresh, 0.5)
        self._worker = threading.Thread(target=self._poll_loop, daemon=True)

    # --- data plane (worker thread) ---
    def _current_target(self) -> tuple:
        """(url, host, port) to poll: the primary server in dynamic mode (re-resolved
        each tick so it follows the single/primary server), else the fixed target."""
        if not self._dynamic:
            return self.url, self.host, self.port
        from . import lifecycle
        host, port = lifecycle.auto_target(None, None)
        return f"http://{host}:{port}", host, port

    def _runinfo(self, host=None, port=None) -> dict | None:
        from . import lifecycle
        return lifecycle.read_run(self.host if host is None else host,
                                  self.port if port is None else port)

    def _resolve_key(self, host=None, port=None) -> str | None:
        return resolve_api_key(self._api_key, self._runinfo(host, port))

    def _poll_loop(self) -> None:
        while not self._stop_evt.is_set():
            url, host, port = self._current_target()
            snap = poll(url, self._resolve_key(host, port), timeout=1.5)
            with self._lock:
                self._snapshot = snap
                self.url, self.host, self.port = url, host, port
            self._stop_evt.wait(self.interval)

    def _spawn(self, fn) -> None:
        threading.Thread(target=fn, daemon=True).start()

    # --- actions (each off the main thread) ---
    def _unload(self, model_id: str) -> None:
        def work():
            try:
                _post_json(_server_root(self.url) + "/unload",
                           {"model": model_id},
                           api_key=self._resolve_key(), timeout=8.0)
            except Exception:
                pass  # fire-and-forget; the next poll shows the real state
        self._spawn(work)

    def _reload(self) -> None:
        def work():
            try:
                _post_json(_server_root(self.url) + "/v1/reload",
                           {}, api_key=self._resolve_key(), timeout=15.0)
            except Exception:
                pass  # fire-and-forget; the next poll shows the real state
        self._spawn(work)

    def _copy_url(self) -> None:
        url = self.url or ""
        self._spawn(lambda: subprocess.run(["pbcopy"], input=url.encode(),
                                           capture_output=True))

    def _edit_config(self, path: str) -> None:
        """Open the floating config editor (main run-loop thread - it is called
        from a menu item). The panel is cached like the transcript panel; a
        changed path (dynamic mode following a different server) rebuilds it."""
        if self._cfg_panel is not None and self._cfg_panel.path != path:
            self._cfg_panel.hide()
            self._cfg_panel = None
        if self._cfg_panel is None:
            from .menubar_config import ConfigPanel
            self._cfg_panel = ConfigPanel(
                path, on_reload=self._reload,
                on_open_editor=lambda: self._open_text_editor(path))
        self._cfg_panel.show()

    def _open_text_editor(self, path: str) -> None:
        # `open -t` = the default plain-text editor; a bare `open` would hand
        # .yaml to whatever claimed the extension (often Xcode, slow to launch).
        self._spawn(lambda: subprocess.run(["open", "-t", path],
                                           capture_output=True))

    def _stop(self) -> None:
        self._notify.expect()            # bar-initiated: the down is expected
        def work():
            from . import lifecycle
            lifecycle.stop(self.host, self.port)
        self._spawn(work)

    def _restart(self) -> None:
        self._notify.expect()            # kickstart/restart dips are expected
        run = self._runinfo()
        def work():
            from . import lifecycle
            if run and run.get("managed_by") == "launchd":
                label = run.get("label") or lifecycle._label(self.host, self.port)
                subprocess.run(
                    ["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/{label}"],
                    capture_output=True)
            else:
                lifecycle.restart(self.host, self.port)
        self._spawn(work)

    def _log_sources(self) -> list:
        """(label, path) pairs for the unified logs panel - the tracked
        server's log plus this process's own, when they exist."""
        from . import lifecycle
        out = []
        run = self._runinfo()
        log = (run or {}).get("log")
        if log and os.path.exists(log):
            out.append(("server", log))
        mlog = str(lifecycle.menubar_log_path())
        if os.path.exists(mlog):
            out.append(("menubar", mlog))
        return out

    def _show_logs(self) -> None:
        if self._logs_panel is None:
            self._logs_panel = _LogsPanel()
        self._logs_panel.tick(self._log_sources())
        self._logs_panel.show()

    def _open_talk(self) -> None:
        """Open a terminal window running `gmlx talk` against the tracked server
        (the typed/slash-command experience; the in-process session below is the
        default click). Runs through this process's interpreter, not a bare
        `gmlx`: the terminal shell's PATH may resolve to a different install
        (missing the talk extra) than the venv this menu bar came from."""
        import shlex
        exe = shlex.quote(os.path.abspath(sys.executable))
        cmd = f"{exe} -m gmlx talk"
        # The terminal shell's cwd won't find the server's config by
        # discovery, and talk's wake word / persona / brain live there.
        cfg = (self._runinfo() or {}).get("config_abspath")
        if cfg:
            cmd += f" --config {shlex.quote(cfg)}"
        if not self._dynamic:
            cmd += f" --base-url {self.url}"
        self._spawn(lambda: open_talk_terminal(cmd))

    # --- in-process voice session -------------------------------------------
    def _start_voice(self) -> None:
        """Run the voice loop headless inside this process: the menu bar is the
        status display (state glyph + controls + transcript panel), earcons are
        the audio feedback - no terminal window."""
        if self._voice is not None:
            return
        sess = _VoiceSession()
        self._voice = sess
        self._voice_timer.start()
        url, host, port = self._current_target()
        run = self._runinfo(host, port)

        def boot():
            try:
                from . import talk as talk_mod
                from .talk_client import ensure_v1_base, probe_capabilities
                # _current_target returns a bare http://host:port; the talk
                # client's audio routes exist only under /v1.
                base = ensure_v1_base(url)
                key = self._resolve_key(host, port)
                talk_cfg = talk_mod._load_talk_cfg(
                    (run or {}).get("config_abspath"))
                s = talk_mod._merged_settings(_no_talk_flags(), talk_cfg)
                if s["mode"] in ("ptt", "text"):
                    s["mode"] = "wake"           # keyboard modes need a terminal
                caps = probe_capabilities(base, key)
                model = talk_mod._pick_model(s["model"], caps)
                if not model:
                    raise talk_mod.TalkSetupError(
                        "no talk.model configured and the server has no "
                        "default model")
                def warn(msg: str) -> None:
                    # Boot-time degradations (wake word falling back to an
                    # open mic, missing extras, MCP servers down) land in the
                    # transcript panel, which is usually closed - notify too.
                    sess.lines.append(f"[talk] {msg}")
                    self._voice_notification(msg)

                loop, cleanup = talk_mod.build_talk_loop(
                    s, base_url=base, api_key=key, model=model,
                    on_display=sess.lines.append, warn=warn)
                loop.backend.gain = self._settings.get("volume", 1.0)
                sess.loop, sess.cleanup = loop, cleanup
                sess.wake = getattr(loop.wake, "name", None)
                sess.lines.append(f"[talk] {model} · voice "
                                  f"{s['voice'] or 'default'}")
                sess.starting = False
                try:
                    loop.run_headless(sess.control)
                finally:
                    cleanup()
            except Exception as e:
                sess.error = str(e)
                sess.lines.append(f"[talk] error: {e}")
                sess.starting = False
                self._voice_notification(f"voice chat failed: {e}")

        sess.thread = threading.Thread(target=boot, daemon=True,
                                       name="menubar-voice")
        sess.thread.start()

    def _voice_notification(self, msg: str) -> None:
        """Post a voice-session notification from the boot worker thread
        (marshaled to the main run loop, like the hotkey fire path)."""
        def post():
            try:
                self._rumps.notification("gmlx voice", None, msg)
            except Exception:
                pass    # notification center unavailable (bare interpreter,
                        # no Info.plist) - the transcript line still has it
        try:
            from PyObjCTools import AppHelper
            AppHelper.callAfter(post)
        except Exception:
            post()

    def _end_voice(self) -> None:
        sess, self._voice = self._voice, None
        self._voice_timer.stop()
        if self._panel is not None:
            self._panel.hide()
        if sess is not None:
            sess.control.put("quit")

    def _voice_command(self, cmd: str) -> None:
        if self._voice is not None:
            self._voice.control.put(cmd)

    def _volume_item(self, volume: float):
        """The output-volume slider, created once and reused across renders:
        rumps registers every slider in a process-lifetime callback map, so
        a fresh one per repaint (two a second during a session) would leak
        NSSliders; a stable instance also keeps the knob where the user left
        it. Never re-set its value programmatically - the user is the only
        writer after creation."""
        if self._volume_slider is None:
            it = self._rumps.SliderMenuItem(
                value=volume * 100.0, min_value=0, max_value=100,
                callback=self._set_volume)
            # Continuous: gain lands per drag event, so the change is heard
            # while the reply is still speaking (not on mouse-up).
            it._slider.setContinuous_(True)
            # rumps installs the bare NSSlider as the item view: no menu
            # indent, and its 15pt default frame clips the ~21pt knob. Re-home
            # it in a padded container matching the text items' inset. Detach
            # it from the item first: replacing an NSMenuItem's view rips the
            # old view out of its superview, so setView_(container) with the
            # slider still installed would empty the container just built.
            from AppKit import NSMakeRect, NSView, NSViewWidthSizable
            slider = it._slider
            it._menuitem.setView_(None)
            slider.setFrame_(NSMakeRect(14, 2, 172, 24))
            slider.setAutoresizingMask_(NSViewWidthSizable)
            container = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, 200, 28))
            container.setAutoresizingMask_(NSViewWidthSizable)
            container.addSubview_(slider)
            it._menuitem.setView_(container)
            self._volume_slider = it
        return self._volume_slider

    def _set_volume(self, slider) -> None:
        vol = min(1.0, max(0.0, slider.value / 100.0))
        self._settings["volume"] = vol
        sess = self._voice
        backend = getattr(getattr(sess, "loop", None), "backend", None)
        if backend is not None:
            backend.gain = vol
        save_menubar_settings(self._settings)

    # --- global tap-to-talk hotkey --------------------------------------------
    def _hotkey_available(self) -> bool:
        """The submenu shows when the voice stack and Quartz are importable -
        server reachability is irrelevant (it's a local setting, and the
        hotkey itself can start a session once the server is back)."""
        if self._hotkey_avail is None:
            import importlib.util
            self._hotkey_avail = (
                sys.platform == "darwin"
                and importlib.util.find_spec("sounddevice") is not None
                and importlib.util.find_spec("Quartz") is not None)
        return self._hotkey_avail

    def _arm_hotkey_async(self, *, alert_on_failure: bool = False) -> None:
        """Start the tap off the menu thread - ``HotkeyTap.start`` blocks on
        the tap thread's readiness (and a raced ``stop`` joins it), so the
        menu must not wait. The outcome lands back on the main run loop."""
        if self._hotkey_arming:
            return
        self._hotkey_arming = True
        from . import hotkey
        tap = hotkey.HotkeyTap(self._ptt_modifier, self._on_hotkey_fire)

        def work():
            try:
                ok = tap.start()
            except Exception:
                ok = False

            def apply():
                self._hotkey_arming = False
                if self._settings.get("hotkey", "off") == "off":
                    self._spawn(tap.stop)    # toggled off while arming
                elif ok:
                    self._hotkey_tap = tap
                    self._hotkey_error = None
                else:
                    # TCC grants can bind to a fresh process only.
                    self._hotkey_error = ("granted - quit and reopen the "
                                          "menu bar to activate")
                    if alert_on_failure:
                        self._rumps.alert(
                            "Tap-to-talk hotkey",
                            "Permission is granted but the hotkey could not "
                            "activate in this process. Quit and reopen the "
                            "menu bar.")
                self._refresh(None)
            try:
                from PyObjCTools import AppHelper
                AppHelper.callAfter(apply)
            except Exception:
                apply()
        self._spawn(work)

    def _rearm_hotkey_silently(self) -> None:
        """At launch: re-arm a persisted hotkey only when the permission is
        already granted. Never prompts - enabling is always a user action.
        When the grant is missing, the "needs permission" note stays until
        :meth:`_refresh` sees the preflight turn true (granting in System
        Settings applies to this live process) and arms."""
        if (self._settings.get("hotkey", "off") == "off"
                or not self._hotkey_available()):
            return
        try:
            from . import hotkey
            if hotkey.preflight():
                self._arm_hotkey_async()
            else:
                self._hotkey_error = "not active - needs permission"
        except Exception as e:
            self._hotkey_error = f"not active - {e}"

    def _repreflight_hotkey(self) -> None:
        """Refresh-tick: an enabled hotkey stuck on "needs permission" arms
        itself once the user grants access in System Settings - no second
        toggle required. Cheap (one AXIsProcessTrusted call per tick)."""
        if (self._settings.get("hotkey", "off") == "off"
                or self._hotkey_tap is not None or self._hotkey_arming
                or "needs permission" not in (self._hotkey_error or "")):
            return
        try:
            from . import hotkey
            if hotkey.preflight():
                self._hotkey_error = None
                self._arm_hotkey_async()
        except Exception:
            pass  # advisory probe; the next tick retries

    def _set_hotkey(self, choice: str) -> None:
        """Menu callback (main run-loop thread): switch the hotkey on or
        off. The one place a TCC prompt can originate. Tap start/stop block
        on their thread handshakes, so both run off this thread; the
        checkmark repaints immediately from the persisted setting."""
        tap, self._hotkey_tap = self._hotkey_tap, None
        if tap is not None:
            self._spawn(tap.stop)
        self._hotkey_error = None
        self._settings["hotkey"] = choice
        save_menubar_settings(self._settings)
        if choice == "off":
            self._refresh(None)
            return
        # Pick up config edits to talk.push_to_talk_modifier on re-enable.
        url_, host, port = self._current_target()
        self._ptt_modifier = ptt_modifier_from_config(self._runinfo(host, port))
        from . import hotkey
        if not (hotkey.preflight() or hotkey.request()):
            self._hotkey_error = "not active - needs permission"
            self._refresh(None)          # paint before the modal alert
            if self._rumps.alert(
                    "Permission needed",
                    "The tap-to-talk hotkey needs Accessibility access for "
                    "gmlx. Grant it in System Settings, then enable the "
                    "hotkey again (or relaunch the menu bar).",
                    ok="Open System Settings", cancel=True) == 1:
                url = hotkey.privacy_pane_url()
                self._spawn(lambda: subprocess.run(["open", url],
                                                   capture_output=True))
            return
        self._arm_hotkey_async(alert_on_failure=True)
        self._refresh(None)

    def _on_hotkey_fire(self) -> None:
        # Tap-thread callback: marshal before touching rumps/session state.
        from PyObjCTools import AppHelper
        AppHelper.callAfter(self._hotkey_fire_main)

    def _hotkey_fire_main(self) -> None:
        if self._voice is None:
            self._start_voice()      # boots into its mode's rest state
        else:
            self._voice.control.put("hotkey")

    def _toggle_hotkey(self) -> None:
        enabled = self._settings.get("hotkey", "off") != "off"
        self._set_hotkey("off" if enabled else "on")

    def _toggle_transcript(self) -> None:
        if self._panel is None:
            self._panel = _TranscriptPanel()
        if self._panel.visible():
            self._panel.hide()
        else:
            if self._voice is not None:
                self._panel.set_text(self._voice.transcript())
            self._panel.show()

    def _show_memory(self) -> None:
        """List stored memories into the transcript panel (opened if hidden);
        the list itself prints through the session's display seam."""
        if self._panel is None:
            self._panel = _TranscriptPanel()
        if self._voice is not None:
            self._panel.set_text(self._voice.transcript())
        self._panel.show()
        self._voice_command("memory")

    def _clear_memory(self) -> None:
        if self._rumps.alert("Clear talk memory?",
                             "Deletes every stored memory for the assistant "
                             "brain. This cannot be undone.",
                             ok="Clear", cancel=True) == 1:
            self._voice_command("memory clear yes")

    # --- view plane (main run-loop thread) ---
    def _refresh(self, _timer) -> None:
        with self._lock:
            snap = dict(self._snapshot)
            host, port = self.host, self.port
        run = self._runinfo(host, port)
        # Death notification: only when the runfile is still there - a clean
        # `gmlx stop` removes it, a crashed server leaves it behind.
        if self._notify.observe(bool(snap.get("reachable"))) and run is not None:
            self._post_down_notification(snap, run)
        session = None
        sess = self._voice
        if sess is not None:
            if not sess.alive() and not sess.error:
                self._end_voice()                # loop exited cleanly (/quit)
            else:
                # An errored session stays visible ("Voice chat: <error>" +
                # End voice chat) so the failure isn't a silent no-op.
                session = sess.snapshot()
                if self._panel is not None and self._panel.visible():
                    self._panel.set_text(sess.transcript())
        if self._logs_panel is not None and self._logs_panel.visible():
            self._logs_panel.tick(self._log_sources())
        hotkey = None
        if self._hotkey_available():
            self._repreflight_hotkey()
            from .hotkey import combo_label
            hotkey = {"enabled": self._settings.get("hotkey", "off") != "off",
                      "available": True, "error": self._hotkey_error,
                      "label": combo_label(self._ptt_modifier)}
        self._render(build_menu_model(snap, run,
                                      talk_model=self._talk_model,
                                      session=session,
                                      fallback_config=_existing_default_config(),
                                      hotkey=hotkey,
                                      volume=self._settings.get("volume")))

    def _post_down_notification(self, snap: dict,
                                run: dict | None = None) -> None:
        from . import lifecycle
        try:
            pid_dead = bool(run) and not lifecycle.identity_ok(run)
        except Exception:
            pid_dead = False
        msg = down_message(snap.get("url") or self.url, run, pid_dead)
        try:
            self._rumps.notification("gmlx", None, msg)
        except Exception:
            pass    # notification center unavailable (bare interpreter, no
                    # Info.plist) - the glyph flip still shows the state

    def _make_unload(self, model_id: str):
        return lambda _sender: self._unload(model_id)

    def _render(self, model: dict) -> None:
        rumps = self._rumps
        self.app.title = model["title"]
        items = [self._disabled(model["header"])]
        if model.get("busy_line"):
            items.append(self._disabled(model["busy_line"]))
        items.append(rumps.separator)

        if model["models"]:
            parent = rumps.MenuItem(model["models_header"])
            for m in model["models"]:
                if m["unloadable"]:
                    parent.add(rumps.MenuItem(f"Unload  {m['label']}",
                                              callback=self._make_unload(m["id"])))
                else:
                    parent.add(self._disabled(m["label"]))
            items.append(parent)
            items.append(rumps.separator)
        elif model["models_header"]:
            items.append(self._disabled(model["models_header"]))
            items.append(rumps.separator)

        sess = model.get("talk_session")
        if sess:
            items.append(self._disabled(sess["line"]))
            if sess["busy"]:
                items.append(rumps.MenuItem(
                    "Stop speaking",
                    callback=lambda _s: self._voice_command("stop")))
            items.append(rumps.MenuItem(
                "Unmute mic" if sess["muted"] else "Mute mic",
                callback=lambda _s: self._voice_command("mute")))
            if sess.get("volume") is not None:
                items.append(self._disabled("Volume"))
                items.append(self._volume_item(sess["volume"]))
            showing = self._panel is not None and self._panel.visible()
            items.append(rumps.MenuItem(
                "Hide transcript" if showing else "Show transcript",
                callback=lambda _s: self._toggle_transcript()))
            if sess.get("has_memory"):
                items.append(rumps.MenuItem(
                    "Show memory",
                    callback=lambda _s: self._show_memory()))
                items.append(rumps.MenuItem(
                    "Clear memory…",
                    callback=lambda _s: self._clear_memory()))
            items.append(rumps.MenuItem(
                "End voice chat", callback=lambda _s: self._end_voice()))
            items.append(rumps.separator)
        if model.get("can_talk"):
            items.append(rumps.MenuItem(model.get("talk_label", "Talk to model…"),
                                        callback=lambda _s: self._start_voice()))
            items.append(rumps.MenuItem("Talk in a terminal…",
                                        callback=lambda _s: self._open_talk()))
        if model["can_reload"]:
            items.append(rumps.MenuItem("Reload config",
                                        callback=lambda _s: self._reload()))
        if model.get("config_path"):
            cfg_path = model["config_path"]
            items.append(rumps.MenuItem(
                "Edit config…",
                callback=lambda _s: self._edit_config(cfg_path)))
        # With the settings group, not the talk actions: the talk items come
        # and go with server state, which would bounce this to the top slot.
        if model.get("hotkey"):
            hk = model["hotkey"]
            it = rumps.MenuItem(f"Tap-to-talk with {hk['label']}",
                                callback=lambda _s: self._toggle_hotkey())
            it.state = 1 if hk["enabled"] else 0
            items.append(it)
            if hk.get("error") and hk["enabled"]:
                items.append(self._disabled(f"  {hk['error']}"))
        if model["can_start"]:
            items.append(rumps.MenuItem("Start server",
                                        callback=lambda _s: self._restart()))
        if model["can_restart"]:
            items.append(rumps.MenuItem("Restart server",
                                        callback=lambda _s: self._restart()))
        if model["can_stop"]:
            items.append(rumps.MenuItem("Stop server",
                                        callback=lambda _s: self._stop()))
        if model["managed_by"] == "launchd":
            items.append(self._disabled("(managed by launchd - "
                                        "`gmlx service uninstall` to remove)"))
        if model["log"]:
            items.append(rumps.MenuItem("Open logs",
                                        callback=lambda _s: self._show_logs()))
        if model.get("url"):
            items.append(rumps.MenuItem("Copy server URL",
                                        callback=lambda _s: self._copy_url()))
        items.append(rumps.separator)
        items.append(rumps.MenuItem("Quit", callback=lambda _s: self._quit()))

        self.app.menu.clear()
        self.app.menu.update(self._dedupe_separators(items))

    def _dedupe_separators(self, items: list) -> list:
        """Collapse adjacent separators and drop a trailing one - an omitted models
        section or control set would otherwise leave doubled / dangling separators."""
        sep = self._rumps.separator
        out: list = []
        for it in items:
            if it is sep and (not out or out[-1] is sep):
                continue
            out.append(it)
        while out and out[-1] is sep:
            out.pop()
        return out

    def _disabled(self, title: str):
        item = self._rumps.MenuItem(title)
        item.set_callback(None)                  # greyed-out, non-clickable label
        return item

    def _quit(self) -> None:
        if self._hotkey_tap is not None:
            self._hotkey_tap.stop()
            self._hotkey_tap = None
        self._stop_evt.set()
        self._rumps.quit_application()

    def run(self) -> int:
        self._worker.start()
        self._rearm_hotkey_silently()
        self._refresh(None)                      # paint once before the first tick
        self._timer.start()
        self.app.run()
        return 0


def cmd_menubar(argv: list | None = None,
                prog: str = "gmlx launch menubar") -> int:
    ap = argparse.ArgumentParser(
        prog=prog,
        description="macOS menu-bar monitor for a backgrounded gmlx server: shows "
                    "status and load, lists resident models with an unload action, "
                    "offers reload / restart / stop / copy-URL / open-logs over the "
                    "existing endpoints, opens a validating editor on the server's "
                    "config, and notifies if the server dies. Detaches by default "
                    "(a background `serve` raises it for you); `--foreground` "
                    "runs the event loop in this terminal.")
    ap.add_argument("-f", "--foreground", action="store_true",
                    help="Run the menu-bar event loop in this process (blocking) "
                         "instead of the default detached start.")
    ap.add_argument("--stop", action="store_true",
                    help="Quit a detached menu-bar monitor and exit.")
    ap.add_argument("--url", default=None, metavar="URL",
                    help="Server base URL (default: the managed server's, else "
                         "http://127.0.0.1:8080).")
    ap.add_argument("--api-key", default=None, metavar="KEY",
                    help="Key for a key-protected server (default: the managed "
                         "server's own `server.api_key`, read from its config).")
    ap.add_argument("--interval", type=float, default=4.0, metavar="S",
                    help="Poll interval in seconds (default 4).")
    ap.add_argument("--host", default=None,
                    help="Managed-server host to track (default: the single one, "
                         "else 127.0.0.1).")
    ap.add_argument("--port", type=int, default=None,
                    help="Managed-server port to track (default: the single one, "
                         "else 8080).")
    # Internal: set by lifecycle.start_menubar(auto=True) so the foreground child
    # records itself as auto-raised (torn down when the last server stops).
    ap.add_argument("--auto-raised", action="store_true", help=argparse.SUPPRESS)
    # Internal: set in the LaunchAgent plist (lifecycle.service_install).
    ap.add_argument("--launchd", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    if a.launchd:
        # launchd-parented: re-exec through the (just refreshed) app bundle
        # so ps and TCC prompts read "gmlx" - see procname.launchd_reexec.
        from . import procname
        procname.launchd_reexec(procname.menubar_bundle,
                                ["launch", "menubar", *argv])

    from . import lifecycle
    if a.stop:
        if lifecycle.menubar_alive() and lifecycle.menubar_agent_loaded():
            print("the menu bar runs as a launchd agent (killing it would "
                  "just respawn it) - quit it from its own menu, or remove "
                  "the agent with `gmlx service uninstall`", file=sys.stderr)
            return 1
        if lifecycle.stop_menubar():
            print("stopped the menu-bar monitor")
        else:
            print("no menu-bar monitor running")
        return 0
    # An explicit --url/--host/--port pins the bar to that one server; otherwise it
    # tracks the primary server (auto_target), following it as servers come and go.
    explicit = a.url is not None or a.host is not None or a.port is not None

    # Default: detach a foreground child so the shell stays free (symmetric with serve).
    if not a.foreground:
        if not lifecycle.gui_session_available():
            print("error: the menu bar needs a macOS GUI session "
                  "(not available over SSH / off macOS)", file=sys.stderr)
            return 2
        extra: list = []
        if a.url:
            extra += ["--url", a.url]
        if a.host:
            extra += ["--host", a.host]
        if a.port is not None:
            extra += ["--port", str(a.port)]
        if a.api_key:
            extra += ["--api-key", a.api_key]
        if a.interval != 4.0:
            extra += ["--interval", str(a.interval)]
        rc = lifecycle.start_menubar(extra=extra)
        if rc == 0:
            if explicit:
                host, port = lifecycle.auto_target(a.host, a.port)
                where = a.url or f"http://{host}:{port}"
            else:
                where = "the primary server"
            print(f"menu bar monitoring {where} (quit it from its own menu)")
        else:
            print("error: could not start the menu bar", file=sys.stderr)
        return rc

    try:
        import rumps  # noqa: F401
    except ImportError:
        print("error: the menu bar needs rumps (macOS only) - install with "
              "`pip install rumps`, or reinstall gmlx on macOS where it's a "
              "default dependency", file=sys.stderr)
        return 2

    if lifecycle.menubar_alive():
        # The detached path no-ops via start_menubar; the foreground path must
        # refuse too - overwriting the machine-wide pidfile would orphan the
        # running bar (invisible to menubar_alive/stop once this process's
        # exit removes the record).
        print("error: a menu bar is already running - stop it first "
              "(gmlx launch menubar --stop)", file=sys.stderr)
        return 1
    lifecycle.write_menubar_run(os.getpid(), auto=a.auto_raised)
    if a.launchd:
        # First bar of this login starts the recorded server (off the GUI
        # thread - launch_detached blocks on server readiness).
        threading.Thread(target=_autostart_server_once, daemon=True,
                         name="menubar-autostart").start()
    try:
        if explicit:
            host, port = a.host, a.port
            if a.url and host is None and port is None:
                host, port = _split_url(a.url)
            host, port = lifecycle.auto_target(host, port)
            url = a.url or f"http://{host}:{port}"
            app = _MenuBarApp(url, a.api_key, a.interval, host, port, dynamic=False)
        else:
            app = _MenuBarApp(None, a.api_key, a.interval, None, None, dynamic=True)
        return app.run()
    finally:
        lifecycle.remove_menubar_run()
