"""Config editor behind the menu bar's "Edit config" item.

The editing session - load, validate, conflict-checked atomic save - lives in
:class:`ConfigDraft`, plain file and parser logic with no AppKit, unit-tested
directly. :class:`ConfigPanel` is the thin floating-window shell over it: an
editable text view plus Validate / Revert / Save / Save & Reload / Open in
Editor, main run-loop thread only (same threading contract as the transcript
panel in :mod:`.menubar`).

Validation runs the server's own parser (:func:`gmlx.config.build_config`),
so the verdict here is exactly the verdict ``gmlx serve`` would give: strict
errors for structural typos, soft warnings for open-ended namespaces, plus a
does-each-model-path-resolve check. That check is the one thing an external
text editor can never offer, and it is the reason this panel exists.
"""
from __future__ import annotations

import os
import stat
import tempfile
import warnings


def _one_line(msg: str, limit: int = 400) -> str:
    """Flatten a (possibly multi-line YAML) error for a single status row."""
    flat = " ".join(str(msg).split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def validate_config_text(text: str) -> tuple:
    """Parse a draft config exactly as the server would -> ``(ok, message)``.

    Mirrors :func:`gmlx.config.load_config` (YAML -> mapping ->
    ``build_config``) on an in-memory string. Soft ``warnings.warn`` emissions
    (unknown sampling/load/cache knobs, unknown family) and model paths that do
    not resolve under ``model_dirs`` are reported in the message but leave
    ``ok`` true - the server itself would still start on such a config."""
    import yaml

    from .config import ConfigError, build_config, resolve_path

    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as e:
        return False, _one_line(f"malformed YAML: {e}")
    if doc is not None and not isinstance(doc, dict):
        return False, f"config root must be a mapping, got {type(doc).__name__}"
    soft: list = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            cfg = build_config(doc or {})
        except ConfigError as e:
            return False, _one_line(str(e))
        except Exception as e:  # a parser bug must not read as "config OK"
            return False, _one_line(f"{type(e).__name__}: {e}")
    soft.extend(str(w.message) for w in caught)
    for mid, m in cfg.models.items():
        try:
            resolve_path(m.path, cfg.model_dirs)
        except Exception as e:
            soft.append(_one_line(f"model {mid!r}: {e}"))
    verdict = f"Config OK - {len(cfg.models)} model(s), {len(cfg.profiles)} profile(s)"
    if soft:
        verdict += f"; {len(soft)} warning(s): {_one_line(soft[0])}"
    return True, verdict


class ConfigDraft:
    """One file's editing session: load, validate, and atomically save with an
    mtime conflict check. ``save`` refuses (returns ``(False, why)``) when the
    file changed on disk after :meth:`load`, unless ``force`` - the caller
    decides whether a retry means "overwrite anyway"."""

    def __init__(self, path: str):
        self.path = str(path)
        self._mtime_ns: int | None = None

    def load(self) -> str:
        with open(self.path) as f:
            text = f.read()
        self._mtime_ns = os.stat(self.path).st_mtime_ns
        return text

    def changed_on_disk(self) -> bool:
        if self._mtime_ns is None:
            return False        # nothing loaded yet: no baseline to conflict with
        try:
            return os.stat(self.path).st_mtime_ns != self._mtime_ns
        except OSError:
            return False        # deleted underneath us: save() just recreates it

    def validate(self, text: str) -> tuple:
        return validate_config_text(text)

    def save(self, text: str, force: bool = False) -> tuple:
        """Atomic write (temp file + rename in the config's directory),
        preserving the file's permission bits - a config may hold an api_key,
        so a fresh file is created 0600."""
        if not force and self.changed_on_disk():
            return False, ("File changed on disk since you loaded it - "
                           "Revert to pick up the changes, or Save again "
                           "to overwrite them.")
        try:
            mode = stat.S_IMODE(os.stat(self.path).st_mode)
        except OSError:
            mode = 0o600
        d = os.path.dirname(os.path.abspath(self.path))
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".gmlx-config-")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(text)
            os.chmod(tmp, mode)
            os.replace(tmp, self.path)
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        self._mtime_ns = os.stat(self.path).st_mtime_ns
        return True, "Saved."


# pyobjc classes are process-global: define the button target once, lazily, so
# the module stays importable (and ConfigDraft testable) off macOS.
_BTN_TARGET_CLS = None


def _target_class():
    global _BTN_TARGET_CLS
    if _BTN_TARGET_CLS is None:
        from Foundation import NSObject

        class _GmlxConfigBtnTarget(NSObject):
            def act_(self, _sender):
                self.handler()

        _BTN_TARGET_CLS = _GmlxConfigBtnTarget
    return _BTN_TARGET_CLS


def _install_edit_menu() -> None:
    """Give the (menu-bar-only) app a main-menu Edit menu so Cmd-C/V/X/Z/A key
    equivalents reach the text view - without one, an LSUIElement app cannot
    paste. The menu is never displayed; it exists purely to route keys."""
    from AppKit import NSApplication, NSMenu, NSMenuItem
    app = NSApplication.sharedApplication()
    main = app.mainMenu()
    if main is not None and main.itemWithTitle_("Edit") is not None:
        return
    if main is None:
        main = NSMenu.alloc().init()
        app.setMainMenu_(main)
    edit = NSMenu.alloc().initWithTitle_("Edit")
    for title, sel, key in (("Undo", "undo:", "z"), ("Redo", "redo:", "Z"),
                            ("Cut", "cut:", "x"), ("Copy", "copy:", "c"),
                            ("Paste", "paste:", "v"),
                            ("Select All", "selectAll:", "a")):
        edit.addItem_(
            NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, sel, key))
    holder = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Edit", None, "")
    main.addItem_(holder)
    main.setSubmenu_forItem_(edit, holder)


class ConfigPanel:
    """Floating editable panel over a :class:`ConfigDraft`. All methods run on
    the main run-loop thread. ``on_reload`` fires the bar's existing
    reload-config action after a successful Save & Reload; ``on_open_editor``
    opens the file in the external default text editor (the escape hatch for
    people who want a real editor)."""

    def __init__(self, path: str, on_reload, on_open_editor):
        from AppKit import (NSBackingStoreBuffered, NSClosableWindowMask,
                            NSFont, NSMakeRect, NSMakeSize, NSPanel,
                            NSResizableWindowMask, NSScrollView, NSTextField,
                            NSTextView, NSTitledWindowMask,
                            NSUtilityWindowMask)
        self.draft = ConfigDraft(path)
        self.path = self.draft.path
        self._on_reload = on_reload
        self._on_open_editor = on_open_editor
        self._baseline: str | None = None   # text at last load/save
        self._force_save = False               # armed by a save conflict
        self._targets: list = []               # retain the button targets
        _install_edit_menu()

        mask = (NSTitledWindowMask | NSClosableWindowMask
                | NSResizableWindowMask | NSUtilityWindowMask)
        self.panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, 640, 460), mask, NSBackingStoreBuffered, False)
        home = os.path.expanduser("~")
        shown = self.path.replace(home, "~", 1) if self.path.startswith(home) \
            else self.path
        self.panel.setTitle_(f"Server config - {shown}")
        self.panel.setFloatingPanel_(True)
        self.panel.setReleasedWhenClosed_(False)   # user close = hide, reusable
        # Utility panels default hidesOnDeactivate=YES: without this the
        # editor vanishes the moment the user clicks any other app.
        self.panel.setHidesOnDeactivate_(False)
        self.panel.setMinSize_(NSMakeSize(480, 320))

        scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(0, 76, 640, 384))
        scroll.setHasVerticalScroller_(True)
        scroll.setAutoresizingMask_(18)            # width + height sizable
        self.text = NSTextView.alloc().initWithFrame_(scroll.bounds())
        self.text.setFont_(NSFont.userFixedPitchFontOfSize_(12.0))
        self.text.setRichText_(False)
        self.text.setAllowsUndo_(True)
        # Smart substitutions would silently corrupt YAML (curly quotes, dashes).
        self.text.setAutomaticQuoteSubstitutionEnabled_(False)
        self.text.setAutomaticDashSubstitutionEnabled_(False)
        self.text.setAutomaticTextReplacementEnabled_(False)
        self.text.setAutomaticSpellingCorrectionEnabled_(False)
        self.text.setContinuousSpellCheckingEnabled_(False)
        self.text.setAutoresizingMask_(18)
        scroll.setDocumentView_(self.text)
        self.panel.contentView().addSubview_(scroll)

        self.status = NSTextField.labelWithString_("")
        self.status.setFrame_(NSMakeRect(12, 46, 616, 22))
        self.status.setFont_(NSFont.systemFontOfSize_(11.0))
        self.status.setSelectable_(True)           # errors should be copyable
        self.status.setAutoresizingMask_(2 | 32)   # width sizable, pinned bottom
        self.panel.contentView().addSubview_(self.status)

        for title, x, w, m, fn in (
                ("Open in Editor", 12, 120, 4 | 32, self._open_editor),
                ("Revert", 252, 80, 1 | 32, self._revert),
                ("Validate", 340, 84, 1 | 32, self._validate),
                ("Save", 432, 70, 1 | 32, self._save),
                ("Save & Reload", 510, 118, 1 | 32, self._save_reload)):
            self._add_button(title, x, w, m, fn)
        self.panel.center()

    def _add_button(self, title, x, w, mask, handler) -> None:
        from AppKit import NSButton, NSMakeRect
        b = NSButton.alloc().initWithFrame_(NSMakeRect(x, 10, w, 28))
        b.setTitle_(title)
        b.setBezelStyle_(1)                        # rounded push button
        t = _target_class().alloc().init()
        t.handler = handler
        b.setTarget_(t)
        b.setAction_("act:")
        b.setAutoresizingMask_(mask)
        self._targets.append(t)
        self.panel.contentView().addSubview_(b)

    # --- view helpers ---
    def _get_text(self) -> str:
        return str(self.text.string())

    def _set_text(self, s: str) -> None:
        self.text.setString_(s)

    def _status(self, msg: str) -> None:
        self.status.setStringValue_(msg)

    def visible(self) -> bool:
        return bool(self.panel.isVisible())

    def hide(self) -> None:
        self.panel.orderOut_(None)

    def show(self) -> None:
        """Open (or re-front) the editor. With no unsaved edits the text is
        re-read from disk, so reopening picks up outside changes; unsaved
        edits are kept."""
        if self._baseline is None or self._get_text() == self._baseline:
            self._load(status="")
        from AppKit import NSApplication
        # An agent (LSUIElement) app never auto-activates; without this the
        # panel fronts but keystrokes keep going to the previous app.
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        self.panel.makeKeyAndOrderFront_(None)

    # --- actions ---
    def _load(self, status: str = "Reverted to the file on disk.") -> None:
        try:
            text = self.draft.load()
        except FileNotFoundError:
            self._baseline = ""
            self._status(f"New file - {self.path} does not exist yet; "
                         "Save will create it.")
            return
        except OSError as e:
            self._status(_one_line(f"Could not read {self.path}: {e}"))
            return
        self._set_text(text)
        self._baseline = text
        self._force_save = False
        if status:
            self._status(status)

    def _revert(self) -> None:
        self._load()

    def _validate(self) -> None:
        _ok, msg = self.draft.validate(self._get_text())
        self._status(msg)

    def _save(self, quiet: bool = False) -> bool:
        text = self._get_text()
        try:
            saved, msg = self.draft.save(text, force=self._force_save)
        except OSError as e:
            self._status(_one_line(f"Could not save: {e}"))
            return False
        if not saved:
            self._force_save = True                # next Save press overwrites
            self._status(msg)
            return False
        self._baseline = text
        self._force_save = False
        if not quiet:
            ok, verdict = self.draft.validate(text)
            self._status("Saved." if ok else f"Saved - but: {verdict}")
        return True

    def _save_reload(self) -> None:
        ok, verdict = self.draft.validate(self._get_text())
        if not ok:
            self._status(f"Not saved - fix first: {verdict}")
            return
        if self._save(quiet=True):
            self._on_reload()
            self._status(f"{verdict}. Saved - server reloading.")

    def _open_editor(self) -> None:
        self._on_open_editor()
