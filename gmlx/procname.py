"""macOS process identity for long-lived children (serve daemon, menu bar).

A venv interpreter resolves to the framework Python.app stub, so backgrounded
processes show up as "Python" in ps / Activity Monitor, and the menu bar's
notification-permission prompt is attributed to "Python" too. The fix is a
copy (not a symlink - execve resolves symlinks before setting the kernel proc
name) of the ~50 KB interpreter stub under a `gmlx` filename. Framework
builds (Homebrew, python.org) link their runtime by absolute path so the copy
runs from anywhere; python-build-standalone builds (uv-managed pythons) link
`@executable_path/../lib/libpythonX.Y.dylib`, which :func:`_copy_stub`
satisfies with a sibling `lib` symlink next to the copy. The copy lives
outside the venv, so children must be spawned with PYTHONEXECUTABLE pointing
at the venv interpreter (framework builds honor it when computing
sys.executable / sys.prefix); `cli.umbrella_main` pops the variable at startup
so it never leaks to grandchildren (MCP tool servers, launchd agents, ...).

The menu bar additionally runs from a minimal app bundle: notification
attribution follows NSBundle.mainBundle of the running executable, so a bare
renamed stub still reports the interpreter's bundle. The bundle is re-signed
ad-hoc best-effort; every helper degrades to ``None`` (caller falls back to
``sys.executable``) rather than raising into the serve path.

TCC (microphone, notifications) keys its grants to the binary's code
signature, and an ad-hoc signature is a per-binary hash: rewriting the copy
on every launch would silently revoke the user's grants on every restart.
Hence the sidecar stamp in :func:`_copy_stub` - the copy is only rewritten
when the source stub itself changed (interpreter upgrade), which necessarily
re-prompts.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROC_NAME = "gmlx"
BUNDLE_ID = "org.gmlx.menubar"

_INFO_PLIST = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleName</key><string>{PROC_NAME}</string>
  <key>CFBundleDisplayName</key><string>{PROC_NAME}</string>
  <key>CFBundleIdentifier</key><string>{BUNDLE_ID}</string>
  <key>CFBundleExecutable</key><string>{PROC_NAME}</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>LSUIElement</key><true/>
  <key>NSMicrophoneUsageDescription</key>
  <string>Voice chat listens for the wake word and your questions.</string>
</dict>
</plist>
"""


def _proc_dir() -> Path:
    return Path(os.path.expanduser("~/.cache/gmlx/proc"))


def _app_dir() -> Path:
    """Home of the gmlx.app bundle. Application Support, not the cache: the
    launchd LaunchAgent references the bundle by absolute path, and cache
    cleaners feel entitled to delete ~/.cache - a broken login item is a
    worse failure than a re-copied stub."""
    return Path(os.path.expanduser("~/Library/Application Support/gmlx"))


def _stub_path() -> str:
    """The executable actually loaded in this process. Not
    ``realpath(sys.executable)``: on Homebrew framework builds that lands on
    the thin bin/python3.x launcher, which immediately re-execs the
    Python.app binary - a copy of the launcher would re-exec too and the
    child would show up as "Python" again. ``_NSGetExecutablePath`` reports
    the post-re-exec binary, which runs standalone and keeps its exec name."""
    import ctypes
    buf = ctypes.create_string_buffer(4096)
    size = ctypes.c_uint32(len(buf))
    dyld = ctypes.CDLL(None)
    if dyld._NSGetExecutablePath(buf, ctypes.byref(size)) == 0:
        return os.path.realpath(buf.value.decode())
    return os.path.realpath(sys.executable)


def _link_runtime_lib(dest: Path, src: str) -> None:
    """python-build-standalone interpreters (uv-managed pythons) link
    libpython as ``@executable_path/../lib/libpythonX.Y.dylib``, so a bare
    copy of the binary aborts in dyld. Keep a ``lib`` symlink beside the
    copy's parent directory resolving that reference back to the source
    interpreter's own lib/. Framework builds link by absolute path and have
    no sibling lib/, so nothing is created (and a link left by a previous
    interpreter is removed). A symlink keeps the copy's bytes - and its
    TCC-keyed CDHash - untouched."""
    target = Path(src).parent.parent / "lib"
    link = dest.parent.parent / "lib"
    if link.is_symlink():
        if target.is_dir() and os.readlink(link) == str(target):
            return
        link.unlink()
    elif link.exists():
        return                       # a real directory - not ours to manage
    if target.is_dir():
        link.symlink_to(target)


def _atomic_write(dest: Path, data, *, mode: int | None = None) -> None:
    """Write ``data`` (str or bytes) to ``dest`` via a unique temp file in the
    same directory + rename. The unique name (not a fixed ``.tmp``) avoids
    racing a concurrent writer, and the temp is removed if the write raises so
    no partial file is left behind."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(dest.parent),
                                    prefix=dest.name + ".", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb" if isinstance(data, bytes) else "w") as f:
            f.write(data)
        if mode is not None:
            tmp.chmod(mode)
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _copy_stub(dest: Path, stamp: Path | None = None) -> bool:
    """Place a current copy of the interpreter stub at ``dest``. Skips the
    copy when one from this exact stub is already there, judged by a sidecar
    stamp of the source's identity - not by the copy's own size/mtime, which
    codesign rewrites in place: re-copying (and re-signing) on every launch
    would give the binary a fresh CDHash each time and silently revoke the
    TCC grants (microphone, notifications) keyed to it. Either way the
    runtime-lib symlink is (re)ensured - see :func:`_link_runtime_lib`.
    Returns True when the file was (re)written."""
    src = _stub_path()
    st = os.stat(src)
    stamp = stamp or dest.with_name(dest.name + ".src")
    want = f"{src} {st.st_size} {st.st_mtime}"
    try:
        if dest.exists() and stamp.read_text() == want:
            _link_runtime_lib(dest, src)
            return False
    except OSError:
        pass
    _atomic_write(dest, Path(src).read_bytes(), mode=0o755)
    stamp.write_text(want)
    _link_runtime_lib(dest, src)
    return True


def child_env() -> dict:
    """Environment for a child exec'd through a renamed stub: the venv
    interpreter path, so getpath still lands in this venv."""
    env = dict(os.environ)
    env["PYTHONEXECUTABLE"] = os.path.abspath(sys.executable)
    return env


def launchd_reexec(refresh, argv_tail: list) -> None:
    """Boot shim for launchd agents: their plists exec the venv interpreter
    (a path that survives interpreter upgrades), and the entry point calls
    this to re-exec the same pid through a freshly-refreshed stub/bundle so
    ps and TCC see "gmlx". Refreshing *before* the exec is the point - a
    plist that execs a copied stub directly crash-loops in dyld forever
    after an interpreter swap, since no gmlx code runs before the exec.

    ``refresh`` is :func:`named_python` or :func:`menubar_bundle`; on None
    or exec failure this returns and the caller keeps running under the
    venv interpreter (shows as "Python" - better than a crash loop). The
    env guard keeps the exec'd process from re-execing in turn."""
    if os.environ.pop("GMLX_LAUNCHD_REEXEC", None) is not None:
        return
    try:
        target = refresh()
    except Exception:
        return
    if not target:
        return
    env = child_env()
    env["GMLX_LAUNCHD_REEXEC"] = "1"
    try:
        os.execve(target, [target, "-m", "gmlx", *argv_tail], env)
    except OSError:
        return


def named_python() -> str | None:
    """Path to a `gmlx`-named interpreter copy, or None (non-macOS / failure).
    Spawn with :func:`child_env` so the child resolves this venv."""
    if sys.platform != "darwin":
        return None
    try:
        dest = _proc_dir() / PROC_NAME
        _copy_stub(dest)
        return str(dest)
    except OSError:
        return None


def menubar_bundle() -> str | None:
    """Executable path inside a minimal gmlx.app bundle (so notification
    prompts read "gmlx"), or None. Spawn with :func:`child_env`."""
    if sys.platform != "darwin":
        return None
    try:
        base = _app_dir()
        contents = base / f"{PROC_NAME}.app" / "Contents"
        plist = contents / "Info.plist"
        if not plist.exists() or plist.read_text() != _INFO_PLIST:
            # Atomic write: an interrupted write must not leave a truncated
            # Info.plist for a later launch to read.
            _atomic_write(plist, _INFO_PLIST, mode=0o644)
        exe = contents / "MacOS" / PROC_NAME
        # Stamp outside the bundle: codesign seals the bundle's subcomponents
        # and refuses to sign with a foreign file in Contents/MacOS.
        if _copy_stub(exe, stamp=base / f"{PROC_NAME}.app.src"):
            # Ad-hoc signature matching the bundle id; attribution still works
            # unsigned on current macOS, so failure is fine.
            subprocess.run(
                ["codesign", "-s", "-", "-f", "--identifier", BUNDLE_ID,
                 str(exe)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                check=False)
        _retire_cache_bundle()
        return str(exe)
    except OSError:
        return None


def agent_trampoline() -> str | None:
    """A signed sh trampoline inside gmlx.app that execs the bundle binary
    with the plist's arguments - launchd agents point their ProgramArguments
    here, or None (non-macOS / failure). Two attribution systems key on two
    different things, and the trampoline is what satisfies both:

    - Login Items (BTM) names an agent after the plist's executable. The
      venv python there reads as a "Python" login item; a script inside
      gmlx.app reads as gmlx.
    - TCC pins a process's identity at its first NON-PLATFORM exec. sh is a
      platform binary and passes through; whatever comes next is what the
      microphone/Accessibility prompts name forever (exec'ing the bundle
      binary afterwards is too late). So the script must exec the bundle
      binary directly, not the venv python - a python in the middle makes
      every prompt say "python3.12".

    The stale-copy hazard (interpreter swap -> dyld abort -> KeepAlive
    crash loop) is handled by probing the bundle binary first: on failure
    the script falls back to the venv python, which boots attributed as
    python for that one session while its entry point refreshes the bundle
    for the next launch (procname.launchd_reexec). The happy path sets the
    reexec guard itself - it is already running the right binary."""
    exe = menubar_bundle()
    if exe is None:
        return None
    try:
        script = Path(exe).parent / f"{PROC_NAME}-agent"
        py = os.path.abspath(sys.executable)
        body = ('#!/bin/sh\n'
                f'BIN="{exe}"\n'
                f'PY="{py}"\n'
                'export PYTHONEXECUTABLE="$PY"\n'
                'if "$BIN" -c "" 2>/dev/null; then\n'
                '    export GMLX_LAUNCHD_REEXEC=1\n'
                '    exec "$BIN" -m gmlx "$@"\n'
                'fi\n'
                'exec "$PY" -m gmlx "$@"\n')
        try:
            if script.read_text() == body:
                return str(script)
        except OSError:
            pass
        _atomic_write(script, body, mode=0o755)
        # The bundle exe's codesign treats MacOS/ siblings as subcomponents
        # and requires them signed; a script signature lives in xattrs, so
        # this never touches the exe's TCC-keyed CDHash.
        subprocess.run(["codesign", "-s", "-", "-f", str(script)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=False)
        return str(script)
    except OSError:
        return None


def _retire_cache_bundle() -> None:
    """Remove the pre-relocation bundle from the cache dir (best-effort) so
    stale launch paths die cleanly instead of running old copies; TCC keys
    on the bundle id + CDHash, so grants follow the relocated bundle."""
    import shutil
    old = _proc_dir() / f"{PROC_NAME}.app"
    if old.exists():
        shutil.rmtree(old, ignore_errors=True)
    try:
        (_proc_dir() / f"{PROC_NAME}.app.src").unlink()
    except OSError:
        pass
