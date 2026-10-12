"""No test writes to the Hugging Face cache of the user running the suite.

A test that needs a cache makes one under tmp_path (see
tests/serve/fake_hub.py). :func:`install` adds an audit hook that refuses
every write under the user's cache folders, as an open for writing, a new
folder, a link, a rename or a removal, by raising PermissionError before
the write happens. Each refused write is recorded in :data:`WRITES`, and
conftest.py fails the test that made it, so code that swallows the error
still fails its test. The hook sees this process only, not the
subprocesses a test starts.
"""
from __future__ import annotations

import os
import sys

# The guarded folders, each ending in a separator.
ROOTS: tuple = ()
# (event, path) of each refused write, until a test's check takes them.
WRITES: list = []

_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
# Audit event -> the indices of its arguments that name a path it writes.
_WRITE_ARGS = {
    "os.mkdir": (0,), "os.remove": (0,), "os.rmdir": (0,),
    "os.truncate": (0,), "os.chmod": (0,), "os.chown": (0,),
    "os.utime": (0,), "os.chflags": (0,), "os.setxattr": (0,),
    "os.removexattr": (0,), "shutil.rmtree": (0,),
    "os.rename": (0, 1), "shutil.move": (0, 1),
    "os.symlink": (1,), "os.link": (1,),
    "shutil.copyfile": (1,), "shutil.copytree": (1,),
}
_installed = False


def roots(environ) -> tuple:
    """The Hugging Face cache folders that ``environ`` names: HF_HOME, by
    default ``~/.cache/huggingface``, and the hub cache in it or at
    HF_HUB_CACHE. Both the given and the resolved form of each."""
    home = environ.get("HF_HOME") or os.path.join(
        environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"),
        "huggingface")
    hub = (environ.get("HF_HUB_CACHE") or environ.get("HUGGINGFACE_HUB_CACHE")
           or os.path.join(home, "hub"))
    out = set()
    for r in (home, hub):
        r = os.path.expanduser(r)
        out |= {os.path.abspath(r) + os.sep, os.path.realpath(r) + os.sep}
    return tuple(sorted(out))


def _hook(event, args):
    if event == "open":
        if len(args) < 3 or not isinstance(args[2], int) \
                or not args[2] & _WRITE_FLAGS:
            return
        paths = (args[0],)
    else:
        index = _WRITE_ARGS.get(event)
        if index is None:
            return
        paths = tuple(args[i] for i in index if i < len(args))
    for p in paths:
        if not isinstance(p, (str, bytes, os.PathLike)):
            continue
        path = os.fsdecode(p)
        if event == "os.mkdir" and os.path.lexists(path):
            continue            # it fails on what is there and writes nothing
        if (os.path.abspath(path) + os.sep).startswith(ROOTS):
            WRITES.append((event, path))
            raise PermissionError(
                f"a test wrote to the Hugging Face cache of the user running "
                f"the suite ({event} {path}); make a cache under tmp_path")


def install(environ) -> None:
    """Guard the caches ``environ`` names, once per process."""
    global ROOTS, _installed
    ROOTS = roots(environ)
    if not _installed:
        sys.addaudithook(_hook)
        _installed = True


def take(start: int) -> list:
    """The writes refused since ``len(WRITES)`` was ``start``, removed."""
    new = WRITES[start:]
    del WRITES[start:]
    return new


def failure(writes: list) -> str | None:
    """The failure message for ``writes``, refused writes as :func:`take`
    returns them, or None when there are none."""
    if not writes:
        return None
    lines = "\n".join(f"  {event} {path}" for event, path in writes[:20])
    more = f"\n  and {len(writes) - 20} more" if len(writes) > 20 else ""
    return (f"{len(writes)} write(s) to the Hugging Face cache of the user "
            f"running the suite were refused; make a cache under tmp_path "
            f"(tests/serve/fake_hub.py):\n{lines}{more}")
