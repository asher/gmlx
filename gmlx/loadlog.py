"""Load-time console output routing.

Model loading emits diagnostic lines (``[arch]``, ``[gguf]``, ``[patch]``, ...).
This module decides where they go: printed verbatim (``verbose=True``),
silenced behind a progress spinner (the default CLI experience), or silenced
entirely (the default for library callers and serve bridges).

State lives in a ContextVar so concurrent loads on worker threads (serve
prefetch) cannot leak each other's verbosity or spinner.
"""
from __future__ import annotations

import functools
import os
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar


class _State:
    __slots__ = ("verbose", "spinner", "base", "facts", "stage_label", "stray")

    def __init__(self, verbose: bool, spinner=None, base: str = ""):
        self.verbose = verbose
        self.spinner = spinner
        self.base = base
        self.facts: dict = {}
        self.stage_label: str | None = None
        self.stray: list[str] | None = None  # capture(): deferred std* writes


_STATE: ContextVar[_State | None] = ContextVar("gmlx_loadlog", default=None)


class _Router:
    """``sys.stdout``/``sys.stderr`` proxy so third-party writes from a
    backgrounded load (HF tokenizer warnings and the like) defer to the join
    instead of smearing the prompt the main thread owns. ``write`` runs in
    the writer's context, so the session ContextVar tells a capture-session
    write (buffer it) from every other write (pass through). Catches
    ``print`` and ``warnings`` output - both look the stream up per call;
    logging handlers that bound the real stream at import time, and C-level
    fd writes, still pass through."""

    def __init__(self, real):
        self._real = real

    def write(self, s):
        state = _STATE.get()
        if state is not None and state.stray is not None:
            state.stray.append(s)
            return len(s)
        return self._real.write(s)

    def flush(self):
        self._real.flush()

    def __getattr__(self, name):
        return getattr(self._real, name)


def _install_router() -> None:
    if not isinstance(sys.stderr, _Router):
        sys.stderr = _Router(sys.stderr)
    if not isinstance(sys.stdout, _Router):
        sys.stdout = _Router(sys.stdout)


def is_verbose() -> bool:
    """True with no active session (direct callers keep their prints) or when
    the session was seeded verbose."""
    state = _STATE.get()
    return True if state is None else state.verbose


def verbose_print(msg: str) -> None:
    """A load diagnostic line: printed verbatim unless a quiet session is active."""
    if is_verbose():
        print(msg)


def warn(msg: str) -> None:
    """Always-visible warning; routed around the spinner when one is animating."""
    state = _STATE.get()
    if state is not None and state.spinner is not None:
        state.spinner.println(msg)
    else:
        print(msg, file=sys.stderr)


def stage(label: str) -> None:
    """Mark a load stage; updates the spinner label when the CLI installed one."""
    state = _STATE.get()
    if state is None:
        return
    state.stage_label = label
    if state.spinner is not None:
        state.spinner.update(f"{state.base}: {label}")


def fact(key: str, value) -> None:
    """Record a summary datum (arch, codec histogram, drafter kind, ...)."""
    state = _STATE.get()
    if state is not None:
        state.facts[key] = value


def facts() -> dict:
    state = _STATE.get()
    return {} if state is None else state.facts


def seeds(fn):
    """Decorate a loader entry point so its ``verbose=`` kwarg seeds a session
    when the CLI has not installed one. Nested loads reuse the outer session.
    Library callers default to quiet; pass ``verbose=True`` for diagnostics."""

    @functools.wraps(fn)
    def wrapper(*args, verbose: bool = False, **kwargs):
        if _STATE.get() is not None:
            return fn(*args, verbose=verbose, **kwargs)
        tok = _STATE.set(_State(verbose=verbose))
        try:
            return fn(*args, verbose=verbose, **kwargs)
        finally:
            _STATE.reset(tok)

    return wrapper


def _summary_line(name: str, elapsed: float, collected: dict) -> str:
    parts = [f"[load] {name}"]
    arch = collected.get("arch")
    if arch:
        parts.append(f"arch {arch}")
    codecs = collected.get("codecs")
    if codecs:
        top = sorted(codecs.items(), key=lambda kv: (-kv[1], kv[0]))
        shown = " ".join(f"{c} x{n}" for c, n in top[:3])
        if len(top) > 3:
            shown += f" +{len(top) - 3} more"
        parts.append(shown)
    else:
        parts.append("no kquant tensors")
    parts.append(f"{elapsed:.1f}s")
    if collected.get("mmproj"):
        parts.append("+mmproj")
    drafter = collected.get("drafter")
    if drafter:
        parts.append(f"drafter {drafter}")
    attn = collected.get("attn")
    if attn:
        parts.append(f"attn {attn}")
    sidecar = collected.get("indexer-sidecar")
    if sidecar:
        parts.append(f"indexer sidecar {sidecar}")
    return " | ".join(parts)


class Capture:
    """Deferred results of a backgrounded load: the ``[load]`` summary line
    on success, the failing stage label on error, and any third-party writes
    the ``_Router`` held back."""

    def __init__(self):
        self.summary: str | None = None
        self.stage: str | None = None
        self.stray: str = ""


@contextmanager
def capture(path: str):
    """``load_ui`` for a load running off the tty (chat's background load):
    quiet session, no spinner, summary/stage handed back on the yielded
    ``Capture`` for the joiner to print once the tty is free. Third-party
    stdout/stderr writes from the load are buffered onto ``Capture.stray``
    (see ``_Router``) for the joiner to replay."""
    name = os.path.basename(path)
    state = _State(verbose=False)
    state.stray = []
    _install_router()
    tok = _STATE.set(state)
    box = Capture()
    t0 = time.perf_counter()
    try:
        yield box
    except Exception:
        box.stage = state.stage_label
        raise
    finally:
        _STATE.reset(tok)
        box.stray = "".join(state.stray)
        box.summary = _summary_line(name, time.perf_counter() - t0, state.facts)


def report_failure(e, stage: str | None) -> None:
    """The one merged load-failure line: stage + reason + next step, flagging
    the exception (``_gmlx_reported``) so the CLI handlers skip their own
    print and the user never sees two stacked error lines for one failure.
    Deliberate refusals (unsupported codec/arch/VLM) pass through silently -
    their messages are written to stand alone; name-matched to keep this
    module free of loader imports. Shared by ``load_ui`` and chat's
    background-load joiner."""
    if type(e).__name__ in ("UnsupportedCodecError", "UnsupportedArchError",
                            "UnsupportedVLMError"):
        return
    where = f" {stage}" if stage else ""
    hint = (" - truncated download or not a GGUF file?"
            if stage == "reading gguf metadata" else "")
    print(
        f"error: load failed{where}: {e}{hint} "
        "(re-run with --verbose for full load diagnostics)",
        file=sys.stderr,
    )
    try:
        e._gmlx_reported = True
    except AttributeError:
        pass                           # exceptions with __slots__: print twice


@contextmanager
def load_ui(verbose: bool, path: str):
    """CLI wrapper for one model load: spinner + summary in quiet mode, a null
    context in verbose mode. Exceptions propagate; quiet mode names the stage
    that failed."""
    if verbose:
        yield
        return
    from .spinner import Spinner

    name = os.path.basename(path)
    base = f"loading {name}"
    spinner = Spinner(base) if sys.stderr.isatty() else None
    state = _State(verbose=False, spinner=spinner, base=base)
    tok = _STATE.set(state)
    t0 = time.perf_counter()
    try:
        if spinner is not None:
            with spinner:
                yield
        else:
            yield
    except Exception as e:
        report_failure(e, state.stage_label)
        raise
    finally:
        _STATE.reset(tok)
    print(_summary_line(name, time.perf_counter() - t0, state.facts))
