"""``gmlx chat`` - interactive chat REPL on a GGUF K-quant model.

Loads the GGUF once (the same loader path as ``gmlx run``) and chats over
a persistent mlx-lm prompt cache, so each turn prefills only the new message.
The basic loop mirrors ``mlx_lm.chat`` (``/exit`` quits, ``/reset`` restarts), and
the terminal it runs in is upgraded:

* with ``prompt_toolkit`` installed (``pip install gmlx[chat]``) the
  prompt gets zsh/fish-grade editing: completion menus that pop **as you
  type** a ``/command``, fish-style ghost suggestions from history (accept
  with the right arrow), a bottom toolbar showing the live sampling settings,
  staged-block count and last reply's tok/s, clean multi-line bracketed
  paste, and Alt-Enter (or Shift-Enter in terminals that send ESC CR for it)
  to insert a newline without submitting. Without it, :mod:`readline` provides line editing and Tab completion
  (the fallback shim);
* up-arrow history is persisted across sessions
  (``$XDG_CACHE_HOME/gmlx/chat_history[.ptk]``); ``--no-history`` keeps
  the session ephemeral (no file read or written; in-session recall still
  works);
* lines starting with ``/`` are commands: ``/history [on|off|clear]`` controls
  persistence at runtime; ``/temp`` / ``/top-p`` / ``/top-k`` / ``/min-p`` /
  ``/max-tokens`` / ``/repetition-penalty`` / ``/presence-penalty`` /
  ``/frequency-penalty`` adjust sampling for subsequent replies (``/sampling``
  shows the current values - all eight are also startup flags, so a model
  card's full sampling recommendation fits on the command line);
* a reasoning model's thinking is stripped of its control markers and streamed
  in the theme's thinking style in a gutter-framed block that closes with a
  ``thought for Xs * N tok``
  payoff (the answer stays normal-weight); ``hide`` collapses it to a live
  spinner that resolves to the same payoff. **Ctrl-O** toggles expand<->collapse
  live during a reply; ``/reasoning [show|hide|raw]`` (startup ``--reasoning``)
  sets the default and reaches ``raw`` (verbatim). The stored turn keeps the raw
  text, so this is display-only;
* ``/load <file>`` prefills the *next* prompt with a text file's contents
  (via the readline startup hook), so it can be edited before Enter sends it;
  Tab completes ``/command`` names, ``/history`` arguments, and file paths
  after ``/load`` and ``/!``;
* ``/! <command>`` runs a shell command and **stages** its output (a fenced
  block with a ``$ command`` header and an ``[exit N in T]`` footer) so it is
  attached to your *next* message - ask a question about the output and both
  land in one turn. Long output is middle-truncated; several ``/!`` blocks
  stack; Enter on an empty prompt sends the staged blocks alone; ``/drop``
  discards them. Ctrl-C interrupts the command, not the session;
* with ``--mmproj`` the session is **multimodal**: ``/image <path>`` and
  ``/audio <path>`` stage media for the next message exactly like ``/!``
  stages text, and **dragging a file from Finder into the terminal works
  directly** - the terminal pastes the path, the REPL recognizes it and
  stages it. Media markers stay on the turn that sent them, so follow-up
  questions reference earlier images correctly. (VLM turns re-prefill the
  conversation each time; the KV-cached fast path is text-only for now.)
* ``/reset`` restarts the conversation; ``/clear`` also wipes the screen;
* sessions **autosave** per turn (schema-v1 JSON under
  ``$XDG_DATA_HOME/gmlx/chats``; ``--no-autosave`` opts out). ``/save``,
  ``/sessions``, ``/load-session <name|N>`` and ``--resume [NAME]`` restore a
  conversation - settings and transcript come back immediately and the history
  prefills with the next message; ``/export [file.md]`` writes a markdown
  transcript with thinking in collapsed ``<details>`` blocks;
* **Esc or Ctrl-C during a reply cancels it** and returns to the prompt (the
  partial reply stays in the cache; ``/reset`` clears it); Ctrl-C at an
  idle prompt or **Ctrl-D** exits.

With ``--speculative`` the session runs MTP speculative decoding (native-head
qwen3.5/3.6, or a ``--draft-gguf`` assistant for gemma4): replies stream through
mlx-vlm's MTP engine over the same persistent cache, for a decode speedup. It's
the text-only path (no ``--mmproj`` / ``--adapter`` / ``--stream-*``) and samples on
temperature/top-p/top-k/min-p only.

Each completed reply prints a one-line ``N tok @ X tok/s`` stat.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time


class _ChatExit(Exception):
    """Deferred-setup failure (raised at the background-load join) carrying
    the exit code the synchronous path would have returned."""

    def __init__(self, code: int):
        super().__init__(code)
        self.code = code


def _build_parser(prog: str = "gmlx chat") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=prog,
        description="Interactive chat REPL on a GGUF K-quant model - multi-turn "
        "over a persistent KV cache, with /commands for sampling, "
        "staging files/shell output, and (with --mmproj) images and "
        "audio. Type '/help' inside the REPL for the command list; "
        "Esc or Ctrl-C cancels a reply.",
    )
    ap.add_argument(
        "gguf", nargs="?", default=None,
        help="Path to the GGUF file (sharded ok), or a config model id. "
        "Optional with --assistant (server default model).",
    )

    # Server-backed assistant mode (no local load; mirrors `gmlx talk`).
    ap.add_argument(
        "--assistant",
        action="store_true",
        help="Chat through the built-in tool-loop assistant on a running "
        "(auto-started) server: MCP tools + long-term memory from the "
        "config's assistant: block. The positional is a served model id.",
    )
    ap.add_argument(
        "--base-url",
        default=None,
        help="Server OpenAI base URL (default http://HOST:PORT/v1).",
    )
    ap.add_argument("--host", default=None, help="Server host.")
    ap.add_argument("--port", type=int, default=None, help="Server port.")
    ap.add_argument("--api-key", default=None, help="Server API key.")
    ap.add_argument(
        "--no-start", action="store_true", help="Never auto-start a server."
    )
    ap.add_argument(
        "--start-timeout",
        type=float,
        default=180.0,
        metavar="S",
        help="Auto-start wait (default 180; 0 = wait forever).",
    )

    # Load options (the `run` subset that applies to an interactive session;
    # shared builders keep the two surfaces in sync).
    from .cli import (
        add_config_profile_args,
        add_kv_cache_args,
        add_load_args,
        add_placement_args,
        add_sampling_args,
        add_speculative_args,
        add_verbosity_arg,
        add_vlm_shared_args,
    )

    ap.add_argument(
        "--hf-source",
        default=None,
        help="Load config from this HF id / local dir instead of "
        "synthesizing it from GGUF metadata.",
    )
    add_config_profile_args(ap)
    add_load_args(ap)
    ap.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Send each turn verbatim, applying no chat template (for base / "
        "non-instruct GGUFs that carry no template). Plain-text models only.",
    )
    ap.add_argument(
        "--adapter",
        default=None,
        metavar="PATH",
        help="GGUF LoRA adapter applied live over the base at "
        "load - base stays K-quant, no merge (text path "
        "only, not --mmproj).",
    )
    ap.add_argument(
        "--mmproj",
        default=None,
        metavar="PATH",
        help="Vision/audio projector GGUF (general.architecture="
        "clip). Enables multimodal chat: /image and /audio "
        "stage media, dragging a file into the terminal "
        "stages it too.",
    )
    # MTP speculative decoding - the same group `gmlx run` exposes (shared builder,
    # so the two surfaces stay in sync and a config's `speculative: true` applies here).
    add_speculative_args(ap)
    add_verbosity_arg(ap)

    # Sampling (all runtime-adjustable in-chat via the matching /command).
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        metavar="N",
        help="Per-reply decode-token cap; default 0 = generate until the "
        "model stops. Pass N to cap (adjustable in-chat via /max-tokens; "
        "0 removes the cap).",
    )
    add_sampling_args(ap)
    ap.add_argument(
        "--stop",
        action="append",
        default=None,
        metavar="STR",
        help="Stop sequence - the reply ends (trimmed) when it appears. Repeatable.",
    )

    # KV cache + prefill memory knobs.
    add_kv_cache_args(ap)

    # VLM-only (--mmproj).
    add_vlm_shared_args(ap)
    ap.add_argument(
        "--reasoning",
        choices=("show", "hide", "raw"),
        default="show",
        help="How to display a reasoning model's thinking: 'show' styles it under "
        "a label and strips the control markers (default), 'hide' drops it and "
        "prints only the answer, 'raw' passes everything through verbatim. "
        "Toggle live with /reasoning.",
    )
    ap.add_argument(
        "--render",
        choices=("auto", "plain", "lite", "rich"),
        default="auto",
        help="Reply rendering: 'rich' full markdown, 'lite' ANSI markdown, "
        "'plain' raw text. Default auto: rich when installed on a color TTY. "
        "Switch live with /render.",
    )
    ap.add_argument(
        "--theme",
        default="dark",
        help="Color theme (see /theme for the list). Default dark.",
    )
    ap.add_argument(
        "--colorblind",
        action="store_true",
        help="Colorblind-friendly accents (Okabe-Ito) for any theme.",
    )

    # Chat-template controls.
    ap.add_argument(
        "--system-prompt",
        default=None,
        help="System message, sent on the first turn (and after each reset).",
    )

    ap.add_argument(
        "--no-autosave",
        action="store_true",
        help="Don't autosave the session after each turn.",
    )
    ap.add_argument(
        "--resume",
        nargs="?",
        const="",
        default=None,
        metavar="NAME",
        help="Resume a saved session (default: this model's latest).",
    )
    ap.add_argument(
        "--no-history",
        action="store_true",
        help="Don't read or write the prompt-history file "
        "(in-session recall still works).",
    )

    # Execution-placement knobs (same surface as `gmlx run`; text-only).
    add_placement_args(ap)
    return ap


def parse_template_config(raw: str | None) -> dict:
    """Parse ``--chat-template-config`` early so a JSON typo fails fast."""
    import json

    if not raw:
        return {}
    try:
        out = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"--chat-template-config is not valid JSON: {e}") from e
    if not isinstance(out, dict):
        raise ValueError("--chat-template-config must be a JSON object")
    return out


def parse_logit_bias(raw: str | None) -> dict | None:
    """Parse ``--logit-bias`` ('{"token_id": bias}') into ``{int: float}``."""
    import json

    if not raw:
        return None
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"--logit-bias is not valid JSON: {e}") from e
    if not isinstance(doc, dict):
        raise ValueError("--logit-bias must be a JSON object of token-id: bias")
    try:
        return {int(k): float(v) for k, v in doc.items()}
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"--logit-bias keys must be token ids, values numbers: {e}"
        ) from e


def parse_resize_shape(raw: str | None) -> list | None:
    """Parse ``--resize-shape`` ('448' or '672x448') into the 1- or 2-int
    list mlx-vlm's ``normalize_resize_shape`` accepts."""
    if not raw:
        return None
    try:
        parts = [int(p) for p in raw.lower().replace("x", " ").split()]
    except ValueError as e:
        raise ValueError(f"--resize-shape must be N or WxH: {e}") from e
    if len(parts) not in (1, 2):
        raise ValueError("--resize-shape must be N or WxH")
    return parts


# Terminal shim: history, /commands, completion


@dataclasses.dataclass
class ChatState:
    """Mutable REPL session state, shared by the input wiring, slash
    commands, turn plumbing, and session save/restore. Field defaults are
    the values reads assume before the field's first assignment."""

    # line editor wiring (_wire_input / _wire_history / _wire_ptk)
    readline: object = None
    history_enabled: bool = False   # /history on|off (persistence at exit)
    history_loaded: bool = False    # history file read once per session
    ptk_session: object = None
    input_fn: object = None      # scripted-input seam for the e2e loop tests

    # staged input for the next turn
    staged: list = dataclasses.field(default_factory=list)         # /! blocks
    staged_images: list = dataclasses.field(default_factory=list)
    staged_audio: list = dataclasses.field(default_factory=list)
    pending_send: str | None = None      # /retry auto-resend
    pending_insert: str | None = None    # /load prompt prefill

    # presentation
    theme: object = None
    colorblind: bool = False
    render: str = "plain"
    reasoning: str = "show"

    # conversation + per-turn plumbing
    vlm: bool = False
    system_prompt: str | None = None
    thinking_budget: int | None = None
    sampling: dict = dataclasses.field(default_factory=dict)
    transcript: list = dataclasses.field(default_factory=list)
    replay_messages: list | None = None  # deferred history prefill (--resume)
    turn_checkpoint: dict | None = None
    ctx_used: int | None = None
    ctx_max: int | None = None
    last_stats: dict | None = None
    last_tps: float | None = None
    last_think_open: bool = False

    # session bookkeeping
    session_stats: dict | None = None
    session_meta: dict | None = None
    session_name: str | None = None
    session_created: str | None = None
    session_list: list | None = None
    model_name: str | None = None
    model_info: dict | None = None
    autosave: object = None
    clipboard_runner: object = None

    # --assistant mode
    assistant_brain: object = None
    assistant_extra: dict | None = None
    assistant_baseline: dict | None = None
    assistant_touched: set | None = None

    def take(self, name: str):
        """Read an optional field and clear it to None - a single-consumer
        handoff (staged sends, deferred replays)."""
        v = getattr(self, name)
        setattr(self, name, None)
        return v

    def take_list(self, name: str) -> list:
        """Read a list field and reset it to a fresh empty list."""
        v = getattr(self, name)
        setattr(self, name, [])
        return v


def _history_path():
    from pathlib import Path

    cache = Path(os.environ.get("XDG_CACHE_HOME") or "~/.cache").expanduser()
    return cache / "gmlx" / "chat_history"


def _wire_history(state: ChatState, readline) -> None:
    """Set up prompt-history persistence on ``state``.

    ``state.history_enabled`` is consulted at exit, so ``/history on|off`` can flip
    persistence mid-session. The file is only read once (``history_loaded``) -
    on startup when enabled, or lazily by ``/history on`` - so a later save
    merges prior history instead of overwriting it with just this session.
    """
    state.readline = readline
    if readline is None:
        return

    import atexit

    hist = _history_path()
    if state.history_enabled:
        try:
            if hist.exists():
                readline.read_history_file(hist)
            state.history_loaded = True
        except OSError:
            pass

    def _save() -> None:
        if not state.history_enabled:
            return
        try:
            hist.parent.mkdir(parents=True, exist_ok=True)
            readline.set_history_length(1000)
            readline.write_history_file(hist)
        except OSError:
            pass

    atexit.register(_save)


# /! - run a shell command, stage its output for the next turn

# Chars of command output kept per /! block (head + tail around a marker), so
# a stray `cat bigfile` can't blow the context.
_SHELL_OUTPUT_LIMIT = 16_000


def _truncate_middle(text: str, limit: int = _SHELL_OUTPUT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return (
        text[:half].rstrip("\n")
        + f"\n[... {len(text) - 2 * half} chars truncated ...]\n"
        + text[-half:].lstrip("\n")
    )


def _run_shell(command: str) -> tuple[str, int, float]:
    """Run ``command`` in a shell; return (merged stdout+stderr, exit code,
    seconds). stdin is /dev/null so interactive commands can't wedge the REPL;
    Ctrl-C interrupts the child's process group and re-raises."""
    import signal
    import subprocess
    import time

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        command,
        shell=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate()
    except KeyboardInterrupt:
        try:
            os.killpg(proc.pid, signal.SIGINT)
            proc.wait(timeout=5)
        except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
            proc.kill()
            proc.wait()
        raise
    return out, proc.returncode, time.perf_counter() - t0


def _shell_block(command: str, out: str, code: int, secs: float) -> str:
    """Format a /! result the way it lands in the chat turn."""
    body = _truncate_middle(out.rstrip("\n"))
    lines = [f"$ {command}"]
    if body:
        lines.append(body)
    lines.append(f"[exit {code} in {secs:.2f}s]")
    return "\n".join(lines)


def _handle_shell(command: str, state: ChatState) -> None:
    if not command:
        print("[chat] usage: /! <command> (stages output for your next message)")
        return
    try:
        out, code, secs = _run_shell(command)
    except KeyboardInterrupt:
        print("\n[chat] command canceled (nothing staged)")
        return
    block = _shell_block(command, out, code, secs)
    print(block)
    state.staged.append(block)
    n = len(state.staged)
    print(
        f"[chat] staged ({n} block{'s' if n > 1 else ''}) - attached to your "
        f"next message; Enter alone sends as-is, '/drop' discards"
    )


def _compose_user_content(state: ChatState, line: str) -> str:
    """Prepend any staged /! blocks to the typed message (or send them alone
    on an empty line). Also records the composed text in the current turn
    checkpoint (the transcript's user message)."""
    staged = state.take_list("staged")
    if not staged:
        content = line
    else:
        blocks = "\n\n".join(f"```\n{b}\n```" for b in staged)
        content = f"{blocks}\n\n{line}" if line.strip() else blocks
    cp = state.turn_checkpoint
    if cp is not None:
        cp["user_content"] = content
    return content


# Media staging (--mmproj): /image, /audio, and dragged-in file paths

_IMAGE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
}
_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aiff", ".aif"}


def _detect_dropped_media(line: str) -> tuple[list[str], list[str]] | None:
    """Recognize a line that is entirely media-file paths - what a terminal
    pastes when files are dragged in (shell-escaped, possibly several).
    Returns ``(images, audios)``, or None if any token is not an existing
    image/audio file."""
    import shlex

    try:
        tokens = shlex.split(line)
    except ValueError:
        return None
    if not tokens:
        return None
    images, audios = [], []
    for t in tokens:
        p = os.path.expanduser(t)
        if not os.path.isfile(p):
            return None
        ext = os.path.splitext(p)[1].lower()
        if ext in _IMAGE_EXTS:
            images.append(p)
        elif ext in _AUDIO_EXTS:
            audios.append(p)
        else:
            return None
    return images, audios


def _stage_media(state: ChatState, images: list[str], audios: list[str]) -> None:
    if not state.vlm:
        print(
            "[chat] this looks like media, but the session is text-only - "
            "restart with --mmproj <projector.gguf> to chat about "
            "images/audio"
        )
        return
    if images:
        state.staged_images.extend(images)
    if audios:
        state.staged_audio.extend(audios)
    n_i = len(state.staged_images)
    n_a = len(state.staged_audio)
    parts = [
        p
        for p, n in (
            (f"{n_i} image{'s' if n_i != 1 else ''}", n_i),
            (f"{n_a} audio", n_a),
        )
        if n
    ]
    print(
        f"[chat] staged {' + '.join(parts)} - attached to your next message "
        f"('/drop' discards)"
    )


def _handle_media_command(cmd: str, arg: str, state: ChatState) -> None:
    """``/image <path...>`` / ``/audio <path...>`` - stage media files."""
    import shlex

    kind = cmd[1:]
    if not arg:
        print(f"[chat] usage: /{kind} <file> (stages it for your next message)")
        return
    try:
        tokens = shlex.split(arg)
    except ValueError as e:
        print(f"[chat] /{kind}: {e}")
        return
    paths = []
    for t in tokens:
        p = os.path.expanduser(t)
        if not os.path.isfile(p):
            print(f"[chat] /{kind}: no such file: {t}")
            return
        paths.append(p)
    _stage_media(
        state, paths if kind == "image" else [], paths if kind == "audio" else []
    )


# Runtime-adjustable sampling knobs: /command -> (state key, parser).
_SAMPLING_COMMANDS = {
    "/temp": ("temp", float),
    "/top-p": ("top_p", float),
    "/top-k": ("top_k", int),
    "/min-p": ("min_p", float),
    "/max-tokens": ("max_tokens", int),
    "/xtc-probability": ("xtc_probability", float),
    "/xtc-threshold": ("xtc_threshold", float),
    "/repetition-penalty": ("repetition_penalty", float),
    "/repetition-context-size": ("repetition_context_size", int),
    "/presence-penalty": ("presence_penalty", float),
    "/frequency-penalty": ("frequency_penalty", float),
}


def _print_shim_help(state: ChatState) -> None:
    status = "on" if state.history_enabled else "off"
    print("[chat] commands:")
    print(f"- '/history [on|off|clear]' control prompt-history saving ({status})")
    print("- '/temp /top-p /top-k /min-p /max-tokens <value>' adjust sampling "
          "(max-tokens 0 = no cap)")
    print("- '/xtc-probability /xtc-threshold <value>' adjust XTC sampling")
    print(
        "- '/repetition-penalty /presence-penalty /frequency-penalty "
        "/repetition-context-size <value>' adjust penalties"
    )
    print("- '/sampling' to show the current sampling settings")
    print(
        "- '/system [text|off]' show or set the system prompt "
        "(setting restarts the chat)"
    )
    print("- '/thinking-budget [N|off]' cap thinking tokens per reply")
    print(
        "- '/reasoning [show|hide|raw]' control how thinking is displayed "
        "(Ctrl-O collapses/expands it live during a reply)"
    )
    print("- '/render [plain|lite|rich]' set reply markdown rendering")
    print("- '/theme [NAME] [cb]' set the color theme ('cb' = colorblind accents)")
    print("- '/model' show the loaded model card - '/stats' session totals")
    print(
        "- '/retry' regenerate the last reply - '/undo' remove the last "
        "exchange entirely"
    )
    print("- '/copy' copy the last answer (thinking stripped) to the clipboard")
    print(
        "- '/save [name]' / '/sessions' / '/load-session <name|N>' save, list, "
        "and restore sessions (autosaved per turn unless --no-autosave)"
    )
    print("- '/export [file.md]' write the conversation as markdown")
    print("- '/load <file>' prefill the next prompt from a text file (edit, Enter)")
    print(
        "- '/! <command>' run a shell command, stage its output for the next "
        "message ('/drop' discards)"
    )
    if state.vlm:
        print(
            "- '/image <file>' / '/audio <file>' stage media for the next "
            "message (or just drag a file in)"
        )
    if state.assistant_brain is not None:
        print(
            "- '/memory [forget ID|clear]' list or edit the assistant's "
            "long-term memory"
        )
    print("- '/reset' restart the conversation - '/clear' also wipes the screen")
    print("- '/exit' or '/quit' (or Ctrl-D) leave the chat")
    if state.ptk_session is not None:
        print(
            "- Alt-Enter inserts a newline (Shift-Enter too, if your terminal "
            "sends ESC CR for it); Tab completes /commands and paths"
        )
    else:
        print("- '/help' to display these commands; Tab completes /commands and paths")


def _print_sampling(state: ChatState) -> None:
    s = state.sampling
    knobs = "  ".join(f"{k.replace('_', '-')}={v}" for k, v in sorted(s.items()))
    print(f"[chat] sampling: {knobs}")


def _cache_tokens(cache) -> int:
    """Tokens held in a KV cache (max per-layer offset); 0 for None/empty."""
    if not cache:
        return 0
    return max((int(getattr(c, "offset", 0)) for c in cache), default=0)


def _fmt_k(n) -> str:
    n = int(n)
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def _can_trim(cache) -> bool:
    """True when every layer cache supports trim() (a rotating cache that has
    wrapped its window does not)."""
    if not cache:
        return False
    return all(bool(getattr(c, "is_trimmable", lambda: False)()) for c in cache)


def _trim_to(cache, checkpoint: int, lm=None) -> bool:
    """Rewind a KV cache to ``checkpoint`` tokens; clears the mrope bookkeeping
    on ``lm`` (mirrors ``_prefill_into_cache``) so the next turn re-derives it."""
    n = _cache_tokens(cache) - int(checkpoint)
    if n <= 0:
        return True
    if not _can_trim(cache):
        return False
    for c in cache:
        c.trim(n)
    if lm is not None:
        for attr in ("_position_ids", "_rope_deltas"):
            if hasattr(lm, attr):
                setattr(lm, attr, None)
    return True


def _begin_turn(state: ChatState, *, cache, first_turn: bool, vlm_lens=(0, 0, 0)) -> None:
    """Checkpoint the pre-turn session shape (for the transcript, /retry, /undo).
    ``_compose_user_content`` fills in the composed user text; staged media are
    captured here, before the generation paths pop them."""
    state.turn_checkpoint = {
        "cache_before": _cache_tokens(cache),
        "first_turn_before": first_turn,
        "vlm_lens": tuple(vlm_lens),
        "images": list(state.staged_images),
        "audios": list(state.staged_audio),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "user_content": "",
    }


def _fmt_stat_line(stats: dict, ctx_used, ctx_max) -> str:
    parts = []
    if stats.get("prompt_tokens"):
        p = f"prompt {_fmt_k(stats['prompt_tokens'])} tok"
        if stats.get("prompt_tps"):
            p += f" @ {stats['prompt_tps']:.0f} tok/s"
        parts.append(p)
    if stats.get("gen_tokens"):
        parts.append(
            f"gen {_fmt_k(stats['gen_tokens'])} tok @ {stats.get('gen_tps', 0.0):.1f} tok/s"
        )
    if stats.get("rounds"):
        parts.append(
            f"accept {stats.get('accept_rate', 0.0) * 100:.0f}% · "
            f"{stats.get('mean_accept_len', 0.0):.1f}/round"
        )
    if ctx_used:
        c = f"ctx {_fmt_k(ctx_used)}"
        if ctx_max:
            c += f"/{_fmt_k(ctx_max)}"
        parts.append(c)
    return "[chat] " + " · ".join(parts)


def _end_turn(state: ChatState, reply: str, canceled: bool, cache=None) -> None:
    """Record the finished turn in the transcript, fold session totals, and
    print the stat line."""
    cp = state.take("turn_checkpoint")
    if cp is None:
        return
    stats = state.last_stats or {}
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    entry = {
        "user": {
            "role": "user",
            "content": cp.get("user_content", ""),
            "ts": cp.get("ts", now),
            "images": cp.get("images", []),
            "audios": cp.get("audios", []),
        },
        "assistant": {
            "role": "assistant",
            "content": reply or "",
            "ts": now,
            "canceled": bool(canceled),
            "think_open": state.last_think_open,
            "stats": stats,
        },
        "cache_before": cp.get("cache_before", 0),
        "first_turn_before": cp.get("first_turn_before", True),
        "vlm_lens": cp.get("vlm_lens", (0, 0, 0)),
    }
    if "brain_before" in cp:      # assistant turns rewind by brain checkpoint
        entry["brain_before"] = cp["brain_before"]
    state.transcript.append(entry)
    ss = state.session_stats
    if ss is not None:
        ss["turns"] += 1
        ss["prompt_tok"] += stats.get("prompt_tokens", 0)
        ss["gen_tok"] += stats.get("gen_tokens", 0)
        if stats.get("gen_tps"):
            ss["gen_time"] += stats.get("gen_tokens", 0) / stats["gen_tps"]
        ss["accepted"] += stats.get("accepted", 0)
        ss["drafted"] += stats.get("drafted", 0)
        ss["rounds"] += stats.get("rounds", 0)
    used = _cache_tokens(cache)
    if not used:
        used = stats.get("prompt_tokens", 0) + stats.get("gen_tokens", 0)
    state.ctx_used = used or None
    if not canceled and stats.get("gen_tokens"):
        line = _fmt_stat_line(stats, state.ctx_used, state.ctx_max)
        theme = state.theme
        print(theme.paint("stat", line) if theme is not None else line)
    hook = state.autosave
    if hook is not None:
        hook()


def _print_session_stats(state: ChatState) -> None:
    ss = state.session_stats
    if not ss or not ss["turns"]:
        print("[chat] no completed turns yet")
        return
    parts = [
        f"{ss['turns']} turn{'s' if ss['turns'] != 1 else ''}",
        f"prompt {_fmt_k(ss['prompt_tok'])} tok",
        f"gen {_fmt_k(ss['gen_tok'])} tok",
    ]
    if ss["gen_time"] > 0:
        parts.append(f"{ss['gen_tok'] / ss['gen_time']:.1f} tok/s avg")
    if ss["drafted"]:
        parts.append(f"accept {ss['accepted'] / ss['drafted'] * 100:.0f}%")
    mins, secs = divmod(int(time.monotonic() - ss["t0"]), 60)
    parts.append(f"{mins}m{secs:02d}s" if mins else f"{secs}s")
    print("[chat] session: " + " · ".join(parts))


def _model_type(config) -> str:
    """``model_type`` from a config that is a dict (text path) or an object
    (mlx-vlm ModelConfig)."""
    if isinstance(config, dict):
        return config.get("model_type", "")
    return getattr(config, "model_type", "")


def _build_model_info(args, config, drafter, vlm_mtp: bool) -> dict:
    """Model card, built once post-load (a header-only GGUF re-read)."""
    info: dict = {"path": os.path.abspath(args.gguf)}
    try:
        from .preflight import preflight

        pf = preflight(args.gguf, arch=args.arch)
        info.update(
            arch=pf.arch,
            n_tensors=pf.n_tensors,
            n_params=pf.n_params,
            codecs=dict(
                sorted(pf.codec_histogram.items(), key=lambda kv: -kv[1])
            ),
            size_bytes=sum(os.stat(s).st_size for s in pf.shards),
            n_shards=len(pf.shards),
        )
    except Exception:
        pass
    info["model_type"] = _model_type(config)
    if args.mmproj:
        info["mmproj"] = os.path.abspath(args.mmproj)
        try:
            info["size_bytes"] = info.get("size_bytes", 0) + os.stat(args.mmproj).st_size
        except OSError:
            pass
    if getattr(args, "adapter", None):
        info["adapter"] = args.adapter
    if drafter is not None:
        kind = "assistant" if args.draft_gguf else "native-head"
        block = getattr(getattr(drafter, "config", None), "block_size", None)
        info["drafter"] = f"{kind} MTP" + (f" (block {block})" if block else "")
        if vlm_mtp:
            info["drafter"] += ", text-only turns"
    return info


def _print_model_info(state: ChatState) -> None:
    info = state.model_info
    if not info:
        print("[chat] model info unavailable")
        return
    name = state.model_name or os.path.basename(info.get("path", ""))
    print(f"[chat] model: {name}")
    shards = f" ({info['n_shards']} shards)" if info.get("n_shards", 1) > 1 else ""
    if info.get("size_bytes"):
        print(f"  path     {info['path']}{shards} · {info['size_bytes'] / 1e9:.1f} GB")
    else:
        print(f"  path     {info['path']}{shards}")
    arch_bits = [info.get("arch", "?")]
    if info.get("model_type") and info.get("model_type") != info.get("arch"):
        arch_bits.append(f"-> {info['model_type']}")
    if info.get("n_params"):
        arch_bits.append(f"· {info['n_params'] / 1e9:.1f}B params")
    if info.get("n_tensors"):
        arch_bits.append(f"· {info['n_tensors']} tensors")
    print(f"  arch     {' '.join(arch_bits)}")
    if info.get("codecs"):
        top = list(info["codecs"].items())
        shown = "  ".join(f"{c} x{n}" for c, n in top[:6])
        if len(top) > 6:
            shown += f"  +{len(top) - 6} more"
        print(f"  codecs   {shown}")
    if state.ctx_max:
        print(f"  context  {_fmt_k(state.ctx_max)} max")
    if info.get("drafter"):
        print(f"  drafter  {info['drafter']}")
    if info.get("mmproj"):
        print(f"  mmproj   {info['mmproj']}")
    if info.get("adapter"):
        print(f"  adapter  {info['adapter']}")


def _strip_thinking(text: str, start_in_thinking: bool = False) -> str:
    """The answer portion of a raw reply (reasoning spans + markers removed)."""
    from .sessions import split_thinking

    return split_thinking(text, start_in_thinking)[1]


def _session_doc(state: ChatState) -> dict:
    """Schema-v1 session document from the live REPL state."""
    msgs = []
    for e in state.transcript:
        msgs.append(dict(e["user"]))
        msgs.append(dict(e["assistant"]))
    doc = {
        "model": dict(state.session_meta or {}),
        "system_prompt": state.system_prompt,
        "sampling": dict(state.sampling),
        "reasoning": state.reasoning,
        "render": state.render,
        "theme": getattr(state.theme, "name", None),
        "colorblind": bool(state.colorblind),
        "thinking_budget": state.thinking_budget,
        "messages": msgs,
    }
    if state.session_created:
        doc["created"] = state.session_created
    return doc


def _copy_to_clipboard(text: str, runner=None) -> str | None:
    """Copy ``text``; returns the mechanism used, or None when nothing worked.
    Tries pbcopy / xclip / wl-copy, then the OSC 52 terminal escape."""
    import shutil
    import subprocess

    def _run(argv, data):
        subprocess.run(argv, input=data.encode(), check=True)

    # An injected runner (tests) also decides availability.
    for argv in (["pbcopy"], ["xclip", "-selection", "clipboard"], ["wl-copy"]):
        if runner is not None or shutil.which(argv[0]):
            try:
                (runner or _run)(argv, text)
                return argv[0]
            except Exception:
                continue
    if sys.stderr.isatty():
        import base64

        payload = base64.b64encode(text.encode()).decode()
        sys.stderr.write(f"\x1b]52;c;{payload}\x07")
        sys.stderr.flush()
        return "osc52"
    return None


def _copy_last_answer(state: ChatState) -> None:
    transcript = state.transcript
    if not transcript:
        print("[chat] nothing to copy yet")
        return
    a = transcript[-1]["assistant"]
    text = _strip_thinking(a.get("content", ""), a.get("think_open", False))
    if not text:
        print("[chat] the last reply has no answer text to copy")
        return
    how = _copy_to_clipboard(text, runner=state.clipboard_runner)
    if how is None:
        print("[chat] no clipboard mechanism available")
    else:
        print(f"[chat] copied {len(text)} chars ({how})")


def _slash_drop(cmd, arg, state):
    n = (
        len(state.take_list("staged"))
        + len(state.take_list("staged_images"))
        + len(state.take_list("staged_audio"))
    )
    print(f"[chat] dropped {n} staged item{'s' if n != 1 else ''}")
    return None


def _slash_media(cmd, arg, state):
    if state.assistant_brain is not None:
        print(f"[chat] {cmd[1:]} attachments are not available with "
              "--assistant")
        return None
    _handle_media_command(cmd, arg, state)
    return None


def _knob_display(key, value):
    """Knob value for display; max-tokens 0 means "no cap"."""
    if key == "max_tokens" and not value:
        return "0 (no cap - replies run until the model stops)"
    return value


def _slash_sampling_knob(cmd, arg, state):
    key, cast = _SAMPLING_COMMANDS[cmd]
    if not arg:
        print(f"[chat] {cmd[1:]} = {_knob_display(key, state.sampling[key])}")
        return None
    try:
        state.sampling[key] = cast(arg)
    except ValueError:
        print(f"[chat] {cmd} needs a {cast.__name__}, got {arg!r}")
        return None
    if (state.assistant_brain is not None
            and key in _ASSISTANT_SAMPLING):
        # An explicit /command rides along even when it lands back on
        # the session baseline; otherwise the knob silently drops out
        # of the forwarded payload and the server default applies.
        state.assistant_touched.add(key)
    print(f"[chat] {cmd[1:]} = {_knob_display(key, state.sampling[key])} "
          f"(next reply)")
    return None


def _slash_sampling(cmd, arg, state):
    _print_sampling(state)
    return None


def _slash_reasoning(cmd, arg, state):
    if not arg:
        print(f"[chat] reasoning = {state.reasoning}")
        return None
    if arg not in ("show", "hide", "raw"):
        print("[chat] /reasoning needs one of: show, hide, raw")
        return None
    state.reasoning = arg
    print(f"[chat] reasoning = {arg} (next reply)")
    return None


def _slash_rewind(cmd, arg, state):
    return cmd[1:]  # the REPL loop owns the cache/history rewind


def _slash_exit(cmd, arg, state):
    return "exit"


def _slash_reset(cmd, arg, state):
    print("[chat] conversation reset")
    return "reset"


def _slash_system(cmd, arg, state):
    if not arg:
        cur = state.system_prompt
        print(f"[chat] system prompt: {cur}" if cur
              else "[chat] no system prompt set")
        return None
    state.system_prompt = None if arg == "off" else arg
    what = "cleared" if arg == "off" else "set"
    print(f"[chat] system prompt {what} (conversation reset)")
    return "reset"


def _slash_memory(cmd, arg, state):
    _chat_memory_cmd(arg, state)
    return None


def _slash_thinking_budget(cmd, arg, state):
    if state.assistant_brain is not None:
        print("[chat] the server owns thinking budgets - not available "
              "with --assistant")
        return None
    if arg and arg != "off":
        try:
            state.thinking_budget = int(arg)
        except ValueError:
            print(f"[chat] /thinking-budget needs an int or 'off', got {arg!r}")
            return None
        print(f"[chat] thinking-budget = {arg} (next reply)")
        return None
    if arg == "off":
        state.thinking_budget = None
        print("[chat] thinking-budget = off (next reply)")
        return None
    tb = state.thinking_budget
    print(f"[chat] thinking-budget = {tb if tb is not None else 'off'}")
    return None


def _slash_copy(cmd, arg, state):
    _copy_last_answer(state)
    return None


def _slash_save(cmd, arg, state):
    from . import sessions

    if not state.transcript:
        print("[chat] nothing to save yet")
        return None
    name = arg or state.session_name or sessions.default_session_name(
        (state.session_meta or {}).get("path", "chat")
    )
    try:
        path = sessions.save_session(_session_doc(state), name)
    except OSError as e:
        print(f"[chat] /save: {e}")
        return None
    state.session_name = name
    print(f"[chat] saved {path}")
    return None


def _slash_export(cmd, arg, state):
    from . import sessions

    if not state.transcript:
        print("[chat] nothing to export yet")
        return None
    name = state.session_name or sessions.default_session_name(
        (state.session_meta or {}).get("path", "chat")
    )
    try:
        path = sessions.export_markdown(_session_doc(state), arg or f"{name}.md")
    except OSError as e:
        print(f"[chat] /export: {e}")
        return None
    print(f"[chat] exported {path}")
    return None


def _slash_sessions(cmd, arg, state):
    from . import sessions

    rows = sessions.list_sessions()
    if not rows:
        print(f"[chat] no saved sessions in {sessions.sessions_dir()}")
        return None
    state.session_list = [r["name"] for r in rows]
    for i, r in enumerate(rows, 1):
        n = r["turns"]
        print(
            f"  {i:2d}. {r['name']}  ({n} turn{'s' if n != 1 else ''}, "
            f"{r['model']}, {r['updated']})"
        )
    print("[chat] '/load-session <name|number>' restores one")
    return None


def _slash_load_session(cmd, arg, state):
    if not arg:
        print("[chat] usage: /load-session <name|number> ('/sessions' lists)")
        return None
    ref = arg
    if arg.isdigit():
        from . import sessions

        names = state.session_list
        if names is None:
            names = [r["name"] for r in sessions.list_sessions()]
        if not (1 <= int(arg) <= len(names)):
            print(f"[chat] no session #{arg} ('/sessions' lists)")
            return None
        ref = names[int(arg) - 1]
    return ("load-session", ref)


def _slash_model(cmd, arg, state):
    _print_model_info(state)
    return None


def _slash_stats(cmd, arg, state):
    _print_session_stats(state)
    return None


def _slash_render(cmd, arg, state):
    from .render import rich_available

    if not arg:
        print(f"[chat] render = {state.render}")
        return None
    if arg not in ("plain", "lite", "rich"):
        print("[chat] /render needs one of: plain, lite, rich")
        return None
    if arg == "rich" and not rich_available():
        print("[chat] rich not installed (pip install 'gmlx[chat]')")
        return None
    state.render = arg
    print(f"[chat] render = {arg} (next reply)")
    return None


def _slash_theme(cmd, arg, state):
    from .reasoning import want_color
    from .theme import list_themes, resolve_theme

    if not arg:
        cur = state.theme
        cb = " +colorblind" if state.colorblind else ""
        print(
            f"[chat] theme = {cur.name if cur else 'dark'}{cb} - "
            f"options: {', '.join(list_themes())} ('/theme NAME [cb]')"
        )
        return None
    parts = arg.split()
    cb = len(parts) > 1 and parts[1] in ("cb", "colorblind")
    try:
        state.theme = resolve_theme(parts[0], colorblind=cb, color=want_color())
    except ValueError as e:
        print(f"[chat] {e}")
        return None
    state.colorblind = cb
    print(f"[chat] theme = {parts[0]}{' +colorblind' if cb else ''} (next reply)")
    return None


def _slash_clear(cmd, arg, state):
    print("\033[2J\033[H", end="")
    print("[chat] conversation reset")
    return "reset"


def _slash_load(cmd, arg, state):
    from pathlib import Path

    if not arg:
        print("[chat] usage: /load <text file> (prefills the next prompt)")
        return None
    try:
        text = Path(arg).expanduser().read_text()
    except (OSError, UnicodeDecodeError) as e:
        print(f"[chat] /load: {e}")
        return None
    state.pending_insert = text.rstrip("\n")
    print(f"[chat] loaded {arg} ({len(text)} chars) - edit, then Enter")
    return None


def _slash_history(cmd, arg, state):
    readline = state.readline
    ptk = state.ptk_session is not None
    if not ptk and readline is None:
        print("[chat] history unavailable (no readline on this build)")
        return None
    if arg == "off":
        state.history_enabled = False
        print("[chat] history off (this session will not be saved)")
    elif arg == "on":
        if ptk:
            _ptk_history_reload(state.ptk_session.history)
        elif not state.history_loaded:
            try:
                if _history_path().exists():
                    readline.read_history_file(_history_path())
                state.history_loaded = True
            except OSError:
                pass
        state.history_enabled = True
        print("[chat] history on")
    elif arg == "clear":
        if ptk:
            state.ptk_session.history.get_strings().clear()
        else:
            readline.clear_history()
        for p in (_history_path(), _history_path().with_suffix(".ptk")):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        print("[chat] history cleared")
    else:
        status = "on" if state.history_enabled else "off"
        path = _history_path().with_suffix(".ptk") if ptk else _history_path()
        print(f"[chat] history is {status} ({path})")
    return None


_SLASH_HANDLERS = {
    **{c: _slash_sampling_knob for c in _SAMPLING_COMMANDS},
    "/drop": _slash_drop,
    "/image": _slash_media,
    "/audio": _slash_media,
    "/sampling": _slash_sampling,
    "/reasoning": _slash_reasoning,
    "/retry": _slash_rewind,
    "/undo": _slash_rewind,
    "/exit": _slash_exit,
    "/quit": _slash_exit,
    "/reset": _slash_reset,
    "/system": _slash_system,
    "/memory": _slash_memory,
    "/thinking-budget": _slash_thinking_budget,
    "/copy": _slash_copy,
    "/save": _slash_save,
    "/export": _slash_export,
    "/sessions": _slash_sessions,
    "/load-session": _slash_load_session,
    "/model": _slash_model,
    "/stats": _slash_stats,
    "/render": _slash_render,
    "/theme": _slash_theme,
    "/clear": _slash_clear,
    "/load": _slash_load,
    "/history": _slash_history,
}


def _handle_slash(line: str, state: ChatState) -> str | tuple | None:
    """Handle a ``/command`` line. Returns a verb the REPL loop acts on
    (``"reset"``, ``"retry"``, ``"undo"``, ``"exit"``, or
    ``("load-session", ref)``), else ``None``. Unknown commands (including
    ``/help``) print the command list."""
    if line.strip().startswith("/!"):
        _handle_shell(line.strip()[2:].strip(), state)
        return None
    cmd, _, arg = line.strip().partition(" ")
    handler = _SLASH_HANDLERS.get(cmd)
    if handler is None:
        _print_shim_help(state)
        return None
    return handler(cmd, arg.strip(), state)


# /! and /help have no table entry: the shell prefix short-circuits before
# dispatch, and /help rides the unknown-command path.
_ALL_COMMANDS = sorted([*_SLASH_HANDLERS, "/!", "/help"])


def _completion_options(buf: str, text: str) -> list[str] | None:
    """Completion candidates for ``text`` (the word being completed) given the
    full line ``buf`` - shared by the readline and prompt_toolkit backends.

    Completes command names on a leading ``/``, the on/off/clear arguments of
    ``/history``, and filesystem paths after ``/load`` and ``/!``. Ordinary
    chat text yields no matches, so completion stays inert outside the
    command surface.
    """
    import glob

    if buf.startswith("/history "):
        return [o for o in ("on", "off", "clear") if o.startswith(text)]
    if buf.startswith("/reasoning "):
        return [o for o in ("show", "hide", "raw") if o.startswith(text)]
    if buf.startswith("/render "):
        return [o for o in ("plain", "lite", "rich") if o.startswith(text)]
    if buf.startswith("/thinking-budget "):
        return [o for o in ("off",) if o.startswith(text)]
    if buf.startswith("/load-session "):
        from .sessions import list_sessions

        return [s["name"] for s in list_sessions() if s["name"].startswith(text)]
    if buf.startswith("/theme "):
        from .theme import list_themes

        head = buf.split(" ", 2)
        if len(head) > 2:  # completing the modifier
            return [o for o in ("cb",) if o.startswith(text)]
        return [o for o in list_themes() if o.startswith(text)]
    if buf.startswith(("/load ", "/! ", "/image ", "/audio ", "/export ")):
        return [
            m + ("/" if os.path.isdir(m) else "")
            for m in sorted(glob.glob(os.path.expanduser(text) + "*"))
        ]
    if text.startswith("/"):
        return [c + " " for c in _ALL_COMMANDS if c.startswith(text)]
    return None


def _make_completer(readline, state: ChatState):
    """A readline completer over :func:`_completion_options`."""

    def complete(text: str, index: int):
        options = _completion_options(readline.get_line_buffer(), text)
        if options is None:
            return None
        return options[index] if index < len(options) else None

    return complete


def _wire_completion(readline, state: ChatState) -> None:
    """Bind Tab to the /command completer (GNU readline or libedit)."""
    readline.set_completer(_make_completer(readline, state))
    # The default delims include "/" - drop it so "/te" and "/load src/f"
    # complete as whole tokens.
    readline.set_completer_delims(" \t\n")
    backend = getattr(readline, "backend", None)  # py3.13+; None earlier
    is_libedit = (
        backend == "editline"
        if backend is not None
        else "libedit" in (readline.__doc__ or "")
    )
    if is_libedit:
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")


def _ptk_history_reload(hist) -> None:
    """Drop prompt_toolkit's one-shot load cache so the next prompt re-reads the
    history file. ``History.load()`` fills ``_loaded_strings`` once, so a session
    started with ``--no-history`` would never pick the file up on ``/history on``.
    Both attributes are prompt_toolkit internals and the ``[chat]`` extra leaves
    it unpinned: if a future release renames them, enabling still works for this
    session's own entries - it just won't backfill the file."""
    if hasattr(hist, "_loaded") and hasattr(hist, "_loaded_strings"):
        hist._loaded = False
        hist._loaded_strings = []


def _ptk_key_bindings():
    """Alt-Enter inserts a newline instead of submitting. Terminals configured
    to send ESC CR for Shift-Enter (iTerm2, VS Code, Windows Terminal) hit the
    same binding; unconfigured ones send a plain CR, which submits."""
    from prompt_toolkit.key_binding import KeyBindings

    kb = KeyBindings()

    @kb.add("escape", "enter")
    def _insert_newline(event):
        event.current_buffer.insert_text("\n")

    return kb


def _wire_ptk(state: ChatState) -> bool:
    """Upgrade the prompt to a prompt_toolkit session when the package is
    installed (the ``[chat]`` extra). Returns False to fall back to readline.

    Adds over the readline shim: completion menus while typing a /command,
    fish-style ghost suggestions from history, a bottom toolbar (live
    sampling settings, staged-block count, last reply tok/s), and proper
    bracketed paste. History persists to ``chat_history.ptk`` (prompt_toolkit
    format) next to readline's file; ``state.history_enabled`` gates both loading
    and storing, so ``/history on|off`` works mid-session.
    """
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from prompt_toolkit.completion import Completer, Completion
        from prompt_toolkit.history import FileHistory
    except ImportError:
        return False

    hist_file = _history_path().with_suffix(".ptk")
    try:
        hist_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False

    class _ToggleableFileHistory(FileHistory):
        def load_history_strings(self):
            if not state.history_enabled:
                return
            yield from super().load_history_strings()

        def store_string(self, string: str) -> None:
            if state.history_enabled:
                super().store_string(string)

    class _SlashCompleter(Completer):
        def get_completions(self, document, complete_event):
            text = document.get_word_before_cursor(WORD=True)
            options = _completion_options(document.text_before_cursor, text)
            for o in options or ():
                yield Completion(o, start_position=-len(text))

    def _toolbar():
        s = state.sampling
        parts = []
        if state.model_name:
            parts.append(state.model_name)
        parts += [
            f"temp={s['temp']:g}",
            f"top-p={s['top_p']:g}",
            f"max-tok={s['max_tokens'] or 'off'}",
        ]
        if state.ctx_used and state.ctx_max:
            parts.append(f"ctx {_fmt_k(state.ctx_used)}/{_fmt_k(state.ctx_max)}")
        if state.staged:
            parts.append(f"+{len(state.staged)} staged")
        if state.staged_images:
            parts.append(f"+{len(state.staged_images)} img")
        if state.staged_audio:
            parts.append(f"+{len(state.staged_audio)} aud")
        if state.last_tps:
            parts.append(f"{state.last_tps:.1f} tok/s")
        return " · ".join(parts)

    state.ptk_session = PromptSession(
        history=_ToggleableFileHistory(hist_file),
        auto_suggest=AutoSuggestFromHistory(),
        completer=_SlashCompleter(),
        complete_while_typing=True,
        bottom_toolbar=_toolbar,
        key_bindings=_ptk_key_bindings(),
    )
    state.history_loaded = True
    return True


def _wire_input(no_history: bool) -> ChatState:
    """Set up the line editor: prompt_toolkit when installed, else readline.
    Piped/scripted stdin skips prompt_toolkit (it needs a real tty) so
    ``printf 'hi\\nq\\n' | gmlx chat ...`` works."""
    state = ChatState(history_enabled=not no_history)
    if sys.stdin.isatty() and _wire_ptk(state):
        return state
    try:
        import readline
    except ImportError:  # pragma: no cover - absent on some builds
        readline = None
    _wire_history(state, readline)
    if readline is not None:
        _wire_completion(readline, state)
    return state


def _read_input(state: ChatState, prompt: str) -> str:
    """Read a prompt line, honoring a pending /load prefill."""
    pending = state.take("pending_insert")
    # Scripted-input seam: a caller may inject ``state.input_fn`` (called
    # with the prompt and any pending /load prefill) to drive the REPL without
    # a terminal - used by the end-to-end loop tests.
    hook = state.input_fn
    if hook is not None:
        return hook(prompt, pending)
    session = state.ptk_session
    if session is not None:
        return session.prompt(prompt, default=pending or "")
    readline = state.readline
    if pending is None:
        return input(prompt)
    if readline is None:
        print("[chat] (no readline: sending the loaded text as-is)")
        return pending
    # Startup hook fires as input() starts: the text lands in the line
    # buffer, editable, and is submitted only on Enter.
    readline.set_startup_hook(lambda t=pending: readline.insert_text(t))
    try:
        return input(prompt)
    finally:
        readline.set_startup_hook(None)


# REPL


def _handoff_esc(state: ChatState) -> bool:
    """Whether an Esc was typed during the prompt->generation handoff.

    prompt_toolkit's app teardown keeps reading the tty while it waits for a
    cursor-position response, so an Esc pressed right after submit never
    reaches the tty queue _EscCancel watches: a lone \\x1b is held in the
    vt100 parser as an ambiguous sequence prefix, and completed keys are
    stored as typeahead for the next prompt. Drain both: an Escape is a
    cancel request; anything else (typed-ahead text) is put back."""
    session = state.ptk_session
    if session is None:
        return False
    try:
        from prompt_toolkit.input.typeahead import get_typeahead, store_typeahead
        from prompt_toolkit.keys import Keys

        keys = get_typeahead(session.input)
        flush_keys = getattr(session.input, "flush_keys", None)
        if flush_keys is not None:
            keys += flush_keys()
    except Exception:
        return False
    esc = any(k.key == Keys.Escape for k in keys)
    if keys and not esc:
        store_typeahead(session.input, keys)
    return esc


class _EscCancel:
    """Watch for control keys while a reply streams.

    The terminal goes into cbreak for the duration (keypresses readable
    immediately, not echoed into the streaming text; Ctrl-C still signals).
    ``pressed()`` polls stdin without blocking and drains whatever arrived,
    so arrow-key escape tails don't leak into the next prompt. Esc cancels;
    Ctrl-O (``\\x0f``) fires ``on_toggle`` (collapse/expand thinking). Inert
    when stdin is not a tty.
    """

    def __init__(self, on_toggle=None):
        self._on_toggle = on_toggle

    def __enter__(self):
        self.fd = None
        if sys.stdin.isatty():
            import termios
            import tty

            self.fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(self.fd)
            # TCSADRAIN, not the default TCSAFLUSH: an Esc typed during the
            # prompt->generation handoff must survive the mode switch so the
            # first pressed() poll cancels at token 1. TCSAFLUSH discards it.
            tty.setcbreak(self.fd, termios.TCSADRAIN)
        return self

    def pressed(self) -> bool:
        import select

        if self.fd is None:
            return False
        hit = False
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = os.read(self.fd, 1)
            if ch == b"\x1b":
                hit = True
            elif ch == b"\x0f" and self._on_toggle is not None:
                self._on_toggle()
        return hit

    def __exit__(self, *exc):
        import termios

        if self.fd is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved)


def _stream_reply(
    chunks,
    state: ChatState,
    stops: list | None = None,
    start_in_thinking: bool = False,
    drafter=None,
) -> tuple[str, bool]:
    """Print a streaming reply; Esc or Ctrl-C cancels it (the session keeps
    running) and a ``stops`` sequence ends it cleanly (trimmed). Reasoning
    ("thinking") spans are stripped of their control markers and dimmed (or
    hidden) per ``state.reasoning`` - ``start_in_thinking`` seeds the case
    where the chat template pre-opens ``<think>`` so only the close is streamed.
    Returns ``(text_so_far, canceled)`` - the *raw* text (markers intact) so
    multi-turn history stays faithful - and records the reply's tok/s for the
    stat line + toolbar when it completes."""
    from .generation import StopScanner
    from .reasoning import ReasoningFilter, ReasoningPrinter, want_color

    scanner = StopScanner(stops) if stops else None
    state.last_think_open = bool(start_in_thinking)
    display = state.reasoning
    rf = None if display == "raw" else ReasoningFilter(start_in_thinking=start_in_thinking)
    theme = state.theme
    renderer = None
    if state.render in ("lite", "rich") and display != "raw":
        from .render import StreamRenderer

        renderer = StreamRenderer(state.render, theme)
    printer = ReasoningPrinter(
        display=display,
        color=want_color(),
        theme=theme,
        answer_sink=renderer.feed if renderer else None,
    )

    def _toggle() -> None:
        # Ctrl-O: collapse<->expand thinking, live for this reply and persisted as
        # the default for the next (raw is left alone - set it via /reasoning).
        if state.reasoning == "raw":
            return
        new = "hide" if state.reasoning == "show" else "show"
        state.reasoning = new
        printer.set_display(new)

    def _show(text: str) -> None:
        if rf is None:
            printer.feed_spans([(text, "answer")])
        else:
            printer.feed_spans(rf.feed(text))

    reply: list[str] = []

    def _accept(text: str) -> str:
        reply.append(text)  # raw (markers intact) -> return value + KV faithful
        _show(text)
        return text

    last, canceled, stopped = None, False, False
    pre_canceled = _handoff_esc(state)
    try:
        with _EscCancel(on_toggle=_toggle) as esc:
            for r in chunks:
                if pre_canceled:
                    canceled = True
                    break
                if scanner is None:
                    _accept(r.text)
                else:
                    out, stopped = scanner.feed(r.text)
                    if out:
                        _accept(out)
                printer.tick()
                last = r
                if stopped or esc.pressed():
                    canceled = not stopped
                    break
            else:
                if scanner is not None:
                    tail = scanner.flush()
                    if tail:
                        _accept(tail)
        if rf is not None:
            printer.feed_spans(rf.flush())
    except KeyboardInterrupt:
        canceled = True
    finally:
        printer.close(canceled=canceled)
        if renderer is not None:
            renderer.finalize()
    stats: dict = {}
    if last is not None:
        stats = {
            "prompt_tokens": int(getattr(last, "prompt_tokens", 0) or 0),
            "prompt_tps": float(getattr(last, "prompt_tps", 0.0) or 0.0),
            "gen_tokens": int(getattr(last, "generation_tokens", 0) or 0),
            "gen_tps": float(getattr(last, "generation_tps", 0.0) or 0.0),
        }
        accepts = list(getattr(drafter, "accept_lens", None) or [])
        drafts = list(getattr(drafter, "draft_lens", None) or [])
        if drafts:
            stats["accepted"] = int(sum(accepts))
            stats["drafted"] = int(sum(drafts))
            stats["rounds"] = len(drafts)
            stats["accept_rate"] = stats["accepted"] / max(1, stats["drafted"])
            stats["mean_accept_len"] = stats["accepted"] / max(1, len(accepts))
        state.last_tps = stats["gen_tps"]
    state.last_stats = stats
    if canceled:
        print("\n[chat] reply canceled - /retry regenerates, /reset restarts the chat")
    else:
        print()
        cap = int((state.sampling or {}).get("max_tokens") or 0)
        # Gate on finish_reason where the stream reports one ("length" = cap
        # hit); the token-count heuristic only backstops chunk shapes without
        # it, so an EOS landing exactly on the cap-th token stays silent.
        finish = getattr(last, "finish_reason", None)
        cap_hit = (finish == "length" if finish is not None
                   else stats.get("gen_tokens", 0) >= cap)
        if cap and not stopped and cap_hit:
            print(f"[chat] note: reply stopped at the max-tokens cap ({cap}) - "
                  "raise it, or `/max-tokens 0` to generate until the model "
                  "stops", file=sys.stderr)
    return "".join(reply), canceled


def _opens_thinking(prompt) -> bool:
    """Whether a rendered ``prompt`` pre-opens a ``<think>`` block (so the model
    streams only the close). Tolerant of token-id prompts - returns False for
    non-strings, so callers pass the string render where they have one."""
    from .thinking_budget import prompt_opens_thinking

    return prompt_opens_thinking(prompt)


def _vlm_message(
    model_type: str,
    content: str,
    role: str = "user",
    n_images: int = 0,
    n_audios: int = 0,
):
    """Render one message the way mlx-vlm's ``apply_chat_template`` would -
    but at the turn that sends it. mlx-vlm places all media markers on the
    *last* user message of a conversation, which breaks multi-turn re-prefill
    (an image sent three turns ago would migrate to the newest message);
    rendering per-turn pins each marker to its own turn."""
    from mlx_vlm.prompt_utils import MODEL_CONFIG, get_message_json

    if model_type not in MODEL_CONFIG:
        return {"role": role, "content": content}
    return get_message_json(
        model_type,
        content,
        role,
        skip_image_token=n_images == 0,
        skip_audio_token=n_audios == 0,
        num_images=n_images,
        num_audios=n_audios,
    )


# --assistant: the server-backed tool-loop turn engine.

# chat sampling key -> chat-completions payload key, forwarded per round when
# the flag was passed or the /command changed it (untouched knobs stay unset
# so the server's own profile sampling applies).
_ASSISTANT_SAMPLING = {
    "temp": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "xtc_probability": "xtc_probability",
    "xtc_threshold": "xtc_threshold",
    "repetition_penalty": "repetition_penalty",
    "repetition_context_size": "repetition_context_size",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
}

# (attr, flag) pairs that cannot be honored server-side; passing one with
# --assistant is a hard error rather than a silently different conversation.
_ASSISTANT_REJECT = (
    ("adapter", "--adapter"),
    ("mmproj", "--mmproj"),
    ("chat_template", "--chat-template"),
    ("no_chat_template", "--no-chat-template"),
    ("chat_template_config", "--chat-template-config"),
)

# (attr, flag, is-set) triples the server owns; each passed one gets a note.
_ASSISTANT_NOOP = (
    ("arch", "--arch"),
    ("hf_source", "--hf-source"),
    ("no_remap", "--no-remap"),
    ("no_zero_copy", "--no-zero-copy"),
    ("max_kv_size", "--max-kv-size"),
    ("speculative", "--speculative"),
    ("no_speculative", "--no-speculative"),
    ("draft_gguf", "--draft-gguf"),
    ("stochastic_mtp", "--stochastic-mtp"),
    ("kv_bits", "--kv-bits"),
    ("prefill_step_size", "--prefill-step-size"),
    ("stream_experts", "--stream-experts"),
    ("stream_cpu", "--stream-cpu"),
    ("moe_experts", "--moe-experts"),
    ("moe_expert_mass", "--moe-expert-mass"),
    ("moe_expert_probe", "--moe-expert-probe"),
    ("moe_miss_shed", "--moe-miss-shed"),
    ("moe_layer_shed", "--moe-layer-shed"),
    ("thinking_budget", "--thinking-budget"),
    ("resize_shape", "--resize-shape"),   # only meaningful with --mmproj (rejected)
)


def _assistant_flag_gate(args, parser) -> int | None:
    """Reject/noop the local-load flags under --assistant. Rejected flags
    exit 2; server-owned ones print one [chat] note and are ignored."""
    for attr, flag in _ASSISTANT_REJECT:
        if getattr(args, attr, None) not in (None, False):
            print(f"error: {flag} is not supported with --assistant "
                  "(the server owns the model and its template)",
                  file=sys.stderr)
            return 2
    noop = [flag for attr, flag in _ASSISTANT_NOOP
            if getattr(args, attr, None) not in (None, False)]
    for attr, flag in (("kv_group_size", "--kv-group-size"),
                       ("quantized_kv_start", "--quantized-kv-start"),
                       ("prefill_feeder", "--prefill-feeder"),
                       ("decode_feeder", "--decode-feeder")):
        if getattr(args, attr, None) != parser.get_default(attr):
            noop.append(flag)
    if noop:
        print(f"[chat] server owns {', '.join(noop)} - ignored with "
              "--assistant")
    return None


def _setup_assistant(args):
    """Resolve the server + served model and build the AssistantBrain from
    the config's shared ``assistant:`` block (the same tools + memory store
    ``gmlx talk`` uses). Returns ``(brain, model_request, base_url, extra)``
    or an int exit code. ``extra`` is the live payload-extras dict the brain's
    stream seam reads every round."""
    import atexit

    from . import launch as launch_mod

    ns = argparse.Namespace(
        harness=None, rerun_label="chat", base_url=args.base_url,
        host=args.host, port=args.port, api_key=args.api_key,
        no_start=args.no_start, start_timeout=args.start_timeout,
        config_only=False)
    rc = launch_mod._ensure_server(ns)
    if rc is not None:
        return rc
    base_url, api_key = ns.base_url, ns.api_key

    from .talk_client import TalkClientError, probe_capabilities
    try:
        caps = probe_capabilities(base_url, api_key)
    except TalkClientError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    requested = args.gguf
    # Inline `<id>@<profile>` addressing: a served id can never contain '@'
    # (config validation rejects it), so split the head off for the
    # served-membership check and route the tail through --profile (mirrors
    # launch._pick_default; profile validity is the server's call -> 400).
    if requested and "@" in requested and not (
            requested.endswith(".gguf")
            or os.path.exists(os.path.expanduser(requested))):
        head, _, tail = requested.rpartition("@")   # last '@', per config.split_address
        if head:   # an empty head ("@coding") is malformed - leave it to fail the served check
            requested = head
            args.profile = args.profile or tail
    if requested and (requested.endswith(".gguf")
                      or os.path.exists(os.path.expanduser(requested))):
        print("error: --assistant chats through the server - pass a served "
              "model id, not a file (add it to your config's models:)",
              file=sys.stderr)
        return 2
    served = caps.get("chat_ids") or []
    model = requested or caps.get("default") or (
        served[0] if len(served) == 1 else None)
    if not model:
        ids = ", ".join(served) or "(none)"
        print(f"error: no model selected and the server has no default - "
              f"pass one of: {ids}", file=sys.stderr)
        return 2
    if requested and requested not in served:
        ids = ", ".join(served) or "(none)"
        print(f"error: {requested!r} is not served - served ids: {ids} "
              "(add it to your config)", file=sys.stderr)
        return 2
    model_request = f"{model}@{args.profile}" if args.profile else model

    from .config import AssistantCfg
    a = AssistantCfg()
    try:
        if args.config:
            from . import config as cfgmod
            a = cfgmod.load_config(args.config).assistant
        else:
            from .launch import _discover_config
            cfg, _path = _discover_config()
            if cfg is not None:
                a = cfg.assistant
    except Exception as e:                    # noqa: BLE001 - degrade
        print(f"[chat] config: {e} - assistant runs tool-less",
              file=sys.stderr)

    from .assistant_brain import AssistantBrain
    from .talk_client import stream_chat as _stream_chat
    from .talk_mcp import connect_servers
    mcp_host, registry, warns = connect_servers(
        a.mcp, call_timeout_s=a.tool_timeout_s)
    for w in warns:
        print(f"[chat] {w}", file=sys.stderr)

    memory = None
    if a.memory.enabled:                      # the same store talk uses
        from .talk_memory import MemoryStore, make_extractor
        extractor = (make_extractor(base_url, model_request, api_key=api_key)
                     if a.memory.extract else None)
        memory = MemoryStore(base_url=base_url, api_key=api_key,
                             path=a.memory.path, top_k=a.memory.top_k,
                             extract=extractor, ttl_days=a.memory.ttl_days,
                             max_items=a.memory.max_items)

    # Usage chunks are gated on stream_options server-side; sampling knobs
    # join this dict per turn (see _sync_assistant_extra).
    extra: dict = {"stream_options": {"include_usage": True}}

    def seam(burl, *, model, messages, max_tokens, api_key=None, tools=None,
             timeout=600.0):
        return _stream_chat(burl, model=model, messages=messages,
                            max_tokens=max_tokens, api_key=api_key,
                            tools=tools, timeout=timeout, extra=dict(extra))

    brain = AssistantBrain(
        base_url=base_url, model=model_request, api_key=api_key,
        system=None, max_tokens=1024, tools=registry,
        max_tool_rounds=a.max_tool_rounds, tool_timeout_s=a.tool_timeout_s,
        memory=memory, stream=seam)

    def cleanup():
        brain.close()                         # closes the memory store
        if mcp_host is not None:
            mcp_host.close()

    atexit.register(cleanup)
    return brain, model_request, base_url, extra


def _sync_assistant_extra(state) -> None:
    """Refresh the forwarded sampling knobs from the live /command values:
    a knob rides along once the CLI set it or a /command moved it off the
    session baseline; everything else stays server-side."""
    extra = state.assistant_extra
    s = state.sampling
    baseline = state.assistant_baseline
    touched = state.assistant_touched
    for key, payload in _ASSISTANT_SAMPLING.items():
        if key in touched or s[key] != baseline[key]:
            extra[payload] = s[key]
        else:
            extra.pop(payload, None)


def _assistant_reply(brain, user_text: str, state: ChatState) -> tuple[str, bool]:
    """One assistant turn through the standard reply pipeline: brain events
    adapt into ``_stream_reply`` chunks (say -> text; status -> a transient
    line cleared on the first spoken span; done -> the stat objects the
    renderer already understands). Esc-cancel closes the adapter, which
    closes the brain turn - partial text commits, tool rounds stay atomic."""
    from types import SimpleNamespace

    from .talk_client import TalkClientError

    status_shown = [False]
    t0 = time.monotonic()

    def _clear_status():
        if status_shown[0]:
            sys.stdout.write("\r\x1b[K")
            sys.stdout.flush()
            status_shown[0] = False

    def chunks():
        try:
            for kind, payload in brain.turn(user_text):
                if kind == "say":
                    _clear_status()
                    yield SimpleNamespace(text=payload)
                elif kind == "status":
                    _clear_status()
                    sys.stdout.write(f"[assistant] {payload}...")
                    sys.stdout.flush()
                    status_shown[0] = True
                elif kind == "done":
                    u = payload or {}
                    n = int(u.get("completion_tokens") or 0)
                    el = time.monotonic() - t0
                    yield SimpleNamespace(
                        text="",
                        prompt_tokens=int(u.get("prompt_tokens") or 0),
                        prompt_tps=0.0,
                        generation_tokens=n,
                        generation_tps=(n / el) if el > 0 and n else 0.0)
        except TalkClientError as e:
            _clear_status()
            print(f"\n[chat] server error: {e}", file=sys.stderr)
        finally:
            _clear_status()

    return _stream_reply(chunks(), state)


def _chat_memory_cmd(arg: str, state: ChatState) -> None:
    """/memory - list / forget ID / clear, against the assistant's store."""
    brain = state.assistant_brain
    mem = getattr(brain, "memory", None) if brain is not None else None
    if mem is None:
        print("[chat] no memory store (--assistant with assistant.memory "
              "enabled only)")
        return
    words = arg.split()
    if not words:
        total = mem.count()
        if total == 0:
            print("[chat] no memories stored")
            return
        rows = mem.list_all(limit=20)
        header = f"[chat] {total} memories"
        if total > len(rows):
            header += f" (newest {len(rows)} shown)"
        lines = [header]
        now = time.time()
        for r in rows:
            text = " ".join(str(r["text"]).split())
            if len(text) > 60:
                text = text[:57] + "..."
            age_s = now - r["created"]
            age = (f"{age_s / 86400:.0f}d" if age_s >= 86400
                   else f"{age_s / 3600:.0f}h" if age_s >= 3600
                   else f"{age_s / 60:.0f}m")
            lines.append(f"  #{r['id']:<5} {age:>4}  x{r['recalled']:<3} "
                         f"{text}")
        print("\n".join(lines))
    elif words[0] == "forget" and len(words) == 2:
        try:
            mem_id = int(words[1].lstrip("#"))
        except ValueError:
            print("[chat] usage: /memory forget ID")
            return
        print(f"[chat] forgot #{mem_id}" if mem.delete(mem_id)
              else f"[chat] no memory #{mem_id}")
    elif words[0] == "clear":
        if words[1:] == ["yes"]:
            print(f"[chat] cleared {mem.clear()} memories")
        else:
            print(f"[chat] this deletes {mem.count()} memories - confirm "
                  "with: /memory clear yes")
    else:
        print("[chat] usage: /memory | /memory forget ID | /memory clear")


def _session_restore_settings(state: ChatState, doc: dict, name: str) -> None:
    """Restore session identity + settings (system prompt, sampling knobs,
    reasoning/render/thinking-budget, theme) from a saved session doc."""
    state.session_name = name
    state.session_created = doc.get("created")
    if doc.get("system_prompt") is not None:
        state.system_prompt = doc["system_prompt"]
    for k, v in (doc.get("sampling") or {}).items():
        if k in state.sampling:
            state.sampling[k] = v
    for k in ("reasoning", "render", "thinking_budget"):
        if doc.get(k) is not None:
            setattr(state, k, doc[k])
    if doc.get("theme"):
        from .reasoning import want_color
        from .theme import resolve_theme

        try:
            state.theme = resolve_theme(
                doc["theme"],
                colorblind=bool(doc.get("colorblind")),
                color=want_color(),
            )
            state.colorblind = bool(doc.get("colorblind"))
        except ValueError:
            pass


def _session_parse_messages(doc: dict, system_prompt) -> tuple[list, list, int]:
    """Pair the saved messages into transcript entries. Returns (replay
    messages, transcript, generated-token total); saved turns carry no cache
    checkpoint, so they never rewind in place."""
    replay, transcript, entry, n_tok = [], [], None, 0
    if system_prompt:
        replay.append({"role": "system", "content": system_prompt})
    for m in doc.get("messages") or []:
        role = m.get("role")
        if role == "user":
            entry = {
                "user": dict(m),
                "assistant": None,
                "cache_before": None,  # unknown checkpoint: no rewind
                "first_turn_before": False,
                "vlm_lens": (0, 0, 0),
            }
            replay.append({"role": "user", "content": m.get("content", "")})
        elif role == "assistant" and entry is not None:
            entry["assistant"] = dict(m)
            n_tok += int((m.get("stats") or {}).get("gen_tokens", 0) or 0)
            transcript.append(entry)
            replay.append(
                {"role": "assistant", "content": m.get("content", "")}
            )
            entry = None
    return replay, transcript, n_tok


def _session_rebuild_vlm_history(
    transcript, state: ChatState, model_type: str,
    vlm_msgs: list, vlm_images: list, vlm_audios: list,
) -> None:
    """Rebuild the VLM conversation lists from a restored transcript, mutating
    ``vlm_msgs``/``vlm_images``/``vlm_audios`` in place. VLM turns re-prefill
    from ``vlm_msgs`` each turn (markers pinned per turn), so the replay path
    ``replay_messages`` uses does not apply; missing media files are warned
    about and skipped."""
    missing = []
    if state.system_prompt:
        vlm_msgs.append(
            {"role": "system", "content": state.system_prompt}
        )
    for e in transcript:
        u = e["user"]
        imgs = list(u.get("images") or [])
        auds = list(u.get("audios") or [])
        missing += [f for f in imgs + auds if not os.path.isfile(f)]
        vlm_msgs.append(
            _vlm_message(
                model_type, u.get("content", ""), "user",
                len(imgs), len(auds),
            )
        )
        vlm_images.extend(imgs)
        vlm_audios.extend(auds)
        if e["assistant"]:
            vlm_msgs.append(
                _vlm_message(
                    model_type, e["assistant"].get("content", ""),
                    "assistant",
                )
            )
    if missing:
        print(
            f"[chat] warning: {len(missing)} media file(s) from the "
            "saved session no longer exist: " + ", ".join(missing[:3])
        )


class _ChatBackend:
    """One chat mode's loaded model stack (assistant / VLM+MTP / VLM / MTP
    text / plain text) plus its deferred-load machinery. ``new_text_cache``
    builds the per-conversation KV cache (None on paths without a local
    one); ``join`` finishes a pending background load and runs the deferred
    model-dependent setup."""

    def __init__(self):
        self.model = None
        self.tok = None
        self.config = None
        self.processor = None
        self.drafter = None
        self.model_type = ""
        self.load_pending = False
        self.new_text_cache = lambda: None
        self._join = None            # set on the background text path

    def join(self) -> bool:
        """Join the background text load; returns True when it ran (the
        caller rebinds model state), False when nothing was pending.
        Raises the load error, or _ChatExit from the deferred setup."""
        if not self.load_pending:
            return False
        self.load_pending = False
        self._join()
        return True


def _require_chat_template(tok, *, verbatim_hint: bool = False) -> None:
    if tok.chat_template is not None:
        return
    extra = (", or --no-chat-template to send turns verbatim"
             if verbatim_hint else "")
    print(
        "error: this GGUF carries no chat template (base model?) - "
        f"pass one with --chat-template{extra}",
        file=sys.stderr,
    )
    raise _ChatExit(1)


def _backend_vlm_mtp(args) -> _ChatBackend:
    # VLM loaded on mlx-vlm classes + an assistant drafter: text-only turns
    # run through the MTP engine over a persistent text KV cache
    # (model.language_model), while image/audio turns use the plain VLM
    # stream.
    from mlx_vlm.models.cache import make_prompt_cache as _mtp_make_cache

    from . import loadlog
    from .mtp_load import load_vlm_mtp_model

    b = _ChatBackend()
    with loadlog.load_ui(args.verbose, args.gguf):
        b.model, b.drafter, b.config, b.tok, b.processor = load_vlm_mtp_model(
            args.gguf,
            args.mmproj,
            arch=args.arch,
            draft_gguf_path=args.draft_gguf,
            chat_template=args.chat_template,
            zero_copy=not args.no_zero_copy,
            verbose=args.verbose,
        )
    b.model_type = _model_type(b.config)
    _require_chat_template(b.tok)
    b.new_text_cache = lambda: _mtp_make_cache(b.model.language_model)
    return b


def _backend_vlm(args) -> _ChatBackend:
    from . import loadlog
    from .vlm import load_vlm_model

    b = _ChatBackend()
    with loadlog.load_ui(args.verbose, args.gguf):
        b.model, b.config, b.processor = load_vlm_model(
            args.gguf,
            args.mmproj,
            hf_source=args.hf_source,
            arch=args.arch,
            zero_copy=not args.no_zero_copy,
            verbose=args.verbose,
        )
    b.model_type = _model_type(b.config)
    return b


def _backend_mtp_text(args, kv_kwargs) -> _ChatBackend:
    # MTP speculative decoding: the target loads on mlx-vlm classes alongside
    # a drafter (native-head for qwen3.5/3.6, or a --draft-gguf assistant).
    # Each reply runs through mlx-vlm's MTP engine; the cache is reused
    # across turns exactly like the plain text path.
    from mlx_vlm.models.cache import make_prompt_cache as _mtp_make_cache

    from . import loadlog
    from .mtp_load import load_mtp_model

    b = _ChatBackend()
    with loadlog.load_ui(args.verbose, args.gguf):
        b.model, b.drafter, b.config, b.tok = load_mtp_model(
            args.gguf,
            arch=args.arch,
            draft_gguf_path=args.draft_gguf,
            chat_template=args.chat_template,
            zero_copy=not args.no_zero_copy,
            verbose=args.verbose,
        )
    _require_chat_template(b.tok)

    # --kv-bits on the MTP path: same pooled-cache packing as the plain
    # path (rollback/replay are watermark moves, storage-agnostic). For
    # drafter models with no pooled caches the flag stays dropped, with
    # an accurate note instead of the generic dropped-flags warning.
    mtp_quantize_pools = None
    if kv_kwargs.get("kv_bits") is not None:
        from .generation import quantize_pooled_caches

        bits = kv_kwargs["kv_bits"]
        group = kv_kwargs.get("kv_group_size", 64)
        probe = _mtp_make_cache(b.model.language_model)
        if quantize_pooled_caches(probe, bits, group):
            mtp_quantize_pools = (bits, group)
            print(
                f"[kv] {bits}-bit pooled KV cache "
                "(sliding windows stay fp16)"
            )
        else:
            print(
                "warning: --kv-bits not applied on the MTP path "
                "(no quantizable caches)",
                file=sys.stderr,
            )
        kv_kwargs["kv_bits"] = None

    def _new_text_cache():
        # Single-stream MTP uses a plain target KV cache the native drafter
        # reads back - one cache, reused across turns (like the text path).
        c = _mtp_make_cache(b.model.language_model)
        if mtp_quantize_pools is not None:
            from .generation import quantize_pooled_caches

            quantize_pooled_caches(c, *mtp_quantize_pools)
        return c

    b.new_text_cache = _new_text_cache
    return b


def _backend_plain_text(args, kv_kwargs) -> _ChatBackend:
    from mlx_lm.models.cache import make_prompt_cache

    from . import loadlog
    from .envflags import env_bool
    from .loader import load_model, preset_native_fp_wire_env

    # Sets GMLX_NATIVE_FP before the load reads it - on the background
    # path load_model runs off-thread, so the env must be in place here.
    preset_native_fp_wire_env(args)

    b = _ChatBackend()

    def _load_text_model():
        return load_model(
            args.gguf,
            arch=args.arch,
            hf_source=args.hf_source,
            chat_template=args.chat_template,
            no_remap=args.no_remap,
            zero_copy=not args.no_zero_copy,
            verbose=args.verbose,
        )

    quantize_pools = None   # set at the load join (kv-bits pooled packing)

    def _new_text_cache():
        c = make_prompt_cache(b.model, args.max_kv_size)
        if quantize_pools is not None:
            from .generation import quantize_pooled_caches

            quantize_pooled_caches(c, *quantize_pools)
        return c

    b.new_text_cache = _new_text_cache

    def _post_text_load():
        # Model-dependent setup, deferred to the load join on the
        # background path. Prints bare, so it must run on the main
        # thread with the tty free.
        nonlocal quantize_pools
        if not args.no_chat_template:
            _require_chat_template(b.tok, verbatim_hint=True)

        from .cli import _apply_placement

        _apply_placement(args, b.model)
        from .loader import _resolve_prefill_step

        step, defaulted = _resolve_prefill_step(
            b.model, kv_kwargs.get("prefill_step_size")
        )
        if defaulted:
            print(
                f"[prefill] streaming model: chunk size defaults to {step} "
                "(--prefill-step-size overrides)"
            )
        if step is not None:
            kv_kwargs["prefill_step_size"] = step

        # The MTP path already warns that --kv-bits is dropped. On plain
        # decoding, mlx-lm's per-step converter would crash on rotating
        # caches -- for arches with growing pooled caches (deepseek4) the
        # flag is honored by packing those at rest instead; the fp16
        # sliding windows are size-capped either way.
        if kv_kwargs.get("kv_bits") is not None:
            from .generation import kv_quantization_unsupported

            reason = kv_quantization_unsupported(b.model)
            if reason:
                from .generation import quantize_pooled_caches

                bits = kv_kwargs["kv_bits"]
                group = kv_kwargs.get("kv_group_size", 64)
                probe = make_prompt_cache(b.model, args.max_kv_size)
                if quantize_pooled_caches(probe, bits, group):
                    quantize_pools = (bits, group)
                    print(
                        f"[kv] {bits}-bit pooled KV cache "
                        "(sliding windows stay fp16)"
                    )
                else:
                    print(
                        f"warning: --kv-bits dropped: {reason}",
                        file=sys.stderr,
                    )
                kv_kwargs["kv_bits"] = None

        if args.adapter:
            from .adapter import apply_gguf_adapter
            from .discovery import header_meta

            adapter = os.path.abspath(os.path.expanduser(args.adapter))
            base_arch = (getattr(args, "arch", None)
                         or (header_meta(args.gguf) or {}).get("arch"))
            n = apply_gguf_adapter(b.model, b.config, adapter,
                                   base_arch=base_arch)
            print(f"[adapter] applied {n}-module GGUF LoRA from {adapter}")

    # Load in the background while the user types their first message.
    # Interactive tty only: piped stdin already holds its input (and the
    # scripted/e2e output order stays deterministic); verbose keeps its
    # inline diagnostics. GMLX_CHAT_BG_LOAD=0 restores the
    # synchronous load.
    b.load_pending = (
        env_bool("GMLX_CHAT_BG_LOAD", True)
        and sys.stdin.isatty()
        and not args.verbose
    )
    if not b.load_pending:
        with loadlog.load_ui(args.verbose, args.gguf):
            b.model, b.config, b.tok = _load_text_model()
        _post_text_load()
        return b

    import threading

    load_box: list = []

    def _bg_load():
        cap_holder = []
        try:
            with loadlog.capture(args.gguf) as cap:
                cap_holder.append(cap)
                result = _load_text_model()
            load_box.append(("ok", cap, result))
        except BaseException as e:
            load_box.append(
                ("err", cap_holder[0] if cap_holder else None, e))

    load_thread = threading.Thread(
        target=_bg_load, name="gmlx-chat-load", daemon=True)
    load_thread.start()

    def _join():
        # Join with a spinner while the load is still running, then run the
        # deferred model-dependent setup.
        if load_thread.is_alive():
            if sys.stderr.isatty():
                from .spinner import Spinner

                with Spinner(f"loading {os.path.basename(args.gguf)}"):
                    load_thread.join()
            else:
                load_thread.join()
        status, cap, payload = load_box[0]
        stray = cap.stray if cap is not None else ""
        if stray:
            # Third-party writes the loadlog router held back during the load.
            sys.stderr.write(stray if stray.endswith("\n") else stray + "\n")
        if status == "err":
            # Same merged reason line + dedupe flag as the run path, so the
            # CLI backstop doesn't stack a second "error:" line.
            from .loadlog import report_failure

            report_failure(payload, cap.stage if cap is not None else None)
            raise payload
        b.model, b.config, b.tok = payload
        if cap.summary:
            print(cap.summary)
        _post_text_load()

    b._join = _join
    return b


def _load_chat_backend(args, kv_kwargs, *, speculative: bool,
                       vlm_mtp: bool) -> _ChatBackend:
    """Load the model stack for the chat mode the flags resolved to.
    Raises _ChatExit for user-facing load errors (missing chat template)."""
    if args.assistant:
        # Server-backed: no local model, tokenizer, or KV cache.
        b = _ChatBackend()
        b.config = {}
        return b
    if vlm_mtp:
        return _backend_vlm_mtp(args)
    if args.mmproj is not None:
        return _backend_vlm(args)
    if speculative:
        return _backend_mtp_text(args, kv_kwargs)
    return _backend_plain_text(args, kv_kwargs)


def cmd_chat(argv: list[str] | None = None, prog: str = "gmlx chat") -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = _build_parser(prog)
    args = parser.parse_args(argv)
    if args.stochastic_mtp:
        from .speculative import set_stoch_accept

        set_stoch_accept(True)
    brain = None                  # --assistant: server-backed turn engine
    model_request = None
    if args.assistant:
        rc = _assistant_flag_gate(args, parser)
        if rc is not None:
            return rc
        if args.gguf:
            from .cli import split_path_intent

            split_path_intent(args)      # id@profile -> id + --profile
        logit_bias = parse_logit_bias(args.logit_bias)
        setup = _setup_assistant(args)
        if isinstance(setup, int):
            return setup
        brain, model_request, base_url, assistant_extra = setup
        args.gguf = model_request        # session naming/matching key
        if args.seed is not None:
            assistant_extra["seed"] = args.seed
        if args.stop:
            assistant_extra["stop"] = list(args.stop)
        if logit_bias:
            assistant_extra["logit_bias"] = logit_bias
        template_kwargs = {}
        resize_shape = None
        speculative = False
        vlm_mtp = False
    else:
        # The server-targeting flags only apply under --assistant; without it
        # chat loads the model in-process. Accepting them silently would leave
        # the user believing they are on the server while a second copy loads.
        for attr, flag in (("base_url", "--base-url"), ("host", "--host"),
                           ("port", "--port"), ("api_key", "--api-key"),
                           ("no_start", "--no-start")):
            if getattr(args, attr, None) not in (None, False):
                parser.error(f"{flag} targets a server and needs --assistant "
                             f"(without it, chat loads the model in-process)")
        if not args.gguf:
            parser.error("the gguf argument is required without --assistant")
        # By-name config fallback (same rules as `gmlx run`): a positional that isn't
        # an on-disk file is resolved against the server config by id/alias, applying that
        # model's path + sampling/template/load settings.
        from .cli import apply_family_defaults, maybe_load_from_config, split_path_intent

        split_path_intent(args)
        rc = maybe_load_from_config(args, parser, argv)
        if rc is not None:
            return rc
        gguf = os.path.expanduser(args.gguf)
        if not os.path.exists(gguf):
            if args.gguf.startswith(("hf:", "http://", "https://")):
                hint = (
                    " (remote refs work with `gmlx validate` / "
                    "`gmlx pull`; chat needs a local file)"
                )
            else:
                hint = (
                    " (not a file, and no matching model id/alias in your config - "
                    "see `gmlx list` or your config's models:)"
                )
            print(f"error: no such file: {args.gguf}{hint}", file=sys.stderr)
            return 2
        args.gguf = gguf
        rc = apply_family_defaults(args, parser, argv)
        if rc is not None:
            return rc
        # chat prints the family note up front: its load is backgrounded and
        # joined mid-REPL, so there is no clean post-load spot for it.
        from .cli import print_family_note
        print_family_note(args)
        if args.adapter and args.mmproj:
            print(
                "error: --adapter (live GGUF LoRA) on a --mmproj base is not "
                "supported yet.",
                file=sys.stderr,
            )
            return 2
        if args.mmproj and (
            args.stream_experts
            or args.stream_cpu
            or args.moe_experts is not None
            or args.moe_expert_mass is not None
            or args.moe_expert_probe
            or args.moe_miss_shed is not None
            or args.moe_layer_shed is not None
        ):
            print(
                "error: --stream-experts/--stream-cpu/--moe-experts/"
                "--moe-expert-mass/--moe-expert-probe/--moe-miss-shed/"
                "--moe-layer-shed are text-only (no VLM "
                "offload plumbing); drop --mmproj or the offload flags.",
                file=sys.stderr,
            )
            return 2
        from .cli import (
            _vlm_mtp_drafter_available,
            mtp_dropped_chat_flags,
            resolve_speculative,
        )

        # Native-head GGUFs auto-enable MTP (a separate --draft-gguf forces it); sampler
        # flags the chat MTP path can't honor are dropped with a warning (--no-mtp honors
        # them via plain decoding). --no-speculative/--no-mtp (or config
        # 'speculative: false') turns it off.
        speculative, mtp_note = resolve_speculative(args, args.gguf)
        if mtp_note:
            print(mtp_note)
        # VLM x MTP: a loaded VLM (--mmproj) still serves text-only turns through the MTP
        # path when a drafter is available -- a --draft-gguf assistant (gemma4) or a
        # native head in the LLM GGUF (qwen3.5/3.6). Image/audio turns (or a conversation
        # that already holds media) fall back to the plain VLM path. (resolve_speculative
        # treats --mmproj as a hard MTP blocker, so the VLM x MTP decision is its own.)
        vlm_mtp = bool(args.mmproj and _vlm_mtp_drafter_available(args))
        if speculative and not vlm_mtp and (
            args.mmproj or args.adapter or args.stream_experts or args.stream_cpu
            or args.moe_experts is not None or args.moe_expert_mass is not None
            or args.moe_expert_probe or args.moe_miss_shed is not None
            or args.moe_layer_shed is not None
        ):
            which = next(
                name
                for name, on in (
                    ("--mmproj", args.mmproj),
                    ("--adapter", args.adapter),
                    ("--stream-experts", args.stream_experts),
                    ("--stream-cpu", args.stream_cpu),
                    ("--moe-experts", args.moe_experts is not None),
                    ("--moe-expert-mass", args.moe_expert_mass is not None),
                    ("--moe-expert-probe", args.moe_expert_probe),
                    ("--moe-miss-shed", args.moe_miss_shed is not None),
                    ("--moe-layer-shed", args.moe_layer_shed is not None),
                )
                if on
            )
            hint = (
                " (add --draft-gguf for text-only MTP turns on this VLM)"
                if which == "--mmproj"
                else " (MTP supports plain text bases only)"
            )
            print(
                f"error: {which} on an MTP base is not supported{hint}.",
                file=sys.stderr,
            )
            return 2
        if speculative or vlm_mtp:
            dropped = mtp_dropped_chat_flags(args)
            if dropped:
                print(
                    f"warning: {', '.join(dropped)} not applied on the MTP path "
                    f"(set --no-mtp to apply via plain decoding)",
                    file=sys.stderr,
                )
        template_kwargs = parse_template_config(args.chat_template_config)
        logit_bias = parse_logit_bias(args.logit_bias)
        resize_shape = parse_resize_shape(args.resize_shape)

    state = _wire_input(no_history=args.no_history)
    state.vlm = args.mmproj is not None
    state.reasoning = args.reasoning
    from .reasoning import want_color as _want_color
    from .render import resolve_render_mode
    from .theme import resolve_theme

    _color = _want_color()
    try:
        state.theme = resolve_theme(
            args.theme, colorblind=args.colorblind, color=_color
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    state.colorblind = args.colorblind
    render_mode, render_note = resolve_render_mode(
        args.render, tty=sys.stdout.isatty(), color=_color
    )
    state.render = render_mode
    if render_note:
        print(f"[chat] {render_note}")
    state.transcript = []
    state.session_stats = {
        "t0": time.monotonic(),
        "turns": 0,
        "prompt_tok": 0,
        "gen_tok": 0,
        "gen_time": 0.0,
        "accepted": 0,
        "drafted": 0,
        "rounds": 0,
    }
    state.sampling = {
        "temp": args.temp,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "max_tokens": args.max_tokens,
        "xtc_probability": args.xtc_probability,
        "xtc_threshold": args.xtc_threshold,
        "repetition_penalty": args.repetition_penalty,
        "repetition_context_size": args.repetition_context_size,
        "presence_penalty": args.presence_penalty,
        "frequency_penalty": args.frequency_penalty,
    }
    # KV-cache + prefill knobs shared by every turn (both backends).
    kv_kwargs = {
        "max_kv_size": args.max_kv_size,
        "kv_bits": args.kv_bits,
        "kv_group_size": args.kv_group_size,
        "quantized_kv_start": args.quantized_kv_start,
    }
    if args.prefill_step_size is not None:
        kv_kwargs["prefill_step_size"] = args.prefill_step_size

    if brain is not None:
        state.assistant_brain = brain
        state.assistant_extra = assistant_extra
        state.assistant_baseline = dict(state.sampling)
        state.assistant_touched = {
            k for k in _ASSISTANT_SAMPLING
            if getattr(args, k) != parser.get_default(k)
        }
    elif args.seed is not None:
        import mlx.core as mx

        mx.random.seed(args.seed)

    try:
        backend = _load_chat_backend(
            args, kv_kwargs, speculative=speculative, vlm_mtp=vlm_mtp)
    except _ChatExit as e:
        return e.code
    model, tok, config = backend.model, backend.tok, backend.config
    processor, drafter = backend.processor, backend.drafter
    model_type = backend.model_type
    cache = None
    if not backend.load_pending:
        cache = backend.new_text_cache()

    def _cfg_get(key):
        for c in (config, (config.get("text_config") if isinstance(config, dict) else None)):
            if c is None:
                continue
            v = c.get(key) if isinstance(c, dict) else getattr(c, key, None)
            if v:
                return v
        return None

    state.system_prompt = getattr(args, "system_prompt", None)
    state.thinking_budget = getattr(args, "thinking_budget", None)
    if args.assistant:
        # The session key is the served id, not a file path.
        model_key = model_request
        state.ctx_max = None
        state.model_name = model_request[:24]
        state.model_info = {"path": f"{model_request} (via {base_url})",
                            "model_type": "assistant"}
    else:
        model_key = os.path.abspath(args.gguf)
        try:
            from .discovery import derive_id

            state.model_name = derive_id(os.path.basename(args.gguf))[0][:24]
        except Exception:
            state.model_name = os.path.basename(args.gguf)[:24]

    def _bind_model_state():
        # Config-dependent state - deferred to the load join on the
        # background text path (the assistant branch bound its own above).
        if args.assistant:
            return
        state.ctx_max = args.max_kv_size or _cfg_get(
            "max_position_embeddings")
        state.model_info = _build_model_info(args, config, drafter, vlm_mtp)
        state.session_meta["arch"] = (
            state.model_info.get("arch") or args.arch)

    state.session_meta = {
        "path": model_key,
        "arch": (state.model_info or {}).get("arch") or args.arch,
        "mmproj": os.path.abspath(args.mmproj) if args.mmproj else None,
        "draft_gguf": os.path.abspath(args.draft_gguf) if args.draft_gguf else None,
    }
    state.session_name = None
    if not backend.load_pending:
        _bind_model_state()
    if not args.no_autosave:

        def _autosave():
            # A turn was recorded, undone, or retried: keep the file current.
            # Skip creating a file until there is something to save.
            if not state.transcript and not state.session_name:
                return
            from . import sessions

            if not state.session_name:
                state.session_name = sessions.default_session_name(args.gguf)
            try:
                sessions.save_session(_session_doc(state), state.session_name)
            except OSError:
                pass

        state.autosave = _autosave

    def _finish_load():
        # Join the background text load, then rebind the model-dependent
        # locals. No-op elsewhere.
        nonlocal model, config, tok, cache
        if not backend.join():
            return
        model, config, tok = backend.model, backend.config, backend.tok
        cache = backend.new_text_cache()
        _bind_model_state()

    editor = (
        "prompt_toolkit"
        if state.ptk_session is not None
        else "readline (pip install 'gmlx[chat]' for menu completion + suggestions)"
    )
    print(
        f"[chat] '/help' lists commands - '/exit' or Ctrl-D quits - "
        f"Esc cancels a reply - editor: {editor}"
    )
    if state.vlm:
        print(
            "[chat] multimodal: /image or /audio stage media - or just drag "
            "a file from Finder into the prompt"
        )
    if drafter is not None:
        kind = "assistant" if args.draft_gguf else "native-head"
        if vlm_mtp:
            print(
                f"[chat] MTP speculative decoding on text-only turns "
                f"({kind} drafter); image/audio turns use the VLM path"
            )
        else:
            print(f"[chat] MTP speculative decoding on ({kind} drafter)")
    if brain is not None:
        tools = ", ".join(brain.tools.names()) or "(none)"
        mem = (f" - memory: {brain.memory.count()} items"
               if brain.memory is not None else "")
        print(f"[chat] assistant mode: {model_request} via {base_url} - "
              f"tools: {tools}{mem}")
    first_turn = True
    vlm_msgs: list = []  # rendered messages (media markers pinned per turn)
    vlm_images: list = []  # media paths, in marker order across all turns
    vlm_audios: list = []

    def _reset_conversation():
        nonlocal cache, first_turn
        if state.vlm:
            vlm_msgs.clear(), vlm_images.clear(), vlm_audios.clear()
            if vlm_mtp:
                cache = backend.new_text_cache()  # drop the text-turn KV history too
        else:
            cache = backend.new_text_cache()
        if brain is not None:
            brain.reset()
        first_turn = True
        state.transcript = []
        state.ctx_used = None
        state.session_name = None
        state.session_created = None
        state.take("replay_messages")  # stale pre-reset history

    def _pop_last_turn(verb: str):
        """Rewind one exchange (KV rewind + VLM history + ledger); the /retry
        and /undo backend. Trims the cache to the turn checkpoint when every
        layer supports it; otherwise (recurrent-state hybrids, a wrapped
        rotating cache, turns restored from a saved session) rebuilds - fresh
        cache, and the remaining history re-prefills with the next message via
        the same deferred-replay path --resume uses. Returns the popped ledger
        entry, or None with a notice."""
        nonlocal cache, first_turn
        transcript = state.transcript
        if not transcript:
            print(f"[chat] nothing to {verb}")
            return None
        entry = transcript[-1]
        rebuilt = False
        cp = entry.get("cache_before")
        if cache is not None and (cp is None or _cache_tokens(cache) > cp):
            lm = getattr(model, "language_model", model)
            if cp is None or not _trim_to(cache, cp, lm=lm):
                cache = backend.new_text_cache()
                rebuilt = True
        m, i, a = entry.get("vlm_lens", (0, 0, 0))
        del vlm_msgs[m:], vlm_images[i:], vlm_audios[a:]
        if brain is not None:
            # Truncate to the turn's brain checkpoint: an assistant turn can
            # append several messages (tool rounds), not just user+reply.
            bb = entry.get("brain_before")
            if bb is not None:
                del brain.messages[bb:]
            else:                        # restored session: rebuild history
                brain.messages = [
                    m2 for e in transcript[:-1] for m2 in (
                        {"role": "user", "content": e["user"].get("content", "")},
                        {"role": "assistant",
                         "content": e["assistant"].get("content", "")})]
        first_turn = entry.get("first_turn_before", first_turn)
        transcript.pop()
        state.ctx_used = (cp or None) if not rebuilt else None
        if rebuilt:
            # The fresh cache invalidates every recorded checkpoint; later
            # rewinds rebuild again rather than mis-trimming.
            for e in transcript:
                e["cache_before"] = None
            state.take("replay_messages")  # stale pre-pop history
            if transcript:
                replay = []
                if state.system_prompt:
                    replay.append(
                        {"role": "system", "content": state.system_prompt}
                    )
                for e in transcript:
                    replay.append(
                        {"role": "user", "content": e["user"].get("content", "")}
                    )
                    replay.append(
                        {
                            "role": "assistant",
                            "content": e["assistant"].get("content", ""),
                        }
                    )
                if not state.vlm or vlm_mtp:
                    state.replay_messages = replay
                first_turn = False
                print(
                    "[chat] this model's cache can't rewind in place - rebuilt; "
                    "the earlier history re-prefills with your next message"
                )
            else:
                first_turn = True
        hook = state.autosave
        if hook is not None:
            hook()
        return entry

    def _apply_session(doc: dict, name: str) -> None:
        """Restore settings + transcript from a saved session. The KV replay is
        deferred: the full history is prepended to the next turn's messages and
        prefills through the existing chunked path in one go."""
        nonlocal first_turn
        _reset_conversation()
        _session_restore_settings(state, doc, name)
        replay, transcript, n_tok = _session_parse_messages(
            doc, state.system_prompt)
        state.transcript = transcript
        if state.vlm:
            _session_rebuild_vlm_history(
                transcript, state, model_type,
                vlm_msgs, vlm_images, vlm_audios,
            )
        if brain is not None:
            # The brain holds the history itself; nothing prefills locally.
            brain.messages = [m for e in transcript for m in (
                {"role": "user", "content": e["user"].get("content", "")},
                {"role": "assistant",
                 "content": e["assistant"].get("content", "")})]
        if transcript:
            first_turn = False
            if (not state.vlm or vlm_mtp) and brain is None:
                state.replay_messages = replay
        n = len(transcript)
        print(
            f"[chat] resumed '{name}': {n} turn{'s' if n != 1 else ''}"
            + (f", {_fmt_k(n_tok)} gen tok" if n_tok else "")
            + (" - the history prefills with your next message"
               if brain is None else "")
        )

    if args.resume is not None:
        try:
            _finish_load()
        except _ChatExit as e:
            return e.code
        from . import sessions

        ref = args.resume or sessions.latest_for_model(model_key)
        if not ref:
            print("[chat] no saved session for this model - starting fresh")
        else:
            try:
                doc, spath = sessions.load_session(ref)
            except (OSError, ValueError) as e:
                print(f"error: --resume: {e}", file=sys.stderr)
                return 2
            mp = (doc.get("model") or {}).get("path")
            if mp and mp != model_key:
                print(
                    f"error: --resume: that session was recorded with {mp}",
                    file=sys.stderr,
                )
                return 2
            _apply_session(doc, spath.stem)

    from .cli import _DIFFUSION_MAX_TOKENS, _UNCAPPED_MAX_TOKENS

    while True:
        parts = []
        if state.staged:
            parts.append(f"+{len(state.staged)}")
        if state.staged_images:
            parts.append(f"+{len(state.staged_images)}img")
        if state.staged_audio:
            parts.append(f"+{len(state.staged_audio)}aud")
        prompt_str = f"({' '.join(parts)}) >> " if parts else ">> "
        pending = state.take("pending_send")
        if pending is not None:
            line = pending
        else:
            try:
                line = _read_input(state, prompt_str)
            except EOFError:
                print()  # Ctrl-D: exit
                return 0
        word = line.strip()
        if word:
            # Before the /command branch: a file dragged from Finder pastes
            # as an absolute path, which also starts with "/".
            dropped = _detect_dropped_media(line)
            if dropped is not None:
                _stage_media(state, *dropped)
                continue
        if word.startswith("/"):
            verb = _handle_slash(line, state)
            if verb == "exit":
                return 0
            if verb is not None:
                # Every non-exit verb rebuilds or trims the KV cache, so it
                # needs the model; plain state commands (None verb) run
                # without joining the background load.
                try:
                    _finish_load()
                except _ChatExit as e:
                    return e.code
            if verb == "reset":
                _reset_conversation()
            elif verb == "undo":
                if _pop_last_turn("undo") is not None:
                    print("[chat] last exchange removed")
            elif verb == "retry":
                if (
                    state.staged
                    or state.staged_images
                    or state.staged_audio
                ):
                    print(
                        "[chat] staged items pending - send them or /drop "
                        "first, then /retry"
                    )
                else:
                    entry = _pop_last_turn("retry")
                    if entry is not None:
                        u = entry["user"]
                        if u.get("images"):
                            state.staged_images = list(u["images"])
                        if u.get("audios"):
                            state.staged_audio = list(u["audios"])
                        state.pending_send = u["content"]
                        print("[chat] regenerating...")
            elif isinstance(verb, tuple) and verb[0] == "load-session":
                from . import sessions

                try:
                    doc, spath = sessions.load_session(verb[1])
                except (OSError, ValueError) as e:
                    print(f"[chat] /load-session: {e}")
                else:
                    mp = (doc.get("model") or {}).get("path")
                    if mp and mp != model_key:
                        print(
                            f"[chat] that session was recorded with {mp} - "
                            "start gmlx chat on that file to resume it"
                        )
                    else:
                        _apply_session(doc, spath.stem)
            continue
        if not word and not (
            state.staged
            or state.staged_images
            or state.staged_audio
        ):
            continue

        try:
            _finish_load()          # taking a turn: join the background load
        except _ChatExit as e:
            return e.code
        _begin_turn(
            state,
            cache=cache,
            first_turn=first_turn,
            vlm_lens=(len(vlm_msgs), len(vlm_images), len(vlm_audios)),
        )
        s = state.sampling
        rep = s["repetition_penalty"]
        # 0 = uncapped ("until the model stops"): the local generation loops
        # still need a finite bound - same sentinel resolution as `gmlx run`.
        eff_max = s["max_tokens"] or _UNCAPPED_MAX_TOKENS

        if brain is not None:
            # Server-backed assistant turn: the brain owns history and the
            # tool loop; /system and the sampling /commands apply per turn.
            state.turn_checkpoint["brain_before"] = len(brain.messages)
            brain.system = state.system_prompt
            # Uncapped omits max_tokens from the request (None) rather than
            # sending the sentinel: an explicit 1<<25 trips the server's
            # MAX_KV_SIZE context-budget check; omitted, the server default
            # (MLX_VLM_MAX_TOKENS) applies.
            brain.max_tokens = s["max_tokens"] or None
            _sync_assistant_extra(state)
            user_text = _compose_user_content(state, line)
            reply, canceled = _assistant_reply(brain, user_text, state)
            first_turn = False
            _end_turn(state, reply, canceled)
            continue

        if state.vlm:
            staged_imgs = state.staged_images
            staged_auds = state.staged_audio
            if (
                vlm_mtp
                and not vlm_images
                and not vlm_audios
                and not staged_imgs
                and not staged_auds
            ):
                # Text-only turn while the conversation holds no media: run the MTP
                # engine over the persistent text cache (the incremental templating +
                # cache append of the plain MTP path). Also record the turn in
                # vlm_msgs so the first image turn can re-prefill the full history on
                # the VLM path; once any media enters, the session stays on that path.
                from .generation import stream_generate_speculative

                messages = list(state.take("replay_messages") or [])
                if first_turn and state.system_prompt:
                    sys_msg = {"role": "system", "content": state.system_prompt}
                    messages.append(sys_msg)
                    vlm_msgs.append(sys_msg)  # same shape the VLM branch records
                user_text = _compose_user_content(state, line)
                messages.append({"role": "user", "content": user_text})
                vlm_msgs.append(_vlm_message(model_type, user_text, "user", 0, 0))
                first_turn = False
                prompt_text = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    **template_kwargs,
                )
                prompt = tok.encode(prompt_text, add_special_tokens=False)
                reply, _canceled = _stream_reply(
                    stream_generate_speculative(
                        model,
                        drafter,
                        tok,
                        prompt,
                        prompt_cache=cache,
                        max_tokens=eff_max,
                        temp=s["temp"],
                        top_p=s["top_p"],
                        top_k=s["top_k"],
                        min_p=s["min_p"],
                        draft_block_size=args.draft_block_size,
                    ),
                    state,
                    stops=args.stop,
                    start_in_thinking=_opens_thinking(prompt_text),
                    drafter=drafter,
                )
                if reply:
                    vlm_msgs.append(_vlm_message(model_type, reply, "assistant"))
                _end_turn(state, reply, _canceled, cache=cache)
                continue
            from mlx_vlm.generate import stream_generate as vlm_stream
            from mlx_vlm.prompt_utils import get_chat_template

            imgs = state.take_list("staged_images")
            auds = state.take_list("staged_audio")
            if first_turn and state.system_prompt:
                vlm_msgs.append({"role": "system", "content": state.system_prompt})
            vlm_msgs.append(
                _vlm_message(
                    model_type,
                    _compose_user_content(state, line),
                    "user",
                    len(imgs),
                    len(auds),
                )
            )
            vlm_images.extend(imgs)
            vlm_audios.extend(auds)
            first_turn = False
            prompt = get_chat_template(
                processor, vlm_msgs, add_generation_prompt=True, **template_kwargs
            )
            # No cross-turn KV cache on the VLM path yet: each turn
            # re-prefills the whole conversation (and re-encodes media).
            reply, _canceled = _stream_reply(
                vlm_stream(
                    model,
                    processor,
                    prompt,
                    image=list(vlm_images) or None,
                    audio=list(vlm_audios) or None,
                    max_tokens=eff_max,
                    temperature=s["temp"],
                    top_p=s["top_p"],
                    top_k=s["top_k"],
                    min_p=s["min_p"],
                    repetition_penalty=None if rep in (0.0, 1.0) else rep,
                    repetition_context_size=s["repetition_context_size"],
                    presence_penalty=s["presence_penalty"] or None,
                    frequency_penalty=s["frequency_penalty"] or None,
                    logit_bias=logit_bias,
                    resize_shape=resize_shape,
                    thinking_budget=state.thinking_budget,
                    **kv_kwargs,
                ),
                state,
                stops=args.stop,
                start_in_thinking=_opens_thinking(prompt),
            )
            if reply:
                vlm_msgs.append(_vlm_message(model_type, reply, "assistant"))
            _end_turn(state, reply, _canceled)
            continue

        if drafter is not None:
            # MTP speculative decoding over the persistent cache (the same per-turn
            # templating + cache append as the plain text path below). Sampling is
            # temp/top-p/top-k/min-p only - mlx-vlm's MTP walk exposes no penalty/
            # bias/stop hooks, so the REPL's other sampling /commands don't apply.
            from .generation import stream_generate_speculative

            messages = list(state.take("replay_messages") or [])
            if first_turn and state.system_prompt:
                messages.append({"role": "system", "content": state.system_prompt})
            messages.append(
                {"role": "user", "content": _compose_user_content(state, line)}
            )
            prompt_text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                **template_kwargs,
            )
            prompt = tok.encode(prompt_text, add_special_tokens=False)
            first_turn = False
            reply, canceled = _stream_reply(
                stream_generate_speculative(
                    model,
                    drafter,
                    tok,
                    prompt,
                    prompt_cache=cache,
                    max_tokens=eff_max,
                    temp=s["temp"],
                    top_p=s["top_p"],
                    top_k=s["top_k"],
                    min_p=s["min_p"],
                    draft_block_size=args.draft_block_size,
                ),
                state,
                stops=args.stop,
                start_in_thinking=_opens_thinking(prompt_text),
                drafter=drafter,
            )
            _end_turn(state, reply, canceled, cache=cache)
            continue

        from .diffusion import is_diffusion_model

        if is_diffusion_model(model):
            # Non-autoregressive: denoise a canvas via mlx-vlm. No cross-turn KV
            # cache, and the AR sampler/stop controls don't apply.
            from .diffusion import stream as diffusion_stream

            messages = list(state.take("replay_messages") or [])
            if first_turn and state.system_prompt:
                messages.append({"role": "system", "content": state.system_prompt})
            messages.append(
                {"role": "user", "content": _compose_user_content(state, line)}
            )
            prompt_text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                **template_kwargs,
            )
            prompt = tok.encode(prompt_text, add_special_tokens=False)
            first_turn = False
            reply, canceled = _stream_reply(
                # Bounded canvas fallback shared with `gmlx run` - the
                # denoising canvas is allocated at max_tokens, so "until the
                # model stops" has no meaning here.
                diffusion_stream(model, tok, prompt,
                                 max_tokens=s["max_tokens"]
                                 or _DIFFUSION_MAX_TOKENS),
                state,
                stops=args.stop,
                start_in_thinking=_opens_thinking(prompt_text),
            )
            _end_turn(state, reply, canceled)
            continue

        from mlx_lm.generate import stream_generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        messages = list(state.take("replay_messages") or [])
        if first_turn and state.system_prompt:
            messages.append({"role": "system", "content": state.system_prompt})
        messages.append({"role": "user", "content": _compose_user_content(state, line)})
        if args.no_chat_template:
            # Base / non-instruct model: no template, just the raw turn(s).
            prompt = "\n".join(
                m["content"] for m in messages if isinstance(m.get("content"), str)
            )
            prompt_text = prompt
        else:
            # Render once, encode the render: token-identical to a tokenize=True
            # template call (which is render + encode(add_special_tokens=False)
            # internally), and the string is reused for thinking-budget seeding
            # + reasoning detection.
            prompt_text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                **template_kwargs,
            )
            prompt = tok.encode(prompt_text, add_special_tokens=False)

        xtc_kwargs = {}
        if s["xtc_probability"] > 0:
            xtc_kwargs = {
                "xtc_probability": s["xtc_probability"],
                "xtc_threshold": s["xtc_threshold"],
                "xtc_special_tokens": tok.encode("\n") + list(tok.eos_token_ids),
            }
        sampler = make_sampler(
            temp=s["temp"],
            top_p=s["top_p"],
            top_k=s["top_k"],
            min_p=s["min_p"],
            **xtc_kwargs,
        )
        from .tokenizer import merge_suppressed_tokens
        logit_bias = merge_suppressed_tokens(logit_bias, tok)
        logits_processors = make_logits_processors(
            logit_bias=logit_bias,
            repetition_penalty=None if rep in (0.0, 1.0) else rep,
            repetition_context_size=s["repetition_context_size"],
            presence_penalty=s["presence_penalty"] or None,
            frequency_penalty=s["frequency_penalty"] or None,
        )
        # Thinking-token cap (fresh per turn - the processor is stateful). Honored
        # regardless of enable_thinking; seed the in-thinking state from whether
        # the rendered prompt actually opens a <think> block (see generation.generate).
        if state.thinking_budget is not None:
            from .thinking_budget import (
                make_thinking_budget_processor,
                prompt_opens_thinking,
            )

            tbp = make_thinking_budget_processor(
                tok,
                state.thinking_budget,
                start_in_thinking=prompt_opens_thinking(prompt_text),
            )
            if tbp is not None:
                logits_processors = list(logits_processors) + [tbp]
        first_turn = False  # the system prompt is in the cache either way
        reply, canceled = _stream_reply(
            stream_generate(
                model,
                tok,
                prompt,
                max_tokens=eff_max,
                sampler=sampler,
                logits_processors=logits_processors,
                prompt_cache=cache,
                **kv_kwargs,
            ),
            state,
            stops=args.stop,
            start_in_thinking=_opens_thinking(prompt_text),
        )
        _end_turn(state, reply, canceled, cache=cache)
