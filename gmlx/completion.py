"""Shell completion: ``gmlx completion zsh`` emits a script, and the hidden
``gmlx __complete`` computes candidates live.

The design is fully dynamic: the emitted zsh script is a thin shim that, on every
Tab, calls ``gmlx __complete <words...>`` and feeds the result to ``compadd``.
All of the logic lives here in Python - so completions always match the installed
version (flags are read from each verb's own ``--help``) and the running config
(model ids/aliases come from the resolved server config). Nothing to regenerate
after an upgrade.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import io
import os
import re
import sys

# Verb one-liners for first-word completion. Mirrors the umbrella help; a test
# asserts every dispatchable verb has an entry so this can't silently drift.
_VERB_DESC = {
    "run": "generate, benchmark, or inspect a GGUF",
    "chat": "interactive multi-turn chat REPL on a GGUF",
    "talk": "voice chat with a served model (wake word, STT, TTS)",
    "serve": "run the batched multi-model OpenAI/Anthropic server",
    "init": "scaffold a starter server config",
    "sync-models": "reconcile a config's models with disk / the hf cache",
    "launch": "point a coding harness at a running server",
    "stop": "stop a backgrounded server",
    "restart": "restart a backgrounded server",
    "status": "show whether a backgrounded server is running",
    "logs": "print or follow a backgrounded server's log",
    "service": "install/uninstall a launchd LaunchAgent",
    "validate": "check a local or remote GGUF will load",
    "pull": "validate a remote GGUF, then download it",
    "rm": "delete a model's files and config entry",
    "list": "list the models your server config defines",
    "ps": "show the models resident in a running server",
    "profiles": "show per-family sampling defaults + @intents",
    "doctor": "check the runtime, config, models, and services",
    "train": "finetune a LoRA adapter on a GGUF base",
    "completion": "print a shell completion script",
}

# Verbs whose first positional is a model (config id/alias or a path on disk).
_MODEL_POSITIONAL_VERBS = frozenset({"run", "chat", "serve", "rm"})
# Verbs whose first positional is a path / remote ref (no config lookup).
_FILE_POSITIONAL_VERBS = frozenset({"validate", "pull"})
_SERVICE_ACTIONS = ("install", "uninstall", "status")


def _canon(verb: str) -> str:
    """Resolve a verb alias (``ls`` -> ``list``) to its canonical form."""
    from .cli import _VERB_ALIASES

    return _VERB_ALIASES.get(verb, verb)


def _known_verbs() -> set[str]:
    from .cli import _VERBS

    return set(_VERBS)


# Flag introspection (drift-free): scrape each verb's own argparse --help.

@functools.lru_cache(maxsize=None)
def _verb_options(verb: str) -> tuple[tuple[str, str, str], ...]:
    """Return ``(option, metavar, help)`` per individual option string for a verb,
    read from its ``--help`` output so the set always matches the real parser.

    ``service`` carries no standard options block (it dispatches sub-actions), so it
    borrows ``serve``'s flags - that is the option set its ``install`` action takes.
    """
    if verb == "service":
        verb = "serve"
    text = _capture_help(verb)
    out: list[list[str]] = []
    pending: list[int] = []                  # out indices awaiting wrapped help
    for line in text.splitlines():
        # Option-defining lines are indented exactly two spaces and start with a dash;
        # usage continuations align far deeper and positionals don't start with a dash.
        if re.match(r"^ {2}-", line):
            head = re.match(r"^ {2}(-\S.*)$", line).group(1)
            parts = re.split(r"\s{2,}", head, maxsplit=1)
            inline = parts[1].strip() if len(parts) > 1 else ""
            pending = []
            for tok in parts[0].split(", "):
                tok = tok.strip()
                if not tok.startswith("-"):
                    continue
                bits = tok.split(None, 1)
                out.append([bits[0], bits[1] if len(bits) > 1 else "", inline])
                if not inline:               # help wrapped onto the next line
                    pending.append(len(out) - 1)
        elif pending and re.match(r"^ {3,}\S", line):
            for idx in pending:
                out[idx][2] = line.strip()
            pending = []
        else:
            pending = []
    return tuple((o[0], o[1], o[2]) for o in out)


def _capture_help(verb: str) -> str:
    """Run ``gmlx <verb> --help`` in-process, capturing stdout (argparse exits
    via SystemExit, which we swallow). Returns ``""`` on any trouble - completion
    must never raise."""
    from .cli import umbrella_main

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            umbrella_main([verb, "--help"])
    except SystemExit:
        pass
    except Exception:  # noqa: BLE001 - a broken verb help must not break completion
        return ""
    return buf.getvalue()


def _is_pathish(metavar: str) -> bool:
    mv = (metavar or "").upper()
    return any(k in mv for k in ("PATH", "FILE", "DIR"))


def _option_for(verb: str, flag: str) -> tuple[str, str, str] | None:
    for opt, metavar, helptext in _verb_options(verb):
        if opt == flag:
            return (opt, metavar, helptext)
    return None


# Dynamic value sources (config model ids/aliases, harnesses).

def _config_path_from(words: list[str]) -> str | None:
    """The ``--config FILE`` value present in ``words``, else the first existing
    default config path."""
    for i, w in enumerate(words):
        if w == "--config" and i + 1 < len(words):
            return words[i + 1]
        if w.startswith("--config="):
            return w.split("=", 1)[1]
    from . import config as cfgmod

    return next((str(p) for p in cfgmod.default_config_paths() if p.exists()), None)


def _model_candidates(words: list[str]) -> list[str]:
    """``id<TAB>desc`` lines for every model, alias, and assistant in the
    resolved config."""
    path = _config_path_from(words)
    if not path or not os.path.exists(os.path.expanduser(path)):
        return []
    from . import config as cfgmod

    try:
        cfg = cfgmod.load_config(os.path.expanduser(path))
    except Exception:  # noqa: BLE001 - a malformed config just yields no model names
        return []
    out: list[str] = []
    for mid, m in cfg.models.items():
        base = os.path.basename(getattr(m, "path", "") or "")
        out.append(f"{mid}\t{base}" if base else mid)
    for name, target in cfg.aliases.items():
        out.append(f"{name}\talias -> {target}")
    for name, alias in cfg.assistants.items():
        out.append(f"{name}\tassistant -> {alias.model}")
    return out


def _harness_candidates() -> list[str]:
    from .launch import _HARNESSES

    out = [f"{h}\tcoding harness" for h in sorted(_HARNESSES)]
    out.append("menubar\tmacOS status-bar monitor")
    return out


def _running_servers() -> list[dict]:
    """Runfile dicts for backgrounded servers (newest first), or ``[]``."""
    try:
        from . import lifecycle

        return list(reversed(lifecycle.list_runs()))
    except Exception:  # noqa: BLE001 - a missing/odd state dir just yields nothing
        return []


def _endpoint_candidates(metavar: str, flag: str) -> list[str]:
    """Live host/port/url values for an endpoint-valued flag, read from the
    runfiles of servers started with ``serve`` (background). ``metavar`` selects
    which field (``PORT`` / ``HOST`` / ``...URL...``); ``flag`` distinguishes a base
    URL (``--base-url`` wants the ``/v1`` form). Empty when nothing is running, so
    the value stays free-form."""
    mv = (metavar or "").upper()
    out: list[str] = []
    seen: set[str] = set()
    for r in _running_servers():
        host = r.get("host") or "127.0.0.1"
        port = r.get("port")
        url = r.get("url") or f"http://{host}:{port}"
        managed = r.get("managed_by") or "detach"
        if mv == "PORT":
            val, desc = str(port or ""), f"{host} ({managed})"
        elif mv == "HOST":
            val, desc = str(host), "running server"
        elif "URL" in mv:
            val = f"{url}/v1" if flag == "--base-url" else url
            desc = f"running server ({managed})"
        else:
            return []
        if not val or val in seen:
            continue
        seen.add(val)
        out.append(f"{val}\t{desc}")
    return out


# Positional bookkeeping: how many positionals are already filled, so we only offer
# a positional's values while the user is actually on it.

def _positionals_filled(verb: str, after_verb: list[str]) -> int:
    value_flags = {opt for opt, mv, _ in _verb_options(verb) if mv}
    count = 0
    i = 0
    n = len(after_verb)
    while i < n:
        t = after_verb[i]
        if t.startswith("-"):
            if "=" in t:
                i += 1
            elif t in value_flags:
                i += 2  # this flag consumes the next token as its value
            else:
                i += 1
        else:
            count += 1
            i += 1
    return count


def _positional_candidates(verb: str, after_verb: list[str]) -> list[str]:
    """Candidates for the positional the user is currently typing (the trailing,
    in-progress word is *not* in ``after_verb``)."""
    filled = _positionals_filled(verb, after_verb)
    if verb in _MODEL_POSITIONAL_VERBS:
        if filled:
            return []                       # the model slot is already taken
        return ["::files", *_model_candidates(after_verb)]
    if verb == "talk":
        # A served model id, never a path on disk - no ::files fallback.
        return [] if filled else _model_candidates(after_verb)
    if verb in _FILE_POSITIONAL_VERBS:
        return [] if filled else ["::files"]
    if verb == "launch":
        return [] if filled else _harness_candidates()
    if verb == "service":
        return [] if filled else [f"{a}\tlaunchd action" for a in _SERVICE_ACTIONS]
    return []


def _complete(argv: list[str]) -> list[str]:
    """Compute candidate lines for the current command line. ``argv`` is the words
    after the program name; its last element is the (possibly empty) word being
    completed."""
    args = list(argv) if argv else [""]
    cur = args[-1]
    pre = args[:-1]

    if not pre:                              # completing the verb itself
        verbs = sorted(_known_verbs() | {"ls"})
        out = []
        for v in verbs:
            desc = _VERB_DESC.get(_canon(v), "")
            out.append(f"{v}\t{desc}" if desc else v)
        return out

    verb = _canon(pre[0])
    if verb not in _known_verbs():
        return []

    if cur.startswith("-"):                  # completing a flag
        return [f"{opt}\t{h}" if h else opt for opt, _mv, h in _verb_options(verb)]

    prev = pre[-1]
    if prev.startswith("-") and "=" not in prev:
        opt = _option_for(verb, prev)
        if opt is not None and opt[1]:       # the previous flag wants a value
            if _is_pathish(opt[1]):
                return ["::files"]
            # An endpoint flag (--host/--port/--url/--base-url) completes from the
            # servers currently running; any other value flag (a temperature, a
            # token count) has nothing to enumerate.
            return _endpoint_candidates(opt[1], opt[0])

    return _positional_candidates(verb, pre[1:])


def cmd_complete(argv: list[str]) -> int:
    """Hidden ``gmlx __complete`` entry point - prints one candidate per line
    (``value`` or ``value\\tdescription``; a leading ``::files`` defers to the
    shell's filename completion). Always exits 0 - a completion path must never
    surface an error to the shell."""
    try:
        for line in _complete(list(argv)):
            print(line)
    except Exception:  # noqa: BLE001 - never let completion fail loudly
        pass
    return 0


# `completion <shell>` - emit a completion script.

_ZSH_SCRIPT = r"""#compdef gmlx
# gmlx zsh completion.
#
# Install (either works):
#   - eval - add to ~/.zshrc:
#       eval "$(gmlx completion zsh)"
#   - fpath - write a function file:
#       mkdir -p ~/.zfunc && gmlx completion zsh > ~/.zfunc/_gmlx
#       # then, before `compinit` in ~/.zshrc:  fpath+=(~/.zfunc)
#
# Candidates are computed live by `gmlx __complete`, so they always match the
# installed version and your server config's models.

_gmlx() {
  local -a _args
  _args=("${(@)words[2,$CURRENT]}")
  (( ${#_args} )) || _args=("")

  local _out
  _out="$(gmlx __complete "${_args[@]}" 2>/dev/null)"

  local -a _lines
  _lines=("${(@f)_out}")

  if [[ ${_lines[1]} == "::files" ]]; then
    _files
    _lines=("${_lines[@]:1}")
  fi

  local -a _vals _disp
  local _l _v _d
  for _l in "${_lines[@]}"; do
    [[ -z $_l ]] && continue
    _v=${_l%%$'\t'*}
    _d=${_l#*$'\t'}
    _vals+=("$_v")
    if [[ -n $_d && $_d != $_v ]]; then
      _disp+=("${_v}  --  ${_d}")
    else
      _disp+=("$_v")
    fi
  done
  (( ${#_vals} )) && compadd -d _disp -a _vals
}

if [[ $funcstack[1] == _gmlx ]]; then
  _gmlx "$@"
else
  # A stock macOS zsh has no compinit in ~/.zshrc; without it compdef doesn't
  # exist and the eval-install one-liner errors out. -i: skip "insecure"
  # (group-writable) fpath dirs instead of prompting y/n at every shell start
  # - a common condition on Homebrew Macs (/usr/local/share/zsh*).
  (( $+functions[compdef] )) || { autoload -Uz compinit && compinit -i }
  compdef _gmlx gmlx
fi
"""

_BASH_SCRIPT = r"""# gmlx bash completion.
#
# Install (either works):
#   - eval - add to ~/.bashrc:
#       eval "$(gmlx completion bash)"
#   - file - drop it where bash-completion looks (needs the bash-completion pkg):
#       gmlx completion bash > ~/.local/share/bash-completion/completions/gmlx
#
# Candidates are computed live by `gmlx __complete`, so they always match the
# installed version and your server config's models.

_gmlx() {
  local cur="${COMP_WORDS[COMP_CWORD]}"
  local -a _args=("${COMP_WORDS[@]:1:COMP_CWORD}")

  local _out
  _out="$(gmlx __complete "${_args[@]}" 2>/dev/null)"

  local -a _cands=()
  local _line _files=0
  while IFS= read -r _line; do
    [[ -z $_line ]] && continue
    if [[ $_line == "::files" ]]; then
      _files=1
      continue
    fi
    _cands+=("${_line%%$'\t'*}")
  done <<< "$_out"

  COMPREPLY=()
  if (( ${#_cands[@]} )); then
    local IFS=$'\n'
    COMPREPLY+=( $(compgen -W "${_cands[*]}" -- "$cur") )
  fi
  if (( _files )); then
    COMPREPLY+=( $(compgen -f -- "$cur") )
    compopt -o filenames 2>/dev/null
  fi
}
complete -F _gmlx gmlx
"""

_FISH_SCRIPT = r"""# gmlx fish completion.
#
# Install:
#   - eval - add to ~/.config/fish/config.fish:
#       gmlx completion fish | source
#   - file - drop it where fish autoloads completions:
#       gmlx completion fish > ~/.config/fish/completions/gmlx.fish
#
# Candidates are computed live by `gmlx __complete`, so they always match the
# installed version and your server config's models.

function __gmlx_complete
    set -l tokens (commandline -opc)
    set -l cur (commandline -ct)
    # Drop the program name; pass the current (possibly partial) token last so
    # `__complete` sees it as the word being completed. `commandline -opc` may or
    # may not include that partial token depending on the cursor, so strip a
    # trailing copy before re-appending it explicitly.
    set -l args $tokens[2..-1]
    if test -n "$cur"; and set -q args[-1]; and test "$args[-1]" = "$cur"
        set -e args[-1]
    end
    for line in (gmlx __complete $args "$cur" 2>/dev/null)
        if test "$line" = '::files'
            __fish_complete_path "$cur"
        else
            printf '%s\n' "$line"
        end
    end
end

complete -c gmlx -f -a '(__gmlx_complete)'
"""

_SHELLS = {"zsh": _ZSH_SCRIPT, "bash": _BASH_SCRIPT, "fish": _FISH_SCRIPT}


def cmd_completion(argv: list[str], prog: str = "gmlx completion") -> int:
    ap = argparse.ArgumentParser(
        prog=prog,
        description="Print a shell completion script (zsh, bash, or fish).",
        epilog="zsh: `eval \"$(gmlx completion zsh)\"` in ~/.zshrc (or write it "
               "onto your fpath as ~/.zfunc/_gmlx). "
               "bash: `eval \"$(gmlx completion bash)\"` in ~/.bashrc. "
               "fish: `gmlx completion fish | source` in config.fish (or write "
               "it to ~/.config/fish/completions/gmlx.fish).",
    )
    ap.add_argument("shell", nargs="?", choices=sorted(_SHELLS),
                    help="Shell to emit a completion script for.")
    a = ap.parse_args(argv)
    if a.shell is None:
        ap.print_help()
        return 0
    # Script only, on stdout. `--help` carries the install instructions;
    # this command runs unattended via `eval` in shell rc files on every
    # login, so it must not print anything extra even to stderr.
    sys.stdout.write(_SHELLS[a.shell])
    return 0
