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
    "launch": "run a client against a running server",
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
    "systemone": "answer a structured-decision request",
    "profiles": "show per-family sampling defaults + @intents",
    "doctor": "check the runtime, config, models, and services",
    "train": "finetune a LoRA adapter on a GGUF base",
    "distill": "offline distillation: gen, filter, cache, align, train, eval, census",
    "completion": "print a shell completion script",
}

# Verbs whose first positional is a model (config id/alias or a path on disk).
_MODEL_POSITIONAL_VERBS = frozenset({"run", "chat", "serve", "rm"})
# Verbs whose first positional is a path / remote ref (no config lookup).
_FILE_POSITIONAL_VERBS = frozenset({"validate", "pull", "systemone"})
_SERVICE_ACTIONS = ("install", "uninstall", "status")
_DISTILL_ACTIONS = ("gen", "filter", "cache", "align", "train", "eval", "census")


# A candidate value is typed into the user's command line, and some come from
# files that another program could write: a config file, a runfile, a folder
# a container session created. Only these characters pass. A model id in a
# config can also hold a space or parentheses, which every completion script
# quotes, so only a runfile value and a private-home name get the stricter set.
_SAFE_NAME = re.compile(r"[A-Za-z0-9._-]+")
_SAFE_VALUE = re.compile(r"[A-Za-z0-9._:/@+=,%~{}\[\]-]+")
_SAFE_SHOWN = re.compile(r"[A-Za-z0-9._:/@+=,%~{}\[\]() -]+")
_UNPRINTABLE = re.compile(r"[\x00-\x1f\x7f]")


def _safe_lines(lines: list[str]) -> list[str]:
    """Drop candidates whose value holds a character outside the safe set,
    and blank out control characters in descriptions."""
    out = []
    for line in lines:
        if line == "::files":
            out.append(line)
            continue
        value, tab, desc = line.partition("\t")
        if not _SAFE_SHOWN.fullmatch(value):
            continue
        out.append(value + tab + _UNPRINTABLE.sub(" ", desc) if tab else value)
    return out


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
    if verb.startswith("distill ") and not text.strip():
        return ()
    out: list[list[str]] = []
    current: list[int] = []                  # out indices whose help is still wrapping
    for line in text.splitlines():
        # Option-defining lines are indented exactly two spaces and start with a dash;
        # usage continuations align far deeper and positionals don't start with a dash.
        if re.match(r"^ {2}-", line):
            head = re.match(r"^ {2}(-\S.*)$", line).group(1)
            parts = re.split(r"\s{2,}", head, maxsplit=1)
            inline = parts[1].strip() if len(parts) > 1 else ""
            current = []
            for tok in parts[0].split(", "):
                tok = tok.strip()
                if not tok.startswith("-"):
                    continue
                bits = tok.split(None, 1)
                out.append([bits[0], bits[1] if len(bits) > 1 else "", inline])
                current.append(len(out) - 1)
        elif current and re.match(r"^ {3,}\S", line):
            for idx in current:
                out[idx][2] = f"{out[idx][2]} {line.strip()}".strip()
        else:
            current = []
    return tuple((o[0], o[1], _first_sentence(o[2])) for o in out)


def _first_sentence(text: str) -> str:
    """The help up to its first full stop, so a description never ends
    where argparse wrapped the line."""
    m = re.search(r"(?<!\be\.g)(?<!\bi\.e)\.(?=\s)", text)
    return text[:m.end()] if m else text


def _capture_help(verb: str) -> str:
    """Run ``gmlx <verb> --help`` in-process, capturing stdout (argparse exits
    via SystemExit, which we swallow). Verbs with a condensed default help
    (run/chat) are scraped via ``--help-all`` so completion still offers the
    full flag surface. Returns ``""`` on any trouble - completion must never
    raise."""
    from .cli import umbrella_main

    flag = "--help-all" if verb in ("run", "chat") else "--help"
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            umbrella_main([*verb.split(), flag])
    except SystemExit:
        pass
    except Exception:  # noqa: BLE001 - a broken verb help must not break completion
        return ""
    return buf.getvalue()


def _choice_values(metavar: str) -> list[str]:
    """The values of a ``{a,b,c}`` choices metavar, else ``[]``."""
    m = re.fullmatch(r"\{([^{}]+)\}", metavar or "")
    if not m:
        return []
    return [v.strip() for v in m.group(1).split(",") if v.strip()]


def _named_value_candidates(flag: str, words: list[str] = ()) -> list[str]:
    """Value candidates for flags whose metavar hides an enumerable set
    (themes, sampling profiles)."""
    try:
        if flag == "--theme":
            from gmlx.tui.theme import list_themes

            return [f"{t}\tcolor theme" for t in list_themes()]
        if flag == "--profile":
            from gmlx.gen.profiles import builtin_intents

            return [f"{i}\tbuilt-in intent" for i in sorted(builtin_intents())]
        if flag == "--dsh-profile":
            return _dsh_profile_candidates(_container_launch(words), words)
    except Exception:  # noqa: BLE001 - value candidates are best-effort
        return []
    return []


def _is_pathish(metavar: str) -> bool:
    mv = (metavar or "").upper()
    return any(k in mv for k in ("PATH", "FILE", "DIR", "GGUF"))


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
    import gmlx.config as cfgmod

    # A completion never prints the ./gmlx.yaml line, which would land in
    # the middle of the command line being typed.
    return next((str(p) for p in cfgmod.default_config_paths(note_local=False)
                 if p.exists()), None)


def _model_candidates(words: list[str]) -> list[str]:
    """``id<TAB>desc`` lines for every model, alias, and assistant in the
    resolved config."""
    path = _config_path_from(words)
    if not path or not os.path.exists(os.path.expanduser(path)):
        return []
    import gmlx.config as cfgmod

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


# Launch targets that are not coding agents, grouped as launch.py groups them.
_HARNESS_KINDS = {
    "hermes": "agent runtime", "goose": "agent runtime",
    "aichat": "chat TUI", "elia": "chat TUI",
    "open-webui": "web app", "dsh": "web app",
}


def _harness_candidates() -> list[str]:
    from .launch import _HARNESSES

    out = [f"{h}\t{_HARNESS_KINDS.get(h, 'coding agent')}"
           for h in sorted(_HARNESSES)]
    out.append("menubar\tmacOS status-bar monitor")
    return out


def _container_launch(words: list[str]) -> bool:
    """Whether a ``gmlx launch dsh`` command line runs in container mode, from
    its flags and then the user-level config. A container-only flag such as
    ``--shell`` implies container mode, as it does for launch itself."""
    from .launch_container import CONTAINER_FLAGS

    words = words[:words.index("--")] if "--" in words else list(words)
    if "--no-container" in words:
        return False
    if "--container" in words:
        return True
    implied = {*CONTAINER_FLAGS.values(), "--no-mount-cwd"}
    if any(w.split("=", 1)[0] in implied for w in words):
        return True
    from gmlx.config import load_launch_settings

    return bool(load_launch_settings(note_local=False).container.for_client("dsh").enabled)


def _dsh_project(words: list[str]) -> str:
    """The project whose private home a dsh launch with a profile of its own
    uses, keyed as launch keys it from the same flags and config: the
    current folder's when the launch shares it, the folder of a --mount or
    mounts: entry that holds it, else the default one. When the launch
    joins the running session of another project, as run_container finds
    it, that project. A launch that would stop, such as one from a folder
    launch never shares, gets the default one."""
    from types import SimpleNamespace

    from gmlx.config import ConfigError, load_launch_settings
    from gmlx.container import settings

    from .launch_container import _session_key

    words = words[:words.index("--")] if "--" in words else list(words)
    mount_cwd, mounts = None, []
    for i, word in enumerate(words):
        if word in ("--mount-cwd", "--no-mount-cwd"):
            mount_cwd = word == "--mount-cwd"
        elif word == "--mount" and i + 1 < len(words):
            mounts.append(words[i + 1])
        elif word.startswith("--mount="):
            mounts.append(word.split("=", 1)[1])
    a = SimpleNamespace(harness="dsh", mount_cwd=mount_cwd, mount=mounts)
    try:
        cfg = load_launch_settings(note_local=False).container.for_client("dsh")
        project, folder = _session_key(a, cfg)
        return _dsh_joined(a, project, folder)
    except (OSError, settings.SettingsError, ConfigError):
        return settings.PROJECT_DEFAULT


def _dsh_joined(a, project: str, folder: str | None) -> str:
    """The project of the dsh session that a launch keyed to ``project``
    joins: its own session first, else the session of another project that
    holds the folder that the launch can join from. Completion runs no
    container query, so a session counts while the launch that its record
    names lives."""
    from gmlx.container import session

    from .launch_container import _holding_sessions, _join_folder

    def lives(key: str, record: dict | None) -> bool:
        return record is not None and session.session_state("dsh", key, record, []) is not None

    here = _join_folder(a, folder)
    if here is None or lives(project, session.read_record("dsh", project)):
        return project
    for other, record in _holding_sessions("dsh", project, here):
        if lives(other, record):
            return other
    return project


def _dsh_profile_candidates(container: bool = False,
                            words: list[str] | None = None) -> list[str]:
    """dsh's shipped profiles plus the profiles under $DSH_HOME, or in
    container mode those in the private home of the current folder's
    project."""
    from .launch import _DSH_PROFILE, _DSH_SHIPPED, _DSH_STDIO, _dsh_home

    names = {n: "shipped dsh profile" for n in _DSH_SHIPPED - _DSH_STDIO}
    names[_DSH_PROFILE] = "gmlx profile (default)"
    if container:
        from gmlx.container import confine
        from gmlx.container.settings import private_home_path

        # The guest owns the private home, so no link in it is followed.
        home = private_home_path("dsh", _dsh_project(words or []))
        root = home / ".dsh" / "profiles"
        found = []
        with confine.confined(home):
            try:
                listed = confine.listdir(root)
            except confine.ConfinedError:
                listed = []
            for name in filter(_SAFE_NAME.fullmatch, listed):
                try:
                    if confine.exists(root / name / "package.json"):
                        found.append(name)
                except confine.ConfinedError:
                    pass
    else:
        root = _dsh_home() / "profiles"
        found = [d.name for d in (root.iterdir() if root.is_dir() else ())
                 if _SAFE_NAME.fullmatch(d.name) and (d / "package.json").is_file()]
    for name in found:
        names.setdefault(name, "dsh profile")
    return [f"{n}\t{h}" for n, h in sorted(names.items())]


def _running_servers() -> list[dict]:
    """Runfile dicts for backgrounded servers (newest first), or ``[]``."""
    try:
        import gmlx.serve.lifecycle as lifecycle

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
        if not _SAFE_VALUE.fullmatch(val) or val in seen:
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
    if verb == "distill":
        return [] if filled else [f"{a}\tdistill action" for a in _DISTILL_ACTIONS]
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
    if "--" in pre[1:]:
        # After a bare --, the words belong to the client, never to gmlx.
        return ["::files"] if verb == "launch" else []
    if verb == "distill" and len(pre) > 1 and pre[1] in _DISTILL_ACTIONS:
        # the action's own parser carries the flags: scrape `distill <action> --help`
        verb = f"distill {pre[1]}"
        pre = [verb, *pre[2:]]

    if cur.startswith("-"):                  # completing a flag
        return [f"{opt}\t{h}" if h else opt for opt, _mv, h in _verb_options(verb)]

    prev = pre[-1]
    if prev.startswith("-") and "=" not in prev:
        opt = _option_for(verb, prev)
        if opt is not None and opt[1]:       # the previous flag wants a value
            choices = _choice_values(opt[1])
            if choices:                      # a {on,off,...} choices flag
                return choices
            if _is_pathish(opt[1]):
                return ["::files"]
            named = _named_value_candidates(opt[0], pre)
            if named:
                return named
            if opt[1] == "MODEL":            # a served model id (launch --model)
                return _model_candidates(pre[1:])
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
        for line in _safe_lines(_complete(list(argv))):
            print(line)
    except Exception:  # noqa: BLE001, S110 - never let completion fail loudly
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

# COMP_WORDS holds each word as typed, with its quotes and backslashes.
# This sets _gmlx_word to the word with them removed, as the command gets
# it, and runs nothing. When the word ends inside a quote, _gmlx_quote is
# that quote and _gmlx_qstart is where the quoted part starts in the word.
_gmlx_unquote() {
  local w=$1 out= q= c i=0 at=0
  while (( i < ${#w} )); do
    c=${w:i:1}
    i=$((i + 1))
    if [[ $q == "'" ]]; then
      if [[ $c == "'" ]]; then q=; else out+=$c; fi
    elif [[ $c == '\' ]]; then
      c=${w:i:1}
      i=$((i + 1))
      # In double quotes, a backslash stays before other characters.
      if [[ $q == '"' ]]; then
        case $c in '$'|'`'|'"'|'\') ;; *) out+='\' ;; esac
      fi
      out+=$c
    elif [[ $c == '"' ]]; then
      if [[ $q == '"' ]]; then q=; else q='"'; at=${#out}; fi
    elif [[ $c == "'" && -z $q ]]; then
      q="'"
      at=${#out}
    else
      out+=$c
    fi
  done
  _gmlx_word=$out
  _gmlx_quote=$q
  _gmlx_qstart=$at
}

# Add a candidate to COMPREPLY in the form the shell inserts. Inside an open
# quote, the shell replaces only the quoted part, and the quote keeps the
# candidate as one word. When the shell quotes file names itself, the
# candidate goes in as it is. Otherwise it goes in quoted, as one word.
# printf %q quotes the candidate, so nothing in it runs.
_gmlx_reply() {
  local q
  if [[ -n $_gmlx_quote ]]; then
    COMPREPLY+=("${1:_gmlx_qstart}")
  elif (( _gmlx_raw )); then
    COMPREPLY+=("$1")
  else
    printf -v q '%q' "$1"
    COMPREPLY+=("$q")
  fi
}

_gmlx() {
  local _gmlx_word= _gmlx_quote= _gmlx_qstart=0 _gmlx_raw=0 _w
  local -a _args=()
  for _w in "${COMP_WORDS[@]:1:COMP_CWORD}"; do
    _gmlx_unquote "$_w"
    _args+=("$_gmlx_word")
  done
  _gmlx_unquote "${COMP_WORDS[COMP_CWORD]}"
  local cur=$_gmlx_word

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

  # With file names among the candidates, bash 4 and later quote every
  # candidate themselves. bash 3.2 has no compopt.
  if (( _files )) && compopt -o filenames 2>/dev/null; then
    _gmlx_raw=1
  fi
  # Match by prefix in bash itself. compgen -W would expand each candidate,
  # running any command substitution a candidate holds.
  COMPREPLY=()
  local _c
  for _c in "${_cands[@]}"; do
    [[ $_c == "$cur"* ]] && _gmlx_reply "$_c"
  done
  if (( _files )); then
    while IFS= read -r _line; do
      [[ -n $_line ]] && _gmlx_reply "$_line"
    done < <(compgen -f -- "$cur")
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
