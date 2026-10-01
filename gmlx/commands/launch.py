"""``gmlx launch <harness>`` - point an external coding harness at a gmlx
server's OpenAI-compatible API, then exec it (starting the server if needed).

Mirrors ``ollama launch``, with one deliberate omission: **no auto-install**. The
harness must already be on ``PATH``; if it isn't we print its install pointer and
exit (we never run a package manager). We probe the server's ``/health`` +
``/v1/models`` over HTTP, render the harness's *own* config - into our own config
namespace, never mutating the user's harness files - and exec the harness pointed
at our ``/v1`` endpoint.

By default, if no server is reachable, ``launch`` auto-starts one in the background
from a default-location config (``~/.config/gmlx/gmlx.yaml``, ``~/.gmlx.yaml``),
polling with a spinner until it answers - when the config
preloads a model the wait spans that model's load. With no config anywhere it points
the user at ``gmlx init``. ``--no-start`` opts out; an explicit ``--base-url`` is
never auto-started.

Supports ten surfaces, each via its own native config - coding harnesses (opencode,
pi, omp, claude-code), agent runtimes (hermes, goose), two chat-focused terminal UIs
that are *not* coding harnesses (aichat, elia), a browser chat app (open-webui), and
the DeepSeek Harness web app (dsh):

- **opencode** - a custom OpenAI-compatible provider, injected via ``OPENCODE_CONFIG``
  so the user's ``~/.config/opencode`` is untouched.
- **pi** - merged non-destructively into ``~/.pi/agent/{models,settings}.json``. pi
  has no documented config-injection env var, so this harness edits the user's own
  files; existing providers/settings are preserved.
- **omp** (oh-my-pi) - merged non-destructively into ``~/.omp/agent/{models,config}.yml``
  (YAML). Provider goes in ``models.yml``; the default model is pinned via
  ``modelRoles.default`` in ``config.yml``. Existing providers/roles are preserved.
- **hermes** (NousResearch hermes-agent) - our ``model``/``providers.custom`` block is
  merged into ``$HERMES_HOME/config.yaml`` (default ``~/.hermes/config.yaml``), the
  only config file hermes 0.19 reads, after a timestamped backup of the previous
  file; ``CUSTOM_BASE_URL`` is exported too. Hermes refuses models with <64k context
  at startup, so launch notes a default model with a smaller window.
- **goose** (Block) - pointer keys (``GOOSE_PROVIDER: openai`` + ``OPENAI_HOST``-family)
  merged non-destructively into ``~/.config/goose/config.yaml``; ``OPENAI_API_KEY`` is
  exec-environment-only (env takes precedence in goose, and the YAML may hold a real
  OpenAI credential we must not overwrite).
- **claude-code** (Anthropic Claude Code) - pure env injection (``ANTHROPIC_BASE_URL`` /
  ``ANTHROPIC_MODEL`` / ``ANTHROPIC_AUTH_TOKEN``); ``~/.claude`` is never touched.
- **aichat** (sigoden/aichat) - a chat-REPL with tools/agents, not a coding harness.
  An ``openai-compatible`` client injected via ``AICHAT_CONFIG_DIR``; every served id is
  flagged ``supports_function_calling`` so its tools work against the server's tool-call
  surface. The user's ``~/.config/aichat`` is untouched.
- **elia** (darrenburns/elia) - a keyboard-centric chat TUI. A fresh ``config.toml``
  injected via ``XDG_CONFIG_HOME`` (each served id an OpenAI-compatible litellm model);
  the user's ``~/.config/elia`` is untouched. Requires the elia config.toml rewrite
  (elia >= 1.x).
- **open-webui** (Open WebUI) - a browser chat app (a web *server*, not a terminal
  client). Pure env injection (``OPENAI_API_BASE_URL`` / ``OPENAI_API_KEY`` /
  ``ENABLE_OLLAMA_API=false`` / ``DATA_DIR``) - Open WebUI has no config file, so
  nothing on disk is mutated. Runs on its own port (3000, since the server holds 8080),
  passed as ``serve --port`` because ``open-webui serve`` does not read the ``PORT`` env
  var (it would otherwise bind 8080 and collide with the gmlx server), with chat
  history + the sqlite DB at a host ``DATA_DIR`` (the reason to prefer this over the
  Docker image's opaque volume); then you open the printed URL.
  Its RAG embedder is pointed back at the gmlx server (``RAG_EMBEDDING_ENGINE=openai``)
  rather than the default local HuggingFace download, so no embedder is fetched at boot
  (and boot stays clean on a host with no cached embedder - ``HF_HUB_OFFLINE=1`` would
  crash there); document-RAG waits on the server's ``/v1/embeddings``. Its audio engines
  (``AUDIO_STT_*`` / ``AUDIO_TTS_*``) are wired at the server too, but only when it
  advertises STT/TTS via its ``/v1/models`` markers (server run with ``--stt`` / ``--tts``);
  a chat-only server keeps Open WebUI's built-in browser audio. Needs a separate install
  (``CLIENT_INSTALL``; Python 3.11/3.12 only, not 3.13).
- **dsh** (DeepSeek Harness, 0.1.7 or newer) - boots a gmlx-owned ``gmlx`` profile in
  ``$DSH_HOME``, created from the ``web`` template on first launch, with the provider,
  default model and compaction policies passed as a ``--patch`` overlay written under
  our namespace. The key reaches dsh as ``GMLX_API_KEY`` in the exec environment.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

from gmlx.container import confine

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8080
_PROVIDER_ID = "gmlx"
_CONFIG_HOME = "~/.config/gmlx"          # our namespace, never the harness's


# The exit codes of a launch that stops before the client starts, as
# docs/cli.md lists them. Once the client runs, the exit status is the
# client's own. The values follow sysexits(3) where one fits.
EXIT_FAILURE = 1                       # a refusal or failure of no class below
EXIT_USAGE = 2                         # a bad flag or flag combination
EXIT_UNAVAILABLE = os.EX_UNAVAILABLE   # 69: something launch needs is missing or down
EXIT_TEMPFAIL = os.EX_TEMPFAIL         # 75: busy or changing state, so try again later
EXIT_CONFIG = os.EX_CONFIG             # 78: no gmlx config, or a config that does not load


class LaunchError(RuntimeError):
    """A user-facing launch failure (server down, harness missing, bad --model).
    Carries a clean message, which the CLI prints, and the exit code."""

    def __init__(self, message: str, code: int = EXIT_FAILURE):
        super().__init__(message)
        self.code = code


def exit_code(e: BaseException) -> int:
    """The exit code of a launch that stops with ``e``."""
    from gmlx.config import ConfigError
    from gmlx.container import cli, settings

    if isinstance(e, LaunchError):
        return e.code
    if isinstance(e, ConfigError):
        return EXIT_CONFIG
    if isinstance(e, cli.Unavailable):
        return EXIT_UNAVAILABLE
    if isinstance(e, settings.Busy):
        return EXIT_TEMPFAIL
    return EXIT_FAILURE


# The program each client runs on the Mac, its name in messages, and the
# command that installs it there, Homebrew first when the project publishes
# a formula. A fourth item, when present, is one more line of advice.
CLIENT_INSTALL: dict[str, tuple[str, ...]] = {
    "claude-code": ("claude", "Claude Code", "brew install --cask claude-code"),
    "opencode": ("opencode", "opencode", "brew install sst/tap/opencode"),
    "pi": ("pi", "pi", "npm install -g @earendil-works/pi-coding-agent"),
    "omp": ("omp", "omp (oh-my-pi)", "brew install can1357/tap/omp"),
    "hermes": ("hermes", "hermes",
               "curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash"),
    "goose": ("goose", "goose", "brew install block-goose-cli"),
    "aichat": ("aichat", "aichat", "brew install aichat"),
    "elia": ("elia", "elia", "uv tool install elia-chat"),
    "open-webui": ("open-webui", "Open WebUI", "uv tool install --python 3.12 open-webui",
                   "Open WebUI needs Python 3.11 or 3.12."),
    "dsh": ("dsh", "dsh (DeepSeek Harness)", "npm install -g @deepseek-ai/dsh@next"),
}


# The section of docs/launch.md about each client.
_CLIENT_ANCHOR = {"claude-code": "claude-code", "opencode": "opencode-pi-and-omp",
                  "pi": "opencode-pi-and-omp", "omp": "opencode-pi-and-omp",
                  "hermes": "hermes", "goose": "goose", "aichat": "aichat", "elia": "elia",
                  "open-webui": "open-webui", "dsh": "dsh"}


def install_advice(client: str) -> str:
    """The lines that say how to get ``client``: the Mac install command, and
    container mode, which installs it in the image."""
    _binary, _label, command, *more = CLIENT_INSTALL[client]
    return "\n".join([f"Install it with:\n  {command}", *more,
                      "Or run it in a container, which installs it for you:",
                      f"  gmlx launch {client} --container"])


def _help_epilog(client: str | None) -> str:
    """The end of ``gmlx launch --help``. With a client named, it says how to
    install that client and where the guide covers it."""
    from gmlx import DOCS_URL

    example = client or "claude-code"
    common = (f"Arguments after -- go to the client, as in gmlx launch {example} -- "
              "--help.")
    if client is None:
        return (common + "\ngmlx launch menubar starts the macOS menu bar monitor for "
                "a running server.")
    label = CLIENT_INSTALL[client][1]
    return (f"{label} is a separate program. {install_advice(client)}\n"
            f"The launch guide covers {label}:\n"
            f"  {DOCS_URL}launch.html#{_CLIENT_ANCHOR[client]}\n\n{common}")


def _named_client(ap: argparse.ArgumentParser, argv: list) -> str | None:
    """The client that ``argv`` names, found before the parse so that
    ``--help`` can describe it. A word that is the value of an option does
    not count."""
    takes_value = {s for act in ap._actions if act.nargs != 0 for s in act.option_strings}
    skip = False
    for word in argv:
        if skip:
            skip = False
        elif word in takes_value:
            skip = True
        elif not word.startswith("-"):
            return word if word in _HARNESSES else None
    return None


def _find_binary(client: str, a):
    """``shutil.which`` + the standard not-on-PATH refusal shared by every
    client (skipped under --config-only, which only writes the config). In
    container mode the client runs in the image, so the bare name stands."""
    name, label = CLIENT_INSTALL[client][:2]
    if getattr(a, "container_mode", False):
        return name
    binary = shutil.which(name)
    if binary is None and not a.config_only:
        raise LaunchError(f"{label} is not on your PATH, and launch does not install "
                          f"clients on the Mac. {install_advice(client)}", EXIT_UNAVAILABLE)
    return binary


# Clients that cannot start without a default model, by the setting their
# refusal names. dsh also takes the server's only chat model.
_NEEDS_DEFAULT = {"claude-code": "ANTHROPIC_MODEL", "goose": "GOOSE_MODEL",
                  "hermes": "model.default", "dsh": "agent-default-model"}


def check_model_choice(client: str | None, models: list,
                       requested: str | None) -> str | None:
    """The default model a launch of ``client`` gets from the server's
    ``models``, after the checks every launch makes: ``--model`` must be
    served, and a client that needs a default model must get one."""
    default_model = _pick_default(models, requested)
    if default_model is None and client == "dsh":
        chat = chat_models(models)
        if len(chat) == 1:
            default_model = chat[0]["id"]
    if default_model is None and client in _NEEDS_DEFAULT:
        raise LaunchError(
            f"{client} needs a default model ({_NEEDS_DEFAULT[client]}), and the server "
            "marks none as its default. Pass --model, or set server.defaults.model in "
            "the server's config.")
    return default_model


def _probe_target(a):
    """The shared client preamble: resolve the server base URL, probe its
    served models, and pick the default, which a client in _NEEDS_DEFAULT
    must get."""
    base_url = a.base_url or f"http://{a.host}:{a.port}/v1"
    client = getattr(a, "harness", None)
    models = probe_models(base_url, a.api_key, client)
    default_model = check_model_choice(client, models, a.model)
    # In container mode the probe runs from the Mac, and the client reaches
    # the server at the guest URL.
    return getattr(a, "guest_base_url", None) or base_url, models, default_model


def _client_key(a) -> str | None:
    """The key a client config gets. In container mode the client reaches a
    local server through a session socket, which needs no key, so the config
    gets a placeholder and never the server's key."""
    return getattr(a, "client_api_key", None) or a.api_key


def _summary(name: str, base_url: str, models: list,
             default_model: str | None, extra: str = "") -> str:
    """The first ``[launch]`` status line every harness prints."""
    return (f"[launch] {name} -> {base_url}  ({len(models)} model(s)"
            + (f", default {default_model}" if default_model else "")
            + extra + ")")


def _finish(a, binary, argv: list, pairs: dict, *, drop=(), exec_fn) -> int:
    """The shared end of every harness: under --config-only, print the command
    that runs the client and return 0; otherwise exec it with ``pairs`` added
    to the environment and the ``drop`` names removed from it. The arguments
    after ``--`` on the launch command line follow ``argv``."""
    extra = list(getattr(a, "passthrough", None) or ())
    sink = getattr(a, "container_sink", None)
    if sink is not None:                  # container mode runs it in the image
        return sink(list(argv), dict(pairs), extra)
    if a.config_only:
        words = ([f"{k}={v}" for k, v in pairs.items()] + list(argv)
                 + [shlex.quote(w) for w in extra])
        print(f"[launch] run it with:  {' '.join(words)}")
        return 0
    env = dict(os.environ, **pairs)
    for name in drop:
        env.pop(name, None)
    return exec_fn(binary, list(argv) + extra, env)


# server probe (HTTP only - launch never imports the model stack)
def _http_get_json(url: str, timeout: float = 5.0, headers: dict | None = None):
    """GET ``url`` and parse JSON (lifecycle.get_json). Seam: monkeypatched in
    tests so the probe is exercised without a live server."""
    from gmlx.serve.lifecycle import get_json

    return get_json(url, timeout=timeout, headers=headers)


def _http_post_json(url: str, body: dict, *, api_key: str | None = None,
                    timeout: float = 3.0):
    """POST ``body`` as JSON and parse the reply (lifecycle.post_json). Seam:
    monkeypatched in tests."""
    from gmlx.serve.lifecycle import post_json

    return post_json(url, body, api_key=api_key, timeout=timeout)


def _http_delete(url: str, *, api_key: str | None = None, timeout: float = 3.0) -> int:
    """DELETE ``url`` and return the status. Seam: monkeypatched in tests."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    req = urllib.request.Request(url, headers=headers, method="DELETE")
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (local server)
        return r.status


def _keep_model(a) -> None:
    """Best-effort: ask the server to keep ``a.model`` resident through its idle-TTL
    reaper (it stays LRU-evictable) and warm-load it, so a coding session's model
    isn't idle-unloaded mid-use. Fire-and-forget - the server warms in the background
    and the harness execs immediately; an older server without ``/v1/keep`` just warns."""
    base = a.base_url or f"http://{a.host}:{a.port}/v1"
    url = base.rstrip("/") + "/keep"
    try:
        _http_post_json(url, {"model": a.model, "warm": True}, api_key=a.api_key)
        print(f"[launch] {a.model} stays loaded while idle. --no-keep turns this off.")
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read())
        except Exception:
            detail = None
        message = detail.get("message") if isinstance(detail, dict) else None
        if e.code == 404:
            # Two 404 shapes: a new server rejecting an unknown id (JSON body
            # with status "unknown_model"), or an old server with no /v1/keep
            # route at all.
            if isinstance(detail, dict) and detail.get("status") == "unknown_model":
                why = "the server does not offer it"
            else:
                why = "this server cannot keep a model. Update gmlx and restart it to keep one"
        elif e.code == 400 and message:
            # A bad profile or an ambiguous default carries a message that
            # says what to change.
            why = f"the server refused to keep it: {message}"
        else:
            why = f"the keep request failed ({e})"
        print(f"[launch] {a.model} can unload while idle, because {why}.")
    except (urllib.error.URLError, OSError, ValueError) as e:
        print(f"[launch] {a.model} can unload while idle, because the keep request "
              f"failed ({e}).")


def _server_root(base_url: str) -> str:
    """The server root for ``/health`` (strip a trailing ``/v1``)."""
    from gmlx.serve.lifecycle import server_root

    return server_root(base_url)


def probe_models(base_url: str, api_key: str | None = None,
                 client: str | None = None) -> list:
    """Confirm the server is up (``/health``) and return its ``/v1/models`` ``data``
    list. Raises :class:`LaunchError` with a start-the-server hint if unreachable.
    ``client`` names the client in the hint for a missing key."""
    root = _server_root(base_url)
    try:
        _http_get_json(root + "/health", timeout=5.0)
    except (urllib.error.URLError, OSError, ValueError) as e:
        from .launch_container import _why_unreachable

        raise LaunchError(f"no gmlx server answers at {root} ({_why_unreachable(e)}). "
                          "Start one with gmlx serve, or check the server address.",
                          EXIT_UNAVAILABLE)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    try:
        payload = _http_get_json(base_url.rstrip("/") + "/models", timeout=5.0,
                                 headers=headers)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            what = "refused the API key" if api_key else "needs an API key"
            raise LaunchError(
                f"the server at {root} {what}. Pass the server.api_key of its config "
                f"with gmlx launch {client or '<client>'} --api-key KEY.")
        raise LaunchError(f"server is up but /v1/models failed: {e}")
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise LaunchError(f"server is up but /v1/models failed: {e}")
    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, list):     # every consumer indexes m["id"]
        data = [m for m in data if isinstance(m, dict) and m.get("id")]
    if not data:
        raise LaunchError(no_models_message(root), EXIT_UNAVAILABLE)
    return data


def no_models_message(root: str) -> str:
    """The refusal for a server with no models. For a server that this Mac
    runs from a config gmlx pull does not find by itself, the gmlx pull
    command names that config."""
    import gmlx.serve.lifecycle as lifecycle
    from gmlx import DOCS_URL

    u = urllib.parse.urlsplit(root)
    try:
        run = lifecycle.read_run(u.hostname, u.port) if u.hostname and u.port else None
    except ValueError:
        run = None
    pull = "gmlx pull" + lifecycle.pull_config_flag(lifecycle.run_config_path(run or {}))
    return (f"the server at {root} has no models yet. Download one with {pull}, "
            "which adds it to the running server. The Quickstart lists models by the "
            f"memory they need:\n  {DOCS_URL}quickstart.html#choosing-a-model")


def _pick_default(models: list, requested: str | None) -> str | None:
    """The model id to make the harness default: an explicit ``--model`` (validated
    against the served ids), else the server's ``default``-marked id, else None.
    An ``id@profile`` form passes with a served head - the profile half is the
    server's to validate (an unknown one 400s, listing the valid names)."""
    ids = [m["id"] for m in models]
    if requested:
        head = requested.rsplit("@", 1)[0]
        if requested not in ids and head not in ids:
            raise LaunchError(f"--model {requested} is not a model the server offers. "
                              f"It offers {', '.join(sorted(ids))}.")
        return requested
    for m in models:
        if m.get("default"):
            return m["id"]
    return None


_SERVICE_MARKERS = ("stt", "tts", "embeddings", "rerank")


def chat_models(models: list) -> list:
    """The ``/v1/models`` entries a harness may offer for chat - the service
    advertisements (``whisper-1``, ``text-embedding-3-small``, ...) answer
    their own endpoints, not ``/v1/chat/completions``."""
    return [m for m in models if not any(m.get(k) for k in _SERVICE_MARKERS)]


# opencode
def _display_name(m: dict) -> str:
    """Display name for a harness's model menu - flag alias presets so a profile
    preset is recognisable next to the real ids it shares weights with."""
    if m.get("alias_of"):
        prof = f", {m['profile']}" if m.get("profile") else ""
        return f"{m['id']} (alias of {m['alias_of']}{prof})"
    return m["id"]


def build_opencode_config(base_url: str, models: list, *,
                          provider_id: str = _PROVIDER_ID,
                          default_model: str | None = None,
                          api_key: str | None = None) -> dict:
    """The opencode config that registers gmlx as a custom OpenAI-compatible
    provider with every served id (real models + alias presets) as a pickable model.
    ``apiKey`` only when the server requires one. Pure - no IO."""
    model_map = {m["id"]: {"name": _display_name(m)} for m in chat_models(models)}
    options: dict = {"baseURL": base_url}
    if api_key:
        options["apiKey"] = api_key
    cfg = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            provider_id: {
                "npm": "@ai-sdk/openai-compatible",
                "name": "gmlx (local)",
                "options": options,
                "models": model_map,
            }
        },
    }
    if default_model:
        cfg["model"] = f"{provider_id}/{default_model}"
    return cfg


def _launch_opencode(a, *, exec_fn) -> int:
    binary = _find_binary("opencode", a)
    base_url, models, default_model = _probe_target(a)
    cfg = build_opencode_config(base_url, models, provider_id=a.provider_id,
                                default_model=default_model, api_key=_client_key(a))

    out = Path(os.path.expanduser(a.config_path or f"{_CONFIG_HOME}/opencode.json"))
    _write_text_atomic(out, json.dumps(cfg, indent=2) + "\n")

    print(_summary("opencode", base_url, models, default_model)
          + "\n" + _files_line(a, "wrote", out))
    return _finish(a, binary, ["opencode"], {"OPENCODE_CONFIG": str(out)},
                   exec_fn=exec_fn)


# pi  (https://github.com/parsfaghfouri/pi - "ollama launch pi")
# pi has no documented config-injection env var, so this is the one harness that
# merges into the user's own files (`~/.pi/agent/{models,settings}.json`). We
# preserve every other provider/setting the user already has.
_PI_AGENT_HOME = "~/.pi/agent"


# Every file a handler touches goes through these helpers and
# gmlx.container.confine. In container mode the private home is the guest's
# to change, so a handler must never follow a link the guest planted there.

def _write_text_atomic(path: Path, text: str) -> None:
    """A new file + rename, creating the folders above it. Several of these
    targets are another tool's live config - a crash or full disk mid-write
    must not leave it truncated - and the file keeps its mode."""
    try:
        confine.write_text(path, text)
    except confine.ConfinedError as e:
        raise LaunchError(str(e)) from None


def _mkdirs(path: Path) -> None:
    try:
        confine.mkdirs(path)
    except confine.ConfinedError as e:
        raise LaunchError(str(e)) from None


def _exists(path: Path) -> bool:
    try:
        return confine.exists(path)
    except confine.ConfinedError as e:
        raise LaunchError(str(e)) from None


def _read_config_text(path: Path) -> str:
    """The read half of the edit-in-place flows, with the same refusal contract
    as the parsers: unreadable/binary -> LaunchError, never a traceback. A
    missing file reads as empty."""
    try:
        return (confine.read_text(path) or "").strip()
    except UnicodeDecodeError:
        raise LaunchError(f"{path} is not a text file, so launch does not overwrite it. "
                          "Fix or move the file, then launch again.")
    except confine.ConfinedError as e:
        raise LaunchError(str(e)) from None
    except OSError as e:
        raise LaunchError(f"cannot read {path} ({e}).")


# The largest client config launch parses. A config the client wrote in a
# container can be crafted to take long to parse, and real ones are small.
CONFIG_PARSE_MAX = 256 << 10


def _parse_text(path: Path) -> str:
    """The text of a config launch merges into, at most
    :data:`CONFIG_PARSE_MAX` bytes. A missing file reads as empty."""
    text = _read_config_text(path)
    if len(text.encode("utf-8", "surrogatepass")) > CONFIG_PARSE_MAX:
        raise LaunchError(f"{path} is larger than {CONFIG_PARSE_MAX >> 10} KiB, so launch "
                          "does not read or overwrite it. Fix or move the file, then "
                          "launch again.")
    return text


def _submap(doc: dict, key: str, what: str) -> dict:
    """A copy of the mapping at ``doc[key]``, or ``{}`` when it is absent.
    Any other value stops the merge, since launch would otherwise replace
    it, or fail on it."""
    value = doc.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise LaunchError(f"{what} has {key} as a {type(value).__name__}, not a mapping, so "
                          "launch does not overwrite it. Fix or move the file, then "
                          "launch again.")
    return dict(value)


def _load_json(path: Path) -> dict:
    """Read a JSON object from ``path``; ``{}`` if it's absent or empty. Raises
    :class:`LaunchError` on malformed JSON (we won't silently clobber a file we
    can't parse)."""
    if not _exists(path):
        return {}
    text = _parse_text(path)
    if not text:
        return {}
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as e:
        raise LaunchError(f"{path} is not valid JSON ({e}), so launch does not overwrite it. "
                          "Fix or move the file, then launch again.")
    except RecursionError:
        raise LaunchError(f"{path} nests too deeply to read, so launch does not overwrite it. "
                          "Fix or move the file, then launch again.")
    except ValueError as e:
        # Such as a number longer than Python converts.
        raise LaunchError(f"{path} cannot be read ({e}), so launch does not overwrite it. "
                          "Fix or move the file, then launch again.")
    if not isinstance(doc, dict):
        raise LaunchError(f"{path} is not a JSON object, so launch does not overwrite it. "
                          "Fix or move the file, then launch again.")
    return doc


# pi-ai clients size an unsized model generously: pi at contextWindow 128000 /
# maxTokens 16384, dsh at 262144 / 32768. A 32k model advertised as 128k never
# compacts and overflows the server instead, and the client pins max_tokens on
# every request, which gmlx's preflight prices into the KV estimate - 16k of
# generation headroom per subagent is real memory. Cap generation at this many
# tokens (or a quarter of the window when smaller, and at most half of it).
_MAX_TOKENS_CAP = 8192
_MAX_TOKENS_FLOOR = 1024

# pi-ai request switches for a gmlx provider: the output cap goes out as
# max_tokens, and the store, developer-role and long prompt-cache fields,
# which the server does not read, stay off the request.
_PI_AI_COMPAT = {
    "maxTokensField": "max_tokens",
    "supportsStore": False,
    "supportsDeveloperRole": False,
    "supportsLongCacheRetention": False,
}


def model_capacity(m: dict) -> tuple | None:
    """``(context window, output cap)`` for a ``/v1/models`` entry, or None when
    it reports no context. The window is the smaller of the trained context and
    what fits at width 1."""
    sizes = [v for v in (m.get("context_length"), m.get("max_context_at_width_1"))
             if isinstance(v, int) and v > 0]
    if not sizes:
        return None
    window = min(sizes)
    out = max(_MAX_TOKENS_FLOOR, min(_MAX_TOKENS_CAP, window // 4))
    return window, min(out, window // 2)


def model_window(models: list, model_id: str | None) -> int | None:
    """The context window of ``model_id`` in a ``/v1/models`` list, from its
    own entry or, for an unlisted ``id@profile``, from its base id's entry.
    None when the server reports no context for it."""
    by_id = {m["id"]: m for m in models}
    m = by_id.get(model_id) if model_id else None
    if m is None and model_id:
        m = by_id.get(model_id.rsplit("@", 1)[0])
    capacity = model_capacity(m) if m is not None else None
    return capacity[0] if capacity else None


def pi_model_entry(m: dict) -> dict:
    """One ``models[]`` entry for pi's ``models.json``: the id, plus
    ``contextWindow`` / ``maxTokens`` when ``/v1/models`` sizes the model."""
    entry = {"id": m["id"]}
    capacity = model_capacity(m)
    if capacity:
        entry["contextWindow"], entry["maxTokens"] = capacity
    return entry


def build_pi_configs(base_url: str, models: list, *,
                     provider_id: str = _PROVIDER_ID,
                     default_model: str | None = None,
                     api_key: str | None = None,
                     existing_models: dict | None = None,
                     existing_settings: dict | None = None) -> tuple:
    """The merged ``(models.json, settings.json)`` pi documents. Registers gmlx
    as an ``openai-completions`` provider with every served id; preserves any other
    providers/settings the user already configured. Pure - no IO."""
    models_doc = dict(existing_models or {})
    providers = _submap(models_doc, "providers", "pi's models.json")
    providers[provider_id] = {
        "baseUrl": base_url,
        "api": "openai-completions",
        # pi requires an apiKey; a placeholder when the server has no auth.
        "apiKey": api_key or provider_id,
        "compat": dict(_PI_AI_COMPAT),
        "models": [pi_model_entry(m) for m in chat_models(models)],
    }
    models_doc["providers"] = providers

    settings_doc = dict(existing_settings or {})
    settings_doc["defaultProvider"] = provider_id
    if default_model:
        settings_doc["defaultModel"] = default_model
    return models_doc, settings_doc


def _launch_pi(a, *, exec_fn) -> int:
    binary = _find_binary("pi", a)
    base_url, models, default_model = _probe_target(a)

    agent_dir = Path(os.path.expanduser(a.config_path or _PI_AGENT_HOME))
    models_path = agent_dir / "models.json"
    settings_path = agent_dir / "settings.json"
    models_doc, settings_doc = build_pi_configs(
        base_url, models, provider_id=a.provider_id, default_model=default_model,
        api_key=_client_key(a),
        existing_models=_load_json(models_path),
        existing_settings=_load_json(settings_path))

    _write_text_atomic(models_path, json.dumps(models_doc, indent=2) + "\n")
    _write_text_atomic(settings_path, json.dumps(settings_doc, indent=2) + "\n")

    print(_summary("pi", base_url, models, default_model)
          + "\n" + _files_line(a, "merged", models_path, settings_path))
    return _finish(a, binary, ["pi"], {}, exec_fn=exec_fn)


# omp  (oh-my-pi - https://github.com/can1357/oh-my-pi - "ollama launch omp")
# omp keeps a YAML provider registry in ~/.omp/agent/models.yml and pins the
# default model by role in ~/.omp/agent/config.yml (`modelRoles.default`). Like
# pi, no config-injection env var, so we merge into the user's own files,
# preserving every other provider/role.
_OMP_AGENT_HOME = "~/.omp/agent"


def _load_yaml(path: Path) -> dict:
    """Read a YAML mapping from ``path``; ``{}`` if absent or empty. Raises
    :class:`LaunchError` on malformed YAML or a non-mapping document (we won't
    clobber a file we can't parse)."""
    if not _exists(path):
        return {}
    text = _parse_text(path)
    if not text:
        return {}
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise LaunchError(f"{path} is not valid YAML ({e}), so launch does not overwrite it. "
                          "Fix or move the file, then launch again.")
    except RecursionError:
        raise LaunchError(f"{path} nests too deeply to read, so launch does not overwrite it. "
                          "Fix or move the file, then launch again.")
    except ValueError as e:
        # Such as a number longer than Python converts.
        raise LaunchError(f"{path} cannot be read ({e}), so launch does not overwrite it. "
                          "Fix or move the file, then launch again.")
    if doc is None:
        return {}
    if not isinstance(doc, dict):
        raise LaunchError(f"{path} is not a YAML mapping, so launch does not overwrite it. "
                          "Fix or move the file, then launch again.")
    return doc


def build_omp_configs(base_url: str, models: list, *,
                      provider_id: str = _PROVIDER_ID,
                      default_model: str | None = None,
                      existing_models: dict | None = None,
                      existing_config: dict | None = None) -> tuple:
    """The merged ``(models.yml, config.yml)`` omp documents. Registers gmlx as
    an unauthenticated ``openai-completions`` provider with every served id, and (if
    a default is known) pins ``modelRoles.default`` to it. Preserves any other
    providers/roles. Pure - no IO."""
    models_doc = dict(existing_models or {})
    providers = _submap(models_doc, "providers", "omp's models.yml")
    providers[provider_id] = {
        "baseUrl": base_url,
        "api": "openai-completions",
        "auth": "none",                        # local server is unauthenticated
        "models": [{"id": m["id"], "name": _display_name(m)}
                   for m in chat_models(models)],
    }
    models_doc["providers"] = providers

    config_doc = dict(existing_config or {})
    if default_model:
        roles = _submap(config_doc, "modelRoles", "omp's config.yml")
        roles["default"] = f"{provider_id}/{default_model}"
        config_doc["modelRoles"] = roles
    return models_doc, config_doc


def _launch_omp(a, *, exec_fn) -> int:
    if a.api_key and _client_key(a) == a.api_key:
        print("[launch] note: launch cannot write an API key into omp's provider "
              "registry, so set up omp's own auth when the server needs a key.",
              file=sys.stderr)
    binary = _find_binary("omp", a)
    base_url, models, default_model = _probe_target(a)

    agent_dir = Path(os.path.expanduser(a.config_path or _OMP_AGENT_HOME))
    models_path = agent_dir / "models.yml"
    config_path = agent_dir / "config.yml"
    models_doc, config_doc = build_omp_configs(
        base_url, models, provider_id=a.provider_id, default_model=default_model,
        existing_models=_load_yaml(models_path),
        existing_config=_load_yaml(config_path))

    _write_text_atomic(models_path, yaml.safe_dump(models_doc, sort_keys=False))
    _write_text_atomic(config_path, yaml.safe_dump(config_doc, sort_keys=False))

    print(_summary("omp", base_url, models, default_model)
          + "\n" + _files_line(a, "merged", models_path, config_path))
    return _finish(a, binary, ["omp"], {}, exec_fn=exec_fn)


# hermes  (NousResearch hermes-agent - https://github.com/NousResearch/hermes-agent)
# hermes 0.19 reads its settings only from ``$HERMES_HOME/config.yaml``, and
# sends an API key to a local server only from that file, so launch merges
# its provider block into it: a merge client, not an injection one. On the
# Mac the previous file is backed up first. In container mode the file is the
# private home's own. The provider *type* is hermes's literal ``custom``
# (``--provider-id`` does not apply); ``CUSTOM_BASE_URL`` is exported too -
# hermes's override for ``provider: custom``. A default model is mandatory
# (``model.default``).


HERMES_BACKUPS = 3
_HERMES_BACKUP_NAME = re.compile(r"\.gmlx-(\d{8}-\d{6})(?:-(\d+))?")


def _hermes_backup(path: Path) -> Path:
    """Copy ``path`` to a new ``<name>.gmlx-<date>-<time>[-n]`` beside it,
    with its mode, then delete all but the newest few of those copies. Only
    names of that form are touched."""
    data = path.read_bytes()
    mode = stat.S_IMODE(os.stat(path).st_mode)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    def age(p: Path):
        m = _HERMES_BACKUP_NAME.fullmatch(p.name[len(path.name):])
        return (m.group(1), int(m.group(2) or 0)) if m else None

    def ours():
        return [p for p in path.parent.iterdir()
                if p.name.startswith(path.name + ".gmlx-") and age(p) is not None
                and not p.is_symlink()]
    # A later backup in the same second always gets a higher number, so the
    # names sort by age.
    first = max((age(p)[1] + 1 for p in ours() if age(p)[0] == stamp), default=0)
    for n in range(first, first + 1000):
        backup = path.with_name(f"{path.name}.gmlx-{stamp}" + (f"-{n}" if n else ""))
        try:
            fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
        except FileExistsError:
            continue
        with os.fdopen(fd, "wb") as f:
            os.fchmod(f.fileno(), mode)
            f.write(data)
        break
    else:
        raise LaunchError(f"cannot find a free backup name beside {path}")
    for old in sorted(ours(), key=age)[:-HERMES_BACKUPS]:
        old.unlink(missing_ok=True)
    return backup


def _hermes_config_path() -> Path:
    home = os.environ.get("HERMES_HOME")
    return Path(os.path.expanduser(home or "~/.hermes")) / "config.yaml"


def build_hermes_config(base_url: str, *, default_model: str,
                        api_key: str | None = None,
                        existing: dict | None = None) -> dict:
    """The merged hermes ``config.yaml`` document: the user's existing settings
    with ``model`` pointed at our ``custom`` provider. Key paths follow
    ``hermes config set model.provider/model.default/model.base_url`` and
    ``providers.<name>.*``; hermes resolves a ``providers.custom`` entry before
    ``model.base_url``, so both name the server. Pure - no IO."""
    cfg = dict(existing or {})
    model = cfg.get("model")
    model = dict(model) if isinstance(model, dict) else {}
    model["provider"] = "custom"
    model["default"] = default_model
    model["base_url"] = base_url
    if api_key:
        model["api_key"] = api_key
    cfg["model"] = model
    providers = _submap(cfg, "providers", "hermes's config.yaml")
    custom = _submap(providers, "custom", "hermes's config.yaml providers")
    custom["base_url"] = base_url
    if api_key:
        custom["api_key"] = api_key
    else:
        custom.setdefault("api_key", _PROVIDER_ID)   # placeholder: no server auth
    providers["custom"] = custom
    cfg["providers"] = providers
    return cfg


# hermes refuses to start on a model with a smaller context window.
_HERMES_MIN_CONTEXT = 65536


def _launch_hermes(a, *, exec_fn) -> int:
    binary = _find_binary("hermes", a)
    if a.config_path:
        raise LaunchError("--config-path does not apply to hermes, which reads only "
                          "$HERMES_HOME/config.yaml. Set HERMES_HOME to use another folder.")
    base_url, models, default_model = _probe_target(a)

    path = _hermes_config_path()
    existing = _load_yaml(path)
    cfg = build_hermes_config(base_url, default_model=default_model,
                              api_key=_client_key(a), existing=existing)
    print(_summary("hermes", base_url, models, default_model))
    if cfg == existing:
        print(_files_line(a, "kept", path) + ", which already points hermes at the server")
    else:
        # The private home is gmlx's own, so only a file on the Mac is backed up.
        backup = None
        if not getattr(a, "container_mode", False) and _exists(path):
            backup = _hermes_backup(path)
            print(f"[launch] backed up {path} to {backup}")
        _write_text_atomic(path, yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        print(_files_line(a, "wrote", path) + (
            ", without its comments and layout, which the backup keeps" if backup else ""))
    window = model_window(models, default_model)
    if window is not None and window < _HERMES_MIN_CONTEXT:
        print(f"[launch] hermes refuses a model with less than 64K tokens of context, and "
              f"{default_model} has {window}. Pass --model with a model that has more.")
    return _finish(a, binary, ["hermes"], {"CUSTOM_BASE_URL": base_url}, exec_fn=exec_fn)


# goose  (Block - https://github.com/block/goose)
# goose's env vars take precedence over its config file, so the exec
# environment alone points the session at our server; the non-secret pointer
# keys are also merged non-destructively into ``~/.config/goose/config.yaml``
# so a later bare ``goose`` keeps working. ``OPENAI_API_KEY`` travels in the
# environment only - never the YAML, where it could clobber a real OpenAI
# credential. The provider is goose's literal ``openai`` engine
# (``--provider-id`` does not apply); ``GOOSE_MODEL`` is mandatory.
_GOOSE_CONFIG = "~/.config/goose/config.yaml"


def build_goose_env(base_url: str, *, default_model: str,
                    api_key: str | None = None) -> dict:
    """The goose provider settings as env-var pairs (also the config.yaml key
    names). ``OPENAI_HOST`` is scheme://host:port only; the API path goes in
    ``OPENAI_BASE_PATH``. Pure - no IO."""
    root = _server_root(base_url)
    path = base_url.rstrip("/")[len(root):].strip("/") or "v1"
    return {
        "GOOSE_PROVIDER": "openai",
        "GOOSE_MODEL": default_model,
        "OPENAI_HOST": root,
        "OPENAI_BASE_PATH": f"{path}/chat/completions",
        "OPENAI_API_KEY": api_key or _PROVIDER_ID,  # placeholder: no server auth
    }


def _launch_goose(a, *, exec_fn) -> int:
    binary = _find_binary("goose", a)
    base_url, models, default_model = _probe_target(a)
    pairs = build_goose_env(base_url, default_model=default_model,
                            api_key=_client_key(a))

    cfg_path = Path(os.path.expanduser(a.config_path or _GOOSE_CONFIG))
    cfg = _load_yaml(cfg_path)
    # Persist only the non-secret pointer keys; OPENAI_API_KEY stays env-only so
    # we never clobber a real credential in the user's config.yaml.
    cfg.update({k: v for k, v in pairs.items() if k != "OPENAI_API_KEY"})
    _write_text_atomic(cfg_path, yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    print(_summary("goose", base_url, models, default_model)
          + "\n" + _files_line(a, "merged", cfg_path)
          + ", without the key, which goose gets in OPENAI_API_KEY")
    return _finish(a, binary, ["goose", "session"], pairs, exec_fn=exec_fn)


# claude-code  (Anthropic Claude Code - https://claude.com/claude-code)
# Claude Code reads its provider entirely from env vars, so this is pure
# injection (the user's ~/.claude is never touched): ``ANTHROPIC_BASE_URL``
# points at the server root (Claude Code appends /v1/messages itself - the
# Anthropic surface mlx-vlm serves), ``ANTHROPIC_MODEL`` pins the default, and
# ``ANTHROPIC_AUTH_TOKEN`` carries the key (a placeholder when the server has
# no auth - it must be non-empty or Claude Code starts its own login flow).
# ``ANTHROPIC_SMALL_FAST_MODEL`` routes the background/haiku-class calls to the
# same local model. Claude Code assumes a 200k window for a model outside its
# catalog, so ``CLAUDE_CODE_MAX_CONTEXT_TOKENS`` gives it the window the server
# reports, unless the user's own value is smaller.
CONTEXT_TOKENS = "CLAUDE_CODE_MAX_CONTEXT_TOKENS"


def build_claude_code_env(base_url: str, *, default_model: str,
                          api_key: str | None = None,
                          context_tokens: str | None = None) -> dict:
    """The Claude Code provider settings as env-var pairs. Pure - no IO."""
    pairs = {
        "ANTHROPIC_BASE_URL": _server_root(base_url),
        "ANTHROPIC_AUTH_TOKEN": api_key or _PROVIDER_ID,  # placeholder: no auth
        "ANTHROPIC_MODEL": default_model,
        "ANTHROPIC_SMALL_FAST_MODEL": default_model,
    }
    if context_tokens:
        pairs[CONTEXT_TOKENS] = context_tokens
    return pairs


def claude_context_tokens(window: int | None, own: str | None,
                          model: str | None) -> tuple[str | None, str | None]:
    """The ``CLAUDE_CODE_MAX_CONTEXT_TOKENS`` that launch sets, and the line
    it prints when that replaces the user's own value ``own``. The smaller
    value wins: a whole number from 1 to the model's window stays, and a
    larger one or one that is not a number gets the window. Spaces around
    ``own`` do not count. With no window known, launch sets nothing and the
    user's value reaches Claude Code."""
    if window is None:
        return None, None
    own = (own or "").strip()
    if not own:
        return str(window), None
    if re.fullmatch(r"[0-9]+", own) and 1 <= int(own) <= window:
        return own, None
    return str(window), (f"[launch] Claude Code gets {CONTEXT_TOKENS}={window}, the window "
                         f"of {model}, in place of your {own}")


def _launch_claude_code(a, *, exec_fn) -> int:
    binary = _find_binary("claude-code", a)
    base_url, models, default_model = _probe_target(a)
    # In container mode the user's value comes only from launch.container.env.
    if getattr(a, "container_sink", None) is not None:
        own = getattr(a, "container_context_tokens", None)
    else:
        own = os.environ.get(CONTEXT_TOKENS)
    tokens, replaced = claude_context_tokens(model_window(models, default_model), own,
                                             default_model)
    pairs = build_claude_code_env(base_url, default_model=default_model,
                                  api_key=_client_key(a), context_tokens=tokens)

    print(_summary("claude-code", f"{pairs['ANTHROPIC_BASE_URL']}/v1/messages",
                   models, default_model))
    if replaced:
        print(replaced)
    elif tokens and a.config_only and getattr(a, "container_sink", None) is not None:
        # The container dry run shows env names only, so the value shows here.
        whose = ("your own value" if own and own.strip() == tokens
                 else f"the window of {default_model}")
        print(f"[launch] Claude Code gets {CONTEXT_TOKENS}={tokens}, {whose}")
    if _prompt_cache_off(a.host, a.port, default_model):
        print("[launch] the server's prompt cache is off, and Claude Code resends a long "
              "system prompt on every turn, so each turn starts slowly. Turn the cache on "
              "with server.cache.enabled in the server's config.")
    # An inherited real key would take precedence over our ANTHROPIC_AUTH_TOKEN.
    return _finish(a, binary, ["claude"], pairs, drop=("ANTHROPIC_API_KEY",),
                   exec_fn=exec_fn)


# aichat  (sigoden/aichat - all-in-one LLM CLI: chat-REPL, roles, sessions, RAG,
# tools/agents - a chat client, not a coding harness). Clean injection like opencode:
# aichat honours ``AICHAT_CONFIG_DIR``, so we write our own config dir and never touch
# ``~/.config/aichat``. gmlx becomes one ``openai-compatible`` client; every served
# id is a model, flagged ``supports_function_calling`` so aichat's tools/agents work
# against the server's tool-call surface. Default + selection use the ``<client>:<model>``
# form (e.g. ``gmlx:qwen3.6-27b``).
_AICHAT_CONFIG_HOME = f"{_CONFIG_HOME}/aichat"


def build_aichat_config(base_url: str, models: list, *,
                        provider_id: str = _PROVIDER_ID,
                        default_model: str | None = None,
                        api_key: str | None = None) -> dict:
    """The aichat ``config.yaml`` registering gmlx as an ``openai-compatible``
    client with every served id (function-calling on; vision flagged for VLM ids).
    ``model`` pins the default as ``<client>:<id>``. Pure - no IO."""
    client: dict = {"type": "openai-compatible", "name": provider_id,
                    "api_base": base_url}
    if api_key:
        client["api_key"] = api_key
    entries = []
    for m in chat_models(models):
        e = {"name": m["id"], "supports_function_calling": True}
        if m.get("vlm"):
            e["supports_vision"] = True
        entries.append(e)
    client["models"] = entries
    cfg: dict = {"clients": [client]}
    if default_model:
        cfg["model"] = f"{provider_id}:{default_model}"
    return cfg


def _launch_aichat(a, *, exec_fn) -> int:
    binary = _find_binary("aichat", a)
    base_url, models, default_model = _probe_target(a)
    cfg = build_aichat_config(base_url, models, provider_id=a.provider_id,
                              default_model=default_model, api_key=_client_key(a))

    cfg_dir = Path(os.path.expanduser(a.config_path or _AICHAT_CONFIG_HOME))
    _mkdirs(cfg_dir)
    cfg_file = cfg_dir / "config.yaml"
    _write_text_atomic(cfg_file, yaml.safe_dump(cfg, sort_keys=False))

    print(_summary("aichat", base_url, models, default_model)
          + "\n" + _files_line(a, "wrote", cfg_file))
    from gmlx.container import notices

    # The note carries no news after the first launch, so it prints once.
    for line in notices.due([notices.Once(
            "[launch] note: tool use in aichat also needs its functions, which the "
            "llm-functions project installs. The server already parses tool calls.",
            "aichat-functions")]):
        print(line)
    return _finish(a, binary, ["aichat"], {"AICHAT_CONFIG_DIR": str(cfg_dir)},
                   exec_fn=exec_fn)


# elia  (darrenburns/elia - a keyboard-centric chat TUI, not a coding harness).
# elia reads ``$XDG_CONFIG_HOME/elia/config.toml`` (via xdg-base-dirs), so we point
# ``XDG_CONFIG_HOME`` at our own namespace and write a fresh config there - the user's
# ``~/.config/elia`` is untouched (opencode-style injection; the chat-history DB lives
# under ``XDG_DATA_HOME``, which we leave alone). Each served id is an OpenAI-compatible
# litellm model (``name = "openai/<id>"`` + ``api_base``); ``id`` is a gmlx-prefixed
# lookup key so selection never collides with the user's own models. Requires the elia
# config.toml rewrite (elia >= 1.x); older builds ignore custom endpoints.
_ELIA_CONFIG_HOME = f"{_CONFIG_HOME}/elia-xdg"


def _toml_basic_string(s: str) -> str:
    """``s`` as a double-quoted TOML basic string (escape backslash, quote,
    and control chars - a stray newline in an api key must not produce an
    unparseable config)."""
    esc = s.replace("\\", "\\\\").replace('"', '\\"')
    return '"' + "".join(f"\\u{ord(c):04X}" if ord(c) < 0x20 or c == "\x7f"
                         else c for c in esc) + '"'


def build_elia_config(base_url: str, models: list, *,
                      provider_id: str = _PROVIDER_ID,
                      default_model: str | None = None,
                      api_key: str | None = None) -> str:
    """The elia ``config.toml`` text: one OpenAI-compatible ``[[models]]`` per served id
    (litellm ``openai/<id>`` routing + ``api_base``), a gmlx-prefixed ``id`` lookup
    key, and ``default_model`` when known. Pure - returns the document text."""
    key = api_key or provider_id                 # litellm wants a non-empty key
    lines: list = ["# Written by `gmlx launch elia` - points elia at a gmlx server.",
                   "# Your own ~/.config/elia/config.toml is untouched.", ""]
    if default_model:
        lines.append(
            f"default_model = {_toml_basic_string(f'{provider_id}/{default_model}')}")
        lines.append("")
    for m in chat_models(models):
        served = m["id"]
        lines += [
            "[[models]]",
            f"name = {_toml_basic_string(f'openai/{served}')}",
            f"id = {_toml_basic_string(f'{provider_id}/{served}')}",
            f"display_name = {_toml_basic_string(_display_name(m))}",
            f"api_base = {_toml_basic_string(base_url)}",
            f"api_key = {_toml_basic_string(key)}",
            "",
        ]
    return "\n".join(lines).rstrip("\n") + "\n"


def _launch_elia(a, *, exec_fn) -> int:
    binary = _find_binary("elia", a)
    base_url, models, default_model = _probe_target(a)
    toml_text = build_elia_config(base_url, models, provider_id=a.provider_id,
                                  default_model=default_model, api_key=_client_key(a))

    xdg_home = Path(os.path.expanduser(a.config_path or _ELIA_CONFIG_HOME))
    out = xdg_home / "elia" / "config.toml"
    _write_text_atomic(out, toml_text)

    print(_summary("elia", base_url, models, default_model)
          + "\n" + _files_line(a, "wrote", out))
    if not getattr(a, "container_mode", False):     # the image installs a current elia
        print("[launch] elia 1.x or newer is needed, and an older elia lists no local "
              "models. Upgrade it with uv tool upgrade elia-chat.")

    argv = ["elia"]
    if default_model:
        argv += ["-m", f"{a.provider_id}/{default_model}"]
    return _finish(a, binary, argv, {"XDG_CONFIG_HOME": str(xdg_home)},
                   exec_fn=exec_fn)


# open-webui  (Open WebUI - a browser chat app, not a terminal client: it runs as its
# own web server you point a browser at). Pure env injection like claude-code - Open
# WebUI has no config file, every knob is an env var - so nothing on disk is mutated.
# We aim its single OpenAI endpoint at the gmlx server, silence the Ollama probe,
# run it on its own port (the server already holds 8080), and pin DATA_DIR so chat
# history + the sqlite DB live at a known host path (the reason to prefer this over the
# Docker image, whose storage is a container volume). Needs a separate install - it's a
# heavy Python app pinned to Python 3.11/3.12.
_OPEN_WEBUI_PORT = 3000
_OPEN_WEBUI_DATA_HOME = "~/.open-webui"
# Open WebUI's openai TTS engine always sends a voice; its default ("alloy") is an
# OpenAI voice that the default TTS model (Kokoro) rejects. Pin a valid Kokoro voice
# so read-aloud works out of the box. An exported AUDIO_TTS_VOICE wins, for a
# non-Kokoro `--tts` model that needs one of its own voices.
_OPEN_WEBUI_TTS_VOICE = "af_heart"


def build_open_webui_env(base_url: str, *, default_model: str | None = None,
                         api_key: str | None = None,
                         port: int = _OPEN_WEBUI_PORT, data_dir: str,
                         stt: bool = False, tts: bool = False,
                         rerank: bool = False) -> dict:
    """Open WebUI backend settings as env-var pairs (it has no config file). Points its
    single OpenAI endpoint at the gmlx server, disables the Ollama probe, pins its
    own port + an on-disk DATA_DIR, preselects the default model when known, and routes
    RAG embeddings at the server. ``stt``/``tts`` additionally route Open WebUI's audio
    engines at the server's ``/v1/audio/*``, and ``rerank`` routes its RAG reranker at
    the server's ``/v1/rerank`` - set each only when the server actually advertises that
    capability (see :func:`_launch_open_webui`), so a chat-only server doesn't break
    Open WebUI's built-in browser TTS / local reranker. Pure - no IO."""
    key = api_key or _PROVIDER_ID
    pairs = {
        "OPENAI_API_BASE_URL": base_url,             # Open WebUI appends /models etc.
        "OPENAI_API_KEY": key,                       # placeholder: no auth
        "ENABLE_OLLAMA_API": "false",                # don't probe a missing Ollama
        "PORT": str(port),
        "DATA_DIR": data_dir,                        # chat DB + uploads on the host fs
        # Point RAG at the gmlx server instead of letting Open WebUI download a
        # local sentence-transformers embedder from HuggingFace at boot. The "openai"
        # engine is lazy (no model load at startup), so this both suppresses that
        # surprise fetch and boots cleanly even on a host with no cached embedder -
        # HF_HUB_OFFLINE=1 would instead hard-crash boot there, since Open WebUI builds
        # an embedding function unconditionally. Document-RAG works the moment the
        # server is run with `--embeddings` (the id below is what /v1/embeddings
        # advertises + accepts); until then it stays inert, chat unaffected.
        "RAG_EMBEDDING_ENGINE": "openai",
        "RAG_OPENAI_API_BASE_URL": base_url,         # Open WebUI appends /embeddings
        "RAG_OPENAI_API_KEY": key,
        "RAG_EMBEDDING_MODEL": "text-embedding-3-small",
        "RAG_EMBEDDING_MODEL_AUTO_UPDATE": "false",  # belt-and-suspenders: no HF check
    }
    if stt:
        pairs.update({
            "AUDIO_STT_ENGINE": "openai",            # mic transcription -> our server
            "AUDIO_STT_OPENAI_API_BASE_URL": base_url,
            "AUDIO_STT_OPENAI_API_KEY": key,
            "AUDIO_STT_MODEL": "whisper-1",          # what /v1/audio/transcriptions accepts
        })
    if tts:
        pairs.update({
            "AUDIO_TTS_ENGINE": "openai",            # read-aloud -> our server
            "AUDIO_TTS_OPENAI_API_BASE_URL": base_url,
            "AUDIO_TTS_OPENAI_API_KEY": key,
            "AUDIO_TTS_MODEL": "tts-1",              # what /v1/audio/speech accepts
            "AUDIO_TTS_VOICE": _OPEN_WEBUI_TTS_VOICE,
        })
    if rerank:
        pairs.update({
            # Route the RAG reranker at our /v1/rerank (Open WebUI POSTs the
            # Cohere/Jina shape here; "reranker" is the id /v1/models advertises).
            # Reranking only runs under hybrid search, so enable that too.
            "RAG_RERANKING_ENGINE": "external",
            "RAG_EXTERNAL_RERANKER_URL": f"{base_url}/rerank",
            "RAG_EXTERNAL_RERANKER_API_KEY": key,
            "RAG_RERANKING_MODEL": "reranker",
            "ENABLE_RAG_HYBRID_SEARCH": "true",
        })
    if default_model:
        pairs["DEFAULT_MODELS"] = default_model
    return pairs


def _launch_open_webui(a, *, exec_fn) -> int:
    binary = _find_binary("open-webui", a)
    base_url, models, default_model = _probe_target(a)
    # Route Open WebUI's audio engines at the server only when it advertises the
    # capability (the /v1/models markers the server sets behind --stt / --tts), so a
    # chat-only server leaves Open WebUI's built-in browser STT/TTS untouched.
    stt = any(m.get("stt") for m in models)
    tts = any(m.get("tts") for m in models)
    rerank = any(m.get("rerank") for m in models)

    webui_port = web_port_for("open-webui", a.port or _DEFAULT_PORT)
    data_dir = os.path.abspath(
        os.path.expanduser(a.config_path or _OPEN_WEBUI_DATA_HOME))
    pairs = build_open_webui_env(base_url, default_model=default_model,
                                 api_key=_client_key(a), port=webui_port, data_dir=data_dir,
                                 stt=stt, tts=tts, rerank=rerank)
    if tts and os.environ.get("AUDIO_TTS_VOICE"):
        # A voice the user exported wins over the Kokoro default.
        pairs["AUDIO_TTS_VOICE"] = os.environ["AUDIO_TTS_VOICE"]

    audio = [name for name, on in (("STT", stt), ("TTS", tts)) if on]
    summary = _summary("open-webui", base_url, models, default_model,
                       extra=((f", audio {'+'.join(audio)}" if audio else "")
                              + (", rerank" if rerank else "")))
    container = getattr(a, "container_mode", False)
    # A login can be turned off only before the first account exists, which
    # a new data folder promises.
    fresh = not _exists(Path(data_dir))
    if container:
        # The container session prints the address and opens the browser.
        print(summary + "\n" + _files_line(
            a, "Open WebUI keeps its chat history and database in", data_dir))
        auth = ("add WEBUI_AUTH=false to launch.container.clients.open-webui.env in your "
                "gmlx config, and launch again")
    else:
        print(summary + f"\n[launch] Open WebUI runs at http://localhost:{webui_port}, "
                        "which you open in a browser. It keeps its chat history and "
                        f"database in {data_dir}.")
        auth = "and run WEBUI_AUTH=false gmlx launch open-webui"
    if fresh:
        print("[launch] To use Open WebUI without a login, stop it before you create an "
              f"account, {auth}.")
    # `open-webui serve` binds via its `--port` CLI option (default 8080) and does
    # not read the PORT env var - so the port must be passed on the command line, or
    # the UI would try 8080 and collide with the gmlx server (crash: address in
    # use). PORT stays in `pairs` only for any self-URL construction Open WebUI does.
    argv = ["open-webui", "serve", "--port", str(webui_port)]
    if getattr(a, "container_mode", False):
        # Inside the guest it listens on loopback, where the entry relays it.
        argv[2:2] = ["--host", "127.0.0.1"]
    # Our single endpoint must win - drop any inherited plural OpenAI vars that
    # Open WebUI would otherwise merge ahead of it.
    return _finish(a, binary, argv, pairs,
                   drop=("OPENAI_API_BASE_URLS", "OPENAI_API_KEYS"),
                   exec_fn=exec_fn)


# dsh  (DeepSeek Harness - https://github.com/deepseek-ai/deepseek-harness)
# dsh composes its runtime from Cordis patch layers. The launch boots a
# gmlx-owned `gmlx` profile in the user's $DSH_HOME, created from the shipped
# `web` template on first launch, and passes its rows as a `--patch` overlay:
# the top layer, which dsh never saves, so no dsh file is edited. The provider row
# uses the llm-pi-ai adapter over /v1/chat/completions, the same pi-ai library
# the pi target drives. A patch layer replaces a row's whole config.
# --dsh-profile boots another profile with the same overlay.
_DSH_PROFILE = "gmlx"
_DSH_TEMPLATE = "web"
# Profiles dsh creates on first use. The stdio ones serve a program, so the
# launch prints their command and does not run them.
_DSH_SHIPPED = frozenset({"web", "headless", "acp", "sdk", "sdk-minimal"})
_DSH_STDIO = frozenset({"acp", "sdk", "sdk-minimal"})
_DSH_WEB_BUNDLE = "@deepseek-ai/dsh-web-app"
_DSH_MIN_VERSION = (0, 1, 7)
_DSH_WEB_PORT = 3080
_DSH_KEY_ENV = "GMLX_API_KEY"
_DSH_UPGRADE = CLIENT_INSTALL["dsh"][2]
# Route fallbacks for a served model /v1/models does not size.
_DSH_DEFAULT_WINDOW = 32768
_DSH_DEFAULT_MAX_TOKENS = 8192
# compaction-basic defaults (dsh-compaction-basic resolveCompactSpec).
_DSH_HEADROOM = 65536
_DSH_THRESHOLD_RATIO = 0.8
_DSH_RETAIN_RATIO = 0.16
_DSH_COMPACT_MIN_WINDOW = 8192
_DSH_SMALL_WINDOW = 16384
# A second route lists the same models with thinking off. pi-ai sends the
# z.ai `thinking` field there, which the server maps onto each template's
# switch, and a call that names no reasoning level gets {"type": "disabled"}.
# pi-ai sends the field only for a model that declares a level beyond off.
_DSH_NOTHINK_SUFFIX = "-nothink"
_DSH_NOTHINK_COMPAT = {**_PI_AI_COMPAT, "thinkingFormat": "zai",
                       "supportsReasoningEffort": False}
_DSH_NOTHINK_EFFORTS = {"off": None, "high": "high"}
# dsh's session-title-llm row, restated because a patch replaces the whole
# config. The title call names no reasoning level, so on the second route it
# runs with thinking off and fits dsh's 64 tokens.
_DSH_TITLE_ROW = {"targetWords": 5, "targetCjkCharacters": 10,
                  "maxInputBytes": 4096, "maxOutputTokens": 64,
                  "timeoutMs": 60000}


def dsh_compaction_resolves(window: int, out: int,
                            headroom: int = _DSH_HEADROOM) -> bool:
    """Whether dsh's compaction-basic can size a policy for a model: the
    pressure budget stays above zero and the retained tail below the
    threshold. dsh logs a warning and skips compaction for the model
    otherwise."""
    budget = window - out
    pressure = budget - headroom
    if pressure <= 0:
        return False
    threshold = int(min(window * _DSH_THRESHOLD_RATIO, pressure))
    return int(budget * _DSH_RETAIN_RATIO) < threshold


def dsh_headroom(window: int, out: int) -> int:
    """compaction-basic ``headroomTokens`` scaled to a window: a quarter of
    the message budget, within 4096 and dsh's default of 65536."""
    return min(_DSH_HEADROOM, max(4096, (window - out) // 4))


def build_dsh_overlay(base_url: str, models: list, *, default_model: str,
                      provider_id: str = _PROVIDER_ID) -> list:
    """The Cordis patch rows that route dsh at the gmlx server: two llm-pi-ai
    providers listing every served chat id, one as the server serves it and
    one with thinking off, the default model, session titles on the
    thinking-off route, and compaction policies sized to each model's window.
    An ``id@profile`` default is listed too, since pi-ai refuses a model id
    its route does not list. Pure - no IO."""
    nothink_id = provider_id + _DSH_NOTHINK_SUFFIX
    heads = {m["id"]: m for m in chat_models(models)}
    listed = [(m, m["id"], _display_name(m)) for m in heads.values()]
    if default_model not in heads:
        head = heads.get(default_model.rsplit("@", 1)[0])
        if head is not None:
            listed.append((head, default_model, default_model))
    entries, policies = [], []
    for m, model_id, name in listed:
        entry: dict = {"id": model_id, "name": name}
        capacity = model_capacity(m)
        if capacity:
            entry["contextWindow"], entry["maxTokens"] = capacity
        if m.get("vlm"):
            entry["input"] = ["text", "image"]
        entries.append(entry)
        window, out = capacity or (_DSH_DEFAULT_WINDOW, _DSH_DEFAULT_MAX_TOKENS)
        if window >= _DSH_COMPACT_MIN_WINDOW:
            policies.append({"provider": provider_id, "model": model_id,
                             "headroomTokens": dsh_headroom(window, out),
                             "maxTokens": out})
    policies += [dict(p, provider=nothink_id) for p in policies]

    def route(display_name: str, compat: dict, route_models: list) -> dict:
        return {
            "displayName": display_name,
            "api": "openai-completions",
            "baseURL": base_url,
            # dsh resolves a key only through an env var, even for a keyless server
            "apiKeyEnv": _DSH_KEY_ENV,
            "compat": dict(compat),
            "defaultContextWindow": _DSH_DEFAULT_WINDOW,
            "defaultMaxTokens": _DSH_DEFAULT_MAX_TOKENS,
            "models": route_models,
        }

    nothink_entries = [dict(e, reasoningEfforts=dict(_DSH_NOTHINK_EFFORTS))
                       for e in entries]
    providers = {
        provider_id: route("gmlx (local)", _PI_AI_COMPAT, entries),
        nothink_id: route("gmlx (thinking off)", _DSH_NOTHINK_COMPAT,
                          nothink_entries),
    }
    rows = [
        {"id": "llm-pi-ai", "config": {"providers": providers}},
        {"id": "agent-default-model",
         "config": {"provider": provider_id, "model": default_model}},
        {"id": "session-title-llm",
         "config": {**_DSH_TITLE_ROW, "provider": nothink_id,
                    "model": default_model}},
    ]
    if policies:
        # The web template disables this host row and runs a default-config
        # copy per agent preset, so these policies apply to headless and acp.
        rows.append({"id": "compaction-basic",
                     "config": {"modelPolicies": policies}})
    return rows


def _dsh_home() -> Path:
    """dsh's home: ``$DSH_HOME`` (blank counts as unset), else ``~/.dsh``."""
    raw = os.environ.get("DSH_HOME", "").strip() or "~/.dsh"
    home = Path(os.path.expanduser(raw))
    # In the private home a link is the guest's, so it is never resolved.
    return home if confine.active() else home.resolve()


def _dsh_version(binary: str) -> str | None:
    """The installed dsh version, or None when it cannot be read: the package
    manifest above the resolved bin script, else ``dsh --version``. Seam:
    monkeypatched in tests."""
    here = Path(os.path.realpath(binary)).parent
    for d in (here, *list(here.parents)[:3]):
        try:
            doc = json.loads((d / "package.json").read_text())
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and doc.get("name") == "@deepseek-ai/dsh":
            return str(doc.get("version") or "") or None
    try:
        done = subprocess.run([binary, "--version"], capture_output=True,
                              text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None


def _check_dsh_version(version: str | None) -> None:
    """Refuse a dsh older than the floor. A prerelease counts as its base
    version, and an unreadable version only warns."""
    floor = ".".join(map(str, _DSH_MIN_VERSION))
    m = re.match(r"\s*v?(\d+)\.(\d+)\.(\d+)", version or "")
    if m is None:
        print(f"[launch] cannot read the dsh version ({version!r}), and launch needs "
              f"dsh {floor} or newer. Upgrade an older dsh with:\n  {_DSH_UPGRADE}",
              file=sys.stderr)
        return
    if tuple(int(g) for g in m.groups()) < _DSH_MIN_VERSION:
        raise LaunchError(
            f"dsh {version} is too old, and launch needs dsh {floor} or newer. "
            f"Upgrade it with:\n  {_DSH_UPGRADE}")


def _dsh_runs_web_app(name: str, manifest: Path) -> bool:
    """Whether a dsh profile boots the web app: its manifest lists the web
    bundle. A profile without a readable manifest is judged by name. In a
    private home the manifest is the guest's, and a web profile gets a Mac
    port and a browser tab, so there only the name counts."""
    if confine.active():
        return name in (_DSH_PROFILE, _DSH_TEMPLATE)
    try:
        return _DSH_WEB_BUNDLE in json.loads(
            confine.read_text(manifest) or "")["dsh"]["profile"]["bundles"]
    except (OSError, ValueError, KeyError, TypeError, confine.ConfinedError):
        return name in (_DSH_PROFILE, _DSH_TEMPLATE)


def _check_dsh_profile_name(name: str, config_only: bool) -> None:
    if name in ("", ".", "..") or "/" in name or "\\" in name:
        raise LaunchError(f"{name!r} is not a dsh profile name")
    if name == "desktop":
        raise LaunchError("the desktop profile belongs to the DeepSeek "
                          "Harness desktop app")
    if name in _DSH_STDIO and not config_only:
        raise LaunchError(
            f"the {name} profile serves a program over stdio. Re-run with "
            f"--config-only and give that program the printed command.")


def _launch_dsh(a, *, exec_fn) -> int:
    profile = _DSH_PROFILE if a.dsh_profile is None else a.dsh_profile
    _check_dsh_profile_name(profile, a.config_only)
    binary = _find_binary("dsh", a)
    if not a.config_only and not getattr(a, "container_mode", False):
        _check_dsh_version(_dsh_version(binary))
    base_url, models, default_model = _probe_target(a)
    assert default_model is not None                 # check_model_choice made sure
    by_id = {m["id"]: m for m in chat_models(models)}
    head = by_id.get(default_model) or by_id.get(default_model.rsplit("@", 1)[0])
    if head is None:
        raise LaunchError(
            f"{default_model!r} is a service model, not a chat model: pass "
            f"--model with one of {sorted(by_id)}")

    profile_dir = _dsh_home() / "profiles" / profile
    manifest = profile_dir / "package.json"
    create = profile == _DSH_PROFILE and not _exists(manifest)
    if create and _exists(profile_dir):
        raise LaunchError(
            f"{profile_dir} exists but is not a dsh profile (no package.json). "
            f"Remove or rename it, then re-run to create the profile.")
    if profile not in _DSH_SHIPPED | {_DSH_PROFILE} and not _exists(manifest):
        raise LaunchError(
            f"dsh has no profile {profile!r} ({manifest} is missing). Set it "
            f"up with dsh first, then re-run.")
    web = _dsh_runs_web_app(profile, manifest)

    rows = build_dsh_overlay(base_url, models, default_model=default_model,
                             provider_id=a.provider_id)
    out = Path(os.path.expanduser(
        a.config_path or f"{_CONFIG_HOME}/dsh/gmlx.cordis.yml"))
    _write_text_atomic(out, yaml.safe_dump(rows, sort_keys=False))

    argv = ["dsh", "--profile", profile]
    if create:
        argv += ["--from-default-profile", _DSH_TEMPLATE]
    argv += ["--patch", str(out)]
    if web and getattr(a, "container_mode", False):
        # The Mac opens the browser; the guest has none.
        argv += ["--no-open", "--port", str(web_port_for("dsh", a.port or _DEFAULT_PORT))]
    elif web and (a.port or _DEFAULT_PORT) == _DSH_WEB_PORT:
        argv += ["--port", str(_DSH_WEB_PORT + 1)]
    key = _client_key(a) or _PROVIDER_ID             # placeholder: no auth

    print(_summary("dsh", base_url, models, default_model)
          + "\n" + _files_line(a, "wrote", out))
    if create:
        print(f"[launch] note: the first launch creates the dsh profile "
              f"{profile_dir} from the {_DSH_TEMPLATE} template")
    window, out_cap = (model_capacity(head)
                       or (_DSH_DEFAULT_WINDOW, _DSH_DEFAULT_MAX_TOKENS))
    if window < _DSH_SMALL_WINDOW:
        print(f"[launch] note: {default_model} has a {window}-token context, "
              f"which is small for an agent, so expect frequent compaction")
    if web and not dsh_compaction_resolves(window, out_cap):
        # The web app compacts at dsh's default headroom only.
        need = out_cap + math.ceil(_DSH_HEADROOM / (1 - _DSH_RETAIN_RATIO))
        print(f"[launch] note: the dsh web app compacts {default_model} only "
              f"after the server reports an overflow. Automatic compaction "
              f"there needs a {need}-token context, and this model has "
              f"{window}")
    return _finish(a, binary, argv, {_DSH_KEY_ENV: key}, exec_fn=exec_fn)


def web_port_for(harness: str, server_port: int) -> int | None:
    """The port a browser app listens on: its usual one, or the next when
    the gmlx server holds it. None for the terminal clients."""
    usual = {"open-webui": _OPEN_WEBUI_PORT, "dsh": _DSH_WEB_PORT}.get(harness)
    if usual is None:
        return None
    return usual + 1 if int(server_port) == usual else usual


# dispatch
_HARNESSES = {
    "opencode": _launch_opencode,
    "pi": _launch_pi,
    "omp": _launch_omp,
    "hermes": _launch_hermes,
    "goose": _launch_goose,
    "claude-code": _launch_claude_code,
    "aichat": _launch_aichat,
    "elia": _launch_elia,
    "open-webui": _launch_open_webui,
    "dsh": _launch_dsh,
}


def _default_exec(binary: str, argv: list, env: dict) -> int:
    """Replace this process with the harness (so signals/TTY are the harness's).
    Seam: tests pass a recording fake instead."""
    os.execvpe(binary, argv, env)        # never returns on success
    return 127                           # unreachable; satisfies the type


# start-if-down orchestration (decision logic; the harness builders stay untouched)
def _server_ready(base_url: str, api_key: str | None = None) -> bool:
    """True iff the server answers ``/health`` and ``/v1/models``. A 401 on the
    models counts, since that server is up and needs a key. A server with no
    models yet is up too, and the model probe says what to do about it. Residency-independent, short-timeout so
    polling stays responsive. Mirrors :func:`lifecycle._ready` through the
    ``_http_get_json`` seam."""
    root = _server_root(base_url)
    try:
        _http_get_json(root + "/health", timeout=1.5)
    except (urllib.error.URLError, OSError, ValueError):
        return False
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    try:
        payload = _http_get_json(base_url.rstrip("/") + "/models", timeout=1.5,
                                 headers=headers)
    except urllib.error.HTTPError as e:
        return e.code == 401
    except (urllib.error.URLError, OSError, ValueError):
        return False
    return isinstance(payload, dict) and isinstance(payload.get("data"), list)


def _auth_required(base_url: str) -> bool:
    """True iff ``/v1/models`` answers 401 without a key."""
    try:
        _http_get_json(base_url.rstrip("/") + "/models", timeout=1.5)
    except urllib.error.HTTPError as e:
        return e.code == 401
    except (urllib.error.URLError, OSError, ValueError):
        return False
    return False


def _warn_if_stale_server(host: str, port) -> None:
    """A reused managed server that predates a source change on disk will
    fail lazy imports mid-request with an opaque 500; say so at connect
    time instead. Compares the runfile's boot-time source stamp against
    the tree now - only managed servers carry one, so a foreign server (or
    a pre-stamp runfile) stays silent."""
    import gmlx.serve.lifecycle as lifecycle

    if lifecycle.source_changed(lifecycle.read_run(host, port)):
        print(f"[launch] the server at http://{host}:{port} started before "
              "the gmlx source on disk changed, so a request may fail with an "
              "import error. Run gmlx restart to load the new code.",
              file=sys.stderr)


def _discover_config():
    """The first existing default-location config, loaded. Returns ``(cfg, cfg_path)``:
    ``(None, None)`` if none exists; ``(None, path)`` if it exists but won't load
    (malformed != absent)."""
    import gmlx.config as config
    for p in config.default_config_paths():
        if p.exists():
            # Absolute, so the runfile names the file wherever it is read.
            cfg_path = os.path.abspath(p)
            try:
                return config.load_config(p), cfg_path
            except config.ConfigError:
                return None, cfg_path
    return None, None


# The largest config file launch reads a key from.
_CONFIG_READ_MAX = 1 << 20


def _served_config(host: str, port) -> tuple[str | None, dict] | None:
    """The config file that the managed server at ``host:port`` runs with,
    as its runfile records it, and the file's YAML document. A server that
    started without a config file gives ``(None, {})``. None when no running
    or launchd-managed server records a full path, or the file does not read.
    The read follows no link and never waits on a file that is not a regular
    file, since the file can be in a folder that a container client shares."""
    import gmlx.serve.lifecycle as lifecycle

    run = lifecycle.read_run(host, port) or {}
    if run.get("managed_by") != "launchd" and not lifecycle.pid_alive(run.get("pid")):
        return None
    path = run.get("config_abspath")
    if not path:
        return None, {}
    if not isinstance(path, str) or not os.path.isabs(path):
        return None
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as f:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode) or st.st_size > _CONFIG_READ_MAX:
                return None
            doc = yaml.safe_load(f.read(_CONFIG_READ_MAX))
    except (OSError, ValueError, yaml.YAMLError):
        return None
    return path, doc if isinstance(doc, dict) else {}


def _runfile_key(host: str, port) -> str | None:
    """``server.api_key`` from the config file that the managed server at
    ``host:port`` records in its runfile, as ``gmlx serve --config`` started
    it. None when there is no such file or key."""
    served = _served_config(host, port)
    srv = served[1].get("server") if served else None
    key = srv.get("api_key") if isinstance(srv, dict) else None
    return str(key) if key else None


def _prompt_cache_off(host: str, port, model_id: str | None) -> bool:
    """Whether the managed server at ``host:port`` runs ``model_id`` without
    the prompt cache: its config leaves ``server.cache.enabled`` off and the
    model's ``overrides`` do not turn it on, or it has no config file. False
    when launch cannot tell."""
    served = _served_config(host, port)
    if served is None:
        return False
    doc = served[1]

    def enabled(block) -> bool | None:
        cache = block.get("cache") if isinstance(block, dict) else None
        value = cache.get("enabled") if isinstance(cache, dict) else None
        return value if isinstance(value, bool) else None

    models = doc.get("models")
    model = (models.get(model_id.rsplit("@", 1)[0])
             if isinstance(models, dict) and model_id else None)
    own = enabled(model.get("overrides")) if isinstance(model, dict) else None
    return not (own if own is not None else enabled(doc.get("server")))


def _shown_path(a, path) -> str:
    """``path`` as a status line names it. In container mode the file is in
    the private home, which is the client's home in the container, so it is
    named from ``~``."""
    if getattr(a, "container_mode", False):
        rel = os.path.relpath(str(path), os.path.expanduser("~"))
        if rel != ".." and not rel.startswith("../"):
            return "~" if rel == "." else f"~/{rel}"
    return str(path)


def _files_line(a, verb: str, *paths) -> str:
    """The status line that names the files a client function wrote."""
    where = " in the private home" if getattr(a, "container_mode", False) else ""
    return (f"[launch] {verb} " + " and ".join(_shown_path(a, p) for p in paths)
            + where)


def _server_key(host: str, port) -> str | None:
    """The key to try on a server at ``host:port`` that needs one when
    launch got none: the key in the config the running server records, else
    the key in the user-level config."""
    key = _runfile_key(host, port)
    if key is None:
        cfg, _path = _discover_config()
        key = getattr(cfg, "api_key", None)
    return key


def _human_size(n: int | None) -> str | None:
    """A human byte-count string (e.g. ``16.8 GB``); ``None`` for falsy/unknown."""
    from gmlx.serve.lifecycle import human_gb

    return human_gb(n) if n else None


def _shard_siblings(path: str) -> list:
    """All existing shards of a ``*-00001-of-00003.gguf`` set if ``path`` is one;
    else ``[path]``. Size is cosmetic, so gaps are tolerated."""
    from gmlx.load.preflight import shard_names
    d, base = os.path.split(path)
    try:
        names = shard_names(base)
    except ValueError:
        return [path]
    if len(names) == 1:
        return [path]
    existing = [p for n in names if os.path.exists(p := os.path.join(d, n))]
    return existing or [path]


def _model_size_bytes(cfg, model_id) -> int | None:
    """Best-effort on-disk size of a configured model's GGUF (sums shards). ``None`` if
    it can't be located - size is cosmetic, never raise."""
    try:
        m = cfg.models.get(model_id)
        if m is None or not getattr(m, "path", None):
            return None
        raw = os.path.expanduser(m.path)
        candidates = [raw]
        if not os.path.isabs(raw):
            for d in getattr(cfg, "model_dirs", None) or []:
                candidates.append(os.path.join(os.path.expanduser(d), raw))
        for c in candidates:
            if os.path.isfile(c):
                return sum(os.path.getsize(s) for s in _shard_siblings(c))
        return None
    except OSError:
        return None


def _preload_descr(cfg):
    """``(preload_id, label)`` for the model the server preloads at startup (pin ->
    default -> sole model), label carrying a human size when the file is locatable;
    ``(None, None)`` when nothing preloads. Cosmetic - never raises."""
    from gmlx.serve.server import _preload_id
    try:
        pid = _preload_id(cfg)
    except Exception:
        return None, None
    if not pid:
        return None, None
    size = _human_size(_model_size_bytes(cfg, pid))
    return pid, (f"{pid} ({size})" if size else pid)


def _guide_to_init(rerun: str | None = None) -> None:
    """Setup guidance printed when nothing is running and no config exists.
    ``rerun`` names the verb in the prefix for non-launch verbs (e.g. talk)."""
    tag = f"[{rerun or 'launch'}]"
    print(
        f"{tag} no gmlx server is running, and no config was found in a default\n"
        "  location (~/.config/gmlx/gmlx.yaml, ~/.gmlx.yaml).\n"
        "  Set one up first:\n"
        "    gmlx init --models-dir <DIR>     # scaffold ~/.config/gmlx/gmlx.yaml from your GGUFs\n"
        "    gmlx init --from-hf-cache        # ...or from models already in your HF cache\n"
        "  then run the same command again.\n"
        "  Already have a server? Add --base-url URL to the command.",
        file=sys.stderr)


def _autostart(*, base, host, port, api_key, cfg, cfg_path, start_timeout, config_only):
    """Spawn a background server from ``cfg_path`` and poll until it answers, showing a
    spinner (named with the preloaded model + size when the config preloads one). Only a
    dead child is a hard failure; with a falsy ``start_timeout`` we wait while the child
    lives (Ctrl-C bails, leaving it running). Returns ``(rc, ready, preload_id)`` - the
    caller execs the harness iff ``ready``."""
    import time
    import gmlx.serve.lifecycle as lifecycle
    import gmlx.spinner as spinner

    preload_id, label = _preload_descr(cfg)
    spin_text = (f"starting the server and loading {label}" if preload_id
                 else f"starting the server from {cfg_path}")
    spawned = lifecycle.start_background_nowait(
        ["--config", cfg_path], host=host, port=port,
        config_abspath=cfg_path, api_key=api_key)
    if spawned is None:                              # refused: a server already holds it
        return ((0, True, preload_id) if _server_ready(base, api_key)
                else (EXIT_TEMPFAIL, False, preload_id))
    proc, log = spawned

    outcome = None                                   # set inside the spinner, acted on after
    try:
        with spinner.Spinner(spin_text):
            start = time.monotonic()
            while outcome is None:
                if proc.poll() is not None:
                    outcome = "died"
                elif _server_ready(base, api_key):
                    outcome = "ready"
                elif start_timeout and time.monotonic() - start > start_timeout:
                    outcome = "timeout"
                else:
                    time.sleep(0.3)
    except KeyboardInterrupt:
        print("[launch] interrupted. The server keeps starting in the background. "
              "Check it with gmlx status, or stop it with gmlx stop.", file=sys.stderr)
        return (130, False, preload_id)

    if outcome == "ready":
        if not config_only and getattr(cfg, "menubar", True) \
                and lifecycle.gui_session_available():
            lifecycle.start_menubar(auto=True)  # one machine-wide bar; tracks the primary
        return (0, True, preload_id)
    if outcome == "timeout":
        print(f"[launch] the server is still starting after {start_timeout:.0f} s. "
              "Read its log with gmlx logs, or stop it with gmlx stop.", file=sys.stderr)
        return (EXIT_TEMPFAIL, False, preload_id)
    tail = lifecycle._log_tail(log, 40).rstrip()     # died
    if lifecycle.report_port_in_use(tail, host, port, tag="[launch]"):
        return (EXIT_TEMPFAIL, False, preload_id)
    print(f"[launch] server exited (code {proc.returncode}) before it was ready.",
          file=sys.stderr)
    if tail and tail != "(no log)":
        print(tail, file=sys.stderr)
    return (EXIT_UNAVAILABLE, False, preload_id)


def _ensure_server(a) -> int | None:
    """Start-if-down: resolve the endpoint onto ``a`` and, when nothing is reachable,
    auto-start a background server from a default-location config (or guide to `init`).
    Returns ``None`` to proceed to the harness, or an int exit code to return directly."""
    import gmlx.serve.lifecycle as lifecycle

    if a.base_url or a.host or a.port:
        url_host = url_port = None
        if a.base_url:
            # Harness clients (talk STT/TTS especially) expect the /v1 base;
            # accept a bare http://host:port like the default path builds.
            from gmlx.talk.client import ensure_v1_base
            a.base_url = ensure_v1_base(a.base_url)
            # The URL's own bind, not 8080: harnesses derive their listen port
            # from a.port and would collide with the server otherwise.
            split = urllib.parse.urlsplit(a.base_url)
            url_host = split.hostname
            try:
                url_port = split.port
            except ValueError:
                # A malformed port (http://h:99999) fails cleanly downstream
                # when the probe can't connect; don't traceback here.
                url_port = None
        host0 = a.host or url_host or _DEFAULT_HOST
        port0 = int(a.port or url_port or _DEFAULT_PORT)
    else:
        # No explicit endpoint: resolve like status/stop/ps do (the single
        # managed server, else the config's host/port, else 8080) so launch
        # never silently binds a harness to whatever answers on 8080.
        host0, port0 = lifecycle.auto_target(None, None)
    base0 = a.base_url or f"http://{host0}:{port0}/v1"
    if _server_ready(base0, a.api_key):              # up: fast path, no engine import
        _warn_if_stale_server(host0, port0)
        if a.api_key is None and not a.base_url and _auth_required(base0):
            # Without a key the launch fails, so a key from a config is the
            # only one to try.
            a.api_key = _server_key(host0, port0)
        a.base_url, a.host, a.port = base0, host0, port0
        return None

    if a.base_url:                                   # explicit endpoint: never auto-start
        a.host, a.port = host0, port0
        return None                                  # the harness probe raises the usual error

    cfg, cfg_path = _discover_config()
    if cfg_path is None:
        _guide_to_init(getattr(a, "rerun_label", None))
        return EXIT_CONFIG
    if cfg is None:
        print(f"[launch] the config {cfg_path} does not load, so launch does not start a "
              "server. Fix the config, or pass --base-url.", file=sys.stderr)
        return EXIT_CONFIG

    host = a.host or cfg.host
    port = int(a.port or cfg.port)
    key = a.api_key or getattr(cfg, "api_key", None)
    base = f"http://{host}:{port}/v1"
    a.base_url, a.host, a.port, a.api_key = base, host, port, key
    if _server_ready(base, key):                     # configured server already up (e.g. non-8080)
        _warn_if_stale_server(host, port)
        return None

    if a.no_start:
        print(f"[launch] no server answers at {base}. Start one with gmlx serve, or "
              "drop --no-start so that launch starts it.", file=sys.stderr)
        return EXIT_UNAVAILABLE

    if (lifecycle.read_run(host, port) or {}).get("managed_by") == "launchd":
        print(f"[launch] the launchd server for {host}:{port} may be restarting. "
              "Check it with gmlx status, and launch again in a moment.", file=sys.stderr)
        return EXIT_TEMPFAIL

    rc, ready, preload_id = _autostart(
        base=base, host=host, port=port, api_key=key, cfg=cfg, cfg_path=cfg_path,
        start_timeout=a.start_timeout, config_only=a.config_only)
    if not ready:
        return rc
    if a.config_only:
        print(f"[launch] left a background server running at {base}. Stop it with "
              "gmlx stop.", file=sys.stderr)
    elif not preload_id:
        print(f"[launch] the server is up at {base} with no model preloaded, so the "
              "first request loads one and takes longer.", file=sys.stderr)
    return None


def cmd_launch(argv: list, *, exec_fn=_default_exec,
               prog: str = "gmlx launch") -> int:
    # The macOS menu-bar monitor rides under `launch` but carries its own option set,
    # so it's dispatched before the harness parser ever sees it.
    if argv and argv[0] == "menubar":
        from .menubar import cmd_menubar
        return cmd_menubar(argv[1:], prog=f"{prog} menubar")
    import gmlx.config as config
    config.note_local_config(argv)
    # Everything after the first `--` goes to the client untouched; argparse
    # would reject the client's own flags.
    argv_given = list(argv)
    passthrough: list = []
    if "--" in argv:
        cut = argv.index("--")
        argv, passthrough = argv[:cut], argv[cut + 1:]

    import textwrap
    ap = argparse.ArgumentParser(
        prog=prog,
        # The epilog holds install commands and a URL, which must not wrap.
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.fill(
            "Configure a client for a gmlx server and run it, on the Mac or in an Apple "
            "container. Launch starts the server from the default config when none "
            "answers, and never installs a client on the Mac.", 78),
    )
    ap.add_argument("harness", nargs="?", choices=sorted(_HARNESSES),
                    help="The client to configure and run: a coding agent, a chat "
                         "TUI (aichat, elia) or a web app (open-webui, dsh). Without "
                         "it, launch prints this help.")
    ap.add_argument("--model", default=None,
                    help="Model id to make the client's default. It must be served "
                         "(default: the server's default-marked model).")
    ap.add_argument("--base-url", default=None,
                    help="Server OpenAI base URL (default http://HOST:PORT/v1).")
    ap.add_argument("--host", default=None,
                    help="Server host (default: the single managed server if "
                         f"there's one, else the config's, else {_DEFAULT_HOST}).")
    ap.add_argument("--port", type=int, default=None,
                    help="Server port (default: the single managed server if "
                         f"there's one, else the config's, else {_DEFAULT_PORT}).")
    ap.add_argument("--api-key", default=None, metavar="KEY",
                    help="API key the client sends, which must match the "
                         "server's server.api_key. Default: the server.api_key of "
                         "the config the running server was started with, else of "
                         "the default config. With --base-url, or when no config "
                         "sets a key, tools get a placeholder.")
    ap.add_argument("--provider-id", default=_PROVIDER_ID,
                    help=f"Provider id written into the client's config "
                         f"(default {_PROVIDER_ID}).")
    ap.add_argument("--config-path", default=None,
                    help=f"Write the client's config at this path. By default it goes "
                         f"under {_CONFIG_HOME} for opencode, aichat, elia and dsh, "
                         f"into the client's own files for pi ({_PI_AGENT_HOME}), omp "
                         f"({_OMP_AGENT_HOME}) and goose ({_GOOSE_CONFIG}), and into "
                         f"{_OPEN_WEBUI_DATA_HOME} for open-webui. claude-code writes no "
                         f"config file. hermes refuses it, and so does container mode, "
                         f"where the config goes in the client's private home.")
    ap.add_argument("--config-only", action="store_true",
                    help="Write the client's config and print the command instead "
                         "of running it. In container mode it is a dry run that prints "
                         "the container run command.")
    ap.add_argument("--no-start", action="store_true",
                    help="Never start a server. Without a reachable server, launch "
                         "stops with an error.")
    ap.add_argument("--start-timeout", type=float, default=0.0, metavar="S",
                    help="Cap the wait for an auto-started server to become ready. "
                         "The default 0 waits as long as the server process runs, and "
                         "Ctrl-C stops the wait.")
    ap.add_argument("--no-keep", action="store_true",
                    help="Let --model unload while idle. By default launch asks the "
                         "server to keep it loaded.")
    ap.add_argument("--dsh-profile", default=None, metavar="NAME",
                    help=f"dsh only: boot this dsh profile with the gmlx overlay "
                         f"instead of the {_DSH_PROFILE} profile, for example "
                         f"headless or a terminal UI profile you set up.")
    box = ap.add_argument_group(
        "container mode",
        textwrap.fill("Run the client in an Apple container that sees only the shared "
                      "folders. The launch.container config block sets the defaults.",
                      76))
    box.add_argument("--container", dest="container", action="store_const", const=True,
                     default=None,
                     help="Run the client in an Apple container, whatever the config says.")
    box.add_argument("--no-container", dest="container", action="store_const", const=False,
                     help="Run the client on the Mac, whatever the config says.")
    box.add_argument("--mount", action="append", default=[], metavar="PATH[:DST][:ro]",
                     help="Share another folder with the container. Repeatable, and "
                          "added to the configured mounts.")
    from gmlx.container.settings import NO_CWD_CLIENTS
    box.add_argument("--mount-cwd", dest="mount_cwd", action="store_const", const=True,
                     default=None,
                     help="Share the current folder with the container. Without this "
                          "flag or launch.container.mount_cwd, every client shares it "
                          f"except {' and '.join(sorted(NO_CWD_CLIENTS))}.")
    box.add_argument("--no-mount-cwd", dest="mount_cwd", action="store_const", const=False,
                     help="Do not share the current folder with the container.")
    box.add_argument("--image", default=None, metavar="REF",
                     help="Run this image instead of the configured or shipped one.")
    box.add_argument("--rebuild", action="store_true",
                     help="Rebuild the client's image, or pull an image: reference again.")
    box.add_argument("--reseed", action="store_true",
                     help="Copy every seed file into the private home again, replacing "
                          "the copies there.")
    box.add_argument("--network", choices=("default", "none"), default=None,
                     help="Set the container's network. With default, it reaches the "
                          "internet and your local network, and with none, only the "
                          "gmlx server and the forwarded ports. launch.container.network "
                          "sets the default.")
    box.add_argument("--shell", action="store_true",
                     help="Open a shell in the container instead of the client, or in "
                          "the running session's container.")
    box.add_argument("--remove-home", action="store_true",
                     help="Remove the private home this launch would use, after a "
                          "question, and start nothing.")
    ap.epilog = _help_epilog(_named_client(ap, argv))
    a = ap.parse_args(argv)
    a.passthrough = passthrough

    # Bare `gmlx launch` -> long-form help, not an argparse "required" error.
    if a.harness is None and "--" in argv_given:
        ap.error("name the client before --, as in: gmlx launch pi -- --help")
    if a.harness is None:
        ap.print_help()
        return 0
    if a.dsh_profile is not None and a.harness != "dsh":
        ap.error("--dsh-profile applies only to dsh")
    if "--container" in argv and "--no-container" in argv:
        ap.error("--container and --no-container cannot go together")
    from gmlx.config import ConfigError
    from gmlx.container.text import printable_lines
    from .launch_container import container_mode, run_container
    try:
        in_container, launch_cfg = container_mode(a, ap)
    except ConfigError as e:
        sys.stdout.flush()
        print(printable_lines(f"[launch] {e}"), file=sys.stderr)
        return EXIT_CONFIG
    if in_container:
        return run_container(a, launch_cfg, exec_fn=exec_fn)

    try:
        # A missing client stops the launch before the server starts or keeps
        # a model for a client that cannot run.
        _find_binary(a.harness, a)
        rc = _ensure_server(a)
        if rc is not None:
            return rc
        if a.model and not a.no_keep and not a.config_only:
            # Validate --model before keeping: an unknown id must produce the
            # single refusal (raised again inside the harness fn), never a
            # keep line followed by that refusal.
            base = a.base_url or f"http://{a.host}:{a.port}/v1"
            _pick_default(probe_models(base, a.api_key, a.harness), a.model)
            _keep_model(a)                       # server is reachable here; best-effort
        return _HARNESSES[a.harness](a, exec_fn=exec_fn)
    except LaunchError as e:
        # A message can name a file or value read from a client's config.
        sys.stdout.flush()
        print(printable_lines(f"[launch] {e}"), file=sys.stderr)
        return e.code
