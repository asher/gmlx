"""``gmlx launch <harness>`` - point an external coding harness at a gmlx
server's OpenAI-compatible API, then exec it (starting the server if needed).

Mirrors ``ollama launch``, with one deliberate omission: **no auto-install**. The
harness must already be on ``PATH``; if it isn't we print its install pointer and
exit (we never run a package manager). We probe the server's ``/health`` +
``/v1/models`` over HTTP, render the harness's *own* config - into our own config
namespace, never mutating the user's harness files - and exec the harness pointed
at our ``/v1`` endpoint.

By default, if no server is reachable, ``launch`` auto-starts one in the background
from a default-location config (``./gmlx.yaml``, ``~/.config/gmlx/gmlx.yaml``,
``~/.gmlx.yaml``), polling with a spinner until it answers - when the config
preloads a model the wait spans that model's load. With no config anywhere it points
the user at ``gmlx init``. ``--no-start`` opts out; an explicit ``--base-url`` is
never auto-started.

Supports nine surfaces, each via its own native config - coding harnesses (opencode,
pi, omp, claude-code), agent runtimes (hermes, goose), two chat-focused terminal UIs
that are *not* coding harnesses (aichat, elia), and a browser chat app (open-webui):

- **opencode** - a custom OpenAI-compatible provider, injected via ``OPENCODE_CONFIG``
  so the user's ``~/.config/opencode`` is untouched.
- **pi** - merged non-destructively into ``~/.pi/agent/{models,settings}.json``. pi
  has no documented config-injection env var, so this harness edits the user's own
  files; existing providers/settings are preserved.
- **omp** (oh-my-pi) - merged non-destructively into ``~/.omp/agent/{models,config}.yml``
  (YAML). Provider goes in ``models.yml``; the default model is pinned via
  ``modelRoles.default`` in ``config.yml``. Existing providers/roles are preserved.
- **hermes** (NousResearch hermes-agent) - the user's ``~/.hermes/config.yaml`` is
  read, merged with our ``inference``/``providers.custom`` block, written into our
  namespace, and injected via ``HERMES_CONFIG`` (plus ``CUSTOM_BASE_URL``) - the
  user's file is never touched. Hermes refuses models with <64k context at startup,
  so launch prints that requirement.
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
  (``pipx install open-webui --python python3.12``; Python 3.11/3.12 only, not 3.13).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8080
_PROVIDER_ID = "gmlx"
_CONFIG_HOME = "~/.config/gmlx"          # our namespace, never the harness's


class LaunchError(RuntimeError):
    """A user-facing launch failure (server down, harness missing, bad --model).
    Carries a clean message; the CLI prints it and exits non-zero."""


def _find_binary(name: str, a, install_hint: str, *,
                 label: str | None = None):
    """``shutil.which`` + the standard not-on-PATH refusal shared by every
    harness (skipped under --config-only, which only writes the config)."""
    binary = shutil.which(name)
    if binary is None and not a.config_only:
        raise LaunchError(
            f"{label or name} is not on PATH (launch does not auto-install).\n"
            f"{install_hint}")
    return binary


def _probe_target(a, *, require_default: str | None = None):
    """The shared harness preamble: resolve the server base URL, probe its
    served models, and pick the default. ``require_default`` names the harness
    setting that makes a default model mandatory (e.g. ``GOOSE_MODEL``)."""
    base_url = a.base_url or f"http://{a.host}:{a.port}/v1"
    models = probe_models(base_url, a.api_key)
    default_model = _pick_default(models, a.model)
    if require_default and not default_model:
        raise LaunchError(
            f"{a.harness} needs a default model ({require_default}): pass "
            f"--model, or mark one default in the server config.")
    return base_url, models, default_model


def _summary(name: str, base_url: str, models: list,
             default_model: str | None, extra: str = "") -> str:
    """The first ``[launch]`` status line every harness prints."""
    return (f"[launch] {name} -> {base_url}  ({len(models)} model(s)"
            + (f", default {default_model}" if default_model else "")
            + extra + ")")


# server probe (HTTP only - launch never imports the model stack)
def _http_get_json(url: str, timeout: float = 5.0, headers: dict | None = None):
    """GET ``url`` and parse JSON (lifecycle.get_json). Seam: monkeypatched in
    tests so the probe is exercised without a live server."""
    from .lifecycle import get_json

    return get_json(url, timeout=timeout, headers=headers)


def _http_post_json(url: str, body: dict, *, api_key: str | None = None,
                    timeout: float = 3.0):
    """POST ``body`` as JSON and parse the reply (lifecycle.post_json). Seam:
    monkeypatched in tests."""
    from .lifecycle import post_json

    return post_json(url, body, api_key=api_key, timeout=timeout)


def _keep_model(a) -> None:
    """Best-effort: ask the server to keep ``a.model`` resident through its idle-TTL
    reaper (it stays LRU-evictable) and warm-load it, so a coding session's model
    isn't idle-unloaded mid-use. Fire-and-forget - the server warms in the background
    and the harness execs immediately; an older server without ``/v1/keep`` just warns."""
    base = a.base_url or f"http://{a.host}:{a.port}/v1"
    url = base.rstrip("/") + "/keep"
    try:
        _http_post_json(url, {"model": a.model, "warm": True}, api_key=a.api_key)
        print(f"[launch] keeping {a.model} resident for this session "
              f"(idle-TTL exempt, still pressure-evictable; --no-keep to opt out)")
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
                print(f"[launch] keep skipped: server does not serve {a.model!r}")
            else:
                print(f"[launch] note: server has no /v1/keep route - {a.model} "
                      f"may be idle-unloaded after its TTL (update the server "
                      f"to enable keep)")
        elif e.code == 400 and message:
            # A bad profile / ambiguous default carries an actionable message
            # (e.g. "unknown profile ... available: [...]"); surface it.
            print(f"[launch] keep skipped: {message}")
        else:
            print(f"[launch] keep request failed ({e}); continuing")
    except (urllib.error.URLError, OSError, ValueError) as e:
        print(f"[launch] keep request failed ({e}); continuing")


def _server_root(base_url: str) -> str:
    """The server root for ``/health`` (strip a trailing ``/v1``)."""
    from .lifecycle import server_root

    return server_root(base_url)


def probe_models(base_url: str, api_key: str | None = None) -> list:
    """Confirm the server is up (``/health``) and return its ``/v1/models`` ``data``
    list. Raises :class:`LaunchError` with a start-the-server hint if unreachable."""
    root = _server_root(base_url)
    try:
        _http_get_json(root + "/health", timeout=5.0)
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise LaunchError(
            f"no gmlx server reachable at {root} ({e}).\n"
            f"Start one first, e.g.:  gmlx serve --config <your.yaml>")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    try:
        payload = _http_get_json(base_url.rstrip("/") + "/models", timeout=5.0,
                                 headers=headers)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise LaunchError(
                "server requires an API key (it runs with --api-key) - pass the "
                "same key:  launch <harness> --api-key <key>")
        raise LaunchError(f"server is up but /v1/models failed: {e}")
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise LaunchError(f"server is up but /v1/models failed: {e}")
    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, list):     # every consumer indexes m["id"]
        data = [m for m in data if isinstance(m, dict) and m.get("id")]
    if not data:
        raise LaunchError(f"server at {root} reports no models")
    return data


def _pick_default(models: list, requested: str | None) -> str | None:
    """The model id to make the harness default: an explicit ``--model`` (validated
    against the served ids), else the server's ``default``-marked id, else None.
    An ``id@profile`` form passes with a served head - the profile half is the
    server's to validate (an unknown one 400s, listing the valid names)."""
    ids = [m["id"] for m in models]
    if requested:
        head = requested.rsplit("@", 1)[0]
        if requested not in ids and head not in ids:
            raise LaunchError(
                f"--model {requested!r} is not served; available: {sorted(ids)}")
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
    binary = _find_binary(
        "opencode", a,
        "Install it, then re-run - see https://opencode.ai/docs/  "
        "(e.g. `npm i -g opencode-ai`, `brew install sst/tap/opencode`, or "
        "`curl -fsSL https://opencode.ai/install | bash`).")
    base_url, models, default_model = _probe_target(a)
    cfg = build_opencode_config(base_url, models, provider_id=a.provider_id,
                                default_model=default_model, api_key=a.api_key)

    out = Path(os.path.expanduser(a.config_path or f"{_CONFIG_HOME}/opencode.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(out, json.dumps(cfg, indent=2) + "\n")

    print(_summary("opencode", base_url, models, default_model)
          + f"\n[launch] wrote {out}")
    if a.config_only:
        print(f"[launch] run it with:  OPENCODE_CONFIG={out} opencode")
        return 0

    env = dict(os.environ, OPENCODE_CONFIG=str(out))
    return exec_fn(binary, ["opencode"], env)


# pi  (https://github.com/parsfaghfouri/pi - "ollama launch pi")
# pi has no documented config-injection env var, so this is the one harness that
# merges into the user's own files (`~/.pi/agent/{models,settings}.json`). We
# preserve every other provider/setting the user already has.
_PI_AGENT_HOME = "~/.pi/agent"


def _write_text_atomic(path: Path, text: str) -> None:
    """tmp + rename. Several of these targets are another tool's live config -
    a crash or full disk mid-write must not leave it truncated."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _read_config_text(path: Path) -> str:
    """The read half of the edit-in-place flows, with the same refusal contract
    as the parsers: unreadable/binary -> LaunchError, never a traceback."""
    try:
        return path.read_text().strip()
    except UnicodeDecodeError:
        raise LaunchError(f"{path} is not a text file; refusing to "
                          f"overwrite it")
    except OSError as e:
        raise LaunchError(f"cannot read {path}: {e}")


def _load_json(path: Path) -> dict:
    """Read a JSON object from ``path``; ``{}`` if it's absent or empty. Raises
    :class:`LaunchError` on malformed JSON (we won't silently clobber a file we
    can't parse)."""
    if not path.exists():
        return {}
    text = _read_config_text(path)
    if not text:
        return {}
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as e:
        raise LaunchError(f"{path} is not valid JSON ({e}); refusing to overwrite it")
    if not isinstance(doc, dict):
        raise LaunchError(f"{path} is not a JSON object; refusing to overwrite it")
    return doc


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
    providers = dict(models_doc.get("providers") or {})
    providers[provider_id] = {
        "baseUrl": base_url,
        "api": "openai-completions",
        # pi requires an apiKey; a placeholder when the server has no auth.
        "apiKey": api_key or provider_id,
        "models": [{"id": m["id"]} for m in chat_models(models)],
    }
    models_doc["providers"] = providers

    settings_doc = dict(existing_settings or {})
    settings_doc["defaultProvider"] = provider_id
    if default_model:
        settings_doc["defaultModel"] = default_model
    return models_doc, settings_doc


def _launch_pi(a, *, exec_fn) -> int:
    binary = _find_binary(
        "pi", a,
        "Install pi first, then re-run (or use --config-only to just write the "
        "config).")
    base_url, models, default_model = _probe_target(a)

    agent_dir = Path(os.path.expanduser(a.config_path or _PI_AGENT_HOME))
    models_path = agent_dir / "models.json"
    settings_path = agent_dir / "settings.json"
    models_doc, settings_doc = build_pi_configs(
        base_url, models, provider_id=a.provider_id, default_model=default_model,
        api_key=a.api_key,
        existing_models=_load_json(models_path),
        existing_settings=_load_json(settings_path))

    agent_dir.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(models_path, json.dumps(models_doc, indent=2) + "\n")
    _write_text_atomic(settings_path, json.dumps(settings_doc, indent=2) + "\n")

    print(_summary("pi", base_url, models, default_model)
          + f"\n[launch] merged {models_path} + {settings_path}")
    if a.config_only:
        print("[launch] run it with:  pi")
        return 0

    return exec_fn(binary, ["pi"], dict(os.environ))


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
    if not path.exists():
        return {}
    text = _read_config_text(path)
    if not text:
        return {}
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise LaunchError(f"{path} is not valid YAML ({e}); refusing to overwrite it")
    if doc is None:
        return {}
    if not isinstance(doc, dict):
        raise LaunchError(f"{path} is not a YAML mapping; refusing to overwrite it")
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
    providers = dict(models_doc.get("providers") or {})
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
        roles = dict(config_doc.get("modelRoles") or {})
        roles["default"] = f"{provider_id}/{default_model}"
        config_doc["modelRoles"] = roles
    return models_doc, config_doc


def _launch_omp(a, *, exec_fn) -> int:
    if a.api_key:
        print("[launch] note: omp's provider registry has no API-key slot we "
              "know to write; if the server requires a key, configure omp's "
              "auth manually", file=sys.stderr)
    binary = _find_binary(
        "omp", a,
        "Install omp first, then re-run (or use --config-only to just write the "
        "config).", label="omp (oh-my-pi)")
    base_url, models, default_model = _probe_target(a)

    agent_dir = Path(os.path.expanduser(a.config_path or _OMP_AGENT_HOME))
    models_path = agent_dir / "models.yml"
    config_path = agent_dir / "config.yml"
    models_doc, config_doc = build_omp_configs(
        base_url, models, provider_id=a.provider_id, default_model=default_model,
        existing_models=_load_yaml(models_path),
        existing_config=_load_yaml(config_path))

    agent_dir.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(models_path, yaml.safe_dump(models_doc, sort_keys=False))
    _write_text_atomic(config_path, yaml.safe_dump(config_doc, sort_keys=False))

    print(_summary("omp", base_url, models, default_model)
          + f"\n[launch] merged {models_path} + {config_path}")
    if a.config_only:
        print("[launch] run it with:  omp")
        return 0

    return exec_fn(binary, ["omp"], dict(os.environ))


# hermes  (NousResearch hermes-agent - https://github.com/NousResearch/hermes-agent)
# hermes reads ``HERMES_CONFIG`` for an alternate config file, so we get
# opencode-style injection: the user's ``~/.hermes/config.yaml`` is read and
# merged with our provider block into our namespace, never written back. The
# provider *type* is hermes's literal ``custom`` (``--provider-id`` does not
# apply); ``CUSTOM_BASE_URL`` is exported too - hermes's documented override
# for ``provider: custom``. A default model is mandatory (``inference.model``).
_HERMES_CONFIG = "~/.hermes/config.yaml"


def build_hermes_config(base_url: str, *, default_model: str,
                        api_key: str | None = None,
                        existing: dict | None = None) -> dict:
    """The merged hermes ``config.yaml`` document: the user's existing settings
    with ``inference`` pointed at our ``custom`` provider. Key paths follow
    ``hermes config set inference.provider/inference.model`` and
    ``providers.<name>.*``. Pure - no IO."""
    cfg = dict(existing or {})
    inference = dict(cfg.get("inference") or {})
    inference["provider"] = "custom"
    inference["model"] = default_model
    cfg["inference"] = inference
    providers = dict(cfg.get("providers") or {})
    custom = dict(providers.get("custom") or {})
    custom["base_url"] = base_url
    if api_key:
        custom["api_key"] = api_key
    else:
        custom.setdefault("api_key", _PROVIDER_ID)   # placeholder: no server auth
    providers["custom"] = custom
    cfg["providers"] = providers
    return cfg


def _launch_hermes(a, *, exec_fn) -> int:
    binary = _find_binary(
        "hermes", a,
        "Install hermes-agent first, then re-run - see "
        "https://github.com/NousResearch/hermes-agent  (e.g. `curl -fsSL "
        "https://hermes-agent.nousresearch.com/install.sh | bash`).")
    base_url, models, default_model = _probe_target(
        a, require_default="its config pins inference.model")

    user_cfg = Path(os.path.expanduser(
        os.environ.get("HERMES_CONFIG") or _HERMES_CONFIG))
    cfg = build_hermes_config(base_url, default_model=default_model,
                              api_key=a.api_key, existing=_load_yaml(user_cfg))

    out = Path(os.path.expanduser(
        a.config_path or f"{_CONFIG_HOME}/hermes-config.yaml"))
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(out, yaml.safe_dump(cfg, sort_keys=False))

    print(_summary("hermes", base_url, models, default_model)
          + f"\n[launch] wrote {out} "
          f"(merged from {user_cfg}, which stays untouched)")
    print("[launch] note: hermes requires >=64k context - serve "
          f"{default_model} with a context window of at least 64k tokens")
    if a.config_only:
        print(f"[launch] run it with:  HERMES_CONFIG={out} "
              f"CUSTOM_BASE_URL={base_url} hermes")
        return 0

    env = dict(os.environ, HERMES_CONFIG=str(out), CUSTOM_BASE_URL=base_url)
    return exec_fn(binary, ["hermes"], env)


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
    binary = _find_binary(
        "goose", a,
        "Install goose first, then re-run - see "
        "https://github.com/block/goose (e.g. `brew install "
        "block-goose-cli` or its download_cli.sh script).")
    base_url, models, default_model = _probe_target(
        a, require_default="GOOSE_MODEL")
    pairs = build_goose_env(base_url, default_model=default_model,
                            api_key=a.api_key)

    cfg_path = Path(os.path.expanduser(a.config_path or _GOOSE_CONFIG))
    cfg = _load_yaml(cfg_path)
    # Persist only the non-secret pointer keys; OPENAI_API_KEY stays env-only so
    # we never clobber a real credential in the user's config.yaml.
    cfg.update({k: v for k, v in pairs.items() if k != "OPENAI_API_KEY"})
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(cfg_path, yaml.safe_dump(cfg, sort_keys=False))

    print(_summary("goose", base_url, models, default_model)
          + f"\n[launch] merged {cfg_path} "
          f"(OPENAI_API_KEY is supplied via the environment, not written)")
    if a.config_only:
        env_line = " ".join(f"{k}={v}" for k, v in pairs.items())
        print(f"[launch] run it with:  {env_line} goose session")
        return 0

    return exec_fn(binary, ["goose", "session"], dict(os.environ, **pairs))


# claude-code  (Anthropic Claude Code - https://claude.com/claude-code)
# Claude Code reads its provider entirely from env vars, so this is pure
# injection (the user's ~/.claude is never touched): ``ANTHROPIC_BASE_URL``
# points at the server root (Claude Code appends /v1/messages itself - the
# Anthropic surface mlx-vlm serves), ``ANTHROPIC_MODEL`` pins the default, and
# ``ANTHROPIC_AUTH_TOKEN`` carries the key (a placeholder when the server has
# no auth - it must be non-empty or Claude Code starts its own login flow).
# ``ANTHROPIC_SMALL_FAST_MODEL`` routes the background/haiku-class calls to the
# same local model.
def build_claude_code_env(base_url: str, *, default_model: str,
                          api_key: str | None = None) -> dict:
    """The Claude Code provider settings as env-var pairs. Pure - no IO."""
    return {
        "ANTHROPIC_BASE_URL": _server_root(base_url),
        "ANTHROPIC_AUTH_TOKEN": api_key or _PROVIDER_ID,  # placeholder: no auth
        "ANTHROPIC_MODEL": default_model,
        "ANTHROPIC_SMALL_FAST_MODEL": default_model,
    }


def _launch_claude_code(a, *, exec_fn) -> int:
    binary = _find_binary(
        "claude", a,
        "Install it first, then re-run - see https://claude.com/claude-code  "
        "(e.g. `npm install -g @anthropic-ai/claude-code` or "
        "`brew install --cask claude-code`).", label="claude (Claude Code)")
    base_url, models, default_model = _probe_target(
        a, require_default="ANTHROPIC_MODEL")
    pairs = build_claude_code_env(base_url, default_model=default_model,
                                  api_key=a.api_key)

    print(_summary("claude-code", f"{pairs['ANTHROPIC_BASE_URL']}/v1/messages",
                   models, default_model))
    print("[launch] note: Claude Code sends a very long system prompt and "
          "frequently rewrites its request prefix (compaction, tool results), "
          "so KV-prefix reuse is limited - expect prefill-dominated turn "
          "latency on local models; serve with the prompt cache (cache:) "
          "enabled to soften repeated prefixes")
    if a.config_only:
        env_line = " ".join(f"{k}={v}" for k, v in pairs.items())
        print(f"[launch] run it with:  {env_line} claude")
        return 0

    env = dict(os.environ, **pairs)
    # An inherited real key would take precedence over our ANTHROPIC_AUTH_TOKEN.
    env.pop("ANTHROPIC_API_KEY", None)
    return exec_fn(binary, ["claude"], env)


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
    binary = _find_binary(
        "aichat", a,
        "Install it first, then re-run - see https://github.com/sigoden/aichat  "
        "(e.g. `brew install aichat` or `cargo install aichat`).")
    base_url, models, default_model = _probe_target(a)
    cfg = build_aichat_config(base_url, models, provider_id=a.provider_id,
                              default_model=default_model, api_key=a.api_key)

    cfg_dir = Path(os.path.expanduser(a.config_path or _AICHAT_CONFIG_HOME))
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_file = cfg_dir / "config.yaml"
    _write_text_atomic(cfg_file, yaml.safe_dump(cfg, sort_keys=False))

    print(_summary("aichat", base_url, models, default_model)
          + f"\n[launch] wrote {cfg_file}")
    print("[launch] note: tool/agent use also needs aichat functions installed "
          "(aichat's llm-functions); the server already parses tool calls.")
    if a.config_only:
        print(f"[launch] run it with:  AICHAT_CONFIG_DIR={cfg_dir} aichat")
        return 0

    env = dict(os.environ, AICHAT_CONFIG_DIR=str(cfg_dir))
    return exec_fn(binary, ["aichat"], env)


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
    binary = _find_binary(
        "elia", a,
        "Install it first, then re-run - see https://github.com/darrenburns/elia  "
        "(e.g. `pipx install elia-chat` or `uv tool install elia-chat`).")
    base_url, models, default_model = _probe_target(a)
    toml_text = build_elia_config(base_url, models, provider_id=a.provider_id,
                                  default_model=default_model, api_key=a.api_key)

    xdg_home = Path(os.path.expanduser(a.config_path or _ELIA_CONFIG_HOME))
    out = xdg_home / "elia" / "config.toml"
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(out, toml_text)

    print(_summary("elia", base_url, models, default_model)
          + f"\n[launch] wrote {out}")
    print("[launch] note: needs the elia config.toml rewrite (elia >= 1.x); older "
          "builds ignore custom endpoints - `pipx upgrade elia-chat` if launch fails.")

    argv = ["elia"]
    if default_model:
        argv += ["-m", f"{a.provider_id}/{default_model}"]
    if a.config_only:
        print(f"[launch] run it with:  XDG_CONFIG_HOME={xdg_home} {' '.join(argv)}")
        return 0

    env = dict(os.environ, XDG_CONFIG_HOME=str(xdg_home))
    return exec_fn(binary, argv, env)


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
# so read-aloud works out of the box; a non-Kokoro `--tts` model needs AUDIO_TTS_VOICE
# overridden to one of its own voices.
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
    binary = _find_binary(
        "open-webui", a,
        "Install it in its own environment first, then re-run - see "
        "https://docs.openwebui.com  (e.g. "
        "`pipx install open-webui --python python3.12`; it needs Python 3.11 "
        "or 3.12, NOT 3.13).")
    base_url, models, default_model = _probe_target(a)
    # Route Open WebUI's audio engines at the server only when it advertises the
    # capability (the /v1/models markers the server sets behind --stt / --tts), so a
    # chat-only server leaves Open WebUI's built-in browser STT/TTS untouched.
    stt = any(m.get("stt") for m in models)
    tts = any(m.get("tts") for m in models)
    rerank = any(m.get("rerank") for m in models)

    server_port = a.port or _DEFAULT_PORT
    webui_port = (_OPEN_WEBUI_PORT if server_port != _OPEN_WEBUI_PORT
                  else _OPEN_WEBUI_PORT + 1)
    data_dir = os.path.abspath(
        os.path.expanduser(a.config_path or _OPEN_WEBUI_DATA_HOME))
    pairs = build_open_webui_env(base_url, default_model=default_model,
                                 api_key=a.api_key, port=webui_port, data_dir=data_dir,
                                 stt=stt, tts=tts, rerank=rerank)

    audio = [name for name, on in (("STT", stt), ("TTS", tts)) if on]
    audio_note = (f" Audio {'+'.join(audio)} routed at this server."
                  if audio else "")
    print(_summary("open-webui", base_url, models, default_model,
                   extra=((f", audio {'+'.join(audio)}" if audio else "")
                          + (", rerank" if rerank else "")))
          + f"\n[launch] web UI on http://localhost:{webui_port}  "
          f"(chat history + DB under {data_dir})")
    print("[launch] note: Open WebUI is a web app - open the URL above in a browser "
          "(it is not a terminal client). RAG points at this server, so no embedder "
          "is downloaded (document-RAG waits on /v1/embeddings; chat works now)."
          + audio_note +
          " For a no-login single-user setup add WEBUI_AUTH=false (only on a fresh "
          "DATA_DIR).")
    # `open-webui serve` binds via its `--port` CLI option (default 8080) and does
    # not read the PORT env var - so the port must be passed on the command line, or
    # the UI would try 8080 and collide with the gmlx server (crash: address in
    # use). PORT stays in `pairs` only for any self-URL construction Open WebUI does.
    argv = ["open-webui", "serve", "--port", str(webui_port)]
    if a.config_only:
        env_line = " ".join(f"{k}={v}" for k, v in pairs.items())
        print(f"[launch] run it with:  {env_line} {' '.join(argv)}")
        return 0

    env = dict(os.environ, **pairs)
    # Our single endpoint must win - drop any inherited plural OpenAI vars that
    # Open WebUI would otherwise merge ahead of it.
    env.pop("OPENAI_API_BASE_URLS", None)
    env.pop("OPENAI_API_KEYS", None)
    return exec_fn(binary, argv, env)


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
}


def _default_exec(binary: str, argv: list, env: dict) -> int:
    """Replace this process with the harness (so signals/TTY are the harness's).
    Seam: tests pass a recording fake instead."""
    os.execvpe(binary, argv, env)        # never returns on success
    return 127                           # unreachable; satisfies the type


# start-if-down orchestration (decision logic; the harness builders stay untouched)
def _server_ready(base_url: str, api_key: str | None = None) -> bool:
    """True iff the server answers ``/health`` and ``/v1/models`` (a 401 on models
    counts - up + auth-gated). Residency-independent, short-timeout so polling stays
    responsive. Mirrors :func:`lifecycle._ready` through the ``_http_get_json`` seam."""
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
    return bool(isinstance(payload, dict) and payload.get("data"))


def _discover_config():
    """The first existing default-location config, loaded. Returns ``(cfg, cfg_path)``:
    ``(None, None)`` if none exists; ``(None, path)`` if it exists but won't load
    (malformed != absent)."""
    from . import config
    for p in config.default_config_paths():
        if p.exists():
            cfg_path = str(p)
            try:
                return config.load_config(p), cfg_path
            except config.ConfigError:
                return None, cfg_path
    return None, None


def _human_size(n: int | None) -> str | None:
    """A human byte-count string (e.g. ``16.8 GB``); ``None`` for falsy/unknown."""
    from .lifecycle import human_gb

    return human_gb(n) if n else None


def _shard_siblings(path: str) -> list:
    """All existing shards of a ``*-00001-of-00003.gguf`` set if ``path`` is one;
    else ``[path]``. Size is cosmetic, so gaps are tolerated."""
    from .preflight import shard_names
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
    from .server import _preload_id
    try:
        pid = _preload_id(cfg)
    except Exception:
        return None, None
    if not pid:
        return None, None
    size = _human_size(_model_size_bytes(cfg, pid))
    return pid, (f"{pid} ({size})" if size else pid)


def _guide_to_init(harness: str | None,
                   rerun: str | None = None) -> None:
    """Setup guidance printed when nothing is running and no config exists.
    ``rerun`` overrides the re-run command for non-launch verbs (e.g. talk)."""
    cmd = rerun or f"launch {harness or '<harness>'}"
    tag = f"[{rerun or 'launch'}]"
    print(
        f"{tag} no gmlx server is running, and no config was found in a default\n"
        "  location (./gmlx.yaml, ~/.config/gmlx/gmlx.yaml, ~/.gmlx.yaml).\n"
        "  Set one up first:\n"
        "    gmlx init --models-dir <DIR>     # scaffold ~/.config/gmlx/gmlx.yaml from your GGUFs\n"
        "    gmlx init --from-hf-cache        # ...or from models already in your HF cache\n"
        f"  then re-run:  gmlx {cmd}\n"
        f"  Already have a server? point at it:    gmlx {cmd} --base-url URL",
        file=sys.stderr)


def _autostart(*, base, host, port, api_key, cfg, cfg_path, start_timeout, config_only):
    """Spawn a background server from ``cfg_path`` and poll until it answers, showing a
    spinner (named with the preloaded model + size when the config preloads one). Only a
    dead child is a hard failure; with a falsy ``start_timeout`` we wait while the child
    lives (Ctrl-C bails, leaving it running). Returns ``(rc, ready, preload_id)`` - the
    caller execs the harness iff ``ready``."""
    import time
    from . import lifecycle, spinner

    preload_id, label = _preload_descr(cfg)
    spin_text = (f"starting server - loading {label}" if preload_id
                 else f"starting server from {cfg_path}")
    spawned = lifecycle.start_background_nowait(
        ["--config", cfg_path], host=host, port=port,
        config_abspath=cfg_path, api_key=api_key)
    if spawned is None:                              # refused: a server already holds it
        return ((0, True, preload_id) if _server_ready(base, api_key)
                else (1, False, preload_id))
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
        print("[launch] interrupted - the server is still starting in the background "
              "(`gmlx status` / `gmlx stop`).", file=sys.stderr)
        return (130, False, preload_id)

    if outcome == "ready":
        if not config_only and getattr(cfg, "menubar", True) \
                and lifecycle.gui_session_available():
            lifecycle.start_menubar(auto=True)  # one machine-wide bar; tracks the primary
        return (0, True, preload_id)
    if outcome == "timeout":
        print(f"[launch] server still starting after {start_timeout:.0f}s - check "
              f"`gmlx logs` / `gmlx stop`.", file=sys.stderr)
        return (1, False, preload_id)
    tail = lifecycle._log_tail(log, 40).rstrip()     # died
    if lifecycle.report_port_in_use(tail, host, port, tag="[launch]"):
        return (1, False, preload_id)
    print(f"[launch] server exited (code {proc.returncode}) before it was ready.",
          file=sys.stderr)
    if tail and tail != "(no log)":
        print(tail, file=sys.stderr)
    return (1, False, preload_id)


def _ensure_server(a) -> int | None:
    """Start-if-down: resolve the endpoint onto ``a`` and, when nothing is reachable,
    auto-start a background server from a default-location config (or guide to `init`).
    Returns ``None`` to proceed to the harness, or an int exit code to return directly."""
    from . import lifecycle

    if a.base_url or a.host or a.port:
        url_host = url_port = None
        if a.base_url:
            # Harness clients (talk STT/TTS especially) expect the /v1 base;
            # accept a bare http://host:port like the default path builds.
            from .talk_client import ensure_v1_base
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
        a.base_url, a.host, a.port = base0, host0, port0
        return None

    if a.base_url:                                   # explicit endpoint: never auto-start
        a.host, a.port = host0, port0
        return None                                  # the harness probe raises the usual error

    cfg, cfg_path = _discover_config()
    if cfg_path is None:
        _guide_to_init(a.harness, getattr(a, "rerun_label", None))
        return 2
    if cfg is None:
        print(f"[launch] config {cfg_path} won't load (malformed) - fix it or pass "
              f"--base-url; not starting a server.", file=sys.stderr)
        return 2

    host = a.host or cfg.host
    port = int(a.port or cfg.port)
    key = a.api_key or getattr(cfg, "api_key", None)
    base = f"http://{host}:{port}/v1"
    a.base_url, a.host, a.port, a.api_key = base, host, port, key
    if _server_ready(base, key):                     # configured server already up (e.g. non-8080)
        return None

    if a.no_start:
        print(f"[launch] no server at {base} - start it (`gmlx serve`) or drop "
              f"--no-start to auto-start.", file=sys.stderr)
        return 1

    if (lifecycle.read_run(host, port) or {}).get("managed_by") == "launchd":
        print(f"[launch] a launchd server for {host}:{port} may be restarting - retry "
              f"shortly (`gmlx status`).", file=sys.stderr)
        return 1

    rc, ready, preload_id = _autostart(
        base=base, host=host, port=port, api_key=key, cfg=cfg, cfg_path=cfg_path,
        start_timeout=a.start_timeout, config_only=a.config_only)
    if not ready:
        return rc
    if a.config_only:
        print(f"[launch] left a background server running at {base} "
              f"(`gmlx stop` to tear it down).", file=sys.stderr)
    elif not preload_id:
        print(f"[launch] server up at {base}; no model is preloaded - your first "
              f"request will load one (that first turn will be slow).", file=sys.stderr)
    return None


def cmd_launch(argv: list, *, exec_fn=_default_exec,
               prog: str = "gmlx launch") -> int:
    # The macOS menu-bar monitor rides under `launch` but carries its own option set,
    # so it's dispatched before the harness parser ever sees it.
    if argv and argv[0] == "menubar":
        from .menubar import cmd_menubar
        return cmd_menubar(argv[1:], prog=f"{prog} menubar")

    ap = argparse.ArgumentParser(
        prog=prog,
        description="Point a coding harness at a gmlx server and run it, starting "
                    "the server from a default-location config if none is reachable "
                    "(no harness auto-install).",
        epilog="Also: `gmlx launch menubar` raises the macOS menu-bar monitor for "
               "a running server.",
    )
    ap.add_argument("harness", nargs="?", choices=sorted(_HARNESSES),
                    help="The coding harness, chat TUI (aichat/elia), or web app "
                         "(open-webui) to configure + launch. Omit it (bare "
                         "`gmlx launch`) to print this help; `menubar` raises the "
                         "macOS status-bar monitor.")
    ap.add_argument("--model", default=None,
                    help="Model id to make the harness default (must be served; "
                         "default: the server's default-marked model).")
    ap.add_argument("--base-url", default=None,
                    help="Server OpenAI base URL (default http://HOST:PORT/v1).")
    ap.add_argument("--host", default=None,
                    help="Server host (default: the single managed server if "
                         f"there's one, else the config's, else {_DEFAULT_HOST}).")
    ap.add_argument("--port", type=int, default=None,
                    help="Server port (default: the single managed server if "
                         f"there's one, else the config's, else {_DEFAULT_PORT}).")
    ap.add_argument("--api-key", default=None, metavar="KEY",
                    help="API key the harness sends - pass the same key the "
                         "server runs with (--api-key / server.api_key). "
                         "Default: a placeholder (fine for a no-auth server).")
    ap.add_argument("--provider-id", default=_PROVIDER_ID,
                    help=f"Provider id written into the harness config "
                         f"(default {_PROVIDER_ID}).")
    ap.add_argument("--config-path", default=None,
                    help=f"Where to write the harness config (default under "
                         f"{_CONFIG_HOME}).")
    ap.add_argument("--config-only", action="store_true",
                    help="Write the harness config and print the run command; do "
                         "not exec the harness.")
    ap.add_argument("--no-start", action="store_true",
                    help="Don't auto-start a server when none is reachable; just error.")
    ap.add_argument("--start-timeout", type=float, default=0.0, metavar="S",
                    help="Cap the wait for an auto-started server to become ready "
                         "(default 0 = wait as long as the child lives; Ctrl-C to bail).")
    ap.add_argument("--no-keep", action="store_true",
                    help="Don't ask the server to keep --model resident through its "
                         "idle TTL (it may be idle-unloaded mid-session).")
    a = ap.parse_args(argv)

    # Bare `gmlx launch` -> long-form help, not an argparse "required" error.
    if a.harness is None:
        ap.print_help()
        return 0

    try:
        rc = _ensure_server(a)
        if rc is not None:
            return rc
        if a.model and not a.no_keep and not a.config_only:
            # Validate --model before keeping: an unknown id must produce the
            # single "not served" error (raised again inside the harness fn),
            # never a "keeping X resident" line followed by that error.
            base = a.base_url or f"http://{a.host}:{a.port}/v1"
            _pick_default(probe_models(base, a.api_key), a.model)
            _keep_model(a)                       # server is reachable here; best-effort
        return _HARNESSES[a.harness](a, exec_fn=exec_fn)
    except LaunchError as e:
        print(f"[launch] {e}", file=sys.stderr)
        return 1
