# SPDX-License-Identifier: Apache-2.0
"""Where the server is and which image models the loaders list.

The server address: the node's widget, then ``GMLX_BASE_URL``, then the
settings file, then ``gmlx status --json``, then the default. With no
server running, ``gmlx status`` gives the host and port of gmlx's config,
where ``gmlx serve`` starts one.

The model list: the image model ids of the server at that address, else
those ``gmlx list --image --json`` prints, else the image models
``gmlx pull`` downloads, the default model first. ComfyUI asks for the
widgets at startup, on a refresh and for each queued prompt, so every
probe has a short timeout and an answer is kept for a few seconds.

This module imports neither torch nor ComfyUI, so the stand-in nodes list
the same widgets.
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import subprocess
import threading
import time
import urllib.parse

from . import settings

PREFIX = "/internal/image-steps/v1"
CONTRACT_HEADER = "X-Gmlx-Steps-Contract"
CONTRACT = (1, 0)
DEFAULT_URL = "http://127.0.0.1:8080"

# The image models `gmlx pull` downloads, with their aliases, in the order
# of gmlx.image.catalog.RECIPES. gmlx's tests hold the two equal.
PULL_NAMES = {
    "z-image-turbo": (),
    "qwen-image": ("qwen-image-2.1",),
    "qwen-image-turbo": ("qwen-image-2.1-turbo",),
    "krea-2-turbo": (),
}

# Timeouts of the widget probes, in seconds, and how long an answer is kept.
STATUS_TIMEOUT = 3.0
SERVER_TIMEOUT = 1.0
LIST_TIMEOUT = 5.0
KEEP = 10.0
KEEP_SLOW = 60.0
SLOW = 2.0

LOOPBACK = ("localhost", "::1")

log = logging.getLogger("gmlx")


def _gmlx_json(args: list, timeout: float) -> dict | None:
    """The JSON a gmlx command prints, or None."""
    exe = settings.gmlx_command()
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, *args], capture_output=True, text=True,
                             stdin=subprocess.DEVNULL, timeout=timeout)
        doc = json.loads(out.stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return doc if isinstance(doc, dict) else None


def _url(host, port) -> str:
    host = str(host)
    if host in ("", "0.0.0.0", "::"):
        host = "127.0.0.1"
    if ":" in host:
        host = f"[{host}]"
    return f"http://{host}:{int(port)}"


def status_url(timeout: float = 10.0) -> str | None:
    """The address ``gmlx status --json`` gives: the running server's, else
    the host and port it names, from gmlx's config when no server ran."""
    doc = _gmlx_json(["status", "--json"], timeout)
    if doc is None:
        return None
    if doc.get("running") and doc.get("url"):
        return str(doc["url"])
    if "servers" in doc:
        live = [s for s in doc.get("servers") or []
                if isinstance(s, dict) and s.get("running") and s.get("url")]
        return str(live[0]["url"]) if len(live) == 1 else None
    try:
        return _url(doc["host"], doc["port"]) if doc.get("host") and doc.get("port") else None
    except (TypeError, ValueError):
        return None


def resolve_base_url(widget: str | None = None, timeout: float = 10.0) -> str:
    """The server address: the node's widget, then ``GMLX_BASE_URL``, then
    the settings file, then ``gmlx status --json``, then the default."""
    for v in (widget, os.environ.get("GMLX_BASE_URL"), settings.load().get("base_url")):
        if v and str(v).strip():
            return str(v).strip().rstrip("/")
    return (status_url(timeout) or DEFAULT_URL).rstrip("/")


def resolve_api_key() -> str | None:
    return os.environ.get("GMLX_API_KEY") or settings.load().get("api_key") or None


def is_loopback(url: str) -> bool:
    try:
        host = urllib.parse.urlparse(url).hostname or ""
    except ValueError:
        return False
    return host.startswith("127.") or host in LOOPBACK


def _server_models(url: str, timeout: float) -> list | None:
    """The image model ids of the server at ``url``, or None when it does
    not answer in time."""
    try:
        u = urllib.parse.urlparse(url)
        port = u.port or 80
    except ValueError:
        return None
    if u.scheme not in ("http", "") or not u.hostname:
        return None
    hdrs = {CONTRACT_HEADER: f"{CONTRACT[0]}.{CONTRACT[1]}"}
    key = resolve_api_key()
    if key:
        hdrs["Authorization"] = f"Bearer {key}"
    conn = http.client.HTTPConnection(u.hostname, port, timeout=timeout)
    try:
        conn.request("GET", (u.path or "").rstrip("/") + PREFIX + "/models", headers=hdrs)
        r = conn.getresponse()
        body = r.read()
        if r.status != 200:
            return None
        doc = json.loads(body)
    except (OSError, ValueError, http.client.HTTPException):
        return None
    finally:
        conn.close()
    models = doc.get("models") if isinstance(doc, dict) else None
    if not isinstance(models, list):
        return None
    return [str(m["model"]) for m in models if isinstance(m, dict) and m.get("model")]


def _configured() -> tuple[list, str | None] | None:
    """The image model ids ``gmlx list --image --json`` prints and the
    default it names, or None."""
    doc = _gmlx_json(["list", "--image", "--json"], LIST_TIMEOUT)
    if doc is None or not isinstance(doc.get("models"), list):
        return None
    ids = [str(r["id"]) for r in doc["models"]
           if isinstance(r, dict) and r.get("kind", "image") == "image" and r.get("id")]
    default = doc.get("default_image_model")
    return ids, (str(default) if default else None)


def _first(ids: list, *names) -> list:
    for name in names:
        if name and name in ids:
            return [name] + [i for i in ids if i != name]
    return ids


def _probe() -> dict:
    url = resolve_base_url(None, timeout=STATUS_TIMEOUT)
    ids, default, source = _server_models(url, SERVER_TIMEOUT), None, "server"
    if ids is None:
        got = _configured()
        ids, default, source = (got[0], got[1], "config") if got else ([], None, None)
    if not ids:
        ids, source = list(PULL_NAMES), "pull"
    rec = settings.load().get("default_image_model")
    ids = _first(ids, os.environ.get("GMLX_DEFAULT_IMAGE_MODEL"), default,
                 rec if isinstance(rec, str) else None, settings.FALLBACK_MODEL)
    return {"url": url, "models": ids, "source": source}


_KEPT: dict = {}
_LOCK = threading.Lock()


def forget() -> None:
    """Drop the kept answer, so the next call probes again."""
    with _LOCK:
        _KEPT.clear()


def widgets() -> dict:
    """``{"url", "models", "source"}``: the address a node without a
    server_url uses, the ids the model list shows and where they came from
    (``server``, ``config`` or ``pull``). Never raises."""
    with _LOCK:
        now = time.monotonic()
        kept = _KEPT.get("w")
        if kept is not None and now < kept[0]:
            return kept[1]
        try:
            w = _probe()
        except Exception as e:  # noqa: BLE001: a widget list never fails ComfyUI
            log.warning("gmlx: listing the image models failed: %r", e)
            w = {"url": DEFAULT_URL, "models": _first(list(PULL_NAMES), settings.default_image_model()),
                 "source": "pull"}
        took = time.monotonic() - now
        _KEPT["w"] = (time.monotonic() + (KEEP_SLOW if took > SLOW else KEEP), w)
        return w


def loader_inputs() -> dict:
    """The INPUT_TYPES of the loaders and the status node."""
    w = widgets()
    models = list(w["models"])
    return {"required": {
        "model_name": (models, {
            "default": models[0],
            "tooltip": "An image model of the gmlx server. With none in its config, the "
                       "list shows the models `gmlx pull` downloads."}),
        "server_url": ("STRING", {
            "default": w["url"],
            "tooltip": "The gmlx server's address. Empty finds it when the node runs, as a "
                       "127.0.0.1 address that does not answer does."}),
        "start_server": ("BOOLEAN", {
            "default": True, "tooltip": "Run gmlx serve when no server answers."})}}


def validate(model_name) -> bool | str:
    """ComfyUI's VALIDATE_INPUTS for model_name. Any name passes, so a
    workflow from another Mac or a template reaches the node, which checks
    the name against its server's models."""
    return True if str(model_name or "").strip() else "model_name is empty"


def pull_name(name: str) -> str | None:
    """The recipe ``gmlx pull`` downloads for ``name``, or None."""
    key = str(name).strip().lower()
    for recipe, aliases in PULL_NAMES.items():
        if key == recipe or key in aliases:
            return key
    return None


def not_on_server(name: str, url: str, ids: list) -> str:
    """The refusal for a model the server at ``url`` does not have."""
    have = ", ".join(ids) if ids else "none"
    head = f"gmlx: the gmlx server at {url} has no image model {name!r} (it has: {have})."
    if pull_name(name):
        return (f"{head} Run `gmlx pull {pull_name(name)}`, which downloads it and adds it to "
                "gmlx's config, then run the workflow again.")
    return (f"{head} Pick one it has, or add one with `gmlx pull`, such as "
            f"`gmlx pull {settings.FALLBACK_MODEL}`.")
