# SPDX-License-Identifier: Apache-2.0
"""Where the server is and which image models the loaders list.

The server address: the node's widget, then ``GMLX_BASE_URL``, then the
settings file, then ``gmlx status --json``, then the default. With no
server running, ``gmlx status`` gives the host and port of gmlx's config,
where ``gmlx serve`` starts one.

The model list: the image model ids of the server at that address, else
those ``gmlx list --image --json`` prints, the default model first. With
none, it holds only the default model's name, and a run with it is
refused with the ``gmlx pull`` that adds it. ComfyUI asks for the
widgets at startup, on a refresh and for each queued prompt, so the probes
run on a background thread and the widgets take the answer kept from the
last one.

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

# The image models `gmlx pull` downloads, with their aliases and their
# family and variant, in the order of gmlx.image.catalog.RECIPES. gmlx's
# tests hold the two equal.
PULL_NAMES = {
    "z-image-turbo": (),
    "qwen-image": ("qwen-image-2.1",),
    "qwen-image-turbo": ("qwen-image-2.1-turbo",),
    "krea-2-turbo": (),
    "moss-image": ("moss-image-sft",),
    "moss-image-base": (),
    "flux-2-klein": (),
    "flux-2-klein-base": (),
    "flux-2-klein-9b": (),
    "flux-2-klein-base-9b": (),
    "flux-2-klein-9b-kv": (),
    "flux-3-klein": (),
    "flux-3-klein-base": (),
}
PULL_KINDS = {
    "z-image-turbo": ("z_image", "turbo"),
    "qwen-image": ("qwen_image21", "base"),
    "qwen-image-turbo": ("qwen_image21", "turbo"),
    "krea-2-turbo": ("krea2", "turbo"),
    "moss-image": ("moss_image", "sft"),
    "moss-image-base": ("moss_image", "base"),
    "flux-2-klein": ("flux2_klein", "distilled"),
    "flux-2-klein-base": ("flux2_klein", "base"),
    "flux-2-klein-9b": ("flux2_klein_9b", "distilled"),
    "flux-2-klein-base-9b": ("flux2_klein_9b", "base"),
    "flux-2-klein-9b-kv": ("flux2_klein_9b_kv", "distilled"),
    "flux-3-klein": ("flux3_klein", "distilled"),
    "flux-3-klein-base": ("flux3_klein", "base"),
}

# Timeouts of the widget probes, in seconds, and how long an answer is kept.
STATUS_TIMEOUT = 3.0
SERVER_TIMEOUT = 1.0
LIST_TIMEOUT = 5.0
KEEP = 10.0

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


def server_state(url: str, timeout: float = STATUS_TIMEOUT) -> tuple[bool | None, str | None]:
    """Whether the managed server on the port of ``url`` runs, by its
    runfile pid as ``gmlx status --json`` reports it, and the path of its
    log. None for the first when gmlx gives no answer or names no server on
    that port."""
    doc = _gmlx_json(["status", "--json"], timeout)
    if doc is None:
        return None, None
    try:
        port = urllib.parse.urlparse(url).port or 80
    except ValueError:
        return None, None
    rows = doc.get("servers") if "servers" in doc else [doc]
    for r in rows or []:
        try:
            if isinstance(r, dict) and int(r.get("port") or 0) == port:
                return bool(r.get("running")), (str(r["log"]) if r.get("log") else None)
        except (TypeError, ValueError):
            continue
    return None, None


def resolve_base_url(widget: str | None = None, timeout: float = 10.0) -> str:
    """The server address: the node's widget, then ``GMLX_BASE_URL``, then
    the settings file, then ``gmlx status --json``, then the default."""
    for v in (widget, os.environ.get("GMLX_BASE_URL"), settings.load().get("base_url")):
        if v and str(v).strip():
            return str(v).strip().rstrip("/")
    return (status_url(timeout) or DEFAULT_URL).rstrip("/")


def resolve_api_key() -> str | None:
    return os.environ.get("GMLX_API_KEY") or settings.load().get("api_key") or None


def _server_models(url: str, timeout: float) -> tuple[list, str | None] | None:
    """The image model ids of the server at ``url`` and its default image
    model, or None when it does not answer in time."""
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
    default = doc.get("default_model")
    return ([str(m["model"]) for m in models if isinstance(m, dict) and m.get("model")],
            str(default) if default else None)


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


def _ordered(ids: list, default: str | None) -> list:
    rec = settings.load().get("default_image_model")
    return _first(ids, os.environ.get("GMLX_DEFAULT_IMAGE_MODEL"), default,
                  rec if isinstance(rec, str) else None, settings.FALLBACK_MODEL)


def _probe() -> dict:
    url = resolve_base_url(None, timeout=STATUS_TIMEOUT)
    got, source = _server_models(url, SERVER_TIMEOUT), "server"
    if got is None:
        got, source = _configured(), "config"
    ids, default = got if got else ([], None)
    if not ids:
        return {"url": url, "models": [settings.default_image_model()], "source": "default"}
    return {"url": url, "models": _ordered(ids, default), "source": source}


def _fallback() -> dict:
    """The widgets before a probe answered, found without running gmlx:
    only the default model's name, since nothing says it is pulled."""
    url = os.environ.get("GMLX_BASE_URL") or settings.load().get("base_url") or DEFAULT_URL
    return {"url": str(url).strip().rstrip("/") or DEFAULT_URL,
            "models": [settings.default_image_model()], "source": "default"}


_KEPT: dict = {}
_LOCK = threading.Lock()
_STATE = {"gen": 0, "thread": None, "thread_gen": -1}


def forget() -> None:
    """Drop the kept answer and any probe running, so the next call probes
    again."""
    with _LOCK:
        _KEPT.clear()
        _STATE["gen"] += 1


def _refresh(gen: int) -> None:
    try:
        w = _probe()
    except Exception as e:  # noqa: BLE001: a widget list never fails ComfyUI
        log.warning("gmlx: listing the image models failed: %r", e)
        w = _fallback()
    with _LOCK:
        if gen == _STATE["gen"]:
            _KEPT["w"] = (time.monotonic() + KEEP, w)


def refresh() -> threading.Thread:
    """Probe on a background thread, unless a probe runs already. Returns
    that thread."""
    with _LOCK:
        t = _STATE["thread"]
        if t is not None and t.is_alive() and _STATE["thread_gen"] == _STATE["gen"]:
            return t
        t = threading.Thread(target=_refresh, args=(_STATE["gen"],), name="gmlx-widgets",
                             daemon=True)
        _STATE["thread"], _STATE["thread_gen"] = t, _STATE["gen"]
        t.start()
        return t


def widgets() -> dict:
    """``{"url", "models", "source"}``: the address a new node shows, the
    ids the model list shows and where they came from (``server``,
    ``config`` or ``default``). Returns at once with the answer kept from
    the last probe, or before the first probe answers with the default
    model's name alone, and starts a probe when the answer is old.
    Never raises."""
    with _LOCK:
        kept = _KEPT.get("w")
    if kept is None or time.monotonic() >= kept[0]:
        refresh()
    return kept[1] if kept is not None else _fallback()


def loader_inputs() -> dict:
    """The INPUT_TYPES of the loaders and the status node."""
    w = widgets()
    models = list(w["models"])
    return {"required": {
        "model_name": (models, {
            "default": models[0],
            "tooltip": "An image model of the gmlx server. With none in its config, the "
                       "list holds the default model, which `gmlx pull` adds."}),
        "server_url": ("STRING", {
            "default": w["url"],
            "tooltip": "The gmlx server's address. Empty, or the default "
                       f"{DEFAULT_URL}, finds the gmlx server when the node runs."}),
        "start_server": ("BOOLEAN", {
            "default": True, "tooltip": "Run gmlx serve when no server answers."})}}


def validate(model_name) -> bool | str:
    """ComfyUI's VALIDATE_INPUTS for model_name. Any name passes, so a
    workflow from another Mac or a template reaches the node, which checks
    the name against its server's models."""
    return True if str(model_name or "").strip() else "model_name is empty"


def pull_name(name: str) -> str | None:
    """The recipe ``gmlx pull`` downloads for ``name``, a recipe or one of
    its aliases, or None."""
    key = str(name).strip().lower()
    for recipe, aliases in PULL_NAMES.items():
        if key == recipe or key in aliases:
            return recipe
    return None


def not_on_server(name: str, url: str, models: list) -> str:
    """The refusal for a model the server at ``url`` does not have.
    ``models`` are the rows of the server's model list. When the server has
    the recipe's model under another id, such as one ``gmlx pull``
    registered under the recipe name for a workflow that names its alias,
    the refusal names that id."""
    ids = [str(m.get("model")) for m in models if isinstance(m, dict) and m.get("model")]
    have = ", ".join(ids) if ids else "none"
    head = f"gmlx: the gmlx server at {url} has no image model {name!r} (it has: {have})."
    recipe = pull_name(name)
    if recipe:
        same = [str(m["model"]) for m in models if isinstance(m, dict) and m.get("model")
                and (m.get("family"), m.get("variant")) == PULL_KINDS[recipe]]
        if same:
            return (f"{head} It has that model as {same[0]!r}: pick {same[0]!r} in the "
                    f"loader's model_name, or run `gmlx pull {recipe}` and pick the id it "
                    "adds.")
        return (f"{head} Run `gmlx pull {recipe}`, which downloads it and adds it to "
                "gmlx's config, then run the workflow again.")
    return (f"{head} Pick one it has, or add one with `gmlx pull`, such as "
            f"`gmlx pull {settings.FALLBACK_MODEL}`.")
