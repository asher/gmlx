# SPDX-License-Identifier: Apache-2.0
"""The client side of the gmlx image step contract.

A body is a safetensors container: an 8 byte little-endian header length,
a compact JSON header naming each tensor's dtype, shape and byte range, then
the raw bytes. Request fields travel as sorted compact JSON under
``__metadata__["gmlx"]``. The header is padded with spaces to a multiple of
8 bytes, so one set of tensors and fields gives one body, the same body the
server writes.

A tensor's hash is the sha256 of its dtype name, its shape as JSON and its
raw bytes. The server keeps conditionings and adapter matrices by hash, so
each is uploaded once.

This module imports torch and numpy only, never ComfyUI.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import logging
import os
import selectors
import shutil
import struct
import subprocess
import threading
import time
import urllib.parse
import weakref

import numpy as np
import torch

PREFIX = "/internal/image-steps/v1"
CONTENT_TYPE = "application/vnd.gmlx.tensors+safetensors"
CONTRACT_HEADER = "X-Gmlx-Steps-Contract"
CONTRACT = (1, 0)
DEFAULT_URL = "http://127.0.0.1:8080"

_TO_NAME = {torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16",
            torch.uint8: "U8", torch.bool: "BOOL"}
_NUMPY = {"F32": "<f4", "F16": "<f2", "BF16": "<u2", "U8": "u1", "BOOL": "?"}
_ITEM = {"F32": 4, "F16": 2, "BF16": 2, "U8": 1, "BOOL": 1}

log = logging.getLogger("gmlx")


class GmlxError(RuntimeError):
    """A request the gmlx server refused or could not serve."""


class GmlxRefusal(GmlxError):
    """Something a gmlx model does not support, refused by name."""


class GmlxVersionError(GmlxError):
    """The node pack and the server speak different step contract majors."""


class GmlxMissing(GmlxError):
    """The server does not hold some tensors or tables a request named."""

    def __init__(self, missing):
        self.missing = list(missing)
        super().__init__(f"the gmlx server does not hold {len(self.missing)} "
                         "tensor(s) or table(s)")


class GmlxInterrupted(GmlxError):
    """The server stopped a step because its run was interrupted or ended."""


# Wire format.

def _raw(t: torch.Tensor) -> tuple[str, list, bytes]:
    if isinstance(t, tuple):
        return t[0], [int(d) for d in t[1]], bytes(t[2])
    if isinstance(t, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(t))
    t = t.detach()
    if t.dtype not in _TO_NAME:
        raise GmlxError(f"dtype {t.dtype} is not one the step contract carries")
    name = _TO_NAME[t.dtype]
    t = t.contiguous().cpu()
    raw = t.view(torch.uint8).numpy().tobytes() if t.numel() else b""
    return name, [int(d) for d in t.shape], raw


def pack(tensors: dict, fields: dict | None = None) -> bytes:
    """A body of ``tensors`` (torch tensors, numpy arrays or
    ``(dtype, shape, bytes)`` triples by name, in order) and ``fields``."""
    header: dict = {}
    chunks = []
    off = 0
    for name, t in tensors.items():
        dt, shape, raw = _raw(t)
        header[name] = {"dtype": dt, "shape": shape,
                        "data_offsets": [off, off + len(raw)]}
        chunks.append(raw)
        off += len(raw)
    if fields is not None:
        header["__metadata__"] = {"gmlx": json.dumps(
            fields, sort_keys=True, separators=(",", ":"))}
    hj = json.dumps(header, separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    return struct.pack("<Q", len(hj)) + hj + b"".join(chunks)


def unpack(body: bytes) -> tuple[dict, dict]:
    """``(tensors, fields)`` of a body, each tensor a torch tensor of its own
    dtype that owns its memory."""
    if len(body) < 8:
        raise GmlxError("the response is shorter than a safetensors header")
    (n,) = struct.unpack("<Q", body[:8])
    if 8 + n > len(body):
        raise GmlxError("the response's safetensors header is out of range")
    header = json.loads(body[8:8 + n])
    meta = header.pop("__metadata__", None) or {}
    base = 8 + n
    out = {}
    for name, spec in header.items():
        dt = spec["dtype"]
        if dt not in _NUMPY:
            raise GmlxError(f"the response carries dtype {dt}")
        start, end = spec["data_offsets"]
        a = np.frombuffer(body, dtype=_NUMPY[dt], count=(end - start) // _ITEM[dt],
                          offset=base + start).reshape(spec["shape"])
        t = torch.from_numpy(a.copy())
        if dt == "BF16":
            t = t.view(torch.bfloat16)
        out[name] = t
    fields = json.loads(meta["gmlx"]) if "gmlx" in meta else {}
    return out, fields


def tensor_hash(t) -> str:
    """The contract's hash of a tensor: sha256 of dtype name, shape and bytes."""
    dt, shape, raw = _raw(t)
    return raw_hash(dt, shape, raw)


def raw_hash(dt: str, shape, raw: bytes) -> str:
    h = hashlib.sha256()
    h.update(dt.encode())
    h.update(json.dumps([int(d) for d in shape]).encode())
    h.update(raw)
    return h.hexdigest()


def table_id(table: dict) -> tuple[str, bytes]:
    """The canonical body of a patch table and its id, the sha256 of it."""
    body = json.dumps(table, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest(), body


def _version(t) -> int | None:
    """A tensor's in-place write counter; None for an inference tensor,
    which ComfyUI's node execution creates and which has none."""
    try:
        return t._version
    except RuntimeError:
        return None


# Discovery.

def settings_path() -> str:
    return os.environ.get("GMLX_COMFY_SETTINGS") or os.path.expanduser(
        "~/.config/gmlx/comfyui.json")


def _settings() -> dict:
    try:
        with open(settings_path()) as f:
            doc = json.load(f)
        return doc if isinstance(doc, dict) else {}
    except (OSError, ValueError):
        return {}


def _gmlx_status_url() -> str | None:
    exe = shutil.which("gmlx")
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, "status", "--json"], capture_output=True,
                             text=True, timeout=10)
        doc = json.loads(out.stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    if isinstance(doc, dict) and doc.get("running") and doc.get("url"):
        return str(doc["url"])
    return None


def resolve_base_url(widget: str | None = None) -> str:
    """The server address: the node's widget, then ``GMLX_BASE_URL``, then
    the settings file, then ``gmlx status --json``, then the default."""
    for v in (widget, os.environ.get("GMLX_BASE_URL"), _settings().get("base_url")):
        if v and str(v).strip():
            return str(v).strip().rstrip("/")
    return (_gmlx_status_url() or DEFAULT_URL).rstrip("/")


def resolve_api_key() -> str | None:
    return os.environ.get("GMLX_API_KEY") or _settings().get("api_key") or None


# HTTP.

def _dropped(conn) -> bool:
    """Whether an idle connection is closed or holds bytes nobody asked
    for, checked without waiting. Either way it cannot carry a request."""
    if conn.sock is None:
        return True
    try:
        with selectors.DefaultSelector() as sel:
            sel.register(conn.sock, selectors.EVENT_READ)
            return bool(sel.select(0))
    except (OSError, ValueError):
        return True


class StepClient:
    """A keep-alive HTTP/1.1 client of one gmlx server, with what it has
    uploaded there. Steps run on short-lived threads, so connections come
    from a small pool."""

    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 600.0):
        u = urllib.parse.urlparse(base_url)
        if u.scheme not in ("http", "") or not u.hostname:
            raise GmlxError(f"the gmlx server address {base_url!r} is not an http URL")
        self.base_url = base_url.rstrip("/")
        self.host, self.port = u.hostname, u.port or 80
        self.root = (u.path or "").rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._pool: list = []
        self._lock = threading.Lock()
        self.known_tensors: set = set()
        self.known_adapters: set = set()
        self.known_tables: set = set()
        self._hash_memo: dict = {}
        self.capabilities: dict | None = None
        self.stats = {"tensor_uploads": 0, "adapter_uploads": 0, "table_puts": 0,
                      "steps": 0, "connections": 0, "missing_retries": 0}

    # Transport.

    def request(self, method: str, path: str, body: bytes | None = None,
                headers: dict | None = None, *, versioned: bool = True,
                timeout: float | None = None):
        hdrs = {}
        if versioned:
            hdrs[CONTRACT_HEADER] = f"{CONTRACT[0]}.{CONTRACT[1]}"
        if self.api_key:
            hdrs["Authorization"] = f"Bearer {self.api_key}"
        hdrs.update(headers or {})
        if body is not None or method in ("POST", "PUT", "PATCH"):
            hdrs["Content-Length"] = str(len(body or b""))
        for attempt in (0, 1):
            conn = None
            if attempt == 0:
                conn = self._take()
            pooled = conn is not None
            if conn is None:
                conn = http.client.HTTPConnection(self.host, self.port,
                                                  timeout=timeout or self.timeout)
                self.stats["connections"] += 1
            else:
                # A pooled connection keeps the timeout of the call that made it.
                conn.timeout = timeout or self.timeout
                if conn.sock is not None:
                    conn.sock.settimeout(conn.timeout)
            try:
                conn.putrequest(method, self.root + path)
                for k, v in hdrs.items():
                    conn.putheader(k, v)
                conn.endheaders()
            except (ConnectionError, http.client.HTTPException, OSError) as e:
                conn.close()
                if attempt or not pooled or isinstance(e, TimeoutError):
                    raise self._lost(method, path, e) from None
                # The headers did not all go out, so the server holds no
                # request to act on: send it once more on a new connection.
                log.debug("gmlx: %s %s on a reused connection failed (%r); retrying",
                          method, path, e)
                continue
            try:
                if body:
                    conn.send(body)
                r = conn.getresponse()
                data = r.read()
            except (ConnectionError, http.client.HTTPException, OSError) as e:
                # The server may have read the request, so it is never resent.
                conn.close()
                raise self._lost(method, path, e) from None
            if r.getheader("Connection", "").lower() == "close":
                conn.close()
            else:
                with self._lock:
                    self._pool.append(conn)
            return r.status, {k.lower(): v for k, v in r.getheaders()}, data
        raise AssertionError("unreachable")

    def _take(self):
        """An idle pooled connection that is still open, or None. One the
        server closed while idle is closed here, before a byte is sent on it."""
        with self._lock:
            while self._pool:
                conn = self._pool.pop()
                if not _dropped(conn):
                    return conn
                conn.close()
        return None

    def _lost(self, method: str, path: str, e: Exception) -> GmlxError:
        return GmlxError(f"the gmlx server at {self.base_url} did not answer "
                         f"{method} {path}: {e}")

    def _fail(self, status: int, data: bytes, what: str):
        try:
            doc = json.loads(data) if data else {}
        except ValueError:
            doc = {}
        if not isinstance(doc, dict):
            doc = {}
        msg = doc.get("error") or data[:300].decode("utf-8", "replace")
        if status == 409 and doc.get("missing") is not None:
            raise GmlxMissing(doc["missing"])
        if status == 409 and doc.get("code") == "interrupted":
            raise GmlxInterrupted(f"{what}: {msg}")
        if status == 400 and "step contract" in str(msg):
            raise GmlxVersionError(f"gmlx: {msg}")
        if status in (401, 403):
            raise GmlxError(f"gmlx: {what} was refused ({status}): {msg}. Set "
                            "GMLX_API_KEY when the server has a key, and run the "
                            "server on this machine")
        raise GmlxError(f"gmlx: {what} failed ({status}): {msg}")

    def call_json(self, method: str, path: str, obj=None, *, what: str = ""):
        body = None if obj is None else json.dumps(obj).encode()
        hdrs = {} if body is None else {"Content-Type": "application/json"}
        s, _, d = self.request(method, PREFIX + path, body, hdrs,
                               versioned=path != "/capabilities")
        if s != 200:
            self._fail(s, d, what or f"{method} {path}")
        return json.loads(d) if d else {}

    def call_tensors(self, path: str, tensors: dict, fields: dict | None, *,
                     what: str = "", method: str = "POST"):
        s, h, d = self.request(method, PREFIX + path, pack(tensors, fields),
                               {"Content-Type": CONTENT_TYPE})
        if s != 200:
            self._fail(s, d, what or path)
        return unpack(d) + (h,)

    # Handshake.

    def health(self, timeout: float = 2.0) -> bool:
        try:
            s, _, _ = self.request("GET", "/health", versioned=False, timeout=timeout)
        except GmlxError:
            return False
        return s == 200

    def handshake(self) -> dict:
        """The server's capabilities, after checking that this pack speaks its
        contract. A major mismatch names the side to upgrade."""
        caps = self.call_json("GET", "/capabilities", what="the capabilities handshake")
        c = caps.get("contract") or {}
        major, minor = int(c.get("major", -1)), int(c.get("minor", 0))
        mc = caps.get("min_client") or {"major": major, "minor": 0}
        if major != CONTRACT[0]:
            side = "the ComfyUI-gmlx node pack" if major > CONTRACT[0] else "the gmlx server"
            raise GmlxVersionError(
                f"gmlx: the server speaks step contract {major}.{minor} and this node "
                f"pack speaks {CONTRACT[0]}.{CONTRACT[1]}. Upgrade {side}.")
        if (int(mc.get("major", 0)), int(mc.get("minor", 0))) > CONTRACT:
            raise GmlxVersionError(
                f"gmlx: the server needs a client of step contract {mc.get('major')}."
                f"{mc.get('minor')} or newer and this node pack speaks "
                f"{CONTRACT[0]}.{CONTRACT[1]}. Upgrade the ComfyUI-gmlx node pack.")
        self.capabilities = caps
        return caps

    # Models.

    def models(self) -> list:
        return self.call_json("GET", "/models").get("models", [])

    def status(self, model: str) -> dict:
        return self.call_json("GET", f"/models/{urllib.parse.quote(model, safe='')}/status",
                              what=f"status of {model!r}")

    def manifest(self, model: str) -> dict:
        return self.call_json("GET", f"/models/{urllib.parse.quote(model, safe='')}/manifest",
                              what=f"manifest of {model!r}")

    def load(self, model: str) -> dict:
        return self.call_json("POST", "/models/load", {"model": model}, what=f"load of {model!r}")

    # Uploads.

    def _memo_hash(self, t: torch.Tensor) -> str:
        """The hash of an adapter matrix, memoized by tensor identity, since
        a LoRA's tensors stay the same objects across clones."""
        key = id(t)
        hit = self._hash_memo.get(key)
        if hit is not None:
            ref, version, h = hit
            if ref() is t and version == _version(t):
                return h
        h = tensor_hash(t)
        try:
            ref = weakref.ref(t, lambda _r, k=key: self._hash_memo.pop(k, None))
        except TypeError:
            return h
        self._hash_memo[key] = (ref, _version(t), h)
        return h

    def put_tensor(self, h: str, t, *, adapter: bool = False) -> None:
        kind = "adapters" if adapter else "tensors"
        s, _, d = self.request("PUT", f"{PREFIX}/{kind}/{h}", pack({"t": t}),
                               {"Content-Type": CONTENT_TYPE})
        if s != 200:
            self._fail(s, d, f"upload of {kind[:-1]} {h[:12]}")
        if adapter:
            self.known_adapters.add(h)
            self.stats["adapter_uploads"] += 1
        else:
            self.known_tensors.add(h)
            self.stats["tensor_uploads"] += 1

    def ensure_tensor(self, t, h: str | None = None) -> str:
        h = h or tensor_hash(t)
        if h not in self.known_tensors:
            self.put_tensor(h, t)
        return h

    def ensure_adapter(self, t: torch.Tensor) -> str:
        h = self._memo_hash(t)
        if h in self.known_adapters:
            return h
        s, _, _ = self.request("HEAD", f"{PREFIX}/adapters/{h}")
        if s == 200:
            self.known_adapters.add(h)
        else:
            self.put_tensor(h, t, adapter=True)
        return h

    def define_table(self, model: str, table: dict, *, force: bool = False) -> str:
        tid, body = table_id(table)
        if tid in self.known_tables and not force:
            return tid
        q = urllib.parse.quote(model, safe="")
        s, _, d = self.request("PUT", f"{PREFIX}/patch-tables/{tid}?model={q}", body,
                               {"Content-Type": "application/json"})
        if s != 200:
            self._fail(s, d, "the patch table")
        self.known_tables.add(tid)
        self.stats["table_puts"] += 1
        return tid

    def forget(self) -> None:
        """Forget what the server was sent, after it answered that it does
        not hold something: it restarted or evicted."""
        self.known_tensors.clear()
        self.known_adapters.clear()
        self.known_tables.clear()

    # Work.

    def encode(self, model: str, text: str, images: list | None = None) -> tuple:
        """Conditioning (1, L, D) f32 and the answer's fields for ``text``,
        with reference images as uint8 (H, W, C) arrays."""
        if images:
            tensors = {f"image_{i}": np.ascontiguousarray(im, dtype=np.uint8)
                       for i, im in enumerate(images)}
            # ComfyUI encodes the reference latents with its own VAE node.
            out, fields, _ = self.call_tensors("/encode", tensors,
                                               {"model": model, "text": text, "refs": False},
                                               what="the text encode")
        else:
            s, _, d = self.request("POST", PREFIX + "/encode",
                                   json.dumps({"model": model, "text": text}).encode(),
                                   {"Content-Type": "application/json"})
            if s != 200:
                self._fail(s, d, "the text encode")
            out, fields = unpack(d)
        if fields.get("cond"):
            self.known_tensors.add(fields["cond"])
        for h in fields.get("ref_latents") or []:
            self.known_tensors.add(h)
        return out, fields

    def step(self, x: torch.Tensor, timestep: torch.Tensor, fields: dict):
        out, _, h = self.call_tensors("/step", {"x": x, "timestep": timestep}, fields,
                                      what="the step")
        self.stats["steps"] += 1
        return out["out"], h

    def vae_decode(self, model: str, latents: torch.Tensor, alpha: bool) -> torch.Tensor:
        out, _, _ = self.call_tensors("/vae/decode", {"latents": latents.float()},
                                      {"model": model, "alpha": bool(alpha)},
                                      what="the VAE decode")
        return out["image"]

    def vae_encode(self, model: str, image: torch.Tensor) -> torch.Tensor:
        out, _, _ = self.call_tensors("/vae/encode", {"image": image.float()},
                                      {"model": model}, what="the VAE encode")
        return out["latents"]

    def run_begin(self, model: str) -> str:
        return self.call_json("POST", "/runs/begin", {"model": model}, what="the run begin")["run"]

    def run_end(self, run: str) -> None:
        self.call_json("POST", f"/runs/{run}/end", {}, what="the run end")

    def interrupt(self, run: str | None) -> None:
        self.call_json("POST", "/interrupt", {"run": run}, what="the interrupt")


_CLIENTS: dict = {}
_CLIENTS_LOCK = threading.Lock()


def get_client(base_url: str | None = None, *, start_server: bool = False,
               start_timeout: float = 180.0) -> StepClient:
    """The shared client of a server, with the handshake done. When the
    server is down and ``start_server`` is set, runs ``gmlx serve``, which
    starts detached, and waits for ``/health``."""
    url = resolve_base_url(base_url)
    key = resolve_api_key()
    with _CLIENTS_LOCK:
        c = _CLIENTS.get((url, key))
        if c is None:
            c = _CLIENTS[(url, key)] = StepClient(url, key)
    if c.capabilities is not None and c.health():
        return c
    if not c.health():
        exe = shutil.which("gmlx")
        if not start_server or exe is None:
            hint = ("run `gmlx serve`" if exe else "install gmlx and run `gmlx serve`")
            raise GmlxError(f"gmlx: no server answers at {url}; {hint}, or set the "
                            "loader's server_url or GMLX_BASE_URL")
        log.info("gmlx: starting `gmlx serve` for %s", url)
        subprocess.Popen([exe, "serve"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
        deadline = time.monotonic() + start_timeout
        while not c.health():
            if time.monotonic() > deadline:
                raise GmlxError(f"gmlx: `gmlx serve` did not answer at {url} within "
                                f"{start_timeout:.0f} s; see `gmlx logs`")
            time.sleep(0.5)
    c.handshake()
    return c
