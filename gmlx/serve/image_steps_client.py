"""A reference client of the image step contract, numpy and the standard
library only.

It speaks the routes under ``/internal/image-steps/v1`` that
``gmlx.serve.patches.image_steps`` serves: it uploads a tensor once by its
hash, defines patch tables by the hash of their JSON, runs a sampling loop
of one step per request and keeps the wall time of each step beside the
forward time the server reports, so the difference is the transport.
``gmlx image bench --via-steps`` and the end-to-end test drive it.

A transport is a callable ``(method, path, body, headers)`` that returns
``(status, headers, body)``; :func:`http_transport` keeps one HTTP/1.1
connection open to a server.
"""

from __future__ import annotations

import http.client
import json
import time
import urllib.parse
from dataclasses import dataclass

import numpy as np

from .image_wire import CONTENT_TYPE, CONTRACT, CONTRACT_HEADER, PREFIX
from .image_wire import read, tensor_hash, write


class StepsClientError(RuntimeError):
    """A refused request, with its status and JSON body."""

    def __init__(self, status: int, body: dict):
        super().__init__(f"{status}: {body.get('error', body)}")
        self.status = status
        self.body = body


class MissingTensors(StepsClientError):
    """A 409 that names tensors or tables the server does not hold."""

    @property
    def missing(self) -> list:
        return list(self.body.get("missing") or [])


def http_transport(base_url: str, *, api_key: str | None = None,
                   timeout: float = 600.0):
    """A transport over one keep-alive HTTP/1.1 connection to
    ``base_url``, reopened when the server closes it."""
    u = urllib.parse.urlsplit(base_url)
    host, port = u.hostname or "127.0.0.1", u.port or 80
    state = {"conn": None}

    def call(method, path, body, headers):
        hdrs = dict(headers)
        if api_key:
            hdrs["Authorization"] = f"Bearer {api_key}"
        for attempt in (0, 1):
            conn = state["conn"]
            if conn is None:
                conn = http.client.HTTPConnection(host, port, timeout=timeout)
                state["conn"] = conn
            try:
                conn.request(method, path, body=body, headers=hdrs)
                r = conn.getresponse()
                data = r.read()
                return r.status, {k.lower(): v for k, v in r.getheaders()}, \
                    data
            except (http.client.RemoteDisconnected, BrokenPipeError,
                    ConnectionResetError):
                conn.close()
                state["conn"] = None
                if attempt:
                    raise
        raise AssertionError("unreachable")

    return call


def array_hash(a) -> str:
    """The contract hash of a numpy array as the wire carries it."""
    (t,) = read(write({"t": a}))[0].values()
    return t.hash()


@dataclass
class StepRecord:
    """The times of one step request, in milliseconds."""
    wall_ms: float
    forward_ms: float
    server_ms: float
    prefix_cache: str

    @property
    def overhead_ms(self) -> float:
        return self.wall_ms - self.forward_ms


class StepsClient:
    def __init__(self, transport, *, version: str | None = None,
                 clock=time.perf_counter):
        self.transport = transport
        self.version = version or f"{CONTRACT[0]}.{CONTRACT[1]}"
        self.clock = clock
        self.records: list[StepRecord] = []
        self._uploaded: set = set()

    # Plumbing.

    def _call(self, method, path, body=None, *, ctype=None, versioned=True):
        headers = {}
        if versioned:
            headers[CONTRACT_HEADER] = self.version
        if ctype:
            headers["Content-Type"] = ctype
        return self.transport(method, f"{PREFIX}/{path}", body, headers)

    @staticmethod
    def _check(status, body) -> None:
        if status < 400:
            return
        try:
            doc = json.loads(body or b"{}")
        except ValueError:
            doc = {"error": body[:200].decode("latin-1")}
        cls = MissingTensors if status == 409 and "missing" in doc else \
            StepsClientError
        raise cls(status, doc)

    def json(self, method, path, doc=None, *, versioned=True) -> dict:
        body = None if doc is None else json.dumps(doc).encode()
        status, _h, data = self._call(method, path, body,
                                      ctype="application/json" if body
                                      else None, versioned=versioned)
        self._check(status, data)
        return json.loads(data or b"{}")

    def container(self, path, tensors, fields=None):
        status, headers, data = self._call("POST", path,
                                           write(tensors, fields),
                                           ctype=CONTENT_TYPE)
        self._check(status, data)
        got, fields_out = read(data)
        return ({k: _owned(t) for k, t in got.items()}, fields_out, headers)

    # Routes.

    def capabilities(self) -> dict:
        return self.json("GET", "capabilities", versioned=False)

    def status(self, model: str) -> dict:
        return self.json("GET", f"models/{model}/status")

    def manifest(self, model: str) -> dict:
        return self.json("GET", f"models/{model}/manifest")

    def begin(self, model: str) -> str:
        return self.json("POST", "runs/begin", {"model": model})["run"]

    def end(self, run: str) -> dict:
        return self.json("POST", f"runs/{run}/end", {})

    def interrupt(self, run: str | None = None) -> int:
        return self.json("POST", "interrupt", {"run": run})["interrupted"]

    def encode(self, model: str, text: str, images=()) -> tuple:
        """``(fields, tensors)`` of POST encode with u8 images (H, W, 3)."""
        tensors = {f"image_{i}": np.ascontiguousarray(im, np.uint8)
                   for i, im in enumerate(images)}
        got, fields, _ = self.container("encode", tensors,
                                        {"model": model, "text": text})
        self._uploaded.update([fields["cond"], *fields["ref_latents"]])
        return fields, got

    def put(self, a, *, kind: str = "tensors") -> str:
        """Upload ``a`` once and return its hash."""
        h = array_hash(a)
        if h in self._uploaded:
            return h
        status, _h, data = self._call("HEAD", f"{kind}/{h}")
        if status != 200:
            status, _h, data = self._call("PUT", f"{kind}/{h}",
                                          write({"t": a}), ctype=CONTENT_TYPE)
            self._check(status, data)
        self._uploaded.add(h)
        return h

    def define_table(self, model: str, body: dict) -> str:
        raw = json.dumps(body, sort_keys=True,
                         separators=(",", ":")).encode()
        import hashlib

        tid = hashlib.sha256(raw).hexdigest()
        status, _h, data = self._call(
            "PUT", f"patch-tables/{tid}?model={urllib.parse.quote(model)}",
            raw, ctype="application/json")
        self._check(status, data)
        return tid

    def step(self, model: str, run: str | None, x, timestep, rows: list,
             patch_table: str | None = None) -> np.ndarray:
        """The model output for latents ``x`` (B, C, h, w) at ``timestep``
        under convention comfy, one row dict per batch row. Records the
        times of the call."""
        t = np.asarray(timestep, np.float32).reshape(-1)
        fields = {"model": model, "run": run, "convention": "comfy",
                  "patch_table": patch_table, "rows": rows}
        t0 = self.clock()
        got, _f, headers = self.container(
            "step", {"x": np.ascontiguousarray(x, np.float32),
                     "timestep": t}, fields)
        wall = (self.clock() - t0) * 1000
        self.records.append(StepRecord(
            wall, float(headers.get("x-gmlx-forward-ms", "nan")),
            float(headers.get("x-gmlx-server-ms", "nan")),
            headers.get("x-gmlx-prefix-cache", "")))
        return got["out"]

    def sample(self, model: str, noise, sigmas, rows: list, *,
               cfg: float = 1.0, neg_rows: list | None = None,
               patch_table: str | None = None, run: str | None = None,
               update=None, on_step=None) -> np.ndarray:
        """Sample from ``noise`` (B, C, h, w) over ``sigmas`` with one step
        request per sigma. With ``cfg`` other than 1 each request carries
        the negative rows after the positive ones and the client guides.
        ``update(x, v, i)`` moves ``x`` from ``sigmas[i]`` to
        ``sigmas[i + 1]``; the default is Euler in float32, as
        ``gmlx.image.samplers.Euler`` computes it. A 409 re-uploads
        nothing by itself: the caller uploads what it names and retries."""
        s = np.asarray(sigmas, np.float32)
        x = np.asarray(noise, np.float32)
        b = x.shape[0]
        guided = cfg != 1.0
        if guided and (neg_rows is None or len(neg_rows) != len(rows)):
            raise ValueError("guidance needs one negative row per row")
        step_rows = rows + (neg_rows if guided else [])
        for i in range(len(s) - 1):
            xin = np.concatenate([x, x]) if guided else x
            ts = np.full((len(step_rows),), s[i], np.float32)
            out = self.step(model, run, xin, ts, step_rows, patch_table)
            v = guide(out[:b], out[b:], cfg) if guided else out
            x = update(x, v, i) if update is not None else euler(x, v, s, i)
            if on_step is not None:
                on_step(i, x)
        return x


def euler(x, v, sigmas, i: int) -> np.ndarray:
    """``x + (sigma_next - sigma) * v`` in float32."""
    dt = np.float32(sigmas[i + 1]) - np.float32(sigmas[i])
    return x.astype(np.float32) + np.float32(dt) * v.astype(np.float32)


def guide(pos, neg, cfg: float) -> np.ndarray:
    """``neg + cfg * (pos - neg)`` in float32."""
    pos, neg = pos.astype(np.float32), neg.astype(np.float32)
    return neg + np.float32(cfg) * (pos - neg)


def overhead_ms_median(records) -> float | None:
    vals = sorted(r.overhead_ms for r in records)
    if not vals:
        return None
    n = len(vals)
    return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2


def _owned(t):
    """A WireTensor as a numpy array that owns its bytes."""
    return t.numpy().copy()


__all__ = ["StepsClient", "StepsClientError", "MissingTensors",
           "http_transport", "array_hash", "euler", "guide",
           "overhead_ms_median", "tensor_hash"]
