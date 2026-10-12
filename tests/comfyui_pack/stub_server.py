#!/usr/bin/env python3
"""A stub gmlx server for the pack's tests: the image step contract with a
stub denoiser, a stub text encoder and a stub VAE. Standard library and
numpy only; it shares no code with the pack, so the tests check the pack's
wire format against an independent reader and writer.

Usage: stub_server.py [--port 0] [--step-delay-ms 0] [--contract 1.0]
                      [--api-key KEY] [--cond-tokens 32] [--gmlx-version stub]
                      [--no-contract-field] [--default-model ID]
It prints one JSON line with its address, then serves until killed.
Test routes outside the contract: GET /stub/log, POST /stub/log/reset,
POST /stub/forget (drop every tensor, adapter and table held).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

PREFIX = "/internal/image-steps/v1"
CONTENT_TYPE = "application/vnd.gmlx.tensors+safetensors"
HEADER = "X-Gmlx-Steps-Contract"
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

COMFY = {
    "z_image": {"model_class": "ZImage", "image_model": "lumina2",
                "latent": {"channels": 16, "stride": 8, "patch": 2},
                "vae": {"channels_out": 3, "stride": 8}, "edit": False,
                "context": ("cap_embedder.1.weight", -1)},
    "qwen_image21": {"model_class": "QwenImage21", "image_model": "qwen_image21",
                     "latent": {"channels": 64, "stride": 16, "patch": 1},
                     "vae": {"channels_out": 4, "stride": 16}, "edit": True,
                     "context": ("txt_in.text_norm.weight", 0)},
    # The text features of Krea 2 are its encoder's 12 tapped layers side by side.
    "krea2": {"model_class": "Krea2", "image_model": "krea2",
              "latent": {"channels": 16, "stride": 8, "patch": 2},
              "vae": {"channels_out": 3, "stride": 8}, "edit": False,
              "context": ("txtfusion.layerwise_blocks.0.prenorm.scale", 0, 12)},
}
TURBO_SIGMAS = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0]
MODELS = {
    "z-image-turbo": ("z_image", "turbo", {"steps": 8, "cfg": 1.0, "shift": 3.0, "sigmas": None}),
    "qwen-image-2.1": ("qwen_image21", "base", {"steps": 40, "cfg": 1.0, "shift": 1.0, "sigmas": None}),
    "qwen-image-2.1-turbo": ("qwen_image21", "turbo",
                             {"steps": 8, "cfg": 1.0, "shift": 1.0, "sigmas": TURBO_SIGMAS}),
    "krea-2-turbo": ("krea2", "turbo", {"steps": 8, "cfg": 1.0, "shift": 1.0, "sigmas": None}),
}

_NP = {"F32": "<f4", "F16": "<f2", "BF16": "<u2", "U8": "u1", "BOOL": "?"}


def read_body(body: bytes):
    n = struct.unpack("<Q", body[:8])[0]
    header = json.loads(body[8:8 + n])
    meta = header.pop("__metadata__", None) or {}
    data = body[8 + n:]
    out = {}
    for name, spec in header.items():
        s, e = spec["data_offsets"]
        raw = data[s:e]
        a = np.frombuffer(raw, dtype=_NP[spec["dtype"]]).reshape(spec["shape"])
        if spec["dtype"] == "BF16":
            a = (a.astype(np.uint32) << 16).view(np.float32)
        out[name] = (spec["dtype"], list(spec["shape"]), raw, a)
    return out, (json.loads(meta["gmlx"]) if "gmlx" in meta else {})


def write_body(arrays: dict, fields: dict | None = None) -> bytes:
    header, chunks, off = {}, [], 0
    for name, a in arrays.items():
        a = np.ascontiguousarray(a)
        dt = {np.dtype("<f4"): "F32", np.dtype("<f2"): "F16", np.dtype("u1"): "U8"}[a.dtype]
        raw = a.tobytes()
        header[name] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [off, off + len(raw)]}
        chunks.append(raw)
        off += len(raw)
    if fields is not None:
        header["__metadata__"] = {"gmlx": json.dumps(fields, sort_keys=True, separators=(",", ":"))}
    hj = json.dumps(header, separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    return struct.pack("<Q", len(hj)) + hj + b"".join(chunks)


def thash(dtype: str, shape, raw: bytes) -> str:
    h = hashlib.sha256()
    h.update(dtype.encode())
    h.update(json.dumps([int(d) for d in shape]).encode())
    h.update(raw)
    return h.hexdigest()


def scalar(s: str) -> float:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def load_manifests() -> dict:
    out = {}
    for fam in COMFY:
        with open(os.path.join(FIXTURES, f"manifest-{fam}.json")) as f:
            out[fam] = json.load(f)
    return out


class Run:
    def __init__(self, model):
        self.id = uuid.uuid4().hex[:16]
        self.model = model
        self.gen = 0
        self.ended = False


class State:
    def __init__(self, a):
        self.contract = tuple(int(v) for v in a.contract.split("."))
        self.api_key = a.api_key
        self.gmlx_version = a.gmlx_version
        self.no_contract_field = a.no_contract_field
        self.delay = a.step_delay_ms / 1000.0
        self.cond_tokens = a.cond_tokens
        self.default_model = a.default_model
        self.manifests = load_manifests()
        self.tensors: dict = {}
        self.adapters: dict = {}
        self.tables: dict = {}
        self.runs: dict = {}
        self.log: list = []
        self.lock = threading.Lock()

    def event(self, **kw):
        kw["t"] = time.time()
        with self.lock:
            self.log.append(kw)

    def context_dim(self, fam):
        name, axis, *times = COMFY[fam]["context"]
        t = next(t for t in self.manifests[fam]["tensors"]
                 if t["name"] == name or t["name"].endswith("." + name))
        return t["shape"][axis] * (times[0] if times else 1)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "gmlx-stub/1"
    disable_nagle_algorithm = True
    st: State = None

    def log_message(self, *a):
        pass

    def _send(self, code, body: bytes, ctype="application/json", headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code, obj, headers=None):
        self._send(code, json.dumps(obj).encode(), headers=headers)

    def _err(self, status, msg, **extra):
        self._json(status, {"error": msg, **extra})

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(n) if n else b""

    def _gate(self, sub: str) -> bool:
        st = self.st
        if st.api_key and self.headers.get("Authorization") != f"Bearer {st.api_key}":
            self._err(401, "a valid API key is required")
            return False
        if sub == "/capabilities":
            return True
        v = self.headers.get(HEADER)
        c = f"{st.contract[0]}.{st.contract[1]}"
        if v is None:
            self._err(400, f"the request has no {HEADER} header; send {c}")
            return False
        major, _, minor = v.partition(".")
        if int(major) != st.contract[0]:
            side = "the gmlx server" if int(major) > st.contract[0] else "the client"
            self._err(400, f"step contract {int(major)}.{int(minor or 0)} is not served; this "
                           f"server speaks {c}. Upgrade {side}.")
            return False
        return True

    def _model(self, mid):
        if mid not in MODELS:
            self._err(404, f"no image model {mid!r}")
            return None
        return MODELS[mid]

    def do_HEAD(self):
        p = self.path
        if not p.startswith(PREFIX) or not self._gate(p[len(PREFIX):]):
            return self._send(404, b"") if not p.startswith(PREFIX) else None
        kind, _, h = p[len(PREFIX) + 1:].partition("/")
        store = {"tensors": self.st.tensors, "adapters": self.st.adapters}.get(kind)
        self._send(200 if store is not None and h in store else 404, b"")

    def do_GET(self):
        st, p = self.st, self.path
        if p == "/health":
            return self._json(200, {"status": "ok"})
        if p == "/stub/log":
            with st.lock:
                return self._json(200, {"log": list(st.log), "tables": sorted(st.tables),
                                        "tensors": len(st.tensors), "adapters": len(st.adapters)})
        if not p.startswith(PREFIX):
            return self._err(404, "not found")
        sub = p[len(PREFIX):]
        if not self._gate(sub):
            return
        if sub == "/capabilities":
            # A server from before the contract names none.
            contract = {} if st.no_contract_field else {
                "contract": {"major": st.contract[0], "minor": st.contract[1]},
                "min_client": {"major": st.contract[0], "minor": 0}}
            return self._json(200, {
                **contract, "gmlx_version": st.gmlx_version,
                "wire": {"container": "safetensors", "dtypes": ["f32", "f16", "bf16", "u8"]},
                "families": sorted(COMFY), "conventions": ["comfy"], "patch_kinds": ["lora"],
                "features": ["prefix_cache", "noise_mask", "rgba_vae", "tiled_vae"],
                "limits": {"max_tensor_bytes": 1 << 30, "max_adapter_bytes": 1 << 30, "max_rows": 16}})
        if sub == "/models":
            return self._json(200, {"models": [{"model": m, "family": f, "variant": v, "resident": True}
                                               for m, (f, v, _s) in MODELS.items()],
                                    "default_model": st.default_model})
        parts = sub.strip("/").split("/")
        if len(parts) == 3 and parts[0] == "models":
            m = self._model(parts[1])
            if m is None:
                return
            fam, variant, sampling = m
            if parts[2] == "manifest":
                return self._json(200, {**st.manifests[fam], "model": parts[1]})
            if parts[2] == "status":
                f = COMFY[fam]
                return self._json(200, {
                    "model": parts[1], "family": fam, "variant": variant, "resident": True,
                    "comfy": {"model_class": f["model_class"], "image_model": f["image_model"]},
                    "latent": dict(f["latent"]), "vae": dict(f["vae"]), "edit_capable": f["edit"],
                    "sampling": dict(sampling), "compute_dtype": "bf16",
                    "memory": {"dit_bytes": 1, "encoder_bytes": 1, "vae_bytes": 1,
                               "activation_reserve_bytes": 0, "resident_bytes": 3}})
        self._err(404, "not found", path=p)

    def do_PUT(self):
        st, p = self.st, self.path
        if not p.startswith(PREFIX):
            return self._err(404, "not found")
        path, _, query = p.partition("?")
        sub = path[len(PREFIX):]
        if not self._gate(sub):
            return
        body = self._body()
        kind, _, h = sub.strip("/").partition("/")
        if kind in ("tensors", "adapters"):
            if self.headers.get("Content-Type", "").split(";")[0] != CONTENT_TYPE:
                return self._err(415, f"the body is {CONTENT_TYPE}")
            tensors, _ = read_body(body)
            if len(tensors) != 1:
                return self._err(400, f"an upload holds one tensor, not {len(tensors)}")
            (dt, shape, raw, a), = tensors.values()
            got = thash(dt, shape, raw)
            if got != h:
                return self._err(400, "the tensor's hash is not the one in the path", got=got)
            (st.tensors if kind == "tensors" else st.adapters)[h] = (dt, shape, a)
            st.event(kind="put_" + kind[:-1], hash=h, dtype=dt, shape=shape)
            return self._json(200, {"hash": h})
        if kind == "patch-tables":
            canon = json.dumps(json.loads(body), sort_keys=True, separators=(",", ":")).encode()
            got = hashlib.sha256(canon).hexdigest()
            if got != h:
                return self._err(400, "the table's id is not the one in the path", got=got)
            table = json.loads(body)
            model = dict(kv.split("=", 1) for kv in query.split("&") if "=" in kv).get("model") \
                or table.get("model")
            if self._model(model) is None:
                return
            missing = sorted({pp.get(s) for e in table["entries"] for pp in e["patches"]
                              for s in ("up", "down") if pp.get(s) not in st.adapters})
            if missing:
                st.event(kind="table_missing", id=h, missing=len(missing))
                return self._err(409, "adapters are not held", missing=missing)
            for e in table["entries"]:
                for pp in e["patches"]:
                    extra = set(pp) - {"adapter", "up", "down", "alpha", "strength",
                                       "strength_model", "kind"}
                    if extra:
                        return self._err(400, f"patch fields {sorted(extra)} are not read")
                    if float(pp.get("strength_model", 1.0)) != 1.0:
                        return self._err(400, "patch table refused: model strength")
            st.tables[h] = table
            st.event(kind="put_table", id=h, model=model, entries=len(table["entries"]),
                     bands=sum(1 for e in table["entries"] if e.get("band")),
                     strengths=sorted({pp["strength"] for e in table["entries"] for pp in e["patches"]}))
            return self._json(200, {"id": h})
        self._err(404, "not found")

    def do_POST(self):
        st, p = self.st, self.path
        if p == "/stub/log/reset":
            with st.lock:
                st.log.clear()
            return self._json(200, {})
        if p == "/stub/forget":
            st.tensors.clear()
            st.adapters.clear()
            st.tables.clear()
            st.event(kind="forget")
            return self._json(200, {})
        if not p.startswith(PREFIX):
            return self._err(404, "not found")
        sub = p[len(PREFIX):]
        if not self._gate(sub):
            return
        t0 = time.perf_counter()
        body = self._body()
        ctype = self.headers.get("Content-Type", "").split(";")[0]
        if sub == "/runs/begin":
            req = json.loads(body or b"{}")
            if self._model(req.get("model")) is None:
                return
            run = Run(req["model"])
            st.runs[run.id] = run
            st.event(kind="run_begin", run=run.id, model=run.model)
            return self._json(200, {"run": run.id, "idle_s": 120.0})
        if sub.startswith("/runs/") and sub.endswith("/end"):
            rid = sub.split("/")[2]
            run = st.runs.pop(rid, None)
            if run is not None:
                run.ended = True
                run.gen += 1
            st.event(kind="run_end", run=rid)
            return self._json(200, {"run": rid, "ended": run is not None})
        if sub == "/interrupt":
            rid = json.loads(body or b"{}").get("run")
            hit = [r for r in st.runs.values() if not rid or r.id == rid]
            for r in hit:
                r.gen += 1
            st.event(kind="interrupt", run=rid)
            return self._json(200, {"interrupted": len(hit)})
        if sub == "/adapters/register":
            return self._err(404, "no such adapter file under the server's model folders")
        if sub == "/models/load":
            mid = json.loads(body or b"{}").get("model")
            if self._model(mid) is None:
                return
            return self._json(200, {"model": mid, "resident": True})
        if sub == "/models/unload":
            mid = json.loads(body or b"{}").get("model")
            if self._model(mid) is None:
                return
            return self._json(200, {"model": mid, "unloaded": True})
        if sub == "/encode":
            return self._encode(body, ctype)
        if sub == "/step":
            return self._step(body, t0)
        if sub in ("/vae/decode", "/vae/encode"):
            return self._vae(sub, body)
        self._err(404, "not found")

    def _encode(self, body, ctype):
        st = self.st
        images = []
        if ctype == CONTENT_TYPE:
            tensors, req = read_body(body)
            for name in sorted(tensors, key=lambda n: (len(n), n)):
                dt, shape, _raw, a = tensors[name]
                if dt != "U8" or len(shape) != 3 or shape[2] not in (3, 4):
                    return self._err(400, f"image {name!r} is u8 (H, W, 3 or 4)")
                images.append(a)
        else:
            req = json.loads(body)
            if req.get("images"):
                return self._err(400, "the stub takes images in a container only")
        m = self._model(req.get("model"))
        if m is None:
            return
        fam = m[0]
        if images and not COMFY[fam]["edit"]:
            return self._err(400, "this family takes no reference images")
        n = int(req.get("num_tokens") or st.cond_tokens)
        rng = np.random.default_rng(int(hashlib.sha256(req["text"].encode()).hexdigest()[:8], 16))
        cond = (rng.standard_normal((1, n, st.context_dim(fam)), dtype=np.float32) * 0.5)
        out = {"cond": cond}
        stride, ch = COMFY[fam]["latent"]["stride"], COMFY[fam]["latent"]["channels"]
        want_refs = req.get("refs", True)
        for i, im in enumerate(images if want_refs else []):
            h, w = -(-im.shape[0] // stride), -(-im.shape[1] // stride)
            out[f"ref_{i}"] = np.full((1, ch, h, w), float(im.mean()) / 255.0, np.float32)
        hashes = {}
        for k, a in out.items():
            a = np.ascontiguousarray(a)
            hashes[k] = thash("F32", a.shape, a.tobytes())
            st.tensors[hashes[k]] = ("F32", list(a.shape), a)
        fields = {"cond": hashes["cond"], "ref_latents": [hashes[k] for k in out if k != "cond"],
                  "image_slots": [2 + 3 * i for i in range(len(images))], "num_tokens": n}
        st.event(kind="encode", model=req["model"], text=req["text"], tokens=n, images=len(images), refs=want_refs)
        self._send(200, write_body(out, fields), ctype=CONTENT_TYPE)

    def _step(self, body, t0):
        st = self.st
        tensors, req = read_body(body)
        rid = req.get("run")
        run = st.runs.get(rid) if rid else None
        if rid and run is None:
            return self._err(404, f"no run {rid!r}; it ended or sat idle past 120 seconds")
        if self._model(req.get("model")) is None:
            return
        x, ts = tensors.get("x"), tensors.get("timestep")
        rows = req.get("rows") or []
        if x is None or ts is None or len(x[1]) != 4 or x[1][0] != len(rows) or not rows:
            return self._err(400, "x is (B, C, h, w) with one row per entry of rows")
        missing = []
        for r in rows:
            for h in [r.get("cond")] + list(r.get("ref_latents") or []):
                if h not in st.tensors:
                    missing.append(h)
        if req.get("patch_table") and req["patch_table"] not in st.tables:
            missing.append(req["patch_table"])
        if missing:
            st.event(kind="step_missing", run=rid, missing=len(set(missing)))
            return self._err(409, "not held", missing=sorted(set(missing)))
        gen = run.gen if run else 0
        deadline = time.perf_counter() + st.delay
        while time.perf_counter() < deadline:
            if run is not None and (run.gen != gen or run.ended):
                st.event(kind="step_interrupted", run=rid, by="run_end" if run.ended else "interrupt")
                return self._err(409, "the step was interrupted", code="interrupted")
            time.sleep(0.005)
        tf = time.perf_counter()
        xa = x[3].astype(np.float32)
        t = np.broadcast_to(ts[3].astype(np.float32).reshape(-1), (len(rows),))
        k = np.array([scalar(r["cond"]) + 0.05 * sum(scalar(h) for h in r.get("ref_latents") or [])
                      for r in rows], np.float32)
        k += np.float32(scalar(req.get("patch_table") or "none") * 0.1)
        out = xa * np.float32(-0.25) + (t * np.float32(0.05) + k * np.float32(0.02)).reshape(-1, 1, 1, 1)
        fwd = (time.perf_counter() - tf) * 1000
        st.event(kind="step", run=rid, model=req["model"], rows=len(rows), x_shape=x[1], x_dtype=x[0],
                 timestep=[float(v) for v in t], convention=req.get("convention"),
                 num_tokens=[r.get("num_tokens") for r in rows],
                 refs=[len(r.get("ref_latents") or []) for r in rows],
                 image_slots=[r.get("image_slots") for r in rows],
                 prefix_cache=[r.get("prefix_cache", True) for r in rows],
                 row_keys=sorted({k for r in rows for k in r}), patch_table=req.get("patch_table"),
                 fields=sorted(req))
        self._send(200, write_body({"out": out.astype(np.float32)}), ctype=CONTENT_TYPE, headers={
            "X-Gmlx-Forward-Ms": f"{fwd:.3f}",
            "X-Gmlx-Server-Ms": f"{(time.perf_counter() - t0) * 1000:.3f}",
            "X-Gmlx-Prefix-Cache": "none"})

    def _vae(self, sub, body):
        tensors, req = read_body(body)
        m = self._model(req.get("model"))
        if m is None:
            return
        fam = m[0]
        stride = COMFY[fam]["vae"]["stride"]
        ch = COMFY[fam]["latent"]["channels"]
        if sub == "/vae/decode":
            lat = tensors.get("latents")
            if lat is None or len(lat[1]) != 4 or lat[1][1] != ch:
                return self._err(400, f"vae/decode takes latents (B, {ch}, h, w)")
            a = lat[3].astype(np.float32)
            g = 1.0 / (1.0 + np.exp(-a.mean(axis=1)))
            img = np.repeat(np.repeat(g, stride, axis=1), stride, axis=2)[..., None]
            c = 4 if req.get("alpha") else 3
            img = np.concatenate([img] * c, axis=-1).astype(np.float32)
            self.st.event(kind="vae_decode", shape=lat[1], alpha=bool(req.get("alpha")))
            return self._send(200, write_body({"image": img}), ctype=CONTENT_TYPE)
        im = tensors.get("image")
        if im is None or len(im[1]) != 4:
            return self._err(400, "vae/encode takes an image (B, H, W, C) in [0, 1]")
        B, H, W, C = im[1]
        if H % stride or W % stride:
            return self._err(400, f"the image is not a multiple of {stride}")
        a = im[3].astype(np.float32)[..., 0].reshape(B, H // stride, stride, W // stride, stride)
        z = a.mean(axis=(2, 4))[:, None].repeat(ch, axis=1).astype(np.float32)
        self.st.event(kind="vae_encode", shape=im[1])
        self._send(200, write_body({"latents": z}), ctype=CONTENT_TYPE)


def serve(a):
    Handler.st = State(a)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    print(json.dumps({"listening": f"http://{a.host}:{srv.server_port}"}), flush=True)
    try:
        srv.serve_forever(poll_interval=0.05)
    except KeyboardInterrupt:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--step-delay-ms", type=float, default=0.0)
    ap.add_argument("--contract", default="1.0")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--gmlx-version", default="stub")
    ap.add_argument("--no-contract-field", action="store_true")
    ap.add_argument("--cond-tokens", type=int, default=32)
    ap.add_argument("--default-model", default=None)
    serve(ap.parse_args())


if __name__ == "__main__":
    main()
