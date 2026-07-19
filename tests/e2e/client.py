"""A tiny stdlib HTTP client for the gmlx / mlx-vlm server endpoints.

urllib only - the harness adds no runtime deps. Every call returns
``(status, parsed_json_or_text)`` and never raises on an HTTP error status, so a
scenario can assert on a 4xx/5xx body (e.g. unknown-id -> 404) the same way it
asserts on a 200.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import urllib.error
import urllib.request
from typing import Optional, Tuple


class Client:
    def __init__(self, base_url: str, *, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # low level
    def _request(self, method: str, path: str, body: Optional[dict] = None,
                 *, timeout: Optional[float] = None) -> Tuple[int, object]:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                raw = resp.read().decode("utf-8", "replace")
                return resp.status, _maybe_json(raw)
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "replace")
            return e.code, _maybe_json(raw)
        except urllib.error.URLError as e:
            return 0, f"URLError: {e.reason}"
        except Exception as e:                       # noqa: BLE001 - report, don't raise
            return -1, f"{type(e).__name__}: {e}"

    def get(self, path: str, *, timeout: Optional[float] = None):
        return self._request("GET", path, timeout=timeout)

    def post(self, path: str, body: Optional[dict] = None,
             *, timeout: Optional[float] = None):
        return self._request("POST", path, body, timeout=timeout)

    # typed endpoints
    def health(self):
        return self.get("/health", timeout=10.0)

    def models(self):
        return self.get("/v1/models", timeout=15.0)

    def metrics(self):
        return self.get("/v1/metrics", timeout=15.0)

    def cache_stats(self):
        return self.get("/v1/cache/stats", timeout=15.0)

    def cache_reset(self):
        return self.post("/v1/cache/reset", {}, timeout=15.0)

    def unload(self, model: Optional[str] = None):
        body = {"model": model} if model else {}
        return self.post("/unload", body, timeout=60.0)

    def reload(self):
        return self.post("/v1/reload", {}, timeout=60.0)

    def chat(self, model: str, messages: list, *, max_tokens: int = 256,
             stream: bool = False, image_paths: Optional[list] = None,
             **sampling) -> Tuple[int, object]:
        """POST /v1/chat/completions. ``image_paths`` attaches local images as
        data-URIs to the last user message (for the VLM tier). Extra ``sampling``
        kwargs (temperature/top_p/top_k/min_p/repetition_penalty/profile/...) ride
        through as request fields (the schema is extra=allow)."""
        msgs = messages
        if image_paths:
            msgs = _attach_images(messages, image_paths)
        body = {"model": model, "messages": msgs, "max_tokens": max_tokens,
                "stream": stream, **{k: v for k, v in sampling.items()
                                     if v is not None}}
        if stream:
            return self._chat_stream(body)
        return self.post("/v1/chat/completions", body)

    def _chat_stream(self, body: dict) -> Tuple[int, object]:
        """Consume an SSE stream and reconstruct a non-stream-shaped body so the
        same floor checks apply. Returns (status, reconstructed_body)."""
        url = f"{self.base_url}/v1/chat/completions"
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(), method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "text/event-stream")
        chunks, finish, n_events = [], None, 0
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for line in resp:
                    line = line.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:"):].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        ev = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    n_events += 1
                    delta = (ev.get("choices") or [{}])[0]
                    piece = (delta.get("delta") or {}).get("content")
                    if piece:
                        chunks.append(piece)
                    fr = delta.get("finish_reason")
                    if fr:
                        finish = fr
                status = resp.status
        except urllib.error.HTTPError as e:
            return e.code, _maybe_json(e.read().decode("utf-8", "replace"))
        except Exception as e:                       # noqa: BLE001
            return -1, f"{type(e).__name__}: {e}"
        text = "".join(chunks)
        body_out = {
            "choices": [{"message": {"role": "assistant", "content": text},
                         "finish_reason": finish or ("stop" if text else None)}],
            "usage": {"prompt_tokens": 1, "completion_tokens": max(1, n_events)},
            "_stream_events": n_events,
        }
        return status, body_out


def _maybe_json(raw: str):
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def _attach_images(messages: list, image_paths: list) -> list:
    """Return a copy of ``messages`` with images appended to the last user turn as
    OpenAI ``image_url`` data-URI parts."""
    parts = []
    for path in image_paths:
        mime = mimetypes.guess_type(path)[0] or "image/jpeg"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        parts.append({"type": "image_url",
                      "image_url": {"url": f"data:{mime};base64,{b64}"}})
    out = [dict(m) for m in messages]
    for m in reversed(out):
        if m["role"] == "user":
            text = m["content"] if isinstance(m["content"], str) else ""
            m["content"] = [{"type": "text", "text": text}, *parts]
            break
    return out
