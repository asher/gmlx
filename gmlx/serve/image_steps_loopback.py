"""The image step routes on a loopback port of this process, over one
loaded pipeline, for ``gmlx image bench --via-steps``.

The routes run on their own app with the server's handlers and the image
worker, without the serve middleware. Every MLX op of the pipeline runs on
the image worker, as in the server: MLX streams belong to the thread that
made them, so the caller loads, encodes and decodes through :func:`work`.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace


def work(fn, *args):
    """``fn(*args)`` on the image worker."""
    from .image_service import WORKER

    return WORKER.submit(fn, *args).result()


class OneModelHost:
    """A host of one served pipeline under the id ``model_id``. It writes
    no request log lines."""

    log_requests = False

    def __init__(self, model_id: str, pipe, paths: dict):
        self.model_id = model_id
        self.spec = SimpleNamespace(id=model_id, path=paths["dit"],
                                    encoder=paths.get("encoder"),
                                    vae=paths.get("vae"), encoder_model=None)
        self.served = SimpleNamespace(
            pipe=pipe, spec=self.spec, adapters=(),
            plan=SimpleNamespace(keep_encoder=True))

    def resolve(self, model):
        if model not in (None, "", self.model_id):
            raise LookupError(f"this server serves {self.model_id!r} only")
        return self.spec.path, self.spec

    def acquire(self, model):
        self.resolve(model)
        served = self.served

        class Hold:
            _entry = SimpleNamespace(pipeline=served)

            def release(self):
                pass

        return Hold()

    def models(self) -> dict:
        return {self.model_id: self.spec}

    def resident(self, path) -> bool:
        return True

    def plan(self, spec):
        return None

    def unload(self, path, spec):
        from .patches.image_steps import _err

        return _err(409, "this server keeps its one model loaded")


@contextmanager
def serve(host):
    """Serve the step routes over ``host`` on a loopback port. Yields the
    base URL."""
    import uvicorn
    from fastapi import FastAPI

    from gmlx.config import ImageServerCfg

    from .patches.image_steps import mount

    app = FastAPI()
    mount(app, ImageServerCfg(), host)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(
        app, log_level="warning", lifespan="off", access_log=False))
    t = threading.Thread(target=server.run, kwargs={"sockets": [sock]},
                         name="image-steps-loopback", daemon=True)
    t.start()
    deadline = time.monotonic() + 30
    while not server.started:
        if not t.is_alive() or time.monotonic() > deadline:
            raise RuntimeError("the loopback step server did not start")
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        t.join(timeout=10)
        sock.close()
        # Back to the server's host, with no runs or uploads left.
        mount(app, ImageServerCfg(steps_api=False))


def model_id(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]
