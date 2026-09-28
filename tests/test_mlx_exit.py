"""Process and thread exits after mx.compile, with no exit guard.

mlx 0.32.1 dropped the compile-cache cleanup, so a compiled wrapper still
alive at interpreter exit, or a worker thread that ran one, crashed with
SIGSEGV as its thread-local cache entry was freed. mlx 0.32.2 fixed both,
and gmlx runs both exits as stock. These children fail if mlx regresses.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)

# A compiled wrapper kept alive past interpreter shutdown.
PROCESS_EXIT = """\
import ctypes
import mlx.core as mx
mx.set_default_device(mx.cpu)
f = mx.compile(lambda x: (x + 1, x * 2))
f(mx.array(1.0))
ctypes.pythonapi.Py_IncRef(ctypes.py_object(f))
"""

# Worker threads that run a live compiled wrapper, then exit. os._exit
# keeps the process exit out of the result.
THREAD_EXIT = """\
import os
import threading
import mlx.core as mx
mx.set_default_device(mx.cpu)
f = mx.compile(lambda x: (x + 1, x * 2))

def work():
    mx.set_default_device(mx.cpu)
    mx.eval(f(mx.array([1.0, 2.0])))

for _ in range(8):
    t = threading.Thread(target=work)
    t.start()
    t.join()
print("threads done", flush=True)
os._exit(0)
"""


def _child(code):
    return subprocess.run([sys.executable, "-c", code], capture_output=True)


def test_process_exit_after_compile_exits_clean():
    r = _child(PROCESS_EXIT)
    assert r.returncode == 0, (r.returncode, r.stderr.decode()[-500:])


def test_thread_exit_after_compile_exits_clean():
    r = _child(THREAD_EXIT)
    assert r.returncode == 0, (r.returncode, r.stderr.decode()[-500:])
    assert b"threads done" in r.stdout


def test_cli_entry_exits_clean():
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-m", "gmlx", "--version"],
                       capture_output=True, env=env)
    assert r.returncode == 0, r.stderr.decode()[-500:]
    assert r.stdout.strip()


def test_the_server_bridge_leaves_the_engine_thread_stock(monkeypatch):
    # The engine thread ends at unload, as stock does.
    import gmlx.serve.bridge_vlm as bridge

    gen = importlib.import_module("mlx_vlm.server.generation")
    monkeypatch.setattr(gen, bridge._BRIDGE_FLAG, False, raising=False)
    monkeypatch.setattr(gen, "load_model_resources", gen.load_model_resources)
    cls = gen.ResponseGenerator
    for name in ("_run", "stop_and_join"):
        monkeypatch.setattr(cls, name, cls.__dict__[name])
    bridge.install_gguf_server_bridge()
    for name in ("_run", "stop_and_join"):
        assert cls.__dict__[name].__module__ == gen.__name__, name
