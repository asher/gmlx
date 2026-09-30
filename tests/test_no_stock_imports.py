# vendor_upstream: skip
"""gmlx imports nothing from a stock mlx-vlm or mlx-lm.

A child process puts poisoned ``mlx_vlm`` and ``mlx_lm`` packages first on
sys.path: each records the import and raises ImportError. The child then
imports gmlx, runs every CLI verb's ``--help``, loads a tiny llama GGUF and
generates from it, prepares the train and distill LoRA students on a tiny
K-quant GGUF, builds tiny VLMs through the copy's loader, constructs
the serve app without binding a port, and imports every other gmlx module
(training and distillation included). No step may reach a poisoned
package, even under an ``except ImportError``, and no stock name may be in
sys.modules after it. The poison works whether or not the stock packages
are installed. The child runs on the CPU device.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_STOCK = ("mlx_vlm", "mlx_lm")

_POISON = '''\
import builtins
import traceback

frames = [f for f in traceback.extract_stack()[:-1]
          if "importlib" not in f.filename]
builtins.__dict__.setdefault("_stock_hits", []).append(
    (__name__, "".join(traceback.format_list(frames[-6:]))))
raise ImportError(f"stock {__name__} is poisoned: gmlx imports "
                  f"gmlx._vendor.{__name__}")
'''

_CHILD = r'''
import builtins
import contextlib
import importlib
import io
import json
import os
import pkgutil
import sys
import tempfile
import traceback

poison, tests = sys.argv[1], sys.argv[2]
sys.path[:0] = [poison, os.path.join(tests, "load")]
builtins._stock_hits = []
STOCK = ("mlx_vlm", "mlx_lm")
report = {"steps": [], "hits": [], "stray": [], "failed": {}, "skipped": []}

import mlx.core as mx  # noqa: E402

mx.set_default_device(mx.cpu)


def step(name, fn):
    seen = len(builtins._stock_hits)
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            fn()
    except SystemExit:
        pass
    except Exception:
        report["failed"][name] = traceback.format_exc()[-2000:]
    report["steps"].append(name)
    report["hits"] += [[name, *h] for h in builtins._stock_hits[seen:]]
    stray = sorted(m for m in sys.modules if m.partition(".")[0] in STOCK)
    if stray:
        report["stray"].append([name, stray])


def verbs():
    step("import gmlx.commands.cli",
         lambda: importlib.import_module("gmlx.commands.cli"))
    cli = sys.modules.get("gmlx.commands.cli")
    for verb in getattr(cli, "_VERBS", ()):
        step(f"gmlx {verb} --help",
             lambda v=verb: cli.umbrella_main([v, "--help"]))


work = tempfile.mkdtemp()
gguf = os.path.join(work, "tiny.gguf")


def load_gguf():
    import gmlx
    from test_loader_api import _mint_tiny_llama

    _mint_tiny_llama(gguf)
    model, _, _ = gmlx.load_model(gguf)
    mx.eval(model(mx.array([[1, 2, 3]])))


def run_gguf():
    from gmlx.commands.cli import umbrella_main

    code = umbrella_main(["run", gguf, "--prompt", "He", "--max-tokens", "3"])
    assert not code, code


def lora_student(keys):
    import gmlx
    from gmlx.tune.lora import prepare_lora_student
    from test_loader_api import _mint_tiny_llama

    path = os.path.join(work, "tiny-q8.gguf")
    if not os.path.exists(path):
        _mint_tiny_llama(path, q8_0=True)
    model, _, _ = gmlx.load_model(path)
    assert prepare_lora_student(model, rank=2, scale=1.0, keys=keys) > 0
    mx.eval(model(mx.array([[1, 2, 3]])))


def train_student():
    lora_student(None)


def distill_student():
    from gmlx.tune.lora import LORA_KEYS

    lora_student(LORA_KEYS)


def build_vlms():
    import test_owned_model_routing as registration

    for cfg in registration._configs().values():
        registration.gvlm.build_vlm_model(cfg)


def serve_app():
    from fastapi.testclient import TestClient

    import gmlx.serve.bridge_vlm as serving
    from gmlx.config import build_config
    from gmlx.serve import patches
    from gmlx.serve.server import _import_serving

    _import_serving()
    cfg = build_config({"server": {"host": "127.0.0.1", "port": 8080},
                        "models": {"tiny": {"path": gguf}}})
    serving.register_resolved_models(cfg)
    serving.install_gguf_server_bridge()
    patches.install_server_patches(cfg, reload_fn=None)
    app = importlib.import_module("gmlx._vendor.mlx_vlm.server.app").app
    r = TestClient(app, base_url="http://127.0.0.1").get("/health")
    assert r.status_code == 200, r.status_code


def every_module():
    import gmlx

    for info in pkgutil.walk_packages(gmlx.__path__, "gmlx.",
                                      onerror=lambda name: None):
        name = info.name
        if name.startswith("gmlx._vendor.") or name.endswith("__main__"):
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                importlib.import_module(name)
        except Exception as e:
            # An optional dependency this venv lacks; a poisoned import
            # shows in the hits either way.
            report["skipped"].append(f"{name}: {type(e).__name__}: {e}"[:300])


step("import gmlx", lambda: importlib.import_module("gmlx"))
verbs()
step("load a llama GGUF", load_gguf)
step("gmlx run on the GGUF", run_gguf)
step("prepare a gmlx train LoRA student on a K-quant GGUF", train_student)
step("prepare a gmlx distill LoRA student on a K-quant GGUF", distill_student)
step("build tiny VLMs", build_vlms)
step("construct the serve app", serve_app)
step("import every gmlx module", every_module)
# A hit no step spans still fails: the parent sees every recorded import.
report["unattributed"] = len(builtins._stock_hits) - len(report["hits"])
print(json.dumps(report))
'''


def test_gmlx_never_imports_stock_packages(tmp_path):
    for name in _STOCK:
        (tmp_path / name).mkdir()
        (tmp_path / name / "__init__.py").write_text(_POISON)
    env = dict(os.environ, PYTHONPATH=str(_ROOT))
    r = subprocess.run(
        [sys.executable, "-c", _CHILD, str(tmp_path), str(_ROOT / "tests")],
        capture_output=True, text=True, timeout=900, env=env, cwd=tmp_path)
    assert r.returncode == 0, r.stderr[-4000:]
    report = json.loads(r.stdout.strip().splitlines()[-1])
    assert len(report["steps"]) > 20, report["steps"]
    hits = [f"{step}: import {name}\n{where}"
            for step, name, where in report["hits"]]
    assert not hits, "\n\n".join(hits[:5])
    assert report["unattributed"] == 0
    assert not report["stray"], report["stray"]
    assert not report["failed"], "\n\n".join(
        f"{k}:\n{v}" for k, v in report["failed"].items())
