# vendor_upstream: skip
"""gmlx imports nothing from a stock mlx-vlm or mlx-lm.

A child process puts poisoned ``mlx_vlm`` and ``mlx_lm`` packages first on
sys.path: each records the import and raises ImportError. The child then
imports gmlx, runs every CLI verb's ``--help``, runs ``python -m gmlx
--version`` and ``python -m gmlx serve --help`` as processes of their own
with the poison first on their path, loads a tiny llama GGUF and generates
from it, prepares the train and distill LoRA students on a tiny K-quant
GGUF, generates from a tiny Pixtral VLM GGUF pair with an image, builds
tiny VLMs through the copy's loader, constructs the serve app without
binding a port, and imports every other gmlx module (training,
distillation and ``__main__`` modules included). No step may reach a
poisoned package, even under an ``except ImportError``, no stock name may
be in sys.modules after it, and only a ``--help`` step may exit, with the
code of a help or of a usage error. A module import may fail only on a
missing optional dependency. The poison works whether or not the stock
packages are installed. The child runs on the CPU device.
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
import json
import os
import traceback

frames = [f for f in traceback.extract_stack()[:-1]
          if "importlib" not in f.filename]
where = "".join(traceback.format_list(frames[-6:]))
builtins.__dict__.setdefault("_stock_hits", []).append((__name__, where))
if os.environ.get("GMLX_POISON_LOG"):
    with open(os.environ["GMLX_POISON_LOG"], "a") as f:
        f.write(json.dumps([__name__, where]) + "\\n")
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
import subprocess
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


def step(name, fn, exits=()):
    """Run ``fn``. An exit code, raised in a SystemExit or returned by a
    CLI entry point, fails the step unless it is in ``exits``. A ``--help``
    step allows the codes of a help and of a usage error."""
    seen = len(builtins._stock_hits)
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            code = fn()
        if isinstance(code, int) and code not in exits:
            report["failed"][name] = f"exit code {code}"
    except SystemExit as e:
        if e.code not in exits:
            report["failed"][name] = f"SystemExit({e.code!r})"
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
             lambda v=verb: cli.umbrella_main([v, "--help"]), exits=(0, 2))


work = tempfile.mkdtemp()
gguf = os.path.join(work, "tiny.gguf")


def module_step(argv, exits=(0,)):
    """``python -m gmlx`` in a process of its own, with the poison first on
    its path. The poison logs each import to a file this step reads."""
    log = os.path.join(work, "poison.log")

    def run():
        open(log, "w").close()
        env = dict(os.environ, GMLX_POISON_LOG=log, KQUANT_FORCE_CPU="1",
                   PYTHONPATH=os.pathsep.join(
                       [poison, os.environ.get("PYTHONPATH", "")]))
        r = subprocess.run([sys.executable, "-m", "gmlx", *argv],
                           capture_output=True, text=True, timeout=300,
                           env=env, cwd=work)
        with open(log) as f:
            builtins._stock_hits += [tuple(json.loads(x)) for x in f]
        assert r.returncode in exits, (r.returncode, r.stderr[-2000:])

    step("python -m gmlx " + " ".join(argv), run)


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


def run_vlm_gguf():
    from gmlx.commands.cli import umbrella_main
    from test_loader_api import _mint_tiny_image, _mint_tiny_pixtral

    text, mmproj, image = (os.path.join(work, n)
                           for n in ("vlm.gguf", "mmproj.gguf", "i.png"))
    _mint_tiny_pixtral(text, mmproj)
    _mint_tiny_image(image)
    code = umbrella_main(["run", text, "--mmproj", mmproj, "--image", image,
                          "--prompt", "He", "--max-tokens", "3"])
    assert not code, code


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


def missing_dependency(e):
    """The name of the optional dependency whose absence raised ``e``, or
    None when ``e`` is not a missing dependency."""
    while e is not None:
        top = (getattr(e, "name", None) or "").partition(".")[0]
        if (isinstance(e, ModuleNotFoundError) and top
                and top not in ("gmlx", *STOCK)):
            return top
        e = e.__cause__ or e.__context__
    return None


def every_module():
    import gmlx

    for info in pkgutil.walk_packages(gmlx.__path__, "gmlx.",
                                      onerror=lambda name: None):
        name = info.name
        if name.startswith("gmlx._vendor."):
            continue
        try:
            # A __main__ module imports without running: its body waits on
            # __name__ == "__main__".
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                importlib.import_module(name)
        except Exception as e:
            dep = missing_dependency(e)
            if dep is None:
                report["failed"][f"import {name}"] = \
                    traceback.format_exc()[-2000:]
            else:
                report["skipped"].append(f"{name}: needs {dep}")


step("import gmlx", lambda: importlib.import_module("gmlx"))
verbs()
module_step(["--version"])
module_step(["serve", "--help"])
step("load a llama GGUF", load_gguf)
step("gmlx run on the GGUF", run_gguf)
step("prepare a gmlx train LoRA student on a K-quant GGUF", train_student)
step("prepare a gmlx distill LoRA student on a K-quant GGUF", distill_student)
step("gmlx run on a Pixtral VLM GGUF pair with an image", run_vlm_gguf)
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
