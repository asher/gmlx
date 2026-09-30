# vendor_upstream: skip
"""The stock mlx-vlm and mlx-lm results the owned-cache tests compare with.

gmlx does not import the stock packages, so each test that checks an owned
class against its stock original reads the stock side from golden/. A
stock run is the test's own child script with its module names reversed
from gmlx._vendor.mlx_vlm and gmlx._vendor.mlx_lm to mlx_vlm and mlx_lm
(``scripts/vendor_upstream.py reverse``), run on the CPU device in a child
with the repository off sys.path. Each golden stores the sha256 of the
script it came from; ``load_golden`` fails when the script has changed
since, or when the copy's versions in gmlx/_vendor/manifest.json differ
from the recorded ones.

To record, run this file with interpreters that have the stock packages
at the copied versions, for example scratch venvs made with
``uv pip install mlx==<pin> mlx-vlm==0.7.3 mlx-lm==0.31.3``:

    python tests/cache/stock_golden.py --python PY [--python PY ...]

Every result must agree across the interpreters, except the Python call
counts of test_owned_generate_loop, which are recorded per Python version.
``--check`` compares with the committed goldens instead of writing them.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GOLDEN = HERE / "golden"
MANIFEST = REPO / "gmlx" / "_vendor" / "manifest.json"
ARCHS = ("llama", "gemma3", "qwen3_next")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _copied_versions() -> dict:
    packages = json.loads(MANIFEST.read_text())["packages"]
    return {name: p["version"] for name, p in packages.items()}


def load_golden(name: str, script: str) -> dict:
    """The golden ``name``, checked against ``script`` and the copy."""
    golden = json.loads((GOLDEN / f"{name}.json").read_text())
    assert golden["script_sha256"] == _sha(script), (
        f"golden/{name}.json was recorded from another version of its "
        f"script: record it again with tests/cache/stock_golden.py")
    assert golden["recorded"]["packages"] == _copied_versions(), (
        golden["recorded"]["packages"])
    return golden


# Recording.

def _vendor_tool():
    path = REPO / "scripts" / "vendor_upstream.py"
    spec = importlib.util.spec_from_file_location("_vendor_upstream", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def script_constant(test_file: str, name: str) -> str:
    """The string a test module assigns to ``name`` at top level."""
    tree = ast.parse((HERE / test_file).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise LookupError(f"{test_file} assigns no {name}")


_PROBE = r'''
import importlib.metadata as md
import json
import sys

import mlx.core as mx
import mlx_lm
import mlx_vlm

print(json.dumps({"python": "%d.%d" % sys.version_info[:2],
                  "mlx": mx.__version__,
                  "packages": {"mlx_vlm": md.version("mlx-vlm"),
                               "mlx_lm": md.version("mlx-lm")},
                  "files": [mlx_vlm.__file__, mlx_lm.__file__]}))
'''


class _Stock:
    """Runs reversed scripts under one interpreter, off the repository."""

    def __init__(self, python: str, tmp: Path):
        self.python, self.tmp = python, tmp
        self.vu = _vendor_tool()
        env = dict(os.environ)
        keep = [p for p in env.get("PYTHONPATH", "").split(os.pathsep)
                if p and Path(p).resolve() != REPO]
        env["PYTHONPATH"] = os.pathsep.join(keep)
        self.env = env
        self.info = self._run(["-c", _PROBE])
        for f in self.info.pop("files"):
            if Path(f).resolve().is_relative_to(REPO):
                raise SystemExit(f"{python}: {f} is in the repository")

    def _run(self, argv: list[str]) -> dict:
        r = subprocess.run([self.python, *argv], capture_output=True,
                           text=True, env=self.env, cwd=self.tmp, timeout=900)
        if r.returncode:
            raise SystemExit(f"{self.python} {argv[-3:]}: {r.stderr[-3000:]}")
        return json.loads(r.stdout.strip().splitlines()[-1])

    def script(self, text: str, *args: str) -> dict:
        return self._run(["-c", self.vu.reverse_text(text), *args])

    def file(self, text: str, *args: str) -> dict:
        path = self.tmp / "stock_driver.py"
        path.write_text(self.vu.reverse_text(text))
        return self._run([str(path), *args])


def _equivalence(runner: _Stock) -> dict:
    text = (HERE / "cache_equivalence_ops.py").read_text()
    out = {}
    for mode, root, field in (("vlm", "mlx_vlm", "full"),
                              ("lm", "mlx_lm", "reduced")):
        run = runner.file(text, mode)
        assert run["loaded"] == [root], run["loaded"]
        out[mode] = {"loaded": run["loaded"], "cache_module": run["cache_module"],
                     "rows": {r["key"]: json.loads(r[field]) for r in run["rows"]}}
    return {"script_sha256": _sha(text), **out}


def _generate_loop(runner: _Stock) -> dict:
    text = script_constant("test_owned_generate_loop.py", "_SCRIPT")
    runs, calls = {}, {}
    for arch in ARCHS:
        run = runner.script(text, "stock", arch, "cpu")
        assert not run["gmlx"] and run["device"] == "cpu", run
        assert all(c.startswith("mlx_lm.models.cache.") for c in run["classes"])
        counts = run.pop("calls")
        if counts is not None:
            calls[arch] = {runner.info["python"]: counts}
        runs[arch] = run
    return {"script_sha256": _sha(text), "runs": runs, "calls": calls}


def _lm_defects(runner: _Stock) -> dict:
    text = script_constant("test_owned_cache_lm_defects.py", "_SCENARIO")
    run = runner.script(text, "stock")
    assert run["module"] == "mlx_lm.models.cache" and not run["gmlx"], run
    return {"script_sha256": _sha(text), "run": run}


def _quantized_sdpa(runner: _Stock) -> dict:
    text = script_constant("test_owned_quantized_sdpa.py", "_STOCK_LM_CHILD")
    return {"script_sha256": _sha(text), "run": runner.script(text)}


RECORDERS = {"cache_equivalence": _equivalence, "generate_loop": _generate_loop,
             "lm_defects": _lm_defects, "quantized_sdpa": _quantized_sdpa}


def record(pythons: list[str]) -> dict:
    goldens, infos = {}, []
    with tempfile.TemporaryDirectory() as tmp:
        runners = [_Stock(py, Path(tmp)) for py in pythons]
        for runner in runners:
            info = runner.info
            if info["packages"] != _copied_versions():
                raise SystemExit(f"{runner.python}: stock {info['packages']}, "
                                 f"the copy holds {_copied_versions()}")
            infos.append(info)
        if len({i["mlx"] for i in infos}) != 1:
            raise SystemExit(f"the interpreters differ in mlx: {infos}")
        recorded = {"packages": infos[0]["packages"], "mlx": infos[0]["mlx"],
                    "python": sorted(i["python"] for i in infos),
                    "device": "cpu"}
        for name, fn in RECORDERS.items():
            first, per_version = None, {}
            for runner in runners:
                got = fn(runner)
                # Profiler call counts depend on the Python version.
                for arch, counts in got.pop("calls", {}).items():
                    per_version.setdefault(arch, {}).update(counts)
                if first is None:
                    first = got
                elif got != first:
                    raise SystemExit(f"{name}: {runner.python} differs from "
                                     f"{runners[0].python}")
            if per_version:
                first["calls"] = per_version
            goldens[name] = {"recorded": recorded, **first}
    return goldens


def render(obj) -> str:
    """JSON with one line per row of a ``rows`` mapping."""
    def enc(o, pad, rows):
        if not isinstance(o, dict) or not o:
            return json.dumps(o, sort_keys=True)
        inner = pad + " "
        if rows:
            items = [f"{inner}{json.dumps(k)}: {json.dumps(v, sort_keys=True)}"
                     for k, v in o.items()]
        else:
            items = [f"{inner}{json.dumps(k)}: {enc(v, inner, k == 'rows')}"
                     for k, v in o.items()]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    return enc(obj, "", False) + "\n"


def _comparable(golden: dict, versions: list[str]) -> dict:
    """``golden`` restricted to the Python versions of a check run."""
    golden = json.loads(json.dumps(golden))
    golden["recorded"]["python"] = None
    for per_version in golden.get("calls", {}).values():
        for version in set(per_version) - set(versions):
            del per_version[version]
    return golden


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--python", action="append",
                    help="an interpreter with the stock packages (repeatable; "
                         "default: this one)")
    ap.add_argument("--check", action="store_true",
                    help="compare with the committed goldens, write nothing")
    args = ap.parse_args(argv)
    goldens = record(args.python or [sys.executable])
    stale = []
    for name, golden in goldens.items():
        path = GOLDEN / f"{name}.json"
        text = render(golden)
        if args.check:
            versions = golden["recorded"]["python"]
            if not path.exists() or (
                    _comparable(json.loads(path.read_text()), versions)
                    != _comparable(golden, versions)):
                stale.append(path.name)
            continue
        GOLDEN.mkdir(exist_ok=True)
        path.write_text(text)
        print(f"wrote {path.relative_to(REPO)} ({len(text)} bytes)")
    if stale:
        print("differs from the recording: " + ", ".join(stale))
        return 1
    if args.check:
        print(f"{len(goldens)} goldens match the recording")
    return 0


if __name__ == "__main__":
    sys.exit(main())
