#!/usr/bin/env python3
"""Run a video oracle script in its oracle environment, out of process.

  python scripts/video_oracle/run.py --python <oracle python> \\
      [--fixtures <dir>] [--threads N] [--log <file>] <script> [args...]

This is scripts/image_oracle/run.py with two changes. The child gets
GMLX_VIDEO_FIXTURES. Each oracle script names the modules it needs in a
module-level ``ORACLE_NEEDS`` tuple, since the video oracles run in two
environments: ComfyUI's model code in one with torch and gguf, the
diffusers scheduler and the transformers encoder in one with diffusers.
The runner refuses an interpreter that cannot import those, that can
import gmlx or mlx, or that is the interpreter running the runner.

The child runs offline, without PYTHONPATH or the user site directory.
``<script>`` is a path, or a file name under scripts/video_oracle/.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURES_ENV = "GMLX_VIDEO_FIXTURES"
NEEDS = "ORACLE_NEEDS"
_FORBIDDEN = ("gmlx", "mlx")

_PROBE = r"""
import importlib.util as u, json, sys
names = json.loads(sys.argv[1])
print(json.dumps({"prefix": sys.prefix,
                  **{n: u.find_spec(n) is not None for n in names}}))
"""


def _image_runner():
    path = HERE.parent / "image_oracle" / "run.py"
    spec = importlib.util.spec_from_file_location("_video_oracle_run", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"error: cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.FIXTURES_ENV = FIXTURES_ENV
    return mod


_img = _image_runner()
child_env = _img.child_env
run = _img.run


def needs(script: Path) -> tuple[str, ...]:
    """The script's ``ORACLE_NEEDS``: a tuple of module names, read from
    its source without running it."""
    tree = ast.parse(script.read_text(), str(script))
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == NEEDS
                        for t in node.targets)):
            value = ast.literal_eval(node.value)
            if (isinstance(value, tuple) and value
                    and all(isinstance(v, str) for v in value)):
                return value
            break
    raise SystemExit(f"error: {script.name} declares no {NEEDS} tuple of "
                     "module names")


def preflight(python: str, needed: tuple[str, ...],
              env: dict | None = None) -> str | None:
    """None when ``python`` can run a script that needs ``needed``, else
    why not."""
    exe = Path(python).expanduser()
    if not exe.exists():
        return f"{python} does not exist"
    names = sorted(set(needed) | set(_FORBIDDEN))
    try:
        out = subprocess.run([str(exe), "-I", "-c", _PROBE,
                              json.dumps(names)], env=env,
                             capture_output=True, text=True, timeout=120,
                             check=True).stdout
        found = json.loads(out.strip().splitlines()[-1])
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        return f"{python} failed the environment probe: {e}"
    if Path(found["prefix"]).resolve() == Path(sys.prefix).resolve():
        return (f"{python} is this interpreter's environment; the oracle "
                "runs in its own environment")
    leaked = [n for n in _FORBIDDEN if found[n]]
    if leaked:
        return (f"{python} can import {' and '.join(leaked)}; the oracle "
                "environment must not hold the code it checks")
    missing = [n for n in needed if not found[n]]
    if missing:
        return f"{python} cannot import {', '.join(missing)}"
    return None


def resolve_script(script: str) -> Path:
    p = Path(script)
    if not p.is_file() and not p.is_absolute():
        p = HERE / script
    if not p.is_file():
        raise SystemExit(f"error: no oracle script {script}")
    return p.resolve()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="Arguments after the script name go to the script.")
    ap.add_argument("--python", required=True,
                    help="interpreter of the oracle environment")
    ap.add_argument("--fixtures", default=os.environ.get(FIXTURES_ENV),
                    help=f"fixture directory (default ${FIXTURES_ENV})")
    ap.add_argument("--threads", type=int,
                    help="CPU threads for the oracle's math libraries")
    ap.add_argument("--log", type=Path, help="append the child's output here")
    ap.add_argument("script", help="oracle script path or file name")
    ap.add_argument("args", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    if not a.fixtures:
        ap.error(f"set --fixtures or {FIXTURES_ENV}")
    fixtures = Path(a.fixtures).expanduser().resolve()
    fixtures.mkdir(parents=True, exist_ok=True)
    script = resolve_script(a.script)
    env = child_env(fixtures, a.threads)
    why = preflight(a.python, needs(script), env)
    if why:
        print(f"error: {why}", file=sys.stderr)
        return 2
    return run(a.python, script, a.args, env, a.log)


if __name__ == "__main__":
    sys.exit(main())
