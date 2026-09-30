#!/usr/bin/env python3
"""Run an image oracle script in the oracle environment, out of process.

  python scripts/image_oracle/run.py --python <oracle python> \\
      [--fixtures <dir>] [--threads N] [--log <file>] <script> [args...]

The oracle environment holds torch, diffusers at the pinned revision,
transformers, gguf, safetensors and numpy, and never gmlx or mlx. The
runner refuses an interpreter that can import gmlx or mlx, that cannot
import torch or diffusers, or that is the interpreter running the runner.

The child runs with HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1, without
PYTHONPATH or the user site directory, and with GMLX_IMAGE_FIXTURES set to
the fixture directory. An oracle run therefore reads only local files and
never imports the gmlx tree. ``<script>`` is a path, or a file name under
scripts/image_oracle/.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURES_ENV = "GMLX_IMAGE_FIXTURES"

# Names the child never inherits: they would put another tree on its path.
_DROP = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV")
_OFFLINE = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DIFFUSERS_GGUF_CUDA_KERNELS": "false",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}
_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS")

_PROBE = r"""
import importlib.util as u, json, sys
print(json.dumps({"prefix": sys.prefix,
                  **{n: u.find_spec(n) is not None
                     for n in ("torch", "diffusers", "gguf", "gmlx", "mlx")}}))
"""


def child_env(fixtures: Path, threads: int | None = None,
              base: dict | None = None) -> dict:
    env = {k: v for k, v in (os.environ if base is None else base).items()
           if k not in _DROP}
    env.update(_OFFLINE)
    env[FIXTURES_ENV] = str(fixtures)
    if threads:
        for k in _THREAD_VARS:
            env[k] = str(threads)
    return env


def preflight(python: str, env: dict | None = None) -> str | None:
    """None when ``python`` is a usable oracle interpreter, else why not."""
    exe = Path(python).expanduser()
    if not exe.exists():
        return f"{python} does not exist"
    try:
        out = subprocess.run([str(exe), "-I", "-c", _PROBE], env=env,
                             capture_output=True, text=True, timeout=120,
                             check=True).stdout
        found = json.loads(out.strip().splitlines()[-1])
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        return f"{python} failed the environment probe: {e}"
    if Path(found["prefix"]).resolve() == Path(sys.prefix).resolve():
        return (f"{python} is this interpreter's environment; the oracle "
                "runs in its own environment")
    leaked = [n for n in ("gmlx", "mlx") if found[n]]
    if leaked:
        return (f"{python} can import {' and '.join(leaked)}; the oracle "
                "environment must not hold the code it checks")
    missing = [n for n in ("torch", "diffusers", "gguf") if not found[n]]
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


def run(python: str, script: Path, args: list[str], env: dict,
        log: Path | None = None) -> int:
    cmd = [str(Path(python).expanduser()), str(script), *args]
    print("[oracle] " + " ".join(cmd), flush=True)
    sink = open(log, "a") if log else None
    try:
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if sink:
                sink.write(line)
                sink.flush()
        return proc.wait()
    finally:
        if sink:
            sink.close()


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
    env = child_env(fixtures, a.threads)
    why = preflight(a.python, env)
    if why:
        print(f"error: {why}", file=sys.stderr)
        return 2
    return run(a.python, resolve_script(a.script), a.args, env, a.log)


if __name__ == "__main__":
    sys.exit(main())
