#!/usr/bin/env python3
"""Run a speech oracle script in the speech oracle environment, out of
process.

  python scripts/speech_oracle/run.py --python <oracle python> \\
      [--fixtures <dir>] [--threads N] [--log <file>] <script> [args...]

The speech oracle environment holds torch, torchaudio, transformers at the
revision oracle.py names, gguf, sentencepiece, librosa, scipy and numpy,
and never gmlx or mlx. The runner is the image oracle's
(scripts/image_oracle/run.py) with this list of packages and
GMLX_SPEECH_FIXTURES as the fixture directory. ``<script>`` is a path, or
a file name under scripts/speech_oracle/.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURES_ENV = "GMLX_SPEECH_FIXTURES"
NEEDS = ("torch", "torchaudio", "transformers", "gguf", "sentencepiece",
         "librosa", "scipy")


def _image_runner():
    path = HERE.parent / "image_oracle" / "run.py"
    spec = importlib.util.spec_from_file_location("_image_oracle_run", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


runner = _image_runner()


def child_env(fixtures: Path, threads: int | None = None,
              base: dict | None = None) -> dict:
    env = runner.child_env(fixtures, threads, base)
    env.pop(runner.FIXTURES_ENV, None)
    env[FIXTURES_ENV] = str(fixtures)
    return env


def preflight(python: str, env: dict | None = None) -> str | None:
    return runner.preflight(python, env, needs=NEEDS)


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
    env = child_env(fixtures, a.threads)
    why = preflight(a.python, env)
    if why:
        print(f"error: {why}", file=sys.stderr)
        return 2
    return runner.run(a.python, resolve_script(a.script), a.args, env, a.log)


if __name__ == "__main__":
    sys.exit(main())
