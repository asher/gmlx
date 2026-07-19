#!/usr/bin/env bash
# Upstream canary: does the LATEST mlx-vlm still satisfy our patched seams?
#
# Builds a disposable venv (never touches the dev venv), installs this
# checkout plus the newest mlx-vlm, and runs the seam contract check
# (gmlx.upstream_seams) and an import smoke. Nonzero exit = drift; the
# report names each moved or rewritten symbol. Run this when upstream cuts a
# release, before users hit it - see docs/upstream-upgrades.md.
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d "${TMPDIR:-/tmp}/gmlx-canary.XXXXXX")"
trap 'rm -rf "$work"' EXIT

python3 -m venv "$work/venv"
py="$work/venv/bin/python"
"$py" -m pip -q install -U pip
"$py" -m pip -q install -e "$repo"
# Lift the exact pin to whatever PyPI serves today.
"$py" -m pip -q install -U mlx-vlm

echo "== versions =="
"$py" -c 'import importlib.metadata as md
for p in ("mlx", "mlx-lm", "mlx-vlm"):
    print(f"  {p} {md.version(p)}")'

echo "== seam check =="
"$py" -m gmlx.upstream_seams

echo "== import smoke =="
"$py" -c "import gmlx.loader, gmlx.modules, gmlx.spec_engine, \
gmlx.server_patches, gmlx.apc_pooling; print('  imports OK')"

echo "canary PASS: latest mlx-vlm matches the pinned seam contract"
