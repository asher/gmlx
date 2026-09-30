#!/usr/bin/env bash
# Upstream drift: what a new mlx-vlm and mlx-lm release changes in the copy
# under gmlx/_vendor.
#
# Downloads the new release and the copied release of each package as
# wheels, with no install and no dependencies, and runs
# scripts/drift_report.py on them. The copied releases come from
# gmlx/_vendor/manifest.json. The report informs a port and the script never
# fails on drift: porting a change is a decision, taken per
# docs/internals/upstream-upgrades.md. It exits nonzero only when a download
# or the report itself fails.
#
# Usage: scripts/upstream_canary.sh [MLX_VLM_VERSION [MLX_LM_VERSION]]
# With no version, the newest release on PyPI. Flags for drift_report.py go
# in DRIFT_ARGS, for example DRIFT_ARGS="--json drift.json".
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
py="${PYTHON:-python3}"
work="$(mktemp -d "${TMPDIR:-/tmp}/gmlx-drift.XXXXXX")"
trap 'rm -rf "$work"' EXIT

copied() {
    "$py" -c 'import json, sys
print(json.load(open(sys.argv[1]))["packages"][sys.argv[2]]["version"])' \
        "$repo/gmlx/_vendor/manifest.json" "$1"
}

# fetch DIST [VERSION] DIR: the wheel of DIST, unpacked into DIR/tree.
fetch() {
    local spec="$1"
    if [ -n "$2" ]; then spec="$1==$2"; fi
    mkdir -p "$3"
    "$py" -m pip download -q --disable-pip-version-check --no-deps \
        --only-binary=:all: -d "$3" "$spec"
    "$py" -m zipfile -e "$3"/*.whl "$3/tree"
}

args=()
new_versions=("${1:-}" "${2:-}")
i=0
for package in mlx_vlm mlx_lm; do
    dist="${package//_/-}"
    fetch "$dist" "${new_versions[$i]}" "$work/new-$package"
    fetch "$dist" "$(copied "$package")" "$work/old-$package"
    args+=(--new "$package=$work/new-$package/tree/$package"
           --old "$package=$work/old-$package/tree/$package")
    i=$((i + 1))
done

# shellcheck disable=SC2086
"$py" "$repo/scripts/drift_report.py" "${args[@]}" ${DRIFT_ARGS:-}
