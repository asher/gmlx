#!/usr/bin/env python3
# Merge serve-bench JSONs into unified per-model results. Later files (by
# mtime) override earlier cells (key = model|runtime|arm|depth|conc), so a
# partial rerun (e.g. --only gmlx, or a single depth) folds onto an existing
# ladder without disturbing untouched cells. Raw samples are preserved; the
# full-length decode filter stays in the analysis layer (plot-bench.py).
#
# Usage: merge.py <root_dir> <out_dir>
#   globs <root_dir>/**/serve-bench-*.json (excluding <out_dir>), sorts by
#   mtime, writes <out_dir>/<model>.json (one per model) + all-models.json.
import glob
import json
import os
import sys

root, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
# Quarantine markers: any path component starting with one of these holds bad
# or out-of-scope data (tainted build, dropped curve) and must NOT fold in.
_QUARANTINE = ("_tainted", "_out-of-scope")
def _quarantined(p):
    return any(seg.startswith(_QUARANTINE) for seg in p.split(os.sep))
paths = [p for p in glob.glob(os.path.join(root, "**", "serve-bench-*.json"), recursive=True)
         if os.path.abspath(out) not in os.path.abspath(p) and not _quarantined(p)]
paths.sort(key=lambda p: os.path.getmtime(p))   # oldest first -> newest wins

# per model: shell (meta/config/models entry) from first sighting; cells last-wins
shell, results, prov = {}, {}, {}
for p in paths:
    d = json.load(open(p))
    for m in d.get("models", []):
        nm = m["name"]
        shell.setdefault(nm, {"meta": d.get("meta", {}), "config": d.get("config", {}),
                              "model": m})
        results.setdefault(nm, {})
        prov.setdefault(nm, {})
    for key, cell in d.get("results", {}).items():
        nm = key.split("|")[0]
        results.setdefault(nm, {})[key] = cell
        prov.setdefault(nm, {})[key] = os.path.relpath(p, root)

combined = {"models": [], "results": {}, "provenance": {}}
for nm in sorted(shell):
    sh = shell[nm]
    doc = {"meta": sh["meta"], "config": sh["config"], "models": [sh["model"]],
           "results": results[nm], "provenance": prov[nm]}
    json.dump(doc, open(os.path.join(out, nm + ".json"), "w"), indent=1)
    combined["models"].append(sh["model"])
    combined["results"].update(results[nm])
    combined["provenance"].update(prov[nm])
json.dump(combined, open(os.path.join(out, "all-models.json"), "w"), indent=1)
print(f"merged {len(paths)} inputs -> {len(shell)} models into {out}")
for nm in sorted(shell):
    ncell = len(results[nm])
    print(f"  {nm:30} {ncell:3} cells")
