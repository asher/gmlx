#!/usr/bin/env python3
"""Run the GPU test gate on this Mac and share its result through git.

  python scripts/gate.py setup       # once per clone: push and fetch the results
  python scripts/gate.py run         # test HEAD on the GPU and record the result
  python scripts/gate.py run --full  # every test on both devices, before a release
  python scripts/gate.py status      # who gated HEAD's tree, or what it inherits
  python scripts/gate.py scope       # list the tests the gate runs
  python scripts/gate.py prune       # drop results older than four weeks

`run` runs the gate set on the GPU: the numerics folders and every other
test file that imports mlx or mlx-kquant. CI runs the whole suite on the
CPU. `run --full` runs every test, nightly ones too, on the GPU with
coverage and then on the CPU. It checks the gate set against the two
reports (`scope --check`) and writes the test map that
scripts/affected_tests.py reads. Either adds the stress tier when
concurrency code changed.

A result is a JSON blob under refs/gmlx-gate-mine/<tree>/<host>. After
`setup`, a plain `git push` publishes it on origin as
refs/gmlx-gate/<tree>/<host>, and a fetch brings every result down to
refs/gmlx-gate/*. The gate workflow reads those refs and sets the gate/gpu
status of a pull request.

`decide --head SHA [--base SHA]` prints the statuses for a commit as JSON.
Results for the commit's own tree settle it, pass or fail. Without one, a
passing tree that differs only in docs passes it, then a change that is
docs-only from its merge-base, and otherwise it is pending.

Stdlib only, plus git.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import affected_tests  # noqa: E402
import changed_paths  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MINE = "refs/gmlx-gate-mine"
MIRROR = "refs/gmlx-gate"
PUSH = f"+{MINE}/*:{MIRROR}/*"
FETCH = f"+{MIRROR}/*:{MIRROR}/*"
WINDOW = timedelta(days=28)
STRESS = ("gmlx/container/", "gmlx/stream/", "gmlx/talk/", "gmlx/serve/lifecycle.py",
          "gmlx/serve/engine_jobs.py", "gmlx/spinner.py", "crates/gmlx-entry/")
DESCRIPTION_MAX = 140
WORKERS = 8
DEVICE_MODULES = ("mlx", "mlx_kquant")
_CHILD_IMPORT = re.compile(r"^\s*(?:import|from)\s+(?:mlx|mlx_kquant)\b", re.M)
# Test files whose cases skip on one device and run on the other, that the
# import scan misses. `scope --check` lists candidates.
GPU_EXTRA: tuple[str, ...] = ()


def git(*args: str, cwd: Path | str | None = None, check: bool = True, stdin: str | None = None) -> str:
    p = subprocess.run(["git", *args], cwd=cwd or ROOT, capture_output=True, text=True, input=stdin)
    if check and p.returncode != 0:
        raise SystemExit(f"git {' '.join(args)}: {p.stderr.strip()}")
    return p.stdout.strip() if p.returncode == 0 else ""


def host_slug() -> str:
    name = subprocess.run(["scutil", "--get", "LocalHostName"], capture_output=True, text=True).stdout.strip() \
        if sys.platform == "darwin" else ""
    return re.sub(r"[^a-z0-9-]+", "-", (name or platform.node()).lower()).strip("-") or "host"


def host_label() -> str:
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout
    return chip.strip().removeprefix("Apple ") or platform.machine()


def tree_of(rev: str, cwd=None) -> str | None:
    out = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{rev}^{{tree}}"],
                         cwd=cwd or ROOT, capture_output=True, text=True)
    return out.stdout.strip() or None


def _when(result: dict) -> datetime:
    try:
        return datetime.fromisoformat(result["when"])
    except (KeyError, TypeError, ValueError):
        return datetime.min.replace(tzinfo=timezone.utc)


def results(cwd=None, namespaces=(MINE, MIRROR)) -> dict[tuple[str, str], dict]:
    """Every readable result, keyed by (tree, host), newest first per key."""
    listing = git("for-each-ref", "--format=%(refname) %(objectname) %(objecttype)",
                  *[f"{n}/" for n in namespaces], cwd=cwd)
    found: dict[tuple[str, str], dict] = {}
    for line in listing.splitlines():
        ref, sha, kind = line.split()
        parts = ref.split("/")
        if kind != "blob" or len(parts) != 4:
            continue
        try:
            result = json.loads(git("cat-file", "blob", sha, cwd=cwd))
        except ValueError:
            continue
        key = (parts[2], parts[3])
        result["ref"] = ref
        if key not in found or _when(result) > _when(found[key]):
            found[key] = result
    return found


def _fresh(found: dict, now: datetime) -> dict:
    return {k: r for k, r in found.items() if now - _when(r) <= WINDOW}


def _duration(result: dict) -> str:
    return f"{round(result.get('duration_s', 0) / 60)}m"


def _decide_one(head_tree: str, own: list[dict], passing: list[tuple[str, dict]],
                docs_change: bool, cwd) -> dict:
    if own:
        good = [r for r in own if r.get("pass")]
        if good:
            return {"state": "success", "description": ", ".join(f"{r['host_slug']} {_duration(r)}" for r in good)}
        return {"state": "failure", "description": "failed on " + ", ".join(r["host_slug"] for r in own)}
    for tree, r in sorted(passing, key=lambda tr: _when(tr[1]), reverse=True):
        # -r lists files; without it a change shows only as its top-level folder.
        diff = subprocess.run(["git", "diff-tree", "-r", "--name-only", "--no-renames", tree, head_tree],
                              cwd=cwd or ROOT, capture_output=True, text=True)
        if diff.returncode != 0:
            continue
        if all(changed_paths.is_docs(p) for p in diff.stdout.split()):
            commit = r.get("commit", tree)
            return {"state": "success", "description": f"inherited from {commit[:10]}, docs-only delta",
                    "inherits": commit}
    if docs_change:
        return {"state": "success", "description": "no code change"}
    return {"state": "pending", "description": "no gate result for this tree"}


def decide(head: str, base: str = "", hosts: list[str] = (), cwd=None,
           now: datetime | None = None) -> list[dict]:
    """The gate/gpu status for ``head``, then one per host in ``hosts``."""
    now = now or datetime.now(timezone.utc)
    head_tree = tree_of(head, cwd)
    if head_tree is None:
        raise SystemExit(f"{head} is not a commit here")
    found = _fresh(results(cwd), now)
    for (tree, host), r in found.items():
        r["host_slug"] = host
    paths = changed_paths.changed(base, head, cwd=cwd) if base else None
    docs_change = paths is not None and not changed_paths.classify(paths)["code"]
    out = []
    for context, wanted in [("gate/gpu", None), *[(f"gate/gpu/{h}", h) for h in hosts]]:
        keep = {k: r for k, r in found.items() if wanted is None or k[1] == wanted}
        own = [r for (tree, _), r in keep.items() if tree == head_tree]
        passing = [(tree, r) for (tree, _), r in keep.items() if tree != head_tree and r.get("pass")]
        status = _decide_one(head_tree, own, passing, docs_change, cwd)
        status["description"] = status["description"][:DESCRIPTION_MAX]
        out.append({"context": context, **status})
    return out


def prune_local(now: datetime | None = None, namespaces=(MINE, MIRROR), cwd=None) -> list[str]:
    """Delete local results older than the window; return their refs."""
    now = now or datetime.now(timezone.utc)
    old = [r["ref"] for r in results(cwd, namespaces).values() if now - _when(r) > WINDOW]
    for ref in old:
        git("update-ref", "-d", ref, cwd=cwd)
    return old


def record(result: dict, tree: str, host: str, cwd=None) -> str:
    blob = git("hash-object", "-w", "--stdin", stdin=json.dumps(result, indent=1) + "\n", cwd=cwd)
    ref = f"{MINE}/{tree}/{host}"
    git("update-ref", ref, blob, cwd=cwd)
    return ref


def _config_values(key: str, cwd=None) -> list[str]:
    return git("config", "--get-all", key, cwd=cwd, check=False).splitlines()


def setup(a) -> int:
    wanted = [("remote.origin.push", "HEAD"), ("remote.origin.push", PUSH), ("remote.origin.fetch", FETCH)]
    for key, value in wanted:
        if value in _config_values(key):
            print(f"{key} {value}: already set")
        else:
            git("config", "--add", key, value)
            print(f"{key} {value}: added")
    print("A plain `git push` now pushes the current branch and this Mac's gate results.")
    return 0


def _summary_lines(log: Path) -> list[str]:
    lines = log.read_text(errors="replace").splitlines() if log.exists() else []
    keep = [ln for ln in lines if ln.startswith(("FAILED", "ERROR"))][:20]
    tail = [ln for ln in lines[-3:] if ln.strip()]
    return keep + tail


def imports_device(path: Path) -> bool:
    """Whether a test file imports mlx or mlx-kquant, itself or in code it
    runs in a child interpreter."""
    try:
        text = path.read_text()
        tree = ast.parse(text, filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return False
    child = "sys.executable" in text
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(a.name.split(".")[0] in DEVICE_MODULES for a in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and not node.level \
                and (node.module or "").split(".")[0] in DEVICE_MODULES:
            return True
        if child and isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and _CHILD_IMPORT.search(node.value):
            return True
    return False


def gate_set(root: Path | None = None) -> list[str]:
    """The numerics folders, then every other test file that imports mlx or
    mlx-kquant, and GPU_EXTRA. Helper modules are not followed: the root
    conftest's helpers import mlx, which would pull in every file."""
    root = root or ROOT
    folders = [f"tests/{d}" for d in changed_paths.NUMERICS if (root / "tests" / d).is_dir()]
    files = {p.relative_to(root).as_posix() for p in (root / "tests").rglob("test_*.py")
             if p.relative_to(root).parts[1] not in changed_paths.NUMERICS and imports_device(p)}
    files |= {f for f in GPU_EXTRA if (root / f).exists()}
    return folders + sorted(files)


def _skips(junit: Path) -> dict[tuple[str, str], bool]:
    return {(c.get("classname", ""), c.get("name", "")): c.find("skipped") is not None
            for c in ET.parse(junit).iter("testcase")}


def _case_file(classname: str, root: Path) -> str | None:
    parts = classname.split(".")
    for i in range(len(parts), 0, -1):
        cand = "/".join(parts[:i]) + ".py"
        if (root / cand).is_file():
            return cand
    return None


def scope_misses(gpu_junit: Path, cpu_junit: Path, root: Path | None = None) -> list[str]:
    """Test files with a case that skips on one device and runs on the
    other, outside the gate set."""
    root = root or ROOT
    gpu, cpu = _skips(gpu_junit), _skips(cpu_junit)
    differ = {_case_file(key[0], root) for key in gpu.keys() & cpu.keys() if gpu[key] != cpu[key]}
    covered = gate_set(root)
    return sorted(f for f in differ - {None}
                  if not any(f == c or f.startswith(c + "/") for c in covered))


def _state_dir() -> Path:
    return Path(git("rev-parse", "--path-format=absolute", "--git-common-dir")) / "gmlx-gate"


def _changed_since_main() -> list[str] | None:
    base = git("merge-base", "HEAD", "origin/main", check=False) or git("merge-base", "HEAD", "main", check=False)
    return changed_paths.changed(base, "HEAD") if base else None


def _tiers(paths: list[str] | None, *, full: bool, stress: bool | None) -> list[str]:
    def touches(prefixes):
        return paths is None or any(p == x or (x.endswith("/") and p.startswith(x)) for p in paths for x in prefixes)
    tiers = ["gpu-full", "cpu-full", "scope"] if full else ["gpu"]
    if stress if stress is not None else touches(STRESS):
        tiers.append("stress")
    return tiers


def _tier_command(tier: str, workers: int, junit: Path) -> tuple[list[str], dict]:
    env = {k: v for k, v in os.environ.items() if k != "KQUANT_TEST_GGUF_DIR"}
    py = sys.executable
    if tier == "stress":
        return [py, str(ROOT / "scripts" / "stress_tests.py"), "--preset", "concurrency",
                "--rounds", "1", "--count", "3", "--out", str(_state_dir() / "stress")], env
    cmd = [py, "-m", "pytest", "-q", "-rfE", "-n", str(workers), "--dist", "worksteal", f"--junitxml={junit}"]
    if tier == "gpu":
        return [*cmd, "-m", "not nightly", *gate_set()], env
    if tier == "gpu-full":
        env["COVERAGE_FILE"] = str(_state_dir() / "coverage" / ".coverage")
        return [*cmd, "--cov=gmlx", "--cov-context=test", "--cov-report="], env
    env["KQUANT_FORCE_CPU"] = "1"
    return cmd, env


def _check_scope(log: Path) -> int:
    reports = [log.with_name(log.name.replace("-scope.log", f"-{t}.xml")) for t in ("gpu-full", "cpu-full")]
    if not all(r.exists() for r in reports):
        log.write_text("no junit report from both full tiers\nexit 1\n")
        return 1
    misses = scope_misses(*reports)
    lines = [f"MISSED {f}: skips on one device only; add it to GPU_EXTRA in scripts/gate.py" for f in misses]
    log.write_text("\n".join([*lines, f"exit {1 if misses else 0}"]) + "\n")
    return 1 if misses else 0


def run_tier(tier: str, workers: int, log: Path) -> tuple[int, float]:
    if tier == "scope":
        return _check_scope(log), 0.0
    cmd, env = _tier_command(tier, workers, log.with_suffix(".xml"))
    if tier == "gpu-full":
        cov = Path(env["COVERAGE_FILE"])
        cov.parent.mkdir(parents=True, exist_ok=True)
        cov.unlink(missing_ok=True)
    start = time.monotonic()
    with open(log, "w") as f:
        f.write("+ " + " ".join(cmd) + "\n")
        f.flush()
        code = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    with open(log, "a") as f:
        f.write(f"exit {code}\n")
    return code, time.monotonic() - start


def run(a) -> int:
    if git("status", "--porcelain"):
        raise SystemExit("the tree has uncommitted changes; commit or stash them first")
    head = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    logs = _state_dir() / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    tiers = _tiers(_changed_since_main(), full=a.full, stress=a.stress)
    print(f"gating {head[:10]} (tree {tree[:10]}): {', '.join(tiers)}", flush=True)
    done, total, passed = [], 0.0, True
    for tier in tiers:
        log = logs / f"{tree[:12]}-{tier}.log"
        print(f"{tier}: log {log}", flush=True)
        code, took = run_tier(tier, a.workers, log)
        total += took
        passed = passed and code == 0
        done.append({"tier": tier, "exit": code, "duration_s": round(took), "log": str(log),
                     "summary": _summary_lines(log)})
        print(f"{tier}: exit {code} in {took / 60:.1f} min", flush=True)
    cov = _state_dir() / "coverage" / ".coverage"
    if "gpu-full" in tiers and cov.exists():
        print(f"test map: {affected_tests.write_map(cov, head, ROOT)}", flush=True)
    prune_local(namespaces=(MINE,))
    result = {"pass": passed, "commit": head, "host": host_label(),
              "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "duration_s": round(total), "tiers": [t["tier"] for t in done],
              "log": [line for t in done for line in [f"{t['tier']}: exit {t['exit']}", *t["summary"]]]}
    ref = record(result, tree, host_slug())
    print(f"{'passed' if passed else 'failed'}; recorded {ref}")
    print("Push it with a plain `git push` (after `scripts/gate.py setup`).")
    return 0 if passed else 1


def status(a) -> int:
    if not a.offline:
        subprocess.run(["git", "fetch", "--prune", "--quiet", "origin", FETCH], cwd=ROOT)
    head = git("rev-parse", a.sha)
    tree = git("rev-parse", f"{head}^{{tree}}")
    mirror = results(namespaces=(MIRROR,))
    for (t, host), r in sorted(results().items()):
        if t != tree:
            continue
        where = "on origin" if (tree, host) in mirror else "only on this Mac"
        print(f"{host}: {'passed' if r.get('pass') else 'failed'} {r.get('when')} "
              f"in {_duration(r)}, {', '.join(r.get('tiers', []))}; {where}")
    base = git("merge-base", head, "origin/main", check=False)
    gate = decide(head, base)[0]
    if "inherits" in gate:
        print(f"inherits from {gate['inherits'][:10]}: a docs-only delta over a gated tree needs no run")
    print(f"gate/gpu would be {gate['state']}: {gate['description']}")
    return 0


def prune(a) -> int:
    subprocess.run(["git", "fetch", "--prune", "--quiet", "origin", FETCH], cwd=ROOT)
    old_mirror = [r["ref"] for r in results(namespaces=(MIRROR,)).values()
                  if datetime.now(timezone.utc) - _when(r) > WINDOW]
    removed = prune_local()
    if old_mirror and not a.local:
        git("push", "origin", *[f":{ref}" for ref in old_mirror])
    print(f"removed {len(removed)} local result(s)"
          + ("" if a.local else f" and {len(old_mirror)} on origin"))
    return 0


def scope(a) -> int:
    if not a.check:
        picked = gate_set()
        print("\n".join(picked))
        print(f"{len(picked)} folders and files", file=sys.stderr)
        return 0
    if not (a.gpu_junit and a.cpu_junit):
        raise SystemExit("scope --check needs --gpu-junit and --cpu-junit")
    misses = scope_misses(Path(a.gpu_junit), Path(a.cpu_junit))
    for f in misses:
        print(f"{f}: skips on one device only; add it to GPU_EXTRA in scripts/gate.py")
    return 1 if misses else 0


def decide_cmd(a) -> int:
    print(json.dumps(decide(a.head, a.base, a.hosts.split()), indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup")
    r = sub.add_parser("run")
    r.add_argument("--workers", type=int, default=WORKERS, help=f"pytest-xdist workers (default {WORKERS})")
    r.add_argument("--full", action="store_true",
                   help="every test on the GPU with coverage, then on the CPU, and the scope check")
    r.add_argument("--stress", action=argparse.BooleanOptionalAction, default=None,
                   help="force the stress tier on or off (default: when concurrency paths change)")
    s = sub.add_parser("status")
    s.add_argument("sha", nargs="?", default="HEAD")
    s.add_argument("--offline", action="store_true", help="skip the fetch from origin")
    c = sub.add_parser("scope")
    c.add_argument("--check", action="store_true", help="list files with a device skip difference outside the set")
    c.add_argument("--gpu-junit", default="")
    c.add_argument("--cpu-junit", default="")
    p = sub.add_parser("prune")
    p.add_argument("--local", action="store_true", help="leave origin alone")
    d = sub.add_parser("decide")
    d.add_argument("--head", required=True)
    d.add_argument("--base", default="")
    d.add_argument("--hosts", default="", help="space-separated host slugs to report one by one")
    a = ap.parse_args(argv)
    return {"setup": setup, "run": run, "status": status, "scope": scope, "prune": prune,
            "decide": decide_cmd}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
