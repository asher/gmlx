#!/usr/bin/env python3
"""Run the pack's suites against one ComfyUI checkout and write the results.

Usage: run_tests.py --comfy DIR --label NAME --out DIR [--pack DIR] [--mps] [--no-headless]

The unit suites (wire, client, comfy) run in one process on the cpu device,
then the headless ComfyUI run. --mps runs only the MPS suite, in place of
both. Writes NAME.json and NAME.md in --out, with every test's outcome and
every skip's reason. --pack tests the pack in another folder, such as the
custom_nodes link that `gmlx comfyui install` made; the default is the
source in the gmlx tree.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import unittest

TESTS = os.path.dirname(os.path.abspath(__file__))
UNIT = ["test_wire", "test_client", "test_comfy"]


class Recorder(unittest.TextTestResult):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.records = []

    def _rec(self, test, status, detail=""):
        self.records.append({"id": test.id(), "status": status, "detail": detail})

    def addSuccess(self, test):
        super().addSuccess(test)
        self._rec(test, "pass")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._rec(test, "fail", self._exc_info_to_string(err, test))

    def addError(self, test, err):
        super().addError(test, err)
        self._rec(test, "error", self._exc_info_to_string(err, test))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._rec(test, "skip", reason)

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            failed = issubclass(err[0], test.failureException)
            self._rec(subtest, "fail" if failed else "error", self._exc_info_to_string(err, subtest))


def run_unit(modules: list, path: str) -> int:
    sys.path.insert(0, TESTS)
    suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
    runner = unittest.TextTestRunner(resultclass=Recorder, verbosity=2, stream=sys.stdout)
    t0 = time.monotonic()
    res = runner.run(suite)
    with open(path, "w") as f:
        json.dump({"modules": modules, "wall_s": round(time.monotonic() - t0, 2),
                   "tests_run": res.testsRun, "records": res.records}, f, indent=1)
    return 0 if res.wasSuccessful() else 1


def sub(args: list, env: dict, log: str) -> int:
    with open(log, "w") as f:
        return subprocess.run([sys.executable, *args], cwd=TESTS, env=env, stdout=f,
                              stderr=subprocess.STDOUT).returncode


def summarize(rec: dict) -> dict:
    counts: dict = {}
    for r in rec["records"]:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return counts


def tail(path: str, n: int = 4000) -> str:
    try:
        with open(path, errors="replace") as f:
            return f.read()[-n:]
    except OSError:
        return ""


def show_failures(name: str, part: dict, log: str) -> None:
    """Print each failure with its traceback to the step log, or the end of
    the suite's own log when it ended before writing its results."""
    bad = [r for r in part.get("records", []) if r["status"] in ("fail", "error")]
    for r in bad:
        print(f"--- {name} {r['status']}: {r['id']}\n{r['detail'].rstrip()}", flush=True)
    if part.get("rc") and not bad:
        print(f"--- {name} exited {part['rc']} with no failure recorded; end of {log}:\n"
              f"{tail(log)}", flush=True)


def markdown(doc: dict) -> str:
    lines = [f"# gmlx node pack tests: {doc['label']}", "",
             f"- ComfyUI: {doc['comfy']} ({doc['comfyui_version']}, {doc['git']})",
             f"- pack: {doc.get('pack')}",
             f"- Python: {doc['python']}", f"- torch: {doc['torch']}", ""]
    for name, part in doc["suites"].items():
        c = part.get("counts", {})
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"- tests run: {part.get('tests_run')}, pass {c.get('pass', 0)}, fail {c.get('fail', 0)}, "
                     f"error {c.get('error', 0)}, skip {c.get('skip', 0)}; exit {part.get('rc')}")
        for r in part.get("records", []):
            if r["status"] == "skip":
                lines.append(f"- skip {r['id']}: {r['detail']}")
        for r in part.get("records", []):
            if r["status"] in ("fail", "error"):
                lines.append(f"- {r['status']} {r['id']}")
                lines += ["", "```", r["detail"].rstrip()[-1500:], "```", ""]
        lines.append("")
    h = doc.get("headless")
    if h:
        lines += ["## headless", "", f"- startup {h.get('startup_s')} s; cases {len(h.get('cases', []))}, "
                  f"passed {h.get('passed')}", ""]
        for c in h.get("cases", []):
            extra = "".join(f", {k} {c[k]}" for k in ("latent_sha256", "x_shape", "step_interrupted", "wall_s")
                            if k in c)
            lines.append(f"- {'ok' if c.get('ok') else 'FAIL'}: {c['case']}{extra}")
            for k in c["checks"]:
                if not k["ok"]:
                    lines.append(f"  - {k['check']}: {str(k['detail'])[:400]}")
        if h.get("fatal"):
            lines += ["", "```", h["fatal"][-2000:], "```"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comfy")
    ap.add_argument("--label")
    ap.add_argument("--out")
    ap.add_argument("--pack")
    ap.add_argument("--mps", action="store_true")
    ap.add_argument("--no-headless", action="store_true")
    ap.add_argument("--unit", nargs="+", help=argparse.SUPPRESS)
    ap.add_argument("--json", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.unit:
        sys.exit(run_unit(a.unit, a.json))
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    comfy = os.path.abspath(a.comfy)
    env = dict(os.environ, COMFYUI_DIR=comfy, PYTHONUNBUFFERED="1")
    env.pop("GMLX_TEST_MPS", None)
    env.pop("GMLX_API_KEY", None)
    if a.pack:
        env["GMLX_PACK_DIR"] = os.path.abspath(a.pack)
    git = subprocess.run(["git", "-C", comfy, "describe", "--tags", "--always"], capture_output=True,
                         text=True).stdout.strip()
    ver = subprocess.run([sys.executable, "-c", "import comfyui_version as v; print(v.__version__)"],
                         cwd=comfy, capture_output=True, text=True).stdout.strip()
    import torch
    doc = {"label": a.label, "comfy": comfy, "git": git, "comfyui_version": ver,
           "pack": env.get("GMLX_PACK_DIR") or "source",
           "python": sys.version.split()[0], "torch": torch.__version__, "suites": {}}
    suites = [] if a.mps else [("unit", UNIT, {})]
    if a.mps:
        suites.append(("mps", ["test_mps"], {"GMLX_TEST_MPS": "1"}))
    for name, mods, extra in suites:
        path = os.path.join(out, f"{a.label}-{name}.json")
        log = os.path.join(out, f"{a.label}-{name}.log")
        rc = sub([os.path.join(TESTS, "run_tests.py"), "--unit", *mods, "--json", path],
                 {**env, **extra}, log)
        part = {"rc": rc}
        if os.path.exists(path):
            with open(path) as f:
                part.update(json.load(f))
            part["counts"] = summarize(part)
        doc["suites"][name] = part
        print(f"{name}: rc {rc} {part.get('counts')}", flush=True)
        show_failures(name, part, log)
    if not a.no_headless and not a.mps:
        hlog = os.path.join(out, f"headless-{a.label}.log")
        rc = sub([os.path.join(TESTS, "headless.py"), "--comfy", comfy, "--out", out, "--label", a.label],
                 env, hlog)
        hp = os.path.join(out, f"headless-{a.label}.json")
        if os.path.exists(hp):
            with open(hp) as f:
                doc["headless"] = json.load(f)
        doc["headless_rc"] = rc
        h = doc.get("headless", {})
        print(f"headless: rc {rc} passed {h.get('passed')} failed {h.get('failed')}", flush=True)
        for c in h.get("cases", []):
            for k in c.get("checks", []):
                if not k["ok"]:
                    print(f"--- headless FAIL {c['case']}: {k['check']}: {k['detail']}", flush=True)
        if h.get("fatal"):
            print(f"--- headless fatal:\n{h['fatal']}", flush=True)
        if rc and not h:
            print(f"--- headless exited {rc} with no results; end of {hlog}:\n{tail(hlog)}",
                  flush=True)
    with open(os.path.join(out, f"{a.label}.json"), "w") as f:
        json.dump(doc, f, indent=1, default=str)
    with open(os.path.join(out, f"{a.label}.md"), "w") as f:
        f.write(markdown(doc))
    bad = any(p.get("rc") for p in doc["suites"].values()) or doc.get("headless_rc")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
