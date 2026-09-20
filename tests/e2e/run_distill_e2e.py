#!/usr/bin/env python3
"""End-to-end offline distillation on a small GGUF pair: cache, align, train,
eval, as a user runs the four verbs.

  1. prep a small text corpus (a bundled paragraph set, no network);
  2. ``gmlx distill cache`` a teacher GGUF over it at a small top-K;
  3. ``gmlx distill align`` the cache to the student GGUF (identity when the
     two share a tokenizer, group projection otherwise);
  4. ``gmlx distill train`` a LoRA adapter on the student against the view;
  5. ``gmlx distill eval`` the student with ``--before``, and assert the
     training loss fell and the eval wrote both reports.

Every verb runs as a real subprocess through the console script. Not
``test_``-prefixed, so pytest skips it: it needs the GPU and two GGUFs. Run
it with the project interpreter::

    python tests/e2e/run_distill_e2e.py                      # qwen3 0.6B Q8 -> Q4
    python tests/e2e/run_distill_e2e.py --teacher-handle gemma3_1b --student-handle qwen3_0_6b_q4
    python tests/e2e/run_distill_e2e.py --keep --out ./distill-e2e-out
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import ModelRegistry           # noqa: E402

PARAGRAPHS = [
    "The tide tables for the harbour are printed every spring and pinned inside the door of the "
    "chandlery, where the ink fades by August and the fishermen read them from memory.",
    "A recipe for bread needs four things: flour, water, salt and time. The first three are bought, "
    "and the last is the one most people try to skip.",
    "The library's oldest map shows the river running east of the mill, which it did until the flood "
    "of the eighteen forties moved it half a mile west in a single night.",
    "To replace the bearing, lift the drum, mark the position of the belt, and keep the four screws "
    "in the order they came out, because two of them are longer than the others.",
    "Most of the cost of a long train journey is the first hour, when the line runs through the "
    "suburbs and the carriage is full of people who will get off before the coast.",
    "The observatory logs cloud cover every hour in tenths, and a clear night is written as a zero, "
    "which is why the best pages of the ledger look empty.",
    "A good knife is kept sharp by little and often, on a stone that is flat, with the same angle "
    "held on both sides and the burr taken off at the end.",
    "The garden wall was built dry, without mortar, and it has stood for two hundred years because "
    "each stone rests on two below it and the wind goes through instead of against.",
] * 6


def _gmlx_cli(python: str) -> str:
    cand = os.path.join(os.path.dirname(python), "gmlx")
    if not os.path.exists(cand):
        raise FileNotFoundError(f"no gmlx console script at {cand}; install the package in this env")
    return cand


def _run(cmd: list[str], log_path: str) -> int:
    print("[e2e] " + " ".join(cmd))
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        print(Path(log_path).read_text()[-3000:])
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-root", default=ModelRegistry.root, help="root for the GGUFs (default ~/llm/gguf)")
    ap.add_argument("--teacher-handle", default="qwen3_0_6b_q8", help="registry handle of the teacher")
    ap.add_argument("--student-handle", default="qwen3_0_6b_q4", help="registry handle of the student")
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", default=None, help="artifact dir (default: temp, removed)")
    ap.add_argument("--keep", action="store_true", help="keep artifacts on disk")
    ap.add_argument("--python", default=sys.executable, help="interpreter for the subprocesses")
    a = ap.parse_args()

    reg = ModelRegistry(root=a.models_root)
    teacher, student = reg.find(a.teacher_handle), reg.find(a.student_handle)
    if teacher is None or student is None:
        print(f"SKIP: handles {a.teacher_handle!r} and {a.student_handle!r} must both resolve under {a.models_root}.")
        reg.print_bootstrap([a.teacher_handle, a.student_handle])
        return 0
    gmlx = _gmlx_cli(a.python)
    tmp = a.out or tempfile.mkdtemp(prefix="gmlx-distill-e2e-")
    Path(tmp).mkdir(parents=True, exist_ok=True)
    print(f"[e2e] teacher={teacher}\n[e2e] student={student}\n[e2e] artifacts={tmp}")
    rc = 1
    try:
        corpus = Path(tmp) / "corpus.jsonl"
        corpus.write_text("".join(json.dumps({"text": p}) + "\n" for p in PARAGRAPHS))
        cache, view = os.path.join(tmp, "cache"), os.path.join(tmp, "view")
        adapter, report = os.path.join(tmp, "student-distill.gguf"), os.path.join(tmp, "train.json")
        if _run([gmlx, "distill", "cache", "--teacher", teacher, "--corpus", str(corpus), "--out", cache,
                 "--top-k", str(a.top_k), "--max-len", str(a.max_len), "--rows-per-shard", "16"],
                os.path.join(tmp, "cache.log")):
            return 1
        if _run([gmlx, "distill", "cache", "--validate", cache], os.path.join(tmp, "validate.log")):
            return 1
        if _run([gmlx, "distill", "align", "--cache", cache, "--student", student, "--out", view],
                os.path.join(tmp, "align.log")):
            return 1
        if _run([gmlx, "distill", "train", "--view", view, "--student", student, "--adapter-out", adapter,
                 "--iters", str(a.iters), "--batch-size", str(a.batch_size), "--report", report,
                 "--report-every", "5", "--val-every", str(a.iters), "--save-every", str(a.iters),
                 "--ckpt-dir", os.path.join(tmp, "ckpt")], os.path.join(tmp, "train.log")):
            return 1
        rec = json.loads(Path(report).read_text())
        losses = [r["loss"] for r in rec["log"] if "loss" in r]
        if len(losses) < 2 or not losses[-1] < losses[0]:
            print(f"FAIL: training loss did not fall: {losses}")
            return 1
        print(f"[e2e] loss {losses[0]:.4f} -> {losses[-1]:.4f} over {rec['state']['iteration']} iterations")
        if not Path(adapter).is_file():
            print(f"FAIL: no adapter at {adapter}")
            return 1
        slice_path = Path(tmp) / "heldout.txt"
        slice_path.write_text("\n\n".join(PARAGRAPHS[:4]))
        md, js = os.path.join(tmp, "eval.md"), os.path.join(tmp, "eval.json")
        if _run([gmlx, "distill", "eval", "--student", student, "--adapter", adapter, "--before",
                 "--slice", f"heldout={slice_path}", "--kld-cache", cache, "--kld-rows", "8",
                 "--max-len", str(a.max_len), "--md", md, "--json", js], os.path.join(tmp, "eval.log")):
            return 1
        ev = json.loads(Path(js).read_text())
        print(f"[e2e] bpb after {ev['after']['bpb']['heldout']['bpb']:.4f} "
              f"before {ev['before']['bpb']['heldout']['bpb']:.4f}; "
              f"kld after {ev['after']['kld']['mean_kld_nats']:.4f} before {ev['before']['kld']['mean_kld_nats']:.4f}")
        print("PASS")
        rc = 0
    finally:
        if not a.keep and not a.out:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print(f"[e2e] artifacts kept under {tmp}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
