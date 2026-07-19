#!/usr/bin/env python3
"""End-to-end LoRA-on-GGUF test: creation through serving.

Exercises the whole GGUF LoRA loop as a user runs it - **GGUF in, GGUF out**, no
safetensors, no merge:

  1. prep a tiny finetune set (the cached ``GPT007/pirate_speak`` turns, or a
     bundled synthetic pirate set so the harness is self-contained);
  2. ``gmlx train`` a LoRA on a small K-quant GGUF base (the trainer runs the
     adapter's gradient through the kquant matmul's vjp; the frozen base carries no
     float copy) and emit a **GGUF** adapter;
  3. ``gmlx serve`` the base alone, then the base ``--adapter`` the trained GGUF,
     and fire the same prompts at each over real HTTP;
  4. assert the adapter shifts the served output - pirate markers appear with the
     adapter and the response diverges from the base - with greedy decoding so the
     check is deterministic.

Both verbs run as real subprocesses (the console script + ``gmlx.server``), so
argparse, the monkeypatch wiring, uvicorn, and the residency pool all get exercised.

Not ``test_``-prefixed, so pytest skips it - it needs the GPU and a base GGUF. Run
it directly with the project interpreter::

    python tests/e2e/run_lora_e2e.py                 # full: prep -> train -> serve
    python tests/e2e/run_lora_e2e.py --iters 60      # quicker train
    python tests/e2e/run_lora_e2e.py --reuse-adapter ADAPTER.gguf   # skip training
    python tests/e2e/run_lora_e2e.py --keep --out ./lora-e2e-out    # keep artifacts
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
from client import Client                  # noqa: E402
from models import ModelRegistry           # noqa: E402
from server_proc import ServerProc         # noqa: E402

# Distinctive pirate tokens - phrases/spellings that don't show up in a plain
# assistant answer, so their presence is a clean signal the LoRA took.
PIRATE_MARKERS = [
    "arr", "ahoy", "matey", "hearty", "heartie", "scurvy", "timbers", "grog",
    "landlubber", "booty", "doubloon", "cap'n", "avast", "scallywag", "aye",
    "yer ", "ye be", "ye'", "swashbuckl", "shiver me", "me timbers", "blimey",
    "walk the plank", "sea dog", "jolly roger", "me hearties",
]

PROMPTS = [
    "What's the weather like today?",
    "Can you help me write an email to my boss?",
]

# A self-contained fallback finetune set (used only if the HF pirate dataset isn't
# cached) - enough varied turns to overfit a small base into a clear pirate voice.
SYNTHETIC_TURNS = [
    ("Hello there!", "Ahoy there, matey! Welcome aboard, ye landlubber!"),
    ("How are you?", "Arrr, I be feelin' fine as a fresh barrel o' grog, thankee!"),
    ("What's the weather like?", "Shiver me timbers, the skies be clear fer sailin', matey!"),
    ("Tell me a joke.", "Arrr! Why couldn't the crew play cards? The cap'n be standin' on the deck!"),
    ("What should I eat for dinner?", "Ye should be feastin' on hardtack and salted fish, ye scurvy dog!"),
    ("Help me write an email.", "Aye, matey! Hoist the words and set sail: greet yer cap'n proper-like!"),
    ("What time is it?", "Arrr, 'tis time to swab the deck and count yer doubloons, matey!"),
    ("Recommend a book.", "Ahoy! Read ye a tale o' buried booty and the high seas, hearty!"),
    ("How do I make coffee?", "Brew it black as the Jolly Roger and strong enough fer a sea dog, aye!"),
    ("What's your favorite color?", "Arrr, the deep blue o' the seven seas, where treasure be waitin', matey!"),
    ("Give me advice.", "Avast! Trust yer compass, guard yer booty, and never cross the cap'n!"),
    ("Tell me about yourself.", "I be a humble pirate o' the code seas, sailin' fer treasure, ye scallywag!"),
    ("What's two plus two?", "Arrr, four doubloons in yer chest, matey, countin' true!"),
    ("How was your day?", "Busy as a bosun, matey - swabbin' decks and chasin' booty all day, aye!"),
    ("Say something nice.", "Ye be the finest hearty to ever walk the plank with me, matey!"),
    ("What's the capital of France?", "Arrr, Paris it be, a fine port fer a thirsty sea dog, matey!"),
]


def count_markers(text: str) -> int:
    low = text.lower()
    return sum(1 for m in PIRATE_MARKERS if m in low)


def prep_data(dest: Path) -> str:
    """Write train.jsonl/valid.jsonl chat records. Prefer the cached HF pirate set;
    fall back to the bundled synthetic turns so the harness needs no network."""
    dest.mkdir(parents=True, exist_ok=True)
    records = _load_pirate_hf() or _synthetic_records()
    split = max(1, len(records) // 10)
    (dest / "valid.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records[:split]))
    (dest / "train.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records[split:]))
    src = "GPT007/pirate_speak" if len(records) > len(SYNTHETIC_TURNS) else "synthetic"
    print(f"[data] {len(records) - split} train / {split} valid from {src}")
    return str(dest)


def _load_pirate_hf():
    """The cached pirate dataset as chat records, or None if it can't be loaded
    (no network is attempted - offline-cache only)."""
    import re
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        from datasets import load_dataset
        ds = load_dataset("GPT007/pirate_speak", split="train")
    except Exception as e:                               # noqa: BLE001
        print(f"[data] HF pirate set unavailable ({type(e).__name__}); using synthetic")
        return None
    turn = re.compile(
        r"user<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>.*?"
        r"assistant<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>", re.DOTALL)
    out = []
    for row in ds:
        m = turn.search(row["text"])
        if m:
            out.append({"messages": [
                {"role": "user", "content": m.group(1).strip()},
                {"role": "assistant", "content": m.group(2).strip()}]})
    return out or None


def _synthetic_records():
    return [{"messages": [{"role": "user", "content": u},
                          {"role": "assistant", "content": a}]}
            for u, a in SYNTHETIC_TURNS]


def _gmlx_cli(python: str) -> str:
    """The ``gmlx`` console script next to the interpreter (how a user runs it)."""
    cand = os.path.join(os.path.dirname(python), "gmlx")
    if not os.path.exists(cand):
        raise FileNotFoundError(
            f"no gmlx console script at {cand} - install the package in this env")
    return cand


def train_adapter(python: str, base: str, data_dir: str, out_path: str, *,
                  iters: int, num_layers: int, rank: int, scale: float,
                  log_path: str) -> str:
    argv = [_gmlx_cli(python), "train", base,
            "--data", data_dir, "--adapter-out", out_path,
            "--iters", str(iters), "--num-layers", str(num_layers),
            "--rank", str(rank), "--scale", str(scale),
            "--batch-size", "4", "--steps-per-report", str(max(10, iters // 4))]
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    print(f"[train] {' '.join(argv)}")
    with open(log_path, "w") as log:
        log.write(f"# argv: {' '.join(argv)}\n")
        log.flush()
        r = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, env=env)
    if r.returncode != 0 or not os.path.exists(out_path):
        tail = Path(log_path).read_text().splitlines()[-40:]
        raise RuntimeError("training failed:\n" + "\n".join(tail))
    print(f"[train] wrote adapter -> {out_path}")
    return out_path


def serve_and_query(python: str, base: str, adapter: str | None,
                    log_path: str) -> list[str]:
    """Serve the base (optionally with the GGUF adapter) and greedily complete each
    prompt over HTTP. Returns the response texts."""
    serve_args = [base] + (["--adapter", adapter] if adapter else [])
    with ServerProc(serve_args, log_path=log_path, python=python) as proc:
        proc.wait_ready()
        client = Client(proc.base_url)
        # The positional single-model serve derives a friendly id from the filename;
        # /v1/models is the source of truth for what to address.
        status, body = client.models()
        if status != 200 or not body.get("data"):
            raise RuntimeError(f"/v1/models failed ({status}): {body}\n"
                               + proc.log_tail())
        model_id = body["data"][0]["id"]
        responses = []
        for prompt in PROMPTS:
            status, body = client.chat(
                model_id, [{"role": "user", "content": prompt}],
                max_tokens=160, temperature=0.0)
            if status != 200:
                raise RuntimeError(f"chat failed ({status}): {body}\n"
                                   + proc.log_tail())
            responses.append(body["choices"][0]["message"]["content"])
        return responses


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-root", default=ModelRegistry.root,
                    help="root for the base GGUF (default ~/llm/gguf)")
    ap.add_argument("--base-handle", default="qwen3_0_6b_q8",
                    help="model registry handle for the base (default qwen3_0_6b_q8)")
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--num-layers", type=int, default=8)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--reuse-adapter", default=None,
                    help="skip training and serve this existing GGUF adapter")
    ap.add_argument("--min-markers", type=int, default=3,
                    help="min pirate markers (summed over prompts) for a pass")
    ap.add_argument("--out", default=None, help="artifact dir (default: temp, removed)")
    ap.add_argument("--keep", action="store_true", help="keep artifacts on disk")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter for the train/serve subprocesses")
    a = ap.parse_args()

    reg = ModelRegistry(root=a.models_root)
    base = reg.find(a.base_handle)
    if base is None:
        print(f"SKIP: base handle {a.base_handle!r} not found under {a.models_root}.")
        reg.print_bootstrap([a.base_handle])
        return 0

    tmp = a.out or tempfile.mkdtemp(prefix="gguf-lora-e2e-")
    Path(tmp).mkdir(parents=True, exist_ok=True)
    print(f"[e2e] base={base}\n[e2e] artifacts={tmp}")
    rc = 1
    try:
        if a.reuse_adapter:
            adapter = os.path.abspath(os.path.expanduser(a.reuse_adapter))
            print(f"[train] reusing adapter {adapter}")
        else:
            data_dir = prep_data(Path(tmp) / "data")
            adapter = train_adapter(
                a.python, base, data_dir, os.path.join(tmp, "pirate-adapter.gguf"),
                iters=a.iters, num_layers=a.num_layers, rank=a.rank, scale=a.scale,
                log_path=os.path.join(tmp, "train.log"))

        print("[serve] base (no adapter) ...")
        base_out = serve_and_query(a.python, base, None,
                                   os.path.join(tmp, "serve-base.log"))
        print("[serve] base + pirate adapter ...")
        adapt_out = serve_and_query(a.python, base, adapter,
                                    os.path.join(tmp, "serve-adapter.log"))

        rc = _grade(base_out, adapt_out, a.min_markers, Path(tmp))
    finally:
        if not a.keep and not a.out:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
        elif a.keep or a.out:
            print(f"[e2e] artifacts kept under {tmp}")
    return rc


def _grade(base_out: list[str], adapt_out: list[str], min_markers: int,
           out_dir: Path) -> int:
    base_m = sum(count_markers(t) for t in base_out)
    adapt_m = sum(count_markers(t) for t in adapt_out)
    diverged = sum(1 for b, a in zip(base_out, adapt_out) if b.strip() != a.strip())

    lines = ["# LoRA-on-GGUF e2e\n",
             f"- base pirate markers : {base_m}",
             f"- adapter markers     : {adapt_m}",
             f"- prompts diverged    : {diverged}/{len(PROMPTS)}\n"]
    for i, prompt in enumerate(PROMPTS):
        lines += [f"## prompt {i + 1}: {prompt}",
                  f"**base:** {base_out[i].strip()[:400]}\n",
                  f"**adapter:** {adapt_out[i].strip()[:400]}\n"]
    report = "\n".join(lines)
    (out_dir / "report.md").write_text(report)
    print("\n" + report)

    checks = {
        "adapter shows pirate voice": adapt_m >= min_markers,
        "adapter more pirate than base": adapt_m > base_m,
        "output diverged": diverged >= 1,
    }
    print("=" * 60)
    ok = True
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print("=" * 60)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
