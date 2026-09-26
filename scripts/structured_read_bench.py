#!/usr/bin/env python
"""Time the parts of a structured read on a DiffusionGemma GGUF.

    python scripts/structured_read_bench.py DIFFUSIONGEMMA.gguf [--rounds 5]

Loads the model in process and times prompt prefill, reads across canvas
widths and sample counts, batched and one sample at a time, denoise steps,
constrained and unconstrained unembedding, prompt extension, a thought, and
whole decisions. Each arm reports the median of ``--rounds`` timed runs
after ``--warmup`` untimed ones, with the arm order reversed on every other
round. Run it on an idle machine. Writes JSON and a markdown table under
``--out``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time

import mlx.core as mx

from gmlx.systemone import ReadRequest, TemplateResolver, decide, jev_schema, jev_state
from gmlx.systemone.engine import BoundReader, ChatTokens, StructuredReader, engine_scope
from gmlx.systemone.template import system_text

_QUESTIONS = {
    "a": {"type": "noul", "instructions": "Does the text describe an outage?"},
    "b": {"type": "choice", "instructions": "Which team owns it?",
          "criteria": {"infra": "outages", "billing": "money", "product": "features"}},
    "c": {"type": "score", "instructions": "How severe is it?",
          "criteria": ["low", "medium", "high"]},
}
_FILLER = ("The service returned errors for several minutes before recovering, "
           "and the customer asked for an explanation of the incident. ")


def _sync():
    mx.synchronize()


def _therm() -> str:
    try:
        return subprocess.run(["pmset", "-g", "therm"], capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _peak_gb() -> float:
    get = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
    return get() / 1e9


class Bench:
    def __init__(self, rounds: int, warmup: int, cooldown: float):
        self.rounds = rounds
        self.warmup = warmup
        self.cooldown = cooldown
        self.rows: list[dict] = []

    def block(self, name: str, arms: list[tuple[str, dict, object]]):
        """``arms``: (label, params, fn) where ``fn()`` runs once and returns
        the milliseconds to record."""
        print(f"\n[{name}]", flush=True)
        times: dict[str, list[float]] = {label: [] for label, _, _ in arms}
        for r in range(self.warmup + self.rounds):
            order = arms if r % 2 == 0 else list(reversed(arms))
            for label, _, fn in order:
                ms = fn()
                if r >= self.warmup:
                    times[label].append(ms)
        for label, params, _ in arms:
            ts = times[label]
            row = {"block": name, "arm": label, **params,
                   "median_ms": statistics.median(ts), "min_ms": min(ts),
                   "max_ms": max(ts), "runs": len(ts)}
            self.rows.append(row)
            print(f"  {label:<40} {row['median_ms']:9.1f} ms "
                  f"(min {row['min_ms']:.1f}, max {row['max_ms']:.1f})", flush=True)
        if self.cooldown:
            time.sleep(self.cooldown)


def _timed(fn):
    _sync()
    t0 = time.perf_counter()
    fn()
    _sync()
    return (time.perf_counter() - t0) * 1e3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("gguf")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--cooldown", type=float, default=15.0,
                    help="seconds of idle between blocks")
    ap.add_argument("--canvas", type=int, default=64)
    ap.add_argument("--prompts", default="200,500,1000,2000")
    ap.add_argument("--out", default=os.path.expanduser(
        "~/.local/state/claude-scratch/gmlx/systemone"))
    a = ap.parse_args()

    from gmlx.gen.diffusion import is_diffusion_model
    from gmlx.load.loader import load_model
    from gmlx.serve.bridge_vlm import _make_text_processor

    therm_start = _therm()
    t0 = time.perf_counter()
    model, _config, tokenizer = load_model(a.gguf, verbose=False)
    load_s = time.perf_counter() - t0
    if not is_diffusion_model(model):
        sys.exit(f"{a.gguf} is not a DiffusionGemma GGUF")
    processor = _make_text_processor(tokenizer)
    gen = importlib.import_module("mlx_vlm.server.generation")
    step_size = int(gen.get_prefill_step_size())
    canvas = min(a.canvas, int(model.config.canvas_length))

    schema = jev_schema({"questions": _QUESTIONS, "state": ""})
    sys_text = system_text(schema)
    base_tokens = ChatTokens(processor, "")
    resolver = TemplateResolver(base_tokens.enc, canvas)
    template, slots = resolver.template_for(schema, resolver.scaffold, "")
    need = resolver.canvas_width(template)
    per_filler = len(base_tokens.enc(_FILLER))
    base_len = len(base_tokens.chat_ids(sys_text, False))

    def state_for(p: int) -> str:
        return _FILLER * max(0, (p - base_len) // per_filler)

    def prompt_ids(p: int) -> list[int]:
        return ChatTokens(processor, state_for(p)).chat_ids(sys_text, False)

    lengths = [int(x) for x in a.prompts.split(",")]
    prompts = {p: prompt_ids(p) for p in lengths}
    mid = 500 if 500 in prompts else lengths[0]
    bench = Bench(a.rounds, a.warmup, a.cooldown)

    with engine_scope(model, 0):
        reader = StructuredReader(model, prefill_step_size=step_size)
        chunked = StructuredReader(model, prefill_step_size=512)

        arms = []
        for p, ids in prompts.items():
            arms.append((f"prefill P={len(ids)}", {"prompt": len(ids), "chunked": False},
                         lambda ids=ids: _timed(lambda: reader.prefill(ids))))
            arms.append((f"prefill P={len(ids)} chunk 512", {"prompt": len(ids), "chunked": True},
                         lambda ids=ids: _timed(lambda: chunked.prefill(ids))))
        bench.block("prefill", arms)

        cache = reader.prefill(prompts[mid])
        seeds = [42 + k * 7919 for k in range(8)]

        def req(width, n, *, steps=1, constrained=True, rows=None):
            return ReadRequest(template=tuple(template), slots=tuple(slots), width=width,
                               seeds=tuple(seeds[:n]), steps=steps,
                               constrained=constrained, rows_per_pass=rows)

        arms = []
        for width in (16, 32, 64):
            if width < need or width > canvas:
                continue
            for n in (1, 2, 4):
                batched = req(width, n, rows=width * n)
                arms.append((f"read W={width} n={n} batched",
                             {"width": width, "samples": n, "batched": True},
                             lambda r=batched: _timed(lambda: reader.read(cache, r))))
                if n > 1:
                    ones = [req(width, 1) for _ in range(n)]
                    ones = [ReadRequest(template=o.template, slots=o.slots, width=width,
                                        seeds=(seeds[k],)) for k, o in enumerate(ones)]
                    arms.append((f"read W={width} n={n} one at a time",
                                 {"width": width, "samples": n, "batched": False},
                                 lambda rs=ones: _timed(
                                     lambda: [reader.read(cache, r) for r in rs])))
        bench.block(f"reads at P={len(prompts[mid])}", arms)

        width = max(32, need)
        arms = []
        for steps in (1, 2, 4, 8):
            for constrained in (True, False):
                r = req(width, 1, steps=steps, constrained=constrained)
                arms.append((f"steps={steps} {'constrained' if constrained else 'full vocab'}",
                             {"width": width, "steps": steps, "constrained": constrained},
                             lambda r=r: _timed(lambda: reader.read(cache, r))))
        for constrained in (True, False):
            r = req(width, 4, constrained=constrained, rows=width * 4)
            arms.append((f"n=4 steps=1 {'constrained' if constrained else 'full vocab'}",
                         {"width": width, "samples": 4, "constrained": constrained},
                         lambda r=r: _timed(lambda: reader.read(cache, r))))
        bench.block(f"steps and unembedding at W={width}", arms)

        extra = base_tokens.enc(" yes\nb: infra\nc: high and more words here")

        def extend_ms(k):
            fresh = reader.prefill(prompts[mid])
            _sync()
            return _timed(lambda: fresh.extend(extra[:k]))

        bench.block("prompt extension", [
            ("extend 1 token", {"tokens": 1}, lambda: extend_ms(1)),
            ("extend 8 tokens", {"tokens": 8}, lambda: extend_ms(8)),
        ])

        think_prompt = (ChatTokens(processor, state_for(mid)).chat_ids(sys_text, True)
                        + list(resolver.thought_open))
        backend = processor.tokenizer
        thought_tokens = []

        def think_ms():
            def run():
                ids, _info = reader.think(think_prompt, 64,
                                          stop_id=resolver.thought_close[0],
                                          canvas_width=canvas, processor=processor,
                                          backend=backend)
                thought_tokens.append(len(ids))
            return _timed(run)

        bench.block("thought", [("think budget 64", {"budget": 64}, think_ms)])

        state = jev_state({"state": state_for(mid)})
        tokens = ChatTokens(processor, state)
        engine = BoundReader(reader, processor=processor, backend=backend)
        reads_seen = {}

        def decide_ms(samples):
            body = {"questions": _QUESTIONS, "state": state, "samples": samples}
            sch = jev_schema(body)

            def run():
                out, _ = decide(sch, state, engine=engine, resolver=resolver,
                                chat_ids=tokens.chat_ids, seed=42, constrained=True,
                                canvas_len=canvas, decode=tokens.decode)
                reads_seen[str(samples)] = out["diagnostics"]["timing"]["reads"]
            return _timed(run)

        bench.block("whole decisions", [
            ("decide samples=1", {"samples": 1}, lambda: decide_ms(1)),
            ("decide samples=4", {"samples": 4}, lambda: decide_ms(4)),
            ('decide samples="auto"', {"samples": "auto"}, lambda: decide_ms("auto")),
        ])

    result = {
        "gguf": os.path.abspath(a.gguf),
        "machine": platform.machine(),
        "chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                               capture_output=True, text=True).stdout.strip(),
        "load_s": load_s,
        "prefill_step_size": step_size,
        "canvas": canvas,
        "template_tokens": len(template),
        "min_width": need,
        "thought_tokens": thought_tokens,
        "decide_reads": reads_seen,
        "peak_gb": _peak_gb(),
        "therm_start": therm_start,
        "therm_end": _therm(),
        "rows": bench.rows,
    }
    os.makedirs(a.out, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    base = os.path.join(a.out, f"structured-read-bench-{stamp}")
    with open(base + ".json", "w") as f:
        json.dump(result, f, indent=2)
    with open(base + ".md", "w") as f:
        f.write("| Block | Arm | Median ms | Min | Max |\n|---|---|---|---|---|\n")
        for row in bench.rows:
            f.write(f"| {row['block']} | {row['arm']} | {row['median_ms']:.1f} | "
                    f"{row['min_ms']:.1f} | {row['max_ms']:.1f} |\n")
    print(f"\npeak memory {result['peak_gb']:.1f} GB, thought tokens {thought_tokens}, "
          f"decide reads {reads_seen}")
    print(f"wrote {base}.json and {base}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
