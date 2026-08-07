"""Teacher-forced KLD harness for KV-cache quantization schemes.

Runs the same corpus through the same model once per cache arm and
measures each arm's logit divergence from the fp16-cache baseline, on two
legs: the prefill leg (chunked prefill logits, exercising the quantized
prefill/materialize paths) and the decode leg (token-by-token from full
prefill depth, exercising the fused decode kernels).

Arms: fp16 (baseline), kv8 (affine QuantizedKVCache, group 64), kvarnN
(KVarNKVCache at N bits; kvarn6 and kvarn4 by default; kvarnk6v5-style
names select mixed K/V widths). Primary metric is median KLD per leg;
same_top is the argmax agreement rate. The routine gate is
median_KLD(kvarn6) <= median_KLD(kv8) on both legs.

Corpus: wikitext test split (local HF cache), tokenized to ctx +
decode-tokens ids. Protocol: idle machine, one model, arms interleaved
per chunk so drift hits all arms equally.

Example:
  PYTHONPATH=~/src/mlx-kquant-kvarn python scripts/kld_harness.py \\
      ~/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf --ctx 16384 \\
      --decode-tokens 1024 --json /tmp/kld.json
"""

from __future__ import annotations

import argparse
import json
import sys

import mlx.core as mx
import numpy as np


def load_corpus_ids(tokenizer, n_tokens: int) -> list[int]:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    ids: list[int] = []
    for row in ds:
        text = row["text"]
        if not text.strip():
            continue
        ids.extend(tokenizer.encode(text))
        if len(ids) >= n_tokens:
            break
    if len(ids) < n_tokens:
        raise SystemExit(f"corpus too small: {len(ids)} < {n_tokens} tokens")
    return ids[:n_tokens]


def build_arm(name: str, model, tail_tokens: int):
    """Hybrid archs quantize/convert their plain-KV layers and leave
    recurrent state untouched, matching the runtime's partial coverage."""
    from mlx_lm.models.cache import KVCache, make_prompt_cache

    pc = make_prompt_cache(model)
    eligible = sum(1 for c in pc if type(c) is KVCache)
    if name == "fp16":
        return pc
    if name == "kv8":
        if not eligible:
            raise SystemExit("kv8 arm: no plain KV layers to quantize")
        return [
            c.to_quantized(group_size=64, bits=8) if type(c) is KVCache else c
            for c in pc
        ]
    if name.startswith("kvarn"):
        import re

        from gmlx.kvarn_cache import convert_prompt_cache
        from gmlx.kvarn_sdpa import install_kvarn_sdpa, kvarn_ops_missing

        reason = kvarn_ops_missing()
        if reason:
            raise SystemExit(f"kvarn arm unavailable: {reason}")
        spec = name[len("kvarn") :]
        m = re.fullmatch(r"k(\d)v(\d)", spec)
        k_bits, v_bits = (
            (int(m.group(1)), int(m.group(2))) if m else (int(spec), int(spec))
        )
        n = convert_prompt_cache(
            pc, k_bits=k_bits, v_bits=v_bits, tail_tokens=tail_tokens
        )
        if n == 0 or n != eligible:
            raise SystemExit(f"kvarn arm converted {n}/{eligible} eligible layers")
        install_kvarn_sdpa()
        return pc
    raise SystemExit(f"unknown arm {name!r}")


def kld_and_top(ref_logits, arm_logits):
    """Per-position KL(ref || arm) in fp32 plus argmax agreement."""
    lp = ref_logits.astype(mx.float32) - mx.logsumexp(
        ref_logits.astype(mx.float32), axis=-1, keepdims=True
    )
    lq = arm_logits.astype(mx.float32) - mx.logsumexp(
        arm_logits.astype(mx.float32), axis=-1, keepdims=True
    )
    kld = (mx.exp(lp) * (lp - lq)).sum(axis=-1)
    same = mx.argmax(lp, axis=-1) == mx.argmax(lq, axis=-1)
    mx.eval(kld, same)
    return np.array(kld).reshape(-1), np.array(same).reshape(-1)


def summarize(klds, tops):
    k = np.concatenate(klds)
    t = np.concatenate(tops)
    return {
        "positions": int(k.size),
        "median_kld": float(np.median(k)),
        "mean_kld": float(k.mean()),
        "p99_kld": float(np.percentile(k, 99)),
        "same_top": float(t.mean()),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("gguf")
    ap.add_argument("--arms", default="fp16,kv8,kvarn6,kvarn4")
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--decode-tokens", type=int, default=1024)
    ap.add_argument("--prefill-chunk", type=int, default=512)
    ap.add_argument(
        "--skip-head",
        type=int,
        default=512,
        help="Prefill positions excluded from the stats (the "
        "region every arm serves fp16; default 512).",
    )
    ap.add_argument("--kv-tail-tokens", type=int, default=1024)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    from gmlx.loader import load_model

    model, _config, tok = load_model(args.gguf)
    info = mx.device_info()
    mx.set_wired_limit(int(info["max_recommended_working_set_size"]))

    total = args.ctx + args.decode_tokens
    ids = load_corpus_ids(tok, total)
    print(
        f"[kld] corpus {total} tokens, ctx {args.ctx}, decode {args.decode_tokens}",
        file=sys.stderr,
    )

    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    if arm_names[0] != "fp16":
        raise SystemExit("the first arm must be fp16 (the baseline)")
    caches = {a: build_arm(a, model, args.kv_tail_tokens) for a in arm_names}
    stats = {a: {"prefill": ([], []), "decode": ([], [])} for a in arm_names[1:]}

    def run_span(a, b, leg):
        x = mx.array(ids[a:b])[None]
        ref = model(x, cache=caches["fp16"])
        mx.eval(ref)
        for name in arm_names[1:]:
            out = model(x, cache=caches[name])
            kld, top = kld_and_top(ref, out)
            keep = (
                slice(max(0, args.skip_head - a), None)
                if leg == "prefill"
                else slice(None)
            )
            stats[name][leg][0].append(kld[keep])
            stats[name][leg][1].append(top[keep])

    for a in range(0, args.ctx, args.prefill_chunk):
        b = min(a + args.prefill_chunk, args.ctx)
        run_span(a, b, "prefill")
        print(f"\r[kld] prefill {b}/{args.ctx}", end="", file=sys.stderr)
    print(file=sys.stderr)
    for i in range(args.decode_tokens):
        run_span(args.ctx + i, args.ctx + i + 1, "decode")
        if i % 64 == 0:
            print(f"\r[kld] decode {i}/{args.decode_tokens}", end="", file=sys.stderr)
    print(file=sys.stderr)

    report = {}
    for name in arm_names[1:]:
        report[name] = {
            leg: summarize(*stats[name][leg]) for leg in ("prefill", "decode")
        }
    for name, legs in report.items():
        for leg, s in legs.items():
            print(
                f"{name:8s} {leg:8s} median {s['median_kld']:.6f}  "
                f"mean {s['mean_kld']:.6f}  p99 {s['p99_kld']:.6f}  "
                f"same_top {s['same_top'] * 100:.2f}%  "
                f"({s['positions']} pos)"
            )
    if "kvarn6" in report and "kv8" in report:
        for leg in ("prefill", "decode"):
            a = report["kvarn6"][leg]["median_kld"]
            b = report["kv8"][leg]["median_kld"]
            verdict = "PASS" if a <= b else "FAIL"
            print(f"gate {leg}: kvarn6 median {a:.6f} vs kv8 {b:.6f} -> {verdict}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[kld] wrote {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
