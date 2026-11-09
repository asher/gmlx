#!/usr/bin/env python3
"""Expert adapters on MoE models, resident and with the experts streamed.

For each model, a rank-1 adapter minted on every expert down projection is
served resident and streamed, and the same greedy chats run with top-20
logprobs at the model's strength and at 0:

* every request answers;
* resident, fused blocks (the default) give the adapter the same effect as
  unfused blocks. They are not bit-identical: the fused block mixes the
  experts in float32;
* under streaming, fused blocks (``GMLX_FUSED_MOE_BLOCK=1``) and unfused
  blocks (``=0``) give identical logprobs, since a streamed fused block runs
  its stock forward;
* the adapter's effect on the first token, on minus off, correlates with
  the resident load's: a slot mapped to the wrong expert gives no
  correlation;
* a streamed load decodes from the arena and counts the adapter tables
  in its budget line. Gemma 4 stores gate and up as one fused tensor, so
  its row checks the feeders on that layout.

Models: a Qwen3.6-35B-A3B file (``qwen35moe``, also with ``--stream-cpu``)
and a Gemma 4 26B-A4B file, each skipped when missing. Not ``test_``-
prefixed, so pytest skips it; run it on an idle Mac::

    python tests/e2e/run_moe_lora_e2e.py
    python tests/e2e/run_moe_lora_e2e.py --qwen PATH --gemma PATH --out DIR

Exit 0 on pass, 1 on a failed check.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from client import Client                  # noqa: E402
from server_proc import ServerProc         # noqa: E402

GGUF = os.path.expanduser("~/llm/gguf")
QWEN = f"{GGUF}/unsloth__Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
GEMMA = f"{GGUF}/unsloth__gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
PROMPTS = [
    "Name three rivers in Europe and one fact about each.",
    "Explain what a hash table is in two sentences.",
    "Write a haiku about autumn rain.",
]
# The least mean correlation of the adapter's effect against the resident
# unfused load: measured 0.78 to 0.97 streamed on the M3 Max, near 0 for a
# slot mapped to the wrong expert.
MIN_CORR = 0.6
_failed: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}",
          flush=True)
    if not ok:
        _failed.append(name)


def mint(base: str, out: str) -> None:
    """A rank-1 adapter on the expert and shared-expert down projections of
    every trunk layer of ``base``."""
    import numpy as np
    from helpers import mint_lora_gguf

    from gmlx.load.headerscan import scan_gguf

    kv = scan_gguf(base, include_tensors=False).kv
    arch = kv["general.architecture"]
    trunk = int(kv[f"{arch}.block_count"]) - int(kv.get(f"{arch}.nextn_predict_layers") or 0)
    shapes = {t.name: t.shape for t in scan_gguf(base).tensors}
    rng = np.random.default_rng(0)
    pairs = {}
    for i in range(trunk):
        name = f"blk.{i}.ffn_down_exps.weight"
        if name in shapes:
            inter, hidden, e = shapes[name]
            pairs[name] = (rng.standard_normal((e, 1, inter)) / np.sqrt(inter),
                           rng.standard_normal((e, hidden, 1)) * 0.5)
        name = f"blk.{i}.ffn_down_shexp.weight"
        if name in shapes:
            inter, hidden = shapes[name]
            pairs[name] = (rng.standard_normal((1, inter)) / np.sqrt(inter),
                           rng.standard_normal((hidden, 1)) * 0.5)
    mint_lora_gguf(out, arch, pairs)


def serve(model: str, adapter: str, flags: list, block: str, strength: float,
          log: str, python: str) -> dict | None:
    """Each prompt at ``strength`` and at 0: its chosen tokens with their
    top-20 logprobs. None when a request fails."""
    env = {"GMLX_FUSED_MOE_BLOCK": block}
    with ServerProc([*flags, "--adapter", adapter, model], env_extra=env,
                    log_path=log, python=python) as srv:
        srv.wait_ready(timeout=900)
        c = Client(srv.base_url)
        mid = c.models()[1]["data"][0]["id"]
        out = {}
        for s in (strength, 0.0):
            for i, p in enumerate(PROMPTS):
                code, body = c.chat(
                    mid, [{"role": "user", "content": p}], max_tokens=24,
                    temperature=0, logprobs=True, top_logprobs=20,
                    chat_template_kwargs={"enable_thinking": False},
                    adapters=[{"id": os.path.basename(adapter), "strength": s}])
                if code != 200:
                    check(f"{os.path.basename(log)} answers", False, f"{code} {body}")
                    return None
                out[(s, i)] = [(t["token"], t["logprob"],
                                [(a["token"], a["logprob"]) for a in t["top_logprobs"]])
                               for t in body["choices"][0]["logprobs"]["content"]]
    with open(log) as f:
        out["log"] = f.read()
    return out


def effect_corr(ref: dict, got: dict, strength: float, i: int) -> float:
    """Correlation of the adapter's first-token effect over the top-20
    tokens both loads share."""
    def eff(d):
        on = dict(d[(strength, i)][0][2])
        off = dict(d[(0.0, i)][0][2])
        return {k: on[k] - off[k] for k in on if k in off}
    a, b = eff(ref), eff(got)
    keys = [k for k in a if k in b]
    xa, xb = [a[k] for k in keys], [b[k] for k in keys]
    ma, mb = sum(xa) / len(xa), sum(xb) / len(xb)
    cov = sum((u - ma) * (v - mb) for u, v in zip(xa, xb))
    den = math.sqrt(sum((u - ma) ** 2 for u in xa) * sum((v - mb) ** 2 for v in xb))
    return cov / den if den else 0.0


def run_model(tag: str, model: str, strength: float, modes: list, arena: bool,
              out: str, python: str) -> None:
    adapter = os.path.join(out, f"{tag}-experts-r1.gguf")
    mint(model, adapter)
    ref = serve(model, adapter, [], "0", strength,
                os.path.join(out, f"{tag}-resident-block0.log"), python)
    if ref is None:
        return
    fused = serve(model, adapter, [], "1", strength,
                  os.path.join(out, f"{tag}-resident-block1.log"), python)
    if fused is not None:
        corr = [effect_corr(ref, fused, strength, i) for i in range(len(PROMPTS))]
        check(f"{tag} resident: fused blocks give the adapter's effect",
              sum(corr) / len(corr) >= MIN_CORR,
              f"correlation {', '.join(f'{c:.2f}' for c in corr)}")
    for flag in modes:
        runs = {b: serve(model, adapter, [flag], b, strength,
                         os.path.join(out, f"{tag}{flag}-block{b}.log"), python)
                for b in ("1", "0")}
        if None in runs.values():
            continue
        fused, unfused = runs["1"], runs["0"]
        same = all(fused[k] == unfused[k] for k in fused if k != "log")
        check(f"{tag} {flag}: fused and unfused blocks match", same)
        corr = [effect_corr(ref, unfused, strength, i) for i in range(len(PROMPTS))]
        mean = sum(corr) / len(corr)
        check(f"{tag} {flag}: adapter effect matches resident", mean >= MIN_CORR,
              f"correlation {', '.join(f'{c:.2f}' for c in corr)}")
        if flag != "--stream-experts" or not arena:
            continue
        log = unfused["log"]
        check(f"{tag} {flag}: decodes from the arena",
              "decode feeder unavailable" not in log)
        check(f"{tag} {flag}: prefills through the ring",
              "feeder prefill unavailable" not in log)
        budget = [ln for ln in log.splitlines() if "[stream] memory budget:" in ln]
        check(f"{tag} {flag}: budget counts the adapter tables",
              bool(budget) and "(adapter tables" in budget[0],
              budget[0].strip()[:120] if budget else "no budget line")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qwen", default=QWEN)
    ap.add_argument("--gemma", default=GEMMA)
    ap.add_argument("--out", default=None, help="log and adapter dir (default: temp)")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()
    out = a.out or tempfile.mkdtemp(prefix="moe-lora-e2e-")
    os.makedirs(out, exist_ok=True)
    print(f"logs in {out}", flush=True)
    for tag, model, strength, modes, arena in (
            ("qwen36", a.qwen, 0.2, ["--stream-experts", "--stream-cpu"], True),
            ("gemma4", a.gemma, 1.0, ["--stream-experts"], True)):
        if not os.path.isfile(model):
            print(f"SKIP  {tag}: {model} not found", flush=True)
            continue
        run_model(tag, model, strength, modes, arena, out, a.python)
    print("FAIL" if _failed else "PASS", flush=True)
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
