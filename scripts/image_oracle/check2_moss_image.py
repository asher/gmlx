#!/usr/bin/env python3
"""Check 2 oracle for MOSS-Image: the Qwen3.5-9B text encoder read after
its final norm.

MOSSImagePipeline.encode_prompt renders each prompt with the tokenizer's
chat template for one user turn, with the generation prompt and thinking
on, tokenizes it with truncation at 4096 tokens, runs it alone through
the text model and keeps last_hidden_state: (L, 4096) per prompt, with no
pad rows. This script runs the same steps on the model that transformers
builds from the encoder GGUF itself (arch qwen35), which keeps each
quantized matrix in its GGUF blocks and dequantizes it on every call.
Before the run, one matrix of each codec is dequantized by transformers
and by gguf-py, and the two must agree bit for bit in float32, which is
ggml's arithmetic.

The bf16 run loads the model again in bfloat16, as the release loads its
encoder, and each GGUF product then runs as a float32 GEMM over the
bfloat16 operands with one rounding, as oracle.emulate_bf16_matmul does
for torch's linear. Each weight is dequantized in float32 and rounded to
bfloat16 once. The tokenizer is the release's, from --tokenizer, loaded
as Qwen2Tokenizer, the class the release's model_index.json names. The
manifest records the class and its pre-tokenizer.

The prompts are check 2's common set, plus a Devanagari and a Thai prompt
whose combining marks split differently under a pre-tokenizer that
keeps marks with their letters and one that does not.

For each prompt the fixture holds the token ids and the fp32 and bf16
features. The manifest records the bf16 against fp32 gap of each prompt
and sets the tolerances: relative L2 at twice that gap, and a per-token
cosine floor of 1 - 4 (1 - c), c the oracle's own smallest per-token
cosine of the prompt.

  python scripts/image_oracle/run.py --python <oracle python> --threads 16 \\
      check2_moss_image.py --encoder Qwen3.5-9B-Q8_0.gguf \\
      --tokenizer <MOSS-Image-SFT>/tokenizer --name check2_moss_image_q8_0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import fixtures
import oracle
from check2_z_image import PROMPTS, token_cosine

NAME = "check2_moss_image"
FAMILY = "moss_image"
SCRIPT = "scripts/image_oracle/check2_moss_image.py"
RULE = ("relative L2 = 2 x the oracle's own bf16 against fp32 relative L2 "
        "of the same prompt; per-token cosine at least 1 - 4 (1 - c), c the "
        "oracle's own smallest per-token cosine of the same prompt")
MAX_LEN = 4096
# Combining marks: a vowel sign and a virama in Devanagari, two vowel signs
# in Thai.
MARKS = {"devanagari": "a temple in \u0928\u092e\u0938\u094d\u0924\u0947 style",
         "thai": "a shop sign that reads \u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35"}


def load_encoder(torch, path: Path, mode: str):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        str(path.parent), gguf_file=path.name, dtype=oracle.torch_dtype(mode),
        local_files_only=True)
    return model.eval()


def gguf_linears(model) -> list:
    from transformers.integrations.gguf.utils import GgufLinear

    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, GgufLinear)]


def census(model) -> dict:
    out: dict = {}
    for _, m in gguf_linears(model):
        out[int(m.ggml_type)] = out.get(int(m.ggml_type), 0) + 1
    return out


def check_dequant(torch, model) -> dict:
    """One matrix of each codec, dequantized by transformers and by
    gguf-py in float32. Raises unless they agree bit for bit."""
    import gguf
    from transformers.integrations.gguf.kernels import dequantize_blocks

    seen: dict = {}
    for name, m in gguf_linears(model):
        t = int(m.ggml_type)
        if t in seen or m.input_permutation is not None:
            continue
        rows = m.weight[:8]
        got = dequantize_blocks(rows, t, rows.shape[0], m.in_features,
                                torch.float32).numpy()
        want = gguf.quants.dequantize(rows.numpy(),
                                      gguf.GGMLQuantizationType(t))
        want = np.asarray(want, np.float32).reshape(got.shape)
        if got.tobytes() != want.tobytes():
            raise SystemExit(f"error: transformers dequantizes {name} "
                             f"({gguf.GGMLQuantizationType(t).name}) away "
                             "from gguf-py")
        seen[t] = name
    return {gguf.GGMLQuantizationType(t).name: n for t, n in seen.items()}


def emulate_bf16_gguf(enable: bool) -> None:
    """Each GGUF product on bfloat16 rows as a float32 GEMM over the
    bfloat16 operands, with one rounding of the result. The weight is
    dequantized in float32 and rounded to bfloat16 once, as a bf16
    checkpoint holds it."""
    import torch
    from transformers.integrations.gguf import utils as gu
    from transformers.integrations.gguf.kernels import (DEQUANT_CHUNK_ELEMS,
                                                        dequantize_blocks)

    native = getattr(gu.GgufLinear, "_oracle_native", None)
    if native is None:
        native = gu.GgufLinear._oracle_native = gu.GgufLinear._unpack_matmul
    if not enable:
        gu.GgufLinear._unpack_matmul = native
        return

    def unpack(self, flat):
        if flat.dtype != torch.bfloat16:
            return native(self, flat)
        step = max(1, DEQUANT_CHUNK_ELEMS // self.in_features)
        out = torch.empty(flat.shape[0], self.out_features,
                          dtype=torch.bfloat16)
        x = flat.float()
        for s in range(0, self.out_features, step):
            rows = self.weight[s:s + step]
            w = dequantize_blocks(rows, self.ggml_type, rows.shape[0],
                                  self.in_features, torch.float32)
            w = w.to(torch.bfloat16).float()
            out[:, s:s + step] = (x @ w.T).to(torch.bfloat16)
        return out

    gu.GgufLinear._unpack_matmul = unpack


def tokenizer(path: Path):
    from transformers import Qwen2Tokenizer

    return Qwen2Tokenizer.from_pretrained(str(path), local_files_only=True)


def pre_tokenizer(tok) -> dict:
    import json

    return json.loads(tok.backend_tokenizer.to_str())["pre_tokenizer"]


def token_ids(tok, prompt: str) -> np.ndarray:
    """The pipeline's ids of one prompt."""
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=True)
    ids = tok(text, return_tensors="np", truncation=True,
              max_length=MAX_LEN)["input_ids"][0]
    return ids.astype(np.int64)


def features(torch, model, ids: np.ndarray) -> dict:
    x = torch.from_numpy(ids[None])
    t0 = time.perf_counter()
    out = model(input_ids=x, use_cache=False, return_dict=True)
    secs = time.perf_counter() - t0
    return {"features": oracle.to_numpy(out.last_hidden_state[0]),
            "seconds": secs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--encoder", required=True, type=Path,
                    help="Qwen3.5-9B text GGUF")
    ap.add_argument("--tokenizer", required=True, type=Path,
                    help="the release's tokenizer folder")
    ap.add_argument("--name", default=NAME)
    ap.add_argument("--prompts", nargs="*",
                    default=[*PROMPTS, *MARKS])
    ap.add_argument("--prompt", nargs=2, action="append", default=[],
                    metavar=("NAME", "TEXT"),
                    help="one more prompt by name, such as the prompt of "
                         "check 8's trajectory")
    a = ap.parse_args(argv)
    known = {**PROMPTS, **MARKS}
    prompts = {**{k: known[k] for k in a.prompts}, **dict(a.prompt)}
    torch = oracle.setup()
    tok = tokenizer(a.tokenizer)
    ids = {k: token_ids(tok, v) for k, v in prompts.items()}
    fx = fixtures.FixtureWriter(a.name, check=2, family=FAMILY,
                                script=SCRIPT)
    fx.add_file("encoder", a.encoder)
    for f in ("tokenizer.json", "tokenizer_config.json",
              "chat_template.jinja"):
        if (a.tokenizer / f).is_file():
            fx.add_file(f"tokenizer/{f}", a.tokenizer / f)
    runs, info = {}, {}
    for mode in ("fp32", "bf16"):
        t0 = time.perf_counter()
        model = load_encoder(torch, a.encoder, mode)
        info[f"{mode}_load_s"] = round(time.perf_counter() - t0, 1)
        if mode == "fp32":
            info["dequant_checked"] = check_dequant(torch, model)
            info["gguf_linears"] = census(model)
        emulate_bf16_gguf(mode == "bf16")
        oracle.emulate_bf16_matmul(mode == "bf16")
        runs[mode] = {k: features(torch, model, v) for k, v in ids.items()}
        print(f"[check2] {mode}: {info[f'{mode}_load_s']}s load, "
              f"{ {k: round(r['seconds'], 1) for k, r in runs[mode].items()} }",
              flush=True)
        del model
    emulate_bf16_gguf(False)
    oracle.emulate_bf16_matmul(False)
    fx.inputs.update(prompts=prompts, max_sequence_length=MAX_LEN,
                     tokenizer_class=type(tok).__name__,
                     pre_tokenizer=pre_tokenizer(tok),
                     read="last_hidden_state, after the final norm",
                     template="the tokenizer's chat template, one user turn, "
                              "add_generation_prompt and enable_thinking",
                     matmul_bf16=("bf16 operands, float32 GEMM, one bf16 "
                                  "rounding"), **info)
    rel, cos, rows, secs = {}, {}, {}, {}
    for k in prompts:
        f32, b16 = runs["fp32"][k], runs["bf16"][k]
        fx.add(f"ids_{k}", ids[k])
        fx.add(f"features_{k}", f32["features"])
        fx.add(f"features_{k}_bf16", b16["features"])
        gap = fixtures.rel_l2(b16["features"], f32["features"])
        c = token_cosine(b16["features"], f32["features"])
        rel[k] = 2 * gap
        cos[k] = float(1 - 4 * (1 - c.min()))
        rows[k] = int(f32["features"].shape[0])
        secs[k] = {m: round(runs[m][k]["seconds"], 1) for m in runs}
        fx.results[f"bf16_vs_fp32_{k}"] = {
            "rel_l2": gap, "token_cosine_min": float(c.min()),
            "worst_token": int(c.argmin())}
        print(f"[check2] {k}: {rows[k]} rows, bf16 vs fp32 rel L2 "
              f"{gap:.3e}, min token cosine {c.min():.5f}", flush=True)
    fx.inputs["rows"] = rows
    fx.results["seconds"] = secs
    fx.tolerances.update(rel_l2=rel, token_cosine_min=cos, rule=RULE)
    fx.note("The bf16 model is the GGUF loaded again in bfloat16, as the "
            "release loads its encoder.")
    print(f"[check2] wrote {fx.write()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
