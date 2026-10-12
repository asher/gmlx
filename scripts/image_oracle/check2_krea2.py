#!/usr/bin/env python3
"""Check 2 oracle for Krea 2: the text encoder tap.

Krea2Pipeline.get_text_hidden_states runs Qwen3-VL-4B on the prompt in
the pipeline's template, padded in the middle to a fixed length with the
pads masked and the positions counting real tokens alone, and stacks the
hidden states of twelve layers per token. This script runs the same
steps at batch 1 and keeps the rows of real tokens after the first 34,
the rows the DiT attends to: (rows, 12 * 2560), layer by layer.

transformers does not read arch qwen3vl from a GGUF, so the script builds
the language model's state from the text GGUF itself, as
check2_qwen_image21.py does: gguf-py dequantizes each tensor to float32,
and the tensors go into Qwen3VLForConditionalGeneration by name. The
model config is the Qwen3-VL config of a local Qwen-Image 2.1 folder with
the 4B widths of the GGUF header, and the tokenizer is that folder's
Qwen3-VL tokenizer. The vision tower is not loaded: a text prompt does
not reach it.

For each prompt the fixture holds the token ids of the kept tokens of the
templated prompt, as the pipeline forms them without its pads, and the
fp32 and bf16 features. The manifest records the bf16 against fp32 gap of
each prompt and sets the tolerances: relative L2 at twice that gap, and a
per-token cosine floor of 1 - 4 (1 - c), c the oracle's own smallest
per-token cosine of the prompt.

  python scripts/image_oracle/run.py --python <oracle python> --threads 16 \\
      check2_krea2.py --encoder Qwen3-VL-4B-Instruct-Q8_0.gguf \\
      --config <Qwen-Image 2.1 folder>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import fixtures
import oracle
from check2_qwen_image21 import dequantized, hf_name
from check2_z_image import PROMPTS, token_cosine

NAME = "check2_krea2"
SCRIPT = "scripts/image_oracle/check2_krea2.py"
RULE = ("relative L2 = 2 x the oracle's own bf16 against fp32 relative L2 "
        "of the same prompt; per-token cosine at least 1 - 4 (1 - c), c the "
        "oracle's own smallest per-token cosine of the same prompt")
PREFIX = ("<|im_start|>system\nDescribe the image by detailing the color, "
          "shape, size, texture, quantity, text, spatial relationships of "
          "the objects and background:<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
DROP, N_SUFFIX, MAX_LEN = 34, 5, 512
LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
# The 4B widths, from the GGUF header.
WIDTHS = {"hidden_size": 2560, "intermediate_size": 9728,
          "num_hidden_layers": 36, "num_attention_heads": 32,
          "num_key_value_heads": 8, "head_dim": 128}


def load_encoder(torch, a):
    import gguf
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    cfg = AutoConfig.from_pretrained(str(a.config / "text_encoder"),
                                     local_files_only=True)
    reader = gguf.GGUFReader(str(a.encoder))
    kv = {f.name: f for f in reader.fields.values()}

    def num(key):
        f = kv[key]
        return f.parts[f.data[0]].tolist()[0]

    head = {"hidden_size": "embedding_length",
            "intermediate_size": "feed_forward_length",
            "num_hidden_layers": "block_count",
            "num_attention_heads": "attention.head_count",
            "num_key_value_heads": "attention.head_count_kv",
            "head_dim": "attention.key_length"}
    for k, v in WIDTHS.items():
        got = num(f"qwen3vl.{head[k]}")
        if got != v:
            raise SystemExit(f"error: {k} is {got} in the GGUF, not {v}")
        setattr(cfg.text_config, k, v)
    cfg.tie_word_embeddings = True
    with init_empty_weights():
        te = Qwen3VLForConditionalGeneration(cfg)
    te.model.visual = torch.nn.Module()
    te.lm_head = torch.nn.Identity()
    lm = te.model.language_model
    want = {n: tuple(p.shape) for n, p in lm.named_parameters()}
    census, seen, left = {}, set(), []
    for t in reader.tensors:
        name = hf_name(t.name)
        if name is None:
            left.append(t.name)
            continue
        value = dequantized(torch, t)
        if tuple(value.shape) != want.get(name):
            raise SystemExit(f"error: {t.name} is {tuple(value.shape)}, "
                             f"{name} wants {want.get(name)}")
        set_module_tensor_to_device(lm, name, "cpu", value=value,
                                    dtype=torch.float32)
        seen.add(name)
        census[t.tensor_type.name] = census.get(t.tensor_type.name, 0) + 1
    if left or seen != set(want):
        raise SystemExit(f"error: GGUF tensors left over {left[:4]}, model "
                         f"parameters not set {sorted(set(want) - seen)[:4]}")
    return te.eval(), census


def tokenizer(config: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(config / "processor"),
                                         local_files_only=True)


def features(torch, te, tok, prompt: str) -> dict:
    """The pipeline's steps at batch 1, and the rows it attends to."""
    text_tokens = tok([PREFIX + prompt], truncation=True,
                      padding="max_length",
                      max_length=MAX_LEN + DROP - N_SUFFIX,
                      return_tensors="pt")
    suffix = tok([SUFFIX], return_tensors="pt")
    ids = torch.cat([text_tokens.input_ids, suffix.input_ids], dim=1)
    mask = torch.cat([text_tokens.attention_mask, suffix.attention_mask],
                     dim=1).bool()
    pos = (mask.long().cumsum(dim=-1) - 1).clamp(min=0)
    pos = pos.unsqueeze(0).expand(3, -1, -1)
    t0 = time.perf_counter()
    out = te.model.language_model(input_ids=ids, attention_mask=mask,
                                  position_ids=pos,
                                  output_hidden_states=True)
    secs = time.perf_counter() - t0
    hs = torch.stack([out.hidden_states[i] for i in LAYERS], dim=2)
    keep = mask[0].clone()
    keep[:DROP] = False
    rows = hs[0][keep]
    return {"ids": ids[0][mask[0]].numpy().astype(np.int64),
            "features": oracle.to_numpy(rows.reshape(rows.shape[0], -1)),
            "seconds": secs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--encoder", required=True, type=Path,
                    help="Qwen3-VL-4B text GGUF")
    ap.add_argument("--config", required=True, type=Path,
                    help="local Qwen-Image 2.1 folder (processor and "
                         "text_encoder config)")
    ap.add_argument("--name", default=NAME)
    ap.add_argument("--prompts", nargs="*", default=list(PROMPTS))
    ap.add_argument("--prompt", nargs=2, action="append", default=[],
                    metavar=("NAME", "TEXT"),
                    help="one more prompt by name, such as the prompt of "
                         "check 8's trajectory")
    a = ap.parse_args(argv)
    prompts = {**{k: PROMPTS[k] for k in a.prompts}, **dict(a.prompt)}
    torch = oracle.setup()
    t0 = time.perf_counter()
    te, census = load_encoder(torch, a)
    tok = tokenizer(a.config)
    print(f"[check2] loaded {a.encoder.name} in "
          f"{time.perf_counter() - t0:.1f}s, codecs {census}", flush=True)
    fx = fixtures.FixtureWriter(a.name, check=2, family="krea2",
                                script=SCRIPT)
    fx.add_file("encoder", a.encoder)
    fx.inputs.update(prompts=prompts,
                     drop_index=DROP, tap_layers=list(LAYERS),
                     max_sequence_length=MAX_LEN, suffix_tokens=N_SUFFIX,
                     gguf_codecs=census, config=str(a.config.name),
                     matmul_bf16=("bf16 operands, float32 GEMM, one bf16 "
                                  "rounding"))
    runs = {}
    for mode in ("fp32", "bf16"):
        if mode == "bf16":
            te = te.to(torch.bfloat16)
            oracle.emulate_bf16_matmul(True)
        runs[mode] = {k: features(torch, te, tok, v)
                      for k, v in prompts.items()}
    oracle.emulate_bf16_matmul(False)
    rel, cos, rows, secs = {}, {}, {}, {}
    for k in prompts:
        f32, b16 = runs["fp32"][k], runs["bf16"][k]
        assert np.array_equal(f32["ids"], b16["ids"])
        fx.add(f"ids_{k}", f32["ids"])
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
    print(f"[check2] wrote {fx.write()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
