#!/usr/bin/env python3
"""Check 2 oracle for FLUX.3 klein: the text encoder tap.

BFL's TextEncoder.encode (src/flux3/text_encoder.py of --reference) runs
Qwen3-VL-4B on the prompt in the chat template of one user turn with the
generation prompt, right-padded with the pad token to a multiple of 80
tokens and at most 4096, with the attention mask, and with the position
of each real token on all four rotary axes and 1 on each pad. It stacks
hidden_states 4, 8, ..., 32 of every row, the pad rows kept, to (L, 8 *
2560), layer by layer, and returns them in bf16 with the text ids (0, 0,
0, row). This script calls that method itself, on an instance whose model
is the GGUF's and whose processor is the official folder's tokenizer and
chat template, and keeps the forward's hidden states to stack them again
without the final bf16 cast. The stack rounded to bf16 must equal the
method's return value bit for bit.

transformers does not read arch qwen3vl from a GGUF, so the script builds
the language model's state from the text GGUF, as check2_krea2.py does:
gguf-py dequantizes each tensor to float32, and the tensors go into
Qwen3VLForConditionalGeneration by name, with the config of the official
text_encoder folder. The vision tower is not loaded: a text prompt does
not reach it. The processor has no image or video processor; the video
processor needs torchvision, which the oracle environment lacks.

For each prompt the fixture holds the ids, mask and positions the method
fed the model, its text ids, and the fp32 and bf16 features of every row.
The manifest records the bf16 against fp32 gap of each prompt, the
largest relative L2 of a pad row of the fp32 run to the first, and the
tolerances: relative L2
at twice that gap, and a per-token cosine floor of 1 - 4 (1 - c), c the
oracle's own smallest per-token cosine of the prompt. A prompt of
--ids-only keeps its ids and mask alone, without a forward: the prompt
past the 4096-token cap.

  python scripts/image_oracle/run.py --python <oracle python> --threads 16 \\
      check2_flux3_klein.py --reference <flux-3-image-klein>/src \\
      --encoder Qwen3-VL-4B-Instruct-Q8_0.gguf \\
      --text-encoder <official repo>/text_encoder
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import fixtures
import oracle
from check2_qwen_image21 import dequantized, hf_name
from check2_z_image import _LONG, PROMPTS, token_cosine

NAME = "check2_flux3_klein"
SCRIPT = "scripts/image_oracle/check2_flux3_klein.py"
RULE = ("relative L2 = 2 x the oracle's own bf16 against fp32 relative L2 "
        "of the same prompt; per-token cosine at least 1 - 4 (1 - c), c the "
        "oracle's own smallest per-token cosine of the same prompt")
# The 4B widths, from the GGUF header.
WIDTHS = {"hidden_size": "embedding_length",
          "intermediate_size": "feed_forward_length",
          "num_hidden_layers": "block_count",
          "num_attention_heads": "attention.head_count",
          "num_key_value_heads": "attention.head_count_kv",
          "head_dim": "attention.key_length"}


def reference(path: Path):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(path))
    from flux3 import text_encoder

    try:
        commit = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return text_encoder, commit


def load_encoder(torch, encoder: Path, folder: Path):
    import gguf
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    cfg = AutoConfig.from_pretrained(str(folder), local_files_only=True)
    reader = gguf.GGUFReader(str(encoder))
    kv = {f.name: f for f in reader.fields.values()}
    for k, key in WIDTHS.items():
        f = kv[f"qwen3vl.{key}"]
        got = f.parts[f.data[0]].tolist()[0]
        if got != getattr(cfg.text_config, k):
            raise SystemExit(f"error: {k} is {got} in the GGUF, "
                             f"{getattr(cfg.text_config, k)} in the config")
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


def processor(folder: Path):
    """The official folder's tokenizer and chat template in the
    reference's processor class, with no image or video processor."""
    from transformers import AutoTokenizer, Qwen3VLProcessor

    tok = AutoTokenizer.from_pretrained(str(folder), local_files_only=True)
    template = json.loads((folder / "chat_template.json").read_text())
    check = Qwen3VLProcessor.check_argument_for_proper_class
    Qwen3VLProcessor.check_argument_for_proper_class = (
        lambda self, name, arg: None if arg is None
        else check(self, name, arg))
    try:
        return Qwen3VLProcessor(
            image_processor=None, tokenizer=tok, video_processor=None,
            chat_template=template["chat_template"])
    finally:
        del Qwen3VLProcessor.check_argument_for_proper_class


class Keep:
    """The model, keeping the inputs and output of its last call."""

    def __init__(self, model):
        self.model = model
        self.last = None

    def __call__(self, **kwargs):
        out = self.model(**kwargs)
        self.last = (kwargs, out)
        return out


def encoder(text_encoder, model, proc):
    """An instance of the reference's TextEncoder on ``model`` and
    ``proc``: its __init__ loads from a folder, so it is set by hand as
    __init__ sets it."""
    enc = object.__new__(text_encoder.TextEncoder)
    enc.model = Keep(model)
    enc.processor = proc
    enc.processor.tokenizer.padding_side = "right"
    enc.device = "cpu"
    return enc


def features(torch, text_encoder, enc, prompt: str) -> dict:
    t0 = time.perf_counter()
    tokens, ids = enc.encode(prompt)
    secs = time.perf_counter() - t0
    kw, out = enc.model.last
    length = kw["input_ids"].shape[1]
    stacked = torch.stack([out.hidden_states[k]
                           for k in text_encoder.OUTPUT_LAYERS], dim=1)
    f = stacked.permute(0, 2, 1, 3).reshape(length, -1)
    if not torch.equal(f.to(torch.bfloat16), tokens):
        raise SystemExit("error: the stack differs from encode's own")
    mask = kw["attention_mask"][0]
    real = int(mask.sum())
    f32 = oracle.to_numpy(f)
    pads = f32[real:].astype(np.float64)
    spread = 0.0
    if len(pads) > 1:
        spread = float((np.linalg.norm(pads - pads[0], axis=1)
                        / np.linalg.norm(pads[0])).max())
    return {"ids": kw["input_ids"][0].numpy().astype(np.int64),
            "mask": mask.numpy().astype(np.int64),
            "positions": kw["position_ids"].numpy().astype(np.int64),
            "text_ids": ids.numpy().astype(np.int64),
            "features": f32, "real": real,
            "pad_spread": spread,
            "seconds": secs}


def ids_only(text_encoder, enc, prompt: str) -> dict:
    """The ids and mask encode would feed the model, without a forward:
    its own steps up to the call."""
    import math

    proc = enc.processor
    text = proc.apply_chat_template([{"role": "user", "content": prompt}],
                                    tokenize=False,
                                    add_generation_prompt=True)
    tok = proc.tokenizer
    real = tok(text, return_tensors="pt", truncation=True,
               max_length=text_encoder.MAX_LENGTH)["input_ids"].shape[1]
    length = min(math.ceil(real / text_encoder.PAD_MULTIPLE)
                 * text_encoder.PAD_MULTIPLE, text_encoder.MAX_LENGTH)
    inputs = tok(text, return_tensors="pt", padding="max_length",
                 truncation=True, max_length=length)
    return {"ids": inputs["input_ids"][0].numpy().astype(np.int64),
            "mask": inputs["attention_mask"][0].numpy().astype(np.int64)}


def eighty(proc) -> str:
    """The longest prefix of the long prompt, cut at a character, whose
    templated prompt is exactly 80 tokens, so its features have no pad
    row."""
    for n in range(len(_LONG), 0, -1):
        p = _LONG[:n].rstrip()
        text = proc.apply_chat_template([{"role": "user", "content": p}],
                                        tokenize=False,
                                        add_generation_prompt=True)
        if len(proc.tokenizer(text)["input_ids"]) == 80:
            return p
    raise SystemExit("error: no prefix of the long prompt is 80 tokens")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reference", required=True, type=Path,
                    help="src directory of BFL's flux-3-image-klein")
    ap.add_argument("--encoder", required=True, type=Path,
                    help="Qwen3-VL-4B text GGUF")
    ap.add_argument("--text-encoder", required=True, type=Path,
                    help="the official repo's text_encoder folder")
    ap.add_argument("--name", default=NAME)
    a = ap.parse_args(argv)
    torch = oracle.setup()
    text_encoder, commit = reference(a.reference.expanduser())
    folder = a.text_encoder.expanduser()
    proc = processor(folder)
    prompts = dict(PROMPTS)
    prompts["eighty"] = eighty(proc)
    capped = {"cap": " ".join([_LONG] * 30)}
    t0 = time.perf_counter()
    te, census = load_encoder(torch, a.encoder, folder)
    print(f"[check2] loaded {a.encoder.name} in "
          f"{time.perf_counter() - t0:.1f}s, codecs {census}", flush=True)
    fx = fixtures.FixtureWriter(a.name, check=2, family="flux3_klein",
                                script=SCRIPT)
    fx.add_file("encoder", a.encoder)
    for f in ("config.json", "tokenizer.json", "tokenizer_config.json",
              "chat_template.json"):
        fx.add_file(f"text_encoder/{f}", folder / f)
    fx.inputs.update(prompts=prompts, ids_only=capped,
                     tap_layers=list(text_encoder.OUTPUT_LAYERS),
                     pad_multiple=text_encoder.PAD_MULTIPLE,
                     max_length=text_encoder.MAX_LENGTH,
                     reference_commit=commit, gguf_codecs=census,
                     matmul_bf16=("bf16 operands, float32 GEMM, one bf16 "
                                  "rounding"))
    enc = encoder(text_encoder, te, proc)
    for k, p in capped.items():
        got = ids_only(text_encoder, enc, p)
        fx.add(f"ids_{k}", got["ids"])
        fx.add(f"mask_{k}", got["mask"])
    runs = {}
    for mode in ("fp32", "bf16"):
        if mode == "bf16":
            te = te.to(torch.bfloat16)
            enc = encoder(text_encoder, te, proc)
            oracle.emulate_bf16_matmul(True)
        runs[mode] = {k: features(torch, text_encoder, enc, v)
                      for k, v in prompts.items()}
    oracle.emulate_bf16_matmul(False)
    rel, cos, rows, secs, pads = {}, {}, {}, {}, {}
    for k in prompts:
        f32, b16 = runs["fp32"][k], runs["bf16"][k]
        for key in ("ids", "mask", "positions", "text_ids"):
            assert np.array_equal(f32[key], b16[key]), (k, key)
            fx.add(f"{key}_{k}", f32[key])
        fx.add(f"features_{k}", f32["features"])
        fx.add(f"features_{k}_bf16", b16["features"])
        gap = fixtures.rel_l2(b16["features"], f32["features"])
        c = token_cosine(b16["features"], f32["features"])
        rel[k] = 2 * gap
        cos[k] = float(1 - 4 * (1 - c.min()))
        rows[k] = {"real": f32["real"], "rows": int(f32["ids"].shape[0])}
        pads[k] = f32["pad_spread"]
        secs[k] = {m: round(runs[m][k]["seconds"], 1) for m in runs}
        fx.results[f"bf16_vs_fp32_{k}"] = {
            "rel_l2": gap, "token_cosine_min": float(c.min()),
            "worst_token": int(c.argmin())}
        print(f"[check2] {k}: {rows[k]}, pad rows within {pads[k]:.1e} of "
              f"the first, bf16 vs "
              f"fp32 rel L2 {gap:.3e}, min token cosine {c.min():.5f}",
              flush=True)
    fx.inputs["rows"] = rows
    fx.results["pad_rows_rel_l2_to_the_first"] = pads
    fx.results["seconds"] = secs
    fx.tolerances.update(rel_l2=rel, token_cosine_min=cos, rule=RULE)
    print(f"[check2] wrote {fx.write()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
