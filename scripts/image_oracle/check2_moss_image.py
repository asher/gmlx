#!/usr/bin/env python3
"""Check 2 oracle for MOSS-Image: the Qwen3.5-9B text encoder read after
its final norm.

MOSSImagePipeline.encode_prompt renders each prompt with the tokenizer's
chat template for one user turn, with the generation prompt and thinking
on, tokenizes it with truncation at 4096 tokens, runs it alone through
the text model and keeps last_hidden_state: (L, 4096) per prompt, with no
pad rows. This script runs the same steps on the model that transformers
builds from the encoder GGUF itself (arch qwen35), on the CPU in float32,
with every quantized tensor dequantized at load by gguf-py in float32,
which is ggml's arithmetic (``exact_reads``). Before the run, one matrix
of each codec must agree bit for bit with the parameter that holds it, so
no transform of the load has changed it.

The bf16 run rounds that model to bfloat16 once, as the release loads its
encoder in bfloat16, and each linear product then runs as a float32 GEMM
over the bfloat16 operands with one rounding (oracle.emulate_bf16_matmul).
The tokenizer is the release's, from --tokenizer, loaded as
Qwen2Tokenizer, the class the release's model_index.json names. The
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

--layers N keeps the first N layers, for a memory ladder.
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


def exact_reads() -> None:
    """Every block tensor of the GGUF dequantized by gguf-py in float32,
    which is ggml's arithmetic, as transformers reads it. transformers
    5.18 without the ``kernels`` package falls back to a dense load whose
    unpack step skips this file's tensors and leaves their bytes in the
    parameters; with the values unpacked here it passes them through."""
    import gguf
    import torch
    from transformers.integrations.gguf import gguf_conversion_mapping as cm
    from transformers.integrations.gguf import reader as rd

    native = rd.LazyGgufTensor.__getitem__
    unpack = cm.Dequantize.convert

    def getitem(self, key):
        if self.ggml_type in rd._TORCH_DTYPE:
            return native(self, key)
        raw = np.copy(self.data).reshape(self.shape)
        values = gguf.quants.dequantize(
            raw, gguf.GGMLQuantizationType(self.ggml_type))
        return torch.from_numpy(np.ascontiguousarray(values, np.float32))

    def convert(self, input_dict, *args, **kwargs):
        if all((t[0] if isinstance(t, list) else t).dtype != torch.uint8
               for t in input_dict.values()):
            return input_dict
        return unpack(self, input_dict, *args, **kwargs)

    rd.LazyGgufTensor.__getitem__ = getitem
    cm.Dequantize.convert = convert


def load_encoder(torch, path: Path, layers: int | None = None):
    """The text model of the GGUF on the CPU in float32, every quantized
    tensor dequantized at load. ``layers`` keeps the first N layers, for a
    memory ladder."""
    from transformers import AutoConfig, AutoModel

    exact_reads()
    kw: dict = dict(gguf_file=path.name, local_files_only=True)
    if layers is not None:
        cfg = AutoConfig.from_pretrained(str(path.parent), **kw)
        kw.update(num_hidden_layers=layers,
                  layer_types=list(cfg.layer_types)[:layers])
    model = AutoModel.from_pretrained(str(path.parent), dtype=torch.float32,
                                      device_map="cpu", **kw).eval()
    bad = {(str(p.device), p.dtype) for p in model.parameters()} - {
        ("cpu", torch.float32)}
    if bad:
        raise SystemExit(f"error: the encoder loaded as {sorted(map(str, bad))}"
                         ", not float32 on the CPU")
    return model


# GGUF tensors the check tries, in order. A parameter holds the first three
# as they are; the others may come with their heads reordered.
_KINDS = ("ffn_down", "ffn_gate", "ffn_up", "attn_output", "ssm_out",
          "ssm_alpha", "ssm_beta", "attn_gate", "attn_qkv")


def gguf_codecs(path: Path) -> dict:
    import gguf

    out: dict = {}
    for t in gguf.GGUFReader(str(path)).tensors:
        out[t.tensor_type.name] = out.get(t.tensor_type.name, 0) + 1
    return out


def check_dequant(model, path: Path, layers: int) -> dict:
    """For each quantized codec, one matrix of the first ``layers`` blocks
    dequantized by gguf-py in float32 must be held by a parameter of the
    same size: bit for bit, or, for a matrix the load reorders by head,
    as the same multiset of float32 values. Raises when a codec of those
    blocks has no such parameter. Returns the codec to the GGUF and
    parameter names and how they matched."""
    import gguf

    params = [(n, p.detach().numpy()) for n, p in model.named_parameters()]
    tensors = {t.name: t for t in gguf.GGUFReader(str(path)).tensors}
    float_types = {"F32", "F16", "BF16"}
    wanted = {t.tensor_type.name for n, t in tensors.items()
              if n.startswith(tuple(f"blk.{i}." for i in range(layers)))
              and t.tensor_type.name not in float_types}
    done: dict = {}
    for kind in _KINDS:
        for i in range(layers):
            t = tensors.get(f"blk.{i}.{kind}.weight")
            if t is None or t.tensor_type.name not in wanted - set(done):
                continue
            want = np.asarray(gguf.quants.dequantize(t.data, t.tensor_type),
                              np.float32)
            same = [n for n, a in params if a.shape == want.shape
                    and a.tobytes() == want.tobytes()]
            if same:
                done[t.tensor_type.name] = [t.name, same[0], "bitwise"]
                continue
            flat = np.sort(want, axis=None)
            same = [n for n, a in params if a.size == want.size
                    and np.sort(a, axis=None).tobytes() == flat.tobytes()]
            if same:
                done[t.tensor_type.name] = [t.name, same[0], "reordered"]
    missing = wanted - set(done)
    if missing:
        raise SystemExit(f"error: no parameter holds a {sorted(missing)} "
                         "matrix as gguf-py dequantizes it")
    return done


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
    ap.add_argument("--layers", type=int,
                    help="keep the first N layers, for a memory ladder; the "
                         "fixture name takes a _<N>l suffix")
    ap.add_argument("--prompts", nargs="*",
                    default=[*PROMPTS, *MARKS])
    ap.add_argument("--prompt", nargs=2, action="append", default=[],
                    metavar=("NAME", "TEXT"),
                    help="one more prompt by name, such as the prompt of "
                         "check 8's trajectory")
    a = ap.parse_args(argv)
    if a.layers is not None:
        a.name = f"{a.name}_{a.layers}l"
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
    t0 = time.perf_counter()
    model = load_encoder(torch, a.encoder, a.layers)
    layers = model.config.num_hidden_layers
    info = {"load_s": round(time.perf_counter() - t0, 1), "layers": layers,
            "gguf_codecs": gguf_codecs(a.encoder),
            "dequant_checked": check_dequant(model, a.encoder, layers)}
    print(f"[check2] loaded {layers} layers in {info['load_s']}s, dequant "
          f"checked {sorted(info['dequant_checked'])}", flush=True)
    runs = {}
    for mode in ("fp32", "bf16"):
        if mode == "bf16":
            # Parameters only: from_pretrained in bfloat16 keeps the
            # float32 RoPE buffers.
            for p in model.parameters():
                p.data = p.data.to(torch.bfloat16)
        oracle.emulate_bf16_matmul(mode == "bf16")
        runs[mode] = {k: features(torch, model, v) for k, v in ids.items()}
        print(f"[check2] {mode}: "
              f"{ {k: round(r['seconds'], 1) for k, r in runs[mode].items()} }",
              flush=True)
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
    fx.note("The bf16 model is the float32 model rounded to bfloat16 once, "
            "as the release loads its encoder in bfloat16.")
    print(f"[check2] wrote {fx.write()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
