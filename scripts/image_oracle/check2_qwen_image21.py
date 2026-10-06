#!/usr/bin/env python3
"""Check 2 oracle for Qwen-Image 2.1: the text encoder tap.

The pinned pipeline's encode_prompt runs Qwen3-VL-8B on each text prompt
in the pipeline's own template at batch 1, with no padding. Its features
are the output of the last decoder layer, before the final norm, less the
rows of the system turn. This script checks them equal, bit for bit, to
the output of decoder layer 35 read by a forward hook.

transformers does not read arch qwen3vl from a GGUF, so the script builds
the language model's state from the text GGUF itself: gguf-py dequantizes
each tensor to float32, whose arithmetic is ggml's, and the tensors go
into Qwen3VLForConditionalGeneration by name. The vision tower and the
output head are not loaded: a text prompt reaches neither. As a check of
the name map the script reads the model's next token for a short chat
turn, with the final norm and the output matrix of the GGUF.

The processor holds the tokenizer and the chat template of the official
pipeline folder and no image or video processor. The video processor
needs torchvision, which the oracle environment lacks.

For each prompt the fixture holds the token ids of the whole templated
prompt and the fp32 and bf16 taps of the rows that the pipeline keeps.
The manifest records the bf16 against fp32 gap of each prompt and the
tolerances: relative L2 at twice that gap, and a per-token cosine floor
of 1 - 4 (1 - c), where c is the oracle's own smallest per-token cosine
of that prompt, which bounds the angle at twice the oracle's own. The
starting value of 0.999 does not hold for this encoder: the row of each
<|im_start|> token differs by a fifth of its norm between the two runs.
The manifest names the row of each prompt's smallest cosine. The bf16
model is the float32 one rounded in place, and it runs after every
float32 prompt, so one copy of the weights is held at a time.

The bf16 run computes each linear product as a float32 GEMM over the
bfloat16 operands, rounded once to bfloat16, since torch's CPU bfloat16
GEMM is single threaded.

  python scripts/image_oracle/run.py --python <oracle python> --threads 16 \\
      check2_qwen_image21.py --encoder Qwen3-VL-8B-Instruct-Q4_K_M.gguf \\
      --config <official repo dir>
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

NAME = "check2_qwen_image21"
STARTING = {"rel_l2": 2e-2, "token_cosine_min": 0.999}
RULE = ("relative L2 = 2 x the oracle's own bf16 against fp32 relative L2 "
        "of the same prompt; per-token cosine at least 1 - 4 (1 - c), c the "
        "oracle's own smallest per-token cosine of the same prompt, and at "
        "most the starting value")
ROWS = 16384            # rows of a large tensor dequantized at a time
SANITY = "The capital of France is"

# GGUF tensor name to the name under model.language_model.
TOP = {"token_embd.weight": "embed_tokens.weight",
       "output_norm.weight": "norm.weight"}
LAYER = {"attn_q.weight": "self_attn.q_proj.weight",
         "attn_k.weight": "self_attn.k_proj.weight",
         "attn_v.weight": "self_attn.v_proj.weight",
         "attn_output.weight": "self_attn.o_proj.weight",
         "attn_q_norm.weight": "self_attn.q_norm.weight",
         "attn_k_norm.weight": "self_attn.k_norm.weight",
         "attn_norm.weight": "input_layernorm.weight",
         "ffn_norm.weight": "post_attention_layernorm.weight",
         "ffn_gate.weight": "mlp.gate_proj.weight",
         "ffn_up.weight": "mlp.up_proj.weight",
         "ffn_down.weight": "mlp.down_proj.weight"}
HEAD = "output.weight"


def hf_name(name: str) -> str | None:
    if name in TOP:
        return TOP[name]
    part = name.split(".", 2)
    if len(part) == 3 and part[0] == "blk" and part[2] in LAYER:
        return f"layers.{int(part[1])}.{LAYER[part[2]]}"
    return None


def dequantized(torch, tensor):
    """A GGUF tensor as a float32 torch tensor, a large one in row
    chunks so the transient stays small."""
    import gguf

    data = tensor.data
    out = torch.empty(tuple(int(d) for d in reversed(tensor.shape)),
                      dtype=torch.float32)
    rows = out.shape[0] if out.ndim > 1 else 1
    if out.ndim == 1:
        out.copy_(torch.from_numpy(np.array(gguf.quants.dequantize(
            data, tensor.tensor_type), dtype=np.float32)))
        return out
    for at in range(0, rows, ROWS):
        part = gguf.quants.dequantize(data[at:at + ROWS], tensor.tensor_type)
        out[at:at + ROWS] = torch.from_numpy(
            np.ascontiguousarray(part, dtype=np.float32))
    return out


def text_processor(config: Path):
    """The pipeline's processor: the tokenizer and the chat template of
    the official folder, with no image or video processor."""
    from transformers import AutoTokenizer, Qwen3VLProcessor

    folder = config / "processor"
    tok = AutoTokenizer.from_pretrained(str(folder), local_files_only=True)
    check = Qwen3VLProcessor.check_argument_for_proper_class
    Qwen3VLProcessor.check_argument_for_proper_class = (
        lambda self, name, arg: None if arg is None
        else check(self, name, arg))
    try:
        return Qwen3VLProcessor(
            image_processor=None, tokenizer=tok, video_processor=None,
            chat_template=(folder / "chat_template.jinja").read_text())
    finally:
        del Qwen3VLProcessor.check_argument_for_proper_class


def load_encoder(torch, a):
    """(model, GGUF reader, codec census). The model holds the language
    model in float32 on the CPU."""
    import gguf
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    cfg = AutoConfig.from_pretrained(str(a.config / "text_encoder"),
                                     local_files_only=True)
    with init_empty_weights():
        te = Qwen3VLForConditionalGeneration(cfg)
    te.model.visual = torch.nn.Module()
    te.lm_head = torch.nn.Identity()
    lm = te.model.language_model
    want = {n: tuple(p.shape) for n, p in lm.named_parameters()}
    reader = gguf.GGUFReader(str(a.encoder))
    census: dict[str, int] = {}
    seen, left = set(), []
    for t in reader.tensors:
        name = hf_name(t.name)
        if name is None:
            left.append(t.name)
            continue
        value = dequantized(torch, t)
        if tuple(value.shape) != want.get(name):
            raise SystemExit(f"error: {t.name} has shape "
                             f"{tuple(value.shape)}, and {name} wants "
                             f"{want.get(name)}")
        # The config names bfloat16 and the empty model takes it, so the
        # dtype is given with each tensor.
        set_module_tensor_to_device(lm, name, "cpu", value=value,
                                    dtype=torch.float32)
        seen.add(name)
        census[t.tensor_type.name] = census.get(t.tensor_type.name, 0) + 1
    if left != [HEAD] or seen != set(want):
        raise SystemExit(f"error: GGUF tensors left over {left}, model "
                         f"parameters not set {sorted(set(want) - seen)}")
    meta = [n for n, p in te.named_parameters() if p.is_meta]
    kinds = {p.dtype for p in te.parameters()} | {
        b.dtype for b in te.buffers() if b.is_floating_point()}
    if meta or kinds != {torch.float32}:
        raise SystemExit(f"error: parameters on no device {meta[:4]}, "
                         f"dtypes {kinds}")
    if len(lm.layers) != 36:
        raise SystemExit(f"error: {len(lm.layers)} layers, expected 36")
    return te.eval(), reader, census


def next_token(torch, te, reader, proc) -> dict:
    """The model's greedy next token for a short chat turn, through the
    final norm and the output matrix of the GGUF, read in row chunks."""
    import gguf

    text = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text":
                                       "Complete the sentence in one word. "
                                       + SANITY}]}],
        tokenize=False, add_generation_prompt=True)
    ids = proc.tokenizer(text, return_tensors="pt").input_ids
    lm = te.model.language_model
    last = lm(input_ids=ids).last_hidden_state[0, -1]
    head = next(t for t in reader.tensors if t.name == HEAD)
    best, at = None, -1
    for lo in range(0, int(head.shape[1]), ROWS):
        w = torch.from_numpy(np.ascontiguousarray(gguf.quants.dequantize(
            head.data[lo:lo + ROWS], head.tensor_type), dtype=np.float32))
        logits = w @ last
        top = int(logits.argmax())
        if best is None or float(logits[top]) > best:
            best, at = float(logits[top]), lo + top
    return {"prompt": text, "token_id": at,
            "token": proc.tokenizer.decode([at]), "logit": best}


class Tap:
    """Output of the last decoder layer."""

    def __init__(self, lm):
        self.out = None
        self._hook = lm.layers[-1].register_forward_hook(self._keep)

    def _keep(self, _mod, _args, out):
        self.out = out[0] if isinstance(out, tuple) else out

    def remove(self):
        self._hook.remove()


def encode(torch, pipe, prompt: str) -> dict:
    """encode_prompt with the layer hook, and the check that ties its
    output to the hooked tap."""
    tap = Tap(pipe.text_encoder.model.language_model)
    try:
        t0 = time.perf_counter()
        embeds, mask, pads = pipe.encode_prompt(prompt,
                                                device=torch.device("cpu"))
        secs = time.perf_counter() - t0
    finally:
        tap.remove()
    text = pipe.prompt_template_t2i.format(prompt or " ")
    ids = pipe.processor(text=[text], padding=True, padding_side="left",
                         return_tensors="pt").input_ids[0]
    if mask is not None or bool(pads.any()) or embeds.shape[0] != 1:
        raise SystemExit("error: a text prompt at batch 1 came back with a "
                         "mask or an image slot")
    drop = pipe._drop_idx
    if tap.out.shape[1] != len(ids) or embeds.shape[1] != len(ids) - drop:
        raise SystemExit(f"error: {tap.out.shape[1]} rows for {len(ids)} "
                         f"ids, {embeds.shape[1]} kept after {drop}")
    return {"text": text, "ids": ids.numpy().astype(np.int64),
            "tap": oracle.to_numpy(embeds[0]), "seconds": secs,
            "hook_equals_features": bool(torch.equal(tap.out[0, drop:],
                                                     embeds[0]))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--encoder", required=True, type=Path,
                    help="Qwen3-VL-8B text GGUF")
    ap.add_argument("--config", required=True, type=Path,
                    help="local copy of Qwen/Qwen-Image-2.1 (processor, "
                         "text_encoder and scheduler configs)")
    ap.add_argument("--config-revision",
                    default="790c92633540aa0cb11d9abf19eb46d861714758",
                    help="revision of the --config copy, for the manifest")
    ap.add_argument("--prompts", nargs="+", default=list(PROMPTS),
                    choices=list(PROMPTS))
    ap.add_argument("--name", default=NAME)
    ap.add_argument("--out", type=Path,
                    help="fixture directory (default $GMLX_IMAGE_FIXTURES)")
    a = ap.parse_args(argv)
    torch = oracle.setup()

    from diffusers import (FlowMatchEulerDiscreteScheduler,
                           QwenImage21Pipeline)

    proc = text_processor(a.config)
    t0 = time.perf_counter()
    te, reader, census = load_encoder(torch, a)
    load_s = time.perf_counter() - t0
    print(f"[check2] loaded in {load_s:.0f} s, {census}", flush=True)
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(a.config), subfolder="scheduler", local_files_only=True)
    pipe = QwenImage21Pipeline(scheduler=sched, vae=None, text_encoder=te,
                               processor=proc, transformer=None)
    sanity = next_token(torch, te, reader, proc)
    print(f"[check2] next token after {SANITY!r}: {sanity['token']!r}",
          flush=True)
    if "Paris" not in sanity["token"]:
        raise SystemExit(f"error: the model continues {SANITY!r} with "
                         f"{sanity['token']!r}")

    fx = fixtures.FixtureWriter(
        a.name, check=2, family="qwen_image21",
        script=f"scripts/image_oracle/{Path(__file__).name}",
        argv=[Path(__file__).name, "--encoder", a.encoder.name, "--config",
              a.config.name])
    fx.add_file("encoder", a.encoder)
    for role, rel in (("tokenizer", "processor/tokenizer.json"),
                      ("chat_template", "processor/chat_template.jinja"),
                      ("text_encoder_config", "text_encoder/config.json")):
        fx.add_file(role, a.config / rel)
    prompts = {k: PROMPTS[k] for k in a.prompts}
    runs = {"fp32": {}, "bf16": {}}
    for key, prompt in prompts.items():
        runs["fp32"][key] = encode(torch, pipe, prompt)
        print(f"[check2] fp32 {key}: {len(runs['fp32'][key]['ids'])} ids, "
              f"{runs['fp32'][key]['seconds']:.1f} s", flush=True)
    t0 = time.perf_counter()
    rotary = te.model.language_model.rotary_emb
    inv_freq = rotary.inv_freq.clone()
    te.to(torch.bfloat16)
    # The cast rounds the rotary frequencies too. They stay float32, so
    # the gap measures the arithmetic and not the position table.
    rotary.inv_freq = inv_freq
    cast_s = time.perf_counter() - t0
    kinds = {p.dtype for p in te.parameters()}
    if kinds != {torch.bfloat16}:
        raise SystemExit(f"error: the bf16 model holds {kinds}")
    oracle.emulate_bf16_matmul(True)
    matmul = oracle.matmul_mode()
    for key, prompt in prompts.items():
        runs["bf16"][key] = encode(torch, pipe, prompt)
        print(f"[check2] bf16 {key}: {runs['bf16'][key]['seconds']:.1f} s",
              flush=True)
    oracle.emulate_bf16_matmul(False)

    gaps, checks, rendered, lengths, kept, secs, tol_rel = ({}, {}, {}, {},
                                                            {}, {}, {})
    tol_cos = {}
    for key in prompts:
        r32, r16 = runs["fp32"][key], runs["bf16"][key]
        if not np.array_equal(r16["ids"], r32["ids"]):
            raise SystemExit(f"error: {key}: ids differ between modes")
        fx.add(f"ids_{key}", r32["ids"])
        fx.add(f"tap_{key}", r32["tap"])
        fx.add(f"tap_bf16_{key}", r16["tap"])
        gap = fixtures.compare(r16["tap"], r32["tap"])
        cos = token_cosine(r16["tap"], r32["tap"])
        norms = np.linalg.norm(r32["tap"].astype(np.float64), axis=-1)
        rel = np.linalg.norm(r16["tap"].astype(np.float64)
                             - r32["tap"], axis=-1) / norms
        row = int(cos.argmin())
        gap["token_cosine_min"] = float(cos.min())
        # The row of the smallest cosine, by its place in the tap and by
        # the token it belongs to, and the spread of the row errors.
        gap["token_cosine_min_row"] = row
        gap["token_cosine_min_token_id"] = int(
            r32["ids"][int(pipe._drop_idx) + row])
        gap["row_rel_l2"] = {"min": float(rel.min()),
                             "median": float(np.median(rel)),
                             "max": float(rel.max())}
        gap["row_norm"] = {"min": float(norms.min()),
                           "median": float(np.median(norms)),
                           "max": float(norms.max())}
        tol_cos[key] = min(STARTING["token_cosine_min"],
                           1.0 - 4.0 * (1.0 - gap["token_cosine_min"]))
        gaps[key] = gap
        checks[key] = {m: r["hook_equals_features"]
                       for m, r in (("fp32", r32), ("bf16", r16))}
        rendered[key] = r32["text"]
        lengths[key] = len(r32["ids"])
        kept[key] = int(r32["tap"].shape[0])
        secs[key] = {"fp32_s": round(r32["seconds"], 1),
                     "bf16_s": round(r16["seconds"], 1)}
        tol_rel[key] = 2 * gap["rel_l2"]
        print(f"[check2] {key}: {lengths[key]} ids, {kept[key]} rows, bf16 "
              f"vs fp32 rel L2 {gap['rel_l2']:.3e}, token cos min "
              f"{gap['token_cosine_min']:.6f}", flush=True)
        if not all(checks[key].values()):
            raise SystemExit(f"error: {key}: tap check failed: "
                             f"{checks[key]}")

    fx.inputs.update(
        prompts=prompts, rendered=rendered, lengths=lengths, rows=kept,
        drop_index=int(pipe._drop_idx),
        tap="output of decoder layer 35 (all 36 layers run), before the "
        "final norm, less the first drop_index rows",
        template="QwenImage21Pipeline.prompt_template_t2i; an empty prompt "
        "is one space",
        padding="none: batch 1",
        config_revision=a.config_revision,
        encoder="Qwen3VLForConditionalGeneration with the language model "
        "set by name from the text GGUF, gguf-py float32 dequantization; "
        "the vision tower and the output head are not loaded; the bf16 "
        "model is the float32 one rounded in place, with its rotary "
        "frequencies left in float32",
        processor="Qwen3VLProcessor with the tokenizer and chat template of "
        "the official processor folder and no image or video processor",
        gguf_codecs=census)
    fx.results.update(
        oracle={"device": "cpu", "threads": torch.get_num_threads(),
                "bf16_matmul": matmul},
        name_map_check=sanity, tap_checks=checks, bf16_vs_fp32=gaps,
        seconds={"load_s": round(load_s, 1), "cast_s": round(cast_s, 1),
                 **secs})
    fx.tolerances.update(rule=RULE, starting=STARTING, rel_l2=tol_rel,
                         token_cosine_min=tol_cos)
    fx.note("The pipeline encodes one prompt at batch 1 with no padding and "
            "no mask, as gmlx encodes a prompt.")
    path = fx.write(a.out)
    print(f"[check2] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
