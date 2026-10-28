#!/usr/bin/env python3
"""Check 2 oracle for MiniMax-H3: the text encoder tap.

H3 conditions on Qwen3-VL-32B-Instruct after decoder layer 50 of 64,
before the final norm, on the prompt's raw token ids. This script takes
each prompt's ids from ComfyUI's MiniMaxH3Tokenizer and runs transformers'
Qwen3-VL text decoder layers on the stock text GGUF with one layer
resident at a time: gguf-py dequantizes the layer's tensors to float32,
whose arithmetic is ggml's, both passes run it on their own hidden states,
and the next layer replaces it. Layers 1 to 50 hold about 25B parameters,
101 GB in float32, so the model is never loaded whole.

Each pass starts from the token embedding rows of the prompt's ids, at
batch 1 with no padding, through transformers' SDPA attention, causal
with no mask. The fp32 pass runs float32 throughout. The bf16 pass holds
each layer in bfloat16, takes the rotary angles in float32 and their cos
and sin in bfloat16, and runs each linear product as a float32 GEMM over
bfloat16 operands, rounded once to bfloat16 (oracle.emulate_bf16_matmul).

As a check of the name map a short prompt also runs layers 51 to 64, the
final norm and the output matrix in float32, and its next token must be
" Paris".

For each prompt the fixture holds ComfyUI's ids and the fp32 and bf16
taps. The manifest records each prompt's bf16 against fp32 gap after
every layer, and the tolerances: relative L2 at twice the gap at the tap,
over every row and over the rows after the first, and a per-token cosine
floor of 1 - 4 (1 - c), c the smallest per-token cosine of the two
passes, at most 0.999. The first row holds nearly all of the tap's norm,
so the gate over every row reads little else.

  python scripts/video_oracle/run.py --python <comfyui venv python> \\
      --threads 16 check2_h3.py --comfy-dir <ComfyUI checkout> \\
      --encoder Qwen3-VL-32B-Instruct-Q4_K_M.gguf
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from collections import Counter
from pathlib import Path

import numpy as np

import fixtures
import oracle

ORACLE_NEEDS = ("torch", "transformers", "gguf", "einops", "comfy_kitchen")

NAME = "check2_h3"
FAMILY = "h3"
SCRIPT = "scripts/video_oracle/check2_h3.py"
TAP = 50                # decoder layers run before the tap
STARTING = {"rel_l2": 2e-2, "token_cosine_min": 0.999}
RULE = ("relative L2 = 2 x the oracle's own bf16 against fp32 relative L2 "
        "of the same prompt, over every row and over the rows after the "
        "first; per-token cosine at least 1 - 4 (1 - c), c the oracle's own "
        "smallest per-token cosine of the same prompt, and at most the "
        "starting value")
ROWS = 16384            # rows of a large tensor dequantized at a time
SANITY = "The capital of France is"
DIALOGUE = ("A man in a grey raincoat stops under a streetlight and says "
            "<d>[English] We should have left an hour ago.</d> The woman "
            "beside him laughs and answers <d>[English] Then we run.</d> "
            "Rain hisses on the pavement as they hurry off.")
# Every token MiniMax's tokenizer adds, two of them side by side.
MARKUP = ("A street singer tunes a guitar under a bridge. <|caption_start|>"
          "Warm dusk light, a slow push in.<|caption_end|> She sings "
          "<|lyrics_start|><d>[English] Hold on to the light.</d>"
          "<|lyrics_end|> and stops.<|cutoff|>")

# GGUF tensor names of a decoder layer, to the names under the
# transformers layer.
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


def _image_script(name: str):
    """An image oracle script loaded by path for its constants and
    helpers. Its ``fixtures`` and ``oracle`` imports bind this
    directory's modules, which have the same functions."""
    path = Path(__file__).resolve().parent.parent / "image_oracle" / name
    spec = importlib.util.spec_from_file_location(f"_img_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"error: cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_z = _image_script("check2_z_image.py")
PROMPTS = {"plain": _z.PROMPTS["plain"], "empty": _z.PROMPTS["empty"],
           "non_ascii": _z.PROMPTS["non_ascii"], "long": _z.PROMPTS["long"],
           "dialogue": DIALOGUE, "markup": MARKUP}
token_cosine = _z.token_cosine


def dequantized(torch, tensor):
    """A GGUF tensor as a float32 torch tensor, a large one in row chunks
    so the transient stays small."""
    import gguf

    data = tensor.data
    out = torch.empty(tuple(int(d) for d in reversed(tensor.shape)),
                      dtype=torch.float32)
    if out.ndim == 1:
        out.copy_(torch.from_numpy(np.array(gguf.quants.dequantize(
            data, tensor.tensor_type), dtype=np.float32)))
        return out
    for at in range(0, out.shape[0], ROWS):
        part = gguf.quants.dequantize(data[at:at + ROWS], tensor.tensor_type)
        out[at:at + ROWS] = torch.from_numpy(
            np.ascontiguousarray(part, dtype=np.float32))
    return out


def rows_of(torch, tensor, ids):
    """Rows ``ids`` of a GGUF matrix, float32."""
    import gguf

    part = gguf.quants.dequantize(tensor.data[np.asarray(ids)],
                                  tensor.tensor_type)
    return torch.from_numpy(np.ascontiguousarray(part, dtype=np.float32))


def text_config(kv: dict):
    """transformers' text config from the GGUF header's values."""
    from transformers import Qwen3VLTextConfig

    def get(key):
        return kv[f"qwen3vl.{key}"].contents()

    sections = [int(s) for s in get("rope.dimension_sections")][:3]
    cfg = Qwen3VLTextConfig(
        hidden_size=int(get("embedding_length")),
        intermediate_size=int(get("feed_forward_length")),
        num_hidden_layers=1,
        num_attention_heads=int(get("attention.head_count")),
        num_key_value_heads=int(get("attention.head_count_kv")),
        head_dim=int(get("attention.key_length")),
        rms_norm_eps=float(get("attention.layer_norm_rms_epsilon")),
        rope_parameters={"rope_type": "default",
                         "rope_theta": float(get("rope.freq_base")),
                         "mrope_section": sections,
                         "mrope_interleaved": True},
        max_position_embeddings=int(get("context_length")),
        attention_bias=False)
    cfg._attn_implementation = "sdpa"
    return cfg, int(get("block_count"))


def comfy_ids(prompts: dict) -> dict:
    """Each prompt's ids from ComfyUI's H3 tokenizer, and the decoder of
    its Qwen tokenizer."""
    from comfy.text_encoders.minimax import MiniMaxH3Tokenizer

    tok = MiniMaxH3Tokenizer()
    out = {}
    for key, text in prompts.items():
        entries = tok.tokenize_with_weights(text)["qwen3vl_32b"]
        if len(entries) != 1:
            raise SystemExit(f"error: {key}: {len(entries)} token batches")
        out[key] = np.asarray([int(e[0]) for e in entries[0]], np.int32)
    return out, tok.qwen3vl_32b.tokenizer


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--comfy-dir", type=Path, required=True,
                    help="ComfyUI checkout (its H3 tokenizer)")
    ap.add_argument("--encoder", type=Path, required=True,
                    help="stock Qwen3-VL-32B-Instruct text GGUF")
    ap.add_argument("--prompts", nargs="+", default=list(PROMPTS),
                    choices=list(PROMPTS))
    ap.add_argument("--name", default=NAME)
    ap.add_argument("--out", type=Path,
                    help="fixture directory (default $GMLX_VIDEO_FIXTURES)")
    a = ap.parse_args(argv)
    torch = oracle.setup()
    commit = oracle.start_comfy(a.comfy_dir)
    import gguf
    from transformers.models.qwen3_vl import modeling_qwen3_vl as hf

    prompts = {k: PROMPTS[k] for k in a.prompts}
    ids, qwen_tok = comfy_ids({**prompts, "sanity": SANITY})
    reader = gguf.GGUFReader(str(a.encoder))
    tensors = {t.name: t for t in reader.tensors}
    cfg, depth = text_config(reader.fields)
    census = dict(Counter(t.tensor_type.name for t in reader.tensors))
    print(f"[check2] {depth} layers, hidden {cfg.hidden_size}, {census}",
          flush=True)

    layer32 = hf.Qwen3VLTextDecoderLayer(cfg, 0).eval()
    layer16 = hf.Qwen3VLTextDecoderLayer(cfg, 0).eval().to(torch.bfloat16)
    rotary = hf.Qwen3VLTextRotaryEmbedding(cfg)
    embed = tensors["token_embd.weight"]
    h32, h16, pe32, pe16 = {}, {}, {}, {}
    with torch.no_grad():
        for key, row in ids.items():
            x = rows_of(torch, embed, row.tolist())[None]
            pos = torch.arange(x.shape[1]).view(1, 1, -1).expand(3, 1, -1)
            h32[key] = x
            pe32[key] = rotary(x, pos)
            if key != "sanity":
                h16[key] = x.to(torch.bfloat16)
                pe16[key] = rotary(h16[key], pos)
    gaps = {k: [] for k in prompts}
    t0 = time.perf_counter()
    taps32, taps16 = {}, {}
    for i in range(depth):
        state = {LAYER[s]: dequantized(torch, tensors[f"blk.{i}.{s}"])
                 for s in LAYER}
        layer32.load_state_dict(state, strict=True)
        keys = ["sanity"] if i >= TAP else list(ids)
        with torch.no_grad():
            for key in keys:
                h32[key] = layer32(h32[key], position_embeddings=pe32[key],
                                   attention_mask=None)
            if i < TAP:
                layer16.load_state_dict(
                    {k: v.to(torch.bfloat16) for k, v in state.items()},
                    strict=True)
                oracle.emulate_bf16_matmul(True)
                for key in prompts:
                    h16[key] = layer16(h16[key],
                                       position_embeddings=pe16[key],
                                       attention_mask=None)
                oracle.emulate_bf16_matmul(False)
                for key in prompts:
                    gaps[key].append(fixtures.rel_l2(
                        h16[key][0].float().numpy(), h32[key][0].numpy()))
        del state
        if i + 1 == TAP:
            taps32 = {k: h32[k][0].numpy().copy() for k in prompts}
            taps16 = {k: h16[k][0].float().numpy() for k in prompts}
        print(f"[check2] layer {i + 1} of {depth}, "
              f"{time.perf_counter() - t0:.0f} s", flush=True)
    layer_s = time.perf_counter() - t0

    with torch.no_grad():
        eps = cfg.rms_norm_eps
        last = h32["sanity"][0, -1]
        norm = dequantized(torch, tensors["output_norm.weight"])
        v = last * torch.rsqrt(last.pow(2).mean() + eps) * norm
        head = tensors["output.weight"]
        best, at = -float("inf"), -1
        for lo in range(0, int(head.shape[1]), ROWS):
            w = rows_of(torch, head, range(lo, min(lo + ROWS,
                                                   int(head.shape[1]))))
            logits = w @ v
            j = int(logits.argmax())
            if float(logits[j]) > best:
                best, at = float(logits[j]), lo + j
    token = qwen_tok.decode([at])
    sanity = {"prompt": SANITY, "token_id": at, "token": token,
              "logit": best}
    print(f"[check2] next token after {SANITY!r}: {token!r}", flush=True)
    if "Paris" not in token:
        raise SystemExit(f"error: the model continues {SANITY!r} with "
                         f"{token!r}")

    fx = fixtures.FixtureWriter(
        a.name, check=2, family=FAMILY, script=SCRIPT,
        argv=[Path(__file__).name, "--encoder", a.encoder.name])
    fx.add_file("encoder", a.encoder)
    stats, tol_rel, tol_rest, tol_cos, lengths = {}, {}, {}, {}, {}
    for key in prompts:
        t32, t16 = taps32[key], taps16[key]
        fx.add(f"ids_{key}", ids[key])
        fx.add(f"tap_{key}", t32)
        fx.add(f"tap_bf16_{key}", t16)
        gap = fixtures.compare(t16, t32)
        cos = token_cosine(t16, t32)
        row = int(cos.argmin())
        gap["token_cosine_min"] = float(cos.min())
        gap["token_cosine_min_row"] = row
        gap["token_cosine_min_token_id"] = int(ids[key][row])
        gap["rel_l2_by_layer"] = [float(g) for g in gaps[key]]
        if len(t32) > 1:
            gap["rel_l2_after_first"] = fixtures.rel_l2(t16[1:], t32[1:])
            tol_rest[key] = 2 * gap["rel_l2_after_first"]
        stats[key] = gap
        lengths[key] = int(len(ids[key]))
        tol_rel[key] = 2 * gap["rel_l2"]
        tol_cos[key] = min(STARTING["token_cosine_min"],
                           1.0 - 4.0 * (1.0 - gap["token_cosine_min"]))
        print(f"[check2] {key}: {lengths[key]} ids, bf16 vs fp32 rel L2 "
              f"{gap['rel_l2']:.3e}, token cos min "
              f"{gap['token_cosine_min']:.6f}", flush=True)
    fx.inputs.update(
        prompts=prompts, lengths=lengths, comfyui_commit=commit,
        tap=f"output of decoder layer {TAP} of {depth}, before the final "
        "norm, every row",
        tokenizer="comfy.text_encoders.minimax.MiniMaxH3Tokenizer: raw "
        "prompt ids, no template, the seven MiniMax tokens at their fixed "
        "ids, an empty prompt the pad token 151643",
        encoder="transformers Qwen3VLTextDecoderLayer, one resident at a "
        "time, weights from the text GGUF by gguf-py float32 "
        "dequantization; SDPA attention, causal, batch 1, no padding",
        config={k: getattr(cfg, k) for k in (
            "hidden_size", "intermediate_size", "num_attention_heads",
            "num_key_value_heads", "head_dim", "rms_norm_eps",
            "rope_parameters")},
        gguf_codecs=census)
    fx.results.update(
        oracle={"device": "cpu", "threads": torch.get_num_threads(),
                "bf16_matmul": "bf16 operands, float32 GEMM, one bf16 "
                "rounding"},
        name_map_check=sanity, bf16_vs_fp32=stats,
        seconds={"layers_s": round(layer_s, 1)})
    fx.tolerances.update(rule=RULE, starting=STARTING, rel_l2=tol_rel,
                         rel_l2_after_first=tol_rest,
                         token_cosine_min=tol_cos)
    path = fx.write(a.out)
    print(f"[check2] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
