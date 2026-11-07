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

A keyframe prompt (--prompts keyframe keyframes, merge 3) is ComfyUI's
fl2va presentation: per image "<Picture i>: " and a vision block, then the
prompt, with ComfyUI's ids, tags and tower input (process_qwen2vl_images
at patch 16, mean and std 0.5) of seeded images at 640x400 and 400x640,
brought to the 512x512 canvas as MiniMaxH3ImageToVideo brings them. transformers'
Qwen3VLVisionModel runs the mmproj beside the encoder (--mmproj, its
tensors in float32 by gguf-py) in float32 in both passes, as ComfyUI and
gmlx run the tower; its merger rows replace the image pad rows, its
deepstack rows join them after decoder layers 1 to 3, and the rows take
transformers' 3D RoPE positions (Qwen3VLModel.get_rope_index). The
fixture holds the images, the tower's input and output, and the tags,
and a report column that does not gate: ComfyUI's own tower on the same
input and weights, and ComfyUI's positions, against transformers'.

  python scripts/video_oracle/run.py --python <comfyui venv python> \\
      --threads 16 check2_h3.py --comfy-dir <ComfyUI checkout> \\
      --encoder Qwen3-VL-32B-Instruct-Q4_K_M.gguf

  python scripts/video_oracle/run.py --python <comfyui venv python> \\
      --threads 16 check2_h3.py --comfy-dir <ComfyUI checkout> \\
      --encoder Qwen3-VL-32B-Instruct-Q4_K_M.gguf \\
      --mmproj mmproj-BF16.gguf --prompts keyframe keyframes \\
      --name check2_h3_keyframes
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

# The keyframe prompts: the text, and the size (width, height) of each
# source image, first then last. MiniMaxH3ImageToVideo brings each to the
# canvas, the first stretched, the last cropped about its center.
KEYFRAME_TEXT = "A red fox trots through fresh snow at dawn, then looks back"
KEYFRAMES = {"keyframe": [(640, 400)], "keyframes": [(640, 400), (400, 640)]}
CANVAS = (512, 512)
CROPS = ("disabled", "center")
PAD = 151655            # the image pad id the tower's rows replace
# mmproj GGUF tensor names of a vision block, to the names under
# transformers' Qwen3VLVisionBlock, and of a merger.
VISION_BLOCK = {"ln1": "norm1", "ln2": "norm2", "attn_qkv": "attn.qkv",
                "attn_out": "attn.proj", "ffn_up": "mlp.linear_fc1",
                "ffn_down": "mlp.linear_fc2"}
MERGER = {"norm": "norm", "fc1": "linear_fc1", "fc2": "linear_fc2"}

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


def keyframe_image(seed: int, w: int, h: int) -> np.ndarray:
    """A seeded picture, (h, w, 3) uint8: smooth color with mild noise,
    as a photo is."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w] / 37.0
    rgb = np.stack([np.sin(x + 0.3 * y), np.cos(0.7 * x - y),
                    np.sin(x * y / 9)], -1)
    rgb = np.clip((rgb + 1) * 120 + rng.normal(0, 4, rgb.shape), 0, 255)
    return rgb.astype(np.uint8)


def canvas_image(torch, source: np.ndarray, crop: str) -> np.ndarray:
    """``source`` at the canvas, as MiniMaxH3ImageToVideo resizes a
    keyframe (crop "disabled" for the first, "center" for the last),
    uint8: the resize's lanczos returns whole 8-bit levels."""
    from comfy_extras.nodes_minimax_h3 import _resize

    x = torch.from_numpy(source.astype(np.float32) / 255.0)[None]
    out = _resize(x, *CANVAS, crop)[0].numpy()
    im = np.round(out * 255.0).astype(np.uint8)
    if not np.array_equal(im.astype(np.float32) / 255.0, out):
        raise SystemExit("error: the resize left a value off the 8-bit "
                         "levels")
    return im


def comfy_keyframe_prompt(torch, key: str) -> dict:
    """ComfyUI's ids, tags, embeds_info and tower input of a keyframe
    prompt: the image entries expanded to their pad rows, as its encode
    expands them."""
    from comfy.text_encoders import minimax
    from comfy.text_encoders.qwen_vl import process_qwen2vl_images

    sources = [keyframe_image(10 + i, w, h)
               for i, (w, h) in enumerate(KEYFRAMES[key])]
    images = [canvas_image(torch, src, crop)
              for src, crop in zip(sources, CROPS)]
    shown = [torch.from_numpy(im.astype(np.float32) / 255.0)[None]
             for im in images]
    tok = minimax.MiniMaxH3Tokenizer()
    entries = tok.tokenize_with_weights(KEYFRAME_TEXT, images=shown)
    (batch,) = entries["qwen3vl_32b"]
    pixels, grids = [], []
    for im in shown:
        px, grid = process_qwen2vl_images(im, patch_size=16,
                                          image_mean=[0.5, 0.5, 0.5],
                                          image_std=[0.5, 0.5, 0.5])
        pixels.append(px.float())
        grids.append([int(v) for v in grid.reshape(-1)])
    ids, info, k = [], [], 0
    for t, _w in batch:
        if isinstance(t, dict):
            n = grids[k][1] * grids[k][2] // 4
            info.append({"type": "image", "index": len(ids), "size": n,
                         "extra": {"grid": torch.tensor([grids[k]])}})
            ids += [PAD] * n
            k += 1
        else:
            ids.append(int(t))
    tags = minimax.token_tags_from_embeds_info(len(ids), info)
    return {"ids": np.asarray(ids, np.int32), "tags": tags.numpy(),
            "sources": sources, "images": images, "pixels": pixels,
            "grids": grids, "info": info}


def vision_config(kv: dict, tensors: dict):
    """transformers' vision config from the mmproj header's values and
    its position table."""
    from transformers.models.qwen3_vl.configuration_qwen3_vl import \
        Qwen3VLVisionConfig

    def get(key):
        return kv[f"clip.vision.{key}"].contents()

    deep = [i for i, d in enumerate(get("is_deepstack_layers")) if d]
    cfg = Qwen3VLVisionConfig(
        depth=int(get("block_count")),
        hidden_size=int(get("embedding_length")),
        intermediate_size=int(get("feed_forward_length")),
        num_heads=int(get("attention.head_count")),
        patch_size=int(get("patch_size")),
        spatial_merge_size=int(get("spatial_merge_size")),
        temporal_patch_size=2, in_channels=3,
        out_hidden_size=int(get("projection_dim")),
        num_position_embeddings=int(
            tensors["v.position_embd.weight"].shape[-1]),
        deepstack_visual_indexes=deep, hidden_act="gelu_pytorch_tanh")
    cfg._attn_implementation = "sdpa"
    return cfg


def vision_state(torch, tensors: dict, cfg) -> dict:
    """The mmproj's tensors in float32 under transformers' names."""
    def t(name):
        return dequantized(torch, tensors[name])

    out = {"patch_embed.proj.weight": torch.stack(
               [t("v.patch_embd.weight"), t("v.patch_embd.weight.1")], 2),
           "patch_embed.proj.bias": t("v.patch_embd.bias"),
           "pos_embed.weight": t("v.position_embd.weight")}
    for i in range(cfg.depth):
        for g, h in VISION_BLOCK.items():
            for kind in ("weight", "bias"):
                out[f"blocks.{i}.{h}.{kind}"] = t(f"v.blk.{i}.{g}.{kind}")
    for kind in ("weight", "bias"):
        out[f"merger.norm.{kind}"] = t(f"v.post_ln.{kind}")
        out[f"merger.linear_fc1.{kind}"] = t(f"mm.0.{kind}")
        out[f"merger.linear_fc2.{kind}"] = t(f"mm.2.{kind}")
        for k, layer in enumerate(cfg.deepstack_visual_indexes):
            for g, h in MERGER.items():
                out[f"deepstack_merger_list.{k}.{h}.{kind}"] = t(
                    f"v.deepstack.{layer}.{g}.{kind}")
    return out


def rope_positions(torch, hf, ids, grids):
    """transformers' 3D RoPE positions (3, 1, L) of a prompt with images,
    the image pad rows its image tokens."""
    from types import SimpleNamespace

    shim = SimpleNamespace(config=SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2)))
    shim.get_vision_position_ids = (
        lambda *a, **k: hf.Qwen3VLModel.get_vision_position_ids(shim, *a,
                                                                **k))
    x = torch.from_numpy(np.asarray(ids, np.int64))[None]
    pos, _ = hf.Qwen3VLModel.get_rope_index(
        shim, x, (x == PAD).int(), image_grid_thw=torch.tensor(grids))
    return pos


def comfy_column(torch, state: dict, prompt: dict, tower: dict) -> dict:
    """ComfyUI's own Qwen3-VL tower on the same input and weights, and
    its positions, against transformers': a report column, not a gate."""
    import comfy.ops
    from comfy.text_encoders.qwen3vl import (QWEN3VL_VISION,
                                             QWEN3VL_VISION_COMMON,
                                             Qwen3VLVisionModel)
    from comfy.text_encoders.qwen_vl import qwen2vl_mrope_position_ids

    cfg = {**QWEN3VL_VISION_COMMON, **QWEN3VL_VISION["qwen3vl_32b"],
           "out_hidden_size": 5120}
    vis = Qwen3VLVisionModel(cfg, device="cpu", dtype=torch.float32,
                             ops=comfy.ops.manual_cast).eval()
    vis.load_state_dict(state, strict=True)
    gaps = []
    with torch.no_grad():
        for i, (px, grid) in enumerate(zip(prompt["pixels"],
                                           prompt["grids"])):
            merged, deep = vis(px, torch.tensor([grid]))
            row = {"merger": fixtures.compare(oracle.to_numpy(merged),
                                              tower[f"merger_{i}"])}
            for k, d in enumerate(deep):
                row[f"deepstack_{k}"] = fixtures.compare(
                    oracle.to_numpy(d), tower[f"deepstack_{i}_{k}"])
            gaps.append(row)
    pos = qwen2vl_mrope_position_ids(prompt["info"], len(prompt["ids"]),
                                     "cpu")
    return {"tower": gaps, "positions_equal": bool(np.array_equal(
        pos.numpy().astype(np.int64), prompt["positions"][:, 0].numpy()))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--comfy-dir", type=Path, required=True,
                    help="ComfyUI checkout (its H3 tokenizer)")
    ap.add_argument("--encoder", type=Path, required=True,
                    help="stock Qwen3-VL-32B-Instruct text GGUF")
    ap.add_argument("--prompts", nargs="+", default=list(PROMPTS),
                    choices=[*PROMPTS, *KEYFRAMES])
    ap.add_argument("--mmproj", type=Path,
                    help="the encoder's mmproj GGUF, for a keyframe prompt")
    ap.add_argument("--name", default=NAME)
    ap.add_argument("--out", type=Path,
                    help="fixture directory (default $GMLX_VIDEO_FIXTURES)")
    a = ap.parse_args(argv)
    torch = oracle.setup()
    commit = oracle.start_comfy(a.comfy_dir)
    import gguf
    from transformers.models.qwen3_vl import modeling_qwen3_vl as hf

    prompts = {k: PROMPTS.get(k, KEYFRAME_TEXT) for k in a.prompts}
    ids, qwen_tok = comfy_ids({k: v for k, v in {**prompts,
                                                  "sanity": SANITY}.items()
                               if k not in KEYFRAMES})
    framed = {k: comfy_keyframe_prompt(torch, k) for k in prompts
              if k in KEYFRAMES}
    if framed and a.mmproj is None:
        ap.error("a keyframe prompt needs --mmproj")
    for k, f in framed.items():
        ids[k] = f["ids"]
    tower, deep, column = {}, {}, {}
    if framed:
        vreader = gguf.GGUFReader(str(a.mmproj))
        vtensors = {t.name: t for t in vreader.tensors}
        vcfg = vision_config(vreader.fields, vtensors)
        state = vision_state(torch, vtensors, vcfg)
        vis = hf.Qwen3VLVisionModel(vcfg).eval()
        vis.load_state_dict(state, strict=True)
        with torch.no_grad():
            for k, f in framed.items():
                merged, stacks = [], []
                for i, (px, grid) in enumerate(zip(f["pixels"], f["grids"])):
                    o = vis(px, grid_thw=torch.tensor([grid]))
                    merged.append(o.pooler_output)
                    stacks.append(o.deepstack_features)
                    tower[k, f"merger_{i}"] = oracle.to_numpy(o.pooler_output)
                    for j, d in enumerate(o.deepstack_features):
                        tower[k, f"deepstack_{i}_{j}"] = oracle.to_numpy(d)
                f["merged"] = torch.cat(merged)
                deep[k] = [torch.cat([s[j] for s in stacks])
                           for j in range(len(stacks[0]))]
                f["positions"] = rope_positions(torch, hf, f["ids"],
                                                f["grids"])
                column[k] = comfy_column(torch, state, f, {
                    n: v for (kk, n), v in tower.items() if kk == k})
                print(f"[check2] {k}: ComfyUI's tower and positions "
                      f"against transformers': {column[k]}", flush=True)
        del vis, state
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
            if key in framed:
                f = framed[key]
                f["mask"] = torch.from_numpy(f["ids"] == PAD)[None]
                x[f["mask"]] = f["merged"]
                pos = f["positions"]
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
                if key in deep and i < len(deep[key]):
                    h32[key] = hf.Qwen3VLTextModel._deepstack_process(
                        None, h32[key], framed[key]["mask"], deep[key][i])
            if i < TAP:
                layer16.load_state_dict(
                    {k: v.to(torch.bfloat16) for k, v in state.items()},
                    strict=True)
                oracle.emulate_bf16_matmul(True)
                for key in prompts:
                    h16[key] = layer16(h16[key],
                                       position_embeddings=pe16[key],
                                       attention_mask=None)
                    if key in deep and i < len(deep[key]):
                        h16[key] = hf.Qwen3VLTextModel._deepstack_process(
                            None, h16[key], framed[key]["mask"],
                            deep[key][i])
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

    argv = [Path(__file__).name, "--encoder", a.encoder.name]
    if framed:
        argv += ["--mmproj", a.mmproj.name, "--prompts", *a.prompts,
                 "--name", a.name]
    fx = fixtures.FixtureWriter(a.name, check=2, family=FAMILY,
                                script=SCRIPT, argv=argv)
    fx.add_file("encoder", a.encoder)
    if framed:
        fx.add_file("mmproj", a.mmproj)
    for k, f in framed.items():
        fx.add(f"tags_{k}", f["tags"])
        fx.add(f"positions_{k}", oracle.to_numpy(f["positions"][:, 0]))
        for i, (src, im, px) in enumerate(zip(f["sources"], f["images"],
                                              f["pixels"])):
            fx.add(f"source_{k}_{i}", src)
            fx.add(f"image_{k}_{i}", im)
            fx.add(f"pixels_{k}_{i}", oracle.to_numpy(px))
    for (k, n), v in tower.items():
        fx.add(f"tower_{k}_{n}", v)
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
    if framed:
        fx.inputs.update(
            keyframes={k: {"text": KEYFRAME_TEXT,
                           "sources": [list(sz) for sz in KEYFRAMES[k]],
                           "sizes": [list(CANVAS)] * len(KEYFRAMES[k]),
                           "crops": list(CROPS[:len(KEYFRAMES[k])]),
                           "grids": framed[k]["grids"]} for k in framed},
            images="keyframe_image(10 + i): seeded smooth color, uint8, "
            "at the source size; MiniMaxH3ImageToVideo's _resize to the "
            "canvas, the first with crop disabled, the last center",
            tower="transformers Qwen3VLVisionModel on the mmproj in "
            "float32 in both passes; its merger rows replace the image pad "
            "rows, its deepstack rows join them after decoder layers 1 to "
            f"{len(next(iter(deep.values())))}; RoPE positions of "
            "Qwen3VLModel.get_rope_index")
        fx.results["comfyui_column"] = column
    fx.tolerances.update(rule=RULE, starting=STARTING, rel_l2=tol_rel,
                         rel_l2_after_first=tol_rest,
                         token_cosine_min=tol_cos)
    path = fx.write(a.out)
    print(f"[check2] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
