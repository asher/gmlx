#!/usr/bin/env python3
"""Check 2 oracle for MiniMax-H3-Image: the text encoder tap on image
prompts.

H3-Image conditions on the encoder of MiniMax-H3, Qwen3-VL-32B-Instruct
read after decoder layer 50 of 64 before the final norm, so this is the
method of scripts/video_oracle/check2_h3.py, whose layer, name map and
GGUF helpers it loads by path. The ids are what ComfyUI's
MiniMaxH3ImageConditioning hands the encoder for a prompt without
images: MiniMaxH3Tokenizer.tokenize_with_weights with no images and the
node's picture label, which is the raw prompt.

The prompts are image prompts: check 2's shared four, one that asks for
lettering, and a prompt of exactly 4096 ids (rows_4096), built from a
fixed list of scene sentences and cut to that length. rows_256 and
rows_1024 are the same prompt at those lengths, for a probe.

Each pass runs one decoder layer resident at a time, from the token
embedding rows, at batch 1 with no padding, through transformers' SDPA,
causal with no mask: the fp32 pass in float32, the bf16 pass on bfloat16
layers with each product a float32 GEMM over bfloat16 operands rounded
once. A short prompt also runs layers 51 to 64, the final norm and the
output matrix, and its next token must be " Paris". The manifest holds
each prompt's gap after every layer and the tolerances of check2_h3.py:
relative L2 at twice the gap at the tap, over every row and over the
rows after the first, and a per-token cosine floor of 1 - 4 (1 - c), at
most 0.999.

--layers N runs the first N layers of the passes, prints the footprint
after each and writes nothing, to step a new prompt length up first.

  python scripts/image_oracle/run.py --python <ComfyUI venv python> \\
      --threads 16 check2_h3_image.py --comfy-dir <ComfyUI checkout> \\
      --encoder Qwen3-VL-32B-Instruct-Q4_K_M.gguf
"""

from __future__ import annotations

import argparse
import importlib.util
import resource
import time
from collections import Counter
from pathlib import Path

import numpy as np

import check3_h3_image as c3
import fixtures
import oracle

ORACLE_NEEDS = ("torch", "transformers", "gguf", "einops", "comfy_kitchen")

NAME = "check2_h3_image"
FAMILY = "h3_image"
SCRIPT = "scripts/image_oracle/check2_h3_image.py"
PICTURE_LABEL = "Picture %d: "     # MiniMaxH3ImageConditioning's
TYPOGRAPHY = ('A hand-painted wooden sign above a bakery door that reads '
              '"FRESH BREAD 7AM" in cream letters, a chalkboard beside it '
              'listing "Rye 4.50" and "Sourdough 6", morning light')
SCENES = (
    "A lighthouse stands on a granite cliff above a cold green sea.",
    "Gulls circle a fishing boat whose red hull is streaked with rust.",
    "Fog lies in the valley below a village of slate roofs and stone walls.",
    "A woman in a yellow raincoat walks a grey dog along the harbor wall.",
    "Lanterns hang from a wire strung between two crooked chimneys.",
    "A market stall sells lemons, figs and bunches of dried lavender.",
    "Children chalk a hopscotch grid on the wet cobbles of a square.",
    "An old tram climbs a steep street lined with tiled facades.",
    "A heron waits in the reeds at the edge of a still brown pond.",
    "Snow dusts the pines on a ridge under a pale winter sun.",
    "A clock tower shows ten past four above a busy railway station.",
    "Copper pots hang over a stove in a narrow kitchen with blue tiles.",
)
ROWS = (256, 1024, 4096)


def _video_script(name: str):
    """A video oracle script loaded by path for its constants and
    helpers."""
    path = Path(__file__).resolve().parent.parent / "video_oracle" / name
    spec = importlib.util.spec_from_file_location(f"_vid_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"error: cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v2 = _video_script("check2_h3.py")
SHARED = ("plain", "empty", "non_ascii", "long")
PROMPTS = {**{k: v2.PROMPTS[k] for k in SHARED}, "typography": TYPOGRAPHY}
DEFAULT = (*PROMPTS, "rows_4096")
CHOICES = (*PROMPTS, *(f"rows_{n}" for n in ROWS))


def _max_rss() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30


def node_ids(tok, text: str) -> np.ndarray:
    """The ids MiniMaxH3ImageConditioning hands the encoder for ``text``
    without images."""
    entries = tok.tokenize_with_weights(
        text, images=[], picture_label=PICTURE_LABEL)["qwen3vl_32b"]
    if len(entries) != 1:
        raise SystemExit(f"error: {len(entries)} token batches")
    return np.asarray([int(e[0]) for e in entries[0]], np.int32)


def rows_prompt(tok, rows: int) -> str:
    """The scene sentences in turn, numbered, cut to the most words of at
    most ``rows`` ids, then " and" until it is ``rows`` ids."""
    words, i = [], 0
    while len(words) < 2 * rows:
        words += f"Panel {i + 1}: {SCENES[i % len(SCENES)]}".split()
        i += 1
    lo, hi = 1, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(node_ids(tok, " ".join(words[:mid]))) <= rows:
            lo = mid
        else:
            hi = mid - 1
    text = " ".join(words[:lo])
    while len(node_ids(tok, text)) < rows:
        text += " and"
    if len(node_ids(tok, text)) != rows:
        raise SystemExit(f"error: no prompt of exactly {rows} ids")
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--comfy-dir", type=Path, required=True,
                    help="ComfyUI checkout (its H3 tokenizer)")
    ap.add_argument("--encoder", type=Path, required=True,
                    help="stock Qwen3-VL-32B-Instruct text GGUF")
    ap.add_argument("--prompts", nargs="+", default=list(DEFAULT),
                    choices=CHOICES)
    ap.add_argument("--layers", type=int,
                    help="run this many layers, print, write nothing")
    ap.add_argument("--name", default=NAME)
    ap.add_argument("--out", type=Path,
                    help="fixture directory (default $GMLX_IMAGE_FIXTURES)")
    a = ap.parse_args(argv)
    torch = oracle.setup()
    commit = c3.comfy_oracle().start_comfy(a.comfy_dir)
    import gguf
    from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
    from transformers.models.qwen3_vl import modeling_qwen3_vl as hf

    tok = MiniMaxH3Tokenizer()
    prompts = {k: PROMPTS[k] if k in PROMPTS
               else rows_prompt(tok, int(k.split("_")[1]))
               for k in a.prompts}
    ids = {k: node_ids(tok, v) for k, v in
           {**prompts, "sanity": v2.SANITY}.items()}
    for k in prompts:
        print(f"[check2] {k}: {len(ids[k])} ids", flush=True)
    reader = gguf.GGUFReader(str(a.encoder))
    tensors = {t.name: t for t in reader.tensors}
    cfg, depth = v2.text_config(reader.fields)
    census = dict(Counter(t.tensor_type.name for t in reader.tensors))
    print(f"[check2] {depth} layers, hidden {cfg.hidden_size}, {census}",
          flush=True)
    run_to = depth if a.layers is None else min(a.layers, v2.TAP)

    layer32 = hf.Qwen3VLTextDecoderLayer(cfg, 0).eval()
    layer16 = hf.Qwen3VLTextDecoderLayer(cfg, 0).eval().to(torch.bfloat16)
    rotary = hf.Qwen3VLTextRotaryEmbedding(cfg)
    embed = tensors["token_embd.weight"]
    h32, h16, pe32, pe16 = {}, {}, {}, {}
    with torch.no_grad():
        for key, row in ids.items():
            x = v2.rows_of(torch, embed, row.tolist())[None]
            pos = torch.arange(x.shape[1]).view(1, 1, -1).expand(3, 1, -1)
            h32[key] = x
            pe32[key] = rotary(x, pos)
            if key != "sanity":
                h16[key] = x.to(torch.bfloat16)
                pe16[key] = rotary(h16[key], pos)
    gaps = {k: [] for k in prompts}
    t0 = time.perf_counter()
    taps32, taps16 = {}, {}
    for i in range(run_to):
        state = {v2.LAYER[s]: v2.dequantized(torch, tensors[f"blk.{i}.{s}"])
                 for s in v2.LAYER}
        layer32.load_state_dict(state, strict=True)
        keys = ["sanity"] if i >= v2.TAP else list(ids)
        with torch.no_grad():
            for key in keys:
                h32[key] = layer32(h32[key], position_embeddings=pe32[key],
                                   attention_mask=None)
            if i < v2.TAP:
                layer16.load_state_dict(
                    {k: v.to(torch.bfloat16) for k, v in state.items()},
                    strict=True)
                oracle.emulate_bf16_matmul(True)
                try:
                    for key in prompts:
                        h16[key] = layer16(h16[key],
                                           position_embeddings=pe16[key],
                                           attention_mask=None)
                finally:
                    oracle.emulate_bf16_matmul(False)
                for key in prompts:
                    gaps[key].append(fixtures.rel_l2(
                        h16[key][0].float().numpy(), h32[key][0].numpy()))
        del state
        if i + 1 == v2.TAP:
            taps32 = {k: h32[k][0].numpy().copy() for k in prompts}
            taps16 = {k: h16[k][0].float().numpy() for k in prompts}
        print(f"[check2] layer {i + 1} of {depth}, "
              f"{time.perf_counter() - t0:.0f} s, max rss "
              f"{_max_rss():.1f} GiB", flush=True)
    layer_s = time.perf_counter() - t0
    if a.layers is not None:
        print(f"[check2] probe: {run_to} layers in {layer_s:.0f} s, max rss "
              f"{_max_rss():.1f} GiB; nothing written", flush=True)
        return 0

    with torch.no_grad():
        last = h32["sanity"][0, -1]
        norm = v2.dequantized(torch, tensors["output_norm.weight"])
        v = last * torch.rsqrt(last.pow(2).mean() + cfg.rms_norm_eps) * norm
        head = tensors["output.weight"]
        best, at = -float("inf"), -1
        for lo in range(0, int(head.shape[1]), v2.ROWS):
            w = v2.rows_of(torch, head, range(lo, min(lo + v2.ROWS,
                                                      int(head.shape[1]))))
            logits = w @ v
            j = int(logits.argmax())
            if float(logits[j]) > best:
                best, at = float(logits[j]), lo + j
    token = tok.qwen3vl_32b.tokenizer.decode([at])
    sanity = {"prompt": v2.SANITY, "token_id": at, "token": token,
              "logit": best}
    print(f"[check2] next token after {v2.SANITY!r}: {token!r}", flush=True)
    if "Paris" not in token:
        raise SystemExit(f"error: the model continues {v2.SANITY!r} with "
                         f"{token!r}")

    argv = [Path(__file__).name, "--encoder", a.encoder.name]
    if list(a.prompts) != list(DEFAULT):
        argv += ["--prompts", *a.prompts, "--name", a.name]
    fx = fixtures.FixtureWriter(a.name, check=2, family=FAMILY,
                                script=SCRIPT, argv=argv)
    fx.add_file("encoder", a.encoder)
    stats, tol_rel, tol_rest, tol_cos, lengths = {}, {}, {}, {}, {}
    for key in prompts:
        t32, t16 = taps32[key], taps16[key]
        fx.add(f"ids_{key}", ids[key])
        fx.add(f"tap_{key}", t32)
        fx.add(f"tap_bf16_{key}", t16)
        gap = fixtures.compare(t16, t32)
        cos = v2.token_cosine(t16, t32)
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
        tol_cos[key] = min(v2.STARTING["token_cosine_min"],
                           1.0 - 4.0 * (1.0 - gap["token_cosine_min"]))
        print(f"[check2] {key}: {lengths[key]} ids, bf16 vs fp32 rel L2 "
              f"{gap['rel_l2']:.3e}, token cos min "
              f"{gap['token_cosine_min']:.6f}", flush=True)
    fx.inputs.update(
        prompts=prompts, lengths=lengths, comfyui_commit=commit,
        tap=f"output of decoder layer {v2.TAP} of {depth}, before the "
        "final norm, every row",
        tokenizer="comfy.text_encoders.minimax.MiniMaxH3Tokenizer."
        "tokenize_with_weights(prompt, images=[], picture_label="
        f"{PICTURE_LABEL!r}), as MiniMaxH3ImageConditioning calls it "
        "without images: the raw prompt ids, the seven MiniMax tokens at "
        "their fixed ids, an empty prompt the pad token 151643",
        rows_prompts="SCENES numbered in turn, cut to the most words of at "
        "most N ids, then ' and' until N ids",
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
        seconds={"layers_s": round(layer_s, 1)},
        max_rss_gib=round(_max_rss(), 2))
    fx.tolerances.update(rule=v2.RULE, starting=v2.STARTING,
                         rel_l2=tol_rel, rel_l2_after_first=tol_rest,
                         token_cosine_min=tol_cos)
    path = fx.write(a.out)
    print(f"[check2] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
